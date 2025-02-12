use crate::aux_schema::BundleMismatchType;
use crate::ExecutionReceiptFor;
use codec::{Decode, Encode};
use domain_block_builder::BlockBuilder;
use domain_runtime_primitives::opaque::AccountId;
use domain_runtime_primitives::CheckExtrinsicsValidityError;
use sc_client_api::{AuxStore, BlockBackend, ExecutorProvider, ProofProvider};
use sc_domains::FPStorageKeyProvider;
use sp_api::{ApiExt, ProvideRuntimeApi};
use sp_blockchain::HeaderBackend;
use sp_core::traits::CodeExecutor;
use sp_core::H256;
use sp_domain_digests::AsPredigest;
use sp_domains::core_api::DomainCoreApi;
use sp_domains::proof_provider_and_verifier::StorageProofProvider;
use sp_domains::{
    DomainId, DomainsApi, ExtrinsicDigest, HeaderHashingFor, InvalidBundleType, RuntimeId,
};
use sp_domains_fraud_proof::execution_prover::ExecutionProver;
use sp_domains_fraud_proof::fraud_proof::{
    ApplyExtrinsicMismatch, DomainRuntimeCodeAt, ExecutionPhase, FinalizeBlockMismatch, FraudProof,
    FraudProofVariant, InvalidBlockFeesProof, InvalidBundlesProof, InvalidBundlesProofData,
    InvalidDomainBlockHashProof, InvalidExtrinsicsRootProof, InvalidStateTransitionProof,
    InvalidTransfersProof, MmrRootProof, ValidBundleDigest, ValidBundleProof,
};
use sp_domains_fraud_proof::storage_proof::{self, *};
use sp_domains_fraud_proof::FraudProofApi;
use sp_messenger::MessengerApi;
use sp_mmr_primitives::MmrApi;
use sp_runtime::generic::BlockId;
use sp_runtime::traits::{Block as BlockT, HashingFor, Header as HeaderT, NumberFor, One};
use sp_runtime::{Digest, DigestItem};
use sp_trie::LayoutV1;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum TraceDiffType {
    Shorter,
    Longer,
    Mismatch,
}

/// Error type for fraud proof generation.
#[derive(Debug, thiserror::Error)]
pub enum FraudProofError {
    #[error(
        "Out of bounds extrinsic index for creating the fraud proof, got: {index}, max: {max}"
    )]
    OutOfBoundsExtrinsicIndex { index: usize, max: usize },
    #[error(transparent)]
    Blockchain(#[from] sp_blockchain::Error),
    #[error(transparent)]
    RuntimeApi(#[from] sp_api::ApiError),
    #[error("Missing bundle at {bundle_index}")]
    MissingBundle { bundle_index: usize },
    #[error("Invalid bundle extrinsic at bundle index[{bundle_index}]")]
    InvalidBundleExtrinsic { bundle_index: usize },
    #[error("Fail to generate the proof of inclusion")]
    FailToGenerateProofOfInclusion,
    #[error("Missing extrinsics in the Consesnsus block")]
    MissingConsensusExtrinsics,
    #[error("Unable to decode opaque bundle's extrinsic at index {extrinsic_index}")]
    UnableToDecodeOpaqueBundleExtrinsic {
        extrinsic_index: usize,
        decoding_error: codec::Error,
    },
    #[error(
        "Invalid extrinsic index for creating illegal tx fraud proof, \
        expected extrinsic index: {index},\
        is_good_invalid_fraud_proof: {is_good_invalid_fraud_proof} \
        and validity response is: {extrinsics_validity_response:?}"
    )]
    InvalidIllegalTxFraudProofExtrinsicIndex {
        index: usize,
        is_good_invalid_fraud_proof: bool,
        extrinsics_validity_response: Result<(), CheckExtrinsicsValidityError>,
    },
    #[error("Fail to generate the storage proof")]
    StorageProof(#[from] storage_proof::GenerationError),
}

pub struct FraudProofGenerator<Block: BlockT, CBlock, Client, CClient, Backend, E> {
    client: Arc<Client>,
    consensus_client: Arc<CClient>,
    backend: Arc<Backend>,
    code_executor: Arc<E>,
    storage_key_provider: FPStorageKeyProvider<CBlock, Block::Header, CClient>,
    _phantom: PhantomData<(Block, CBlock)>,
}

impl<Block: BlockT, CBlock, Client, CClient, Backend, E> Clone
    for FraudProofGenerator<Block, CBlock, Client, CClient, Backend, E>
{
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            consensus_client: self.consensus_client.clone(),
            backend: self.backend.clone(),
            code_executor: self.code_executor.clone(),
            storage_key_provider: self.storage_key_provider.clone(),
            _phantom: self._phantom,
        }
    }
}

type FraudProofFor<CBlock, DomainHeader> =
    FraudProof<NumberFor<CBlock>, <CBlock as BlockT>::Hash, DomainHeader, H256>;

impl<Block, CBlock, Client, CClient, Backend, E>
    FraudProofGenerator<Block, CBlock, Client, CClient, Backend, E>
where
    Block: BlockT,
    Block::Hash: Into<H256>,
    CBlock: BlockT,
    Client: HeaderBackend<Block>
        + BlockBackend<Block>
        + AuxStore
        + ProvideRuntimeApi<Block>
        + ProofProvider<Block>
        + ExecutorProvider<Block>
        + 'static,
    Client::Api: sp_block_builder::BlockBuilder<Block>
        + sp_api::ApiExt<Block>
        + DomainCoreApi<Block>
        + MessengerApi<Block, NumberFor<CBlock>, CBlock::Hash>,
    CClient: HeaderBackend<CBlock>
        + BlockBackend<CBlock>
        + ProvideRuntimeApi<CBlock>
        + ProofProvider<CBlock>
        + 'static,
    CClient::Api: DomainsApi<CBlock, Block::Header>
        + MmrApi<CBlock, H256, NumberFor<CBlock>>
        + FraudProofApi<CBlock, Block::Header>,
    Backend: sc_client_api::Backend<Block> + Send + Sync + 'static,
    E: CodeExecutor,
{
    pub fn new(
        client: Arc<Client>,
        consensus_client: Arc<CClient>,
        backend: Arc<Backend>,
        code_executor: Arc<E>,
    ) -> Self {
        Self {
            client,
            consensus_client: consensus_client.clone(),
            backend,
            code_executor,
            storage_key_provider: FPStorageKeyProvider::new(consensus_client),
            _phantom: Default::default(),
        }
    }

    #[allow(clippy::type_complexity)]
    fn maybe_generate_domain_runtime_code_proof_for_receipt(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
    ) -> Result<Option<DomainRuntimeCodeAt<NumberFor<CBlock>, CBlock::Hash, H256>>, FraudProofError>
    {
        // NOTE: domain runtime code is take affect in the next block, so we need to get
        // the doamin runtime code of the parent block, which is what used to derived the
        // ER.
        let (parent_consensus_number, parent_consensus_hash) = {
            let consensus_header = self.consensus_header(local_receipt.consensus_block_hash)?;
            (
                local_receipt.consensus_block_number - One::one(),
                *consensus_header.parent_hash(),
            )
        };

        let best_hash = self.consensus_client.info().best_hash;
        let runtime_api = self.consensus_client.runtime_api();
        let runtime_id = runtime_api
            .runtime_id(best_hash, domain_id)?
            .ok_or_else(|| {
                sp_blockchain::Error::Application(
                    "Failed to get domain runtime id".to_string().into(),
                )
            })?;
        let is_domain_runtime_upgraded_since = runtime_api
            .is_domain_runtime_upgraded_since(best_hash, domain_id, parent_consensus_number)?
            .ok_or_else(|| {
                sp_blockchain::Error::Application(
                    "Failed to get domain runtime object".to_string().into(),
                )
            })?;

        if is_domain_runtime_upgraded_since {
            let mmr_proof =
                sc_domains::generate_mmr_proof(&self.consensus_client, parent_consensus_number)?;
            let domain_runtime_code_proof = DomainRuntimeCodeProof::generate(
                self.consensus_client.as_ref(),
                parent_consensus_hash,
                runtime_id,
                &self.storage_key_provider,
            )?;
            Ok(Some(DomainRuntimeCodeAt {
                mmr_proof,
                domain_runtime_code_proof,
            }))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn generate_invalid_state_transition_proof(
        &self,
        domain_id: DomainId,
        execution_phase: ExecutionPhase,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
        bad_receipt_trace_length: usize,
        bad_receipt_hash: Block::Hash,
    ) -> Result<FraudProofFor<CBlock, Block::Header>, FraudProofError> {
        let block_hash = local_receipt.domain_block_hash;
        let block_number = local_receipt.domain_block_number;
        let header = self.header(block_hash)?;
        let parent_header = self.header(*header.parent_hash())?;

        let prover = ExecutionProver::new(self.backend.clone(), self.code_executor.clone());

        let inherent_digests = Digest {
            logs: vec![DigestItem::consensus_block_info(
                local_receipt.consensus_block_hash,
            )],
        };

        let extrinsics = self.block_body(block_hash)?;
        let max_extrinsic_index = extrinsics.len() - 1;
        let encoded_extrinsics: Vec<_> = extrinsics.iter().map(Encode::encode).collect();

        let mut block_builder = BlockBuilder::new(
            self.client.clone(),
            parent_header.hash(),
            *parent_header.number(),
            inherent_digests.clone(),
            self.backend.clone(),
            self.code_executor.clone(),
            extrinsics.into(),
            // NOTE: the inherent extrinsic is already contained in the above `extrinsics`, which
            // is getting from the block body, thus it is okay to pass `maybe_inherent_data` as
            // `None`.
            None,
        )?;

        let (storage_changes, call_data) = match &execution_phase {
            ExecutionPhase::InitializeBlock => (
                None,
                Block::Header::new(
                    block_number,
                    Default::default(),
                    Default::default(),
                    parent_header.hash(),
                    inherent_digests,
                )
                .encode(),
            ),
            ExecutionPhase::ApplyExtrinsic { mismatch, .. } => {
                let extrinsic_index = match mismatch {
                    ApplyExtrinsicMismatch::StateRoot(trace_mismatch_index) => {
                        (trace_mismatch_index - 1) as usize
                    }
                    ApplyExtrinsicMismatch::Shorter => (bad_receipt_trace_length - 1) - 1,
                };

                let target_extrinsic = encoded_extrinsics.get(extrinsic_index).ok_or(
                    FraudProofError::OutOfBoundsExtrinsicIndex {
                        index: extrinsic_index,
                        max: max_extrinsic_index,
                    },
                )?;

                (
                    Some(block_builder.prepare_storage_changes_before(extrinsic_index)?),
                    target_extrinsic.clone(),
                )
            }
            ExecutionPhase::FinalizeBlock { .. } => (
                Some(block_builder.prepare_storage_changes_before_finalize_block()?),
                Vec::new(),
            ),
        };

        let delta_changes = storage_changes.map(|storage_changes| {
            (
                storage_changes.storage_changes.transaction,
                storage_changes.storage_changes.transaction_storage_root,
            )
        });

        let execution_proof = prover.prove_execution(
            parent_header.hash(),
            &execution_phase,
            call_data.as_slice(),
            delta_changes,
        )?;

        let maybe_domain_runtime_code_proof =
            self.maybe_generate_domain_runtime_code_proof_for_receipt(domain_id, local_receipt)?;

        let invalid_state_transition_proof = FraudProof {
            domain_id,
            bad_receipt_hash,
            maybe_mmr_proof: None,
            maybe_domain_runtime_code_proof,
            proof: FraudProofVariant::InvalidStateTransition(InvalidStateTransitionProof {
                execution_proof,
                execution_phase,
            }),
        };

        Ok(invalid_state_transition_proof)
    }

    pub(crate) fn generate_valid_bundle_proof(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
        bundle_index: usize,
        bad_receipt_hash: Block::Hash,
    ) -> Result<FraudProofFor<CBlock, Block::Header>, FraudProofError> {
        let consensus_block_hash = local_receipt.consensus_block_hash;
        let consensus_block_number = local_receipt.consensus_block_number;
        let consensus_extrinsics = self
            .consensus_client
            .block_body(consensus_block_hash)?
            .ok_or(FraudProofError::MissingConsensusExtrinsics)?;

        let mut bundles = self
            .consensus_client
            .runtime_api()
            .extract_successful_bundles(consensus_block_hash, domain_id, consensus_extrinsics)?;

        if bundle_index >= bundles.len() {
            return Err(FraudProofError::MissingBundle { bundle_index });
        }
        let bundle = bundles.swap_remove(bundle_index);

        let bundle_with_proof = OpaqueBundleWithProof::generate(
            &self.storage_key_provider,
            self.consensus_client.as_ref(),
            domain_id,
            consensus_block_hash,
            bundle,
            bundle_index as u32,
        )?;

        let mmr_proof =
            sc_domains::generate_mmr_proof(&self.consensus_client, consensus_block_number)?;

        let maybe_domain_runtime_code_proof =
            self.maybe_generate_domain_runtime_code_proof_for_receipt(domain_id, local_receipt)?;

        let valid_bundle_proof = FraudProof {
            domain_id,
            bad_receipt_hash,
            maybe_mmr_proof: Some(mmr_proof),
            maybe_domain_runtime_code_proof,
            proof: FraudProofVariant::ValidBundle(ValidBundleProof { bundle_with_proof }),
        };

        Ok(valid_bundle_proof)
    }

    pub(crate) fn generate_invalid_domain_extrinsics_root_proof(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
        bad_receipt_hash: Block::Hash,
    ) -> Result<FraudProofFor<CBlock, Block::Header>, FraudProofError> {
        let consensus_block_hash = local_receipt.consensus_block_hash;
        let consensus_block_number = local_receipt.consensus_block_number;

        let valid_bundle_digests =
            self.generate_valid_bundle_digest_for_receipt(domain_id, local_receipt)?;

        let mmr_proof =
            sc_domains::generate_mmr_proof(&self.consensus_client, consensus_block_number)?;

        let maybe_domain_runtime_code_proof =
            self.maybe_generate_domain_runtime_code_proof_for_receipt(domain_id, local_receipt)?;

        let maybe_runtime_id =
            self.is_domain_runtime_upgraded_at(domain_id, consensus_block_hash)?;

        let invalid_inherent_extrinsic_proofs = InvalidInherentExtrinsicDataProof::generate(
            self.consensus_client.as_ref(),
            consensus_block_hash,
            (),
            &self.storage_key_provider,
        )?;

        let maybe_domain_runtime_upgraded_proof = MaybeDomainRuntimeUpgradedProof::generate(
            &self.storage_key_provider,
            self.consensus_client.as_ref(),
            consensus_block_hash,
            maybe_runtime_id,
        )?;

        let domain_chain_allowlist_proof = DomainChainsAllowlistUpdateStorageProof::generate(
            self.consensus_client.as_ref(),
            consensus_block_hash,
            domain_id,
            &self.storage_key_provider,
        )?;

        let domain_sudo_call_proof = DomainSudoCallStorageProof::generate(
            self.consensus_client.as_ref(),
            consensus_block_hash,
            domain_id,
            &self.storage_key_provider,
        )?;

        let evm_domain_contract_creation_allowed_by_call_proof =
            EvmDomainContractCreationAllowedByCallStorageProof::generate(
                self.consensus_client.as_ref(),
                consensus_block_hash,
                domain_id,
                &self.storage_key_provider,
            )?;

        let invalid_domain_extrinsics_root_proof = FraudProof {
            domain_id,
            bad_receipt_hash,
            maybe_mmr_proof: Some(mmr_proof),
            maybe_domain_runtime_code_proof,
            proof: FraudProofVariant::InvalidExtrinsicsRoot(InvalidExtrinsicsRootProof {
                valid_bundle_digests,
                invalid_inherent_extrinsic_proofs,
                maybe_domain_runtime_upgraded_proof,
                domain_chain_allowlist_proof,
                domain_sudo_call_proof,
                evm_domain_contract_creation_allowed_by_call_proof,
            }),
        };

        Ok(invalid_domain_extrinsics_root_proof)
    }

    pub fn is_domain_runtime_upgraded_at(
        &self,
        domain_id: DomainId,
        at: CBlock::Hash,
    ) -> Result<Option<RuntimeId>, FraudProofError> {
        let runtime_id = self
            .consensus_client
            .runtime_api()
            .runtime_id(at, domain_id)?
            .ok_or(sp_blockchain::Error::Application(Box::from(format!(
                "No RuntimeId found for {domain_id:?}"
            ))))?;
        // This API is only present in API versions 2 and later, but it is safe to call
        // unconditionally, because:
        // - on Mainnet, there are no domains yet, and
        // - on Taurus, there are no invalid execution receipts yet.
        let runtime_upgrades = self.consensus_client.runtime_api().runtime_upgrades(at)?;

        let is_runtime_upgraded = runtime_upgrades.contains(&runtime_id);

        Ok(is_runtime_upgraded.then_some(runtime_id))
    }

    pub(crate) fn generate_valid_bundle_digest_for_receipt(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
    ) -> Result<Vec<ValidBundleDigest>, FraudProofError> {
        let consensus_block_hash = local_receipt.consensus_block_hash;
        let consensus_extrinsics = self
            .consensus_client
            .block_body(consensus_block_hash)?
            .ok_or_else(|| {
                sp_blockchain::Error::Backend(format!(
                    "BlockBody of {consensus_block_hash:?} unavailable"
                ))
            })?;

        let bundles = self
            .consensus_client
            .runtime_api()
            .extract_successful_bundles(consensus_block_hash, domain_id, consensus_extrinsics)?;

        let domain_runtime_api = self.client.runtime_api();
        let mut valid_bundle_digests = Vec::new();
        for bundle_index in local_receipt.valid_bundle_indexes() {
            let bundle =
                bundles
                    .get(bundle_index as usize)
                    .ok_or(FraudProofError::MissingBundle {
                        bundle_index: bundle_index as usize,
                    })?;

            let mut exts = Vec::with_capacity(bundle.extrinsics.len());
            for opaque_extrinsic in &bundle.extrinsics {
                let extrinsic = Block::Extrinsic::decode(&mut opaque_extrinsic.encode().as_slice())
                    .map_err(|_| FraudProofError::InvalidBundleExtrinsic {
                        bundle_index: bundle_index as usize,
                    })?;

                exts.push(extrinsic)
            }

            let bundle_digest = domain_runtime_api
                .extract_signer(local_receipt.domain_block_hash, exts)?
                .into_iter()
                .map(|(signer, ext)| {
                    (
                        signer,
                        ExtrinsicDigest::new::<LayoutV1<HeaderHashingFor<Block::Header>>>(
                            ext.encode(),
                        ),
                    )
                })
                .collect::<Vec<(Option<AccountId>, ExtrinsicDigest)>>();
            valid_bundle_digests.push(ValidBundleDigest {
                bundle_index,
                bundle_digest,
            });
        }

        Ok(valid_bundle_digests)
    }

    /// Generates and returns an invalid bundle proof.
    pub(crate) fn generate_invalid_bundle_proof(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
        mismatch_type: BundleMismatchType,
        bundle_index: u32,
        bad_receipt_hash: Block::Hash,
    ) -> Result<FraudProofFor<CBlock, Block::Header>, FraudProofError> {
        self.generate_invalid_bundle_proof_inner(
            domain_id,
            local_receipt,
            mismatch_type,
            bundle_index,
            bad_receipt_hash,
            false,
        )
    }

    /// Test-only function for generating an invalid bundle proof.
    #[cfg(test)]
    pub(crate) fn generate_invalid_bundle_proof_for_test(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
        mismatch_type: BundleMismatchType,
        bundle_index: u32,
        bad_receipt_hash: Block::Hash,
        allow_invalid_proof_in_tests: bool,
    ) -> Result<FraudProofFor<CBlock, Block::Header>, FraudProofError> {
        self.generate_invalid_bundle_proof_inner(
            domain_id,
            local_receipt,
            mismatch_type,
            bundle_index,
            bad_receipt_hash,
            allow_invalid_proof_in_tests,
        )
    }

    /// Inner function for generating invalid bundle proofs, with test support code.
    fn generate_invalid_bundle_proof_inner(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
        mismatch_type: BundleMismatchType,
        bundle_index: u32,
        bad_receipt_hash: Block::Hash,
        // Whether to generate an invalid proof against a valid ER,
        // only used in tests, ignored in production
        mut allow_invalid_proof_in_tests: bool,
    ) -> Result<FraudProofFor<CBlock, Block::Header>, FraudProofError> {
        if cfg!(not(test)) {
            allow_invalid_proof_in_tests = false;
        }

        let consensus_block_hash = local_receipt.consensus_block_hash;
        let consensus_block_number = local_receipt.consensus_block_number;

        let bundle = {
            let consensus_extrinsics = self
                .consensus_client
                .block_body(consensus_block_hash)?
                .ok_or(FraudProofError::MissingConsensusExtrinsics)?;

            let mut bundles = self
                .consensus_client
                .runtime_api()
                .extract_successful_bundles(
                    consensus_block_hash,
                    domain_id,
                    consensus_extrinsics,
                )?;
            if bundles.len() <= bundle_index as usize {
                return Err(FraudProofError::MissingBundle {
                    bundle_index: bundle_index as usize,
                });
            }

            bundles.swap_remove(bundle_index as usize)
        };

        let (invalid_type, is_good_invalid_fraud_proof) = match mismatch_type {
            BundleMismatchType::GoodInvalid(invalid_type) => (invalid_type, true),
            BundleMismatchType::BadInvalid(invalid_type) => (invalid_type, false),
            BundleMismatchType::ValidBundleContents => {
                return Err(sp_blockchain::Error::Application(
                    "Unexpected bundle mismatch type, this should not happen"
                        .to_string()
                        .into(),
                )
                .into());
            }
        };

        let proof_data = if invalid_type
            .extrinsic_index()
            .is_some_and(|idx| bundle.extrinsics.len() as u32 <= idx)
        {
            // The bad receipt claims a non-exist extrinsic is invalid, in this case, generate a
            // `bundle_with_proof` as proof data is enough
            let bundle_with_proof = OpaqueBundleWithProof::generate(
                &self.storage_key_provider,
                self.consensus_client.as_ref(),
                domain_id,
                consensus_block_hash,
                bundle,
                bundle_index,
            )?;
            InvalidBundlesProofData::Bundle(bundle_with_proof)
        } else {
            match invalid_type {
                InvalidBundleType::IllegalTx(expected_extrinsic_index) => {
                    if expected_extrinsic_index as usize >= bundle.extrinsics.len() {
                        return Err(FraudProofError::OutOfBoundsExtrinsicIndex {
                            index: expected_extrinsic_index as usize,
                            max: bundle.extrinsics.len(),
                        });
                    }

                    let domain_block_parent_hash = *self
                        .client
                        .header(local_receipt.domain_block_hash)?
                        .ok_or_else(|| {
                            FraudProofError::Blockchain(sp_blockchain::Error::MissingHeader(
                                format!("{:?}", local_receipt.domain_block_hash),
                            ))
                        })?
                        .parent_hash();

                    let domain_block_parent_number = self
                        .client
                        .block_number_from_id(&BlockId::Hash(domain_block_parent_hash))?
                        .ok_or_else(|| {
                            FraudProofError::Blockchain(sp_blockchain::Error::Backend(format!(
                                "unable to get block number for domain block:{:?}",
                                domain_block_parent_hash
                            )))
                        })?;

                    let mut runtime_api_instance = self.client.runtime_api();
                    runtime_api_instance.record_proof();
                    let proof_recorder = runtime_api_instance
                        .proof_recorder()
                        .expect("we enabled proof recording just above; qed");

                    let mut block_extrinsics = vec![];
                    for (extrinsic_index, extrinsic) in bundle
                        .extrinsics
                        .iter()
                        .enumerate()
                        .take((expected_extrinsic_index + 1) as usize)
                    {
                        let encoded_extrinsic = extrinsic.encode();
                        block_extrinsics.push(
                            Block::Extrinsic::decode(&mut encoded_extrinsic.as_slice()).map_err(
                                |decoding_error| {
                                    FraudProofError::UnableToDecodeOpaqueBundleExtrinsic {
                                        extrinsic_index,
                                        decoding_error,
                                    }
                                },
                            )?,
                        );
                    }

                    let validation_response = runtime_api_instance
                        .check_extrinsics_and_do_pre_dispatch(
                            domain_block_parent_hash,
                            block_extrinsics,
                            domain_block_parent_number,
                            domain_block_parent_hash,
                        )?;

                    // Check the validation response and extrinsic indexes:
                    // If are proving the bundle is actually invalid, the expected validation response is Err.
                    // If we are proving the invalid bundle proof is wrong, the expected validation response is Ok.
                    // If we are proving the bundle is actually invalid, the extrinsic indexes also must match.
                    if !allow_invalid_proof_in_tests
                        && ((is_good_invalid_fraud_proof == validation_response.is_ok())
                            || (is_good_invalid_fraud_proof
                                && validation_response
                                    .as_ref()
                                    .is_err_and(|e| e.extrinsic_index != expected_extrinsic_index)))
                    {
                        return Err(FraudProofError::InvalidIllegalTxFraudProofExtrinsicIndex {
                            index: expected_extrinsic_index as usize,
                            is_good_invalid_fraud_proof,
                            extrinsics_validity_response: validation_response,
                        });
                    }

                    let execution_proof = proof_recorder.drain_storage_proof();

                    let bundle_with_proof = OpaqueBundleWithProof::generate(
                        &self.storage_key_provider,
                        self.consensus_client.as_ref(),
                        domain_id,
                        consensus_block_hash,
                        bundle,
                        bundle_index,
                    )?;

                    InvalidBundlesProofData::BundleAndExecution {
                        bundle_with_proof,
                        execution_proof,
                    }
                }
                InvalidBundleType::OutOfRangeTx(_) | InvalidBundleType::InvalidBundleWeight => {
                    let bundle_with_proof = OpaqueBundleWithProof::generate(
                        &self.storage_key_provider,
                        self.consensus_client.as_ref(),
                        domain_id,
                        consensus_block_hash,
                        bundle,
                        bundle_index,
                    )?;

                    InvalidBundlesProofData::Bundle(bundle_with_proof)
                }
                InvalidBundleType::UndecodableTx(extrinsic_index)
                | InvalidBundleType::InherentExtrinsic(extrinsic_index) => {
                    let encoded_extrinsics: Vec<_> =
                        bundle.extrinsics.iter().map(Encode::encode).collect();

                    let extrinsic_proof = StorageProofProvider::<
                        LayoutV1<HeaderHashingFor<Block::Header>>,
                    >::generate_enumerated_proof_of_inclusion(
                        encoded_extrinsics.as_slice(),
                        extrinsic_index,
                    )
                    .ok_or(FraudProofError::FailToGenerateProofOfInclusion)?;

                    InvalidBundlesProofData::Extrinsic(extrinsic_proof)
                }
                InvalidBundleType::InvalidXDM(extrinsic_index) => {
                    let encoded_extrinsic = bundle.extrinsics[extrinsic_index as usize].encode();
                    let extrinsic = Block::Extrinsic::decode(&mut encoded_extrinsic.as_slice())
                        .map_err(|decoding_error| {
                            FraudProofError::UnableToDecodeOpaqueBundleExtrinsic {
                                extrinsic_index: extrinsic_index as usize,
                                decoding_error,
                            }
                        })?;

                    let maybe_xdm_mmr_proof = self
                        .client
                        .runtime_api()
                        .extract_xdm_mmr_proof(local_receipt.domain_block_hash, &extrinsic)?;

                    let mmr_root_proof = match maybe_xdm_mmr_proof {
                        // `None` this is not an XDM so not need to generate mmr root proof
                        None => None,
                        Some(xdm_mmr_proof) => {
                            let xdm_mmr_proof_constructed_at_number =
                                xdm_mmr_proof.consensus_block_number;

                            // The MMR leaf is added to the state at the next block
                            let mmr_root_state_added_at_number =
                                xdm_mmr_proof_constructed_at_number + One::one();
                            let mmr_root_state_added_at_hash = self.consensus_client.hash(mmr_root_state_added_at_number)?.ok_or_else(|| {
                                sp_blockchain::Error::Backend(format!(
                                    "Consensus block hash for #{mmr_root_state_added_at_number:?} not found",
                                ))
                            })?;

                            let mmr_proof = sc_domains::generate_mmr_proof(
                                &self.consensus_client,
                                mmr_root_state_added_at_number,
                            )?;
                            let mmr_root_storage_proof = MmrRootStorageProof::generate(
                                self.consensus_client.as_ref(),
                                mmr_root_state_added_at_hash,
                                xdm_mmr_proof_constructed_at_number,
                                &self.storage_key_provider,
                            )?;
                            Some(MmrRootProof {
                                mmr_proof,
                                mmr_root_storage_proof,
                            })
                        }
                    };

                    let extrinsic_proof = {
                        let encoded_extrinsics: Vec<_> =
                            bundle.extrinsics.iter().map(Encode::encode).collect();

                        StorageProofProvider::<
                            LayoutV1<HeaderHashingFor<Block::Header>>,
                        >::generate_enumerated_proof_of_inclusion(
                            encoded_extrinsics.as_slice(),
                            extrinsic_index,
                        )
                        .ok_or(FraudProofError::FailToGenerateProofOfInclusion)?
                    };

                    InvalidBundlesProofData::InvalidXDMProofData {
                        extrinsic_proof,
                        mmr_root_proof,
                    }
                }
            }
        };

        let mmr_proof =
            sc_domains::generate_mmr_proof(&self.consensus_client, consensus_block_number)?;

        let maybe_domain_runtime_code_proof =
            self.maybe_generate_domain_runtime_code_proof_for_receipt(domain_id, local_receipt)?;

        let invalid_bundle_proof = FraudProof {
            domain_id,
            bad_receipt_hash,
            maybe_mmr_proof: Some(mmr_proof),
            maybe_domain_runtime_code_proof,
            proof: FraudProofVariant::InvalidBundles(InvalidBundlesProof {
                bundle_index,
                invalid_bundle_type: invalid_type,
                is_good_invalid_fraud_proof,
                proof_data,
            }),
        };
        Ok(invalid_bundle_proof)
    }

    pub(crate) fn generate_invalid_block_fees_proof(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
        bad_receipt_hash: Block::Hash,
    ) -> Result<FraudProofFor<CBlock, Block::Header>, FraudProofError> {
        let block_hash = local_receipt.domain_block_hash;
        let runtime_api = self.client.runtime_api();

        let key = runtime_api.block_fees_storage_key(block_hash)?;
        let storage_proof = self
            .client
            .read_proof(block_hash, &mut [key.as_slice()].into_iter())?;

        let maybe_domain_runtime_code_proof =
            self.maybe_generate_domain_runtime_code_proof_for_receipt(domain_id, local_receipt)?;

        let invalid_block_fees_proof = FraudProof {
            domain_id,
            bad_receipt_hash,
            maybe_mmr_proof: None,
            maybe_domain_runtime_code_proof,
            proof: FraudProofVariant::InvalidBlockFees(InvalidBlockFeesProof { storage_proof }),
        };
        Ok(invalid_block_fees_proof)
    }

    pub(crate) fn generate_invalid_transfers_proof(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
        bad_receipt_hash: Block::Hash,
    ) -> Result<FraudProofFor<CBlock, Block::Header>, FraudProofError> {
        let block_hash = local_receipt.domain_block_hash;
        let runtime_api = self.client.runtime_api();
        let key = runtime_api.transfers_storage_key(block_hash)?;
        let storage_proof = self
            .client
            .read_proof(block_hash, &mut [key.as_slice()].into_iter())?;

        let maybe_domain_runtime_code_proof =
            self.maybe_generate_domain_runtime_code_proof_for_receipt(domain_id, local_receipt)?;

        let invalid_transfers_proof = FraudProof {
            domain_id,
            bad_receipt_hash,
            maybe_mmr_proof: None,
            maybe_domain_runtime_code_proof,
            proof: FraudProofVariant::InvalidTransfers(InvalidTransfersProof { storage_proof }),
        };
        Ok(invalid_transfers_proof)
    }

    pub(crate) fn generate_invalid_domain_block_hash_proof(
        &self,
        domain_id: DomainId,
        local_receipt: &ExecutionReceiptFor<Block, CBlock>,
        bad_receipt_hash: Block::Hash,
    ) -> Result<FraudProofFor<CBlock, Block::Header>, FraudProofError> {
        let block_hash = local_receipt.domain_block_hash;
        let digest_key = sp_domains::system_digest_final_key();
        let digest_storage_proof = self
            .client
            .read_proof(block_hash, &mut [digest_key.as_slice()].into_iter())?;

        let invalid_domain_block_hash = FraudProof {
            domain_id,
            bad_receipt_hash,
            maybe_mmr_proof: None,
            maybe_domain_runtime_code_proof: None,
            proof: FraudProofVariant::InvalidDomainBlockHash(InvalidDomainBlockHashProof {
                digest_storage_proof,
            }),
        };
        Ok(invalid_domain_block_hash)
    }

    fn header(&self, hash: Block::Hash) -> Result<Block::Header, sp_blockchain::Error> {
        self.client
            .header(hash)?
            .ok_or_else(|| sp_blockchain::Error::Backend(format!("Header not found for {hash:?}")))
    }

    fn consensus_header(&self, hash: CBlock::Hash) -> Result<CBlock::Header, sp_blockchain::Error> {
        self.consensus_client.header(hash)?.ok_or_else(|| {
            sp_blockchain::Error::Backend(format!("Consensus header not found for {hash:?}"))
        })
    }

    fn block_body(&self, at: Block::Hash) -> Result<Vec<Block::Extrinsic>, sp_blockchain::Error> {
        self.client.block_body(at)?.ok_or_else(|| {
            sp_blockchain::Error::Backend(format!("Block body not found for {at:?}"))
        })
    }

    fn generate_execution_phase(
        &self,
        local_receipt_domain_block_hash: Block::Hash,
        local_trace_length: usize,
        mismatch: (TraceDiffType, u32),
    ) -> Result<ExecutionPhase, FraudProofError> {
        let extrinsics = self.block_body(local_receipt_domain_block_hash)?;
        let encoded_extrinsics: Vec<_> = extrinsics.iter().map(Encode::encode).collect();

        match mismatch {
            (_, 0) => Ok(ExecutionPhase::InitializeBlock),
            (TraceDiffType::Longer, mismatch_trace_index) => Ok(ExecutionPhase::FinalizeBlock {
                mismatch: FinalizeBlockMismatch::Longer(mismatch_trace_index),
            }),
            (TraceDiffType::Mismatch, mismatch_trace_index)
                if mismatch_trace_index as usize == local_trace_length - 1 =>
            {
                Ok(ExecutionPhase::FinalizeBlock {
                    mismatch: FinalizeBlockMismatch::StateRoot,
                })
            }
            (TraceDiffType::Mismatch, mismatch_trace_index)
            | (TraceDiffType::Shorter, mismatch_trace_index) => {
                // There is a trace root of the `initialize_block` in the head of the trace so we
                // need to minus one to get the correct `extrinsic_index`
                let extrinsic_index = mismatch_trace_index as usize - 1;

                if extrinsic_index >= encoded_extrinsics.len() {
                    return Err(FraudProofError::OutOfBoundsExtrinsicIndex {
                        index: extrinsic_index,
                        max: encoded_extrinsics.len(),
                    });
                }

                let proof_of_inclusion = StorageProofProvider::<
                    LayoutV1<HashingFor<Block>>,
                >::generate_enumerated_proof_of_inclusion(
                    encoded_extrinsics.as_slice(),
                    extrinsic_index as u32,
                )
                .ok_or(FraudProofError::FailToGenerateProofOfInclusion)?;

                Ok(ExecutionPhase::ApplyExtrinsic {
                    extrinsic_proof: proof_of_inclusion,
                    mismatch: if mismatch.0 == TraceDiffType::Mismatch {
                        ApplyExtrinsicMismatch::StateRoot(mismatch_trace_index)
                    } else {
                        ApplyExtrinsicMismatch::Shorter
                    },
                })
            }
        }
    }

    /// Returns first mismatched ExecutionPhase between the receipts `local` and `other` if any.
    /// If local trace length > other trace length then we provide storage proof as usual but add a flag in fraud proof
    /// indicating that this is length mismatch, so we create a storage proof with ApplyExtrinsic execution phase
    /// and prove that this state transition is valid, that means execution should not stop here.
    ///
    /// if other trace length > local trace length then we create a storage proof with FinalizeBlock execution phase
    /// to prove that it is valid, so that means execution should stop here.
    pub(crate) fn find_mismatched_execution_phase(
        &self,
        local_receipt_domain_block_hash: Block::Hash,
        local_trace: &[Block::Hash],
        other_trace: &[Block::Hash],
    ) -> Result<Option<ExecutionPhase>, FraudProofError> {
        let state_root_mismatch = local_trace
            .iter()
            .enumerate()
            .zip(other_trace.iter().enumerate())
            .find_map(|((local_index, local_root), (_, other_root))| {
                if local_root != other_root {
                    Some((TraceDiffType::Mismatch, local_index as u32))
                } else {
                    None
                }
            });

        let mismatch = if let Some(mismatch) = state_root_mismatch {
            mismatch
        } else if local_trace.len() > other_trace.len() {
            (TraceDiffType::Shorter, (other_trace.len() - 1) as u32)
        } else if local_trace.len() < other_trace.len() {
            (TraceDiffType::Longer, (local_trace.len() - 1) as u32)
        } else {
            return Ok(None);
        };

        self.generate_execution_phase(local_receipt_domain_block_hash, local_trace.len(), mismatch)
            .map(Some)
    }
}
