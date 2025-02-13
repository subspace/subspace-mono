#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::{
    DomainInherentExtrinsic, DomainInherentExtrinsicData, DomainStorageKeyRequest,
    StatelessDomainRuntimeCall,
};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use codec::{Codec, Decode, Encode};
use domain_block_preprocessor::stateless_runtime::StatelessRuntime;
use domain_runtime_primitives::{
    BlockNumber, CheckExtrinsicsValidityError, CHECK_EXTRINSICS_AND_DO_PRE_DISPATCH_METHOD_NAME,
};
use hash_db::{HashDB, Hasher};
use sc_client_api::execution_extensions::ExtensionsFactory;
use sc_client_api::BlockBackend;
use sc_executor::RuntimeVersionOf;
use sp_api::ProvideRuntimeApi;
use sp_blockchain::HeaderBackend;
use sp_core::traits::{CallContext, CodeExecutor, FetchRuntimeCode, RuntimeCode};
use sp_core::H256;
use sp_domains::{BundleProducerElectionApi, DomainsApi, ExtrinsicDigest};
use sp_externalities::Extensions;
use sp_messenger::MessengerApi;
use sp_runtime::traits::{Block as BlockT, Hash as HashT, HashingFor, NumberFor};
use sp_runtime::OpaqueExtrinsic;
use sp_state_machine::{LayoutV1, OverlayedChanges, StateMachine, TrieBackend, TrieBackendBuilder};
use sp_trie::{MemoryDB, StorageProof};
use sp_weights::Weight;
use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;
use subspace_runtime_primitives::Balance;

struct DomainRuntimeCodeFetcher(Vec<u8>);

impl FetchRuntimeCode for DomainRuntimeCodeFetcher {
    fn fetch_runtime_code(&self) -> Option<Cow<[u8]>> {
        Some(Cow::Borrowed(self.0.as_ref()))
    }
}

/// Trait to query and verify Domains Fraud proof.
pub trait FraudProofHostFunctions: Send + Sync {
    /// Derive the bundle digest for the given bundle body.
    fn derive_bundle_digest(
        &self,
        domain_runtime_code: Vec<u8>,
        bundle_body: Vec<OpaqueExtrinsic>,
    ) -> Option<H256>;

    /// Check the execution proof
    fn execution_proof_check(
        &self,
        domain_block_id: (BlockNumber, H256),
        pre_state_root: H256,
        // TODO: implement `PassBy` for `sp_trie::StorageProof` in upstream to pass it directly here
        encoded_proof: Vec<u8>,
        execution_method: &str,
        call_data: &[u8],
        domain_runtime_code: Vec<u8>,
    ) -> Option<Vec<u8>>;

    fn check_extrinsics_in_single_context(
        &self,
        domain_runtime_code: Vec<u8>,
        domain_block_id: (BlockNumber, H256),
        domain_block_state_root: H256,
        bundle_extrinsics: Vec<OpaqueExtrinsic>,
        // TODO: implement `PassBy` for `sp_trie::StorageProof` in upstream to pass it directly here
        encoded_proof: Vec<u8>,
    ) -> Option<Option<u32>>;

    fn construct_domain_inherent_extrinsic(
        &self,
        domain_runtime_code: Vec<u8>,
        domain_inherent_extrinsic_data: DomainInherentExtrinsicData,
    ) -> Option<DomainInherentExtrinsic>;

    fn domain_storage_key(
        &self,
        domain_runtime_code: Vec<u8>,
        req: DomainStorageKeyRequest,
    ) -> Option<Vec<u8>>;

    fn domain_runtime_call(
        &self,
        domain_runtime_code: Vec<u8>,
        call: StatelessDomainRuntimeCall,
    ) -> Option<bool>;

    fn bundle_weight(
        &self,
        domain_runtime_code: Vec<u8>,
        bundle_body: Vec<OpaqueExtrinsic>,
    ) -> Option<Weight>;

    fn extract_xdm_mmr_proof(
        &self,
        domain_runtime_code: Vec<u8>,
        opaque_extrinsic: Vec<u8>,
    ) -> Option<Option<Vec<u8>>>;
}

sp_externalities::decl_extension! {
    /// Domains fraud proof host function
    pub struct FraudProofExtension(std::sync::Arc<dyn FraudProofHostFunctions>);
}

impl FraudProofExtension {
    /// Create a new instance of [`FraudProofExtension`].
    pub fn new(inner: std::sync::Arc<dyn FraudProofHostFunctions>) -> Self {
        Self(inner)
    }
}

/// Trait Impl to query and verify Domains Fraud proof.
pub struct FraudProofHostFunctionsImpl<Block, Client, DomainBlock, Executor, EFC> {
    consensus_client: Arc<Client>,
    domain_executor: Arc<Executor>,
    domain_extensions_factory_creator: EFC,
    _phantom: PhantomData<(Block, DomainBlock)>,
}

impl<Block, Client, DomainBlock, Executor, EFC>
    FraudProofHostFunctionsImpl<Block, Client, DomainBlock, Executor, EFC>
{
    pub fn new(
        consensus_client: Arc<Client>,
        domain_executor: Arc<Executor>,
        domain_extensions_factory_creator: EFC,
    ) -> Self {
        FraudProofHostFunctionsImpl {
            consensus_client,
            domain_executor,
            domain_extensions_factory_creator,
            _phantom: Default::default(),
        }
    }
}

impl<Block, Client, DomainBlock, Executor, EFC> FraudProofHostFunctions
    for FraudProofHostFunctionsImpl<Block, Client, DomainBlock, Executor, EFC>
where
    Block: BlockT,
    Block::Hash: From<H256>,
    DomainBlock: BlockT,
    DomainBlock::Hash: From<H256> + Into<H256>,
    NumberFor<DomainBlock>: From<BlockNumber>,
    Client: BlockBackend<Block> + HeaderBackend<Block> + ProvideRuntimeApi<Block>,
    Client::Api: DomainsApi<Block, DomainBlock::Header>
        + BundleProducerElectionApi<Block, Balance>
        + MessengerApi<Block, NumberFor<Block>, Block::Hash>,
    Executor: CodeExecutor + RuntimeVersionOf,
    EFC: Fn(Arc<Client>, Arc<Executor>) -> Box<dyn ExtensionsFactory<DomainBlock>> + Send + Sync,
{
    fn derive_bundle_digest(
        &self,
        domain_runtime_code: Vec<u8>,
        bundle_body: Vec<OpaqueExtrinsic>,
    ) -> Option<H256> {
        let mut extrinsics = Vec::with_capacity(bundle_body.len());
        for opaque_extrinsic in bundle_body {
            let ext =
                DomainBlock::Extrinsic::decode(&mut opaque_extrinsic.encode().as_slice()).ok()?;
            extrinsics.push(ext);
        }

        let domain_stateless_runtime = StatelessRuntime::<Block, DomainBlock, _>::new(
            self.domain_executor.clone(),
            domain_runtime_code.into(),
        );

        let ext_signers: Vec<_> = domain_stateless_runtime
            .extract_signer(extrinsics)
            .ok()?
            .into_iter()
            .map(|(signer, tx)| {
                (
                    signer,
                    ExtrinsicDigest::new::<LayoutV1<HashingFor<DomainBlock>>>(tx.encode()),
                )
            })
            .collect();

        Some(HashingFor::<DomainBlock>::hash_of(&ext_signers).into())
    }

    fn execution_proof_check(
        &self,
        domain_block_id: (BlockNumber, H256),
        pre_state_root: H256,
        encoded_proof: Vec<u8>,
        execution_method: &str,
        call_data: &[u8],
        domain_runtime_code: Vec<u8>,
    ) -> Option<Vec<u8>> {
        let proof: StorageProof = Decode::decode(&mut encoded_proof.as_ref()).ok()?;

        let domain_runtime_code_fetcher = DomainRuntimeCodeFetcher(domain_runtime_code);
        let runtime_code = RuntimeCode {
            code_fetcher: &domain_runtime_code_fetcher,
            hash: b"Hash of the code does not matter in terms of the execution proof check"
                .to_vec(),
            heap_pages: None,
        };

        let (domain_block_number, domain_block_hash) = domain_block_id;
        let mut domain_extensions = (self.domain_extensions_factory_creator)(
            self.consensus_client.clone(),
            self.domain_executor.clone(),
        )
        .extensions_for(domain_block_hash.into(), domain_block_number.into());

        execution_proof_check::<HashingFor<DomainBlock>, _>(
            pre_state_root.into(),
            proof,
            &mut Default::default(),
            self.domain_executor.as_ref(),
            execution_method,
            call_data,
            &runtime_code,
            &mut domain_extensions,
        )
        .ok()
    }

    fn check_extrinsics_in_single_context(
        &self,
        domain_runtime_code: Vec<u8>,
        domain_block_id: (BlockNumber, H256),
        domain_block_state_root: H256,
        bundle_extrinsics: Vec<OpaqueExtrinsic>,
        encoded_proof: Vec<u8>,
    ) -> Option<Option<u32>> {
        let storage_proof: StorageProof = Decode::decode(&mut encoded_proof.as_ref()).ok()?;
        let (domain_block_number, domain_block_hash) = domain_block_id;

        let raw_response = self.execution_proof_check(
            domain_block_id,
            domain_block_state_root,
            storage_proof.encode(),
            CHECK_EXTRINSICS_AND_DO_PRE_DISPATCH_METHOD_NAME,
            // The call data must be encoded form of arguments to `DomainCoreApi::check_extrinsic_and_do_pre_dispatch`
            &(&bundle_extrinsics, &domain_block_number, &domain_block_hash).encode(),
            domain_runtime_code,
        )?;

        let bundle_extrinsics_validity_response: Result<(), CheckExtrinsicsValidityError> =
            Decode::decode(&mut raw_response.as_slice()).ok()?;
        if let Err(bundle_extrinsic_validity_error) = bundle_extrinsics_validity_response {
            Some(Some(bundle_extrinsic_validity_error.extrinsic_index))
        } else {
            Some(None)
        }
    }

    fn construct_domain_inherent_extrinsic(
        &self,
        domain_runtime_code: Vec<u8>,
        domain_inherent_extrinsic_data: DomainInherentExtrinsicData,
    ) -> Option<DomainInherentExtrinsic> {
        let DomainInherentExtrinsicData {
            timestamp,
            maybe_domain_runtime_upgrade,
            consensus_transaction_byte_fee,
            domain_chain_allowlist,
            maybe_sudo_runtime_call,
            maybe_evm_domain_contract_creation_allowed_by_call,
        } = domain_inherent_extrinsic_data;

        let domain_stateless_runtime = StatelessRuntime::<Block, DomainBlock, _>::new(
            self.domain_executor.clone(),
            domain_runtime_code.into(),
        );

        let domain_timestamp_extrinsic = domain_stateless_runtime
            .construct_timestamp_extrinsic(timestamp)
            .ok()
            .map(|ext| ext.encode())?;

        let consensus_chain_byte_fee_extrinsic = domain_stateless_runtime
            .construct_consensus_chain_byte_fee_extrinsic(consensus_transaction_byte_fee)
            .ok()
            .map(|ext| ext.encode())?;

        let maybe_domain_chain_allowlist_extrinsic = if !domain_chain_allowlist.is_empty() {
            Some(
                domain_stateless_runtime
                    .construct_domain_update_chain_allowlist_extrinsic(domain_chain_allowlist)
                    .ok()
                    .map(|ext| ext.encode())?,
            )
        } else {
            None
        };

        let maybe_domain_set_code_extrinsic = match maybe_domain_runtime_upgrade {
            None => None,
            Some(upgraded_runtime_code) => Some(
                domain_stateless_runtime
                    .construct_set_code_extrinsic(upgraded_runtime_code)
                    .ok()
                    .map(|ext| ext.encode())?,
            ),
        };

        let maybe_domain_sudo_call_extrinsic = match maybe_sudo_runtime_call {
            None => None,
            Some(call) => Some(
                domain_stateless_runtime
                    .construct_domain_sudo_extrinsic(call)
                    .ok()
                    .map(|call| call.encode())?,
            ),
        };

        let maybe_evm_domain_contract_creation_allowed_by_call_extrinsic =
            match maybe_evm_domain_contract_creation_allowed_by_call {
                None => None,
                Some(call) => Some(
                    domain_stateless_runtime
                        .construct_evm_contract_creation_allowed_by_extrinsic(call)
                        .ok()
                        .map(|call| call.encode())?,
                ),
            };

        Some(DomainInherentExtrinsic {
            domain_timestamp_extrinsic,
            maybe_domain_chain_allowlist_extrinsic,
            consensus_chain_byte_fee_extrinsic,
            maybe_domain_set_code_extrinsic,
            maybe_domain_sudo_call_extrinsic,
            maybe_evm_domain_contract_creation_allowed_by_call_extrinsic,
        })
    }

    fn domain_storage_key(
        &self,
        domain_runtime_code: Vec<u8>,
        req: DomainStorageKeyRequest,
    ) -> Option<Vec<u8>> {
        let domain_stateless_runtime = StatelessRuntime::<Block, DomainBlock, _>::new(
            self.domain_executor.clone(),
            domain_runtime_code.into(),
        );
        let key = match req {
            DomainStorageKeyRequest::BlockFees => domain_stateless_runtime.block_fees_storage_key(),
            DomainStorageKeyRequest::Transfers => domain_stateless_runtime.transfers_storage_key(),
        }
        .ok()?;
        Some(key)
    }

    fn domain_runtime_call(
        &self,
        domain_runtime_code: Vec<u8>,
        call: StatelessDomainRuntimeCall,
    ) -> Option<bool> {
        let domain_stateless_runtime = StatelessRuntime::<Block, DomainBlock, _>::new(
            self.domain_executor.clone(),
            domain_runtime_code.into(),
        );

        match call {
            StatelessDomainRuntimeCall::IsTxInRange {
                opaque_extrinsic,
                bundle_vrf_hash,
                domain_tx_range,
            } => {
                let encoded_extrinsic = opaque_extrinsic.encode();
                let extrinsic =
                    DomainBlock::Extrinsic::decode(&mut encoded_extrinsic.as_slice()).ok()?;
                domain_stateless_runtime
                    .is_within_tx_range(&extrinsic, &bundle_vrf_hash, &domain_tx_range)
                    .ok()
            }
            StatelessDomainRuntimeCall::IsInherentExtrinsic(opaque_extrinsic) => {
                let encoded_extrinsic = opaque_extrinsic.encode();
                let extrinsic =
                    DomainBlock::Extrinsic::decode(&mut encoded_extrinsic.as_slice()).ok()?;
                domain_stateless_runtime
                    .is_inherent_extrinsic(&extrinsic)
                    .ok()
            }
            StatelessDomainRuntimeCall::IsDecodableExtrinsic(opaque_extrinsic) => Some(matches!(
                domain_stateless_runtime.decode_extrinsic(opaque_extrinsic),
                Ok(Ok(_))
            )),
            StatelessDomainRuntimeCall::IsValidDomainSudoCall(encoded_extrinsic) => {
                domain_stateless_runtime
                    .is_valid_sudo_call(encoded_extrinsic)
                    .ok()
            }
            // Check if the EVM domain has been upgraded with the relevant runtime calls,
            // and that EVM runtime implementation accepts this allow list value.
            StatelessDomainRuntimeCall::IsValidEvmDomainContractCreationAllowedByCall(
                decoded_argument,
            ) => domain_stateless_runtime
                .is_valid_evm_contract_creation_allowed_by_call(decoded_argument)
                .ok(),
        }
    }

    fn bundle_weight(
        &self,
        domain_runtime_code: Vec<u8>,
        bundle_body: Vec<OpaqueExtrinsic>,
    ) -> Option<Weight> {
        let domain_stateless_runtime = StatelessRuntime::<Block, DomainBlock, _>::new(
            self.domain_executor.clone(),
            domain_runtime_code.into(),
        );
        let mut estimated_bundle_weight = Weight::default();
        for opaque_extrinsic in bundle_body {
            let encoded_extrinsic = opaque_extrinsic.encode();
            let extrinsic =
                DomainBlock::Extrinsic::decode(&mut encoded_extrinsic.as_slice()).ok()?;
            let tx_weight = domain_stateless_runtime.extrinsic_weight(&extrinsic).ok()?;
            estimated_bundle_weight = estimated_bundle_weight.saturating_add(tx_weight);
        }
        Some(estimated_bundle_weight)
    }

    fn extract_xdm_mmr_proof(
        &self,
        domain_runtime_code: Vec<u8>,
        opaque_extrinsic: Vec<u8>,
    ) -> Option<Option<Vec<u8>>> {
        let domain_stateless_runtime = StatelessRuntime::<Block, DomainBlock, _>::new(
            self.domain_executor.clone(),
            domain_runtime_code.into(),
        );
        let extrinsic = DomainBlock::Extrinsic::decode(&mut opaque_extrinsic.as_slice()).ok()?;
        domain_stateless_runtime
            .extract_xdm_mmr_proof(&extrinsic)
            .ok()
    }
}

type CreateProofCheckBackedResult<H> = Result<
    (TrieBackend<MemoryDB<H>, H>, sp_trie::recorder::Recorder<H>),
    Box<dyn sp_state_machine::Error>,
>;

/// Creates a memory db backed with a recorder.
/// Fork of `sp_state_machine::create_proof_check_backend` with recorder enabled.
fn create_proof_check_backend_with_recorder<H>(
    root: H::Out,
    proof: StorageProof,
) -> CreateProofCheckBackedResult<H>
where
    H: Hasher,
    H::Out: Codec,
{
    let db = proof.into_memory_db();
    let recorder = sp_trie::recorder::Recorder::<H>::default();

    if db.contains(&root, hash_db::EMPTY_PREFIX) {
        let backend = TrieBackendBuilder::new(db, root)
            .with_recorder(recorder.clone())
            .build();
        Ok((backend, recorder))
    } else {
        Err(Box::new(sp_state_machine::ExecutionError::InvalidProof))
    }
}

/// Execution Proof check error.
#[derive(Debug)]
pub(crate) enum ExecutionProofCheckError {
    /// Holds the actual execution proof error
    // While we don't read the error, we'd still like it to be present in case we need it later
    #[allow(dead_code)]
    ExecutionError(Box<dyn sp_state_machine::Error>),
    /// Error when storage proof contains unused node keys.
    UnusedNodes,
}

impl From<Box<dyn sp_state_machine::Error>> for ExecutionProofCheckError {
    fn from(value: Box<dyn sp_state_machine::Error>) -> Self {
        Self::ExecutionError(value)
    }
}

#[allow(clippy::too_many_arguments)]
/// Executes the given proof using the runtime
/// The only difference between sp_state_machine::execution_proof_check is Extensions
pub(crate) fn execution_proof_check<H, Exec>(
    root: H::Out,
    proof: StorageProof,
    overlay: &mut OverlayedChanges<H>,
    exec: &Exec,
    method: &str,
    call_data: &[u8],
    runtime_code: &RuntimeCode,
    extensions: &mut Extensions,
) -> Result<Vec<u8>, ExecutionProofCheckError>
where
    H: Hasher,
    H::Out: Ord + 'static + Codec,
    Exec: CodeExecutor + Clone + 'static,
{
    let expected_nodes_to_be_read = proof.iter_nodes().count();
    let (trie_backend, recorder) = create_proof_check_backend_with_recorder::<H>(root, proof)?;
    let result = StateMachine::<_, H, Exec>::new(
        &trie_backend,
        overlay,
        exec,
        method,
        call_data,
        extensions,
        runtime_code,
        CallContext::Offchain,
    )
    .execute()?;

    let recorded_proof = recorder.drain_storage_proof();
    if recorded_proof.iter_nodes().count() != expected_nodes_to_be_read {
        return Err(ExecutionProofCheckError::UnusedNodes);
    }

    Ok(result)
}
