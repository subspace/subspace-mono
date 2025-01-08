//! This crate provides a preprocessor for the domain block, which is used to construct
//! domain extrinsics from the consensus block.
//!
//! The workflow is as follows:
//! 1. Extract domain-specific bundles from the consensus block.
//! 2. Compile the domain bundles into a list of extrinsics.
//! 3. Shuffle the extrisnics using the seed from the consensus chain.
//! 4. Filter out the invalid xdm extrinsics.
//! 5. Push back the potential new domain runtime extrisnic.

#![warn(rust_2018_idioms)]
#![feature(let_chains)]

pub mod inherents;
pub mod stateless_runtime;

use crate::inherents::is_runtime_upgraded;
use codec::Encode;
use domain_runtime_primitives::opaque::AccountId;
use sc_client_api::BlockBackend;
use sp_api::{ApiError, Core, ProvideRuntimeApi};
use sp_blockchain::HeaderBackend;
use sp_core::H256;
use sp_domains::core_api::DomainCoreApi;
use sp_domains::extrinsics::deduplicate_and_shuffle_extrinsics;
use sp_domains::{
    DomainId, DomainsApi, ExecutionReceipt, ExtrinsicDigest, HeaderHashingFor, InboxedBundle,
    InvalidBundleType, OpaqueBundle, OpaqueBundles, ReceiptValidity,
};
use sp_messenger::MessengerApi;
use sp_mmr_primitives::MmrApi;
use sp_runtime::traits::{Block as BlockT, Hash as HashT, Header, NumberFor};
use sp_runtime::DigestItem;
use sp_state_machine::LayoutV1;
use sp_subspace_mmr::ConsensusChainMmrLeafProof;
use sp_weights::Weight;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;
use subspace_core_primitives::{Randomness, U256};
use subspace_runtime_primitives::Balance;

type DomainBlockElements<CBlock> = (Vec<<CBlock as BlockT>::Extrinsic>, Randomness);

enum BundleValidity<Extrinsic> {
    Valid(Vec<Extrinsic>),
    Invalid(InvalidBundleType),
}

/// Extracts the raw materials for building a new domain block from the primary block.
fn prepare_domain_block_elements<Block, CBlock, CClient>(
    consensus_client: &CClient,
    block_hash: CBlock::Hash,
) -> sp_blockchain::Result<DomainBlockElements<CBlock>>
where
    Block: BlockT,
    CBlock: BlockT,
    CClient: HeaderBackend<CBlock> + BlockBackend<CBlock> + ProvideRuntimeApi<CBlock> + Send + Sync,
    CClient::Api: DomainsApi<CBlock, Block::Header>,
{
    let extrinsics = consensus_client.block_body(block_hash)?.ok_or_else(|| {
        sp_blockchain::Error::Backend(format!("BlockBody of {block_hash:?} unavailable"))
    })?;

    let shuffling_seed = consensus_client
        .runtime_api()
        .extrinsics_shuffling_seed(block_hash)?;

    Ok((extrinsics, shuffling_seed))
}

pub struct PreprocessResult<Block: BlockT> {
    pub extrinsics: VecDeque<Block::Extrinsic>,
    pub bundles: Vec<InboxedBundle<Block::Hash>>,
}

pub struct DomainBlockPreprocessor<Block, CBlock, Client, CClient, ReceiptValidator> {
    domain_id: DomainId,
    client: Arc<Client>,
    consensus_client: Arc<CClient>,
    receipt_validator: ReceiptValidator,
    _phantom_data: PhantomData<(Block, CBlock)>,
}

impl<Block, CBlock, Client, CClient, ReceiptValidator: Clone> Clone
    for DomainBlockPreprocessor<Block, CBlock, Client, CClient, ReceiptValidator>
{
    fn clone(&self) -> Self {
        Self {
            domain_id: self.domain_id,
            client: self.client.clone(),
            consensus_client: self.consensus_client.clone(),
            receipt_validator: self.receipt_validator.clone(),
            _phantom_data: self._phantom_data,
        }
    }
}

pub trait ValidateReceipt<Block, CBlock>
where
    Block: BlockT,
    CBlock: BlockT,
{
    fn validate_receipt(
        &self,
        receipt: &ExecutionReceipt<
            NumberFor<CBlock>,
            CBlock::Hash,
            NumberFor<Block>,
            Block::Hash,
            Balance,
        >,
    ) -> sp_blockchain::Result<ReceiptValidity>;
}

impl<Block, CBlock, Client, CClient, ReceiptValidator>
    DomainBlockPreprocessor<Block, CBlock, Client, CClient, ReceiptValidator>
where
    Block: BlockT,
    Block::Hash: Into<H256>,
    CBlock: BlockT,
    CBlock::Hash: From<Block::Hash>,
    NumberFor<CBlock>: From<NumberFor<Block>>,
    Client: HeaderBackend<Block> + ProvideRuntimeApi<Block> + 'static,
    Client::Api: DomainCoreApi<Block> + MessengerApi<Block, NumberFor<CBlock>, CBlock::Hash>,
    CClient: HeaderBackend<CBlock>
        + BlockBackend<CBlock>
        + ProvideRuntimeApi<CBlock>
        + Send
        + Sync
        + 'static,
    CClient::Api: DomainsApi<CBlock, Block::Header>
        + MessengerApi<CBlock, NumberFor<CBlock>, CBlock::Hash>
        + MmrApi<CBlock, H256, NumberFor<CBlock>>,
    ReceiptValidator: ValidateReceipt<Block, CBlock>,
{
    pub fn new(
        domain_id: DomainId,
        client: Arc<Client>,
        consensus_client: Arc<CClient>,
        receipt_validator: ReceiptValidator,
    ) -> Self {
        Self {
            domain_id,
            client,
            consensus_client,
            receipt_validator,
            _phantom_data: Default::default(),
        }
    }

    pub fn preprocess_consensus_block(
        &self,
        consensus_block_hash: CBlock::Hash,
        parent_domain_block: (Block::Hash, NumberFor<Block>),
    ) -> sp_blockchain::Result<Option<PreprocessResult<Block>>> {
        let (primary_extrinsics, shuffling_seed) = prepare_domain_block_elements::<Block, CBlock, _>(
            &*self.consensus_client,
            consensus_block_hash,
        )?;

        let bundles = self
            .consensus_client
            .runtime_api()
            .extract_successful_bundles(consensus_block_hash, self.domain_id, primary_extrinsics)?;

        if bundles.is_empty()
            && !is_runtime_upgraded::<_, _, Block>(
                &self.consensus_client,
                consensus_block_hash,
                self.domain_id,
            )?
        {
            return Ok(None);
        }

        let tx_range = self
            .consensus_client
            .runtime_api()
            .domain_tx_range(consensus_block_hash, self.domain_id)?;

        let (inboxed_bundles, extrinsics) = self.compile_bundles_to_extrinsics(
            bundles,
            tx_range,
            parent_domain_block,
            consensus_block_hash,
        )?;

        let extrinsics =
            deduplicate_and_shuffle_extrinsics::<Block::Extrinsic>(extrinsics, shuffling_seed);

        Ok(Some(PreprocessResult {
            extrinsics,
            bundles: inboxed_bundles,
        }))
    }

    /// Filter out the invalid bundles first and then convert the remaining valid ones to
    /// a list of extrinsics.
    #[allow(clippy::type_complexity)]
    fn compile_bundles_to_extrinsics(
        &self,
        bundles: OpaqueBundles<CBlock, Block::Header, Balance>,
        tx_range: U256,
        (parent_domain_hash, parent_domain_number): (Block::Hash, NumberFor<Block>),
        at_consensus_hash: CBlock::Hash,
    ) -> sp_blockchain::Result<(
        Vec<InboxedBundle<Block::Hash>>,
        Vec<(Option<AccountId>, Block::Extrinsic)>,
    )> {
        let consensus_spec_version = self
            .consensus_client
            .runtime_api()
            .version(at_consensus_hash)
            .map_err(sp_blockchain::Error::RuntimeApiError)?
            .spec_version;

        let mut inboxed_bundles = Vec::with_capacity(bundles.len());
        let mut valid_extrinsics = Vec::new();

        let runtime_api = self.client.runtime_api();
        for bundle in bundles {
            // For the honest operator the validity of the extrinsic of the bundle is committed
            // to (or say verified against) the receipt that is submitted with the bundle, the
            // consensus runtime should only accept the bundle if the receipt is derived from
            // the parent domain block. If it is not then either there is a bug in the consensus
            // runtime (for validating the bundle) or in the domain client (for finding the parent
            // domain block).
            //
            // NOTE: The receipt's `domain_block_number` is verified by the consensus runtime while
            // the `domain_block_hash` is not (which is take care of by the fraud proof) so we can't
            // check the parent domain block hash here.
            // TODO: remove consensus runtime version check before next network
            if consensus_spec_version >= 6
                && bundle.receipt().domain_block_number != parent_domain_number
            {
                // If there consensus runtime just upgraded to spec version 6, which bring the receipt
                // gap check, the bundle that included in the same block is doesn't perform the check
                // because the new runtime take effect in the next block so skip the check here too.
                let is_consensus_runtime_upgraded_to_6 = {
                    let consensus_header = self
                        .consensus_client
                        .header(at_consensus_hash)?
                        .ok_or_else(|| {
                            sp_blockchain::Error::Backend(format!(
                                "Consensus block header of {at_consensus_hash:?} unavailable"
                            ))
                        })?;

                    let runtime_upgraded = consensus_header
                        .digest()
                        .logs()
                        .iter()
                        .any(|di| di == &DigestItem::RuntimeEnvironmentUpdated);

                    runtime_upgraded && consensus_spec_version == 6
                };
                if !is_consensus_runtime_upgraded_to_6 {
                    return Err(sp_blockchain::Error::RuntimeApiError(
                        ApiError::Application(
                            format!(
                                "Unexpected bundle in consensus block: {:?}, something must be wrong",
                                at_consensus_hash
                            )
                            .into(),
                        ),
                    ));
                }
            }

            let extrinsic_root = bundle.extrinsics_root();
            match self.check_bundle_validity(
                &bundle,
                &tx_range,
                (parent_domain_hash, parent_domain_number),
                at_consensus_hash,
            )? {
                BundleValidity::Valid(extrinsics) => {
                    let extrinsics: Vec<_> = match runtime_api
                        .extract_signer(parent_domain_hash, extrinsics)
                    {
                        Ok(res) => res,
                        Err(e) => {
                            tracing::error!(error = ?e, "Error at calling runtime api: extract_signer");
                            return Err(e.into());
                        }
                    };
                    let bundle_digest: Vec<_> = extrinsics
                        .iter()
                        .map(|(signer, tx)| {
                            (
                                signer.clone(),
                                ExtrinsicDigest::new::<LayoutV1<HeaderHashingFor<Block::Header>>>(
                                    tx.encode(),
                                ),
                            )
                        })
                        .collect();
                    inboxed_bundles.push(InboxedBundle::valid(
                        HeaderHashingFor::<Block::Header>::hash_of(&bundle_digest),
                        extrinsic_root,
                    ));
                    valid_extrinsics.extend(extrinsics);
                }
                BundleValidity::Invalid(invalid_bundle_type) => {
                    inboxed_bundles
                        .push(InboxedBundle::invalid(invalid_bundle_type, extrinsic_root));
                }
            }
        }

        Ok((inboxed_bundles, valid_extrinsics))
    }

    fn check_bundle_validity(
        &self,
        bundle: &OpaqueBundle<NumberFor<CBlock>, CBlock::Hash, Block::Header, Balance>,
        tx_range: &U256,
        (parent_domain_hash, parent_domain_number): (Block::Hash, NumberFor<Block>),
        at_consensus_hash: CBlock::Hash,
    ) -> sp_blockchain::Result<BundleValidity<Block::Extrinsic>> {
        let bundle_vrf_hash =
            U256::from_be_bytes(*bundle.sealed_header.header.proof_of_election.vrf_hash());

        let mut extrinsics = Vec::with_capacity(bundle.extrinsics.len());
        let mut estimated_bundle_weight = Weight::default();

        let runtime_api = self.client.runtime_api();
        let consensus_runtime_api = self.consensus_client.runtime_api();

        // Check the validity of each extrinsic
        //
        // NOTE: for each extrinsic the checking order must follow `InvalidBundleType::checking_order`
        for (index, opaque_extrinsic) in bundle.extrinsics.iter().enumerate() {
            let decode_result =
                runtime_api.decode_extrinsic(parent_domain_hash, opaque_extrinsic.clone())?;
            let extrinsic = match decode_result {
                Ok(extrinsic) => extrinsic,
                Err(err) => {
                    tracing::error!(
                        ?opaque_extrinsic,
                        ?err,
                        "Undecodable extrinsic in bundle({})",
                        bundle.hash()
                    );
                    return Ok(BundleValidity::Invalid(InvalidBundleType::UndecodableTx(
                        index as u32,
                    )));
                }
            };

            let is_within_tx_range = runtime_api.is_within_tx_range(
                parent_domain_hash,
                &extrinsic,
                &bundle_vrf_hash,
                tx_range,
            )?;

            if !is_within_tx_range {
                return Ok(BundleValidity::Invalid(InvalidBundleType::OutOfRangeTx(
                    index as u32,
                )));
            }

            // Check if this extrinsic is an inherent extrinsic.
            // If so, this is an invalid bundle since these extrinsics should not be included in the
            // bundle. Extrinsic is always decodable due to the check above.
            if runtime_api.is_inherent_extrinsic(parent_domain_hash, &extrinsic)? {
                return Ok(BundleValidity::Invalid(
                    InvalidBundleType::InherentExtrinsic(index as u32),
                ));
            }

            if let Some(xdm_mmr_proof) =
                runtime_api.extract_xdm_mmr_proof(parent_domain_hash, &extrinsic)?
            {
                let ConsensusChainMmrLeafProof {
                    opaque_mmr_leaf,
                    proof,
                    ..
                } = xdm_mmr_proof;

                if consensus_runtime_api
                    .verify_proof(at_consensus_hash, vec![opaque_mmr_leaf], proof)?
                    .is_err()
                {
                    return Ok(BundleValidity::Invalid(InvalidBundleType::InvalidXDM(
                        index as u32,
                    )));
                }
            }

            // Using one instance of runtime_api throughout the loop in order to maintain context
            // between them.
            // Using `check_extrinsics_and_do_pre_dispatch` instead of `check_transaction_validity`
            // to maintain side-effect in the storage buffer.
            // let is_legal_tx = runtime_api
            //     .check_extrinsics_and_do_pre_dispatch(
            //         parent_domain_hash,
            //         vec![extrinsic.clone()],
            //         parent_domain_number,
            //         parent_domain_hash,
            //     )?
            //     .is_ok();

            // if !is_legal_tx {
            //     return Ok(BundleValidity::Invalid(InvalidBundleType::IllegalTx(
            //         index as u32,
            //     )));
            // }

            let tx_weight = runtime_api.extrinsic_weight(parent_domain_hash, &extrinsic)?;
            estimated_bundle_weight = estimated_bundle_weight.saturating_add(tx_weight);

            extrinsics.push(extrinsic);
        }

        if estimated_bundle_weight != bundle.estimated_weight() {
            return Ok(BundleValidity::Invalid(
                InvalidBundleType::InvalidBundleWeight,
            ));
        }

        Ok(BundleValidity::Valid(extrinsics))
    }
}

#[cfg(test)]
mod tests {
    use sp_domains::extrinsics::shuffle_extrinsics;
    use sp_keyring::sr25519::Keyring;
    use sp_runtime::traits::{BlakeTwo256, Hash as HashT};
    use subspace_core_primitives::Randomness;

    #[test]
    fn shuffle_extrinsics_should_work() {
        let alice = Keyring::Alice.to_account_id();
        let bob = Keyring::Bob.to_account_id();
        let charlie = Keyring::Charlie.to_account_id();

        let extrinsics = vec![
            (Some(alice.clone()), 10),
            (None, 100),
            (Some(bob.clone()), 1),
            (Some(bob), 2),
            (Some(charlie.clone()), 30),
            (Some(alice.clone()), 11),
            (Some(charlie), 31),
            (None, 101),
            (None, 102),
            (Some(alice), 12),
        ];

        let dummy_seed = Randomness::from(BlakeTwo256::hash_of(&[1u8; 64]).to_fixed_bytes());
        let shuffled_extrinsics = shuffle_extrinsics(extrinsics, dummy_seed);

        assert_eq!(
            shuffled_extrinsics,
            vec![100, 30, 10, 1, 11, 101, 31, 12, 102, 2]
        );
    }
}
