use crate::aux_schema::BundleMismatchType;
use crate::fraud_proof::FraudProofGenerator;
use crate::utils::{DomainBlockImportNotification, DomainImportNotificationSinks};
use crate::ExecutionReceiptFor;
use codec::Encode;
use domain_block_builder::{BlockBuilder, BuiltBlock, CollectedStorageChanges};
use domain_block_preprocessor::inherents::get_inherent_data;
use domain_block_preprocessor::PreprocessResult;
use sc_client_api::{AuxStore, BlockBackend, ExecutorProvider, Finalizer, ProofProvider};
use sc_consensus::{
    BlockImportParams, BoxBlockImport, ForkChoiceStrategy, ImportResult, StateAction,
    StorageChanges,
};
use sc_transaction_pool_api::OffchainTransactionPoolFactory;
use sp_api::{ApiExt, ProvideRuntimeApi};
use sp_blockchain::{HashAndNumber, HeaderBackend, HeaderMetadata};
use sp_consensus::{BlockOrigin, SyncOracle};
use sp_core::traits::CodeExecutor;
use sp_core::H256;
use sp_domains::core_api::DomainCoreApi;
use sp_domains::merkle_tree::MerkleTree;
use sp_domains::{BundleValidity, DomainId, DomainsApi, ExecutionReceipt, HeaderHashingFor};
use sp_domains_fraud_proof::fraud_proof::FraudProof;
use sp_domains_fraud_proof::FraudProofApi;
use sp_messenger::MessengerApi;
use sp_mmr_primitives::MmrApi;
use sp_runtime::traits::{Block as BlockT, Header as HeaderT, NumberFor, One, Zero};
use sp_runtime::{Digest, Saturating};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::sync::Arc;

struct DomainBlockBuildResult<Block>
where
    Block: BlockT,
{
    header_number: NumberFor<Block>,
    header_hash: Block::Hash,
    state_root: Block::Hash,
    extrinsics_root: Block::Hash,
    intermediate_roots: Vec<Block::Hash>,
}

pub(crate) struct DomainBlockResult<Block, CBlock>
where
    Block: BlockT,
    CBlock: BlockT,
{
    pub header_hash: Block::Hash,
    pub header_number: NumberFor<Block>,
    pub execution_receipt: ExecutionReceiptFor<Block, CBlock>,
}

/// An abstracted domain block processor.
pub(crate) struct DomainBlockProcessor<Block, CBlock, Client, CClient, Backend, Executor>
where
    Block: BlockT,
    CBlock: BlockT,
{
    pub(crate) domain_id: DomainId,
    pub(crate) domain_created_at: NumberFor<CBlock>,
    pub(crate) client: Arc<Client>,
    pub(crate) consensus_client: Arc<CClient>,
    pub(crate) backend: Arc<Backend>,
    pub(crate) domain_confirmation_depth: NumberFor<Block>,
    pub(crate) block_import: Arc<BoxBlockImport<Block>>,
    pub(crate) import_notification_sinks: DomainImportNotificationSinks<Block, CBlock>,
    pub(crate) domain_sync_oracle: Arc<dyn SyncOracle + Send + Sync>,
    pub(crate) domain_executor: Arc<Executor>,
}

impl<Block, CBlock, Client, CClient, Backend, Executor> Clone
    for DomainBlockProcessor<Block, CBlock, Client, CClient, Backend, Executor>
where
    Block: BlockT,
    CBlock: BlockT,
{
    fn clone(&self) -> Self {
        Self {
            domain_id: self.domain_id,
            domain_created_at: self.domain_created_at,
            client: self.client.clone(),
            consensus_client: self.consensus_client.clone(),
            backend: self.backend.clone(),
            domain_confirmation_depth: self.domain_confirmation_depth,
            block_import: self.block_import.clone(),
            import_notification_sinks: self.import_notification_sinks.clone(),
            domain_sync_oracle: self.domain_sync_oracle.clone(),
            domain_executor: self.domain_executor.clone(),
        }
    }
}

/// A list of consensus blocks waiting to be processed by operator on each imported consensus block
/// notification.
///
/// Usually, each new domain block is built on top of the current best domain block, with the block
/// content extracted from the incoming consensus block. However, an incoming imported consensus block
/// notification can also imply multiple pending consensus blocks in case of the consensus chain re-org.
#[derive(Debug)]
pub(crate) struct PendingConsensusBlocks<Block: BlockT, CBlock: BlockT> {
    /// Base block used to build new domain blocks derived from the consensus blocks below.
    pub initial_parent: (Block::Hash, NumberFor<Block>),
    /// Pending consensus blocks that need to be processed sequentially.
    pub consensus_imports: Vec<HashAndNumber<CBlock>>,
}

impl<Block, CBlock, Client, CClient, Backend, Executor>
    DomainBlockProcessor<Block, CBlock, Client, CClient, Backend, Executor>
where
    Block: BlockT,
    CBlock: BlockT,
    NumberFor<CBlock>: Into<NumberFor<Block>>,
    Client: HeaderBackend<Block>
        + BlockBackend<Block>
        + AuxStore
        + ProvideRuntimeApi<Block>
        + Finalizer<Block, Backend>
        + ExecutorProvider<Block>
        + 'static,
    Client::Api:
        DomainCoreApi<Block> + sp_block_builder::BlockBuilder<Block> + sp_api::ApiExt<Block>,
    CClient: HeaderBackend<CBlock>
        + HeaderMetadata<CBlock, Error = sp_blockchain::Error>
        + BlockBackend<CBlock>
        + ProvideRuntimeApi<CBlock>
        + 'static,
    CClient::Api: DomainsApi<CBlock, Block::Header>
        + MessengerApi<CBlock, NumberFor<CBlock>, CBlock::Hash>
        + 'static,
    Backend: sc_client_api::Backend<Block> + 'static,
    Executor: CodeExecutor,
{
    /// Returns a list of consensus blocks waiting to be processed if any.
    ///
    /// It's possible to have multiple pending consensus blocks that need to be processed in case
    /// the consensus chain re-org occurs.
    pub(crate) fn pending_imported_consensus_blocks(
        &self,
        consensus_block_hash: CBlock::Hash,
        consensus_block_number: NumberFor<CBlock>,
    ) -> sp_blockchain::Result<Option<PendingConsensusBlocks<Block, CBlock>>> {
        match consensus_block_number.cmp(&(self.domain_created_at + One::one())) {
            // Consensus block at `domain_created_at + 1` is the first block that possibly contains
            // bundle, thus we can safely skip any consensus block that is smaller than this block.
            Ordering::Less => return Ok(None),
            // For consensus block at `domain_created_at + 1` use the genesis domain block as the
            // initial parent block directly to avoid further processing.
            Ordering::Equal => {
                return Ok(Some(PendingConsensusBlocks {
                    initial_parent: (self.client.info().genesis_hash, Zero::zero()),
                    consensus_imports: vec![HashAndNumber {
                        hash: consensus_block_hash,
                        number: consensus_block_number,
                    }],
                }));
            }
            // For consensus block > `domain_created_at + 1`, there is potential existing fork
            // thus we need to handle it carefully as following.
            Ordering::Greater => {}
        }

        let best_hash = self.client.info().best_hash;
        let best_number = self.client.info().best_number;

        // When there are empty consensus blocks multiple consensus block could map to the same
        // domain block, thus use `latest_consensus_block_hash_for` to find the latest consensus
        // block that map to the best domain block.
        let consensus_block_hash_for_best_domain_hash =
            match crate::aux_schema::latest_consensus_block_hash_for(&*self.backend, &best_hash)? {
                // If the auxiliary storage is empty and the best domain block is the genesis block
                // this consensus block could be the first block being processed thus just use the
                // consensus block at `domain_created_at` directly, otherwise the auxiliary storage
                // is expected to be non-empty thus return an error.
                //
                // NOTE: it is important to check the auxiliary storage first before checking if it
                // is the genesis block otherwise we may repeatedly processing the empty consensus
                // block, see https://github.com/autonomys/subspace/issues/1763 for more detail.
                None if best_number.is_zero() => self
                    .consensus_client
                    .hash(self.domain_created_at)?
                    .ok_or_else(|| {
                        sp_blockchain::Error::Backend(format!(
                            "Consensus hash for block #{} not found",
                            self.domain_created_at
                        ))
                    })?,
                None => {
                    return Err(sp_blockchain::Error::Backend(format!(
                        "Consensus hash for domain hash #{best_number},{best_hash} not found"
                    )));
                }
                Some(b) => b,
            };

        let consensus_from = consensus_block_hash_for_best_domain_hash;
        let consensus_to = consensus_block_hash;

        if consensus_from == consensus_to {
            tracing::debug!("Consensus block {consensus_block_hash:?} has already been processed");
            return Ok(None);
        }

        let route =
            sp_blockchain::tree_route(&*self.consensus_client, consensus_from, consensus_to)?;

        let retracted = route.retracted();
        let enacted = route.enacted();

        tracing::trace!(
            ?retracted,
            ?enacted,
            common_block = ?route.common_block(),
            "Calculating PendingConsensusBlocks on #{best_number},{best_hash:?}"
        );

        match (retracted.is_empty(), enacted.is_empty()) {
            (true, false) => {
                // New tip, A -> B
                Ok(Some(PendingConsensusBlocks {
                    initial_parent: (best_hash, best_number),
                    consensus_imports: enacted.to_vec(),
                }))
            }
            (false, true) => {
                tracing::debug!("Consensus blocks {retracted:?} have been already processed");
                Ok(None)
            }
            (true, true) => {
                unreachable!(
                    "Tree route is not empty as `consensus_from` and `consensus_to` in tree_route() \
                    are checked above to be not the same; qed",
                );
            }
            (false, false) => {
                let (common_block_number, common_block_hash) =
                    (route.common_block().number, route.common_block().hash);

                // The `common_block` is smaller than the consensus block that the domain was created at, thus
                // we can safely skip any consensus block that is smaller than `domain_created_at` and start at
                // the consensus block `domain_created_at + 1`, which the first block that possibly contains bundle,
                // and use the genesis domain block as the initial parent block.
                if common_block_number <= self.domain_created_at {
                    let consensus_imports = enacted
                        .iter()
                        .skip_while(|block| block.number <= self.domain_created_at)
                        .cloned()
                        .collect();
                    return Ok(Some(PendingConsensusBlocks {
                        initial_parent: (self.client.info().genesis_hash, Zero::zero()),
                        consensus_imports,
                    }));
                }

                // Get the domain block that is derived from the common consensus block and use it as
                // the initial domain parent block
                let domain_block_hash: Block::Hash = crate::aux_schema::best_domain_hash_for(
                    &*self.client,
                    &common_block_hash,
                )?
                    .ok_or_else(
                        || {
                            sp_blockchain::Error::Backend(format!(
                                "Hash of domain block derived from consensus block #{common_block_number},{common_block_hash} not found"
                            ))
                        },
                    )?;
                let parent_header = self.client.header(domain_block_hash)?.ok_or_else(|| {
                    sp_blockchain::Error::Backend(format!(
                        "Domain block header for #{domain_block_hash:?} not found",
                    ))
                })?;

                Ok(Some(PendingConsensusBlocks {
                    initial_parent: (parent_header.hash(), *parent_header.number()),
                    consensus_imports: enacted.to_vec(),
                }))
            }
        }
    }

    pub(crate) async fn process_domain_block(
        &self,
        (consensus_block_hash, consensus_block_number): (CBlock::Hash, NumberFor<CBlock>),
        (parent_hash, parent_number): (Block::Hash, NumberFor<Block>),
        preprocess_result: PreprocessResult<Block>,
        inherent_digests: Digest,
    ) -> Result<DomainBlockResult<Block, CBlock>, sp_blockchain::Error> {
        let PreprocessResult {
            extrinsics,
            bundles,
        } = preprocess_result;

        // The fork choice of the domain chain should follow exactly as the consensus chain,
        // so always use `Custom(false)` here to not set the domain block as new best block
        // immediately, and if the domain block is indeed derive from the best consensus fork,
        // it will be reset as the best domain block, see `BundleProcessor::process_bundles`.
        let fork_choice = ForkChoiceStrategy::Custom(false);
        let inherent_data = get_inherent_data::<_, _, Block>(
            self.consensus_client.clone(),
            consensus_block_hash,
            parent_hash,
            self.domain_id,
        )
        .await?;

        let DomainBlockBuildResult {
            extrinsics_root,
            state_root,
            header_number,
            header_hash,
            intermediate_roots,
        } = self
            .build_and_import_block(
                parent_hash,
                parent_number,
                extrinsics,
                fork_choice,
                inherent_digests,
                inherent_data,
            )
            .await?;

        tracing::debug!(
            "Built new domain block #{header_number},{header_hash} from \
            consensus block #{consensus_block_number},{consensus_block_hash} \
            on top of parent domain block #{parent_number},{parent_hash}"
        );

        // if let Some(to_finalize_block_number) =
        //     header_number.checked_sub(&self.domain_confirmation_depth)
        // {
        //     if to_finalize_block_number > self.client.info().finalized_number {
        //         let to_finalize_block_hash =
        //             self.client.hash(to_finalize_block_number)?.ok_or_else(|| {
        //                 sp_blockchain::Error::Backend(format!(
        //                     "Header for #{to_finalize_block_number} not found"
        //                 ))
        //             })?;
        //         self.client
        //             .finalize_block(to_finalize_block_hash, None, true)?;
        //         tracing::debug!("Successfully finalized block: #{to_finalize_block_number},{to_finalize_block_hash}");
        //     }
        // }

        let roots: Vec<[u8; 32]> = intermediate_roots
            .iter()
            .map(|v| {
                v.encode().try_into().expect(
                    "State root uses the same Block hash type which must fit into [u8; 32]; qed",
                )
            })
            .collect();
        let trace_root = MerkleTree::from_leaves(&roots).root().ok_or_else(|| {
            sp_blockchain::Error::Application(Box::from("Failed to get merkle root of trace"))
        })?;

        tracing::trace!(
            ?intermediate_roots,
            ?trace_root,
            "Trace root calculated for #{header_number},{header_hash}"
        );

        let parent_receipt = if parent_number.is_zero() {
            let genesis_hash = self.client.info().genesis_hash;
            let genesis_header = self.client.header(genesis_hash)?.ok_or_else(|| {
                sp_blockchain::Error::Backend(format!(
                    "Domain block header for #{genesis_hash:?} not found",
                ))
            })?;
            ExecutionReceipt::genesis(
                *genesis_header.state_root(),
                *genesis_header.extrinsics_root(),
                genesis_hash,
            )
        } else {
            crate::load_execution_receipt_by_domain_hash::<Block, CBlock, _>(
                &*self.client,
                parent_hash,
                parent_number,
            )?
        };

        // Get the accumulated transaction fee of all transactions included in the block
        // and used as the operator reward
        let runtime_api = self.client.runtime_api();
        let block_fees = runtime_api.block_fees(header_hash)?;
        let transfers = runtime_api.transfers(header_hash)?;

        let execution_receipt = ExecutionReceipt {
            domain_block_number: header_number,
            domain_block_hash: header_hash,
            domain_block_extrinsic_root: extrinsics_root,
            parent_domain_block_receipt_hash: parent_receipt
                .hash::<HeaderHashingFor<Block::Header>>(),
            consensus_block_number,
            consensus_block_hash,
            inboxed_bundles: bundles,
            final_state_root: state_root,
            execution_trace: intermediate_roots,
            execution_trace_root: sp_core::H256(trace_root),
            block_fees,
            transfers,
        };

        Ok(DomainBlockResult {
            header_hash,
            header_number,
            execution_receipt,
        })
    }

    async fn build_and_import_block(
        &self,
        parent_hash: Block::Hash,
        parent_number: NumberFor<Block>,
        extrinsics: VecDeque<Block::Extrinsic>,
        fork_choice: ForkChoiceStrategy,
        inherent_digests: Digest,
        inherent_data: sp_inherents::InherentData,
    ) -> Result<DomainBlockBuildResult<Block>, sp_blockchain::Error> {
        let block_builder = BlockBuilder::new(
            self.client.clone(),
            parent_hash,
            parent_number,
            inherent_digests,
            self.backend.clone(),
            self.domain_executor.clone(),
            extrinsics,
            Some(inherent_data),
        )?;

        let BuiltBlock {
            block,
            storage_changes,
        } = block_builder.build()?;

        let CollectedStorageChanges {
            storage_changes,
            intermediate_roots,
        } = storage_changes;

        let (header, body) = block.deconstruct();
        let state_root = *header.state_root();
        let extrinsics_root = *header.extrinsics_root();
        let header_hash = header.hash();
        let header_number = *header.number();

        let block_import_params = {
            let block_origin = if self.domain_sync_oracle.is_major_syncing() {
                // The domain block is derived from the consensus block, if the consensus chain is
                // in major sync then we should also consider the domain block is `NetworkInitialSync`
                BlockOrigin::NetworkInitialSync
            } else {
                BlockOrigin::Own
            };
            let mut import_block = BlockImportParams::new(block_origin, header);
            import_block.body = Some(body);
            import_block.state_action =
                StateAction::ApplyChanges(StorageChanges::Changes(storage_changes));
            import_block.fork_choice = Some(fork_choice);
            import_block
        };
        self.import_domain_block(block_import_params).await?;

        Ok(DomainBlockBuildResult {
            header_hash,
            header_number,
            state_root,
            extrinsics_root,
            intermediate_roots,
        })
    }

    pub(crate) async fn import_domain_block(
        &self,
        block_import_params: BlockImportParams<Block>,
    ) -> Result<(), sp_blockchain::Error> {
        let (header_number, header_hash, parent_hash) = (
            *block_import_params.header.number(),
            block_import_params.header.hash(),
            *block_import_params.header.parent_hash(),
        );

        let import_result = self.block_import.import_block(block_import_params).await?;

        match import_result {
            ImportResult::Imported(..) => {}
            ImportResult::AlreadyInChain => {
                tracing::debug!("Block #{header_number},{header_hash:?} is already in chain");
            }
            ImportResult::KnownBad => {
                return Err(sp_consensus::Error::ClientImport(format!(
                    "Bad block #{header_number}({header_hash:?})"
                ))
                .into());
            }
            ImportResult::UnknownParent => {
                return Err(sp_consensus::Error::ClientImport(format!(
                    "Block #{header_number}({header_hash:?}) has an unknown parent: {parent_hash:?}"
                ))
                .into());
            }
            ImportResult::MissingState => {
                return Err(sp_consensus::Error::ClientImport(format!(
                    "Parent state of block #{header_number}({header_hash:?}) is missing, parent: {parent_hash:?}"
                ))
                    .into());
            }
        }

        Ok(())
    }

    pub(crate) fn on_consensus_block_processed(
        &self,
        consensus_block_hash: CBlock::Hash,
        domain_block_result: Option<DomainBlockResult<Block, CBlock>>,
    ) -> sp_blockchain::Result<()> {
        let domain_hash = match domain_block_result {
            Some(DomainBlockResult {
                header_hash,
                header_number: _,
                execution_receipt,
            }) => {
                let oldest_unconfirmed_receipt_number = self
                    .consensus_client
                    .runtime_api()
                    .oldest_unconfirmed_receipt_number(consensus_block_hash, self.domain_id)?;
                crate::aux_schema::write_execution_receipt::<_, Block, CBlock>(
                    &*self.client,
                    oldest_unconfirmed_receipt_number,
                    &execution_receipt,
                )?;

                // Notify the imported domain block when the receipt processing is done.
                let domain_import_notification = DomainBlockImportNotification {
                    domain_block_hash: header_hash,
                    consensus_block_hash,
                };

                self.import_notification_sinks.lock().retain(|sink| {
                    sink.unbounded_send(domain_import_notification.clone())
                        .is_ok()
                });

                header_hash
            }
            None => {
                // No new domain block produced, thus this consensus block should map to the same
                // domain block as its parent block
                let consensus_header = self
                    .consensus_client
                    .header(consensus_block_hash)?
                    .ok_or_else(|| {
                        sp_blockchain::Error::Backend(format!(
                            "Header for consensus block {consensus_block_hash:?} not found"
                        ))
                    })?;
                if *consensus_header.number() > self.domain_created_at + One::one() {
                    let consensus_parent_hash = consensus_header.parent_hash();
                    crate::aux_schema::best_domain_hash_for(&*self.client, consensus_parent_hash)?
                        .ok_or_else(|| {
                        sp_blockchain::Error::Backend(format!(
                            "Domain hash for consensus block {consensus_parent_hash:?} not found",
                        ))
                    })?
                } else {
                    self.client.info().genesis_hash
                }
            }
        };

        crate::aux_schema::track_domain_hash_and_consensus_hash(
            &*self.client,
            domain_hash,
            consensus_block_hash,
        )?;

        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct InboxedBundleMismatchInfo {
    bundle_index: u32,
    mismatch_type: BundleMismatchType,
}

// Find the first mismatch of the `InboxedBundle` in the `ER::inboxed_bundles` list
pub(crate) fn find_inboxed_bundles_mismatch<Block, CBlock>(
    local_receipt: &ExecutionReceiptFor<Block, CBlock>,
    external_receipt: &ExecutionReceiptFor<Block, CBlock>,
) -> Result<Option<InboxedBundleMismatchInfo>, sp_blockchain::Error>
where
    Block: BlockT,
    CBlock: BlockT,
{
    if local_receipt.inboxed_bundles == external_receipt.inboxed_bundles {
        return Ok(None);
    }

    // The `bundles_extrinsics_roots` should be checked in the runtime when the consensus block is
    // constructed/imported thus the `external_receipt` must have the same `bundles_extrinsics_roots`
    //
    // NOTE: this also check `local_receipt.inboxed_bundles` and `external_receipt.inboxed_bundles`
    // have the same length
    if local_receipt.bundles_extrinsics_roots() != external_receipt.bundles_extrinsics_roots() {
        return Err(sp_blockchain::Error::Application(format!(
            "Found mismatch of `ER::bundles_extrinsics_roots`, this should not happen, local: {:?}, external: {:?}",
            local_receipt.bundles_extrinsics_roots(),
            external_receipt.bundles_extrinsics_roots(),
        ).into()));
    }

    // Get the first mismatch of `ER::inboxed_bundles`
    let (bundle_index, (local_bundle, external_bundle)) = local_receipt
        .inboxed_bundles
        .iter()
        .zip(external_receipt.inboxed_bundles.iter())
        .enumerate()
        .find(|(_, (local_bundle, external_bundle))| local_bundle != external_bundle)
        .expect(
            "The `local_receipt.inboxed_bundles` and `external_receipt.inboxed_bundles` are checked to have the \
            same length and being non-equal; qed",
        );

    // The `local_bundle` and `external_bundle` are checked to have the same `extrinsic_root`
    // thus they must being mismatch due to bundle validity
    let mismatch_type = match (local_bundle.bundle.clone(), external_bundle.bundle.clone()) {
        (
            BundleValidity::Invalid(local_invalid_type),
            BundleValidity::Invalid(external_invalid_type),
        ) => {
            match local_invalid_type
                .checking_order()
                .cmp(&external_invalid_type.checking_order())
            {
                // The `external_invalid_type` claim a prior check is pass while the `local_invalid_type` think
                // it is failed, so generate a fraud proof to prove that check is truely invalid
                Ordering::Less => BundleMismatchType::TrueInvalid(local_invalid_type),
                // The `external_invalid_type` claim a prior check is failed while the `local_invalid_type` think
                // it is pass, so generate a fraud proof to prove that check is falsely invalid
                Ordering::Greater => BundleMismatchType::FalseInvalid(external_invalid_type),
                Ordering::Equal => unreachable!(
                    "bundle validity must be different as the local/external bundle are checked to be different \
                    and they have the same `extrinsic_root`"
                ),
            }
        }
        (BundleValidity::Valid(_), BundleValidity::Valid(_)) => BundleMismatchType::Valid,
        (BundleValidity::Valid(_), BundleValidity::Invalid(invalid_type)) => {
            BundleMismatchType::FalseInvalid(invalid_type)
        }
        (BundleValidity::Invalid(invalid_type), BundleValidity::Valid(_)) => {
            BundleMismatchType::TrueInvalid(invalid_type)
        }
    };

    Ok(Some(InboxedBundleMismatchInfo {
        mismatch_type,
        bundle_index: bundle_index as u32,
    }))
}

pub(crate) struct ReceiptsChecker<Block, Client, CBlock, CClient, Backend, E>
where
    Block: BlockT,
    CBlock: BlockT,
{
    pub(crate) domain_id: DomainId,
    pub(crate) client: Arc<Client>,
    pub(crate) consensus_client: Arc<CClient>,
    pub(crate) domain_sync_oracle: Arc<dyn SyncOracle + Send + Sync>,
    pub(crate) fraud_proof_generator:
        FraudProofGenerator<Block, CBlock, Client, CClient, Backend, E>,
    pub(crate) consensus_offchain_tx_pool_factory: OffchainTransactionPoolFactory<CBlock>,
}

impl<Block, CBlock, Client, CClient, Backend, E> Clone
    for ReceiptsChecker<Block, Client, CBlock, CClient, Backend, E>
where
    Block: BlockT,
    CBlock: BlockT,
{
    fn clone(&self) -> Self {
        Self {
            domain_id: self.domain_id,
            client: self.client.clone(),
            consensus_client: self.consensus_client.clone(),
            domain_sync_oracle: self.domain_sync_oracle.clone(),
            fraud_proof_generator: self.fraud_proof_generator.clone(),
            consensus_offchain_tx_pool_factory: self.consensus_offchain_tx_pool_factory.clone(),
        }
    }
}

pub struct MismatchedReceipts<Block, CBlock>
where
    Block: BlockT,
    CBlock: BlockT,
{
    local_receipt: ExecutionReceiptFor<Block, CBlock>,
    bad_receipt: ExecutionReceiptFor<Block, CBlock>,
}

impl<Block, Client, CBlock, CClient, Backend, E>
    ReceiptsChecker<Block, Client, CBlock, CClient, Backend, E>
where
    Block: BlockT,
    Block::Hash: Into<H256>,
    CBlock: BlockT,
    NumberFor<CBlock>: Into<NumberFor<Block>>,
    Client: HeaderBackend<Block>
        + BlockBackend<Block>
        + ProofProvider<Block>
        + AuxStore
        + ExecutorProvider<Block>
        + ProvideRuntimeApi<Block>
        + 'static,
    Client::Api: DomainCoreApi<Block>
        + sp_block_builder::BlockBuilder<Block>
        + sp_api::ApiExt<Block>
        + MessengerApi<Block, NumberFor<CBlock>, CBlock::Hash>,
    CClient: HeaderBackend<CBlock>
        + BlockBackend<CBlock>
        + ProofProvider<CBlock>
        + ProvideRuntimeApi<CBlock>
        + 'static,
    CClient::Api: DomainsApi<CBlock, Block::Header>
        + FraudProofApi<CBlock, Block::Header>
        + MmrApi<CBlock, H256, NumberFor<CBlock>>,
    Backend: sc_client_api::Backend<Block> + 'static,
    E: CodeExecutor,
{
    pub(crate) fn maybe_submit_fraud_proof(
        &self,
        consensus_block_hash: CBlock::Hash,
    ) -> sp_blockchain::Result<()> {
        if self.domain_sync_oracle.is_major_syncing() {
            tracing::debug!(
                "Skip reporting unconfirmed bad receipt as the consensus node is still major syncing..."
            );
            return Ok(());
        }

        if let Some(mismatched_receipts) = self.find_mismatch_receipt(consensus_block_hash)? {
            let fraud_proof = self.generate_fraud_proof(mismatched_receipts)?;
            tracing::info!("Submit fraud proof: {fraud_proof:?}");

            let consensus_best_hash = self.consensus_client.info().best_hash;
            let mut consensus_runtime_api = self.consensus_client.runtime_api();
            consensus_runtime_api.register_extension(
                self.consensus_offchain_tx_pool_factory
                    .offchain_transaction_pool(consensus_best_hash),
            );
            consensus_runtime_api.submit_fraud_proof_unsigned(consensus_best_hash, fraud_proof)?;
        }

        Ok(())
    }

    pub fn find_mismatch_receipt(
        &self,
        consensus_block_hash: CBlock::Hash,
    ) -> sp_blockchain::Result<Option<MismatchedReceipts<Block, CBlock>>> {
        let mut oldest_mismatch = None;
        let mut to_check = self
            .consensus_client
            .runtime_api()
            .head_receipt_number(consensus_block_hash, self.domain_id)?;
        let oldest_unconfirmed_receipt_number = match self
            .consensus_client
            .runtime_api()
            .oldest_unconfirmed_receipt_number(consensus_block_hash, self.domain_id)?
        {
            Some(er_number) => er_number,
            // Early return if there is no non-confirmed ER exist
            None => return Ok(None),
        };

        while !to_check.is_zero() && oldest_unconfirmed_receipt_number <= to_check {
            let onchain_receipt_hash = self
                .consensus_client
                .runtime_api()
                .receipt_hash(consensus_block_hash, self.domain_id, to_check)?
                .ok_or_else(|| {
                    sp_blockchain::Error::Application(
                        format!("Receipt hash for #{to_check:?} not found").into(),
                    )
                })?;
            let local_receipt = {
                // Get the domain block hash corresponding to `to_check` in the
                // domain canonical chain
                let domain_hash = self.client.hash(to_check)?.ok_or_else(|| {
                    sp_blockchain::Error::Backend(format!(
                        "Domain block hash for #{to_check:?} not found"
                    ))
                })?;
                crate::load_execution_receipt_by_domain_hash::<Block, CBlock, _>(
                    &*self.client,
                    domain_hash,
                    to_check,
                )?
            };
            if local_receipt.hash::<HeaderHashingFor<Block::Header>>() != onchain_receipt_hash {
                oldest_mismatch.replace((local_receipt, onchain_receipt_hash));
                to_check = to_check.saturating_sub(One::one());
            } else {
                break;
            }
        }

        match oldest_mismatch {
            None => Ok(None),
            Some((local_receipt, bad_receipt_hash)) => {
                let bad_receipt = self
                    .consensus_client
                    .runtime_api()
                    .execution_receipt(consensus_block_hash, bad_receipt_hash)?
                    .ok_or_else(|| {
                        sp_blockchain::Error::Application(
                            format!("Receipt for #{bad_receipt_hash:?} not found").into(),
                        )
                    })?;
                debug_assert_eq!(
                    local_receipt.consensus_block_hash,
                    bad_receipt.consensus_block_hash,
                );
                Ok(Some(MismatchedReceipts {
                    local_receipt,
                    bad_receipt,
                }))
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_fraud_proof(
        &self,
        mismatched_receipts: MismatchedReceipts<Block, CBlock>,
    ) -> sp_blockchain::Result<FraudProof<NumberFor<CBlock>, CBlock::Hash, Block::Header, H256>>
    {
        let MismatchedReceipts {
            local_receipt,
            bad_receipt,
        } = mismatched_receipts;

        let bad_receipt_hash = bad_receipt.hash::<HeaderHashingFor<Block::Header>>();

        // NOTE: the checking order MUST follow exactly as the dependency order of fraud proof
        // see https://github.com/autonomys/subspace/issues/1892
        if let Some(InboxedBundleMismatchInfo {
            bundle_index,
            mismatch_type,
        }) = find_inboxed_bundles_mismatch::<Block, CBlock>(&local_receipt, &bad_receipt)?
        {
            return match mismatch_type {
                BundleMismatchType::Valid => self
                    .fraud_proof_generator
                    .generate_valid_bundle_proof(
                        self.domain_id,
                        &local_receipt,
                        bundle_index as usize,
                        bad_receipt_hash,
                    )
                    .map_err(|err| {
                        sp_blockchain::Error::Application(Box::from(format!(
                            "Failed to generate valid bundles fraud proof: {err}"
                        )))
                    }),
                _ => self
                    .fraud_proof_generator
                    .generate_invalid_bundle_proof(
                        self.domain_id,
                        &local_receipt,
                        mismatch_type,
                        bundle_index,
                        bad_receipt_hash,
                    )
                    .map_err(|err| {
                        sp_blockchain::Error::Application(Box::from(format!(
                            "Failed to generate invalid bundles fraud proof: {err}"
                        )))
                    }),
            };
        }

        if bad_receipt.domain_block_extrinsic_root != local_receipt.domain_block_extrinsic_root {
            return self
                .fraud_proof_generator
                .generate_invalid_domain_extrinsics_root_proof(
                    self.domain_id,
                    &local_receipt,
                    bad_receipt_hash,
                )
                .map_err(|err| {
                    sp_blockchain::Error::Application(Box::from(format!(
                        "Failed to generate invalid domain extrinsics root fraud proof: {err}"
                    )))
                });
        }

        if let Some(execution_phase) = self
            .fraud_proof_generator
            .find_mismatched_execution_phase(
                local_receipt.domain_block_hash,
                &local_receipt.execution_trace,
                &bad_receipt.execution_trace,
            )
            .map_err(|err| {
                sp_blockchain::Error::Application(Box::from(format!(
                    "Failed to find mismatched execution phase: {err}"
                )))
            })?
        {
            return self
                .fraud_proof_generator
                .generate_invalid_state_transition_proof(
                    self.domain_id,
                    execution_phase,
                    &local_receipt,
                    bad_receipt.execution_trace.len(),
                    bad_receipt_hash,
                )
                .map_err(|err| {
                    sp_blockchain::Error::Application(Box::from(format!(
                        "Failed to generate invalid state transition fraud proof: {err}"
                    )))
                });
        }

        if bad_receipt.block_fees != local_receipt.block_fees {
            return self
                .fraud_proof_generator
                .generate_invalid_block_fees_proof(self.domain_id, &local_receipt, bad_receipt_hash)
                .map_err(|err| {
                    sp_blockchain::Error::Application(Box::from(format!(
                        "Failed to generate invalid block rewards fraud proof: {err}"
                    )))
                });
        }

        if bad_receipt.transfers != local_receipt.transfers {
            return self
                .fraud_proof_generator
                .generate_invalid_transfers_proof(self.domain_id, &local_receipt, bad_receipt_hash)
                .map_err(|err| {
                    sp_blockchain::Error::Application(Box::from(format!(
                        "Failed to generate invalid transfers fraud proof: {err}"
                    )))
                });
        }

        if bad_receipt.domain_block_hash != local_receipt.domain_block_hash {
            return self
                .fraud_proof_generator
                .generate_invalid_domain_block_hash_proof(
                    self.domain_id,
                    &local_receipt,
                    bad_receipt_hash,
                )
                .map_err(|err| {
                    sp_blockchain::Error::Application(Box::from(format!(
                        "Failed to generate invalid domain block hash fraud proof: {err}"
                    )))
                });
        }

        Err(sp_blockchain::Error::Application(Box::from(format!(
            "No fraudulent field found for the mismatched ER, this should not happen, \
            local_receipt {local_receipt:?}, bad_receipt {bad_receipt:?}"
        ))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use domain_test_service::evm_domain_test_runtime::Block;
    use sp_domains::{InboxedBundle, InvalidBundleType};
    use subspace_test_runtime::Block as CBlock;

    fn create_test_execution_receipt(
        inboxed_bundles: Vec<InboxedBundle<<Block as BlockT>::Hash>>,
    ) -> ExecutionReceiptFor<Block, CBlock>
    where
        Block: BlockT,
        CBlock: BlockT,
    {
        ExecutionReceipt {
            domain_block_number: Zero::zero(),
            domain_block_hash: Default::default(),
            domain_block_extrinsic_root: Default::default(),
            parent_domain_block_receipt_hash: Default::default(),
            consensus_block_hash: Default::default(),
            consensus_block_number: Zero::zero(),
            inboxed_bundles,
            final_state_root: Default::default(),
            execution_trace: vec![],
            execution_trace_root: Default::default(),
            block_fees: Default::default(),
            transfers: Default::default(),
        }
    }

    #[test]
    fn er_bundles_mismatch_detection() {
        // If empty invalid receipt field on both should result in no fraud proof
        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![]),
                &create_test_execution_receipt(vec![]),
            )
            .unwrap(),
            None
        );

        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![InboxedBundle::invalid(
                    InvalidBundleType::UndecodableTx(0),
                    Default::default(),
                )]),
                &create_test_execution_receipt(vec![InboxedBundle::invalid(
                    InvalidBundleType::UndecodableTx(0),
                    Default::default(),
                )]),
            )
            .unwrap(),
            None
        );

        // Mismatch in valid bundle
        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![
                    InboxedBundle::invalid(InvalidBundleType::UndecodableTx(0), Default::default()),
                    InboxedBundle::valid(H256::random(), Default::default()),
                ]),
                &create_test_execution_receipt(vec![
                    InboxedBundle::invalid(InvalidBundleType::UndecodableTx(0), Default::default()),
                    InboxedBundle::valid(H256::random(), Default::default()),
                ]),
            )
            .unwrap(),
            Some(InboxedBundleMismatchInfo {
                mismatch_type: BundleMismatchType::Valid,
                bundle_index: 1,
            })
        );

        // Mismatch in invalid extrinsic index
        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(InvalidBundleType::UndecodableTx(1), Default::default()),
                ]),
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(InvalidBundleType::UndecodableTx(2), Default::default()),
                ]),
            )
            .unwrap(),
            Some(InboxedBundleMismatchInfo {
                mismatch_type: BundleMismatchType::TrueInvalid(InvalidBundleType::UndecodableTx(1)),
                bundle_index: 1,
            })
        );
        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(InvalidBundleType::UndecodableTx(4), Default::default()),
                ]),
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(InvalidBundleType::UndecodableTx(3), Default::default()),
                ]),
            )
            .unwrap(),
            Some(InboxedBundleMismatchInfo {
                mismatch_type: BundleMismatchType::FalseInvalid(InvalidBundleType::UndecodableTx(
                    3
                )),
                bundle_index: 1,
            })
        );
        // Even the invalid type is mismatch, the extrinsic index mismatch should be considered first
        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(InvalidBundleType::UndecodableTx(4), Default::default()),
                ]),
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(InvalidBundleType::IllegalTx(3), Default::default()),
                ]),
            )
            .unwrap(),
            Some(InboxedBundleMismatchInfo {
                mismatch_type: BundleMismatchType::FalseInvalid(InvalidBundleType::IllegalTx(3)),
                bundle_index: 1,
            })
        );

        // Mismatch in invalid type
        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(InvalidBundleType::IllegalTx(3), Default::default()),
                ]),
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(
                        InvalidBundleType::InherentExtrinsic(3),
                        Default::default()
                    ),
                ]),
            )
            .unwrap(),
            Some(InboxedBundleMismatchInfo {
                mismatch_type: BundleMismatchType::FalseInvalid(
                    InvalidBundleType::InherentExtrinsic(3)
                ),
                bundle_index: 1,
            })
        );

        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(
                        InvalidBundleType::InherentExtrinsic(3),
                        Default::default()
                    ),
                ]),
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(Default::default(), Default::default()),
                    InboxedBundle::invalid(InvalidBundleType::IllegalTx(3), Default::default()),
                ]),
            )
            .unwrap(),
            Some(InboxedBundleMismatchInfo {
                mismatch_type: BundleMismatchType::TrueInvalid(
                    InvalidBundleType::InherentExtrinsic(3)
                ),
                bundle_index: 1,
            })
        );

        // Only first mismatch is detected
        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(H256::random(), Default::default()),
                    InboxedBundle::invalid(
                        InvalidBundleType::InherentExtrinsic(3),
                        Default::default()
                    ),
                ]),
                &create_test_execution_receipt(vec![
                    InboxedBundle::valid(H256::random(), Default::default()),
                    InboxedBundle::invalid(InvalidBundleType::IllegalTx(3), Default::default()),
                ]),
            )
            .unwrap(),
            Some(InboxedBundleMismatchInfo {
                mismatch_type: BundleMismatchType::Valid,
                bundle_index: 0,
            })
        );

        // Taking valid bundle as invalid
        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![InboxedBundle::valid(
                    H256::random(),
                    Default::default(),
                ),]),
                &create_test_execution_receipt(vec![InboxedBundle::invalid(
                    InvalidBundleType::IllegalTx(3),
                    Default::default(),
                ),]),
            )
            .unwrap(),
            Some(InboxedBundleMismatchInfo {
                mismatch_type: BundleMismatchType::FalseInvalid(InvalidBundleType::IllegalTx(3)),
                bundle_index: 0,
            })
        );

        // Taking invalid as valid
        assert_eq!(
            find_inboxed_bundles_mismatch::<Block, CBlock>(
                &create_test_execution_receipt(vec![InboxedBundle::invalid(
                    InvalidBundleType::IllegalTx(3),
                    Default::default(),
                ),]),
                &create_test_execution_receipt(vec![InboxedBundle::valid(
                    H256::random(),
                    Default::default(),
                ),]),
            )
            .unwrap(),
            Some(InboxedBundleMismatchInfo {
                mismatch_type: BundleMismatchType::TrueInvalid(InvalidBundleType::IllegalTx(3)),
                bundle_index: 0,
            })
        );
    }
}
