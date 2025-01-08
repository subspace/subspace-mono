use crate::domain_block_processor::{
    DomainBlockProcessor, PendingConsensusBlocks, ReceiptsChecker,
};
use crate::ExecutionReceiptFor;
use domain_block_preprocessor::DomainBlockPreprocessor;
use sc_client_api::{AuxStore, BlockBackend, Finalizer, ProofProvider};
use sc_consensus::{BlockImportParams, ForkChoiceStrategy, StateAction};
use sp_api::ProvideRuntimeApi;
use sp_blockchain::{HeaderBackend, HeaderMetadata};
use sp_consensus::BlockOrigin;
use sp_core::traits::CodeExecutor;
use sp_core::H256;
use sp_domain_digests::AsPredigest;
use sp_domains::core_api::DomainCoreApi;
use sp_domains::{DomainId, DomainsApi, ReceiptValidity};
use sp_domains_fraud_proof::FraudProofApi;
use sp_messenger::MessengerApi;
use sp_mmr_primitives::MmrApi;
use sp_runtime::traits::{Block as BlockT, NumberFor, Zero};
use sp_runtime::{Digest, DigestItem};
use sp_weights::constants::WEIGHT_REF_TIME_PER_MILLIS;
use std::sync::Arc;
use std::time::Instant;

// The slow log threshold for consensus block preprocessing
const SLOW_PREPROCESS_MILLIS: u64 = 500;

// The slow log threshold for domain block execution: `reference_duration_ms * 1.2 + 200ms`,
// where `reference_duration_ms * 0.2` as buffer of the slow extrinsic execution (i.e. slower
// machine than the reference machine) and 200ms as buffer of the rest of the processing.
fn slow_domain_block_execution_threshold(reference_duration_ms: u64) -> u64 {
    reference_duration_ms + (reference_duration_ms / 5) + 200
}

// The slow log threshold for post domain block execution
const SLOW_POST_DOMAIN_BLOCK_EXECUTION_MILLIS: u64 = 250;

type DomainReceiptsChecker<Block, CBlock, Client, CClient, Backend, E> =
    ReceiptsChecker<Block, Client, CBlock, CClient, Backend, E>;

pub(crate) struct BundleProcessor<Block, CBlock, Client, CClient, Backend, E>
where
    Block: BlockT,
    CBlock: BlockT,
{
    domain_id: DomainId,
    consensus_client: Arc<CClient>,
    client: Arc<Client>,
    backend: Arc<Backend>,
    domain_receipts_checker: DomainReceiptsChecker<Block, CBlock, Client, CClient, Backend, E>,
    domain_block_preprocessor:
        DomainBlockPreprocessor<Block, CBlock, Client, CClient, ReceiptValidator<Client>>,
    domain_block_processor: DomainBlockProcessor<Block, CBlock, Client, CClient, Backend>,
}

impl<Block, CBlock, Client, CClient, Backend, E> Clone
    for BundleProcessor<Block, CBlock, Client, CClient, Backend, E>
where
    Block: BlockT,
    CBlock: BlockT,
{
    fn clone(&self) -> Self {
        Self {
            domain_id: self.domain_id,
            consensus_client: self.consensus_client.clone(),
            client: self.client.clone(),
            backend: self.backend.clone(),
            domain_receipts_checker: self.domain_receipts_checker.clone(),
            domain_block_preprocessor: self.domain_block_preprocessor.clone(),
            domain_block_processor: self.domain_block_processor.clone(),
        }
    }
}

struct ReceiptValidator<Client> {
    client: Arc<Client>,
}

impl<Client> Clone for ReceiptValidator<Client> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
        }
    }
}

impl<Client> ReceiptValidator<Client> {
    pub fn new(client: Arc<Client>) -> Self {
        Self { client }
    }
}

impl<Block, CBlock, Client> domain_block_preprocessor::ValidateReceipt<Block, CBlock>
    for ReceiptValidator<Client>
where
    Block: BlockT,
    CBlock: BlockT,
    Client: AuxStore,
{
    fn validate_receipt(
        &self,
        receipt: &ExecutionReceiptFor<Block, CBlock>,
    ) -> sp_blockchain::Result<ReceiptValidity> {
        // Skip genesis receipt as it has been already verified by the consensus chain.
        if receipt.domain_block_number.is_zero() {
            return Ok(ReceiptValidity::Valid);
        }

        let consensus_block_hash = receipt.consensus_block_hash;
        let _local_receipt = crate::aux_schema::load_execution_receipt::<_, Block, CBlock>(
            &*self.client,
            consensus_block_hash,
        )?
        .ok_or_else(|| {
            sp_blockchain::Error::Backend(format!(
                "Receipt for consensus block {consensus_block_hash} not found"
            ))
        })?;

        Ok(ReceiptValidity::Valid)
    }
}

impl<Block, CBlock, Client, CClient, Backend, E>
    BundleProcessor<Block, CBlock, Client, CClient, Backend, E>
where
    Block: BlockT,
    Block::Hash: Into<H256>,
    CBlock: BlockT,
    NumberFor<CBlock>: From<NumberFor<Block>> + Into<NumberFor<Block>>,
    CBlock::Hash: From<Block::Hash>,
    Client: HeaderBackend<Block>
        + BlockBackend<Block>
        + AuxStore
        + ProvideRuntimeApi<Block>
        + ProofProvider<Block>
        + Finalizer<Block, Backend>
        + 'static,
    Client::Api: DomainCoreApi<Block>
        + MessengerApi<Block, NumberFor<CBlock>, CBlock::Hash>
        + sp_block_builder::BlockBuilder<Block>
        + sp_api::ApiExt<Block>,
    CClient: HeaderBackend<CBlock>
        + HeaderMetadata<CBlock, Error = sp_blockchain::Error>
        + BlockBackend<CBlock>
        + ProofProvider<CBlock>
        + ProvideRuntimeApi<CBlock>
        + 'static,
    CClient::Api: DomainsApi<CBlock, Block::Header>
        + MessengerApi<CBlock, NumberFor<CBlock>, CBlock::Hash>
        + FraudProofApi<CBlock, Block::Header>
        + MmrApi<CBlock, H256, NumberFor<CBlock>>
        + 'static,
    Backend: sc_client_api::Backend<Block> + 'static,
    E: CodeExecutor,
{
    pub(crate) fn new(
        domain_id: DomainId,
        consensus_client: Arc<CClient>,
        client: Arc<Client>,
        backend: Arc<Backend>,
        domain_receipts_checker: DomainReceiptsChecker<Block, CBlock, Client, CClient, Backend, E>,
        domain_block_processor: DomainBlockProcessor<Block, CBlock, Client, CClient, Backend>,
    ) -> Self {
        let domain_block_preprocessor = DomainBlockPreprocessor::new(
            domain_id,
            client.clone(),
            consensus_client.clone(),
            ReceiptValidator::new(client.clone()),
        );
        Self {
            domain_id,
            consensus_client,
            client,
            backend,
            domain_receipts_checker,
            domain_block_preprocessor,
            domain_block_processor,
        }
    }

    // TODO: Handle the returned error properly, ref to https://github.com/autonomys/subspace/pull/695#discussion_r926721185
    pub(crate) async fn process_bundles(
        self,
        consensus_block_info: (CBlock::Hash, NumberFor<CBlock>, bool),
    ) -> sp_blockchain::Result<()> {
        let (consensus_block_hash, consensus_block_number, is_new_best) = consensus_block_info;

        // Skip processing the blocks of the non-canonical chain, these blocks will be processed if
        // the chain becomes canonical later
        if !is_new_best {
            return Ok(());
        }

        tracing::debug!(
            "Processing consensus block #{consensus_block_number},{consensus_block_hash}"
        );

        let maybe_pending_consensus_blocks = self
            .domain_block_processor
            .pending_imported_consensus_blocks(consensus_block_hash, consensus_block_number)?;

        if let Some(PendingConsensusBlocks {
            initial_parent,
            consensus_imports,
        }) = maybe_pending_consensus_blocks
        {
            tracing::trace!(
                ?initial_parent,
                ?consensus_imports,
                "Pending consensus blocks to process"
            );

            let mut domain_parent = initial_parent;

            for consensus_info in consensus_imports {
                if let Some(next_domain_parent) = self
                    .process_bundles_at((consensus_info.hash, consensus_info.number), domain_parent)
                    .await?
                {
                    domain_parent = next_domain_parent;
                }
            }

            // The domain branch driving from the best consensus branch should also be the best domain branch even
            // if it is no the longest domain branch. Thus re-import the tip of the best domain branch to make it
            // the new best block if it isn't.
            //
            // Note: this may cause the best domain fork switch to a shorter fork or in some case the best domain
            // block become the ancestor block of the current best block.
            let domain_tip = domain_parent.0;
            if self.client.info().best_hash != domain_tip {
                let header = self.client.header(domain_tip)?.ok_or_else(|| {
                    sp_blockchain::Error::Backend(format!("Header for #{:?} not found", domain_tip))
                })?;
                let block_origin = if self
                    .domain_block_processor
                    .domain_sync_oracle
                    .is_major_syncing()
                {
                    // The domain block is derived from the consensus block, if the consensus chain is
                    // in major sync then we should also consider the domain block is `NetworkInitialSync`
                    BlockOrigin::NetworkInitialSync
                } else {
                    BlockOrigin::Own
                };
                let block_import_params = {
                    let mut import_block = BlockImportParams::new(block_origin, header);
                    import_block.import_existing = true;
                    import_block.fork_choice = Some(ForkChoiceStrategy::Custom(true));
                    import_block.state_action = StateAction::Skip;
                    import_block
                };
                self.domain_block_processor
                    .import_domain_block(block_import_params)
                    .await?;
                assert_eq!(domain_tip, self.client.info().best_hash);
            }

            // Check the ER submitted to consensus chain and submit fraud proof if there is bad ER
            // NOTE: this have to be done after the recorrect of the best domain fork happen above
            self.domain_receipts_checker
                .maybe_submit_fraud_proof(consensus_block_hash)?;
        }

        Ok(())
    }

    async fn process_bundles_at(
        &self,
        consensus_block_info: (CBlock::Hash, NumberFor<CBlock>),
        parent_info: (Block::Hash, NumberFor<Block>),
    ) -> sp_blockchain::Result<Option<(Block::Hash, NumberFor<Block>)>> {
        let (consensus_block_hash, consensus_block_number) = consensus_block_info;
        let (parent_hash, parent_number) = parent_info;
        let start = Instant::now();

        tracing::debug!(
            "Building a new domain block from consensus block #{consensus_block_number},{consensus_block_hash} \
            on top of parent block #{parent_number},{parent_hash}"
        );

        let maybe_preprocess_result = self
            .domain_block_preprocessor
            .preprocess_consensus_block(consensus_block_hash, (parent_hash, parent_number))?;

        let preprocess_took = start.elapsed().as_millis();
        // if preprocess_took >= SLOW_PREPROCESS_MILLIS.into() {
        tracing::warn!(
            ?consensus_block_info,
            "Slow consensus block preprocessing, took {preprocess_took}ms"
        );
        // }

        let Some(preprocess_result) = maybe_preprocess_result else {
            tracing::debug!(
                "Skip building new domain block, no bundles and runtime upgrade for this domain \
                    in consensus block #{consensus_block_number:?},{consensus_block_hash}"
            );
            self.domain_block_processor
                .on_consensus_block_processed(consensus_block_hash, None)?;
            return Ok(None);
        };

        let inherent_digests = Digest {
            logs: vec![DigestItem::consensus_block_info(consensus_block_hash)],
        };

        let domain_block_result = self
            .domain_block_processor
            .process_domain_block(
                (consensus_block_hash, consensus_block_number),
                (parent_hash, parent_number),
                preprocess_result,
                inherent_digests,
            )
            .await?;

        let head_receipt_number = self
            .consensus_client
            .runtime_api()
            .head_receipt_number(consensus_block_hash, self.domain_id)?;
        assert!(
            domain_block_result.header_number > head_receipt_number,
            "Domain chain number must larger than the head number of the receipt chain \
            (which is maintained on the consensus chain) by at least 1"
        );

        let built_block_info = (
            domain_block_result.header_hash,
            domain_block_result.header_number,
        );

        let block_execution_took = start.elapsed().as_millis().saturating_sub(preprocess_took);

        let reference_block_execution_duration_ms = self
            .client
            .runtime_api()
            .block_weight(domain_block_result.header_hash)?
            .ref_time()
            / WEIGHT_REF_TIME_PER_MILLIS;
        // if block_execution_took
        //     >= slow_domain_block_execution_threshold(reference_block_execution_duration_ms).into()
        // {
        tracing::warn!(
            ?consensus_block_info,
            ?built_block_info,
            ?reference_block_execution_duration_ms,
            "Slow domain block execution, took {block_execution_took}ms"
        );
        // }

        self.domain_block_processor
            .on_consensus_block_processed(consensus_block_hash, Some(domain_block_result))?;

        let post_block_execution_took = start
            .elapsed()
            .as_millis()
            .saturating_sub(preprocess_took + block_execution_took);
        if post_block_execution_took >= SLOW_POST_DOMAIN_BLOCK_EXECUTION_MILLIS.into() {
            tracing::warn!(
                ?consensus_block_info,
                ?built_block_info,
                "Slow post domain block execution, took {post_block_execution_took}ms"
            );
        }

        Ok(Some(built_block_info))
    }
}
