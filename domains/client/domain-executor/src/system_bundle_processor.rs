use crate::domain_block_processor::{
    preprocess_primary_block, DomainBlockProcessor, PendingPrimaryBlocks,
};
use crate::parent_chain::{ParentChainInterface, SystemDomainParentChain};
use crate::utils::translate_number_type;
use crate::TransactionFor;
use domain_runtime_primitives::{AccountId, DomainCoreApi, DomainExtrinsicApi};
use sc_client_api::{AuxStore, BlockBackend, StateBackendFor};
use sc_consensus::{BlockImport, ForkChoiceStrategy};
use sp_api::{NumberFor, ProvideRuntimeApi};
use sp_blockchain::{HeaderBackend, HeaderMetadata};
use sp_core::traits::CodeExecutor;
use sp_domain_digests::AsPredigest;
use sp_domains::{DomainId, ExecutorApi};
use sp_keystore::SyncCryptoStorePtr;
use sp_runtime::traits::{Block as BlockT, HashFor};
use sp_runtime::{Digest, DigestItem};
use std::sync::Arc;
use system_runtime_primitives::SystemDomainApi;

pub(crate) struct SystemBundleProcessor<Block, PBlock, Client, PClient, Backend, E>
where
    PBlock: BlockT,
{
    primary_chain_client: Arc<PClient>,
    client: Arc<Client>,
    parent_chain: SystemDomainParentChain<PClient, Block, PBlock>,
    backend: Arc<Backend>,
    keystore: SyncCryptoStorePtr,
    domain_block_processor:
        DomainBlockProcessor<Block, PBlock, Block, Client, PClient, Client, Backend, E>,
}

impl<Block, PBlock, Client, PClient, Backend, E> Clone
    for SystemBundleProcessor<Block, PBlock, Client, PClient, Backend, E>
where
    PBlock: BlockT,
{
    fn clone(&self) -> Self {
        Self {
            primary_chain_client: self.primary_chain_client.clone(),
            client: self.client.clone(),
            parent_chain: self.parent_chain.clone(),
            backend: self.backend.clone(),
            keystore: self.keystore.clone(),
            domain_block_processor: self.domain_block_processor.clone(),
        }
    }
}

impl<Block, PBlock, Client, PClient, Backend, E>
    SystemBundleProcessor<Block, PBlock, Client, PClient, Backend, E>
where
    Block: BlockT,
    PBlock: BlockT,
    Client:
        HeaderBackend<Block> + BlockBackend<Block> + AuxStore + ProvideRuntimeApi<Block> + 'static,
    Client::Api: DomainCoreApi<Block, AccountId>
        + DomainExtrinsicApi<Block, NumberFor<PBlock>, PBlock::Hash>
        + sp_block_builder::BlockBuilder<Block>
        + sp_api::ApiExt<Block, StateBackend = StateBackendFor<Backend, Block>>
        + SystemDomainApi<Block, NumberFor<PBlock>, PBlock::Hash>,
    for<'b> &'b Client: BlockImport<
        Block,
        Transaction = sp_api::TransactionFor<Client, Block>,
        Error = sp_consensus::Error,
    >,
    PClient: HeaderBackend<PBlock>
        + HeaderMetadata<PBlock, Error = sp_blockchain::Error>
        + BlockBackend<PBlock>
        + ProvideRuntimeApi<PBlock>
        + 'static,
    PClient::Api: ExecutorApi<PBlock, Block::Hash> + 'static,
    Backend: sc_client_api::Backend<Block> + 'static,
    TransactionFor<Backend, Block>: sp_trie::HashDBT<HashFor<Block>, sp_trie::DBValue>,
    E: CodeExecutor,
{
    pub(crate) fn new(
        primary_chain_client: Arc<PClient>,
        client: Arc<Client>,
        backend: Arc<Backend>,
        keystore: SyncCryptoStorePtr,
        domain_block_processor: DomainBlockProcessor<
            Block,
            PBlock,
            Block,
            Client,
            PClient,
            Client,
            Backend,
            E,
        >,
    ) -> Self {
        let parent_chain = SystemDomainParentChain::new(primary_chain_client.clone());
        Self {
            primary_chain_client,
            client,
            parent_chain,
            backend,
            keystore,
            domain_block_processor,
        }
    }

    // TODO: Handle the returned error properly, ref to https://github.com/subspace/subspace/pull/695#discussion_r926721185
    pub(crate) async fn process_bundles(
        self,
        primary_info: (PBlock::Hash, NumberFor<PBlock>, ForkChoiceStrategy),
    ) -> Result<(), sp_blockchain::Error> {
        tracing::debug!(?primary_info, "Processing imported primary block");

        let (primary_hash, primary_number, fork_choice) = primary_info;

        let maybe_pending_primary_blocks = self
            .domain_block_processor
            .pending_imported_primary_blocks(primary_hash, primary_number)?;

        if let Some(PendingPrimaryBlocks {
            initial_parent,
            primary_imports,
        }) = maybe_pending_primary_blocks
        {
            tracing::trace!(
                ?initial_parent,
                ?primary_imports,
                "Pending primary blocks to process"
            );

            let mut domain_parent = initial_parent;

            for (i, primary_info) in primary_imports.iter().enumerate() {
                // Use the origin fork_choice for the target primary block,
                // the intermediate ones use `Custom(false)`.
                let fork_choice = if i == primary_imports.len() - 1 {
                    fork_choice
                } else {
                    ForkChoiceStrategy::Custom(false)
                };

                domain_parent = self
                    .process_bundles_at(
                        (primary_info.hash, primary_info.number, fork_choice),
                        domain_parent,
                    )
                    .await?;
            }
        }

        Ok(())
    }

    async fn process_bundles_at(
        &self,
        primary_info: (PBlock::Hash, NumberFor<PBlock>, ForkChoiceStrategy),
        parent_info: (Block::Hash, NumberFor<Block>),
    ) -> Result<(Block::Hash, NumberFor<Block>), sp_blockchain::Error> {
        tracing::debug!(?primary_info, ?parent_info, "Building a new domain block");

        let (primary_hash, primary_number, fork_choice) = primary_info;
        let (parent_hash, parent_number) = parent_info;

        let (bundles, shuffling_seed, maybe_new_runtime) =
            preprocess_primary_block(DomainId::SYSTEM, &*self.primary_chain_client, primary_hash)?;

        let extrinsics = self.domain_block_processor.bundles_to_extrinsics(
            parent_hash,
            bundles,
            shuffling_seed,
        )?;

        let digests = {
            let mut digest = Digest::default();
            if let Some(state_root_update) = self
                .domain_block_processor
                .system_domain_state_root_update_digest(parent_hash)?
            {
                digest.push(state_root_update);
            }
            let primary_block_info = DigestItem::primary_block_info((primary_number, primary_hash));
            digest.push(primary_block_info);
            digest
        };

        let domain_block_result = self
            .domain_block_processor
            .process_domain_block(
                (primary_hash, primary_number),
                (parent_hash, parent_number),
                extrinsics,
                maybe_new_runtime,
                fork_choice,
                digests,
            )
            .await?;

        let head_receipt_number = {
            let n = self.parent_chain.head_receipt_number(primary_hash)?;
            translate_number_type::<NumberFor<PBlock>, NumberFor<Block>>(n)
        };

        assert!(
            domain_block_result.header_number > head_receipt_number,
            "Consensus chain number must larger than execution chain number by at least 1"
        );

        let oldest_receipt_number = {
            let n = self.parent_chain.oldest_receipt_number(primary_hash)?;
            translate_number_type::<NumberFor<PBlock>, NumberFor<Block>>(n)
        };

        let built_block_info = (
            domain_block_result.header_hash,
            domain_block_result.header_number,
        );

        if let Some(fraud_proof) = self.domain_block_processor.on_domain_block_processed(
            primary_hash,
            domain_block_result,
            head_receipt_number,
            oldest_receipt_number,
        )? {
            self.parent_chain.submit_fraud_proof_unsigned(fraud_proof)?;
        }

        Ok(built_block_info)
    }
}
