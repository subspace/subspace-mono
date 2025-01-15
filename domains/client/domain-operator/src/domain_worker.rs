// Copyright 2020 Parity Technologies (UK) Ltd.
// This file is part of Polkadot.

// Polkadot is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Polkadot is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Polkadot.  If not, see <http://www.gnu.org/licenses/>.

use crate::bundle_processor::BundleProcessor;
use crate::domain_bundle_producer::{BundleProducer, DomainProposal};
use crate::utils::{BlockInfo, OperatorSlotInfo};
use crate::{NewSlotNotification, OperatorStreams};
use futures::channel::mpsc;
use futures::{SinkExt, Stream, StreamExt};
use sc_client_api::{
    AuxStore, BlockBackend, BlockImportNotification, BlockchainEvents, ExecutorProvider, Finalizer,
    ProofProvider,
};
use sc_transaction_pool_api::OffchainTransactionPoolFactory;
use sp_api::{ApiExt, ProvideRuntimeApi};
use sp_block_builder::BlockBuilder;
use sp_blockchain::{HeaderBackend, HeaderMetadata};
use sp_core::traits::{CodeExecutor, SpawnEssentialNamed};
use sp_core::H256;
use sp_domains::core_api::DomainCoreApi;
use sp_domains::{BundleProducerElectionApi, DomainsApi, OpaqueBundle, OperatorId};
use sp_domains_fraud_proof::FraudProofApi;
use sp_messenger::MessengerApi;
use sp_mmr_primitives::MmrApi;
use sp_runtime::traits::{Block as BlockT, Header as HeaderT, NumberFor};
use sp_transaction_pool::runtime_api::TaggedTransactionQueue;
use std::pin::{pin, Pin};
use std::sync::Arc;
use std::task::{Context, Poll};
use subspace_runtime_primitives::Balance;
use tracing::{info, Instrument};

pub type OpaqueBundleFor<Block, CBlock> =
    OpaqueBundle<NumberFor<CBlock>, <CBlock as BlockT>::Hash, <Block as BlockT>::Header, Balance>;

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(super) async fn start_worker<
    Block,
    CBlock,
    Client,
    CClient,
    Backend,
    IBNS,
    CIBNS,
    NSNS,
    ASS,
    E,
>(
    spawn_essential: Box<dyn SpawnEssentialNamed>,
    consensus_client: Arc<CClient>,
    consensus_offchain_tx_pool_factory: OffchainTransactionPoolFactory<CBlock>,
    maybe_operator_id: Option<OperatorId>,
    mut bundle_producer: Box<dyn BundleProducer<Block, CBlock> + Send>,
    bundle_processor: BundleProcessor<Block, CBlock, Client, CClient, Backend, E>,
    operator_streams: OperatorStreams<CBlock, IBNS, CIBNS, NSNS, ASS>,
) where
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
        + ExecutorProvider<Block>
        + 'static,
    Client::Api: DomainCoreApi<Block>
        + MessengerApi<Block, NumberFor<CBlock>, CBlock::Hash>
        + BlockBuilder<Block>
        + sp_api::ApiExt<Block>
        + TaggedTransactionQueue<Block>,
    CClient: HeaderBackend<CBlock>
        + HeaderMetadata<CBlock, Error = sp_blockchain::Error>
        + BlockBackend<CBlock>
        + ProofProvider<CBlock>
        + ProvideRuntimeApi<CBlock>
        + BlockchainEvents<CBlock>
        + 'static,
    CClient::Api: DomainsApi<CBlock, Block::Header>
        + MessengerApi<CBlock, NumberFor<CBlock>, CBlock::Hash>
        + BundleProducerElectionApi<CBlock, Balance>
        + FraudProofApi<CBlock, Block::Header>
        + MmrApi<CBlock, H256, NumberFor<CBlock>>,
    Backend: sc_client_api::Backend<Block> + 'static,
    IBNS: Stream<Item = (NumberFor<CBlock>, mpsc::Sender<()>)> + Send + 'static,
    CIBNS: Stream<Item = BlockImportNotification<CBlock>> + Send + 'static,
    NSNS: Stream<Item = NewSlotNotification> + Send + 'static,
    ASS: Stream<Item = mpsc::Sender<()>> + Send + 'static,
    E: CodeExecutor,
{
    let span = tracing::Span::current();

    let OperatorStreams {
        consensus_block_import_throttling_buffer_size,
        block_importing_notification_stream,
        imported_block_notification_stream,
        new_slot_notification_stream,
        acknowledgement_sender_stream,
        _phantom,
    } = operator_streams;

    let mut throttled_block_import_notification_stream =
        throttling_block_import_notifications::<Block, _, _, _, _>(
            spawn_essential,
            consensus_client.clone(),
            Box::pin(block_importing_notification_stream),
            Box::pin(imported_block_notification_stream),
            consensus_block_import_throttling_buffer_size,
        );

    if let Some(operator_id) = maybe_operator_id {
        info!("👷 Running as Operator[{operator_id}]...");
        let mut latest_slot_notification_stream =
            LatestItemStream::new(new_slot_notification_stream);
        let mut acknowledgement_sender_stream = pin!(acknowledgement_sender_stream);
        loop {
            tokio::select! {
                // Ensure any new slot/block import must handle first before the `acknowledgement_sender_stream`
                // NOTE: this is only necessary for the test.
                biased;

                Some((slot, proof_of_time)) = latest_slot_notification_stream.next() => {
                    let res = bundle_producer
                        .produce_bundle(
                            operator_id,
                            OperatorSlotInfo {
                                slot,
                                proof_of_time,
                            },
                        )
                        .instrument(span.clone())
                        .await;
                    match res {
                        Err(err) => {
                            tracing::error!(?slot, ?err, "Error at producing bundle.");
                        }
                        Ok(Some(domain_proposal)) => {
                            let best_hash = consensus_client.info().best_hash;
                            let mut runtime_api = consensus_client.runtime_api();
                            runtime_api.register_extension(consensus_offchain_tx_pool_factory.offchain_transaction_pool(best_hash));

                            match domain_proposal {
                                DomainProposal::Bundle(opaque_bundle) => {
                                    if let Err(err) = runtime_api.submit_bundle_unsigned(best_hash, opaque_bundle) {
                                        tracing::error!(?slot, ?err, "Error at submitting bundle.");
                                    }
                                },
                                DomainProposal::Receipt(singleton_receipt) => {
                                    if let Err(err) = runtime_api.submit_receipt_unsigned(best_hash, singleton_receipt) {
                                        tracing::error!(?slot, ?err, "Error at submitting receipt.");
                                    }
                                },
                            }
                        }
                        Ok(None) => {}
                    }
                }
                Some(maybe_block_info) = throttled_block_import_notification_stream.next() => {
                    if let Some(block_info) = maybe_block_info {
                        if let Err(error) = bundle_processor
                            .clone()
                            .process_bundles((
                                block_info.hash,
                                block_info.number,
                                block_info.is_new_best,
                            ))
                            .instrument(span.clone())
                            .await
                        {
                            tracing::error!(?error, "Failed to process consensus block");
                            // Bring down the service as bundles processor is an essential task.
                            // TODO: more graceful shutdown.
                            break;
                        }
                    }
                }
                // In production the `acknowledgement_sender_stream` is an empty stream, it only set to
                // real stream in test
                Some(mut acknowledgement_sender) = acknowledgement_sender_stream.next() => {
                    if let Err(err) = acknowledgement_sender.send(()).await {
                        tracing::error!(
                            ?err,
                            "Failed to send acknowledgement"
                        );
                    }
                }
            }
        }
    } else {
        info!("🧑‍ Running as Full node...");
        drop(new_slot_notification_stream);
        drop(acknowledgement_sender_stream);
        while let Some(maybe_block_info) = throttled_block_import_notification_stream.next().await {
            if let Some(block_info) = maybe_block_info {
                if let Err(error) = bundle_processor
                    .clone()
                    .process_bundles((block_info.hash, block_info.number, block_info.is_new_best))
                    .instrument(span.clone())
                    .await
                {
                    tracing::error!(?error, "Failed to process consensus block");
                    // Bring down the service as bundles processor is an essential task.
                    // TODO: more graceful shutdown.
                    break;
                }
            }
        }
    }
}

/// Throttle the consensus block import notification based on the `consensus_block_import_throttling_buffer_size`
/// to pause the consensus block import in case the consensus chain runs much faster than the domain.
///
/// Return the throttled block import notification stream
#[allow(clippy::too_many_arguments)]
fn throttling_block_import_notifications<Block, CBlock, CClient, BlocksImporting, BlocksImported>(
    spawn_essential: Box<dyn SpawnEssentialNamed>,
    consensus_client: Arc<CClient>,
    mut blocks_importing: BlocksImporting,
    mut blocks_imported: BlocksImported,
    consensus_block_import_throttling_buffer_size: u32,
) -> mpsc::Receiver<Option<BlockInfo<CBlock>>>
where
    Block: BlockT,
    CBlock: BlockT,
    CClient: HeaderBackend<CBlock>
        + BlockBackend<CBlock>
        + ProvideRuntimeApi<CBlock>
        + BlockchainEvents<CBlock>
        + 'static,
    CClient::Api: DomainsApi<CBlock, Block::Header>,
    BlocksImporting: Stream<Item = (NumberFor<CBlock>, mpsc::Sender<()>)> + Unpin + Send + 'static,
    BlocksImported: Stream<Item = BlockImportNotification<CBlock>> + Unpin + Send + 'static,
{
    // The consensus chain can be ahead of the domain by up to `block_import_throttling_buffer_size/2`
    // blocks, for there are two notifications per block sent to this buffer (one will be actually
    // consumed by the domain processor, the other from `sc-consensus-subspace` is used to discontinue
    // the consensus block import in case the consensus chain runs much faster than the domain.).
    let (mut block_info_sender, block_info_receiver) = mpsc::channel::<Option<BlockInfo<CBlock>>>(
        consensus_block_import_throttling_buffer_size as usize,
    );

    spawn_essential.spawn_essential(
        "consensus-block-import-throttler",
        None,
        Box::pin(async move {
            loop {
                tokio::select! {
                    // Ensure the `blocks_imported` branch will be checked before the `blocks_importing` branch.
                    // Currently this is only necessary for the test to ensure when both `block_imported` notification
                    // and `blocks_importing` notification are arrived, the `block_imported` notification will be processed
                    // first, such that we can ensure when the `blocks_importing` acknowledgement is responded, the
                    // imported block must being processed by the executor.
                    // Please see https://github.com/autonomys/subspace/pull/1363#discussion_r1162571291 for more details.
                    biased;

                    maybe_block_imported = blocks_imported.next() => {
                        let block_imported = match maybe_block_imported {
                            Some(block_imported) => block_imported,
                            None => {
                                // Can be None on graceful shutdown.
                                break;
                            }
                        };
                        let header = match consensus_client.header(block_imported.hash) {
                            Ok(Some(header)) => header,
                            res => {
                                tracing::error!(
                                    result = ?res,
                                    header = ?block_imported.header,
                                    "Imported consensus block header not found",
                                );
                                return;
                            }
                        };
                        let block_info = BlockInfo {
                            hash: header.hash(),
                            number: *header.number(),
                            is_new_best: block_imported.is_new_best,
                        };
                        let _ = block_info_sender.feed(Some(block_info)).await;
                    }
                    maybe_block_importing = blocks_importing.next() => {
                        // TODO: remove the `block_number` from the notification since it is not used
                        let (_block_number, mut acknowledgement_sender) =
                            match maybe_block_importing {
                                Some(block_importing) => block_importing,
                                None => {
                                    // Can be None on graceful shutdown.
                                    break;
                                }
                            };
                        // Pause the consensus block import when the sink is full.
                        let _ = block_info_sender.feed(None).await;
                        let _ = acknowledgement_sender.send(()).await;
                    }
                }
            }
        }),
    );

    block_info_receiver
}

struct LatestItemStream<S: Stream> {
    inner: Pin<Box<S>>,
}

impl<S: Stream> LatestItemStream<S> {
    fn new(stream: S) -> Self {
        Self {
            inner: Box::pin(stream),
        }
    }
}

impl<S> Stream for LatestItemStream<S>
where
    S: Stream,
{
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut last_item = None;
        while let Poll::Ready(poll) = self.inner.as_mut().poll_next(cx) {
            match poll {
                Some(item) => {
                    last_item = Some(item);
                }
                None => {
                    return Poll::Ready(last_item);
                }
            }
        }

        if last_item.is_some() {
            Poll::Ready(last_item)
        } else {
            Poll::Pending
        }
    }
}
