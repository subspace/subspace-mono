//! Farming cluster cache
//!
//! Cache is responsible for caching pieces within allocated space to accelerate plotting and serve
//! pieces in response to DSN requests.
//!
//! This module exposes some data structures for NATS communication, custom piece cache
//! implementation designed to work with cluster cache and a service function to drive the backend
//! part of the cache.

use crate::cluster::controller::ClusterControllerCacheIdentifyBroadcast;
use crate::cluster::nats_client::{
    GenericBroadcast, GenericRequest, GenericStreamRequest, NatsClient,
};
use crate::farm::{CacheId, FarmError, PieceCache, PieceCacheId, PieceCacheOffset};
use anyhow::anyhow;
use async_trait::async_trait;
use futures::stream::FuturesUnordered;
use futures::{select, stream, FutureExt, Stream, StreamExt};
use parity_scale_codec::{Decode, Encode};
use std::collections::BTreeSet;
use std::pin::Pin;
use std::task::Poll;
use std::time::{Duration, Instant};
use subspace_core_primitives::pieces::{Piece, PieceIndex};
use tokio::time::MissedTickBehavior;
use tracing::{debug, error, info, info_span, trace, warn, Instrument};

const MIN_CACHE_IDENTIFICATION_INTERVAL: Duration = Duration::from_secs(1);

/// Request cache details from cache
#[derive(Debug, Clone, Encode, Decode)]
pub struct ClusterCacheDetailsRequest;

impl GenericStreamRequest for ClusterCacheDetailsRequest {
    /// `*` here stands for cache ID
    const SUBJECT: &'static str = "subspace.cache.*.details";
    type Response = ClusterPieceCacheDetails;
}

/// Cache details
#[derive(Debug, Clone, Encode, Decode)]
pub struct ClusterPieceCacheDetails {
    /// Piece Cache ID
    pub piece_cache_id: PieceCacheId,
    /// Max number of elements in this cache
    pub max_num_elements: u32,
}

/// Broadcast with identification details by caches
#[derive(Debug, Clone, Encode, Decode)]
pub struct ClusterCacheIdentifyBroadcast {
    /// Cache ID
    pub cache_id: CacheId,
}

impl GenericBroadcast for ClusterCacheIdentifyBroadcast {
    /// `*` here stands for cache group
    const SUBJECT: &'static str = "subspace.cache.*.cache-identify";
}

/// Write piece into cache
#[derive(Debug, Clone, Encode, Decode)]
struct ClusterCacheWritePieceRequest {
    offset: PieceCacheOffset,
    piece_index: PieceIndex,
    piece: Piece,
}

impl GenericRequest for ClusterCacheWritePieceRequest {
    /// `*` here stands for cache ID
    const SUBJECT: &'static str = "subspace.cache.*.write-piece";
    type Response = Result<(), String>;
}

/// Read piece index from cache
#[derive(Debug, Clone, Encode, Decode)]
struct ClusterCacheReadPieceIndexRequest {
    offset: PieceCacheOffset,
}

impl GenericRequest for ClusterCacheReadPieceIndexRequest {
    /// `*` here stands for cache ID
    const SUBJECT: &'static str = "subspace.cache.*.read-piece-index";
    type Response = Result<Option<PieceIndex>, String>;
}

/// Read piece from cache
#[derive(Debug, Clone, Encode, Decode)]
pub(super) struct ClusterCacheReadPieceRequest {
    pub(super) offset: PieceCacheOffset,
}

impl GenericRequest for ClusterCacheReadPieceRequest {
    /// `*` here stands for cache ID
    const SUBJECT: &'static str = "subspace.cache.*.read-piece";
    type Response = Result<Option<(PieceIndex, Piece)>, String>;
}

/// Read piece from cache
#[derive(Debug, Clone, Encode, Decode)]
pub(super) struct ClusterCacheReadPiecesRequest {
    pub(super) offsets: Vec<PieceCacheOffset>,
}

impl GenericStreamRequest for ClusterCacheReadPiecesRequest {
    /// `*` here stands for cache ID
    const SUBJECT: &'static str = "subspace.cache.*.read-pieces";
    type Response = Result<(PieceCacheOffset, Option<(PieceIndex, Piece)>), String>;
}

/// Collect plotted pieces from farmer
#[derive(Debug, Clone, Encode, Decode)]
struct ClusterCacheContentsRequest;

impl GenericStreamRequest for ClusterCacheContentsRequest {
    /// `*` here stands for cache ID
    const SUBJECT: &'static str = "subspace.cache.*.contents";
    type Response = Result<(PieceCacheOffset, Option<PieceIndex>), String>;
}

/// Cluster cache implementation
#[derive(Debug)]
pub struct ClusterPieceCache {
    piece_cache_id: PieceCacheId,
    piece_cache_id_string: String,
    max_num_elements: u32,
    nats_client: NatsClient,
}

#[async_trait]
impl PieceCache for ClusterPieceCache {
    fn id(&self) -> &PieceCacheId {
        &self.piece_cache_id
    }

    #[inline]
    fn max_num_elements(&self) -> u32 {
        self.max_num_elements
    }

    async fn contents(
        &self,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<(PieceCacheOffset, Option<PieceIndex>), FarmError>>
                + Unpin
                + Send
                + '_,
        >,
        FarmError,
    > {
        Ok(Box::new(
            self.nats_client
                .stream_request(
                    &ClusterCacheContentsRequest,
                    Some(&self.piece_cache_id_string),
                )
                .await?
                .map(|response| response.map_err(FarmError::from)),
        ))
    }

    async fn write_piece(
        &self,
        offset: PieceCacheOffset,
        piece_index: PieceIndex,
        piece: &Piece,
    ) -> Result<(), FarmError> {
        Ok(self
            .nats_client
            .request(
                &ClusterCacheWritePieceRequest {
                    offset,
                    piece_index,
                    piece: piece.clone(),
                },
                Some(&self.piece_cache_id_string),
            )
            .await??)
    }

    async fn read_piece_index(
        &self,
        offset: PieceCacheOffset,
    ) -> Result<Option<PieceIndex>, FarmError> {
        Ok(self
            .nats_client
            .request(
                &ClusterCacheReadPieceIndexRequest { offset },
                Some(&self.piece_cache_id_string),
            )
            .await??)
    }

    async fn read_piece(
        &self,
        offset: PieceCacheOffset,
    ) -> Result<Option<(PieceIndex, Piece)>, FarmError> {
        Ok(self
            .nats_client
            .request(
                &ClusterCacheReadPieceRequest { offset },
                Some(&self.piece_cache_id_string),
            )
            .await??)
    }

    async fn read_pieces(
        &self,
        offsets: Box<dyn Iterator<Item = PieceCacheOffset> + Send>,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<(PieceCacheOffset, Option<(PieceIndex, Piece)>), FarmError>>
                + Send
                + Unpin
                + '_,
        >,
        FarmError,
    > {
        let offsets = offsets.collect::<Vec<_>>();
        let mut offsets_set = BTreeSet::from_iter(offsets.iter().copied());
        let mut stream = self
            .nats_client
            .stream_request(
                &ClusterCacheReadPiecesRequest { offsets },
                Some(&self.piece_cache_id_string),
            )
            .await?
            .map(|response| response.map_err(FarmError::from))
            .fuse();
        Ok(Box::new(stream::poll_fn(move |cx| {
            if !stream.is_done() {
                match stream.poll_next_unpin(cx) {
                    Poll::Ready(Some(response)) => {
                        return Poll::Ready(Some(response.inspect(|(offset, _)| {
                            offsets_set.remove(offset);
                        })));
                    }
                    Poll::Ready(None) => {
                        // Handled as a general case below
                    }
                    Poll::Pending => {
                        return Poll::Pending;
                    }
                }
            }

            // Uphold invariant of the trait that some result should be returned for every unique
            // provided offset
            match offsets_set.pop_first() {
                Some(offset) => Poll::Ready(Some(Ok((offset, None)))),
                None => Poll::Ready(None),
            }
        })))
    }
}

impl ClusterPieceCache {
    /// Create new instance using information from previously received
    /// [`ClusterCacheIdentifyBroadcast`]
    #[inline]
    pub fn new(
        piece_cache_id: PieceCacheId,
        max_num_elements: u32,
        nats_client: NatsClient,
    ) -> ClusterPieceCache {
        Self {
            piece_cache_id,
            piece_cache_id_string: piece_cache_id.to_string(),
            max_num_elements,
            nats_client,
        }
    }
}

#[derive(Debug)]
struct CacheDetails<'a, C> {
    piece_cache_id: PieceCacheId,
    piece_cache_id_string: String,
    cache: &'a C,
}

impl<C> CacheDetails<'_, C>
where
    C: PieceCache,
{
    fn derive_identification(&self) -> ClusterPieceCacheDetails {
        ClusterPieceCacheDetails {
            piece_cache_id: self.piece_cache_id,
            max_num_elements: self.cache.max_num_elements(),
        }
    }
}

/// Create cache service for specified caches that will be processing incoming requests and send
/// periodic identify notifications
pub async fn cache_service<C>(
    nats_client: NatsClient,
    caches: &[C],
    cache_group: &str,
    identification_broadcast_interval: Duration,
    primary_instance: bool,
) -> anyhow::Result<()>
where
    C: PieceCache,
{
    let cache_id = CacheId::new();
    let cache_id_string = cache_id.to_string();

    let caches_details = caches
        .iter()
        .map(|cache| {
            let piece_cache_id = *cache.id();
            if primary_instance {
                info!(%piece_cache_id, max_num_elements = %cache.max_num_elements(), "Created piece cache");
            }

            CacheDetails {
                piece_cache_id,
                piece_cache_id_string: piece_cache_id.to_string(),
                cache,
            }
        })
        .collect::<Vec<_>>();

    if primary_instance {
        select! {
            result = identify_responder(
                &nats_client,
                cache_id,
                cache_group,
                identification_broadcast_interval
            ).fuse() => {
                result
            },
            result = caches_details_responder(
                &nats_client,
                &cache_id_string,
                &caches_details
            ).fuse() => {
                result
            },
            result = write_piece_responder(&nats_client, &caches_details).fuse() => {
                result
            },
            result = read_piece_index_responder(&nats_client, &caches_details).fuse() => {
                result
            },
            result = read_piece_responder(&nats_client, &caches_details).fuse() => {
                result
            },
            result = read_pieces_responder(&nats_client, &caches_details).fuse() => {
                result
            },
            result = contents_responder(&nats_client, &caches_details).fuse() => {
                result
            },
        }
    } else {
        select! {
            result = write_piece_responder(&nats_client, &caches_details).fuse() => {
                result
            },
            result = read_piece_index_responder(&nats_client, &caches_details).fuse() => {
                result
            },
            result = read_piece_responder(&nats_client, &caches_details).fuse() => {
                result
            },
            result = read_pieces_responder(&nats_client, &caches_details).fuse() => {
                result
            },
            result = contents_responder(&nats_client, &caches_details).fuse() => {
                result
            },
        }
    }
}

/// Listen for cache identification broadcast from controller and publish identification
/// broadcast in response, also send periodic notifications reminding that cache exists.
///
/// Implementation is using concurrency with multiple tokio tasks, but can be started multiple times
/// per controller instance in order to parallelize more work across threads if needed.
async fn identify_responder(
    nats_client: &NatsClient,
    cache_id: CacheId,
    cache_group: &str,
    identification_broadcast_interval: Duration,
) -> anyhow::Result<()> {
    let mut subscription = nats_client
        .subscribe_to_broadcasts::<ClusterControllerCacheIdentifyBroadcast>(Some(cache_group), None)
        .await
        .map_err(|error| {
            anyhow!("Failed to subscribe to cache identify broadcast requests: {error}")
        })?
        .fuse();

    // Also send periodic updates in addition to the subscription response
    let mut interval = tokio::time::interval(identification_broadcast_interval);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

    let mut last_identification = Instant::now();

    loop {
        select! {
            maybe_message = subscription.next() => {
                let Some(message) = maybe_message else {
                    debug!("Identify broadcast stream ended");
                    break;
                };

                trace!(?message, "Cache received identify broadcast message");

                if last_identification.elapsed() < MIN_CACHE_IDENTIFICATION_INTERVAL {
                    // Skip too frequent identification requests
                    continue;
                }

                last_identification = Instant::now();
                send_identify_broadcast(nats_client, cache_id, cache_group).await;
                interval.reset();
            }
            _ = interval.tick().fuse() => {
                last_identification = Instant::now();
                trace!("Cache self-identification");

                send_identify_broadcast(nats_client, cache_id, cache_group).await;
            }
        }
    }

    Ok(())
}

async fn send_identify_broadcast(nats_client: &NatsClient, cache_id: CacheId, cache_group: &str) {
    if let Err(error) = nats_client
        .broadcast(&ClusterCacheIdentifyBroadcast { cache_id }, cache_group)
        .await
    {
        warn!(%cache_id, %error, "Failed to send cache identify notification");
    }
}

async fn caches_details_responder<C>(
    nats_client: &NatsClient,
    cache_id_string: &str,
    caches_details: &[CacheDetails<'_, C>],
) -> anyhow::Result<()>
where
    C: PieceCache,
{
    nats_client
        .stream_request_responder(
            Some(cache_id_string),
            Some(cache_id_string.to_string()),
            |_request: ClusterCacheDetailsRequest| async {
                Some(stream::iter(
                    caches_details
                        .iter()
                        .map(CacheDetails::derive_identification),
                ))
            },
        )
        .await
}

async fn write_piece_responder<C>(
    nats_client: &NatsClient,
    caches_details: &[CacheDetails<'_, C>],
) -> anyhow::Result<()>
where
    C: PieceCache,
{
    caches_details
        .iter()
        .map(|cache_details| async move {
            nats_client
                .request_responder(
                    Some(cache_details.piece_cache_id_string.as_str()),
                    Some(cache_details.piece_cache_id_string.clone()),
                    |request: ClusterCacheWritePieceRequest| async move {
                        Some(
                            cache_details
                                .cache
                                .write_piece(request.offset, request.piece_index, &request.piece)
                                .await
                                .map_err(|error| error.to_string()),
                        )
                    },
                )
                .instrument(info_span!("", piece_cache_id = %cache_details.piece_cache_id))
                .await
        })
        .collect::<FuturesUnordered<_>>()
        .next()
        .await
        .ok_or_else(|| anyhow!("No caches"))?
}

async fn read_piece_index_responder<C>(
    nats_client: &NatsClient,
    caches_details: &[CacheDetails<'_, C>],
) -> anyhow::Result<()>
where
    C: PieceCache,
{
    caches_details
        .iter()
        .map(|cache_details| async move {
            nats_client
                .request_responder(
                    Some(cache_details.piece_cache_id_string.as_str()),
                    Some(cache_details.piece_cache_id_string.clone()),
                    |request: ClusterCacheReadPieceIndexRequest| async move {
                        Some(
                            cache_details
                                .cache
                                .read_piece_index(request.offset)
                                .await
                                .map_err(|error| error.to_string()),
                        )
                    },
                )
                .instrument(info_span!("", piece_cache_id = %cache_details.piece_cache_id))
                .await
        })
        .collect::<FuturesUnordered<_>>()
        .next()
        .await
        .ok_or_else(|| anyhow!("No caches"))?
}

async fn read_piece_responder<C>(
    nats_client: &NatsClient,
    caches_details: &[CacheDetails<'_, C>],
) -> anyhow::Result<()>
where
    C: PieceCache,
{
    caches_details
        .iter()
        .map(|cache_details| async move {
            nats_client
                .request_responder(
                    Some(cache_details.piece_cache_id_string.as_str()),
                    Some(cache_details.piece_cache_id_string.clone()),
                    |request: ClusterCacheReadPieceRequest| async move {
                        Some(
                            cache_details
                                .cache
                                .read_piece(request.offset)
                                .await
                                .map_err(|error| error.to_string()),
                        )
                    },
                )
                .instrument(info_span!("", piece_cache_id = %cache_details.piece_cache_id))
                .await
        })
        .collect::<FuturesUnordered<_>>()
        .next()
        .await
        .ok_or_else(|| anyhow!("No caches"))?
}

async fn read_pieces_responder<C>(
    nats_client: &NatsClient,
    caches_details: &[CacheDetails<'_, C>],
) -> anyhow::Result<()>
where
    C: PieceCache,
{
    caches_details
        .iter()
        .map(|cache_details| async move {
            nats_client
                .stream_request_responder::<_, _, Pin<Box<dyn Stream<Item = _> + Send>>, _>(
                    Some(cache_details.piece_cache_id_string.as_str()),
                    Some(cache_details.piece_cache_id_string.clone()),
                    |ClusterCacheReadPiecesRequest { offsets }| async move {
                        Some(
                            match cache_details
                                .cache
                                .read_pieces(Box::new(offsets.into_iter()))
                                .await
                            {
                                Ok(contents) => Box::pin(contents.map(|maybe_cache_element| {
                                    maybe_cache_element.map_err(|error| error.to_string())
                                })) as _,
                                Err(error) => {
                                    error!(%error, "Failed to read pieces");

                                    Box::pin(stream::once(async move {
                                        Err(format!("Failed to read pieces: {error}"))
                                    })) as _
                                }
                            },
                        )
                    },
                )
                .instrument(info_span!("", piece_cache_id = %cache_details.piece_cache_id))
                .await
        })
        .collect::<FuturesUnordered<_>>()
        .next()
        .await
        .ok_or_else(|| anyhow!("No caches"))?
}

async fn contents_responder<C>(
    nats_client: &NatsClient,
    caches_details: &[CacheDetails<'_, C>],
) -> anyhow::Result<()>
where
    C: PieceCache,
{
    caches_details
        .iter()
        .map(|cache_details| async move {
            nats_client
                .stream_request_responder::<_, _, Pin<Box<dyn Stream<Item = _> + Send>>, _>(
                    Some(cache_details.piece_cache_id_string.as_str()),
                    Some(cache_details.piece_cache_id_string.clone()),
                    |_request: ClusterCacheContentsRequest| async move {
                        Some(match cache_details.cache.contents().await {
                            Ok(contents) => Box::pin(contents.map(|maybe_cache_element| {
                                maybe_cache_element.map_err(|error| error.to_string())
                            })) as _,
                            Err(error) => {
                                error!(%error, "Failed to get contents");

                                Box::pin(stream::once(async move {
                                    Err(format!("Failed to get contents: {error}"))
                                })) as _
                            }
                        })
                    },
                )
                .instrument(info_span!("", piece_cache_id = %cache_details.piece_cache_id))
                .await
        })
        .collect::<FuturesUnordered<_>>()
        .next()
        .await
        .ok_or_else(|| anyhow!("No caches"))?
}
