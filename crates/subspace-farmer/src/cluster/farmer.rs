//! Farming cluster farmer
//!
//! Farmer is responsible for maintaining farms, doing audits and generating proofs when solution is
//! found in one of the plots.
//!
//! This module exposes some data structures for NATS communication, custom farm implementation
//! designed to work with cluster farmer and a service function to drive the backend part
//! of the farmer.

use crate::cluster::controller::ClusterControllerFarmerIdentifyBroadcast;
use crate::cluster::nats_client::{
    GenericBroadcast, GenericRequest, GenericStreamRequest, NatsClient,
};
use crate::farm::{
    Farm, FarmError, FarmId, FarmerId, FarmingNotification, HandlerFn, HandlerId, PieceReader,
    PlottedSectors, SectorUpdate,
};
use crate::utils::AsyncJoinOnDrop;
use anyhow::anyhow;
use async_trait::async_trait;
use event_listener_primitives::Bag;
use futures::channel::mpsc;
use futures::stream::FuturesUnordered;
use futures::{select, stream, FutureExt, Stream, StreamExt};
use parity_scale_codec::{Decode, Encode};
use std::future::Future;
use std::pin::{pin, Pin};
use std::sync::Arc;
use std::time::{Duration, Instant};
use subspace_core_primitives::hashes::{blake3_hash_list, Blake3Hash};
use subspace_core_primitives::pieces::{Piece, PieceOffset};
use subspace_core_primitives::sectors::SectorIndex;
use subspace_farmer_components::plotting::PlottedSector;
use subspace_rpc_primitives::SolutionResponse;
use tokio::time::MissedTickBehavior;
use tracing::{debug, error, info_span, trace, warn, Instrument};

const BROADCAST_NOTIFICATIONS_BUFFER: usize = 1000;
const MIN_FARMER_IDENTIFICATION_INTERVAL: Duration = Duration::from_secs(1);

type Handler<A> = Bag<HandlerFn<A>, A>;

/// Broadcast with cluster farmer id for identification
#[derive(Debug, Clone, Encode, Decode)]
pub struct ClusterFarmerIdentifyBroadcast {
    /// Cluster farmer ID
    pub farmer_id: FarmerId,
    /// Farmer fingerprint changes when something about internal farm changes (like allocated space)
    pub fingerprint: Blake3Hash,
}

impl GenericBroadcast for ClusterFarmerIdentifyBroadcast {
    /// `*` here stands for cluster farmer ID
    const SUBJECT: &'static str = "subspace.farmer.*.farmer-identify";
}

/// Request farm details from farmer
#[derive(Debug, Clone, Encode, Decode)]
pub struct ClusterFarmerFarmDetailsRequest;

impl GenericStreamRequest for ClusterFarmerFarmDetailsRequest {
    /// `*` here stands for cluster farmer ID
    const SUBJECT: &'static str = "subspace.farmer.*.farm.details";
    type Response = ClusterFarmerFarmDetails;
}

/// Farm details
#[derive(Debug, Clone, Encode, Decode)]
pub struct ClusterFarmerFarmDetails {
    /// Farm ID
    pub farm_id: FarmId,
    /// Total number of sectors in the farm
    pub total_sectors_count: SectorIndex,
    /// Farm fingerprint changes when something about farm changes (like allocated space)
    pub fingerprint: Blake3Hash,
}

/// Broadcast with sector updates by farmers
#[derive(Debug, Clone, Encode, Decode)]
struct ClusterFarmerSectorUpdateBroadcast {
    /// Farm ID
    farm_id: FarmId,
    /// Sector index
    sector_index: SectorIndex,
    /// Sector update
    sector_update: SectorUpdate,
}

impl GenericBroadcast for ClusterFarmerSectorUpdateBroadcast {
    /// `*` here stands for single farm ID
    const SUBJECT: &'static str = "subspace.farmer.*.sector-update";
}

/// Broadcast with farming notifications by farmers
#[derive(Debug, Clone, Encode, Decode)]
struct ClusterFarmerFarmingNotificationBroadcast {
    /// Farm ID
    farm_id: FarmId,
    /// Farming notification
    farming_notification: FarmingNotification,
}

impl GenericBroadcast for ClusterFarmerFarmingNotificationBroadcast {
    /// `*` here stands for single farm ID
    const SUBJECT: &'static str = "subspace.farmer.*.farming-notification";
}

/// Broadcast with solutions by farmers
#[derive(Debug, Clone, Encode, Decode)]
struct ClusterFarmerSolutionBroadcast {
    /// Farm ID
    farm_id: FarmId,
    /// Solution response
    solution_response: SolutionResponse,
}

impl GenericBroadcast for ClusterFarmerSolutionBroadcast {
    /// `*` here stands for single farm ID
    const SUBJECT: &'static str = "subspace.farmer.*.solution-response";
}

/// Read piece from farm
#[derive(Debug, Clone, Encode, Decode)]
struct ClusterFarmerReadPieceRequest {
    sector_index: SectorIndex,
    piece_offset: PieceOffset,
}

impl GenericRequest for ClusterFarmerReadPieceRequest {
    /// `*` here stands for single farm ID
    const SUBJECT: &'static str = "subspace.farmer.*.farm.read-piece";
    type Response = Result<Option<Piece>, String>;
}

/// Request plotted sectors from farmer
#[derive(Debug, Clone, Encode, Decode)]
struct ClusterFarmerPlottedSectorsRequest;

impl GenericStreamRequest for ClusterFarmerPlottedSectorsRequest {
    /// `*` here stands for single farm ID
    const SUBJECT: &'static str = "subspace.farmer.*.farm.plotted-sectors";
    type Response = Result<PlottedSector, String>;
}

#[derive(Debug)]
struct ClusterPlottedSectors {
    farm_id_string: String,
    nats_client: NatsClient,
}

#[async_trait]
impl PlottedSectors for ClusterPlottedSectors {
    async fn get(
        &self,
    ) -> Result<
        Box<dyn Stream<Item = Result<PlottedSector, FarmError>> + Unpin + Send + '_>,
        FarmError,
    > {
        Ok(Box::new(
            self.nats_client
                .stream_request(
                    &ClusterFarmerPlottedSectorsRequest,
                    Some(&self.farm_id_string),
                )
                .await?
                .map(|response| response.map_err(FarmError::from)),
        ))
    }
}

#[derive(Debug)]
struct ClusterPieceReader {
    farm_id_string: String,
    nats_client: NatsClient,
}

#[async_trait]
impl PieceReader for ClusterPieceReader {
    async fn read_piece(
        &self,
        sector_index: SectorIndex,
        piece_offset: PieceOffset,
    ) -> Result<Option<Piece>, FarmError> {
        Ok(self
            .nats_client
            .request(
                &ClusterFarmerReadPieceRequest {
                    sector_index,
                    piece_offset,
                },
                Some(&self.farm_id_string),
            )
            .await??)
    }
}

#[derive(Default, Debug)]
struct Handlers {
    sector_update: Handler<(SectorIndex, SectorUpdate)>,
    farming_notification: Handler<FarmingNotification>,
    solution: Handler<SolutionResponse>,
}

/// Cluster farm implementation
#[derive(Debug)]
pub struct ClusterFarm {
    farm_id: FarmId,
    farm_id_string: String,
    total_sectors_count: SectorIndex,
    nats_client: NatsClient,
    handlers: Arc<Handlers>,
    background_tasks: AsyncJoinOnDrop<()>,
}

#[async_trait(?Send)]
impl Farm for ClusterFarm {
    fn id(&self) -> &FarmId {
        &self.farm_id
    }

    fn total_sectors_count(&self) -> SectorIndex {
        self.total_sectors_count
    }

    fn plotted_sectors(&self) -> Arc<dyn PlottedSectors + 'static> {
        Arc::new(ClusterPlottedSectors {
            farm_id_string: self.farm_id_string.clone(),
            nats_client: self.nats_client.clone(),
        })
    }

    fn piece_reader(&self) -> Arc<dyn PieceReader + 'static> {
        Arc::new(ClusterPieceReader {
            farm_id_string: self.farm_id_string.clone(),
            nats_client: self.nats_client.clone(),
        })
    }

    fn on_sector_update(
        &self,
        callback: HandlerFn<(SectorIndex, SectorUpdate)>,
    ) -> Box<dyn HandlerId> {
        Box::new(self.handlers.sector_update.add(callback))
    }

    fn on_farming_notification(
        &self,
        callback: HandlerFn<FarmingNotification>,
    ) -> Box<dyn HandlerId> {
        Box::new(self.handlers.farming_notification.add(callback))
    }

    fn on_solution(&self, callback: HandlerFn<SolutionResponse>) -> Box<dyn HandlerId> {
        Box::new(self.handlers.solution.add(callback))
    }

    fn run(self: Box<Self>) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>> {
        Box::pin((*self).run())
    }
}

impl ClusterFarm {
    /// Create new instance using information from previously received
    /// [`ClusterFarmerIdentifyBroadcast`]
    pub async fn new(
        farm_id: FarmId,
        total_sectors_count: SectorIndex,
        nats_client: NatsClient,
    ) -> anyhow::Result<Self> {
        let farm_id_string = farm_id.to_string();

        let sector_updates_subscription = nats_client
            .subscribe_to_broadcasts::<ClusterFarmerSectorUpdateBroadcast>(
                Some(&farm_id_string),
                None,
            )
            .await
            .map_err(|error| anyhow!("Failed to subscribe to sector updates broadcast: {error}"))?;
        let farming_notifications_subscription = nats_client
            .subscribe_to_broadcasts::<ClusterFarmerFarmingNotificationBroadcast>(
                Some(&farm_id_string),
                None,
            )
            .await
            .map_err(|error| {
                anyhow!("Failed to subscribe to farming notifications broadcast: {error}")
            })?;
        let solution_subscription = nats_client
            .subscribe_to_broadcasts::<ClusterFarmerSolutionBroadcast>(Some(&farm_id_string), None)
            .await
            .map_err(|error| {
                anyhow!("Failed to subscribe to solution responses broadcast: {error}")
            })?;

        let handlers = Arc::<Handlers>::default();
        // Run background tasks and fire corresponding notifications
        let background_tasks = {
            let handlers = Arc::clone(&handlers);

            async move {
                let mut sector_updates_subscription = pin!(sector_updates_subscription);
                let mut farming_notifications_subscription =
                    pin!(farming_notifications_subscription);
                let mut solution_subscription = pin!(solution_subscription);

                let sector_updates_fut = async {
                    while let Some(ClusterFarmerSectorUpdateBroadcast {
                        sector_index,
                        sector_update,
                        ..
                    }) = sector_updates_subscription.next().await
                    {
                        handlers
                            .sector_update
                            .call_simple(&(sector_index, sector_update));
                    }
                };
                let farming_notifications_fut = async {
                    while let Some(ClusterFarmerFarmingNotificationBroadcast {
                        farming_notification,
                        ..
                    }) = farming_notifications_subscription.next().await
                    {
                        handlers
                            .farming_notification
                            .call_simple(&farming_notification);
                    }
                };
                let solutions_fut = async {
                    while let Some(ClusterFarmerSolutionBroadcast {
                        solution_response, ..
                    }) = solution_subscription.next().await
                    {
                        handlers.solution.call_simple(&solution_response);
                    }
                };

                select! {
                    _ = sector_updates_fut.fuse() => {}
                    _ = farming_notifications_fut.fuse() => {}
                    _ = solutions_fut.fuse() => {}
                }
            }
        };

        Ok(Self {
            farm_id,
            farm_id_string,
            total_sectors_count,
            nats_client,
            handlers,
            background_tasks: AsyncJoinOnDrop::new(tokio::spawn(background_tasks), true),
        })
    }

    /// Run and wait for background tasks to exit or return an error
    pub async fn run(self) -> anyhow::Result<()> {
        Ok(self.background_tasks.await?)
    }
}

#[derive(Debug)]
struct FarmDetails {
    farm_id: FarmId,
    farm_id_string: String,
    total_sectors_count: SectorIndex,
    piece_reader: Arc<dyn PieceReader + 'static>,
    plotted_sectors: Arc<dyn PlottedSectors + 'static>,
    _background_tasks: Option<AsyncJoinOnDrop<()>>,
}

impl FarmDetails {
    fn derive_identification(&self) -> ClusterFarmerFarmDetails {
        ClusterFarmerFarmDetails {
            farm_id: self.farm_id,
            total_sectors_count: self.total_sectors_count,
            fingerprint: self.derive_fingerprint(),
        }
    }

    fn derive_fingerprint(&self) -> Blake3Hash {
        blake3_hash_list(&[
            &self.farm_id.encode(),
            &self.total_sectors_count.to_le_bytes(),
        ])
    }
}

/// Create farmer service for specified farms that will be processing incoming requests and send
/// periodic identify notifications.
///
/// Implementation is using concurrency with multiple tokio tasks, but can be started multiple times
/// per controller instance in order to parallelize more work across threads if needed.
pub fn farmer_service<F>(
    nats_client: NatsClient,
    farms: &[F],
    identification_broadcast_interval: Duration,
    primary_instance: bool,
) -> impl Future<Output = anyhow::Result<()>> + Send + 'static
where
    F: Farm,
{
    let farmer_id = FarmerId::new();
    let farmer_id_string = farmer_id.to_string();

    // For each farm start forwarding notifications as broadcast messages and create farm details
    // that can be used to respond to incoming requests
    let farms_details = farms
        .iter()
        .map(|farm| {
            let farm_id = *farm.id();
            let nats_client = nats_client.clone();

            let background_tasks = if primary_instance {
                let (sector_updates_sender, mut sector_updates_receiver) =
                    mpsc::channel(BROADCAST_NOTIFICATIONS_BUFFER);
                let (farming_notifications_sender, mut farming_notifications_receiver) =
                    mpsc::channel(BROADCAST_NOTIFICATIONS_BUFFER);
                let (solutions_sender, mut solutions_receiver) =
                    mpsc::channel(BROADCAST_NOTIFICATIONS_BUFFER);

                let sector_updates_handler_id =
                    farm.on_sector_update(Arc::new(move |(sector_index, sector_update)| {
                        if let Err(error) = sector_updates_sender.clone().try_send(
                            ClusterFarmerSectorUpdateBroadcast {
                                farm_id,
                                sector_index: *sector_index,
                                sector_update: sector_update.clone(),
                            },
                        ) {
                            warn!(%farm_id, %error, "Failed to send sector update notification");
                        }
                    }));

                let farming_notifications_handler_id =
                    farm.on_farming_notification(Arc::new(move |farming_notification| {
                        if let Err(error) = farming_notifications_sender.clone().try_send(
                            ClusterFarmerFarmingNotificationBroadcast {
                                farm_id,
                                farming_notification: farming_notification.clone(),
                            },
                        ) {
                            warn!(%farm_id, %error, "Failed to send farming notification");
                        }
                    }));

                let solutions_handler_id = farm.on_solution(Arc::new(move |solution_response| {
                    if let Err(error) =
                        solutions_sender
                            .clone()
                            .try_send(ClusterFarmerSolutionBroadcast {
                                farm_id,
                                solution_response: solution_response.clone(),
                            })
                    {
                        warn!(%farm_id, %error, "Failed to send solution notification");
                    }
                }));

                Some(AsyncJoinOnDrop::new(
                    tokio::spawn(async move {
                        let farm_id_string = farm_id.to_string();

                        let sector_updates_fut = async {
                            while let Some(broadcast) = sector_updates_receiver.next().await {
                                if let Err(error) =
                                    nats_client.broadcast(&broadcast, &farm_id_string).await
                                {
                                    warn!(%farm_id, %error, "Failed to broadcast sector update");
                                }
                            }
                        };
                        let farming_notifications_fut = async {
                            while let Some(broadcast) = farming_notifications_receiver.next().await
                            {
                                if let Err(error) =
                                    nats_client.broadcast(&broadcast, &farm_id_string).await
                                {
                                    warn!(
                                        %farm_id,
                                        %error,
                                        "Failed to broadcast farming notification"
                                    );
                                }
                            }
                        };
                        let solutions_fut = async {
                            while let Some(broadcast) = solutions_receiver.next().await {
                                if let Err(error) =
                                    nats_client.broadcast(&broadcast, &farm_id_string).await
                                {
                                    warn!(%farm_id, %error, "Failed to broadcast solution");
                                }
                            }
                        };

                        select! {
                            _ = sector_updates_fut.fuse() => {}
                            _ = farming_notifications_fut.fuse() => {}
                            _ = solutions_fut.fuse() => {}
                        }

                        drop(sector_updates_handler_id);
                        drop(farming_notifications_handler_id);
                        drop(solutions_handler_id);
                    }),
                    true,
                ))
            } else {
                None
            };

            FarmDetails {
                farm_id,
                farm_id_string: farm_id.to_string(),
                total_sectors_count: farm.total_sectors_count(),
                piece_reader: farm.piece_reader(),
                plotted_sectors: farm.plotted_sectors(),
                _background_tasks: background_tasks,
            }
        })
        .collect::<Vec<_>>();

    async move {
        if primary_instance {
            select! {
                result = identify_responder(
                    &nats_client,
                    farmer_id,
                    &farmer_id_string,
                    &farms_details,
                    identification_broadcast_interval
                ).fuse() => {
                    result
                },
                result = farms_details_responder(
                    &nats_client,
                    &farmer_id_string,
                    &farms_details
                ).fuse() => {
                    result
                },
                result = plotted_sectors_responder(&nats_client, &farms_details).fuse() => {
                    result
                },
                result = read_piece_responder(&nats_client, &farms_details).fuse() => {
                    result
                },
            }
        } else {
            select! {
                result = plotted_sectors_responder(&nats_client, &farms_details).fuse() => {
                    result
                },
                result = read_piece_responder(&nats_client, &farms_details).fuse() => {
                    result
                },
            }
        }
    }
}

/// Listen for farmer identification broadcast from controller and publish identification
/// broadcast in response, also send periodic notifications reminding that farm exists
async fn identify_responder(
    nats_client: &NatsClient,
    farmer_id: FarmerId,
    farmer_id_string: &str,
    farms_details: &[FarmDetails],
    identification_broadcast_interval: Duration,
) -> anyhow::Result<()> {
    let mut subscription = nats_client
        .subscribe_to_broadcasts::<ClusterControllerFarmerIdentifyBroadcast>(None, None)
        .await
        .map_err(|error| {
            anyhow!("Failed to subscribe to farmer identify broadcast requests: {error}")
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

                trace!(?message, "Farmer received identify broadcast message");

                if last_identification.elapsed() < MIN_FARMER_IDENTIFICATION_INTERVAL {
                    // Skip too frequent identification requests
                    continue;
                }

                last_identification = Instant::now();
                send_identify_broadcast(nats_client, farmer_id, farmer_id_string, farms_details).await;
                interval.reset();
            }
            _ = interval.tick().fuse() => {
                last_identification = Instant::now();
                trace!("Farmer self-identification");

                send_identify_broadcast(nats_client, farmer_id, farmer_id_string, farms_details).await;
            }
        }
    }

    Ok(())
}

async fn send_identify_broadcast(
    nats_client: &NatsClient,
    farmer_id: FarmerId,
    farmer_id_string: &str,
    farms_details: &[FarmDetails],
) {
    if let Err(error) = nats_client
        .broadcast(
            &new_identify_message(farmer_id, farms_details),
            farmer_id_string,
        )
        .await
    {
        warn!(%farmer_id, %error, "Failed to send farmer identify notification");
    }
}

fn new_identify_message(
    farmer_id: FarmerId,
    farms_details: &[FarmDetails],
) -> ClusterFarmerIdentifyBroadcast {
    let farmer_id_bytes = farmer_id.encode();
    let farms_sectors_counts = farms_details
        .iter()
        .map(|farm_details| farm_details.total_sectors_count.to_le_bytes())
        .collect::<Vec<_>>();
    let mut farms_sectors_counts = farms_sectors_counts
        .iter()
        .map(AsRef::as_ref)
        .collect::<Vec<_>>();
    farms_sectors_counts.push(farmer_id_bytes.as_slice());
    let fingerprint = blake3_hash_list(farms_sectors_counts.as_slice());

    ClusterFarmerIdentifyBroadcast {
        farmer_id,
        fingerprint,
    }
}

async fn farms_details_responder(
    nats_client: &NatsClient,
    farmer_id_string: &str,
    farms_details: &[FarmDetails],
) -> anyhow::Result<()> {
    nats_client
        .stream_request_responder(
            Some(farmer_id_string),
            Some(farmer_id_string.to_string()),
            |_request: ClusterFarmerFarmDetailsRequest| async {
                Some(stream::iter(
                    farms_details.iter().map(FarmDetails::derive_identification),
                ))
            },
        )
        .await
}

async fn plotted_sectors_responder(
    nats_client: &NatsClient,
    farms_details: &[FarmDetails],
) -> anyhow::Result<()> {
    farms_details
        .iter()
        .map(|farm_details| async move {
            nats_client
                .stream_request_responder::<_, _, Pin<Box<dyn Stream<Item = _> + Send>>, _>(
                    Some(&farm_details.farm_id_string),
                    Some(farm_details.farm_id_string.clone()),
                    |_request: ClusterFarmerPlottedSectorsRequest| async move {
                        Some(match farm_details.plotted_sectors.get().await {
                            Ok(plotted_sectors) => {
                                Box::pin(plotted_sectors.map(|maybe_plotted_sector| {
                                    maybe_plotted_sector.map_err(|error| error.to_string())
                                })) as _
                            }
                            Err(error) => {
                                error!(
                                    %error,
                                    farm_id = %farm_details.farm_id,
                                    "Failed to get plotted sectors"
                                );

                                Box::pin(stream::once(async move {
                                    Err(format!("Failed to get plotted sectors: {error}"))
                                })) as _
                            }
                        })
                    },
                )
                .instrument(info_span!("", cache_id = %farm_details.farm_id))
                .await
        })
        .collect::<FuturesUnordered<_>>()
        .next()
        .await
        .ok_or_else(|| anyhow!("No farms"))?
}

async fn read_piece_responder(
    nats_client: &NatsClient,
    farms_details: &[FarmDetails],
) -> anyhow::Result<()> {
    farms_details
        .iter()
        .map(|farm_details| async move {
            nats_client
                .request_responder(
                    Some(farm_details.farm_id_string.as_str()),
                    Some(farm_details.farm_id_string.clone()),
                    |request: ClusterFarmerReadPieceRequest| async move {
                        Some(
                            farm_details
                                .piece_reader
                                .read_piece(request.sector_index, request.piece_offset)
                                .await
                                .map_err(|error| error.to_string()),
                        )
                    },
                )
                .instrument(info_span!("", cache_id = %farm_details.farm_id))
                .await
        })
        .collect::<FuturesUnordered<_>>()
        .next()
        .await
        .ok_or_else(|| anyhow!("No farms"))?
}
