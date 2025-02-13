//! This module exposed implementation of farms maintenance.
//!
//! The goal is to observe farms in a cluster and keep controller's data structures
//! about which pieces are plotted in which sectors of which farm up to date. Implementation
//! automatically handles dynamic farm addition and removal, etc.

#[cfg(test)]
mod tests;

use crate::cluster::controller::ClusterControllerFarmerIdentifyBroadcast;
use crate::cluster::farmer::{ClusterFarm, ClusterFarmerIdentifyFarmBroadcast};
use crate::cluster::nats_client::NatsClient;
use crate::farm::plotted_pieces::PlottedPieces;
use crate::farm::{Farm, FarmId, SectorPlottingDetails, SectorUpdate};
use anyhow::anyhow;
use async_lock::RwLock as AsyncRwLock;
use futures::channel::oneshot;
use futures::stream::{FusedStream, FuturesUnordered};
use futures::{select, FutureExt, Stream, StreamExt};
use parking_lot::Mutex;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::mem;
use std::pin::{pin, Pin};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use subspace_core_primitives::hashes::Blake3Hash;
use subspace_core_primitives::sectors::SectorIndex;
use tokio::task;
use tokio::time::MissedTickBehavior;
use tokio_stream::StreamMap;
use tracing::{error, info, trace, warn};

/// Number of farms in a cluster is currently limited to 2^16
pub type FarmIndex = u16;

type AddRemoveResult = Option<(FarmIndex, oneshot::Receiver<()>, ClusterFarm)>;
type AddRemoveFuture<'a, R> = Pin<Box<dyn Future<Output = R> + 'a>>;
type AddRemoveStream<'a, R> = Pin<Box<dyn Stream<Item = R> + Unpin + 'a>>;

/// A FarmsAddRemovetreamMap that keeps track of futures that are currently being processed for each `FarmIndex`.
struct FarmsAddRemoveStreamMap<'a, R> {
    in_progress: StreamMap<FarmIndex, AddRemoveStream<'a, R>>,
    farms_to_add_remove: HashMap<FarmIndex, VecDeque<AddRemoveFuture<'a, R>>>,
    is_terminated: bool,
}

impl<R> Default for FarmsAddRemoveStreamMap<'_, R> {
    fn default() -> Self {
        Self {
            in_progress: StreamMap::default(),
            farms_to_add_remove: HashMap::default(),
            is_terminated: true,
        }
    }
}

impl<'a, R: 'a> FarmsAddRemoveStreamMap<'a, R> {
    /// When pushing a new task, it first checks if there is already a future for the given `FarmIndex` in `in_progress`.
    ///   - If there is, the task is added to `farms_to_add_remove`.
    ///   - If not, the task is directly added to `in_progress`.
    fn push(&mut self, farm_index: FarmIndex, fut: AddRemoveFuture<'a, R>) {
        // Reset termination flag since there are new task to execute
        self.is_terminated = false;

        if self.in_progress.contains_key(&farm_index) {
            let queue = self.farms_to_add_remove.entry(farm_index).or_default();
            queue.push_back(fut);
        } else {
            self.in_progress
                .insert(farm_index, Box::pin(fut.into_stream()) as _);
        }
    }

    /// Polls the next entry in `in_progress` and moves the next task from `farms_to_add_remove` to `in_progress` if there is any.
    /// If there are no more tasks to execute, returns `None`.
    fn poll_next_entry(&mut self, cx: &mut Context<'_>) -> Poll<Option<R>> {
        if let Some((farm_index, res)) = std::task::ready!(self.in_progress.poll_next_unpin(cx)) {
            // Current task completed, remove from in_progress queue, check if there are more tasks to execute
            self.in_progress.remove(&farm_index);
            if self.farms_to_add_remove.is_empty() && self.in_progress.is_empty() {
                // No more tasks to execute
                self.is_terminated = true;
                return Poll::Ready(Some(res));
            }

            let Some(mut next_entry) = self.farms_to_add_remove.remove(&farm_index) else {
                // Current index no more tasks to execute
                return Poll::Ready(Some(res));
            };
            if let Some(fut) = next_entry.pop_front() {
                self.in_progress
                    .insert(farm_index, Box::pin(fut.into_stream()) as _);
            }

            // Re-insert back into farms_to_add_remove if there are more tasks to execute
            if !next_entry.is_empty() {
                self.farms_to_add_remove.insert(farm_index, next_entry);
            }

            Poll::Ready(Some(res))
        } else {
            // All tasks completed
            if self.farms_to_add_remove.is_empty() {
                // No more tasks to execute
                self.is_terminated = true;
                return Poll::Ready(None);
            }

            // Push tasks into in_progress queue
            for (farm_index, futs) in self.farms_to_add_remove.iter_mut() {
                if let Some(fut) = futs.pop_front() {
                    self.in_progress
                        .insert(*farm_index, Box::pin(fut.into_stream()) as _);
                }
            }
            Poll::Pending
        }
    }
}

impl<'a, R: 'a> Stream for FarmsAddRemoveStreamMap<'a, R> {
    type Item = R;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.poll_next_entry(cx)
    }
}

impl<'a, R: 'a> FusedStream for FarmsAddRemoveStreamMap<'a, R> {
    fn is_terminated(&self) -> bool {
        self.is_terminated
    }
}

#[derive(Debug)]
struct KnownFarm {
    farm_id: FarmId,
    fingerprint: Blake3Hash,
    last_identification: Instant,
    expired_sender: oneshot::Sender<()>,
}

enum KnownFarmInsertResult {
    Inserted {
        farm_index: FarmIndex,
        expired_receiver: oneshot::Receiver<()>,
    },
    FingerprintUpdated {
        farm_index: FarmIndex,
        expired_receiver: oneshot::Receiver<()>,
    },
    NotInserted,
}

#[derive(Debug)]
struct KnownFarms {
    identification_broadcast_interval: Duration,
    known_farms: HashMap<FarmIndex, KnownFarm>,
}

impl KnownFarms {
    fn new(identification_broadcast_interval: Duration) -> Self {
        Self {
            identification_broadcast_interval,
            known_farms: HashMap::new(),
        }
    }

    fn insert_or_update(
        &mut self,
        farm_id: FarmId,
        fingerprint: Blake3Hash,
    ) -> KnownFarmInsertResult {
        if let Some(existing_result) =
            self.known_farms
                .iter_mut()
                .find_map(|(&farm_index, known_farm)| {
                    if known_farm.farm_id == farm_id {
                        if known_farm.fingerprint == fingerprint {
                            known_farm.last_identification = Instant::now();
                            Some(KnownFarmInsertResult::NotInserted)
                        } else {
                            let (expired_sender, expired_receiver) = oneshot::channel();

                            known_farm.fingerprint = fingerprint;
                            known_farm.expired_sender = expired_sender;

                            Some(KnownFarmInsertResult::FingerprintUpdated {
                                farm_index,
                                expired_receiver,
                            })
                        }
                    } else {
                        None
                    }
                })
        {
            return existing_result;
        }

        for farm_index in FarmIndex::MIN..=FarmIndex::MAX {
            if let Entry::Vacant(entry) = self.known_farms.entry(farm_index) {
                let (expired_sender, expired_receiver) = oneshot::channel();

                entry.insert(KnownFarm {
                    farm_id,
                    fingerprint,
                    last_identification: Instant::now(),
                    expired_sender,
                });

                return KnownFarmInsertResult::Inserted {
                    farm_index,
                    expired_receiver,
                };
            }
        }

        warn!(%farm_id, max_supported_farm_index = %FarmIndex::MAX, "Too many farms, ignoring");
        KnownFarmInsertResult::NotInserted
    }

    fn remove_expired(&mut self) -> impl Iterator<Item = (FarmIndex, KnownFarm)> + '_ {
        self.known_farms.extract_if(|_farm_index, known_farm| {
            known_farm.last_identification.elapsed() > self.identification_broadcast_interval * 2
        })
    }

    fn remove(&mut self, farm_index: FarmIndex) {
        self.known_farms.remove(&farm_index);
    }
}

/// Utility function for maintaining farms by controller in a cluster environment
pub async fn maintain_farms(
    instance: &str,
    nats_client: &NatsClient,
    plotted_pieces: &Arc<AsyncRwLock<PlottedPieces<FarmIndex>>>,
    identification_broadcast_interval: Duration,
) -> anyhow::Result<()> {
    let mut known_farms = KnownFarms::new(identification_broadcast_interval);
    // Stream map for adding/removing farms
    let mut farms_to_add_remove = FarmsAddRemoveStreamMap::default();
    let mut farms = FuturesUnordered::new();

    let farmer_identify_subscription = pin!(nats_client
        .subscribe_to_broadcasts::<ClusterFarmerIdentifyFarmBroadcast>(None, None)
        .await
        .map_err(|error| anyhow!(
            "Failed to subscribe to farmer identify farm broadcast: {error}"
        ))?);

    // Request farmer to identify themselves
    if let Err(error) = nats_client
        .broadcast(&ClusterControllerFarmerIdentifyBroadcast, instance)
        .await
    {
        warn!(%error, "Failed to send farmer identification broadcast");
    }

    let mut farmer_identify_subscription = farmer_identify_subscription.fuse();
    let mut farm_pruning_interval = tokio::time::interval_at(
        (Instant::now() + identification_broadcast_interval * 2).into(),
        identification_broadcast_interval * 2,
    );
    farm_pruning_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        select! {
            (farm_index, result) = farms.select_next_some() => {
                known_farms.remove(farm_index);
                farms_to_add_remove.push(farm_index, Box::pin(async move {
                    let plotted_pieces = Arc::clone(plotted_pieces);

                    let delete_farm_fut = task::spawn_blocking(move || {
                        plotted_pieces.write_blocking().delete_farm(farm_index);
                    });
                    if let Err(error) = delete_farm_fut.await {
                        error!(
                            %farm_index,
                            %error,
                            "Failed to delete farm that exited"
                        );
                    }

                    None
                }));

                match result {
                    Ok(()) => {
                        info!(%farm_index, "Farm exited successfully");
                    }
                    Err(error) => {
                        error!(%farm_index, %error, "Farm exited with error");
                    }
                }
            }
            maybe_identify_message = farmer_identify_subscription.next() => {
                let Some(identify_message) = maybe_identify_message else {
                    return Err(anyhow!("Farmer identify stream ended"));
                };

                process_farm_identify_message(
                    identify_message,
                    nats_client,
                    &mut known_farms,
                    &mut farms_to_add_remove,
                    plotted_pieces,
                );
            }
            _ = farm_pruning_interval.tick().fuse() => {
                for (farm_index, removed_farm) in known_farms.remove_expired() {
                    let farm_id = removed_farm.farm_id;

                    if removed_farm.expired_sender.send(()).is_ok() {
                        warn!(
                            %farm_index,
                            %farm_id,
                            "Farm expired and removed"
                        );
                    } else {
                        warn!(
                            %farm_index,
                            %farm_id,
                            "Farm exited before expiration notification"
                        );
                    }

                    farms_to_add_remove.push(farm_index, Box::pin(async move {
                        let plotted_pieces = Arc::clone(plotted_pieces);

                        let delete_farm_fut = task::spawn_blocking(move || {
                            plotted_pieces.write_blocking().delete_farm(farm_index);
                        });
                        if let Err(error) = delete_farm_fut.await {
                            error!(
                                %farm_index,
                                %farm_id,
                                %error,
                                "Failed to delete farm that expired"
                            );
                        }

                        None
                    }));
                }
            }
            result = farms_to_add_remove.select_next_some() => {
                if let Some((farm_index, expired_receiver, farm)) = result {
                    farms.push(async move {
                        select! {
                            result = farm.run().fuse() => {
                                (farm_index, result)
                            }
                            _ = expired_receiver.fuse() => {
                                // Nothing to do
                                (farm_index, Ok(()))
                            }
                        }
                    });
                }
            }
        }
    }
}

fn process_farm_identify_message<'a>(
    identify_message: ClusterFarmerIdentifyFarmBroadcast,
    nats_client: &'a NatsClient,
    known_farms: &mut KnownFarms,
    farms_to_add_remove: &mut FarmsAddRemoveStreamMap<'a, AddRemoveResult>,
    plotted_pieces: &'a Arc<AsyncRwLock<PlottedPieces<FarmIndex>>>,
) {
    let ClusterFarmerIdentifyFarmBroadcast {
        farm_id,
        total_sectors_count,
        fingerprint,
    } = identify_message;
    let (farm_index, expired_receiver, add, remove) =
        match known_farms.insert_or_update(farm_id, fingerprint) {
            KnownFarmInsertResult::Inserted {
                farm_index,
                expired_receiver,
            } => {
                info!(
                    %farm_index,
                    %farm_id,
                    "Discovered new farm, initializing"
                );

                (farm_index, expired_receiver, true, false)
            }
            KnownFarmInsertResult::FingerprintUpdated {
                farm_index,
                expired_receiver,
            } => {
                info!(
                    %farm_index,
                    %farm_id,
                    "Farm fingerprint updated, re-initializing"
                );

                (farm_index, expired_receiver, true, true)
            }
            KnownFarmInsertResult::NotInserted => {
                trace!(
                    %farm_id,
                    "Received identification for already known farm"
                );
                // Nothing to do here
                return;
            }
        };

    if remove {
        farms_to_add_remove.push(
            farm_index,
            Box::pin(async move {
                let plotted_pieces = Arc::clone(plotted_pieces);

                let delete_farm_fut = task::spawn_blocking(move || {
                    plotted_pieces.write_blocking().delete_farm(farm_index);
                });
                if let Err(error) = delete_farm_fut.await {
                    error!(
                        %farm_index,
                        %farm_id,
                        %error,
                        "Failed to delete farm that was replaced"
                    );
                }

                None
            }),
        );
    }

    if add {
        farms_to_add_remove.push(
            farm_index,
            Box::pin(async move {
                match initialize_farm(
                    farm_index,
                    farm_id,
                    total_sectors_count,
                    Arc::clone(plotted_pieces),
                    nats_client,
                )
                .await
                {
                    Ok(farm) => {
                        if remove {
                            info!(
                                %farm_index,
                                %farm_id,
                                "Farm re-initialized successfully"
                            );
                        } else {
                            info!(
                                %farm_index,
                                %farm_id,
                                "Farm initialized successfully"
                            );
                        }

                        Some((farm_index, expired_receiver, farm))
                    }
                    Err(error) => {
                        warn!(
                            %error,
                            "Failed to initialize farm {farm_id}"
                        );
                        None
                    }
                }
            }),
        );
    }
}

async fn initialize_farm(
    farm_index: FarmIndex,
    farm_id: FarmId,
    total_sectors_count: SectorIndex,
    plotted_pieces: Arc<AsyncRwLock<PlottedPieces<FarmIndex>>>,
    nats_client: &NatsClient,
) -> anyhow::Result<ClusterFarm> {
    let farm = ClusterFarm::new(farm_id, total_sectors_count, nats_client.clone())
        .await
        .map_err(|error| anyhow!("Failed instantiate cluster farm {farm_id}: {error}"))?;

    // Buffer sectors that are plotted while already plotted sectors are being iterated over
    let plotted_sectors_buffer = Arc::new(Mutex::new(Vec::new()));
    let sector_update_handler = farm.on_sector_update(Arc::new({
        let plotted_sectors_buffer = Arc::clone(&plotted_sectors_buffer);

        move |(_sector_index, sector_update)| {
            if let SectorUpdate::Plotting(SectorPlottingDetails::Finished {
                plotted_sector,
                old_plotted_sector,
                ..
            }) = sector_update
            {
                plotted_sectors_buffer
                    .lock()
                    .push((plotted_sector.clone(), old_plotted_sector.clone()));
            }
        }
    }));

    // Add plotted sectors of the farm to global plotted pieces
    let plotted_sectors = farm.plotted_sectors();
    let mut plotted_sectors = plotted_sectors
        .get()
        .await
        .map_err(|error| anyhow!("Failed to get plotted sectors for farm {farm_id}: {error}"))?;

    {
        plotted_pieces
            .write()
            .await
            .add_farm(farm_index, farm.piece_reader());

        while let Some(plotted_sector_result) = plotted_sectors.next().await {
            let plotted_sector = plotted_sector_result.map_err(|error| {
                anyhow!("Failed to get plotted sector for farm {farm_id}: {error}")
            })?;

            let mut plotted_pieces_guard = plotted_pieces.write().await;
            plotted_pieces_guard.add_sector(farm_index, &plotted_sector);

            // Drop the guard immediately to make sure other tasks are able to access the plotted pieces
            drop(plotted_pieces_guard);

            task::yield_now().await;
        }
    }

    // Add sectors that were plotted while above iteration was happening to plotted sectors
    // too
    drop(sector_update_handler);
    let plotted_sectors_buffer = mem::take(&mut *plotted_sectors_buffer.lock());
    let add_buffered_sectors_fut = task::spawn_blocking(move || {
        let mut plotted_pieces = plotted_pieces.write_blocking();

        for (plotted_sector, old_plotted_sector) in plotted_sectors_buffer {
            if let Some(old_plotted_sector) = old_plotted_sector {
                plotted_pieces.delete_sector(farm_index, &old_plotted_sector);
            }
            // Call delete first to avoid adding duplicates
            plotted_pieces.delete_sector(farm_index, &plotted_sector);
            plotted_pieces.add_sector(farm_index, &plotted_sector);
        }
    });

    add_buffered_sectors_fut
        .await
        .map_err(|error| anyhow!("Failed to add buffered sectors for farm {farm_id}: {error}"))?;

    Ok(farm)
}
