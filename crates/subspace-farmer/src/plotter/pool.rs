//! Pool plotter

use crate::plotter::{Plotter, SectorPlottingProgress};
use async_trait::async_trait;
use futures::channel::mpsc;
use std::any::type_name_of_val;
use std::time::Duration;
use subspace_core_primitives::sectors::SectorIndex;
use subspace_core_primitives::PublicKey;
use subspace_farmer_components::FarmerProtocolInfo;
use tracing::{error, trace};

/// Pool plotter.
///
/// This plotter implementation relies on retries and should only be used with local plotter
/// implementations (like CPU and GPU).
#[derive(Debug)]
pub struct PoolPlotter {
    plotters: Vec<Box<dyn Plotter + Send + Sync>>,
    retry_interval: Duration,
}

#[async_trait]
impl Plotter for PoolPlotter {
    async fn has_free_capacity(&self) -> Result<bool, String> {
        for (index, plotter) in self.plotters.iter().enumerate() {
            match plotter.has_free_capacity().await {
                Ok(result) => {
                    if result {
                        return Ok(true);
                    }
                }
                Err(error) => {
                    error!(
                        %error,
                        %index,
                        r#type = type_name_of_val(plotter),
                        "Failed to check free capacity for plotter"
                    );
                }
            }
        }

        Ok(false)
    }

    async fn plot_sector(
        &self,
        public_key: PublicKey,
        sector_index: SectorIndex,
        farmer_protocol_info: FarmerProtocolInfo,
        pieces_in_sector: u16,
        replotting: bool,
        progress_sender: mpsc::Sender<SectorPlottingProgress>,
    ) {
        loop {
            for plotter in &self.plotters {
                if plotter
                    .try_plot_sector(
                        public_key,
                        sector_index,
                        farmer_protocol_info,
                        pieces_in_sector,
                        replotting,
                        progress_sender.clone(),
                    )
                    .await
                {
                    return;
                }
            }

            trace!(
                retry_interval = ?self.retry_interval,
                "All plotters are busy, will wait and try again later"
            );
            tokio::time::sleep(self.retry_interval).await;
        }
    }

    async fn try_plot_sector(
        &self,
        public_key: PublicKey,
        sector_index: SectorIndex,
        farmer_protocol_info: FarmerProtocolInfo,
        pieces_in_sector: u16,
        replotting: bool,
        progress_sender: mpsc::Sender<SectorPlottingProgress>,
    ) -> bool {
        for plotter in &self.plotters {
            if plotter
                .try_plot_sector(
                    public_key,
                    sector_index,
                    farmer_protocol_info,
                    pieces_in_sector,
                    replotting,
                    progress_sender.clone(),
                )
                .await
            {
                return true;
            }
        }

        false
    }
}

impl PoolPlotter {
    /// Create new instance
    pub fn new(plotters: Vec<Box<dyn Plotter + Send + Sync>>, retry_interval: Duration) -> Self {
        Self {
            plotters,
            retry_interval,
        }
    }
}
