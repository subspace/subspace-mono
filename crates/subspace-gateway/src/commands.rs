//! Gateway subcommands.

pub(crate) mod http;
pub(crate) mod network;
pub(crate) mod rpc;

use crate::commands::http::HttpCommandOptions;
use crate::commands::network::{configure_network, NetworkArgs};
use crate::commands::rpc::RpcCommandOptions;
use crate::node_client::RpcNodeClient;
use crate::piece_getter::DsnPieceGetter;
use crate::piece_validator::SegmentCommitmentPieceValidator;
use anyhow::anyhow;
use async_lock::Semaphore;
use clap::Parser;
use std::num::NonZeroUsize;
use std::panic;
use std::process::exit;
use std::sync::Arc;
use subspace_core_primitives::pieces::Record;
use subspace_data_retrieval::object_fetcher::ObjectFetcher;
use subspace_erasure_coding::ErasureCoding;
use subspace_kzg::Kzg;
use subspace_networking::utils::piece_provider::PieceProvider;
use subspace_networking::NodeRunner;
use tokio::signal;
use tracing::{debug, warn};

/// The default size limit, based on the maximum consensus block size.
pub const DEFAULT_MAX_SIZE: usize = 5 * 1024 * 1024;
/// Multiplier on top of outgoing connections number for piece downloading purposes
const PIECE_PROVIDER_MULTIPLIER: usize = 10;

/// Commands for working with a gateway.
#[derive(Debug, Parser)]
#[clap(about, version)]
pub enum Command {
    /// Run data gateway with RPC server
    Rpc(RpcCommandOptions),
    /// Run data gateway with HTTP server
    Http(HttpCommandOptions),
    // TODO: subcommand to run various benchmarks
}

/// Options for running a gateway
#[derive(Debug, Parser)]
pub(crate) struct GatewayOptions {
    /// Enable development mode.
    ///
    /// Implies following flags (unless customized):
    /// * `--allow-private-ips`
    #[arg(long, verbatim_doc_comment)]
    dev: bool,

    /// The maximum object size to fetch.
    /// Larger objects will return an error.
    #[arg(long, default_value_t = DEFAULT_MAX_SIZE)]
    max_size: usize,

    #[clap(flatten)]
    dsn_options: NetworkArgs,
}

/// Install a panic handler which exits on panics, rather than unwinding. Unwinding can hang the
/// tokio runtime waiting for stuck tasks or threads.
pub(crate) fn set_exit_on_panic() {
    let default_panic_hook = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        default_panic_hook(panic_info);
        exit(1);
    }));
}

pub(crate) fn raise_fd_limit() {
    match fdlimit::raise_fd_limit() {
        Ok(fdlimit::Outcome::LimitRaised { from, to }) => {
            debug!(
                "Increased file descriptor limit from previous (most likely soft) limit {} to \
                new (most likely hard) limit {}",
                from, to
            );
        }
        Ok(fdlimit::Outcome::Unsupported) => {
            // Unsupported platform (a platform other than Linux or macOS)
        }
        Err(error) => {
            warn!(
                "Failed to increase file descriptor limit for the process due to an error: {}.",
                error
            );
        }
    }
}

#[cfg(unix)]
pub(crate) async fn shutdown_signal() {
    use futures::FutureExt;
    use std::pin::pin;

    futures::future::select(
        pin!(signal::unix::signal(signal::unix::SignalKind::interrupt())
            .expect("Setting signal handlers must never fail")
            .recv()
            .map(|_| {
                tracing::info!("Received SIGINT, shutting down gateway...");
            }),),
        pin!(signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Setting signal handlers must never fail")
            .recv()
            .map(|_| {
                tracing::info!("Received SIGTERM, shutting down gateway...");
            }),),
    )
    .await;
}

#[cfg(not(unix))]
pub(crate) async fn shutdown_signal() {
    signal::ctrl_c()
        .await
        .expect("Setting signal handlers must never fail");

    tracing::info!("Received Ctrl+C, shutting down gateway...");
}

/// Configures and returns object fetcher and DSN node runner.
pub async fn initialize_object_fetcher(
    options: GatewayOptions,
) -> anyhow::Result<(
    ObjectFetcher<DsnPieceGetter<SegmentCommitmentPieceValidator<RpcNodeClient>>>,
    NodeRunner,
)> {
    let GatewayOptions {
        dev,
        max_size,
        mut dsn_options,
    } = options;
    // Development mode handling is limited to this section
    {
        if dev {
            dsn_options.allow_private_ips = true;
        }
    }

    let kzg = Kzg::new();
    let erasure_coding = ErasureCoding::new(
        NonZeroUsize::new(Record::NUM_S_BUCKETS.next_power_of_two().ilog2() as usize)
            .expect("Not zero; qed"),
    )
    .map_err(|error| anyhow!("Failed to instantiate erasure coding: {error}"))?;

    let out_connections = dsn_options.out_connections;
    // TODO: move this service code into its own function, in a new library part of this crate
    let (dsn_node, dsn_node_runner, node_client) = configure_network(dsn_options).await?;

    let piece_provider = PieceProvider::new(
        dsn_node.clone(),
        SegmentCommitmentPieceValidator::new(dsn_node, node_client, kzg),
        Arc::new(Semaphore::new(
            out_connections as usize * PIECE_PROVIDER_MULTIPLIER,
        )),
    );
    let piece_getter = DsnPieceGetter::new(piece_provider);
    let object_fetcher = ObjectFetcher::new(piece_getter.into(), erasure_coding, max_size);

    Ok((object_fetcher, dsn_node_runner))
}
