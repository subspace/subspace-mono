use crate::commitments::Commitments;
use crate::dsn;
use crate::dsn::{PieceIndexHashNumber, SyncOptions};
use crate::farming::Farming;
use crate::identity::Identity;
use crate::plot::{Plot, PlotError};
use crate::plotting::plot_pieces;
use crate::rpc_client::RpcClient;
use futures::FutureExt;
use num_traits::{WrappingAdd, WrappingSub};
use parking_lot::Mutex;
use std::future::Future;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use subspace_core_primitives::{PieceIndexHash, PublicKey};
use subspace_networking::libp2p::identity::sr25519;
use subspace_networking::libp2p::multiaddr::Protocol;
use subspace_networking::libp2p::{Multiaddr, PeerId};
use subspace_networking::multimess::MultihashCode;
use subspace_networking::{
    libp2p, Config, Node, NodeRunner, PiecesByRangeRequest, PiecesByRangeResponse, PiecesToPlot,
};
use subspace_solving::SubspaceCodec;
use tokio::runtime::Handle;
use tracing::info;

const SYNC_PIECES_AT_ONCE: u64 = 5000;

pub(crate) struct SinglePlotFarmOptions<C, NewPlot>
where
    C: RpcClient,
    NewPlot: Fn(usize, PublicKey, u64) -> Result<Plot, PlotError> + Clone + Send + 'static,
{
    pub(crate) base_directory: PathBuf,
    pub(crate) plot_index: usize,
    pub(crate) max_plot_pieces: u64,
    pub(crate) farming_client: C,
    pub(crate) new_plot: NewPlot,
    pub(crate) listen_on: Vec<Multiaddr>,
    pub(crate) bootstrap_nodes: Vec<Multiaddr>,
    pub(crate) first_listen_on: Arc<Mutex<Option<Vec<Multiaddr>>>>,
    pub(crate) start_farmings: bool,
    pub(crate) reward_address: PublicKey,
}

// TODO: Make fields private
pub struct SinglePlotFarm {
    pub(crate) codec: SubspaceCodec,
    pub plot: Plot,
    pub commitments: Commitments,
    pub(crate) farming: Option<Farming>,
    pub(crate) node: Node,
    /// Might be `None` if was already taken out before
    pub node_runner: Option<NodeRunner>,
}

#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Range expected to sync
    pub expected_range: Range<PieceIndexHashNumber>,
    /// Range of pieces which was actually synced
    pub got_range: Range<PieceIndexHashNumber>,
    /// Public key of the plot
    pub public_key: PublicKey,
    /// Total number of pieces synced
    pub total_pieces: u64,
}

impl SinglePlotFarm {
    pub(crate) fn new<C, NewPlot>(
        SinglePlotFarmOptions {
            base_directory,
            plot_index,
            max_plot_pieces,
            farming_client,
            new_plot,
            mut listen_on,
            mut bootstrap_nodes,
            first_listen_on,
            start_farmings,
            reward_address,
        }: SinglePlotFarmOptions<C, NewPlot>,
    ) -> anyhow::Result<Self>
    where
        C: RpcClient,
        NewPlot: Fn(usize, PublicKey, u64) -> Result<Plot, PlotError> + Clone + Send + 'static,
    {
        std::fs::create_dir_all(&base_directory)?;

        let identity = Identity::open_or_create(&base_directory)?;
        let public_key = identity.public_key().to_bytes().into();

        // TODO: This doesn't account for the fact that node can
        // have a completely different history to what farmer expects
        info!("Opening plot");
        let plot = new_plot(plot_index, public_key, max_plot_pieces)?;

        info!("Opening commitments");
        let commitments = Commitments::new(base_directory.join("commitments"))?;

        // Start the farming task
        let farming = start_farmings.then(|| {
            Farming::start(
                plot.clone(),
                commitments.clone(),
                farming_client.clone(),
                identity.clone(),
                reward_address,
            )
        });

        for multiaddr in &mut listen_on {
            if let Some(Protocol::Tcp(starting_port)) = multiaddr.pop() {
                multiaddr.push(Protocol::Tcp(starting_port + plot_index as u16));
            } else {
                return Err(anyhow::anyhow!("Unknown protocol {}", multiaddr));
            }
        }
        {
            let mut first_listen_on = first_listen_on.lock();
            // Only add the first instance to bootstrap nodes of others
            match first_listen_on.as_ref() {
                Some(first_listen_on) => {
                    bootstrap_nodes.extend_from_slice(first_listen_on);
                }
                None => {
                    let public_key = sr25519::PublicKey::from(*identity.public_key());
                    let public_key = libp2p::identity::PublicKey::Sr25519(public_key);
                    let peer_id = PeerId::from(public_key);

                    first_listen_on.replace(
                        listen_on
                            .clone()
                            .into_iter()
                            .map(|listen_on| listen_on.with(Protocol::P2p(peer_id.into())))
                            .collect(),
                    );
                }
            }
        }

        let codec = SubspaceCodec::new(&plot.public_key());
        let create_networking_fut = subspace_networking::create(Config {
            bootstrap_nodes,
            // TODO: Do we still need it?
            value_getter: Arc::new({
                let plot = plot.clone();
                let codec = codec.clone();
                move |key| {
                    let code = key.code();

                    if code != u64::from(MultihashCode::Piece)
                        && code != u64::from(MultihashCode::PieceIndex)
                    {
                        return None;
                    }

                    let piece_index = u64::from_le_bytes(
                        key.digest()[..std::mem::size_of::<u64>()].try_into().ok()?,
                    );
                    plot.read(piece_index)
                        .ok()
                        .and_then(|mut piece| {
                            codec
                                .decode(&mut piece, piece_index)
                                .ok()
                                .map(move |()| piece)
                        })
                        .map(|piece| piece.to_vec())
                }
            }),
            pieces_by_range_request_handler: Arc::new({
                let plot = plot.clone();
                let codec = codec.clone();

                // TODO: also ask for how many pieces to read
                move |&PiecesByRangeRequest { from, to }| {
                    let mut pieces_and_indexes =
                        plot.get_sequential_pieces(from, SYNC_PIECES_AT_ONCE).ok()?;

                    if pieces_and_indexes.len() == 1 {
                        let (index, mut piece) = pieces_and_indexes
                            .into_iter()
                            .next()
                            .expect("Always present as length is exactly one");
                        codec.decode(&mut piece, index).ok()?;
                        return Some(PiecesByRangeResponse {
                            pieces: PiecesToPlot {
                                piece_indexes: vec![index],
                                pieces: piece.into(),
                            },
                            next_piece_index_hash: None,
                        });
                    }

                    let next_piece_index_hash = if let Some(idx) = pieces_and_indexes
                        .iter()
                        .position(|(piece_index, _)| PieceIndexHash::from(*piece_index) >= to)
                    {
                        pieces_and_indexes.truncate(idx);
                        None
                    } else {
                        pieces_and_indexes
                            .pop()
                            .map(|(index, _)| Some(PieceIndexHash::from(index)))
                            .unwrap_or_default()
                    };

                    let (piece_indexes, pieces) = pieces_and_indexes
                        .into_iter()
                        .flat_map(|(index, mut piece)| {
                            codec.decode(&mut piece, index).ok()?;
                            Some((index, piece))
                        })
                        .unzip();

                    Some(PiecesByRangeResponse {
                        pieces: PiecesToPlot {
                            piece_indexes,
                            pieces,
                        },
                        next_piece_index_hash,
                    })
                }
            }),
            allow_non_globals_in_dht: true,
            listen_on,
            ..Config::with_keypair(sr25519::Keypair::from(
                sr25519::SecretKey::from_bytes(identity.secret_key().to_bytes())
                    .expect("Always valid"),
            ))
        });

        let (node, node_runner) = Handle::current().block_on(create_networking_fut)?;

        node.on_new_listener(Arc::new({
            let node_id = node.id();

            move |multiaddr| {
                info!(
                    "Listening on {}",
                    multiaddr.clone().with(Protocol::P2p(node_id.into()))
                );
            }
        }))
        .detach();

        Ok::<_, anyhow::Error>(SinglePlotFarm {
            codec,
            plot,
            commitments,
            farming,
            node,
            node_runner: Some(node_runner),
        })
    }

    pub fn dsn_sync(
        &self,
        max_plot_size: u64,
        total_pieces: u64,
        range_size: PieceIndexHashNumber,
    ) -> impl Future<Output = anyhow::Result<SyncResult>> {
        let plot = self.plot.clone();
        let commitments = self.commitments.clone();
        let codec = self.codec.clone();
        let node = self.node.clone();

        let options = SyncOptions {
            range_size,
            public_key: plot.public_key(),
            max_plot_size,
            total_pieces,
        };
        let mut plot_pieces = plot_pieces(codec, &plot, commitments);

        dsn::sync(node, options, move |pieces, piece_indexes| {
            if !plot_pieces(PiecesToPlot {
                pieces,
                piece_indexes,
            }) {
                return Err(anyhow::anyhow!("Failed to plot pieces in archiving"));
            }
            Ok(())
        })
        .map({
            let plot = self.plot.clone();
            move |result| {
                let sync_sector_size = if total_pieces < max_plot_size {
                    PieceIndexHashNumber::MAX
                } else {
                    PieceIndexHashNumber::MAX / total_pieces * max_plot_size
                };
                let expected_range = Range {
                    start: PieceIndexHashNumber::from_big_endian(&plot.public_key())
                        .wrapping_sub(&(sync_sector_size / 2)),
                    end: PieceIndexHashNumber::from_big_endian(&plot.public_key())
                        .wrapping_add(&(sync_sector_size / 2)),
                };
                let got_range = plot.get_piece_range().map(|Range { start, end }| Range {
                    start: start.into(),
                    end: end.into(),
                })?;
                result.map(|()| SyncResult {
                    expected_range,
                    got_range,
                    public_key: plot.public_key(),
                    total_pieces: plot.piece_count(),
                })
            }
        })
    }
}
