use futures::StreamExt;
use std::cmp::{Ordering, Reverse};
use std::collections::BTreeMap;
use std::error::Error;
use subspace_core_primitives::{Blake2b256Hash, RootBlock, SegmentIndex};
use subspace_networking::libp2p::PeerId;
use subspace_networking::{Node, RootBlockRequest, RootBlockResponse};
use tracing::{debug, error, trace, warn};

const ROOT_BLOCK_NUMBER_PER_REQUEST: u64 = 10;
/// Minimum peers number to participate in root block election.
const ROOT_BLOCK_CONSENSUS_MIN_SET: u64 = 2; //TODO: change the value
/// Threshold for the root block election success (minimum peer number with the same root block).
const ROOT_BLOCK_CONSENSUS_THRESHOLD: u64 = 2; //TODO: change the value

/// Helps gathering root blocks from DSN
pub struct RootBlockHandler {
    dsn_node: Node,
}

impl RootBlockHandler {
    pub fn new(dsn_node: Node) -> Self {
        Self { dsn_node }
    }

    /// Returns root blocks known to DSN.
    pub async fn get_root_blocks(&self) -> Result<Vec<RootBlock>, Box<dyn Error>> {
        trace!("Getting root blocks...");

        let mut result = Vec::new();
        let mut last_root_block = self.get_last_root_block().await?;
        debug!(
            "Getting root blocks starting from segment_index={}",
            last_root_block.segment_index()
        );

        result.push(last_root_block);

        while last_root_block.segment_index() > 0 {
            let segment_indexes: Vec<_> = (0..last_root_block.segment_index())
                .rev()
                .take(ROOT_BLOCK_NUMBER_PER_REQUEST as usize)
                .collect();

            let mut root_blocks = self.get_root_blocks_batch(segment_indexes).await?;
            root_blocks.sort_by(|rb1, rb2| compare_optional_root_blocks(rb1, rb2).reverse());

            for root_block in root_blocks {
                match root_block {
                    None => {
                        warn!("Root block request returned None.");
                    }
                    Some(root_block) => {
                        //TODO: add peer ban
                        if root_block.hash() != last_root_block.prev_root_block_hash() {
                            error!(
                                hash=?root_block.hash(),
                                prev_hash=?last_root_block.prev_root_block_hash(),
                                "Root block hash doesn't match expected hash from the last block."
                            );

                            return Err(
                                "Root block hash doesn't match expected hash from the last block."
                                    .into(),
                            );
                        }

                        last_root_block = root_block;
                        result.push(root_block);
                    }
                }
            }
        }

        Ok(result)
    }

    //TODO: add peer ban
    /// Return last root block known to DSN. We ask several peers for the highest root block
    /// known to them. Target root block should be known to at least ROOT_BLOCK_CONSENSUS_THRESHOLD
    /// among peer set with minimum size of ROOT_BLOCK_CONSENSUS_MIN_SET peers.
    async fn get_last_root_block(&self) -> Result<RootBlock, Box<dyn Error>> {
        let mut root_block_score: BTreeMap<Blake2b256Hash, u64> = BTreeMap::new();
        let mut root_block_dict: BTreeMap<Blake2b256Hash, RootBlock> = BTreeMap::new();
        let mut participating_peers = 0u64;
        trace!("Getting last root block...");

        // Get random peers. Some of them could be bootstrap nodes with no support for
        // request-response protocol for records root.
        let get_peers_result = self
            .dsn_node
            .get_closest_peers(PeerId::random().into())
            .await;

        match get_peers_result {
            Ok(mut get_peers_stream) => {
                while let Some(peer_id) = get_peers_stream.next().await {
                    trace!(%peer_id, "get_closest_peers returned an item");

                    let request_result = self
                        .dsn_node
                        .send_generic_request(
                            peer_id,
                            RootBlockRequest::LastRootBlocks {
                                root_block_number: ROOT_BLOCK_NUMBER_PER_REQUEST,
                            },
                        )
                        .await;

                    match request_result {
                        Ok(RootBlockResponse { root_blocks }) => {
                            trace!(%peer_id, "Last root block request succeeded.");
                            if root_blocks.iter().any(|rb| rb.is_some()) {
                                participating_peers += 1;
                            }

                            let root_block_deduplication_map = BTreeMap::from_iter(
                                root_blocks
                                    .iter()
                                    .filter_map(|rb| rb.map(|rb| (rb.hash(), rb))),
                            );
                            let sanitized_root_blocks = root_block_deduplication_map
                            .values()
                            .collect::<Vec<_>>();

                            // Collect root block votes from the peer.
                            for root_block in sanitized_root_blocks {
                                trace!(
                                        %peer_id,
                                        segment_index=root_block.segment_index(),
                                        hash=?root_block.hash(),
                                        "Last root block was obtained."
                                    );

                                root_block_score
                                    .entry(root_block.hash())
                                    .and_modify(|val| *val += 1)
                                    .or_insert(1);
                                root_block_dict
                                    .entry(root_block.hash())
                                    .or_insert(*root_block);
                            }
                        }
                        Err(error) => {
                            debug!(%peer_id, ?error, "Last root block request failed.");
                        }
                    };
                }
            }
            Err(err) => {
                warn!(?err, "get_closest_peers returned an error");

                return Err(err.into());
            }
        }

        // TODO: Consider adding attempts to increase the initial peer set.
        if participating_peers < ROOT_BLOCK_CONSENSUS_MIN_SET {
            return Err(format!(
                "Root block consensus failed: not enough peers ({}).",
                root_block_score.len()
            )
            .into());
        }

        // Sort the collection to get highest blocks first.
        let mut root_blocks = root_block_dict.values().collect::<Vec<_>>();
        root_blocks.sort_by_key(|rb| Reverse(rb.segment_index()));

        for root_block in root_blocks {
            let score = root_block_score
                .get(&root_block.hash())
                .expect("Must be present because of the manual adding.");

            if *score >= ROOT_BLOCK_CONSENSUS_THRESHOLD {
                return Ok(*root_block);
            }
        }

        Err("Root block consensus failed: can't pass the threshold.".into())
    }

    async fn get_root_blocks_batch(
        &self,
        segment_indexes: Vec<SegmentIndex>,
    ) -> Result<Vec<Option<RootBlock>>, Box<dyn Error>> {
        trace!(?segment_indexes, "Getting root block batch...");

        // Get random peers. Some of them could be bootstrap nodes with no support for
        // request-response protocol for records root.
        let get_peers_result = self
            .dsn_node
            .get_closest_peers(PeerId::random().into())
            .await;

        match get_peers_result {
            Ok(mut get_peers_stream) => {
                while let Some(peer_id) = get_peers_stream.next().await {
                    trace!(%peer_id, "get_closest_peers returned an item");

                    let request_result = self
                        .dsn_node
                        .send_generic_request(
                            peer_id,
                            RootBlockRequest::SegmentIndexes {
                                segment_indexes: segment_indexes.clone(),
                            },
                        )
                        .await;

                    match request_result {
                        Ok(RootBlockResponse { root_blocks }) => {
                            trace!(%peer_id, ?segment_indexes, "Root block request succeeded.");

                            return Ok(root_blocks);
                        }
                        Err(error) => {
                            debug!(%peer_id, ?segment_indexes, ?error, "Root block request failed.");
                        }
                    };
                }
                Err("No more peers for root blocks.".into())
            }
            Err(err) => {
                warn!(?err, "get_closest_peers returned an error");

                Err(err.into())
            }
        }
    }
}

/// Compares two root blocks by segment indexes. None are less then Some.
fn compare_optional_root_blocks(rb1: &Option<RootBlock>, rb2: &Option<RootBlock>) -> Ordering {
    match (rb1, rb2) {
        (None, None) => Ordering::Equal,
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (Some(rb1), Some(rb2)) => rb1.segment_index().cmp(&rb2.segment_index()),
    }
}
