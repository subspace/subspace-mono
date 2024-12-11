//! Abstract farm API
//!
//! This module provides a bunch of traits and simple data structures that serve as a layer of
//! abstraction that improves composition without having assumptions about implementation details.
//!
//! Implementations can be local (backed by local disk) and remote (connected via network in some
//! way). This crate provides a few of such implementations, but more can be created externally as
//! well if needed without modifying the library itself.

use async_trait::async_trait;
use derive_more::{Display, From};
use futures::Stream;
use parity_scale_codec::{Decode, Encode, EncodeLike, Input, Output};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use std::{fmt, io};
use subspace_core_primitives::pieces::{Piece, PieceIndex, PieceOffset};
use subspace_core_primitives::sectors::SectorIndex;
use subspace_core_primitives::segments::SegmentIndex;
use subspace_farmer_components::auditing::AuditingError;
use subspace_farmer_components::plotting::PlottedSector;
use subspace_farmer_components::proving::ProvingError;
use subspace_networking::libp2p::kad::RecordKey;
use subspace_rpc_primitives::SolutionResponse;
use thiserror::Error;
use ulid::Ulid;

pub mod plotted_pieces;

/// Erased error type
pub type FarmError = Box<dyn std::error::Error + Send + Sync + 'static>;
/// Type alias used for event handlers
pub type HandlerFn<A> = Arc<dyn Fn(&A) + Send + Sync + 'static>;

/// Getter for plotted sectors
#[async_trait]
pub trait PlottedSectors: Send + Sync + fmt::Debug {
    /// Get already plotted sectors
    async fn get(
        &self,
    ) -> Result<
        Box<dyn Stream<Item = Result<PlottedSector, FarmError>> + Unpin + Send + '_>,
        FarmError,
    >;
}

/// An identifier for a cache, can be used for in logs, thread names, etc.
#[derive(
    Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Serialize, Deserialize, Display, From,
)]
#[serde(untagged)]
pub enum CacheId {
    /// Cache ID
    Ulid(Ulid),
}

impl Encode for CacheId {
    #[inline]
    fn size_hint(&self) -> usize {
        1_usize
            + match self {
                CacheId::Ulid(ulid) => 0_usize.saturating_add(Encode::size_hint(&ulid.0)),
            }
    }

    #[inline]
    fn encode_to<O: Output + ?Sized>(&self, output: &mut O) {
        match self {
            CacheId::Ulid(ulid) => {
                output.push_byte(0);
                Encode::encode_to(&ulid.0, output);
            }
        }
    }
}

impl EncodeLike for CacheId {}

impl Decode for CacheId {
    #[inline]
    fn decode<I: Input>(input: &mut I) -> Result<Self, parity_scale_codec::Error> {
        match input
            .read_byte()
            .map_err(|e| e.chain("Could not decode `CacheId`, failed to read variant byte"))?
        {
            0 => u128::decode(input)
                .map(|ulid| CacheId::Ulid(Ulid(ulid)))
                .map_err(|e| e.chain("Could not decode `CacheId::Ulid.0`")),
            _ => Err("Could not decode `CacheId`, variant doesn't exist".into()),
        }
    }
}

#[allow(clippy::new_without_default)]
impl CacheId {
    /// Creates new ID
    #[inline]
    pub fn new() -> Self {
        Self::Ulid(Ulid::new())
    }
}

/// An identifier for a piece cache, can be used for in logs, thread names, etc.
#[derive(
    Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Serialize, Deserialize, Display, From,
)]
#[serde(untagged)]
pub enum PieceCacheId {
    /// Piece Cache ID
    Ulid(Ulid),
}

impl Encode for PieceCacheId {
    #[inline]
    fn size_hint(&self) -> usize {
        1_usize
            + match self {
                PieceCacheId::Ulid(ulid) => 0_usize.saturating_add(Encode::size_hint(&ulid.0)),
            }
    }

    #[inline]
    fn encode_to<O: Output + ?Sized>(&self, output: &mut O) {
        match self {
            PieceCacheId::Ulid(ulid) => {
                output.push_byte(0);
                Encode::encode_to(&ulid.0, output);
            }
        }
    }
}

impl EncodeLike for PieceCacheId {}

impl Decode for PieceCacheId {
    #[inline]
    fn decode<I: Input>(input: &mut I) -> Result<Self, parity_scale_codec::Error> {
        match input
            .read_byte()
            .map_err(|e| e.chain("Could not decode `PieceCacheId`, failed to read variant byte"))?
        {
            0 => u128::decode(input)
                .map(|ulid| PieceCacheId::Ulid(Ulid(ulid)))
                .map_err(|e| e.chain("Could not decode `PieceCacheId::Ulid.0`")),
            _ => Err("Could not decode `PieceCacheId`, variant doesn't exist".into()),
        }
    }
}

#[allow(clippy::new_without_default)]
impl PieceCacheId {
    /// Creates new ID
    #[inline]
    pub fn new() -> Self {
        Self::Ulid(Ulid::new())
    }
}

/// Offset wrapper for pieces in [`PieceCache`]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Encode, Decode)]
#[repr(transparent)]
pub struct PieceCacheOffset(pub(crate) u32);

/// Abstract piece cache implementation.
///
/// Piece cache is a simple container that stores concatenated pieces in a flat file at specific
/// offsets. Implementation doesn't have to be local though, cache can be remote somewhere on the
/// network, APIs are intentionally async to account for that.
#[async_trait]
pub trait PieceCache: Send + Sync + fmt::Debug {
    /// ID of this cache
    fn id(&self) -> &PieceCacheId;

    /// Max number of elements in this cache
    fn max_num_elements(&self) -> u32;

    /// Contents of this piece cache.
    ///
    /// NOTE: it is possible to do concurrent reads and writes, higher level logic must ensure this
    /// doesn't happen for the same piece being accessed!
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
    >;

    /// Store piece in cache at specified offset, replacing existing piece if there is any.
    ///
    /// NOTE: it is possible to do concurrent reads and writes, higher level logic must ensure this
    /// doesn't happen for the same piece being accessed!
    async fn write_piece(
        &self,
        offset: PieceCacheOffset,
        piece_index: PieceIndex,
        piece: &Piece,
    ) -> Result<(), FarmError>;

    /// Read piece index from cache at specified offset.
    ///
    /// Returns `None` if offset is out of range.
    ///
    /// NOTE: it is possible to do concurrent reads and writes, higher level logic must ensure this
    /// doesn't happen for the same piece being accessed!
    async fn read_piece_index(
        &self,
        offset: PieceCacheOffset,
    ) -> Result<Option<PieceIndex>, FarmError>;

    /// Read piece from cache at specified offset.
    ///
    /// Returns `None` if offset is out of range.
    ///
    /// NOTE: it is possible to do concurrent reads and writes, higher level logic must ensure this
    /// doesn't happen for the same piece being accessed!
    async fn read_piece(
        &self,
        offset: PieceCacheOffset,
    ) -> Result<Option<(PieceIndex, Piece)>, FarmError>;

    /// Read pieces from cache at specified offsets.
    ///
    /// Number of elements in returned stream is the same as number of unique `offsets`.
    /// Returns `None` for offsets that are out of range.
    ///
    /// NOTE: it is possible to do concurrent reads and writes, higher level logic must ensure this
    /// doesn't happen for the same pieces being accessed!
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
    >;
}

/// Result of piece storing check
#[derive(Debug, Copy, Clone, Encode, Decode)]
pub enum MaybePieceStoredResult {
    /// Definitely not stored
    No,
    /// Maybe has vacant slot to store
    Vacant,
    /// Maybe still stored
    Yes,
}

/// Abstract plot cache implementation.
///
/// Plot cache is a cache that exploits space towards the end of the plot that is not yet occupied
/// by sectors in order to increase effective caching space, which helps with plotting speed for
/// small farmers since they don't need to retrieve the same pieces from the network over and over
/// again, which is slower and uses a lot of Internet bandwidth.
#[async_trait]
pub trait PlotCache: Send + Sync + fmt::Debug {
    /// Check if piece is potentially stored in this cache (not guaranteed to be because it might be
    /// overridden with sector any time)
    async fn is_piece_maybe_stored(
        &self,
        key: &RecordKey,
    ) -> Result<MaybePieceStoredResult, FarmError>;

    /// Store piece in cache if there is free space, otherwise `Ok(false)` is returned
    async fn try_store_piece(
        &self,
        piece_index: PieceIndex,
        piece: &Piece,
    ) -> Result<bool, FarmError>;

    /// Read piece from cache.
    ///
    /// Returns `None` if not cached.
    async fn read_piece(&self, key: &RecordKey) -> Result<Option<Piece>, FarmError>;
}

/// Auditing details
#[derive(Debug, Copy, Clone, Encode, Decode)]
pub struct AuditingDetails {
    /// Number of sectors that were audited
    pub sectors_count: SectorIndex,
    /// Audit duration
    pub time: Duration,
}

/// Result of the proving
#[derive(Debug, Copy, Clone, Encode, Decode)]
pub enum ProvingResult {
    /// Proved successfully and accepted by the node
    Success,
    /// Proving took too long
    Timeout,
    /// Managed to prove within time limit, but node rejected solution, likely due to timeout on its
    /// end
    Rejected,
    /// Proving failed altogether
    Failed,
}

impl fmt::Display for ProvingResult {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Success => "Success",
            Self::Timeout => "Timeout",
            Self::Rejected => "Rejected",
            Self::Failed => "Failed",
        })
    }
}

/// Proving details
#[derive(Debug, Copy, Clone, Encode, Decode)]
pub struct ProvingDetails {
    /// Whether proving ended up being successful
    pub result: ProvingResult,
    /// Audit duration
    pub time: Duration,
}

/// Special decoded farming error
#[derive(Debug, Encode, Decode)]
pub struct DecodedFarmingError {
    /// String representation of an error
    error: String,
    /// Whether error is fatal
    is_fatal: bool,
}

impl fmt::Display for DecodedFarmingError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

/// Errors that happen during farming
#[derive(Debug, Error)]
pub enum FarmingError {
    /// Failed to subscribe to slot info notifications
    #[error("Failed to subscribe to slot info notifications: {error}")]
    FailedToSubscribeSlotInfo {
        /// Lower-level error
        error: anyhow::Error,
    },
    /// Failed to retrieve farmer info
    #[error("Failed to retrieve farmer info: {error}")]
    FailedToGetFarmerInfo {
        /// Lower-level error
        error: anyhow::Error,
    },
    /// Slot info notification stream ended
    #[error("Slot info notification stream ended")]
    SlotNotificationStreamEnded,
    /// Low-level auditing error
    #[error("Low-level auditing error: {0}")]
    LowLevelAuditing(#[from] AuditingError),
    /// Low-level proving error
    #[error("Low-level proving error: {0}")]
    LowLevelProving(#[from] ProvingError),
    /// I/O error occurred
    #[error("Farming I/O error: {0}")]
    Io(#[from] io::Error),
    /// Decoded farming error
    #[error("Decoded farming error {0}")]
    Decoded(DecodedFarmingError),
}

impl Encode for FarmingError {
    #[inline]
    fn encode_to<O: Output + ?Sized>(&self, dest: &mut O) {
        let error = DecodedFarmingError {
            error: self.to_string(),
            is_fatal: self.is_fatal(),
        };

        error.encode_to(dest)
    }
}

impl Decode for FarmingError {
    #[inline]
    fn decode<I: Input>(input: &mut I) -> Result<Self, parity_scale_codec::Error> {
        DecodedFarmingError::decode(input).map(FarmingError::Decoded)
    }
}

impl FarmingError {
    /// String variant of the error, primarily for monitoring purposes
    #[inline]
    pub fn str_variant(&self) -> &str {
        match self {
            FarmingError::FailedToSubscribeSlotInfo { .. } => "FailedToSubscribeSlotInfo",
            FarmingError::FailedToGetFarmerInfo { .. } => "FailedToGetFarmerInfo",
            FarmingError::LowLevelAuditing(_) => "LowLevelAuditing",
            FarmingError::LowLevelProving(_) => "LowLevelProving",
            FarmingError::Io(_) => "Io",
            FarmingError::Decoded(_) => "Decoded",
            FarmingError::SlotNotificationStreamEnded => "SlotNotificationStreamEnded",
        }
    }

    /// Whether this error is fatal and makes farm unusable
    pub fn is_fatal(&self) -> bool {
        match self {
            FarmingError::FailedToSubscribeSlotInfo { .. } => true,
            FarmingError::FailedToGetFarmerInfo { .. } => true,
            FarmingError::LowLevelAuditing(_) => true,
            FarmingError::LowLevelProving(error) => error.is_fatal(),
            FarmingError::Io(_) => true,
            FarmingError::Decoded(error) => error.is_fatal,
            FarmingError::SlotNotificationStreamEnded => true,
        }
    }
}

/// Various farming notifications
#[derive(Debug, Clone, Encode, Decode)]
pub enum FarmingNotification {
    /// Auditing
    Auditing(AuditingDetails),
    /// Proving
    Proving(ProvingDetails),
    /// Non-fatal farming error
    NonFatalError(Arc<FarmingError>),
}

/// Details about sector currently being plotted
#[derive(Debug, Clone, Encode, Decode)]
pub enum SectorPlottingDetails {
    /// Starting plotting of a sector
    Starting {
        /// Progress so far in % (not including this sector)
        progress: f32,
        /// Whether sector is being replotted
        replotting: bool,
        /// Whether this is the last sector queued so far
        last_queued: bool,
    },
    /// Downloading sector pieces
    Downloading,
    /// Downloaded sector pieces
    Downloaded(Duration),
    /// Encoding sector pieces
    Encoding,
    /// Encoded sector pieces
    Encoded(Duration),
    /// Writing sector
    Writing,
    /// Written sector
    Written(Duration),
    /// Finished plotting
    Finished {
        /// Information about plotted sector
        plotted_sector: PlottedSector,
        /// Information about old plotted sector that was replaced
        old_plotted_sector: Option<PlottedSector>,
        /// How much time it took to plot a sector
        time: Duration,
    },
    /// Plotting failed
    Error(String),
}

/// Details about sector expiration
#[derive(Debug, Clone, Encode, Decode)]
pub enum SectorExpirationDetails {
    /// Sector expiration became known
    Determined {
        /// Segment index at which sector expires
        expires_at: SegmentIndex,
    },
    /// Sector will expire at the next segment index and should be replotted
    AboutToExpire,
    /// Sector already expired
    Expired,
}

/// Various sector updates
#[derive(Debug, Clone, Encode, Decode)]
pub enum SectorUpdate {
    /// Sector is being plotted
    Plotting(SectorPlottingDetails),
    /// Sector expiration information updated
    Expiration(SectorExpirationDetails),
}

/// Abstract piece reader implementation
#[async_trait]
pub trait PieceReader: Send + Sync + fmt::Debug {
    /// Read piece from sector by offset, `None` means input parameters are incorrect or piece
    /// reader was shut down
    async fn read_piece(
        &self,
        sector_index: SectorIndex,
        piece_offset: PieceOffset,
    ) -> Result<Option<Piece>, FarmError>;
}

/// Opaque handler ID for event handlers, once dropped handler will be removed automatically
pub trait HandlerId: Send + Sync + fmt::Debug {
    /// Consumes [`HandlerId`] and prevents handler from being removed automatically.
    fn detach(&self);
}

impl HandlerId for event_listener_primitives::HandlerId {
    #[inline]
    fn detach(&self) {
        self.detach();
    }
}

/// An identifier for a farm, can be used for in logs, thread names, etc.
#[derive(
    Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Serialize, Deserialize, Display, From,
)]
#[serde(untagged)]
pub enum FarmId {
    /// Farm ID
    Ulid(Ulid),
}

impl Encode for FarmId {
    #[inline]
    fn size_hint(&self) -> usize {
        1_usize
            + match self {
                FarmId::Ulid(ulid) => 0_usize.saturating_add(Encode::size_hint(&ulid.0)),
            }
    }

    #[inline]
    fn encode_to<O: Output + ?Sized>(&self, output: &mut O) {
        match self {
            FarmId::Ulid(ulid) => {
                output.push_byte(0);
                Encode::encode_to(&ulid.0, output);
            }
        }
    }
}

impl EncodeLike for FarmId {}

impl Decode for FarmId {
    #[inline]
    fn decode<I: Input>(input: &mut I) -> Result<Self, parity_scale_codec::Error> {
        match input
            .read_byte()
            .map_err(|e| e.chain("Could not decode `FarmId`, failed to read variant byte"))?
        {
            0 => u128::decode(input)
                .map(|ulid| FarmId::Ulid(Ulid(ulid)))
                .map_err(|e| e.chain("Could not decode `FarmId::Ulid.0`")),
            _ => Err("Could not decode `FarmId`, variant doesn't exist".into()),
        }
    }
}

#[allow(clippy::new_without_default)]
impl FarmId {
    /// Creates new ID
    #[inline]
    pub fn new() -> Self {
        Self::Ulid(Ulid::new())
    }
}

/// Abstract farm implementation
#[async_trait(?Send)]
pub trait Farm {
    /// ID of this farm
    fn id(&self) -> &FarmId;

    /// Number of sectors in this farm
    fn total_sectors_count(&self) -> SectorIndex;

    /// Get plotted sectors instance
    fn plotted_sectors(&self) -> Arc<dyn PlottedSectors + 'static>;

    /// Get piece reader to read plotted pieces later
    fn piece_reader(&self) -> Arc<dyn PieceReader + 'static>;

    /// Subscribe to sector updates
    fn on_sector_update(
        &self,
        callback: HandlerFn<(SectorIndex, SectorUpdate)>,
    ) -> Box<dyn HandlerId>;

    /// Subscribe to farming notifications
    fn on_farming_notification(
        &self,
        callback: HandlerFn<FarmingNotification>,
    ) -> Box<dyn HandlerId>;

    /// Subscribe to new solution notification
    fn on_solution(&self, callback: HandlerFn<SolutionResponse>) -> Box<dyn HandlerId>;

    /// Run and wait for background threads to exit or return an error
    fn run(self: Box<Self>) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>;
}

#[async_trait]
impl<T> Farm for Box<T>
where
    T: Farm + ?Sized,
{
    #[inline]
    fn id(&self) -> &FarmId {
        self.as_ref().id()
    }

    #[inline]
    fn total_sectors_count(&self) -> SectorIndex {
        self.as_ref().total_sectors_count()
    }

    #[inline]
    fn plotted_sectors(&self) -> Arc<dyn PlottedSectors + 'static> {
        self.as_ref().plotted_sectors()
    }

    #[inline]
    fn piece_reader(&self) -> Arc<dyn PieceReader + 'static> {
        self.as_ref().piece_reader()
    }

    #[inline]
    fn on_sector_update(
        &self,
        callback: HandlerFn<(SectorIndex, SectorUpdate)>,
    ) -> Box<dyn HandlerId> {
        self.as_ref().on_sector_update(callback)
    }

    #[inline]
    fn on_farming_notification(
        &self,
        callback: HandlerFn<FarmingNotification>,
    ) -> Box<dyn HandlerId> {
        self.as_ref().on_farming_notification(callback)
    }

    #[inline]
    fn on_solution(&self, callback: HandlerFn<SolutionResponse>) -> Box<dyn HandlerId> {
        self.as_ref().on_solution(callback)
    }

    #[inline]
    fn run(self: Box<Self>) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>> {
        (*self).run()
    }
}
