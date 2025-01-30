//! Pieces-related data structures.

#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::segments::{ArchivedHistorySegment, RecordedHistorySegment, SegmentIndex};
use crate::ScalarBytes;
#[cfg(feature = "serde")]
use ::serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
use ::serde::{Deserializer, Serializer};
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bytes::{Bytes, BytesMut};
use core::array::TryFromSliceError;
use core::hash::{Hash, Hasher};
use core::iter::Step;
use core::{fmt, mem, slice};
use derive_more::{
    Add, AddAssign, AsMut, AsRef, Deref, DerefMut, Display, Div, DivAssign, From, Into, Mul,
    MulAssign, Sub, SubAssign,
};
use parity_scale_codec::{Decode, Encode, EncodeLike, Input, MaxEncodedLen, Output};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scale_info::build::Fields;
use scale_info::{Path, Type, TypeInfo};
#[cfg(feature = "serde")]
use serde_big_array::BigArray;

/// Piece index in consensus
#[derive(
    Debug,
    Display,
    Default,
    Copy,
    Clone,
    Ord,
    PartialOrd,
    Eq,
    PartialEq,
    Hash,
    Encode,
    Decode,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
    TypeInfo,
    MaxEncodedLen,
)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct PieceIndex(u64);

impl Step for PieceIndex {
    #[inline]
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
        u64::steps_between(&start.0, &end.0)
    }

    #[inline]
    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        u64::forward_checked(start.0, count).map(Self)
    }

    #[inline]
    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        u64::backward_checked(start.0, count).map(Self)
    }
}

impl From<u64> for PieceIndex {
    #[inline]
    fn from(original: u64) -> Self {
        Self(original)
    }
}

impl From<PieceIndex> for u64 {
    #[inline]
    fn from(original: PieceIndex) -> Self {
        original.0
    }
}

impl PieceIndex {
    /// Size in bytes.
    pub const SIZE: usize = mem::size_of::<u64>();
    /// Piece index 0.
    pub const ZERO: PieceIndex = PieceIndex(0);
    /// Piece index 1.
    pub const ONE: PieceIndex = PieceIndex(1);

    /// Create new instance
    #[inline]
    pub const fn new(n: u64) -> Self {
        Self(n)
    }

    /// Create piece index from bytes.
    #[inline]
    pub const fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        Self(u64::from_le_bytes(bytes))
    }

    /// Convert piece index to bytes.
    #[inline]
    pub const fn to_bytes(self) -> [u8; Self::SIZE] {
        self.0.to_le_bytes()
    }

    /// Segment index piece index corresponds to
    #[inline]
    pub const fn segment_index(&self) -> SegmentIndex {
        SegmentIndex::new(self.0 / ArchivedHistorySegment::NUM_PIECES as u64)
    }

    /// Position of a piece in a segment
    #[inline]
    pub const fn position(&self) -> u32 {
        // Position is statically guaranteed to fit into u32
        (self.0 % ArchivedHistorySegment::NUM_PIECES as u64) as u32
    }

    /// Position of a source piece in the source pieces for a segment.
    /// Panics if the piece is not a source piece.
    #[inline]
    pub const fn source_position(&self) -> u32 {
        assert!(self.is_source());

        let source_start = self.position() / RecordedHistorySegment::ERASURE_CODING_RATE.1 as u32
            * RecordedHistorySegment::ERASURE_CODING_RATE.0 as u32;
        let source_offset = self.position() % RecordedHistorySegment::ERASURE_CODING_RATE.1 as u32;

        source_start + source_offset
    }

    /// Returns the piece index for a source position and segment index.
    /// Overflows to the next segment if the position is greater than the last source position.
    #[inline]
    pub const fn from_source_position(
        source_position: u32,
        segment_index: SegmentIndex,
    ) -> PieceIndex {
        let source_position = source_position as u64;
        let start = source_position / RecordedHistorySegment::ERASURE_CODING_RATE.0 as u64
            * RecordedHistorySegment::ERASURE_CODING_RATE.1 as u64;
        let offset = source_position % RecordedHistorySegment::ERASURE_CODING_RATE.0 as u64;

        PieceIndex(segment_index.first_piece_index().0 + start + offset)
    }

    /// Is this piece index a source piece?
    #[inline]
    pub const fn is_source(&self) -> bool {
        // Source pieces are interleaved with parity pieces, source first
        self.0 % (RecordedHistorySegment::ERASURE_CODING_RATE.1 as u64)
            < (RecordedHistorySegment::ERASURE_CODING_RATE.0 as u64)
    }

    /// Returns the next source piece index.
    /// Panics if the piece is not a source piece.
    #[inline]
    pub const fn next_source_index(&self) -> PieceIndex {
        PieceIndex::from_source_position(self.source_position() + 1, self.segment_index())
    }
}

/// Piece offset in sector
#[derive(
    Debug,
    Display,
    Default,
    Copy,
    Clone,
    Ord,
    PartialOrd,
    Eq,
    PartialEq,
    Hash,
    Encode,
    Decode,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
    TypeInfo,
    MaxEncodedLen,
)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct PieceOffset(u16);

impl Step for PieceOffset {
    #[inline]
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
        u16::steps_between(&start.0, &end.0)
    }

    #[inline]
    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        u16::forward_checked(start.0, count).map(Self)
    }

    #[inline]
    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        u16::backward_checked(start.0, count).map(Self)
    }
}

impl From<u16> for PieceOffset {
    #[inline]
    fn from(original: u16) -> Self {
        Self(original)
    }
}

impl From<PieceOffset> for u16 {
    #[inline]
    fn from(original: PieceOffset) -> Self {
        original.0
    }
}

impl From<PieceOffset> for u32 {
    #[inline]
    fn from(original: PieceOffset) -> Self {
        Self::from(original.0)
    }
}

impl From<PieceOffset> for u64 {
    #[inline]
    fn from(original: PieceOffset) -> Self {
        Self::from(original.0)
    }
}

impl From<PieceOffset> for usize {
    #[inline]
    fn from(original: PieceOffset) -> Self {
        usize::from(original.0)
    }
}

impl PieceOffset {
    /// Piece index 0.
    pub const ZERO: PieceOffset = PieceOffset(0);
    /// Piece index 1.
    pub const ONE: PieceOffset = PieceOffset(1);

    /// Convert piece offset to bytes.
    #[inline]
    pub const fn to_bytes(self) -> [u8; mem::size_of::<u16>()] {
        self.0.to_le_bytes()
    }
}

/// Raw record contained within recorded history segment before archiving is applied.
///
/// NOTE: This is a stack-allocated data structure and can cause stack overflow!
#[derive(Copy, Clone, Eq, PartialEq, Deref, DerefMut)]
#[repr(transparent)]
pub struct RawRecord([[u8; ScalarBytes::SAFE_BYTES]; Self::NUM_CHUNKS]);

impl fmt::Debug for RawRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0.as_flattened()))
    }
}

impl Default for RawRecord {
    #[inline]
    fn default() -> Self {
        Self([Default::default(); Self::NUM_CHUNKS])
    }
}

impl AsRef<[u8]> for RawRecord {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice().as_flattened()
    }
}

impl AsMut<[u8]> for RawRecord {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.0.as_mut_slice().as_flattened_mut()
    }
}

impl From<&RawRecord> for &[[u8; ScalarBytes::SAFE_BYTES]; RawRecord::NUM_CHUNKS] {
    #[inline]
    fn from(value: &RawRecord) -> Self {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&[[u8; ScalarBytes::SAFE_BYTES]; RawRecord::NUM_CHUNKS]> for &RawRecord {
    #[inline]
    fn from(value: &[[u8; ScalarBytes::SAFE_BYTES]; RawRecord::NUM_CHUNKS]) -> Self {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut RawRecord> for &mut [[u8; ScalarBytes::SAFE_BYTES]; RawRecord::NUM_CHUNKS] {
    #[inline]
    fn from(value: &mut RawRecord) -> Self {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut [[u8; ScalarBytes::SAFE_BYTES]; RawRecord::NUM_CHUNKS]> for &mut RawRecord {
    #[inline]
    fn from(value: &mut [[u8; ScalarBytes::SAFE_BYTES]; RawRecord::NUM_CHUNKS]) -> Self {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&RawRecord> for &[u8; ScalarBytes::SAFE_BYTES * RawRecord::NUM_CHUNKS] {
    #[inline]
    fn from(value: &RawRecord) -> Self {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout as inner array, while array of byte arrays has the same alignment as a single byte
        unsafe { mem::transmute(value) }
    }
}

impl From<&[u8; ScalarBytes::SAFE_BYTES * RawRecord::NUM_CHUNKS]> for &RawRecord {
    #[inline]
    fn from(value: &[u8; ScalarBytes::SAFE_BYTES * RawRecord::NUM_CHUNKS]) -> Self {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout as inner array, while array of byte arrays has the same alignment as a single byte
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut RawRecord> for &mut [u8; ScalarBytes::SAFE_BYTES * RawRecord::NUM_CHUNKS] {
    #[inline]
    fn from(value: &mut RawRecord) -> Self {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout as inner array, while array of byte arrays has the same alignment as a single byte
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut [u8; ScalarBytes::SAFE_BYTES * RawRecord::NUM_CHUNKS]> for &mut RawRecord {
    #[inline]
    fn from(value: &mut [u8; ScalarBytes::SAFE_BYTES * RawRecord::NUM_CHUNKS]) -> Self {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout as inner array, while array of byte arrays has the same alignment as a single byte
        unsafe { mem::transmute(value) }
    }
}

impl RawRecord {
    /// Number of chunks (scalars) within one raw record.
    pub const NUM_CHUNKS: usize = 2_usize.pow(15);
    /// Size of raw record in bytes, is guaranteed to be a multiple of [`ScalarBytes::SAFE_BYTES`].
    pub const SIZE: usize = ScalarBytes::SAFE_BYTES * Self::NUM_CHUNKS;

    /// Create boxed value without hitting stack overflow
    #[inline]
    pub fn new_boxed() -> Box<Self> {
        // TODO: Should have been just `::new()`, but https://github.com/rust-lang/rust/issues/53827
        // SAFETY: Data structure filled with zeroes is a valid invariant
        unsafe { Box::new_zeroed().assume_init() }
    }

    /// Convenient conversion from slice of record to underlying representation for efficiency
    /// purposes.
    #[inline]
    pub fn slice_to_repr(value: &[Self]) -> &[[[u8; ScalarBytes::SAFE_BYTES]; Self::NUM_CHUNKS]] {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }

    /// Convenient conversion from slice of underlying representation to record for efficiency
    /// purposes.
    #[inline]
    pub fn slice_from_repr(value: &[[[u8; ScalarBytes::SAFE_BYTES]; Self::NUM_CHUNKS]]) -> &[Self] {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }

    /// Convenient conversion from mutable slice of record to underlying representation for
    /// efficiency purposes.
    #[inline]
    pub fn slice_mut_to_repr(
        value: &mut [Self],
    ) -> &mut [[[u8; ScalarBytes::SAFE_BYTES]; Self::NUM_CHUNKS]] {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }

    /// Convenient conversion from mutable slice of underlying representation to record for
    /// efficiency purposes.
    #[inline]
    pub fn slice_mut_from_repr(
        value: &mut [[[u8; ScalarBytes::SAFE_BYTES]; Self::NUM_CHUNKS]],
    ) -> &mut [Self] {
        // SAFETY: `RawRecord` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

/// Record contained within a piece.
///
/// NOTE: This is a stack-allocated data structure and can cause stack overflow!
#[derive(Copy, Clone, Eq, PartialEq, Deref, DerefMut)]
#[repr(transparent)]
pub struct Record([[u8; ScalarBytes::FULL_BYTES]; Self::NUM_CHUNKS]);

impl fmt::Debug for Record {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0.as_flattened()))
    }
}

impl Default for Record {
    #[inline]
    fn default() -> Self {
        Self([Default::default(); Self::NUM_CHUNKS])
    }
}

impl AsRef<[u8]> for Record {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.0.as_flattened()
    }
}

impl AsMut<[u8]> for Record {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.0.as_flattened_mut()
    }
}

impl From<&Record> for &[[u8; ScalarBytes::FULL_BYTES]; Record::NUM_CHUNKS] {
    #[inline]
    fn from(value: &Record) -> Self {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&[[u8; ScalarBytes::FULL_BYTES]; Record::NUM_CHUNKS]> for &Record {
    #[inline]
    fn from(value: &[[u8; ScalarBytes::FULL_BYTES]; Record::NUM_CHUNKS]) -> Self {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut Record> for &mut [[u8; ScalarBytes::FULL_BYTES]; Record::NUM_CHUNKS] {
    #[inline]
    fn from(value: &mut Record) -> Self {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut [[u8; ScalarBytes::FULL_BYTES]; Record::NUM_CHUNKS]> for &mut Record {
    #[inline]
    fn from(value: &mut [[u8; ScalarBytes::FULL_BYTES]; Record::NUM_CHUNKS]) -> Self {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&Record> for &[u8; ScalarBytes::FULL_BYTES * Record::NUM_CHUNKS] {
    #[inline]
    fn from(value: &Record) -> Self {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        // as inner array, while array of byte arrays has the same alignment as a single byte
        unsafe { mem::transmute(value) }
    }
}

impl From<&[u8; ScalarBytes::FULL_BYTES * Record::NUM_CHUNKS]> for &Record {
    #[inline]
    fn from(value: &[u8; ScalarBytes::FULL_BYTES * Record::NUM_CHUNKS]) -> Self {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        // as inner array, while array of byte arrays has the same alignment as a single byte
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut Record> for &mut [u8; ScalarBytes::FULL_BYTES * Record::NUM_CHUNKS] {
    #[inline]
    fn from(value: &mut Record) -> Self {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        // as inner array, while array of byte arrays has the same alignment as a single byte
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut [u8; ScalarBytes::FULL_BYTES * Record::NUM_CHUNKS]> for &mut Record {
    #[inline]
    fn from(value: &mut [u8; ScalarBytes::FULL_BYTES * Record::NUM_CHUNKS]) -> Self {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        // as inner array, while array of byte arrays has the same alignment as a single byte
        unsafe { mem::transmute(value) }
    }
}

impl Record {
    /// Number of chunks (scalars) within one record.
    pub const NUM_CHUNKS: usize = RawRecord::NUM_CHUNKS;
    /// Number of s-buckets contained within one record (and by extension sector).
    ///
    /// Essentially we chunk records into scalars and erasure code them.
    pub const NUM_S_BUCKETS: usize = Self::NUM_CHUNKS
        * RecordedHistorySegment::ERASURE_CODING_RATE.1
        / RecordedHistorySegment::ERASURE_CODING_RATE.0;
    /// Size of a segment record given the global piece size (in bytes) after erasure coding
    /// [`RawRecord`], is guaranteed to be a multiple of [`ScalarBytes::FULL_BYTES`].
    pub const SIZE: usize = ScalarBytes::FULL_BYTES * Self::NUM_CHUNKS;

    /// Create boxed value without hitting stack overflow
    #[inline]
    pub fn new_boxed() -> Box<Self> {
        // TODO: Should have been just `::new()`, but https://github.com/rust-lang/rust/issues/53827
        // SAFETY: Data structure filled with zeroes is a valid invariant
        unsafe { Box::new_zeroed().assume_init() }
    }

    /// Create vector filled with zeroe records without hitting stack overflow
    #[inline]
    pub fn new_zero_vec(length: usize) -> Vec<Self> {
        // TODO: Should have been just `::new()`, but https://github.com/rust-lang/rust/issues/53827
        let mut records = Vec::with_capacity(length);
        {
            let slice = records.spare_capacity_mut();
            // SAFETY: Same memory layout due to `#[repr(transparent)]` on `Record` and
            // `MaybeUninit<[[T; M]; N]>` is guaranteed to have the same layout as
            // `[[MaybeUninit<T>; M]; N]`
            let slice = unsafe {
                slice::from_raw_parts_mut(
                    slice.as_mut_ptr()
                        as *mut [[mem::MaybeUninit<u8>; ScalarBytes::FULL_BYTES]; Self::NUM_CHUNKS],
                    length,
                )
            };
            for byte in slice.as_flattened_mut().as_flattened_mut() {
                byte.write(0);
            }
        }
        // SAFETY: All values are initialized above.
        unsafe {
            records.set_len(records.capacity());
        }

        records
    }

    /// Convenient conversion from slice of record to underlying representation for efficiency
    /// purposes.
    #[inline]
    pub fn slice_to_repr(value: &[Self]) -> &[[[u8; ScalarBytes::FULL_BYTES]; Self::NUM_CHUNKS]] {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        unsafe { mem::transmute(value) }
    }

    /// Convenient conversion from slice of underlying representation to record for efficiency
    /// purposes.
    #[inline]
    pub fn slice_from_repr(value: &[[[u8; ScalarBytes::FULL_BYTES]; Self::NUM_CHUNKS]]) -> &[Self] {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        unsafe { mem::transmute(value) }
    }

    /// Convenient conversion from mutable slice of record to underlying representation for
    /// efficiency purposes.
    #[inline]
    pub fn slice_mut_to_repr(
        value: &mut [Self],
    ) -> &mut [[[u8; ScalarBytes::FULL_BYTES]; Self::NUM_CHUNKS]] {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        unsafe { mem::transmute(value) }
    }

    /// Convenient conversion from mutable slice of underlying representation to record for
    /// efficiency purposes.
    #[inline]
    pub fn slice_mut_from_repr(
        value: &mut [[[u8; ScalarBytes::FULL_BYTES]; Self::NUM_CHUNKS]],
    ) -> &mut [Self] {
        // SAFETY: `Record` is `#[repr(transparent)]` and guaranteed to have the same memory layout
        unsafe { mem::transmute(value) }
    }

    /// Convert from a record to its raw bytes, assumes dealing with source record that only stores
    /// safe bytes in its chunks.
    #[inline]
    pub fn to_raw_record_chunks(
        &self,
    ) -> impl Iterator<Item = &'_ [u8; ScalarBytes::SAFE_BYTES]> + '_ {
        // We have zero byte padding from [`ScalarBytes::SAFE_BYTES`] to
        // [`ScalarBytes::FULL_BYTES`] that we need to skip
        self.iter()
            .map(|bytes| bytes[1..].try_into().expect("Correct length; qed"))
    }

    /// Convert from a record to mutable raw bytes, assumes dealing with source record that only stores
    /// safe bytes in its chunks.
    #[inline]
    pub fn to_mut_raw_record_chunks(
        &mut self,
    ) -> impl Iterator<Item = &'_ mut [u8; ScalarBytes::SAFE_BYTES]> + '_ {
        self.iter_mut()
            .map(|bytes| (&mut bytes[1..]).try_into().expect("Correct length; qed"))
    }
}

/// Record commitment contained within a piece.
#[derive(
    Copy,
    Clone,
    Eq,
    PartialEq,
    Hash,
    Deref,
    DerefMut,
    From,
    Into,
    Encode,
    Decode,
    TypeInfo,
    MaxEncodedLen,
)]
pub struct RecordCommitment([u8; RecordCommitment::SIZE]);

impl fmt::Debug for RecordCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct RecordCommitmentBinary(#[serde(with = "BigArray")] [u8; RecordCommitment::SIZE]);

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct RecordCommitmentHex(#[serde(with = "hex")] [u8; RecordCommitment::SIZE]);

#[cfg(feature = "serde")]
impl Serialize for RecordCommitment {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            RecordCommitmentHex(self.0).serialize(serializer)
        } else {
            RecordCommitmentBinary(self.0).serialize(serializer)
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for RecordCommitment {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self(if deserializer.is_human_readable() {
            RecordCommitmentHex::deserialize(deserializer)?.0
        } else {
            RecordCommitmentBinary::deserialize(deserializer)?.0
        }))
    }
}

impl Default for RecordCommitment {
    #[inline]
    fn default() -> Self {
        Self([0; Self::SIZE])
    }
}

impl TryFrom<&[u8]> for RecordCommitment {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(slice: &[u8]) -> Result<Self, Self::Error> {
        <[u8; Self::SIZE]>::try_from(slice).map(Self)
    }
}

impl AsRef<[u8]> for RecordCommitment {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for RecordCommitment {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl From<&RecordCommitment> for &[u8; RecordCommitment::SIZE] {
    #[inline]
    fn from(value: &RecordCommitment) -> Self {
        // SAFETY: `RecordCommitment` is `#[repr(transparent)]` and guaranteed to have the same
        // memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&[u8; RecordCommitment::SIZE]> for &RecordCommitment {
    #[inline]
    fn from(value: &[u8; RecordCommitment::SIZE]) -> Self {
        // SAFETY: `RecordCommitment` is `#[repr(transparent)]` and guaranteed to have the same
        // memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut RecordCommitment> for &mut [u8; RecordCommitment::SIZE] {
    #[inline]
    fn from(value: &mut RecordCommitment) -> Self {
        // SAFETY: `RecordCommitment` is `#[repr(transparent)]` and guaranteed to have the same
        // memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut [u8; RecordCommitment::SIZE]> for &mut RecordCommitment {
    #[inline]
    fn from(value: &mut [u8; RecordCommitment::SIZE]) -> Self {
        // SAFETY: `RecordCommitment` is `#[repr(transparent)]` and guaranteed to have the same
        // memory layout
        unsafe { mem::transmute(value) }
    }
}

impl RecordCommitment {
    /// Size of record commitment in bytes.
    pub const SIZE: usize = 48;
}

/// Record witness contained within a piece.
#[derive(
    Copy,
    Clone,
    Eq,
    PartialEq,
    Hash,
    Deref,
    DerefMut,
    From,
    Into,
    Encode,
    Decode,
    TypeInfo,
    MaxEncodedLen,
)]
pub struct RecordWitness([u8; RecordWitness::SIZE]);

impl fmt::Debug for RecordWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct RecordWitnessBinary(#[serde(with = "BigArray")] [u8; RecordWitness::SIZE]);

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct RecordWitnessHex(#[serde(with = "hex")] [u8; RecordWitness::SIZE]);

#[cfg(feature = "serde")]
impl Serialize for RecordWitness {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            RecordWitnessHex(self.0).serialize(serializer)
        } else {
            RecordWitnessBinary(self.0).serialize(serializer)
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for RecordWitness {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self(if deserializer.is_human_readable() {
            RecordWitnessHex::deserialize(deserializer)?.0
        } else {
            RecordWitnessBinary::deserialize(deserializer)?.0
        }))
    }
}

impl Default for RecordWitness {
    #[inline]
    fn default() -> Self {
        Self([0; Self::SIZE])
    }
}

impl TryFrom<&[u8]> for RecordWitness {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(slice: &[u8]) -> Result<Self, Self::Error> {
        <[u8; Self::SIZE]>::try_from(slice).map(Self)
    }
}

impl AsRef<[u8]> for RecordWitness {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for RecordWitness {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl From<&RecordWitness> for &[u8; RecordWitness::SIZE] {
    #[inline]
    fn from(value: &RecordWitness) -> Self {
        // SAFETY: `RecordWitness` is `#[repr(transparent)]` and guaranteed to have the same
        // memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&[u8; RecordWitness::SIZE]> for &RecordWitness {
    #[inline]
    fn from(value: &[u8; RecordWitness::SIZE]) -> Self {
        // SAFETY: `RecordWitness` is `#[repr(transparent)]` and guaranteed to have the same
        // memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut RecordWitness> for &mut [u8; RecordWitness::SIZE] {
    #[inline]
    fn from(value: &mut RecordWitness) -> Self {
        // SAFETY: `RecordWitness` is `#[repr(transparent)]` and guaranteed to have the same
        // memory layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut [u8; RecordWitness::SIZE]> for &mut RecordWitness {
    #[inline]
    fn from(value: &mut [u8; RecordWitness::SIZE]) -> Self {
        // SAFETY: `RecordWitness` is `#[repr(transparent)]` and guaranteed to have the same
        // memory layout
        unsafe { mem::transmute(value) }
    }
}

impl RecordWitness {
    /// Size of record witness in bytes.
    pub const SIZE: usize = 48;
}

enum CowBytes {
    Shared(Bytes),
    Owned(BytesMut),
}

impl fmt::Debug for CowBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.as_ref()))
    }
}

impl PartialEq for CowBytes {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl Eq for CowBytes {}

impl Hash for CowBytes {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl Clone for CowBytes {
    fn clone(&self) -> Self {
        match self {
            Self::Shared(bytes) => Self::Shared(bytes.clone()),
            // Always return shared clone
            Self::Owned(bytes) => Self::Shared(Bytes::copy_from_slice(bytes)),
        }
    }
}

impl AsRef<[u8]> for CowBytes {
    fn as_ref(&self) -> &[u8] {
        match self {
            CowBytes::Shared(bytes) => bytes.as_ref(),
            CowBytes::Owned(bytes) => bytes.as_ref(),
        }
    }
}

impl AsMut<[u8]> for CowBytes {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        match self {
            CowBytes::Shared(bytes) => {
                *self = CowBytes::Owned(BytesMut::from(mem::take(bytes)));

                let CowBytes::Owned(bytes) = self else {
                    unreachable!("Just replaced; qed");
                };

                bytes.as_mut()
            }
            CowBytes::Owned(bytes) => bytes.as_mut(),
        }
    }
}

/// A piece of archival history in Subspace Network.
///
/// This version is allocated on the heap, for stack-allocated piece see [`PieceArray`].
///
/// Internally piece contains a record and corresponding witness that together with segment
/// commitment of the segment this piece belongs to can be used to verify that a piece belongs to
/// the actual archival history of the blockchain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Piece(CowBytes);

impl Encode for Piece {
    #[inline]
    fn size_hint(&self) -> usize {
        self.as_ref().size_hint()
    }

    #[inline]
    fn encode_to<O: Output + ?Sized>(&self, output: &mut O) {
        self.as_ref().encode_to(output)
    }

    #[inline]
    fn encode(&self) -> Vec<u8> {
        self.as_ref().encode()
    }

    #[inline]
    fn using_encoded<R, F: FnOnce(&[u8]) -> R>(&self, f: F) -> R {
        self.as_ref().using_encoded(f)
    }
}

impl EncodeLike for Piece {}

impl Decode for Piece {
    fn decode<I: Input>(input: &mut I) -> Result<Self, parity_scale_codec::Error> {
        let bytes =
            Bytes::decode(input).map_err(|error| error.chain("Could not decode `Piece`"))?;

        if bytes.len() != Self::SIZE {
            return Err(
                parity_scale_codec::Error::from("Incorrect Piece length").chain(format!(
                    "Expected {} bytes, found {} bytes",
                    Self::SIZE,
                    bytes.len()
                )),
            );
        }

        Ok(Piece(CowBytes::Shared(bytes)))
    }
}

impl TypeInfo for Piece {
    type Identity = Self;

    fn type_info() -> Type {
        Type::builder()
            .path(Path::new("Piece", module_path!()))
            .docs(&["A piece of archival history in Subspace Network"])
            .composite(
                Fields::unnamed().field(|f| f.ty::<[u8; Piece::SIZE]>().type_name("PieceArray")),
            )
    }
}

#[cfg(feature = "serde")]
impl Serialize for Piece {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = match &self.0 {
            CowBytes::Shared(bytes) => bytes.as_ref(),
            CowBytes::Owned(bytes) => bytes.as_ref(),
        };

        if serializer.is_human_readable() {
            hex::serde::serialize(bytes, serializer)
        } else {
            bytes.serialize(serializer)
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Piece {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = if deserializer.is_human_readable() {
            hex::serde::deserialize::<_, Vec<u8>>(deserializer).and_then(|bytes| {
                if bytes.len() == Piece::SIZE {
                    Ok(Bytes::from(bytes))
                } else {
                    Err(serde::de::Error::invalid_length(
                        bytes.len(),
                        &format!("Expected {} bytes", Piece::SIZE).as_str(),
                    ))
                }
            })?
        } else {
            Bytes::deserialize(deserializer)?
        };

        Ok(Piece(CowBytes::Shared(bytes)))
    }
}

impl Default for Piece {
    #[inline]
    fn default() -> Self {
        Self(CowBytes::Owned(BytesMut::zeroed(Self::SIZE)))
    }
}

impl From<Piece> for Vec<u8> {
    #[inline]
    fn from(piece: Piece) -> Self {
        match piece.0 {
            CowBytes::Shared(bytes) => bytes.to_vec(),
            CowBytes::Owned(bytes) => Vec::from(bytes),
        }
    }
}

impl TryFrom<&[u8]> for Piece {
    type Error = ();

    #[inline]
    fn try_from(slice: &[u8]) -> Result<Self, Self::Error> {
        if slice.len() != Self::SIZE {
            return Err(());
        }

        Ok(Self(CowBytes::Shared(Bytes::copy_from_slice(slice))))
    }
}

impl TryFrom<Vec<u8>> for Piece {
    type Error = ();

    #[inline]
    fn try_from(vec: Vec<u8>) -> Result<Self, Self::Error> {
        if vec.len() != Self::SIZE {
            return Err(());
        }

        Ok(Self(CowBytes::Shared(Bytes::from(vec))))
    }
}

impl TryFrom<Bytes> for Piece {
    type Error = ();

    #[inline]
    fn try_from(bytes: Bytes) -> Result<Self, Self::Error> {
        if bytes.len() != Self::SIZE {
            return Err(());
        }

        Ok(Self(CowBytes::Shared(bytes)))
    }
}

impl TryFrom<BytesMut> for Piece {
    type Error = ();

    #[inline]
    fn try_from(bytes: BytesMut) -> Result<Self, Self::Error> {
        if bytes.len() != Self::SIZE {
            return Err(());
        }

        Ok(Self(CowBytes::Owned(bytes)))
    }
}

impl From<&PieceArray> for Piece {
    #[inline]
    fn from(value: &PieceArray) -> Self {
        Self(CowBytes::Shared(Bytes::copy_from_slice(value.as_ref())))
    }
}

impl Deref for Piece {
    type Target = PieceArray;

    #[inline]
    fn deref(&self) -> &Self::Target {
        <&[u8; Self::SIZE]>::try_from(self.as_ref())
            .expect("Slice of memory has correct length; qed")
            .into()
    }
}

impl DerefMut for Piece {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        <&mut [u8; Self::SIZE]>::try_from(self.as_mut())
            .expect("Slice of memory has correct length; qed")
            .into()
    }
}

impl AsRef<[u8]> for Piece {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl AsMut<[u8]> for Piece {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.0.as_mut()
    }
}

impl Piece {
    /// Size of a piece (in bytes).
    pub const SIZE: usize = Record::SIZE + RecordCommitment::SIZE + RecordWitness::SIZE;

    /// Ensure piece contains cheaply cloneable shared data.
    ///
    /// Internally piece uses CoW mechanism and can store either mutable owned data or data that is
    /// cheap to clone, calling this method will ensure further clones will not result in additional
    /// memory allocations.
    pub fn to_shared(self) -> Self {
        Self(match self.0 {
            CowBytes::Shared(bytes) => CowBytes::Shared(bytes),
            CowBytes::Owned(bytes) => CowBytes::Shared(bytes.freeze()),
        })
    }
}

/// A piece of archival history in Subspace Network.
///
/// This version is allocated on the stack, for heap-allocated piece see [`Piece`].
///
/// Internally piece contains a record and corresponding witness that together with segment
/// commitment of the segment this piece belongs to can be used to verify that a piece belongs to
/// the actual archival history of the blockchain.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Deref, DerefMut, AsRef, AsMut)]
#[repr(transparent)]
pub struct PieceArray([u8; Piece::SIZE]);

impl fmt::Debug for PieceArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl Default for PieceArray {
    #[inline]
    fn default() -> Self {
        Self([0u8; Piece::SIZE])
    }
}

impl AsRef<[u8]> for PieceArray {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for PieceArray {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl From<&PieceArray> for &[u8; Piece::SIZE] {
    #[inline]
    fn from(value: &PieceArray) -> Self {
        // SAFETY: `PieceArray` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&[u8; Piece::SIZE]> for &PieceArray {
    #[inline]
    fn from(value: &[u8; Piece::SIZE]) -> Self {
        // SAFETY: `PieceArray` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut PieceArray> for &mut [u8; Piece::SIZE] {
    #[inline]
    fn from(value: &mut PieceArray) -> Self {
        // SAFETY: `PieceArray` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

impl From<&mut [u8; Piece::SIZE]> for &mut PieceArray {
    #[inline]
    fn from(value: &mut [u8; Piece::SIZE]) -> Self {
        // SAFETY: `PieceArray` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

impl PieceArray {
    /// Create boxed value without hitting stack overflow
    #[inline]
    pub fn new_boxed() -> Box<Self> {
        // TODO: Should have been just `::new()`, but https://github.com/rust-lang/rust/issues/53827
        // SAFETY: Data structure filled with zeroes is a valid invariant
        unsafe { Box::<Self>::new_zeroed().assume_init() }
    }

    /// Split piece into underlying components.
    #[inline]
    pub fn split(&self) -> (&Record, &RecordCommitment, &RecordWitness) {
        let (record, extra) = self.0.split_at(Record::SIZE);
        let (commitment, witness) = extra.split_at(RecordCommitment::SIZE);

        let record = <&[u8; Record::SIZE]>::try_from(record)
            .expect("Slice of memory has correct length; qed");
        let commitment = <&[u8; RecordCommitment::SIZE]>::try_from(commitment)
            .expect("Slice of memory has correct length; qed");
        let witness = <&[u8; RecordWitness::SIZE]>::try_from(witness)
            .expect("Slice of memory has correct length; qed");

        (record.into(), commitment.into(), witness.into())
    }

    /// Split piece into underlying mutable components.
    #[inline]
    pub fn split_mut(&mut self) -> (&mut Record, &mut RecordCommitment, &mut RecordWitness) {
        let (record, extra) = self.0.split_at_mut(Record::SIZE);
        let (commitment, witness) = extra.split_at_mut(RecordCommitment::SIZE);

        let record = <&mut [u8; Record::SIZE]>::try_from(record)
            .expect("Slice of memory has correct length; qed");
        let commitment = <&mut [u8; RecordCommitment::SIZE]>::try_from(commitment)
            .expect("Slice of memory has correct length; qed");
        let witness = <&mut [u8; RecordWitness::SIZE]>::try_from(witness)
            .expect("Slice of memory has correct length; qed");

        (record.into(), commitment.into(), witness.into())
    }

    /// Record contained within a piece.
    #[inline]
    pub fn record(&self) -> &Record {
        self.split().0
    }

    /// Mutable record contained within a piece.
    #[inline]
    pub fn record_mut(&mut self) -> &mut Record {
        self.split_mut().0
    }

    /// Commitment contained within a piece.
    #[inline]
    pub fn commitment(&self) -> &RecordCommitment {
        self.split().1
    }

    /// Mutable commitment contained within a piece.
    #[inline]
    pub fn commitment_mut(&mut self) -> &mut RecordCommitment {
        self.split_mut().1
    }

    /// Witness contained within a piece.
    #[inline]
    pub fn witness(&self) -> &RecordWitness {
        self.split().2
    }

    /// Mutable witness contained within a piece.
    #[inline]
    pub fn witness_mut(&mut self) -> &mut RecordWitness {
        self.split_mut().2
    }

    /// Convenient conversion from slice of piece array to underlying representation for efficiency
    /// purposes.
    #[inline]
    pub fn slice_to_repr(value: &[Self]) -> &[[u8; Piece::SIZE]] {
        // SAFETY: `PieceArray` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }

    /// Convenient conversion from slice of underlying representation to piece array for efficiency
    /// purposes.
    #[inline]
    pub fn slice_from_repr(value: &[[u8; Piece::SIZE]]) -> &[Self] {
        // SAFETY: `PieceArray` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }

    /// Convenient conversion from mutable slice of piece array to underlying representation for
    /// efficiency purposes.
    #[inline]
    pub fn slice_mut_to_repr(value: &mut [Self]) -> &mut [[u8; Piece::SIZE]] {
        // SAFETY: `PieceArray` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }

    /// Convenient conversion from mutable slice of underlying representation to piece array for
    /// efficiency purposes.
    #[inline]
    pub fn slice_mut_from_repr(value: &mut [[u8; Piece::SIZE]]) -> &mut [Self] {
        // SAFETY: `PieceArray` is `#[repr(transparent)]` and guaranteed to have the same memory
        // layout
        unsafe { mem::transmute(value) }
    }
}

impl From<Box<PieceArray>> for Vec<u8> {
    fn from(value: Box<PieceArray>) -> Self {
        let mut value = mem::ManuallyDrop::new(value);
        // SAFETY: Always contains fixed allocation of bytes
        unsafe { Vec::from_raw_parts(value.as_mut_ptr(), Piece::SIZE, Piece::SIZE) }
    }
}

/// Flat representation of multiple pieces concatenated for more efficient for processing
#[derive(Clone, PartialEq, Eq)]
pub struct FlatPieces(CowBytes);

impl fmt::Debug for FlatPieces {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlatPieces").finish_non_exhaustive()
    }
}

impl Deref for FlatPieces {
    type Target = [PieceArray];

    #[inline]
    fn deref(&self) -> &Self::Target {
        let bytes = self.0.as_ref();
        // SAFETY: Bytes slice has length of multiples of piece size and lifetimes of returned data
        // are preserved
        let pieces = unsafe {
            slice::from_raw_parts(
                bytes.as_ptr() as *const [u8; Piece::SIZE],
                bytes.len() / Piece::SIZE,
            )
        };
        PieceArray::slice_from_repr(pieces)
    }
}

impl DerefMut for FlatPieces {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let bytes = self.0.as_mut();
        // SAFETY: Bytes slice has length of multiples of piece size and lifetimes of returned data
        // are preserved
        let pieces = unsafe {
            slice::from_raw_parts_mut(
                bytes.as_mut_ptr() as *mut [u8; Piece::SIZE],
                bytes.len() / Piece::SIZE,
            )
        };
        PieceArray::slice_mut_from_repr(pieces)
    }
}

impl FlatPieces {
    /// Allocate `FlatPieces` that will hold `piece_count` pieces filled with zeroes
    #[inline]
    pub fn new(piece_count: usize) -> Self {
        Self(CowBytes::Owned(BytesMut::zeroed(piece_count * Piece::SIZE)))
    }

    /// Iterate over all pieces.
    ///
    /// NOTE: Unless [`Self::to_shared`] was called first, iterator may have to allocate each piece
    /// from scratch, which is rarely a desired behavior.
    #[inline]
    pub fn pieces(&self) -> Box<dyn ExactSizeIterator<Item = Piece> + '_> {
        match &self.0 {
            CowBytes::Shared(bytes) => Box::new(
                bytes
                    .chunks_exact(Piece::SIZE)
                    .map(|slice| Piece(CowBytes::Shared(bytes.slice_ref(slice)))),
            ),
            CowBytes::Owned(bytes) => Box::new(
                bytes
                    .chunks_exact(Piece::SIZE)
                    .map(|slice| Piece(CowBytes::Shared(Bytes::copy_from_slice(slice)))),
            ),
        }
    }

    /// Iterator over source pieces (even indices)
    #[inline]
    pub fn source_pieces(&self) -> impl ExactSizeIterator<Item = Piece> + '_ {
        self.pieces().step_by(2)
    }

    /// Iterator over source pieces (even indices)
    #[inline]
    pub fn source(&self) -> impl ExactSizeIterator<Item = &'_ PieceArray> + '_ {
        self.iter().step_by(2)
    }

    /// Mutable iterator over source pieces (even indices)
    #[inline]
    pub fn source_mut(&mut self) -> impl ExactSizeIterator<Item = &'_ mut PieceArray> + '_ {
        self.iter_mut().step_by(2)
    }

    /// Iterator over parity pieces (odd indices)
    #[inline]
    pub fn parity_pieces(&self) -> impl ExactSizeIterator<Item = Piece> + '_ {
        self.pieces().skip(1).step_by(2)
    }

    /// Iterator over parity pieces (odd indices)
    #[inline]
    pub fn parity(&self) -> impl ExactSizeIterator<Item = &'_ PieceArray> + '_ {
        self.iter().skip(1).step_by(2)
    }

    /// Mutable iterator over parity pieces (odd indices)
    #[inline]
    pub fn parity_mut(&mut self) -> impl ExactSizeIterator<Item = &'_ mut PieceArray> + '_ {
        self.iter_mut().skip(1).step_by(2)
    }

    /// Ensure flat pieces contains cheaply cloneable shared data.
    ///
    /// Internally flat pieces uses CoW mechanism and can store either mutable owned data or data
    /// that is cheap to clone, calling this method will ensure further clones and returned pieces
    /// will not result in additional memory allocations.
    pub fn to_shared(self) -> Self {
        Self(match self.0 {
            CowBytes::Shared(bytes) => CowBytes::Shared(bytes),
            CowBytes::Owned(bytes) => CowBytes::Shared(bytes.freeze()),
        })
    }
}

#[cfg(feature = "parallel")]
impl FlatPieces {
    /// Parallel iterator over source pieces (even indices)
    #[inline]
    pub fn par_source(&self) -> impl IndexedParallelIterator<Item = &'_ PieceArray> + '_ {
        self.par_iter().step_by(2)
    }

    /// Mutable parallel iterator over source pieces (even indices)
    #[inline]
    pub fn par_source_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = &'_ mut PieceArray> + '_ {
        self.par_iter_mut().step_by(2)
    }

    /// Parallel iterator over parity pieces (odd indices)
    #[inline]
    pub fn par_parity(&self) -> impl IndexedParallelIterator<Item = &'_ PieceArray> + '_ {
        self.par_iter().skip(1).step_by(2)
    }

    /// Mutable parallel iterator over parity pieces (odd indices)
    #[inline]
    pub fn par_parity_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = &'_ mut PieceArray> + '_ {
        self.par_iter_mut().skip(1).step_by(2)
    }
}
