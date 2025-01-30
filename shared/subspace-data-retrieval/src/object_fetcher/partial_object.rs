//! Object fetching internals for partial objects.
//!
//! Partial objects can contain potential segment padding at the start, middle, or end of the
//! object. They never contain segment headers, which are stripped before reconstruction.
//!
//! Objects are reconstructed from 3 kinds of pieces:
//! - pieces at the start of a segment, which have a variable-length segment header,
//! - pieces in the middle of a segment, which have no header or padding, and
//! - pieces at the end of a segment, which have variable-length padding.
//!
//! This module provides a uniform interface for handling these different kinds of pieces:
//! - `SegmentDataLength` is used to track the length of the partial object data before and in
//!   potential segment padding. It also tracks the presence of segment headers.
//! - `PartialData` is used to store the partial object data (including potential segment padding)
//!   before the object length is known, and before each new piece is added to the partial object.
//! - `PartialObject` is used to store the partial object data after the object length is known.

use crate::object_fetcher::segment_header::{strip_segment_header, MAX_SEGMENT_PADDING};
use crate::object_fetcher::{decode_data_length, Error, MAX_ENCODED_LENGTH_SIZE};
use parity_scale_codec::{Decode, Input};
use std::collections::BTreeSet;
use std::iter;
use std::iter::TrustedLen;
use subspace_core_primitives::hashes::{blake3_hash, Blake3Hash};
use subspace_core_primitives::objects::GlobalObject;
use subspace_core_primitives::pieces::{PieceIndex, RawRecord};
use subspace_core_primitives::segments::RecordedHistorySegment;
use tracing::{debug, trace};

/// The fixed value of every padding byte.
const PADDING_BYTE_VALUE: u8 = 0;

/// The segment prefix data and padding lengths for a partial object.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SegmentDataLength {
    /// `true` if we need to manually skip the segment header in the next piece.
    pub need_to_skip_header: bool,

    /// The length of the object data before potential segment padding.
    /// This can be zero.
    pub prefix_data_length: usize,

    /// The length of potential segment padding after the prefix data.
    /// This can be outside the actual object (but we won't know that until we read the object's
    /// length).
    pub maybe_padding_data_length: usize,
}

impl SegmentDataLength {
    /// Returns the segment padding length for a mapping piece and offset.
    pub fn new(piece_index: PieceIndex, offset: u32) -> SegmentDataLength {
        // The last few bytes might contain padding if a piece is the last piece in the segment.
        let piece_position_in_segment = piece_index.source_position() as usize;
        let data_shards = RecordedHistorySegment::NUM_RAW_RECORDS;
        let last_data_piece_in_segment = piece_position_in_segment >= data_shards - 1;

        // How much bytes are definitely available starting at `piece_index` and `offset` without
        // crossing a segment boundary.
        //
        // The last few bytes might contain padding if a piece is the last piece in the segment.
        let bytes_available_in_segment =
            (data_shards - piece_position_in_segment) * RawRecord::SIZE - offset as usize;

        if !last_data_piece_in_segment || bytes_available_in_segment > MAX_SEGMENT_PADDING {
            // If it's not the last data piece, or the object starts outside the padding, the
            // object can have the entire segment (minus the padding)
            SegmentDataLength {
                need_to_skip_header: false,
                prefix_data_length: bytes_available_in_segment - MAX_SEGMENT_PADDING,
                maybe_padding_data_length: MAX_SEGMENT_PADDING,
            }
        } else {
            // If the object starts inside the padding, there's no prefix before the padding
            SegmentDataLength {
                need_to_skip_header: false,
                prefix_data_length: 0,
                maybe_padding_data_length: bytes_available_in_segment,
            }
        }
    }
}

/// The data before an object's length is known.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PartialData {
    /// The object data before segment padding, starting with the encoded object length.
    ///
    /// This can be empty for an object which starts in segment padding.
    prefix_data: Vec<u8>,

    /// The potential padding data. The bytes in this data can have any values.
    maybe_padding_data: Vec<u8>,

    /// The partial object data after segment padding.
    /// For objects which don't overlap the end of a segment, this is empty.
    ///
    /// The encoded object length might overlap with the start of the suffix data.
    suffix_data: Vec<u8>,
}

impl PartialData {
    /// Create a new empty `PartialData` object.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the supplied `piece_data` to the `PartialData` object, and updates the segment data lengths.
    ///
    /// The mapping is only used for error reporting.
    pub fn add_piece_data(
        &mut self,
        segment_data_length: &mut SegmentDataLength,
        piece_index: PieceIndex,
        mut piece_data: Vec<u8>,
        mapping: GlobalObject,
    ) -> Result<(), Error> {
        if segment_data_length.need_to_skip_header {
            let (piece_data, max_remaining_object_bytes) =
                strip_segment_header(piece_data, piece_index.segment_index(), mapping)?;

            // Objects are currently much smaller than segments, so we know there's no further
            // headers or segment padding in an object that's near the start of a segment.
            // TODO: support objects that span multiple segments
            segment_data_length.need_to_skip_header = false;
            segment_data_length.prefix_data_length = max_remaining_object_bytes;
            segment_data_length.maybe_padding_data_length = 0;

            self.add_piece_data_without_padding(piece_data);
        } else if piece_data.len() <= segment_data_length.prefix_data_length
            || segment_data_length.maybe_padding_data_length == 0
        {
            // There's no header and no padding in this piece - the typical case
            piece_data.truncate(segment_data_length.prefix_data_length);
            segment_data_length.prefix_data_length -= piece_data.len();

            self.add_piece_data_without_padding(piece_data);
        } else {
            // We only download whole pieces, and we only strip data off the front of pieces.
            // So if we have any padding, we have all remaining padding in the segment.
            let maybe_padding_data = piece_data
                .split_off(piece_data.len() - segment_data_length.maybe_padding_data_length);
            assert_eq!(segment_data_length.prefix_data_length, piece_data.len());

            // After reading padding, we need to skip the segment header in the next piece.
            segment_data_length.need_to_skip_header = true;
            // Segment headers are variable length, so we don't know how much data is in the next
            // segment
            segment_data_length.prefix_data_length = 0;
            segment_data_length.maybe_padding_data_length = 0;

            self.add_piece_data_with_padding(&piece_data, &maybe_padding_data);
        }

        Ok(())
    }

    /// Add data with potential padding to the end of this partial data.
    /// This method can also handle data that doesn't actually have any padding.
    ///
    /// Panics if the object already has padding.
    fn add_piece_data_with_padding(
        &mut self,
        new_prefix_data: &[u8],
        new_maybe_padding_data: &[u8],
    ) {
        // Doesn't actually have padding
        if new_maybe_padding_data.is_empty() {
            self.add_piece_data_without_padding(new_prefix_data.to_vec());
            return;
        }

        assert!(
            !self.has_padding(),
            "add_piece_data_with_padding() can only be called once: {self:?}, \
             {new_prefix_data:?}, {new_maybe_padding_data:?}"
        );

        self.prefix_data.extend_from_slice(new_prefix_data);
        // The padding was empty, so this does assignment (but without allocating another vector)
        self.maybe_padding_data
            .extend_from_slice(new_maybe_padding_data);
    }

    /// Add padding-less `piece_data` to the end of this partial object.
    fn add_piece_data_without_padding(&mut self, mut piece_data: Vec<u8>) {
        if self.has_padding() {
            // If there might be padding, or there is suffix data, new data must be added at the end
            // of the suffix data.
            self.suffix_data.append(&mut piece_data);
        } else {
            // Otherwise, the data belongs to the existing prefix data.
            self.prefix_data.append(&mut piece_data);
        }
    }

    /// Returns true if this object has padding, or might have padding.
    fn has_padding(&self) -> bool {
        !self.maybe_padding_data.is_empty() || !self.suffix_data.is_empty()
    }
}

/// A possible data length and segment padding length for an object.
/// Sorts by data length first, then padding length.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ObjectLength {
    /// The length of the object data, including the encoded length.
    data_length: usize,

    /// The number of zero-valued bytes between the prefix and suffix data.
    /// These bytes could be segment padding, or part of the object data.
    padding_length: usize,
}

/// The state of a single object that is being fetched.
/// Each object is contained within a single block.
///
/// An object at the end of a segment needs to ignore any segment padding. To avoid reconstructing
/// the entire segment, we try each possible padding against the hash instead. We don't need to
/// store the padding bytes, because padding is always zero-valued.
///
/// Objects also need to ignore the parent segment header and `BlockContinuation` header at the
/// start of a segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartialObject {
    /// The object data before segment padding, starting with the encoded object length.
    ///
    /// This can be empty for an object which starts in segment padding.
    /// (Such objects are always zero length, because padding is always zero-valued.)
    prefix_data: Vec<u8>,

    /// The partial object data after segment padding.
    /// For objects which don't overlap the end of a segment, this is empty.
    ///
    /// The encoded object length might overlap with the start of the suffix data.
    suffix_data: Vec<u8>,

    /// The possible object lengths and their corresponding padding lengths.
    /// For objects which aren't at the end of a segment, there is only one data length, and the
    /// padding length is zero.
    lengths: BTreeSet<ObjectLength>,
}

impl PartialObject {
    /// Given prefix data, potential padding data, and suffix data, returns a new `PartialObject`.
    /// Returns `Ok(None)` if more data is needed, or an error if object retrieval failed.
    ///
    /// This method can also handle data that doesn't actually have any padding.
    ///
    /// The mapping is only used for error reporting.
    ///
    /// Panics if an object has a suffix, but no padding.
    pub fn new_with_padding(
        partial_data: &PartialData,
        max_object_len: usize,
        mapping: GlobalObject,
    ) -> Result<Option<Self>, Error> {
        let PartialData {
            prefix_data,
            maybe_padding_data,
            suffix_data,
        } = partial_data;

        // No padding, so the data is continuous
        if maybe_padding_data.is_empty() {
            assert!(
                suffix_data.is_empty(),
                "suffix_data can only have bytes if there might be padding bytes"
            );
            return Self::new_without_padding(prefix_data, max_object_len, mapping);
        }

        // Count the padding bytes
        let (non_padding_data, object_max_segment_padding_length) =
            Self::count_padding_bytes(maybe_padding_data);

        // Keep the remaining non-padding bytes
        let mut prefix_data = prefix_data.to_vec();
        prefix_data.extend_from_slice(non_padding_data);

        // Find the possible lengths for the object, with and without each padding byte
        let mut lengths = BTreeSet::new();
        let mut last_error = None;

        for padding_length in 0..=object_max_segment_padding_length {
            let length_prefix = padded_data(&prefix_data, padding_length, suffix_data)
                .take(MAX_ENCODED_LENGTH_SIZE)
                .collect::<Vec<u8>>();

            match decode_data_length(length_prefix.as_slice(), max_object_len, mapping) {
                // A valid length
                Ok(Some((length_prefix_len, data_length))) => {
                    lengths.insert(ObjectLength {
                        data_length: length_prefix_len + data_length,
                        padding_length,
                    });
                }
                Ok(None) => {
                    // Not enough data to decode all the lengths yet (this is a rare edge case).
                    return Ok(None);
                }
                Err(err) => {
                    // This length is invalid and needs to be skipped.
                    last_error = Some(err);
                    continue;
                }
            }
        }

        if lengths.is_empty() {
            // All lengths were invalid
            Err(last_error.expect("last_error is set if lengths is empty; qed"))
        } else {
            Ok(Some(Self {
                prefix_data,
                suffix_data: suffix_data.to_vec(),
                lengths,
            }))
        }
    }

    /// Given the data for an object without any potential segment padding, returns a new
    /// `PartialObject`.
    ///
    /// Returns `Ok(None)` if more data is needed, or an error if object retrieval failed.
    ///
    /// The mapping is only used for error reporting.
    fn new_without_padding(
        data: &[u8],
        max_object_len: usize,
        mapping: GlobalObject,
    ) -> Result<Option<Self>, Error> {
        let Some((length_prefix_len, data_length)) =
            decode_data_length(data, max_object_len, mapping)?
        else {
            // Not enough data yet (this is a rare edge case).
            return Ok(None);
        };

        Ok(Some(Self {
            prefix_data: data.to_vec(),
            suffix_data: Vec::new(),
            lengths: [ObjectLength {
                data_length: length_prefix_len + data_length,
                padding_length: 0,
            }]
            .into(),
        }))
    }

    /// Count the actual padding bytes in `maybe_padding_data`.
    /// Returns the remaining non-padding data, and the number of padding bytes.
    fn count_padding_bytes(mut maybe_padding_data: &[u8]) -> (&[u8], usize) {
        // Count the padding bytes backwards from the end of the segment
        let mut object_max_segment_padding_length = 0;
        while let Some(maybe_padding_byte) = maybe_padding_data.take_last() {
            if *maybe_padding_byte != PADDING_BYTE_VALUE {
                break;
            }

            object_max_segment_padding_length += 1;
        }

        (maybe_padding_data, object_max_segment_padding_length)
    }

    /// Add `piece_data` with padding to the end of this partial object.
    /// This method can also handle data that doesn't actually have any padding.
    pub fn add_piece_data_with_padding(&mut self, mut new_data: PartialData) {
        // Doesn't actually have padding
        if new_data.maybe_padding_data.is_empty() {
            let mut piece_data = new_data.prefix_data.to_vec();
            piece_data.extend_from_slice(&new_data.suffix_data);

            self.add_piece_data_without_padding(piece_data);
            return;
        }

        assert!(
            !self.has_padding(),
            "add_piece_data_with_padding() can only be called once: {self:?}, {new_data:?}"
        );

        // Count the padding bytes
        let (non_padding_data, object_max_segment_padding_length) =
            Self::count_padding_bytes(&new_data.maybe_padding_data);

        // Keep the remaining non-padding bytes
        new_data.prefix_data.extend_from_slice(non_padding_data);

        // Add the new data to the existing data
        self.prefix_data.append(&mut new_data.prefix_data);
        self.suffix_data = new_data.suffix_data;

        // There's only one data length, but now we have multiple padding lengths
        let data_length = self.shortest_object_length();
        // Avoid a range panic if there's actually no padding, by overwriting the existing length
        for padding_length in 0..=object_max_segment_padding_length {
            self.lengths.insert(ObjectLength {
                data_length,
                padding_length,
            });
        }
    }

    /// Add padding-less `piece_data` to the end of this partial object.
    pub fn add_piece_data_without_padding(&mut self, mut piece_data: Vec<u8>) {
        if self.has_padding() {
            // If there might be padding, or there is suffix data, new data must be added at the end
            // of the suffix data.
            self.suffix_data.append(&mut piece_data);
        } else {
            // For performance, add the data to the existing prefix data.
            self.prefix_data.append(&mut piece_data);
        }
    }

    /// Returns true if this object has padding, or might have padding.
    ///
    /// Objects can have no padding in three different ways:
    /// - they aren't at the end of a segment
    /// - they are at the end of a segment, but all potential padding bytes are non-zero
    /// - we've checked other padding lengths and they were invalid
    ///
    /// Panics if the object has no valid lengths left.
    fn has_padding(&self) -> bool {
        !self.suffix_data.is_empty()
            || self.lengths.len() > 1
            || self
                .lengths
                .first()
                .expect("other methods return an error if lengths becomes empty; qed")
                .padding_length
                == 0
    }

    /// Returns the shortest possible amount of fetched data for the object, based on potential segment padding.
    ///
    /// Panics if the object has no valid lengths left.
    pub fn shortest_fetched_data_length(&self) -> usize {
        self.prefix_data.len() + self.shortest_object_padding_length() + self.suffix_data.len()
    }

    /// Returns the shortest possible length for the object, based on potential segment padding.
    ///
    /// If the segment padding is outside the length prefix, the object length is known, and only
    /// the object data varies.
    ///
    /// Panics if the object has no valid lengths left.
    pub fn shortest_object_length(&self) -> usize {
        self.lengths
            .first()
            .expect("other methods return an error if lengths becomes empty; qed")
            .data_length
    }

    /// Returns the padding length for the shortest data length.
    /// Adding padding can change the object length, so this might not be the shortest number of
    /// added padding bytes.
    ///
    /// Panics if the object has no valid lengths left.
    pub fn shortest_object_padding_length(&self) -> usize {
        self.lengths
            .first()
            .expect("other methods return an error if lengths becomes empty; qed")
            .padding_length
    }

    /// Returns the longest possible length for the object, based on potential segment padding.
    ///
    /// Panics if the object has no valid lengths left.
    pub fn longest_object_length(&self) -> usize {
        self.lengths
            .last()
            .expect("other methods return an error if lengths becomes empty; qed")
            .data_length
    }

    /// Returns the shortest possible data for the object, based on potential segment padding.
    ///
    /// If the segment padding is outside the length prefix, the object length is known, and only
    /// the object data varies. These ties are broken using the shortest padding length.
    ///
    /// Panics if the object has no valid lengths left.
    pub fn shortest_object_data(&self) -> ObjectDataInput<impl TrustedLen<Item = u8> + use<'_>> {
        let length = self
            .lengths
            .first()
            .expect("other methods return an error if lengths becomes empty; qed");

        ObjectDataInput(
            padded_data(&self.prefix_data, length.padding_length, &self.suffix_data)
                .take(self.shortest_object_length()),
        )
    }

    /// Check the hash against all possible objects with enough data.
    /// If a possible object's data doesn't match the mapping hash, its length is removed.
    ///
    /// Returns valid object data, `Ok(None)` if we need more data, or an error if all possible
    /// objects have invalid hashes.
    pub fn check_available_objects(
        &mut self,
        mapping: GlobalObject,
    ) -> Result<Option<Vec<u8>>, Error> {
        while self.shortest_object_length() <= self.shortest_fetched_data_length() {
            let data = Vec::<u8>::decode(&mut self.shortest_object_data())
                .map_err(|source| Error::ObjectDecoding { source, mapping })?;

            let data_hash = blake3_hash(&data);
            if data_hash == mapping.hash {
                return Ok(Some(data));
            } else {
                trace!(data = %hex::encode(&data), "Potential object data");

                // If we've run out of lengths to try, return a hash mismatch error
                self.mark_shortest_object_hash_invalid(data_hash, mapping)?;
            }
        }

        Ok(None)
    }

    /// Remove the shortest data, because it has an invalid hash.
    /// Call this method if the shortest data has an incorrect hash.
    ///
    /// The mapping is only used for error reporting.
    ///
    /// Returns an error if there are no object lengths left to try.
    fn mark_shortest_object_hash_invalid(
        &mut self,
        data_hash: Blake3Hash,
        mapping: GlobalObject,
    ) -> Result<(), Error> {
        if self.lengths.len() > 1 {
            // We still have more lengths to try.
            let ignored_err = Error::InvalidDataHash {
                data_hash,
                data_length: self.shortest_object_length(),
                mapping,
            };
            debug!(
                ?ignored_err,
                "Invalid data hash, trying next padding length"
            );

            self.lengths.pop_first();
            Ok(())
        } else {
            // There are no lengths left, so we know all the possible data/padding combinations are invalid.
            Err(Error::InvalidDataHash {
                data_hash,
                data_length: self.shortest_object_length(),
                mapping,
            })
        }
    }
}

/// A wrapper struct which impls Input for a PartialObject's data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectDataInput<I>(I)
where
    I: TrustedLen<Item = u8>;

impl<I> Input for ObjectDataInput<I>
where
    I: TrustedLen<Item = u8>,
{
    fn remaining_len(&mut self) -> Result<Option<usize>, parity_scale_codec::Error> {
        Ok(Some(self.len()))
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<(), parity_scale_codec::Error> {
        if buf.len() > self.len() {
            return Err("Not enough data to fill buffer".into());
        }

        let buf_vec = (&mut self.0).take(buf.len()).collect::<Vec<u8>>();
        buf.copy_from_slice(&buf_vec);

        Ok(())
    }
}

impl<I> ObjectDataInput<I>
where
    I: TrustedLen<Item = u8>,
{
    /// Returns the remaining length of the data.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        let (lower, upper) = self.0.size_hint();

        assert_eq!(
            lower,
            upper.expect("max object length is much less than usize::MAX; qed"),
            "ObjectDataInput is a TrustedLen; qed"
        );

        lower
    }
}

/// Returns an iterator over the padded data for `prefix_data` and `suffix_data`.
fn padded_data<'data>(
    prefix_data: &'data [u8],
    padding_length: usize,
    suffix_data: &'data [u8],
) -> impl TrustedLen<Item = u8> + 'data {
    prefix_data
        .iter()
        .chain(iter::repeat_n(&PADDING_BYTE_VALUE, padding_length))
        .chain(suffix_data.iter())
        .copied()
}

#[cfg(test)]
mod test {
    use super::*;
    use parity_scale_codec::Encode;
    use subspace_archiving::archiver::SegmentItem;

    #[test]
    fn padding_byte_value_constant() {
        let padding_byte = SegmentItem::Padding.encode();
        assert_eq!(padding_byte, vec![PADDING_BYTE_VALUE]);
    }
}
