//! Object fetching internals for segment headers.
//!
//! This module implements segment header stripping. Segment headers are stripped from pieces
//! before they are used to reconstruct objects.

use crate::object_fetcher::{decode_data_length, Error};
use parity_scale_codec::{Decode, Encode, Input, IoReader};
use std::io::Cursor;
use subspace_archiving::archiver::SegmentItem;
use subspace_core_primitives::hashes::Blake3Hash;
use subspace_core_primitives::objects::GlobalObject;
use subspace_core_primitives::segments::{
    ArchivedBlockProgress, LastArchivedBlock, SegmentCommitment, SegmentHeader, SegmentIndex,
};
use subspace_runtime_primitives::MAX_BLOCK_LENGTH;

/// The maximum amount of segment padding.
///
/// This is the difference between the lengths of the compact encodings of the minimum and maximum
/// block sizes in the consensus chain. As of January 2025, the minimum block size is (potentially)
/// 63 or less, and the maximum block size is in the range 2^14 to 2^30 - 1.
/// <https://docs.substrate.io/reference/scale-codec/#fn-1>
pub const MAX_SEGMENT_PADDING: usize = 3;

/// The minimum size of a segment header.
#[inline]
pub fn min_segment_header_encoded_size() -> usize {
    let min_segment_header = SegmentHeader::V0 {
        segment_index: 0.into(),
        segment_commitment: SegmentCommitment::default(),
        prev_segment_header_hash: Blake3Hash::default(),
        last_archived_block: LastArchivedBlock {
            number: 0,
            archived_progress: ArchivedBlockProgress::Complete,
        },
    };

    min_segment_header.encoded_size()
}

/// The maximum size of the segment header.
pub fn max_segment_header_encoded_size() -> usize {
    let max_segment_header = SegmentHeader::V0 {
        segment_index: u64::MAX.into(),
        segment_commitment: SegmentCommitment::default(),
        prev_segment_header_hash: Blake3Hash::default(),
        last_archived_block: LastArchivedBlock {
            number: u32::MAX,
            archived_progress: ArchivedBlockProgress::Partial(u32::MAX),
        },
    };

    max_segment_header.encoded_size()
}

/// Removes the segment header from the start of a piece, and returns the remaining data.
/// Also returns the maximum remaining bytes in the object.
///
/// The maximum remaining bytes is the length of the data in the block continuation containing the
/// object. This block continuation might span multiple pieces, and can contain multiple objects
/// (or other data).
///
/// Returns an error if the data is too short to contain a segment header, or if the header is
/// invalid.
///
/// The segment index and mapping are only used for error reporting.
pub fn strip_segment_header(
    piece_data: Vec<u8>,
    segment_index: SegmentIndex,
    mapping: GlobalObject,
) -> Result<(Vec<u8>, usize), Error> {
    let mut piece_data = IoReader(Cursor::new(piece_data));

    // Decode::decode() wants to read the entire segment here, so we have to decode it manually.
    // In SCALE encoding, variants are always one byte.
    let segment_variant = piece_data
        .read_byte()
        .map_err(|source| Error::SegmentDecoding {
            source,
            segment_index,
            mapping,
        })?;
    // We only know how to decode variant 0.
    if segment_variant != 0 {
        return Err(Error::UnknownSegmentVariant {
            segment_variant,
            segment_index,
            mapping,
        });
    }

    // Variant 0 consists of a list of items, with no length prefix.
    let segment_item =
        SegmentItem::decode(&mut piece_data).map_err(|source| Error::SegmentDecoding {
            source,
            segment_index,
            mapping,
        })?;

    // The parent segment header is always first.
    let SegmentItem::ParentSegmentHeader(_) = segment_item else {
        return Err(Error::UnexpectedSegmentItem {
            segment_progress: piece_data.0.position() as usize,
            segment_index,
            segment_item: Box::new(segment_item),
            mapping,
        });
    };

    // Since we're reading a continuing object, the next item must be a block continuation.
    // We want to discard its header and keep its data. But the block continuation might span
    // multiple pieces. So we need to read its header manually, too.
    let segment_item_variant = piece_data
        .read_byte()
        .map_err(|source| Error::SegmentDecoding {
            source,
            segment_index,
            mapping,
        })?;

    // Now strip off the header so we can read the block continuation length.
    let header_bytes = piece_data.0.position() as usize;
    let mut piece_data = piece_data.0.into_inner().split_off(header_bytes);
    let segment_item_lengths = decode_data_length(&piece_data, MAX_BLOCK_LENGTH as usize, mapping)?;

    // Block continuations are variant 3
    if segment_item_variant != 3 || segment_item_lengths.is_none() {
        return Err(Error::UnexpectedSegmentItemVariant {
            segment_progress: header_bytes,
            segment_index,
            segment_item_variant,
            segment_item_lengths,
            mapping,
        });
    }

    let (segment_item_prefix_len, segment_item_data_len) =
        segment_item_lengths.expect("just checked length is Some; qed");
    // Now strip off the length prefix, and any bytes that aren't in the block continuation.
    let mut piece_data = piece_data.split_off(segment_item_prefix_len);
    piece_data.truncate(segment_item_data_len);

    Ok((piece_data, segment_item_data_len))
}

#[cfg(test)]
mod test {
    use super::*;
    use parity_scale_codec::{Compact, CompactLen};
    use subspace_runtime_primitives::MAX_BLOCK_LENGTH;

    #[test]
    fn max_segment_padding_constant() {
        assert_eq!(
            MAX_SEGMENT_PADDING,
            Compact::compact_len(&MAX_BLOCK_LENGTH) - Compact::<u32>::compact_len(&1)
        );
    }

    #[test]
    fn segment_header_length_constants() {
        assert!(
            min_segment_header_encoded_size() < max_segment_header_encoded_size(),
            "min_segment_header_encoded_size: {} must be less than max_segment_header_encoded_size: {}",
            min_segment_header_encoded_size(),
            max_segment_header_encoded_size()
        );
    }
}
