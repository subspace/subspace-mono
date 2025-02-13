//! Fetching objects stored in the archived history of Subspace Network.

use crate::piece_fetcher::download_pieces;
use crate::piece_getter::PieceGetter;
use crate::segment_downloading::{download_segment, SegmentDownloadingError};
use parity_scale_codec::{Compact, CompactLen, Decode, Encode};
use std::sync::Arc;
use subspace_archiving::archiver::{Segment, SegmentItem};
use subspace_core_primitives::hashes::{blake3_hash, Blake3Hash};
use subspace_core_primitives::objects::{GlobalObject, GlobalObjectMapping};
use subspace_core_primitives::pieces::{Piece, PieceIndex, RawRecord};
use subspace_core_primitives::segments::{RecordedHistorySegment, SegmentIndex};
use subspace_erasure_coding::ErasureCoding;
use tracing::{debug, trace, warn};

/// The maximum amount of segment padding.
///
/// This is the difference between the lengths of the compact encodings of the minimum and maximum
/// block sizes in the consensus chain. As of January 2025, the minimum block size is (potentially)
/// 63 or less, and the maximum block size is in the range 2^14 to 2^30 - 1.
/// <https://docs.substrate.io/reference/scale-codec/#fn-1>
pub const MAX_SEGMENT_PADDING: usize = 3;

/// The maximum object length this module can handle.
///
/// Currently objects are limited by the largest block size on the consensus chain, which is 5 MB.
/// But this implementation supports the maximum length of the 4 byte scale encoding.
pub const MAX_SUPPORTED_OBJECT_LENGTH: usize = 1024 * 1024 * 1024 - 1;

/// Object fetching errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Supplied piece index is not a source piece
    #[error("Piece index is not a source piece, object: {mapping:?}")]
    NotSourcePiece { mapping: GlobalObject },

    /// Supplied piece offset is too large
    #[error(
        "Piece offset is too large, must be less than {}, object: {mapping:?}",
        RawRecord::SIZE
    )]
    PieceOffsetTooLarge { mapping: GlobalObject },

    /// No item in segment at offset
    #[error(
        "Offset {offset_in_segment} in segment {segment_index} is not an item, \
         current progress: {progress}, object: {mapping:?}"
    )]
    NoSegmentItem {
        progress: usize,
        offset_in_segment: usize,
        segment_index: SegmentIndex,
        mapping: GlobalObject,
    },

    /// Unexpected item in first segment at offset
    #[error(
        "Offset {offset_in_segment} in first segment {segment_index} has unexpected item, \
         current progress: {segment_progress}, object: {mapping:?}, item: {segment_item:?}"
    )]
    UnexpectedFirstSegmentItem {
        segment_progress: usize,
        offset_in_segment: usize,
        segment_index: SegmentIndex,
        segment_item: Box<SegmentItem>,
        mapping: GlobalObject,
    },

    /// Unexpected item in continuing segment at offset
    #[error(
        "Continuing segment {segment_index} has unexpected item, \
         collected data: {collected_data}, object: {mapping:?}, item: {segment_item:?}"
    )]
    UnexpectedContinuingSegmentItem {
        collected_data: usize,
        segment_index: SegmentIndex,
        segment_item: Box<SegmentItem>,
        mapping: GlobalObject,
    },

    /// Object not found after downloading expected number of segments
    #[error(
        "Object segment range {first_segment_index}..={last_segment_index} did not contain \
         full object, object: {mapping:?}"
    )]
    TooManySegments {
        first_segment_index: SegmentIndex,
        last_segment_index: SegmentIndex,
        mapping: GlobalObject,
    },

    /// Object is too large error
    #[error(
        "Data length {data_length} exceeds maximum object size {max_object_len} \
         for object: {mapping:?}"
    )]
    ObjectTooLarge {
        data_length: usize,
        max_object_len: usize,
        mapping: GlobalObject,
    },

    /// Length prefix is too large error
    #[error(
        "Length prefix length {length_prefix_len} exceeds maximum object size {max_object_len} \
         for object: {mapping:?}"
    )]
    LengthPrefixTooLarge {
        length_prefix_len: usize,
        max_object_len: usize,
        mapping: GlobalObject,
    },

    /// Hash doesn't match data
    #[error("Incorrect data hash {data_hash:?} for {data_length} byte object: {mapping:?}")]
    InvalidDataHash {
        data_hash: Blake3Hash,
        data_length: usize,
        mapping: GlobalObject,
    },

    /// Object decoding error
    #[error("Object data decoding error: {source:?}, object: {mapping:?}")]
    ObjectDecoding {
        source: parity_scale_codec::Error,
        mapping: GlobalObject,
    },

    /// Segment getter error
    #[error("Getting segment failed: {source:?}, object: {mapping:?}")]
    SegmentGetter {
        source: SegmentDownloadingError,
        mapping: GlobalObject,
    },

    /// Piece getter error
    #[error("Getting piece caused an error: {source:?}, object: {mapping:?}")]
    PieceGetterError {
        source: anyhow::Error,
        mapping: GlobalObject,
    },

    /// Piece getter couldn't find the piece
    #[error("Piece {piece_index:?} was not found by piece getter")]
    PieceNotFound { piece_index: PieceIndex },
}

/// Object fetcher for the Subspace DSN.
pub struct ObjectFetcher<PG>
where
    PG: PieceGetter + Send + Sync,
{
    /// The piece getter used to fetch pieces.
    piece_getter: Arc<PG>,

    /// The erasure coding configuration of those pieces.
    erasure_coding: ErasureCoding,

    /// The maximum number of data bytes we'll read for a single object.
    max_object_len: usize,
}

impl<PG> ObjectFetcher<PG>
where
    PG: PieceGetter + Send + Sync,
{
    /// Create a new object fetcher with the given configuration.
    ///
    /// `max_object_len` is the amount of data bytes we'll read for a single object before giving
    /// up and returning an error. In this implementation, it is limited to
    /// [`MAX_SUPPORTED_OBJECT_LENGTH`], which is much larger than the largest block on any domain
    /// (as of December 2024).
    pub fn new(
        piece_getter: Arc<PG>,
        erasure_coding: ErasureCoding,
        mut max_object_len: usize,
    ) -> Self {
        if max_object_len > MAX_SUPPORTED_OBJECT_LENGTH {
            warn!(
                max_object_len,
                MAX_SUPPORTED_OBJECT_LENGTH,
                "Object fetcher size limit exceeds maximum supported object size, \
                limiting to implementation-supported size"
            );

            max_object_len = MAX_SUPPORTED_OBJECT_LENGTH;
        }

        Self {
            piece_getter,
            erasure_coding,
            max_object_len,
        }
    }

    /// Assemble the objects in `mapping` by fetching necessary pieces using the piece getter, and
    /// putting the objects' bytes together.
    ///
    /// Checks the objects' hashes to make sure the correct bytes are returned.
    pub async fn fetch_objects(
        &self,
        mappings: GlobalObjectMapping,
    ) -> Result<Vec<Vec<u8>>, Error> {
        let mut objects = Vec::with_capacity(mappings.objects().len());

        // TODO: sort mappings in piece index order, and keep pieces until they're no longer needed
        for &mapping in mappings.objects() {
            let GlobalObject {
                hash,
                piece_index,
                offset,
            } = mapping;

            // Validate parameters
            if !piece_index.is_source() {
                debug!(
                    ?mapping,
                    "Invalid piece index for object: must be a source piece",
                );

                // Parity pieces contain effectively random data, and can't be used to fetch objects
                return Err(Error::NotSourcePiece { mapping });
            }

            if offset >= RawRecord::SIZE as u32 {
                debug!(
                    ?mapping,
                    RawRecord_SIZE = RawRecord::SIZE,
                    "Invalid piece offset for object: must be less than the size of a raw record",
                );

                return Err(Error::PieceOffsetTooLarge { mapping });
            }

            // Try fast object assembling from individual pieces,
            // then regular object assembling from segments
            let data = match self.fetch_object_fast(mapping).await? {
                Some(data) => data,
                None => {
                    let data = self.fetch_object_regular(mapping).await?;

                    debug!(
                        ?mapping,
                        len = %data.len(),
                        "Fetched object using regular object assembling",

                    );

                    data
                }
            };

            let data_hash = blake3_hash(&data);
            if data_hash != hash {
                debug!(
                    ?data_hash,
                    data_length = %data.len(),
                    ?mapping,
                    "Retrieved data doesn't match requested mapping hash"
                );
                trace!(data = %hex::encode(&data), "Retrieved data");

                return Err(Error::InvalidDataHash {
                    data_hash,
                    data_length: data.len(),
                    mapping,
                });
            }

            objects.push(data);
        }

        Ok(objects)
    }

    /// Fast object fetching and assembling where the object doesn't cross piece (super fast) or
    /// segment (just fast) boundaries, returns `Ok(None)` if fast retrieval is not guaranteed.
    // TODO: return already downloaded pieces from fetch_object_fast() and pass them to fetch_object_regular()
    async fn fetch_object_fast(&self, mapping: GlobalObject) -> Result<Option<Vec<u8>>, Error> {
        let GlobalObject {
            piece_index,
            offset,
            ..
        } = mapping;

        // If the offset is before the last few bytes of a segment, we might be able to do very
        // fast object retrieval without assembling and processing the whole segment.
        //
        // The last few bytes might contain padding if a piece is the last piece in the segment.
        let before_max_padding = offset as usize <= RawRecord::SIZE - 1 - MAX_SEGMENT_PADDING;
        let piece_position_in_segment = piece_index.source_position();
        let data_shards = RecordedHistorySegment::NUM_RAW_RECORDS as u32;
        let last_data_piece_in_segment = piece_position_in_segment >= data_shards - 1;

        if last_data_piece_in_segment && !before_max_padding {
            trace!(
                piece_position_in_segment,
                ?mapping,
                "Fast object retrieval not possible: last source piece in segment, \
                and start of object length bytes is in potential segment padding",
            );

            // Fast retrieval possibility is not guaranteed
            return Ok(None);
        }

        // How much bytes are definitely available starting at `piece_index` and `offset` without
        // crossing a segment boundary.
        //
        // The last few bytes might contain padding if a piece is the last piece in the segment.
        let bytes_available_in_segment =
            (data_shards - piece_position_in_segment) * RawRecord::SIZE as u32 - offset;
        let Some(bytes_available_in_segment) =
            bytes_available_in_segment.checked_sub(MAX_SEGMENT_PADDING as u32)
        else {
            // We need to reconstruct the full segment and discard padding before reading the length.
            return Ok(None);
        };

        // Data from pieces that were already read, starting with piece at index `piece_index`
        let mut read_records_data = Vec::<u8>::with_capacity(RawRecord::SIZE * 2);
        let mut next_source_piece_index = piece_index;

        let piece = self.read_piece(next_source_piece_index, mapping).await?;
        next_source_piece_index = next_source_piece_index.next_source_index();
        // Discard piece data before the offset
        read_records_data.extend(
            piece
                .record()
                .to_raw_record_chunks()
                .flatten()
                .skip(offset as usize)
                .copied(),
        );

        if last_data_piece_in_segment {
            // The last few bytes might contain segment padding, so we can't use them for object
            // length or object data.
            read_records_data.truncate(read_records_data.len() - MAX_SEGMENT_PADDING);
        }

        let data_lengths = decode_data_length(&read_records_data, self.max_object_len, mapping)?;

        let data_length = if let Some((length_prefix_len, data_length)) = data_lengths {
            length_prefix_len + data_length
        } else if !last_data_piece_in_segment {
            // Need the next piece to read the length of data, but we can only use it if there was
            // no segment padding
            trace!(
                %next_source_piece_index,
                piece_position_in_segment,
                bytes_available_in_segment,
                ?mapping,
                "Part of object length bytes is in next piece, fetching",
            );

            let piece = self.read_piece(next_source_piece_index, mapping).await?;
            next_source_piece_index = next_source_piece_index.next_source_index();
            read_records_data.extend(piece.record().to_raw_record_chunks().flatten().copied());

            let (length_prefix_len, data_length) =
                decode_data_length(&read_records_data, self.max_object_len, mapping)?
                    .expect("Extra RawRecord is larger than the length encoding; qed");

            length_prefix_len + data_length
        } else {
            trace!(
                piece_position_in_segment,
                bytes_available_in_segment,
                ?mapping,
                "Fast object retrieval not possible: last source piece in segment, \
                and part of object length bytes is in potential segment padding",
            );

            // Super fast read is not possible, because we removed potential segment padding, so
            // the piece bytes are not guaranteed to be continuous
            return Ok(None);
        };

        if data_length > bytes_available_in_segment as usize {
            trace!(
                data_length,
                bytes_available_in_segment,
                piece_position_in_segment,
                ?mapping,
                "Fast object retrieval not possible: part of object data bytes is in \
                potential segment padding",
            );

            // Not enough data without crossing segment boundary
            return Ok(None);
        }

        // Read more pieces until we have enough data
        if data_length as usize > read_records_data.len() {
            let remaining_piece_count =
                (data_length as usize - read_records_data.len()).div_ceil(RawRecord::SIZE);
            let remaining_piece_indexes = (next_source_piece_index..)
                .filter(|i| i.is_source())
                .take(remaining_piece_count)
                .collect();
            self.read_pieces(remaining_piece_indexes, mapping)
                .await?
                .into_iter()
                .for_each(|piece| {
                    read_records_data
                        .extend(piece.record().to_raw_record_chunks().flatten().copied())
                });
        }

        // Decode the data, and return it if it's valid
        let read_records_data = Vec::<u8>::decode(&mut read_records_data.as_slice())
            .map_err(|source| Error::ObjectDecoding { source, mapping })?;

        debug!(
            ?mapping,
            len = %read_records_data.len(),
            "Fetched object using fast object assembling",
        );

        Ok(Some(read_records_data))
    }

    /// Fetch and assemble an object that can cross segment boundaries, which requires assembling
    /// and iterating over full segments.
    async fn fetch_object_regular(&self, mapping: GlobalObject) -> Result<Vec<u8>, Error> {
        let GlobalObject {
            piece_index,
            offset,
            ..
        } = mapping;

        let mut segment_index = piece_index.segment_index();
        let piece_position_in_segment = piece_index.source_position();
        // Used to access the data after it is converted to raw bytes
        let offset_in_segment =
            piece_position_in_segment as usize * RawRecord::SIZE + offset as usize;

        trace!(
            %segment_index,
            offset_in_segment,
            piece_position_in_segment,
            ?mapping,
            "Fetching object from segment(s)",
        );

        let mut data = {
            let items = self
                .read_segment(segment_index, mapping)
                .await?
                .into_items();
            // Go through the segment until we reach the offset.
            // Unconditional progress is enum variant, always 1 byte in SCALE encoding.
            // (Segments do not have an item count, to make incremental writing easier.)
            let mut progress = 1;
            let segment_item = items
                .into_iter()
                .find(|item| {
                    // Add number of bytes in encoded version of segment item
                    progress += item.encoded_size();

                    // Our data is within another segment item, which will have wrapping data
                    // structure, hence strictly `>` here
                    progress > offset_in_segment
                })
                .ok_or_else(|| {
                    debug!(
                        progress,
                        offset_in_segment,
                        ?segment_index,
                        ?mapping,
                        "Failed to find item at offset in segment"
                    );

                    Error::NoSegmentItem {
                        progress,
                        offset_in_segment,
                        segment_index,
                        mapping,
                    }
                })?;

            trace!(
                progress,
                %segment_index,
                offset_in_segment,
                piece_position_in_segment,
                ?mapping,
                segment_item = format!("{segment_item:?}").chars().take(50).collect::<String>(),
                "Found item at offset in first segment",
            );

            // Look at the item after the offset, collecting block bytes
            match segment_item {
                SegmentItem::Block { bytes, .. }
                | SegmentItem::BlockStart { bytes, .. }
                | SegmentItem::BlockContinuation { bytes, .. } => {
                    // Rewind back progress to the beginning of this item
                    progress -= bytes.len();
                    // Get a chunk of the bytes starting at the offset in this item
                    Vec::from(&bytes[offset_in_segment - progress..])
                }
                segment_item @ SegmentItem::Padding
                | segment_item @ SegmentItem::ParentSegmentHeader(_) => {
                    // TODO: create a Display impl for SegmentItem that is shorter than the entire
                    // data contained in it
                    debug!(
                        segment_progress = progress,
                        offset_in_segment,
                        %segment_index,
                        ?mapping,
                        segment_item = format!("{segment_item:?}").chars().take(50).collect::<String>(),
                        "Unexpected segment item in first segment",
                    );

                    return Err(Error::UnexpectedFirstSegmentItem {
                        segment_progress: progress,
                        offset_in_segment,
                        segment_index,
                        mapping,
                        segment_item: Box::new(segment_item),
                    });
                }
            }
        };

        trace!(
            %segment_index,
            offset_in_segment,
            piece_position_in_segment,
            ?mapping,
            data_len = data.len(),
            "Got data at offset in first segment",
        );

        // Return an error if the length is unreasonably large, before we get the next segment
        if let Some((length_prefix_len, data_length)) =
            decode_data_length(data.as_slice(), self.max_object_len, mapping)?
        {
            let data_length = length_prefix_len + data_length;

            // If we have the whole object, decode and return it.
            // TODO: use tokio Bytes type to re-use the same allocation by stripping the length at the start
            if data.len() >= data_length {
                return Vec::<u8>::decode(&mut data.as_slice())
                    .map_err(|source| Error::ObjectDecoding { source, mapping });
            }
        }

        // We need to read extra segments to get the object length, or the full object.
        // We don't attempt to calculate the number of segments needed, because it involves
        // headers and optional padding.
        loop {
            segment_index += SegmentIndex::ONE;
            let items = self
                .read_segment(segment_index, mapping)
                .await?
                .into_items();
            for segment_item in items {
                match segment_item {
                    SegmentItem::BlockContinuation { bytes, .. } => {
                        data.extend_from_slice(&bytes);

                        if let Some((length_prefix_len, data_length)) =
                            decode_data_length(data.as_slice(), self.max_object_len, mapping)?
                        {
                            let data_length = length_prefix_len + data_length;

                            if data.len() >= data_length {
                                return Vec::<u8>::decode(&mut data.as_slice())
                                    .map_err(|source| Error::ObjectDecoding { source, mapping });
                            }
                        }
                    }

                    // Padding at the end of segments, and segment headers can be skipped,
                    // they're not part of the object data
                    SegmentItem::Padding | SegmentItem::ParentSegmentHeader(_) => {}

                    // We should not see these items while collecting data for a single object
                    SegmentItem::Block { .. } | SegmentItem::BlockStart { .. } => {
                        debug!(
                            collected_data = ?data.len(),
                            %segment_index,
                            ?mapping,
                            segment_item = format!("{segment_item:?}").chars().take(50).collect::<String>(),
                            "Unexpected segment item in continuing segment",
                        );

                        return Err(Error::UnexpectedContinuingSegmentItem {
                            collected_data: data.len(),
                            segment_index,
                            mapping,
                            segment_item: Box::new(segment_item),
                        });
                    }
                }
            }
        }
    }

    /// Read the whole segment by its index (just records, skipping witnesses).
    ///
    /// The mapping is only used for error reporting.
    async fn read_segment(
        &self,
        segment_index: SegmentIndex,
        mapping: GlobalObject,
    ) -> Result<Segment, Error> {
        download_segment(
            segment_index,
            &self.piece_getter,
            self.erasure_coding.clone(),
        )
        .await
        .map_err(|source| Error::SegmentGetter { source, mapping })
    }

    /// Concurrently read multiple pieces, and return them in the supplied order.
    ///
    /// The mapping is only used for error reporting.
    async fn read_pieces(
        &self,
        piece_indexes: Arc<[PieceIndex]>,
        mapping: GlobalObject,
    ) -> Result<Vec<Piece>, Error> {
        download_pieces(piece_indexes.clone(), &self.piece_getter)
            .await
            .map_err(|source| {
                debug!(
                    ?piece_indexes,
                    error = ?source,
                    ?mapping,
                    "Error fetching pieces during object assembling"
                );

                Error::PieceGetterError { source, mapping }
            })
    }

    /// Read and return a single piece.
    ///
    /// The mapping is only used for error reporting.
    async fn read_piece(
        &self,
        piece_index: PieceIndex,
        mapping: GlobalObject,
    ) -> Result<Piece, Error> {
        download_pieces(vec![piece_index].into(), &self.piece_getter)
            .await
            .map(|pieces| {
                pieces
                    .first()
                    .expect("download_pieces always returns exact pieces or error")
                    .clone()
            })
            .map_err(|source| {
                debug!(
                    %piece_index,
                    error = ?source,
                    ?mapping,
                    "Error fetching piece during object assembling"
                );

                Error::PieceGetterError { source, mapping }
            })
    }
}

/// Validate and decode the encoded length of `data`, including the encoded length bytes.
/// `data` may be incomplete.
///
/// Returns `Ok(Some((data_length_encoded_length, data_length)))` if the length is valid,
/// `Ok(None)` if there aren't enough bytes to decode the length, otherwise an error.
///
/// The mapping is only used for error reporting.
fn decode_data_length(
    mut data: &[u8],
    max_object_len: usize,
    mapping: GlobalObject,
) -> Result<Option<(usize, usize)>, Error> {
    let data_length = match Compact::<u64>::decode(&mut data) {
        Ok(Compact(data_length)) => {
            let data_length = data_length as usize;
            if data_length > max_object_len {
                debug!(
                    data_length,
                    max_object_len,
                    ?mapping,
                    "Data length exceeds object size limit for object fetcher"
                );

                return Err(Error::ObjectTooLarge {
                    data_length,
                    max_object_len,
                    mapping,
                });
            }

            data_length
        }
        Err(err) => {
            // Parity doesn't have an easily matched error enum, and all bit sequences are
            // valid compact encodings. So we assume we don't have enough bytes to decode the
            // length, unless we already have enough bytes to decode the maximum length.
            if data.len() >= Compact::<u64>::compact_len(&(max_object_len as u64)) {
                debug!(
                    length_prefix_len = data.len(),
                    max_object_len,
                    ?mapping,
                    "Length prefix exceeds object size limit for object fetcher"
                );

                return Err(Error::LengthPrefixTooLarge {
                    length_prefix_len: data.len(),
                    max_object_len,
                    mapping,
                });
            }

            debug!(
                ?err,
                ?mapping,
                "Not enough bytes to decode data length for object"
            );

            return Ok(None);
        }
    };

    let data_length_encoded_length = Compact::<u64>::compact_len(&(data_length as u64));

    trace!(
        data_length,
        data_length_encoded_length,
        ?mapping,
        "Decoded data length for object"
    );

    Ok(Some((data_length_encoded_length, data_length)))
}
