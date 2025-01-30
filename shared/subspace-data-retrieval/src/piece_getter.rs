//! Getting object pieces from the Subspace Distributed Storage Network, or various caches.

use async_trait::async_trait;
use futures::{stream, Stream, StreamExt};
use std::fmt;
use std::future::Future;
use std::sync::Arc;
use subspace_archiving::archiver::NewArchivedSegment;
use subspace_core_primitives::pieces::{Piece, PieceIndex};

/// Trait representing a way to get pieces
#[async_trait]
pub trait PieceGetter: fmt::Debug {
    /// Get piece by index.
    ///
    /// Returns `Ok(None)` if the piece is not found.
    /// Returns `Err(_)` if trying to get the piece caused an error.
    async fn get_piece(&self, piece_index: PieceIndex) -> anyhow::Result<Option<Piece>>;

    /// Get pieces with provided indices.
    ///
    /// The number of elements in the returned stream is the same as the number of unique
    /// `piece_indices`.
    async fn get_pieces<'a>(
        &'a self,
        piece_indices: Vec<PieceIndex>,
    ) -> anyhow::Result<
        Box<dyn Stream<Item = (PieceIndex, anyhow::Result<Option<Piece>>)> + Send + Unpin + 'a>,
    >;
}

#[async_trait]
impl<T> PieceGetter for Arc<T>
where
    T: PieceGetter + Send + Sync + ?Sized,
{
    #[inline]
    async fn get_piece(&self, piece_index: PieceIndex) -> anyhow::Result<Option<Piece>> {
        self.as_ref().get_piece(piece_index).await
    }

    #[inline]
    async fn get_pieces<'a>(
        &'a self,
        piece_indices: Vec<PieceIndex>,
    ) -> anyhow::Result<
        Box<dyn Stream<Item = (PieceIndex, anyhow::Result<Option<Piece>>)> + Send + Unpin + 'a>,
    > {
        self.as_ref().get_pieces(piece_indices).await
    }
}

// Convenience methods, mainly used in testing
#[async_trait]
impl PieceGetter for NewArchivedSegment {
    async fn get_piece(&self, piece_index: PieceIndex) -> anyhow::Result<Option<Piece>> {
        if piece_index.segment_index() == self.segment_header.segment_index() {
            return Ok(Some(
                self.pieces
                    .pieces()
                    .nth(piece_index.position() as usize)
                    .expect("Piece position always exists in a segment; qed"),
            ));
        }

        Ok(None)
    }

    async fn get_pieces<'a>(
        &'a self,
        piece_indices: Vec<PieceIndex>,
    ) -> anyhow::Result<
        Box<dyn Stream<Item = (PieceIndex, anyhow::Result<Option<Piece>>)> + Send + Unpin + 'a>,
    > {
        get_pieces_individually(|piece_index| self.get_piece(piece_index), piece_indices)
    }
}

#[async_trait]
impl PieceGetter for (PieceIndex, Piece) {
    async fn get_piece(&self, piece_index: PieceIndex) -> anyhow::Result<Option<Piece>> {
        if self.0 == piece_index {
            return Ok(Some(self.1.clone()));
        }

        Ok(None)
    }

    async fn get_pieces<'a>(
        &'a self,
        piece_indices: Vec<PieceIndex>,
    ) -> anyhow::Result<
        Box<dyn Stream<Item = (PieceIndex, anyhow::Result<Option<Piece>>)> + Send + Unpin + 'a>,
    > {
        get_pieces_individually(|piece_index| self.get_piece(piece_index), piece_indices)
    }
}

#[async_trait]
impl PieceGetter for Vec<(PieceIndex, Piece)> {
    async fn get_piece(&self, piece_index: PieceIndex) -> anyhow::Result<Option<Piece>> {
        Ok(self.iter().find_map(|(index, piece)| {
            if *index == piece_index {
                Some(piece.clone())
            } else {
                None
            }
        }))
    }

    async fn get_pieces<'a>(
        &'a self,
        piece_indices: Vec<PieceIndex>,
    ) -> anyhow::Result<
        Box<dyn Stream<Item = (PieceIndex, anyhow::Result<Option<Piece>>)> + Send + Unpin + 'a>,
    > {
        get_pieces_individually(|piece_index| self.get_piece(piece_index), piece_indices)
    }
}

/// A default implementation which gets each piece individually, using the `get_piece` async
/// function.
///
/// This is mainly used for testing, most production implementations can fetch multiple pieces more
/// efficiently.
#[expect(clippy::type_complexity, reason = "type matches trait signature")]
pub fn get_pieces_individually<'a, PieceIndices, Func, Fut>(
    // TODO: replace with AsyncFn(PieceIndex) -> anyhow::Result<Option<Piece>> once it stabilises
    // https://github.com/rust-lang/rust/issues/62290
    get_piece: Func,
    piece_indices: PieceIndices,
) -> anyhow::Result<
    Box<dyn Stream<Item = (PieceIndex, anyhow::Result<Option<Piece>>)> + Send + Unpin + 'a>,
>
where
    PieceIndices: IntoIterator<Item = PieceIndex, IntoIter: Send> + Send + 'a,
    Func: Fn(PieceIndex) -> Fut + Clone + Send + 'a,
    Fut: Future<Output = anyhow::Result<Option<Piece>>> + Send + Unpin + 'a,
{
    Ok(Box::new(Box::pin(stream::iter(piece_indices).then(
        move |piece_index| {
            let get_piece = get_piece.clone();
            async move { (piece_index, get_piece(piece_index).await) }
        },
    ))))
}
