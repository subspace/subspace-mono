use event_listener_primitives::{Bag, HandlerId};
use log::error;
use rocksdb::DB;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Weak};
use subspace_core_primitives::{
    FlatPieces, Piece, PieceIndex, PieceIndexHash, PublicKey, RootBlock, PIECE_SIZE,
};
use subspace_solving::PieceDistance;

use super::{Error, PieceOffset, Result};

const LAST_ROOT_BLOCK_KEY: &[u8] = b"last_root_block";

#[derive(Debug, Default)]
pub struct WriteResult {
    pieces: Arc<FlatPieces>,
    piece_offsets: Vec<Option<PieceOffset>>,
    evicted_pieces: Vec<Piece>,
}

impl WriteResult {
    /// Iterator over tuple of piece offset and piece itself as memory slice
    pub fn to_recommitment_iterator(&self) -> impl Iterator<Item = (PieceOffset, &[u8])> {
        self.piece_offsets
            .iter()
            .zip(self.pieces.as_pieces())
            .filter_map(|(maybe_piece_offset, piece)| {
                maybe_piece_offset.map(|piece_offset| (piece_offset, piece))
            })
    }

    pub fn evicted_pieces(&self) -> &[Piece] {
        &self.evicted_pieces
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PlottedPieces {
    pub plotted_piece_count: usize,
}

#[derive(Default, Debug)]
struct Handlers {
    progress_change: Bag<Arc<dyn Fn(&PlottedPieces) + Send + Sync + 'static>, PlottedPieces>,
}

#[derive(Debug)]
enum Request {
    ReadEncoding {
        index_hash: PieceIndexHash,
        result_sender: mpsc::Sender<io::Result<Piece>>,
    },
    ReadEncodingWithIndex {
        piece_offset: PieceOffset,
        result_sender: mpsc::Sender<io::Result<(Piece, PieceIndex)>>,
    },
    ReadEncodings {
        /// Can be from 0 to the `piece_count`
        piece_offset: PieceOffset,
        count: u64,
        /// Vector containing all of the pieces as contiguous block of memory
        result_sender: mpsc::Sender<io::Result<Vec<u8>>>,
    },
    WriteEncodings {
        encodings: Arc<FlatPieces>,
        piece_indexes: Vec<PieceIndex>,
        /// Returns offsets of all new pieces and pieces which were replaced
        result_sender: mpsc::Sender<io::Result<WriteResult>>,
    },
    Exit {
        result_sender: mpsc::Sender<()>,
    },
}

#[derive(Debug)]
enum RequestPriority {
    Low,
    High,
}

#[derive(Debug)]
struct RequestWithPriority {
    request: Request,
    priority: RequestPriority,
}

struct Inner {
    handlers: Handlers,
    requests_sender: mpsc::SyncSender<RequestWithPriority>,
    plot_metadata_db: Arc<DB>,
    piece_count: Arc<AtomicU64>,
    address: PublicKey,
    base_directory: PathBuf,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let (result_sender, result_receiver) = mpsc::channel();

        if self
            .requests_sender
            .send(RequestWithPriority {
                request: Request::Exit { result_sender },
                priority: RequestPriority::Low,
            })
            .is_ok()
        {
            // We don't care why this returns
            let _ = result_receiver.recv();
        }
    }
}

/// `SinglePlot` struct is an abstraction on top of both single repica plot and tags database.
///
/// It converts requests to internal reads/writes to the plot and tags database. It
/// prioritizes reads over writes by having separate queues for reads and writes requests, read
/// requests are executed until exhausted after which at most 1 write request is handled and the
/// cycle repeats. This allows finding solution with as little delay as possible while introducing
/// changes to the plot at the same time (re-plotting on salt changes or extending plot size).
#[derive(Clone)]
pub struct SinglePlot {
    inner: Arc<Inner>,
}

impl SinglePlot {
    /// Creates a new plot for persisting encoded pieces to disk
    pub fn open_or_create<B: AsRef<Path>>(
        base_directory: B,
        address: PublicKey,
        max_piece_count: u64,
    ) -> Result<Self> {
        let plot_worker = SinglePlotWorker::from_base_directory(
            base_directory.as_ref(),
            address,
            max_piece_count,
        )?;

        let plot_metadata_db = Arc::new(
            DB::open_default(base_directory.as_ref().join("plot-metadata"))
                .map_err(Error::MetadataDbOpen)?,
        );

        let (requests_sender, requests_receiver) = mpsc::sync_channel(100);

        let piece_count = Arc::clone(&plot_worker.piece_count);
        tokio::task::spawn_blocking(move || plot_worker.run(requests_receiver));

        let inner = Inner {
            handlers: Handlers::default(),
            requests_sender,
            plot_metadata_db,
            piece_count,
            address,
            base_directory: base_directory.as_ref().into(),
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Returns address for which pieces are plotted
    pub(crate) fn address(&self) -> PublicKey {
        self.inner.address
    }

    /// Returns base directory for plot
    pub(crate) fn base_directory(&'_ self) -> impl AsRef<Path> + '_ {
        &self.inner.base_directory
    }

    /// How many pieces are there in the plot
    pub(crate) fn piece_count(&self) -> PieceOffset {
        self.inner.piece_count.load(Ordering::Acquire)
    }

    /// Whether plot doesn't have anything in it
    pub(crate) fn is_empty(&self) -> bool {
        self.piece_count() == 0
    }

    /// Reads a piece from plot by index
    pub(crate) fn read(&self, index_hash: impl Into<PieceIndexHash>) -> io::Result<Piece> {
        let (result_sender, result_receiver) = mpsc::channel();
        let index_hash = index_hash.into();

        self.inner
            .requests_sender
            .send(RequestWithPriority {
                request: Request::ReadEncoding {
                    index_hash,
                    result_sender,
                },
                priority: RequestPriority::High,
            })
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed sending read encoding request: {}", error),
                )
            })?;

        result_receiver.recv().map_err(|error| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Read encoding result sender was dropped: {}", error),
            )
        })?
    }

    /// Writes a piece/s to the plot by index, will overwrite if piece exists (updates)
    pub fn write_many(
        &self,
        encodings: Arc<FlatPieces>,
        piece_indexes: Vec<PieceIndex>,
    ) -> io::Result<WriteResult> {
        if encodings.is_empty() {
            return Ok(Default::default());
        }
        self.inner
            .handlers
            .progress_change
            .call_simple(&PlottedPieces {
                plotted_piece_count: encodings.len(),
            });

        let (result_sender, result_receiver) = mpsc::channel();

        self.inner
            .requests_sender
            .send(RequestWithPriority {
                request: Request::WriteEncodings {
                    encodings,
                    piece_indexes,
                    result_sender,
                },
                priority: RequestPriority::Low,
            })
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed sending write many request: {}", error),
                )
            })?;

        result_receiver.recv().map_err(|error| {
            io::Error::other(format!("Write many result sender was dropped: {error}"))
        })?
    }

    /// Get last root block
    pub(crate) fn get_last_root_block(&self) -> Result<Option<RootBlock>, rocksdb::Error> {
        self.inner
            .plot_metadata_db
            .get(LAST_ROOT_BLOCK_KEY)
            .map(|maybe_last_root_block| {
                maybe_last_root_block.as_ref().map(|last_root_block| {
                    serde_json::from_slice(last_root_block)
                        .expect("Database contains incorrect last root block")
                })
            })
    }

    /// Store last root block
    pub(crate) fn set_last_root_block(
        &self,
        last_root_block: &RootBlock,
    ) -> Result<(), rocksdb::Error> {
        let last_root_block = serde_json::to_vec(&last_root_block).unwrap();
        self.inner
            .plot_metadata_db
            .put(LAST_ROOT_BLOCK_KEY, last_root_block)
    }

    pub(crate) fn downgrade(&self) -> WeakSinglePlot {
        WeakSinglePlot {
            inner: Arc::downgrade(&self.inner),
        }
    }

    pub fn read_piece(&self, index_hash: impl Into<PieceIndexHash>) -> io::Result<Vec<u8>> {
        self.read(index_hash)
            .map(|piece| <[u8; PIECE_SIZE]>::from(piece).to_vec())
    }

    pub(crate) fn read_piece_with_index(
        &self,
        piece_offset: PieceOffset,
    ) -> io::Result<(Piece, PieceIndex)> {
        let (result_sender, result_receiver) = mpsc::channel();

        self.inner
            .requests_sender
            .send(RequestWithPriority {
                request: Request::ReadEncodingWithIndex {
                    piece_offset,
                    result_sender,
                },
                priority: RequestPriority::High,
            })
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed sending read encodings request: {}", error),
                )
            })?;

        result_receiver.recv().map_err(|error| {
            io::Error::other(format!(
                "Read encodings result sender was dropped: {}",
                error
            ))
        })?
    }

    /// Returns pieces packed one after another in contiguous `Vec<u8>`
    pub(crate) fn read_pieces(&self, piece_offset: PieceOffset, count: u64) -> io::Result<Vec<u8>> {
        let (result_sender, result_receiver) = mpsc::channel();

        self.inner
            .requests_sender
            .send(RequestWithPriority {
                request: Request::ReadEncodings {
                    piece_offset,
                    count,
                    result_sender,
                },
                priority: RequestPriority::High,
            })
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed sending read encodings request: {}", error),
                )
            })?;

        result_receiver.recv().map_err(|error| {
            io::Error::other(format!(
                "Read encodings result sender was dropped: {}",
                error
            ))
        })?
    }

    pub fn on_progress_change(
        &self,
        callback: Arc<dyn Fn(&PlottedPieces) + Send + Sync + 'static>,
    ) -> HandlerId {
        self.inner.handlers.progress_change.add(callback)
    }
}

#[derive(Clone)]
pub(crate) struct WeakSinglePlot {
    inner: Weak<Inner>,
}

impl WeakSinglePlot {
    pub(crate) fn upgrade(&self) -> Option<SinglePlot> {
        self.inner.upgrade().map(|inner| SinglePlot { inner })
    }
}

#[derive(Debug)]
struct IndexHashToOffsetDB {
    inner: DB,
    address: PublicKey,
    max_distance: Option<PieceDistance>,
}

impl IndexHashToOffsetDB {
    fn open_default(path: impl AsRef<Path>, address: PublicKey) -> Result<Self> {
        let inner = DB::open_default(path.as_ref()).map_err(Error::IndexDbOpen)?;
        let max_distance = {
            let mut iter = inner.raw_iterator();
            iter.seek_to_last();
            iter.key().map(PieceDistance::from_big_endian)
        };
        Ok(Self {
            inner,
            address,
            max_distance,
        })
    }

    fn get(&self, index_hash: &PieceIndexHash) -> io::Result<Option<PieceOffset>> {
        let distance = PieceDistance::xor_distance(index_hash, &self.address);
        self.inner
            .get(&distance.to_bytes())
            .map_err(io::Error::other)
            .and_then(|opt_val| {
                opt_val
                    .map(|val| <[u8; 8]>::try_from(val).map(PieceOffset::from_le_bytes))
                    .transpose()
                    .map_err(|_| io::Error::other("Offsets in rocksdb supposed to be 8 bytes long"))
            })
    }

    /// Returns `true` if piece plot will not result in exceeding plot size and doesn't exist
    /// already
    fn should_store(&self, index_hash: &PieceIndexHash) -> io::Result<bool> {
        Ok(match self.max_distance {
            Some(max_distance) => {
                PieceDistance::xor_distance(index_hash, &self.address) < max_distance
                    && self.get(index_hash)?.is_none()
            }
            None => false,
        })
    }

    fn remove_furthest(&mut self) -> io::Result<Option<PieceOffset>> {
        let max_distance = match self.max_distance {
            Some(max_distance) => max_distance,
            None => return Ok(None),
        };
        let result = self
            .inner
            .get(&max_distance.to_bytes())
            .map_err(io::Error::other)?
            .map(|buffer| *<&[u8; 8]>::try_from(&*buffer).unwrap())
            .map(PieceOffset::from_le_bytes);
        self.inner
            .delete(&max_distance.to_bytes())
            .map_err(io::Error::other)?;

        let mut iter = self.inner.raw_iterator();
        iter.seek_to_last();
        self.max_distance = iter.key().map(PieceDistance::from_big_endian);

        Ok(result)
    }

    fn put(&mut self, index_hash: &PieceIndexHash, offset: PieceOffset) -> io::Result<()> {
        let distance = PieceDistance::xor_distance(index_hash, &self.address);
        self.inner
            .put(&distance.to_bytes(), offset.to_le_bytes())
            .map_err(io::Error::other)?;

        match self.max_distance {
            Some(old_distance) => {
                if old_distance < distance {
                    self.max_distance.replace(distance);
                }
            }
            None => {
                self.max_distance.replace(distance);
            }
        };

        Ok(())
    }
}

struct SinglePlotWorker {
    plot: File,
    piece_index_hash_to_offset_db: IndexHashToOffsetDB,
    piece_offset_to_index: File,
    piece_count: Arc<AtomicU64>,
    max_piece_count: u64,
}

impl SinglePlotWorker {
    fn from_base_directory(
        base_directory: impl AsRef<Path>,
        address: PublicKey,
        max_piece_count: u64,
    ) -> Result<Self> {
        let plot = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(base_directory.as_ref().join("plot.bin"))
            .map_err(Error::PlotOpen)?;

        let plot_size = plot
            .metadata()
            .map(|metadata| metadata.len())
            .map_err(Error::PlotOpen)?;

        let piece_count = Arc::new(AtomicU64::new(plot_size / PIECE_SIZE as u64));

        let piece_offset_to_index = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(base_directory.as_ref().join("plot-offset-to-index.bin"))
            .map_err(Error::OffsetDbOpen)?;

        // TODO: handle `piece_count.load() > max_piece_count`, we should discard some of the pieces
        //  here

        let piece_index_hash_to_offset_db = IndexHashToOffsetDB::open_default(
            base_directory.as_ref().join("plot-index-to-offset"),
            address,
        )?;

        Ok(Self {
            plot,
            piece_index_hash_to_offset_db,
            piece_offset_to_index,
            piece_count,
            max_piece_count,
        })
    }

    fn read_encoding(&mut self, piece_index_hash: PieceIndexHash) -> io::Result<Piece> {
        let mut buffer = Piece::default();
        let offset = self
            .piece_index_hash_to_offset_db
            .get(&piece_index_hash)?
            .ok_or_else(|| {
                io::Error::other(format!("Piece with hash {piece_index_hash:?} not found"))
            })?;
        self.plot
            .seek(SeekFrom::Start(offset * PIECE_SIZE as u64))?;
        self.plot.read_exact(buffer.as_mut()).map(|()| buffer)
    }

    fn get_piece_index(&mut self, offset: PieceOffset) -> io::Result<PieceIndex> {
        let mut buf = [0; 8];
        self.piece_offset_to_index.seek(SeekFrom::Start(
            offset * std::mem::size_of::<PieceIndex>() as u64,
        ))?;
        self.piece_offset_to_index.read_exact(&mut buf)?;
        Ok(PieceIndex::from_le_bytes(buf))
    }

    fn put_piece_index(&mut self, offset: PieceOffset, piece_index: PieceIndex) -> io::Result<()> {
        self.piece_offset_to_index.seek(SeekFrom::Start(
            offset * std::mem::size_of::<PieceIndex>() as u64,
        ))?;
        self.piece_offset_to_index
            .write_all(&piece_index.to_le_bytes())
    }

    // TODO: Add error recovery
    fn write_encodings(
        &mut self,
        pieces: Arc<FlatPieces>,
        piece_indexes: Vec<PieceIndex>,
    ) -> io::Result<WriteResult> {
        let current_piece_count = self.piece_count.load(Ordering::SeqCst);
        let pieces_left_until_full_plot =
            (self.max_piece_count - current_piece_count).min(pieces.count() as u64);

        // Split pieces and indexes in those that can be appended to the end of plot (thus written
        // sequentially) and those that need to be checked individually and plotted one by one in
        // place of old pieces
        let (sequential_pieces, _) =
            pieces.split_at(pieces_left_until_full_plot as usize * PIECE_SIZE);
        // Iterator is more convenient for random pieces, otherwise we could take it from above
        let random_pieces = pieces
            .as_pieces()
            .skip(pieces_left_until_full_plot as usize);
        let (sequential_piece_indexes, random_piece_indexes) =
            piece_indexes.split_at(pieces_left_until_full_plot as usize);

        // Process sequential pieces
        {
            self.plot
                .seek(SeekFrom::Start(current_piece_count * PIECE_SIZE as u64))?;
            self.plot.write_all(sequential_pieces)?;

            for (piece_offset, &piece_index) in
                (current_piece_count..).zip(sequential_piece_indexes)
            {
                self.piece_index_hash_to_offset_db
                    .put(&piece_index.into(), piece_offset)?;
                self.put_piece_index(piece_offset, piece_index)?;
            }

            self.piece_count
                .fetch_add(pieces_left_until_full_plot, Ordering::AcqRel);
        }

        let mut piece_offsets = Vec::<Option<PieceOffset>>::with_capacity(pieces.count());
        piece_offsets.extend(
            (current_piece_count..)
                .take(pieces_left_until_full_plot as usize)
                .map(Some),
        );
        piece_offsets.resize(piece_offsets.capacity(), None);
        let mut evicted_pieces =
            Vec::with_capacity(pieces.count() - pieces_left_until_full_plot as usize);

        // Process random pieces
        for ((piece, &piece_index), maybe_piece_offset) in
            random_pieces.zip(random_piece_indexes).zip(
                piece_offsets
                    .iter_mut()
                    .skip(pieces_left_until_full_plot as usize),
            )
        {
            // Check if piece is out of plot range or if it is in the plot
            if !self
                .piece_index_hash_to_offset_db
                .should_store(&piece_index.into())?
            {
                continue;
            }

            let piece_offset = self
                .piece_index_hash_to_offset_db
                .remove_furthest()?
                .expect("Must be always present as plot is non-empty; qed");

            let mut old_piece = Piece::default();
            self.plot
                .seek(SeekFrom::Start(piece_offset * PIECE_SIZE as u64))?;
            self.plot.read_exact(&mut old_piece)?;

            self.plot
                .seek(SeekFrom::Start(piece_offset * PIECE_SIZE as u64))?;
            self.plot.write_all(piece)?;

            self.piece_index_hash_to_offset_db
                .put(&piece_index.into(), piece_offset)?;
            self.put_piece_index(piece_offset, piece_index)?;

            // TODO: This is a bit inefficient when pieces from previous iterations of this loop are
            //  evicted, causing extra tags overrides during recommitment
            maybe_piece_offset.replace(piece_offset);
            evicted_pieces.push(old_piece);
        }

        Ok(WriteResult {
            pieces,
            piece_offsets,
            evicted_pieces,
        })
    }

    fn run(mut self, requests_receiver: mpsc::Receiver<RequestWithPriority>) {
        let mut low_priority_requests = VecDeque::new();
        let mut exit_result_sender = None;

        // Process as many high priority as possible, interleaved with single low priority request
        // in case no high priority requests are available.
        'outer: while let Ok(request_with_priority) = requests_receiver.recv() {
            let RequestWithPriority {
                mut request,
                mut priority,
            } = request_with_priority;

            loop {
                if matches!(priority, RequestPriority::Low) {
                    low_priority_requests.push_back(request);
                } else {
                    match request {
                        Request::ReadEncoding {
                            index_hash,
                            result_sender,
                        } => {
                            let _ = result_sender.send(self.read_encoding(index_hash));
                        }
                        Request::ReadEncodingWithIndex {
                            piece_offset,
                            result_sender,
                        } => {
                            let result = try {
                                let mut buffer = Piece::default();
                                self.plot
                                    .seek(SeekFrom::Start(piece_offset * PIECE_SIZE as u64))?;
                                self.plot.read_exact(buffer.as_mut())?;
                                let index = self.get_piece_index(piece_offset)?;
                                (buffer, index)
                            };
                            let _ = result_sender.send(result);
                        }
                        Request::ReadEncodings {
                            piece_offset,
                            count,
                            result_sender,
                        } => {
                            let result = try {
                                self.plot
                                    .seek(SeekFrom::Start(piece_offset * PIECE_SIZE as u64))?;
                                let mut buffer = Vec::with_capacity(count as usize * PIECE_SIZE);
                                buffer.resize(buffer.capacity(), 0);
                                self.plot.read_exact(&mut buffer)?;
                                buffer
                            };
                            let _ = result_sender.send(result);
                        }
                        Request::WriteEncodings {
                            encodings,
                            piece_indexes,
                            result_sender,
                        } => {
                            let _ =
                                result_sender.send(self.write_encodings(encodings, piece_indexes));
                        }
                        Request::Exit { result_sender } => {
                            exit_result_sender.replace(result_sender);
                            break 'outer;
                        }
                    }
                }

                match requests_receiver.try_recv() {
                    Ok(some_request_with_priority) => {
                        request = some_request_with_priority.request;
                        priority = some_request_with_priority.priority;
                        continue;
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        // If no high priority requests available, process one low priority request.
                        if let Some(low_priority_request) = low_priority_requests.pop_front() {
                            request = low_priority_request;
                            priority = RequestPriority::High;
                            continue;
                        }
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        // Ignore
                    }
                }

                break;
            }
        }

        if let Err(error) = self.plot.sync_all() {
            error!("Failed to sync plot file before exit: {}", error);
        }

        if let Err(error) = self.piece_offset_to_index.sync_all() {
            error!(
                "Failed to sync piece offset to index file before exit: {}",
                error
            );
        }

        // Close the rest of databases
        drop(self);
    }
}
