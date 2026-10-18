//! Chunked layout: chunk index lookup, per-chunk decoding and copying.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use super::DatasetStorage;
use super::contiguous::{read_block_runs, to_usize};
use super::selection::{
    DimRuns, HyperslabDim, Resolved, block_runs, copy_group, for_each_group, row_major_strides,
};
use crate::error::{Error, Result};
use crate::filters::{FilterContext, PipelineStep, decode_pipeline};
use crate::format::UNDEFINED_ADDRESS;
use crate::format::btree_v1::{ChunkEntry, chunk_btree_entries, find_chunk};
use crate::format::btree_v2::{BTreeV2, BTreeV2Header, Record, RecordContext};
use crate::format::chunk_index::{
    ChunkIndexContext, ChunkRecord, ImplicitIndex, chunks_per_dim, extensible_linear_index,
    implicit_chunk_address, linear_index, total_chunks, unlinearize,
};
use crate::format::extensible_array::ExtensibleArray;
use crate::format::fixed_array::FixedArray;
use crate::format::messages::dataspace::{DataspaceKind, UNLIMITED};
use crate::format::messages::layout::{
    ChunkIndex, ChunkedLayout, DONT_FILTER_PARTIAL_BOUND_CHUNKS, FilteredChunk,
};

/// Below this many chunks, B-tree and extensible array indexes are queried
/// chunk by chunk instead of being read whole.
const POINT_LOOKUP_MAX: usize = 64;

/// One allocated chunk, as listed by [`DatasetStorage::chunks`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChunkInfo {
    /// Logical element coordinates of the chunk's first element.
    pub offset: Vec<u64>,
    /// Scaled coordinates (`offset / chunk dims`), i.e. the chunk's
    /// position in the chunk grid.
    pub scaled: Vec<u64>,
    /// File address of the stored chunk.
    pub address: u64,
    /// Stored size in bytes (after filtering).
    pub size: u64,
    /// Filter mask: bit `n` set means filter `n` was skipped.
    pub filter_mask: u32,
}

/// Validated chunk geometry of a dataset.
struct Geometry {
    /// Chunk dimensions (elements), one per dataspace dimension.
    chunk_dims: Vec<u64>,
    /// Current dataset dimensions.
    dims: Vec<u64>,
    /// Decoded size of one chunk in bytes.
    chunk_bytes: usize,
    /// Whether the dataset has a (non-empty) filter pipeline.
    filtered: bool,
}

impl Geometry {
    fn new(storage: &DatasetStorage<'_>, c: &ChunkedLayout) -> Result<Self> {
        let space = storage.dataspace;
        if space.kind != DataspaceKind::Simple {
            return Err(Error::invalid("chunked layout requires a simple dataspace"));
        }
        if c.dims.len() != space.rank() {
            return Err(Error::invalid(
                "chunk rank does not match the dataspace rank",
            ));
        }
        if c.dims.contains(&0) {
            return Err(Error::invalid("chunk dimension of size zero"));
        }
        if c.element_size != storage.element_size as u64 {
            return Err(Error::invalid(
                "chunked layout element size does not match the datatype size",
            ));
        }
        let chunk_bytes = c
            .dims
            .iter()
            .try_fold(storage.element_size as u64, |acc, &d| acc.checked_mul(d))
            .and_then(|b| usize::try_from(b).ok())
            .filter(|&b| b <= storage.limits.max_chunk_bytes)
            .ok_or(Error::LimitExceeded("chunk size"))?;
        Ok(Geometry {
            chunk_dims: c.dims.clone(),
            dims: space.dims.clone(),
            chunk_bytes,
            filtered: storage.pipeline.is_some_and(|p| !p.filters.is_empty()),
        })
    }

    /// Element offset of the chunk at `scaled`.
    fn offset_of(&self, scaled: &[u64]) -> Result<Vec<u64>> {
        scaled
            .iter()
            .zip(&self.chunk_dims)
            .map(|(&s, &c)| {
                s.checked_mul(c)
                    .ok_or_else(|| Error::invalid("chunk offset overflows"))
            })
            .collect()
    }

    /// Scaled coordinates of a chunk given its element offset, which must
    /// be chunk-aligned.
    fn scaled_of(&self, offset: &[u64]) -> Result<Vec<u64>> {
        if offset.len() != self.chunk_dims.len() {
            return Err(Error::invalid("chunk coordinate rank mismatch"));
        }
        offset
            .iter()
            .zip(&self.chunk_dims)
            .map(|(&o, &c)| {
                if o % c != 0 {
                    Err(Error::invalid("chunk offset not aligned to the chunk grid"))
                } else {
                    Ok(o / c)
                }
            })
            .collect()
    }

    /// Whether the chunk at `offset` extends beyond the current dimensions.
    fn is_partial(&self, offset: &[u64]) -> bool {
        offset
            .iter()
            .zip(&self.chunk_dims)
            .zip(&self.dims)
            .any(|((&o, &c), &d)| o.saturating_add(c) > d)
    }

    fn index_context(&self, storage: &DatasetStorage<'_>) -> ChunkIndexContext {
        if self.filtered {
            ChunkIndexContext::filtered(storage.sizes)
        } else {
            ChunkIndexContext::unfiltered(storage.sizes)
        }
    }
}

/// Chunk grid from the maximum dimensions (fixed array, implicit index).
fn max_grid(storage: &DatasetStorage<'_>, g: &Geometry) -> Result<Vec<u64>> {
    let space = storage.dataspace;
    let max: Vec<u64> = (0..space.rank())
        .map(|i| space.max_dim(i).unwrap_or(0))
        .collect();
    if max.contains(&UNLIMITED) {
        return Err(Error::invalid(
            "fixed-size chunk index used with an unlimited dimension",
        ));
    }
    if max.iter().zip(&g.dims).any(|(m, d)| m < d) {
        return Err(Error::invalid(
            "dataspace maximum smaller than current size",
        ));
    }
    chunks_per_dim(&max, &g.chunk_dims)
}

/// Chunk grid of an extensible array index and its unlimited dimension.
/// The grid's entry for the unlimited dimension is unused (set to the
/// current number of chunks).
fn ea_grid(storage: &DatasetStorage<'_>, g: &Geometry) -> Result<(Vec<u64>, usize)> {
    let space = storage.dataspace;
    let unlimited: Vec<usize> = (0..space.rank())
        .filter(|&i| space.is_unlimited(i))
        .collect();
    let [unlim] = unlimited.as_slice() else {
        return Err(Error::invalid(
            "extensible array index requires exactly one unlimited dimension",
        ));
    };
    let max: Vec<u64> = (0..space.rank())
        .map(|i| {
            if i == *unlim {
                g.dims[i]
            } else {
                space.max_dim(i).unwrap_or(0)
            }
        })
        .collect();
    Ok((chunks_per_dim(&max, &g.chunk_dims)?, *unlim))
}

/// Inverse of [`extensible_linear_index`].
fn extensible_unlinearize(mut index: u64, grid: &[u64], unlim: usize) -> Result<Vec<u64>> {
    let mut out = vec![0u64; grid.len()];
    for (i, (slot, &n)) in out.iter_mut().zip(grid).enumerate().rev() {
        if i == unlim {
            continue;
        }
        if n == 0 {
            return Err(Error::invalid("chunk index outside chunk grid"));
        }
        *slot = index % n;
        index /= n;
    }
    if let Some(slot) = out.get_mut(unlim) {
        *slot = index;
    }
    Ok(out)
}

fn v1_record(g: &Geometry, e: &ChunkEntry) -> ChunkRecord {
    ChunkRecord {
        address: e.address,
        size: g.filtered.then_some(u64::from(e.size)),
        filter_mask: e.filter_mask,
    }
}

fn v2_record(r: &Record) -> Option<(&[u64], ChunkRecord)> {
    match r {
        Record::Chunk {
            address,
            scaled_offsets,
        } => Some((
            scaled_offsets,
            ChunkRecord {
                address: *address,
                size: None,
                filter_mask: 0,
            },
        )),
        Record::FilteredChunk {
            address,
            chunk_size,
            filter_mask,
            scaled_offsets,
        } => Some((
            scaled_offsets,
            ChunkRecord {
                address: *address,
                size: Some(*chunk_size),
                filter_mask: *filter_mask,
            },
        )),
        _ => None,
    }
}

/// Open the v2 B-tree chunk index at `addr`, deriving the record layout
/// from its header.
fn open_btree_v2(storage: &DatasetStorage<'_>, g: &Geometry, addr: u64) -> Result<BTreeV2> {
    let sizes = storage.sizes;
    let buf = storage
        .store
        .read_vec_at(addr, BTreeV2Header::encoded_size(sizes))?;
    let header = BTreeV2Header::parse(&buf, sizes)?;
    let rank = g.chunk_dims.len();
    let expected_type = if g.filtered { 11 } else { 10 };
    if header.tree_type != expected_type {
        return Err(Error::invalid(
            "v2 B-tree chunk index type does not match the filter pipeline",
        ));
    }
    let chunk_size_len = if g.filtered {
        usize::from(header.record_size)
            .checked_sub(usize::from(sizes.offset) + 4 + 8 * rank)
            .filter(|w| (1..=8).contains(w))
            .ok_or_else(|| Error::invalid("bad v2 B-tree filtered chunk record size"))?
    } else {
        0
    };
    let ctx = RecordContext {
        heap_id_len: None,
        chunk_ndims: rank,
        chunk_size_len,
    };
    BTreeV2::open(storage.store, addr, sizes, ctx)
}

// ------------------------------------------------------- chunk index --

/// How [`FullIndex::get`] finds a record.
enum Lookup {
    /// Every grid position has a record (fixed and extensible arrays).
    Linear {
        records: Vec<ChunkRecord>,
        grid: Vec<u64>,
        /// The unlimited dimension of an extensible array index.
        unlim: Option<usize>,
    },
    /// Sparse: scaled coordinates -> position in `allocated`.
    Map(HashMap<Vec<u64>, usize>),
}

/// A completely loaded chunk index (fixed array, extensible array, v1 or
/// v2 B-tree), kept by the dataset's [`ChunkCache`](super::ChunkCache).
pub(crate) struct FullIndex {
    /// Every allocated chunk: scaled coordinates and record.
    allocated: Vec<(Vec<u64>, ChunkRecord)>,
    lookup: Lookup,
}

impl FullIndex {
    fn get(&self, scaled: &[u64]) -> Result<ChunkRecord> {
        Ok(match &self.lookup {
            Lookup::Linear {
                records,
                grid,
                unlim,
            } => {
                let li = match unlim {
                    Some(u) => extensible_linear_index(scaled, grid, *u)?,
                    None => linear_index(scaled, grid)?,
                };
                usize::try_from(li)
                    .ok()
                    .and_then(|i| records.get(i).copied())
                    .unwrap_or(ChunkRecord::UNALLOCATED)
            }
            Lookup::Map(map) => map
                .get(scaled)
                .and_then(|&i| self.allocated.get(i))
                .map_or(ChunkRecord::UNALLOCATED, |(_, r)| *r),
        })
    }

    fn sparse(allocated: Vec<(Vec<u64>, ChunkRecord)>) -> FullIndex {
        let map = allocated
            .iter()
            .enumerate()
            .map(|(i, (s, _))| (s.clone(), i))
            .collect();
        FullIndex {
            allocated,
            lookup: Lookup::Map(map),
        }
    }
}

/// Read the whole chunk index of `c` (not for single chunk and implicit
/// indexes, which need none).
fn load_full_index(
    storage: &DatasetStorage<'_>,
    c: &ChunkedLayout,
    g: &Geometry,
) -> Result<FullIndex> {
    let store = storage.store;
    match c.index {
        ChunkIndex::FixedArray { address, .. } => {
            let grid = max_grid(storage, g)?;
            let fa = FixedArray::open(store, address, &g.index_context(storage))?;
            if fa.header.max_num_entries != total_chunks(&grid)? {
                return Err(Error::invalid(
                    "fixed array size does not match the chunk grid",
                ));
            }
            let records = fa.records(store)?;
            let mut allocated = Vec::new();
            for (i, r) in records.iter().enumerate() {
                if r.is_allocated() {
                    allocated.push((unlinearize(i as u64, &grid)?, *r));
                }
            }
            Ok(FullIndex {
                allocated,
                lookup: Lookup::Linear {
                    records,
                    grid,
                    unlim: None,
                },
            })
        }
        ChunkIndex::ExtensibleArray { address, .. } => {
            let (grid, unlim) = ea_grid(storage, g)?;
            let ea = ExtensibleArray::open(store, address, &g.index_context(storage))?;
            let records = ea.records(store, ea.header.max_index_set)?;
            let mut allocated = Vec::new();
            for (i, r) in records.iter().enumerate() {
                if r.is_allocated() {
                    allocated.push((extensible_unlinearize(i as u64, &grid, unlim)?, *r));
                }
            }
            Ok(FullIndex {
                allocated,
                lookup: Lookup::Linear {
                    records,
                    grid,
                    unlim: Some(unlim),
                },
            })
        }
        ChunkIndex::BTreeV1 { address } => {
            let entries = chunk_btree_entries(store, address, storage.sizes, g.chunk_dims.len())?;
            let allocated = entries
                .iter()
                .map(|e| Ok((g.scaled_of(&e.offsets)?, v1_record(g, e))))
                .collect::<Result<Vec<_>>>()?;
            Ok(FullIndex::sparse(allocated))
        }
        ChunkIndex::BTreeV2 { address, .. } => {
            let tree = open_btree_v2(storage, g, address)?;
            let allocated = tree
                .records(store)?
                .iter()
                .filter_map(|r| v2_record(r).map(|(k, rec)| (k.to_vec(), rec)))
                .collect();
            Ok(FullIndex::sparse(allocated))
        }
        ChunkIndex::SingleChunk { .. } | ChunkIndex::Implicit { .. } => Err(Error::invalid(
            "single chunk and implicit indexes have no index structure",
        )),
    }
}

/// The whole chunk index, from the cache when possible.
fn full_index(
    storage: &DatasetStorage<'_>,
    c: &ChunkedLayout,
    g: &Geometry,
) -> Result<Arc<FullIndex>> {
    if let Some(f) = storage.cache.and_then(|k| k.index().full.clone()) {
        return Ok(f);
    }
    let full = Arc::new(load_full_index(storage, c, g)?);
    if let Some(k) = storage.cache {
        let mut st = k.index();
        st.full = Some(full.clone());
        st.points = HashMap::new();
    }
    Ok(full)
}

/// Look up a few chunks individually (extensible array and B-tree
/// indexes), using and filling the cache's memo. `None` when the index
/// should be loaded whole instead.
fn point_records(
    storage: &DatasetStorage<'_>,
    c: &ChunkedLayout,
    g: &Geometry,
    needed: &[Vec<u64>],
) -> Result<Option<Vec<ChunkRecord>>> {
    if !matches!(
        c.index,
        ChunkIndex::ExtensibleArray { .. }
            | ChunkIndex::BTreeV1 { .. }
            | ChunkIndex::BTreeV2 { .. }
    ) {
        return Ok(None);
    }
    let mut found: Vec<Option<ChunkRecord>> = vec![None; needed.len()];
    if let Some(k) = storage.cache {
        let st = k.index();
        for (slot, s) in found.iter_mut().zip(needed) {
            *slot = st.points.get(s).copied();
        }
        let missing = found.iter().filter(|f| f.is_none()).count();
        if !st.points_allowed(missing) {
            return Ok(None);
        }
    }
    let missing: Vec<&Vec<u64>> = needed
        .iter()
        .zip(&found)
        .filter(|(_, f)| f.is_none())
        .map(|(s, _)| s)
        .collect();
    let store = storage.store;
    let looked_up: Vec<ChunkRecord> = if missing.is_empty() {
        Vec::new()
    } else {
        match c.index {
            ChunkIndex::ExtensibleArray { address, .. } => {
                let (grid, unlim) = ea_grid(storage, g)?;
                let ea = ExtensibleArray::open(store, address, &g.index_context(storage))?;
                missing
                    .iter()
                    .map(|s| {
                        let li = extensible_linear_index(s, &grid, unlim)?;
                        if li >= ea.header.max_index_set {
                            Ok(ChunkRecord::UNALLOCATED)
                        } else {
                            ea.get(store, li)
                        }
                    })
                    .collect::<Result<_>>()?
            }
            ChunkIndex::BTreeV1 { address } => missing
                .iter()
                .map(|s| {
                    let off = g.offset_of(s)?;
                    Ok(
                        match find_chunk(store, address, storage.sizes, &g.chunk_dims, &off)? {
                            Some(e) => v1_record(g, &e),
                            None => ChunkRecord::UNALLOCATED,
                        },
                    )
                })
                .collect::<Result<_>>()?,
            ChunkIndex::BTreeV2 { address, .. } => {
                let tree = open_btree_v2(storage, g, address)?;
                missing
                    .iter()
                    .map(|s| {
                        let hit = tree.find(store, |r| match v2_record(r) {
                            Some((key, _)) => key.cmp(s.as_slice()),
                            None => Ordering::Less,
                        })?;
                        Ok(hit
                            .as_ref()
                            .and_then(v2_record)
                            .map_or(ChunkRecord::UNALLOCATED, |(_, rec)| rec))
                    })
                    .collect::<Result<_>>()?
            }
            _ => return Ok(None),
        }
    };
    if let Some(k) = storage.cache {
        let mut st = k.index();
        st.lookups = st.lookups.saturating_add(missing.len());
        for (s, r) in missing.iter().zip(&looked_up) {
            st.points.insert((*s).clone(), *r);
        }
    }
    let mut it = looked_up.into_iter();
    Ok(Some(
        found
            .into_iter()
            .map(|f| f.or_else(|| it.next()).unwrap_or(ChunkRecord::UNALLOCATED))
            .collect(),
    ))
}

/// List every allocated chunk of the dataset.
pub(super) fn list_chunks(
    storage: &DatasetStorage<'_>,
    c: &ChunkedLayout,
) -> Result<Vec<ChunkInfo>> {
    let g = Geometry::new(storage, c)?;
    let mut found: Vec<(Vec<u64>, ChunkRecord)> = Vec::new();
    if c.index.address() != UNDEFINED_ADDRESS {
        match c.index {
            ChunkIndex::SingleChunk { address, filtered } => {
                found.push((
                    vec![0u64; g.chunk_dims.len()],
                    single_chunk_record(address, filtered),
                ));
            }
            ChunkIndex::Implicit { address } => {
                let grid = max_grid(storage, &g)?;
                let idx = ImplicitIndex {
                    address,
                    chunk_bytes: g.chunk_bytes as u64,
                    num_chunks: total_chunks(&grid)?,
                };
                for (i, r) in idx.records()?.into_iter().enumerate() {
                    found.push((unlinearize(i as u64, &grid)?, r));
                }
            }
            _ => found = full_index(storage, c, &g)?.allocated.clone(),
        }
    }
    found
        .into_iter()
        .filter(|(_, r)| r.is_allocated())
        .map(|(scaled, r)| {
            Ok(ChunkInfo {
                offset: g.offset_of(&scaled)?,
                scaled,
                address: r.address,
                size: r.size.unwrap_or(g.chunk_bytes as u64),
                filter_mask: r.filter_mask,
            })
        })
        .collect()
}

fn single_chunk_record(address: u64, filtered: Option<FilteredChunk>) -> ChunkRecord {
    ChunkRecord {
        address,
        size: filtered.map(|f| f.size),
        filter_mask: filtered.map_or(0, |f| f.filter_mask),
    }
}

// ------------------------------------------------------ chunk decoding --

/// Read the stored (filtered) bytes of a chunk.
fn read_stored(storage: &DatasetStorage<'_>, rec: &ChunkRecord) -> Result<Vec<u8>> {
    let size = rec
        .size
        .ok_or_else(|| Error::invalid("filtered chunk without a stored size"))?;
    storage.store.read_vec_at(rec.address, to_usize(size)?)
}

/// Everything needed to undo the filter pipeline on a chunk, shareable
/// between threads.
struct Decoder<'a> {
    registry: &'a crate::filters::FilterRegistry,
    steps: Vec<PipelineStep<'a>>,
    ctx: FilterContext,
    chunk_bytes: usize,
}

impl Decoder<'_> {
    fn decode(&self, data: Vec<u8>, mask: u32) -> Result<Vec<u8>> {
        let out = decode_pipeline(self.registry, &self.steps, mask, data, &self.ctx)?;
        if out.len() != self.chunk_bytes {
            return Err(Error::invalid(format!(
                "decoded chunk has {} bytes, expected {}",
                out.len(),
                self.chunk_bytes
            )));
        }
        Ok(out)
    }

    /// Decode several chunks, in parallel with the `parallel` feature.
    fn decode_all(&self, items: Vec<(Vec<u8>, u32)>) -> Vec<Result<Vec<u8>>> {
        #[cfg(feature = "parallel")]
        {
            // At least MIN_THREAD_WORK decoded bytes per thread.
            let work = items.len().saturating_mul(self.chunk_bytes) / MIN_THREAD_WORK;
            let threads = decode_threads().min(items.len()).min(work.max(1));
            if threads > 1 {
                return self.decode_parallel(items, threads);
            }
        }
        items.into_iter().map(|(d, m)| self.decode(d, m)).collect()
    }

    #[cfg(feature = "parallel")]
    fn decode_parallel(&self, items: Vec<(Vec<u8>, u32)>, threads: usize) -> Vec<Result<Vec<u8>>> {
        let per = items.len().div_ceil(threads);
        let mut parts: Vec<Vec<(Vec<u8>, u32)>> = Vec::new();
        let mut it = items.into_iter().peekable();
        while it.peek().is_some() {
            parts.push(it.by_ref().take(per).collect());
        }
        std::thread::scope(|s| {
            let handles: Vec<_> = parts
                .into_iter()
                .map(|part| {
                    let n = part.len();
                    let h = s.spawn(move || {
                        part.into_iter()
                            .map(|(d, m)| self.decode(d, m))
                            .collect::<Vec<_>>()
                    });
                    (n, h)
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|(n, h)| {
                    h.join().unwrap_or_else(|_| {
                        (0..n)
                            .map(|_| Err(Error::invalid("chunk decoding thread panicked")))
                            .collect()
                    })
                })
                .collect()
        })
    }
}

/// Bound on the decoded bytes of one batch of chunks decoded together (in
/// parallel with the `parallel` feature).
const MAX_BATCH_BYTES: usize = 256 << 20;

/// Smallest amount of decoded data worth a decoding thread.
#[cfg(feature = "parallel")]
const MIN_THREAD_WORK: usize = 256 << 10;

/// Most threads decoding the chunks of one read.
fn decode_threads() -> usize {
    // `available_parallelism` costs system calls: ask once.
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(32)
    })
}

/// Copy the runs of a decoded chunk into the output.
fn copy_runs(data: &[u8], runs: &[DimRuns], esize: usize, out: &mut [u8]) -> Result<()> {
    for_each_group(runs, esize, &mut |g| copy_group(data, 0, g, out))
}

/// A filtered chunk waiting to be decoded.
struct Pending {
    address: u64,
    runs: Vec<DimRuns>,
    stored: Vec<u8>,
    mask: u32,
}

/// Visits the chunks a read needs and copies their selected elements.
struct Visitor<'s, 'a, 'o> {
    storage: &'s DatasetStorage<'a>,
    c: &'s ChunkedLayout,
    g: &'s Geometry,
    sel: &'s Resolved,
    out: &'o mut [u8],
    chunk_strides: Vec<u64>,
    store_size: u64,
    decoder: Option<Decoder<'a>>,
    pending: Vec<Pending>,
    batch: usize,
}

impl<'s, 'a, 'o> Visitor<'s, 'a, 'o> {
    fn new(
        storage: &'s DatasetStorage<'a>,
        c: &'s ChunkedLayout,
        g: &'s Geometry,
        sel: &'s Resolved,
        out: &'o mut [u8],
    ) -> Result<Self> {
        let decoder = storage.pipeline.filter(|_| g.filtered).map(|p| Decoder {
            registry: storage.registry,
            steps: p
                .filters
                .iter()
                .map(|f| PipelineStep {
                    id: f.id,
                    optional: f.optional,
                    client_data: &f.client_data,
                })
                .collect(),
            ctx: FilterContext::new(storage.element_size)
                .with_max_output(storage.limits.max_chunk_bytes)
                .with_expected_size(Some(g.chunk_bytes)),
            chunk_bytes: g.chunk_bytes,
        });
        let batch = if cfg!(feature = "parallel") {
            (MAX_BATCH_BYTES / g.chunk_bytes.max(1)).clamp(1, decode_threads().saturating_mul(4))
        } else {
            1
        };
        Ok(Visitor {
            storage,
            c,
            g,
            sel,
            out,
            chunk_strides: row_major_strides(&g.chunk_dims),
            store_size: storage.store.size()?,
            decoder,
            pending: Vec::new(),
            batch,
        })
    }

    /// Copy the selected part of the chunk at `scaled` (if allocated).
    fn visit(&mut self, scaled: &[u64], rec: ChunkRecord) -> Result<()> {
        if !rec.is_allocated() {
            return Ok(());
        }
        let g = self.g;
        let origin = g.offset_of(scaled)?;
        let extent: Vec<u64> = origin
            .iter()
            .zip(&g.chunk_dims)
            .zip(&g.dims)
            .map(|((&o, &c), &d)| c.min(d.saturating_sub(o)))
            .collect();
        let Some(runs) = block_runs(self.sel, &origin, &extent, &self.chunk_strides) else {
            return Ok(());
        };
        let esize = self.storage.element_size;
        if self.decoder.is_none() {
            // Unfiltered: read the selected bytes straight from the file.
            rec.address
                .checked_add(g.chunk_bytes as u64)
                .filter(|&end| end <= self.store_size)
                .ok_or(Error::UnexpectedEof)?;
            return read_block_runs(self.storage.store, rec.address, &runs, esize, self.out);
        }
        let registry = self.storage.registry.generation();
        if let Some(data) = self
            .storage
            .cache
            .and_then(|k| k.get(rec.address, registry))
        {
            return copy_runs(&data, &runs, esize, self.out);
        }
        let mut mask = rec.filter_mask;
        if self.c.flags & DONT_FILTER_PARTIAL_BOUND_CHUNKS != 0 && g.is_partial(&origin) {
            mask = u32::MAX;
        }
        let stored = read_stored(self.storage, &rec)?;
        self.pending.push(Pending {
            address: rec.address,
            runs,
            stored,
            mask,
        });
        if self.pending.len() >= self.batch {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        let Some(decoder) = &self.decoder else {
            return Ok(());
        };
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut meta = Vec::with_capacity(self.pending.len());
        let mut items = Vec::with_capacity(self.pending.len());
        for p in self.pending.drain(..) {
            meta.push((p.address, p.runs));
            items.push((p.stored, p.mask));
        }
        let esize = self.storage.element_size;
        for ((address, runs), data) in meta.into_iter().zip(decoder.decode_all(items)) {
            let data = data?;
            copy_runs(&data, &runs, esize, self.out)?;
            if let Some(k) = self.storage.cache
                && k.accepts(data.len())
            {
                k.insert(address, self.storage.registry.generation(), Arc::new(data));
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.flush()
    }
}

// --------------------------------------------------------- selection --

/// Indices of the chunks (along one dimension, chunk size `cd`) that hold
/// at least one element of the normalized selection `h`, increasing.
/// `None` when there are more than `cap`.
fn chunk_indices(h: &HyperslabDim, cd: u64, cap: usize) -> Option<Vec<u64>> {
    let mut out: Vec<u64> = Vec::new();
    let mut k = 0u64;
    while k < h.count {
        // In range: validated selections end within the dataspace.
        let bs = h.start + k * h.stride;
        let first = bs / cd;
        let last = (bs + h.block - 1) / cd;
        let from = match out.last() {
            Some(&prev) if prev >= first => prev + 1,
            _ => first,
        };
        if from <= last {
            let n = usize::try_from(last - from + 1).ok()?;
            if n > cap.saturating_sub(out.len()) {
                return None;
            }
            out.extend(from..=last);
        }
        // Skip the blocks starting inside chunk `last`, except the final
        // one (which may reach further chunks): their chunks are covered.
        let next_chunk = (last + 1).saturating_mul(cd);
        let skip_to = if next_chunk > h.start && h.stride > 0 {
            ((next_chunk - 1 - h.start) / h.stride).min(h.count - 1)
        } else {
            0
        };
        k = (k + 1).max(skip_to);
    }
    Some(out)
}

/// Per-dimension lists of the chunks a selection touches; `None` when
/// they would describe more than `cap` chunks.
fn needed_chunks(sel: &Resolved, g: &Geometry, cap: usize) -> Option<Vec<Vec<u64>>> {
    let mut total = 1usize;
    let mut per_dim = Vec::with_capacity(sel.dims.len());
    for (h, &cd) in sel.dims.iter().zip(&g.chunk_dims) {
        let v = chunk_indices(h, cd, cap)?;
        total = total.saturating_mul(v.len());
        if total > cap {
            return None;
        }
        per_dim.push(v);
    }
    Some(per_dim)
}

/// Call `f` with every element of the cartesian product of `per_dim`, in
/// row-major order.
fn for_each_coord(per_dim: &[Vec<u64>], mut f: impl FnMut(&[u64]) -> Result<()>) -> Result<()> {
    if per_dim.iter().any(Vec::is_empty) {
        return Ok(());
    }
    let mut idx = vec![0usize; per_dim.len()];
    let mut coord: Vec<u64> = per_dim.iter().map(|v| v[0]).collect();
    loop {
        f(&coord)?;
        let mut k = idx.len();
        loop {
            if k == 0 {
                return Ok(());
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] < per_dim[k].len() {
                coord[k] = per_dim[k][idx[k]];
                break;
            }
            idx[k] = 0;
            coord[k] = per_dim[k][0];
        }
    }
}

/// Chunked layout: copy the selected elements of every intersecting chunk.
pub(super) fn read_chunked(
    storage: &DatasetStorage<'_>,
    c: &ChunkedLayout,
    sel: &Resolved,
    out: &mut [u8],
) -> Result<()> {
    let g = Geometry::new(storage, c)?;
    if c.index.address() == UNDEFINED_ADDRESS || sel.count == 0 {
        // Nothing allocated: the output already holds the fill value.
        return Ok(());
    }
    let mut v = Visitor::new(storage, c, &g, sel, out)?;
    match c.index {
        ChunkIndex::SingleChunk { address, filtered } => {
            let zero = vec![0u64; g.chunk_dims.len()];
            v.visit(&zero, single_chunk_record(address, filtered))?;
        }
        ChunkIndex::Implicit { address } => {
            let grid = max_grid(storage, &g)?;
            // Every chunk is allocated, at distinct addresses: a selection
            // touching more chunks than fit in the file cannot be read.
            let cap = usize::try_from(v.store_size / g.chunk_bytes.max(1) as u64)
                .unwrap_or(usize::MAX)
                .saturating_add(1);
            let per_dim = needed_chunks(sel, &g, cap).ok_or(Error::UnexpectedEof)?;
            for_each_coord(&per_dim, |s| {
                let li = linear_index(s, &grid)?;
                let rec = ChunkRecord {
                    address: implicit_chunk_address(address, li, g.chunk_bytes as u64)?,
                    size: None,
                    filter_mask: 0,
                };
                v.visit(s, rec)
            })?;
        }
        _ => {
            let cached = storage.cache.and_then(|k| k.index().full.clone());
            let full = match cached {
                Some(f) => f,
                None => {
                    if let Some(per_dim) = needed_chunks(sel, &g, POINT_LOOKUP_MAX) {
                        let mut needed = Vec::new();
                        for_each_coord(&per_dim, |s| {
                            needed.push(s.to_vec());
                            Ok(())
                        })?;
                        if let Some(recs) = point_records(storage, c, &g, &needed)? {
                            for (s, r) in needed.iter().zip(recs) {
                                v.visit(s, r)?;
                            }
                            return v.finish();
                        }
                    }
                    full_index(storage, c, &g)?
                }
            };
            // Walk whichever is smaller: the chunks the selection touches,
            // or the allocated chunks.
            match needed_chunks(sel, &g, full.allocated.len()) {
                Some(per_dim) => for_each_coord(&per_dim, |s| v.visit(s, full.get(s)?))?,
                None => {
                    for (s, r) in &full.allocated {
                        v.visit(s, *r)?;
                    }
                }
            }
        }
    }
    v.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(h: &HyperslabDim, cd: u64) -> Vec<u64> {
        let mut v: Vec<u64> = (0..h.count)
            .flat_map(|k| {
                let bs = h.start + k * h.stride;
                (bs..bs + h.block).map(move |e| e / cd)
            })
            .collect();
        v.dedup();
        v
    }

    #[test]
    fn chunk_indices_match_brute_force() {
        for start in 0..5 {
            for stride in 1..9 {
                for block in 1..=stride {
                    for count in 1..12 {
                        for cd in 1..7 {
                            let h = HyperslabDim::new(start, stride, count, block);
                            let got = chunk_indices(&h, cd, usize::MAX).unwrap();
                            assert_eq!(got, brute(&h, cd), "{h:?} cd={cd}");
                        }
                    }
                }
            }
        }
        let h = HyperslabDim::new(0, 1, 1, 100);
        assert!(chunk_indices(&h, 1, 99).is_none());
        assert_eq!(chunk_indices(&h, 1, 100).unwrap().len(), 100);
    }

    #[test]
    fn cartesian_product() {
        let mut got = Vec::new();
        for_each_coord(&[vec![1, 2], vec![5, 7, 9]], |c| {
            got.push(c.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(got.len(), 6);
        assert_eq!(got[0], [1, 5]);
        assert_eq!(got[5], [2, 9]);
        let mut n = 0;
        for_each_coord(&[], |_| {
            n += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(n, 1);
    }
}
