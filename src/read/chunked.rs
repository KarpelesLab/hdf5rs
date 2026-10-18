//! Chunked layout: chunk index lookup, per-chunk decoding and copying.

use std::cmp::Ordering;
use std::collections::HashMap;

use super::DatasetStorage;
use super::contiguous::{byte_off, to_usize};
use super::selection::{HyperslabDim, Resolved, block_runs, for_each_run, row_major_strides};
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
    ChunkIndex, ChunkedLayout, DONT_FILTER_PARTIAL_BOUND_CHUNKS,
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

/// Look up the records of the chunks at the given scaled coordinates
/// (unallocated chunks yield [`ChunkRecord::UNALLOCATED`]).
fn lookup(
    storage: &DatasetStorage<'_>,
    c: &ChunkedLayout,
    g: &Geometry,
    needed: &[Vec<u64>],
) -> Result<Vec<ChunkRecord>> {
    let store = storage.store;
    let unallocated = || vec![ChunkRecord::UNALLOCATED; needed.len()];
    if c.index.address() == UNDEFINED_ADDRESS {
        return Ok(unallocated());
    }
    let few = needed.len() <= POINT_LOOKUP_MAX;
    match c.index {
        ChunkIndex::SingleChunk { address, filtered } => {
            let rec = ChunkRecord {
                address,
                size: filtered.map(|f| f.size),
                filter_mask: filtered.map_or(0, |f| f.filter_mask),
            };
            Ok(needed
                .iter()
                .map(|s| {
                    if s.iter().all(|&v| v == 0) {
                        rec
                    } else {
                        ChunkRecord::UNALLOCATED
                    }
                })
                .collect())
        }
        ChunkIndex::Implicit { address } => {
            let grid = max_grid(storage, g)?;
            needed
                .iter()
                .map(|s| {
                    let li = linear_index(s, &grid)?;
                    Ok(ChunkRecord {
                        address: implicit_chunk_address(address, li, g.chunk_bytes as u64)?,
                        size: None,
                        filter_mask: 0,
                    })
                })
                .collect()
        }
        ChunkIndex::FixedArray { address, .. } => {
            let grid = max_grid(storage, g)?;
            let fa = FixedArray::open(store, address, &g.index_context(storage))?;
            if fa.header.max_num_entries != total_chunks(&grid)? {
                return Err(Error::invalid(
                    "fixed array size does not match the chunk grid",
                ));
            }
            let records = fa.records(store)?;
            needed
                .iter()
                .map(|s| {
                    let li = linear_index(s, &grid)?;
                    usize::try_from(li)
                        .ok()
                        .and_then(|i| records.get(i).copied())
                        .ok_or_else(|| Error::invalid("chunk outside fixed array"))
                })
                .collect()
        }
        ChunkIndex::ExtensibleArray { address, .. } => {
            let (grid, unlim) = ea_grid(storage, g)?;
            let ea = ExtensibleArray::open(store, address, &g.index_context(storage))?;
            let indices = needed
                .iter()
                .map(|s| extensible_linear_index(s, &grid, unlim))
                .collect::<Result<Vec<u64>>>()?;
            if few {
                return indices
                    .iter()
                    .map(|&li| {
                        if li >= ea.header.max_index_set {
                            Ok(ChunkRecord::UNALLOCATED)
                        } else {
                            ea.get(store, li)
                        }
                    })
                    .collect();
            }
            let records = ea.records(store, ea.header.max_index_set)?;
            Ok(indices
                .iter()
                .map(|&li| {
                    usize::try_from(li)
                        .ok()
                        .and_then(|i| records.get(i).copied())
                        .unwrap_or(ChunkRecord::UNALLOCATED)
                })
                .collect())
        }
        ChunkIndex::BTreeV1 { address } => {
            if few {
                return needed
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
                    .collect();
            }
            let entries = chunk_btree_entries(store, address, storage.sizes, g.chunk_dims.len())?;
            let mut map = HashMap::with_capacity(entries.len());
            for e in &entries {
                map.insert(g.scaled_of(&e.offsets)?, v1_record(g, e));
            }
            Ok(needed
                .iter()
                .map(|s| map.get(s).copied().unwrap_or(ChunkRecord::UNALLOCATED))
                .collect())
        }
        ChunkIndex::BTreeV2 { address, .. } => {
            let tree = open_btree_v2(storage, g, address)?;
            if few {
                return needed
                    .iter()
                    .map(|s| {
                        let found = tree.find(store, |r| match v2_record(r) {
                            Some((key, _)) => key.cmp(s.as_slice()),
                            None => Ordering::Less,
                        })?;
                        Ok(found
                            .as_ref()
                            .and_then(v2_record)
                            .map_or(ChunkRecord::UNALLOCATED, |(_, rec)| rec))
                    })
                    .collect();
            }
            let mut map = HashMap::new();
            for r in tree.records(store)? {
                if let Some((key, rec)) = v2_record(&r) {
                    map.insert(key.to_vec(), rec);
                }
            }
            Ok(needed
                .iter()
                .map(|s| map.get(s).copied().unwrap_or(ChunkRecord::UNALLOCATED))
                .collect())
        }
    }
}

/// List every allocated chunk of the dataset.
pub(super) fn list_chunks(
    storage: &DatasetStorage<'_>,
    c: &ChunkedLayout,
) -> Result<Vec<ChunkInfo>> {
    let g = Geometry::new(storage, c)?;
    let store = storage.store;
    let mut found: Vec<(Vec<u64>, ChunkRecord)> = Vec::new();
    if c.index.address() != UNDEFINED_ADDRESS {
        match c.index {
            ChunkIndex::SingleChunk { .. } => {
                let zero = vec![0u64; g.chunk_dims.len()];
                let rec = lookup(storage, c, &g, std::slice::from_ref(&zero))?;
                found.extend(rec.into_iter().map(|r| (zero.clone(), r)));
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
            ChunkIndex::FixedArray { address, .. } => {
                let grid = max_grid(storage, &g)?;
                let records = FixedArray::read(store, address, &g.index_context(storage))?;
                for (i, r) in records.into_iter().enumerate() {
                    if r.is_allocated() {
                        found.push((unlinearize(i as u64, &grid)?, r));
                    }
                }
            }
            ChunkIndex::ExtensibleArray { address, .. } => {
                let (grid, unlim) = ea_grid(storage, &g)?;
                let records = ExtensibleArray::read(store, address, &g.index_context(storage))?;
                for (i, r) in records.into_iter().enumerate() {
                    if r.is_allocated() {
                        found.push((extensible_unlinearize(i as u64, &grid, unlim)?, r));
                    }
                }
            }
            ChunkIndex::BTreeV1 { address } => {
                for e in chunk_btree_entries(store, address, storage.sizes, g.chunk_dims.len())? {
                    found.push((g.scaled_of(&e.offsets)?, v1_record(&g, &e)));
                }
            }
            ChunkIndex::BTreeV2 { address, .. } => {
                let tree = open_btree_v2(storage, &g, address)?;
                for r in tree.records(store)? {
                    if let Some((key, rec)) = v2_record(&r) {
                        found.push((key.to_vec(), rec));
                    }
                }
            }
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

/// Read and decode one allocated chunk.
fn load_chunk(
    storage: &DatasetStorage<'_>,
    c: &ChunkedLayout,
    g: &Geometry,
    rec: &ChunkRecord,
    offset: &[u64],
) -> Result<Vec<u8>> {
    let store = storage.store;
    let Some(pipeline) = storage.pipeline.filter(|_| g.filtered) else {
        return store.read_vec_at(rec.address, g.chunk_bytes);
    };
    let size = rec
        .size
        .ok_or_else(|| Error::invalid("filtered chunk without a stored size"))?;
    let mut mask = rec.filter_mask;
    if c.flags & DONT_FILTER_PARTIAL_BOUND_CHUNKS != 0 && g.is_partial(offset) {
        mask = u32::MAX;
    }
    let data = store.read_vec_at(rec.address, to_usize(size)?)?;
    let steps: Vec<PipelineStep<'_>> = pipeline
        .filters
        .iter()
        .map(|f| PipelineStep {
            id: f.id,
            optional: f.optional,
            client_data: &f.client_data,
        })
        .collect();
    let ctx = FilterContext::new(storage.element_size)
        .with_max_output(storage.limits.max_chunk_bytes)
        .with_expected_size(Some(g.chunk_bytes));
    let out = decode_pipeline(storage.registry, &steps, mask, data, &ctx)?;
    if out.len() != g.chunk_bytes {
        return Err(Error::invalid(format!(
            "decoded chunk has {} bytes, expected {}",
            out.len(),
            g.chunk_bytes
        )));
    }
    Ok(out)
}

/// Indices of the chunks (along one dimension, chunk size `cd`) that hold
/// at least one element of the normalized selection `h`, increasing.
fn chunk_indices(h: &HyperslabDim, cd: u64) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    for k in 0..h.count {
        // In range: validated selections end within the dataspace.
        let bs = h.start + k * h.stride;
        let first = bs / cd;
        let last = (bs + h.block - 1) / cd;
        let from = match out.last() {
            Some(&prev) if prev >= first => prev + 1,
            _ => first,
        };
        out.extend(from..=last);
    }
    out
}

/// Chunked layout: copy the selected elements of every intersecting chunk.
pub(super) fn read_chunked(
    storage: &DatasetStorage<'_>,
    c: &ChunkedLayout,
    sel: &Resolved,
    out: &mut [u8],
) -> Result<()> {
    let g = Geometry::new(storage, c)?;
    let per_dim: Vec<Vec<u64>> = sel
        .dims
        .iter()
        .zip(&g.chunk_dims)
        .map(|(h, &cd)| chunk_indices(h, cd))
        .collect();
    if per_dim.iter().any(Vec::is_empty) {
        return Ok(());
    }
    // Cartesian product of the per-dimension chunk indices.
    let mut needed: Vec<Vec<u64>> = Vec::new();
    let mut idx = vec![0usize; per_dim.len()];
    'outer: loop {
        needed.push(idx.iter().zip(&per_dim).map(|(&i, v)| v[i]).collect());
        let mut k = idx.len();
        loop {
            if k == 0 {
                break 'outer;
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] < per_dim[k].len() {
                break;
            }
            idx[k] = 0;
        }
    }
    let records = lookup(storage, c, &g, &needed)?;
    let chunk_strides = row_major_strides(&g.chunk_dims);
    let esize = storage.element_size;
    for (scaled, rec) in needed.iter().zip(&records) {
        if !rec.is_allocated() {
            continue;
        }
        let origin = g.offset_of(scaled)?;
        let extent: Vec<u64> = origin
            .iter()
            .zip(&g.chunk_dims)
            .zip(&g.dims)
            .map(|((&o, &c), &d)| c.min(d.saturating_sub(o)))
            .collect();
        let Some(runs) = block_runs(sel, &origin, &extent, &chunk_strides) else {
            continue;
        };
        let data = load_chunk(storage, c, &g, rec, &origin)?;
        for_each_run(&runs, &mut |s, o, n| {
            let s = to_usize(byte_off(s, esize)?)?;
            let o = to_usize(byte_off(o, esize)?)?;
            let n = to_usize(byte_off(n, esize)?)?;
            let from = s
                .checked_add(n)
                .and_then(|e| data.get(s..e))
                .ok_or(Error::UnexpectedEof)?;
            let to = o
                .checked_add(n)
                .and_then(|e| out.get_mut(o..e))
                .ok_or_else(|| Error::invalid("selection run outside output"))?;
            to.copy_from_slice(from);
            Ok(())
        })?;
    }
    Ok(())
}
