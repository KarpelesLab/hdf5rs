//! Chunked dataset storage: chunk encoding (filters), a write-back cache
//! for partially written chunks, and the chunk index built when the file
//! is finished.
//!
//! Chunks are written as soon as they are complete: a write covering every
//! in-bounds element of a chunk stores it immediately (edge chunks padded
//! with the fill value). Partially covered chunks are kept in memory and
//! merged with later writes; they are written out when the cache exceeds
//! its budget (only if the storage can read them back later), on
//! [`flush`](super::DatasetWriter::flush) and when the file is finished.
//! Rewriting a stored chunk reuses its space when the new encoding fits,
//! otherwise new space is allocated (the old space is not reused).

use std::collections::BTreeMap;

use super::storage::{Allocator, WriteAt};
use crate::error::{Error, Result};
use crate::filters::{
    FilterContext, FilterRegistry, PipelineStep, decode_pipeline, encode_pipeline,
};
use crate::format::btree_v1::{ChunkEntry, ChunkKey, encode_chunk_btree};
use crate::format::btree_v2::{BTreeV2Header, Record, encode_btree_v2};
use crate::format::chunk_index::{
    ChunkIndexContext, ChunkRecord, EncodedIndex, chunks_per_dim, extensible_linear_index,
    filtered_chunk_size_width_v4, linear_index, total_chunks,
};
use crate::format::extensible_array::{ExtensibleArrayParams, encode_extensible_array};
use crate::format::fixed_array::encode_fixed_array;
use crate::format::messages::dataspace::UNLIMITED;
use crate::format::messages::filter_pipeline::FilterPipeline;
use crate::format::messages::layout::{
    ChunkIndex, ChunkedLayout, DONT_FILTER_PARTIAL_BOUND_CHUNKS, DataLayout, FilteredChunk,
    SINGLE_INDEX_WITH_FILTER,
};
use crate::format::{Sizes, UNDEFINED_ADDRESS};
use crate::read::{Resolved, block_runs, for_each_run, row_major_strides};

/// K of chunk v1 B-trees (the reference library's default).
pub(crate) const CHUNK_BTREE_K: u16 = 32;
/// log2 of the fixed array page size (the reference library's default).
const FIXED_ARRAY_PAGE_BITS: u8 = 10;
/// v2 B-tree chunk index node size (the reference library's default).
const BTREE_V2_NODE_SIZE: u32 = 2048;
/// Default budget of the partially-written chunk cache, per dataset.
pub(crate) const DEFAULT_CACHE_BYTES: usize = 32 << 20;

/// Which chunk index a dataset uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndexKind {
    /// Version 1 B-tree (layout version 3).
    BTreeV1,
    /// The dataset is a single chunk.
    Single,
    /// Fixed-size, unfiltered, early-allocated: chunks stored contiguously.
    Implicit,
    /// Fixed maximum dimensions.
    FixedArray,
    /// Exactly one unlimited dimension.
    ExtensibleArray,
    /// Several unlimited dimensions.
    BTreeV2,
}

impl IndexKind {
    /// Choose the index the reference library would use for a dataset with
    /// maximum dimensions `max`, chunks `chunk` (layout version 4).
    pub(crate) fn choose_v4(max: &[u64], chunk: &[u64], filtered: bool, early: bool) -> Self {
        let unlimited = max.iter().filter(|&&m| m == UNLIMITED).count();
        match unlimited {
            0 => {
                let single = max.iter().zip(chunk).all(|(&m, &c)| m <= c);
                if single {
                    IndexKind::Single
                } else if early && !filtered {
                    IndexKind::Implicit
                } else {
                    IndexKind::FixedArray
                }
            }
            1 => IndexKind::ExtensibleArray,
            _ => IndexKind::BTreeV2,
        }
    }
}

/// A stored chunk: its index record and the bytes reserved for it.
#[derive(Debug, Clone, Copy)]
struct Stored {
    record: ChunkRecord,
    capacity: u64,
}

/// Per-call view of the dataset a chunk operation needs.
pub(crate) struct Env<'a> {
    pub(crate) sizes: Sizes,
    pub(crate) storage: &'a mut dyn WriteAt,
    pub(crate) alloc: &'a mut Allocator,
    pub(crate) registry: &'a FilterRegistry,
    /// Element size in bytes.
    pub(crate) element_size: usize,
    /// One element of fill value.
    pub(crate) fill: &'a [u8],
    /// Current dimensions.
    pub(crate) dims: &'a [u64],
    /// Maximum dimensions ([`UNLIMITED`] for unlimited ones).
    pub(crate) max_dims: &'a [u64],
    /// Cache budget in bytes.
    pub(crate) cache_limit: usize,
}

/// Chunked storage state of one dataset.
#[derive(Debug, Clone)]
pub(crate) struct ChunkedSpec {
    /// Chunk dimensions (elements).
    pub(crate) chunk: Vec<u64>,
    /// Filter pipeline (may be empty).
    pub(crate) pipeline: FilterPipeline,
    pub(crate) kind: IndexKind,
    /// Store partial edge chunks unfiltered.
    pub(crate) partial_unfiltered: bool,
    /// Allocate every chunk (written with the fill value) at the end.
    pub(crate) early: bool,
    /// Size in bytes of one unfiltered chunk.
    pub(crate) chunk_bytes: usize,
    stored: BTreeMap<Vec<u64>, Stored>,
    cache: BTreeMap<Vec<u64>, Vec<u8>>,
    cache_bytes: usize,
    /// Base address of an implicit index.
    implicit_base: u64,
    /// The finished index (set by [`ChunkedSpec::finalize`]).
    index: Option<ChunkIndex>,
}

impl ChunkedSpec {
    pub(crate) fn new(
        chunk: Vec<u64>,
        pipeline: FilterPipeline,
        kind: IndexKind,
        partial_unfiltered: bool,
        early: bool,
        element_size: usize,
    ) -> Result<Self> {
        let chunk_bytes = chunk
            .iter()
            .try_fold(element_size as u64, |a, &c| a.checked_mul(c))
            .filter(|&b| b <= u64::from(u32::MAX))
            .and_then(|b| usize::try_from(b).ok())
            .ok_or(Error::LimitExceeded("chunk larger than 4 GiB"))?;
        Ok(ChunkedSpec {
            chunk,
            pipeline,
            kind,
            partial_unfiltered,
            early,
            chunk_bytes,
            stored: BTreeMap::new(),
            cache: BTreeMap::new(),
            cache_bytes: 0,
            implicit_base: UNDEFINED_ADDRESS,
            index: None,
        })
    }

    fn filtered(&self) -> bool {
        !self.pipeline.filters.is_empty()
    }

    /// Layout message version used for this dataset.
    pub(crate) fn layout_version(&self) -> u8 {
        if self.kind == IndexKind::BTreeV1 {
            3
        } else {
            4
        }
    }

    /// Reserve (and fill) the contiguous chunk array of an implicit index,
    /// sized from the maximum dimensions.
    pub(crate) fn allocate_implicit(&mut self, env: &mut Env<'_>) -> Result<()> {
        let grid = chunks_per_dim(env.max_dims, &self.chunk)?;
        let n = total_chunks(&grid)?;
        let total = n
            .checked_mul(self.chunk_bytes as u64)
            .ok_or(Error::LimitExceeded("chunk array size"))?;
        if total == 0 {
            return Ok(());
        }
        self.implicit_base = env.alloc.alloc(total)?;
        if env.fill.iter().any(|&b| b != 0) {
            let chunk = self.fill_chunk(env);
            for i in 0..n {
                env.storage
                    .write_all_at(self.implicit_base + i * self.chunk_bytes as u64, &chunk)?;
            }
        }
        Ok(())
    }

    fn fill_chunk(&self, env: &Env<'_>) -> Vec<u8> {
        if env.fill.is_empty() {
            return vec![0; self.chunk_bytes];
        }
        env.fill.repeat(self.chunk_bytes / env.fill.len())
    }

    fn origin(&self, scaled: &[u64]) -> Vec<u64> {
        scaled
            .iter()
            .zip(&self.chunk)
            .map(|(&s, &c)| s * c)
            .collect()
    }

    /// In-bounds extent of the chunk at `origin` for dimensions `dims`.
    fn extent(&self, origin: &[u64], dims: &[u64]) -> Vec<u64> {
        origin
            .iter()
            .zip(&self.chunk)
            .zip(dims)
            .map(|((&o, &c), &d)| c.min(d.saturating_sub(o)))
            .collect()
    }

    fn is_partial(&self, scaled: &[u64], dims: &[u64]) -> bool {
        self.origin(scaled)
            .iter()
            .zip(&self.chunk)
            .zip(dims)
            .any(|((&o, &c), &d)| o.saturating_add(c) > d)
    }

    fn steps(&self) -> Vec<PipelineStep<'_>> {
        self.pipeline
            .filters
            .iter()
            .map(|f| PipelineStep {
                id: f.id,
                optional: f.optional,
                client_data: &f.client_data,
            })
            .collect()
    }

    fn all_filters_mask(&self) -> u32 {
        let n = self.pipeline.filters.len() as u32;
        if n >= 32 { u32::MAX } else { (1u32 << n) - 1 }
    }

    /// Encode and write one chunk (unfiltered bytes `data`).
    fn store(&mut self, env: &mut Env<'_>, scaled: &[u64], data: Vec<u8>) -> Result<()> {
        if data.len() != self.chunk_bytes {
            return Err(Error::invalid("chunk buffer size mismatch"));
        }
        if self.kind == IndexKind::Implicit {
            let grid = chunks_per_dim(env.max_dims, &self.chunk)?;
            let li = linear_index(scaled, &grid)?;
            let addr = self.implicit_base + li * self.chunk_bytes as u64;
            env.storage.write_all_at(addr, &data)?;
            self.stored.insert(
                scaled.to_vec(),
                Stored {
                    record: ChunkRecord {
                        address: addr,
                        size: None,
                        filter_mask: 0,
                    },
                    capacity: self.chunk_bytes as u64,
                },
            );
            return Ok(());
        }
        let (bytes, mask) = if !self.filtered() {
            (data, 0)
        } else if self.partial_unfiltered && self.is_partial(scaled, env.dims) {
            (data, self.all_filters_mask())
        } else {
            let ctx = FilterContext::new(env.element_size);
            encode_pipeline(env.registry, &self.steps(), data, &ctx)?
        };
        let len = bytes.len() as u64;
        if self.kind == IndexKind::BTreeV1 && len > u64::from(u32::MAX) {
            return Err(Error::LimitExceeded("filtered chunk larger than 4 GiB"));
        }
        let (address, capacity) = match self.stored.get(scaled) {
            Some(s) if s.capacity >= len => (s.record.address, s.capacity),
            _ => (env.alloc.alloc(len)?, len),
        };
        env.storage.write_all_at(address, &bytes)?;
        self.stored.insert(
            scaled.to_vec(),
            Stored {
                record: ChunkRecord {
                    address,
                    size: self.filtered().then_some(len),
                    filter_mask: mask,
                },
                capacity,
            },
        );
        Ok(())
    }

    /// The current (unfiltered) contents of a chunk: from the cache, read
    /// back from storage, or the fill value.
    fn load(&mut self, env: &mut Env<'_>, scaled: &[u64]) -> Result<Vec<u8>> {
        if let Some(buf) = self.cache.remove(scaled) {
            self.cache_bytes = self.cache_bytes.saturating_sub(buf.len());
            return Ok(buf);
        }
        let Some(s) = self.stored.get(scaled).copied() else {
            return Ok(self.fill_chunk(env));
        };
        if !env.storage.supports_read_back() {
            return Err(Error::unsupported(
                "rewriting a stored chunk requires storage that can be read back",
            ));
        }
        let len = s.record.size.unwrap_or(self.chunk_bytes as u64);
        let len = usize::try_from(len).map_err(|_| Error::LimitExceeded("chunk size"))?;
        let mut buf = vec![0u8; len];
        env.storage.read_exact_at(s.record.address, &mut buf)?;
        if !self.filtered() || self.kind == IndexKind::Implicit {
            return Ok(buf);
        }
        let ctx = FilterContext::new(env.element_size).with_expected_size(Some(self.chunk_bytes));
        let out = decode_pipeline(env.registry, &self.steps(), s.record.filter_mask, buf, &ctx)?;
        if out.len() != self.chunk_bytes {
            return Err(Error::invalid("decoded chunk has the wrong size"));
        }
        Ok(out)
    }

    fn cache_put(&mut self, scaled: Vec<u64>, buf: Vec<u8>) {
        self.cache_bytes += buf.len();
        if let Some(old) = self.cache.insert(scaled, buf) {
            self.cache_bytes = self.cache_bytes.saturating_sub(old.len());
        }
    }

    /// Write every cached chunk out.
    pub(crate) fn flush(&mut self, env: &mut Env<'_>) -> Result<()> {
        let cache = std::mem::take(&mut self.cache);
        self.cache_bytes = 0;
        for (scaled, buf) in cache {
            self.store(env, &scaled, buf)?;
        }
        Ok(())
    }

    /// Write the selected elements (`data`: packed row-major, `sel.count`
    /// elements) into the chunks they fall in.
    pub(crate) fn write(&mut self, env: &mut Env<'_>, sel: &Resolved, data: &[u8]) -> Result<()> {
        let esize = env.element_size;
        if data.len() as u64 != sel.count.saturating_mul(esize as u64) {
            return Err(Error::invalid("data size does not match the selection"));
        }
        if sel.count == 0 {
            return Ok(());
        }
        // Chunk coordinates touched along each dimension.
        let per_dim: Vec<Vec<u64>> = sel
            .dims
            .iter()
            .zip(&self.chunk)
            .map(|(h, &c)| {
                let mut v: Vec<u64> = Vec::new();
                for k in 0..h.count {
                    let start = h.start + k * h.stride;
                    let (a, b) = (start / c, (start + h.block - 1) / c);
                    let from = match v.last() {
                        Some(&last) if last >= a => last + 1,
                        _ => a,
                    };
                    v.extend(from..=b);
                }
                v
            })
            .collect();
        let strides = row_major_strides(&self.chunk);
        let mut idx = vec![0usize; per_dim.len()];
        loop {
            let scaled: Vec<u64> = idx.iter().zip(&per_dim).map(|(&i, v)| v[i]).collect();
            let origin = self.origin(&scaled);
            let extent = self.extent(&origin, env.dims);
            if let Some(runs) = block_runs(sel, &origin, &extent, &strides) {
                let mut covered = 0u64;
                for_each_run(&runs, &mut |_, _, n| {
                    covered += n;
                    Ok(())
                })?;
                let full = covered == extent.iter().product::<u64>();
                let mut buf = if full {
                    if let Some(old) = self.cache.remove(&scaled) {
                        self.cache_bytes = self.cache_bytes.saturating_sub(old.len());
                    }
                    self.fill_chunk(env)
                } else {
                    self.load(env, &scaled)?
                };
                for_each_run(&runs, &mut |s, o, n| {
                    let (s, o, n) = (s as usize * esize, o as usize * esize, n as usize * esize);
                    let src = data
                        .get(o..o + n)
                        .ok_or_else(|| Error::invalid("selection run outside data"))?;
                    buf.get_mut(s..s + n)
                        .ok_or_else(|| Error::invalid("selection run outside chunk"))?
                        .copy_from_slice(src);
                    Ok(())
                })?;
                if full {
                    self.store(env, &scaled, buf)?;
                } else {
                    self.cache_put(scaled, buf);
                }
            }
            // Odometer over the per-dimension chunk lists.
            let mut k = idx.len();
            loop {
                if k == 0 {
                    if self.cache_bytes > env.cache_limit && env.storage.supports_read_back() {
                        self.flush(env)?;
                    }
                    return Ok(());
                }
                k -= 1;
                idx[k] += 1;
                if idx[k] < per_dim[k].len() {
                    break;
                }
                idx[k] = 0;
            }
        }
    }

    /// Prepare for a change of the current dimensions from `env.dims` to
    /// `new_dims`: chunks entirely outside the new extent are dropped, the
    /// parts of boundary chunks that fall outside are reset to the fill
    /// value (so that growing again exposes fill values), and with
    /// [`partial_unfiltered`](Self::partial_unfiltered) chunks whose edge
    /// status changes are queued for re-encoding.
    pub(crate) fn resize(&mut self, env: &mut Env<'_>, new_dims: &[u64]) -> Result<()> {
        let keys: Vec<Vec<u64>> = self
            .stored
            .keys()
            .chain(self.cache.keys())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let shrinks = new_dims.iter().zip(env.dims).any(|(n, o)| n < o);
        let strides = row_major_strides(&self.chunk);
        for scaled in keys {
            let origin = self.origin(&scaled);
            let outside = origin.iter().zip(new_dims).any(|(&o, &d)| o >= d);
            if outside && self.kind != IndexKind::Implicit {
                if let Some(old) = self.cache.remove(&scaled) {
                    self.cache_bytes = self.cache_bytes.saturating_sub(old.len());
                }
                self.stored.remove(&scaled);
                continue;
            }
            let old_ext = self.extent(&origin, env.dims);
            let new_ext = self.extent(&origin, new_dims);
            let cut = shrinks && new_ext.iter().zip(&old_ext).any(|(n, o)| n < o);
            let edge_change = self.partial_unfiltered
                && self.filtered()
                && self.is_partial(&scaled, env.dims) != self.is_partial(&scaled, new_dims);
            if !cut && !edge_change {
                continue;
            }
            let mut buf = self.load(env, &scaled)?;
            if cut {
                // Keep only the new in-bounds region.
                let mut kept = self.fill_chunk(env);
                let ranges: Vec<std::ops::Range<u64>> = origin
                    .iter()
                    .zip(&new_ext)
                    .map(|(&o, &e)| o..o + e)
                    .collect();
                let space = crate::format::messages::dataspace::Dataspace::simple(new_dims, None);
                let sel = crate::read::Selection::from_ranges(&ranges).resolve(&space)?;
                if let Some(runs) = block_runs(&sel, &origin, &new_ext, &strides) {
                    let esize = env.element_size;
                    for_each_run(&runs, &mut |s, _, n| {
                        let (s, n) = (s as usize * esize, n as usize * esize);
                        let src = buf.get(s..s + n).ok_or(Error::UnexpectedEof)?;
                        kept.get_mut(s..s + n)
                            .ok_or(Error::UnexpectedEof)?
                            .copy_from_slice(src);
                        Ok(())
                    })?;
                }
                buf = kept;
            }
            self.cache_put(scaled, buf);
        }
        Ok(())
    }

    /// Write out cached chunks, allocate fill chunks for early allocation,
    /// and build the chunk index.
    pub(crate) fn finalize(&mut self, env: &mut Env<'_>) -> Result<()> {
        self.flush(env)?;
        if self.early && self.kind != IndexKind::Implicit {
            let grid = chunks_per_dim(env.dims, &self.chunk)?;
            let n = total_chunks(&grid)?;
            if n > (1 << 24) {
                return Err(Error::LimitExceeded("too many chunks to allocate early"));
            }
            for li in 0..n {
                let scaled = crate::format::chunk_index::unlinearize(li, &grid)?;
                if !self.stored.contains_key(&scaled) {
                    let fill = self.fill_chunk(env);
                    self.store(env, &scaled, fill)?;
                }
            }
        }
        let sizes = env.sizes;
        let filtered = self.filtered();
        let width = filtered_chunk_size_width_v4(self.chunk_bytes as u64);
        let ctx = ChunkIndexContext {
            sizes,
            filtered,
            chunk_size_width: Some(width),
        };
        let alloc = &mut *env.alloc;
        let mut alloc_fn = |n: u64| alloc.alloc(n);
        let encoded: Option<EncodedIndex> = if self.stored.is_empty() {
            None
        } else {
            match self.kind {
                IndexKind::Single | IndexKind::Implicit => None,
                IndexKind::FixedArray => {
                    let grid = chunks_per_dim(env.max_dims, &self.chunk)?;
                    let n = total_chunks(&grid)?;
                    let n = usize::try_from(n)
                        .ok()
                        .filter(|&n| n <= 1 << 28)
                        .ok_or(Error::LimitExceeded("too many chunks for a fixed array"))?;
                    let mut recs = vec![ChunkRecord::UNALLOCATED; n];
                    for (scaled, s) in &self.stored {
                        let li = linear_index(scaled, &grid)? as usize;
                        if let Some(slot) = recs.get_mut(li) {
                            *slot = s.record;
                        }
                    }
                    Some(encode_fixed_array(
                        &ctx,
                        width,
                        FIXED_ARRAY_PAGE_BITS,
                        &recs,
                        &mut alloc_fn,
                    )?)
                }
                IndexKind::ExtensibleArray => {
                    let unlim = env
                        .max_dims
                        .iter()
                        .position(|&m| m == UNLIMITED)
                        .ok_or_else(|| Error::invalid("no unlimited dimension"))?;
                    let maxes: Vec<u64> = env
                        .max_dims
                        .iter()
                        .zip(env.dims)
                        .map(|(&m, &d)| if m == UNLIMITED { d } else { m })
                        .collect();
                    let grid = chunks_per_dim(&maxes, &self.chunk)?;
                    let mut by_index = BTreeMap::new();
                    for (scaled, s) in &self.stored {
                        by_index.insert(extensible_linear_index(scaled, &grid, unlim)?, s.record);
                    }
                    let n = by_index.keys().next_back().map_or(0, |&k| k + 1);
                    let n = usize::try_from(n).ok().filter(|&n| n <= 1 << 28).ok_or(
                        Error::LimitExceeded("too many chunks for an extensible array"),
                    )?;
                    let mut recs = vec![ChunkRecord::UNALLOCATED; n];
                    for (li, r) in by_index {
                        if let Some(slot) = recs.get_mut(li as usize) {
                            *slot = r;
                        }
                    }
                    Some(encode_extensible_array(
                        &ctx,
                        width,
                        &ExtensibleArrayParams::default(),
                        &recs,
                        &mut alloc_fn,
                    )?)
                }
                IndexKind::BTreeV2 => {
                    let rank = self.chunk.len();
                    let record_size = usize::from(sizes.offset)
                        + if filtered { usize::from(width) + 4 } else { 0 }
                        + 8 * rank;
                    let template = BTreeV2Header {
                        tree_type: if filtered { 11 } else { 10 },
                        node_size: BTREE_V2_NODE_SIZE,
                        record_size: u16::try_from(record_size)
                            .map_err(|_| Error::LimitExceeded("chunk record size"))?,
                        depth: 0,
                        split_percent: 100,
                        merge_percent: 40,
                        root_address: UNDEFINED_ADDRESS,
                        root_num_records: 0,
                        total_records: 0,
                    };
                    let records: Vec<Record> = self
                        .stored
                        .iter()
                        .map(|(scaled, s)| {
                            let r = s.record;
                            if filtered {
                                Record::FilteredChunk {
                                    address: r.address,
                                    chunk_size: r.size.unwrap_or(0),
                                    filter_mask: r.filter_mask,
                                    scaled_offsets: scaled.clone(),
                                }
                            } else {
                                Record::Chunk {
                                    address: r.address,
                                    scaled_offsets: scaled.clone(),
                                }
                            }
                        })
                        .collect();
                    Some(encode_btree_v2(
                        &template,
                        sizes,
                        &records,
                        usize::from(width),
                        &mut alloc_fn,
                    )?)
                }
                IndexKind::BTreeV1 => {
                    let entries = self
                        .stored
                        .iter()
                        .map(|(scaled, s)| {
                            Ok(ChunkEntry {
                                offsets: self.origin(scaled),
                                size: u32::try_from(
                                    s.record.size.unwrap_or(self.chunk_bytes as u64),
                                )
                                .map_err(|_| Error::LimitExceeded("chunk size"))?,
                                filter_mask: s.record.filter_mask,
                                address: s.record.address,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut end = entries
                        .last()
                        .map(|e| e.offsets.clone())
                        .ok_or_else(|| Error::invalid("empty chunk B-tree"))?;
                    if let (Some(e0), Some(c0)) = (end.first_mut(), self.chunk.first()) {
                        *e0 += c0;
                    }
                    let end_key = ChunkKey {
                        size: 0,
                        filter_mask: 0,
                        offsets: end,
                    };
                    Some(encode_chunk_btree(
                        sizes,
                        CHUNK_BTREE_K,
                        &entries,
                        &end_key,
                        env.element_size as u64,
                        &mut alloc_fn,
                    )?)
                }
            }
        };
        if let Some(e) = &encoded {
            for b in &e.blocks {
                env.storage.write_all_at(b.address, &b.bytes)?;
            }
        }
        let address = encoded.as_ref().map_or(UNDEFINED_ADDRESS, |e| e.address);
        self.index = Some(match self.kind {
            IndexKind::BTreeV1 => ChunkIndex::BTreeV1 { address },
            IndexKind::Single => {
                let zero = vec![0u64; self.chunk.len()];
                let rec = self.stored.get(&zero).map(|s| s.record);
                ChunkIndex::SingleChunk {
                    address: rec.map_or(UNDEFINED_ADDRESS, |r| r.address),
                    filtered: filtered.then(|| FilteredChunk {
                        size: rec.and_then(|r| r.size).unwrap_or(0),
                        filter_mask: rec.map_or(0, |r| r.filter_mask),
                    }),
                }
            }
            IndexKind::Implicit => ChunkIndex::Implicit {
                address: self.implicit_base,
            },
            IndexKind::FixedArray => ChunkIndex::FixedArray {
                address,
                page_bits: FIXED_ARRAY_PAGE_BITS,
            },
            IndexKind::ExtensibleArray => {
                let p = ExtensibleArrayParams::default();
                ChunkIndex::ExtensibleArray {
                    address,
                    max_bits: p.max_nelmts_bits,
                    index_elements: p.index_block_elements,
                    min_pointers: p.secondary_block_min_data_ptrs,
                    min_elements: p.data_block_min_elements,
                    page_bits: p.max_data_block_page_elements_bits,
                }
            }
            IndexKind::BTreeV2 => ChunkIndex::BTreeV2 {
                address,
                node_size: BTREE_V2_NODE_SIZE,
                split_percent: 100,
                merge_percent: 40,
            },
        });
        Ok(())
    }

    /// The data layout message. Only meaningful after
    /// [`finalize`](Self::finalize), which sets the index address.
    pub(crate) fn layout(&self, element_size: u32) -> DataLayout {
        let mut flags = 0;
        if self.layout_version() >= 4 {
            if self.partial_unfiltered && self.filtered() {
                flags |= DONT_FILTER_PARTIAL_BOUND_CHUNKS;
            }
            if self.kind == IndexKind::Single && self.filtered() {
                flags |= SINGLE_INDEX_WITH_FILTER;
            }
        }
        let index = self.index.unwrap_or(match self.kind {
            IndexKind::BTreeV1 => ChunkIndex::BTreeV1 {
                address: UNDEFINED_ADDRESS,
            },
            _ => ChunkIndex::Implicit {
                address: UNDEFINED_ADDRESS,
            },
        });
        DataLayout::Chunked(ChunkedLayout {
            dims: self.chunk.clone(),
            element_size: u64::from(element_size),
            flags,
            index,
            layout_version: self.layout_version(),
        })
    }
}

/// Pick chunk dimensions for a dataset of `dims` (maximum `max`, with
/// [`UNLIMITED`] entries) and `element_size`-byte elements: start from the
/// current size (at least 1024 along unlimited dimensions, at least 1
/// everywhere), then halve the largest dimension until a chunk is at most
/// 64 KiB, never exceeding a fixed maximum.
pub(crate) fn guess_chunk(dims: &[u64], max: &[u64], element_size: usize) -> Vec<u64> {
    const TARGET: u64 = 64 * 1024;
    let mut chunk: Vec<u64> = dims
        .iter()
        .zip(max)
        .map(|(&d, &m)| {
            if m == UNLIMITED {
                d.max(1024)
            } else {
                d.min(m).max(1)
            }
        })
        .collect();
    let bytes = |c: &[u64]| {
        c.iter()
            .try_fold(element_size.max(1) as u64, |a, &x| a.checked_mul(x))
            .unwrap_or(u64::MAX)
    };
    while bytes(&chunk) > TARGET {
        let Some((i, _)) = chunk
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 1)
            .max_by_key(|&(_, &c)| c)
        else {
            break;
        };
        chunk[i] = chunk[i].div_ceil(2);
    }
    chunk
}
