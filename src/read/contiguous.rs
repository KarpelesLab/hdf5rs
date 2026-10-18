//! Compact and contiguous layouts (including external raw data files).

use super::selection::{
    DimRuns, Resolved, RunGroup, block_runs, copy_group, for_each_group, row_major_strides,
};
use super::{DatasetStorage, dataset_bytes};
use crate::error::{Error, Result};
use crate::format::UNDEFINED_ADDRESS;
use crate::format::local_heap::LocalHeap;
use crate::format::messages::dataspace::UNLIMITED;
use crate::format::messages::external_files::ExternalFiles;
use crate::io::ReadAt;

/// Source gaps up to this many bytes are read through rather than split
/// into separate reads.
const MAX_GAP: u64 = 16 << 10;
/// Upper bound of one coalesced read.
const MAX_SPAN: u64 = 4 << 20;

/// `u64` byte position/length to `usize`.
pub(super) fn to_usize(v: u64) -> Result<usize> {
    usize::try_from(v).map_err(|_| Error::LimitExceeded("size exceeds address space"))
}

/// Copy the selected elements of an in-memory array covering the whole
/// dataspace (row-major) into `out`.
pub(super) fn copy_from_full(
    storage: &DatasetStorage<'_>,
    src: &[u8],
    sel: &Resolved,
    out: &mut [u8],
) -> Result<()> {
    let dims = &storage.dataspace.dims;
    let origin = vec![0u64; dims.len()];
    let Some(runs) = block_runs(sel, &origin, dims, &row_major_strides(dims)) else {
        return Ok(());
    };
    for_each_group(&runs, storage.element_size, &mut |g| {
        copy_group(src, 0, g, out)
    })
}

/// Compact layout: the raw data is stored in the layout message.
pub(super) fn read_compact(
    storage: &DatasetStorage<'_>,
    data: &[u8],
    sel: &Resolved,
    out: &mut [u8],
) -> Result<()> {
    if (data.len() as u64) < dataset_bytes(storage)? {
        return Err(Error::invalid("compact data smaller than the dataset"));
    }
    copy_from_full(storage, data, sel, out)
}

/// Contiguous layout.
pub(super) fn read_contiguous(
    storage: &DatasetStorage<'_>,
    address: u64,
    size: Option<u64>,
    sel: &Resolved,
    out: &mut [u8],
) -> Result<()> {
    let needed = dataset_bytes(storage)?;
    if let Some(ext) = storage.external {
        let Some(resolver) = storage.resolver else {
            return Err(Error::unsupported(
                "external raw data files (no resolver configured)",
            ));
        };
        let ext_store = ExternalStore::open(storage, ext, resolver)?;
        if ext_store.size()? < needed {
            return Err(Error::invalid("external files smaller than the dataset"));
        }
        return read_runs(storage, &ext_store, 0, sel, out);
    }
    if address == UNDEFINED_ADDRESS {
        // Never written: the output already holds the fill value.
        return Ok(());
    }
    if size.is_some_and(|s| s < needed) {
        return Err(Error::invalid(
            "contiguous storage smaller than the dataset",
        ));
    }
    address
        .checked_add(needed)
        .filter(|&end| end != UNDEFINED_ADDRESS)
        .ok_or_else(|| Error::invalid("contiguous storage address overflows"))?;
    read_runs(storage, storage.store, address, sel, out)
}

/// Read the selected elements of a row-major array stored at `base` in
/// `store`, coalescing nearby runs into larger reads.
fn read_runs(
    storage: &DatasetStorage<'_>,
    store: &dyn ReadAt,
    base: u64,
    sel: &Resolved,
    out: &mut [u8],
) -> Result<()> {
    let dims = &storage.dataspace.dims;
    let origin = vec![0u64; dims.len()];
    let Some(runs) = block_runs(sel, &origin, dims, &row_major_strides(dims)) else {
        return Ok(());
    };
    read_block_runs(store, base, &runs, storage.element_size, out)
}

/// Read the runs of a row-major block of `esize`-byte elements stored at
/// `base` in `store` into `out`, coalescing nearby runs into larger reads.
pub(super) fn read_block_runs(
    store: &dyn ReadAt,
    base: u64,
    runs: &[DimRuns],
    esize: usize,
    out: &mut [u8],
) -> Result<()> {
    let mut span = Span {
        store,
        base,
        start: 0,
        end: 0,
        groups: Vec::new(),
        tmp: Vec::new(),
    };
    for_each_group(runs, esize, &mut |g| span.push_group(g, out))?;
    span.flush(out)
}

/// A pending coalesced read: byte range `start..end` (relative to `base`)
/// and the run groups it serves.
struct Span<'s> {
    store: &'s dyn ReadAt,
    base: u64,
    start: u64,
    end: u64,
    groups: Vec<RunGroup>,
    tmp: Vec<u8>,
}

impl Span<'_> {
    /// Queue a group, split so that every queued piece is worth reading
    /// through (small gaps) and spans at most [`MAX_SPAN`] bytes.
    fn push_group(&mut self, g: &RunGroup, out: &mut [u8]) -> Result<()> {
        if g.count > 1 && g.src_step.saturating_sub(g.len) > MAX_GAP {
            // Far apart: one direct read per run.
            self.flush(out)?;
            let len = to_usize(g.len)?;
            let (mut src, mut o) = (g.src, to_usize(g.out)?);
            let (src_step, out_step) = (g.src_step, to_usize(g.out_step)?);
            for _ in 0..g.count {
                let dst = o
                    .checked_add(len)
                    .and_then(|e| out.get_mut(o..e))
                    .ok_or_else(|| Error::invalid("selection run outside output"))?;
                let pos = self
                    .base
                    .checked_add(src)
                    .ok_or(Error::LimitExceeded("file position overflows"))?;
                self.store.read_exact_at(pos, dst)?;
                src = src.saturating_add(src_step);
                o = o.saturating_add(out_step);
            }
            return Ok(());
        }
        let per = if g.count <= 1 {
            1
        } else {
            (MAX_SPAN.saturating_sub(g.len) / g.src_step.max(1)).saturating_add(1)
        };
        let mut k = 0u64;
        while k < g.count {
            let n = per.min(g.count - k);
            let at = |step: u64, from: u64| {
                k.checked_mul(step)
                    .and_then(|d| d.checked_add(from))
                    .ok_or(Error::LimitExceeded("byte offset overflows"))
            };
            let sub = RunGroup {
                src: at(g.src_step, g.src)?,
                out: at(g.out_step, g.out)?,
                count: n,
                ..*g
            };
            self.push(sub, out)?;
            k += n;
        }
        Ok(())
    }

    fn push(&mut self, g: RunGroup, out: &mut [u8]) -> Result<()> {
        let (start, end) = (g.src, g.src_end()?);
        let mergeable = !self.groups.is_empty()
            && start >= self.end
            && start - self.end <= MAX_GAP
            && end - self.start <= MAX_SPAN;
        if !mergeable {
            self.flush(out)?;
            self.start = start;
        }
        self.end = end;
        self.groups.push(g);
        Ok(())
    }

    fn flush(&mut self, out: &mut [u8]) -> Result<()> {
        let pos = |off: u64| {
            self.base
                .checked_add(off)
                .ok_or(Error::LimitExceeded("file position overflows"))
        };
        match self.groups.as_slice() {
            [] => {}
            [g] if g.count == 1 => {
                let o = to_usize(g.out)?;
                let len = to_usize(g.len)?;
                let dst = o
                    .checked_add(len)
                    .and_then(|e| out.get_mut(o..e))
                    .ok_or_else(|| Error::invalid("selection run outside output"))?;
                self.store.read_exact_at(pos(g.src)?, dst)?;
            }
            groups => {
                let span_len = to_usize(self.end - self.start)?;
                self.tmp.clear();
                self.tmp.resize(span_len, 0);
                self.store.read_exact_at(pos(self.start)?, &mut self.tmp)?;
                for g in groups {
                    copy_group(&self.tmp, self.start, g, out)?;
                }
            }
        }
        self.groups.clear();
        Ok(())
    }
}

/// One external file segment mapped into the dataset's address space.
struct Segment {
    /// First dataset byte covered.
    start: u64,
    /// Number of dataset bytes covered (may be [`u64::MAX`] = unlimited).
    size: u64,
    /// Offset of the data within the external file.
    file_offset: u64,
    file: Box<dyn ReadAt>,
}

/// The concatenation of a dataset's external file segments, viewed as one
/// byte store. Bytes reserved in a segment but beyond the end of its file
/// read as zeros.
struct ExternalStore {
    segments: Vec<Segment>,
    total: u64,
}

impl ExternalStore {
    fn open(
        storage: &DatasetStorage<'_>,
        ext: &ExternalFiles,
        resolver: &dyn super::ExternalFileResolver,
    ) -> Result<Self> {
        let heap = LocalHeap::read(storage.store, ext.heap_address, storage.sizes)?;
        let mut segments = Vec::with_capacity(ext.slots.len());
        let mut total = 0u64;
        for slot in &ext.slots {
            let name = heap.get_string_lossy(slot.name_offset)?;
            let file = resolver.open(&name)?;
            segments.push(Segment {
                start: total,
                size: slot.size,
                file_offset: slot.file_offset,
                file,
            });
            if slot.size == UNLIMITED {
                total = u64::MAX;
                break;
            }
            total = total.saturating_add(slot.size);
        }
        Ok(ExternalStore { segments, total })
    }
}

impl ReadAt for ExternalStore {
    fn read_exact_at(&self, mut pos: u64, mut buf: &mut [u8]) -> Result<()> {
        while !buf.is_empty() {
            let seg = self
                .segments
                .iter()
                .find(|s| pos >= s.start && pos - s.start < s.size)
                .ok_or(Error::UnexpectedEof)?;
            let rel = pos - seg.start;
            let avail = seg.size - rel;
            let take = usize::try_from(avail).unwrap_or(usize::MAX).min(buf.len());
            let (now, rest) = buf.split_at_mut(take);
            let fpos = seg
                .file_offset
                .checked_add(rel)
                .ok_or(Error::UnexpectedEof)?;
            let fsize = seg.file.size()?;
            // Part present in the file, then zeros for the reserved tail.
            let present = usize::try_from(fsize.saturating_sub(fpos))
                .unwrap_or(usize::MAX)
                .min(take);
            let (head, tail) = now.split_at_mut(present);
            if !head.is_empty() {
                seg.file.read_exact_at(fpos, head)?;
            }
            tail.fill(0);
            buf = rest;
            pos = pos.checked_add(take as u64).ok_or(Error::UnexpectedEof)?;
        }
        Ok(())
    }

    fn size(&self) -> Result<u64> {
        Ok(self.total)
    }
}
