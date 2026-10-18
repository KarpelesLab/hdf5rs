//! Dataset selections and the run-based copy engine.
//!
//! A [`Selection`] picks elements of a dataspace. Internally every
//! selection is normalized to one regular hyperslab per dimension; copying
//! between a source block (a chunk, or the whole contiguous dataset) and
//! the packed output is expressed as *runs*: maximal stretches of elements
//! that are contiguous both in the source and in the output.

use std::ops::Range;

use crate::error::{Error, Result};
use crate::format::messages::dataspace::{Dataspace, DataspaceKind};

/// A regular hyperslab along one dimension: `count` blocks of `block`
/// consecutive elements, the first starting at `start`, successive blocks
/// `stride` elements apart (HDF5's `start`/`stride`/`count`/`block`).
///
/// Blocks may not overlap: when `count > 1`, `stride` must be at least
/// `block`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HyperslabDim {
    /// Index of the first selected element.
    pub start: u64,
    /// Distance between the starts of successive blocks.
    pub stride: u64,
    /// Number of blocks.
    pub count: u64,
    /// Number of consecutive elements per block.
    pub block: u64,
}

impl HyperslabDim {
    /// A hyperslab from its four parameters.
    pub fn new(start: u64, stride: u64, count: u64, block: u64) -> Self {
        HyperslabDim {
            start,
            stride,
            count,
            block,
        }
    }

    /// The contiguous range `range.start..range.end` (empty if
    /// `range.end <= range.start`).
    pub fn range(range: Range<u64>) -> Self {
        let len = range.end.saturating_sub(range.start);
        if len == 0 {
            HyperslabDim::new(range.start, 1, 0, 1)
        } else {
            HyperslabDim::new(range.start, len, 1, len)
        }
    }

    /// Every `step`-th element of `range` (`step` >= 1).
    pub fn strided(range: Range<u64>, step: u64) -> Self {
        let len = range.end.saturating_sub(range.start);
        let step = step.max(1);
        HyperslabDim::new(range.start, step, len.div_ceil(step), 1)
    }

    /// Number of selected elements (`count * block`).
    pub fn len(&self) -> Result<u64> {
        self.count
            .checked_mul(self.block)
            .ok_or(Error::LimitExceeded("selection size overflows"))
    }

    /// Whether no element is selected.
    pub fn is_empty(&self) -> bool {
        self.count == 0 || self.block == 0
    }

    /// Coordinate of the `i`-th selected element (normalized, in-range
    /// selections only).
    pub(crate) fn coord(&self, i: u64) -> u64 {
        let block = self.block.max(1);
        self.start
            .saturating_add((i / block).saturating_mul(self.stride))
            .saturating_add(i % block)
    }

    /// Position within the selection of coordinate `c`, if selected.
    pub(crate) fn index_of(&self, c: u64) -> Option<u64> {
        if self.is_empty() || c < self.start {
            return None;
        }
        let off = c - self.start;
        let stride = self.stride.max(1);
        let (b, w) = (off / stride, off % stride);
        if b >= self.count || w >= self.block {
            return None;
        }
        b.checked_mul(self.block)?.checked_add(w)
    }

    /// One past the last selected coordinate (normalized selections).
    pub(crate) fn end(&self) -> u64 {
        if self.is_empty() {
            return self.start;
        }
        (self.count - 1)
            .saturating_mul(self.stride)
            .saturating_add(self.start)
            .saturating_add(self.block)
    }

    /// Check the hyperslab against a dimension of size `dim` and return its
    /// canonical form (adjacent blocks merged; empty selections become
    /// `count == 0`).
    fn normalize(&self, dim: u64, axis: usize) -> Result<HyperslabDim> {
        let bad =
            |what: &str| -> Error { Error::invalid(format!("selection dimension {axis}: {what}")) };
        if self.is_empty() {
            return Ok(HyperslabDim::new(0, 1, 0, 1));
        }
        if self.count > 1 && self.stride < self.block {
            return Err(bad("stride smaller than block (overlapping blocks)"));
        }
        let end = (self.count - 1)
            .checked_mul(self.stride)
            .and_then(|v| v.checked_add(self.start))
            .and_then(|v| v.checked_add(self.block))
            .ok_or_else(|| bad("extent overflows"))?;
        if end > dim {
            return Err(bad("extends beyond the dataspace"));
        }
        if self.count == 1 || self.stride == self.block {
            // One contiguous block. `len <= dim`, so it cannot overflow.
            let len = self.count * self.block;
            return Ok(HyperslabDim::new(self.start, len, 1, len));
        }
        Ok(*self)
    }

    /// Append to `out` the parts of this (normalized) selection that fall
    /// inside the block of elements `b0 .. b0 + ext`: source position
    /// relative to `b0`, length, and position in the selection's output.
    pub(crate) fn pieces_in(&self, b0: u64, ext: u64, out: &mut Vec<Piece>) {
        if self.count == 0 || ext == 0 {
            return;
        }
        // Validated selections end within the dataspace, so none of the
        // arithmetic below on in-range values can overflow; saturate anyway.
        let b1 = b0.saturating_add(ext);
        let sel_end = self.start + (self.count - 1) * self.stride + self.block;
        if b1 <= self.start || b0 >= sel_end {
            return;
        }
        // First block whose end is after b0.
        let k_min = if b0 < self.start + self.block {
            0
        } else {
            (b0 - self.start - self.block) / self.stride + 1
        };
        // Last block whose start is before b1.
        let k_max = ((b1 - 1 - self.start) / self.stride).min(self.count - 1);
        let mut k = k_min;
        while k <= k_max {
            let bs = self.start + k * self.stride;
            let a = bs.max(b0);
            let z = (bs + self.block).min(b1);
            if a < z {
                out.push(Piece {
                    src: a - b0,
                    len: z - a,
                    out: k * self.block + (a - bs),
                });
            }
            k += 1;
        }
    }
}

/// A selection of dataset elements.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum Selection {
    /// Every element (the only valid selection of scalar and null
    /// dataspaces).
    #[default]
    All,
    /// One regular hyperslab per dimension; the selected elements are the
    /// cartesian product of the per-dimension selections.
    Hyperslab(Vec<HyperslabDim>),
}

impl Selection {
    /// Select `ranges[i]` along dimension `i` (half-open ranges).
    pub fn from_ranges(ranges: &[Range<u64>]) -> Selection {
        Selection::Hyperslab(ranges.iter().cloned().map(HyperslabDim::range).collect())
    }

    /// Select the half-open range `range` of a one-dimensional dataspace.
    pub fn range(range: Range<u64>) -> Selection {
        Selection::Hyperslab(vec![HyperslabDim::range(range)])
    }

    /// A hyperslab selection from per-dimension parameters.
    pub fn hyperslab(dims: impl Into<Vec<HyperslabDim>>) -> Selection {
        Selection::Hyperslab(dims.into())
    }

    /// Validate against `space` and return the canonical per-dimension form.
    pub(crate) fn resolve(&self, space: &Dataspace) -> Result<Resolved> {
        match space.kind {
            DataspaceKind::Null => {
                return match self {
                    Selection::All => Ok(Resolved::empty()),
                    Selection::Hyperslab(d) if d.is_empty() => Ok(Resolved::empty()),
                    _ => Err(Error::invalid("hyperslab selection on a null dataspace")),
                };
            }
            DataspaceKind::Scalar => {
                return match self {
                    Selection::All => Ok(Resolved::scalar()),
                    Selection::Hyperslab(d) if d.is_empty() => Ok(Resolved::scalar()),
                    _ => Err(Error::invalid("hyperslab selection on a scalar dataspace")),
                };
            }
            DataspaceKind::Simple => {}
        }
        let dims: Vec<HyperslabDim> = match self {
            Selection::All => space
                .dims
                .iter()
                .map(|&d| HyperslabDim::range(0..d))
                .collect(),
            Selection::Hyperslab(h) => {
                if h.len() != space.rank() {
                    return Err(Error::invalid(format!(
                        "selection rank {} does not match dataspace rank {}",
                        h.len(),
                        space.rank()
                    )));
                }
                h.clone()
            }
        };
        let dims = dims
            .iter()
            .zip(&space.dims)
            .enumerate()
            .map(|(i, (h, &d))| h.normalize(d, i))
            .collect::<Result<Vec<_>>>()?;
        let shape = dims
            .iter()
            .map(HyperslabDim::len)
            .collect::<Result<Vec<_>>>()?;
        let count = shape.iter().try_fold(1u64, |acc, &n| {
            acc.checked_mul(n)
                .ok_or(Error::LimitExceeded("selection size overflows"))
        })?;
        Ok(Resolved {
            dims,
            shape,
            count,
            scalar: false,
        })
    }

    /// Shape of the packed output for dataspace `space` (empty for scalar
    /// and null dataspaces).
    pub fn output_shape(&self, space: &Dataspace) -> Result<Vec<u64>> {
        Ok(self.resolve(space)?.shape)
    }

    /// Number of elements selected in `space`.
    pub fn num_elements(&self, space: &Dataspace) -> Result<u64> {
        Ok(self.resolve(space)?.count)
    }
}

/// A validated, normalized selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Resolved {
    /// Normalized per-dimension selections (empty for scalar/null).
    pub dims: Vec<HyperslabDim>,
    /// Output shape.
    pub shape: Vec<u64>,
    /// Number of selected elements.
    pub count: u64,
    /// The dataspace is scalar (rank 0, one element).
    pub scalar: bool,
}

impl Resolved {
    fn empty() -> Self {
        Resolved {
            dims: Vec::new(),
            shape: Vec::new(),
            count: 0,
            scalar: false,
        }
    }

    fn scalar() -> Self {
        Resolved {
            dims: Vec::new(),
            shape: Vec::new(),
            count: 1,
            scalar: true,
        }
    }

    /// Row-major strides (in elements) of the output.
    pub fn out_strides(&self) -> Vec<u64> {
        row_major_strides(&self.shape)
    }
}

/// Row-major strides of an array of shape `shape`, in elements. The caller
/// guarantees the total element count fits in a `u64`.
pub(crate) fn row_major_strides(shape: &[u64]) -> Vec<u64> {
    let mut out = vec![1u64; shape.len()];
    let mut acc = 1u64;
    for (slot, &n) in out.iter_mut().zip(shape).rev() {
        *slot = acc;
        acc = acc.saturating_mul(n);
    }
    out
}

/// A selected stretch along one dimension of a source block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Piece {
    /// Start, relative to the source block.
    pub src: u64,
    /// Number of elements.
    pub len: u64,
    /// Start in the output along this dimension.
    pub out: u64,
}

/// The pieces of one dimension, with the source and output strides of that
/// dimension (in elements).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DimRuns {
    pub pieces: Vec<Piece>,
    pub src_stride: u64,
    pub out_stride: u64,
}

/// Build the per-dimension pieces for copying the block with origin
/// `origin` (element coordinates), in-bounds extent `extent` and source
/// strides `src_strides` into the output described by `sel`. Returns `None`
/// when the block contains no selected element.
pub(crate) fn block_runs(
    sel: &Resolved,
    origin: &[u64],
    extent: &[u64],
    src_strides: &[u64],
) -> Option<Vec<DimRuns>> {
    let out_strides = sel.out_strides();
    let mut dims = Vec::with_capacity(sel.dims.len());
    for (i, h) in sel.dims.iter().enumerate() {
        let mut pieces = Vec::new();
        h.pieces_in(origin[i], extent[i], &mut pieces);
        if pieces.is_empty() {
            return None;
        }
        dims.push(DimRuns {
            pieces,
            src_stride: src_strides[i],
            out_stride: out_strides[i],
        });
    }
    collapse(&mut dims);
    Some(dims)
}

/// Merge trailing dimensions that are copied whole into their parent so
/// that runs become as long as possible.
pub(crate) fn collapse(dims: &mut Vec<DimRuns>) {
    while dims.len() >= 2 {
        let j = dims.len() - 1;
        let [p] = dims[j].pieces.as_slice() else {
            return;
        };
        let p = *p;
        let n = p.len;
        let full = p.src == 0
            && p.out == 0
            && dims[j - 1].src_stride == n.saturating_mul(dims[j].src_stride)
            && dims[j - 1].out_stride == n.saturating_mul(dims[j].out_stride);
        if !full {
            return;
        }
        let scaled: Option<Vec<Piece>> = dims[j - 1]
            .pieces
            .iter()
            .map(|q| {
                Some(Piece {
                    src: q.src.checked_mul(n)?,
                    len: q.len.checked_mul(n)?,
                    out: q.out.checked_mul(n)?,
                })
            })
            .collect();
        let Some(scaled) = scaled else {
            return;
        };
        let last = dims.pop().unwrap_or(DimRuns {
            pieces: Vec::new(),
            src_stride: 1,
            out_stride: 1,
        });
        let parent = &mut dims[j - 1];
        parent.pieces = scaled;
        parent.src_stride = last.src_stride;
        parent.out_stride = last.out_stride;
    }
}

/// Call `f(src_offset, out_offset, len)` (all in elements) for every run,
/// in row-major order. Source offsets are strictly increasing. A rank-0
/// list yields the single run `(0, 0, 1)`.
pub(crate) fn for_each_run(
    dims: &[DimRuns],
    f: &mut dyn FnMut(u64, u64, u64) -> Result<()>,
) -> Result<()> {
    let Some((last, outer)) = dims.split_last() else {
        return f(0, 0, 1);
    };
    // Per outer dimension: (piece index, position within the piece).
    let mut idx = vec![(0usize, 0u64); outer.len()];
    if outer
        .iter()
        .any(|d| d.pieces.is_empty() || d.pieces.iter().any(|p| p.len == 0))
    {
        return Ok(());
    }
    loop {
        let mut src = 0u64;
        let mut out = 0u64;
        for (d, &(i, w)) in outer.iter().zip(&idx) {
            let p = d.pieces[i];
            src = src.saturating_add(p.src.saturating_add(w).saturating_mul(d.src_stride));
            out = out.saturating_add(p.out.saturating_add(w).saturating_mul(d.out_stride));
        }
        for p in &last.pieces {
            f(
                src.saturating_add(p.src.saturating_mul(last.src_stride)),
                out.saturating_add(p.out.saturating_mul(last.out_stride)),
                p.len,
            )?;
        }
        // Odometer increment over the outer dimensions.
        let mut k = outer.len();
        loop {
            if k == 0 {
                return Ok(());
            }
            k -= 1;
            let (i, w) = &mut idx[k];
            *w += 1;
            if *w < outer[k].pieces[*i].len {
                break;
            }
            *w = 0;
            *i += 1;
            if *i < outer[k].pieces.len() {
                break;
            }
            *i = 0;
        }
    }
}

/// `count` equally spaced runs of `len` bytes: run `k` copies source bytes
/// `src + k * src_step ..` to output bytes `out + k * out_step ..`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunGroup {
    pub src: u64,
    pub out: u64,
    pub len: u64,
    pub count: u64,
    pub src_step: u64,
    pub out_step: u64,
}

impl RunGroup {
    /// One past the last source byte.
    pub fn src_end(&self) -> Result<u64> {
        (self.count.saturating_sub(1))
            .checked_mul(self.src_step)
            .and_then(|v| v.checked_add(self.src))
            .and_then(|v| v.checked_add(self.len))
            .ok_or(Error::LimitExceeded("byte offset overflows"))
    }
}

/// Like [`for_each_run`], but in bytes (elements of `esize` bytes) and
/// with equally spaced runs of the last dimension grouped, so strided
/// selections cost one callback per group rather than per element.
pub(crate) fn for_each_group(
    dims: &[DimRuns],
    esize: usize,
    f: &mut dyn FnMut(&RunGroup) -> Result<()>,
) -> Result<()> {
    let es = esize as u64;
    let bytes = |v: u64| {
        v.checked_mul(es)
            .ok_or(Error::LimitExceeded("byte offset overflows"))
    };
    let Some((last, outer)) = dims.split_last() else {
        return f(&RunGroup {
            src: 0,
            out: 0,
            len: es,
            count: 1,
            src_step: 0,
            out_step: 0,
        });
    };
    // Group the last dimension's pieces (relative byte offsets).
    let mut groups: Vec<RunGroup> = Vec::new();
    for p in &last.pieces {
        let src = bytes(p.src.saturating_mul(last.src_stride))?;
        let out = bytes(p.out.saturating_mul(last.out_stride))?;
        let len = bytes(p.len)?;
        if let Some(g) = groups.last_mut()
            && g.len == len
            && src > g.src
            && out > g.out
        {
            let (ds, dout) = (src - g.src, out - g.out);
            if g.count == 1 {
                g.src_step = ds;
                g.out_step = dout;
                g.count = 2;
                continue;
            }
            if g.count.checked_mul(g.src_step) == Some(ds)
                && g.count.checked_mul(g.out_step) == Some(dout)
            {
                g.count += 1;
                continue;
            }
        }
        groups.push(RunGroup {
            src,
            out,
            len,
            count: 1,
            src_step: 0,
            out_step: 0,
        });
    }
    if outer
        .iter()
        .any(|d| d.pieces.is_empty() || d.pieces.iter().any(|p| p.len == 0))
    {
        return Ok(());
    }
    // One element run per row (e.g. a column): group along the next
    // dimension instead.
    let mut outer = outer;
    if let [g] = groups.as_slice()
        && g.count == 1
        && let Some((next, rest)) = outer.split_last()
    {
        let g = *g;
        let add = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or(Error::LimitExceeded("byte offset overflows"))
        };
        groups = next
            .pieces
            .iter()
            .map(|p| {
                Ok(RunGroup {
                    src: add(bytes(p.src.saturating_mul(next.src_stride))?, g.src)?,
                    out: add(bytes(p.out.saturating_mul(next.out_stride))?, g.out)?,
                    len: g.len,
                    count: p.len,
                    src_step: bytes(next.src_stride)?,
                    out_step: bytes(next.out_stride)?,
                })
            })
            .collect::<Result<_>>()?;
        outer = rest;
    }
    let mut idx = vec![(0usize, 0u64); outer.len()];
    loop {
        let mut src = 0u64;
        let mut out = 0u64;
        for (d, &(i, w)) in outer.iter().zip(&idx) {
            let p = d.pieces[i];
            src = src.saturating_add(p.src.saturating_add(w).saturating_mul(d.src_stride));
            out = out.saturating_add(p.out.saturating_add(w).saturating_mul(d.out_stride));
        }
        let (src, out) = (bytes(src)?, bytes(out)?);
        for g in &groups {
            let shifted = RunGroup {
                src: src
                    .checked_add(g.src)
                    .ok_or(Error::LimitExceeded("byte offset overflows"))?,
                out: out
                    .checked_add(g.out)
                    .ok_or(Error::LimitExceeded("byte offset overflows"))?,
                ..*g
            };
            f(&shifted)?;
        }
        let mut k = outer.len();
        loop {
            if k == 0 {
                return Ok(());
            }
            k -= 1;
            let (i, w) = &mut idx[k];
            *w += 1;
            if *w < outer[k].pieces[*i].len {
                break;
            }
            *w = 0;
            *i += 1;
            if *i < outer[k].pieces.len() {
                break;
            }
            *i = 0;
        }
    }
}

/// Copy the runs of `g` from `src` (whose first byte is source offset
/// `src_base`) into `out`.
pub(crate) fn copy_group(src: &[u8], src_base: u64, g: &RunGroup, out: &mut [u8]) -> Result<()> {
    let usz =
        |v: u64| usize::try_from(v).map_err(|_| Error::LimitExceeded("size exceeds address space"));
    let start = usz(g.src.checked_sub(src_base).ok_or(Error::UnexpectedEof)?)?;
    let (len, ss, os) = (usz(g.len)?, usz(g.src_step)?, usz(g.out_step)?);
    let o = usz(g.out)?;
    match len {
        1 => copy_fixed::<1>(src, out, start, o, ss, os, g.count),
        2 => copy_fixed::<2>(src, out, start, o, ss, os, g.count),
        4 => copy_fixed::<4>(src, out, start, o, ss, os, g.count),
        8 => copy_fixed::<8>(src, out, start, o, ss, os, g.count),
        _ => {
            let (mut s, mut o) = (start, o);
            for _ in 0..g.count {
                let from = s
                    .checked_add(len)
                    .and_then(|e| src.get(s..e))
                    .ok_or(Error::UnexpectedEof)?;
                let to = o
                    .checked_add(len)
                    .and_then(|e| out.get_mut(o..e))
                    .ok_or_else(|| Error::invalid("selection run outside output"))?;
                to.copy_from_slice(from);
                s = s.saturating_add(ss);
                o = o.saturating_add(os);
            }
            Ok(())
        }
    }
}

/// [`copy_group`] for runs of exactly `N` bytes.
fn copy_fixed<const N: usize>(
    src: &[u8],
    out: &mut [u8],
    mut s: usize,
    mut o: usize,
    ss: usize,
    os: usize,
    count: u64,
) -> Result<()> {
    for _ in 0..count {
        let from: &[u8; N] = s
            .checked_add(N)
            .and_then(|e| src.get(s..e))
            .and_then(|b| b.try_into().ok())
            .ok_or(Error::UnexpectedEof)?;
        let to: &mut [u8; N] = o
            .checked_add(N)
            .and_then(|e| out.get_mut(o..e))
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| Error::invalid("selection run outside output"))?;
        *to = *from;
        s = s.saturating_add(ss);
        o = o.saturating_add(os);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_match_runs() {
        let space = Dataspace {
            version: 2,
            kind: DataspaceKind::Simple,
            dims: vec![7, 9, 11],
            max_dims: None,
            permutation: None,
        };
        let sels = [
            Selection::All,
            Selection::hyperslab(vec![
                HyperslabDim::new(1, 2, 3, 1),
                HyperslabDim::new(0, 3, 3, 2),
                HyperslabDim::new(1, 3, 3, 1),
            ]),
            Selection::from_ranges(&[2..5, 0..9, 4..5]),
            Selection::hyperslab(vec![
                HyperslabDim::range(0..7),
                HyperslabDim::new(0, 4, 3, 1),
                HyperslabDim::new(0, 2, 6, 1),
            ]),
        ];
        for sel in sels {
            let r = sel.resolve(&space).unwrap();
            for (origin, extent) in [([0, 0, 0], [7, 9, 11]), ([2, 3, 4], [4, 4, 4])] {
                let strides = row_major_strides(&extent);
                let Some(runs) = block_runs(&r, &origin, &extent, &strides) else {
                    continue;
                };
                for esize in [1usize, 3, 8] {
                    let mut want = Vec::new();
                    for_each_run(&runs, &mut |s, o, n| {
                        for k in 0..n {
                            want.push(((s + k) * esize as u64, (o + k) * esize as u64));
                        }
                        Ok(())
                    })
                    .unwrap();
                    let mut got = Vec::new();
                    for_each_group(&runs, esize, &mut |g| {
                        for k in 0..g.count {
                            for b in (0..g.len).step_by(esize) {
                                got.push((g.src + k * g.src_step + b, g.out + k * g.out_step + b));
                            }
                        }
                        Ok(())
                    })
                    .unwrap();
                    assert_eq!(got, want, "{sel:?} {origin:?} esize {esize}");
                    // And the copies agree.
                    let n: u64 = extent.iter().product();
                    let src: Vec<u8> = (0..n as usize * esize).map(|i| i as u8).collect();
                    let mut a = vec![0u8; r.count as usize * esize];
                    let mut b = a.clone();
                    for (s, o) in &want {
                        let (s, o) = (*s as usize, *o as usize);
                        a[o..o + esize].copy_from_slice(&src[s..s + esize]);
                    }
                    for_each_group(&runs, esize, &mut |g| copy_group(&src, 0, g, &mut b)).unwrap();
                    assert_eq!(a, b);
                }
            }
        }
    }

    fn simple(dims: &[u64]) -> Dataspace {
        Dataspace {
            version: 2,
            kind: DataspaceKind::Simple,
            dims: dims.to_vec(),
            max_dims: None,
            permutation: None,
        }
    }

    #[test]
    fn normalize_and_validate() {
        let ds = simple(&[10, 8]);
        let r = Selection::from_ranges(&[2..5, 0..8]).resolve(&ds).unwrap();
        assert_eq!(r.shape, vec![3, 8]);
        assert_eq!(r.count, 24);
        assert_eq!(r.dims[0], HyperslabDim::new(2, 3, 1, 3));
        // adjacent blocks merge
        let r = Selection::hyperslab(vec![
            HyperslabDim::new(1, 2, 3, 2),
            HyperslabDim::range(0..1),
        ])
        .resolve(&ds)
        .unwrap();
        assert_eq!(r.dims[0], HyperslabDim::new(1, 6, 1, 6));
        // errors
        for bad in [
            Selection::from_ranges(&[0..11, 0..8]),
            Selection::range(0..1),
            Selection::hyperslab(vec![
                HyperslabDim::new(0, 1, 2, 2),
                HyperslabDim::range(0..1),
            ]),
            Selection::hyperslab(vec![
                HyperslabDim::new(u64::MAX, 1, 1, 1),
                HyperslabDim::range(0..1),
            ]),
            Selection::hyperslab(vec![
                HyperslabDim::new(0, u64::MAX, 3, 1),
                HyperslabDim::range(0..1),
            ]),
        ] {
            assert!(bad.resolve(&ds).is_err(), "{bad:?}");
        }
        // empty selections are fine anywhere
        let r = Selection::from_ranges(&[5..5, 100..200]);
        assert!(r.resolve(&ds).is_err()); // 100..200 out of range
        let r = Selection::from_ranges(&[5..5, 1..2]).resolve(&ds).unwrap();
        assert_eq!(r.count, 0);
        assert_eq!(Selection::All.num_elements(&simple(&[3, 0])).unwrap(), 0);
    }

    #[test]
    fn scalar_and_null() {
        let mut s = simple(&[]);
        s.kind = DataspaceKind::Scalar;
        assert_eq!(Selection::All.num_elements(&s).unwrap(), 1);
        assert!(Selection::range(0..1).resolve(&s).is_err());
        s.kind = DataspaceKind::Null;
        assert_eq!(Selection::All.num_elements(&s).unwrap(), 0);
    }

    #[test]
    fn pieces() {
        let h = HyperslabDim::new(1, 4, 5, 2); // 1,2 5,6 9,10 13,14 17,18
        let mut v = Vec::new();
        h.pieces_in(4, 4, &mut v); // block 4..8
        assert_eq!(
            v,
            vec![Piece {
                src: 1,
                len: 2,
                out: 2
            }]
        );
        v.clear();
        h.pieces_in(0, 20, &mut v);
        assert_eq!(v.len(), 5);
        assert_eq!(
            v[4],
            Piece {
                src: 17,
                len: 2,
                out: 8
            }
        );
        v.clear();
        h.pieces_in(6, 4, &mut v); // 6..10: 6, 9
        assert_eq!(
            v,
            vec![
                Piece {
                    src: 0,
                    len: 1,
                    out: 3
                },
                Piece {
                    src: 3,
                    len: 1,
                    out: 4
                }
            ]
        );
        v.clear();
        h.pieces_in(19, 5, &mut v);
        assert!(v.is_empty());
        h.pieces_in(3, 2, &mut v); // 3..5: nothing
        assert!(v.is_empty());
    }

    #[test]
    fn collapse_full_rows() {
        let ds = simple(&[4, 5]);
        let sel = Selection::All.resolve(&ds).unwrap();
        let runs = block_runs(&sel, &[0, 0], &[4, 5], &[5, 1]).unwrap();
        assert_eq!(runs.len(), 1);
        let mut got = Vec::new();
        for_each_run(&runs, &mut |s, o, l| {
            got.push((s, o, l));
            Ok(())
        })
        .unwrap();
        assert_eq!(got, vec![(0, 0, 20)]);
        // a chunk 2x3 at (2,3) of a 4x5 dataset, stored 2x3 with edge clip
        let runs = block_runs(&sel, &[2, 3], &[2, 2], &[3, 1]).unwrap();
        let mut got = Vec::new();
        for_each_run(&runs, &mut |s, o, l| {
            got.push((s, o, l));
            Ok(())
        })
        .unwrap();
        assert_eq!(got, vec![(0, 13, 2), (3, 18, 2)]);
    }
}
