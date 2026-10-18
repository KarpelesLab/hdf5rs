//! Virtual dataset reading: assembling the selected elements from the
//! mapped source datasets.
//!
//! Each mapping pairs a selection of a source dataset with a selection of
//! the virtual dataset holding the same number of elements; the `k`-th
//! selected source element (row-major order of the source selection) goes
//! to the `k`-th selected virtual element. Elements not covered by any
//! mapping (or whose source is unavailable) read as the fill value. Later
//! mappings overwrite earlier ones where they overlap.

use super::selection::{Piece, Resolved};
use super::{DatasetStorage, Selection, filled};
use crate::error::{Error, Result};
use crate::format::messages::dataspace::DataspaceKind;
use crate::format::vds::VirtualMapping;

/// Read `selection` of a virtual dataset described by `storage` (dataspace,
/// element size, fill value, limits) and its `mappings`.
///
/// `read_source(mapping, source_selection)` returns the raw bytes of the
/// source selection of a mapping, or `None` when the source is unavailable
/// (its region then reads as the fill value).
pub(crate) fn read_virtual<F>(
    storage: &DatasetStorage<'_>,
    mappings: &[VirtualMapping],
    selection: &Selection,
    mut read_source: F,
) -> Result<Vec<u8>>
where
    F: FnMut(&VirtualMapping, &Selection) -> Result<Option<Vec<u8>>>,
{
    let space = storage.dataspace;
    if space.kind != DataspaceKind::Simple || space.rank() == 0 {
        return Err(Error::unsupported(
            "virtual dataset with a scalar or null dataspace",
        ));
    }
    let esize = storage.element_size;
    if esize == 0 {
        return Err(Error::invalid("element size is zero"));
    }
    let user = selection.resolve(space)?;
    let out_len = usize::try_from(user.count)
        .ok()
        .and_then(|n| n.checked_mul(esize))
        .filter(|&n| n <= storage.limits.max_output_bytes)
        .ok_or(Error::LimitExceeded("selection output size"))?;
    let mut out = filled(out_len, storage.fill_bytes()?);
    if out_len == 0 {
        return Ok(out);
    }
    for m in mappings {
        let Some(vsel) = m.virtual_selection.to_read_selection()? else {
            continue;
        };
        reject_unlimited(&vsel)?;
        let v = vsel.resolve(space)?;
        if v.count == 0 || !overlaps(&v, &user) {
            continue;
        }
        let Some(ssel) = m.source_selection.to_read_selection()? else {
            continue;
        };
        reject_unlimited(&ssel)?;
        let Some(src) = read_source(m, &ssel)? else {
            continue;
        };
        let want = usize::try_from(v.count)
            .ok()
            .and_then(|n| n.checked_mul(esize));
        if want != Some(src.len()) {
            return Err(Error::invalid(format!(
                "virtual mapping of {}:{} selects {} source bytes for {} virtual elements",
                m.source_file,
                m.source_dataset,
                src.len(),
                v.count
            )));
        }
        scatter(&v, &user, &src, esize, &mut out)?;
    }
    Ok(out)
}

/// Unlimited (printf-style growing) mappings are not supported.
fn reject_unlimited(sel: &Selection) -> Result<()> {
    if let Selection::Hyperslab(dims) = sel
        && dims
            .iter()
            .any(|h| h.count == u64::MAX || h.block == u64::MAX)
    {
        return Err(Error::unsupported("unlimited virtual dataset mappings"));
    }
    Ok(())
}

/// Whether the bounding boxes of two selections of the same dataspace
/// intersect.
fn overlaps(a: &Resolved, b: &Resolved) -> bool {
    a.dims
        .iter()
        .zip(&b.dims)
        .all(|(x, y)| x.start < y.end() && y.start < x.end())
}

/// Copy the elements of `src` (packed in the order of virtual selection
/// `v`) that fall in the user selection `user` into `out`.
fn scatter(v: &Resolved, user: &Resolved, src: &[u8], esize: usize, out: &mut [u8]) -> Result<()> {
    let rank = v.dims.len();
    let last = rank - 1;
    let out_strides = user.out_strides();
    let (vl, ul) = (v.dims[last], user.dims[last]);
    let row_len = v.shape[last];
    let mut idx = vec![0u64; last];
    let mut row = 0u64;
    let mut pieces: Vec<Piece> = Vec::new();
    let oob = || Error::invalid("virtual mapping copy out of bounds");
    loop {
        // Output offset of this row, if every outer coordinate is selected.
        let mut base = Some(0u64);
        for d in 0..last {
            let c = v.dims[d].coord(idx[d]);
            base = match (base, user.dims[d].index_of(c)) {
                (Some(b), Some(i)) => i.checked_mul(out_strides[d]).and_then(|x| x.checked_add(b)),
                _ => None,
            };
            if base.is_none() {
                break;
            }
        }
        if let Some(base) = base {
            let k_row = row.checked_mul(row_len).ok_or_else(oob)?;
            for j in 0..vl.count {
                let bs = vl.start + j * vl.stride;
                pieces.clear();
                ul.pieces_in(bs, vl.block, &mut pieces);
                for p in &pieces {
                    let k = (k_row + j * vl.block + p.src) as usize;
                    let o = (base + p.out) as usize;
                    let n = p.len as usize;
                    let from = src.get(k * esize..(k + n) * esize).ok_or_else(oob)?;
                    out.get_mut(o * esize..(o + n) * esize)
                        .ok_or_else(oob)?
                        .copy_from_slice(from);
                }
            }
        }
        // Advance the odometer over the outer dimensions.
        row += 1;
        let mut d = last;
        loop {
            if d == 0 {
                return Ok(());
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < v.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}
