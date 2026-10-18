//! Decoding of reference datatype elements and dataspace selections.
//!
//! Spec: "VIII.B. Reference Encoding (Revised)", "VIII.C. Reference
//! Encoding (Backward Compatibility)" and "VIII.A. Dataspace Encoding"
//! (Dataspace Selection).

use super::DecodeContext;
use super::vlen::{fetch, parse_heap_id};
use crate::error::{Error, Result};
use crate::format::ByteReader;
use crate::format::messages::datatype::{Datatype, DatatypeClass, ReferenceKind};

/// Highest dataspace rank accepted in selections.
const MAX_RANK: u32 = 32;

/// A decoded reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reference {
    /// A null (unset) reference.
    Null,
    /// An object reference (`H5R_OBJECT1`): the object header address.
    Object(u64),
    /// A dataset region reference (`H5R_DATASET_REGION1`).
    Region(RegionReference),
    /// A revised reference (`H5R_OBJECT2`, `H5R_DATASET_REGION2`,
    /// `H5R_ATTR`; datatype version 4).
    Revised(RevisedReference),
}

impl Reference {
    /// The address of the referenced object header, when it is in this
    /// file and known.
    pub fn object_address(&self) -> Option<u64> {
        match self {
            Reference::Null => None,
            Reference::Object(a) => Some(*a),
            Reference::Region(r) => Some(r.object),
            Reference::Revised(r) if r.external_file.is_none() => r.object_address(),
            Reference::Revised(_) => None,
        }
    }
}

/// A dataset region reference: an object plus a selection of its
/// dataspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionReference {
    /// Address of the dataset's object header.
    pub object: u64,
    /// The selected region.
    pub selection: Selection,
}

/// A revised (version 4 datatype) reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevisedReference {
    /// Kind, as recorded in the element's reference header (which may be
    /// more specific than the datatype's).
    pub kind: ReferenceKind,
    /// The object token (for the native file format, the object header
    /// address in little-endian order).
    pub token: Vec<u8>,
    /// Name of the external file holding the object, if any.
    pub external_file: Option<String>,
    /// Selection, for dataset region references.
    pub selection: Option<Selection>,
    /// Attribute name, for attribute references.
    pub attribute: Option<String>,
}

impl RevisedReference {
    /// The token interpreted as a little-endian object address (tokens of
    /// at most 8 bytes).
    pub fn object_address(&self) -> Option<u64> {
        if self.token.len() > 8 || self.token.is_empty() {
            return None;
        }
        let mut b = [0u8; 8];
        b[..self.token.len()].copy_from_slice(&self.token);
        Some(u64::from_le_bytes(b))
    }
}

/// One block of an irregular hyperslab selection (inclusive corners).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperslabBlock {
    /// Coordinates of the first element.
    pub start: Vec<u64>,
    /// Coordinates of the last element (inclusive).
    pub end: Vec<u64>,
}

/// A dataspace selection (spec "Dataspace Selection").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// Nothing selected.
    None,
    /// The entire extent.
    All,
    /// A list of points.
    Points {
        /// Dataspace rank.
        rank: u32,
        /// Point coordinates, `rank` values each.
        points: Vec<Vec<u64>>,
    },
    /// A hyperslab given as a list of blocks.
    Hyperslab {
        /// Dataspace rank.
        rank: u32,
        /// The blocks.
        blocks: Vec<HyperslabBlock>,
    },
    /// A regular hyperslab (start / stride / count / block per dimension).
    RegularHyperslab {
        /// Start offsets.
        start: Vec<u64>,
        /// Strides.
        stride: Vec<u64>,
        /// Block counts.
        count: Vec<u64>,
        /// Block sizes.
        block: Vec<u64>,
    },
}

impl Selection {
    /// Parse an encoded selection (starting at the 4-byte selection type).
    /// Trailing bytes are ignored.
    pub fn parse(data: &[u8]) -> Result<Selection> {
        let mut r = ByteReader::new(data);
        let sel_type = r.read_u32()?;
        let version = r.read_u32()?;
        match sel_type {
            0 | 3 => {
                if version != 1 {
                    return Err(Error::unsupported(format!(
                        "selection info version {version}"
                    )));
                }
                Ok(if sel_type == 0 {
                    Selection::None
                } else {
                    Selection::All
                })
            }
            1 => parse_points(&mut r, version),
            2 => parse_hyperslab(&mut r, version),
            other => Err(Error::invalid(format!("unknown selection type {other}"))),
        }
    }
}

fn read_rank(r: &mut ByteReader<'_>) -> Result<u32> {
    let rank = r.read_u32()?;
    if rank == 0 || rank > MAX_RANK {
        return Err(Error::invalid(format!("selection rank {rank}")));
    }
    Ok(rank)
}

fn read_enc_size(r: &mut ByteReader<'_>) -> Result<usize> {
    match r.read_u8()? {
        n @ (2 | 4 | 8) => Ok(n as usize),
        other => Err(Error::invalid(format!("selection encode size {other}"))),
    }
}

/// Read `count` groups of `per` values of `width` bytes, checking the
/// total against the remaining data before allocating.
fn read_coords(
    r: &mut ByteReader<'_>,
    count: u64,
    per: usize,
    width: usize,
) -> Result<Vec<Vec<u64>>> {
    let need = count
        .checked_mul((per * width) as u64)
        .filter(|&n| n <= r.remaining() as u64)
        .ok_or(Error::UnexpectedEof)?;
    let mut out = Vec::with_capacity((need / (per * width).max(1) as u64) as usize);
    for _ in 0..count {
        let mut v = Vec::with_capacity(per);
        for _ in 0..per {
            v.push(r.read_uint(width)?);
        }
        out.push(v);
    }
    Ok(out)
}

fn parse_points(r: &mut ByteReader<'_>, version: u32) -> Result<Selection> {
    let (rank, count, width) = match version {
        1 => {
            r.skip(4)?; // reserved
            let _length = r.read_u32()?;
            let rank = read_rank(r)?;
            (rank, u64::from(r.read_u32()?), 4)
        }
        2 => {
            let width = read_enc_size(r)?;
            let rank = read_rank(r)?;
            (rank, r.read_uint(width)?, width)
        }
        v => return Err(Error::unsupported(format!("point selection version {v}"))),
    };
    let points = read_coords(r, count, rank as usize, width)?;
    Ok(Selection::Points { rank, points })
}

fn blocks(r: &mut ByteReader<'_>, rank: u32, count: u64, width: usize) -> Result<Selection> {
    let corners = count.checked_mul(2).ok_or(Error::UnexpectedEof)?;
    let mut flat = read_coords(r, corners, rank as usize, width)?.into_iter();
    let mut blocks = Vec::with_capacity(flat.len() / 2);
    while let (Some(start), Some(end)) = (flat.next(), flat.next()) {
        blocks.push(HyperslabBlock { start, end });
    }
    Ok(Selection::Hyperslab { rank, blocks })
}

fn regular(r: &mut ByteReader<'_>, rank: u32, width: usize) -> Result<Selection> {
    let rows = read_coords(r, u64::from(rank), 4, width)?;
    let col = |i: usize| rows.iter().map(|row| row[i]).collect::<Vec<u64>>();
    Ok(Selection::RegularHyperslab {
        start: col(0),
        stride: col(1),
        count: col(2),
        block: col(3),
    })
}

fn parse_hyperslab(r: &mut ByteReader<'_>, version: u32) -> Result<Selection> {
    match version {
        1 => {
            r.skip(4)?; // reserved
            let _length = r.read_u32()?;
            let rank = read_rank(r)?;
            let count = u64::from(r.read_u32()?);
            blocks(r, rank, count, 4)
        }
        2 => {
            let flags = r.read_u8()?;
            let _length = r.read_u32()?;
            let rank = read_rank(r)?;
            if flags & 1 == 0 {
                return Err(Error::unsupported("irregular version 2 hyperslab"));
            }
            regular(r, rank, 8)
        }
        3 => {
            let flags = r.read_u8()?;
            let width = read_enc_size(r)?;
            let rank = read_rank(r)?;
            if flags & 1 != 0 {
                regular(r, rank, width)
            } else {
                let count = r.read_uint(width)?;
                blocks(r, rank, count, width)
            }
        }
        v => Err(Error::unsupported(format!(
            "hyperslab selection version {v}"
        ))),
    }
}

/// Read a little-endian unsigned integer of `bytes.len()` (≤ 8) bytes.
fn le_uint(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .rev()
        .fold(0u64, |acc, &b| (acc << 8) | u64::from(b))
}

/// Decode one reference element.
pub(crate) fn decode_reference(
    dt: &Datatype,
    bytes: &[u8],
    ctx: &DecodeContext<'_>,
) -> Result<Reference> {
    let DatatypeClass::Reference(rt) = &dt.class else {
        return Err(Error::invalid("not a reference datatype"));
    };
    if bytes.len() != dt.size as usize {
        return Err(Error::invalid("reference element size mismatch"));
    }
    match rt.kind {
        ReferenceKind::Object => {
            if bytes.is_empty() || bytes.len() > 8 {
                return Err(Error::invalid(format!(
                    "object reference of {} bytes",
                    bytes.len()
                )));
            }
            if bytes.iter().all(|&b| b == 0) || bytes.iter().all(|&b| b == 0xff) {
                return Ok(Reference::Null);
            }
            Ok(Reference::Object(le_uint(bytes)))
        }
        ReferenceKind::DatasetRegion => {
            let id = parse_heap_id(bytes, ctx.sizes)?;
            if id.is_null() {
                return Ok(Reference::Null);
            }
            let data = fetch(ctx, &id)?;
            let mut r = ByteReader::with_sizes(&data, ctx.sizes);
            let object = r.read_offset()?;
            let selection = Selection::parse(r.rest())?;
            Ok(Reference::Region(RegionReference { object, selection }))
        }
        ReferenceKind::Object2 | ReferenceKind::DatasetRegion2 | ReferenceKind::Attribute => {
            decode_revised(bytes, ctx)
        }
    }
}

fn decode_revised(bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Reference> {
    if bytes.iter().all(|&b| b == 0) {
        return Ok(Reference::Null);
    }
    let mut r = ByteReader::with_sizes(bytes, ctx.sizes);
    let kind = match r.read_u8()? {
        2 => ReferenceKind::Object2,
        3 => ReferenceKind::DatasetRegion2,
        4 => ReferenceKind::Attribute,
        other => return Err(Error::invalid(format!("revised reference type {other}"))),
    };
    let flags = r.read_u8()?;
    let external = flags & 1 != 0;
    if kind == ReferenceKind::Object2 && !external {
        // The reference block is stored inline.
        return parse_block(&mut r, kind, false);
    }
    // Otherwise: block size, then a global heap ID locating the block.
    let len = r.read_u32()? as usize;
    let id_len = crate::format::global_heap::GlobalHeapId::encoded_size(ctx.sizes.offset);
    let id = parse_heap_id(r.read_bytes(id_len)?, ctx.sizes)?;
    if id.is_null() {
        return Err(Error::invalid("revised reference with a null heap ID"));
    }
    let data = fetch(ctx, &id)?;
    let block = data
        .get(..len)
        .ok_or_else(|| Error::invalid("revised reference block truncated"))?;
    parse_block(
        &mut ByteReader::with_sizes(block, ctx.sizes),
        kind,
        external,
    )
}

fn parse_block(r: &mut ByteReader<'_>, kind: ReferenceKind, external: bool) -> Result<Reference> {
    let token_len = r.read_u8()? as usize;
    let token = r.read_bytes(token_len)?.to_vec();
    let name = |r: &mut ByteReader<'_>| -> Result<String> {
        let len = r.read_u16()? as usize;
        let raw = r.read_bytes(len)?;
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        Ok(String::from_utf8_lossy(&raw[..end]).into_owned())
    };
    let external_file = if external { Some(name(r)?) } else { None };
    let selection = if kind == ReferenceKind::DatasetRegion2 {
        let size = r.read_u32()? as usize;
        let _rank = r.read_u32()?;
        Some(Selection::parse(r.read_bytes(size)?)?)
    } else {
        None
    };
    let attribute = if kind == ReferenceKind::Attribute {
        Some(name(r)?)
    } else {
        None
    };
    Ok(Reference::Revised(RevisedReference {
        kind,
        token,
        external_file,
        selection,
        attribute,
    }))
}
