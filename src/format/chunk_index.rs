//! Shared pieces of the version 4 Data Layout chunk indexes (spec: "Appendix
//! C: Types of Indexes for Dataset Chunks").
//!
//! This module holds:
//!
//! * [`ChunkRecord`], the per-chunk element shared by the Fixed Array and
//!   Extensible Array indexes, and [`ChunkIndexContext`], the decoding
//!   parameters those indexes need;
//! * pure chunk-grid math: chunks per dimension, scaled offsets and the
//!   linear chunk index used by the Implicit, Fixed Array and Extensible
//!   Array indexes;
//! * the Implicit index address computation and the Single Chunk index
//!   representation, which need no on-disk structure beyond the layout
//!   message itself.
//!
//! # Chunk dimensions
//!
//! All functions here take dimension lists of the *dataspace rank*. The
//! chunked layout message stores dimension sizes in units of elements; when
//! it carries an extra trailing "dimension" for the datatype size (as layout
//! versions 1–3 do), the caller must drop it first.
//!
//! # Which dimensions to linearize against
//!
//! The Implicit index section of the spec computes `nchunks` from the
//! *current* dimensions, but it also states that the chunk array is
//! allocated from the *maximum* dimension sizes, and the Fixed Array is
//! sized from "the maximum number of chunks". A linear index that changed
//! whenever the dataset is extended would be useless for a pre-allocated
//! array, so callers should pass the maximum dimensions for these indexes;
//! the two coincide whenever the dataset has never been resized.
//!
//! For the Extensible Array (exactly one unlimited dimension) see
//! [`extensible_linear_index`].

use super::{ByteReader, Sizes, UNDEFINED_ADDRESS};
use crate::checksum::metadata_checksum;
use crate::error::{Error, Result};

/// Number of records an index may synthesize without on-disk backing (for
/// unallocated blocks), unless the file itself is larger than this many
/// bytes. Guards against absurd element counts in malformed headers.
pub const MAX_SYNTHETIC_RECORDS: u64 = 1 << 20;

/// One chunk entry of a chunk index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRecord {
    /// File address of the chunk, or [`UNDEFINED_ADDRESS`] when the chunk has
    /// not been allocated (it then reads as the fill value).
    pub address: u64,
    /// Stored (filtered) size of the chunk in bytes. `None` for indexes of
    /// non-filtered datasets, where every chunk occupies exactly the nominal
    /// chunk byte size.
    pub size: Option<u64>,
    /// Filter mask: bit `n` set means filter `n` of the pipeline was skipped
    /// for this chunk. Always 0 for non-filtered datasets.
    pub filter_mask: u32,
}

impl ChunkRecord {
    /// A record for an unallocated chunk.
    pub const UNALLOCATED: ChunkRecord = ChunkRecord {
        address: UNDEFINED_ADDRESS,
        size: None,
        filter_mask: 0,
    };

    /// Whether storage has been allocated for this chunk.
    pub fn is_allocated(&self) -> bool {
        self.address != UNDEFINED_ADDRESS
    }
}

/// Decoding parameters for Fixed Array / Extensible Array chunk indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkIndexContext {
    /// Offset/length widths from the superblock.
    pub sizes: Sizes,
    /// Whether the dataset has a filter pipeline (filtered chunk elements
    /// carry a size and a filter mask). Checked against the structure's
    /// client ID.
    pub filtered: bool,
    /// Width in bytes (1..=8) of the "Chunk Size" field of filtered
    /// elements, if known. `None` derives it from the structure's element
    /// size (`element size - size of offsets - 4`); `Some` additionally
    /// checks the structure against it. Ignored for non-filtered datasets.
    pub chunk_size_width: Option<u8>,
}

impl ChunkIndexContext {
    /// Context for a dataset without filters.
    pub fn unfiltered(sizes: Sizes) -> Self {
        ChunkIndexContext {
            sizes,
            filtered: false,
            chunk_size_width: None,
        }
    }

    /// Context for a filtered dataset, deriving the chunk size width from
    /// the on-disk element size.
    pub fn filtered(sizes: Sizes) -> Self {
        ChunkIndexContext {
            sizes,
            filtered: true,
            chunk_size_width: None,
        }
    }
}

/// Decoder for one array element ("Data Block Element for (Non-)filtered
/// Dataset Chunk"), shared by the Fixed and Extensible Array code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ElementCodec {
    pub(crate) sizes: Sizes,
    pub(crate) filtered: bool,
    pub(crate) size_width: u8,
    pub(crate) element_size: usize,
}

impl ElementCodec {
    /// Validate a structure's client ID and element size against `ctx`.
    pub(crate) fn new(ctx: &ChunkIndexContext, client_id: u8, element_size: u8) -> Result<Self> {
        let filtered = match client_id {
            0 => false,
            1 => true,
            other => {
                return Err(Error::invalid(format!(
                    "unknown chunk index client ID {other}"
                )));
            }
        };
        if filtered != ctx.filtered {
            return Err(Error::invalid(
                "chunk index client ID does not match dataset filter pipeline",
            ));
        }
        let offset = ctx.sizes.offset;
        let size_width = if filtered {
            let w = element_size
                .checked_sub(offset)
                .and_then(|v| v.checked_sub(4))
                .filter(|w| (1..=8).contains(w))
                .ok_or_else(|| Error::invalid("bad filtered chunk index element size"))?;
            if let Some(expected) = ctx.chunk_size_width
                && expected != w
            {
                return Err(Error::invalid(
                    "chunk index element size disagrees with chunk size width",
                ));
            }
            w
        } else {
            if element_size != offset {
                return Err(Error::invalid(
                    "non-filtered chunk index element size must equal size of offsets",
                ));
            }
            0
        };
        Ok(ElementCodec {
            sizes: ctx.sizes,
            filtered,
            size_width,
            element_size: usize::from(element_size),
        })
    }

    /// Decode one element. Elements with an undefined address decode as
    /// [`ChunkRecord::UNALLOCATED`].
    pub(crate) fn decode(&self, r: &mut ByteReader<'_>) -> Result<ChunkRecord> {
        let address = r.read_offset()?;
        if address == UNDEFINED_ADDRESS {
            // Skip the rest of the element: an unallocated chunk has no
            // meaningful size or filter mask.
            r.skip(self.element_size - usize::from(self.sizes.offset))?;
            return Ok(ChunkRecord::UNALLOCATED);
        }
        if !self.filtered {
            return Ok(ChunkRecord {
                address,
                size: None,
                filter_mask: 0,
            });
        }
        let size = r.read_uint(usize::from(self.size_width))?;
        let filter_mask = r.read_u32()?;
        Ok(ChunkRecord {
            address,
            size: Some(size),
            filter_mask,
        })
    }

    /// Decode `n` consecutive elements from `bytes`, appending to `out`.
    pub(crate) fn decode_many(
        &self,
        bytes: &[u8],
        n: usize,
        out: &mut Vec<ChunkRecord>,
    ) -> Result<()> {
        let mut r = ByteReader::with_sizes(bytes, self.sizes);
        for _ in 0..n {
            out.push(self.decode(&mut r)?);
        }
        Ok(())
    }
}

/// Verify a trailing 4-byte metadata checksum over everything before it.
pub(crate) fn verify_checksum(buf: &[u8], what: &'static str) -> Result<()> {
    let split = buf.len().checked_sub(4).ok_or(Error::UnexpectedEof)?;
    let (body, tail) = buf.split_at(split);
    let expected = ByteReader::new(tail).read_u32()?;
    let actual = metadata_checksum(body);
    if expected != actual {
        return Err(Error::ChecksumMismatch {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Maximum number of records that may be materialized for a store: the
/// larger of [`MAX_SYNTHETIC_RECORDS`] and the store size in bytes.
pub(crate) fn record_limit(store_size: u64) -> u64 {
    store_size.max(MAX_SYNTHETIC_RECORDS)
}

/// Convert a file-derived count to `usize`, enforcing `limit`.
pub(crate) fn checked_count(n: u64, limit: u64, what: &'static str) -> Result<usize> {
    if n > limit {
        return Err(Error::LimitExceeded(what));
    }
    usize::try_from(n).map_err(|_| Error::LimitExceeded(what))
}

/// Whether bit `i` of a page-initialization bitmap is set.
///
/// The spec does not state the bit order of these bitmaps. We use
/// most-significant-bit-first within each byte (page 0 is bit 7 of byte 0).
/// Bits beyond the bitmap read as clear.
pub(crate) fn bitmap_bit(bitmap: &[u8], i: u64) -> bool {
    let Ok(byte) = usize::try_from(i / 8) else {
        return false;
    };
    bitmap
        .get(byte)
        .is_some_and(|b| b & (0x80u8 >> (i % 8)) != 0)
}

fn check_rank(a: &[u64], b: &[u64]) -> Result<()> {
    if a.len() != b.len() {
        return Err(Error::invalid("chunk index dimension count mismatch"));
    }
    Ok(())
}

/// Number of chunks along each dimension: `ceil(dims[i] / chunk_dims[i])`.
///
/// A zero chunk dimension is invalid; a zero dataset dimension yields zero
/// chunks.
pub fn chunks_per_dim(dims: &[u64], chunk_dims: &[u64]) -> Result<Vec<u64>> {
    check_rank(dims, chunk_dims)?;
    dims.iter()
        .zip(chunk_dims)
        .map(|(&d, &c)| {
            if c == 0 {
                return Err(Error::invalid("zero chunk dimension"));
            }
            Ok(d.div_ceil(c))
        })
        .collect()
}

/// Total number of chunks in a grid (product of `nchunks`; 1 for rank 0).
pub fn total_chunks(nchunks: &[u64]) -> Result<u64> {
    nchunks.iter().try_fold(1u64, |acc, &n| {
        acc.checked_mul(n)
            .ok_or(Error::LimitExceeded("chunk count overflows u64"))
    })
}

/// Scaled offset of a chunk: element offset divided by the chunk size in
/// each dimension. Offsets need not be chunk-aligned (any element inside the
/// chunk maps to it).
pub fn scaled_offset(offset: &[u64], chunk_dims: &[u64]) -> Result<Vec<u64>> {
    check_rank(offset, chunk_dims)?;
    offset
        .iter()
        .zip(chunk_dims)
        .map(|(&o, &c)| {
            o.checked_div(c)
                .ok_or_else(|| Error::invalid("zero chunk dimension"))
        })
        .collect()
}

/// "Down chunks": the row-major stride (in chunks) of each dimension, the
/// last (fastest-changing) dimension having stride 1.
pub fn down_chunks(nchunks: &[u64]) -> Result<Vec<u64>> {
    let mut out = vec![0u64; nchunks.len()];
    let mut acc = 1u64;
    for (slot, &n) in out.iter_mut().zip(nchunks).rev() {
        *slot = acc;
        acc = acc
            .checked_mul(n)
            .ok_or(Error::LimitExceeded("chunk count overflows u64"))?;
    }
    Ok(out)
}

/// Linear (row-major) chunk index of the chunk at `scaled` in a grid of
/// `nchunks` chunks per dimension, as defined in the Implicit index section
/// and used by the Implicit and Fixed Array indexes.
///
/// Errors if a coordinate lies outside the grid.
pub fn linear_index(scaled: &[u64], nchunks: &[u64]) -> Result<u64> {
    check_rank(scaled, nchunks)?;
    let mut index = 0u64;
    for (&s, &n) in scaled.iter().zip(nchunks) {
        if s >= n {
            return Err(Error::invalid("chunk coordinate outside chunk grid"));
        }
        // Horner form of sum(down_chunks[i] * scaled[i]).
        index = index
            .checked_mul(n)
            .and_then(|v| v.checked_add(s))
            .ok_or(Error::LimitExceeded("chunk index overflows u64"))?;
    }
    Ok(index)
}

/// Inverse of [`linear_index`]: chunk grid coordinates of a linear index.
pub fn unlinearize(mut index: u64, nchunks: &[u64]) -> Result<Vec<u64>> {
    let mut out = vec![0u64; nchunks.len()];
    for (slot, &n) in out.iter_mut().zip(nchunks).rev() {
        if n == 0 {
            return Err(Error::invalid("chunk index outside chunk grid"));
        }
        *slot = index % n;
        index /= n;
    }
    if index != 0 {
        return Err(Error::invalid("chunk index outside chunk grid"));
    }
    Ok(out)
}

/// Linear chunk index for an Extensible Array index (dataset with exactly
/// one unlimited dimension, `unlimited_dim`).
///
/// The spec does not spell this mapping out. An extensible array only grows
/// at its end, so the index must stay stable when the unlimited dimension is
/// extended: that dimension therefore has to be the *slowest-changing* one.
/// We linearize row-major over the permuted coordinate list
/// `[unlimited, d0, d1, … (remaining dims in their original order)]`, using
/// `nchunks` (computed from the **maximum** dimensions) for the fixed
/// dimensions; the unlimited dimension's entry in `nchunks` is ignored.
/// When the unlimited dimension is already dimension 0 this is exactly
/// [`linear_index`].
///
/// This is an inference from the structure's design; verify against real
/// files.
pub fn extensible_linear_index(
    scaled: &[u64],
    nchunks: &[u64],
    unlimited_dim: usize,
) -> Result<u64> {
    check_rank(scaled, nchunks)?;
    let unlim = *scaled
        .get(unlimited_dim)
        .ok_or_else(|| Error::invalid("unlimited dimension out of range"))?;
    let mut index = unlim;
    for (i, (&s, &n)) in scaled.iter().zip(nchunks).enumerate() {
        if i == unlimited_dim {
            continue;
        }
        if s >= n {
            return Err(Error::invalid("chunk coordinate outside chunk grid"));
        }
        index = index
            .checked_mul(n)
            .and_then(|v| v.checked_add(s))
            .ok_or(Error::LimitExceeded("chunk index overflows u64"))?;
    }
    Ok(index)
}

/// Address of a chunk in an Implicit index: `base + linear_index *
/// chunk_bytes` (spec: "The Implicit Index").
pub fn implicit_chunk_address(base: u64, linear_index: u64, chunk_bytes: u64) -> Result<u64> {
    if base == UNDEFINED_ADDRESS {
        return Ok(UNDEFINED_ADDRESS);
    }
    linear_index
        .checked_mul(chunk_bytes)
        .and_then(|off| base.checked_add(off))
        .filter(|&a| a != UNDEFINED_ADDRESS)
        .ok_or(Error::LimitExceeded("implicit chunk address overflows"))
}

/// An Implicit chunk index: all chunks of a non-filtered, early-allocated
/// dataset stored contiguously in linear chunk index order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImplicitIndex {
    /// Address of the chunk array (from the layout message); may be
    /// undefined if storage is not allocated.
    pub address: u64,
    /// Size in bytes of one (unfiltered) chunk.
    pub chunk_bytes: u64,
    /// Number of chunks in the array (normally from the maximum dims).
    pub num_chunks: u64,
}

impl ImplicitIndex {
    /// The record of chunk `index`.
    pub fn record(&self, index: u64) -> Result<ChunkRecord> {
        if index >= self.num_chunks {
            return Err(Error::invalid("implicit chunk index out of range"));
        }
        Ok(ChunkRecord {
            address: implicit_chunk_address(self.address, index, self.chunk_bytes)?,
            size: None,
            filter_mask: 0,
        })
    }

    /// All records, in linear chunk index order. Validates that the last
    /// chunk's address computation does not overflow.
    pub fn records(&self) -> Result<Vec<ChunkRecord>> {
        checked_count(
            self.num_chunks,
            MAX_SYNTHETIC_RECORDS,
            "too many implicit index chunks",
        )?;
        if let Some(last) = self.num_chunks.checked_sub(1) {
            self.record(last)?;
        }
        (0..self.num_chunks).map(|i| self.record(i)).collect()
    }
}

/// Flag bit of the chunked layout (v4) "Flags" field marking a filtered
/// single chunk (`SINGLE_INDEX_WITH_FILTER`).
pub const SINGLE_INDEX_WITH_FILTER: u8 = 0x02;

/// A Single Chunk index: the dataset's only chunk, stored at the address
/// given in the layout message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingleChunkIndex {
    /// The chunk's record (`size` is `Some` for a filtered chunk).
    pub record: ChunkRecord,
}

impl SingleChunkIndex {
    /// Parse the Single Chunk "Indexing Type Information" from `r` (which
    /// must be positioned on it) and combine it with the layout message's
    /// trailing `address`.
    ///
    /// The information ("Size of filtered chunk" as a length, then the
    /// 4-byte "Filters for chunk" mask) is present only for a filtered
    /// chunk. The spec's prose names bit 0 (`DONT_FILTER_PARTIAL_BOUND_CHUNKS`)
    /// here, but its flag table defines bit 1 (`SINGLE_INDEX_WITH_FILTER`)
    /// as "a filtered chunk for Single Chunk indexing"; we follow the table.
    pub fn parse_info(r: &mut ByteReader<'_>, flags: u8) -> Result<Option<(u64, u32)>> {
        if flags & SINGLE_INDEX_WITH_FILTER == 0 {
            return Ok(None);
        }
        let size = r.read_length()?;
        let mask = r.read_u32()?;
        Ok(Some((size, mask)))
    }

    /// Build the index from the layout message's address and the parsed
    /// filtered-chunk information.
    pub fn new(address: u64, filtered: Option<(u64, u32)>) -> Self {
        let (size, filter_mask) = match filtered {
            Some((s, m)) => (Some(s), m),
            None => (None, 0),
        };
        SingleChunkIndex {
            record: ChunkRecord {
                address,
                size,
                filter_mask,
            },
        }
    }
}

/// Helpers for building index structures byte by byte in tests.
#[cfg(test)]
pub(crate) mod testutil {
    use crate::checksum::metadata_checksum;

    /// Append `v` as a `width`-byte little-endian integer.
    pub fn put(buf: &mut Vec<u8>, v: u64, width: usize) {
        buf.extend_from_slice(&v.to_le_bytes()[..width.min(8)]);
        buf.extend(std::iter::repeat_n(0u8, width.saturating_sub(8)));
    }

    /// Append an address of `width` bytes (`u64::MAX` = undefined).
    pub fn put_addr(buf: &mut Vec<u8>, v: u64, width: usize) {
        if v == u64::MAX {
            buf.extend(std::iter::repeat_n(0xffu8, width));
        } else {
            put(buf, v, width);
        }
    }

    /// Append the checksum of everything from `start` to the end of `buf`.
    pub fn seal(buf: &mut Vec<u8>, start: usize) {
        let c = metadata_checksum(&buf[start..]);
        buf.extend_from_slice(&c.to_le_bytes());
    }

    /// Set bit `i` (MSB first) of `bitmap`.
    pub fn set_bit(bitmap: &mut [u8], i: usize) {
        bitmap[i / 8] |= 0x80 >> (i % 8);
    }

    /// Tiny xorshift PRNG for mutation tests.
    pub struct Rng(pub u64);

    impl Rng {
        /// Next pseudo-random value.
        pub fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Pseudo-random value in `0..n` (`n > 0`).
        pub fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_example() {
        // Implicit index example: dims [4,5], chunks [3,2], offset [3,4].
        let n = chunks_per_dim(&[4, 5], &[3, 2]).unwrap();
        assert_eq!(n, vec![2, 3]);
        assert_eq!(total_chunks(&n).unwrap(), 6);
        assert_eq!(down_chunks(&n).unwrap(), vec![3, 1]);
        let s = scaled_offset(&[3, 4], &[3, 2]).unwrap();
        assert_eq!(s, vec![1, 2]);
        assert_eq!(linear_index(&s, &n).unwrap(), 5);
        assert_eq!(unlinearize(5, &n).unwrap(), vec![1, 2]);
        assert!(unlinearize(6, &n).is_err());
        assert!(linear_index(&[2, 0], &n).is_err());
    }

    #[test]
    fn grid_edge_cases() {
        assert!(chunks_per_dim(&[4], &[0]).is_err());
        assert!(chunks_per_dim(&[4], &[1, 2]).is_err());
        assert_eq!(chunks_per_dim(&[0, 7], &[3, 7]).unwrap(), vec![0, 1]);
        assert_eq!(total_chunks(&[]).unwrap(), 1);
        assert_eq!(linear_index(&[], &[]).unwrap(), 0);
        assert!(total_chunks(&[u64::MAX, 2]).is_err());
        assert!(chunks_per_dim(&[u64::MAX], &[1]).is_ok());
        assert!(linear_index(&[1, 1], &[u64::MAX, u64::MAX]).is_err());
        assert!(scaled_offset(&[5], &[0]).is_err());
    }

    #[test]
    fn extensible_index_permutes_unlimited_dim() {
        // Unlimited dim first: identical to row-major.
        let n = [999, 3, 4];
        for s in [[0, 0, 0], [5, 2, 3], [1, 0, 2]] {
            assert_eq!(
                extensible_linear_index(&s, &n, 0).unwrap(),
                linear_index(&s, &[u64::MAX, 3, 4]).unwrap()
            );
        }
        // Unlimited dim last: it becomes the slowest-changing.
        let n = [3, 4, 0];
        assert_eq!(extensible_linear_index(&[0, 0, 0], &n, 2).unwrap(), 0);
        assert_eq!(extensible_linear_index(&[0, 1, 0], &n, 2).unwrap(), 1);
        assert_eq!(extensible_linear_index(&[1, 0, 0], &n, 2).unwrap(), 4);
        assert_eq!(extensible_linear_index(&[0, 0, 1], &n, 2).unwrap(), 12);
        assert_eq!(
            extensible_linear_index(&[2, 3, 7], &n, 2).unwrap(),
            7 * 12 + 11
        );
        assert!(extensible_linear_index(&[3, 0, 0], &n, 2).is_err());
        assert!(extensible_linear_index(&[0, 0, 0], &n, 3).is_err());
    }

    #[test]
    fn implicit() {
        assert_eq!(implicit_chunk_address(1000, 3, 64).unwrap(), 1192);
        assert_eq!(
            implicit_chunk_address(UNDEFINED_ADDRESS, 3, 64).unwrap(),
            UNDEFINED_ADDRESS
        );
        assert!(implicit_chunk_address(1, u64::MAX, 2).is_err());
        let idx = ImplicitIndex {
            address: 4096,
            chunk_bytes: 100,
            num_chunks: 3,
        };
        let r = idx.records().unwrap();
        assert_eq!(
            r.iter().map(|r| r.address).collect::<Vec<_>>(),
            vec![4096, 4196, 4296]
        );
        assert!(idx.record(3).is_err());
        let huge = ImplicitIndex {
            address: 0,
            chunk_bytes: u64::MAX,
            num_chunks: 3,
        };
        assert!(huge.records().is_err());
    }

    #[test]
    fn single_chunk() {
        let mut data = Vec::new();
        data.extend_from_slice(&1234u64.to_le_bytes());
        data.extend_from_slice(&5u32.to_le_bytes());
        let mut r = ByteReader::new(&data);
        let info = SingleChunkIndex::parse_info(&mut r, SINGLE_INDEX_WITH_FILTER).unwrap();
        assert_eq!(info, Some((1234, 5)));
        assert!(r.is_empty());
        let idx = SingleChunkIndex::new(800, info);
        assert_eq!(idx.record.size, Some(1234));
        assert_eq!(idx.record.filter_mask, 5);
        let mut r = ByteReader::new(&data);
        assert_eq!(SingleChunkIndex::parse_info(&mut r, 0x01).unwrap(), None);
        assert_eq!(r.position(), 0);
        assert!(
            !SingleChunkIndex::new(UNDEFINED_ADDRESS, None)
                .record
                .is_allocated()
        );
    }

    #[test]
    fn bitmap_order() {
        let bm = [0b1000_0001u8, 0b0100_0000];
        assert!(bitmap_bit(&bm, 0));
        assert!(!bitmap_bit(&bm, 1));
        assert!(bitmap_bit(&bm, 7));
        assert!(bitmap_bit(&bm, 9));
        assert!(!bitmap_bit(&bm, 16));
        assert!(!bitmap_bit(&bm, u64::MAX));
    }

    #[test]
    fn codec_validation() {
        let s = Sizes::DEFAULT;
        assert!(ElementCodec::new(&ChunkIndexContext::unfiltered(s), 0, 8).is_ok());
        assert!(ElementCodec::new(&ChunkIndexContext::unfiltered(s), 0, 9).is_err());
        assert!(ElementCodec::new(&ChunkIndexContext::unfiltered(s), 1, 16).is_err());
        assert!(ElementCodec::new(&ChunkIndexContext::unfiltered(s), 2, 8).is_err());
        let c = ElementCodec::new(&ChunkIndexContext::filtered(s), 1, 16).unwrap();
        assert_eq!(c.size_width, 4);
        assert!(ElementCodec::new(&ChunkIndexContext::filtered(s), 1, 12).is_err());
        assert!(ElementCodec::new(&ChunkIndexContext::filtered(s), 1, 21).is_err());
        assert!(ElementCodec::new(&ChunkIndexContext::filtered(s), 1, 3).is_err());
        let mut ctx = ChunkIndexContext::filtered(s);
        ctx.chunk_size_width = Some(2);
        assert!(ElementCodec::new(&ctx, 1, 16).is_err());
        assert!(ElementCodec::new(&ctx, 1, 14).is_ok());
    }
}
