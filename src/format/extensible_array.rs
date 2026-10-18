//! Extensible Array chunk index (spec: "The Extensible Array Index").
//!
//! # Structure
//!
//! A header (`EAHD`) points at an index block (`EAIB`). The index block
//! holds the first few elements inline, then the addresses of the data
//! blocks (`EADB`) of the smallest super blocks, then the addresses of
//! secondary blocks (`EASB`), one per remaining super block. A secondary
//! block lists the addresses of its super block's data blocks. Large data
//! blocks are split into pages laid out contiguously after the data block
//! header, each page followed by its own checksum; which pages have been
//! initialized is recorded by a bitmap in the secondary block.
//!
//! # Geometry
//!
//! The spec lists the creation parameters but not how they shape the
//! array; the layout below follows from the description ("the first few
//! \[blocks\] are (conceptually) stored in parent elements", data blocks that
//! grow, a "pre-balanced internal structure" of fixed depth) together with
//! the requirement that the field counts be derivable from the header alone.
//! Let
//!
//! * `I` = index block elements, `m` = data block min elements,
//! * `p` = secondary block min data pointers, `B` = max nelmts bits.
//!
//! Array indices `0..I` live in the index block. The rest of the address
//! space is split into *super blocks* `s = 0, 1, 2, …`. Super block `s`
//! holds `2^floor(s/2)` data blocks of `m * 2^ceil(s/2)` elements each, i.e.
//! `m * 2^s` elements: sizes double every super block, alternating between
//! doubling the number of data blocks and doubling their size:
//!
//! ```text
//! s          0   1   2   3   4   5   6  ...
//! dblks      1   1   2   2   4   4   8
//! dblk size  m  2m  2m  4m  4m  8m  8m
//! ```
//!
//! Super block `s` therefore starts at array index `I + m * (2^s - 1)`.
//! Super blocks with fewer than `p` data blocks do not get a secondary
//! block; their data block addresses sit directly in the index block. With
//! `p` a power of two, these are super blocks `0 .. 2*log2(p)`, holding
//! `2 * (1 + 2 + … + p/2) = 2 * (p - 1)` data blocks. The number of super
//! blocks is `1 + B - log2(m)` (enough for `2^B` elements), so the index
//! block has `1 + B - log2(m) - 2*log2(p)` secondary block address slots.
//!
//! The "Block Offset" fields are encoded in `ceil(B / 8)` bytes (by analogy
//! with the fractal heap's block offsets, the spec gives no width); they are
//! parsed but, being "principally used for file integrity checking", not
//! validated.
//!
//! A data block is paged when it has more than `2^page_bits` elements; it
//! then has `elements / 2^page_bits` pages. The secondary block's page
//! bitmap has one bit per page of each of its data blocks, in order
//! (data block 0's pages first).

use super::chunk_index::{
    ChunkIndexContext, ChunkRecord, ElementCodec, bitmap_bit, checked_count, record_limit,
    verify_checksum,
};
use super::{ByteReader, Sizes, UNDEFINED_ADDRESS};
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Parsed Extensible Array header (`EAHD`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensibleArrayHeader {
    /// Address the header was read from.
    pub address: u64,
    /// Client ID: 0 = non-filtered chunks, 1 = filtered chunks.
    pub client_id: u8,
    /// Size in bytes of one element.
    pub element_size: u8,
    /// Number of bits needed to store the maximum number of elements.
    pub max_nelmts_bits: u8,
    /// Number of elements stored directly in the index block.
    pub index_block_elements: u8,
    /// Minimum number of elements per data block.
    pub data_block_min_elements: u8,
    /// Minimum number of data block pointers for a secondary block.
    pub secondary_block_min_data_ptrs: u8,
    /// log2 of the maximum number of elements in a data block page.
    pub max_data_block_page_elements_bits: u8,
    /// Statistics: number of secondary blocks created.
    pub num_secondary_blocks: u64,
    /// Statistics: size of the secondary blocks created.
    pub secondary_block_size: u64,
    /// Statistics: number of data blocks created.
    pub num_data_blocks: u64,
    /// Statistics: size of the data blocks created.
    pub data_block_size: u64,
    /// One more than the highest element index set (taken as the logical
    /// element count by [`ExtensibleArray::read`]).
    pub max_index_set: u64,
    /// Statistics: number of elements realized.
    pub num_elements: u64,
    /// Address of the index block, or [`UNDEFINED_ADDRESS`].
    pub index_block_address: u64,
}

impl ExtensibleArrayHeader {
    /// Encoded size of the header in bytes.
    pub fn encoded_size(sizes: Sizes) -> usize {
        4 + 4 + 4 + 6 * usize::from(sizes.length) + usize::from(sizes.offset) + 4
    }

    /// Read and verify the header at `addr`.
    pub fn read<R: ReadAt + ?Sized>(store: &R, addr: u64, sizes: Sizes) -> Result<Self> {
        let buf = store.read_vec_at(addr, Self::encoded_size(sizes))?;
        let mut r = ByteReader::with_sizes(&buf, sizes);
        r.expect_signature(b"EAHD")?;
        verify_checksum(&buf, "extensible array header")?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "extensible array header version {version}"
            )));
        }
        Ok(ExtensibleArrayHeader {
            address: addr,
            client_id: r.read_u8()?,
            element_size: r.read_u8()?,
            max_nelmts_bits: r.read_u8()?,
            index_block_elements: r.read_u8()?,
            data_block_min_elements: r.read_u8()?,
            secondary_block_min_data_ptrs: r.read_u8()?,
            max_data_block_page_elements_bits: r.read_u8()?,
            num_secondary_blocks: r.read_length()?,
            secondary_block_size: r.read_length()?,
            num_data_blocks: r.read_length()?,
            data_block_size: r.read_length()?,
            max_index_set: r.read_length()?,
            num_elements: r.read_length()?,
            index_block_address: r.read_offset()?,
        })
    }
}

/// `2^k`, or `None` if it does not fit in a `u64`.
fn pow2(k: u32) -> Option<u64> {
    1u64.checked_shl(k)
}

/// Where an array element lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementLocation {
    /// Stored inline in the index block, at this position.
    IndexBlock(u64),
    /// Stored in a data block.
    DataBlock {
        /// Super block number.
        super_block: u32,
        /// Data block number within the super block.
        data_block: u64,
        /// Element number within the data block.
        element: u64,
    },
}

/// The shape of an extensible array, derived from its creation parameters
/// (see the [module documentation](self) for the derivation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensibleArrayGeometry {
    /// Elements stored in the index block (`I`).
    pub index_block_elements: u64,
    /// Minimum elements per data block (`m`, a power of two).
    pub data_block_min_elements: u64,
    /// Minimum data block pointers per secondary block (`p`, a power of two).
    pub secondary_block_min_data_ptrs: u64,
    /// Bits needed for the maximum number of elements (`B`).
    pub max_nelmts_bits: u32,
    /// log2 of the maximum number of elements per data block page.
    pub page_bits: u32,
}

impl ExtensibleArrayGeometry {
    /// Validate and derive the geometry from a header.
    pub fn from_header(h: &ExtensibleArrayHeader) -> Result<Self> {
        let m = h.data_block_min_elements;
        let p = h.secondary_block_min_data_ptrs;
        if !m.is_power_of_two() {
            return Err(Error::invalid(
                "extensible array data block min elements is not a power of two",
            ));
        }
        if !p.is_power_of_two() {
            return Err(Error::invalid(
                "extensible array secondary block min pointers is not a power of two",
            ));
        }
        let bits = u32::from(h.max_nelmts_bits);
        if bits == 0 || bits > 64 || m.trailing_zeros() > bits {
            return Err(Error::invalid("bad extensible array max nelmts bits"));
        }
        Ok(ExtensibleArrayGeometry {
            index_block_elements: u64::from(h.index_block_elements),
            data_block_min_elements: u64::from(m),
            secondary_block_min_data_ptrs: u64::from(p),
            max_nelmts_bits: bits,
            page_bits: u32::from(h.max_data_block_page_elements_bits),
        })
    }

    /// Number of super blocks: `1 + B - log2(m)`.
    pub fn super_block_count(&self) -> u32 {
        1 + self.max_nelmts_bits - self.data_block_min_elements.trailing_zeros()
    }

    /// Number of super blocks whose data blocks are referenced directly from
    /// the index block: `2 * log2(p)` (possibly more than actually exist).
    pub fn index_block_super_blocks(&self) -> u32 {
        2 * self.secondary_block_min_data_ptrs.trailing_zeros()
    }

    /// Number of data block address slots in the index block: `2 * (p - 1)`.
    pub fn index_block_data_block_addresses(&self) -> u64 {
        2 * (self.secondary_block_min_data_ptrs - 1)
    }

    /// Number of secondary block address slots in the index block.
    pub fn index_block_secondary_block_addresses(&self) -> u64 {
        u64::from(
            self.super_block_count()
                .saturating_sub(self.index_block_super_blocks()),
        )
    }

    /// Width in bytes of "Block Offset" fields: `ceil(B / 8)`.
    pub fn block_offset_width(&self) -> usize {
        self.max_nelmts_bits.div_ceil(8) as usize
    }

    /// Number of data blocks in super block `s`: `2^floor(s/2)`.
    pub fn data_blocks_in_super_block(&self, s: u32) -> Result<u64> {
        pow2(s / 2).ok_or(Error::LimitExceeded("extensible array super block"))
    }

    /// Number of elements per data block of super block `s`:
    /// `m * 2^ceil(s/2)`.
    pub fn data_block_elements(&self, s: u32) -> Result<u64> {
        pow2(s.div_ceil(2))
            .and_then(|v| v.checked_mul(self.data_block_min_elements))
            .ok_or(Error::LimitExceeded("extensible array data block"))
    }

    /// Array index of the first element of super block `s`:
    /// `I + m * (2^s - 1)`.
    pub fn super_block_start(&self, s: u32) -> Result<u64> {
        pow2(s)
            .map(|v| v - 1)
            .and_then(|v| v.checked_mul(self.data_block_min_elements))
            .and_then(|v| v.checked_add(self.index_block_elements))
            .ok_or(Error::LimitExceeded("extensible array super block"))
    }

    /// Elements per data block page, if super block `s`'s data blocks are
    /// paged, together with the number of pages per data block.
    pub fn data_block_paging(&self, s: u32) -> Result<Option<(u64, u64)>> {
        let n = self.data_block_elements(s)?;
        Ok(match pow2(self.page_bits) {
            Some(pe) if n > pe => Some((pe, n / pe)),
            _ => None,
        })
    }

    /// Position, within the index block's data block address list, of the
    /// first data block of super block `s` (`s` < index block super blocks).
    fn index_block_dblk_base(&self, s: u32) -> Result<u64> {
        (0..s).try_fold(0u64, |acc, t| Ok(acc + self.data_blocks_in_super_block(t)?))
    }

    /// Locate array element `index`.
    pub fn locate(&self, index: u64) -> Result<ElementLocation> {
        let Some(mut rel) = index.checked_sub(self.index_block_elements) else {
            return Ok(ElementLocation::IndexBlock(index));
        };
        for s in 0..self.super_block_count() {
            // Super block size m * 2^s; saturate: an overflowing size holds
            // every remaining representable index.
            let size = pow2(s)
                .and_then(|v| v.checked_mul(self.data_block_min_elements))
                .unwrap_or(u64::MAX);
            if rel < size {
                let per = self.data_block_elements(s)?;
                return Ok(ElementLocation::DataBlock {
                    super_block: s,
                    data_block: rel / per,
                    element: rel % per,
                });
            }
            rel -= size;
        }
        Err(Error::invalid("index beyond extensible array capacity"))
    }
}

/// A parsed secondary block (`EASB`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecondaryBlock {
    /// The "Block Offset" field (not validated).
    pub block_offset: u64,
    /// Page initialization bitmap (empty if the data blocks are not paged).
    pub page_bitmap: Vec<u8>,
    /// Data block addresses (some may be undefined).
    pub data_block_addresses: Vec<u64>,
}

/// An opened Extensible Array chunk index: header plus parsed index block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensibleArray {
    /// The parsed header.
    pub header: ExtensibleArrayHeader,
    /// The derived geometry.
    pub geometry: ExtensibleArrayGeometry,
    codec: ElementCodec,
    index_elements: Vec<ChunkRecord>,
    data_block_addresses: Vec<u64>,
    secondary_block_addresses: Vec<u64>,
}

/// Size of the common block prefix: signature, version, client ID, header
/// address.
fn prefix_len(sizes: Sizes) -> usize {
    4 + 2 + usize::from(sizes.offset)
}

fn size_err() -> Error {
    Error::LimitExceeded("extensible array block size")
}

impl ExtensibleArray {
    /// Read the header and index block at `addr`, validating against `ctx`.
    pub fn open<R: ReadAt + ?Sized>(store: &R, addr: u64, ctx: &ChunkIndexContext) -> Result<Self> {
        let header = ExtensibleArrayHeader::read(store, addr, ctx.sizes)?;
        let geometry = ExtensibleArrayGeometry::from_header(&header)?;
        let codec = ElementCodec::new(ctx, header.client_id, header.element_size)?;
        let mut ea = ExtensibleArray {
            header,
            geometry,
            codec,
            index_elements: Vec::new(),
            data_block_addresses: Vec::new(),
            secondary_block_addresses: Vec::new(),
        };
        if header.index_block_address != UNDEFINED_ADDRESS {
            ea.read_index_block(store)?;
        }
        Ok(ea)
    }

    /// Read the whole index at `addr`: records for array indices
    /// `0..max_index_set`, in linear chunk index order.
    pub fn read<R: ReadAt + ?Sized>(
        store: &R,
        addr: u64,
        ctx: &ChunkIndexContext,
    ) -> Result<Vec<ChunkRecord>> {
        let ea = Self::open(store, addr, ctx)?;
        ea.records(store, ea.header.max_index_set)
    }

    fn read_index_block<R: ReadAt + ?Sized>(&mut self, store: &R) -> Result<()> {
        let g = &self.geometry;
        let o = usize::from(self.codec.sizes.offset);
        // All counts here are small: they derive from one-byte parameters.
        let nel = g.index_block_elements as usize;
        let ndb = g.index_block_data_block_addresses() as usize;
        let nsb = g.index_block_secondary_block_addresses() as usize;
        let len =
            prefix_len(self.codec.sizes) + nel * self.codec.element_size + (ndb + nsb) * o + 4;
        let buf = store.read_vec_at(self.header.index_block_address, len)?;
        let mut r = self.check_prefix(&buf, b"EAIB", "extensible array index block")?;
        let mut elems = Vec::with_capacity(nel);
        for _ in 0..nel {
            elems.push(self.codec.decode(&mut r)?);
        }
        let dblks = (0..ndb)
            .map(|_| r.read_offset())
            .collect::<Result<Vec<_>>>()?;
        let sblks = (0..nsb)
            .map(|_| r.read_offset())
            .collect::<Result<Vec<_>>>()?;
        self.index_elements = elems;
        self.data_block_addresses = dblks;
        self.secondary_block_addresses = sblks;
        Ok(())
    }

    /// Check signature, checksum, version, client ID and header address of
    /// a block; returns a reader positioned after the header address.
    fn check_prefix<'a>(
        &self,
        buf: &'a [u8],
        sig: &[u8; 4],
        what: &'static str,
    ) -> Result<ByteReader<'a>> {
        let mut r = ByteReader::with_sizes(buf, self.codec.sizes);
        r.expect_signature(sig)?;
        verify_checksum(buf, what)?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!("{what} version {version}")));
        }
        if r.read_u8()? != self.header.client_id {
            return Err(Error::invalid(format!("{what}: client ID mismatch")));
        }
        if r.read_offset()? != self.header.address {
            return Err(Error::invalid(format!("{what}: header address mismatch")));
        }
        Ok(r)
    }

    /// Read the secondary block of super block `s` at `addr`.
    pub fn read_secondary_block<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        addr: u64,
        s: u32,
    ) -> Result<SecondaryBlock> {
        let g = &self.geometry;
        let ndblks = g.data_blocks_in_super_block(s)?;
        let bitmap_bits = match g.data_block_paging(s)? {
            Some((_, npages)) => ndblks.checked_mul(npages).ok_or_else(size_err)?,
            None => 0,
        };
        let bitmap_len = usize::try_from(bitmap_bits.div_ceil(8)).map_err(|_| size_err())?;
        let ndblks_usize = usize::try_from(ndblks).map_err(|_| size_err())?;
        let len = ndblks_usize
            .checked_mul(usize::from(self.codec.sizes.offset))
            .and_then(|v| v.checked_add(bitmap_len))
            .and_then(|v| v.checked_add(prefix_len(self.codec.sizes) + g.block_offset_width() + 4))
            .ok_or_else(size_err)?;
        // `read_vec_at` checks the length against the store before allocating.
        let buf = store.read_vec_at(addr, len)?;
        let mut r = self.check_prefix(&buf, b"EASB", "extensible array secondary block")?;
        let block_offset = r.read_uint(g.block_offset_width())?;
        let page_bitmap = r.read_bytes(bitmap_len)?.to_vec();
        let data_block_addresses = (0..ndblks_usize)
            .map(|_| r.read_offset())
            .collect::<Result<Vec<_>>>()?;
        Ok(SecondaryBlock {
            block_offset,
            page_bitmap,
            data_block_addresses,
        })
    }

    /// Append elements `range` of the data block of super
    /// block `s` at `addr` to `out`. `pages` is the page bitmap and the bit
    /// index of this data block's first page, for blocks referenced from a
    /// secondary block.
    fn read_data_block<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        addr: u64,
        s: u32,
        pages: Option<(&[u8], u64)>,
        range: std::ops::Range<u64>,
        out: &mut Vec<ChunkRecord>,
    ) -> Result<()> {
        let g = &self.geometry;
        let nelmts = g.data_block_elements(s)?;
        let (start, end) = (range.start, range.end);
        let count = end.saturating_sub(start);
        if end > nelmts {
            return Err(Error::invalid("extensible array data block range"));
        }
        let esize = self.codec.element_size;
        let hdr_len = prefix_len(self.codec.sizes) + g.block_offset_width();
        let what = "extensible array data block";
        let Some((per_page, _)) = g.data_block_paging(s)? else {
            let elems = usize::try_from(nelmts)
                .ok()
                .and_then(|n| n.checked_mul(esize))
                .ok_or_else(size_err)?;
            let len = elems.checked_add(hdr_len + 4).ok_or_else(size_err)?;
            let buf = store.read_vec_at(addr, len)?;
            let mut r = self.check_prefix(&buf, b"EADB", what)?;
            let _block_offset = r.read_uint(g.block_offset_width())?;
            // start < nelmts, so start * esize <= elems.
            r.skip(start as usize * esize)?;
            for _ in 0..count {
                out.push(self.codec.decode(&mut r)?);
            }
            return Ok(());
        };
        let Some((bitmap, first_bit)) = pages else {
            return Err(Error::unsupported(
                "paged extensible array data block referenced from the index block",
            ));
        };
        let buf = store.read_vec_at(addr, hdr_len + 4)?;
        let mut r = self.check_prefix(&buf, b"EADB", what)?;
        let _block_offset = r.read_uint(g.block_offset_width())?;
        // per_page < nelmts and nelmts * esize was not yet bounded, so use
        // checked arithmetic for page positions.
        let page_len = usize::try_from(per_page)
            .ok()
            .and_then(|n| n.checked_mul(esize))
            .and_then(|v| v.checked_add(4))
            .ok_or_else(size_err)?;
        let mut idx = start;
        while idx < end {
            let page = idx / per_page;
            let in_page = idx % per_page;
            let take = (per_page - in_page).min(end - idx);
            let bit = first_bit.checked_add(page).ok_or_else(size_err)?;
            if !bitmap_bit(bitmap, bit) {
                let n = usize::try_from(take).map_err(|_| size_err())?;
                out.resize(out.len() + n, ChunkRecord::UNALLOCATED);
            } else {
                let pos = (page_len as u64)
                    .checked_mul(page)
                    .and_then(|v| v.checked_add((hdr_len + 4) as u64))
                    .and_then(|v| v.checked_add(addr))
                    .ok_or(Error::UnexpectedEof)?;
                let pbuf = store.read_vec_at(pos, page_len)?;
                verify_checksum(&pbuf, "extensible array data block page")?;
                let mut pr = ByteReader::with_sizes(&pbuf, self.codec.sizes);
                pr.skip(in_page as usize * esize)?;
                for _ in 0..take {
                    out.push(self.codec.decode(&mut pr)?);
                }
            }
            idx += take;
        }
        Ok(())
    }

    /// Look up the record of array element `index` (unallocated elements
    /// yield [`ChunkRecord::UNALLOCATED`]).
    pub fn get<R: ReadAt + ?Sized>(&self, store: &R, index: u64) -> Result<ChunkRecord> {
        let (s, d, e) = match self.geometry.locate(index)? {
            ElementLocation::IndexBlock(i) => {
                return Ok(usize::try_from(i)
                    .ok()
                    .and_then(|i| self.index_elements.get(i).copied())
                    .unwrap_or(ChunkRecord::UNALLOCATED));
            }
            ElementLocation::DataBlock {
                super_block,
                data_block,
                element,
            } => (super_block, data_block, element),
        };
        let mut out = Vec::with_capacity(1);
        if s < self.geometry.index_block_super_blocks() {
            let pos = self.geometry.index_block_dblk_base(s)? + d;
            let addr = self.iblock_dblk(pos);
            if addr == UNDEFINED_ADDRESS {
                return Ok(ChunkRecord::UNALLOCATED);
            }
            self.read_data_block(store, addr, s, None, e..e + 1, &mut out)?;
        } else {
            let saddr = self.iblock_sblk(s);
            if saddr == UNDEFINED_ADDRESS {
                return Ok(ChunkRecord::UNALLOCATED);
            }
            let sb = self.read_secondary_block(store, saddr, s)?;
            let addr = usize::try_from(d)
                .ok()
                .and_then(|d| sb.data_block_addresses.get(d).copied())
                .ok_or_else(|| Error::invalid("extensible array data block index"))?;
            if addr == UNDEFINED_ADDRESS {
                return Ok(ChunkRecord::UNALLOCATED);
            }
            let first_bit = self.first_page_bit(s, d)?;
            self.read_data_block(
                store,
                addr,
                s,
                Some((&sb.page_bitmap, first_bit)),
                e..e + 1,
                &mut out,
            )?;
        }
        out.pop()
            .ok_or_else(|| Error::invalid("extensible array element missing"))
    }

    fn iblock_dblk(&self, pos: u64) -> u64 {
        usize::try_from(pos)
            .ok()
            .and_then(|p| self.data_block_addresses.get(p).copied())
            .unwrap_or(UNDEFINED_ADDRESS)
    }

    fn iblock_sblk(&self, s: u32) -> u64 {
        let slot = s - self.geometry.index_block_super_blocks();
        self.secondary_block_addresses
            .get(slot as usize)
            .copied()
            .unwrap_or(UNDEFINED_ADDRESS)
    }

    fn first_page_bit(&self, s: u32, d: u64) -> Result<u64> {
        Ok(match self.geometry.data_block_paging(s)? {
            Some((_, npages)) => d.checked_mul(npages).ok_or_else(size_err)?,
            None => 0,
        })
    }

    /// Records for array indices `0..count`, in order, reading each block
    /// once.
    pub fn records<R: ReadAt + ?Sized>(&self, store: &R, count: u64) -> Result<Vec<ChunkRecord>> {
        let n = checked_count(
            count,
            record_limit(store.size()?),
            "too many extensible array elements",
        )?;
        let g = &self.geometry;
        let mut out = Vec::with_capacity(n);
        let remaining = |out: &Vec<ChunkRecord>| (n - out.len()) as u64;

        for i in 0..(g.index_block_elements as usize).min(n) {
            out.push(
                self.index_elements
                    .get(i)
                    .copied()
                    .unwrap_or(ChunkRecord::UNALLOCATED),
            );
        }
        let iblock_sblks = g.index_block_super_blocks();
        let mut iblock_pos = 0u64;
        for s in 0..g.super_block_count() {
            if out.len() >= n {
                break;
            }
            let ndblks = g.data_blocks_in_super_block(s)?;
            let per = g.data_block_elements(s)?;
            let secondary = if s < iblock_sblks {
                None
            } else {
                let saddr = self.iblock_sblk(s);
                if saddr == UNDEFINED_ADDRESS {
                    let take = ndblks.saturating_mul(per).min(remaining(&out));
                    out.resize(out.len() + take as usize, ChunkRecord::UNALLOCATED);
                    continue;
                }
                Some(self.read_secondary_block(store, saddr, s)?)
            };
            for d in 0..ndblks {
                if out.len() >= n {
                    break;
                }
                let take = per.min(remaining(&out));
                let addr = match &secondary {
                    None => {
                        let a = self.iblock_dblk(iblock_pos);
                        iblock_pos += 1;
                        a
                    }
                    Some(sb) => sb
                        .data_block_addresses
                        .get(d as usize)
                        .copied()
                        .unwrap_or(UNDEFINED_ADDRESS),
                };
                if addr == UNDEFINED_ADDRESS {
                    out.resize(out.len() + take as usize, ChunkRecord::UNALLOCATED);
                    continue;
                }
                let pages = match &secondary {
                    Some(sb) => Some((sb.page_bitmap.as_slice(), self.first_page_bit(s, d)?)),
                    None => None,
                };
                self.read_data_block(store, addr, s, pages, 0..take, &mut out)?;
            }
        }
        if out.len() < n {
            return Err(Error::invalid("count beyond extensible array capacity"));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Encoding (write side)
// ---------------------------------------------------------------------------

impl ExtensibleArrayHeader {
    /// Encode the header (`EAHD` through checksum).
    pub fn encode(&self, sizes: Sizes) -> Result<Vec<u8>> {
        let mut w = super::ByteWriter::with_sizes(sizes);
        w.bytes(b"EAHD")
            .u8(0)
            .u8(self.client_id)
            .u8(self.element_size)
            .u8(self.max_nelmts_bits)
            .u8(self.index_block_elements)
            .u8(self.data_block_min_elements)
            .u8(self.secondary_block_min_data_ptrs)
            .u8(self.max_data_block_page_elements_bits);
        for v in [
            self.num_secondary_blocks,
            self.secondary_block_size,
            self.num_data_blocks,
            self.data_block_size,
            self.max_index_set,
            self.num_elements,
        ] {
            w.write_length(v)?;
        }
        w.write_offset(self.index_block_address)?;
        Ok(super::chunk_index::seal_block(w))
    }
}

/// Creation parameters of an extensible array (as stored in the layout
/// message).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensibleArrayParams {
    /// Bits needed to store the maximum number of elements.
    pub max_nelmts_bits: u8,
    /// Elements stored in the index block.
    pub index_block_elements: u8,
    /// Minimum data block pointers of a secondary block (power of two).
    pub secondary_block_min_data_ptrs: u8,
    /// Minimum elements per data block (power of two).
    pub data_block_min_elements: u8,
    /// log2 of the maximum number of elements in a data block page.
    pub max_data_block_page_elements_bits: u8,
}

impl Default for ExtensibleArrayParams {
    /// The parameters the reference library uses for chunk indexes.
    fn default() -> Self {
        ExtensibleArrayParams {
            max_nelmts_bits: 32,
            index_block_elements: 4,
            secondary_block_min_data_ptrs: 4,
            data_block_min_elements: 16,
            max_data_block_page_elements_bits: 10,
        }
    }
}

/// Encode an Extensible Array chunk index holding `records` (array indices
/// `0..records.len()`; "max index set" is `records.len()`).
///
/// Data blocks are created only when they hold an allocated record, and
/// secondary blocks only when one of their data blocks exists, as the
/// reference library does when elements are set. Paged data blocks are
/// written in full, with the bitmap bit of every page holding an allocated
/// record set.
///
/// "Block Offset" fields follow what the reference library writes: the
/// super block's first element index (index block elements excluded) plus
/// the data block's position in its parent's address list times the data
/// block size; for data blocks listed in the index block that position
/// counts every index block data block before it.
pub fn encode_extensible_array(
    ctx: &ChunkIndexContext,
    size_width: u8,
    params: &ExtensibleArrayParams,
    records: &[ChunkRecord],
    alloc: &mut super::chunk_index::AllocFn<'_>,
) -> Result<super::chunk_index::EncodedIndex> {
    use super::chunk_index::{EncodedBlock, EncodedIndex, seal_block, set_bitmap_bit};
    let sizes = ctx.sizes;
    let codec = ElementCodec::for_encoding(sizes, ctx.filtered, size_width)?;
    let mut header = ExtensibleArrayHeader {
        address: 0,
        client_id: u8::from(ctx.filtered),
        element_size: u8::try_from(codec.element_size)
            .map_err(|_| Error::invalid("extensible array element size"))?,
        max_nelmts_bits: params.max_nelmts_bits,
        index_block_elements: params.index_block_elements,
        data_block_min_elements: params.data_block_min_elements,
        secondary_block_min_data_ptrs: params.secondary_block_min_data_ptrs,
        max_data_block_page_elements_bits: params.max_data_block_page_elements_bits,
        num_secondary_blocks: 0,
        secondary_block_size: 0,
        num_data_blocks: 0,
        data_block_size: 0,
        max_index_set: records.len() as u64,
        num_elements: u64::from(params.index_block_elements),
        index_block_address: UNDEFINED_ADDRESS,
    };
    let g = ExtensibleArrayGeometry::from_header(&header)?;
    let n = records.len() as u64;
    if g.max_nelmts_bits < 64 && n > (1u64 << g.max_nelmts_bits) {
        return Err(Error::LimitExceeded("too many extensible array elements"));
    }
    header.address = alloc(ExtensibleArrayHeader::encoded_size(sizes) as u64)?;
    let o = usize::from(sizes.offset);
    let nel = g.index_block_elements as usize;
    let ndb = g.index_block_data_block_addresses() as usize;
    let nsb = g.index_block_secondary_block_addresses() as usize;
    let iblock_len = prefix_len(sizes) + nel * codec.element_size + (ndb + nsb) * o + 4;
    header.index_block_address = alloc(iblock_len as u64)?;
    let rec_at = |i: u64| -> ChunkRecord {
        usize::try_from(i)
            .ok()
            .and_then(|i| records.get(i).copied())
            .unwrap_or(ChunkRecord::UNALLOCATED)
    };
    let any_allocated = |start: u64, len: u64| -> bool {
        let end = start.saturating_add(len).min(n);
        (start..end).any(|i| rec_at(i).is_allocated())
    };
    let prefix = |w: &mut super::ByteWriter, sig: &[u8; 4]| -> Result<()> {
        w.bytes(sig).u8(0).u8(header.client_id);
        w.write_offset(header.address)?;
        Ok(())
    };
    let bow = g.block_offset_width();
    // Encode one data block of super block `s` starting at array index
    // `start` (`per` elements); returns its bytes and page bitmap bits.
    let data_block =
        |s: u32, start: u64, per: u64, block_offset: u64| -> Result<(Vec<u8>, Vec<bool>)> {
            let mut w = super::ByteWriter::with_sizes(sizes);
            prefix(&mut w, b"EADB")?;
            w.write_uint(block_offset, bow)?;
            match g.data_block_paging(s)? {
                None => {
                    for i in start..start + per {
                        codec.encode(&mut w, &rec_at(i))?;
                    }
                    Ok((seal_block(w), Vec::new()))
                }
                Some((pe, npages)) => {
                    let mut bytes = seal_block(w);
                    let mut bits = Vec::new();
                    for p in 0..npages {
                        let pstart = start + p * pe;
                        bits.push(any_allocated(pstart, pe));
                        let mut pw = super::ByteWriter::with_sizes(sizes);
                        for i in pstart..pstart + pe {
                            codec.encode(&mut pw, &rec_at(i))?;
                        }
                        bytes.extend_from_slice(&seal_block(pw));
                    }
                    Ok((bytes, bits))
                }
            }
        };

    let mut blocks = Vec::new();
    let mut iblock_dblks = vec![UNDEFINED_ADDRESS; ndb];
    let mut iblock_sblks = vec![UNDEFINED_ADDRESS; nsb];
    let mut iblock_pos = 0usize;
    let iblock_sblk_count = g.index_block_super_blocks();
    for s in 0..g.super_block_count() {
        let start = g.super_block_start(s)?;
        if start >= n {
            break;
        }
        let rel = start - g.index_block_elements;
        let ndblks = g.data_blocks_in_super_block(s)?;
        let per = g.data_block_elements(s)?;
        if s < iblock_sblk_count {
            for d in 0..ndblks {
                let dstart = start + d * per;
                let pos = iblock_pos;
                iblock_pos += 1;
                if !any_allocated(dstart, per) {
                    continue;
                }
                let off = rel + pos as u64 * per;
                let (bytes, _) = data_block(s, dstart, per, off)?;
                let addr = alloc(bytes.len() as u64)?;
                header.num_data_blocks += 1;
                header.data_block_size += bytes.len() as u64;
                header.num_elements += per;
                if let Some(slot) = iblock_dblks.get_mut(pos) {
                    *slot = addr;
                }
                blocks.push(EncodedBlock {
                    address: addr,
                    bytes,
                });
            }
            continue;
        }
        // Super block with a secondary block.
        if !any_allocated(start, ndblks.saturating_mul(per)) {
            continue;
        }
        let npages = g.data_block_paging(s)?.map_or(0, |(_, np)| np);
        let nbits = ndblks
            .checked_mul(npages)
            .ok_or(Error::LimitExceeded("extensible array page bitmap"))?;
        let mut bitmap = vec![
            0u8;
            usize::try_from(nbits.div_ceil(8)).map_err(|_| {
                Error::LimitExceeded("extensible array page bitmap")
            })?
        ];
        let mut dblk_addrs = Vec::new();
        for d in 0..ndblks {
            let dstart = start + d * per;
            if !any_allocated(dstart, per) {
                dblk_addrs.push(UNDEFINED_ADDRESS);
                continue;
            }
            let (bytes, bits) = data_block(s, dstart, per, rel + d * per)?;
            for (p, set) in bits.into_iter().enumerate() {
                if set {
                    set_bitmap_bit(&mut bitmap, d * npages + p as u64);
                }
            }
            let addr = alloc(bytes.len() as u64)?;
            header.num_data_blocks += 1;
            header.data_block_size += bytes.len() as u64;
            header.num_elements += per;
            dblk_addrs.push(addr);
            blocks.push(EncodedBlock {
                address: addr,
                bytes,
            });
        }
        let mut w = super::ByteWriter::with_sizes(sizes);
        prefix(&mut w, b"EASB")?;
        w.write_uint(rel, bow)?;
        w.bytes(&bitmap);
        for a in &dblk_addrs {
            w.write_offset(*a)?;
        }
        let bytes = seal_block(w);
        let addr = alloc(bytes.len() as u64)?;
        header.num_secondary_blocks += 1;
        header.secondary_block_size += bytes.len() as u64;
        let slot = (s - iblock_sblk_count) as usize;
        *iblock_sblks
            .get_mut(slot)
            .ok_or_else(|| Error::invalid("extensible array secondary block slot"))? = addr;
        blocks.push(EncodedBlock {
            address: addr,
            bytes,
        });
    }
    // Index block.
    let mut w = super::ByteWriter::with_sizes(sizes);
    prefix(&mut w, b"EAIB")?;
    for i in 0..g.index_block_elements {
        codec.encode(&mut w, &rec_at(i))?;
    }
    for a in iblock_dblks.iter().chain(&iblock_sblks) {
        w.write_offset(*a)?;
    }
    let bytes = seal_block(w);
    if bytes.len() != iblock_len {
        return Err(Error::invalid("extensible array index block size"));
    }
    blocks.push(EncodedBlock {
        address: header.index_block_address,
        bytes,
    });
    blocks.push(EncodedBlock {
        address: header.address,
        bytes: header.encode(sizes)?,
    });
    Ok(EncodedIndex {
        address: header.address,
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::super::chunk_index::testutil::{Rng, put, put_addr, seal, set_bit};
    use super::*;

    const HDR: u64 = 0x40;

    #[derive(Clone, Copy)]
    struct Spec {
        sizes: Sizes,
        filtered: bool,
        size_width: usize,
        idx_elmts: u8,
        min_elmts: u8,
        min_ptrs: u8,
        max_bits: u8,
        page_bits: u8,
        /// Indices `0..n` hold `value(i)`; the rest are unallocated.
        n: u64,
    }

    fn value(spec: &Spec, i: u64) -> ChunkRecord {
        ChunkRecord {
            address: 10_000 + i,
            size: spec.filtered.then_some(i + 1),
            filter_mask: if spec.filtered { (i % 3) as u32 } else { 0 },
        }
    }

    fn expected(spec: &Spec, count: u64) -> Vec<ChunkRecord> {
        (0..count)
            .map(|i| {
                if i < spec.n {
                    value(spec, i)
                } else {
                    ChunkRecord::UNALLOCATED
                }
            })
            .collect()
    }

    /// Build a file with an extensible array header at `HDR`. Blocks are
    /// allocated only when they contain an index `< n`; pages likewise.
    fn build(spec: &Spec) -> Vec<u8> {
        let o = usize::from(spec.sizes.offset);
        let l = usize::from(spec.sizes.length);
        let esize = if spec.filtered {
            o + spec.size_width + 4
        } else {
            o
        };
        let client = u8::from(spec.filtered);
        let hdr = ExtensibleArrayHeader {
            address: HDR,
            client_id: client,
            element_size: esize as u8,
            max_nelmts_bits: spec.max_bits,
            index_block_elements: spec.idx_elmts,
            data_block_min_elements: spec.min_elmts,
            secondary_block_min_data_ptrs: spec.min_ptrs,
            max_data_block_page_elements_bits: spec.page_bits,
            num_secondary_blocks: 0,
            secondary_block_size: 0,
            num_data_blocks: 0,
            data_block_size: 0,
            max_index_set: spec.n,
            num_elements: 0,
            index_block_address: 0,
        };
        let g = ExtensibleArrayGeometry::from_header(&hdr).unwrap();
        let bow = g.block_offset_width();
        let enc = |f: &mut Vec<u8>, i: u64| {
            let e = if i < spec.n {
                value(spec, i)
            } else {
                ChunkRecord::UNALLOCATED
            };
            put_addr(f, e.address, o);
            if spec.filtered {
                put(f, e.size.unwrap_or(0), spec.size_width);
                f.extend_from_slice(&e.filter_mask.to_le_bytes());
            }
        };
        let block_start = |f: &mut Vec<u8>, sig: &[u8; 4]| {
            f.extend_from_slice(sig);
            f.extend_from_slice(&[0, client]);
            put_addr(f, HDR, o);
        };

        let hdr_len = ExtensibleArrayHeader::encoded_size(spec.sizes);
        let iblk_addr = HDR as usize + hdr_len;
        let ndb = g.index_block_data_block_addresses() as usize;
        let nsb = g.index_block_secondary_block_addresses() as usize;
        let iblk_len = 6 + o + spec.idx_elmts as usize * esize + (ndb + nsb) * o + 4;
        let mut f = vec![0u8; iblk_addr + iblk_len];

        // Data blocks and secondary blocks, appended after the index block.
        let mut iblock_dblks = vec![u64::MAX; ndb];
        let mut sblks = vec![u64::MAX; nsb];
        let mut iblock_pos = 0;
        for s in 0..g.super_block_count() {
            let Ok(start) = g.super_block_start(s) else {
                break;
            };
            let ndblks = g.data_blocks_in_super_block(s).unwrap();
            let per = g.data_block_elements(s).unwrap();
            let paging = g.data_block_paging(s).unwrap();
            let in_iblock = s < g.index_block_super_blocks();
            if !in_iblock && start >= spec.n {
                break;
            }
            let mut addrs = vec![u64::MAX; ndblks as usize];
            let npages = paging.map_or(0, |p| p.1);
            let mut bitmap = vec![0u8; (ndblks * npages).div_ceil(8) as usize];
            for d in 0..ndblks {
                let dstart = start + d * per;
                if dstart >= spec.n {
                    if in_iblock {
                        iblock_pos += 1;
                    }
                    continue;
                }
                let a = f.len();
                block_start(&mut f, b"EADB");
                put(&mut f, start - spec.idx_elmts as u64, bow);
                match paging {
                    None => {
                        for i in dstart..dstart + per {
                            enc(&mut f, i);
                        }
                        seal(&mut f, a);
                    }
                    Some((pe, np)) => {
                        assert!(!in_iblock);
                        seal(&mut f, a);
                        for p in 0..np {
                            let pstart = dstart + p * pe;
                            let ps = f.len();
                            if pstart < spec.n {
                                set_bit(&mut bitmap, (d * np + p) as usize);
                                for i in pstart..pstart + pe {
                                    enc(&mut f, i);
                                }
                                seal(&mut f, ps);
                            } else {
                                f.resize(ps + pe as usize * esize + 4, 0x5A);
                            }
                        }
                    }
                }
                if in_iblock {
                    iblock_dblks[iblock_pos] = a as u64;
                    iblock_pos += 1;
                } else {
                    addrs[d as usize] = a as u64;
                }
            }
            if !in_iblock {
                let a = f.len();
                block_start(&mut f, b"EASB");
                put(&mut f, start - spec.idx_elmts as u64, bow);
                f.extend_from_slice(&bitmap);
                for &d in &addrs {
                    put_addr(&mut f, d, o);
                }
                seal(&mut f, a);
                sblks[(s - g.index_block_super_blocks()) as usize] = a as u64;
            }
        }

        // Index block.
        let mut ib = Vec::new();
        ib.extend_from_slice(b"EAIB");
        ib.extend_from_slice(&[0, client]);
        put_addr(&mut ib, HDR, o);
        for i in 0..u64::from(spec.idx_elmts) {
            enc(&mut ib, i);
        }
        for &a in iblock_dblks.iter().chain(&sblks) {
            put_addr(&mut ib, a, o);
        }
        seal(&mut ib, 0);
        assert_eq!(ib.len(), iblk_len);
        f[iblk_addr..iblk_addr + iblk_len].copy_from_slice(&ib);

        // Header.
        let mut h = Vec::new();
        h.extend_from_slice(b"EAHD");
        h.extend_from_slice(&[
            0,
            client,
            esize as u8,
            spec.max_bits,
            spec.idx_elmts,
            spec.min_elmts,
            spec.min_ptrs,
            spec.page_bits,
        ]);
        for v in [0, 0, 0, 0, spec.n, spec.n] {
            put(&mut h, v, l);
        }
        put_addr(&mut h, iblk_addr as u64, o);
        seal(&mut h, 0);
        f[HDR as usize..HDR as usize + hdr_len].copy_from_slice(&h);
        f
    }

    fn ctx(spec: &Spec) -> ChunkIndexContext {
        if spec.filtered {
            ChunkIndexContext::filtered(spec.sizes)
        } else {
            ChunkIndexContext::unfiltered(spec.sizes)
        }
    }

    fn check(spec: &Spec) -> Vec<u8> {
        let f = build(spec);
        let c = ctx(spec);
        assert_eq!(
            ExtensibleArray::read(&f, HDR, &c).unwrap(),
            expected(spec, spec.n)
        );
        let ea = ExtensibleArray::open(&f, HDR, &c).unwrap();
        // Past max_index_set: trailing unallocated records.
        let extra = spec.n + 9;
        assert_eq!(ea.records(&f, extra).unwrap(), expected(spec, extra));
        for i in 0..extra {
            assert_eq!(
                ea.get(&f, i).unwrap(),
                expected(spec, extra)[i as usize],
                "index {i}"
            );
        }
        f
    }

    const BASE: Spec = Spec {
        sizes: Sizes::DEFAULT,
        filtered: false,
        size_width: 0,
        idx_elmts: 4,
        min_elmts: 16,
        min_ptrs: 4,
        max_bits: 32,
        page_bits: 10,
        n: 3,
    };

    #[test]
    fn geometry_table() {
        let hdr = ExtensibleArrayHeader {
            address: 0,
            client_id: 0,
            element_size: 8,
            max_nelmts_bits: 32,
            index_block_elements: 4,
            data_block_min_elements: 16,
            secondary_block_min_data_ptrs: 4,
            max_data_block_page_elements_bits: 10,
            num_secondary_blocks: 0,
            secondary_block_size: 0,
            num_data_blocks: 0,
            data_block_size: 0,
            max_index_set: 0,
            num_elements: 0,
            index_block_address: UNDEFINED_ADDRESS,
        };
        let g = ExtensibleArrayGeometry::from_header(&hdr).unwrap();
        assert_eq!(g.super_block_count(), 29);
        assert_eq!(g.index_block_super_blocks(), 4);
        assert_eq!(g.index_block_data_block_addresses(), 6);
        assert_eq!(g.index_block_secondary_block_addresses(), 25);
        assert_eq!(g.block_offset_width(), 4);
        let nd: Vec<_> = (0..7)
            .map(|s| g.data_blocks_in_super_block(s).unwrap())
            .collect();
        assert_eq!(nd, [1, 1, 2, 2, 4, 4, 8]);
        let de: Vec<_> = (0..7).map(|s| g.data_block_elements(s).unwrap()).collect();
        assert_eq!(de, [16, 32, 32, 64, 64, 128, 128]);
        let st: Vec<_> = (0..5).map(|s| g.super_block_start(s).unwrap()).collect();
        assert_eq!(st, [4, 20, 52, 116, 244]);
        // Index-block data blocks: 1 + 1 + 2 + 2 = 2 * (4 - 1).
        assert_eq!(g.index_block_dblk_base(4).unwrap(), 6);
        assert_eq!(g.locate(3).unwrap(), ElementLocation::IndexBlock(3));
        assert_eq!(
            g.locate(4).unwrap(),
            ElementLocation::DataBlock {
                super_block: 0,
                data_block: 0,
                element: 0
            }
        );
        assert_eq!(
            g.locate(52 + 32 + 5).unwrap(),
            ElementLocation::DataBlock {
                super_block: 2,
                data_block: 1,
                element: 5
            }
        );
        // Page size 1024 first exceeded by 2048-element blocks (s = 13).
        assert_eq!(g.data_block_paging(12).unwrap(), None);
        assert_eq!(g.data_block_paging(13).unwrap(), Some((1024, 2)));
        assert_eq!(g.data_block_paging(15).unwrap(), Some((1024, 4)));
        // Capacity ends after the last super block.
        let last = g.super_block_count() - 1;
        let cap = g.super_block_start(last).unwrap() + 16 * (1u64 << last);
        assert!(g.locate(cap - 1).is_ok());
        assert!(g.locate(cap).is_err());
        // max bits 64 must not overflow anywhere.
        let mut h64 = hdr;
        h64.max_nelmts_bits = 64;
        let g = ExtensibleArrayGeometry::from_header(&h64).unwrap();
        assert!(g.locate(u64::MAX).is_ok());
        assert!(g.super_block_start(g.super_block_count() - 1).is_ok());
        assert!(g.super_block_start(g.super_block_count()).is_err());
        // Invalid parameters.
        for (m, p, b) in [
            (0, 4, 32),
            (3, 4, 32),
            (16, 0, 32),
            (16, 6, 32),
            (16, 4, 0),
            (16, 4, 65),
            (16, 4, 3),
        ] {
            let mut h = hdr;
            h.data_block_min_elements = m;
            h.secondary_block_min_data_ptrs = p;
            h.max_nelmts_bits = b;
            assert!(
                ExtensibleArrayGeometry::from_header(&h).is_err(),
                "{m} {p} {b}"
            );
        }
    }

    #[test]
    fn index_block_only() {
        check(&BASE);
        check(&Spec { n: 4, ..BASE });
        check(&Spec { n: 0, ..BASE });
    }

    #[test]
    fn index_block_data_blocks() {
        // 4 inline + 16 + 32 + part of super block 2.
        check(&Spec {
            n: 4 + 16 + 32 + 40,
            ..BASE
        });
    }

    #[test]
    fn secondary_blocks() {
        // Up into super block 5 (secondary blocks for s = 4, 5).
        check(&Spec { n: 600, ..BASE });
        // Small parameters: p = 2 -> 2 inline super blocks, 2 slots.
        let small = Spec {
            idx_elmts: 3,
            min_elmts: 2,
            min_ptrs: 2,
            max_bits: 10,
            n: 22,
            ..BASE
        };
        check(&small);
        check(&Spec {
            filtered: true,
            size_width: 2,
            sizes: Sizes {
                offset: 4,
                length: 4,
            },
            ..small
        });
    }

    #[test]
    fn paged_data_blocks() {
        // 4 elements per page; super blocks >= 3 have 8+ element blocks.
        let spec = Spec {
            idx_elmts: 1,
            min_elmts: 2,
            min_ptrs: 2,
            max_bits: 12,
            page_bits: 2,
            n: 70,
            ..BASE
        };
        check(&spec);
        check(&Spec {
            filtered: true,
            size_width: 8,
            n: 66,
            ..spec
        });
    }

    #[test]
    fn undefined_index_block() {
        let mut f = vec![0u8; HDR as usize];
        f.extend_from_slice(b"EAHD");
        f.extend_from_slice(&[0, 0, 8, 32, 4, 16, 4, 10]);
        for v in [0u64, 0, 0, 0, 0, 0] {
            put(&mut f, v, 8);
        }
        put_addr(&mut f, UNDEFINED_ADDRESS, 8);
        seal(&mut f, HDR as usize);
        let c = ChunkIndexContext::unfiltered(Sizes::DEFAULT);
        assert!(ExtensibleArray::read(&f, HDR, &c).unwrap().is_empty());
        let ea = ExtensibleArray::open(&f, HDR, &c).unwrap();
        assert_eq!(
            ea.records(&f, 100).unwrap(),
            vec![ChunkRecord::UNALLOCATED; 100]
        );
        assert_eq!(ea.get(&f, 50).unwrap(), ChunkRecord::UNALLOCATED);
        assert!(ea.records(&f, u64::MAX).is_err());
    }

    #[test]
    fn corruption_detected() {
        let spec = Spec { n: 600, ..BASE };
        let f = build(&spec);
        let c = ctx(&spec);
        // Flip one byte in every block region; each must be caught (the
        // flipped byte is inside some checksummed block or unused padding).
        let mut caught = 0;
        for pos in (HDR as usize..f.len()).step_by(7) {
            let mut g = f.clone();
            g[pos] ^= 0x10;
            if ExtensibleArray::read(&g, HDR, &c).is_err() {
                caught += 1;
            } else {
                // Only unused bytes may be changed without detection.
                assert_eq!(
                    ExtensibleArray::read(&g, HDR, &c).unwrap(),
                    expected(&spec, spec.n)
                );
            }
        }
        assert!(caught > 0);
        let mut g = f.clone();
        g[HDR as usize + 5] ^= 1;
        assert!(matches!(
            ExtensibleArray::open(&g, HDR, &c),
            Err(Error::ChecksumMismatch { .. })
        ));
        for cut in [HDR as usize + 3, HDR as usize + 60, f.len() - 1] {
            assert!(ExtensibleArray::read(&f[..cut], HDR, &c).is_err());
        }
        assert!(ExtensibleArray::read(&f, HDR, &ChunkIndexContext::filtered(spec.sizes)).is_err());
    }

    #[test]
    fn mutation_never_panics() {
        let bases = [
            build(&Spec { n: 100, ..BASE }),
            build(&Spec {
                idx_elmts: 1,
                min_elmts: 2,
                min_ptrs: 2,
                max_bits: 12,
                page_bits: 2,
                n: 70,
                filtered: true,
                size_width: 4,
                ..BASE
            }),
        ];
        let hdr_len = ExtensibleArrayHeader::encoded_size(Sizes::DEFAULT);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for base in &bases {
            for iter in 0..3000 {
                let mut g = base.clone();
                for _ in 0..1 + rng.below(4) {
                    let i = HDR as usize + rng.below(g.len() - HDR as usize);
                    g[i] = rng.next() as u8;
                }
                if iter % 2 == 0 {
                    // Re-seal the header so mutated parameters get parsed.
                    let h = HDR as usize;
                    let c = crate::checksum::metadata_checksum(&g[h..h + hdr_len - 4]);
                    g[h + hdr_len - 4..h + hdr_len].copy_from_slice(&c.to_le_bytes());
                }
                if rng.below(8) == 0 {
                    g.truncate(rng.below(g.len()));
                }
                for c in [
                    ChunkIndexContext::filtered(Sizes::DEFAULT),
                    ChunkIndexContext::unfiltered(Sizes::DEFAULT),
                ] {
                    let _ = ExtensibleArray::read(&g, HDR, &c);
                    if let Ok(ea) = ExtensibleArray::open(&g, HDR, &c) {
                        let _ = ea.records(&g, 200);
                        for _ in 0..4 {
                            let _ = ea.get(&g, rng.next() % 300);
                            let _ = ea.get(&g, rng.next());
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod encode_tests {
    use super::super::chunk_index::encode_testutil::{apply, bump};
    use super::*;

    fn records(n: u64, filtered: bool, hole: impl Fn(u64) -> bool) -> Vec<ChunkRecord> {
        (0..n)
            .map(|i| {
                if hole(i) {
                    ChunkRecord::UNALLOCATED
                } else {
                    ChunkRecord {
                        address: 1_000_000 + i * 8,
                        size: filtered.then_some(i % 300 + 1),
                        filter_mask: if filtered { (i % 3) as u32 } else { 0 },
                    }
                }
            })
            .collect()
    }

    fn ctx(sizes: Sizes, filtered: bool) -> ChunkIndexContext {
        if filtered {
            ChunkIndexContext::filtered(sizes)
        } else {
            ChunkIndexContext::unfiltered(sizes)
        }
    }

    fn round_trip(
        sizes: Sizes,
        filtered: bool,
        params: &ExtensibleArrayParams,
        recs: &[ChunkRecord],
    ) -> (Vec<u8>, ExtensibleArray) {
        let c = ctx(sizes, filtered);
        let mut alloc = bump(64);
        let idx = encode_extensible_array(&c, 3, params, recs, &mut alloc).unwrap();
        let mut f = vec![0u8; 64];
        apply(&idx, &mut f);
        let ea = ExtensibleArray::open(&f, idx.address, &c).unwrap();
        assert_eq!(ea.records(&f, recs.len() as u64).unwrap(), recs);
        for (i, r) in recs.iter().enumerate().step_by(7) {
            assert_eq!(ea.get(&f, i as u64).unwrap(), *r);
        }
        (f, ea)
    }

    #[test]
    fn matches_reference_statistics() {
        // 1500 chunks with the default parameters: statistics and block
        // offsets as in a file written by the reference library.
        let p = ExtensibleArrayParams::default();
        let recs = records(1500, false, |_| false);
        let (f, ea) = round_trip(Sizes::DEFAULT, false, &p, &recs);
        let h = ea.header;
        assert_eq!(h.num_secondary_blocks, 3);
        assert_eq!(h.secondary_block_size, 194);
        assert_eq!(h.num_data_blocks, 18);
        assert_eq!(h.data_block_size, 12556);
        assert_eq!(h.max_index_set, 1500);
        assert_eq!(h.num_elements, 1524);
        // Block offsets of the index block's data blocks.
        let offs: Vec<u64> = ea
            .data_block_addresses
            .iter()
            .map(|&a| {
                let p = a as usize + prefix_len(Sizes::DEFAULT);
                u64::from(u32::from_le_bytes(f[p..p + 4].try_into().unwrap()))
            })
            .collect();
        assert_eq!(offs, [0, 48, 112, 144, 368, 432]);
    }

    #[test]
    fn sparse_filtered_and_small_sizes() {
        let p = ExtensibleArrayParams::default();
        let recs = records(700, true, |i| i % 5 != 0 || (100..400).contains(&i));
        round_trip(Sizes::DEFAULT, true, &p, &recs);
        let s4 = Sizes {
            offset: 4,
            length: 4,
        };
        round_trip(s4, false, &p, &records(3, false, |_| false));
        round_trip(s4, true, &p, &records(90, true, |i| i == 50));
    }

    #[test]
    fn paged_data_blocks() {
        // Tiny pages so that secondary-block data blocks are paged.
        let p = ExtensibleArrayParams {
            max_nelmts_bits: 16,
            index_block_elements: 2,
            secondary_block_min_data_ptrs: 2,
            data_block_min_elements: 2,
            max_data_block_page_elements_bits: 2,
        };
        let recs = records(300, false, |i| (40..90).contains(&i) || i % 11 == 3);
        round_trip(Sizes::DEFAULT, false, &p, &recs);
        let recs = records(150, true, |i| i < 60);
        round_trip(Sizes::DEFAULT, true, &p, &recs);
    }
}
