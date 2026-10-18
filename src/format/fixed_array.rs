//! Fixed Array chunk index (spec: "The Fixed Array Index").
//!
//! A Fixed Array header (`FAHD`) points at one data block (`FADB`). When the
//! maximum number of entries exceeds `2^page_bits`, the data block is
//! *paged*: it holds only a page-initialization bitmap, and the entries live
//! in `ceil(entries / 2^page_bits)` data block pages laid out contiguously
//! right after the data block, each followed by its own checksum. The last
//! page holds the remainder. Pages whose bitmap bit is clear have never been
//! written and describe unallocated chunks.

use super::chunk_index::{
    ChunkIndexContext, ChunkRecord, ElementCodec, bitmap_bit, checked_count, record_limit,
    verify_checksum,
};
use super::{ByteReader, Sizes, UNDEFINED_ADDRESS};
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Parsed Fixed Array header (`FAHD`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedArrayHeader {
    /// Address the header was read from.
    pub address: u64,
    /// Client ID: 0 = non-filtered chunks, 1 = filtered chunks.
    pub client_id: u8,
    /// Size in bytes of one entry.
    pub entry_size: u8,
    /// log2 of the maximum number of entries in a data block page.
    pub page_bits: u8,
    /// Maximum (and actual) number of entries in the array.
    pub max_num_entries: u64,
    /// Address of the data block, or [`UNDEFINED_ADDRESS`] if no chunk has
    /// been written yet.
    pub data_block_address: u64,
}

impl FixedArrayHeader {
    /// Encoded size of the header in bytes.
    pub fn encoded_size(sizes: Sizes) -> usize {
        4 + 4 + usize::from(sizes.length) + usize::from(sizes.offset) + 4
    }

    /// Read and verify the header at `addr`.
    pub fn read<R: ReadAt + ?Sized>(store: &R, addr: u64, sizes: Sizes) -> Result<Self> {
        let buf = store.read_vec_at(addr, Self::encoded_size(sizes))?;
        let mut r = ByteReader::with_sizes(&buf, sizes);
        r.expect_signature(b"FAHD")?;
        verify_checksum(&buf, "fixed array header")?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "fixed array header version {version}"
            )));
        }
        Ok(FixedArrayHeader {
            address: addr,
            client_id: r.read_u8()?,
            entry_size: r.read_u8()?,
            page_bits: r.read_u8()?,
            max_num_entries: r.read_length()?,
            data_block_address: r.read_offset()?,
        })
    }

    /// Number of entries per data block page, if the data block is paged.
    pub fn page_entries(&self) -> Option<u64> {
        let per_page = 1u64.checked_shl(u32::from(self.page_bits))?;
        (self.max_num_entries > per_page).then_some(per_page)
    }
}

/// A Fixed Array chunk index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedArray {
    /// The parsed header.
    pub header: FixedArrayHeader,
    codec: ElementCodec,
}

impl FixedArray {
    /// Read the header at `addr` and validate it against `ctx`.
    pub fn open<R: ReadAt + ?Sized>(store: &R, addr: u64, ctx: &ChunkIndexContext) -> Result<Self> {
        let header = FixedArrayHeader::read(store, addr, ctx.sizes)?;
        let codec = ElementCodec::new(ctx, header.client_id, header.entry_size)?;
        Ok(FixedArray { header, codec })
    }

    /// Read the whole index at `addr`: one record per chunk, in linear chunk
    /// index order. Unallocated chunks have an undefined address.
    pub fn read<R: ReadAt + ?Sized>(
        store: &R,
        addr: u64,
        ctx: &ChunkIndexContext,
    ) -> Result<Vec<ChunkRecord>> {
        Self::open(store, addr, ctx)?.records(store)
    }

    /// Byte size of the data block prefix: signature, version, client ID,
    /// header address, optional page bitmap and checksum.
    fn prefix_size(&self, bitmap_len: usize) -> Result<usize> {
        (4 + 2 + usize::from(self.codec.sizes.offset) + 4)
            .checked_add(bitmap_len)
            .ok_or(Error::LimitExceeded("fixed array data block size"))
    }

    /// Read every record of the array.
    pub fn records<R: ReadAt + ?Sized>(&self, store: &R) -> Result<Vec<ChunkRecord>> {
        let h = &self.header;
        let store_size = store.size()?;
        let n = checked_count(
            h.max_num_entries,
            record_limit(store_size),
            "too many fixed array entries",
        )?;
        if h.data_block_address == UNDEFINED_ADDRESS {
            return Ok(vec![ChunkRecord::UNALLOCATED; n]);
        }
        let esize = self.codec.element_size;
        let elems_len = n
            .checked_mul(esize)
            .ok_or(Error::LimitExceeded("fixed array data block size"))?;

        let Some(per_page) = h.page_entries() else {
            // Unpaged: elements sit inside the data block, before its checksum.
            let len = self
                .prefix_size(0)?
                .checked_add(elems_len)
                .ok_or(Error::LimitExceeded("fixed array data block size"))?;
            let buf = store.read_vec_at(h.data_block_address, len)?;
            let body = self.check_block_prefix(&buf)?;
            let mut out = Vec::with_capacity(n);
            self.codec.decode_many(body, n, &mut out)?;
            return Ok(out);
        };

        // Paged: n > per_page >= 1, so per_page fits in usize.
        let per_page =
            usize::try_from(per_page).map_err(|_| Error::LimitExceeded("fixed array page size"))?;
        let npages = n.div_ceil(per_page);
        let bitmap_len = npages.div_ceil(8);
        let prefix = self.prefix_size(bitmap_len)?;
        // The data block and all its pages are allocated as one contiguous
        // region; make sure it fits in the file before reading anything.
        let total = npages
            .checked_mul(4)
            .and_then(|c| c.checked_add(elems_len))
            .and_then(|v| v.checked_add(prefix))
            .ok_or(Error::LimitExceeded("fixed array data block size"))?;
        let end = h
            .data_block_address
            .checked_add(total as u64)
            .ok_or(Error::UnexpectedEof)?;
        if end > store_size {
            return Err(Error::UnexpectedEof);
        }
        let buf = store.read_vec_at(h.data_block_address, prefix)?;
        let bitmap = self.check_block_prefix(&buf)?;

        let full_page_len = per_page
            .checked_mul(esize)
            .and_then(|v| v.checked_add(4))
            .ok_or(Error::LimitExceeded("fixed array page size"))?;
        let mut out = Vec::with_capacity(n);
        for page in 0..npages {
            let count = per_page.min(n - page * per_page);
            if !bitmap_bit(bitmap, page as u64) {
                out.resize(out.len() + count, ChunkRecord::UNALLOCATED);
                continue;
            }
            // Bounded by `total`, which was checked against the file size.
            let pos = h.data_block_address + (prefix + page * full_page_len) as u64;
            let page_buf = store.read_vec_at(pos, count * esize + 4)?;
            verify_checksum(&page_buf, "fixed array data block page")?;
            self.codec.decode_many(&page_buf, count, &mut out)?;
        }
        Ok(out)
    }

    /// Verify a data block buffer (prefix through checksum) and return the
    /// bytes between the header address and the checksum.
    fn check_block_prefix<'a>(&self, buf: &'a [u8]) -> Result<&'a [u8]> {
        let mut r = ByteReader::with_sizes(buf, self.codec.sizes);
        r.expect_signature(b"FADB")?;
        verify_checksum(buf, "fixed array data block")?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "fixed array data block version {version}"
            )));
        }
        if r.read_u8()? != self.header.client_id {
            return Err(Error::invalid("fixed array data block client ID mismatch"));
        }
        if r.read_offset()? != self.header.address {
            return Err(Error::invalid(
                "fixed array data block header address mismatch",
            ));
        }
        let rest = r.rest();
        rest.get(..rest.len().saturating_sub(4))
            .ok_or(Error::UnexpectedEof)
    }
}

#[cfg(test)]
mod tests {
    use super::super::chunk_index::testutil::{Rng, put, put_addr, seal, set_bit};
    use super::*;

    const HDR: u64 = 0x40;
    const DBLK: u64 = 0x100;

    /// Build a file containing a fixed array at `HDR` with its data block at
    /// `DBLK`. `entries[i] = None` leaves a whole page uninitialized when
    /// paged (all entries of an uninitialized page must be `None`).
    fn build(
        sizes: Sizes,
        filtered: bool,
        size_width: usize,
        page_bits: u8,
        entries: &[Option<ChunkRecord>],
    ) -> Vec<u8> {
        let o = usize::from(sizes.offset);
        let entry_size = if filtered { o + size_width + 4 } else { o };
        let client = u8::from(filtered);
        let mut f = vec![0u8; HDR as usize];
        f.extend_from_slice(b"FAHD");
        f.extend_from_slice(&[0, client, entry_size as u8, page_bits]);
        put(&mut f, entries.len() as u64, usize::from(sizes.length));
        put_addr(&mut f, DBLK, o);
        seal(&mut f, HDR as usize);
        f.resize(DBLK as usize, 0);

        let enc = |f: &mut Vec<u8>, e: &Option<ChunkRecord>| {
            let e = e.unwrap_or(ChunkRecord::UNALLOCATED);
            put_addr(f, e.address, o);
            if filtered {
                put(f, e.size.unwrap_or(0), size_width);
                f.extend_from_slice(&e.filter_mask.to_le_bytes());
            }
        };

        f.extend_from_slice(b"FADB");
        f.extend_from_slice(&[0, client]);
        put_addr(&mut f, HDR, o);
        let per_page = 1usize << page_bits;
        if entries.len() <= per_page {
            for e in entries {
                enc(&mut f, e);
            }
            seal(&mut f, DBLK as usize);
        } else {
            let pages: Vec<_> = entries.chunks(per_page).collect();
            let mut bitmap = vec![0u8; pages.len().div_ceil(8)];
            for (i, p) in pages.iter().enumerate() {
                if p.iter().any(Option::is_some) {
                    set_bit(&mut bitmap, i);
                }
            }
            f.extend_from_slice(&bitmap);
            seal(&mut f, DBLK as usize);
            for p in pages {
                let start = f.len();
                if p.iter().any(Option::is_some) {
                    for e in p {
                        enc(&mut f, e);
                    }
                    seal(&mut f, start);
                } else {
                    // Never-written page: garbage on disk.
                    f.resize(start + p.len() * entry_size + 4, 0xA5);
                }
            }
        }
        f
    }

    fn rec(address: u64, size: Option<u64>, mask: u32) -> ChunkRecord {
        ChunkRecord {
            address,
            size,
            filter_mask: mask,
        }
    }

    #[test]
    fn unpaged_unfiltered() {
        let s = Sizes::DEFAULT;
        let entries: Vec<_> = (0..5)
            .map(|i| {
                Some(rec(
                    if i == 2 { UNDEFINED_ADDRESS } else { 1000 + i },
                    None,
                    0,
                ))
            })
            .collect();
        let f = build(s, false, 0, 10, &entries);
        let got = FixedArray::read(&f, HDR, &ChunkIndexContext::unfiltered(s)).unwrap();
        assert_eq!(got.len(), 5);
        assert_eq!(got[0].address, 1000);
        assert!(!got[2].is_allocated());
        assert_eq!(got[4].address, 1004);
        // Wrong filter expectation.
        assert!(FixedArray::read(&f, HDR, &ChunkIndexContext::filtered(s)).is_err());
    }

    #[test]
    fn unpaged_filtered_small_sizes() {
        let s = Sizes {
            offset: 4,
            length: 4,
        };
        let entries: Vec<_> = (0..3)
            .map(|i| Some(rec(500 + i * 10, Some(77 + i), i as u32)))
            .collect();
        let f = build(s, true, 3, 4, &entries);
        let got = FixedArray::read(&f, HDR, &ChunkIndexContext::filtered(s)).unwrap();
        assert_eq!(got, entries.iter().map(|e| e.unwrap()).collect::<Vec<_>>());
        let mut ctx = ChunkIndexContext::filtered(s);
        ctx.chunk_size_width = Some(3);
        assert!(FixedArray::read(&f, HDR, &ctx).is_ok());
        ctx.chunk_size_width = Some(4);
        assert!(FixedArray::read(&f, HDR, &ctx).is_err());
    }

    #[test]
    fn paged_with_uninitialized_page() {
        let s = Sizes::DEFAULT;
        // page_bits 2 -> 4 entries/page; 11 entries -> 3 pages (4,4,3).
        let mut entries: Vec<Option<ChunkRecord>> = (0..11)
            .map(|i| Some(rec(2000 + i, Some(10 * i), 0)))
            .collect();
        for e in &mut entries[4..8] {
            *e = None;
        }
        let f = build(s, true, 8, 2, &entries);
        let fa = FixedArray::open(&f, HDR, &ChunkIndexContext::filtered(s)).unwrap();
        assert_eq!(fa.header.page_entries(), Some(4));
        let got = fa.records(&f).unwrap();
        assert_eq!(got.len(), 11);
        for (i, g) in got.iter().enumerate() {
            if (4..8).contains(&i) {
                assert_eq!(*g, ChunkRecord::UNALLOCATED);
            } else {
                assert_eq!(*g, rec(2000 + i as u64, Some(10 * i as u64), 0));
            }
        }
    }

    #[test]
    fn paged_unfiltered_many_pages() {
        let s = Sizes::DEFAULT;
        // 20 pages of 2 entries -> bitmap spans 3 bytes.
        let entries: Vec<_> = (0..40).map(|i| Some(rec(9000 + i, None, 0))).collect();
        let f = build(s, false, 0, 1, &entries);
        let got = FixedArray::read(&f, HDR, &ChunkIndexContext::unfiltered(s)).unwrap();
        assert_eq!(
            got.iter().map(|r| r.address).collect::<Vec<_>>(),
            (9000..9040).collect::<Vec<_>>()
        );
    }

    #[test]
    fn exactly_one_page_is_unpaged() {
        let s = Sizes::DEFAULT;
        let entries: Vec<_> = (0..4).map(|i| Some(rec(i, None, 0))).collect();
        let f = build(s, false, 0, 2, &entries);
        let fa = FixedArray::open(&f, HDR, &ChunkIndexContext::unfiltered(s)).unwrap();
        assert_eq!(fa.header.page_entries(), None);
        assert_eq!(fa.records(&f).unwrap().len(), 4);
    }

    #[test]
    fn undefined_data_block() {
        let s = Sizes::DEFAULT;
        let mut f = vec![0u8; HDR as usize];
        f.extend_from_slice(b"FAHD");
        f.extend_from_slice(&[0, 0, 8, 10]);
        put(&mut f, 7, 8);
        put_addr(&mut f, UNDEFINED_ADDRESS, 8);
        seal(&mut f, HDR as usize);
        let got = FixedArray::read(&f, HDR, &ChunkIndexContext::unfiltered(s)).unwrap();
        assert_eq!(got, vec![ChunkRecord::UNALLOCATED; 7]);
    }

    #[test]
    fn corruption_detected() {
        let s = Sizes::DEFAULT;
        let entries: Vec<_> = (0..11).map(|i| Some(rec(2000 + i, None, 0))).collect();
        let f = build(s, false, 0, 2, &entries);
        let ctx = ChunkIndexContext::unfiltered(s);
        assert!(FixedArray::read(&f, HDR, &ctx).is_ok());
        // Header checksum.
        let mut g = f.clone();
        g[HDR as usize + 8] ^= 1;
        assert!(matches!(
            FixedArray::read(&g, HDR, &ctx),
            Err(Error::ChecksumMismatch { .. })
        ));
        // Data block checksum.
        let mut g = f.clone();
        g[DBLK as usize + 6] ^= 1;
        assert!(matches!(
            FixedArray::read(&g, HDR, &ctx),
            Err(Error::ChecksumMismatch { .. })
        ));
        // Last page checksum.
        let mut g = f.clone();
        let n = g.len();
        g[n - 5] ^= 1;
        assert!(matches!(
            FixedArray::read(&g, HDR, &ctx),
            Err(Error::ChecksumMismatch { .. })
        ));
        // Truncation.
        for cut in [HDR as usize + 10, DBLK as usize + 3, f.len() - 1] {
            assert!(FixedArray::read(&f[..cut], HDR, &ctx).is_err());
        }
    }

    #[test]
    fn mutation_never_panics() {
        let s = Sizes::DEFAULT;
        let mut entries: Vec<Option<ChunkRecord>> =
            (0..11).map(|i| Some(rec(2000 + i, Some(i), 1))).collect();
        entries[8] = None;
        entries[9] = None;
        entries[10] = None;
        let bases = [
            build(s, true, 4, 2, &entries),
            build(s, false, 0, 10, &[Some(rec(5, None, 0)); 6]),
        ];
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for base in &bases {
            for iter in 0..3000 {
                let mut g = base.clone();
                for _ in 0..1 + rng.below(4) {
                    let i = HDR as usize + rng.below(g.len() - HDR as usize);
                    g[i] = rng.next() as u8;
                }
                if iter % 2 == 0 {
                    // Re-seal the header so mutated parameters get parsed.
                    let (h, n) = (HDR as usize, FixedArrayHeader::encoded_size(s));
                    let c = crate::checksum::metadata_checksum(&g[h..h + n - 4]);
                    g[h + n - 4..h + n].copy_from_slice(&c.to_le_bytes());
                }
                if rng.below(8) == 0 {
                    g.truncate(rng.below(g.len()));
                }
                for ctx in [
                    ChunkIndexContext::filtered(s),
                    ChunkIndexContext::unfiltered(s),
                ] {
                    let _ = FixedArray::read(&g, HDR, &ctx);
                }
            }
        }
    }
}
