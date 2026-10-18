//! Local Heap (spec: "Disk Format: Level 1D - Local Heap").
//!
//! A local heap holds the link names (and soft-link values) of an
//! old-style group. Strings are NUL-terminated and addressed by their byte
//! offset within the heap's data segment.

use std::borrow::Cow;

use super::{ByteReader, Sizes, UNDEFINED_ADDRESS};
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Signature of a local heap header.
pub const LOCAL_HEAP_SIGNATURE: [u8; 4] = *b"HEAP";

/// One block of the local heap's free list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeBlock {
    /// Offset of the block within the data segment.
    pub offset: u64,
    /// Size of the block in bytes (including its two header fields).
    pub size: u64,
}

/// A local heap: its header fields plus the loaded data segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalHeap {
    /// Size in bytes of the data segment, as declared in the header.
    pub data_segment_size: u64,
    /// Offset of the first free block, or `None` when the free list is
    /// empty (encoded as the undefined address).
    pub free_list_head: Option<u64>,
    /// File address of the data segment.
    pub data_segment_address: u64,
    sizes: Sizes,
    data: Vec<u8>,
}

impl LocalHeap {
    /// Size in bytes of the encoded header for the given widths.
    pub fn header_size(sizes: Sizes) -> usize {
        8 + 2 * sizes.length as usize + sizes.offset as usize
    }

    /// Read the heap header at `addr` and load its data segment.
    ///
    /// The data segment must lie entirely within the store.
    pub fn read(store: &(impl ReadAt + ?Sized), addr: u64, sizes: Sizes) -> Result<Self> {
        let header = store.read_vec_at(addr, Self::header_size(sizes))?;
        let mut r = ByteReader::with_sizes(&header, sizes);
        r.expect_signature(&LOCAL_HEAP_SIGNATURE)?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!("local heap version {version}")));
        }
        r.skip(3)?;
        let data_segment_size = r.read_length()?;
        let free_raw = r.read_bytes(sizes.length as usize)?;
        let free_list_head = if free_raw.iter().all(|&b| b == 0xff) {
            None
        } else {
            Some(ByteReader::new(free_raw).read_uint(free_raw.len())?)
        };
        let data_segment_address = r.read_offset()?;
        if data_segment_address == UNDEFINED_ADDRESS {
            return Err(Error::invalid("local heap data segment address undefined"));
        }
        if data_segment_size > store.size()? {
            return Err(Error::LimitExceeded("local heap larger than file"));
        }
        let len = usize::try_from(data_segment_size)
            .map_err(|_| Error::LimitExceeded("local heap size"))?;
        let data = store.read_vec_at(data_segment_address, len)?;
        Ok(LocalHeap {
            data_segment_size,
            free_list_head,
            data_segment_address,
            sizes,
            data,
        })
    }

    /// The raw data segment.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The bytes of the NUL-terminated string at `offset`, without the NUL.
    ///
    /// Fails if `offset` is outside the data segment or no NUL terminator
    /// follows it within the segment.
    pub fn get_bytes(&self, offset: u64) -> Result<&[u8]> {
        let start = usize::try_from(offset)
            .ok()
            .filter(|&s| s < self.data.len())
            .ok_or_else(|| Error::invalid("local heap offset out of range"))?;
        let tail = &self.data[start..];
        let end = tail
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| Error::invalid("unterminated string in local heap"))?;
        Ok(&tail[..end])
    }

    /// The string at `offset`, which must be valid UTF-8.
    pub fn get_str(&self, offset: u64) -> Result<&str> {
        std::str::from_utf8(self.get_bytes(offset)?)
            .map_err(|_| Error::invalid("local heap string is not valid UTF-8"))
    }

    /// The string at `offset`, replacing invalid UTF-8 sequences with
    /// U+FFFD.
    pub fn get_string_lossy(&self, offset: u64) -> Result<Cow<'_, str>> {
        Ok(String::from_utf8_lossy(self.get_bytes(offset)?))
    }

    /// Walk the free list.
    ///
    /// Each block must lie within the data segment and be at least
    /// 2 × "Size of Lengths" bytes; a next-offset of 1 ends the list. The
    /// walk is bounded by the number of blocks that could fit, so a cyclic
    /// list yields an error.
    pub fn free_blocks(&self) -> Result<Vec<FreeBlock>> {
        let mut out = Vec::new();
        let Some(mut offset) = self.free_list_head else {
            return Ok(out);
        };
        let field = self.sizes.length as usize;
        let min = 2 * field;
        let max_blocks = self.data.len() / min.max(1) + 1;
        loop {
            if out.len() >= max_blocks {
                return Err(Error::invalid("local heap free list is cyclic"));
            }
            let start = usize::try_from(offset)
                .ok()
                .filter(|&s| s.checked_add(min).is_some_and(|e| e <= self.data.len()))
                .ok_or_else(|| Error::invalid("local heap free block out of range"))?;
            let mut r = ByteReader::with_sizes(&self.data[start..], self.sizes);
            let next = r.read_length()?;
            let size = r.read_length()?;
            let fits = offset
                .checked_add(size)
                .is_some_and(|e| e <= self.data.len() as u64);
            if size < min as u64 || !fits {
                return Err(Error::invalid("local heap free block has bad size"));
            }
            out.push(FreeBlock { offset, size });
            if next == 1 {
                return Ok(out);
            }
            offset = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::{FileBuilder, Rng};

    fn heap_file(sizes: Sizes) -> FileBuilder {
        let mut f = FileBuilder::new(sizes);
        // Segment: "" at 0, "foo" at 8, then a free block at 16 of 16+ bytes.
        let mut seg = vec![0u8; 8];
        seg.extend(b"foo\0\0\0\0\0");
        let l = sizes.length;
        let mut free = FileBuilder::uint(1, l);
        free.extend(FileBuilder::uint(2 * l as u64 + 8, l));
        free.resize(2 * l as usize + 8, 0);
        seg.extend(free);
        let hs = LocalHeap::header_size(sizes) as u64;
        f.local_heap(16, 16 + hs, &seg, 16);
        f
    }

    #[test]
    fn reads_heap_and_strings() {
        for sizes in [
            Sizes::DEFAULT,
            Sizes {
                offset: 4,
                length: 4,
            },
        ] {
            let f = heap_file(sizes);
            let h = LocalHeap::read(&f.data, 16, sizes).unwrap();
            assert_eq!(h.get_str(0).unwrap(), "");
            assert_eq!(h.get_str(8).unwrap(), "foo");
            assert_eq!(h.get_bytes(9).unwrap(), b"oo");
            assert_eq!(h.get_string_lossy(8).unwrap(), "foo");
            assert!(h.get_bytes(h.data().len() as u64).is_err());
            assert!(h.get_bytes(u64::MAX).is_err());
            let fb = h.free_blocks().unwrap();
            assert_eq!(fb.len(), 1);
            assert_eq!(fb[0].offset, 16);
        }
    }

    #[test]
    fn unterminated_and_non_utf8() {
        let mut f = FileBuilder::new(Sizes::DEFAULT);
        f.local_heap(0, 64, b"\xffab\0xyz", u64::MAX);
        let h = LocalHeap::read(&f.data, 0, Sizes::DEFAULT).unwrap();
        assert_eq!(h.free_list_head, None);
        assert!(h.free_blocks().unwrap().is_empty());
        assert!(h.get_str(0).is_err());
        assert_eq!(h.get_string_lossy(0).unwrap(), "\u{fffd}ab");
        assert!(h.get_bytes(4).is_err());
    }

    #[test]
    fn rejects_bad_headers() {
        let f = heap_file(Sizes::DEFAULT);
        let mut bad = f.data.clone();
        bad[16] = b'X';
        assert!(LocalHeap::read(&bad, 16, Sizes::DEFAULT).is_err());
        let mut bad = f.data.clone();
        bad[20] = 1;
        assert!(LocalHeap::read(&bad, 16, Sizes::DEFAULT).is_err());
        // Huge segment size.
        let mut bad = f.data.clone();
        bad[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(LocalHeap::read(&bad, 16, Sizes::DEFAULT).is_err());
        assert!(LocalHeap::read(&f.data[..40], 16, Sizes::DEFAULT).is_err());
    }

    #[test]
    fn cyclic_free_list() {
        let mut seg = vec![0u8; 32];
        seg[0..8].copy_from_slice(&16u64.to_le_bytes());
        seg[8..16].copy_from_slice(&16u64.to_le_bytes());
        seg[16..24].copy_from_slice(&0u64.to_le_bytes());
        seg[24..32].copy_from_slice(&16u64.to_le_bytes());
        let mut f = FileBuilder::new(Sizes::DEFAULT);
        f.local_heap(0, 64, &seg, 0);
        let h = LocalHeap::read(&f.data, 0, Sizes::DEFAULT).unwrap();
        assert!(h.free_blocks().is_err());
    }

    #[test]
    fn mutations_never_panic() {
        let f = heap_file(Sizes::DEFAULT);
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..5000 {
            let d = rng.mutate(&f.data);
            if let Ok(h) = LocalHeap::read(&d, 16, Sizes::DEFAULT) {
                let _ = h.free_blocks();
                for o in 0..40 {
                    let _ = h.get_string_lossy(o);
                }
            }
        }
    }
}
