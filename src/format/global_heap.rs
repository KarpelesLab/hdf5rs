//! Global Heap (spec: "Disk Format: Level 1E - Global Heap").
//!
//! Global heap collections store variable-length data and some reference
//! targets; they are addressed through a [`GlobalHeapId`].

use super::{ByteReader, Sizes, UNDEFINED_ADDRESS};
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Signature of a global heap collection.
pub const GLOBAL_HEAP_SIGNATURE: [u8; 4] = *b"GCOL";

/// Upper bound on a collection's size (sanity limit, besides file size).
pub const MAX_COLLECTION_SIZE: u64 = 1 << 32;

/// A pointer into a global heap collection (spec: "Global Heap ID").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalHeapId {
    /// Address of the collection holding the object.
    pub collection_address: u64,
    /// Index of the object within the collection.
    pub index: u32,
}

impl GlobalHeapId {
    /// Encoded size in bytes for the given offset width.
    pub fn encoded_size(offset_size: u8) -> usize {
        offset_size as usize + 4
    }

    /// Parse an ID: a collection address of "Size of Offsets" bytes
    /// followed by a 32-bit object index.
    pub fn parse(r: &mut ByteReader<'_>) -> Result<Self> {
        Ok(GlobalHeapId {
            collection_address: r.read_offset()?,
            index: r.read_u32()?,
        })
    }

    /// Whether this ID refers to nothing (undefined address or a zero
    /// collection address, as used e.g. for empty variable-length values).
    pub fn is_null(&self) -> bool {
        self.collection_address == UNDEFINED_ADDRESS || self.collection_address == 0
    }
}

/// One object stored in a collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalHeapObject {
    /// Heap object index (never 0; 0 denotes free space).
    pub index: u16,
    /// Reference count.
    pub reference_count: u16,
    /// Object data (exactly "Object Size" bytes, padding removed).
    pub data: Vec<u8>,
}

/// A parsed global heap collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalHeapCollection {
    /// Total size of the collection in bytes, as declared.
    pub collection_size: u64,
    /// Objects, sorted by index.
    objects: Vec<GlobalHeapObject>,
}

impl GlobalHeapCollection {
    /// Read and parse the collection at `addr`.
    pub fn read(store: &(impl ReadAt + ?Sized), addr: u64, sizes: Sizes) -> Result<Self> {
        let head_len = 8 + sizes.length as usize;
        let head = store.read_vec_at(addr, head_len)?;
        let mut r = ByteReader::with_sizes(&head, sizes);
        r.expect_signature(&GLOBAL_HEAP_SIGNATURE)?;
        let version = r.read_u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!(
                "global heap collection version {version}"
            )));
        }
        r.skip(3)?;
        let size = r.read_length()?;
        if size < head_len as u64 {
            return Err(Error::invalid("global heap collection too small"));
        }
        if size > MAX_COLLECTION_SIZE || size > store.size()? {
            return Err(Error::LimitExceeded("global heap collection size"));
        }
        let len = usize::try_from(size).map_err(|_| Error::LimitExceeded("collection size"))?;
        Self::parse(&store.read_vec_at(addr, len)?, sizes)
    }

    /// Parse a collection from `data`, which starts at the signature and
    /// is at least "Collection Size" bytes long.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        r.expect_signature(&GLOBAL_HEAP_SIGNATURE)?;
        let version = r.read_u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!(
                "global heap collection version {version}"
            )));
        }
        r.skip(3)?;
        let collection_size = r.read_length()?;
        let end = usize::try_from(collection_size)
            .ok()
            .filter(|&e| e <= data.len() && e >= r.position())
            .ok_or_else(|| Error::invalid("global heap collection size out of range"))?;
        let mut r = ByteReader::with_sizes(&data[..end], sizes);
        // The collection header and each object header are padded to a
        // multiple of 8 bytes (only visible with 4-byte lengths, where they
        // are 12 bytes long; verified on files written by the reference
        // library).
        let align8 = |n: usize| n.div_ceil(8) * 8;
        r.seek(align8(8 + sizes.length as usize).min(end))?;
        let obj_head = align8(8 + sizes.length as usize);
        let mut objects = Vec::new();
        // A trailing free-space object whose header does not fit is implied.
        while r.remaining() >= obj_head {
            let start = r.position();
            let index = r.read_u16()?;
            if index == 0 {
                break;
            }
            let reference_count = r.read_u16()?;
            r.skip(4)?;
            let size = usize::try_from(r.read_length()?)
                .map_err(|_| Error::invalid("global heap object too large"))?;
            r.seek(start + obj_head)?;
            let data = r.read_bytes(size)?.to_vec();
            // Data is padded to a multiple of 8; the padding of the last
            // object may be clipped by the collection end.
            let pad = (8 - size % 8) % 8;
            r.skip(pad.min(r.remaining()))?;
            objects.push(GlobalHeapObject {
                index,
                reference_count,
                data,
            });
        }
        objects.sort_by_key(|o| o.index);
        if objects.windows(2).any(|w| w[0].index == w[1].index) {
            return Err(Error::invalid("duplicate global heap object index"));
        }
        Ok(GlobalHeapCollection {
            collection_size,
            objects,
        })
    }

    /// The data of object `index`, if present.
    pub fn get(&self, index: u32) -> Option<&[u8]> {
        self.object(index).map(|o| o.data.as_slice())
    }

    /// The full object record for `index`, if present.
    pub fn object(&self, index: u32) -> Option<&GlobalHeapObject> {
        let index = u16::try_from(index).ok()?;
        self.objects
            .binary_search_by_key(&index, |o| o.index)
            .ok()
            .and_then(|i| self.objects.get(i))
    }

    /// All objects, sorted by index.
    pub fn objects(&self) -> &[GlobalHeapObject] {
        &self.objects
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::{FileBuilder, Rng};

    fn object(sizes: Sizes, index: u16, data: &[u8]) -> Vec<u8> {
        let mut v = index.to_le_bytes().to_vec();
        v.extend(1u16.to_le_bytes());
        v.extend([0; 4]);
        v.extend(FileBuilder::uint(data.len() as u64, sizes.length));
        // The header and the data are padded to a multiple of 8.
        v.resize(v.len().div_ceil(8) * 8, 0);
        let mut d = data.to_vec();
        d.resize(d.len().div_ceil(8) * 8, 0);
        v.extend(d);
        v
    }

    fn collection(sizes: Sizes, size: u64, objs: &[(u16, &[u8])], free: bool) -> Vec<u8> {
        let mut v = b"GCOL".to_vec();
        v.extend([1, 0, 0, 0]);
        v.extend(FileBuilder::uint(size, sizes.length));
        v.resize(v.len().div_ceil(8) * 8, 0);
        for (i, d) in objs {
            v.extend(object(sizes, *i, d));
        }
        if free {
            let rest = size - v.len() as u64;
            let mut f = object(sizes, 0, &[]);
            f.truncate(8);
            f.extend(FileBuilder::uint(rest, sizes.length));
            v.extend(f);
        }
        v.resize(size as usize, 0);
        v
    }

    #[test]
    fn reads_collection() {
        for sizes in [
            Sizes::DEFAULT,
            Sizes {
                offset: 4,
                length: 4,
            },
        ] {
            let mut file = vec![0u8; 100];
            file.extend(collection(
                sizes,
                4096,
                &[(1, b"hello"), (3, b"12345678"), (2, b"")],
                true,
            ));
            let c = GlobalHeapCollection::read(&file, 100, sizes).unwrap();
            assert_eq!(c.collection_size, 4096);
            assert_eq!(c.get(1).unwrap(), b"hello");
            assert_eq!(c.get(2).unwrap(), b"");
            assert_eq!(c.get(3).unwrap(), b"12345678");
            assert!(c.get(0).is_none());
            assert!(c.get(4).is_none());
            assert!(c.get(0x1_0001).is_none());
            assert_eq!(c.objects().len(), 3);
            assert_eq!(c.object(1).unwrap().reference_count, 1);
        }
    }

    #[test]
    fn implied_free_space_and_exact_fit() {
        let sizes = Sizes::DEFAULT;
        // 16 header + 16 + 8 data = 40, leave 8 bytes: implied free header.
        let c = collection(sizes, 48, &[(1, b"abcdefgh")], false);
        let g = GlobalHeapCollection::read(&c, 0, sizes).unwrap();
        assert_eq!(g.get(1).unwrap(), b"abcdefgh");
        // Exact fit with no free space.
        let c = collection(sizes, 40, &[(1, b"abc")], false);
        assert_eq!(
            GlobalHeapCollection::read(&c, 0, sizes)
                .unwrap()
                .get(1)
                .unwrap(),
            b"abc"
        );
    }

    #[test]
    fn rejects_bad() {
        let sizes = Sizes::DEFAULT;
        let c = collection(sizes, 64, &[(1, b"x")], true);
        let mut bad = c.clone();
        bad[4] = 0;
        assert!(GlobalHeapCollection::read(&bad, 0, sizes).is_err());
        // Collection larger than file.
        let mut bad = c.clone();
        bad[8] = 0xff;
        assert!(GlobalHeapCollection::read(&bad, 0, sizes).is_err());
        // Object overrunning the collection.
        let mut bad = c.clone();
        bad[24] = 0xff;
        assert!(GlobalHeapCollection::read(&bad, 0, sizes).is_err());
        // Duplicate index.
        let dup = collection(sizes, 64, &[(1, b"x"), (1, b"y")], false);
        assert!(GlobalHeapCollection::read(&dup, 0, sizes).is_err());
        // Too small.
        let mut bad = c.clone();
        bad[8] = 4;
        assert!(GlobalHeapCollection::read(&bad, 0, sizes).is_err());
    }

    #[test]
    fn heap_id() {
        let mut d = 0x1234u64.to_le_bytes().to_vec();
        d.extend(7u32.to_le_bytes());
        let id = GlobalHeapId::parse(&mut ByteReader::new(&d)).unwrap();
        assert_eq!(
            id,
            GlobalHeapId {
                collection_address: 0x1234,
                index: 7
            }
        );
        assert!(!id.is_null());
        assert_eq!(GlobalHeapId::encoded_size(8), d.len());
        assert!(GlobalHeapId::parse(&mut ByteReader::new(&d[..10])).is_err());
    }

    #[test]
    fn mutations_never_panic() {
        let sizes = Sizes::DEFAULT;
        let c = collection(sizes, 128, &[(1, b"hello"), (2, b"world!!!!")], true);
        let mut rng = Rng(7);
        for _ in 0..5000 {
            let d = rng.mutate(&c);
            if let Ok(g) = GlobalHeapCollection::read(&d, 0, sizes) {
                let _ = g.get(1);
            }
        }
    }
}
