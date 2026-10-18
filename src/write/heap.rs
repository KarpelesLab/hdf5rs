//! Global heap writing: variable-length data is appended to an open
//! collection whose address is reserved up front, so heap IDs are known
//! immediately; a collection is written out when full or at the end.

use super::storage::{Allocator, WriteAt};
use crate::error::{Error, Result};
use crate::format::Sizes;
use crate::format::global_heap::{
    DEFAULT_COLLECTION_SIZE, GlobalHeapCollection, GlobalHeapId, GlobalHeapObject,
};

#[derive(Debug)]
struct OpenCollection {
    address: u64,
    size: u64,
    used: u64,
    objects: Vec<GlobalHeapObject>,
}

/// The writer's global heap state.
#[derive(Debug, Default)]
pub(crate) struct HeapWriter {
    current: Option<OpenCollection>,
}

impl HeapWriter {
    /// Store `data` as a new heap object and return its ID.
    pub(crate) fn add(
        &mut self,
        data: &[u8],
        sizes: Sizes,
        alloc: &mut Allocator,
        storage: &mut dyn WriteAt,
    ) -> Result<GlobalHeapId> {
        let need = GlobalHeapCollection::object_size(data.len() as u64, sizes);
        let fits = self.current.as_ref().is_some_and(|c| {
            c.used.checked_add(need).is_some_and(|u| u <= c.size) && c.objects.len() < 0xffff
        });
        if !fits {
            self.flush(sizes, storage)?;
            let header = GlobalHeapCollection::header_size(sizes);
            let size = header
                .checked_add(need)
                .ok_or(Error::LimitExceeded("global heap object size"))?
                .max(DEFAULT_COLLECTION_SIZE);
            let address = alloc.alloc(size)?;
            self.current = Some(OpenCollection {
                address,
                size,
                used: header,
                objects: Vec::new(),
            });
        }
        let c = self
            .current
            .as_mut()
            .ok_or_else(|| Error::invalid("no open global heap collection"))?;
        let index = (c.objects.len() + 1) as u16;
        c.objects.push(GlobalHeapObject {
            index,
            // The reference library writes 0 for variable-length data.
            reference_count: 0,
            data: data.to_vec(),
        });
        c.used += need;
        Ok(GlobalHeapId {
            collection_address: c.address,
            index: u32::from(index),
        })
    }

    /// Write out the open collection, if any.
    pub(crate) fn flush(&mut self, sizes: Sizes, storage: &mut dyn WriteAt) -> Result<()> {
        if let Some(c) = self.current.take() {
            let col = GlobalHeapCollection::from_objects(c.size, c.objects);
            storage.write_all_at(c.address, &col.encode(sizes)?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_and_rolls_over() {
        let sizes = Sizes::DEFAULT;
        let mut alloc = Allocator::new(0, sizes);
        let mut out: Vec<u8> = Vec::new();
        let mut h = HeapWriter::default();
        let a = h.add(b"abc", sizes, &mut alloc, &mut out).unwrap();
        let b = h.add(&[7u8; 4000], sizes, &mut alloc, &mut out).unwrap();
        let c = h.add(&[1u8; 10000], sizes, &mut alloc, &mut out).unwrap();
        h.flush(sizes, &mut out).unwrap();
        assert_eq!(a.collection_address, 0);
        assert_eq!((a.index, b.index), (1, 2));
        assert_eq!(b.collection_address, 0);
        assert_eq!(c.collection_address, 4096);
        assert_eq!(c.index, 1);
        assert_eq!(out.len() as u64, alloc.eoa());
        let col = GlobalHeapCollection::parse(&out, sizes).unwrap();
        assert_eq!(col.get(1), Some(&b"abc"[..]));
        assert_eq!(col.get(2).map(<[u8]>::len), Some(4000));
        let col2 = GlobalHeapCollection::parse(&out[4096..], sizes).unwrap();
        assert_eq!(col2.get(1).map(<[u8]>::len), Some(10000));
    }
}
