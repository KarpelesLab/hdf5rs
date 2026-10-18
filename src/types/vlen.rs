//! Variable-length data and other global-heap-backed values.
//!
//! A variable-length element (sequence or string, spec "Variable-Length"
//! datatype class and "Disk Format: Level 2B - Data Object Data Storage")
//! is stored inline as a 32-bit little-endian count followed by a
//! [`GlobalHeapId`] (collection address of "Size of Offsets" bytes plus a
//! 32-bit object index). The count is the number of base-type elements for
//! sequences and the number of bytes for strings; the heap object holds the
//! packed elements. An empty value has count 0 and may carry a null heap ID.

use std::sync::{Arc, Mutex};

use super::DecodeContext;
use crate::error::{Error, Result};
use crate::format::global_heap::{GlobalHeapCollection, GlobalHeapId};
use crate::format::{ByteReader, Sizes};
use crate::io::ReadAt;

/// Access to global heap objects, needed to decode variable-length data
/// and dataset region references.
pub trait VlenResolver {
    /// Return the data of the global heap object `id`.
    fn global_heap_object(&self, id: &GlobalHeapId) -> Result<Vec<u8>>;
}

impl<T: VlenResolver + ?Sized> VlenResolver for &T {
    fn global_heap_object(&self, id: &GlobalHeapId) -> Result<Vec<u8>> {
        (**self).global_heap_object(id)
    }
}

/// Number of parsed collections kept by [`GlobalHeapResolver`].
const CACHE_ENTRIES: usize = 16;

/// A [`VlenResolver`] reading global heap collections from a store,
/// keeping the most recently used collections parsed in memory.
///
/// `R` may be any [`ReadAt`], including `&dyn ReadAt`. Addresses are
/// relative to the store (wrap the file in an
/// [`OffsetReader`](crate::io::OffsetReader) when the superblock has a
/// non-zero base address).
#[derive(Debug)]
pub struct GlobalHeapResolver<R> {
    store: R,
    sizes: Sizes,
    cache: Mutex<Vec<(u64, Arc<GlobalHeapCollection>)>>,
}

impl<R: ReadAt> GlobalHeapResolver<R> {
    /// Create a resolver over `store` with the file's offset/length widths.
    pub fn new(store: R, sizes: Sizes) -> Self {
        GlobalHeapResolver {
            store,
            sizes,
            cache: Mutex::new(Vec::new()),
        }
    }

    /// The offset/length widths in use.
    pub fn sizes(&self) -> Sizes {
        self.sizes
    }

    /// The parsed collection at `addr` (cached).
    pub fn collection(&self, addr: u64) -> Result<Arc<GlobalHeapCollection>> {
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(i) = cache.iter().position(|(a, _)| *a == addr) {
                let entry = cache.remove(i);
                let coll = Arc::clone(&entry.1);
                cache.push(entry);
                return Ok(coll);
            }
        }
        let coll = Arc::new(GlobalHeapCollection::read(&self.store, addr, self.sizes)?);
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= CACHE_ENTRIES {
            cache.remove(0);
        }
        cache.push((addr, Arc::clone(&coll)));
        Ok(coll)
    }
}

impl<R: ReadAt> VlenResolver for GlobalHeapResolver<R> {
    fn global_heap_object(&self, id: &GlobalHeapId) -> Result<Vec<u8>> {
        if id.is_null() {
            return Err(Error::invalid("null global heap ID"));
        }
        let coll = self.collection(id.collection_address)?;
        coll.get(id.index).map(<[u8]>::to_vec).ok_or_else(|| {
            Error::invalid(format!(
                "global heap object {} missing from collection at {:#x}",
                id.index, id.collection_address
            ))
        })
    }
}

/// The inline part of a variable-length element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VlenElement {
    /// Number of base elements (sequences) or bytes (strings).
    pub length: u32,
    /// Location of the data in the global heap.
    pub heap_id: GlobalHeapId,
}

impl VlenElement {
    /// Encoded size for the given offset width.
    pub fn encoded_size(sizes: Sizes) -> usize {
        4 + GlobalHeapId::encoded_size(sizes.offset)
    }

    /// Parse a variable-length element; `bytes` must be exactly
    /// [`encoded_size`](Self::encoded_size) long.
    pub fn parse(bytes: &[u8], sizes: Sizes) -> Result<VlenElement> {
        if bytes.len() != Self::encoded_size(sizes) {
            return Err(Error::invalid(format!(
                "variable-length element of {} bytes, expected {} for {}-byte offsets",
                bytes.len(),
                Self::encoded_size(sizes),
                sizes.offset
            )));
        }
        let mut r = ByteReader::with_sizes(bytes, sizes);
        let length = r.read_u32()?;
        let heap_id = GlobalHeapId::parse(&mut r)?;
        Ok(VlenElement { length, heap_id })
    }
}

/// Parse a global heap ID occupying all of `bytes`.
pub(crate) fn parse_heap_id(bytes: &[u8], sizes: Sizes) -> Result<GlobalHeapId> {
    if bytes.len() != GlobalHeapId::encoded_size(sizes.offset) {
        return Err(Error::invalid(format!(
            "global heap ID of {} bytes, expected {}",
            bytes.len(),
            GlobalHeapId::encoded_size(sizes.offset)
        )));
    }
    GlobalHeapId::parse(&mut ByteReader::with_sizes(bytes, sizes))
}

/// Fetch a heap object through the context's resolver.
pub(crate) fn fetch(ctx: &DecodeContext<'_>, id: &GlobalHeapId) -> Result<Vec<u8>> {
    let resolver = ctx.resolver.ok_or_else(|| {
        Error::unsupported("global heap data needs a VlenResolver in the DecodeContext")
    })?;
    resolver.global_heap_object(id)
}

/// Fetch the packed data of a variable-length element whose base elements
/// are `base_size` bytes: exactly `length * base_size` bytes.
pub(crate) fn fetch_vlen(bytes: &[u8], base_size: u32, ctx: &DecodeContext<'_>) -> Result<Vec<u8>> {
    let el = VlenElement::parse(bytes, ctx.sizes)?;
    if el.length == 0 {
        return Ok(Vec::new());
    }
    if el.heap_id.is_null() {
        return Err(Error::invalid(
            "non-empty variable-length element with a null heap ID",
        ));
    }
    if base_size == 0 {
        return Err(Error::invalid("variable-length base type of size zero"));
    }
    let need = usize::try_from(u64::from(el.length) * u64::from(base_size))
        .map_err(|_| Error::LimitExceeded("variable-length element size"))?;
    let mut data = fetch(ctx, &el.heap_id)?;
    if data.len() < need {
        return Err(Error::invalid(format!(
            "variable-length element needs {need} bytes, heap object has {}",
            data.len()
        )));
    }
    data.truncate(need);
    Ok(data)
}
