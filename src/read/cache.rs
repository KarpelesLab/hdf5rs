//! Per-dataset caches: decoded chunks (LRU by bytes) and the chunk index.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use super::chunked::FullIndex;
use crate::format::chunk_index::ChunkRecord;

/// Default capacity of a dataset's decoded-chunk cache: 16 MiB.
pub const DEFAULT_CHUNK_CACHE_BYTES: usize = 16 << 20;

/// Most chunk records remembered from individual index lookups before the
/// whole index is loaded instead.
const MAX_POINT_RECORDS: usize = 1024;

/// A cache of decoded chunks and chunk index lookups for **one** chunked
/// dataset.
///
/// Decompressing a chunk is usually the most expensive part of a read;
/// with a cache, repeated small reads that hit the same chunks decode each
/// chunk once. Decoded chunks are kept up to a byte capacity and evicted
/// least recently used first; chunks larger than the capacity are never
/// cached, and a capacity of 0 disables chunk caching. Unfiltered chunks
/// are read straight from the file and not cached.
///
/// The chunk index is loaded lazily: reads touching few chunks look them
/// up individually (and remember the answers); larger reads, or many
/// small ones, load the whole index once and keep it.
///
/// Every [`Dataset`](crate::Dataset) owns one (shared by its clones), sized
/// by [`FileAccess::chunk_cache_bytes`](crate::FileAccess); keep the
/// dataset handle around between reads to benefit from it (opening the
/// dataset again starts with an empty cache). When using
/// [`DatasetStorage`](super::DatasetStorage) directly, attach one with
/// [`with_chunk_cache`](super::DatasetStorage::with_chunk_cache); a cache
/// must only ever be used with a single dataset of a single file, since
/// entries are keyed by file address and chunk coordinates.
///
/// `ChunkCache` is `Send + Sync`; concurrent readers share it.
pub struct ChunkCache {
    chunks: Mutex<Chunks>,
    index: Mutex<IndexState>,
}

/// Counters of a [`ChunkCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ChunkCacheStats {
    /// Chunk reads served from the cache.
    pub hits: u64,
    /// Chunk reads that had to be decoded.
    pub misses: u64,
    /// Decoded chunks currently cached.
    pub chunks: usize,
    /// Bytes of decoded chunks currently cached.
    pub bytes: usize,
    /// Capacity in bytes.
    pub capacity: usize,
    /// Whether the whole chunk index is loaded.
    pub index_loaded: bool,
}

struct Entry {
    data: Arc<Vec<u8>>,
    tick: u64,
}

struct Chunks {
    /// Generation of the filter registry that decoded the cached chunks.
    registry: u64,
    capacity: usize,
    bytes: usize,
    tick: u64,
    map: HashMap<u64, Entry>,
    /// Recency order: tick -> chunk address.
    order: BTreeMap<u64, u64>,
    hits: u64,
    misses: u64,
}

#[derive(Default)]
pub(crate) struct IndexState {
    pub full: Option<Arc<FullIndex>>,
    pub points: HashMap<Vec<u64>, ChunkRecord>,
    /// Individual lookups performed so far.
    pub lookups: usize,
}

impl IndexState {
    /// Whether further individual lookups are still worthwhile.
    pub fn points_allowed(&self, more: usize) -> bool {
        self.lookups.saturating_add(more) <= MAX_POINT_RECORDS
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl ChunkCache {
    /// A cache holding up to `capacity_bytes` of decoded chunks.
    pub fn new(capacity_bytes: usize) -> Self {
        ChunkCache {
            chunks: Mutex::new(Chunks {
                registry: 0,
                capacity: capacity_bytes,
                bytes: 0,
                tick: 0,
                map: HashMap::new(),
                order: BTreeMap::new(),
                hits: 0,
                misses: 0,
            }),
            index: Mutex::new(IndexState::default()),
        }
    }

    /// The capacity in bytes.
    pub fn capacity(&self) -> usize {
        lock(&self.chunks).capacity
    }

    /// Change the capacity, evicting chunks as needed (0 disables chunk
    /// caching).
    pub fn set_capacity(&self, capacity_bytes: usize) {
        let mut c = lock(&self.chunks);
        c.capacity = capacity_bytes;
        c.evict();
    }

    /// Drop every cached chunk and the cached chunk index.
    pub fn clear(&self) {
        let mut c = lock(&self.chunks);
        c.map.clear();
        c.order.clear();
        c.bytes = 0;
        drop(c);
        *lock(&self.index) = IndexState::default();
    }

    /// Current counters.
    pub fn stats(&self) -> ChunkCacheStats {
        let c = lock(&self.chunks);
        ChunkCacheStats {
            hits: c.hits,
            misses: c.misses,
            chunks: c.map.len(),
            bytes: c.bytes,
            capacity: c.capacity,
            index_loaded: lock(&self.index).full.is_some(),
        }
    }

    /// The decoded chunk stored at `address`, if cached by a read using
    /// the filter registry of generation `registry`.
    pub(crate) fn get(&self, address: u64, registry: u64) -> Option<Arc<Vec<u8>>> {
        let mut c = lock(&self.chunks);
        c.tick += 1;
        let tick = c.tick;
        if c.registry != registry {
            c.misses += 1;
            return None;
        }
        let Some(e) = c.map.get_mut(&address) else {
            c.misses += 1;
            return None;
        };
        let old = std::mem::replace(&mut e.tick, tick);
        let data = e.data.clone();
        c.order.remove(&old);
        c.order.insert(tick, address);
        c.hits += 1;
        Some(data)
    }

    /// Whether a chunk of `len` bytes would be cached at all.
    pub(crate) fn accepts(&self, len: usize) -> bool {
        len <= lock(&self.chunks).capacity
    }

    /// Remember the decoded chunk stored at `address`, decoded with the
    /// filter registry of generation `registry` (entries from another
    /// registry are dropped).
    pub(crate) fn insert(&self, address: u64, registry: u64, data: Arc<Vec<u8>>) {
        let mut c = lock(&self.chunks);
        let len = data.len();
        if len > c.capacity {
            return;
        }
        if c.registry != registry {
            c.map.clear();
            c.order.clear();
            c.bytes = 0;
            c.registry = registry;
        }
        c.tick += 1;
        let tick = c.tick;
        if let Some(old) = c.map.insert(address, Entry { data, tick }) {
            c.order.remove(&old.tick);
            c.bytes -= old.data.len();
        }
        c.order.insert(tick, address);
        c.bytes += len;
        c.evict();
    }

    pub(crate) fn index(&self) -> MutexGuard<'_, IndexState> {
        lock(&self.index)
    }
}

impl Chunks {
    fn evict(&mut self) {
        while self.bytes > self.capacity {
            let Some((_, addr)) = self.order.pop_first() else {
                break;
            };
            if let Some(e) = self.map.remove(&addr) {
                self.bytes -= e.data.len();
            }
        }
    }
}

impl Default for ChunkCache {
    fn default() -> Self {
        ChunkCache::new(DEFAULT_CHUNK_CACHE_BYTES)
    }
}

impl fmt::Debug for ChunkCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkCache")
            .field("stats", &self.stats())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_by_bytes() {
        let c = ChunkCache::new(100);
        c.insert(1, 7, Arc::new(vec![0; 40]));
        c.insert(2, 7, Arc::new(vec![0; 40]));
        assert!(c.get(1, 7).is_some()); // 1 is now most recent
        c.insert(3, 7, Arc::new(vec![0; 40])); // evicts 2
        assert!(c.get(2, 7).is_none());
        assert!(c.get(1, 7).is_some() && c.get(3, 7).is_some());
        c.insert(4, 7, Arc::new(vec![0; 101])); // too big: ignored
        assert!(c.get(4, 7).is_none());
        let s = c.stats();
        assert_eq!((s.chunks, s.bytes, s.capacity), (2, 80, 100));
        c.set_capacity(50);
        assert_eq!(c.stats().chunks, 1);
        c.set_capacity(0);
        assert_eq!(c.stats().bytes, 0);
        c.insert(5, 7, Arc::new(vec![0; 1]));
        assert!(c.get(5, 7).is_none());
        c.set_capacity(10);
        c.insert(5, 7, Arc::new(vec![0; 1]));
        c.insert(5, 7, Arc::new(vec![0; 2])); // replace
        assert_eq!(c.stats().bytes, 2);
        assert!(c.get(5, 8).is_none()); // other registry
        c.insert(6, 8, Arc::new(vec![0; 1])); // drops registry 7 entries
        assert!(c.get(5, 7).is_none());
        assert!(c.get(6, 8).is_some());
        c.clear();
        assert_eq!(c.stats().chunks, 0);
    }
}
