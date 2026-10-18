//! Helpers for unit tests: hand-building byte-exact structures in memory.

use super::Sizes;

/// A growable in-memory "file" with little-endian field writers.
#[derive(Debug, Clone)]
pub(crate) struct FileBuilder {
    pub data: Vec<u8>,
    pub sizes: Sizes,
}

impl FileBuilder {
    pub fn new(sizes: Sizes) -> Self {
        FileBuilder {
            data: Vec::new(),
            sizes,
        }
    }

    /// Write `bytes` at `addr`, growing the buffer with zeros as needed.
    pub fn put(&mut self, addr: u64, bytes: &[u8]) {
        let addr = addr as usize;
        if self.data.len() < addr + bytes.len() {
            self.data.resize(addr + bytes.len(), 0);
        }
        self.data[addr..addr + bytes.len()].copy_from_slice(bytes);
    }

    /// Encode an unsigned value with the given width.
    pub fn uint(value: u64, width: u8) -> Vec<u8> {
        let mut v = value.to_le_bytes().to_vec();
        v.resize(width as usize, 0);
        v
    }

    pub fn off(&self, value: u64) -> Vec<u8> {
        if value == u64::MAX {
            vec![0xff; self.sizes.offset as usize]
        } else {
            Self::uint(value, self.sizes.offset)
        }
    }

    pub fn len(&self, value: u64) -> Vec<u8> {
        Self::uint(value, self.sizes.length)
    }

    /// Encode a symbol table entry with cache type 0.
    pub fn entry(&self, name_offset: u64, header: u64) -> Vec<u8> {
        let mut v = self.off(name_offset);
        v.extend(self.off(header));
        v.extend([0u8; 8]);
        v.extend([0u8; 16]);
        v
    }

    /// Write a local heap header at `addr` with its data segment at
    /// `data_addr`.
    pub fn local_heap(&mut self, addr: u64, data_addr: u64, segment: &[u8], free_head: u64) {
        let mut h = b"HEAP".to_vec();
        h.extend([0, 0, 0, 0]);
        h.extend(self.len(segment.len() as u64));
        if free_head == u64::MAX {
            h.extend(vec![0xff; self.sizes.length as usize]);
        } else {
            h.extend(self.len(free_head));
        }
        h.extend(self.off(data_addr));
        self.put(addr, &h);
        self.put(data_addr, segment);
    }

    /// Write a symbol table node with the given encoded entries.
    pub fn snod(&mut self, addr: u64, entries: &[Vec<u8>]) {
        let mut h = b"SNOD".to_vec();
        h.extend([1, 0]);
        h.extend((entries.len() as u16).to_le_bytes());
        for e in entries {
            h.extend(e);
        }
        self.put(addr, &h);
    }

    /// Write a v1 B-tree node. `keys` are pre-encoded (N+1 of them).
    pub fn btree(
        &mut self,
        addr: u64,
        node_type: u8,
        level: u8,
        keys: &[Vec<u8>],
        children: &[u64],
    ) {
        assert_eq!(keys.len(), children.len() + 1);
        let mut h = b"TREE".to_vec();
        h.push(node_type);
        h.push(level);
        h.extend((children.len() as u16).to_le_bytes());
        h.extend(self.off(u64::MAX));
        h.extend(self.off(u64::MAX));
        for (i, c) in children.iter().enumerate() {
            h.extend(&keys[i]);
            h.extend(self.off(*c));
        }
        h.extend(&keys[children.len()]);
        self.put(addr, &h);
    }

    /// Encode a group-node key (heap offset).
    pub fn group_key(&self, heap_offset: u64) -> Vec<u8> {
        self.len(heap_offset)
    }

    /// Encode a chunk-node key; a trailing zero offset is appended.
    pub fn chunk_key(size: u32, mask: u32, offsets: &[u64]) -> Vec<u8> {
        let mut v = size.to_le_bytes().to_vec();
        v.extend(mask.to_le_bytes());
        for o in offsets {
            v.extend(o.to_le_bytes());
        }
        v.extend(0u64.to_le_bytes());
        v
    }
}

/// A tiny deterministic PRNG (xorshift64*) for mutation tests.
pub(crate) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Apply a few random byte mutations / a random truncation.
    pub fn mutate(&mut self, data: &[u8]) -> Vec<u8> {
        let mut d = data.to_vec();
        let n = 1 + self.below(4);
        for _ in 0..n {
            match self.below(4) {
                0 => {
                    let i = self.below(d.len());
                    d[i] = self.next() as u8;
                }
                1 => {
                    let i = self.below(d.len());
                    d[i] ^= 1 << self.below(8);
                }
                2 => {
                    let i = self.below(d.len());
                    d[i] = [0, 0xff, 1, 0x7f][self.below(4)];
                }
                _ => {
                    if self.below(8) == 0 {
                        let l = self.below(d.len() + 1);
                        d.truncate(l.max(1));
                    }
                }
            }
        }
        d
    }
}
