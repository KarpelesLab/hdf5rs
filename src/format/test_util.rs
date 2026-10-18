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

/// A little-endian byte writer.
#[derive(Debug, Default, Clone)]
pub(crate) struct W {
    pub buf: Vec<u8>,
    pub sizes: Sizes,
}

impl W {
    pub fn new() -> Self {
        W::default()
    }

    pub fn with_sizes(sizes: Sizes) -> Self {
        W {
            buf: Vec::new(),
            sizes,
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }

    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn uint(&mut self, v: u64, width: usize) -> &mut Self {
        let b = v.to_le_bytes();
        for i in 0..width {
            self.buf.push(b.get(i).copied().unwrap_or(0));
        }
        self
    }

    pub fn bytes(&mut self, v: &[u8]) -> &mut Self {
        self.buf.extend_from_slice(v);
        self
    }

    pub fn zeros(&mut self, n: usize) -> &mut Self {
        self.buf.resize(self.buf.len() + n, 0);
        self
    }

    /// Write an address in the configured offset width
    /// (`u64::MAX` writes the all-ones undefined address).
    pub fn offset(&mut self, v: u64) -> &mut Self {
        let w = self.sizes.offset as usize;
        if v == u64::MAX {
            self.buf.resize(self.buf.len() + w, 0xff);
            self
        } else {
            self.uint(v, w)
        }
    }

    pub fn length(&mut self, v: u64) -> &mut Self {
        self.uint(v, self.sizes.length as usize)
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
        (self.next() % n.max(1) as u64) as usize
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

/// Apply 1..=4 random byte mutations (flip, set, truncate) to a copy.
pub(crate) fn mutate(rng: &mut Rng, src: &[u8]) -> Vec<u8> {
    let mut v = src.to_vec();
    let n = 1 + rng.below(4);
    for _ in 0..n {
        if v.is_empty() {
            break;
        }
        match rng.below(6) {
            0 => {
                let len = rng.below(v.len());
                v.truncate(len);
            }
            1 => {
                let i = rng.below(v.len());
                v[i] = 0xff;
            }
            2 => {
                let i = rng.below(v.len());
                v[i] = 0;
            }
            _ => {
                let i = rng.below(v.len());
                v[i] ^= 1 << rng.below(8);
            }
        }
    }
    v
}
