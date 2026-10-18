//! Dense link / attribute storage: a fractal heap holding the encoded
//! messages, indexed by name hash in a version 2 B-tree.
//!
//! Layout decisions mirror what the reference library produces (observed
//! black-box): table width 4, a 512-byte (links) or 1024-byte (attributes)
//! starting block size, 64 KiB maximum direct block size, 4096-byte
//! maximum managed object size, checksummed direct blocks, 7-byte (links)
//! or 8-byte (attributes) heap IDs and 512-byte B-tree nodes. Messages
//! larger than the managed limit become "huge" objects indexed by a
//! type 1 v2 B-tree.
//!
//! Every structure's size depends only on the message sizes, so the plan
//! is computed (and space allocated) before addresses inside the messages
//! are known; [`DensePlan::write`] then encodes the final bytes.

use super::storage::{Allocator, WriteAt};
use crate::checksum::{lookup3, metadata_checksum};
use crate::error::{Error, Result};
use crate::format::btree_v2::{BTreeV2Header, ChildPointer, Record, encode_node};
use crate::format::fractal_heap::FractalHeapHeader;
use crate::format::{ByteWriter, Sizes, UNDEFINED_ADDRESS};

/// What a dense store indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseKind {
    /// Link messages of a group (name index: B-tree type 5).
    Links,
    /// Attribute messages of an object (name index: B-tree type 8).
    Attributes,
}

const TABLE_WIDTH: u64 = 4;
const MAX_MANAGED: u64 = 4096;
const BTREE_NODE_SIZE: u32 = 512;
/// Largest node size tried when a depth-1 tree with 512-byte nodes cannot
/// hold every record.
const MAX_BTREE_NODE_SIZE: u32 = 1 << 16;
/// Largest maximum direct block size used when the default one is too
/// small for the heap (root indirect blocks have direct rows only).
const MAX_DIRECT_LIMIT: u64 = 1 << 30;
/// Flag: direct blocks are checksummed.
const FLAG_CHECKSUM_DIRECT: u8 = 0x02;
/// Creation order stored in attribute name records when creation order is
/// not tracked (as the reference library does).
const UNTRACKED_CREATION_ORDER: u32 = 0xffff;

impl DenseKind {
    fn heap_id_len(self) -> usize {
        match self {
            DenseKind::Links => 7,
            DenseKind::Attributes => 8,
        }
    }
    fn heap_bits(self) -> u16 {
        match self {
            DenseKind::Links => 32,
            DenseKind::Attributes => 40,
        }
    }
    fn start_block(self) -> u64 {
        match self {
            DenseKind::Links => 512,
            DenseKind::Attributes => 1024,
        }
    }
    fn tree_type(self) -> u8 {
        match self {
            DenseKind::Links => 5,
            DenseKind::Attributes => 8,
        }
    }
    fn record_size(self) -> u16 {
        match self {
            DenseKind::Links => 4 + 7,
            DenseKind::Attributes => 8 + 1 + 4 + 4,
        }
    }
}

/// Where one stored message lives.
#[derive(Debug, Clone, Copy)]
enum Placement {
    /// Managed object in direct block `block` at heap offset `offset`.
    Managed { block: usize, offset: u64 },
    /// Huge object with this ID (index into `huge`).
    Huge { id: u64 },
}

/// A planned v2 B-tree: node structure and allocated addresses.
#[derive(Debug, Clone)]
struct TreePlan {
    header: BTreeV2Header,
    /// Leaves (record index ranges) and, for depth 1, the separator
    /// record indices between consecutive leaves.
    leaves: Vec<std::ops::Range<usize>>,
    separators: Vec<usize>,
    header_addr: u64,
    leaf_addrs: Vec<u64>,
    root_addr: u64,
}

impl TreePlan {
    /// Plan a tree of `n` records of `record_size` bytes.
    fn new(tree_type: u8, record_size: u16, n: usize, sizes: Sizes) -> Result<TreePlan> {
        let mut header = BTreeV2Header {
            tree_type,
            node_size: BTREE_NODE_SIZE,
            record_size,
            depth: 0,
            split_percent: 100,
            merge_percent: 40,
            root_address: UNDEFINED_ADDRESS,
            root_num_records: 0,
            total_records: n as u64,
        };
        let (leaf_max, _, _) = header.node_capacity(sizes, 0)?;
        if n as u64 <= leaf_max {
            header.root_num_records = n as u16;
            return Ok(TreePlan {
                header,
                leaves: std::iter::once(0..n).collect(),
                separators: Vec::new(),
                header_addr: 0,
                leaf_addrs: Vec::new(),
                root_addr: 0,
            });
        }
        // Depth 1: the smallest node size whose root can index all records.
        header.depth = 1;
        loop {
            let (leaf_max, _, _) = header.node_capacity(sizes, 0)?;
            let (root_max, _, _) = header.node_capacity(sizes, 1)?;
            let leaves = (n as u64 + 1).div_ceil(leaf_max + 1);
            if leaves - 1 <= root_max && leaves - 1 <= u64::from(u16::MAX) {
                let k = leaves as usize;
                let in_leaves = n - (k - 1);
                let (base, extra) = (in_leaves / k, in_leaves % k);
                let mut ranges = Vec::with_capacity(k);
                let mut separators = Vec::with_capacity(k - 1);
                let mut pos = 0;
                for i in 0..k {
                    let len = base + usize::from(i < extra);
                    ranges.push(pos..pos + len);
                    pos += len;
                    if i + 1 < k {
                        separators.push(pos);
                        pos += 1;
                    }
                }
                header.root_num_records = (k - 1) as u16;
                return Ok(TreePlan {
                    header,
                    leaves: ranges,
                    separators,
                    header_addr: 0,
                    leaf_addrs: Vec::new(),
                    root_addr: 0,
                });
            }
            if header.node_size >= MAX_BTREE_NODE_SIZE {
                return Err(Error::LimitExceeded("too many entries for dense storage"));
            }
            header.node_size *= 2;
        }
    }

    fn allocate(&mut self, alloc: &mut Allocator, sizes: Sizes) -> Result<()> {
        self.header_addr = alloc.alloc(BTreeV2Header::encoded_size(sizes) as u64)?;
        let node = u64::from(self.header.node_size);
        if self.header.depth == 0 {
            if self.header.total_records == 0 {
                self.root_addr = UNDEFINED_ADDRESS;
            } else {
                self.root_addr = alloc.alloc(node)?;
            }
        } else {
            self.leaf_addrs = (0..self.leaves.len())
                .map(|_| alloc.alloc(node))
                .collect::<Result<_>>()?;
            self.root_addr = alloc.alloc(node)?;
        }
        self.header.root_address = self.root_addr;
        Ok(())
    }

    fn write(&self, records: &[Record], sizes: Sizes, storage: &mut dyn WriteAt) -> Result<()> {
        storage.write_all_at(self.header_addr, &self.header.encode(sizes)?)?;
        if self.root_addr == UNDEFINED_ADDRESS {
            return Ok(());
        }
        let slice = |r: &std::ops::Range<usize>| {
            records
                .get(r.clone())
                .ok_or_else(|| Error::invalid("B-tree plan out of range"))
        };
        if self.header.depth == 0 {
            let leaf = encode_node(&self.header, sizes, 0, slice(&(0..records.len()))?, &[], 0)?;
            return storage.write_all_at(self.root_addr, &leaf);
        }
        let mut children = Vec::with_capacity(self.leaves.len());
        for (range, &addr) in self.leaves.iter().zip(&self.leaf_addrs) {
            let leaf = encode_node(&self.header, sizes, 0, slice(range)?, &[], 0)?;
            storage.write_all_at(addr, &leaf)?;
            children.push(ChildPointer {
                address: addr,
                num_records: range.len() as u64,
                total_records: range.len() as u64,
            });
        }
        let seps = self
            .separators
            .iter()
            .map(|&i| {
                records
                    .get(i)
                    .cloned()
                    .ok_or_else(|| Error::invalid("B-tree plan out of range"))
            })
            .collect::<Result<Vec<_>>>()?;
        let root = encode_node(&self.header, sizes, 1, &seps, &children, 0)?;
        storage.write_all_at(self.root_addr, &root)
    }
}

/// One direct block of the doubling table.
#[derive(Debug, Clone, Copy)]
struct Block {
    /// Offset in the heap's linear address space.
    offset: u64,
    size: u64,
    used: u64,
}

/// A complete plan for one dense store.
#[derive(Debug, Clone)]
pub(crate) struct DensePlan {
    kind: DenseKind,
    sizes: Sizes,
    /// Encoded message lengths, in input order.
    lens: Vec<u64>,
    placements: Vec<Placement>,
    blocks: Vec<Block>,
    /// Rows of the root indirect block (0: the root is a direct block).
    root_rows: u16,
    max_direct: u64,
    /// Order of objects in the name index (sorted by hash).
    order: Vec<usize>,
    hashes: Vec<u32>,
    huge: Vec<usize>,
    tree: TreePlan,
    huge_tree: Option<TreePlan>,
    // Addresses, set by `allocate`.
    heap_addr: u64,
    root_addr: u64,
    block_addrs: Vec<u64>,
    huge_addrs: Vec<u64>,
}

/// Least number of bytes able to encode `v` (1..=8).
fn bytes_for(v: u64) -> usize {
    (64 - v.leading_zeros() as usize).div_ceil(8).max(1)
}

impl DensePlan {
    /// Plan the storage of messages `(name, encoded length)`.
    pub(crate) fn new(kind: DenseKind, entries: &[(String, usize)], sizes: Sizes) -> Result<Self> {
        let start = kind.start_block();
        let offset_width = usize::from(kind.heap_bits()).div_ceil(8);
        let header_len = 5 + u64::from(sizes.offset) + offset_width as u64 + 4;
        let mut max_direct = 1u64 << 16;
        let (placements, blocks, huge, root_rows) = loop {
            let max_rows = max_direct.trailing_zeros() - start.trailing_zeros() + 2;
            let mut placements = Vec::with_capacity(entries.len());
            let mut blocks: Vec<Block> = Vec::new();
            let mut huge = Vec::new();
            let mut next = 0u64; // index of the next unallocated block
            let mut too_big = false;
            for (i, (_, len)) in entries.iter().enumerate() {
                let len = *len as u64;
                if len == 0 {
                    return Err(Error::invalid("empty dense storage object"));
                }
                if len > MAX_MANAGED {
                    huge.push(i);
                    placements.push(Placement::Huge {
                        id: huge.len() as u64,
                    });
                    continue;
                }
                let fits = |b: &Block| b.size - header_len - b.used >= len;
                if blocks.last().is_none_or(|b| !fits(b)) {
                    // Open the next block large enough for the object.
                    loop {
                        let row = next / TABLE_WIDTH;
                        if row >= u64::from(max_rows) {
                            too_big = true;
                            break;
                        }
                        let size = start << row.saturating_sub(1);
                        let offset = if row == 0 {
                            next * start
                        } else {
                            ((TABLE_WIDTH * start) << (row - 1)) + (next % TABLE_WIDTH) * size
                        };
                        next += 1;
                        let b = Block {
                            offset,
                            size,
                            used: 0,
                        };
                        let ok = fits(&b);
                        blocks.push(b);
                        if ok {
                            break;
                        }
                    }
                    if too_big {
                        break;
                    }
                }
                let block = blocks.len().wrapping_sub(1);
                let Some(b) = blocks.last_mut() else {
                    return Err(Error::invalid("no heap block"));
                };
                placements.push(Placement::Managed {
                    block,
                    offset: b.offset + header_len + b.used,
                });
                b.used += len;
            }
            if too_big {
                if max_direct >= MAX_DIRECT_LIMIT {
                    return Err(Error::LimitExceeded("dense storage heap too large"));
                }
                max_direct *= 2;
                continue;
            }
            let root_rows = if blocks.len() <= 1 {
                0
            } else {
                (next.div_ceil(TABLE_WIDTH)) as u16
            };
            break (placements, blocks, huge, root_rows);
        };
        let hashes: Vec<u32> = entries
            .iter()
            .map(|(name, _)| lookup3(name.as_bytes(), 0))
            .collect();
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by(|&a, &b| {
            hashes[a]
                .cmp(&hashes[b])
                .then_with(|| entries[a].0.cmp(&entries[b].0))
        });
        let tree = TreePlan::new(kind.tree_type(), kind.record_size(), entries.len(), sizes)?;
        let huge_tree = if huge.is_empty() {
            None
        } else {
            let rec = u16::from(sizes.offset) + 2 * u16::from(sizes.length);
            Some(TreePlan::new(1, rec, huge.len(), sizes)?)
        };
        Ok(DensePlan {
            kind,
            sizes,
            lens: entries.iter().map(|(_, l)| *l as u64).collect(),
            placements,
            blocks,
            root_rows,
            max_direct,
            order,
            hashes,
            huge,
            tree,
            huge_tree,
            heap_addr: 0,
            root_addr: UNDEFINED_ADDRESS,
            block_addrs: Vec::new(),
            huge_addrs: Vec::new(),
        })
    }

    fn offset_width(&self) -> usize {
        usize::from(self.kind.heap_bits()).div_ceil(8)
    }

    fn header_len(&self) -> u64 {
        5 + u64::from(self.sizes.offset) + self.offset_width() as u64 + 4
    }

    fn indirect_block_size(&self) -> u64 {
        let max_rows = u64::from(
            self.max_direct.trailing_zeros() - self.kind.start_block().trailing_zeros() + 2,
        );
        let drows = u64::from(self.root_rows).min(max_rows);
        5 + u64::from(self.sizes.offset)
            + self.offset_width() as u64
            + drows * TABLE_WIDTH * u64::from(self.sizes.offset)
            + 4
    }

    /// Reserve space for every structure.
    pub(crate) fn allocate(&mut self, alloc: &mut Allocator) -> Result<()> {
        let sizes = self.sizes;
        self.heap_addr = alloc.alloc(FractalHeapHeader::encoded_size(sizes) as u64)?;
        self.block_addrs = self
            .blocks
            .iter()
            .map(|b| alloc.alloc(b.size))
            .collect::<Result<_>>()?;
        self.root_addr = if self.root_rows > 0 {
            alloc.alloc(self.indirect_block_size())?
        } else {
            self.block_addrs
                .first()
                .copied()
                .unwrap_or(UNDEFINED_ADDRESS)
        };
        self.huge_addrs = self
            .huge
            .iter()
            .map(|&i| alloc.alloc(self.lens[i]))
            .collect::<Result<_>>()?;
        self.tree.allocate(alloc, sizes)?;
        if let Some(t) = &mut self.huge_tree {
            t.allocate(alloc, sizes)?;
        }
        Ok(())
    }

    /// Address of the fractal heap header.
    pub(crate) fn heap_address(&self) -> u64 {
        self.heap_addr
    }

    /// Address of the name index B-tree header.
    pub(crate) fn name_index_address(&self) -> u64 {
        self.tree.header_addr
    }

    fn heap_id(&self, i: usize) -> Result<Vec<u8>> {
        let mut w = ByteWriter::with_sizes(self.sizes);
        match self.placements.get(i) {
            Some(Placement::Managed { offset, .. }) => {
                let len = self.lens.get(i).copied().unwrap_or(0);
                w.u8(0);
                w.write_uint(*offset, self.offset_width())?;
                w.write_uint(len, bytes_for(self.max_direct.min(MAX_MANAGED)))?;
            }
            Some(Placement::Huge { id }) => {
                let key_len = usize::from(self.sizes.length).min(self.kind.heap_id_len() - 1);
                w.u8(0x10);
                w.write_uint(*id, key_len)?;
            }
            None => return Err(Error::invalid("dense object index out of range")),
        }
        let id_len = self.kind.heap_id_len();
        if w.len() > id_len {
            return Err(Error::invalid("heap ID longer than its fixed length"));
        }
        w.zeros(id_len - w.len());
        Ok(w.into_vec())
    }

    /// Encode and write everything. `objects` are the final message
    /// encodings, in the order given to [`DensePlan::new`].
    pub(crate) fn write(&self, objects: &[Vec<u8>], storage: &mut dyn WriteAt) -> Result<()> {
        let sizes = self.sizes;
        if objects.len() != self.lens.len()
            || objects
                .iter()
                .zip(&self.lens)
                .any(|(o, &l)| o.len() as u64 != l)
        {
            return Err(Error::invalid("dense object sizes changed after planning"));
        }
        // Direct blocks.
        let hlen = self.header_len() as usize;
        let mut images: Vec<Vec<u8>> = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let mut w = ByteWriter::with_sizes(sizes);
            w.bytes(b"FHDB").u8(0);
            w.write_offset(self.heap_addr)?;
            w.write_uint(b.offset, self.offset_width())?;
            w.u32(0); // checksum placeholder
            let size = usize::try_from(b.size).map_err(|_| Error::LimitExceeded("heap block"))?;
            w.zeros(size - w.len());
            images.push(w.into_vec());
        }
        for (i, p) in self.placements.iter().enumerate() {
            if let Placement::Managed { block, offset } = *p {
                let b = self.blocks.get(block).ok_or(Error::UnexpectedEof)?;
                let img = images.get_mut(block).ok_or(Error::UnexpectedEof)?;
                let at = (offset - b.offset) as usize;
                let obj = &objects[i];
                img.get_mut(at..at + obj.len())
                    .ok_or(Error::UnexpectedEof)?
                    .copy_from_slice(obj);
            }
        }
        for (img, &addr) in images.iter_mut().zip(&self.block_addrs) {
            let sum = metadata_checksum(img);
            img.get_mut(hlen - 4..hlen)
                .ok_or(Error::UnexpectedEof)?
                .copy_from_slice(&sum.to_le_bytes());
            storage.write_all_at(addr, img)?;
        }
        // Root indirect block.
        if self.root_rows > 0 {
            let mut w = ByteWriter::with_sizes(sizes);
            w.bytes(b"FHIB").u8(0);
            w.write_offset(self.heap_addr)?;
            w.write_uint(0, self.offset_width())?;
            let entries = u64::from(self.root_rows) * TABLE_WIDTH;
            for e in 0..entries as usize {
                w.write_offset(
                    self.block_addrs
                        .get(e)
                        .copied()
                        .unwrap_or(UNDEFINED_ADDRESS),
                )?;
            }
            let sum = metadata_checksum(w.as_slice());
            w.u32(sum);
            if w.len() as u64 != self.indirect_block_size() {
                return Err(Error::invalid("indirect block size mismatch"));
            }
            storage.write_all_at(self.root_addr, w.as_slice())?;
        }
        // Huge objects and their index.
        let mut huge_records = Vec::with_capacity(self.huge.len());
        for (k, (&i, &addr)) in self.huge.iter().zip(&self.huge_addrs).enumerate() {
            storage.write_all_at(addr, &objects[i])?;
            huge_records.push(Record::HugeIndirect {
                address: addr,
                length: self.lens[i],
                id: k as u64 + 1,
            });
        }
        if let Some(t) = &self.huge_tree {
            t.write(&huge_records, sizes, storage)?;
        }
        // Heap header.
        let allocated: u64 = self.blocks.iter().map(|b| b.size).sum();
        let free: u64 = self
            .blocks
            .iter()
            .map(|b| b.size - self.header_len() - b.used)
            .sum();
        let start = self.kind.start_block();
        let (managed_space, iterator) = if self.root_rows > 0 {
            let span = (TABLE_WIDTH * start) << (self.root_rows - 1);
            let end = self.blocks.last().map_or(0, |b| b.offset + b.size);
            (span, end)
        } else if self.blocks.is_empty() {
            (0, 0)
        } else {
            (start, 0)
        };
        let managed_count = self.lens.len() - self.huge.len();
        let header = FractalHeapHeader {
            heap_id_len: self.kind.heap_id_len() as u16,
            io_filter_len: 0,
            flags: FLAG_CHECKSUM_DIRECT,
            max_managed_object_size: MAX_MANAGED as u32,
            next_huge_id: self.huge.len() as u64,
            huge_btree_address: self
                .huge_tree
                .as_ref()
                .map_or(UNDEFINED_ADDRESS, |t| t.header_addr),
            free_space: free,
            free_space_manager_address: UNDEFINED_ADDRESS,
            managed_space,
            allocated_managed_space: allocated,
            direct_block_iterator_offset: iterator,
            managed_object_count: managed_count as u64,
            huge_object_size: self.huge.iter().map(|&i| self.lens[i]).sum(),
            huge_object_count: self.huge.len() as u64,
            tiny_object_size: 0,
            tiny_object_count: 0,
            table_width: TABLE_WIDTH as u16,
            starting_block_size: start,
            max_direct_block_size: self.max_direct,
            max_heap_size_bits: self.kind.heap_bits(),
            starting_root_rows: 1,
            root_address: self.root_addr,
            current_root_rows: self.root_rows,
            filtered_root_size: None,
            root_filter_mask: None,
            filter_info: Vec::new(),
        };
        storage.write_all_at(self.heap_addr, &header.encode(sizes)?)?;
        // Name index.
        let mut records = Vec::with_capacity(self.order.len());
        for &i in &self.order {
            let heap_id = self.heap_id(i)?;
            let hash = self.hashes[i];
            records.push(match self.kind {
                DenseKind::Links => Record::LinkName { hash, heap_id },
                DenseKind::Attributes => Record::AttributeName {
                    heap_id,
                    message_flags: 0,
                    creation_order: UNTRACKED_CREATION_ORDER,
                    hash,
                },
            });
        }
        self.tree.write(&records, sizes, storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::btree_v2::{BTreeV2, RecordContext};
    use crate::format::fractal_heap::FractalHeap;

    fn round_trip(kind: DenseKind, entries: Vec<(String, Vec<u8>)>) -> DensePlan {
        let sizes = Sizes::DEFAULT;
        let spec: Vec<(String, usize)> =
            entries.iter().map(|(n, d)| (n.clone(), d.len())).collect();
        let mut plan = DensePlan::new(kind, &spec, sizes).unwrap();
        let mut alloc = Allocator::new(0, sizes);
        plan.allocate(&mut alloc).unwrap();
        let mut out: Vec<u8> = Vec::new();
        let objects: Vec<Vec<u8>> = entries.iter().map(|(_, d)| d.clone()).collect();
        plan.write(&objects, &mut out).unwrap();
        out.resize(alloc.eoa() as usize, 0);
        let heap = FractalHeap::open(&out, plan.heap_address(), sizes).unwrap();
        let tree = BTreeV2::open(
            &out,
            plan.name_index_address(),
            sizes,
            RecordContext::default(),
        )
        .unwrap();
        let records = tree.records(&out).unwrap();
        assert_eq!(records.len(), entries.len());
        let mut last = 0u32;
        for r in &records {
            let (hash, id) = match r {
                Record::LinkName { hash, heap_id } => (*hash, heap_id),
                Record::AttributeName { hash, heap_id, .. } => (*hash, heap_id),
                _ => panic!("unexpected record"),
            };
            assert!(hash >= last);
            last = hash;
            let data = heap.get(&out, id).unwrap();
            let (name, expect) = entries
                .iter()
                .find(|(n, _)| lookup3(n.as_bytes(), 0) == hash)
                .unwrap();
            assert_eq!(&data, expect, "object {name}");
        }
        plan
    }

    #[test]
    fn small_heap_has_direct_root() {
        let entries = (0..10)
            .map(|i| (format!("link{i}"), vec![i as u8; 20 + i]))
            .collect();
        let plan = round_trip(DenseKind::Links, entries);
        assert_eq!(plan.root_rows, 0);
        assert_eq!(plan.tree.header.depth, 0);
    }

    #[test]
    fn larger_heap_and_tree() {
        let entries = (0..3000)
            .map(|i| (format!("name{i}"), vec![(i % 251) as u8; 30 + i % 50]))
            .collect();
        let plan = round_trip(DenseKind::Links, entries);
        assert!(plan.root_rows > 1);
        assert_eq!(plan.tree.header.depth, 1);
        assert!(plan.tree.header.node_size > BTREE_NODE_SIZE);
    }

    #[test]
    fn huge_and_managed_attributes() {
        let entries = vec![
            ("big".to_string(), vec![7u8; 80000]),
            ("small".to_string(), vec![1u8; 40]),
            ("mid".to_string(), vec![2u8; 4000]),
            ("big2".to_string(), vec![9u8; 5000]),
        ];
        let plan = round_trip(DenseKind::Attributes, entries);
        assert_eq!(plan.huge.len(), 2);
        assert!(plan.huge_tree.is_some());
    }

    #[test]
    fn heap_grows_max_direct_block() {
        // ~1 MiB of 4000-byte objects exceeds the default doubling table.
        let entries = (0..300)
            .map(|i| (format!("a{i}"), vec![(i % 256) as u8; 4000]))
            .collect();
        let plan = round_trip(DenseKind::Attributes, entries);
        assert!(plan.max_direct > 1 << 16);
    }
}
