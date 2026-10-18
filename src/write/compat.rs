//! "Compatibility" (earliest) format pieces: old-style groups stored as a
//! symbol table (local heap of names + v1 B-tree of symbol table nodes),
//! and datatype version downgrades.

use super::storage::{Allocator, WriteAt};
use crate::error::{Error, Result};
use crate::format::btree_v1::{BTreeNode, Keys, NodeType};
use crate::format::local_heap::LocalHeap;
use crate::format::messages::datatype::{ByteOrder, Datatype, DatatypeClass, ReferenceKind};
use crate::format::symbol_table_node::SymbolTableNode;
use crate::format::{ByteWriter, ScratchPad, Sizes, SymbolTableEntry, UNDEFINED_ADDRESS};

/// Group leaf node K written in the superblock (symbol table nodes hold
/// up to `2 * K` entries).
pub(crate) const GROUP_LEAF_K: u16 = 4;
/// Group internal node K (B-tree nodes hold up to `2 * K` children).
pub(crate) const GROUP_INTERNAL_K: u16 = 16;

/// The lowest datatype versions able to encode `dt` (1 for most classes,
/// 2 for arrays and anything containing them), applied recursively.
pub(crate) fn downgrade(dt: &Datatype) -> Datatype {
    let mut out = dt.clone();
    out.version = match &mut out.class {
        DatatypeClass::Compound { members } => {
            let mut v = 1;
            for m in members.iter_mut() {
                m.datatype = downgrade(&m.datatype);
                v = v.max(m.datatype.version);
            }
            v
        }
        DatatypeClass::Enumeration { base, .. } | DatatypeClass::VariableLength { base, .. } => {
            **base = downgrade(base);
            base.version.max(1)
        }
        DatatypeClass::Array { base, .. } => {
            **base = downgrade(base);
            base.version.max(2)
        }
        // Complex numbers and revised references have no older encoding.
        DatatypeClass::Complex { base, .. } => {
            **base = downgrade(base);
            dt.version
        }
        DatatypeClass::Reference(r) => match r.kind {
            ReferenceKind::Object | ReferenceKind::DatasetRegion => 1,
            _ => dt.version,
        },
        DatatypeClass::FloatingPoint(f) if f.byte_order == ByteOrder::Vax => dt.version.max(3),
        _ => 1,
    };
    out
}

/// What a symbol table entry points to.
#[derive(Debug, Clone)]
pub(crate) enum EntryTarget {
    /// Hard link to an object header (address resolved at write time),
    /// identified by its node index.
    Hard(usize),
    /// Soft link value.
    Soft(String),
}

/// Resolves a node index to its object header address and, for groups,
/// their symbol table's (B-tree, local heap) addresses.
pub(crate) type HeaderLookup<'a> = dyn Fn(usize) -> Result<(u64, Option<(u64, u64)>)> + 'a;

/// Plan of one old-style group's symbol table.
#[derive(Debug, Clone)]
pub(crate) struct SymTabPlan {
    sizes: Sizes,
    /// Entries sorted by name: (name offset, target, soft value offset).
    entries: Vec<(u64, EntryTarget, u64)>,
    heap_data: Vec<u8>,
    /// Offset of the free block at the end of the data segment.
    free_offset: u64,
    /// B-tree levels, bottom first; each node lists its child indices in
    /// the level below (SNOD indices for level 0).
    levels: Vec<Vec<std::ops::Range<usize>>>,
    snod_count: usize,
    // Addresses.
    heap_addr: u64,
    heap_data_addr: u64,
    snod_addrs: Vec<u64>,
    node_addrs: Vec<Vec<u64>>,
}

fn push_string(data: &mut Vec<u8>, s: &[u8]) -> u64 {
    let off = data.len() as u64;
    data.extend_from_slice(s);
    data.push(0);
    data.resize(data.len().next_multiple_of(8), 0);
    off
}

impl SymTabPlan {
    /// Plan a symbol table for `links` (name, target).
    pub(crate) fn new(links: &[(String, EntryTarget)], sizes: Sizes) -> Result<SymTabPlan> {
        let mut sorted: Vec<&(String, EntryTarget)> = links.iter().collect();
        sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        // Offset 0 holds the empty string (the B-tree's left-most key).
        let mut heap_data = vec![0u8; 8];
        let mut entries = Vec::with_capacity(sorted.len());
        for (name, target) in sorted {
            if name.as_bytes().contains(&0) {
                return Err(Error::invalid("link name contains NUL"));
            }
            let off = push_string(&mut heap_data, name.as_bytes());
            entries.push((off, target.clone(), 0));
        }
        for e in &mut entries {
            if let EntryTarget::Soft(v) = &e.1 {
                if v.as_bytes().contains(&0) {
                    return Err(Error::invalid("soft link value contains NUL"));
                }
                let off = push_string(&mut heap_data, v.as_bytes());
                if u32::try_from(off).is_err() {
                    return Err(Error::LimitExceeded("local heap too large"));
                }
                e.2 = off;
            }
        }
        // End the data segment with a free block (next = 1 marks the end
        // of the free list), as the reference library does; it rejects a
        // heap whose free list head is the undefined address.
        let free_offset = heap_data.len() as u64;
        let l = usize::from(sizes.length);
        let mut free = ByteWriter::with_sizes(sizes);
        free.write_length(1)?;
        free.write_length(2 * l as u64)?;
        heap_data.extend_from_slice(free.as_slice());
        let per_snod = 2 * usize::from(GROUP_LEAF_K);
        let snod_count = entries.len().div_ceil(per_snod);
        let fanout = 2 * usize::from(GROUP_INTERNAL_K);
        let mut levels = Vec::new();
        let mut width = snod_count;
        loop {
            let nodes = width.div_ceil(fanout).max(1);
            let level: Vec<std::ops::Range<usize>> = (0..nodes)
                .map(|i| i * fanout..((i + 1) * fanout).min(width))
                .collect();
            levels.push(level);
            if nodes == 1 {
                break;
            }
            if levels.len() > 255 {
                return Err(Error::LimitExceeded("group B-tree too deep"));
            }
            width = nodes;
        }
        Ok(SymTabPlan {
            sizes,
            entries,
            heap_data,
            free_offset,
            levels,
            snod_count,
            heap_addr: 0,
            heap_data_addr: 0,
            snod_addrs: Vec::new(),
            node_addrs: Vec::new(),
        })
    }

    /// Reserve space for the heap, symbol table nodes and B-tree nodes.
    pub(crate) fn allocate(&mut self, alloc: &mut Allocator) -> Result<()> {
        let sizes = self.sizes;
        self.heap_addr = alloc.alloc(LocalHeap::header_size(sizes) as u64)?;
        self.heap_data_addr = alloc.alloc(self.heap_data.len() as u64)?;
        let snod = SymbolTableNode::node_size(sizes, GROUP_LEAF_K) as u64;
        self.snod_addrs = (0..self.snod_count)
            .map(|_| alloc.alloc(snod))
            .collect::<Result<_>>()?;
        let node = BTreeNode::group_node_size(sizes, GROUP_INTERNAL_K) as u64;
        self.node_addrs = self
            .levels
            .iter()
            .map(|l| {
                l.iter()
                    .map(|_| alloc.alloc(node))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// Address of the B-tree root node.
    pub(crate) fn btree_address(&self) -> u64 {
        self.node_addrs
            .last()
            .and_then(|l| l.first())
            .copied()
            .unwrap_or(UNDEFINED_ADDRESS)
    }

    /// Address of the local heap header.
    pub(crate) fn heap_address(&self) -> u64 {
        self.heap_addr
    }

    /// Write everything. `header_of(node)` gives a hard link target's
    /// object header address and, for groups, their (B-tree, heap)
    /// addresses for the entry's scratch pad.
    pub(crate) fn write(
        &self,
        header_of: &HeaderLookup<'_>,
        storage: &mut dyn WriteAt,
    ) -> Result<()> {
        let sizes = self.sizes;
        storage.write_all_at(
            self.heap_addr,
            &LocalHeap::encode_header(
                self.heap_data.len() as u64,
                Some(self.free_offset),
                self.heap_data_addr,
                sizes,
            )?,
        )?;
        storage.write_all_at(self.heap_data_addr, &self.heap_data)?;
        let per_snod = 2 * usize::from(GROUP_LEAF_K);
        // Last name offset of each SNOD: the B-tree keys.
        let mut snod_keys = Vec::with_capacity(self.snod_count);
        for (i, chunk) in self.entries.chunks(per_snod).enumerate() {
            let mut node = SymbolTableNode {
                version: 1,
                entries: Vec::with_capacity(chunk.len()),
            };
            for (name_off, target, soft_off) in chunk {
                let entry = match target {
                    EntryTarget::Hard(id) => {
                        let (address, group) = header_of(*id)?;
                        SymbolTableEntry {
                            link_name_offset: *name_off,
                            object_header_address: address,
                            scratch: match group {
                                Some((btree_address, heap_address)) => ScratchPad::Group {
                                    btree_address,
                                    heap_address,
                                },
                                None => ScratchPad::None,
                            },
                        }
                    }
                    EntryTarget::Soft(_) => SymbolTableEntry {
                        link_name_offset: *name_off,
                        object_header_address: UNDEFINED_ADDRESS,
                        scratch: ScratchPad::SoftLink {
                            link_value_offset: *soft_off as u32,
                        },
                    },
                };
                node.entries.push(entry);
            }
            let addr = *self.snod_addrs.get(i).ok_or(Error::UnexpectedEof)?;
            storage.write_all_at(addr, &node.encode(sizes, GROUP_LEAF_K)?)?;
            snod_keys.push(chunk.last().map_or(0, |e| e.0));
        }
        // B-tree levels: (first key, last key) of every node of the level
        // below, starting with the SNODs (whose first key is the previous
        // SNOD's last key, or 0).
        let mut below: Vec<(u64, u64)> = snod_keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (if i == 0 { 0 } else { snod_keys[i - 1] }, k))
            .collect();
        let mut below_addrs = self.snod_addrs.clone();
        for (lvl, (level, addrs)) in self.levels.iter().zip(&self.node_addrs).enumerate() {
            let mut next = Vec::with_capacity(level.len());
            for (j, range) in level.iter().enumerate() {
                let kids = below
                    .get(range.clone())
                    .ok_or_else(|| Error::invalid("group B-tree plan out of range"))?;
                let mut keys = Vec::with_capacity(kids.len() + 1);
                keys.push(kids.first().map_or(0, |k| k.0));
                keys.extend(kids.iter().map(|k| k.1));
                let node = BTreeNode {
                    node_type: NodeType::Group,
                    level: lvl as u8,
                    left_sibling: if j == 0 {
                        UNDEFINED_ADDRESS
                    } else {
                        addrs[j - 1]
                    },
                    right_sibling: addrs.get(j + 1).copied().unwrap_or(UNDEFINED_ADDRESS),
                    keys: Keys::Group(keys.clone()),
                    children: below_addrs
                        .get(range.clone())
                        .ok_or_else(|| Error::invalid("group B-tree plan out of range"))?
                        .to_vec(),
                };
                storage.write_all_at(addrs[j], &node.encode_group(sizes, GROUP_INTERNAL_K)?)?;
                next.push((keys[0], keys.last().copied().unwrap_or(0)));
            }
            below = next;
            below_addrs = addrs.clone();
        }
        Ok(())
    }
}

/// Encode the root symbol table entry stored in a version 0 superblock.
pub(crate) fn root_entry(header: u64, btree: u64, heap: u64) -> SymbolTableEntry {
    SymbolTableEntry {
        link_name_offset: 0,
        object_header_address: header,
        scratch: ScratchPad::Group {
            btree_address: btree,
            heap_address: heap,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::btree_v1::group_btree_children;

    #[test]
    fn symbol_table_round_trip() {
        let sizes = Sizes::DEFAULT;
        for n in [0usize, 1, 8, 9, 300, 3000] {
            let links: Vec<(String, EntryTarget)> = (0..n)
                .map(|i| {
                    let t = if i % 7 == 3 {
                        EntryTarget::Soft(format!("/target/{i}"))
                    } else {
                        EntryTarget::Hard(i)
                    };
                    (format!("n{i:05}"), t)
                })
                .collect();
            let mut plan = SymTabPlan::new(&links, sizes).unwrap();
            let mut alloc = Allocator::new(0, sizes);
            plan.allocate(&mut alloc).unwrap();
            let mut out: Vec<u8> = Vec::new();
            plan.write(&|id| Ok((10_000 + id as u64, None)), &mut out)
                .unwrap();
            out.resize(alloc.eoa() as usize, 0);
            let heap = LocalHeap::read(&out, plan.heap_address(), sizes).unwrap();
            assert_eq!(heap.free_blocks().unwrap().len(), 1);
            let snods = group_btree_children(&out, plan.btree_address(), sizes).unwrap();
            let mut seen = Vec::new();
            for addr in snods {
                let node = SymbolTableNode::read(&out, addr, sizes).unwrap();
                for e in node.entries {
                    let name = heap.get_str(e.link_name_offset).unwrap().to_owned();
                    let i: usize = name[1..].parse().unwrap();
                    match e.scratch {
                        ScratchPad::SoftLink { link_value_offset } => {
                            assert_eq!(
                                heap.get_str(u64::from(link_value_offset)).unwrap(),
                                format!("/target/{i}")
                            );
                        }
                        _ => assert_eq!(e.object_header_address, 10_000 + i as u64),
                    }
                    seen.push(name);
                }
            }
            let mut expect: Vec<String> = links.iter().map(|l| l.0.clone()).collect();
            expect.sort();
            assert_eq!(seen, expect, "n = {n}");
        }
    }

    #[test]
    fn downgrades_versions() {
        let i32le = Datatype::integer(4, true, ByteOrder::LittleEndian);
        let c = Datatype::packed_compound(vec![
            ("a".into(), i32le.clone()),
            ("b".into(), Datatype::array(i32le.clone(), &[2])),
        ])
        .unwrap();
        let d = downgrade(&c);
        assert_eq!(d.version, 2);
        let DatatypeClass::Compound { members } = &d.class else {
            panic!()
        };
        assert_eq!(members[1].datatype.version, 2);
        assert_eq!(downgrade(&Datatype::bool_enum()).version, 1);
        let e = d.encode().unwrap();
        let back = Datatype::parse(&e).unwrap();
        assert_eq!(back.size, c.size);
    }
}
