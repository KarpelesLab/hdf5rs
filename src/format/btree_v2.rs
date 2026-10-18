//! Version 2 B-trees (spec: "Version 2 B-trees", level 1A2).
//!
//! A v2 B-tree consists of a header (`BTHD`), internal nodes (`BTIN`) and
//! leaf nodes (`BTLF`). Every node occupies `node size` bytes on disk, but
//! only the used prefix (records, child pointers) is covered by the checksum
//! that immediately follows it.
//!
//! # Variable-width child pointer fields
//!
//! In an internal node, each child pointer is a triplet
//! `(address, number of records in child, total records in child subtree)`;
//! the last field is omitted in "twig" nodes (depth 1, just above leaves).
//! The two count fields use the fewest bytes able to represent the largest
//! value they could hold, derived level by level from the bottom:
//!
//! * leaf (depth 0): `max(0) = (node_size - 10) / record_size`, where 10 is
//!   the fixed node overhead (signature 4 + version 1 + type 1 + checksum 4);
//!   `cum(0) = max(0)`.
//! * internal node at depth `d >= 1`:
//!   `nrec_width(d) = bytes(max(d-1))`,
//!   `total_width(d) = bytes(cum(d-1))` if `d > 1`, else 0,
//!   `ptr(d) = size_of_offsets + nrec_width(d) + total_width(d)`,
//!   `max(d) = (node_size - 10 - ptr(d)) / (record_size + ptr(d))`
//!   (one pointer more than records), and
//!   `cum(d) = max(d) + (max(d) + 1) * cum(d-1)`.
//!
//! `bytes(v)` is the least number of bytes that can encode `v` (at least 1,
//! at most 8). The spec describes `cum` informally as "multiplying these two
//! values together"; we use the exact maximum subtree size above, which is
//! what the field is defined to bound.

use std::cmp::Ordering;
use std::collections::HashSet;

use super::{ByteReader, Sizes, UNDEFINED_ADDRESS};
use crate::checksum::metadata_checksum;
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Fixed per-node overhead: signature, version, type and checksum.
const NODE_OVERHEAD: u64 = 10;
/// Maximum supported tree depth.
const MAX_DEPTH: u16 = 64;
/// Maximum supported node size (bytes).
const MAX_NODE_SIZE: u32 = 16 << 20;
/// Maximum dataset rank for chunk records (types 10/11).
const MAX_CHUNK_NDIMS: usize = 32;

/// Default heap ID width for link records (types 5 and 6).
pub const LINK_HEAP_ID_LEN: usize = 7;
/// Default heap ID width for SOHM and attribute records (types 7, 8, 9).
pub const MESSAGE_HEAP_ID_LEN: usize = 8;

/// Extra information required to decode some record types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecordContext {
    /// Width of heap ID fields in record types 5–9. `None` uses the spec's
    /// fixed widths ([`LINK_HEAP_ID_LEN`] for types 5/6,
    /// [`MESSAGE_HEAP_ID_LEN`] for types 7–9); otherwise the fractal heap's
    /// "Heap ID Length".
    pub heap_id_len: Option<usize>,
    /// Number of scaled offsets stored per chunk record (types 10/11), i.e.
    /// the dataset rank.
    pub chunk_ndims: usize,
    /// Width in bytes of the "Chunk Size" field of type 11 records (1..=8).
    pub chunk_size_len: usize,
}

/// One decoded v2 B-tree record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    /// Type 1: indirectly accessed, non-filtered huge fractal heap object.
    HugeIndirect {
        /// Address of the object in the file.
        address: u64,
        /// Length of the object in the file.
        length: u64,
        /// Huge object ID (the key stored in the heap ID).
        id: u64,
    },
    /// Type 2: indirectly accessed, filtered huge fractal heap object.
    HugeIndirectFiltered {
        /// Address of the filtered object in the file.
        address: u64,
        /// Length of the filtered object in the file.
        length: u64,
        /// Mask of skipped filters.
        filter_mask: u32,
        /// Size of the de-filtered object.
        memory_size: u64,
        /// Huge object ID (the key stored in the heap ID).
        id: u64,
    },
    /// Type 3: directly accessed, non-filtered huge fractal heap object.
    HugeDirect {
        /// Address of the object in the file.
        address: u64,
        /// Length of the object in the file.
        length: u64,
    },
    /// Type 4: directly accessed, filtered huge fractal heap object.
    HugeDirectFiltered {
        /// Address of the filtered object in the file.
        address: u64,
        /// Length of the filtered object in the file.
        length: u64,
        /// Mask of skipped filters.
        filter_mask: u32,
        /// Size of the de-filtered object.
        memory_size: u64,
    },
    /// Type 5: link name index of an indexed group.
    LinkName {
        /// lookup3 hash of the link name.
        hash: u32,
        /// Heap ID of the link message in the group's fractal heap.
        heap_id: Vec<u8>,
    },
    /// Type 6: link creation order index of an indexed group.
    LinkCreationOrder {
        /// Creation order of the link.
        creation_order: u64,
        /// Heap ID of the link message in the group's fractal heap.
        heap_id: Vec<u8>,
    },
    /// Type 7, sub-type 0: shared message stored in the SOHM heap.
    SharedMessageInHeap {
        /// lookup3 hash of the encoded message.
        hash: u32,
        /// Number of objects referencing the message.
        reference_count: u32,
        /// Heap ID of the message in the index's fractal heap.
        heap_id: Vec<u8>,
    },
    /// Type 7, sub-type 1: shared message stored in an object header.
    SharedMessageInObjectHeader {
        /// lookup3 hash of the encoded message.
        hash: u32,
        /// Object header message type.
        message_type: u8,
        /// The message is the n-th of its type in the header.
        object_header_index: u16,
        /// Address of the object header holding the message.
        object_header_address: u64,
    },
    /// Type 8: attribute name index.
    AttributeName {
        /// Heap ID of the attribute message in the attribute heap.
        heap_id: Vec<u8>,
        /// Object header message flags of the attribute message.
        message_flags: u8,
        /// Creation order of the attribute.
        creation_order: u32,
        /// lookup3 hash of the attribute name.
        hash: u32,
    },
    /// Type 9: attribute creation order index.
    AttributeCreationOrder {
        /// Heap ID of the attribute message in the attribute heap.
        heap_id: Vec<u8>,
        /// Object header message flags of the attribute message.
        message_flags: u8,
        /// Creation order of the attribute.
        creation_order: u32,
    },
    /// Type 10: non-filtered dataset chunk.
    Chunk {
        /// Address of the chunk.
        address: u64,
        /// Chunk offset divided by chunk dimensions, slowest dimension first.
        scaled_offsets: Vec<u64>,
    },
    /// Type 11: filtered dataset chunk.
    FilteredChunk {
        /// Address of the chunk.
        address: u64,
        /// Stored (filtered) size of the chunk.
        chunk_size: u64,
        /// Mask of skipped filters.
        filter_mask: u32,
        /// Chunk offset divided by chunk dimensions, slowest dimension first.
        scaled_offsets: Vec<u64>,
    },
}

impl Record {
    /// Minimal encoded size of a record of `tree_type`.
    fn min_size(tree_type: u8, sizes: Sizes, ctx: &RecordContext) -> Result<usize> {
        let o = sizes.offset as usize;
        let l = sizes.length as usize;
        let hid = |default: usize| ctx.heap_id_len.unwrap_or(default);
        Ok(match tree_type {
            1 => o + 2 * l,
            2 => o + 3 * l + 4,
            3 => o + l,
            4 => o + 2 * l + 4,
            5 => 4 + hid(LINK_HEAP_ID_LEN),
            6 => 8 + hid(LINK_HEAP_ID_LEN),
            // Sub-type 0 is 1 + 4 + 4 + id; sub-type 1 is 1 + 4 + 4 + O.
            7 => 9 + hid(MESSAGE_HEAP_ID_LEN).min(o),
            8 => hid(MESSAGE_HEAP_ID_LEN) + 9,
            9 => hid(MESSAGE_HEAP_ID_LEN) + 5,
            10 | 11 => {
                if ctx.chunk_ndims == 0 || ctx.chunk_ndims > MAX_CHUNK_NDIMS {
                    return Err(Error::invalid("bad chunk dimensionality for v2 B-tree"));
                }
                let base = o + 8 * ctx.chunk_ndims;
                if tree_type == 11 {
                    if !(1..=8).contains(&ctx.chunk_size_len) {
                        return Err(Error::invalid("bad chunk size field width"));
                    }
                    base + ctx.chunk_size_len + 4
                } else {
                    base
                }
            }
            0 => return Err(Error::unsupported("v2 B-tree type 0 (testing only)")),
            t => return Err(Error::invalid(format!("unknown v2 B-tree type {t}"))),
        })
    }

    /// Decode one record of `tree_type` from `data` (which may be longer than
    /// the record).
    pub fn parse(tree_type: u8, data: &[u8], sizes: Sizes, ctx: &RecordContext) -> Result<Record> {
        let mut r = ByteReader::with_sizes(data, sizes);
        let heap_id = |r: &mut ByteReader<'_>, default: usize| -> Result<Vec<u8>> {
            Ok(r.read_bytes(ctx.heap_id_len.unwrap_or(default))?.to_vec())
        };
        let scaled = |r: &mut ByteReader<'_>| -> Result<Vec<u64>> {
            if ctx.chunk_ndims == 0 || ctx.chunk_ndims > MAX_CHUNK_NDIMS {
                return Err(Error::invalid("bad chunk dimensionality for v2 B-tree"));
            }
            (0..ctx.chunk_ndims).map(|_| r.read_u64()).collect()
        };
        Ok(match tree_type {
            1 => Record::HugeIndirect {
                address: r.read_offset()?,
                length: r.read_length()?,
                id: r.read_length()?,
            },
            2 => Record::HugeIndirectFiltered {
                address: r.read_offset()?,
                length: r.read_length()?,
                filter_mask: r.read_u32()?,
                memory_size: r.read_length()?,
                id: r.read_length()?,
            },
            3 => Record::HugeDirect {
                address: r.read_offset()?,
                length: r.read_length()?,
            },
            4 => Record::HugeDirectFiltered {
                address: r.read_offset()?,
                length: r.read_length()?,
                filter_mask: r.read_u32()?,
                memory_size: r.read_length()?,
            },
            5 => Record::LinkName {
                hash: r.read_u32()?,
                heap_id: heap_id(&mut r, LINK_HEAP_ID_LEN)?,
            },
            6 => Record::LinkCreationOrder {
                creation_order: r.read_u64()?,
                heap_id: heap_id(&mut r, LINK_HEAP_ID_LEN)?,
            },
            7 => match r.read_u8()? {
                0 => Record::SharedMessageInHeap {
                    hash: r.read_u32()?,
                    reference_count: r.read_u32()?,
                    heap_id: heap_id(&mut r, MESSAGE_HEAP_ID_LEN)?,
                },
                1 => {
                    let hash = r.read_u32()?;
                    let _reserved = r.read_u8()?;
                    Record::SharedMessageInObjectHeader {
                        hash,
                        message_type: r.read_u8()?,
                        object_header_index: r.read_u16()?,
                        object_header_address: r.read_offset()?,
                    }
                }
                other => {
                    return Err(Error::invalid(format!(
                        "unknown shared message location {other}"
                    )));
                }
            },
            8 => Record::AttributeName {
                heap_id: heap_id(&mut r, MESSAGE_HEAP_ID_LEN)?,
                message_flags: r.read_u8()?,
                creation_order: r.read_u32()?,
                hash: r.read_u32()?,
            },
            9 => Record::AttributeCreationOrder {
                heap_id: heap_id(&mut r, MESSAGE_HEAP_ID_LEN)?,
                message_flags: r.read_u8()?,
                creation_order: r.read_u32()?,
            },
            10 => Record::Chunk {
                address: r.read_offset()?,
                scaled_offsets: scaled(&mut r)?,
            },
            11 => {
                let address = r.read_offset()?;
                if !(1..=8).contains(&ctx.chunk_size_len) {
                    return Err(Error::invalid("bad chunk size field width"));
                }
                let chunk_size = r.read_uint(ctx.chunk_size_len)?;
                Record::FilteredChunk {
                    address,
                    chunk_size,
                    filter_mask: r.read_u32()?,
                    scaled_offsets: scaled(&mut r)?,
                }
            }
            0 => return Err(Error::unsupported("v2 B-tree type 0 (testing only)")),
            t => return Err(Error::invalid(format!("unknown v2 B-tree type {t}"))),
        })
    }
}

/// The v2 B-tree header (`BTHD`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BTreeV2Header {
    /// B-tree type (1..=11), selecting the record format.
    pub tree_type: u8,
    /// Size in bytes of every node.
    pub node_size: u32,
    /// Size in bytes of one record.
    pub record_size: u16,
    /// Depth of the tree (0: the root is a leaf).
    pub depth: u16,
    /// Split threshold, percent.
    pub split_percent: u8,
    /// Merge threshold, percent.
    pub merge_percent: u8,
    /// Address of the root node ([`UNDEFINED_ADDRESS`] when empty).
    pub root_address: u64,
    /// Number of records in the root node.
    pub root_num_records: u16,
    /// Total number of records in the tree.
    pub total_records: u64,
}

impl BTreeV2Header {
    /// Encoded size of the header for the given widths.
    pub fn encoded_size(sizes: Sizes) -> usize {
        4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + sizes.offset as usize + 2 + sizes.length as usize + 4
    }

    /// Parse and checksum-verify a header from `data`.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        r.expect_signature(b"BTHD")?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "v2 B-tree header version {version}"
            )));
        }
        let h = BTreeV2Header {
            tree_type: r.read_u8()?,
            node_size: r.read_u32()?,
            record_size: r.read_u16()?,
            depth: r.read_u16()?,
            split_percent: r.read_u8()?,
            merge_percent: r.read_u8()?,
            root_address: r.read_offset()?,
            root_num_records: r.read_u16()?,
            total_records: r.read_length()?,
        };
        verify_checksum(&mut r, "v2 B-tree header")?;
        Ok(h)
    }
}

/// Read a stored checksum at the reader position and compare it against the
/// checksum of all preceding bytes.
fn verify_checksum(r: &mut ByteReader<'_>, what: &'static str) -> Result<()> {
    let end = r.position();
    let actual = metadata_checksum(r.data().get(..end).ok_or(Error::UnexpectedEof)?);
    let expected = r.read_u32()?;
    if expected != actual {
        return Err(Error::ChecksumMismatch {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Least number of bytes able to encode `v` (1..=8).
fn bytes_for(v: u64) -> usize {
    (64 - v.leading_zeros() as usize).div_ceil(8).max(1)
}

/// Per-depth node geometry, see the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Level {
    /// Maximum number of records in a node at this depth.
    max_records: u64,
    /// Maximum number of records in a subtree rooted at this depth.
    cum_max_records: u64,
    /// Width of "number of records in child" fields in nodes at this depth.
    nrec_width: usize,
    /// Width of "total number of records in child" fields (0 if absent).
    total_width: usize,
}

fn compute_levels(header: &BTreeV2Header, sizes: Sizes) -> Result<Vec<Level>> {
    let node = u64::from(header.node_size);
    let rec = u64::from(header.record_size);
    let usable = node
        .checked_sub(NODE_OVERHEAD)
        .ok_or_else(|| Error::invalid("v2 B-tree node size too small"))?;
    let leaf_max = usable / rec;
    if leaf_max == 0 {
        return Err(Error::invalid("v2 B-tree node cannot hold a record"));
    }
    let mut levels = vec![Level {
        max_records: leaf_max,
        cum_max_records: leaf_max,
        nrec_width: 0,
        total_width: 0,
    }];
    for d in 1..=usize::from(header.depth) {
        let below = levels[d - 1];
        let nrec_width = bytes_for(below.max_records);
        let total_width = if d > 1 {
            bytes_for(below.cum_max_records)
        } else {
            0
        };
        let ptr = u64::from(sizes.offset) + nrec_width as u64 + total_width as u64;
        let max_records = usable
            .checked_sub(ptr)
            .map(|avail| avail / (rec + ptr))
            .unwrap_or(0);
        if max_records == 0 {
            return Err(Error::invalid(
                "v2 B-tree internal node cannot hold a record",
            ));
        }
        let cum_max_records = max_records
            .checked_add(1)
            .and_then(|c| c.checked_mul(below.cum_max_records))
            .and_then(|c| c.checked_add(max_records))
            .unwrap_or(u64::MAX);
        levels.push(Level {
            max_records,
            cum_max_records,
            nrec_width,
            total_width,
        });
    }
    Ok(levels)
}

/// A child pointer of an internal node.
#[derive(Debug, Clone, Copy)]
struct ChildPtr {
    address: u64,
    num_records: u64,
}

/// A decoded node.
#[derive(Debug)]
struct Node {
    records: Vec<Record>,
    children: Vec<ChildPtr>,
}

/// Traversal bookkeeping: cycle detection and work limits.
struct Walk {
    visited: HashSet<u64>,
    nodes_left: u64,
}

impl Walk {
    fn enter(&mut self, addr: u64) -> Result<()> {
        if !self.visited.insert(addr) {
            return Err(Error::invalid("cycle in v2 B-tree"));
        }
        self.nodes_left = self
            .nodes_left
            .checked_sub(1)
            .ok_or(Error::LimitExceeded("too many v2 B-tree nodes"))?;
        Ok(())
    }
}

/// An opened version 2 B-tree.
#[derive(Debug, Clone)]
pub struct BTreeV2 {
    /// Address of the header.
    pub address: u64,
    /// The decoded header.
    pub header: BTreeV2Header,
    sizes: Sizes,
    ctx: RecordContext,
    levels: Vec<Level>,
}

impl BTreeV2 {
    /// Read and validate the header at `addr`.
    pub fn open<R: ReadAt + ?Sized>(
        store: &R,
        addr: u64,
        sizes: Sizes,
        ctx: RecordContext,
    ) -> Result<Self> {
        let buf = store.read_vec_at(addr, BTreeV2Header::encoded_size(sizes))?;
        let header = BTreeV2Header::parse(&buf, sizes)?;
        if header.depth > MAX_DEPTH {
            return Err(Error::LimitExceeded("v2 B-tree depth"));
        }
        if header.node_size > MAX_NODE_SIZE {
            return Err(Error::LimitExceeded("v2 B-tree node size"));
        }
        let min = Record::min_size(header.tree_type, sizes, &ctx)?;
        if usize::from(header.record_size) < min {
            return Err(Error::invalid("v2 B-tree record size too small for type"));
        }
        let levels = compute_levels(&header, sizes)?;
        if header.root_address == UNDEFINED_ADDRESS {
            if header.total_records != 0 || header.root_num_records != 0 {
                return Err(Error::invalid("empty v2 B-tree with records"));
            }
        } else {
            let top = levels[usize::from(header.depth)];
            if u64::from(header.root_num_records) > top.max_records
                || header.total_records > top.cum_max_records
            {
                return Err(Error::invalid("v2 B-tree record count exceeds capacity"));
            }
            // Every record occupies `record_size` distinct bytes of the file.
            let bytes = header
                .total_records
                .checked_mul(u64::from(header.record_size));
            if bytes.is_none_or(|b| b > store.size().unwrap_or(u64::MAX)) {
                return Err(Error::LimitExceeded("v2 B-tree record count"));
            }
        }
        Ok(BTreeV2 {
            address: addr,
            header,
            sizes,
            ctx,
            levels,
        })
    }

    /// Widths of the (number of records, total number of records) fields in
    /// child pointers of nodes at `depth`, or `None` for leaves / out of
    /// range depths.
    pub fn child_field_widths(&self, depth: u16) -> Option<(usize, usize)> {
        match self.levels.get(usize::from(depth)) {
            Some(l) if depth > 0 => Some((l.nrec_width, l.total_width)),
            _ => None,
        }
    }

    /// Maximum number of records a node at `depth` can hold.
    pub fn max_records(&self, depth: u16) -> Option<u64> {
        self.levels.get(usize::from(depth)).map(|l| l.max_records)
    }

    fn new_walk(&self) -> Walk {
        Walk {
            visited: HashSet::new(),
            // Each node holds at least one record, except possibly the root.
            nodes_left: self
                .header
                .total_records
                .saturating_mul(2)
                .saturating_add(2),
        }
    }

    fn read_node<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        addr: u64,
        depth: u16,
        num_records: u64,
    ) -> Result<Node> {
        if addr == UNDEFINED_ADDRESS {
            return Err(Error::invalid("undefined v2 B-tree node address"));
        }
        let level = self
            .levels
            .get(usize::from(depth))
            .ok_or_else(|| Error::invalid("v2 B-tree depth out of range"))?;
        if num_records > level.max_records {
            return Err(Error::invalid("v2 B-tree node record count too large"));
        }
        let buf = store.read_vec_at(addr, self.header.node_size as usize)?;
        let mut r = ByteReader::with_sizes(&buf, self.sizes);
        r.expect_signature(if depth == 0 { b"BTLF" } else { b"BTIN" })?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "v2 B-tree node version {version}"
            )));
        }
        if r.read_u8()? != self.header.tree_type {
            return Err(Error::invalid("v2 B-tree node type mismatch"));
        }
        // `num_records <= max_records`, which fits in the node, so this is
        // a small value.
        let n = num_records as usize;
        let mut records = Vec::with_capacity(n);
        for _ in 0..n {
            let raw = r.read_bytes(usize::from(self.header.record_size))?;
            records.push(Record::parse(
                self.header.tree_type,
                raw,
                self.sizes,
                &self.ctx,
            )?);
        }
        let mut children = Vec::new();
        if depth > 0 {
            let below = self.levels[usize::from(depth) - 1];
            children.reserve(n + 1);
            for _ in 0..=n {
                let address = r.read_offset()?;
                let num_records = r.read_uint(level.nrec_width)?;
                let total = if level.total_width > 0 {
                    r.read_uint(level.total_width)?
                } else {
                    num_records
                };
                if num_records > below.max_records
                    || total < num_records
                    || total > below.cum_max_records
                {
                    return Err(Error::invalid("v2 B-tree child record count invalid"));
                }
                children.push(ChildPtr {
                    address,
                    num_records,
                });
            }
        }
        verify_checksum(
            &mut r,
            if depth == 0 {
                "v2 B-tree leaf node"
            } else {
                "v2 B-tree internal node"
            },
        )?;
        Ok(Node { records, children })
    }

    /// Visit all records in key order. The visitor returns `Ok(false)` to
    /// stop early.
    pub fn visit<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        visitor: &mut dyn FnMut(&Record) -> Result<bool>,
    ) -> Result<()> {
        if self.header.root_address == UNDEFINED_ADDRESS {
            return Ok(());
        }
        let mut walk = self.new_walk();
        self.visit_node(
            store,
            self.header.root_address,
            self.header.depth,
            u64::from(self.header.root_num_records),
            &mut walk,
            visitor,
        )?;
        Ok(())
    }

    fn visit_node<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        addr: u64,
        depth: u16,
        num_records: u64,
        walk: &mut Walk,
        visitor: &mut dyn FnMut(&Record) -> Result<bool>,
    ) -> Result<bool> {
        walk.enter(addr)?;
        let node = self.read_node(store, addr, depth, num_records)?;
        for (i, rec) in node.records.iter().enumerate() {
            if let Some(c) = node.children.get(i)
                && !self.visit_node(store, c.address, depth - 1, c.num_records, walk, visitor)?
            {
                return Ok(false);
            }
            if !visitor(rec)? {
                return Ok(false);
            }
        }
        if let Some(c) = node.children.last() {
            return self.visit_node(store, c.address, depth - 1, c.num_records, walk, visitor);
        }
        Ok(true)
    }

    /// All records, in key order.
    pub fn records<R: ReadAt + ?Sized>(&self, store: &R) -> Result<Vec<Record>> {
        let mut out = Vec::new();
        let limit = self.header.total_records;
        self.visit(store, &mut |rec| {
            if out.len() as u64 >= limit {
                return Err(Error::invalid("v2 B-tree holds more records than declared"));
            }
            out.push(rec.clone());
            Ok(true)
        })?;
        Ok(out)
    }

    /// All records for which `cmp` returns [`Ordering::Equal`], in key order.
    ///
    /// `cmp(record)` must return the ordering of `record` relative to the
    /// searched key (as for [`slice::binary_search_by`]) and be consistent
    /// with the tree's key order. Subtrees that cannot contain matches are
    /// skipped. Useful for hash-keyed trees where several records may share
    /// a hash.
    pub fn find_all<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        cmp: impl Fn(&Record) -> Ordering,
    ) -> Result<Vec<Record>> {
        let mut out = Vec::new();
        self.search(store, &cmp, &mut out, false)?;
        Ok(out)
    }

    /// The first record (in key order) for which `cmp` returns
    /// [`Ordering::Equal`]; see [`find_all`](Self::find_all).
    pub fn find<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        cmp: impl Fn(&Record) -> Ordering,
    ) -> Result<Option<Record>> {
        let mut out = Vec::new();
        self.search(store, &cmp, &mut out, true)?;
        Ok(out.into_iter().next())
    }

    fn search<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        cmp: &dyn Fn(&Record) -> Ordering,
        out: &mut Vec<Record>,
        first_only: bool,
    ) -> Result<()> {
        if self.header.root_address == UNDEFINED_ADDRESS {
            return Ok(());
        }
        let mut walk = self.new_walk();
        self.search_node(
            store,
            self.header.root_address,
            self.header.depth,
            u64::from(self.header.root_num_records),
            &mut walk,
            cmp,
            out,
            first_only,
        )?;
        Ok(())
    }

    /// Returns `false` once the search is complete.
    #[allow(clippy::too_many_arguments)]
    fn search_node<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        addr: u64,
        depth: u16,
        num_records: u64,
        walk: &mut Walk,
        cmp: &dyn Fn(&Record) -> Ordering,
        out: &mut Vec<Record>,
        first_only: bool,
    ) -> Result<bool> {
        walk.enter(addr)?;
        let node = self.read_node(store, addr, depth, num_records)?;
        let ords: Vec<Ordering> = node.records.iter().map(cmp).collect();
        let n = ords.len();
        for i in 0..=n {
            // Child `i` holds keys between record `i - 1` and record `i`.
            let lower_ok = i == 0 || ords[i - 1] != Ordering::Greater;
            let upper_ok = i == n || ords[i] != Ordering::Less;
            if lower_ok
                && upper_ok
                && let Some(c) = node.children.get(i)
                && !self.search_node(
                    store,
                    c.address,
                    depth - 1,
                    c.num_records,
                    walk,
                    cmp,
                    out,
                    first_only,
                )?
            {
                return Ok(false);
            }
            if i < n {
                match ords[i] {
                    Ordering::Equal => {
                        out.push(node.records[i].clone());
                        if first_only {
                            return Ok(false);
                        }
                    }
                    Ordering::Greater => return Ok(false),
                    Ordering::Less => {}
                }
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A growable in-memory "file" for building test structures.
    #[derive(Default)]
    pub(crate) struct FileBuilder {
        pub data: Vec<u8>,
    }

    impl FileBuilder {
        pub fn new() -> Self {
            // Start with some padding so address 0 is never used.
            FileBuilder { data: vec![0; 16] }
        }

        /// Append `bytes` (8-byte aligned) and return their address.
        pub fn place(&mut self, bytes: &[u8]) -> u64 {
            while !self.data.len().is_multiple_of(8) {
                self.data.push(0);
            }
            let addr = self.data.len() as u64;
            self.data.extend_from_slice(bytes);
            addr
        }

        /// Reserve `len` zero bytes and return their address.
        pub fn reserve(&mut self, len: usize) -> u64 {
            self.place(&vec![0; len])
        }

        pub fn write_at(&mut self, addr: u64, bytes: &[u8]) {
            let a = addr as usize;
            self.data[a..a + bytes.len()].copy_from_slice(bytes);
        }
    }

    pub(crate) fn push_uint(out: &mut Vec<u8>, v: u64, width: usize) {
        out.extend_from_slice(&v.to_le_bytes()[..width]);
    }

    pub(crate) fn seal(mut v: Vec<u8>) -> Vec<u8> {
        let c = metadata_checksum(&v);
        v.extend_from_slice(&c.to_le_bytes());
        v
    }

    pub(crate) struct HeaderSpec {
        pub tree_type: u8,
        pub node_size: u32,
        pub record_size: u16,
        pub depth: u16,
        pub root: u64,
        pub root_n: u16,
        pub total: u64,
    }

    pub(crate) fn header_bytes(h: &HeaderSpec) -> Vec<u8> {
        let mut v = b"BTHD".to_vec();
        v.push(0);
        v.push(h.tree_type);
        v.extend_from_slice(&h.node_size.to_le_bytes());
        v.extend_from_slice(&h.record_size.to_le_bytes());
        v.extend_from_slice(&h.depth.to_le_bytes());
        v.push(100);
        v.push(40);
        v.extend_from_slice(&h.root.to_le_bytes());
        v.extend_from_slice(&h.root_n.to_le_bytes());
        v.extend_from_slice(&h.total.to_le_bytes());
        seal(v)
    }

    pub(crate) fn leaf_bytes(tree_type: u8, node_size: u32, records: &[Vec<u8>]) -> Vec<u8> {
        let mut v = b"BTLF".to_vec();
        v.push(0);
        v.push(tree_type);
        for r in records {
            v.extend_from_slice(r);
        }
        let mut v = seal(v);
        v.resize(node_size as usize, 0);
        v
    }

    /// `children`: (address, nrecords, total) triplets.
    pub(crate) fn internal_bytes(
        tree_type: u8,
        node_size: u32,
        records: &[Vec<u8>],
        children: &[(u64, u64, u64)],
        widths: (usize, usize),
    ) -> Vec<u8> {
        let mut v = b"BTIN".to_vec();
        v.push(0);
        v.push(tree_type);
        for r in records {
            v.extend_from_slice(r);
        }
        for &(a, n, t) in children {
            push_uint(&mut v, a, 8);
            push_uint(&mut v, n, widths.0);
            if widths.1 > 0 {
                push_uint(&mut v, t, widths.1);
            }
        }
        let mut v = seal(v);
        v.resize(node_size as usize, 0);
        v
    }

    /// Type 9 record (attribute creation order), 13 bytes.
    fn rec9(order: u32) -> Vec<u8> {
        let mut v = vec![0xa0, 0, 0, 0, 0, 0, 0, order as u8];
        v.push(0);
        v.extend_from_slice(&order.to_le_bytes());
        v
    }

    fn order_of(r: &Record) -> u32 {
        match r {
            Record::AttributeCreationOrder { creation_order, .. } => *creation_order,
            _ => panic!("unexpected record {r:?}"),
        }
    }

    const NODE: u32 = 512;

    /// Depth-2 tree of type 9 records 0..=10:
    /// root [5] -> internal [2] (leaves [0,1], [3,4]) and [8] (leaves
    /// [6,7], [9,10]).
    fn depth2_tree() -> (FileBuilder, u64) {
        let mut f = FileBuilder::new();
        let leaves: Vec<u64> = [[0, 1], [3, 4], [6, 7], [9, 10]]
            .iter()
            .map(|pair| {
                let recs: Vec<_> = pair.iter().map(|&o| rec9(o)).collect();
                f.place(&leaf_bytes(9, NODE, &recs))
            })
            .collect();
        let i1 = f.place(&internal_bytes(
            9,
            NODE,
            &[rec9(2)],
            &[(leaves[0], 2, 2), (leaves[1], 2, 2)],
            (1, 0),
        ));
        let i2 = f.place(&internal_bytes(
            9,
            NODE,
            &[rec9(8)],
            &[(leaves[2], 2, 2), (leaves[3], 2, 2)],
            (1, 0),
        ));
        let root = f.place(&internal_bytes(
            9,
            NODE,
            &[rec9(5)],
            &[(i1, 1, 5), (i2, 1, 5)],
            (1, 2),
        ));
        let hdr = f.place(&header_bytes(&HeaderSpec {
            tree_type: 9,
            node_size: NODE,
            record_size: 13,
            depth: 2,
            root,
            root_n: 1,
            total: 11,
        }));
        (f, hdr)
    }

    #[test]
    fn widths_follow_spec_derivation() {
        let h = BTreeV2Header {
            tree_type: 9,
            node_size: 512,
            record_size: 13,
            depth: 3,
            split_percent: 100,
            merge_percent: 40,
            root_address: 0,
            root_num_records: 0,
            total_records: 0,
        };
        let levels = compute_levels(&h, Sizes::DEFAULT).unwrap();
        // Leaf: (512 - 10) / 13 = 38.
        assert_eq!(levels[0].max_records, 38);
        // Twig: pointer 8 + 1 = 9 bytes; (502 - 9) / (13 + 9) = 22.
        assert_eq!((levels[1].nrec_width, levels[1].total_width), (1, 0));
        assert_eq!(levels[1].max_records, 22);
        assert_eq!(levels[1].cum_max_records, 22 + 23 * 38);
        // Depth 2: pointer 8 + 1 + 2 = 11; (502 - 11) / 24 = 20.
        assert_eq!((levels[2].nrec_width, levels[2].total_width), (1, 2));
        assert_eq!(levels[2].max_records, 20);
        let cum2 = 20 + 21 * (22 + 23 * 38);
        assert_eq!(levels[2].cum_max_records, cum2);
        // Depth 3: totals still fit in 2 bytes (cum2 = 18_836).
        assert_eq!((levels[3].nrec_width, levels[3].total_width), (1, 2));
        assert_eq!(bytes_for(cum2), 2);
        assert_eq!(bytes_for(0), 1);
        assert_eq!(bytes_for(255), 1);
        assert_eq!(bytes_for(256), 2);
        assert_eq!(bytes_for(u64::MAX), 8);
    }

    #[test]
    fn depth0_tree() {
        let mut f = FileBuilder::new();
        let recs: Vec<_> = (0..5).map(rec9).collect();
        let leaf = f.place(&leaf_bytes(9, NODE, &recs));
        let hdr = f.place(&header_bytes(&HeaderSpec {
            tree_type: 9,
            node_size: NODE,
            record_size: 13,
            depth: 0,
            root: leaf,
            root_n: 5,
            total: 5,
        }));
        let t = BTreeV2::open(&f.data, hdr, Sizes::DEFAULT, RecordContext::default()).unwrap();
        let all = t.records(&f.data).unwrap();
        assert_eq!(
            all.iter().map(order_of).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
        let hit = t.find(&f.data, |r| order_of(r).cmp(&3)).unwrap().unwrap();
        assert_eq!(order_of(&hit), 3);
        assert!(t.find(&f.data, |r| order_of(r).cmp(&9)).unwrap().is_none());
    }

    #[test]
    fn depth1_tree() {
        let mut f = FileBuilder::new();
        let l0 = f.place(&leaf_bytes(9, NODE, &[rec9(0), rec9(1)]));
        let l1 = f.place(&leaf_bytes(9, NODE, &[rec9(3)]));
        let l2 = f.place(&leaf_bytes(9, NODE, &[rec9(5), rec9(6), rec9(7)]));
        let root = f.place(&internal_bytes(
            9,
            NODE,
            &[rec9(2), rec9(4)],
            &[(l0, 2, 2), (l1, 1, 1), (l2, 3, 3)],
            (1, 0),
        ));
        let hdr = f.place(&header_bytes(&HeaderSpec {
            tree_type: 9,
            node_size: NODE,
            record_size: 13,
            depth: 1,
            root,
            root_n: 2,
            total: 8,
        }));
        let t = BTreeV2::open(&f.data, hdr, Sizes::DEFAULT, RecordContext::default()).unwrap();
        assert_eq!(t.child_field_widths(1), Some((1, 0)));
        let all = t.records(&f.data).unwrap();
        assert_eq!(
            all.iter().map(order_of).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        for k in 0..8 {
            let hit = t.find(&f.data, |r| order_of(r).cmp(&k)).unwrap().unwrap();
            assert_eq!(order_of(&hit), k);
        }
        // Range query: records 3..=5.
        let range = t
            .find_all(&f.data, |r| {
                let o = order_of(r);
                if o < 3 {
                    Ordering::Less
                } else if o > 5 {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .unwrap();
        assert_eq!(range.iter().map(order_of).collect::<Vec<_>>(), [3, 4, 5]);
    }

    #[test]
    fn depth2_tree_with_wide_totals() {
        let (f, hdr) = depth2_tree();
        let t = BTreeV2::open(&f.data, hdr, Sizes::DEFAULT, RecordContext::default()).unwrap();
        assert_eq!(t.child_field_widths(2), Some((1, 2)));
        let all = t.records(&f.data).unwrap();
        assert_eq!(
            all.iter().map(order_of).collect::<Vec<_>>(),
            (0..=10).collect::<Vec<_>>()
        );
        for k in 0..=10 {
            let hit = t.find(&f.data, |r| order_of(r).cmp(&k)).unwrap().unwrap();
            assert_eq!(order_of(&hit), k);
        }
        // Early stop.
        let mut seen = 0;
        t.visit(&f.data, &mut |_| {
            seen += 1;
            Ok(seen < 4)
        })
        .unwrap();
        assert_eq!(seen, 4);
    }

    #[test]
    fn empty_tree() {
        let mut f = FileBuilder::new();
        let hdr = f.place(&header_bytes(&HeaderSpec {
            tree_type: 5,
            node_size: NODE,
            record_size: 11,
            depth: 0,
            root: UNDEFINED_ADDRESS,
            root_n: 0,
            total: 0,
        }));
        let t = BTreeV2::open(&f.data, hdr, Sizes::DEFAULT, RecordContext::default()).unwrap();
        assert!(t.records(&f.data).unwrap().is_empty());
        assert!(t.find(&f.data, |_| Ordering::Equal).unwrap().is_none());
    }

    #[test]
    fn checksum_and_truncation_errors() {
        let (f, hdr) = depth2_tree();
        // Corrupt the header checksum.
        let mut bad = f.data.clone();
        let end = hdr as usize + BTreeV2Header::encoded_size(Sizes::DEFAULT) - 1;
        bad[end] ^= 1;
        assert!(matches!(
            BTreeV2::open(&bad, hdr, Sizes::DEFAULT, RecordContext::default()),
            Err(Error::ChecksumMismatch { .. })
        ));
        // Corrupt a record inside the first leaf.
        let mut bad = f.data.clone();
        bad[16 + 8] ^= 0x40;
        let t = BTreeV2::open(&bad, hdr, Sizes::DEFAULT, RecordContext::default()).unwrap();
        assert!(matches!(
            t.records(&bad),
            Err(Error::ChecksumMismatch { .. })
        ));
        // Truncated header.
        let short = &f.data[..hdr as usize + 10];
        assert!(BTreeV2::open(short, hdr, Sizes::DEFAULT, RecordContext::default()).is_err());
        // Truncated node: drop the last leaf while keeping the header.
        let t = BTreeV2::open(&f.data, hdr, Sizes::DEFAULT, RecordContext::default()).unwrap();
        let cut = &f.data[..16 + NODE as usize];
        assert!(t.records(cut).is_err());
    }

    #[test]
    fn cycle_is_detected() {
        let mut f = FileBuilder::new();
        let root = f.reserve(NODE as usize);
        let leaf = f.place(&leaf_bytes(9, NODE, &[rec9(1)]));
        // The root's first child points back at the root itself.
        f.write_at(
            root,
            &internal_bytes(9, NODE, &[rec9(0)], &[(root, 1, 1), (leaf, 1, 1)], (1, 0)),
        );
        let hdr = f.place(&header_bytes(&HeaderSpec {
            tree_type: 9,
            node_size: NODE,
            record_size: 13,
            depth: 1,
            root,
            root_n: 1,
            total: 3,
        }));
        let t = BTreeV2::open(&f.data, hdr, Sizes::DEFAULT, RecordContext::default()).unwrap();
        assert!(t.records(&f.data).is_err());

        // Depth 2 with an internal node referencing itself.
        let mut f = FileBuilder::new();
        let leaf = f.place(&leaf_bytes(9, NODE, &[rec9(1)]));
        let root = f.reserve(NODE as usize);
        f.write_at(
            root,
            &internal_bytes(9, NODE, &[rec9(0)], &[(root, 1, 3), (leaf, 1, 1)], (1, 2)),
        );
        let hdr = f.place(&header_bytes(&HeaderSpec {
            tree_type: 9,
            node_size: NODE,
            record_size: 13,
            depth: 2,
            root,
            root_n: 1,
            total: 5,
        }));
        let t = BTreeV2::open(&f.data, hdr, Sizes::DEFAULT, RecordContext::default()).unwrap();
        let e = t.records(&f.data).unwrap_err();
        assert!(e.to_string().contains("cycle"), "{e}");
    }

    #[test]
    fn record_types_parse() {
        let s = Sizes::DEFAULT;
        let ctx = RecordContext::default();
        let mut v = Vec::new();
        push_uint(&mut v, 0x1000, 8);
        push_uint(&mut v, 77, 8);
        push_uint(&mut v, 3, 8);
        assert_eq!(
            Record::parse(1, &v, s, &ctx).unwrap(),
            Record::HugeIndirect {
                address: 0x1000,
                length: 77,
                id: 3
            }
        );
        assert_eq!(
            Record::parse(3, &v, s, &ctx).unwrap(),
            Record::HugeDirect {
                address: 0x1000,
                length: 77
            }
        );
        let mut v = Vec::new();
        push_uint(&mut v, 0x2000, 8);
        push_uint(&mut v, 10, 8);
        push_uint(&mut v, 1, 4);
        push_uint(&mut v, 20, 8);
        push_uint(&mut v, 9, 8);
        assert_eq!(
            Record::parse(2, &v, s, &ctx).unwrap(),
            Record::HugeIndirectFiltered {
                address: 0x2000,
                length: 10,
                filter_mask: 1,
                memory_size: 20,
                id: 9
            }
        );
        assert_eq!(
            Record::parse(4, &v, s, &ctx).unwrap(),
            Record::HugeDirectFiltered {
                address: 0x2000,
                length: 10,
                filter_mask: 1,
                memory_size: 20
            }
        );
        let v = [1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16];
        assert_eq!(
            Record::parse(5, &v, s, &ctx).unwrap(),
            Record::LinkName {
                hash: 0x0403_0201,
                heap_id: vec![10, 11, 12, 13, 14, 15, 16]
            }
        );
        let wide = RecordContext {
            heap_id_len: Some(3),
            ..ctx
        };
        assert_eq!(
            Record::parse(5, &v, s, &wide).unwrap(),
            Record::LinkName {
                hash: 0x0403_0201,
                heap_id: vec![10, 11, 12]
            }
        );
        let mut v = 42u64.to_le_bytes().to_vec();
        v.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(
            Record::parse(6, &v, s, &ctx).unwrap(),
            Record::LinkCreationOrder {
                creation_order: 42,
                heap_id: vec![1, 2, 3, 4, 5, 6, 7]
            }
        );
        let mut v = vec![0];
        v.extend_from_slice(&5u32.to_le_bytes());
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&[9; 8]);
        assert_eq!(
            Record::parse(7, &v, s, &ctx).unwrap(),
            Record::SharedMessageInHeap {
                hash: 5,
                reference_count: 2,
                heap_id: vec![9; 8]
            }
        );
        let mut v = vec![1];
        v.extend_from_slice(&5u32.to_le_bytes());
        v.extend_from_slice(&[0, 0x0c, 3, 0]);
        v.extend_from_slice(&0x300u64.to_le_bytes());
        assert_eq!(
            Record::parse(7, &v, s, &ctx).unwrap(),
            Record::SharedMessageInObjectHeader {
                hash: 5,
                message_type: 0x0c,
                object_header_index: 3,
                object_header_address: 0x300
            }
        );
        assert!(Record::parse(7, &[2; 20], s, &ctx).is_err());
        let mut v = vec![7; 8];
        v.push(1);
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(&0xabcdu32.to_le_bytes());
        assert_eq!(
            Record::parse(8, &v, s, &ctx).unwrap(),
            Record::AttributeName {
                heap_id: vec![7; 8],
                message_flags: 1,
                creation_order: 4,
                hash: 0xabcd
            }
        );
        assert_eq!(
            Record::parse(9, &v, s, &ctx).unwrap(),
            Record::AttributeCreationOrder {
                heap_id: vec![7; 8],
                message_flags: 1,
                creation_order: 4
            }
        );
        let chunk_ctx = RecordContext {
            chunk_ndims: 2,
            chunk_size_len: 3,
            ..ctx
        };
        let mut v = 0x5000u64.to_le_bytes().to_vec();
        v.extend_from_slice(&1u64.to_le_bytes());
        v.extend_from_slice(&2u64.to_le_bytes());
        assert_eq!(
            Record::parse(10, &v, s, &chunk_ctx).unwrap(),
            Record::Chunk {
                address: 0x5000,
                scaled_offsets: vec![1, 2]
            }
        );
        assert!(Record::parse(10, &v, s, &ctx).is_err());
        let mut v = 0x5000u64.to_le_bytes().to_vec();
        v.extend_from_slice(&[0x10, 0x20, 0x03]);
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&4u64.to_le_bytes());
        v.extend_from_slice(&5u64.to_le_bytes());
        assert_eq!(
            Record::parse(11, &v, s, &chunk_ctx).unwrap(),
            Record::FilteredChunk {
                address: 0x5000,
                chunk_size: 0x032010,
                filter_mask: 2,
                scaled_offsets: vec![4, 5]
            }
        );
        assert!(Record::parse(0, &v, s, &ctx).is_err());
        assert!(Record::parse(12, &v, s, &ctx).is_err());
        assert!(Record::parse(1, &[0; 5], s, &ctx).is_err());
    }

    /// Small xorshift PRNG for mutation tests.
    pub(crate) struct Rng(pub u64);

    impl Rng {
        pub fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    fn mutations_never_panic() {
        let (f, hdr) = depth2_tree();
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..3000 {
            let mut data = f.data.clone();
            let n = 1 + rng.next() % 4;
            for _ in 0..n {
                let i = (rng.next() % data.len() as u64) as usize;
                data[i] = rng.next() as u8;
            }
            if rng.next().is_multiple_of(8) {
                data.truncate((rng.next() % data.len() as u64) as usize);
            }
            if let Ok(t) = BTreeV2::open(&data, hdr, Sizes::DEFAULT, RecordContext::default()) {
                let _ = t.records(&data);
                let _ = t.find(&data, |r| match r {
                    Record::AttributeCreationOrder { creation_order, .. } => creation_order.cmp(&7),
                    _ => Ordering::Less,
                });
            }
        }
    }

    #[test]
    fn mutated_nodes_with_valid_checksums_never_panic() {
        // Re-seal mutated nodes so that parsing gets past the checksum.
        let (f, hdr) = depth2_tree();
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        // (node address, length of the checksummed prefix).
        let leaf_used = 6 + 2 * 13;
        let nodes = [
            (16, leaf_used),
            (16 + NODE as usize, leaf_used),
            (16 + 2 * NODE as usize, leaf_used),
            (16 + 3 * NODE as usize, leaf_used),
            (16 + 4 * NODE as usize, 6 + 13 + 2 * 9),
            (16 + 5 * NODE as usize, 6 + 13 + 2 * 9),
            (16 + 6 * NODE as usize, 6 + 13 + 2 * 11),
        ];
        for _ in 0..3000 {
            let mut data = f.data.clone();
            let (node, used) = nodes[(rng.next() % nodes.len() as u64) as usize];
            let i = node + 4 + (rng.next() % (used - 4) as u64) as usize;
            data[i] = rng.next() as u8;
            let pos = node + used;
            let c = metadata_checksum(&data[node..pos]);
            data[pos..pos + 4].copy_from_slice(&c.to_le_bytes());
            if let Ok(t) = BTreeV2::open(&data, hdr, Sizes::DEFAULT, RecordContext::default()) {
                let _ = t.records(&data);
            }
        }
    }
}
