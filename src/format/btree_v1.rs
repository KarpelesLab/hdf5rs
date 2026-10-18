//! Version 1 B-trees (spec: "Disk Format: Level 1A1 - Version 1 B-trees").
//!
//! Node type 0 indexes the symbol table nodes of an old-style group; node
//! type 1 indexes the raw data chunks of a chunked dataset (layout message
//! versions 1–3).

use std::cmp::Ordering;
use std::collections::HashSet;

use super::{ByteReader, ByteWriter, Sizes, UNDEFINED_ADDRESS};
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Signature of a v1 B-tree node.
pub const BTREE_V1_SIGNATURE: [u8; 4] = *b"TREE";

/// Maximum dataset rank accepted for chunk B-trees (the spec caps
/// dataspace rank at 32).
pub const MAX_CHUNK_RANK: usize = 32;

/// Maximum number of distinct nodes visited by one traversal.
pub const MAX_NODES: usize = 1 << 20;

/// Maximum number of leaf entries returned by one traversal.
pub const MAX_ENTRIES: usize = 1 << 24;

/// The kind of objects a v1 B-tree indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// Type 0: group nodes (children are symbol table nodes).
    Group,
    /// Type 1: raw data chunks.
    Chunk,
}

/// A key of a chunk (type 1) B-tree node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkKey {
    /// Size of the (possibly filtered) chunk in bytes.
    pub size: u32,
    /// Bit `i` set means filter `i` of the pipeline was skipped.
    pub filter_mask: u32,
    /// Chunk offset in each dataset dimension (in elements).
    ///
    /// The trailing datatype-offset field stored on disk (which should be
    /// zero) is not included.
    pub offsets: Vec<u64>,
}

/// The keys of a node; there is one more key than children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Keys {
    /// Group keys: byte offsets into the group's local heap.
    Group(Vec<u64>),
    /// Chunk keys.
    Chunk(Vec<ChunkKey>),
}

impl Keys {
    /// Number of keys.
    pub fn len(&self) -> usize {
        match self {
            Keys::Group(k) => k.len(),
            Keys::Chunk(k) => k.len(),
        }
    }

    /// Whether there are no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A parsed v1 B-tree node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BTreeNode {
    /// What the tree indexes.
    pub node_type: NodeType,
    /// Level of the node; leaves are level 0.
    pub level: u8,
    /// Address of the left sibling, or [`UNDEFINED_ADDRESS`].
    pub left_sibling: u64,
    /// Address of the right sibling, or [`UNDEFINED_ADDRESS`].
    pub right_sibling: u64,
    /// The `children.len() + 1` valid keys.
    pub keys: Keys,
    /// Child addresses: sub-tree nodes when `level > 0`, otherwise symbol
    /// table nodes (type 0) or raw data chunks (type 1).
    pub children: Vec<u64>,
}

/// One raw data chunk found in a chunk B-tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkEntry {
    /// Chunk offset in each dataset dimension (in elements).
    pub offsets: Vec<u64>,
    /// Stored chunk size in bytes.
    pub size: u32,
    /// Filter mask (bit `i` set = filter `i` skipped).
    pub filter_mask: u32,
    /// File address of the chunk data.
    pub address: u64,
}

fn header_size(sizes: Sizes) -> usize {
    8 + 2 * sizes.offset as usize
}

fn key_size(node_type: NodeType, sizes: Sizes, ndims: usize) -> usize {
    match node_type {
        NodeType::Group => sizes.length as usize,
        NodeType::Chunk => 8 + 8 * (ndims + 1),
    }
}

fn check_rank(ndims: usize) -> Result<()> {
    if ndims > MAX_CHUNK_RANK {
        return Err(Error::LimitExceeded("chunk B-tree rank"));
    }
    Ok(())
}

impl BTreeNode {
    /// Parse a node from `data`, which starts at the signature.
    ///
    /// `ndims` is the dataset rank (number of dimensions, *excluding* the
    /// extra datatype dimension); it is only used for chunk nodes, whose
    /// keys hold `ndims + 1` 64-bit offsets.
    pub fn parse(data: &[u8], sizes: Sizes, ndims: usize) -> Result<Self> {
        check_rank(ndims)?;
        let mut r = ByteReader::with_sizes(data, sizes);
        r.expect_signature(&BTREE_V1_SIGNATURE)?;
        let node_type = match r.read_u8()? {
            0 => NodeType::Group,
            1 => NodeType::Chunk,
            t => return Err(Error::invalid(format!("unknown v1 B-tree node type {t}"))),
        };
        let level = r.read_u8()?;
        let used = r.read_u16()? as usize;
        let left_sibling = r.read_offset()?;
        let right_sibling = r.read_offset()?;
        let need = used
            .checked_add(1)
            .and_then(|k| k.checked_mul(key_size(node_type, sizes, ndims)))
            .and_then(|n| n.checked_add(used.checked_mul(sizes.offset as usize)?))
            .ok_or(Error::UnexpectedEof)?;
        if need > r.remaining() {
            return Err(Error::UnexpectedEof);
        }
        let mut children = Vec::with_capacity(used);
        let keys = match node_type {
            NodeType::Group => {
                let mut keys = Vec::with_capacity(used + 1);
                for _ in 0..used {
                    keys.push(r.read_length()?);
                    children.push(r.read_offset()?);
                }
                keys.push(r.read_length()?);
                Keys::Group(keys)
            }
            NodeType::Chunk => {
                let mut keys = Vec::with_capacity(used + 1);
                for _ in 0..used {
                    keys.push(parse_chunk_key(&mut r, ndims)?);
                    children.push(r.read_offset()?);
                }
                keys.push(parse_chunk_key(&mut r, ndims)?);
                Keys::Chunk(keys)
            }
        };
        Ok(BTreeNode {
            node_type,
            level,
            left_sibling,
            right_sibling,
            keys,
            children,
        })
    }

    /// Read and parse the node at `addr` (see [`parse`](Self::parse) for
    /// `ndims`).
    pub fn read(
        store: &(impl ReadAt + ?Sized),
        addr: u64,
        sizes: Sizes,
        ndims: usize,
    ) -> Result<Self> {
        check_rank(ndims)?;
        if addr == UNDEFINED_ADDRESS {
            return Err(Error::invalid("v1 B-tree node address undefined"));
        }
        let head = store.read_vec_at(addr, header_size(sizes))?;
        let node_type = match head[4] {
            0 => NodeType::Group,
            1 => NodeType::Chunk,
            t => return Err(Error::invalid(format!("unknown v1 B-tree node type {t}"))),
        };
        let used = u16::from_le_bytes([head[6], head[7]]) as usize;
        // used <= 65535 and key size is small, so this cannot overflow.
        let len = header_size(sizes)
            + (used + 1) * key_size(node_type, sizes, ndims)
            + used * sizes.offset as usize;
        Self::parse(&store.read_vec_at(addr, len)?, sizes, ndims)
    }
}

fn parse_chunk_key(r: &mut ByteReader<'_>, ndims: usize) -> Result<ChunkKey> {
    let size = r.read_u32()?;
    let filter_mask = r.read_u32()?;
    let offsets = (0..ndims)
        .map(|_| r.read_u64())
        .collect::<Result<Vec<_>>>()?;
    let _datatype_offset = r.read_u64()?;
    Ok(ChunkKey {
        size,
        filter_mask,
        offsets,
    })
}

/// Shared traversal state: cycle detection and bounds.
struct Walk<'s, S: ReadAt + ?Sized> {
    store: &'s S,
    sizes: Sizes,
    ndims: usize,
    node_type: NodeType,
    visited: HashSet<u64>,
}

impl<S: ReadAt + ?Sized> Walk<'_, S> {
    /// Load a node, checking type, level and that it was not seen before.
    fn load(&mut self, addr: u64, expected_level: Option<u8>) -> Result<BTreeNode> {
        if self.visited.len() >= MAX_NODES {
            return Err(Error::LimitExceeded("v1 B-tree node count"));
        }
        if !self.visited.insert(addr) {
            return Err(Error::invalid("cycle in v1 B-tree"));
        }
        let node = BTreeNode::read(self.store, addr, self.sizes, self.ndims)?;
        if node.node_type != self.node_type {
            return Err(Error::invalid("unexpected v1 B-tree node type"));
        }
        if expected_level.is_some_and(|l| l != node.level) {
            return Err(Error::invalid("inconsistent v1 B-tree node level"));
        }
        Ok(node)
    }

    /// Depth-first, left-to-right visit of all leaf entries. Levels must
    /// decrease by one per step, which bounds the depth to 256.
    fn visit(
        &mut self,
        addr: u64,
        expected_level: Option<u8>,
        on_leaf: &mut dyn FnMut(&BTreeNode, usize) -> Result<()>,
    ) -> Result<()> {
        let node = self.load(addr, expected_level)?;
        if node.level == 0 {
            for i in 0..node.children.len() {
                on_leaf(&node, i)?;
            }
        } else {
            for &child in &node.children {
                self.visit(child, Some(node.level - 1), on_leaf)?;
            }
        }
        Ok(())
    }
}

/// Collect the addresses of all symbol table nodes of a group B-tree, in
/// stored (left-to-right) order.
pub fn group_btree_children(
    store: &(impl ReadAt + ?Sized),
    root_addr: u64,
    sizes: Sizes,
) -> Result<Vec<u64>> {
    let mut walk = Walk {
        store,
        sizes,
        ndims: 0,
        node_type: NodeType::Group,
        visited: HashSet::new(),
    };
    let mut out = Vec::new();
    walk.visit(root_addr, None, &mut |node, i| {
        if out.len() >= MAX_ENTRIES {
            return Err(Error::LimitExceeded("v1 B-tree entry count"));
        }
        let addr = node.children.get(i).copied().ok_or(Error::UnexpectedEof)?;
        out.push(addr);
        Ok(())
    })?;
    Ok(out)
}

/// Collect all chunks of a chunk B-tree, in stored (left-to-right) order.
///
/// `ndims` is the dataset rank (the layout message's dimensionality minus
/// one).
pub fn chunk_btree_entries(
    store: &(impl ReadAt + ?Sized),
    root_addr: u64,
    sizes: Sizes,
    ndims: usize,
) -> Result<Vec<ChunkEntry>> {
    check_rank(ndims)?;
    let mut walk = Walk {
        store,
        sizes,
        ndims,
        node_type: NodeType::Chunk,
        visited: HashSet::new(),
    };
    let mut out = Vec::new();
    walk.visit(root_addr, None, &mut |node, i| {
        if out.len() >= MAX_ENTRIES {
            return Err(Error::LimitExceeded("v1 B-tree entry count"));
        }
        out.push(chunk_entry(node, i)?);
        Ok(())
    })?;
    Ok(out)
}

fn chunk_entry(node: &BTreeNode, i: usize) -> Result<ChunkEntry> {
    let Keys::Chunk(keys) = &node.keys else {
        return Err(Error::invalid("unexpected v1 B-tree node type"));
    };
    let key = keys.get(i).ok_or(Error::UnexpectedEof)?;
    let address = node.children.get(i).copied().ok_or(Error::UnexpectedEof)?;
    Ok(ChunkEntry {
        offsets: key.offsets.clone(),
        size: key.size,
        filter_mask: key.filter_mask,
        address,
    })
}

/// Find the chunk containing the element at `coords` by descending a chunk
/// B-tree using its keys.
///
/// `chunk_dims` gives the chunk size in elements for each dataset
/// dimension (its length is the dataset rank). Returns `Ok(None)` when no
/// chunk covering `coords` is allocated.
pub fn find_chunk(
    store: &(impl ReadAt + ?Sized),
    root_addr: u64,
    sizes: Sizes,
    chunk_dims: &[u64],
    coords: &[u64],
) -> Result<Option<ChunkEntry>> {
    let ndims = chunk_dims.len();
    check_rank(ndims)?;
    if coords.len() != ndims {
        return Err(Error::invalid("chunk coordinate rank mismatch"));
    }
    // Align the coordinates to the chunk grid: the chunk we look for has
    // exactly these offsets.
    let target = coords
        .iter()
        .zip(chunk_dims)
        .map(|(&c, &d)| {
            if d == 0 {
                Err(Error::invalid("zero chunk dimension"))
            } else {
                Ok(c - c % d)
            }
        })
        .collect::<Result<Vec<u64>>>()?;
    let mut walk = Walk {
        store,
        sizes,
        ndims,
        node_type: NodeType::Chunk,
        visited: HashSet::new(),
    };
    let mut addr = root_addr;
    let mut expected_level = None;
    loop {
        let node = walk.load(addr, expected_level)?;
        let Keys::Chunk(keys) = &node.keys else {
            return Err(Error::invalid("unexpected v1 B-tree node type"));
        };
        // In chunk trees key[i] is the least chunk in child[i]: pick the
        // last child whose left key is <= target.
        let n = node.children.len();
        let mut pick = None;
        for (i, key) in keys.iter().take(n).enumerate() {
            if key.offsets.as_slice().cmp(target.as_slice()) == Ordering::Greater {
                break;
            }
            pick = Some(i);
        }
        let Some(i) = pick else {
            return Ok(None);
        };
        if node.level == 0 {
            let entry = chunk_entry(&node, i)?;
            return Ok((entry.offsets == target).then_some(entry));
        }
        addr = node.children.get(i).copied().ok_or(Error::UnexpectedEof)?;
        expected_level = Some(node.level - 1);
    }
}

impl BTreeNode {
    /// Size of a group (type 0) node with room for `2 * k` children.
    pub fn group_node_size(sizes: Sizes, k: u16) -> usize {
        let k2 = 2 * usize::from(k);
        header_size(sizes) + (k2 + 1) * sizes.length as usize + k2 * sizes.offset as usize
    }

    /// Encode a group (type 0) node at its full size for group internal
    /// node K `k` (unused key/child slots are zero).
    pub fn encode_group(&self, sizes: Sizes, k: u16) -> Result<Vec<u8>> {
        let Keys::Group(keys) = &self.keys else {
            return Err(Error::unsupported("encoding chunk B-tree nodes"));
        };
        if self.node_type != NodeType::Group
            || keys.len() != self.children.len() + 1
            || self.children.len() > 2 * usize::from(k)
        {
            return Err(Error::invalid("malformed group B-tree node"));
        }
        let mut w = ByteWriter::with_sizes(sizes);
        w.bytes(&BTREE_V1_SIGNATURE)
            .u8(0)
            .u8(self.level)
            .u16(self.children.len() as u16);
        w.write_offset(self.left_sibling)?;
        w.write_offset(self.right_sibling)?;
        for (key, child) in keys.iter().zip(&self.children) {
            w.write_length(*key)?;
            w.write_offset(*child)?;
        }
        if let Some(last) = keys.last() {
            w.write_length(*last)?;
        }
        w.zeros(Self::group_node_size(sizes, k) - w.len());
        Ok(w.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::{FileBuilder, Rng};

    const S4: Sizes = Sizes {
        offset: 4,
        length: 4,
    };

    /// Two-level group tree: root at 0 with two leaves; leaves point to
    /// fake SNOD addresses.
    fn group_tree(sizes: Sizes) -> FileBuilder {
        let mut f = FileBuilder::new(sizes);
        let k = |f: &FileBuilder, v| f.group_key(v);
        let keys = [k(&f, 0), k(&f, 10), k(&f, 20)];
        f.btree(0, 0, 1, &keys, &[200, 400]);
        let keys = [k(&f, 0), k(&f, 5), k(&f, 10)];
        f.btree(200, 0, 0, &keys, &[1000, 1100]);
        let keys = [k(&f, 10), k(&f, 15), k(&f, 18), k(&f, 20)];
        f.btree(400, 0, 0, &keys, &[1200, 1300, 1400]);
        f
    }

    #[test]
    fn group_traversal() {
        for sizes in [Sizes::DEFAULT, S4] {
            let f = group_tree(sizes);
            let snods = group_btree_children(&f.data, 0, sizes).unwrap();
            assert_eq!(snods, vec![1000, 1100, 1200, 1300, 1400]);
            let root = BTreeNode::read(&f.data, 0, sizes, 0).unwrap();
            assert_eq!(root.node_type, NodeType::Group);
            assert_eq!(root.level, 1);
            assert_eq!(root.left_sibling, UNDEFINED_ADDRESS);
            assert_eq!(root.keys, Keys::Group(vec![0, 10, 20]));
            // Wrong tree kind is rejected.
            assert!(chunk_btree_entries(&f.data, 0, sizes, 2).is_err());
        }
    }

    #[test]
    fn empty_root() {
        let mut f = FileBuilder::new(Sizes::DEFAULT);
        let key = f.group_key(0);
        f.btree(0, 0, 0, &[key], &[]);
        assert!(
            group_btree_children(&f.data, 0, Sizes::DEFAULT)
                .unwrap()
                .is_empty()
        );
    }

    /// Three-level 2-D chunk tree over a 4x4 grid of 10x10 chunks, with
    /// chunk (1, 2) missing.
    fn chunk_tree() -> (FileBuilder, Vec<Vec<u64>>) {
        let mut f = FileBuilder::new(Sizes::DEFAULT);
        let mut chunks = Vec::new();
        for r in 0..4u64 {
            for c in 0..4u64 {
                if (r, c) != (1, 2) {
                    chunks.push(vec![r * 10, c * 10]);
                }
            }
        }
        let addr_of = |o: &[u64]| 100_000 + o[0] * 100 + o[1];
        let sentinel = FileBuilder::chunk_key(0, 0, &[40, 0]);
        let key = |o: &[u64]| FileBuilder::chunk_key(64, 0b10, o);
        // Leaves: groups of 3 chunks.
        let mut leaves = Vec::new();
        for (n, group) in chunks.chunks(3).enumerate() {
            let a = 200 + n as u64 * 200;
            let mut keys: Vec<_> = group.iter().map(|o| key(o)).collect();
            keys.push(sentinel.clone());
            let kids: Vec<_> = group.iter().map(|o| addr_of(o)).collect();
            f.btree(a, 1, 0, &keys, &kids);
            leaves.push((a, group[0].clone()));
        }
        // Level 1: groups of 2 leaves.
        let mut mids = Vec::new();
        for (n, group) in leaves.chunks(2).enumerate() {
            let a = 1400 + n as u64 * 200;
            let mut keys: Vec<_> = group.iter().map(|(_, o)| key(o)).collect();
            keys.push(sentinel.clone());
            let kids: Vec<_> = group.iter().map(|(a, _)| *a).collect();
            f.btree(a, 1, 1, &keys, &kids);
            mids.push((a, group[0].1.clone()));
        }
        let mut keys: Vec<_> = mids.iter().map(|(_, o)| key(o)).collect();
        keys.push(sentinel);
        let kids: Vec<_> = mids.iter().map(|(a, _)| *a).collect();
        f.btree(0, 1, 2, &keys, &kids);
        (f, chunks)
    }

    #[test]
    fn chunk_traversal_and_lookup() {
        let (f, chunks) = chunk_tree();
        let entries = chunk_btree_entries(&f.data, 0, Sizes::DEFAULT, 2).unwrap();
        assert_eq!(entries.len(), 15);
        for (e, o) in entries.iter().zip(&chunks) {
            assert_eq!(&e.offsets, o);
            assert_eq!(e.size, 64);
            assert_eq!(e.filter_mask, 0b10);
            assert_eq!(e.address, 100_000 + o[0] * 100 + o[1]);
        }
        let dims = [10, 10];
        for r in 0..45u64 {
            for c in 0..45u64 {
                let got = find_chunk(&f.data, 0, Sizes::DEFAULT, &dims, &[r, c]).unwrap();
                let (cr, cc) = (r / 10, c / 10);
                if cr >= 4 || cc >= 4 || (cr, cc) == (1, 2) {
                    assert!(got.is_none(), "({r},{c})");
                } else {
                    let e = got.unwrap();
                    assert_eq!(e.offsets, vec![cr * 10, cc * 10]);
                }
            }
        }
        assert!(find_chunk(&f.data, 0, Sizes::DEFAULT, &dims, &[1]).is_err());
        assert!(find_chunk(&f.data, 0, Sizes::DEFAULT, &[0, 1], &[1, 1]).is_err());
        // Wrong rank changes the key size and breaks the structure.
        assert!(chunk_btree_entries(&f.data, 0, Sizes::DEFAULT, 1).is_err());
        assert!(chunk_btree_entries(&f.data, 0, Sizes::DEFAULT, 99).is_err());
    }

    #[test]
    fn cycle_is_error() {
        let mut f = FileBuilder::new(Sizes::DEFAULT);
        // Root (level 1) points to itself: level check catches it.
        let keys = [f.group_key(0), f.group_key(1)];
        f.btree(0, 0, 1, &keys, &[0]);
        assert!(group_btree_children(&f.data, 0, Sizes::DEFAULT).is_err());
        // Two level-1 nodes pointing to the same leaf twice: visited check.
        let mut f = FileBuilder::new(Sizes::DEFAULT);
        let k3 = [f.group_key(0), f.group_key(1), f.group_key(2)];
        f.btree(0, 0, 1, &k3, &[200, 200]);
        let k2 = [f.group_key(0), f.group_key(1)];
        f.btree(200, 0, 0, &k2, &[999]);
        assert!(matches!(
            group_btree_children(&f.data, 0, Sizes::DEFAULT),
            Err(Error::InvalidFormat(_))
        ));
        // Chunk lookup cycle.
        let mut f = FileBuilder::new(Sizes::DEFAULT);
        let keys = [
            FileBuilder::chunk_key(1, 0, &[0]),
            FileBuilder::chunk_key(0, 0, &[9]),
        ];
        f.btree(0, 1, 1, &keys, &[0]);
        assert!(find_chunk(&f.data, 0, Sizes::DEFAULT, &[1], &[0]).is_err());
        assert!(chunk_btree_entries(&f.data, 0, Sizes::DEFAULT, 1).is_err());
    }

    #[test]
    fn truncated_and_garbage() {
        let f = group_tree(Sizes::DEFAULT);
        for len in 0..f.data.len() {
            let _ = group_btree_children(&f.data[..len], 0, Sizes::DEFAULT);
        }
        assert!(group_btree_children(&f.data[..300], 0, Sizes::DEFAULT).is_err());
        let mut bad = f.data.clone();
        bad[4] = 7;
        assert!(group_btree_children(&bad, 0, Sizes::DEFAULT).is_err());
        assert!(group_btree_children(&f.data, 3, Sizes::DEFAULT).is_err());
        assert!(group_btree_children(&f.data, UNDEFINED_ADDRESS, Sizes::DEFAULT).is_err());
        assert!(group_btree_children(&vec![0xffu8; 4096], 0, Sizes::DEFAULT).is_err());
    }

    #[test]
    fn mutations_never_panic() {
        let g = group_tree(Sizes::DEFAULT);
        let (c, _) = chunk_tree();
        let mut rng = Rng(0xdead_beef);
        for _ in 0..3000 {
            let d = rng.mutate(&g.data);
            let _ = group_btree_children(&d, 0, Sizes::DEFAULT);
            let d = rng.mutate(&c.data);
            let _ = chunk_btree_entries(&d, 0, Sizes::DEFAULT, 2);
            let r = rng.below(50) as u64;
            let _ = find_chunk(&d, 0, Sizes::DEFAULT, &[10, 10], &[r, 49 - r]);
        }
    }

    #[test]
    fn encode_group_round_trip() {
        let sizes = Sizes::DEFAULT;
        let node = BTreeNode {
            node_type: NodeType::Group,
            level: 0,
            left_sibling: UNDEFINED_ADDRESS,
            right_sibling: 4000,
            keys: Keys::Group(vec![0, 8, 24]),
            children: vec![1000, 2000],
        };
        let enc = node.encode_group(sizes, 16).unwrap();
        assert_eq!(enc.len(), 544);
        assert_eq!(BTreeNode::parse(&enc, sizes, 0).unwrap(), node);
        assert!(node.encode_group(sizes, 0).is_err());
    }
}
