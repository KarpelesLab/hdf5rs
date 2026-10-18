//! Symbol Table Node (spec: "Disk Format: Level 1B - Group Symbol Table
//! Nodes"), the leaves of an old-style group's v1 B-tree.

use super::{ByteReader, ByteWriter, Sizes, SymbolTableEntry};
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Signature of a symbol table node.
pub const SYMBOL_TABLE_NODE_SIGNATURE: [u8; 4] = *b"SNOD";

/// Size of the fixed part of a symbol table node.
const HEADER_SIZE: usize = 8;

/// A symbol table node: the valid entries it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolTableNode {
    /// Node version (always 1).
    pub version: u8,
    /// The valid entries ("Number of Symbols" of them), in stored order.
    pub entries: Vec<SymbolTableEntry>,
}

impl SymbolTableNode {
    /// Parse a node from `data`, which starts at the signature.
    ///
    /// Only the first "Number of Symbols" entries are decoded; the unused
    /// tail of the node (whose length depends on the superblock's K) need
    /// not be present.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        r.expect_signature(&SYMBOL_TABLE_NODE_SIGNATURE)?;
        let version = r.read_u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!(
                "symbol table node version {version}"
            )));
        }
        let _reserved = r.read_u8()?;
        let count = r.read_u16()? as usize;
        let need = count
            .checked_mul(SymbolTableEntry::encoded_size(sizes.offset))
            .ok_or(Error::UnexpectedEof)?;
        if need > r.remaining() {
            return Err(Error::UnexpectedEof);
        }
        let entries = (0..count)
            .map(|_| SymbolTableEntry::parse(&mut r))
            .collect::<Result<Vec<_>>>()?;
        Ok(SymbolTableNode { version, entries })
    }

    /// Read and parse the node at `addr`.
    pub fn read(store: &(impl ReadAt + ?Sized), addr: u64, sizes: Sizes) -> Result<Self> {
        let header = store.read_vec_at(addr, HEADER_SIZE)?;
        let count = u16::from_le_bytes([header[6], header[7]]) as usize;
        let len = count
            .checked_mul(SymbolTableEntry::encoded_size(sizes.offset))
            .and_then(|n| n.checked_add(HEADER_SIZE))
            .ok_or(Error::UnexpectedEof)?;
        Self::parse(&store.read_vec_at(addr, len)?, sizes)
    }
}

impl SymbolTableNode {
    /// Size of a node holding up to `2 * leaf_k` entries.
    pub fn node_size(sizes: Sizes, leaf_k: u16) -> usize {
        HEADER_SIZE + 2 * usize::from(leaf_k) * SymbolTableEntry::encoded_size(sizes.offset)
    }

    /// Encode the node at its full size for group leaf node K `leaf_k`
    /// (unused entry slots are zero).
    pub fn encode(&self, sizes: Sizes, leaf_k: u16) -> Result<Vec<u8>> {
        if self.entries.len() > 2 * usize::from(leaf_k) {
            return Err(Error::invalid("too many entries for a symbol table node"));
        }
        let mut w = ByteWriter::with_sizes(sizes);
        w.bytes(&SYMBOL_TABLE_NODE_SIGNATURE)
            .u8(1)
            .u8(0)
            .u16(self.entries.len() as u16);
        for e in &self.entries {
            e.encode(&mut w)?;
        }
        w.zeros(Self::node_size(sizes, leaf_k) - w.len());
        Ok(w.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::ScratchPad;
    use crate::format::test_util::{FileBuilder, Rng};

    #[test]
    fn reads_node() {
        for sizes in [
            Sizes::DEFAULT,
            Sizes {
                offset: 4,
                length: 8,
            },
        ] {
            let mut f = FileBuilder::new(sizes);
            let e1 = f.entry(8, 1000);
            let e2 = f.entry(16, 2000);
            f.snod(32, &[e1, e2]);
            let n = SymbolTableNode::read(&f.data, 32, sizes).unwrap();
            assert_eq!(n.version, 1);
            assert_eq!(n.entries.len(), 2);
            assert_eq!(n.entries[0].link_name_offset, 8);
            assert_eq!(n.entries[1].object_header_address, 2000);
            assert_eq!(n.entries[1].scratch, ScratchPad::None);
        }
    }

    #[test]
    fn rejects_bad() {
        let mut f = FileBuilder::new(Sizes::DEFAULT);
        let e = f.entry(8, 1000);
        f.snod(0, &[e]);
        let mut bad = f.data.clone();
        bad[4] = 2;
        assert!(SymbolTableNode::read(&bad, 0, Sizes::DEFAULT).is_err());
        let mut bad = f.data.clone();
        bad[0] = b'X';
        assert!(SymbolTableNode::read(&bad, 0, Sizes::DEFAULT).is_err());
        let mut bad = f.data.clone();
        bad[6] = 2;
        assert!(SymbolTableNode::read(&bad, 0, Sizes::DEFAULT).is_err());
        assert!(SymbolTableNode::read(&f.data[..20], 0, Sizes::DEFAULT).is_err());
        assert!(SymbolTableNode::read(&f.data, u64::MAX, Sizes::DEFAULT).is_err());
    }

    #[test]
    fn mutations_never_panic() {
        let mut f = FileBuilder::new(Sizes::DEFAULT);
        let entries: Vec<_> = (0..3).map(|i| f.entry(8 * i, 100 * i)).collect();
        f.snod(0, &entries);
        let mut rng = Rng(42);
        for _ in 0..5000 {
            let d = rng.mutate(&f.data);
            let _ = SymbolTableNode::read(&d, 0, Sizes::DEFAULT);
        }
    }

    #[test]
    fn encode_round_trip() {
        let sizes = Sizes::DEFAULT;
        let node = SymbolTableNode {
            version: 1,
            entries: vec![
                SymbolTableEntry {
                    link_name_offset: 8,
                    object_header_address: 800,
                    scratch: ScratchPad::None,
                },
                SymbolTableEntry {
                    link_name_offset: 16,
                    object_header_address: crate::format::UNDEFINED_ADDRESS,
                    scratch: ScratchPad::SoftLink {
                        link_value_offset: 24,
                    },
                },
            ],
        };
        let enc = node.encode(sizes, 4).unwrap();
        assert_eq!(enc.len(), 8 + 8 * 40);
        assert_eq!(SymbolTableNode::parse(&enc, sizes).unwrap(), node);
        assert!(node.encode(sizes, 0).is_err());
    }
}
