//! Symbol Table message (type 0x0011, spec IV.A.2.r).

use crate::error::Result;
use crate::format::{ByteReader, ByteWriter, Sizes};

/// Locates an old-style group's v1 B-tree and local heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolTableMessage {
    /// Address of the v1 B-tree holding the group's symbol table nodes.
    pub btree_address: u64,
    /// Address of the local heap holding the link names.
    pub local_heap_address: u64,
}

impl SymbolTableMessage {
    /// Decode the message body.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        Ok(SymbolTableMessage {
            btree_address: r.read_offset()?,
            local_heap_address: r.read_offset()?,
        })
    }
}

impl SymbolTableMessage {
    /// Encode the message body.
    pub fn encode(&self, sizes: Sizes) -> Result<Vec<u8>> {
        let mut w = ByteWriter::with_sizes(sizes);
        w.write_offset(self.btree_address)?;
        w.write_offset(self.local_heap_address)?;
        Ok(w.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::W;

    #[test]
    fn parse() {
        let mut w = W::new();
        w.offset(136).offset(680);
        let m = SymbolTableMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(m.btree_address, 136);
        assert_eq!(m.local_heap_address, 680);
        assert!(SymbolTableMessage::parse(&w.buf[..15], Sizes::DEFAULT).is_err());
    }

    #[test]
    fn encode_round_trip() {
        let m = SymbolTableMessage {
            btree_address: 136,
            local_heap_address: 680,
        };
        let s = Sizes {
            offset: 4,
            length: 4,
        };
        assert_eq!(
            SymbolTableMessage::parse(&m.encode(s).unwrap(), s).unwrap(),
            m
        );
    }
}
