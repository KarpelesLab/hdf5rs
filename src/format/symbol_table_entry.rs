use super::ByteReader;
use crate::error::{Error, Result};

/// Cached information held in a symbol table entry's scratch-pad space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScratchPad {
    /// Cache type 0: nothing cached.
    None,
    /// Cache type 1: the object is a group; addresses of its v1 B-tree and
    /// local heap.
    Group {
        /// Address of the group's v1 B-tree.
        btree_address: u64,
        /// Address of the group's local heap.
        heap_address: u64,
    },
    /// Cache type 2: the entry is a soft link; offset of the link value in
    /// the local heap.
    SoftLink {
        /// Offset of the link target string in the local heap.
        link_value_offset: u32,
    },
}

/// A symbol table entry (spec: "Symbol Table Entry"), used in symbol table
/// nodes and in version 0/1 superblocks for the root group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolTableEntry {
    /// Offset of the link name in the group's local heap.
    pub link_name_offset: u64,
    /// Address of the object's header.
    pub object_header_address: u64,
    /// Cached scratch-pad information.
    pub scratch: ScratchPad,
}

impl SymbolTableEntry {
    /// Encoded size in bytes for the given offset width.
    pub fn encoded_size(offset_size: u8) -> usize {
        2 * offset_size as usize + 4 + 4 + 16
    }

    /// Parse one entry from `r`.
    pub fn parse(r: &mut ByteReader<'_>) -> Result<Self> {
        let link_name_offset = r.read_offset()?;
        let object_header_address = r.read_offset()?;
        let cache_type = r.read_u32()?;
        let _reserved = r.read_u32()?;
        let pad = r.read_bytes(16)?;
        let mut p = ByteReader::with_sizes(pad, r.sizes());
        let scratch = match cache_type {
            0 => ScratchPad::None,
            1 => ScratchPad::Group {
                btree_address: p.read_offset()?,
                heap_address: p.read_offset()?,
            },
            2 => ScratchPad::SoftLink {
                link_value_offset: p.read_u32()?,
            },
            other => {
                return Err(Error::invalid(format!(
                    "unknown symbol table entry cache type {other}"
                )));
            }
        };
        Ok(SymbolTableEntry {
            link_name_offset,
            object_header_address,
            scratch,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_group_entry() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&96u64.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&136u64.to_le_bytes());
        data.extend_from_slice(&680u64.to_le_bytes());
        assert_eq!(data.len(), SymbolTableEntry::encoded_size(8));
        let e = SymbolTableEntry::parse(&mut ByteReader::new(&data)).unwrap();
        assert_eq!(e.object_header_address, 96);
        assert_eq!(
            e.scratch,
            ScratchPad::Group {
                btree_address: 136,
                heap_address: 680
            }
        );
    }
}
