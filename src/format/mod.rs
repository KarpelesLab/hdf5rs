//! Low-level decoding of on-disk HDF5 structures.
//!
//! Everything in this module mirrors the HDF5 File Format Specification
//! closely; the friendly public API is built on top of it.

pub mod btree_v1;
pub mod btree_v2;
pub mod chunk_index;
pub mod extensible_array;
pub mod fixed_array;
pub mod fractal_heap;
pub mod global_heap;
pub mod group_v1;
pub mod local_heap;
pub mod messages;
pub mod object_header;
mod reader;
pub mod superblock;
mod symbol_table_entry;
pub mod symbol_table_node;
#[cfg(test)]
pub(crate) mod test_util;

pub use reader::ByteReader;
pub use symbol_table_entry::{ScratchPad, SymbolTableEntry};

/// The "undefined address" value, after widening to 64 bits.
///
/// On disk it is encoded as all ones in the file's offset width;
/// [`ByteReader::read_offset`] maps that encoding to this value.
pub const UNDEFINED_ADDRESS: u64 = u64::MAX;

/// Widths of file offsets and lengths, as declared by the superblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizes {
    /// Size in bytes of file addresses ("size of offsets"): 2, 4, 8 (or 16).
    pub offset: u8,
    /// Size in bytes of object sizes ("size of lengths"): 2, 4, 8 (or 16).
    pub length: u8,
}

impl Sizes {
    /// The common 8-byte offsets / 8-byte lengths configuration.
    pub const DEFAULT: Sizes = Sizes {
        offset: 8,
        length: 8,
    };
}

impl Default for Sizes {
    fn default() -> Self {
        Sizes::DEFAULT
    }
}
