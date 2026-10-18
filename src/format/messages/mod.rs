//! Object header message decoders (spec: "Disk Format: Level 2A2 - Data
//! Object Header Messages").
//!
//! Each parser takes the raw message body (as found in
//! [`RawMessage::data`](super::object_header::RawMessage::data)) and the
//! file's [`Sizes`](super::Sizes). Bodies of version 1 object headers may
//! carry trailing alignment padding, which parsers ignore.

pub mod attribute;
pub mod attribute_info;
pub mod btree_k;
pub mod comment;
pub mod continuation;
pub mod dataspace;
pub mod datatype;
pub mod driver_info;
pub mod external_files;
pub mod fill_value;
pub mod filter_pipeline;
pub mod group_info;
pub mod layout;
pub mod link;
pub mod link_info;
pub mod modification_time;
pub mod ref_count;
pub mod shared;
pub mod symbol_table;

/// Header message type identifiers.
///
/// Version 1 object headers store the type in 2 bytes, version 2 headers
/// in 1 byte; both are widened to `u16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageType;

impl MessageType {
    /// NIL: padding / deleted message, ignored by readers.
    pub const NIL: u16 = 0x0000;
    /// Dataspace.
    pub const DATASPACE: u16 = 0x0001;
    /// Link Info (new-style groups).
    pub const LINK_INFO: u16 = 0x0002;
    /// Datatype.
    pub const DATATYPE: u16 = 0x0003;
    /// Fill Value (old).
    pub const FILL_VALUE_OLD: u16 = 0x0004;
    /// Fill Value.
    pub const FILL_VALUE: u16 = 0x0005;
    /// Link.
    pub const LINK: u16 = 0x0006;
    /// Data Storage - External Data Files.
    pub const EXTERNAL_FILES: u16 = 0x0007;
    /// Data Storage - Layout.
    pub const LAYOUT: u16 = 0x0008;
    /// Bogus (testing only; never valid in a file).
    pub const BOGUS: u16 = 0x0009;
    /// Group Info (new-style groups).
    pub const GROUP_INFO: u16 = 0x000A;
    /// Data Storage - Filter Pipeline.
    pub const FILTER_PIPELINE: u16 = 0x000B;
    /// Attribute.
    pub const ATTRIBUTE: u16 = 0x000C;
    /// Object Comment.
    pub const COMMENT: u16 = 0x000D;
    /// Object Modification Time (Old), ASCII encoded.
    pub const MODIFICATION_TIME_OLD: u16 = 0x000E;
    /// Shared Message Table (superblock extension only).
    pub const SHARED_MESSAGE_TABLE: u16 = 0x000F;
    /// Object Header Continuation.
    pub const CONTINUATION: u16 = 0x0010;
    /// Symbol Table (old-style groups).
    pub const SYMBOL_TABLE: u16 = 0x0011;
    /// Object Modification Time.
    pub const MODIFICATION_TIME: u16 = 0x0012;
    /// B-tree 'K' Values (superblock extension only).
    pub const BTREE_K: u16 = 0x0013;
    /// Driver Info (superblock extension only).
    pub const DRIVER_INFO: u16 = 0x0014;
    /// Attribute Info.
    pub const ATTRIBUTE_INFO: u16 = 0x0015;
    /// Object Reference Count.
    pub const REF_COUNT: u16 = 0x0016;
    /// File Space Info (superblock extension only).
    pub const FILE_SPACE_INFO: u16 = 0x0017;

    /// Human-readable name of a message type, or `"Unknown"`.
    pub fn name(id: u16) -> &'static str {
        match id {
            Self::NIL => "NIL",
            Self::DATASPACE => "Dataspace",
            Self::LINK_INFO => "Link Info",
            Self::DATATYPE => "Datatype",
            Self::FILL_VALUE_OLD => "Fill Value (old)",
            Self::FILL_VALUE => "Fill Value",
            Self::LINK => "Link",
            Self::EXTERNAL_FILES => "External Data Files",
            Self::LAYOUT => "Data Storage - Layout",
            Self::BOGUS => "Bogus",
            Self::GROUP_INFO => "Group Info",
            Self::FILTER_PIPELINE => "Data Storage - Filter Pipeline",
            Self::ATTRIBUTE => "Attribute",
            Self::COMMENT => "Object Comment",
            Self::MODIFICATION_TIME_OLD => "Object Modification Time (Old)",
            Self::SHARED_MESSAGE_TABLE => "Shared Message Table",
            Self::CONTINUATION => "Object Header Continuation",
            Self::SYMBOL_TABLE => "Symbol Table",
            Self::MODIFICATION_TIME => "Object Modification Time",
            Self::BTREE_K => "B-tree 'K' Values",
            Self::DRIVER_INFO => "Driver Info",
            Self::ATTRIBUTE_INFO => "Attribute Info",
            Self::REF_COUNT => "Object Reference Count",
            Self::FILE_SPACE_INFO => "File Space Info",
            _ => "Unknown",
        }
    }
}

/// Read a NUL-terminated string from `data`, or the whole slice if it has
/// no NUL. Returns the bytes before the terminator.
pub(crate) fn until_nul(data: &[u8]) -> &[u8] {
    match data.iter().position(|&b| b == 0) {
        Some(i) => &data[..i],
        None => data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(MessageType::name(MessageType::LINK), "Link");
        assert_eq!(MessageType::name(0x0017), "File Space Info");
        assert_eq!(MessageType::name(0x00ff), "Unknown");
        assert_eq!(until_nul(b"ab\0cd"), b"ab");
        assert_eq!(until_nul(b"ab"), b"ab");
    }

    /// Every parser must survive arbitrary input.
    #[test]
    fn parsers_never_panic() {
        use crate::format::Sizes;
        use crate::format::test_util::Rng;
        let mut rng = Rng(42);
        let sizes = [
            Sizes::DEFAULT,
            Sizes {
                offset: 4,
                length: 2,
            },
        ];
        for _ in 0..20000 {
            let len = rng.below(48);
            let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let s = sizes[rng.below(2)];
            let _ = btree_k::BtreeKMessage::parse(&data, s);
            let _ = comment::CommentMessage::parse(&data, s);
            let _ = continuation::ContinuationMessage::parse(&data, s);
            let _ = driver_info::DriverInfoMessage::parse(&data, s);
            let _ = group_info::GroupInfoMessage::parse(&data, s);
            let _ = link::LinkMessage::parse(&data, s);
            let _ = link_info::LinkInfoMessage::parse(&data, s);
            let _ = modification_time::ModificationTimeMessage::parse(&data, s);
            let _ = modification_time::ModificationTimeOldMessage::parse(&data, s);
            let _ = ref_count::RefCountMessage::parse(&data, s);
            let _ = shared::SharedMessage::parse(&data, s);
            let _ = symbol_table::SymbolTableMessage::parse(&data, s);
        }
    }
}
