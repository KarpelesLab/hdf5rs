//! Shared object header messages (SOHM): the Shared Message Table message
//! (type 0x000F, found in the superblock extension), the master table
//! (`SMTB`) and list indexes (`SMLI`).
//!
//! Spec: "III.I. Shared Object Header Message Table" and "IV.A.2.p. The
//! Shared Message Table Message". B-tree indexes use v2 B-tree record
//! type 7 (see [`super::btree_v2`]).
//!
//! Readers rarely need the indexes themselves: a shared message reference
//! (version 3, type 1) carries the fractal heap ID of the message, and the
//! heap is found through the index tracking that message type.

use super::messages::MessageType;
use super::{ByteReader, Sizes};
use crate::checksum::metadata_checksum;
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Maximum number of indexes in a master table (the spec allows at most 8).
pub const MAX_INDEXES: usize = 8;

/// Message type flag bits used by [`SohmIndex::message_types`].
///
/// The specification describes bit 0 as dataspace, bit 1 as datatype and
/// so on, but files written by the reference library (observed as a black
/// box) set bit `n` for header message type `n`: an index holding
/// dataspaces and datatypes stores `0x000A`, one holding fill values,
/// filter pipelines and attributes stores `0x1820`. We follow the files.
pub mod flags {
    /// Dataspace messages (type 0x0001).
    pub const DATASPACE: u16 = 1 << 0x0001;
    /// Datatype messages (type 0x0003).
    pub const DATATYPE: u16 = 1 << 0x0003;
    /// Fill value messages (type 0x0005).
    pub const FILL_VALUE: u16 = 1 << 0x0005;
    /// Filter pipeline messages (type 0x000B).
    pub const FILTER_PIPELINE: u16 = 1 << 0x000B;
    /// Attribute messages (type 0x000C).
    pub const ATTRIBUTE: u16 = 1 << 0x000C;
}

/// The flag bit for an object header message type, if it is shareable.
pub fn message_type_flag(msg_type: u16) -> Option<u16> {
    Some(match msg_type {
        MessageType::DATASPACE => flags::DATASPACE,
        MessageType::DATATYPE => flags::DATATYPE,
        MessageType::FILL_VALUE => flags::FILL_VALUE,
        MessageType::FILTER_PIPELINE => flags::FILTER_PIPELINE,
        MessageType::ATTRIBUTE => flags::ATTRIBUTE,
        _ => return None,
    })
}

/// Shared Message Table message (type 0x000F).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMessageTableMessage {
    /// Message version (0).
    pub version: u8,
    /// Address of the master table (`SMTB`).
    pub table_address: u64,
    /// Number of indexes in the master table.
    pub num_indexes: u8,
}

impl SharedMessageTableMessage {
    /// Decode the message body.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "shared message table message version {version}"
            )));
        }
        Ok(SharedMessageTableMessage {
            version,
            table_address: r.read_offset()?,
            num_indexes: r.read_u8()?,
        })
    }
}

/// Storage form of one SOHM index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SohmIndexType {
    /// Unsorted list (`SMLI`).
    List,
    /// Version 2 B-tree (record type 7).
    BTree,
}

/// One index of the master table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SohmIndex {
    /// Index version (0).
    pub version: u8,
    /// List or B-tree.
    pub index_type: SohmIndexType,
    /// Message types tracked by this index (see [`flags`]).
    pub message_types: u16,
    /// Messages smaller than this are not shared.
    pub min_message_size: u32,
    /// Count above which a list index converts to a B-tree.
    pub list_cutoff: u16,
    /// Count below which a B-tree index converts to a list.
    pub btree_cutoff: u16,
    /// Number of messages in the index.
    pub num_messages: u16,
    /// Address of the list or B-tree.
    pub index_address: u64,
    /// Address of the fractal heap holding the shared messages.
    pub heap_address: u64,
}

impl SohmIndex {
    /// Whether this index tracks messages of `msg_type`.
    pub fn tracks(&self, msg_type: u16) -> bool {
        message_type_flag(msg_type).is_some_and(|f| self.message_types & f != 0)
    }
}

/// The master table (`SMTB`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmTable {
    /// The indexes, in stored order.
    pub indexes: Vec<SohmIndex>,
}

impl SohmTable {
    /// Encoded size of a table with `n` indexes.
    pub fn encoded_size(n: usize, sizes: Sizes) -> usize {
        // signature, per index (version 1, type 1, flags 2, min size 4,
        // cutoffs 2 + 2, count 2, two addresses), checksum
        4 + n * (14 + 2 * sizes.offset as usize) + 4
    }

    /// Read the table at `addr` holding `num_indexes` indexes.
    pub fn read<R: ReadAt + ?Sized>(
        store: &R,
        addr: u64,
        num_indexes: usize,
        sizes: Sizes,
    ) -> Result<Self> {
        if num_indexes > MAX_INDEXES {
            return Err(Error::LimitExceeded("SOHM index count"));
        }
        let buf = store.read_vec_at(addr, Self::encoded_size(num_indexes, sizes))?;
        Self::parse(&buf, num_indexes, sizes)
    }

    /// Parse a table with `num_indexes` indexes from `data`.
    pub fn parse(data: &[u8], num_indexes: usize, sizes: Sizes) -> Result<Self> {
        if num_indexes > MAX_INDEXES {
            return Err(Error::LimitExceeded("SOHM index count"));
        }
        let mut r = ByteReader::with_sizes(data, sizes);
        r.expect_signature(b"SMTB")?;
        let mut indexes = Vec::with_capacity(num_indexes);
        for _ in 0..num_indexes {
            let version = r.read_u8()?;
            if version != 0 {
                return Err(Error::unsupported(format!("SOHM index version {version}")));
            }
            let index_type = match r.read_u8()? {
                0 => SohmIndexType::List,
                1 => SohmIndexType::BTree,
                t => return Err(Error::invalid(format!("unknown SOHM index type {t}"))),
            };
            indexes.push(SohmIndex {
                version,
                index_type,
                message_types: r.read_u16()?,
                min_message_size: r.read_u32()?,
                list_cutoff: r.read_u16()?,
                btree_cutoff: r.read_u16()?,
                num_messages: r.read_u16()?,
                index_address: r.read_offset()?,
                heap_address: r.read_offset()?,
            });
        }
        verify_checksum(&mut r, "shared message table")?;
        Ok(SohmTable { indexes })
    }

    /// The index tracking messages of `msg_type`, if any.
    pub fn index_for(&self, msg_type: u16) -> Option<&SohmIndex> {
        self.indexes.iter().find(|i| i.tracks(msg_type))
    }
}

/// One shared message record (list entry or v2 B-tree type 7 record).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SohmRecord {
    /// The message lives in the index's fractal heap.
    InHeap {
        /// lookup3 hash of the encoded message.
        hash: u32,
        /// Number of objects using the message.
        reference_count: u32,
        /// Fractal heap ID of the message.
        heap_id: [u8; 8],
    },
    /// The message lives in an object header.
    InObjectHeader {
        /// lookup3 hash of the encoded message.
        hash: u32,
        /// Message type.
        message_type: u8,
        /// Index of the message among messages in the header.
        creation_index: u16,
        /// Address of the object header.
        object_header_address: u64,
    },
}

impl SohmRecord {
    /// Size of one list record: both forms are padded to the larger one.
    pub fn encoded_size(sizes: Sizes) -> usize {
        // location (1) + hash (4) + max(refcount 4 + heap id 8,
        // reserved 1 + type 1 + index 2 + address O)
        5 + 12usize.max(4 + sizes.offset as usize)
    }

    /// Decode one record from the start of `data`.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        match r.read_u8()? {
            0 => Ok(SohmRecord::InHeap {
                hash: r.read_u32()?,
                reference_count: r.read_u32()?,
                heap_id: r.read_array::<8>()?,
            }),
            1 => {
                let hash = r.read_u32()?;
                r.skip(1)?;
                Ok(SohmRecord::InObjectHeader {
                    hash,
                    message_type: r.read_u8()?,
                    creation_index: r.read_u16()?,
                    object_header_address: r.read_offset()?,
                })
            }
            l => Err(Error::invalid(format!(
                "unknown shared message location {l}"
            ))),
        }
    }
}

/// A list index (`SMLI`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmList {
    /// The records, in stored order.
    pub records: Vec<SohmRecord>,
}

impl SohmList {
    /// Read the list at `addr` holding `count` records.
    pub fn read<R: ReadAt + ?Sized>(
        store: &R,
        addr: u64,
        count: usize,
        sizes: Sizes,
    ) -> Result<Self> {
        let len = count
            .checked_mul(SohmRecord::encoded_size(sizes))
            .and_then(|n| n.checked_add(8))
            .ok_or(Error::LimitExceeded("SOHM list size"))?;
        if len as u64 > store.size()? {
            return Err(Error::LimitExceeded("SOHM list size"));
        }
        let buf = store.read_vec_at(addr, len)?;
        Self::parse(&buf, count, sizes)
    }

    /// Parse a list of `count` records from `data`.
    pub fn parse(data: &[u8], count: usize, sizes: Sizes) -> Result<Self> {
        let rec = SohmRecord::encoded_size(sizes);
        let mut r = ByteReader::with_sizes(data, sizes);
        r.expect_signature(b"SMLI")?;
        let mut records = Vec::new();
        for _ in 0..count {
            records.push(SohmRecord::parse(r.read_bytes(rec)?, sizes)?);
        }
        verify_checksum(&mut r, "shared message list")?;
        Ok(SohmList { records })
    }
}

fn verify_checksum(r: &mut ByteReader<'_>, what: &'static str) -> Result<()> {
    let covered = r.position();
    let expected = r.read_u32()?;
    let actual = metadata_checksum(r.data().get(..covered).ok_or(Error::UnexpectedEof)?);
    if expected != actual {
        return Err(Error::ChecksumMismatch {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::{Rng, W};

    fn seal(w: &mut W) {
        let sum = metadata_checksum(&w.buf);
        w.u32(sum);
    }

    #[test]
    fn table_message() {
        let mut w = W::new();
        w.u8(0).offset(0x1234).u8(2);
        let m = SharedMessageTableMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(m.table_address, 0x1234);
        assert_eq!(m.num_indexes, 2);
        assert!(SharedMessageTableMessage::parse(&[1, 0, 0], Sizes::DEFAULT).is_err());
    }

    #[test]
    fn table_and_list() {
        let mut w = W::new();
        w.bytes(b"SMTB");
        w.u8(0).u8(0).u16(0x000a).u32(1).u16(50).u16(40).u16(2);
        w.offset(500).offset(600);
        w.u8(0).u8(1).u16(0x1820).u32(1).u16(50).u16(40).u16(9);
        w.offset(700).offset(800);
        seal(&mut w);
        assert_eq!(w.len(), SohmTable::encoded_size(2, Sizes::DEFAULT));
        let t = SohmTable::parse(&w.buf, 2, Sizes::DEFAULT).unwrap();
        assert_eq!(t.indexes[0].index_type, SohmIndexType::List);
        assert_eq!(
            t.index_for(MessageType::DATATYPE).unwrap().heap_address,
            600
        );
        assert_eq!(
            t.index_for(MessageType::ATTRIBUTE).unwrap().heap_address,
            800
        );
        assert_eq!(
            t.index_for(MessageType::ATTRIBUTE).unwrap().index_type,
            SohmIndexType::BTree
        );
        assert!(t.index_for(MessageType::LINK).is_none());
        let mut bad = w.buf.clone();
        bad[6] ^= 1;
        assert!(matches!(
            SohmTable::parse(&bad, 2, Sizes::DEFAULT),
            Err(Error::ChecksumMismatch { .. })
        ));

        let mut l = W::new();
        l.bytes(b"SMLI");
        l.u8(0).u32(0xaabb).u32(3).bytes(&[1, 2, 3, 4, 5, 6, 7, 8]);
        l.u8(1).u32(0xccdd).u8(0).u8(3).u16(1).offset(0x99);
        seal(&mut l);
        let list = SohmList::parse(&l.buf, 2, Sizes::DEFAULT).unwrap();
        assert_eq!(
            list.records[0],
            SohmRecord::InHeap {
                hash: 0xaabb,
                reference_count: 3,
                heap_id: [1, 2, 3, 4, 5, 6, 7, 8]
            }
        );
        assert_eq!(
            list.records[1],
            SohmRecord::InObjectHeader {
                hash: 0xccdd,
                message_type: 3,
                creation_index: 1,
                object_header_address: 0x99
            }
        );
    }

    #[test]
    fn never_panics() {
        let mut rng = Rng(7);
        for _ in 0..5000 {
            let len = rng.below(120);
            let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let _ = SohmTable::parse(&data, rng.below(10), Sizes::DEFAULT);
            let _ = SohmList::parse(&data, rng.below(10), Sizes::DEFAULT);
            let _ = SharedMessageTableMessage::parse(&data, Sizes::DEFAULT);
        }
    }
}
