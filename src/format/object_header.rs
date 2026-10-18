//! Data object headers (spec: "Disk Format: Level 2A - Data Object
//! Headers"), versions 1 and 2, including continuation blocks.

use std::collections::{HashSet, VecDeque};

use super::messages::MessageType;
use super::messages::continuation::ContinuationMessage;
use super::messages::ref_count::RefCountMessage;
use super::{ByteReader, ByteWriter, Sizes, UNDEFINED_ADDRESS};
use crate::checksum::metadata_checksum;
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Maximum number of chunks (header block plus continuation blocks).
pub const MAX_CHUNKS: usize = 1 << 16;
/// Maximum number of messages (including NIL and continuation messages).
pub const MAX_MESSAGES: usize = 1 << 20;

/// Size of the version 1 prefix, including its trailing alignment padding.
const V1_PREFIX_SIZE: u64 = 16;
/// Largest possible version 2 prefix: signature, version, flags, four
/// times, two phase change values and an 8-byte chunk size.
const V2_MAX_PREFIX_SIZE: u64 = 4 + 1 + 1 + 16 + 4 + 8;

const V2_FLAG_CHUNK0_SIZE: u8 = 0x03;
const V2_FLAG_CREATION_ORDER_TRACKED: u8 = 0x04;
const V2_FLAG_PHASE_CHANGE: u8 = 0x10;
const V2_FLAG_TIMES: u8 = 0x20;

/// One header message, undecoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawMessage {
    /// Message type (see [`MessageType`]).
    pub msg_type: u16,
    /// Message flags (see the `FLAG_*` constants).
    pub flags: u8,
    /// Creation order, present in version 2 headers that track attribute
    /// creation order.
    pub creation_order: Option<u16>,
    /// The message body. For version 1 headers this includes any trailing
    /// alignment padding. When [`is_shared`](Self::is_shared) is true this
    /// is a [`SharedMessage`](super::messages::shared::SharedMessage)
    /// record rather than the message itself.
    pub data: Vec<u8>,
}

impl RawMessage {
    /// Flag bit 0: the message data is constant.
    pub const FLAG_CONSTANT: u8 = 0x01;
    /// Flag bit 1: the message is shared; `data` is a shared message record.
    pub const FLAG_SHARED: u8 = 0x02;
    /// Flag bit 2: the message must not be shared.
    pub const FLAG_DONT_SHARE: u8 = 0x04;
    /// Flag bit 3: fail if the type is unknown and the file is writable.
    pub const FLAG_FAIL_IF_UNKNOWN_WRITE: u8 = 0x08;
    /// Flag bit 4: set bit 5 if the type is unknown and the object is
    /// modified.
    pub const FLAG_MARK_IF_UNKNOWN: u8 = 0x10;
    /// Flag bit 5: modified by software that did not understand the type.
    pub const FLAG_WAS_UNKNOWN: u8 = 0x20;
    /// Flag bit 6: the message is shareable.
    pub const FLAG_SHAREABLE: u8 = 0x40;
    /// Flag bit 7: always fail if the type is unknown.
    pub const FLAG_FAIL_IF_UNKNOWN_ALWAYS: u8 = 0x80;

    /// Whether the message body is stored elsewhere (flag bit 1).
    pub fn is_shared(&self) -> bool {
        self.flags & Self::FLAG_SHARED != 0
    }
}

/// Object timestamps stored in a version 2 header (seconds since the epoch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectTimes {
    /// Last raw data access.
    pub access: u32,
    /// Last raw data modification.
    pub modification: u32,
    /// Last metadata change.
    pub change: u32,
    /// Creation.
    pub birth: u32,
}

/// Non-default attribute storage phase change values (version 2 headers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttributePhaseChange {
    /// Maximum number of attributes stored compactly.
    pub max_compact: u16,
    /// Minimum number of attributes stored densely.
    pub min_dense: u16,
}

/// Location of one chunk of an object header as stored in the file.
///
/// Chunk 0 spans the whole header block (prefix, messages and, for
/// version 2, the checksum); continuation chunks are the
/// `(offset, length)` given by their continuation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkSpan {
    /// Address of the chunk.
    pub address: u64,
    /// Size of the chunk in bytes.
    pub size: u64,
}

/// A fully loaded object header.
///
/// `messages` lists every message from all chunks in the order they were
/// encountered (chunk 0 first, then continuation chunks breadth-first).
/// NIL messages are skipped, and continuation messages are consumed while
/// loading and **not** included; their targets are recorded in `chunks`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectHeader {
    /// Address the header was read from.
    pub address: u64,
    /// Header version (1 or 2).
    pub version: u8,
    /// Version 2 header flags (0 for version 1).
    pub flags: u8,
    /// Hard-link reference count. For version 2 headers this comes from
    /// the Object Reference Count message, defaulting to 1.
    pub ref_count: u32,
    /// Timestamps (version 2, flag bit 5).
    pub times: Option<ObjectTimes>,
    /// Attribute phase change values (version 2, flag bit 4).
    pub attribute_phase_change: Option<AttributePhaseChange>,
    /// "Total number of header messages" declared by a version 1 header
    /// (not enforced: includes NIL and continuation messages).
    pub declared_message_count: Option<u16>,
    /// All chunks that make up this header.
    pub chunks: Vec<ChunkSpan>,
    /// The header messages.
    pub messages: Vec<RawMessage>,
}

impl ObjectHeader {
    /// Read the object header at `address`.
    ///
    /// `address` and continuation addresses are positions in `store`; for
    /// a file with a user block, pass a store produced by
    /// [`Superblock::relative_store`](super::superblock::Superblock::relative_store).
    pub fn read<R: ReadAt + ?Sized>(store: &R, address: u64, sizes: Sizes) -> Result<Self> {
        if address == UNDEFINED_ADDRESS {
            return Err(Error::invalid("object header at undefined address"));
        }
        let mut loader = Loader {
            store,
            sizes,
            file_size: store.size()?,
            total_bytes: 0,
            message_count: 0,
            visited: HashSet::new(),
            pending: VecDeque::new(),
            chunks: Vec::new(),
            messages: Vec::new(),
        };
        let sig = store.read_vec_at(address, 4)?;
        let mut header = if sig == b"OHDR" {
            loader.read_v2(address)?
        } else if sig.first() == Some(&1) {
            loader.read_v1(address)?
        } else {
            return Err(Error::invalid("unrecognized object header version"));
        };
        loader.drain(header.version, header.flags)?;
        header.chunks = loader.chunks;
        header.messages = loader.messages;
        if header.version == 2 {
            header.ref_count = match header
                .messages
                .iter()
                .find(|m| m.msg_type == MessageType::REF_COUNT && !m.is_shared())
            {
                Some(m) => RefCountMessage::parse(&m.data, sizes)?.ref_count,
                None => 1,
            };
        }
        Ok(header)
    }

    /// Iterate over messages of one type.
    pub fn messages_of_type(&self, msg_type: u16) -> impl Iterator<Item = &RawMessage> {
        self.messages.iter().filter(move |m| m.msg_type == msg_type)
    }

    /// The first message of a given type, if any.
    pub fn find_message(&self, msg_type: u16) -> Option<&RawMessage> {
        self.messages_of_type(msg_type).next()
    }

    /// Whether attribute creation order is tracked (version 2, flag bit 2),
    /// which is also what makes per-message creation order fields present.
    pub fn attribute_creation_order_tracked(&self) -> bool {
        self.version == 2 && self.flags & V2_FLAG_CREATION_ORDER_TRACKED != 0
    }

    /// Whether attribute creation order is indexed (version 2, flag bit 3).
    pub fn attribute_creation_order_indexed(&self) -> bool {
        self.version == 2 && self.flags & 0x08 != 0
    }
}

/// Settings of a version 2 object header being encoded (see
/// [`ObjectHeader::encode_v2`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct V2HeaderOptions {
    /// Store object timestamps (flag bit 5).
    pub times: Option<ObjectTimes>,
    /// Store non-default attribute phase change values (flag bit 4).
    pub attribute_phase_change: Option<AttributePhaseChange>,
    /// Track attribute creation order (flag bit 2); every message then
    /// carries a creation order field (`RawMessage::creation_order`,
    /// defaulting to 0).
    pub attribute_creation_order_tracked: bool,
    /// Index attribute creation order (flag bit 3; requires tracking).
    pub attribute_creation_order_indexed: bool,
}

impl ObjectHeader {
    /// Encode a single-chunk version 2 object header (`OHDR`, prefix,
    /// messages and checksum).
    ///
    /// Messages are written in order with their `msg_type`, `flags` and
    /// `data`; continuation messages are not generated, so everything must
    /// fit in chunk 0 (whose size field is sized automatically). A message
    /// body longer than 65535 bytes is an error.
    pub fn encode_v2(opts: &V2HeaderOptions, messages: &[RawMessage]) -> Result<Vec<u8>> {
        let tracked = opts.attribute_creation_order_tracked;
        if opts.attribute_creation_order_indexed && !tracked {
            return Err(Error::invalid(
                "attribute creation order indexed but not tracked",
            ));
        }
        let mut body = ByteWriter::new();
        for m in messages {
            let ty = u8::try_from(m.msg_type)
                .map_err(|_| Error::invalid("message type does not fit a version 2 header"))?;
            let size = u16::try_from(m.data.len())
                .map_err(|_| Error::LimitExceeded("header message larger than 65535 bytes"))?;
            body.u8(ty).u16(size).u8(m.flags);
            if tracked {
                body.u16(m.creation_order.unwrap_or(0));
            }
            body.bytes(&m.data);
        }
        let body = body.into_vec();
        let (size_bits, width) = match body.len() {
            0..=0xff => (0u8, 1usize),
            0x100..=0xffff => (1, 2),
            0x1_0000..=0xffff_ffff => (2, 4),
            _ => (3, 8),
        };
        let mut flags = size_bits;
        if tracked {
            flags |= V2_FLAG_CREATION_ORDER_TRACKED;
        }
        if opts.attribute_creation_order_indexed {
            flags |= 0x08;
        }
        if opts.attribute_phase_change.is_some() {
            flags |= V2_FLAG_PHASE_CHANGE;
        }
        if opts.times.is_some() {
            flags |= V2_FLAG_TIMES;
        }
        let mut w = ByteWriter::new();
        w.bytes(b"OHDR").u8(2).u8(flags);
        if let Some(t) = opts.times {
            w.u32(t.access)
                .u32(t.modification)
                .u32(t.change)
                .u32(t.birth);
        }
        if let Some(p) = opts.attribute_phase_change {
            w.u16(p.max_compact).u16(p.min_dense);
        }
        w.write_uint(body.len() as u64, width)?;
        w.bytes(&body);
        let sum = metadata_checksum(w.as_slice());
        w.u32(sum);
        Ok(w.into_vec())
    }
}

impl ObjectHeader {
    /// Encode a single-chunk version 1 object header: the 16-byte prefix
    /// (with `ref_count`) followed by the messages, each padded to a
    /// multiple of eight bytes.
    pub fn encode_v1(ref_count: u32, messages: &[RawMessage]) -> Result<Vec<u8>> {
        let mut body = ByteWriter::new();
        for m in messages {
            let padded = m.data.len().next_multiple_of(8);
            let size = u16::try_from(padded)
                .map_err(|_| Error::LimitExceeded("header message larger than 65535 bytes"))?;
            body.u16(m.msg_type).u16(size).u8(m.flags).zeros(3);
            body.bytes(&m.data).zeros(padded - m.data.len());
        }
        let count = u16::try_from(messages.len())
            .map_err(|_| Error::LimitExceeded("too many header messages"))?;
        let size = u32::try_from(body.len())
            .map_err(|_| Error::LimitExceeded("object header larger than 4 GiB"))?;
        let mut w = ByteWriter::new();
        w.u8(1).u8(0).u16(count).u32(ref_count).u32(size).zeros(4);
        w.bytes(body.as_slice());
        Ok(w.into_vec())
    }
}

struct Loader<'a, R: ?Sized> {
    store: &'a R,
    sizes: Sizes,
    file_size: u64,
    total_bytes: u64,
    message_count: usize,
    visited: HashSet<u64>,
    pending: VecDeque<ContinuationMessage>,
    chunks: Vec<ChunkSpan>,
    messages: Vec<RawMessage>,
}

impl<R: ReadAt + ?Sized> Loader<'_, R> {
    /// Account for a new chunk and load its bytes.
    fn load_chunk(&mut self, address: u64, size: u64) -> Result<Vec<u8>> {
        if !self.visited.insert(address) {
            return Err(Error::invalid("object header continuation cycle"));
        }
        if self.chunks.len() >= MAX_CHUNKS {
            return Err(Error::LimitExceeded("object header chunk count"));
        }
        // Chunks never overlap in a valid file, so their total cannot
        // exceed the file size.
        self.total_bytes = self
            .total_bytes
            .checked_add(size)
            .filter(|&t| t <= self.file_size)
            .ok_or(Error::LimitExceeded("object header size exceeds file size"))?;
        let len = usize::try_from(size).map_err(|_| Error::UnexpectedEof)?;
        let data = self.store.read_vec_at(address, len)?;
        self.chunks.push(ChunkSpan { address, size });
        Ok(data)
    }

    fn read_v1(&mut self, address: u64) -> Result<ObjectHeader> {
        let prefix = self.store.read_vec_at(address, V1_PREFIX_SIZE as usize)?;
        let mut r = ByteReader::new(&prefix);
        let version = r.read_u8()?;
        debug_assert_eq!(version, 1);
        r.skip(1)?;
        let declared = r.read_u16()?;
        let ref_count = r.read_u32()?;
        let size = u64::from(r.read_u32()?);
        let total = size
            .checked_add(V1_PREFIX_SIZE)
            .ok_or(Error::UnexpectedEof)?;
        let block = self.load_chunk(address, total)?;
        let body = block
            .get(V1_PREFIX_SIZE as usize..)
            .ok_or(Error::UnexpectedEof)?;
        self.parse_v1_messages(body)?;
        Ok(ObjectHeader {
            address,
            version: 1,
            flags: 0,
            ref_count,
            times: None,
            attribute_phase_change: None,
            declared_message_count: Some(declared),
            chunks: Vec::new(),
            messages: Vec::new(),
        })
    }

    fn read_v2(&mut self, address: u64) -> Result<ObjectHeader> {
        let avail = self.file_size.saturating_sub(address);
        let head = self
            .store
            .read_vec_at(address, avail.min(V2_MAX_PREFIX_SIZE) as usize)?;
        let mut r = ByteReader::new(&head);
        r.expect_signature(b"OHDR")?;
        let version = r.read_u8()?;
        if version != 2 {
            return Err(Error::unsupported(format!(
                "object header version {version}"
            )));
        }
        let flags = r.read_u8()?;
        if flags & 0xc0 != 0 {
            return Err(Error::invalid("reserved object header flags set"));
        }
        let times = if flags & V2_FLAG_TIMES != 0 {
            Some(ObjectTimes {
                access: r.read_u32()?,
                modification: r.read_u32()?,
                change: r.read_u32()?,
                birth: r.read_u32()?,
            })
        } else {
            None
        };
        let attribute_phase_change = if flags & V2_FLAG_PHASE_CHANGE != 0 {
            Some(AttributePhaseChange {
                max_compact: r.read_u16()?,
                min_dense: r.read_u16()?,
            })
        } else {
            None
        };
        let chunk0_size = r.read_uint(1 << (flags & V2_FLAG_CHUNK0_SIZE))?;
        let prefix_len = r.position();
        let total = chunk0_size
            .checked_add(prefix_len as u64 + 4)
            .ok_or(Error::UnexpectedEof)?;
        let block = self.load_chunk(address, total)?;
        let body = verify_v2_chunk(&block, "object header")?;
        let body = body.get(prefix_len..).ok_or(Error::UnexpectedEof)?;
        self.parse_v2_messages(body, flags)?;
        Ok(ObjectHeader {
            address,
            version: 2,
            flags,
            ref_count: 1,
            times,
            attribute_phase_change,
            declared_message_count: None,
            chunks: Vec::new(),
            messages: Vec::new(),
        })
    }

    /// Follow continuation messages until none remain.
    fn drain(&mut self, version: u8, flags: u8) -> Result<()> {
        while let Some(cont) = self.pending.pop_front() {
            if cont.offset == UNDEFINED_ADDRESS {
                return Err(Error::invalid("continuation to undefined address"));
            }
            let block = self.load_chunk(cont.offset, cont.length)?;
            if version == 1 {
                if cont.length == 0 {
                    return Err(Error::invalid("empty object header continuation"));
                }
                self.parse_v1_messages(&block)?;
            } else {
                let body = verify_v2_chunk(&block, "object header continuation")?;
                let mut r = ByteReader::new(body);
                r.expect_signature(b"OCHK")?;
                self.parse_v2_messages(r.rest(), flags)?;
            }
        }
        Ok(())
    }

    fn count_message(&mut self) -> Result<()> {
        self.message_count += 1;
        if self.message_count > MAX_MESSAGES {
            return Err(Error::LimitExceeded("object header message count"));
        }
        Ok(())
    }

    fn push(&mut self, msg: RawMessage) -> Result<()> {
        self.count_message()?;
        match msg.msg_type {
            MessageType::NIL => {}
            MessageType::CONTINUATION => {
                self.pending
                    .push_back(ContinuationMessage::parse(&msg.data, self.sizes)?);
            }
            _ => self.messages.push(msg),
        }
        Ok(())
    }

    /// Version 1 message sequence: type (2), size (2), flags (1), reserved
    /// (3), data; each message starts on an 8-byte boundary relative to the
    /// block start. A tail shorter than a message prefix is ignored.
    fn parse_v1_messages(&mut self, block: &[u8]) -> Result<()> {
        let mut r = ByteReader::new(block);
        while r.remaining() >= 8 {
            let msg_type = r.read_u16()?;
            let size = r.read_u16()? as usize;
            let flags = r.read_u8()?;
            r.skip(3)?;
            let data = r.read_bytes(size)?.to_vec();
            self.push(RawMessage {
                msg_type,
                flags,
                creation_order: None,
                data,
            })?;
            let aligned = r.position().next_multiple_of(8).min(block.len());
            r.seek(aligned)?;
        }
        Ok(())
    }

    /// Version 2 message sequence: type (1), size (2), flags (1), optional
    /// creation order (2), data. A tail shorter than a message prefix is a
    /// gap and is ignored.
    fn parse_v2_messages(&mut self, body: &[u8], header_flags: u8) -> Result<()> {
        let has_order = header_flags & V2_FLAG_CREATION_ORDER_TRACKED != 0;
        let prefix = if has_order { 6 } else { 4 };
        let mut r = ByteReader::new(body);
        while r.remaining() >= prefix {
            let msg_type = u16::from(r.read_u8()?);
            let size = r.read_u16()? as usize;
            let flags = r.read_u8()?;
            let creation_order = if has_order { Some(r.read_u16()?) } else { None };
            let data = r.read_bytes(size)?.to_vec();
            self.push(RawMessage {
                msg_type,
                flags,
                creation_order,
                data,
            })?;
        }
        Ok(())
    }
}

/// Verify a version 2 chunk's trailing checksum and return everything
/// before it.
fn verify_v2_chunk<'a>(block: &'a [u8], what: &'static str) -> Result<&'a [u8]> {
    let split = block.len().checked_sub(4).ok_or(Error::UnexpectedEof)?;
    let (body, sum) = block.split_at(split);
    let mut sr = ByteReader::new(sum);
    let expected = sr.read_u32()?;
    let actual = metadata_checksum(body);
    if expected != actual {
        return Err(Error::ChecksumMismatch {
            what,
            expected,
            actual,
        });
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::{Rng, W, mutate};

    /// A version 1 message with its body padded to 8 bytes.
    fn v1_msg(w: &mut W, ty: u16, flags: u8, data: &[u8]) {
        let padded = data.len().next_multiple_of(8);
        w.u16(ty).u16(padded as u16).u8(flags).zeros(3).bytes(data);
        w.zeros(padded - data.len());
    }

    fn cont_body(offset: u64, length: u64) -> Vec<u8> {
        let mut w = W::new();
        w.offset(offset).length(length);
        w.buf
    }

    /// Build a file with a v1 header at 0 and one continuation block.
    fn v1_file() -> Vec<u8> {
        let mut msgs = W::new();
        v1_msg(
            &mut msgs,
            MessageType::SYMBOL_TABLE,
            0,
            &cont_body(100, 200),
        );
        v1_msg(&mut msgs, MessageType::NIL, 0, &[0; 8]);
        // Continuation at 256, 24 bytes.
        v1_msg(&mut msgs, MessageType::CONTINUATION, 0, &cont_body(256, 24));
        let mut w = W::new();
        w.u8(1).u8(0).u16(4).u32(2).u32(msgs.len() as u32).u32(0);
        w.bytes(&msgs.buf);
        w.zeros(256 - w.len());
        let mut cont = W::new();
        v1_msg(&mut cont, MessageType::COMMENT, 0, b"hi\0");
        cont.zeros(8); // a zero-length NIL message
        assert_eq!(cont.len(), 24);
        w.bytes(&cont.buf);
        w.buf
    }

    #[test]
    fn v1_with_continuation() {
        let data = v1_file();
        let h = ObjectHeader::read(&data, 0, Sizes::DEFAULT).unwrap();
        assert_eq!(h.version, 1);
        assert_eq!(h.ref_count, 2);
        assert_eq!(h.declared_message_count, Some(4));
        assert_eq!(h.messages.len(), 2);
        assert_eq!(h.messages[0].msg_type, MessageType::SYMBOL_TABLE);
        assert_eq!(h.messages[0].data.len(), 16);
        assert_eq!(h.messages[1].msg_type, MessageType::COMMENT);
        assert_eq!(h.messages[1].data, b"hi\0\0\0\0\0\0");
        assert_eq!(
            h.chunks,
            vec![
                ChunkSpan {
                    address: 0,
                    size: 16 + 24 + 16 + 24
                },
                ChunkSpan {
                    address: 256,
                    size: 24
                }
            ]
        );
        assert!(h.find_message(MessageType::COMMENT).is_some());
    }

    #[test]
    fn v1_unpadded_sizes_are_realigned() {
        let mut msgs = W::new();
        msgs.u16(MessageType::COMMENT).u16(3).u8(0).zeros(3);
        msgs.bytes(b"ab\0").zeros(5);
        msgs.u16(MessageType::COMMENT).u16(2).u8(0).zeros(3);
        msgs.bytes(b"c\0").zeros(6);
        let mut w = W::new();
        w.u8(1).u8(0).u16(2).u32(1).u32(msgs.len() as u32).u32(0);
        w.bytes(&msgs.buf);
        let h = ObjectHeader::read(&w.buf, 0, Sizes::DEFAULT).unwrap();
        assert_eq!(h.messages.len(), 2);
        assert_eq!(h.messages[1].data, b"c\0");
    }

    #[test]
    fn v1_cycle_detected() {
        let mut data = v1_file();
        // Make the continuation block point back at itself.
        let mut cont = W::new();
        v1_msg(&mut cont, MessageType::CONTINUATION, 0, &cont_body(256, 24));
        data.truncate(256);
        data.extend_from_slice(&cont.buf);
        let e = ObjectHeader::read(&data, 0, Sizes::DEFAULT).unwrap_err();
        assert!(matches!(e, Error::InvalidFormat(_)), "{e}");
        // Pointing back at the header itself is also a cycle.
        let mut cont = W::new();
        v1_msg(&mut cont, MessageType::CONTINUATION, 0, &cont_body(0, 24));
        data.truncate(256);
        data.extend_from_slice(&cont.buf);
        assert!(ObjectHeader::read(&data, 0, Sizes::DEFAULT).is_err());
    }

    /// A version 2 chunk 0 with the given flags and raw message bytes.
    fn v2_chunk0(flags: u8, msgs: &[u8]) -> Vec<u8> {
        let mut w = W::new();
        w.bytes(b"OHDR").u8(2).u8(flags);
        if flags & V2_FLAG_TIMES != 0 {
            w.u32(1).u32(2).u32(3).u32(4);
        }
        if flags & V2_FLAG_PHASE_CHANGE != 0 {
            w.u16(8).u16(6);
        }
        w.uint(msgs.len() as u64, 1 << (flags & 3));
        w.bytes(msgs);
        let c = metadata_checksum(&w.buf);
        w.u32(c);
        w.buf
    }

    fn v2_msg(w: &mut W, ty: u8, flags: u8, order: Option<u16>, data: &[u8]) {
        w.u8(ty).u16(data.len() as u16).u8(flags);
        if let Some(o) = order {
            w.u16(o);
        }
        w.bytes(data);
    }

    fn ochk(msgs: &[u8]) -> Vec<u8> {
        let mut w = W::new();
        w.bytes(b"OCHK").bytes(msgs);
        let c = metadata_checksum(&w.buf);
        w.u32(c);
        w.buf
    }

    /// v2 header at 0 (with times, phase change, creation order and a
    /// 2-byte chunk size) plus an OCHK block at 512.
    fn v2_file() -> Vec<u8> {
        let flags = 0x01 | 0x04 | 0x10 | 0x20;
        let mut cont = W::new();
        v2_msg(&mut cont, 0x0C, 0, Some(1), b"attr-bytes");
        v2_msg(&mut cont, 0x00, 0, Some(0), &[0; 3]);
        cont.zeros(3); // gap smaller than a 6-byte prefix
        let block = ochk(&cont.buf);

        let mut msgs = W::new();
        v2_msg(&mut msgs, 0x16, 0, Some(0), &[0, 5, 0, 0, 0]);
        v2_msg(&mut msgs, 0x0C, 0x02, Some(0), b"xyz");
        v2_msg(
            &mut msgs,
            0x10,
            0,
            Some(0),
            &cont_body(512, block.len() as u64),
        );
        let mut data = v2_chunk0(flags, &msgs.buf);
        data.resize(512, 0);
        data.extend_from_slice(&block);
        data
    }

    #[test]
    fn v2_with_continuation() {
        let data = v2_file();
        let h = ObjectHeader::read(&data, 0, Sizes::DEFAULT).unwrap();
        assert_eq!(h.version, 2);
        assert_eq!(h.ref_count, 5);
        assert_eq!(
            h.times,
            Some(ObjectTimes {
                access: 1,
                modification: 2,
                change: 3,
                birth: 4
            })
        );
        assert_eq!(
            h.attribute_phase_change,
            Some(AttributePhaseChange {
                max_compact: 8,
                min_dense: 6
            })
        );
        assert!(h.attribute_creation_order_tracked());
        assert!(!h.attribute_creation_order_indexed());
        let types: Vec<u16> = h.messages.iter().map(|m| m.msg_type).collect();
        assert_eq!(types, vec![0x16, 0x0C, 0x0C]);
        assert!(h.messages[1].is_shared());
        assert_eq!(h.messages[2].creation_order, Some(1));
        assert_eq!(h.messages[2].data, b"attr-bytes");
        assert_eq!(h.chunks.len(), 2);
        assert_eq!(h.chunks[1].address, 512);
        assert_eq!(h.messages_of_type(0x0C).count(), 2);
    }

    #[test]
    fn v2_minimal() {
        let mut msgs = W::new();
        v2_msg(&mut msgs, 0x01, 1, None, &[1, 2]);
        msgs.zeros(2); // gap
        let data = v2_chunk0(0, &msgs.buf);
        let h = ObjectHeader::read(&data, 0, Sizes::DEFAULT).unwrap();
        assert_eq!(h.ref_count, 1);
        assert_eq!(h.times, None);
        assert_eq!(h.messages.len(), 1);
        assert_eq!(h.messages[0].creation_order, None);
        assert_eq!(h.messages[0].flags, 1);
        assert_eq!(h.chunks[0].size, data.len() as u64);
    }

    #[test]
    fn v2_checksum_mismatch() {
        let mut data = v2_file();
        data[30] ^= 0x40;
        assert!(matches!(
            ObjectHeader::read(&data, 0, Sizes::DEFAULT),
            Err(Error::ChecksumMismatch {
                what: "object header",
                ..
            })
        ));
        let mut data = v2_file();
        data[520] ^= 0x01;
        assert!(matches!(
            ObjectHeader::read(&data, 0, Sizes::DEFAULT),
            Err(Error::ChecksumMismatch {
                what: "object header continuation",
                ..
            })
        ));
    }

    #[test]
    fn v2_cycle_detected() {
        // An OCHK block whose continuation points to itself.
        let mut cont = W::new();
        v2_msg(&mut cont, 0x10, 0, None, &cont_body(64, 4 + 20 + 4));
        let block = ochk(&cont.buf);
        assert_eq!(block.len(), 28);
        let mut msgs = W::new();
        v2_msg(&mut msgs, 0x10, 0, None, &cont_body(64, 28));
        let mut data = v2_chunk0(0, &msgs.buf);
        data.resize(64, 0);
        data.extend_from_slice(&block);
        let e = ObjectHeader::read(&data, 0, Sizes::DEFAULT).unwrap_err();
        assert!(matches!(e, Error::InvalidFormat(_)), "{e}");
    }

    #[test]
    fn encode_v1_round_trip() {
        let msgs = vec![
            RawMessage {
                msg_type: 0x11,
                flags: 0,
                creation_order: None,
                data: vec![1; 16],
            },
            RawMessage {
                msg_type: 0x0c,
                flags: 1,
                creation_order: None,
                data: vec![2; 13],
            },
        ];
        let mut data = vec![0u8; 8];
        data.extend(ObjectHeader::encode_v1(3, &msgs).unwrap());
        let h = ObjectHeader::read(&data, 8, Sizes::DEFAULT).unwrap();
        assert_eq!(h.version, 1);
        assert_eq!(h.ref_count, 3);
        assert_eq!(h.declared_message_count, Some(2));
        assert_eq!(h.messages[0], msgs[0]);
        assert_eq!(h.messages[1].data[..13], msgs[1].data[..]);
        assert_eq!(h.messages[1].data.len(), 16);
    }

    #[test]
    fn encode_v2_round_trip() {
        let msg = |ty: u16, flags: u8, order: Option<u16>, len: usize| RawMessage {
            msg_type: ty,
            flags,
            creation_order: order,
            data: (0..len).map(|i| i as u8).collect(),
        };
        let cases = [
            (
                V2HeaderOptions::default(),
                vec![msg(1, 0, None, 8), msg(3, 1, None, 12)],
            ),
            (
                V2HeaderOptions {
                    times: Some(ObjectTimes {
                        access: 1,
                        modification: 2,
                        change: 3,
                        birth: 4,
                    }),
                    attribute_phase_change: Some(AttributePhaseChange {
                        max_compact: 20,
                        min_dense: 10,
                    }),
                    attribute_creation_order_tracked: true,
                    attribute_creation_order_indexed: true,
                },
                vec![msg(12, 0, Some(0), 300), msg(12, 0, Some(1), 3)],
            ),
            (
                V2HeaderOptions::default(),
                vec![msg(12, 0, None, 65535), msg(1, 0, None, 4)],
            ),
        ];
        for (opts, msgs) in cases {
            let mut data = vec![0u8; 16];
            data.extend(ObjectHeader::encode_v2(&opts, &msgs).unwrap());
            let h = ObjectHeader::read(&data, 16, Sizes::DEFAULT).unwrap();
            assert_eq!(h.version, 2);
            assert_eq!(h.times, opts.times);
            assert_eq!(h.attribute_phase_change, opts.attribute_phase_change);
            assert_eq!(
                h.attribute_creation_order_tracked(),
                opts.attribute_creation_order_tracked
            );
            assert_eq!(h.messages, msgs);
            assert_eq!(h.chunks.len(), 1);
            assert_eq!(h.chunks[0].size, data.len() as u64 - 16);
        }
        assert!(
            ObjectHeader::encode_v2(&V2HeaderOptions::default(), &[msg(1, 0, None, 65536)])
                .is_err()
        );
        assert!(
            ObjectHeader::encode_v2(&V2HeaderOptions::default(), &[msg(0x100, 0, None, 1)])
                .is_err()
        );
    }

    #[test]
    fn truncated() {
        let data = v2_file();
        for n in [0, 3, 5, 10, 40, 100, 511, 520, data.len() - 1] {
            assert!(
                ObjectHeader::read(&data[..n], 0, Sizes::DEFAULT).is_err(),
                "len {n}"
            );
        }
        let data = v1_file();
        for n in [0, 4, 15, 30, 255, 270, data.len() - 1] {
            assert!(
                ObjectHeader::read(&data[..n], 0, Sizes::DEFAULT).is_err(),
                "len {n}"
            );
        }
    }

    #[test]
    fn bad_headers() {
        assert!(ObjectHeader::read(&[0u8; 32][..], 0, Sizes::DEFAULT).is_err());
        assert!(ObjectHeader::read(&[0u8; 32][..], UNDEFINED_ADDRESS, Sizes::DEFAULT).is_err());
        // Message larger than the chunk.
        let mut msgs = W::new();
        msgs.u8(1).u16(100).u8(0);
        let data = v2_chunk0(0, &msgs.buf);
        assert!(ObjectHeader::read(&data, 0, Sizes::DEFAULT).is_err());
        // Huge chunk-0 size.
        let mut w = W::new();
        w.bytes(b"OHDR").u8(2).u8(3).u64(u64::MAX - 2);
        assert!(ObjectHeader::read(&w.buf, 0, Sizes::DEFAULT).is_err());
        // Reserved flags.
        let data = v2_chunk0(0x80, &[]);
        assert!(ObjectHeader::read(&data, 0, Sizes::DEFAULT).is_err());
    }

    #[test]
    fn many_chunks_limited_by_file_size() {
        // A chain of tiny v1 continuation blocks, each pointing to the next;
        // fine as long as the budget allows.
        let mut w = W::new();
        let mut first = W::new();
        v1_msg(&mut first, MessageType::CONTINUATION, 0, &cont_body(64, 24));
        w.u8(1).u8(0).u16(0).u32(1).u32(first.len() as u32).u32(0);
        w.bytes(&first.buf);
        w.zeros(64 - w.len());
        let n = 50u64;
        for i in 0..n {
            let next = 64 + (i + 1) * 24;
            let mut c = W::new();
            if i + 1 < n {
                v1_msg(&mut c, MessageType::CONTINUATION, 0, &cont_body(next, 24));
            } else {
                v1_msg(&mut c, MessageType::COMMENT, 0, b"end\0");
                c.zeros(8);
            }
            w.bytes(&c.buf);
        }
        let h = ObjectHeader::read(&w.buf, 0, Sizes::DEFAULT).unwrap();
        assert_eq!(h.chunks.len(), n as usize + 1);
        assert_eq!(h.messages.len(), 1);
    }

    #[test]
    fn mutation_never_panics() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        for seed in [v2_file(), v1_file()] {
            for _ in 0..5000 {
                let m = mutate(&mut rng, &seed);
                let _ = ObjectHeader::read(&m, 0, Sizes::DEFAULT);
            }
        }
        // Mutations that keep checksums valid exercise the message parser.
        let mut msgs = W::new();
        v2_msg(&mut msgs, 0x0C, 0, None, b"abcdef");
        v2_msg(&mut msgs, 0x10, 0, None, &cont_body(0, 30));
        v2_msg(&mut msgs, 0x16, 0, None, &[0, 1, 0, 0, 0]);
        for _ in 0..5000 {
            let body = mutate(&mut rng, &msgs.buf);
            let flags = (rng.next() as u8) & 0x3f;
            let data = v2_chunk0(flags, &body);
            let _ = ObjectHeader::read(&data, 0, Sizes::DEFAULT);
        }
    }
}
