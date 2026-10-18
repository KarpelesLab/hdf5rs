//! Harnesses for the individual low-level parsers, fed arbitrary bytes.
//!
//! Shared by the stress test (`tests/stress.rs`) and the cargo-fuzz
//! targets (`fuzz/`). Each harness interprets a few leading bytes as
//! parameters (offset/length widths, rank, filter id, …) and hands the
//! rest to the parser; results are discarded, only panics, hangs and
//! runaway allocations matter.

#![allow(dead_code)]

use hdf5rs::filters::{FilterContext, FilterRegistry};
use hdf5rs::format::Sizes;
use hdf5rs::format::btree_v1::{chunk_btree_entries, group_btree_children};
use hdf5rs::format::btree_v2::{BTreeV2, RecordContext};
use hdf5rs::format::chunk_index::ChunkIndexContext;
use hdf5rs::format::extensible_array::ExtensibleArray;
use hdf5rs::format::fixed_array::FixedArray;
use hdf5rs::format::fractal_heap::FractalHeap;
use hdf5rs::format::global_heap::GlobalHeapCollection;
use hdf5rs::format::local_heap::LocalHeap;
use hdf5rs::format::messages::MessageType;
use hdf5rs::format::messages::attribute::Attribute;
use hdf5rs::format::messages::attribute_info::AttributeInfo;
use hdf5rs::format::messages::btree_k::BtreeKMessage;
use hdf5rs::format::messages::comment::CommentMessage;
use hdf5rs::format::messages::continuation::ContinuationMessage;
use hdf5rs::format::messages::dataspace::Dataspace;
use hdf5rs::format::messages::datatype::Datatype;
use hdf5rs::format::messages::driver_info::DriverInfoMessage;
use hdf5rs::format::messages::external_files::ExternalFiles;
use hdf5rs::format::messages::fill_value::{FillValue, OldFillValue};
use hdf5rs::format::messages::filter_pipeline::FilterPipeline;
use hdf5rs::format::messages::group_info::GroupInfoMessage;
use hdf5rs::format::messages::layout::DataLayout;
use hdf5rs::format::messages::link::LinkMessage;
use hdf5rs::format::messages::link_info::LinkInfoMessage;
use hdf5rs::format::messages::modification_time::{
    ModificationTimeMessage, ModificationTimeOldMessage,
};
use hdf5rs::format::messages::ref_count::RefCountMessage;
use hdf5rs::format::messages::shared::SharedMessage;
use hdf5rs::format::messages::symbol_table::SymbolTableMessage;
use hdf5rs::format::object_header::ObjectHeader;
use hdf5rs::format::sohm::{SharedMessageTableMessage, SohmTable};
use hdf5rs::format::superblock::Superblock;
use hdf5rs::format::symbol_table_node::SymbolTableNode;
use hdf5rs::format::vds::parse_virtual_mappings;
use hdf5rs::types::{DecodeContext, GlobalHeapResolver, Value, decode_all};

/// A harness: arbitrary bytes in, nothing out.
pub type Harness = fn(&[u8]);

/// The parser harnesses, by name.
pub const TARGETS: &[(&str, Harness)] = &[
    ("object_header", object_header),
    ("message", message),
    ("datatype", datatype),
    ("superblock", superblock),
    ("fractal_heap", fractal_heap),
    ("btree_v1", btree_v1),
    ("btree_v2", btree_v2),
    ("chunk_index", chunk_index),
    ("heaps", heaps),
    ("filters", filters),
];

/// Offset/length widths chosen by a parameter byte.
fn sizes(b: u8) -> Sizes {
    let w = |x: u8| [8u8, 4, 2, 8][usize::from(x & 3)];
    Sizes {
        offset: w(b),
        length: w(b >> 2),
    }
}

fn split(data: &[u8]) -> (u8, &[u8]) {
    match data.split_first() {
        Some((&b, rest)) => (b, rest),
        None => (0, &[]),
    }
}

/// A little-endian 32-bit structure address, then the file image.
fn addr_split(data: &[u8]) -> (u64, &[u8]) {
    match data.split_first_chunk::<4>() {
        Some((a, rest)) => (u64::from(u32::from_le_bytes(*a)), rest),
        None => (0, &[]),
    }
}

/// Build an input for the address-taking harnesses: parameter byte(s),
/// then the address of the structure within `image`, then `image`.
pub fn with_address(params: &[u8], addr: u32, image: &[u8]) -> Vec<u8> {
    let mut v = params.to_vec();
    v.extend(addr.to_le_bytes());
    v.extend_from_slice(image);
    v
}

/// Where each harness expects its structure: signatures to look for in
/// a file image, and the number of parameter bytes before the address.
pub const SIGNATURES: &[(&str, &[u8], usize)] = &[
    ("object_header", b"OHDR", 1),
    ("fractal_heap", b"FRHP", 1),
    ("btree_v1", b"TREE", 1),
    ("btree_v2", b"BTHD", 2),
    ("chunk_index", b"FAHD", 1),
    ("chunk_index", b"EAHD", 1),
    ("heaps", b"HEAP", 1),
    ("heaps", b"GCOL", 1),
    ("heaps", b"SNOD", 1),
    ("heaps", b"SMTB", 1),
];

/// Parse a header message body of type `t` with the matching parser.
pub fn parse_message(t: u16, body: &[u8], sz: Sizes) {
    match t {
        MessageType::DATASPACE => drop(Dataspace::parse(body, sz)),
        MessageType::LINK_INFO => drop(LinkInfoMessage::parse(body, sz)),
        MessageType::DATATYPE => {
            if let Ok(dt) = Datatype::parse(body) {
                let _ = dt.to_string();
            }
        }
        MessageType::FILL_VALUE_OLD => drop(OldFillValue::parse(body, sz)),
        MessageType::FILL_VALUE => drop(FillValue::parse(body, sz)),
        MessageType::LINK => drop(LinkMessage::parse(body, sz)),
        MessageType::EXTERNAL_FILES => drop(ExternalFiles::parse(body, sz)),
        MessageType::LAYOUT => drop(DataLayout::parse(body, sz)),
        MessageType::GROUP_INFO => drop(GroupInfoMessage::parse(body, sz)),
        MessageType::FILTER_PIPELINE => drop(FilterPipeline::parse(body, sz)),
        MessageType::ATTRIBUTE => drop(Attribute::parse(body, sz)),
        MessageType::COMMENT => drop(CommentMessage::parse(body, sz)),
        MessageType::MODIFICATION_TIME_OLD => drop(ModificationTimeOldMessage::parse(body, sz)),
        MessageType::SHARED_MESSAGE_TABLE => drop(SharedMessageTableMessage::parse(body, sz)),
        MessageType::CONTINUATION => drop(ContinuationMessage::parse(body, sz)),
        MessageType::SYMBOL_TABLE => drop(SymbolTableMessage::parse(body, sz)),
        MessageType::MODIFICATION_TIME => drop(ModificationTimeMessage::parse(body, sz)),
        MessageType::BTREE_K => drop(BtreeKMessage::parse(body, sz)),
        MessageType::DRIVER_INFO => drop(DriverInfoMessage::parse(body, sz)),
        MessageType::ATTRIBUTE_INFO => drop(AttributeInfo::parse(body, sz)),
        MessageType::REF_COUNT => drop(RefCountMessage::parse(body, sz)),
        _ => drop(SharedMessage::parse(body, sz)),
    }
}

/// An object header, then every message it holds.
pub fn object_header(data: &[u8]) {
    let (p, rest) = split(data);
    let (addr, rest) = addr_split(rest);
    let sz = sizes(p);
    if let Ok(h) = ObjectHeader::read(rest, addr, sz) {
        for m in &h.messages {
            parse_message(m.msg_type, &m.data, sz);
        }
    }
}

/// One header message: type byte, then the body.
pub fn message(data: &[u8]) {
    let (p, rest) = split(data);
    let (t, body) = split(rest);
    parse_message(u16::from(t % 0x18), body, sizes(p));
    // Virtual dataset mappings are another global-heap-stored structure.
    let _ = parse_virtual_mappings(body, sizes(p));
}

/// A datatype message, then elements of that type decoded as `Value`s.
pub fn datatype(data: &[u8]) {
    let (n, rest) = split(data);
    let cut = usize::from(n).min(rest.len());
    let (desc, payload) = rest.split_at(rest.len() - cut);
    let Ok(dt) = Datatype::parse(desc) else {
        return;
    };
    let _ = dt.to_string();
    let size = dt.size as usize;
    if size == 0 || payload.len() < size {
        return;
    }
    let whole = payload.len() - payload.len() % size;
    // Variable-length data resolves against the payload as a global heap.
    let resolver = GlobalHeapResolver::new(payload, Sizes::DEFAULT);
    let ctx = DecodeContext::with_resolver(Sizes::DEFAULT, &resolver);
    if let Ok(v) = decode_all::<Value>(&dt, &payload[..whole], &ctx) {
        for x in v.iter().take(8) {
            let _ = x.to_string();
        }
    }
    let _ = decode_all::<f64>(&dt, &payload[..whole], &ctx);
    let _ = decode_all::<String>(&dt, &payload[..whole], &ctx);
}

/// Superblock discovery and parsing (also superblock extension lookups).
pub fn superblock(data: &[u8]) {
    let _ = Superblock::read(data);
}

/// A fractal heap; the last bytes are used as heap IDs.
pub fn fractal_heap(data: &[u8]) {
    let (p, rest) = split(data);
    let (addr, rest) = addr_split(rest);
    let sz = sizes(p);
    let Ok(heap) = FractalHeap::open(rest, addr, sz) else {
        return;
    };
    for idlen in [4usize, 7, 8, 12, 17] {
        if rest.len() >= idlen {
            let _ = heap.get(rest, &rest[rest.len() - idlen..]);
        }
    }
}

/// v1 B-trees (group and chunk).
pub fn btree_v1(data: &[u8]) {
    let (p, rest) = split(data);
    let (addr, rest) = addr_split(rest);
    let sz = sizes(p);
    let _ = group_btree_children(rest, addr, sz);
    let _ = chunk_btree_entries(rest, addr, sz, usize::from(p >> 4) + 1);
}

/// A v2 B-tree (every record type).
pub fn btree_v2(data: &[u8]) {
    let (p, rest) = split(data);
    let (q, rest) = split(rest);
    let ctx = RecordContext {
        heap_id_len: (q & 0x80 != 0).then_some(usize::from(q & 0x1f)),
        chunk_ndims: usize::from(q >> 5) + 1,
        chunk_size_len: usize::from(p >> 5) + 1,
    };
    let (addr, rest) = addr_split(rest);
    if let Ok(tree) = BTreeV2::open(rest, addr, sizes(p), ctx) {
        let _ = tree.records(rest);
    }
}

/// Fixed and extensible array chunk indexes.
pub fn chunk_index(data: &[u8]) {
    let (p, rest) = split(data);
    let (addr, rest) = addr_split(rest);
    let sz = sizes(p);
    let ctx = if p & 0x80 != 0 {
        ChunkIndexContext::filtered(sz)
    } else {
        ChunkIndexContext::unfiltered(sz)
    };
    let _ = FixedArray::read(rest, addr, &ctx);
    let _ = ExtensibleArray::read(rest, addr, &ctx);
}

/// Local heap, global heap collection, symbol table node and SOHM table.
pub fn heaps(data: &[u8]) {
    let (p, rest) = split(data);
    let (addr, rest) = addr_split(rest);
    let sz = sizes(p);
    if let Ok(h) = LocalHeap::read(rest, addr, sz) {
        let _ = h.get_string_lossy(u64::from(p));
    }
    if let Ok(c) = GlobalHeapCollection::read(rest, addr, sz) {
        let _ = c.get(1);
    }
    let _ = SymbolTableNode::read(rest, addr, sz);
    let _ = SohmTable::read(rest, addr, usize::from(p >> 4), sz);
}

/// One filter decode: filter selector, client data count, client data,
/// element size, then the chunk.
pub fn filters(data: &[u8]) {
    thread_local! {
        static REGISTRY: FilterRegistry = FilterRegistry::with_builtin();
    }
    let (sel, rest) = split(data);
    let (ncd, rest) = split(rest);
    let ncd = usize::from(ncd % 24);
    let cd_bytes = (ncd * 4).min(rest.len());
    let (cd_raw, rest) = rest.split_at(cd_bytes);
    let cd: Vec<u32> = cd_raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let (es, chunk) = split(rest);
    REGISTRY.with(|r| {
        let ids: Vec<u16> = r.ids().collect();
        let Some(&id) = ids.get(usize::from(sel) % ids.len().max(1)) else {
            return;
        };
        let Some(f) = r.get(id) else { return };
        let ctx = FilterContext::new(usize::from(es % 17))
            .with_max_output(4 << 20)
            .with_expected_size(Some(chunk.len().saturating_mul(4)));
        let _ = f.decode(&cd, chunk.to_vec(), &ctx);
    });
}
