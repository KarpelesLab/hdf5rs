//! Modifying existing files: [`FileWriter::open_rw`] and friends.
//!
//! An existing file is opened by loading the objects it needs into the
//! writer's object model on demand (when a path is resolved through them):
//! groups with their links, datasets with their storage description,
//! committed datatypes. Each such node keeps an [`Origin`]: its object
//! header as found and what was changed. Links and attributes that are not
//! touched are carried over byte for byte.
//!
//! [`FileWriter::finish`] then
//!
//! 1. writes new objects exactly like for a new file (past the old end of
//!    file), builds the chunk indexes of modified chunked datasets, the
//!    new dense link/attribute storage and symbol tables of modified
//!    groups, and the new continuation chunks of modified object headers
//!    (all in newly allocated space);
//! 2. updates the end-of-file address in the superblock;
//! 3. rewrites in place, last, the few structures whose address must not
//!    change: the first chunk of each modified object header, and the
//!    local heap header and root B-tree node of modified symbol-table
//!    groups;
//! 4. clears the "open for writing" flag of version 3 superblocks (set
//!    when the file was opened).
//!
//! See the [module documentation](super) for the guarantees.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::build::{self, Built};
use super::chunked::{ChunkedSpec, IndexKind};
use super::dense::{DenseKind, DensePlan};
use super::storage::{Allocator, Based, WriteAt};
use super::{
    DatasetSpec, FileWriter, LayoutSpec, LinkSpec, LinkTargetSpec, Node, NodeKind, Patched,
    WriteOptions,
};
use crate::checksum::metadata_checksum;
use crate::error::{Error, Result};
use crate::file::File;
use crate::filters::FilterRegistry;
use crate::format::btree_v1::{BTreeNode, Keys, NodeType, group_btree_children};
use crate::format::btree_v2::{BTreeV2, BTreeV2Header, Record, RecordContext, encode_btree_v2};
use crate::format::chunk_index::ChunkRecord;
use crate::format::fractal_heap::FractalHeap;
use crate::format::local_heap::LocalHeap;
use crate::format::messages::MessageType;
use crate::format::messages::attribute::Attribute as AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfo;
use crate::format::messages::btree_k::BtreeKMessage;
use crate::format::messages::datatype::Datatype;
use crate::format::messages::fill_value::{AllocTime, FillValue};
use crate::format::messages::group_info::GroupInfoMessage;
use crate::format::messages::layout::{ChunkIndex, DONT_FILTER_PARTIAL_BOUND_CHUNKS, DataLayout};
use crate::format::messages::link::{CharSet, LinkMessage, LinkTarget};
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::modification_time::ModificationTimeMessage;
use crate::format::messages::ref_count::RefCountMessage;
use crate::format::messages::symbol_table::SymbolTableMessage;
use crate::format::object_header::{ObjectHeader, ObjectTimes, RawMessage};
use crate::format::symbol_table_node::SymbolTableNode;
use crate::format::{ByteWriter, ScratchPad, Sizes, SymbolTableEntry, UNDEFINED_ADDRESS};
use crate::io::ReadAt;
use crate::object::{Object, ObjectKind};

/// Default link / attribute phase change values (maximum compact,
/// minimum dense).
const DEFAULT_PHASE_CHANGE: (u16, u16) = (8, 6);
/// Largest header message body kept compact before forcing dense storage.
const MAX_COMPACT_MESSAGE: usize = 64 * 1024 - 1;
/// Node size of creation-order index B-trees.
const ORDER_INDEX_NODE_SIZE: u32 = 512;
/// Superblock version 3 consistency flag: open for write access.
const SB_WRITE_ACCESS: u8 = 0x01;
/// Superblock version 3 consistency flag: open for SWMR write access.
const SB_SWMR_WRITE: u8 = 0x04;

/// File-level state of a file opened read-write.
#[derive(Debug)]
pub(crate) struct ExistingFile {
    /// Reader over the file as it was when opened (only structures that
    /// existed then are read through it).
    pub(crate) reader: File,
    /// Node of each loaded object, by object header address.
    pub(crate) by_addr: HashMap<u64, usize>,
    /// Targets of removed hard links (one entry per removed link).
    pub(crate) unlinked: Vec<u64>,
    /// The superblock as stored (patched when finishing).
    sb_bytes: Vec<u8>,
    /// In-place rewrites of existing structures have started.
    switching: bool,
}

/// How an existing group stores its links.
#[derive(Debug, Clone)]
pub(crate) enum GroupStorage {
    /// Old-style group: local heap of names + B-tree of symbol table nodes.
    SymbolTable(SymbolTableMessage),
    /// New-style group (Link Info / Link messages, possibly dense).
    Links {
        info: Option<LinkInfoMessage>,
        /// (maximum compact, minimum dense) link counts.
        phase: (u16, u16),
    },
}

/// A link that existed when the file was opened.
#[derive(Debug, Clone)]
pub(crate) struct StoredLink {
    /// The decoded link.
    pub(crate) msg: LinkMessage,
    /// Its link message encoding as stored (new-style groups).
    raw: Option<Vec<u8>>,
    /// Its symbol table entry as stored (old-style groups).
    entry: Option<SymbolTableEntry>,
}

impl StoredLink {
    /// The link message encoding.
    pub(crate) fn encoded(&self, sizes: Sizes) -> Result<Vec<u8>> {
        match &self.raw {
            Some(r) => Ok(r.clone()),
            None => self.msg.encode(sizes),
        }
    }

    /// Target address of a hard link.
    pub(crate) fn hard_address(&self) -> Option<u64> {
        match self.msg.target {
            LinkTarget::Hard { address } => Some(address),
            _ => None,
        }
    }
}

/// An attribute that existed when the file was opened.
#[derive(Debug, Clone)]
pub(crate) struct StoredAttr {
    name: String,
    /// Attribute message body as stored (a shared message record when
    /// `flags` has the shared bit).
    body: Vec<u8>,
    /// Header message flags.
    flags: u8,
    creation_order: Option<u32>,
}

/// On-disk state of an existing object and what changed.
#[derive(Debug, Clone)]
pub(crate) struct Origin {
    /// Object header address.
    pub(crate) addr: u64,
    header: Arc<ObjectHeader>,
    group: Option<GroupStorage>,
    /// Existing attributes still present (`None` until loaded).
    attrs: Option<Vec<StoredAttr>>,
    pub(crate) links_dirty: bool,
    pub(crate) attrs_dirty: bool,
    pub(crate) layout_dirty: bool,
    pub(crate) dataspace_dirty: bool,
    /// Chunk records of a chunked dataset have been loaded.
    chunks_loaded: bool,
}

/// Seconds since the UNIX epoch, saturated to 32 bits.
fn now() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u32::try_from(d.as_secs()).unwrap_or(u32::MAX))
        .unwrap_or(0)
}

/// Whether two datatypes describe the same values (encoding versions
/// aside).
pub(crate) fn same_type(a: &Datatype, b: &Datatype) -> bool {
    a == b || super::compat::downgrade(a) == super::compat::downgrade(b)
}

/// Position of the end-of-file address within a superblock.
fn eof_position(version: u8, sizes: Sizes) -> usize {
    let o = usize::from(sizes.offset);
    match version {
        0 => 24 + 2 * o,
        1 => 28 + 2 * o,
        _ => 12 + 2 * o,
    }
}

impl FileWriter<std::fs::File> {
    /// Open the existing file at `path` for modification.
    ///
    /// Objects are created, deleted and modified through the same handles
    /// as for a new file ([`root`](Self::root), [`group`](Self::group),
    /// [`dataset`](Self::dataset)); [`finish`](Self::finish) writes the
    /// changes. See the [module documentation](super#modifying-existing-files)
    /// for what is supported and the consistency guarantees.
    pub fn open_rw(path: impl AsRef<Path>) -> Result<Self> {
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        let reader = f.try_clone()?;
        FileWriter::open_rw_with(f, reader)
    }
}

impl FileWriter<Vec<u8>> {
    /// Modify a file held in memory; [`finish`](Self::finish) returns the
    /// modified bytes.
    pub fn open_rw_bytes(bytes: Vec<u8>) -> Result<Self> {
        let snapshot = bytes.clone();
        FileWriter::open_rw_with(bytes, snapshot)
    }
}

impl<S: WriteAt> FileWriter<S> {
    /// Modify the existing file held by `storage`. `reader` must present
    /// the same bytes (for example a second handle on the same file, or a
    /// copy); it is only used to read structures that exist when the file
    /// is opened, which are never overwritten before
    /// [`finish`](Self::finish). `storage` should support
    /// [read-back](WriteAt::supports_read_back) to modify chunked datasets.
    pub fn open_rw_with<R: ReadAt + Send + Sync + 'static>(storage: S, reader: R) -> Result<Self> {
        let reader = File::from_reader(reader)?;
        let sb = reader.superblock().clone();
        if sb.version > 3 {
            return Err(Error::unsupported(format!(
                "modifying files with superblock version {}",
                sb.version
            )));
        }
        let ok = |v: u8| matches!(v, 2 | 4 | 8);
        if !ok(sb.sizes.offset) || !ok(sb.sizes.length) {
            return Err(Error::unsupported(
                "modifying files whose offsets/lengths are not 2, 4 or 8 bytes",
            ));
        }
        if sb.driver_info.is_some() || sb.driver_info_address != UNDEFINED_ADDRESS {
            return Err(Error::unsupported(
                "modifying files that use a non-default file driver",
            ));
        }
        let flags = (sb.consistency_flags & 0xff) as u8;
        if sb.version == 3 && flags & (SB_WRITE_ACCESS | SB_SWMR_WRITE) != 0 {
            return Err(Error::unsupported(
                "the file is marked as open for writing (in use, or not closed cleanly)",
            ));
        }
        let sizes = sb.sizes;
        let mut group_k = (
            sb.group_leaf_k.unwrap_or(super::compat::GROUP_LEAF_K),
            sb.group_internal_k
                .unwrap_or(super::compat::GROUP_INTERNAL_K),
        );
        let mut chunk_k = sb
            .indexed_storage_k
            .unwrap_or(super::chunked::CHUNK_BTREE_K);
        if let Some(ext) = reader.superblock_extension()? {
            for m in &ext.messages {
                match m.msg_type {
                    MessageType::FILE_SPACE_INFO => check_file_space_info(&m.data)?,
                    MessageType::DRIVER_INFO => {
                        return Err(Error::unsupported(
                            "modifying files that use a non-default file driver",
                        ));
                    }
                    MessageType::BTREE_K => {
                        let k = BtreeKMessage::parse(&m.data, sizes)?;
                        group_k = (k.group_leaf_k, k.group_internal_k);
                        chunk_k = k.indexed_storage_internal_k;
                    }
                    _ => {}
                }
            }
        }
        if group_k.0 == 0 || group_k.1 == 0 || chunk_k == 0 {
            return Err(Error::invalid("B-tree K of zero"));
        }
        let sb_len = match sb.version {
            0 | 1 => eof_position(sb.version, sizes) + usize::from(sizes.offset),
            _ => 12 + 4 * usize::from(sizes.offset) + 4,
        };
        let sb_bytes = reader.store().read_vec_at(0, sb_len)?;
        let file_end = reader.store().size()?;
        // The stored end-of-file address is absolute (it includes any user
        // block), allocations are relative.
        let start = sb.eof_address.saturating_sub(sb.base()).max(file_end);
        let opts = WriteOptions {
            superblock_version: if sb.version < 2 { 0 } else { sb.version },
            sizes,
            track_times: false,
        };
        let root_addr = sb.root_object_header_address();
        let mut storage = Based::new(storage);
        storage.base = sb.base();
        let mut w = FileWriter {
            storage,
            opts,
            alloc: Allocator::new(start, sizes),
            heap: super::heap::HeapWriter::default(),
            nodes: Vec::new(),
            data_refs: Vec::new(),
            filters: FilterRegistry::with_builtin(),
            chunk_cache_bytes: super::chunked::DEFAULT_CACHE_BYTES,
            group_k,
            chunk_k,
            existing: Some(Box::new(ExistingFile {
                reader,
                by_addr: HashMap::new(),
                unlinked: Vec::new(),
                sb_bytes,
                switching: false,
            })),
        };
        let root = w.load_existing(root_addr)?;
        if root != 0
            || !matches!(
                w.nodes.first().map(|n| &n.kind),
                Some(NodeKind::Group { .. })
            )
        {
            return Err(Error::invalid("root object is not a group"));
        }
        // New objects record times when the root group does.
        if let Some(o) = w.nodes.first().and_then(|n| n.origin.as_ref()) {
            w.opts.track_times = o.header.times.is_some()
                || o.header
                    .find_message(MessageType::MODIFICATION_TIME)
                    .is_some();
        }
        if sb.version == 3 {
            // First action of a writer: mark the file as open for writing.
            w.write_superblock(None, flags | SB_WRITE_ACCESS)?;
        }
        Ok(w)
    }

    /// The file as it was when opened read-write (`None` for new files).
    /// Reading through it shows the metadata as it was before any change;
    /// raw data written in place to existing contiguous datasets may or
    /// may not be visible.
    pub fn original(&self) -> Option<&File> {
        self.existing.as_ref().map(|e| &e.reader)
    }

    /// Fail early if the dataspace of existing dataset `id` cannot be
    /// rewritten (it is a shared message).
    pub(crate) fn check_dataspace_writable(&self, id: usize) -> Result<()> {
        let shared = self
            .nodes
            .get(id)
            .and_then(|n| n.origin.as_ref())
            .and_then(|o| o.header.find_message(MessageType::DATASPACE))
            .is_some_and(RawMessage::is_shared);
        if shared {
            return Err(Error::unsupported(
                "resizing a dataset whose dataspace is a shared message",
            ));
        }
        Ok(())
    }

    /// Called when `finish` fails: if no existing structure was rewritten
    /// yet, the file still holds its old metadata, so the "open for
    /// writing" flag of a version 3 superblock is cleared again (best
    /// effort).
    pub(crate) fn abandon_edits(&mut self) {
        let Some(ex) = self.existing.as_ref() else {
            return;
        };
        if ex.switching || ex.sb_bytes.get(8).copied() != Some(3) {
            return;
        }
        let flags = ex.sb_bytes.get(11).copied().unwrap_or(0);
        let _ = self.write_superblock(None, flags & !(SB_WRITE_ACCESS | SB_SWMR_WRITE));
    }

    fn reader(&self) -> Result<File> {
        self.existing
            .as_ref()
            .map(|e| e.reader.clone())
            .ok_or_else(|| Error::invalid("not an existing file"))
    }

    /// Patch the end-of-file address (when given) and consistency flags of
    /// the stored superblock and write it back.
    fn write_superblock(&mut self, eof: Option<u64>, flags: u8) -> Result<()> {
        let sizes = self.opts.sizes;
        let Some(ex) = self.existing.as_mut() else {
            return Ok(());
        };
        let version = ex.sb_bytes.get(8).copied().unwrap_or(0);
        if let Some(eof) = eof {
            let pos = eof_position(version, sizes);
            let width = usize::from(sizes.offset);
            let bytes = eof.to_le_bytes();
            ex.sb_bytes
                .get_mut(pos..pos + width)
                .ok_or(Error::UnexpectedEof)?
                .copy_from_slice(bytes.get(..width).ok_or(Error::UnexpectedEof)?);
        }
        if version >= 2 {
            *ex.sb_bytes.get_mut(11).ok_or(Error::UnexpectedEof)? = flags;
            let end = ex.sb_bytes.len() - 4;
            let sum = metadata_checksum(ex.sb_bytes.get(..end).ok_or(Error::UnexpectedEof)?);
            ex.sb_bytes
                .get_mut(end..)
                .ok_or(Error::UnexpectedEof)?
                .copy_from_slice(&sum.to_le_bytes());
        }
        let bytes = ex.sb_bytes.clone();
        self.storage.write_all_at(0, &bytes)
    }

    /// The node of the existing object whose header is at `addr`, loading
    /// it if needed.
    pub(crate) fn load_existing(&mut self, addr: u64) -> Result<usize> {
        let reader = {
            let ex = self
                .existing
                .as_ref()
                .ok_or_else(|| Error::invalid("not an existing file"))?;
            if let Some(&id) = ex.by_addr.get(&addr) {
                return Ok(id);
            }
            ex.reader.clone()
        };
        let header = reader.header(addr)?;
        let (kind, group) = match ObjectKind::of_header(&header)? {
            ObjectKind::Group => {
                let (storage, links) = read_group(&reader, &header)?;
                (NodeKind::Group { links }, Some(storage))
            }
            ObjectKind::Dataset => {
                let Object::Dataset(ds) = reader.object_at(addr, "")? else {
                    return Err(Error::invalid("object is not a dataset"));
                };
                (NodeKind::Dataset(self.dataset_from(&ds)?), None)
            }
            ObjectKind::NamedDatatype => {
                let m = header
                    .find_message(MessageType::DATATYPE)
                    .ok_or_else(|| Error::invalid("committed datatype without a datatype"))?;
                let body = reader.message_body(m)?;
                (NodeKind::NamedDatatype(Datatype::parse(&body)?), None)
            }
        };
        let id = self.nodes.len();
        let mut node = Node::new(kind);
        node.origin = Some(Box::new(Origin {
            addr,
            header,
            group,
            attrs: None,
            links_dirty: false,
            attrs_dirty: false,
            layout_dirty: false,
            dataspace_dirty: false,
            chunks_loaded: false,
        }));
        self.nodes.push(node);
        if let Some(ex) = self.existing.as_mut() {
            ex.by_addr.insert(addr, id);
        }
        Ok(id)
    }

    /// The writer's model of an existing dataset.
    fn dataset_from(&self, ds: &crate::Dataset) -> Result<DatasetSpec> {
        let datatype = ds.datatype().clone();
        let esize = u64::from(datatype.size);
        let dataspace = ds.dataspace().clone();
        let alloc_time = ds.fill_value().map_or(AllocTime::Late, |f| f.alloc_time);
        let fill = FillValue::new_v3(alloc_time, ds.fill_value_bytes().map(<[u8]>::to_vec));
        let layout = if ds.external_files_message().is_some() {
            LayoutSpec::Foreign("writing datasets stored in external files".into())
        } else {
            match ds.layout() {
                DataLayout::Compact { data, .. } => LayoutSpec::Compact(Patched {
                    data: data.clone(),
                    refs: Vec::new(),
                }),
                DataLayout::Contiguous { address, size, .. } => {
                    let total = dataspace
                        .element_count()?
                        .checked_mul(esize)
                        .ok_or(Error::LimitExceeded("dataset size"))?;
                    LayoutSpec::Contiguous {
                        address: *address,
                        size: size.unwrap_or(total),
                    }
                }
                DataLayout::Virtual { .. } => {
                    LayoutSpec::Foreign("writing virtual datasets".into())
                }
                DataLayout::Chunked(c) => {
                    let kind = match c.index {
                        ChunkIndex::BTreeV1 { .. } => IndexKind::BTreeV1,
                        ChunkIndex::SingleChunk { .. } => IndexKind::Single,
                        ChunkIndex::Implicit { .. } => IndexKind::Implicit,
                        ChunkIndex::FixedArray { .. } => IndexKind::FixedArray,
                        ChunkIndex::ExtensibleArray { .. } => IndexKind::ExtensibleArray,
                        ChunkIndex::BTreeV2 { .. } => IndexKind::BTreeV2,
                    };
                    let pipeline = ds.filters().cloned().unwrap_or_default();
                    let mut spec = ChunkedSpec::new(
                        c.dims.clone(),
                        pipeline,
                        kind,
                        c.flags & DONT_FILTER_PARTIAL_BOUND_CHUNKS != 0,
                        alloc_time == AllocTime::Early || kind == IndexKind::Implicit,
                        datatype.size as usize,
                    )?;
                    spec.btree_k = self.chunk_k;
                    if let ChunkIndex::Implicit { address } = c.index {
                        spec.set_implicit_base(address);
                    }
                    LayoutSpec::Chunked(Box::new(spec))
                }
            }
        };
        Ok(DatasetSpec {
            datatype,
            committed: None,
            dataspace,
            layout,
            fill,
        })
    }

    pub(crate) fn origin_mut(&mut self, id: usize) -> Option<&mut Origin> {
        self.nodes.get_mut(id).and_then(|n| n.origin.as_deref_mut())
    }

    /// Load the chunk records of an existing chunked dataset (once).
    pub(crate) fn ensure_chunks(&mut self, id: usize) -> Result<()> {
        let Some(o) = self.nodes.get(id).and_then(|n| n.origin.as_ref()) else {
            return Ok(());
        };
        if o.chunks_loaded {
            return Ok(());
        }
        let addr = o.addr;
        let reader = self.reader()?;
        let Object::Dataset(ds) = reader.object_at(addr, "")? else {
            return Err(Error::invalid("object is not a dataset"));
        };
        let filtered = ds.filters().is_some_and(|p| !p.filters.is_empty());
        let chunks = ds.chunks()?;
        let d = self.dataset_spec_mut(id)?;
        if let LayoutSpec::Chunked(spec) = &mut d.layout {
            for c in chunks {
                let record = ChunkRecord {
                    address: c.address,
                    size: filtered.then_some(c.size),
                    filter_mask: c.filter_mask,
                };
                // Filtered chunks are never overwritten in place: the old
                // index stays valid until the new one is in place.
                let capacity = if filtered { 0 } else { c.size };
                spec.insert_stored(c.scaled, record, capacity);
            }
        }
        if let Some(o) = self.origin_mut(id) {
            o.chunks_loaded = true;
        }
        Ok(())
    }

    /// Load the attributes of an existing object (once).
    fn ensure_attrs(&mut self, id: usize) -> Result<()> {
        let Some(o) = self.nodes.get(id).and_then(|n| n.origin.as_ref()) else {
            return Ok(());
        };
        if o.attrs.is_some() {
            return Ok(());
        }
        let header = o.header.clone();
        let reader = self.reader()?;
        let attrs = read_attributes(&reader, &header)?;
        if let Some(o) = self.origin_mut(id) {
            o.attrs = Some(attrs);
        }
        Ok(())
    }

    /// Prepare for setting attribute `name` on node `id`: an existing
    /// attribute of that name is replaced.
    pub(crate) fn before_put_attr(&mut self, id: usize, name: &str) -> Result<()> {
        if self.nodes.get(id).is_some_and(|n| n.origin.is_some()) {
            self.ensure_attrs(id)?;
            if let Some(o) = self.origin_mut(id) {
                if let Some(a) = o.attrs.as_mut() {
                    a.retain(|a| a.name != name);
                }
                o.attrs_dirty = true;
            }
        }
        Ok(())
    }

    /// Delete attribute `name` of node `id`.
    pub(crate) fn delete_attr_on(&mut self, id: usize, name: &str) -> Result<()> {
        let mut found = false;
        if self.nodes.get(id).is_some_and(|n| n.origin.is_some()) {
            self.ensure_attrs(id)?;
            if let Some(o) = self.origin_mut(id)
                && let Some(a) = o.attrs.as_mut()
            {
                let before = a.len();
                a.retain(|a| a.name != name);
                if a.len() != before {
                    found = true;
                    o.attrs_dirty = true;
                }
            }
        }
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| Error::invalid("unknown object"))?;
        let before = node.attrs.len();
        node.attrs.retain(|a| a.name != name);
        found |= node.attrs.len() != before;
        if !found {
            return Err(Error::NotFound(format!("attribute {name:?}")));
        }
        Ok(())
    }

    /// Names of the attributes of node `id`.
    pub(crate) fn attr_names(&mut self, id: usize) -> Result<Vec<String>> {
        self.ensure_attrs(id)?;
        let node = self
            .nodes
            .get(id)
            .ok_or_else(|| Error::invalid("unknown object"))?;
        let mut names: Vec<String> = node
            .origin
            .as_ref()
            .and_then(|o| o.attrs.as_ref())
            .map(|a| a.iter().map(|a| a.name.clone()).collect())
            .unwrap_or_default();
        names.extend(node.attrs.iter().map(|a| a.name.clone()));
        names.sort();
        Ok(names)
    }

    /// Remove link `name` from group `parent`.
    pub(crate) fn delete_link(&mut self, parent: usize, name: &str) -> Result<()> {
        let Some(NodeKind::Group { links }) = self.nodes.get_mut(parent).map(|n| &mut n.kind)
        else {
            return Err(Error::WrongObjectType("link parent is not a group".into()));
        };
        let pos = links
            .iter()
            .position(|l| l.name == name)
            .ok_or_else(|| Error::NotFound(name.to_owned()))?;
        let link = links.remove(pos);
        if let LinkTargetSpec::Stored(s) = &link.target
            && let Some(addr) = s.hard_address()
            && let Some(ex) = self.existing.as_mut()
        {
            ex.unlinked.push(addr);
        }
        if let Some(o) = self.origin_mut(parent) {
            o.links_dirty = true;
        }
        Ok(())
    }

    /// Apply every change to existing objects and update the superblock
    /// (called by `finish` after new objects were written).
    pub(crate) fn apply_edits(&mut self, built: &Built) -> Result<()> {
        let mut delta: HashMap<usize, i64> = HashMap::new();
        for (id, node) in self.nodes.iter().enumerate() {
            if node.origin.is_some()
                && let Some(&c) = built.ref_counts.get(id)
                && c > 0
            {
                *delta.entry(id).or_default() += i64::from(c);
            }
        }
        let unlinked = self
            .existing
            .as_mut()
            .map(|e| std::mem::take(&mut e.unlinked))
            .unwrap_or_default();
        for addr in unlinked {
            let id = self.load_existing(addr)?;
            *delta.entry(id).or_default() -= 1;
        }
        // Phase 1: everything written to new space; in-place writes queued.
        let mut in_place: Vec<(u64, Vec<u8>)> = Vec::new();
        for id in 0..self.nodes.len() {
            if self.nodes.get(id).is_some_and(|n| n.origin.is_some()) {
                let d = delta.get(&id).copied().unwrap_or(0);
                self.apply_node(id, &built.addresses, d, &mut in_place)?;
            }
        }
        // Phase 2: the new end of file, then the pointer switches.
        if let Some(ex) = self.existing.as_mut() {
            ex.switching = true;
        }
        let eof = self
            .alloc
            .eoa()
            .checked_add(self.storage.base)
            .ok_or(Error::LimitExceeded("file size"))?;
        let flags = self
            .existing
            .as_ref()
            .and_then(|e| e.sb_bytes.get(11).copied())
            .unwrap_or(0);
        self.write_superblock(Some(eof), flags)?;
        for (addr, bytes) in in_place {
            self.storage.write_all_at(addr, &bytes)?;
        }
        // Phase 3: clear "open for writing" (version 3) as the last action.
        let version = self
            .existing
            .as_ref()
            .and_then(|e| e.sb_bytes.get(8).copied())
            .unwrap_or(0);
        if version == 3 {
            self.write_superblock(None, flags & !(SB_WRITE_ACCESS | SB_SWMR_WRITE))?;
        }
        Ok(())
    }

    fn apply_node(
        &mut self,
        id: usize,
        addrs: &[u64],
        rc_delta: i64,
        in_place: &mut Vec<(u64, Vec<u8>)>,
    ) -> Result<()> {
        let Some(origin) = self
            .nodes
            .get(id)
            .and_then(|n| n.origin.as_deref())
            .cloned()
        else {
            return Ok(());
        };
        if !(origin.links_dirty
            || origin.attrs_dirty
            || origin.layout_dirty
            || origin.dataspace_dirty
            || rc_delta != 0)
        {
            return Ok(());
        }
        let sizes = self.opts.sizes;
        let header = origin.header.clone();
        let mut msgs = header.messages.clone();
        let mut changed = false;
        if origin.links_dirty {
            match &origin.group {
                Some(GroupStorage::SymbolTable(st)) => {
                    self.rebuild_symtab(id, st, addrs, in_place)?;
                }
                Some(GroupStorage::Links { info, phase }) => {
                    self.relink(id, info.as_ref(), *phase, &header, &mut msgs, addrs)?;
                    changed = true;
                }
                None => return Err(Error::invalid("links changed on a non-group")),
            }
        }
        if origin.attrs_dirty {
            self.reattr(id, &origin, &mut msgs, addrs)?;
            changed = true;
        }
        if origin.dataspace_dirty || origin.layout_dirty {
            let d = self.dataset_spec(id)?;
            if origin.dataspace_dirty {
                let body = d.dataspace.encode(sizes)?;
                replace_message(&mut msgs, MessageType::DATASPACE, body)?;
            }
            if origin.layout_dirty {
                let old_version = msgs
                    .iter()
                    .find(|m| m.msg_type == MessageType::LAYOUT)
                    .and_then(|m| m.data.first().copied())
                    .unwrap_or(3);
                let body = match &d.layout {
                    LayoutSpec::Compact(p) => DataLayout::Compact {
                        data: p.data.clone(),
                        dims: None,
                    }
                    .encode(old_version.clamp(3, 4), sizes)?,
                    LayoutSpec::Contiguous { address, size } => DataLayout::Contiguous {
                        address: *address,
                        size: Some(*size),
                        dims: None,
                    }
                    .encode(old_version.clamp(3, 4), sizes)?,
                    LayoutSpec::Chunked(c) => c
                        .layout(d.datatype.size)
                        .encode(c.layout_version(), sizes)?,
                    LayoutSpec::Foreign(why) => return Err(Error::unsupported(why.clone())),
                };
                replace_message(&mut msgs, MessageType::LAYOUT, body)?;
            }
            changed = true;
        }
        let mut ref_count = header.ref_count;
        if rc_delta != 0 {
            let rc = (i64::from(header.ref_count) + rc_delta).clamp(1, i64::from(u32::MAX));
            ref_count = rc as u32;
            if ref_count != header.ref_count {
                changed = true;
                if header.version == 2 {
                    msgs.retain(|m| m.msg_type != MessageType::REF_COUNT);
                    if ref_count > 1 {
                        msgs.push(RawMessage {
                            msg_type: MessageType::REF_COUNT,
                            flags: RawMessage::FLAG_DONT_SHARE,
                            creation_order: None,
                            data: RefCountMessage { ref_count }.encode()?,
                        });
                    }
                }
            }
        }
        if !changed {
            return Ok(());
        }
        let t = now();
        for m in &mut msgs {
            if m.msg_type == MessageType::MODIFICATION_TIME && !m.is_shared() {
                m.data = ModificationTimeMessage { seconds: t }.encode()?;
            }
        }
        let times = header.times.map(|old| ObjectTimes {
            modification: t,
            change: t,
            ..old
        });
        let image = layout_header(&header, &msgs, ref_count, times, sizes, &mut self.alloc)?;
        for (addr, bytes) in image.extra {
            self.storage.write_all_at(addr, &bytes)?;
        }
        in_place.push(image.first);
        Ok(())
    }

    /// Final link messages of a new-style group, (re)building dense
    /// storage when needed.
    fn relink(
        &mut self,
        id: usize,
        info: Option<&LinkInfoMessage>,
        phase: (u16, u16),
        header: &ObjectHeader,
        msgs: &mut Vec<RawMessage>,
        addrs: &[u64],
    ) -> Result<()> {
        let sizes = self.opts.sizes;
        let mut info = info
            .cloned()
            .unwrap_or_else(|| LinkInfoMessage::compact(false));
        let tracked = info.creation_order_tracked;
        let mut next = info.max_creation_index.unwrap_or(0);
        let Some(NodeKind::Group { links }) = self.nodes.get(id).map(|n| &n.kind) else {
            return Err(Error::invalid("not a group"));
        };
        // (name, encoded link message, creation order)
        let mut entries: Vec<(String, Vec<u8>, Option<u64>)> = Vec::with_capacity(links.len());
        for l in links {
            let target = match &l.target {
                LinkTargetSpec::Stored(s) => {
                    entries.push((l.name.clone(), s.encoded(sizes)?, s.msg.creation_order));
                    continue;
                }
                LinkTargetSpec::Hard(t) => LinkTarget::Hard {
                    address: *addrs
                        .get(t.0)
                        .ok_or_else(|| Error::invalid("link to an unknown object"))?,
                },
                LinkTargetSpec::Soft(p) => LinkTarget::Soft { path: p.clone() },
                LinkTargetSpec::External { file, path } => LinkTarget::External {
                    version: 0,
                    flags: 0,
                    file: file.clone(),
                    path: path.clone(),
                },
            };
            let order = tracked.then(|| {
                let o = next;
                next = next.saturating_add(1);
                o
            });
            let m = LinkMessage {
                creation_order: order,
                char_set: if l.name.is_ascii() {
                    CharSet::Ascii
                } else {
                    CharSet::Utf8
                },
                name: l.name.clone(),
                target,
            };
            entries.push((l.name.clone(), m.encode(sizes)?, order));
        }
        msgs.retain(|m| m.msg_type != MessageType::LINK);
        let n = entries.len();
        let big = entries.iter().any(|e| e.1.len() > MAX_COMPACT_MESSAGE);
        let dense = n > 0
            && (big
                || if info.is_dense() {
                    n >= usize::from(phase.1)
                } else {
                    n > usize::from(phase.0)
                });
        if dense {
            let lens: Vec<(String, usize)> =
                entries.iter().map(|e| (e.0.clone(), e.1.len())).collect();
            let mut plan = DensePlan::new(DenseKind::Links, &lens, sizes)?;
            plan.allocate(&mut self.alloc)?;
            let objs: Vec<Vec<u8>> = entries.iter().map(|e| e.1.clone()).collect();
            plan.write(&objs, &mut self.storage)?;
            info.fractal_heap_address = plan.heap_address();
            info.name_index_address = plan.name_index_address();
            if info.creation_order_indexed {
                let mut recs = Vec::with_capacity(n);
                for (i, e) in entries.iter().enumerate() {
                    recs.push(Record::LinkCreationOrder {
                        creation_order: e.2.unwrap_or(0),
                        heap_id: plan.heap_id(i)?,
                    });
                }
                recs.sort_by_key(|r| match r {
                    Record::LinkCreationOrder { creation_order, .. } => *creation_order,
                    _ => 0,
                });
                let addr = self.write_order_index(6, 8 + 7, &recs)?;
                info.creation_order_index_address = Some(addr);
            }
        } else {
            info.fractal_heap_address = UNDEFINED_ADDRESS;
            info.name_index_address = UNDEFINED_ADDRESS;
            if info.creation_order_indexed {
                info.creation_order_index_address = Some(UNDEFINED_ADDRESS);
            }
            let order = header.attribute_creation_order_tracked().then_some(0);
            for e in &entries {
                msgs.push(RawMessage {
                    msg_type: MessageType::LINK,
                    flags: 0,
                    creation_order: order,
                    data: e.1.clone(),
                });
            }
        }
        if tracked {
            info.max_creation_index = Some(next);
        }
        let body = info.encode(sizes)?;
        if msgs.iter().any(|m| m.msg_type == MessageType::LINK_INFO) {
            replace_message(msgs, MessageType::LINK_INFO, body)?;
        } else {
            msgs.insert(
                0,
                RawMessage {
                    msg_type: MessageType::LINK_INFO,
                    flags: 0,
                    creation_order: None,
                    data: body,
                },
            );
            if !msgs.iter().any(|m| m.msg_type == MessageType::GROUP_INFO) {
                msgs.insert(
                    1,
                    RawMessage {
                        msg_type: MessageType::GROUP_INFO,
                        flags: RawMessage::FLAG_CONSTANT,
                        creation_order: None,
                        data: GroupInfoMessage {
                            link_phase_change: None,
                            estimated_entries: None,
                        }
                        .encode()?,
                    },
                );
            }
        }
        Ok(())
    }

    /// Write a creation order index (v2 B-tree of `tree_type`) holding
    /// `records` (sorted); returns its header address.
    fn write_order_index(
        &mut self,
        tree_type: u8,
        record_size: u16,
        recs: &[Record],
    ) -> Result<u64> {
        let template = BTreeV2Header {
            tree_type,
            node_size: ORDER_INDEX_NODE_SIZE,
            record_size,
            depth: 0,
            split_percent: 100,
            merge_percent: 40,
            root_address: UNDEFINED_ADDRESS,
            root_num_records: 0,
            total_records: 0,
        };
        let sizes = self.opts.sizes;
        let alloc = &mut self.alloc;
        let mut alloc_fn = |n: u64| alloc.alloc(n);
        let enc = encode_btree_v2(&template, sizes, recs, 0, &mut alloc_fn)?;
        for b in &enc.blocks {
            self.storage.write_all_at(b.address, &b.bytes)?;
        }
        Ok(enc.address)
    }

    /// Final attribute messages of an existing object, (re)building dense
    /// storage when needed.
    fn reattr(
        &mut self,
        id: usize,
        origin: &Origin,
        msgs: &mut Vec<RawMessage>,
        addrs: &[u64],
    ) -> Result<()> {
        let sizes = self.opts.sizes;
        let compat = self.is_compat();
        let header = &origin.header;
        let reader = self.reader()?;
        let old_info = match header.find_message(MessageType::ATTRIBUTE_INFO) {
            Some(m) => Some(AttributeInfo::parse(&reader.message_body(m)?, sizes)?),
            None => None,
        };
        let tracked = header.attribute_creation_order_tracked();
        let indexed = header.attribute_creation_order_indexed();
        let stored = origin.attrs.clone().unwrap_or_default();
        let mut next: u32 = match old_info.as_ref().and_then(|i| i.max_creation_index) {
            Some(m) => u32::from(m),
            None => stored
                .iter()
                .filter_map(|a| a.creation_order)
                .max()
                .map_or(0, |m| m.saturating_add(1)),
        };
        // (name, body, flags, creation order)
        let mut entries: Vec<(String, Vec<u8>, u8, Option<u32>)> = Vec::new();
        for a in &stored {
            entries.push((a.name.clone(), a.body.clone(), a.flags, a.creation_order));
        }
        let node = self
            .nodes
            .get(id)
            .ok_or_else(|| Error::invalid("unknown object"))?;
        for a in &node.attrs {
            let body = build::encode_attribute(a, addrs, sizes, compat)?;
            let order = tracked.then(|| {
                let o = next;
                next = next.saturating_add(1);
                o
            });
            entries.push((a.name.clone(), body, 0, order));
        }
        msgs.retain(|m| m.msg_type != MessageType::ATTRIBUTE);
        let n = entries.len();
        let was_dense = old_info
            .as_ref()
            .is_some_and(|i| i.fractal_heap_address != UNDEFINED_ADDRESS);
        let can_dense = header.version == 2 || was_dense;
        let phase = header
            .attribute_phase_change
            .map_or(DEFAULT_PHASE_CHANGE, |p| (p.max_compact, p.min_dense));
        let big = entries.iter().any(|e| e.1.len() > MAX_COMPACT_MESSAGE);
        if big && !can_dense {
            return Err(Error::LimitExceeded(
                "attribute too large for a version 1 object header (64 KiB)",
            ));
        }
        let dense = can_dense
            && n > 0
            && (big
                || if was_dense {
                    n >= usize::from(phase.1)
                } else {
                    n > usize::from(phase.0)
                });
        let mut info = old_info.unwrap_or_else(|| {
            let mut i = AttributeInfo::compact(tracked);
            if indexed {
                i.creation_order_btree_address = Some(UNDEFINED_ADDRESS);
            }
            i
        });
        if dense {
            let lens: Vec<(String, usize)> =
                entries.iter().map(|e| (e.0.clone(), e.1.len())).collect();
            let mut plan = DensePlan::new(DenseKind::Attributes, &lens, sizes)?;
            plan.set_record_info(
                entries
                    .iter()
                    .map(|e| (e.2, e.3.unwrap_or(0xffff)))
                    .collect(),
            );
            plan.allocate(&mut self.alloc)?;
            let objs: Vec<Vec<u8>> = entries.iter().map(|e| e.1.clone()).collect();
            plan.write(&objs, &mut self.storage)?;
            info.fractal_heap_address = plan.heap_address();
            info.name_btree_address = plan.name_index_address();
            if info.creation_order_btree_address.is_some() {
                let mut recs = Vec::with_capacity(n);
                for (i, e) in entries.iter().enumerate() {
                    recs.push(Record::AttributeCreationOrder {
                        heap_id: plan.heap_id(i)?,
                        message_flags: e.2,
                        creation_order: e.3.unwrap_or(0),
                    });
                }
                recs.sort_by_key(|r| match r {
                    Record::AttributeCreationOrder { creation_order, .. } => *creation_order,
                    _ => 0,
                });
                let addr = self.write_order_index(9, 8 + 1 + 4, &recs)?;
                info.creation_order_btree_address = Some(addr);
            }
        } else {
            info.fractal_heap_address = UNDEFINED_ADDRESS;
            info.name_btree_address = UNDEFINED_ADDRESS;
            if info.creation_order_btree_address.is_some() {
                info.creation_order_btree_address = Some(UNDEFINED_ADDRESS);
            }
            for e in &entries {
                msgs.push(RawMessage {
                    msg_type: MessageType::ATTRIBUTE,
                    flags: e.2,
                    creation_order: if tracked {
                        Some(e.3.map_or(0, |o| o.min(u32::from(u16::MAX)) as u16))
                    } else {
                        None
                    },
                    data: e.1.clone(),
                });
            }
        }
        if tracked {
            info.max_creation_index = Some(u16::try_from(next).unwrap_or(u16::MAX));
        }
        if old_info.is_some() || dense {
            let body = info.encode(sizes)?;
            if old_info.is_some() {
                replace_message(msgs, MessageType::ATTRIBUTE_INFO, body)?;
            } else {
                msgs.push(RawMessage {
                    msg_type: MessageType::ATTRIBUTE_INFO,
                    flags: RawMessage::FLAG_DONT_SHARE,
                    creation_order: None,
                    data: body,
                });
            }
        }
        Ok(())
    }

    /// Rebuild the symbol table of an old-style group, keeping the
    /// addresses of its local heap header and B-tree root (cached in
    /// parent entries and the superblock). The heap's data segment is
    /// copied (existing names keep their offsets, new names are appended),
    /// so the old B-tree stays valid until the root node is replaced.
    fn rebuild_symtab(
        &mut self,
        id: usize,
        st: &SymbolTableMessage,
        addrs: &[u64],
        in_place: &mut Vec<(u64, Vec<u8>)>,
    ) -> Result<()> {
        let sizes = self.opts.sizes;
        let (leaf_k, internal_k) = self.group_k;
        let reader = self.reader()?;
        let heap = LocalHeap::read(reader.store().as_ref(), st.local_heap_address, sizes)?;
        let mut data = heap.data().to_vec();
        data.resize(data.len().next_multiple_of(8), 0);
        let push = |data: &mut Vec<u8>, s: &[u8]| -> Result<u64> {
            if s.contains(&0) {
                return Err(Error::invalid("name contains NUL"));
            }
            let off = data.len() as u64;
            data.extend_from_slice(s);
            data.push(0);
            data.resize(data.len().next_multiple_of(8), 0);
            Ok(off)
        };
        let empty = if data.first() == Some(&0) {
            0
        } else {
            push(&mut data, b"")?
        };
        let Some(NodeKind::Group { links }) = self.nodes.get(id).map(|n| &n.kind) else {
            return Err(Error::invalid("not a group"));
        };
        let mut entries: Vec<(String, SymbolTableEntry)> = Vec::with_capacity(links.len());
        for l in links {
            let entry = match &l.target {
                LinkTargetSpec::Stored(s) if s.entry.is_some() => {
                    s.entry.ok_or_else(|| Error::invalid("missing entry"))?
                }
                LinkTargetSpec::Hard(t) => SymbolTableEntry {
                    link_name_offset: push(&mut data, l.name.as_bytes())?,
                    object_header_address: *addrs
                        .get(t.0)
                        .ok_or_else(|| Error::invalid("link to an unknown object"))?,
                    scratch: ScratchPad::None,
                },
                LinkTargetSpec::Soft(p) => {
                    let name = push(&mut data, l.name.as_bytes())?;
                    let value = push(&mut data, p.as_bytes())?;
                    SymbolTableEntry {
                        link_name_offset: name,
                        object_header_address: UNDEFINED_ADDRESS,
                        scratch: ScratchPad::SoftLink {
                            link_value_offset: u32::try_from(value)
                                .map_err(|_| Error::LimitExceeded("local heap too large"))?,
                        },
                    }
                }
                LinkTargetSpec::Stored(s) => match &s.msg.target {
                    LinkTarget::Hard { address } => SymbolTableEntry {
                        link_name_offset: push(&mut data, l.name.as_bytes())?,
                        object_header_address: *address,
                        scratch: ScratchPad::None,
                    },
                    _ => {
                        return Err(Error::unsupported("this link type in a symbol-table group"));
                    }
                },
                LinkTargetSpec::External { .. } => {
                    return Err(Error::unsupported("external links in symbol-table groups"));
                }
            };
            entries.push((l.name.clone(), entry));
        }
        entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        // Trailing free block (next = 1 ends the list).
        let free_offset = data.len() as u64;
        let mut free = ByteWriter::with_sizes(sizes);
        free.write_length(1)?;
        free.write_length(2 * u64::from(sizes.length))?;
        data.extend_from_slice(free.as_slice());
        let data_addr = self.alloc.alloc(data.len() as u64)?;
        self.storage.write_all_at(data_addr, &data)?;
        // Symbol table nodes.
        let per_snod = 2 * usize::from(leaf_k);
        let mut below: Vec<(u64, u64)> = Vec::new(); // (left key, right key)
        let mut below_addrs: Vec<u64> = Vec::new();
        let mut left = empty;
        for chunk in entries.chunks(per_snod) {
            let node = SymbolTableNode {
                version: 1,
                entries: chunk.iter().map(|e| e.1).collect(),
            };
            let bytes = node.encode(sizes, leaf_k)?;
            let addr = self.alloc.alloc(bytes.len() as u64)?;
            self.storage.write_all_at(addr, &bytes)?;
            let right = chunk.last().map_or(empty, |e| e.1.link_name_offset);
            below.push((left, right));
            below_addrs.push(addr);
            left = right;
        }
        // B-tree levels; the single top node replaces the old root.
        let fanout = 2 * usize::from(internal_k);
        let node_size = BTreeNode::group_node_size(sizes, internal_k) as u64;
        let mut level: u8 = 0;
        loop {
            let count = below.len().div_ceil(fanout).max(1);
            let top = count == 1;
            let addrs_lvl: Vec<u64> = if top {
                vec![st.btree_address]
            } else {
                (0..count)
                    .map(|_| self.alloc.alloc(node_size))
                    .collect::<Result<_>>()?
            };
            let mut next = Vec::with_capacity(count);
            for (j, &addr) in addrs_lvl.iter().enumerate() {
                let range = j * fanout..((j + 1) * fanout).min(below.len());
                let kids = below.get(range.clone()).unwrap_or(&[]);
                let mut keys = Vec::with_capacity(kids.len() + 1);
                keys.push(kids.first().map_or(empty, |k| k.0));
                keys.extend(kids.iter().map(|k| k.1));
                let node = BTreeNode {
                    node_type: NodeType::Group,
                    level,
                    left_sibling: j
                        .checked_sub(1)
                        .and_then(|i| addrs_lvl.get(i).copied())
                        .unwrap_or(UNDEFINED_ADDRESS),
                    right_sibling: addrs_lvl.get(j + 1).copied().unwrap_or(UNDEFINED_ADDRESS),
                    keys: Keys::Group(keys.clone()),
                    children: below_addrs.get(range).unwrap_or(&[]).to_vec(),
                };
                let bytes = node.encode_group(sizes, internal_k)?;
                if top {
                    in_place.push((addr, bytes));
                } else {
                    self.storage.write_all_at(addr, &bytes)?;
                }
                let first = keys.first().copied().unwrap_or(empty);
                next.push((first, keys.last().copied().unwrap_or(empty)));
            }
            if top {
                break;
            }
            below = next;
            below_addrs = addrs_lvl;
            level = level
                .checked_add(1)
                .ok_or(Error::LimitExceeded("group B-tree too deep"))?;
        }
        // The heap header switch comes before the root switch: the old
        // tree is valid with the new heap.
        let hdr = LocalHeap::encode_header(data.len() as u64, Some(free_offset), data_addr, sizes)?;
        let root = in_place.pop();
        in_place.push((st.local_heap_address, hdr));
        if let Some(r) = root {
            in_place.push(r);
        }
        Ok(())
    }
}

/// Refuse files whose free space is tracked persistently (or paged): their
/// free-space managers would not know about the space this writer uses.
fn check_file_space_info(data: &[u8]) -> Result<()> {
    let version = data.first().copied().unwrap_or(0);
    let strategy = data.get(1).copied().unwrap_or(0);
    let refuse = match version {
        0 => strategy == 1,
        _ => strategy == 1 || data.get(2).copied().unwrap_or(0) != 0,
    };
    if refuse {
        return Err(Error::unsupported(
            "modifying files with persistent free-space managers or paged allocation",
        ));
    }
    Ok(())
}

/// Replace the body of the first message of `msg_type`.
fn replace_message(msgs: &mut [RawMessage], msg_type: u16, body: Vec<u8>) -> Result<()> {
    let m = msgs
        .iter_mut()
        .find(|m| m.msg_type == msg_type)
        .ok_or_else(|| {
            Error::invalid(format!(
                "no {} message to update",
                MessageType::name(msg_type)
            ))
        })?;
    if m.is_shared() {
        return Err(Error::unsupported(format!(
            "modifying a shared {} message",
            MessageType::name(msg_type)
        )));
    }
    m.data = body;
    Ok(())
}

/// Read the links of an existing group, keeping their stored encodings.
fn read_group(file: &File, header: &ObjectHeader) -> Result<(GroupStorage, Vec<LinkSpec>)> {
    let sizes = file.sizes();
    let store = file.store().as_ref();
    let link = |msg: LinkMessage, raw: Option<Vec<u8>>, entry: Option<SymbolTableEntry>| LinkSpec {
        name: msg.name.clone(),
        target: LinkTargetSpec::Stored(Box::new(StoredLink { msg, raw, entry })),
    };
    if let Some(m) = header.find_message(MessageType::SYMBOL_TABLE) {
        let st = SymbolTableMessage::parse(&m.data, sizes)?;
        let heap = LocalHeap::read(store, st.local_heap_address, sizes)?;
        let mut links = Vec::new();
        for snod in group_btree_children(store, st.btree_address, sizes)? {
            let node = SymbolTableNode::read(store, snod, sizes)?;
            for entry in node.entries {
                let name = heap.get_str(entry.link_name_offset)?.to_owned();
                let target = match entry.scratch {
                    ScratchPad::SoftLink { link_value_offset } => LinkTarget::Soft {
                        path: heap.get_str(u64::from(link_value_offset))?.to_owned(),
                    },
                    _ => LinkTarget::Hard {
                        address: entry.object_header_address,
                    },
                };
                let msg = LinkMessage {
                    creation_order: None,
                    char_set: if name.is_ascii() {
                        CharSet::Ascii
                    } else {
                        CharSet::Utf8
                    },
                    name,
                    target,
                };
                links.push(link(msg, None, Some(entry)));
            }
        }
        return Ok((GroupStorage::SymbolTable(st), links));
    }
    let info = match header.find_message(MessageType::LINK_INFO) {
        Some(m) => Some(LinkInfoMessage::parse(&m.data, sizes)?),
        None => None,
    };
    let phase = match header.find_message(MessageType::GROUP_INFO) {
        Some(m) => GroupInfoMessage::parse(&file.message_body(m)?, sizes)?
            .link_phase_change
            .unwrap_or(DEFAULT_PHASE_CHANGE),
        None => DEFAULT_PHASE_CHANGE,
    };
    let mut links = Vec::new();
    for m in header.messages_of_type(MessageType::LINK) {
        let msg = LinkMessage::parse(&m.data, sizes)?;
        links.push(link(msg, Some(m.data.clone()), None));
    }
    if let Some(i) = info.as_ref().filter(|i| i.is_dense()) {
        let heap = FractalHeap::open(store, i.fractal_heap_address, sizes)?;
        let ctx = RecordContext {
            heap_id_len: Some(usize::from(heap.header.heap_id_len)),
            ..RecordContext::default()
        };
        let tree = BTreeV2::open(store, i.name_index_address, sizes, ctx)?;
        for rec in tree.records(store)? {
            if let Record::LinkName { heap_id, .. } = rec {
                let raw = heap.get(store, &heap_id)?;
                let msg = LinkMessage::parse(&raw, sizes)?;
                links.push(link(msg, Some(raw), None));
            }
        }
    }
    Ok((GroupStorage::Links { info, phase }, links))
}

/// Read the attributes of an existing object as stored.
fn read_attributes(file: &File, header: &ObjectHeader) -> Result<Vec<StoredAttr>> {
    let sizes = file.sizes();
    let store = file.store().as_ref();
    let name_of = |body: &[u8], flags: u8| -> Result<String> {
        let resolved;
        let b = if flags & RawMessage::FLAG_SHARED != 0 {
            resolved = file.shared_message_body(MessageType::ATTRIBUTE, body)?;
            &resolved[..]
        } else {
            body
        };
        Ok(AttributeMessage::parse(b, sizes)?.name)
    };
    let mut out = Vec::new();
    for m in header.messages_of_type(MessageType::ATTRIBUTE) {
        out.push(StoredAttr {
            name: name_of(&m.data, m.flags)?,
            body: m.data.clone(),
            flags: m.flags,
            creation_order: m.creation_order.map(u32::from),
        });
    }
    if let Some(m) = header.find_message(MessageType::ATTRIBUTE_INFO) {
        let info = AttributeInfo::parse(&file.message_body(m)?, sizes)?;
        if info.fractal_heap_address != UNDEFINED_ADDRESS
            && info.name_btree_address != UNDEFINED_ADDRESS
        {
            let heap = FractalHeap::open(store, info.fractal_heap_address, sizes)?;
            let ctx = RecordContext {
                heap_id_len: Some(usize::from(heap.header.heap_id_len)),
                ..RecordContext::default()
            };
            let tree = BTreeV2::open(store, info.name_btree_address, sizes, ctx)?;
            for rec in tree.records(store)? {
                if let Record::AttributeName {
                    heap_id,
                    message_flags,
                    creation_order,
                    ..
                } = rec
                {
                    let body = heap.get(store, &heap_id)?;
                    out.push(StoredAttr {
                        name: name_of(&body, message_flags)?,
                        body,
                        flags: message_flags,
                        creation_order: Some(creation_order),
                    });
                }
            }
        }
    }
    Ok(out)
}

/// A laid-out object header: the first chunk (rewritten in place) and new
/// continuation chunks.
struct HeaderImage {
    first: (u64, Vec<u8>),
    extra: Vec<(u64, Vec<u8>)>,
}

/// Message types this crate knows (for the "unknown message" flags).
fn known_type(t: u16) -> bool {
    t <= MessageType::FILE_SPACE_INFO
}

/// Lay out `msgs` in the first chunk of `header` (whose size is fixed),
/// spilling into one new continuation chunk when they do not fit. Old
/// continuation chunks are abandoned.
fn layout_header(
    header: &ObjectHeader,
    msgs: &[RawMessage],
    ref_count: u32,
    times: Option<ObjectTimes>,
    sizes: Sizes,
    alloc: &mut Allocator,
) -> Result<HeaderImage> {
    let v1 = header.version == 1;
    let tracked = header.attribute_creation_order_tracked();
    let first = *header
        .chunks
        .first()
        .ok_or_else(|| Error::invalid("object header without chunks"))?;
    let mut msgs: Vec<RawMessage> = msgs.to_vec();
    for m in &mut msgs {
        if !known_type(m.msg_type) {
            if m.flags & RawMessage::FLAG_FAIL_IF_UNKNOWN_WRITE != 0 {
                return Err(Error::unsupported(format!(
                    "modifying an object with unknown header message type {:#x}",
                    m.msg_type
                )));
            }
            if m.flags & RawMessage::FLAG_MARK_IF_UNKNOWN != 0 {
                m.flags |= RawMessage::FLAG_WAS_UNKNOWN;
            }
        }
    }
    let msg_prefix: u64 = if v1 {
        8
    } else if tracked {
        6
    } else {
        4
    };
    let size_of = |len: usize| -> u64 {
        if v1 {
            8 + (len as u64).next_multiple_of(8)
        } else {
            msg_prefix + len as u64
        }
    };
    let cont_body = usize::from(sizes.offset) + usize::from(sizes.length);
    let cont_size = size_of(cont_body);
    // Prefix and capacity of the first chunk.
    let (prefix_len, tail) = if v1 {
        (16u64, 0u64)
    } else {
        let width = 1u64 << (header.flags & 0x03);
        let p = 6
            + if header.times.is_some() { 16 } else { 0 }
            + if header.attribute_phase_change.is_some() {
                4
            } else {
                0
            }
            + width;
        (p, 4)
    };
    let cap0 = first
        .size
        .checked_sub(prefix_len + tail)
        .ok_or_else(|| Error::invalid("object header chunk too small"))?;
    let mut used = 0u64;
    let mut split = msgs.len();
    for (i, m) in msgs.iter().enumerate() {
        let s = size_of(m.data.len());
        let reserve = if i + 1 < msgs.len() { cont_size } else { 0 };
        if used + s + reserve > cap0 {
            split = i;
            break;
        }
        used += s;
    }
    let mut extra = Vec::new();
    let mut first_msgs: Vec<RawMessage> = msgs.get(..split).unwrap_or(&[]).to_vec();
    let mut count_extra = 0usize;
    if split < msgs.len() {
        if used + cont_size > cap0 {
            return Err(Error::unsupported(
                "object header too small to hold a continuation message",
            ));
        }
        let rest = msgs.get(split..).unwrap_or(&[]);
        let body_len: u64 = rest.iter().map(|m| size_of(m.data.len())).sum();
        let chunk_len = if v1 { body_len } else { 4 + body_len + 4 };
        let addr = alloc.alloc(chunk_len)?;
        let mut w = ByteWriter::with_sizes(sizes);
        if !v1 {
            w.bytes(b"OCHK");
        }
        for m in rest {
            encode_message(&mut w, m, v1, tracked)?;
        }
        if !v1 {
            let sum = metadata_checksum(w.as_slice());
            w.u32(sum);
        }
        if w.len() as u64 != chunk_len {
            return Err(Error::invalid("continuation chunk size mismatch"));
        }
        count_extra = rest.len();
        extra.push((addr, w.into_vec()));
        let mut cont = ByteWriter::with_sizes(sizes);
        cont.write_offset(addr)?;
        cont.write_length(chunk_len)?;
        first_msgs.push(RawMessage {
            msg_type: MessageType::CONTINUATION,
            flags: 0,
            creation_order: None,
            data: cont.into_vec(),
        });
        used += cont_size;
    }
    // Fill the rest of the first chunk with NIL messages.
    let mut left = cap0 - used;
    while left >= msg_prefix {
        let body = (left - msg_prefix).min(if v1 { 0xfff8 } else { 0xffff });
        first_msgs.push(RawMessage {
            msg_type: MessageType::NIL,
            flags: 0,
            creation_order: None,
            data: vec![0; body as usize],
        });
        left -= msg_prefix + body;
    }
    let mut w = ByteWriter::with_sizes(sizes);
    if v1 {
        let count = u16::try_from(first_msgs.len() + count_extra)
            .map_err(|_| Error::LimitExceeded("too many header messages"))?;
        let size = u32::try_from(cap0).map_err(|_| Error::LimitExceeded("object header size"))?;
        w.u8(1).u8(0).u16(count).u32(ref_count).u32(size).zeros(4);
    } else {
        w.bytes(b"OHDR").u8(2).u8(header.flags);
        if let Some(t) = times {
            w.u32(t.access)
                .u32(t.modification)
                .u32(t.change)
                .u32(t.birth);
        }
        if let Some(p) = header.attribute_phase_change {
            w.u16(p.max_compact).u16(p.min_dense);
        }
        w.write_uint(cap0, 1 << (header.flags & 0x03))?;
    }
    for m in &first_msgs {
        encode_message(&mut w, m, v1, tracked)?;
    }
    if w.len() as u64 > prefix_len + cap0 {
        return Err(Error::invalid("object header overflow"));
    }
    w.zeros((prefix_len + cap0) as usize - w.len());
    if !v1 {
        let sum = metadata_checksum(w.as_slice());
        w.u32(sum);
    }
    if w.len() as u64 != first.size {
        return Err(Error::invalid("object header size changed"));
    }
    Ok(HeaderImage {
        first: (first.address, w.into_vec()),
        extra,
    })
}

fn encode_message(w: &mut ByteWriter, m: &RawMessage, v1: bool, tracked: bool) -> Result<()> {
    if v1 {
        let padded = m.data.len().next_multiple_of(8);
        let size = u16::try_from(padded)
            .map_err(|_| Error::LimitExceeded("header message larger than 65535 bytes"))?;
        w.u16(m.msg_type).u16(size).u8(m.flags).zeros(3);
        w.bytes(&m.data).zeros(padded - m.data.len());
    } else {
        let ty = u8::try_from(m.msg_type)
            .map_err(|_| Error::invalid("message type does not fit a version 2 header"))?;
        let size = u16::try_from(m.data.len())
            .map_err(|_| Error::LimitExceeded("header message larger than 65535 bytes"))?;
        w.u8(ty).u16(size).u8(m.flags);
        if tracked {
            w.u16(m.creation_order.unwrap_or(0));
        }
        w.bytes(&m.data);
    }
    Ok(())
}
