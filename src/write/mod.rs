//! Creating new HDF5 files.
//!
//! ```
//! use hdf5rs::write::FileWriter;
//!
//! let mut f = FileWriter::in_memory();
//! let mut root = f.root();
//! root.set_attr("title", "example")?;
//! let mut g = root.create_group("measurements")?;
//! let mut ds = g
//!     .new_dataset::<f64>("temperature")
//!     .shape(&[2, 3])
//!     .write(&[20.5, 21.0, 21.5, 22.0, 22.5, 23.0])?;
//! ds.set_attr("units", "celsius")?;
//! g.new_dataset::<i32>("counts").write(&[1, 2, 3])?;
//! g.link_soft("latest", "/measurements/temperature")?;
//! let bytes = f.finish()?;
//!
//! let file = hdf5rs::File::from_bytes(bytes)?;
//! assert_eq!(file.dataset("/measurements/temperature")?.shape(), &[2, 3]);
//! # Ok::<(), hdf5rs::Error>(())
//! ```
//!
//! # Design
//!
//! By default files use the "latest" format: a version 2 or 3 superblock,
//! version 2 (`OHDR`) object headers, new-style groups (Link Info / Group
//! Info / Link messages), version 3 attribute messages with UTF-8 names,
//! version 3 fill value messages and version 3 layout messages.
//!
//! Storage is written through the [`WriteAt`] trait with an append-only
//! allocator:
//!
//! - raw data (contiguous datasets, complete chunks) and global heap
//!   collections (variable-length data) are written as soon as they are
//!   produced; chunk indexes are built by [`FileWriter::finish`];
//! - object headers are kept in memory as a model of the object graph and
//!   serialized by [`FileWriter::finish`]: every header is encoded once to
//!   learn its size (sizes do not depend on address values), addresses are
//!   allocated, then headers are encoded for real, which supports
//!   arbitrary hard-link graphs; object references in raw data are patched
//!   once addresses are known;
//! - the superblock (with the end-of-file address) is written last, at
//!   offset 0, whose space is reserved from the start.
//!
//! Space is never reused and the file is complete only after `finish`;
//! dropping a writer leaves an invalid file. In-place modification of
//! existing files can later build on the same model by loading headers
//! into it and allocating new space past the existing end of file.
//!
//! # Compatibility format
//!
//! With [`WriteOptions::superblock_version`] set to 0 the file uses the
//! oldest encodings, like `libver="earliest"`: a version 0 superblock,
//! version 1 object headers, old-style groups (symbol table: local heap
//! of names + v1 B-tree of symbol table nodes, K = 4 / 16), version 1
//! dataspaces and attributes, version 1/2 datatypes and version 2 fill
//! values. External links, dense attribute storage (so attributes are
//! limited to about 64 KiB) and null dataspaces are not available there.
//!
//! # Chunked datasets
//!
//! Asking for chunks ([`DatasetBuilder::chunk`]), filters
//! ([`deflate`](DatasetBuilder::deflate), [`shuffle`](DatasetBuilder::shuffle),
//! [`fletcher32`](DatasetBuilder::fletcher32), [`lzf`](DatasetBuilder::lzf),
//! [`nbit`](DatasetBuilder::nbit), [`scale_offset`](DatasetBuilder::scale_offset),
//! or any registered filter with [`filter`](DatasetBuilder::filter)) or
//! maximum dimensions ([`max_shape`](DatasetBuilder::max_shape)) makes a
//! dataset chunked; chunk dimensions are guessed (about 64 KiB per chunk)
//! when not given. Edge chunks are stored at full size, padded with the fill
//! value. Chunked datasets can be grown or shrunk with
//! [`DatasetWriter::resize`], extended with [`DatasetWriter::append`] and
//! partially rewritten with [`DatasetWriter::write_selection`] until the
//! file is finished; chunks that are only partly written are kept in memory
//! (or read back from storage) and merged.
//!
//! ```
//! use hdf5rs::write::FileWriter;
//!
//! let mut f = FileWriter::in_memory();
//! let mut root = f.root();
//! let mut log = root
//!     .new_dataset::<f64>("log")
//!     .shape(&[0])
//!     .max_shape(&[None])
//!     .chunk(&[1024])
//!     .shuffle()
//!     .deflate(6)
//!     .create()?;
//! log.append(&[1.0, 2.0, 3.0])?;
//! log.append(&[4.0])?;
//! let file = hdf5rs::File::from_bytes(f.finish()?)?;
//! assert_eq!(file.dataset("/log")?.read::<f64>()?, [1.0, 2.0, 3.0, 4.0]);
//! # Ok::<(), hdf5rs::Error>(())
//! ```
//!
//! The chunk index follows what the reference library picks for the file
//! format in use:
//!
//! - superblock version 3 (HDF5 1.10+): version 4 layout messages with a
//!   single-chunk index when one chunk covers the maximum extent, an
//!   implicit index for unfiltered fixed-size datasets with
//!   [`AllocTime::Early`], a fixed array (paged beyond 1024 chunks) for
//!   other fixed-size datasets, an extensible array for one unlimited
//!   dimension and a version 2 B-tree for several (with 4-byte lengths,
//!   datasets with unlimited dimensions keep the version 1 B-tree, which is
//!   the only index the reference library can open there);
//! - superblock versions 2 and 0: version 3 layout messages with a
//!   version 1 B-tree (K = 32), readable by HDF5 1.8.
//!
//! Filter pipelines are version 2 messages (version 1 in the
//! compatibility format). An optional filter that fails on a chunk (e.g.
//! LZF on incompressible data) is skipped for that chunk and recorded in
//! its filter mask. Datasets of reference types cannot be chunked yet.
//!
//! # Groups and attributes
//!
//! In the latest format, following the reference library's default phase-change values, up to 8
//! links / attributes are stored compactly as header messages; beyond
//! that (or when an attribute message exceeds 64 KiB) they are stored
//! densely: in a fractal heap indexed by a version 2 B-tree on the name
//! hash (attributes larger than 4 KiB become "huge" heap objects). Link
//! and attribute creation order are not tracked.

mod build;
mod chunked;
mod compat;
mod dense;
mod heap;
mod storage;
mod value;

pub use storage::{SeekWriter, WriteAt};
pub use value::{AttrValue, EncodeContext, FixedAscii, FixedUtf8, H5Write, ObjectRef};

use std::marker::PhantomData;
use std::path::Path;

use crate::error::{Error, Result};
use crate::filters::{Filter, FilterRegistry};
use crate::format::Sizes;
use crate::format::UNDEFINED_ADDRESS;
use crate::format::messages::dataspace::{Dataspace, UNLIMITED};
use crate::format::messages::datatype::{Datatype, DatatypeClass};
pub use crate::format::messages::fill_value::AllocTime;
use crate::format::messages::fill_value::FillValue;
use crate::format::messages::filter_pipeline::{FilterInfo, FilterPipeline};
use crate::format::superblock::Superblock;
use crate::read::{Resolved, Selection, block_runs, for_each_run, row_major_strides};
use chunked::{ChunkedSpec, Env, IndexKind};
use heap::HeapWriter;
use storage::Allocator;

/// Largest compact dataset payload accepted (the layout message must fit a
/// 16-bit header message size together with its own fields).
pub const MAX_COMPACT_DATA: usize = 65535 - 64;

/// Options for a new file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteOptions {
    /// Superblock version: 2 (default; readable by HDF5 1.8+), 3
    /// (HDF5 1.10+, what `libver="latest"` produces) or 0 for the
    /// compatibility format (see the module documentation).
    pub superblock_version: u8,
    /// Offset/length widths (2, 4 or 8 bytes each; default 8/8).
    pub sizes: Sizes,
    /// Store access/modification/change/birth times in object headers.
    pub track_times: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        WriteOptions {
            superblock_version: 2,
            sizes: Sizes::DEFAULT,
            track_times: false,
        }
    }
}

/// Identifies an object (group, dataset or committed datatype) created by
/// a [`FileWriter`]; used for hard links and object references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId(pub(crate) usize);

/// How a new dataset stores its raw data. Chunked storage is selected by
/// [`DatasetBuilder::chunk`], filters or
/// [`DatasetBuilder::max_shape`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// One contiguous block in the file (the default).
    #[default]
    Contiguous,
    /// Inside the object header (at most [`MAX_COMPACT_DATA`] bytes).
    Compact,
}

/// Bytes with pending object-reference patches (positions within `data`).
#[derive(Debug, Clone, Default)]
pub(crate) struct Patched {
    pub(crate) data: Vec<u8>,
    pub(crate) refs: Vec<(usize, ObjectId)>,
}

#[derive(Debug, Clone)]
pub(crate) enum LinkTargetSpec {
    Hard(ObjectId),
    Soft(String),
    External { file: String, path: String },
}

#[derive(Debug, Clone)]
pub(crate) struct LinkSpec {
    pub(crate) name: String,
    pub(crate) target: LinkTargetSpec,
}

#[derive(Debug, Clone)]
pub(crate) struct AttrSpec {
    pub(crate) name: String,
    pub(crate) datatype: Datatype,
    pub(crate) dataspace: Dataspace,
    pub(crate) value: Patched,
}

#[derive(Debug, Clone)]
pub(crate) enum LayoutSpec {
    Compact(Patched),
    Contiguous { address: u64, size: u64 },
    Chunked(Box<ChunkedSpec>),
}

#[derive(Debug, Clone)]
pub(crate) struct DatasetSpec {
    pub(crate) datatype: Datatype,
    pub(crate) committed: Option<ObjectId>,
    pub(crate) dataspace: Dataspace,
    pub(crate) layout: LayoutSpec,
    pub(crate) fill: FillValue,
}

#[derive(Debug, Clone)]
pub(crate) enum NodeKind {
    Group { links: Vec<LinkSpec> },
    Dataset(DatasetSpec),
    NamedDatatype(Datatype),
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub(crate) kind: NodeKind,
    pub(crate) attrs: Vec<AttrSpec>,
}

/// Writer for a new HDF5 file.
///
/// Create objects through the [`root`](FileWriter::root) group handle,
/// then call [`finish`](FileWriter::finish).
#[derive(Debug)]
pub struct FileWriter<S: WriteAt> {
    storage: S,
    opts: WriteOptions,
    alloc: Allocator,
    heap: HeapWriter,
    nodes: Vec<Node>,
    /// Object references inside contiguous raw data: (absolute position,
    /// target).
    data_refs: Vec<(u64, ObjectId)>,
    /// Filters available for chunked datasets.
    filters: FilterRegistry,
    /// Budget of the partially-written chunk cache, per dataset.
    chunk_cache_bytes: usize,
}

impl FileWriter<std::fs::File> {
    /// Create (or truncate) the file at `path` with default options.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Self::create_with(path, WriteOptions::default())
    }

    /// Create (or truncate) the file at `path`.
    pub fn create_with(path: impl AsRef<Path>, options: WriteOptions) -> Result<Self> {
        // Readable too: chunks may be read back for read-modify-write.
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        FileWriter::new(f, options)
    }
}

impl FileWriter<Vec<u8>> {
    /// A writer producing the file in memory; [`finish`](Self::finish)
    /// returns the bytes.
    pub fn in_memory() -> Self {
        FileWriter {
            storage: Vec::new(),
            opts: WriteOptions::default(),
            alloc: Allocator::new(
                Superblock::encoded_size_v23(Sizes::DEFAULT) as u64,
                Sizes::DEFAULT,
            ),
            heap: HeapWriter::default(),
            nodes: vec![Node::group()],
            data_refs: Vec::new(),
            filters: FilterRegistry::with_builtin(),
            chunk_cache_bytes: chunked::DEFAULT_CACHE_BYTES,
        }
    }
}

impl Node {
    fn group() -> Node {
        Node {
            kind: NodeKind::Group { links: Vec::new() },
            attrs: Vec::new(),
        }
    }
}

/// Validate a link or attribute name.
fn check_name(name: &str, what: &str) -> Result<()> {
    if name.is_empty() || name == "." || name.contains('/') || name.contains('\0') {
        return Err(Error::invalid(format!("invalid {what} name {name:?}")));
    }
    Ok(())
}

fn element_count(dims: &[u64]) -> Result<u64> {
    dims.iter().try_fold(1u64, |a, &d| {
        a.checked_mul(d)
            .ok_or(Error::LimitExceeded("element count overflows"))
    })
}

fn contains_vlen_or_ref(dt: &Datatype) -> bool {
    match &dt.class {
        DatatypeClass::VariableLength { .. } | DatatypeClass::Reference(_) => true,
        DatatypeClass::Compound { members } => {
            members.iter().any(|m| contains_vlen_or_ref(&m.datatype))
        }
        DatatypeClass::Array { base, .. }
        | DatatypeClass::Enumeration { base, .. }
        | DatatypeClass::Complex { base, .. } => contains_vlen_or_ref(base),
        _ => false,
    }
}

impl<S: WriteAt> FileWriter<S> {
    /// A writer emitting the file into `storage` (which should be empty).
    pub fn new(storage: S, options: WriteOptions) -> Result<Self> {
        if !matches!(options.superblock_version, 0 | 2 | 3) {
            return Err(Error::unsupported(format!(
                "writing superblock version {}",
                options.superblock_version
            )));
        }
        let ok = |v: u8| matches!(v, 2 | 4 | 8);
        if !ok(options.sizes.offset) || !ok(options.sizes.length) {
            return Err(Error::unsupported("offset/length sizes must be 2, 4 or 8"));
        }
        let start = if options.superblock_version == 0 {
            Superblock::encoded_size_v0(options.sizes)
        } else {
            Superblock::encoded_size_v23(options.sizes)
        } as u64;
        Ok(FileWriter {
            storage,
            opts: options,
            alloc: Allocator::new(start, options.sizes),
            heap: HeapWriter::default(),
            nodes: vec![Node::group()],
            data_refs: Vec::new(),
            filters: FilterRegistry::with_builtin(),
            chunk_cache_bytes: chunked::DEFAULT_CACHE_BYTES,
        })
    }

    /// Make `filter` available to chunked datasets of this file (replacing
    /// a built-in filter with the same ID). Filters requested by ID must be
    /// registered before the dataset is created.
    pub fn register_filter(&mut self, filter: Box<dyn Filter>) {
        self.filters.register(filter);
    }

    /// Set the per-dataset budget, in bytes, of the cache holding
    /// partially written chunks (default 32 MiB). When exceeded, cached
    /// chunks are written out, provided the storage can read them back.
    pub fn set_chunk_cache_size(&mut self, bytes: usize) {
        self.chunk_cache_bytes = bytes;
    }

    /// Offset/length widths of the file.
    pub fn sizes(&self) -> Sizes {
        self.opts.sizes
    }

    /// The options in use.
    pub fn options(&self) -> WriteOptions {
        self.opts
    }

    /// The root group.
    pub fn root(&mut self) -> GroupWriter<'_, S> {
        GroupWriter { w: self, id: 0 }
    }

    /// An existing group, by absolute path.
    pub fn group(&mut self, path: &str) -> Result<GroupWriter<'_, S>> {
        let id = self.resolve(path)?;
        match self.nodes.get(id).map(|n| &n.kind) {
            Some(NodeKind::Group { .. }) => Ok(GroupWriter { w: self, id }),
            _ => Err(Error::WrongObjectType(format!("{path} is not a group"))),
        }
    }

    /// An existing dataset, by absolute path.
    pub fn dataset(&mut self, path: &str) -> Result<DatasetWriter<'_, S>> {
        let id = self.resolve(path)?;
        match self.nodes.get(id).map(|n| &n.kind) {
            Some(NodeKind::Dataset(_)) => Ok(DatasetWriter { w: self, id }),
            _ => Err(Error::WrongObjectType(format!("{path} is not a dataset"))),
        }
    }

    /// The object at an absolute path (hard links only).
    pub fn object_id(&self, path: &str) -> Result<ObjectId> {
        self.resolve(path).map(ObjectId)
    }

    /// Set (or replace) an attribute on the object at `path`.
    pub fn set_attr<V: AttrValue + ?Sized>(
        &mut self,
        path: &str,
        name: &str,
        value: &V,
    ) -> Result<()> {
        let id = self.resolve(path)?;
        self.set_attr_on(id, name, value)
    }

    /// Whether the compatibility (superblock version 0) format is used.
    fn is_compat(&self) -> bool {
        self.opts.superblock_version < 2
    }

    fn resolve(&self, path: &str) -> Result<usize> {
        self.resolve_from(0, path)
    }

    fn resolve_from(&self, start: usize, path: &str) -> Result<usize> {
        let mut cur = if path.starts_with('/') { 0 } else { start };
        for comp in path.split('/').filter(|c| !c.is_empty() && *c != ".") {
            let Some(NodeKind::Group { links }) = self.nodes.get(cur).map(|n| &n.kind) else {
                return Err(Error::NotFound(path.to_owned()));
            };
            match links.iter().find(|l| l.name == comp).map(|l| &l.target) {
                Some(LinkTargetSpec::Hard(id)) => cur = id.0,
                Some(_) => {
                    return Err(Error::unsupported(format!(
                        "writer path {path:?} goes through a soft or external link"
                    )));
                }
                None => return Err(Error::NotFound(path.to_owned())),
            }
        }
        Ok(cur)
    }

    fn ctx(&mut self) -> EncodeContext<'_> {
        EncodeContext {
            sizes: self.opts.sizes,
            heap: &mut self.heap,
            alloc: &mut self.alloc,
            storage: &mut self.storage,
            refs: Vec::new(),
        }
    }

    fn add_link(&mut self, parent: usize, name: &str, target: LinkTargetSpec) -> Result<()> {
        check_name(name, "link")?;
        let Some(NodeKind::Group { links }) = self.nodes.get_mut(parent).map(|n| &mut n.kind)
        else {
            return Err(Error::WrongObjectType("link parent is not a group".into()));
        };
        if links.iter().any(|l| l.name == name) {
            return Err(Error::invalid(format!("link {name:?} already exists")));
        }
        links.push(LinkSpec {
            name: name.to_owned(),
            target,
        });
        Ok(())
    }

    fn add_node(&mut self, parent: usize, name: &str, node: Node) -> Result<usize> {
        check_name(name, "link")?;
        let id = self.nodes.len();
        self.nodes.push(node);
        if let Err(e) = self.add_link(parent, name, LinkTargetSpec::Hard(ObjectId(id))) {
            self.nodes.pop();
            return Err(e);
        }
        Ok(id)
    }

    fn set_attr_on<V: AttrValue + ?Sized>(
        &mut self,
        id: usize,
        name: &str,
        value: &V,
    ) -> Result<()> {
        let sizes = self.opts.sizes;
        let datatype = value.attr_datatype(sizes);
        let dataspace = match value.attr_shape() {
            None => Dataspace::scalar(),
            Some(dims) => Dataspace::simple(&dims, None),
        };
        let mut ctx = self.ctx();
        let mut data = Vec::new();
        value.attr_encode(&mut data, &mut ctx)?;
        let refs = std::mem::take(&mut ctx.refs);
        self.put_attr(
            id,
            AttrSpec {
                name: name.to_owned(),
                datatype,
                dataspace,
                value: Patched { data, refs },
            },
        )
    }

    fn put_attr(&mut self, id: usize, attr: AttrSpec) -> Result<()> {
        check_name(&attr.name, "attribute")?;
        let expect = attr
            .dataspace
            .element_count()?
            .checked_mul(u64::from(attr.datatype.size));
        if expect != Some(attr.value.data.len() as u64) {
            return Err(Error::invalid(format!(
                "attribute {:?}: data size does not match datatype and shape",
                attr.name
            )));
        }
        // Validate the encoding now rather than at finish.
        let compat = self.is_compat();
        let msg = build::attribute_message(&attr, self.opts.sizes, compat)?;
        if compat && msg.len() > usize::from(u16::MAX) - 8 {
            return Err(Error::LimitExceeded(
                "attribute too large for the compatibility format (64 KiB)",
            ));
        }
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| Error::invalid("unknown object"))?;
        match node.attrs.iter_mut().find(|a| a.name == attr.name) {
            Some(slot) => *slot = attr,
            None => node.attrs.push(attr),
        }
        Ok(())
    }

    /// Write all metadata and the superblock, returning the storage.
    pub fn finish(mut self) -> Result<S> {
        let sizes = self.opts.sizes;
        for id in 0..self.nodes.len() {
            if let Some(NodeKind::Dataset(DatasetSpec {
                layout: LayoutSpec::Chunked(_),
                ..
            })) = self.nodes.get(id).map(|n| &n.kind)
            {
                self.with_chunks(id, |c, env| c.finalize(env))?;
            }
        }
        self.heap.flush(sizes, &mut self.storage)?;
        let root = build::write_headers(&mut self)?;
        let width = sizes.offset as usize;
        let addrs = root.addresses;
        for &(pos, target) in &self.data_refs {
            let addr = *addrs
                .get(target.0)
                .ok_or_else(|| Error::invalid("object reference to an unknown object"))?;
            self.storage
                .write_all_at(pos, &addr.to_le_bytes()[..width.min(8)])?;
        }
        let eof = self.alloc.eoa();
        let sb = match root.root_symtab {
            Some((btree, heap)) => Superblock::new_v0(
                sizes,
                compat::root_entry(root.root_address, btree, heap),
                eof,
            ),
            None => {
                Superblock::new_v23(self.opts.superblock_version, sizes, root.root_address, eof)
            }
        };
        self.storage.write_all_at(0, &sb.encode()?)?;
        self.storage.finish(eof)?;
        Ok(self.storage)
    }
}

/// Handle to a group being written.
#[derive(Debug)]
pub struct GroupWriter<'w, S: WriteAt> {
    w: &'w mut FileWriter<S>,
    id: usize,
}

impl<'w, S: WriteAt> GroupWriter<'w, S> {
    /// This group's object ID.
    pub fn id(&self) -> ObjectId {
        ObjectId(self.id)
    }

    /// Create a child group.
    pub fn create_group(&mut self, name: &str) -> Result<GroupWriter<'_, S>> {
        let id = self.w.add_node(self.id, name, Node::group())?;
        Ok(GroupWriter { w: self.w, id })
    }

    /// An existing group, by path relative to this group (or absolute).
    pub fn group(&mut self, path: &str) -> Result<GroupWriter<'_, S>> {
        let id = self.w.resolve_from(self.id, path)?;
        match self.w.nodes.get(id).map(|n| &n.kind) {
            Some(NodeKind::Group { .. }) => Ok(GroupWriter { w: self.w, id }),
            _ => Err(Error::WrongObjectType(format!("{path} is not a group"))),
        }
    }

    /// Start a dataset whose elements are `T`.
    pub fn new_dataset<T: H5Write>(&mut self, name: &str) -> DatasetBuilder<'_, S, T> {
        let dt = T::datatype(self.w.opts.sizes);
        self.new_dataset_with_type(name, dt).with_element_type()
    }

    /// Start a dataset with an explicit datatype. Write it with
    /// [`DatasetBuilder::write_raw`], or tag it with a matching Rust type
    /// using [`DatasetBuilder::with_element_type`].
    pub fn new_dataset_with_type(
        &mut self,
        name: &str,
        datatype: Datatype,
    ) -> DatasetBuilder<'_, S> {
        DatasetBuilder {
            w: self.w,
            parent: self.id,
            name: name.to_owned(),
            datatype,
            committed: None,
            shape: None,
            layout: Layout::Contiguous,
            fill: None,
            chunk: None,
            max_shape: None,
            filters: Vec::new(),
            partial_unfiltered: false,
            alloc_time: None,
            _elem: PhantomData,
        }
    }

    /// Start a dataset using the committed datatype `datatype` (created
    /// with [`commit_datatype`](Self::commit_datatype)).
    pub fn new_dataset_committed(
        &mut self,
        name: &str,
        datatype: ObjectId,
    ) -> Result<DatasetBuilder<'_, S>> {
        let Some(NodeKind::NamedDatatype(dt)) = self.w.nodes.get(datatype.0).map(|n| &n.kind)
        else {
            return Err(Error::WrongObjectType("not a committed datatype".into()));
        };
        let dt = dt.clone();
        let mut b = self.new_dataset_with_type(name, dt);
        b.committed = Some(datatype);
        Ok(b)
    }

    /// Store `datatype` as a named (committed) datatype object.
    pub fn commit_datatype(&mut self, name: &str, datatype: Datatype) -> Result<ObjectId> {
        datatype.encode()?;
        let id = self.w.add_node(
            self.id,
            name,
            Node {
                kind: NodeKind::NamedDatatype(datatype),
                attrs: Vec::new(),
            },
        )?;
        Ok(ObjectId(id))
    }

    /// Add a hard link to an existing object.
    pub fn link_hard(&mut self, name: &str, target: ObjectId) -> Result<()> {
        if target.0 >= self.w.nodes.len() {
            return Err(Error::invalid("hard link to an unknown object"));
        }
        self.w.add_link(self.id, name, LinkTargetSpec::Hard(target))
    }

    /// Add a soft link holding `target_path`.
    pub fn link_soft(&mut self, name: &str, target_path: &str) -> Result<()> {
        if target_path.is_empty() {
            return Err(Error::invalid("empty soft link target"));
        }
        self.w
            .add_link(self.id, name, LinkTargetSpec::Soft(target_path.to_owned()))
    }

    /// Add an external link to `object_path` in the file `file`.
    pub fn link_external(&mut self, name: &str, file: &str, object_path: &str) -> Result<()> {
        if file.is_empty() || object_path.is_empty() {
            return Err(Error::invalid("empty external link file or path"));
        }
        if self.w.is_compat() {
            return Err(Error::unsupported(
                "external links in the compatibility format",
            ));
        }
        self.w.add_link(
            self.id,
            name,
            LinkTargetSpec::External {
                file: file.to_owned(),
                path: object_path.to_owned(),
            },
        )
    }

    /// Set (or replace) an attribute.
    pub fn set_attr<V: AttrValue + ?Sized>(&mut self, name: &str, value: &V) -> Result<()> {
        self.w.set_attr_on(self.id, name, value)
    }

    /// Set (or replace) an attribute from raw element bytes; `shape` of
    /// `None` makes it scalar.
    pub fn set_attr_raw(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: Option<&[u64]>,
        data: &[u8],
    ) -> Result<()> {
        set_attr_raw(self.w, self.id, name, datatype, shape, data)
    }
}

fn set_attr_raw<S: WriteAt>(
    w: &mut FileWriter<S>,
    id: usize,
    name: &str,
    datatype: Datatype,
    shape: Option<&[u64]>,
    data: &[u8],
) -> Result<()> {
    if contains_vlen_or_ref(&datatype) {
        return Err(Error::unsupported(
            "raw attributes with variable-length or reference types",
        ));
    }
    let dataspace = match shape {
        None => Dataspace::scalar(),
        Some(dims) => Dataspace::simple(dims, None),
    };
    w.put_attr(
        id,
        AttrSpec {
            name: name.to_owned(),
            datatype,
            dataspace,
            value: Patched {
                data: data.to_vec(),
                refs: Vec::new(),
            },
        },
    )
}

/// Handle to a dataset that has been created.
///
/// Chunked datasets can be grown with [`resize`](Self::resize) /
/// [`append`](Self::append) and partially rewritten with
/// [`write_selection`](Self::write_selection) until the file is finished.
#[derive(Debug)]
pub struct DatasetWriter<'w, S: WriteAt> {
    w: &'w mut FileWriter<S>,
    id: usize,
}

/// Whether `dt` holds object or region references.
fn contains_ref(dt: &Datatype) -> bool {
    match &dt.class {
        DatatypeClass::Reference(_) => true,
        DatatypeClass::Compound { members } => members.iter().any(|m| contains_ref(&m.datatype)),
        DatatypeClass::Array { base, .. }
        | DatatypeClass::Enumeration { base, .. }
        | DatatypeClass::Complex { base, .. }
        | DatatypeClass::VariableLength { base, .. } => contains_ref(base),
        _ => false,
    }
}

/// Maximum dimensions of a dataspace ([`UNLIMITED`] for unlimited ones).
fn max_dims_of(space: &Dataspace) -> Vec<u64> {
    (0..space.rank())
        .map(|i| space.max_dim(i).unwrap_or(0))
        .collect()
}

impl<S: WriteAt> FileWriter<S> {
    fn dataset_spec(&self, id: usize) -> Result<&DatasetSpec> {
        match self.nodes.get(id).map(|n| &n.kind) {
            Some(NodeKind::Dataset(d)) => Ok(d),
            _ => Err(Error::WrongObjectType("not a dataset".into())),
        }
    }

    fn dataset_spec_mut(&mut self, id: usize) -> Result<&mut DatasetSpec> {
        match self.nodes.get_mut(id).map(|n| &mut n.kind) {
            Some(NodeKind::Dataset(d)) => Ok(d),
            _ => Err(Error::WrongObjectType("not a dataset".into())),
        }
    }

    /// Run `f` on the chunked storage state of dataset `id`.
    fn with_chunks<R>(
        &mut self,
        id: usize,
        f: impl FnOnce(&mut ChunkedSpec, &mut Env<'_>) -> Result<R>,
    ) -> Result<R> {
        let FileWriter {
            storage,
            opts,
            alloc,
            nodes,
            filters,
            chunk_cache_bytes,
            ..
        } = self;
        let Some(NodeKind::Dataset(d)) = nodes.get_mut(id).map(|n| &mut n.kind) else {
            return Err(Error::WrongObjectType("not a dataset".into()));
        };
        let esize = d.datatype.size as usize;
        let fill = d.fill.value.clone().unwrap_or_else(|| vec![0; esize]);
        let dims = d.dataspace.dims.clone();
        let max = max_dims_of(&d.dataspace);
        let LayoutSpec::Chunked(spec) = &mut d.layout else {
            return Err(Error::unsupported("operation requires a chunked dataset"));
        };
        let mut env = Env {
            sizes: opts.sizes,
            storage,
            alloc,
            registry: filters,
            element_size: esize,
            fill: &fill,
            dims: &dims,
            max_dims: &max,
            cache_limit: *chunk_cache_bytes,
        };
        f(spec, &mut env)
    }

    /// Write packed element bytes into the selected elements of dataset `id`.
    fn write_selection_bytes(&mut self, id: usize, sel: &Selection, data: &[u8]) -> Result<()> {
        let d = self.dataset_spec(id)?;
        let esize = u64::from(d.datatype.size);
        let resolved = sel.resolve(&d.dataspace)?;
        if data.len() as u64 != resolved.count.saturating_mul(esize) {
            return Err(Error::invalid(format!(
                "{} bytes given for {} selected elements",
                data.len(),
                resolved.count
            )));
        }
        if resolved.count == 0 {
            return Ok(());
        }
        match &d.layout {
            LayoutSpec::Chunked(_) => self.with_chunks(id, |c, env| c.write(env, &resolved, data)),
            LayoutSpec::Compact(_) => {
                let dims = d.dataspace.dims.clone();
                let d = self.dataset_spec_mut(id)?;
                let LayoutSpec::Compact(p) = &mut d.layout else {
                    return Err(Error::invalid("layout changed"));
                };
                copy_runs(&resolved, &dims, esize as usize, data, &mut |pos, bytes| {
                    p.data
                        .get_mut(pos..pos + bytes.len())
                        .ok_or_else(|| Error::invalid("selection outside compact data"))?
                        .copy_from_slice(bytes);
                    Ok(())
                })
            }
            LayoutSpec::Contiguous { address, size } => {
                let (mut address, size) = (*address, *size);
                let dims = d.dataspace.dims.clone();
                let fill = d.fill.value.clone();
                if address == UNDEFINED_ADDRESS {
                    // First write: allocate and initialize with the fill value.
                    address = self.alloc.alloc(size)?;
                    if let Some(f) = fill.filter(|f| f.iter().any(|&b| b != 0)) {
                        let per = (1usize << 20) / f.len().max(1);
                        let block = f.repeat(per.max(1));
                        let mut pos = 0u64;
                        while pos < size {
                            let n = (size - pos).min(block.len() as u64) as usize;
                            self.storage.write_all_at(address + pos, &block[..n])?;
                            pos += n as u64;
                        }
                    } else if size > 0 {
                        // Make sure the region exists even if never written.
                        self.storage.write_all_at(address + size - 1, &[0])?;
                    }
                    let d = self.dataset_spec_mut(id)?;
                    d.layout = LayoutSpec::Contiguous { address, size };
                }
                let storage = &mut self.storage;
                copy_runs(&resolved, &dims, esize as usize, data, &mut |pos, bytes| {
                    storage.write_all_at(address + pos as u64, bytes)
                })
            }
        }
    }
}

/// Call `put(byte offset in the dataset, bytes)` for every run of the
/// selection `sel` over a row-major array of shape `dims`.
fn copy_runs(
    sel: &Resolved,
    dims: &[u64],
    esize: usize,
    data: &[u8],
    put: &mut dyn FnMut(usize, &[u8]) -> Result<()>,
) -> Result<()> {
    let origin = vec![0u64; dims.len()];
    let strides = row_major_strides(dims);
    let Some(runs) = block_runs(sel, &origin, dims, &strides) else {
        return Ok(());
    };
    let to = |v: u64| -> Result<usize> {
        usize::try_from(v)
            .ok()
            .and_then(|v| v.checked_mul(esize))
            .ok_or(Error::LimitExceeded("dataset size"))
    };
    for_each_run(&runs, &mut |s, o, n| {
        let (s, o, n) = (to(s)?, to(o)?, to(n)?);
        let src = data
            .get(o..o + n)
            .ok_or_else(|| Error::invalid("selection run outside data"))?;
        put(s, src)
    })
}

impl<S: WriteAt> DatasetWriter<'_, S> {
    /// This dataset's object ID.
    pub fn id(&self) -> ObjectId {
        ObjectId(self.id)
    }

    /// Set (or replace) an attribute.
    pub fn set_attr<V: AttrValue + ?Sized>(&mut self, name: &str, value: &V) -> Result<()> {
        self.w.set_attr_on(self.id, name, value)
    }

    /// Set (or replace) an attribute from raw element bytes.
    pub fn set_attr_raw(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: Option<&[u64]>,
        data: &[u8],
    ) -> Result<()> {
        set_attr_raw(self.w, self.id, name, datatype, shape, data)
    }

    /// Current dimensions.
    pub fn shape(&self) -> Result<Vec<u64>> {
        Ok(self.w.dataset_spec(self.id)?.dataspace.dims.clone())
    }

    /// Maximum dimensions (`None` = unlimited).
    pub fn max_shape(&self) -> Result<Vec<Option<u64>>> {
        let d = self.w.dataset_spec(self.id)?;
        Ok(max_dims_of(&d.dataspace)
            .into_iter()
            .map(|m| (m != UNLIMITED).then_some(m))
            .collect())
    }

    /// Write `data` (row-major, one value per selected element) into the
    /// selected elements. Works with every layout; elements of chunks never
    /// written read as the fill value. Not available for types containing
    /// references.
    pub fn write_selection<T: H5Write>(&mut self, selection: &Selection, data: &[T]) -> Result<()> {
        let dt = self.w.dataset_spec(self.id)?.datatype.clone();
        if T::datatype(self.w.opts.sizes) != dt {
            return Err(Error::invalid(
                "Rust type does not match the dataset datatype",
            ));
        }
        if contains_ref(&dt) {
            return Err(Error::unsupported("partial writes of reference data"));
        }
        let mut ctx = self.w.ctx();
        let mut bytes = Vec::new();
        T::encode_slice(data, &mut bytes, &mut ctx)?;
        self.w.write_selection_bytes(self.id, selection, &bytes)
    }

    /// Like [`write_selection`](Self::write_selection), from raw element
    /// bytes. Not available for types containing variable-length data or
    /// references.
    pub fn write_selection_raw(&mut self, selection: &Selection, data: &[u8]) -> Result<()> {
        if contains_vlen_or_ref(&self.w.dataset_spec(self.id)?.datatype) {
            return Err(Error::unsupported(
                "raw writes of variable-length or reference types",
            ));
        }
        self.w.write_selection_bytes(self.id, selection, data)
    }

    /// Change the current dimensions of a chunked dataset, within its
    /// maximum dimensions. New elements read as the fill value; when
    /// shrinking, elements outside the new extent are discarded.
    pub fn resize(&mut self, dims: &[u64]) -> Result<()> {
        let d = self.w.dataset_spec(self.id)?;
        if !matches!(d.layout, LayoutSpec::Chunked(_)) {
            return Err(Error::unsupported("resizing a dataset that is not chunked"));
        }
        let max = max_dims_of(&d.dataspace);
        if dims.len() != max.len() {
            return Err(Error::invalid("resize: rank mismatch"));
        }
        if dims
            .iter()
            .zip(&max)
            .any(|(&n, &m)| n == UNLIMITED || (m != UNLIMITED && n > m))
        {
            return Err(Error::invalid(
                "resize: new size exceeds the maximum dimensions",
            ));
        }
        element_count(dims)?;
        self.w.with_chunks(self.id, |c, env| c.resize(env, dims))?;
        let d = self.w.dataset_spec_mut(self.id)?;
        d.dataspace.dims = dims.to_vec();
        Ok(())
    }

    /// Append `data` along the first unlimited dimension of a chunked
    /// dataset: the data is a row-major block whose other dimensions match
    /// the current ones (for a 1-D dataset, simply the new elements).
    pub fn append<T: H5Write>(&mut self, data: &[T]) -> Result<()> {
        let sel = self.grow_for_append(data.len() as u64)?;
        self.write_selection(&sel, data)
    }

    /// Like [`append`](Self::append), from raw element bytes.
    pub fn append_raw(&mut self, data: &[u8]) -> Result<()> {
        let size = u64::from(self.w.dataset_spec(self.id)?.datatype.size);
        if size == 0 || !(data.len() as u64).is_multiple_of(size) {
            return Err(Error::invalid("raw data is not a whole number of elements"));
        }
        let sel = self.grow_for_append(data.len() as u64 / size)?;
        self.write_selection_raw(&sel, data)
    }

    /// Grow the first unlimited dimension to make room for `count`
    /// elements; returns the selection of the new region.
    fn grow_for_append(&mut self, count: u64) -> Result<Selection> {
        let d = self.w.dataset_spec(self.id)?;
        let max = max_dims_of(&d.dataspace);
        let axis = max
            .iter()
            .position(|&m| m == UNLIMITED)
            .ok_or_else(|| Error::invalid("append needs an unlimited dimension"))?;
        let mut dims = d.dataspace.dims.clone();
        let slab: u64 = dims
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .try_fold(1u64, |a, (_, &n)| a.checked_mul(n))
            .ok_or(Error::LimitExceeded("element count overflows"))?;
        if slab == 0 || !count.is_multiple_of(slab) {
            return Err(Error::invalid(
                "appended data does not fill whole slices of the other dimensions",
            ));
        }
        let old = dims[axis];
        let new = old
            .checked_add(count / slab)
            .ok_or(Error::LimitExceeded("dimension overflows"))?;
        dims[axis] = new;
        let ranges: Vec<std::ops::Range<u64>> = dims
            .iter()
            .enumerate()
            .map(|(i, &n)| if i == axis { old..new } else { 0..n })
            .collect();
        self.resize(&dims)?;
        Ok(Selection::from_ranges(&ranges))
    }

    /// Write out partially written chunks held in memory.
    pub fn flush(&mut self) -> Result<()> {
        if !matches!(self.w.dataset_spec(self.id)?.layout, LayoutSpec::Chunked(_)) {
            return Ok(());
        }
        self.w.with_chunks(self.id, |c, env| c.flush(env))
    }
}

/// Element marker of a [`DatasetBuilder`] whose datatype was given
/// explicitly rather than derived from a Rust type.
#[derive(Debug, Clone, Copy)]
pub enum Untyped {}

/// Parameters of the scale-offset filter (see [`DatasetBuilder::scale_offset`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleOffsetMode {
    /// Integer data: keep `min_bits` bits per value (0 = as many as each
    /// chunk's range needs; lossless).
    Integer {
        /// Bits per value, 0 for automatic.
        min_bits: u32,
    },
    /// Floating-point data scaled by `10^decimal_digits` and rounded
    /// (lossy, "D-scaling").
    FloatDecimal {
        /// Decimal digits kept.
        decimal_digits: i32,
    },
}

/// A filter requested on a [`DatasetBuilder`], resolved when the dataset is
/// created.
#[derive(Debug, Clone)]
enum FilterRequest {
    Deflate(u32),
    Shuffle,
    Fletcher32,
    Lzf,
    NBit,
    ScaleOffset(ScaleOffsetMode),
    Other {
        id: u16,
        optional: bool,
        client_data: Vec<u32>,
    },
}

/// Builder for a new dataset with elements of Rust type `T`; finish with
/// [`write`](Self::write), [`write_raw`](Self::write_raw) or
/// [`create`](Self::create).
///
/// Requesting chunks, filters or maximum dimensions makes the dataset
/// chunked (chunk dimensions are guessed when not given).
#[derive(Debug)]
pub struct DatasetBuilder<'w, S: WriteAt, T: ?Sized = Untyped> {
    w: &'w mut FileWriter<S>,
    parent: usize,
    name: String,
    datatype: Datatype,
    committed: Option<ObjectId>,
    shape: Option<Vec<u64>>,
    layout: Layout,
    fill: Option<Vec<u8>>,
    chunk: Option<Vec<u64>>,
    max_shape: Option<Vec<u64>>,
    filters: Vec<FilterRequest>,
    partial_unfiltered: bool,
    alloc_time: Option<AllocTime>,
    _elem: PhantomData<fn() -> Box<T>>,
}

impl<'w, S: WriteAt, T: H5Write> DatasetBuilder<'w, S, T> {
    /// Set the fill value (read for elements never written).
    pub fn fill_value(mut self, value: T) -> Result<Self> {
        self.check_type::<T>()?;
        if contains_vlen_or_ref(&self.datatype) {
            return Err(Error::unsupported(
                "fill values of variable-length or reference types",
            ));
        }
        let mut ctx = self.w.ctx();
        let mut bytes = Vec::new();
        value.encode(&mut bytes, &mut ctx)?;
        self.fill = Some(bytes);
        Ok(self)
    }

    /// Create the dataset and write `data` (all elements, row-major).
    pub fn write(self, data: &[T]) -> Result<DatasetWriter<'w, S>> {
        self.check_type::<T>()?;
        let mut ctx = self.w.ctx();
        let mut bytes = Vec::new();
        T::encode_slice(data, &mut bytes, &mut ctx)?;
        let refs = std::mem::take(&mut ctx.refs);
        let count = data.len() as u64;
        self.finish(Some((Patched { data: bytes, refs }, count)))
    }
}

impl<'w, S: WriteAt, T: ?Sized> DatasetBuilder<'w, S, T> {
    /// Re-tag the builder with element type `U`; its datatype must match
    /// the dataset's (checked when writing).
    pub fn with_element_type<U: ?Sized>(self) -> DatasetBuilder<'w, S, U> {
        DatasetBuilder {
            w: self.w,
            parent: self.parent,
            name: self.name,
            datatype: self.datatype,
            committed: self.committed,
            shape: self.shape,
            layout: self.layout,
            fill: self.fill,
            chunk: self.chunk,
            max_shape: self.max_shape,
            filters: self.filters,
            partial_unfiltered: self.partial_unfiltered,
            alloc_time: self.alloc_time,
            _elem: PhantomData,
        }
    }

    /// Set the dimensions (slowest-changing first); an empty shape makes
    /// a scalar dataset. Defaults to one dimension of the data's length.
    pub fn shape(mut self, dims: &[u64]) -> Self {
        self.shape = Some(dims.to_vec());
        self
    }

    /// Choose the storage layout.
    pub fn layout(mut self, layout: Layout) -> Self {
        self.layout = layout;
        self
    }

    /// Shorthand for `layout(Layout::Compact)`.
    pub fn compact(self) -> Self {
        self.layout(Layout::Compact)
    }

    /// Store the dataset in chunks of `dims` elements (one entry per
    /// dimension; along fixed-size dimensions at most the maximum size).
    pub fn chunk(mut self, dims: &[u64]) -> Self {
        self.chunk = Some(dims.to_vec());
        self
    }

    /// Set the maximum dimensions (`None` = unlimited), making the dataset
    /// resizable (and chunked).
    pub fn max_shape(mut self, dims: &[Option<u64>]) -> Self {
        self.max_shape = Some(dims.iter().map(|d| d.unwrap_or(UNLIMITED)).collect());
        self
    }

    /// Add the deflate (gzip) filter at `level` (0-9).
    pub fn deflate(mut self, level: u32) -> Self {
        self.filters.push(FilterRequest::Deflate(level.min(9)));
        self
    }

    /// Add the byte shuffle filter (best placed before compression).
    pub fn shuffle(mut self) -> Self {
        self.filters.push(FilterRequest::Shuffle);
        self
    }

    /// Add the Fletcher-32 checksum filter.
    pub fn fletcher32(mut self) -> Self {
        self.filters.push(FilterRequest::Fletcher32);
        self
    }

    /// Add the LZF compression filter (ID 32000).
    pub fn lzf(mut self) -> Self {
        self.filters.push(FilterRequest::Lzf);
        self
    }

    /// Add the N-bit filter (packs values to their datatype precision).
    pub fn nbit(mut self) -> Self {
        self.filters.push(FilterRequest::NBit);
        self
    }

    /// Add the scale-offset filter.
    pub fn scale_offset(mut self, mode: ScaleOffsetMode) -> Self {
        self.filters.push(FilterRequest::ScaleOffset(mode));
        self
    }

    /// Add filter `id` (which must be registered, see
    /// [`FileWriter::register_filter`]) with `client_data`, as a mandatory
    /// filter: if it fails, writing fails.
    pub fn filter(mut self, id: u16, client_data: &[u32]) -> Self {
        self.filters.push(FilterRequest::Other {
            id,
            optional: false,
            client_data: client_data.to_vec(),
        });
        self
    }

    /// Like [`filter`](Self::filter), but optional: chunks the filter fails
    /// on are stored without it (recorded in their filter mask).
    pub fn optional_filter(mut self, id: u16, client_data: &[u32]) -> Self {
        self.filters.push(FilterRequest::Other {
            id,
            optional: true,
            client_data: client_data.to_vec(),
        });
        self
    }

    /// Store partial edge chunks without applying the filters.
    pub fn dont_filter_partial_chunks(mut self) -> Self {
        self.partial_unfiltered = true;
        self
    }

    /// Set the space allocation time of a chunked dataset:
    /// [`AllocTime::Early`] allocates every chunk (filled with the fill
    /// value); unfiltered fixed-size datasets then use the implicit index.
    /// The default allocates chunks as they are written.
    pub fn alloc_time(mut self, time: AllocTime) -> Self {
        self.alloc_time = Some(time);
        self
    }

    /// Set the fill value from raw element bytes.
    pub fn fill_value_raw(mut self, bytes: &[u8]) -> Result<Self> {
        if bytes.len() != self.datatype.size as usize {
            return Err(Error::invalid(
                "fill value size differs from the element size",
            ));
        }
        self.fill = Some(bytes.to_vec());
        Ok(self)
    }

    fn check_type<U: H5Write>(&self) -> Result<()> {
        if U::datatype(self.w.opts.sizes) != self.datatype {
            return Err(Error::invalid(format!(
                "dataset {:?}: Rust type does not match the dataset datatype",
                self.name
            )));
        }
        Ok(())
    }

    /// Create the dataset from raw element bytes (row-major, exactly
    /// `element count * element size` bytes). Not available for types
    /// containing variable-length data or references.
    pub fn write_raw(self, data: &[u8]) -> Result<DatasetWriter<'w, S>> {
        if contains_vlen_or_ref(&self.datatype) {
            return Err(Error::unsupported(
                "raw writes of variable-length or reference types",
            ));
        }
        let size = u64::from(self.datatype.size);
        if size == 0 || !(data.len() as u64).is_multiple_of(size) {
            return Err(Error::invalid("raw data is not a whole number of elements"));
        }
        let count = data.len() as u64 / size;
        self.finish(Some((
            Patched {
                data: data.to_vec(),
                refs: Vec::new(),
            },
            count,
        )))
    }

    /// Create the dataset without writing data: elements read as the fill
    /// value (contiguous storage is left unallocated). Write it later
    /// through the returned [`DatasetWriter`].
    pub fn create(self) -> Result<DatasetWriter<'w, S>> {
        if self.shape.is_none() {
            return Err(Error::invalid(format!(
                "dataset {:?}: shape required when creating without data",
                self.name
            )));
        }
        self.finish(None)
    }

    /// Resolve the requested filters into a pipeline.
    fn pipeline(&self, chunk: &[u64], registry: &FilterRegistry) -> Result<FilterPipeline> {
        let chunk_elements = chunk
            .iter()
            .try_fold(1u64, |a, &c| a.checked_mul(c))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(Error::LimitExceeded("chunk element count"))?;
        let esize = self.datatype.size;
        let mut filters = Vec::with_capacity(self.filters.len());
        for req in &self.filters {
            let (id, optional, client_data) = match req {
                FilterRequest::Deflate(level) => (crate::filters::DEFLATE, true, vec![*level]),
                FilterRequest::Shuffle => (crate::filters::SHUFFLE, true, vec![esize]),
                FilterRequest::Fletcher32 => (crate::filters::FLETCHER32, false, Vec::new()),
                FilterRequest::Lzf => {
                    let bytes = u64::from(chunk_elements) * u64::from(esize);
                    let bytes =
                        u32::try_from(bytes).map_err(|_| Error::LimitExceeded("chunk size"))?;
                    // Filter version 4, LZF version 0x0105, chunk size.
                    (crate::filters::LZF, true, vec![4, 0x0105, bytes])
                }
                FilterRequest::NBit => (
                    crate::filters::NBIT,
                    true,
                    crate::filters::nbit::client_data(&self.datatype, chunk_elements)?,
                ),
                FilterRequest::ScaleOffset(mode) => {
                    let (ty, factor) = match *mode {
                        ScaleOffsetMode::Integer { min_bits } => (
                            crate::filters::scaleoffset::SCALE_INT,
                            i32::try_from(min_bits)
                                .map_err(|_| Error::invalid("scale-offset min bits"))?,
                        ),
                        ScaleOffsetMode::FloatDecimal { decimal_digits } => (
                            crate::filters::scaleoffset::SCALE_FLOAT_DSCALE,
                            decimal_digits,
                        ),
                    };
                    (
                        crate::filters::SCALEOFFSET,
                        true,
                        crate::filters::scaleoffset::client_data(
                            &self.datatype,
                            chunk_elements,
                            ty,
                            factor,
                            self.fill.as_deref(),
                        )?,
                    )
                }
                FilterRequest::Other {
                    id,
                    optional,
                    client_data,
                } => (*id, *optional, client_data.clone()),
            };
            let filter = registry.get(id).ok_or_else(|| {
                Error::unsupported(format!("filter id {id} is not available for writing"))
            })?;
            filters.push(FilterInfo {
                id,
                name: Some(filter.name().to_owned()),
                optional,
                client_data,
            });
        }
        if filters.len() > crate::filters::MAX_PIPELINE_LEN {
            return Err(Error::invalid("more than 32 filters"));
        }
        Ok(FilterPipeline { filters })
    }

    fn finish(self, data: Option<(Patched, u64)>) -> Result<DatasetWriter<'w, S>> {
        check_name(&self.name, "link")?;
        self.datatype.encode()?;
        if self.datatype.size == 0 {
            return Err(Error::invalid("datatype size is zero"));
        }
        let dims = match (&self.shape, &data) {
            (Some(s), _) => s.clone(),
            (None, Some((_, n))) => vec![*n],
            (None, None) => Vec::new(),
        };
        let count = element_count(&dims)?;
        let total = count
            .checked_mul(u64::from(self.datatype.size))
            .ok_or(Error::LimitExceeded("dataset size overflows"))?;
        if let Some((d, n)) = &data
            && (*n != count || d.data.len() as u64 != total)
        {
            return Err(Error::invalid(format!(
                "dataset {:?}: {n} elements given for a shape of {count}",
                self.name
            )));
        }
        if let Some(n) = self.w.nodes.get(self.parent)
            && let NodeKind::Group { links } = &n.kind
            && links.iter().any(|l| l.name == self.name)
        {
            return Err(Error::invalid(format!(
                "link {:?} already exists",
                self.name
            )));
        }
        let chunked = self.chunk.is_some() || self.max_shape.is_some() || !self.filters.is_empty();
        if chunked {
            return self.finish_chunked(dims, data);
        }
        let DatasetBuilder {
            w,
            parent,
            name,
            datatype,
            committed,
            layout,
            fill,
            ..
        } = self;
        let dataspace = Dataspace::simple(&dims, None);
        let (layout, alloc_time) = match layout {
            Layout::Compact => {
                let patched = match data {
                    Some((p, _)) => p,
                    None => {
                        let elem = fill
                            .clone()
                            .unwrap_or_else(|| vec![0; datatype.size as usize]);
                        let n = usize::try_from(count)
                            .map_err(|_| Error::LimitExceeded("compact dataset size"))?;
                        if total > MAX_COMPACT_DATA as u64 {
                            return Err(Error::LimitExceeded("compact dataset larger than 64 KiB"));
                        }
                        Patched {
                            data: elem.repeat(n),
                            refs: Vec::new(),
                        }
                    }
                };
                if patched.data.len() > MAX_COMPACT_DATA {
                    return Err(Error::LimitExceeded("compact dataset larger than 64 KiB"));
                }
                (LayoutSpec::Compact(patched), AllocTime::Early)
            }
            Layout::Contiguous => {
                let address = match data {
                    Some((p, _)) if total > 0 => {
                        let addr = w.alloc.alloc(total)?;
                        w.storage.write_all_at(addr, &p.data)?;
                        for (pos, target) in p.refs {
                            w.data_refs.push((addr + pos as u64, target));
                        }
                        addr
                    }
                    _ => UNDEFINED_ADDRESS,
                };
                (
                    LayoutSpec::Contiguous {
                        address,
                        size: total,
                    },
                    AllocTime::Late,
                )
            }
        };
        let spec = DatasetSpec {
            datatype,
            committed,
            dataspace,
            layout,
            fill: FillValue::new_v3(alloc_time, fill),
        };
        let id = w.add_node(
            parent,
            &name,
            Node {
                kind: NodeKind::Dataset(spec),
                attrs: Vec::new(),
            },
        )?;
        Ok(DatasetWriter { w, id })
    }

    /// Create a chunked dataset (and write `data`, if given).
    fn finish_chunked(
        self,
        dims: Vec<u64>,
        data: Option<(Patched, u64)>,
    ) -> Result<DatasetWriter<'w, S>> {
        if self.layout == Layout::Compact {
            return Err(Error::invalid(
                "chunks, filters and maximum dimensions need chunked storage, not compact",
            ));
        }
        if dims.is_empty() {
            return Err(Error::invalid("a scalar dataset cannot be chunked"));
        }
        if contains_ref(&self.datatype) {
            return Err(Error::unsupported("chunked datasets of reference types"));
        }
        let max = match &self.max_shape {
            Some(m) => {
                if m.len() != dims.len() {
                    return Err(Error::invalid("maximum dimensions rank mismatch"));
                }
                if m.iter().zip(&dims).any(|(&mx, &d)| mx < d) {
                    return Err(Error::invalid("maximum dimension below current size"));
                }
                m.clone()
            }
            None => dims.clone(),
        };
        let esize = self.datatype.size as usize;
        let chunk = match &self.chunk {
            Some(c) => c.clone(),
            None => chunked::guess_chunk(&dims, &max, esize),
        };
        if chunk.len() != dims.len() {
            return Err(Error::invalid("chunk rank does not match the dataset rank"));
        }
        if chunk.contains(&0) {
            return Err(Error::invalid("chunk dimension of size zero"));
        }
        if chunk
            .iter()
            .zip(&max)
            .any(|(&c, &m)| m != UNLIMITED && c > m.max(1))
        {
            return Err(Error::invalid(
                "chunk dimension exceeds a fixed maximum dimension",
            ));
        }
        let early = self.alloc_time == Some(AllocTime::Early);
        let pipeline = self.pipeline(&chunk, &self.w.filters)?;
        let filtered = !pipeline.filters.is_empty();
        // With lengths narrower than 8 bytes the reference library reads
        // "unlimited" maximum dimensions back as finite values and then
        // rejects the extensible array / v2 B-tree indexes, so such
        // datasets keep the v1 B-tree.
        let narrow_unlimited = self.w.opts.sizes.length < 8 && max.contains(&UNLIMITED);
        let kind = if self.w.opts.superblock_version >= 3 && !narrow_unlimited {
            IndexKind::choose_v4(&max, &chunk, filtered, early)
        } else {
            if chunk.iter().any(|&c| c > u64::from(u32::MAX)) {
                return Err(Error::LimitExceeded("chunk dimension (layout version 3)"));
            }
            IndexKind::BTreeV1
        };
        let spec = ChunkedSpec::new(chunk, pipeline, kind, self.partial_unfiltered, early, esize)?;
        let DatasetBuilder {
            w,
            parent,
            name,
            datatype,
            committed,
            fill,
            ..
        } = self;
        let has_max = max != dims;
        let dataspace = Dataspace::simple(&dims, has_max.then_some(max.as_slice()));
        let alloc_time = if early {
            AllocTime::Early
        } else {
            AllocTime::Incremental
        };
        let spec = DatasetSpec {
            datatype,
            committed,
            dataspace,
            layout: LayoutSpec::Chunked(Box::new(spec)),
            fill: FillValue::new_v3(alloc_time, fill),
        };
        let id = w.add_node(
            parent,
            &name,
            Node {
                kind: NodeKind::Dataset(spec),
                attrs: Vec::new(),
            },
        )?;
        if kind == IndexKind::Implicit {
            w.with_chunks(id, |c, env| c.allocate_implicit(env))?;
        }
        if let Some((p, _)) = data {
            w.write_selection_bytes(id, &Selection::All, &p.data)?;
        }
        Ok(DatasetWriter { w, id })
    }
}

#[cfg(test)]
mod chunk_tests;
#[cfg(test)]
mod tests;
