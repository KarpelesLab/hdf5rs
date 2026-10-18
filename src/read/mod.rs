//! Raw data reading engine.
//!
//! This module turns the low-level description of a dataset (layout,
//! dataspace, filter pipeline, fill value, …) into the bytes of selected
//! elements. It performs **no type conversion**: every element is returned
//! exactly as stored on disk (`element_size` bytes, in the file's byte
//! order), packed in row-major order of the selection.
//!
//! # Usage
//!
//! Gather the dataset's header messages, bundle them in a
//! [`DatasetStorage`], then call [`read_raw`] (or
//! [`DatasetStorage::read`]) with a [`Selection`]:
//!
//! ```no_run
//! # fn demo(
//! #     store: &dyn hdf5rs::ReadAt,
//! #     sizes: hdf5rs::format::Sizes,
//! #     layout: &hdf5rs::format::messages::layout::DataLayout,
//! #     space: &hdf5rs::format::messages::dataspace::Dataspace,
//! #     pipeline: Option<&hdf5rs::format::messages::filter_pipeline::FilterPipeline>,
//! # ) -> hdf5rs::Result<()> {
//! use hdf5rs::read::{DatasetStorage, Selection, read_raw};
//!
//! let storage = DatasetStorage::new(store, sizes, layout, space, 4)
//!     .with_filters(pipeline);
//! let all = read_raw(&storage, &Selection::All)?;
//! let rows = read_raw(&storage, &Selection::from_ranges(&[2..4, 0..10]))?;
//! # Ok(()) }
//! ```
//!
//! # Supported storage
//!
//! * compact (data inside the layout message);
//! * contiguous, including never-written datasets (undefined address, read
//!   as the fill value) and external raw data files (through an
//!   [`ExternalFileResolver`]);
//! * chunked, with every chunk index: version 1 B-tree (layout versions
//!   1–3), single chunk (filtered or not), implicit, fixed array,
//!   extensible array and version 2 B-tree. Unallocated chunks read as the
//!   fill value; filters are applied per chunk honouring each chunk's
//!   filter mask.
//!
//! Virtual datasets need access to other datasets (and possibly other
//! files), so [`read_raw`] rejects them with [`Error::Unsupported`]; read
//! them through [`Dataset`](crate::Dataset), which resolves the mappings.
//!
//! # Chunk coordinates
//!
//! Version 1 B-tree keys hold element offsets; the version 4 indexes use
//! *scaled* coordinates (element offset divided by the chunk size). The
//! implicit and fixed array indexes linearize scaled coordinates row-major
//! over the grid computed from the dataspace's **maximum** dimensions; the
//! extensible array linearizes with its single unlimited dimension as the
//! slowest-changing one (see
//! [`extensible_linear_index`](crate::format::chunk_index::extensible_linear_index)).
//! Both rules were verified against files written by the reference
//! library.

mod chunked;
mod contiguous;
mod selection;
mod virtual_layout;
// The synthetic files use deflate-compressed chunks.
#[cfg(all(test, feature = "deflate"))]
mod tests;

use std::sync::OnceLock;

pub use chunked::ChunkInfo;
pub use selection::{HyperslabDim, Selection};
pub(crate) use selection::{Resolved, block_runs, for_each_run, row_major_strides};
pub(crate) use virtual_layout::read_virtual;

use crate::error::{Error, Result};
use crate::filters::FilterRegistry;
use crate::format::Sizes;
use crate::format::messages::dataspace::Dataspace;
use crate::format::messages::external_files::ExternalFiles;
use crate::format::messages::fill_value::{FillValue, FillValueState, OldFillValue};
use crate::format::messages::filter_pipeline::FilterPipeline;
use crate::format::messages::layout::DataLayout;
use crate::io::ReadAt;

/// Default cap on the size of one read's output: 1 GiB.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 1 << 30;

/// Default cap on the decoded size of one chunk: 1 GiB.
pub const DEFAULT_MAX_CHUNK_BYTES: usize = 1 << 30;

/// Resource limits applied while reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReadLimits {
    /// Maximum number of bytes a single read may return.
    pub max_output_bytes: usize,
    /// Maximum decoded size of one chunk (also bounds filter output).
    pub max_chunk_bytes: usize,
}

impl ReadLimits {
    /// Limits with the given maximum output size and the default chunk cap.
    pub fn with_max_output(max_output_bytes: usize) -> Self {
        ReadLimits {
            max_output_bytes,
            ..ReadLimits::default()
        }
    }

    /// Replace the maximum decoded chunk size.
    pub fn max_chunk_bytes(mut self, max_chunk_bytes: usize) -> Self {
        self.max_chunk_bytes = max_chunk_bytes;
        self
    }
}

impl Default for ReadLimits {
    fn default() -> Self {
        ReadLimits {
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_chunk_bytes: DEFAULT_MAX_CHUNK_BYTES,
        }
    }
}

/// Opens the external raw data files of a dataset (External Data Files
/// message).
///
/// `name` is the file name as stored in the dataset's local heap. How it is
/// resolved (relative to the HDF5 file, a search path, a sandbox, …) is up
/// to the implementation; returning an error aborts the read.
pub trait ExternalFileResolver {
    /// Open the external file `name` for reading.
    fn open(&self, name: &str) -> Result<Box<dyn ReadAt>>;
}

impl<F> ExternalFileResolver for F
where
    F: Fn(&str) -> Result<Box<dyn ReadAt>>,
{
    fn open(&self, name: &str) -> Result<Box<dyn ReadAt>> {
        self(name)
    }
}

/// The built-in filter registry ([`FilterRegistry::with_builtin`]), created
/// on first use.
pub fn builtin_registry() -> &'static FilterRegistry {
    static REGISTRY: OnceLock<FilterRegistry> = OnceLock::new();
    REGISTRY.get_or_init(FilterRegistry::with_builtin)
}

/// Everything needed to read a dataset's raw data.
///
/// Built with [`DatasetStorage::new`] from the mandatory pieces, then
/// refined with the `with_*` methods for the optional header messages.
/// All addresses are interpreted relative to `store` (use
/// [`Superblock::relative_store`](crate::format::superblock::Superblock::relative_store)
/// for files with a user block).
#[derive(Clone, Copy)]
pub struct DatasetStorage<'a> {
    store: &'a dyn ReadAt,
    sizes: Sizes,
    layout: &'a DataLayout,
    dataspace: &'a Dataspace,
    element_size: usize,
    pipeline: Option<&'a FilterPipeline>,
    fill_value: Option<&'a FillValue>,
    old_fill_value: Option<&'a OldFillValue>,
    external: Option<&'a ExternalFiles>,
    resolver: Option<&'a dyn ExternalFileResolver>,
    registry: &'a FilterRegistry,
    limits: ReadLimits,
}

impl std::fmt::Debug for DatasetStorage<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatasetStorage")
            .field("sizes", &self.sizes)
            .field("layout", self.layout)
            .field("dataspace", self.dataspace)
            .field("element_size", &self.element_size)
            .field("pipeline", &self.pipeline)
            .field("fill_value", &self.fill_value)
            .field("old_fill_value", &self.old_fill_value)
            .field("external", &self.external)
            .field("has_resolver", &self.resolver.is_some())
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl<'a> DatasetStorage<'a> {
    /// Describe a dataset stored in `store`.
    ///
    /// `element_size` is the size in bytes of one element (the datatype
    /// message's size). The built-in filter registry and default limits are
    /// used unless replaced.
    pub fn new(
        store: &'a dyn ReadAt,
        sizes: Sizes,
        layout: &'a DataLayout,
        dataspace: &'a Dataspace,
        element_size: usize,
    ) -> Self {
        DatasetStorage {
            store,
            sizes,
            layout,
            dataspace,
            element_size,
            pipeline: None,
            fill_value: None,
            old_fill_value: None,
            external: None,
            resolver: None,
            registry: builtin_registry(),
            limits: ReadLimits::default(),
        }
    }

    /// Set the filter pipeline message (chunked datasets only).
    pub fn with_filters(mut self, pipeline: Option<&'a FilterPipeline>) -> Self {
        self.pipeline = pipeline;
        self
    }

    /// Set the fill value message (type 0x0005).
    pub fn with_fill_value(mut self, fill: Option<&'a FillValue>) -> Self {
        self.fill_value = fill;
        self
    }

    /// Set the old fill value message (type 0x0004), used when no new-style
    /// fill value message defines a value.
    pub fn with_old_fill_value(mut self, fill: Option<&'a OldFillValue>) -> Self {
        self.old_fill_value = fill;
        self
    }

    /// Set the external data files message and the resolver used to open
    /// the files. Without a resolver, reading such a dataset fails with
    /// [`Error::Unsupported`].
    pub fn with_external(
        mut self,
        external: Option<&'a ExternalFiles>,
        resolver: Option<&'a dyn ExternalFileResolver>,
    ) -> Self {
        self.external = external;
        self.resolver = resolver;
        self
    }

    /// Use `registry` to look up filters instead of the built-in registry.
    pub fn with_registry(mut self, registry: &'a FilterRegistry) -> Self {
        self.registry = registry;
        self
    }

    /// Replace the resource limits.
    pub fn with_limits(mut self, limits: ReadLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The dataspace.
    pub fn dataspace(&self) -> &'a Dataspace {
        self.dataspace
    }

    /// The layout.
    pub fn layout(&self) -> &'a DataLayout {
        self.layout
    }

    /// Element size in bytes.
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// The limits in force.
    pub fn limits(&self) -> ReadLimits {
        self.limits
    }

    /// Read the selected elements; see [`read_raw`].
    pub fn read(&self, selection: &Selection) -> Result<Vec<u8>> {
        read_raw(self, selection)
    }

    /// List the allocated chunks of a chunked dataset (diagnostics, tests,
    /// writers). See [`ChunkInfo`]. Fails for other layouts.
    pub fn chunks(&self) -> Result<Vec<ChunkInfo>> {
        match self.layout {
            DataLayout::Chunked(c) => chunked::list_chunks(self, c),
            _ => Err(Error::invalid("dataset is not chunked")),
        }
    }

    /// The bytes of one element of fill value: the defined fill value, or
    /// `None` when elements that were never written read as zero bytes.
    ///
    /// A defined fill value whose size differs from the element size is an
    /// error.
    pub fn fill_bytes(&self) -> Result<Option<&'a [u8]>> {
        let value = match self.fill_value {
            Some(f) => match f.state {
                FillValueState::Defined => f.value.as_deref(),
                FillValueState::Undefined => None,
                // Version 1/2 messages may still carry a value; otherwise
                // fall back to the old message.
                FillValueState::Default => f
                    .value
                    .as_deref()
                    .or(self.old_fill_value.map(|o| o.value.as_slice())),
            },
            None => self.old_fill_value.map(|o| o.value.as_slice()),
        };
        match value {
            Some([]) => Ok(None),
            Some(v) if v.len() != self.element_size => Err(Error::invalid(
                "fill value size does not match the element size",
            )),
            Some(v) if v.iter().all(|&b| b == 0) => Ok(None),
            other => Ok(other),
        }
    }
}

/// Read the elements of `selection` from `storage`.
///
/// Returns `selection.num_elements() * element_size` bytes: the selected
/// elements in row-major order of the selection, each exactly as stored on
/// disk. A scalar dataspace yields one element, a null dataspace nothing.
///
/// Only the parts of the file needed for the selection are read: for
/// chunked datasets only the chunks intersecting the selection, for
/// contiguous datasets only the byte ranges covering it (nearby ranges are
/// coalesced into larger reads).
pub fn read_raw(storage: &DatasetStorage<'_>, selection: &Selection) -> Result<Vec<u8>> {
    let sel = selection.resolve(storage.dataspace)?;
    if storage.element_size == 0 {
        return Err(Error::invalid("element size is zero"));
    }
    let out_len = usize::try_from(sel.count)
        .ok()
        .and_then(|n| n.checked_mul(storage.element_size))
        .filter(|&n| n <= storage.limits.max_output_bytes)
        .ok_or(Error::LimitExceeded("selection output size"))?;
    if let DataLayout::Virtual { .. } = storage.layout {
        return Err(Error::unsupported("virtual dataset layout"));
    }
    if out_len == 0 {
        return Ok(Vec::new());
    }
    let mut out = filled(out_len, storage.fill_bytes()?);
    match storage.layout {
        DataLayout::Compact { data, .. } => {
            contiguous::read_compact(storage, data, &sel, &mut out)?
        }
        DataLayout::Contiguous { address, size, .. } => {
            contiguous::read_contiguous(storage, *address, *size, &sel, &mut out)?
        }
        DataLayout::Chunked(c) => chunked::read_chunked(storage, c, &sel, &mut out)?,
        DataLayout::Virtual { .. } => return Err(Error::unsupported("virtual dataset layout")),
    }
    Ok(out)
}

/// A buffer of `len` bytes initialized with repetitions of `fill` (zeros
/// when `None`).
fn filled(len: usize, fill: Option<&[u8]>) -> Vec<u8> {
    match fill {
        None => vec![0; len],
        Some(pattern) => pattern.iter().copied().cycle().take(len).collect(),
    }
}

/// Number of elements of the dataspace and its byte size.
fn dataset_bytes(storage: &DatasetStorage<'_>) -> Result<u64> {
    storage
        .dataspace
        .element_count()?
        .checked_mul(storage.element_size as u64)
        .ok_or(Error::LimitExceeded("dataset size overflows"))
}
