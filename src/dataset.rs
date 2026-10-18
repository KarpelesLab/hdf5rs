//! Datasets and committed (named) datatypes.

use std::sync::Arc;

use crate::attribute::resolve_datatype;
use crate::error::{Error, Result};
use crate::format::global_heap::GlobalHeapId;
use crate::format::local_heap::LocalHeap;
use crate::format::messages::MessageType;
use crate::format::messages::attribute::MaybeShared;
use crate::format::messages::dataspace::Dataspace;
use crate::format::messages::datatype::Datatype;
use crate::format::messages::external_files::ExternalFiles;
use crate::format::messages::fill_value::{FillValue, FillValueState, OldFillValue};
use crate::format::messages::filter_pipeline::FilterPipeline;
use crate::format::messages::layout::DataLayout;
use crate::format::object_header::RawMessage;
use crate::format::vds::{VirtualMapping, parse_virtual_mappings};
use crate::object::{Node, object_common};
use crate::read::{ChunkCache, ChunkInfo, Selection};
use crate::reader::{Array, DatasetReader};
use crate::types::{GlobalHeapResolver, H5Type, Value, VlenResolver};

/// One segment of a dataset stored in external files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalFile {
    /// File name (resolved from the local heap), as stored.
    pub name: String,
    /// Byte offset of the segment within that file.
    pub file_offset: u64,
    /// Size of the segment in bytes ([`u64::MAX`]-like "unlimited" values
    /// are passed through as stored).
    pub size: u64,
}

/// Decoded metadata of a dataset.
#[derive(Debug)]
struct Meta {
    datatype: Datatype,
    committed_datatype: Option<u64>,
    dataspace: Dataspace,
    layout: DataLayout,
    filters: Option<FilterPipeline>,
    fill_value: Option<FillValue>,
    fill_value_old: Option<OldFillValue>,
    external_files: Option<Vec<ExternalFile>>,
    external_message: Option<ExternalFiles>,
    /// Decoded chunks and chunk index (shared by clones).
    cache: ChunkCache,
}

/// A dataset: a multidimensional array of elements of one datatype.
///
/// All storage-describing messages are decoded (and shared messages
/// resolved) when the dataset is opened. Cheap to clone; `Send + Sync`.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub(crate) node: Node,
    meta: Arc<Meta>,
}

object_common!(Dataset);

impl Dataset {
    pub(crate) fn new(node: Node) -> Result<Dataset> {
        let file = &node.file;
        let sizes = file.sizes();
        let header = &node.header;
        let body = |t: u16| -> Result<Option<Vec<u8>>> {
            header
                .find_message(t)
                .map(|m| file.message_body(m))
                .transpose()
        };
        let required = |t: u16| -> Result<Vec<u8>> {
            body(t)?.ok_or_else(|| {
                Error::invalid(format!(
                    "dataset {} has no {} message",
                    node.path,
                    MessageType::name(t)
                ))
            })
        };

        let dt_msg = header
            .find_message(MessageType::DATATYPE)
            .ok_or_else(|| Error::invalid(format!("dataset {} has no datatype", node.path)))?;
        let field = if dt_msg.is_shared() {
            MaybeShared::Shared(dt_msg.data.clone())
        } else {
            MaybeShared::Inline(Datatype::parse(&dt_msg.data)?)
        };
        let (datatype, committed_datatype) = resolve_datatype(file, field)?;
        let dataspace = Dataspace::parse(&required(MessageType::DATASPACE)?, sizes)?;
        let layout = DataLayout::parse(&required(MessageType::LAYOUT)?, sizes)?;
        let filters = body(MessageType::FILTER_PIPELINE)?
            .map(|b| FilterPipeline::parse(&b, sizes))
            .transpose()?;
        let fill_value = body(MessageType::FILL_VALUE)?
            .map(|b| FillValue::parse(&b, sizes))
            .transpose()?;
        let fill_value_old = body(MessageType::FILL_VALUE_OLD)?
            .map(|b| OldFillValue::parse(&b, sizes))
            .transpose()?;
        let external_message = body(MessageType::EXTERNAL_FILES)?
            .map(|b| ExternalFiles::parse(&b, sizes))
            .transpose()?;
        let external_files = match &external_message {
            None => None,
            Some(ef) => {
                let heap = LocalHeap::read(file.store().as_ref(), ef.heap_address, sizes)?;
                Some(
                    ef.slots
                        .iter()
                        .map(|s| {
                            Ok(ExternalFile {
                                name: heap.get_string_lossy(s.name_offset)?.into_owned(),
                                file_offset: s.file_offset,
                                size: s.size,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            }
        };
        let meta = Meta {
            datatype,
            committed_datatype,
            dataspace,
            layout,
            filters,
            fill_value,
            fill_value_old,
            external_files,
            external_message,
            cache: ChunkCache::new(file.access().chunk_cache_bytes),
        };
        Ok(Dataset {
            node,
            meta: Arc::new(meta),
        })
    }

    /// The element datatype (resolved if committed or shared).
    pub fn datatype(&self) -> &Datatype {
        &self.meta.datatype
    }

    /// Address of the committed datatype object used by this dataset, if
    /// its datatype is committed.
    pub fn committed_datatype_address(&self) -> Option<u64> {
        self.meta.committed_datatype
    }

    /// The dataspace.
    pub fn dataspace(&self) -> &Dataspace {
        &self.meta.dataspace
    }

    /// Current dimensions (empty for scalar and null dataspaces).
    pub fn shape(&self) -> &[u64] {
        &self.meta.dataspace.dims
    }

    /// Maximum dimensions ([`UNLIMITED`](crate::format::messages::dataspace::UNLIMITED)
    /// for unlimited ones); equal to [`shape`](Self::shape) when not stored.
    pub fn max_shape(&self) -> Vec<u64> {
        match &self.meta.dataspace.max_dims {
            Some(m) => m.clone(),
            None => self.meta.dataspace.dims.clone(),
        }
    }

    /// Number of elements.
    pub fn element_count(&self) -> Result<u64> {
        self.meta.dataspace.element_count()
    }

    /// The storage layout.
    pub fn layout(&self) -> &DataLayout {
        &self.meta.layout
    }

    /// The filter pipeline, if any.
    pub fn filters(&self) -> Option<&FilterPipeline> {
        self.meta.filters.as_ref()
    }

    /// The fill value message (type 0x0005), if present.
    pub fn fill_value(&self) -> Option<&FillValue> {
        self.meta.fill_value.as_ref()
    }

    /// The old fill value message (type 0x0004), if present.
    pub fn fill_value_old(&self) -> Option<&OldFillValue> {
        self.meta.fill_value_old.as_ref()
    }

    /// The effective fill value bytes: the new message's value when one is
    /// defined, else the old message's value. `None` means the default
    /// (all zero bytes) applies.
    pub fn fill_value_bytes(&self) -> Option<&[u8]> {
        if let Some(fv) = &self.meta.fill_value {
            if fv.state == FillValueState::Defined {
                return fv.value.as_deref();
            }
            if fv.value.is_some() {
                return fv.value.as_deref();
            }
        }
        self.meta
            .fill_value_old
            .as_ref()
            .map(|f| f.value.as_slice())
            .filter(|v| !v.is_empty())
    }

    /// External data file segments, when the raw data lives outside the
    /// HDF5 file. Names are relative to [`File::path`](crate::File::path)'s
    /// directory unless absolute.
    pub fn external_files(&self) -> Option<&[ExternalFile]> {
        self.meta.external_files.as_deref()
    }

    /// Chunk dimensions for chunked datasets.
    pub fn chunk_shape(&self) -> Option<&[u64]> {
        match &self.meta.layout {
            DataLayout::Chunked(c) => Some(&c.dims),
            _ => None,
        }
    }

    /// Whether the data is stored in the object header.
    pub fn is_compact(&self) -> bool {
        matches!(self.meta.layout, DataLayout::Compact { .. })
    }

    /// Whether the data is one contiguous block.
    pub fn is_contiguous(&self) -> bool {
        matches!(self.meta.layout, DataLayout::Contiguous { .. })
    }

    /// Whether the data is chunked.
    pub fn is_chunked(&self) -> bool {
        matches!(self.meta.layout, DataLayout::Chunked(_))
    }

    /// Whether this is a virtual dataset.
    pub fn is_virtual(&self) -> bool {
        matches!(self.meta.layout, DataLayout::Virtual { .. })
    }

    /// The raw (possibly shared) header message of a given type, for
    /// messages without a dedicated accessor.
    pub fn raw_message(&self, msg_type: u16) -> Option<&RawMessage> {
        self.node.header.find_message(msg_type)
    }

    /// The parsed External Data Files message, if any.
    pub fn external_files_message(&self) -> Option<&ExternalFiles> {
        self.meta.external_message.as_ref()
    }

    /// This dataset's cache of decoded chunks and chunk index lookups
    /// (shared by clones of this handle; sized by
    /// [`FileAccess::chunk_cache_bytes`](crate::FileAccess::chunk_cache_bytes),
    /// adjustable with [`ChunkCache::set_capacity`]).
    pub fn chunk_cache(&self) -> &ChunkCache {
        &self.meta.cache
    }

    /// The mappings of a virtual dataset (empty for other layouts).
    pub fn virtual_mappings(&self) -> Result<Vec<VirtualMapping>> {
        let DataLayout::Virtual {
            heap_address,
            heap_index,
        } = self.meta.layout
        else {
            return Ok(Vec::new());
        };
        let file = &self.node.file;
        let resolver = GlobalHeapResolver::new(file.store().as_ref(), file.sizes());
        let block = resolver.global_heap_object(&GlobalHeapId {
            collection_address: heap_address,
            index: heap_index,
        })?;
        parse_virtual_mappings(&block, file.sizes())
    }

    // ------------------------------------------------------------ reading

    /// A configurable reader (custom filters, limits, external file
    /// resolver). The `read*` methods of `Dataset` use the defaults.
    pub fn reader(&self) -> DatasetReader<'_> {
        DatasetReader::new(self)
    }

    /// Read every element into `T`, in row-major order.
    ///
    /// Variable-length data and region references are resolved through the
    /// file; virtual datasets are assembled from their sources; external
    /// raw data files are opened relative to the file (see
    /// [`FileAccess`](crate::FileAccess)).
    pub fn read<T: H5Type>(&self) -> Result<Vec<T>> {
        self.reader().read()
    }

    /// Read the selected elements into `T`, in row-major order of the
    /// selection.
    pub fn read_selection<T: H5Type>(&self, selection: &Selection) -> Result<Vec<T>> {
        self.reader().read_selection(selection)
    }

    /// Read every element together with the dataset's shape.
    pub fn read_nd<T: H5Type>(&self) -> Result<Array<T>> {
        self.reader().read_nd()
    }

    /// Read the selected elements together with the selection's shape.
    pub fn read_selection_nd<T: H5Type>(&self, selection: &Selection) -> Result<Array<T>> {
        self.reader().read_selection_nd(selection)
    }

    /// Read the single element of a scalar (or one-element) dataset.
    pub fn read_scalar<T: H5Type>(&self) -> Result<T> {
        self.reader().read_scalar()
    }

    /// Read every element as a dynamic [`Value`].
    pub fn read_values(&self) -> Result<Vec<Value>> {
        self.reader().read_values()
    }

    /// Read the raw bytes of every element, in file byte order.
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        self.reader().read_raw()
    }

    /// Read the raw bytes of the selected elements, in file byte order.
    pub fn read_raw_selection(&self, selection: &Selection) -> Result<Vec<u8>> {
        self.reader().read_raw_selection(selection)
    }

    /// The allocated chunks of a chunked dataset (diagnostics; fails for
    /// other layouts).
    pub fn chunks(&self) -> Result<Vec<ChunkInfo>> {
        self.reader().chunks()
    }
}

/// A committed (named) datatype stored as an object in the file.
#[derive(Debug, Clone)]
pub struct NamedDatatype {
    pub(crate) node: Node,
    datatype: Arc<Datatype>,
}

object_common!(NamedDatatype);

impl NamedDatatype {
    pub(crate) fn new(node: Node) -> Result<NamedDatatype> {
        let m = node
            .header
            .find_message(MessageType::DATATYPE)
            .ok_or_else(|| Error::invalid("committed datatype without a datatype message"))?;
        let body = node.file.message_body(m)?;
        let datatype = Arc::new(Datatype::parse(&body)?);
        Ok(NamedDatatype { node, datatype })
    }

    /// The datatype.
    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }
}
