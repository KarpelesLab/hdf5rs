//! The typed read API: [`DatasetReader`], [`Array`] and element decoding
//! with the file's global heap.

use std::cell::Cell;
use std::fmt;

use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::file::File;
use crate::filters::FilterRegistry;
use crate::format::global_heap::GlobalHeapId;
use crate::format::messages::datatype::Datatype;
use crate::format::messages::layout::DataLayout;
use crate::format::vds::{SAME_FILE, VirtualMapping};
use crate::io::ReadAt;
use crate::read::{ChunkInfo, DatasetStorage, ExternalFileResolver, ReadLimits, Selection};
use crate::types::{DecodeContext, GlobalHeapResolver, H5Type, Value, VlenResolver, decode_all};

/// Default maximum nesting of virtual datasets whose sources are virtual
/// datasets (see [`ReadLimits::max_virtual_depth`]; also stops
/// self-referencing mappings).
pub const MAX_VIRTUAL_DEPTH: u32 = 8;

/// An n-dimensional array of decoded elements: the data in row-major order
/// and its shape.
///
/// The shape is empty for scalar (one element) and null (no element)
/// dataspaces.
#[derive(Debug, Clone, PartialEq)]
pub struct Array<T> {
    /// Elements, row-major (last dimension fastest).
    pub data: Vec<T>,
    /// Dimensions.
    pub shape: Vec<u64>,
}

impl<T> Array<T> {
    /// Number of dimensions (0 for scalars).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The elements as a flat, row-major slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The element at multi-dimensional `index` (`None` when out of bounds
    /// or of the wrong rank). A scalar is indexed by `&[]`.
    pub fn get(&self, index: &[u64]) -> Option<&T> {
        if index.len() != self.shape.len() {
            return None;
        }
        let mut lin = 0u64;
        for (&i, &n) in index.iter().zip(&self.shape) {
            if i >= n {
                return None;
            }
            lin = lin.checked_mul(n)?.checked_add(i)?;
        }
        self.data.get(usize::try_from(lin).ok()?)
    }

    /// The flat data, dropping the shape.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Apply `f` to every element, keeping the shape.
    pub fn map<U>(self, f: impl FnMut(T) -> U) -> Array<U> {
        Array {
            data: self.data.into_iter().map(f).collect(),
            shape: self.shape,
        }
    }
}

/// A [`VlenResolver`] that fails once more than a byte budget of global
/// heap data has been resolved (amplification guard: many elements may
/// point to the same large heap object).
struct BudgetResolver<'a> {
    inner: GlobalHeapResolver<&'a dyn ReadAt>,
    left: Cell<usize>,
}

impl VlenResolver for BudgetResolver<'_> {
    fn global_heap_object(&self, id: &GlobalHeapId) -> Result<Vec<u8>> {
        let data = self.inner.global_heap_object(id)?;
        let left = self
            .left
            .get()
            .checked_sub(data.len())
            .ok_or(Error::LimitExceeded("variable-length data size"))?;
        self.left.set(left);
        Ok(data)
    }
}

/// Decode packed elements of `dt`, resolving variable-length data and
/// region references through `file`'s global heap (at most `max_heap_bytes`
/// of heap data in total).
pub(crate) fn decode_elements<T: H5Type>(
    file: &File,
    dt: &Datatype,
    raw: &[u8],
    max_heap_bytes: usize,
) -> Result<Vec<T>> {
    if raw.is_empty() {
        T::check(dt)?;
        return Ok(Vec::new());
    }
    let resolver = BudgetResolver {
        inner: GlobalHeapResolver::new(file.store().as_ref(), file.sizes()),
        left: Cell::new(max_heap_bytes),
    };
    let ctx = DecodeContext::with_resolver(file.sizes(), &resolver);
    decode_all(dt, raw, &ctx)
}

/// Exactly one element out of `v`, or an error naming `what`.
pub(crate) fn single<T>(mut v: Vec<T>, what: &str) -> Result<T> {
    if v.len() != 1 {
        return Err(Error::WrongObjectType(format!(
            "{what} has {} elements, expected exactly one",
            v.len()
        )));
    }
    v.pop()
        .ok_or_else(|| Error::WrongObjectType(format!("{what} is empty")))
}

/// A configurable read of a [`Dataset`], created by [`Dataset::reader`].
///
/// ```no_run
/// use hdf5rs::{File, Selection};
/// use hdf5rs::filters::FilterRegistry;
/// use hdf5rs::read::ReadLimits;
///
/// let file = File::open("data.h5")?;
/// let ds = file.dataset("/images")?;
/// let registry = FilterRegistry::with_builtin(); // plus your own filters
/// let block: Vec<u16> = ds
///     .reader()
///     .filters(&registry)
///     .limits(ReadLimits::with_max_output(64 << 20))
///     .read_selection(&Selection::from_ranges(&[0..1, 0..512, 0..512]))?;
/// # Ok::<(), hdf5rs::Error>(())
/// ```
#[derive(Clone, Copy)]
pub struct DatasetReader<'a> {
    dataset: &'a Dataset,
    registry: Option<&'a FilterRegistry>,
    limits: ReadLimits,
    resolver: Option<&'a dyn ExternalFileResolver>,
    use_cache: bool,
}

impl fmt::Debug for DatasetReader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DatasetReader")
            .field("dataset", &self.dataset.path())
            .field("custom_filters", &self.registry.is_some())
            .field("limits", &self.limits)
            .field("custom_resolver", &self.resolver.is_some())
            .field("use_chunk_cache", &self.use_cache)
            .finish()
    }
}

impl<'a> DatasetReader<'a> {
    pub(crate) fn new(dataset: &'a Dataset) -> Self {
        DatasetReader {
            dataset,
            registry: None,
            limits: dataset.file().access().limits,
            resolver: None,
            use_cache: true,
        }
    }

    /// Look up filters in `registry` instead of the built-in registry.
    pub fn filters(mut self, registry: &'a FilterRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Replace the resource limits (output size, decoded chunk size); the
    /// default comes from the file's
    /// [`FileAccess::read_limits`](crate::FileAccess::read_limits).
    pub fn limits(mut self, limits: ReadLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Use the dataset's [`ChunkCache`](crate::read::ChunkCache) (the
    /// default) or bypass it, e.g. for a one-off scan that should not
    /// evict cached chunks.
    pub fn use_chunk_cache(mut self, enabled: bool) -> Self {
        self.use_cache = enabled;
        self
    }

    /// Open external raw data files with `resolver` instead of the default
    /// (relative to the HDF5 file's directory, per its
    /// [`FileAccess`](crate::FileAccess) policy).
    pub fn external_resolver(mut self, resolver: &'a dyn ExternalFileResolver) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// The raw bytes of every element, in file byte order (see
    /// [`read_raw`](crate::read::read_raw)).
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        self.raw(&Selection::All, 0)
    }

    /// The raw bytes of the selected elements, in file byte order.
    pub fn read_raw_selection(&self, selection: &Selection) -> Result<Vec<u8>> {
        self.raw(selection, 0)
    }

    /// Every element, decoded into `T`, in row-major order.
    pub fn read<T: H5Type>(&self) -> Result<Vec<T>> {
        self.read_selection(&Selection::All)
    }

    /// The selected elements, decoded into `T`, in row-major order of the
    /// selection.
    pub fn read_selection<T: H5Type>(&self, selection: &Selection) -> Result<Vec<T>> {
        let dt = self.dataset.datatype();
        T::check(dt)?;
        let raw = self.raw(selection, 0)?;
        decode_elements(self.dataset.file(), dt, &raw, self.limits.max_output_bytes)
    }

    /// Every element with the dataset's shape.
    pub fn read_nd<T: H5Type>(&self) -> Result<Array<T>> {
        self.read_selection_nd(&Selection::All)
    }

    /// The selected elements with the selection's shape.
    pub fn read_selection_nd<T: H5Type>(&self, selection: &Selection) -> Result<Array<T>> {
        let shape = selection.output_shape(self.dataset.dataspace())?;
        let data = self.read_selection(selection)?;
        Ok(Array { data, shape })
    }

    /// The single element of a scalar (or one-element) dataset.
    pub fn read_scalar<T: H5Type>(&self) -> Result<T> {
        let n = self.dataset.element_count()?;
        if n != 1 {
            return Err(Error::WrongObjectType(format!(
                "dataset {} has {n} elements, expected exactly one",
                self.dataset.path()
            )));
        }
        single(self.read::<T>()?, self.dataset.path())
    }

    /// Every element as a dynamic [`Value`].
    pub fn read_values(&self) -> Result<Vec<Value>> {
        self.read()
    }

    /// The allocated chunks of a chunked dataset (diagnostics).
    pub fn chunks(&self) -> Result<Vec<ChunkInfo>> {
        let ds = self.dataset;
        let file = ds.file();
        let mut storage = DatasetStorage::new(
            file.store().as_ref(),
            file.sizes(),
            ds.layout(),
            ds.dataspace(),
            ds.datatype().size as usize,
        )
        .with_filters(ds.filters())
        .with_limits(self.limits);
        if let Some(r) = self.registry {
            storage = storage.with_registry(r);
        }
        if self.use_cache {
            storage = storage.with_chunk_cache(ds.chunk_cache());
        }
        storage.chunks()
    }

    fn raw(&self, selection: &Selection, depth: u32) -> Result<Vec<u8>> {
        let ds = self.dataset;
        let file = ds.file();
        let default_resolver = match self.resolver {
            Some(_) => None,
            None => file.external_resolver(),
        };
        let resolver: Option<&dyn ExternalFileResolver> = match self.resolver {
            Some(r) => Some(r),
            None => default_resolver
                .as_ref()
                .map(|r| r as &dyn ExternalFileResolver),
        };
        let mut storage = DatasetStorage::new(
            file.store().as_ref(),
            file.sizes(),
            ds.layout(),
            ds.dataspace(),
            ds.datatype().size as usize,
        )
        .with_filters(ds.filters())
        .with_fill_value(ds.fill_value())
        .with_old_fill_value(ds.fill_value_old())
        .with_external(ds.external_files_message(), resolver)
        .with_limits(self.limits);
        if let Some(r) = self.registry {
            storage = storage.with_registry(r);
        }
        if self.use_cache {
            storage = storage.with_chunk_cache(ds.chunk_cache());
        }
        match ds.layout() {
            DataLayout::Virtual { .. } => {
                if depth >= self.limits.max_virtual_depth {
                    return Err(Error::LimitExceeded("virtual dataset nesting depth"));
                }
                let mappings = ds.virtual_mappings()?;
                crate::read::read_virtual(&storage, &mappings, selection, |m, sel| {
                    self.read_source(m, sel, depth)
                })
            }
            _ => storage.read(selection),
        }
    }

    /// Raw bytes of the source selection of a virtual dataset mapping;
    /// `None` when the source file or dataset does not exist.
    fn read_source(
        &self,
        m: &VirtualMapping,
        selection: &Selection,
        depth: u32,
    ) -> Result<Option<Vec<u8>>> {
        let ds = self.dataset;
        let file = if m.source_file == SAME_FILE {
            ds.file().clone()
        } else {
            match ds.file().open_related(&m.source_file) {
                Ok(f) => f,
                Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => return Err(e),
            }
        };
        let src = match file.dataset(&m.source_dataset) {
            Ok(d) => d,
            Err(Error::NotFound(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        if src.datatype().size != ds.datatype().size {
            return Err(Error::unsupported(format!(
                "virtual dataset source {}:{} has a different element size",
                m.source_file, m.source_dataset
            )));
        }
        let reader = DatasetReader {
            dataset: &src,
            registry: self.registry,
            limits: self.limits,
            resolver: self.resolver,
            use_cache: self.use_cache,
        };
        reader.raw(selection, depth + 1).map(Some)
    }
}
