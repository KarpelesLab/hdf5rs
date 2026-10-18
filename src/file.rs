//! The [`File`] handle: opening a store, object header caching and shared
//! message resolution.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::format::fractal_heap::FractalHeap;
use crate::format::messages::MessageType;
use crate::format::messages::shared::{SharedLocation, SharedMessage};
use crate::format::object_header::{ObjectHeader, RawMessage};
use crate::format::sohm::{SharedMessageTableMessage, SohmTable};
use crate::format::superblock::Superblock;
use crate::format::{Sizes, UNDEFINED_ADDRESS};
use crate::group::Group;
use crate::io::ReadAt;
use crate::object::Object;
use crate::read::{DEFAULT_CHUNK_CACHE_BYTES, ExternalFileResolver, ReadLimits};

/// A thread-safe, shareable byte store.
pub type SharedStore = Arc<dyn ReadAt + Send + Sync>;

/// Number of parsed object headers kept in the cache before it is reset.
const HEADER_CACHE_CAPACITY: usize = 1024;

/// Maximum chain length when a shared message points to another shared
/// message.
const MAX_SHARED_DEPTH: u32 = 8;

/// Policy for accessing files other than the HDF5 file itself.
///
/// Three features may open other files, always resolved relative to the
/// directory of the file opened with [`File::open`] / [`File::open_with`]
/// (files opened from memory or a custom reader cannot open other files):
///
/// - **external raw data files** (External Data Files message);
/// - **virtual dataset sources** stored in other files;
/// - **external links**, which are only followed automatically when
///   [`follow_external_links`](Self::follow_external_links) is enabled, or
///   explicitly through [`Group::follow_external`].
///
/// # Security
///
/// The file names come from the HDF5 file. By default they are confined to
/// the file's directory and its subdirectories: absolute names and names
/// with `..` components are rejected, so a hostile file cannot make the
/// reader return the contents of arbitrary files (symbolic links inside
/// the directory are still followed). Files that legitimately point
/// elsewhere, as the reference library allows, need
/// `confine_to_directory(false)`; only use it for trusted files. To forbid
/// opening other files at all, use `external_files(false)`.
///
/// # Resource limits
///
/// [`read_limits`](Self::read_limits) sets the default [`ReadLimits`] of
/// every read of the file (datasets and attributes); see [`ReadLimits`]
/// for the complete list of bounds applied to untrusted input.
/// [`chunk_cache_bytes`](Self::chunk_cache_bytes) sizes each dataset's
/// decoded-chunk cache.
///
/// ```
/// use hdf5rs::FileAccess;
/// use hdf5rs::read::ReadLimits;
///
/// // Settings for files from an untrusted source.
/// let access = FileAccess::default()
///     .external_files(false)
///     .read_limits(ReadLimits::with_max_output(64 << 20).max_chunk_bytes(16 << 20))
///     .chunk_cache_bytes(4 << 20);
/// # let _ = access;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct FileAccess {
    /// Allow opening other files at all (default `true`).
    pub external_files: bool,
    /// Reject absolute names and `..` components (default `true`).
    pub confine_to_directory: bool,
    /// Follow external links during path resolution (default `false`).
    pub follow_external_links: bool,
    /// Default limits of every read (default [`ReadLimits::default`]).
    pub limits: ReadLimits,
    /// Capacity in bytes of each dataset's decoded-chunk cache (default
    /// [`DEFAULT_CHUNK_CACHE_BYTES`]; 0 disables caching).
    pub chunk_cache_bytes: usize,
}

impl Default for FileAccess {
    fn default() -> Self {
        FileAccess {
            external_files: true,
            confine_to_directory: true,
            follow_external_links: false,
            limits: ReadLimits::default(),
            chunk_cache_bytes: DEFAULT_CHUNK_CACHE_BYTES,
        }
    }
}

impl FileAccess {
    /// Allow or forbid opening other files.
    pub fn external_files(mut self, allow: bool) -> Self {
        self.external_files = allow;
        self
    }

    /// Confine other files to the HDF5 file's directory tree (the
    /// default); `false` also honours absolute names and `..` components.
    pub fn confine_to_directory(mut self, confine: bool) -> Self {
        self.confine_to_directory = confine;
        self
    }

    /// Follow external links automatically when resolving paths.
    pub fn follow_external_links(mut self, follow: bool) -> Self {
        self.follow_external_links = follow;
        self
    }

    /// Set the default limits of every read of the file.
    pub fn read_limits(mut self, limits: ReadLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Set the capacity of each dataset's decoded-chunk cache (0 disables
    /// it).
    pub fn chunk_cache_bytes(mut self, bytes: usize) -> Self {
        self.chunk_cache_bytes = bytes;
        self
    }
}

/// An open HDF5 file (read-only).
///
/// Cloning is cheap: clones share the underlying store and caches. `File`
/// is `Send + Sync`.
#[derive(Clone)]
pub struct File {
    inner: Arc<FileInner>,
}

struct FileInner {
    /// Store whose positions are relative file addresses.
    store: SharedStore,
    superblock: Superblock,
    path: Option<PathBuf>,
    access: FileAccess,
    headers: Mutex<HashMap<u64, Arc<ObjectHeader>>>,
    /// `None` = not loaded yet; `Some(None)` = file has no SOHM table.
    sohm: Mutex<Option<Option<Arc<SohmTable>>>>,
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File")
            .field("path", &self.inner.path)
            .field("superblock_version", &self.inner.superblock.version)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl File {
    /// Open the HDF5 file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<File> {
        Self::open_with(path, FileAccess::default())
    }

    /// Open the HDF5 file at `path` with a policy for accessing other files
    /// (see [`FileAccess`]).
    pub fn open_with(path: impl AsRef<Path>, access: FileAccess) -> Result<File> {
        let path = path.as_ref();
        let f = std::fs::File::open(path)?;
        Self::with_store(Arc::new(f), Some(path.to_path_buf()), access)
    }

    /// Open an HDF5 file held in memory.
    pub fn from_bytes(data: Vec<u8>) -> Result<File> {
        Self::from_bytes_with(data, FileAccess::default())
    }

    /// Open an HDF5 file held in memory with the given access policy and
    /// limits (other files can never be opened from such a file).
    pub fn from_bytes_with(data: Vec<u8>, access: FileAccess) -> Result<File> {
        Self::with_store(Arc::new(data), None, access)
    }

    /// Open an HDF5 file from any positional reader.
    pub fn from_reader<R: ReadAt + Send + Sync + 'static>(reader: R) -> Result<File> {
        Self::from_reader_with(reader, FileAccess::default())
    }

    /// Open an HDF5 file from any positional reader with the given access
    /// policy and limits (other files can never be opened from such a
    /// file).
    pub fn from_reader_with<R: ReadAt + Send + Sync + 'static>(
        reader: R,
        access: FileAccess,
    ) -> Result<File> {
        Self::with_store(Arc::new(reader), None, access)
    }

    /// The policy for accessing other files.
    pub fn access(&self) -> FileAccess {
        self.inner.access
    }

    /// Resolve `name` (a file name stored in this file: external data file,
    /// virtual dataset source or external link target) against this file's
    /// directory, honouring the [`FileAccess`] policy.
    ///
    /// Fails with [`Error::Unsupported`] when the file was not opened from
    /// a path or access to other files is disabled, and with
    /// [`Error::InvalidFormat`] when a confined name escapes the directory.
    pub fn resolve_related_path(&self, name: &str) -> Result<PathBuf> {
        let access = self.inner.access;
        if !access.external_files {
            return Err(Error::unsupported(format!(
                "access to other files is disabled (needed for {name:?})"
            )));
        }
        let Some(path) = self.inner.path.as_deref() else {
            return Err(Error::unsupported(format!(
                "cannot open {name:?}: the file was not opened from a path"
            )));
        };
        let dir = path.parent().unwrap_or_else(|| Path::new(""));
        let rel = Path::new(name);
        if name.is_empty() {
            return Err(Error::invalid("empty external file name"));
        }
        if access.confine_to_directory {
            use std::path::Component;
            if rel.components().any(|c| {
                matches!(
                    c,
                    Component::RootDir | Component::Prefix(_) | Component::ParentDir
                )
            }) {
                return Err(Error::invalid(format!(
                    "external file name {name:?} escapes the file's directory"
                )));
            }
        }
        Ok(dir.join(rel))
    }

    /// Open another HDF5 file named in this one (virtual dataset source,
    /// external link target), relative to this file's directory and with
    /// the same [`FileAccess`] policy. See
    /// [`resolve_related_path`](Self::resolve_related_path).
    pub fn open_related(&self, name: &str) -> Result<File> {
        let p = self.resolve_related_path(name)?;
        File::open_with(p, self.inner.access)
    }

    /// The default resolver for external raw data files: names relative to
    /// this file's directory, per the [`FileAccess`] policy. `None` when
    /// other files cannot be opened.
    pub(crate) fn external_resolver(&self) -> Option<RelativeResolver> {
        if !self.inner.access.external_files || self.inner.path.is_none() {
            return None;
        }
        Some(RelativeResolver { file: self.clone() })
    }

    fn with_store(raw: SharedStore, path: Option<PathBuf>, access: FileAccess) -> Result<File> {
        let superblock = Superblock::read(&raw)?;
        let store: SharedStore = if superblock.base() == 0 {
            raw
        } else {
            Arc::new(superblock.relative_store(raw))
        };
        Ok(File {
            inner: Arc::new(FileInner {
                store,
                superblock,
                path,
                access,
                headers: Mutex::new(HashMap::new()),
                sohm: Mutex::new(None),
            }),
        })
    }

    /// The parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.inner.superblock
    }

    /// Offset and length widths used by this file.
    pub fn sizes(&self) -> Sizes {
        self.inner.superblock.sizes
    }

    /// The underlying store, positioned so that file addresses can be used
    /// directly (any user block is skipped).
    pub fn store(&self) -> &SharedStore {
        &self.inner.store
    }

    /// The path the file was opened from, if opened with [`File::open`].
    /// Useful to resolve external data files relative to it.
    pub fn path(&self) -> Option<&Path> {
        self.inner.path.as_deref()
    }

    /// The root group.
    pub fn root(&self) -> Result<Group> {
        let addr = self.inner.superblock.root_object_header_address();
        match self.object_at(addr, "/")? {
            Object::Group(g) => Ok(g),
            _ => Err(Error::invalid("root object is not a group")),
        }
    }

    /// The object at an absolute (or root-relative) `path`.
    pub fn object(&self, path: &str) -> Result<Object> {
        self.root()?.object(path)
    }

    /// The group at `path`.
    pub fn group(&self, path: &str) -> Result<Group> {
        self.root()?.group(path)
    }

    /// The dataset at `path`.
    pub fn dataset(&self, path: &str) -> Result<Dataset> {
        self.root()?.dataset(path)
    }

    /// Open the object whose header is at `address`, recording `path` as
    /// its name.
    pub fn object_at(&self, address: u64, path: &str) -> Result<Object> {
        let header = self.header(address)?;
        Object::from_header(self.clone(), header, path.into())
    }

    /// The parsed object header at `address` (cached).
    pub fn header(&self, address: u64) -> Result<Arc<ObjectHeader>> {
        if address == UNDEFINED_ADDRESS {
            return Err(Error::invalid("object header at undefined address"));
        }
        if let Some(h) = lock(&self.inner.headers).get(&address) {
            return Ok(h.clone());
        }
        let h = Arc::new(ObjectHeader::read(
            self.inner.store.as_ref(),
            address,
            self.sizes(),
        )?);
        let mut cache = lock(&self.inner.headers);
        if cache.len() >= HEADER_CACHE_CAPACITY {
            cache.clear();
        }
        cache.insert(address, h.clone());
        Ok(h)
    }

    /// The superblock extension object header (versions 2/3), if any.
    pub fn superblock_extension(&self) -> Result<Option<Arc<ObjectHeader>>> {
        let addr = self.inner.superblock.extension_address;
        if addr == UNDEFINED_ADDRESS {
            return Ok(None);
        }
        self.header(addr).map(Some)
    }

    /// The shared object header message table, if the file has one.
    pub fn shared_message_table(&self) -> Result<Option<Arc<SohmTable>>> {
        if let Some(t) = lock(&self.inner.sohm).as_ref() {
            return Ok(t.clone());
        }
        let table = match self.superblock_extension()? {
            None => None,
            Some(ext) => match ext.find_message(MessageType::SHARED_MESSAGE_TABLE) {
                None => None,
                Some(m) => {
                    let msg = SharedMessageTableMessage::parse(&m.data, self.sizes())?;
                    Some(Arc::new(SohmTable::read(
                        self.inner.store.as_ref(),
                        msg.table_address,
                        usize::from(msg.num_indexes),
                        self.sizes(),
                    )?))
                }
            },
        };
        *lock(&self.inner.sohm) = Some(table.clone());
        Ok(table)
    }

    /// The body of `msg` (of type `msg_type`), resolving it if it is a
    /// shared message stored elsewhere.
    pub fn message_body(&self, msg: &RawMessage) -> Result<Vec<u8>> {
        if msg.is_shared() {
            self.shared_message_body(msg.msg_type, &msg.data)
        } else {
            Ok(msg.data.clone())
        }
    }

    /// Resolve a shared message encoding (`shared`) of a message of type
    /// `msg_type` to the message body it refers to.
    pub fn shared_message_body(&self, msg_type: u16, shared: &[u8]) -> Result<Vec<u8>> {
        self.resolve_shared(msg_type, shared, 0)
    }

    /// The object header address a shared message encoding points to, if
    /// it is shared through an object header (e.g. a committed datatype).
    pub fn shared_message_address(&self, shared: &[u8]) -> Result<Option<u64>> {
        Ok(match SharedMessage::parse(shared, self.sizes())?.location {
            SharedLocation::ObjectHeader(a) => Some(a),
            _ => None,
        })
    }

    fn resolve_shared(&self, msg_type: u16, shared: &[u8], depth: u32) -> Result<Vec<u8>> {
        if depth >= MAX_SHARED_DEPTH {
            return Err(Error::LimitExceeded("shared message chain length"));
        }
        match SharedMessage::parse(shared, self.sizes())?.location {
            SharedLocation::ObjectHeader(addr) => {
                let header = self.header(addr)?;
                let m = header.find_message(msg_type).ok_or_else(|| {
                    Error::invalid(format!(
                        "shared {} message not found in object header {addr:#x}",
                        MessageType::name(msg_type)
                    ))
                })?;
                if m.is_shared() {
                    self.resolve_shared(msg_type, &m.data, depth + 1)
                } else {
                    Ok(m.data.clone())
                }
            }
            SharedLocation::SharedHeap(id) => {
                let table = self.shared_message_table()?.ok_or_else(|| {
                    Error::invalid("message stored in the shared message heap, but no SOHM table")
                })?;
                let index = table.index_for(msg_type).ok_or_else(|| {
                    Error::invalid(format!(
                        "no SOHM index for {} messages",
                        MessageType::name(msg_type)
                    ))
                })?;
                let store = self.inner.store.as_ref();
                let heap = FractalHeap::open(store, index.heap_address, self.sizes())?;
                heap.get(store, &id)
            }
            SharedLocation::NotShared | SharedLocation::Shareable => Err(Error::invalid(
                "message flagged shared but encoding says it is not shared",
            )),
        }
    }
}

/// Opens external raw data files relative to an HDF5 file's directory.
pub(crate) struct RelativeResolver {
    file: File,
}

impl ExternalFileResolver for RelativeResolver {
    fn open(&self, name: &str) -> Result<Box<dyn ReadAt>> {
        let p = self.file.resolve_related_path(name)?;
        Ok(Box::new(std::fs::File::open(p)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_sync() {
        fn check<T: Send + Sync + Clone>() {}
        check::<File>();
        check::<Group>();
        check::<Dataset>();
        check::<Object>();
        check::<crate::Attribute>();
    }
}
