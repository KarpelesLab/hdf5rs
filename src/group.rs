//! Groups: link listing, lookup and path resolution.

use std::cmp::Ordering;

use crate::checksum::lookup3;
use crate::dataset::{Dataset, NamedDatatype};
use crate::error::{Error, Result};
use crate::format::btree_v1::{MAX_ENTRIES, group_btree_children};
use crate::format::btree_v2::{BTreeV2, Record, RecordContext};
use crate::format::fractal_heap::FractalHeap;
use crate::format::local_heap::LocalHeap;
use crate::format::messages::MessageType;
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::symbol_table::SymbolTableMessage;
use crate::format::symbol_table_node::SymbolTableNode;
use crate::format::{ScratchPad, UNDEFINED_ADDRESS};
use crate::link::{Link, LinkEntry, LinkStorage};
use crate::object::{Node, Object, object_common};

/// Maximum number of soft links followed while resolving one path.
pub const MAX_SOFT_LINK_HOPS: usize = 40;

/// A group: a collection of named links.
///
/// Cheap to clone; `Send + Sync`.
#[derive(Debug, Clone)]
pub struct Group {
    pub(crate) node: Node,
}

object_common!(Group);

/// Where a group's links live, decoded from its header.
enum Storage {
    SymbolTable(SymbolTableMessage),
    Links(Option<LinkInfoMessage>),
}

impl Group {
    pub(crate) fn new(node: Node) -> Group {
        Group { node }
    }

    fn storage(&self) -> Result<Storage> {
        let h = &self.node.header;
        let sizes = self.node.file.sizes();
        if let Some(m) = h.find_message(MessageType::SYMBOL_TABLE) {
            return Ok(Storage::SymbolTable(SymbolTableMessage::parse(
                &m.data, sizes,
            )?));
        }
        match h.find_message(MessageType::LINK_INFO) {
            Some(m) => Ok(Storage::Links(Some(LinkInfoMessage::parse(
                &m.data, sizes,
            )?))),
            None => Ok(Storage::Links(None)),
        }
    }

    /// How this group stores its links.
    pub fn link_storage(&self) -> Result<LinkStorage> {
        Ok(match self.storage()? {
            Storage::SymbolTable(_) => LinkStorage::SymbolTable,
            Storage::Links(Some(info)) if info.is_dense() => LinkStorage::Dense,
            Storage::Links(_) => LinkStorage::Compact,
        })
    }

    /// Whether link creation order is tracked.
    pub fn creation_order_tracked(&self) -> Result<bool> {
        Ok(match self.storage()? {
            Storage::Links(Some(info)) => info.creation_order_tracked,
            _ => false,
        })
    }

    /// All links, sorted by name (byte order).
    pub fn members(&self) -> Result<Vec<LinkEntry>> {
        let mut v = self.read_links()?;
        v.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
        Ok(v)
    }

    /// All links in creation order. Fails with [`Error::Unsupported`] if
    /// the group does not track creation order.
    pub fn members_by_creation_order(&self) -> Result<Vec<LinkEntry>> {
        if !self.creation_order_tracked()? {
            return Err(Error::unsupported(
                "link creation order is not tracked for this group",
            ));
        }
        let mut v = self.read_links()?;
        v.sort_by(|a, b| match (a.creation_order, b.creation_order) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.name.as_bytes().cmp(b.name.as_bytes()),
        });
        Ok(v)
    }

    /// Link names, sorted (byte order).
    pub fn link_names(&self) -> Result<Vec<String>> {
        Ok(self.members()?.into_iter().map(|e| e.name).collect())
    }

    /// Number of links in the group.
    pub fn len(&self) -> Result<usize> {
        Ok(self.read_links()?.len())
    }

    /// Whether the group has no links.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Whether a link called `name` exists (`name` must not contain `/`).
    pub fn contains(&self, name: &str) -> Result<bool> {
        match self.link(name) {
            Ok(_) => Ok(true),
            Err(Error::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// The link called `name` in this group (a single path component).
    pub fn link(&self, name: &str) -> Result<Link> {
        if name.is_empty() || name.contains('/') {
            return Err(Error::invalid(format!("invalid link name {name:?}")));
        }
        let not_found = || Error::NotFound(self.child_path(name));
        match self.storage()? {
            Storage::Links(Some(info)) if info.is_dense() => self
                .dense_lookup(&info, name)?
                .map(|e| e.link)
                .ok_or_else(not_found),
            _ => self
                .read_links()?
                .into_iter()
                .find(|e| e.name == name)
                .map(|e| e.link)
                .ok_or_else(not_found),
        }
    }

    /// The object at `path`, relative to this group (or absolute when it
    /// starts with `/`). Soft links are followed (up to
    /// [`MAX_SOFT_LINK_HOPS`]); external links fail with
    /// [`Error::Unsupported`].
    pub fn object(&self, path: &str) -> Result<Object> {
        let mut hops = 0;
        self.resolve(path, &mut hops)
    }

    /// The group at `path`.
    pub fn group(&self, path: &str) -> Result<Group> {
        self.object(path)?.into_group()
    }

    /// The dataset at `path`.
    pub fn dataset(&self, path: &str) -> Result<Dataset> {
        self.object(path)?.into_dataset()
    }

    /// The committed datatype at `path`.
    pub fn named_datatype(&self, path: &str) -> Result<NamedDatatype> {
        self.object(path)?.into_named_datatype()
    }

    /// Open the targets of all hard and soft links (sorted by name).
    /// Dangling soft links, external and user-defined links are skipped.
    pub fn objects(&self) -> Result<Vec<Object>> {
        let mut out = Vec::new();
        for e in self.members()? {
            match e.link {
                Link::Hard { address } => out.push(
                    self.node
                        .file
                        .object_at(address, &self.child_path(&e.name))?,
                ),
                Link::Soft { .. } => match self.object(&e.name) {
                    Ok(o) => out.push(o),
                    Err(Error::NotFound(_)) => {}
                    Err(e) => return Err(e),
                },
                _ => {}
            }
        }
        Ok(out)
    }

    fn child_path(&self, name: &str) -> String {
        if self.node.path.ends_with('/') {
            format!("{}{name}", self.node.path)
        } else {
            format!("{}/{name}", self.node.path)
        }
    }

    fn resolve(&self, path: &str, hops: &mut usize) -> Result<Object> {
        let mut current: Object = if path.starts_with('/') {
            Object::Group(self.node.file.root()?)
        } else {
            Object::Group(self.clone())
        };
        for comp in path.split('/').filter(|c| !c.is_empty() && *c != ".") {
            let group = match &current {
                Object::Group(g) => g.clone(),
                o => {
                    return Err(Error::NotFound(format!(
                        "{path} ({} is not a group)",
                        o.path()
                    )));
                }
            };
            current = match group.link(comp)? {
                Link::Hard { address } => group
                    .node
                    .file
                    .object_at(address, &group.child_path(comp))?,
                Link::Soft { path: target } => {
                    *hops += 1;
                    if *hops > MAX_SOFT_LINK_HOPS {
                        return Err(Error::LimitExceeded("soft link hops"));
                    }
                    // Report the object under the path it was reached by.
                    group
                        .resolve(&target, hops)?
                        .with_path(group.child_path(comp).into())
                }
                Link::External { file, path: p } => {
                    return Err(Error::unsupported(format!(
                        "external link {} -> {file}:{p} is not followed",
                        group.child_path(comp)
                    )));
                }
                Link::UserDefined { link_type, .. } => {
                    return Err(Error::unsupported(format!(
                        "user-defined link type {link_type} at {}",
                        group.child_path(comp)
                    )));
                }
            };
        }
        Ok(current)
    }

    /// Read every link, in storage order.
    fn read_links(&self) -> Result<Vec<LinkEntry>> {
        match self.storage()? {
            Storage::SymbolTable(st) => self.read_symbol_table(&st),
            Storage::Links(Some(info)) if info.is_dense() => self.read_dense(&info),
            Storage::Links(_) => self.read_compact(),
        }
    }

    fn read_symbol_table(&self, st: &SymbolTableMessage) -> Result<Vec<LinkEntry>> {
        let file = &self.node.file;
        let store = file.store().as_ref();
        let sizes = file.sizes();
        let heap = LocalHeap::read(store, st.local_heap_address, sizes)?;
        let mut out = Vec::new();
        for snod in group_btree_children(store, st.btree_address, sizes)? {
            let node = SymbolTableNode::read(store, snod, sizes)?;
            if out.len().saturating_add(node.entries.len()) > MAX_ENTRIES {
                return Err(Error::LimitExceeded("group member count"));
            }
            for entry in node.entries {
                let name = heap.get_string_lossy(entry.link_name_offset)?.into_owned();
                let link = match entry.scratch {
                    ScratchPad::SoftLink { link_value_offset } => Link::Soft {
                        path: heap
                            .get_string_lossy(u64::from(link_value_offset))?
                            .into_owned(),
                    },
                    _ => Link::Hard {
                        address: entry.object_header_address,
                    },
                };
                out.push(LinkEntry {
                    name,
                    link,
                    creation_order: None,
                });
            }
        }
        Ok(out)
    }

    fn read_compact(&self) -> Result<Vec<LinkEntry>> {
        let sizes = self.node.file.sizes();
        self.node
            .header
            .messages_of_type(MessageType::LINK)
            .map(|m| Ok(LinkEntry::from_message(LinkMessage::parse(&m.data, sizes)?)))
            .collect()
    }

    fn dense_index(&self, info: &LinkInfoMessage) -> Result<(FractalHeap, BTreeV2)> {
        let file = &self.node.file;
        let store = file.store().as_ref();
        let sizes = file.sizes();
        let heap = FractalHeap::open(store, info.fractal_heap_address, sizes)?;
        if info.name_index_address == UNDEFINED_ADDRESS {
            return Err(Error::invalid("dense group without a name index"));
        }
        let ctx = RecordContext {
            heap_id_len: Some(usize::from(heap.header.heap_id_len)),
            ..RecordContext::default()
        };
        let tree = BTreeV2::open(store, info.name_index_address, sizes, ctx)?;
        if tree.header.tree_type != 5 {
            return Err(Error::invalid("link name index is not a type 5 v2 B-tree"));
        }
        Ok((heap, tree))
    }

    fn read_dense(&self, info: &LinkInfoMessage) -> Result<Vec<LinkEntry>> {
        let file = &self.node.file;
        let store = file.store().as_ref();
        let (heap, tree) = self.dense_index(info)?;
        let mut out = Vec::new();
        for rec in tree.records(store)? {
            if let Record::LinkName { heap_id, .. } = rec {
                let data = heap.get(store, &heap_id)?;
                out.push(LinkEntry::from_message(LinkMessage::parse(
                    &data,
                    file.sizes(),
                )?));
            }
        }
        Ok(out)
    }

    fn dense_lookup(&self, info: &LinkInfoMessage, name: &str) -> Result<Option<LinkEntry>> {
        let file = &self.node.file;
        let store = file.store().as_ref();
        let (heap, tree) = self.dense_index(info)?;
        let hash = lookup3(name.as_bytes(), 0);
        let candidates = tree.find_all(store, |r| match r {
            Record::LinkName { hash: h, .. } => h.cmp(&hash),
            _ => Ordering::Less,
        })?;
        for rec in candidates {
            if let Record::LinkName { heap_id, .. } = rec {
                let data = heap.get(store, &heap_id)?;
                let m = LinkMessage::parse(&data, file.sizes())?;
                if m.name == name {
                    return Ok(Some(LinkEntry::from_message(m)));
                }
            }
        }
        Ok(None)
    }
}
