//! Serialization of the object model into object headers (and dense
//! link/attribute storage).

use super::compat::{self, EntryTarget, SymTabPlan};
use super::dense::{DenseKind, DensePlan};
use super::{
    AttrSpec, DatasetSpec, FileWriter, LayoutSpec, LinkTargetSpec, Node, NodeKind, Patched, WriteAt,
};
use crate::error::{Error, Result};
use crate::format::messages::MessageType;
use crate::format::messages::attribute::{Attribute, MaybeShared};
use crate::format::messages::attribute_info::AttributeInfo;
use crate::format::messages::dataspace::{Dataspace, DataspaceKind};
use crate::format::messages::datatype::CharacterSet;
use crate::format::messages::fill_value::FillValue;
use crate::format::messages::group_info::GroupInfoMessage;
use crate::format::messages::layout::DataLayout;
use crate::format::messages::link::{CharSet, LinkMessage, LinkTarget};
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::modification_time::ModificationTimeMessage;
use crate::format::messages::ref_count::RefCountMessage;
use crate::format::messages::shared::{SharedLocation, SharedMessage};
use crate::format::messages::symbol_table::SymbolTableMessage;
use crate::format::object_header::{ObjectHeader, ObjectTimes, RawMessage, V2HeaderOptions};
use crate::format::{Sizes, UNDEFINED_ADDRESS};

/// Default maximum number of compactly stored links / attributes (the
/// reference library's default phase change value); more are stored
/// densely.
pub(crate) const MAX_COMPACT: usize = 8;
/// Largest attribute message kept in the object header; larger ones force
/// dense attribute storage.
const MAX_COMPACT_ATTRIBUTE: usize = 64 * 1024 - 1;

/// Result of laying out the object headers.
pub(crate) struct Built {
    /// Object header address of every node, indexed by node ID.
    pub(crate) addresses: Vec<u64>,
    /// Address of the root group's header.
    pub(crate) root_address: u64,
    /// (B-tree, local heap) addresses of the root group's symbol table
    /// (compatibility format only).
    pub(crate) root_symtab: Option<(u64, u64)>,
}

/// The attribute message for `attr` with `data` as its value: version 3
/// (UTF-8 capable) in the latest format, version 1 in the compatibility
/// format.
fn attribute(attr: &AttrSpec, data: Vec<u8>, compat: bool) -> Result<Attribute> {
    if !compat {
        return Ok(Attribute::new_v3(
            &attr.name,
            attr.datatype.clone(),
            attr.dataspace.clone(),
            data,
        ));
    }
    Ok(Attribute {
        version: 1,
        name: attr.name.clone(),
        name_charset: CharacterSet::Ascii,
        datatype: MaybeShared::Inline(compat::downgrade(&attr.datatype)),
        dataspace: MaybeShared::Inline(compat_dataspace(&attr.dataspace)?),
        data,
    })
}

/// A version 1 dataspace (the compatibility format has no null dataspace).
fn compat_dataspace(ds: &Dataspace) -> Result<Dataspace> {
    if ds.kind == DataspaceKind::Null {
        return Err(Error::unsupported(
            "null dataspaces in the compatibility format",
        ));
    }
    Ok(Dataspace {
        version: 1,
        ..ds.clone()
    })
}

/// Encode an attribute message with object references left as zeros.
pub(crate) fn attribute_message(attr: &AttrSpec, sizes: Sizes, compat: bool) -> Result<Vec<u8>> {
    attribute(attr, attr.value.data.clone(), compat)?.encode(sizes)
}

/// Apply object reference patches to a copy of `p.data`.
fn patched(p: &Patched, addrs: &[u64], sizes: Sizes) -> Result<Vec<u8>> {
    let mut data = p.data.clone();
    let width = sizes.offset as usize;
    for &(pos, target) in &p.refs {
        let addr = *addrs
            .get(target.0)
            .ok_or_else(|| Error::invalid("object reference to an unknown object"))?;
        let bytes = addr.to_le_bytes();
        let end = pos.checked_add(width).ok_or(Error::UnexpectedEof)?;
        data.get_mut(pos..end)
            .ok_or(Error::UnexpectedEof)?
            .copy_from_slice(&bytes[..width.min(8)]);
    }
    Ok(data)
}

fn raw(msg_type: u16, flags: u8, data: Vec<u8>) -> RawMessage {
    RawMessage {
        msg_type,
        flags,
        creation_order: None,
        data,
    }
}

/// The filter pipeline (chunked datasets with filters; message version
/// `pipeline_version`) and data layout messages of a dataset.
fn storage_messages(
    cx: &Ctx<'_>,
    d: &DatasetSpec,
    pipeline_version: u8,
) -> Result<Vec<RawMessage>> {
    let sizes = cx.sizes;
    let mut msgs = Vec::new();
    let (layout, version) = match &d.layout {
        LayoutSpec::Compact(p) => (
            DataLayout::Compact {
                data: patched(p, cx.addrs, sizes)?,
                dims: None,
            },
            3,
        ),
        LayoutSpec::Contiguous { address, size } => (
            DataLayout::Contiguous {
                address: *address,
                size: Some(*size),
                dims: None,
            },
            3,
        ),
        LayoutSpec::Chunked(c) => {
            if !c.pipeline.filters.is_empty() {
                msgs.push(raw(
                    MessageType::FILTER_PIPELINE,
                    0,
                    c.pipeline.encode(pipeline_version)?,
                ));
            }
            (c.layout(d.datatype.size), c.layout_version())
        }
    };
    msgs.push(raw(MessageType::LAYOUT, 0, layout.encode(version, sizes)?));
    Ok(msgs)
}

struct Ctx<'a> {
    nodes: &'a [Node],
    addrs: &'a [u64],
    ref_counts: &'a [u32],
    sizes: Sizes,
    times: Option<ObjectTimes>,
    /// Compatibility (superblock version 0) format.
    compat: bool,
}

impl Ctx<'_> {
    fn node(&self, id: usize) -> Result<&Node> {
        self.nodes
            .get(id)
            .ok_or_else(|| Error::invalid("unknown object"))
    }

    fn addr_of(&self, target: usize) -> Result<u64> {
        self.addrs
            .get(target)
            .copied()
            .ok_or_else(|| Error::invalid("link to an unknown object"))
    }

    /// Encoded link messages of a group (empty for other objects).
    fn link_messages(&self, node: &Node) -> Result<Vec<(String, Vec<u8>)>> {
        let NodeKind::Group { links } = &node.kind else {
            return Ok(Vec::new());
        };
        links
            .iter()
            .map(|l| {
                let target = match &l.target {
                    LinkTargetSpec::Hard(t) => LinkTarget::Hard {
                        address: self.addr_of(t.0)?,
                    },
                    LinkTargetSpec::Soft(p) => LinkTarget::Soft { path: p.clone() },
                    LinkTargetSpec::External { file, path } => LinkTarget::External {
                        version: 0,
                        flags: 0,
                        file: file.clone(),
                        path: path.clone(),
                    },
                };
                let m = LinkMessage {
                    creation_order: None,
                    char_set: if l.name.is_ascii() {
                        CharSet::Ascii
                    } else {
                        CharSet::Utf8
                    },
                    name: l.name.clone(),
                    target,
                };
                Ok((l.name.clone(), m.encode(self.sizes)?))
            })
            .collect()
    }

    /// Encoded attribute messages.
    fn attribute_messages(&self, node: &Node) -> Result<Vec<(String, Vec<u8>)>> {
        node.attrs
            .iter()
            .map(|a| {
                let msg = attribute(a, patched(&a.value, self.addrs, self.sizes)?, self.compat)?;
                Ok((a.name.clone(), msg.encode(self.sizes)?))
            })
            .collect()
    }
}

/// Addresses of a dense store as referenced from the object header.
#[derive(Debug, Clone, Copy)]
struct DenseRef {
    heap: u64,
    name_index: u64,
}

impl DenseRef {
    fn of(plan: Option<&DensePlan>) -> Option<DenseRef> {
        plan.map(|p| DenseRef {
            heap: p.heap_address(),
            name_index: p.name_index_address(),
        })
    }
}

/// Build and encode the object header of node `id`; links / attributes
/// go to the dense stores when given.
fn encode_node(
    cx: &Ctx<'_>,
    id: usize,
    dense_links: Option<DenseRef>,
    dense_attrs: Option<DenseRef>,
    symtab: Option<(u64, u64)>,
) -> Result<Vec<u8>> {
    if cx.compat {
        return encode_node_v1(cx, id, symtab);
    }
    let node = cx.node(id)?;
    let sizes = cx.sizes;
    let mut msgs = Vec::new();
    let attribute_info = |dense: Option<DenseRef>| -> Result<RawMessage> {
        let mut info = AttributeInfo::compact(false);
        if let Some(d) = dense {
            info.fractal_heap_address = d.heap;
            info.name_btree_address = d.name_index;
        }
        Ok(raw(
            MessageType::ATTRIBUTE_INFO,
            RawMessage::FLAG_DONT_SHARE,
            info.encode(sizes)?,
        ))
    };
    match &node.kind {
        NodeKind::Group { .. } => {
            let mut info = LinkInfoMessage::compact(false);
            if let Some(d) = dense_links {
                info.fractal_heap_address = d.heap;
                info.name_index_address = d.name_index;
            }
            msgs.push(raw(MessageType::LINK_INFO, 0, info.encode(sizes)?));
            let ginfo = GroupInfoMessage {
                link_phase_change: None,
                estimated_entries: None,
            };
            msgs.push(raw(
                MessageType::GROUP_INFO,
                RawMessage::FLAG_CONSTANT,
                ginfo.encode()?,
            ));
            msgs.push(attribute_info(dense_attrs)?);
            if dense_links.is_none() {
                for (_, m) in cx.link_messages(node)? {
                    msgs.push(raw(MessageType::LINK, 0, m));
                }
            }
        }
        NodeKind::Dataset(d) => {
            msgs.push(raw(MessageType::DATASPACE, 0, d.dataspace.encode(sizes)?));
            match d.committed {
                Some(t) => {
                    // Version 2 shared message, as the reference library
                    // writes for committed datatypes.
                    let shared = SharedMessage {
                        version: 2,
                        location: SharedLocation::ObjectHeader(cx.addr_of(t.0)?),
                    };
                    msgs.push(raw(
                        MessageType::DATATYPE,
                        RawMessage::FLAG_CONSTANT | RawMessage::FLAG_SHARED,
                        shared.encode(sizes)?,
                    ));
                }
                None => msgs.push(raw(
                    MessageType::DATATYPE,
                    RawMessage::FLAG_CONSTANT,
                    d.datatype.encode()?,
                )),
            }
            msgs.push(raw(
                MessageType::FILL_VALUE,
                RawMessage::FLAG_CONSTANT,
                d.fill.encode()?,
            ));
            msgs.extend(storage_messages(cx, d, 2)?);
            if !node.attrs.is_empty() {
                msgs.push(attribute_info(dense_attrs)?);
            }
        }
        NodeKind::NamedDatatype(dt) => {
            msgs.push(raw(
                MessageType::DATATYPE,
                RawMessage::FLAG_CONSTANT | RawMessage::FLAG_DONT_SHARE,
                dt.encode()?,
            ));
            if dense_attrs.is_some() {
                msgs.push(attribute_info(dense_attrs)?);
            }
        }
    }
    if dense_attrs.is_none() {
        for (_, m) in cx.attribute_messages(node)? {
            msgs.push(raw(MessageType::ATTRIBUTE, 0, m));
        }
    }
    let rc = cx.ref_counts.get(id).copied().unwrap_or(1);
    if rc > 1 {
        msgs.push(raw(
            MessageType::REF_COUNT,
            RawMessage::FLAG_DONT_SHARE,
            RefCountMessage { ref_count: rc }.encode()?,
        ));
    }
    let opts = V2HeaderOptions {
        times: cx.times,
        ..V2HeaderOptions::default()
    };
    ObjectHeader::encode_v2(&opts, &msgs)
}

/// The version 1 object header of node `id` (compatibility format).
fn encode_node_v1(cx: &Ctx<'_>, id: usize, symtab: Option<(u64, u64)>) -> Result<Vec<u8>> {
    let node = cx.node(id)?;
    let sizes = cx.sizes;
    let mut msgs = Vec::new();
    match &node.kind {
        NodeKind::Group { .. } => {
            let (btree_address, local_heap_address) =
                symtab.ok_or_else(|| Error::invalid("group without a symbol table"))?;
            let m = SymbolTableMessage {
                btree_address,
                local_heap_address,
            };
            msgs.push(raw(MessageType::SYMBOL_TABLE, 0, m.encode(sizes)?));
        }
        NodeKind::Dataset(d) => {
            msgs.push(raw(
                MessageType::DATASPACE,
                0,
                compat_dataspace(&d.dataspace)?.encode(sizes)?,
            ));
            match d.committed {
                Some(t) => {
                    let shared = SharedMessage {
                        version: 2,
                        location: SharedLocation::ObjectHeader(cx.addr_of(t.0)?),
                    };
                    msgs.push(raw(
                        MessageType::DATATYPE,
                        RawMessage::FLAG_CONSTANT | RawMessage::FLAG_SHARED,
                        shared.encode(sizes)?,
                    ));
                }
                None => msgs.push(raw(
                    MessageType::DATATYPE,
                    RawMessage::FLAG_CONSTANT,
                    compat::downgrade(&d.datatype).encode()?,
                )),
            }
            // Version 2 fill value message, "defined" (an empty value means
            // the default of zero bytes), as the reference library writes.
            let fill = FillValue {
                version: 2,
                state: crate::format::messages::fill_value::FillValueState::Defined,
                ..d.fill.clone()
            };
            msgs.push(raw(
                MessageType::FILL_VALUE,
                RawMessage::FLAG_CONSTANT,
                fill.encode()?,
            ));
            msgs.extend(storage_messages(cx, d, 1)?);
        }
        NodeKind::NamedDatatype(dt) => {
            msgs.push(raw(
                MessageType::DATATYPE,
                RawMessage::FLAG_CONSTANT | RawMessage::FLAG_DONT_SHARE,
                compat::downgrade(dt).encode()?,
            ));
        }
    }
    for (_, m) in cx.attribute_messages(node)? {
        msgs.push(raw(MessageType::ATTRIBUTE, 0, m));
    }
    if let Some(t) = cx.times {
        msgs.push(raw(
            MessageType::MODIFICATION_TIME,
            0,
            ModificationTimeMessage {
                seconds: t.modification,
            }
            .encode()?,
        ));
    }
    let rc = cx.ref_counts.get(id).copied().unwrap_or(1);
    ObjectHeader::encode_v1(rc, &msgs)
}

/// Dense storage decisions and plans for one node.
#[derive(Default)]
struct NodePlan {
    links: Option<DensePlan>,
    attrs: Option<DensePlan>,
    /// Symbol table of an old-style group (compatibility format).
    symtab: Option<SymTabPlan>,
}

impl NodePlan {
    fn symtab_addrs(&self) -> Option<(u64, u64)> {
        self.symtab
            .as_ref()
            .map(|p| (p.btree_address(), p.heap_address()))
    }
}

fn plan_node(cx: &Ctx<'_>, id: usize) -> Result<NodePlan> {
    let node = cx.node(id)?;
    let sizes = cx.sizes;
    let lens = |msgs: Vec<(String, Vec<u8>)>| -> Vec<(String, usize)> {
        msgs.into_iter().map(|(n, m)| (n, m.len())).collect()
    };
    let mut plan = NodePlan::default();
    if cx.compat {
        if let NodeKind::Group { links } = &node.kind {
            let entries = links
                .iter()
                .map(|l| match &l.target {
                    LinkTargetSpec::Hard(t) => Ok((l.name.clone(), EntryTarget::Hard(t.0))),
                    LinkTargetSpec::Soft(p) => Ok((l.name.clone(), EntryTarget::Soft(p.clone()))),
                    LinkTargetSpec::External { .. } => Err(Error::unsupported(
                        "external links in the compatibility format",
                    )),
                })
                .collect::<Result<Vec<_>>>()?;
            plan.symtab = Some(SymTabPlan::new(&entries, sizes)?);
        }
        return Ok(plan);
    }
    let links = cx.link_messages(node)?;
    if links.len() > MAX_COMPACT {
        plan.links = Some(DensePlan::new(DenseKind::Links, &lens(links), sizes)?);
    }
    let attrs = cx.attribute_messages(node)?;
    if attrs.len() > MAX_COMPACT || attrs.iter().any(|(_, m)| m.len() > MAX_COMPACT_ATTRIBUTE) {
        plan.attrs = Some(DensePlan::new(DenseKind::Attributes, &lens(attrs), sizes)?);
    }
    Ok(plan)
}

/// Lay out, encode and write every object header.
pub(crate) fn write_headers<S: WriteAt>(w: &mut FileWriter<S>) -> Result<Built> {
    let sizes = w.opts.sizes;
    let compat = w.opts.superblock_version < 2;
    let n = w.nodes.len();
    // Hard-link reference counts: incoming hard links, the superblock for
    // the root, and datasets using a committed datatype.
    let mut ref_counts = vec![0u32; n];
    if let Some(r) = ref_counts.first_mut() {
        *r = 1;
    }
    let mut bump = |t: usize| -> Result<()> {
        let c = ref_counts
            .get_mut(t)
            .ok_or_else(|| Error::invalid("link to an unknown object"))?;
        *c = c.saturating_add(1);
        Ok(())
    };
    for node in &w.nodes {
        match &node.kind {
            NodeKind::Group { links } => {
                for l in links {
                    if let LinkTargetSpec::Hard(t) = l.target {
                        bump(t.0)?;
                    }
                }
            }
            NodeKind::Dataset(d) => {
                if let Some(t) = d.committed {
                    bump(t.0)?;
                }
            }
            NodeKind::NamedDatatype(_) => {}
        }
    }
    let times = w.opts.track_times.then(|| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| u32::try_from(d.as_secs()).unwrap_or(u32::MAX))
            .unwrap_or(0);
        ObjectTimes {
            access: now,
            modification: now,
            change: now,
            birth: now,
        }
    });
    // Pass 1: plan and size everything (addresses do not affect sizes),
    // then allocate.
    let placeholder = vec![0u64; n];
    let dummy = DenseRef {
        heap: 0,
        name_index: 0,
    };
    let mut plans = Vec::with_capacity(n);
    let mut lens = Vec::with_capacity(n);
    {
        let cx = Ctx {
            nodes: &w.nodes,
            addrs: &placeholder,
            ref_counts: &ref_counts,
            sizes,
            times,
            compat,
        };
        for id in 0..n {
            let plan = plan_node(&cx, id)?;
            let len = encode_node(
                &cx,
                id,
                plan.links.as_ref().map(|_| dummy),
                plan.attrs.as_ref().map(|_| dummy),
                plan.symtab.as_ref().map(|_| (0, 0)),
            )?
            .len();
            plans.push(plan);
            lens.push(len);
        }
    }
    let mut addresses = Vec::with_capacity(n);
    for (&len, plan) in lens.iter().zip(&mut plans) {
        addresses.push(w.alloc.alloc(len as u64)?);
        if let Some(p) = &mut plan.links {
            p.allocate(&mut w.alloc)?;
        }
        if let Some(p) = &mut plan.attrs {
            p.allocate(&mut w.alloc)?;
        }
        if let Some(p) = &mut plan.symtab {
            p.allocate(&mut w.alloc)?;
        }
    }
    // Pass 2: real encoding.
    let cx = Ctx {
        nodes: &w.nodes,
        addrs: &addresses,
        ref_counts: &ref_counts,
        sizes,
        times,
        compat,
    };
    let header_of = |t: usize| -> Result<(u64, Option<(u64, u64)>)> {
        let addr = cx.addr_of(t)?;
        Ok((addr, plans.get(t).and_then(NodePlan::symtab_addrs)))
    };
    for (id, ((&addr, &len), plan)) in addresses.iter().zip(&lens).zip(&plans).enumerate() {
        let bytes = encode_node(
            &cx,
            id,
            DenseRef::of(plan.links.as_ref()),
            DenseRef::of(plan.attrs.as_ref()),
            plan.symtab_addrs(),
        )?;
        if bytes.len() != len {
            return Err(Error::invalid("object header size changed between passes"));
        }
        w.storage.write_all_at(addr, &bytes)?;
        let node = cx.node(id)?;
        if let Some(p) = &plan.links {
            let objs: Vec<Vec<u8>> = cx
                .link_messages(node)?
                .into_iter()
                .map(|(_, m)| m)
                .collect();
            p.write(&objs, &mut w.storage)?;
        }
        if let Some(p) = &plan.attrs {
            let objs: Vec<Vec<u8>> = cx
                .attribute_messages(node)?
                .into_iter()
                .map(|(_, m)| m)
                .collect();
            p.write(&objs, &mut w.storage)?;
        }
        if let Some(p) = &plan.symtab {
            p.write(&header_of, &mut w.storage)?;
        }
    }
    let root_address = addresses
        .first()
        .copied()
        .filter(|&a| a != UNDEFINED_ADDRESS)
        .ok_or_else(|| Error::invalid("no root group"))?;
    let root_symtab = plans.first().and_then(NodePlan::symtab_addrs);
    Ok(Built {
        addresses,
        root_address,
        root_symtab,
    })
}
