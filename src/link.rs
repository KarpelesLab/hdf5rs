//! Links: the named edges of the group hierarchy.

use crate::format::messages::link::{LinkMessage, LinkTarget};

/// Where a link points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Link {
    /// A hard link to the object header at `address`.
    Hard {
        /// Address of the target object header.
        address: u64,
    },
    /// A soft (symbolic) link holding a path, resolved when traversed.
    Soft {
        /// Target path (absolute, or relative to the containing group).
        path: String,
    },
    /// A link to an object in another file. Followed only on request (see
    /// [`Group::follow_external`](crate::Group::follow_external)).
    External {
        /// Name of the target file.
        file: String,
        /// Path of the target object within that file.
        path: String,
    },
    /// A user-defined link type (not interpreted).
    UserDefined {
        /// The link type (65..=255).
        link_type: u8,
        /// The raw link value.
        data: Vec<u8>,
    },
}

/// How a group stores its links.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkStorage {
    /// Old-style group: symbol table (v1 B-tree + local heap).
    SymbolTable,
    /// New-style group with Link messages in the object header.
    Compact,
    /// New-style group with links in a fractal heap indexed by v2 B-trees.
    Dense,
}

/// A named link in a group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkEntry {
    /// Link name.
    pub name: String,
    /// Link target.
    pub link: Link,
    /// Creation order, when the group tracks it.
    pub creation_order: Option<u64>,
}

impl LinkEntry {
    pub(crate) fn from_message(m: LinkMessage) -> LinkEntry {
        let link = match m.target {
            LinkTarget::Hard { address } => Link::Hard { address },
            LinkTarget::Soft { path } => Link::Soft { path },
            LinkTarget::External { file, path, .. } => Link::External { file, path },
            LinkTarget::UserDefined { link_type, data } => Link::UserDefined { link_type, data },
        };
        LinkEntry {
            name: m.name,
            link,
            creation_order: m.creation_order,
        }
    }
}
