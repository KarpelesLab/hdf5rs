//! Objects in the group hierarchy: groups, datasets and committed
//! datatypes, plus the metadata they share (attributes, times, comments).

use std::sync::Arc;

use crate::attribute::{self, Attribute};
use crate::dataset::{Dataset, NamedDatatype};
use crate::error::{Error, Result};
use crate::file::File;
use crate::format::messages::MessageType;
use crate::format::messages::comment::CommentMessage;
use crate::format::messages::modification_time::{
    ModificationTimeMessage, ModificationTimeOldMessage,
};
use crate::format::object_header::{ObjectHeader, ObjectTimes};
use crate::group::Group;

/// The kind of an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectKind {
    /// A group.
    Group,
    /// A dataset.
    Dataset,
    /// A committed (named) datatype.
    NamedDatatype,
}

impl ObjectKind {
    /// Classify an object by the messages in its header.
    pub fn of_header(h: &ObjectHeader) -> Result<ObjectKind> {
        let has = |t| h.find_message(t).is_some();
        if has(MessageType::LAYOUT) {
            Ok(ObjectKind::Dataset)
        } else if has(MessageType::SYMBOL_TABLE)
            || has(MessageType::LINK_INFO)
            || has(MessageType::LINK)
            || has(MessageType::GROUP_INFO)
        {
            Ok(ObjectKind::Group)
        } else if has(MessageType::DATATYPE) && !has(MessageType::DATASPACE) {
            Ok(ObjectKind::NamedDatatype)
        } else {
            Err(Error::unsupported(format!(
                "cannot determine the kind of object at {:#x}",
                h.address
            )))
        }
    }
}

/// State common to every object handle: the file, the header and the path
/// the object was reached by.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub file: File,
    pub header: Arc<ObjectHeader>,
    pub path: Arc<str>,
}

impl Node {
    pub fn name(&self) -> &str {
        match self.path.rfind('/') {
            Some(i) if self.path.len() > 1 => &self.path[i + 1..],
            _ => &self.path,
        }
    }

    pub fn modification_time(&self) -> Result<Option<i64>> {
        if let Some(t) = self.header.times {
            return Ok(Some(i64::from(t.modification)));
        }
        if let Some(m) = self.header.find_message(MessageType::MODIFICATION_TIME) {
            let m = ModificationTimeMessage::parse(&m.data, self.file.sizes())?;
            return Ok(Some(i64::from(m.seconds)));
        }
        if let Some(m) = self.header.find_message(MessageType::MODIFICATION_TIME_OLD) {
            let m = ModificationTimeOldMessage::parse(&m.data, self.file.sizes())?;
            return Ok(Some(m.to_unix_seconds()));
        }
        Ok(None)
    }

    pub fn comment(&self) -> Result<Option<String>> {
        match self.header.find_message(MessageType::COMMENT) {
            Some(m) => Ok(Some(
                CommentMessage::parse(&m.data, self.file.sizes())?.comment,
            )),
            None => Ok(None),
        }
    }
}

/// Generates the accessors shared by all object handle types.
macro_rules! object_common {
    ($t:ty) => {
        impl $t {
            /// The file this object belongs to.
            pub fn file(&self) -> &$crate::File {
                &self.node.file
            }

            /// Address of the object header (unique per object in a file).
            pub fn address(&self) -> u64 {
                self.node.header.address
            }

            /// The path this object was opened by (e.g. `/a/b`).
            pub fn path(&self) -> &str {
                &self.node.path
            }

            /// The last component of [`path`](Self::path) (`/` for the root).
            pub fn name(&self) -> &str {
                self.node.name()
            }

            /// The parsed object header.
            pub fn header(&self) -> &$crate::format::object_header::ObjectHeader {
                &self.node.header
            }

            /// All attributes, sorted by name (byte order).
            pub fn attributes(&self) -> $crate::Result<Vec<$crate::Attribute>> {
                $crate::attribute::read_all(&self.node.file, &self.node.header, false)
            }

            /// All attributes in creation order; fails if the object does
            /// not track attribute creation order.
            pub fn attributes_by_creation_order(&self) -> $crate::Result<Vec<$crate::Attribute>> {
                $crate::attribute::read_all(&self.node.file, &self.node.header, true)
            }

            /// Attribute names, sorted (byte order).
            pub fn attribute_names(&self) -> $crate::Result<Vec<String>> {
                Ok(self
                    .attributes()?
                    .into_iter()
                    .map(|a| a.name().to_owned())
                    .collect())
            }

            /// The attribute called `name`.
            pub fn attribute(&self, name: &str) -> $crate::Result<$crate::Attribute> {
                $crate::attribute::find(&self.node.file, &self.node.header, name)
            }

            /// Last modification time in seconds since the UNIX epoch, if
            /// recorded.
            pub fn modification_time(&self) -> $crate::Result<Option<i64>> {
                self.node.modification_time()
            }

            /// Access/modification/change/birth times (version 2 headers
            /// that store them).
            pub fn times(&self) -> Option<$crate::format::object_header::ObjectTimes> {
                self.node.header.times
            }

            /// The object comment, if any.
            pub fn comment(&self) -> $crate::Result<Option<String>> {
                self.node.comment()
            }
        }
    };
}
pub(crate) use object_common;

/// Any object in the hierarchy.
#[derive(Debug, Clone)]
pub enum Object {
    /// A group.
    Group(Group),
    /// A dataset.
    Dataset(Dataset),
    /// A committed datatype.
    NamedDatatype(NamedDatatype),
}

impl Object {
    pub(crate) fn from_header(
        file: File,
        header: Arc<ObjectHeader>,
        path: Arc<str>,
    ) -> Result<Self> {
        let node = Node { file, header, path };
        Ok(match ObjectKind::of_header(&node.header)? {
            ObjectKind::Group => Object::Group(Group::new(node)),
            ObjectKind::Dataset => Object::Dataset(Dataset::new(node)?),
            ObjectKind::NamedDatatype => Object::NamedDatatype(NamedDatatype::new(node)?),
        })
    }

    pub(crate) fn with_path(mut self, path: Arc<str>) -> Self {
        match &mut self {
            Object::Group(g) => g.node.path = path,
            Object::Dataset(d) => d.node.path = path,
            Object::NamedDatatype(t) => t.node.path = path,
        }
        self
    }

    fn node(&self) -> &Node {
        match self {
            Object::Group(g) => &g.node,
            Object::Dataset(d) => &d.node,
            Object::NamedDatatype(t) => &t.node,
        }
    }

    /// The kind of this object.
    pub fn kind(&self) -> ObjectKind {
        match self {
            Object::Group(_) => ObjectKind::Group,
            Object::Dataset(_) => ObjectKind::Dataset,
            Object::NamedDatatype(_) => ObjectKind::NamedDatatype,
        }
    }

    /// The file this object belongs to (differs from the starting file for
    /// objects reached through external links).
    pub fn file(&self) -> &File {
        &self.node().file
    }

    /// Address of the object header.
    pub fn address(&self) -> u64 {
        self.node().header.address
    }

    /// The path this object was opened by.
    pub fn path(&self) -> &str {
        &self.node().path
    }

    /// The last path component.
    pub fn name(&self) -> &str {
        self.node().name()
    }

    /// The parsed object header.
    pub fn header(&self) -> &ObjectHeader {
        &self.node().header
    }

    /// All attributes, sorted by name.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        attribute::read_all(&self.node().file, &self.node().header, false)
    }

    /// Attribute names, sorted.
    pub fn attribute_names(&self) -> Result<Vec<String>> {
        Ok(self
            .attributes()?
            .into_iter()
            .map(|a| a.name().to_owned())
            .collect())
    }

    /// The attribute called `name`.
    pub fn attribute(&self, name: &str) -> Result<Attribute> {
        attribute::find(&self.node().file, &self.node().header, name)
    }

    /// Last modification time (seconds since the UNIX epoch), if recorded.
    pub fn modification_time(&self) -> Result<Option<i64>> {
        self.node().modification_time()
    }

    /// Timestamps stored in a version 2 header.
    pub fn times(&self) -> Option<ObjectTimes> {
        self.node().header.times
    }

    /// The object comment, if any.
    pub fn comment(&self) -> Result<Option<String>> {
        self.node().comment()
    }

    /// The group, if this is one.
    pub fn as_group(&self) -> Option<&Group> {
        match self {
            Object::Group(g) => Some(g),
            _ => None,
        }
    }

    /// The dataset, if this is one.
    pub fn as_dataset(&self) -> Option<&Dataset> {
        match self {
            Object::Dataset(d) => Some(d),
            _ => None,
        }
    }

    /// The committed datatype, if this is one.
    pub fn as_named_datatype(&self) -> Option<&NamedDatatype> {
        match self {
            Object::NamedDatatype(t) => Some(t),
            _ => None,
        }
    }

    /// Convert into a group, or fail with [`Error::WrongObjectType`].
    pub fn into_group(self) -> Result<Group> {
        match self {
            Object::Group(g) => Ok(g),
            o => Err(Error::WrongObjectType(format!(
                "{} is not a group",
                o.path()
            ))),
        }
    }

    /// Convert into a dataset.
    pub fn into_dataset(self) -> Result<Dataset> {
        match self {
            Object::Dataset(d) => Ok(d),
            o => Err(Error::WrongObjectType(format!(
                "{} is not a dataset",
                o.path()
            ))),
        }
    }

    /// Convert into a committed datatype.
    pub fn into_named_datatype(self) -> Result<NamedDatatype> {
        match self {
            Object::NamedDatatype(t) => Ok(t),
            o => Err(Error::WrongObjectType(format!(
                "{} is not a committed datatype",
                o.path()
            ))),
        }
    }
}
