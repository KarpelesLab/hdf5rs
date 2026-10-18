//! Attributes: small named values attached to objects, stored compactly
//! (Attribute messages) or densely (fractal heap + v2 B-tree).

use std::cmp::Ordering;

use crate::checksum::lookup3;
use crate::error::{Error, Result};
use crate::file::File;
use crate::format::UNDEFINED_ADDRESS;
use crate::format::btree_v2::{BTreeV2, Record, RecordContext};
use crate::format::fractal_heap::FractalHeap;
use crate::format::messages::MessageType;
use crate::format::messages::attribute::{Attribute as AttributeMessage, MaybeShared};
use crate::format::messages::attribute_info::AttributeInfo;
use crate::format::messages::dataspace::{Dataspace, DataspaceKind};
use crate::format::messages::datatype::{CharacterSet, Datatype};
use crate::format::object_header::{ObjectHeader, RawMessage};
use crate::reader::{Array, decode_elements, single};
use crate::types::{H5Type, Value};

/// An attribute, fully loaded: name, datatype, dataspace and raw value.
///
/// Shared datatypes/dataspaces (committed datatypes, SOHM) are resolved.
/// Cheap to clone relative to its value size; `Send + Sync`.
#[derive(Debug, Clone)]
pub struct Attribute {
    file: File,
    name: String,
    name_charset: CharacterSet,
    datatype: Datatype,
    dataspace: Dataspace,
    data: Vec<u8>,
    creation_order: Option<u32>,
    committed_datatype: Option<u64>,
}

impl Attribute {
    /// Attribute name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Character set of the name.
    pub fn name_charset(&self) -> CharacterSet {
        self.name_charset
    }

    /// The element datatype.
    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    /// The dataspace (shape).
    pub fn dataspace(&self) -> &Dataspace {
        &self.dataspace
    }

    /// Current dimensions (empty for scalar and null dataspaces).
    pub fn shape(&self) -> &[u64] {
        &self.dataspace.dims
    }

    /// Number of elements (1 for scalar, 0 for null).
    pub fn element_count(&self) -> Result<u64> {
        self.dataspace.element_count()
    }

    /// The raw, undecoded value: `element_count * datatype.size` bytes in
    /// file byte order. Variable-length elements are global heap
    /// references, to be resolved through [`file`](Self::file).
    pub fn raw_value(&self) -> &[u8] {
        &self.data
    }

    /// Creation order, when tracked by the owning object.
    pub fn creation_order(&self) -> Option<u32> {
        self.creation_order
    }

    /// Address of the committed datatype this attribute uses, if any.
    pub fn committed_datatype_address(&self) -> Option<u64> {
        self.committed_datatype
    }

    /// The file this attribute belongs to.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Decode every element into `T`, in row-major order (variable-length
    /// data and region references are resolved through the file).
    pub fn read<T: H5Type>(&self) -> Result<Vec<T>> {
        let budget = self.file.access().limits.max_output_bytes;
        decode_elements(&self.file, &self.datatype, &self.data, budget)
    }

    /// Decode every element together with the attribute's shape.
    pub fn read_nd<T: H5Type>(&self) -> Result<Array<T>> {
        Ok(Array {
            data: self.read()?,
            shape: self.dataspace.dims.clone(),
        })
    }

    /// Decode the single element of a scalar (or one-element) attribute.
    pub fn read_scalar<T: H5Type>(&self) -> Result<T> {
        single(self.read()?, &format!("attribute {:?}", self.name))
    }

    /// Every element as a dynamic [`Value`].
    pub fn values(&self) -> Result<Vec<Value>> {
        self.read()
    }

    /// The value as one dynamic [`Value`]: the element itself for scalar
    /// attributes, a [`Value::Array`] of the elements (nested, slowest
    /// dimension outermost) for simple dataspaces, and an empty
    /// [`Value::Array`] for null dataspaces.
    pub fn value(&self) -> Result<Value> {
        let values = self.values()?;
        match self.dataspace.kind {
            DataspaceKind::Scalar => single(values, &format!("attribute {:?}", self.name)),
            DataspaceKind::Null => Ok(Value::Array(Vec::new())),
            DataspaceKind::Simple => nest(values, &self.dataspace.dims),
        }
    }
}

/// Most empty sub-arrays built for a value with zero-size inner dimensions.
const MAX_EMPTY_ROWS: usize = 1 << 20;

/// Nest a flat row-major list of values according to `dims`.
fn nest(values: Vec<Value>, dims: &[u64]) -> Result<Value> {
    let Some((_, rest)) = dims.split_first() else {
        return single(values, "value");
    };
    if rest.is_empty() {
        return Ok(Value::Array(values));
    }
    let inner = rest
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))
        .ok_or(Error::LimitExceeded("attribute shape"))?;
    if inner == 0 {
        // Zero-size inner dimensions: `dims[0]` empty sub-arrays.
        let n = usize::try_from(dims[0])
            .ok()
            .filter(|&n| n <= MAX_EMPTY_ROWS)
            .ok_or(Error::LimitExceeded("attribute shape"))?;
        let empty = nest(Vec::new(), rest)?;
        return Ok(Value::Array(vec![empty; n]));
    }
    let mut out = Vec::with_capacity(values.len() / inner);
    let mut it = values.into_iter();
    loop {
        let chunk: Vec<Value> = it.by_ref().take(inner).collect();
        if chunk.is_empty() {
            break;
        }
        out.push(nest(chunk, rest)?);
    }
    Ok(Value::Array(out))
}

/// Resolve a datatype field of an attribute, returning the type and, for
/// committed datatypes, the address of the named datatype object.
pub(crate) fn resolve_datatype(
    file: &File,
    field: MaybeShared<Datatype>,
) -> Result<(Datatype, Option<u64>)> {
    match field {
        MaybeShared::Inline(t) => Ok((t, None)),
        MaybeShared::Shared(raw) => {
            let body = file.shared_message_body(MessageType::DATATYPE, &raw)?;
            Ok((Datatype::parse(&body)?, file.shared_message_address(&raw)?))
        }
    }
}

fn decode(file: &File, body: &[u8], creation_order: Option<u32>) -> Result<Attribute> {
    let sizes = file.sizes();
    let msg = AttributeMessage::parse(body, sizes)?;
    let both_inline = matches!(msg.datatype, MaybeShared::Inline(_))
        && matches!(msg.dataspace, MaybeShared::Inline(_));
    let (datatype, committed_datatype) = resolve_datatype(file, msg.datatype)?;
    let dataspace = match msg.dataspace {
        MaybeShared::Inline(s) => s,
        MaybeShared::Shared(raw) => {
            let body = file.shared_message_body(MessageType::DATASPACE, &raw)?;
            Dataspace::parse(&body, sizes)?
        }
    };
    let mut data = msg.data;
    if !both_inline {
        let len = dataspace
            .element_count()?
            .checked_mul(u64::from(datatype.size))
            .ok_or(Error::LimitExceeded("attribute data size"))?;
        let len = usize::try_from(len).map_err(|_| Error::UnexpectedEof)?;
        if data.len() < len {
            return Err(Error::UnexpectedEof);
        }
        data.truncate(len);
    }
    Ok(Attribute {
        file: file.clone(),
        name: msg.name,
        name_charset: msg.name_charset,
        datatype,
        dataspace,
        data,
        creation_order,
        committed_datatype,
    })
}

fn from_raw_message(file: &File, m: &RawMessage) -> Result<Attribute> {
    let body = file.message_body(m)?;
    decode(file, &body, m.creation_order.map(u32::from))
}

/// Attribute storage of one object header.
struct Dense {
    heap: FractalHeap,
    tree: BTreeV2,
}

fn dense(file: &File, header: &ObjectHeader) -> Result<Option<Dense>> {
    let Some(m) = header.find_message(MessageType::ATTRIBUTE_INFO) else {
        return Ok(None);
    };
    let info = AttributeInfo::parse(&file.message_body(m)?, file.sizes())?;
    if info.fractal_heap_address == UNDEFINED_ADDRESS
        || info.name_btree_address == UNDEFINED_ADDRESS
    {
        return Ok(None);
    }
    let store = file.store().as_ref();
    let heap = FractalHeap::open(store, info.fractal_heap_address, file.sizes())?;
    let ctx = RecordContext {
        heap_id_len: Some(usize::from(heap.header.heap_id_len)),
        ..RecordContext::default()
    };
    let tree = BTreeV2::open(store, info.name_btree_address, file.sizes(), ctx)?;
    if tree.header.tree_type != 8 {
        return Err(Error::invalid(
            "attribute name index is not a type 8 v2 B-tree",
        ));
    }
    Ok(Some(Dense { heap, tree }))
}

fn dense_record(file: &File, d: &Dense, rec: &Record) -> Result<Option<Attribute>> {
    let Record::AttributeName {
        heap_id,
        message_flags,
        creation_order,
        ..
    } = rec
    else {
        return Ok(None);
    };
    let store = file.store().as_ref();
    let mut body = d.heap.get(store, heap_id)?;
    if message_flags & RawMessage::FLAG_SHARED != 0 {
        body = file.shared_message_body(MessageType::ATTRIBUTE, &body)?;
    }
    decode(file, &body, Some(*creation_order)).map(Some)
}

/// Load every attribute of `header`, sorted by name or by creation order.
pub(crate) fn read_all(
    file: &File,
    header: &ObjectHeader,
    by_creation_order: bool,
) -> Result<Vec<Attribute>> {
    if by_creation_order && !header.attribute_creation_order_tracked() {
        return Err(Error::unsupported(
            "attribute creation order is not tracked for this object",
        ));
    }
    let mut out = header
        .messages_of_type(MessageType::ATTRIBUTE)
        .map(|m| from_raw_message(file, m))
        .collect::<Result<Vec<_>>>()?;
    if let Some(d) = dense(file, header)? {
        for rec in d.tree.records(file.store().as_ref())? {
            if let Some(a) = dense_record(file, &d, &rec)? {
                out.push(a);
            }
        }
    }
    if by_creation_order {
        out.sort_by_key(|a| a.creation_order.unwrap_or(u32::MAX));
    } else {
        out.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    }
    Ok(out)
}

/// Find the attribute called `name` on `header`.
pub(crate) fn find(file: &File, header: &ObjectHeader, name: &str) -> Result<Attribute> {
    for m in header.messages_of_type(MessageType::ATTRIBUTE) {
        let a = from_raw_message(file, m)?;
        if a.name == name {
            return Ok(a);
        }
    }
    if let Some(d) = dense(file, header)? {
        let hash = lookup3(name.as_bytes(), 0);
        let store = file.store().as_ref();
        let candidates = d.tree.find_all(store, |r| match r {
            Record::AttributeName { hash: h, .. } => h.cmp(&hash),
            _ => Ordering::Less,
        })?;
        for rec in candidates {
            if let Some(a) = dense_record(file, &d, &rec)?
                && a.name == name
            {
                return Ok(a);
            }
        }
    }
    Err(Error::NotFound(format!("attribute {name:?}")))
}
