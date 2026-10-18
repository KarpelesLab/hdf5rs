//! A dynamic value model covering every datatype class.

use std::fmt;

use super::DecodeContext;
use super::bits::{self, Int};
use super::h5type::{array_parts, int_value, member_type, string_bytes};
use super::reference::{Reference, decode_reference};
use super::vlen::fetch_vlen;
use crate::error::{Error, Result};
use crate::format::messages::datatype::{ComplexForm, Datatype, DatatypeClass, VariableLengthKind};

/// Maximum nesting of decoded values (bounded by the datatype depth limit
/// in practice; kept as a defensive guard).
const MAX_VALUE_DEPTH: usize = 64;

/// A decoded element of any datatype.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A signed integer.
    Int(i64),
    /// An unsigned integer.
    UInt(u64),
    /// A floating-point number (non-IEEE and extended layouts are rounded
    /// to the nearest `f64`).
    Float(f64),
    /// A boolean: an enumeration with exactly the members `FALSE = 0` and
    /// `TRUE = 1` (h5py's encoding of `bool`).
    Bool(bool),
    /// A fixed or variable-length string that is valid UTF-8 (padding
    /// removed).
    String(String),
    /// A string that is not valid UTF-8 (padding removed).
    Bytes(Vec<u8>),
    /// An enumeration value: the member name if the value matches one, and
    /// the integer value.
    Enum {
        /// Matching member name.
        name: Option<String>,
        /// Underlying integer value.
        value: i128,
    },
    /// The significant bits of a bitfield.
    Bitfield(u64),
    /// Opaque bytes.
    Opaque(Vec<u8>),
    /// A time value (seconds since the epoch, per the time class).
    Time(i64),
    /// A complex number (rectangular form).
    Complex {
        /// Real part.
        re: f64,
        /// Imaginary part.
        im: f64,
    },
    /// An array datatype element; multi-dimensional arrays are nested,
    /// slowest-changing dimension outermost.
    Array(Vec<Value>),
    /// A compound element: members in datatype order.
    Compound(Vec<(String, Value)>),
    /// A variable-length sequence.
    VarLen(Vec<Value>),
    /// A reference.
    Reference(Reference),
}

impl Value {
    /// The value as `i64`, for integers, enumerations, booleans and times
    /// that fit.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(v) | Value::Time(v) => Some(*v),
            Value::UInt(v) => i64::try_from(*v).ok(),
            Value::Enum { value, .. } => i64::try_from(*value).ok(),
            Value::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }

    /// The value as `u64`, for non-negative integers, enumerations,
    /// booleans and bitfields.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::UInt(v) | Value::Bitfield(v) => Some(*v),
            Value::Int(v) | Value::Time(v) => u64::try_from(*v).ok(),
            Value::Enum { value, .. } => u64::try_from(*value).ok(),
            Value::Bool(b) => Some(u64::from(*b)),
            _ => None,
        }
    }

    /// The value as `f64` (lossy for large integers).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Int(v) | Value::Time(v) => Some(*v as f64),
            Value::UInt(v) => Some(*v as f64),
            Value::Enum { value, .. } => Some(*value as f64),
            Value::Bool(b) => Some(f64::from(u8::from(*b))),
            _ => None,
        }
    }

    /// The value as `bool`: booleans, and integers / enumerations (zero is
    /// `false`).
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            Value::Int(v) => Some(*v != 0),
            Value::UInt(v) => Some(*v != 0),
            Value::Enum { value, .. } => Some(*value != 0),
            _ => None,
        }
    }

    /// The string, if this is a valid UTF-8 string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// The elements of an array or variable-length sequence.
    pub fn as_slice(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) | Value::VarLen(v) => Some(v),
            _ => None,
        }
    }

    /// Member `name` of a compound value.
    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Compound(m) => m.iter().find(|(n, _)| n == name).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// Decode one element of `dt` into a [`Value`].
///
/// `bytes` must be exactly `dt.size` bytes. Variable-length data and
/// dataset region references need a resolver in `ctx`.
pub fn decode_value(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Value> {
    decode(dt, bytes, ctx, 0)
}

fn int_to_value(v: Int) -> Result<Value> {
    match v {
        Int::Signed(s) => i64::try_from(s).map(Value::Int).ok(),
        Int::Unsigned(u) => u64::try_from(u).map(Value::UInt).ok(),
    }
    .ok_or_else(|| Error::unsupported("integer value wider than 64 bits"))
}

fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>, depth: usize) -> Result<Value> {
    if depth > MAX_VALUE_DEPTH {
        return Err(Error::LimitExceeded("value nesting depth"));
    }
    if bytes.len() != dt.size as usize {
        return Err(Error::invalid(format!(
            "element of {} bytes for a {}-byte datatype",
            bytes.len(),
            dt.size
        )));
    }
    match &dt.class {
        DatatypeClass::FixedPoint(_) => int_to_value(int_value(dt, bytes)?),
        DatatypeClass::FloatingPoint(fp) => {
            bits::validate_float(fp, dt.size)?;
            if fp.is_ieee(dt.size) && dt.size == 2 {
                let raw = bits::uint_from_bytes(bytes, fp.byte_order)? as u16;
                return Ok(Value::Float(f64::from(bits::f16_to_f32(raw))));
            }
            if fp.is_ieee(dt.size) && dt.size == 4 {
                let raw = bits::uint_from_bytes(bytes, fp.byte_order)? as u32;
                return Ok(Value::Float(f64::from(f32::from_bits(raw))));
            }
            if fp.is_ieee(dt.size) && dt.size == 8 {
                let raw = bits::uint_from_bytes(bytes, fp.byte_order)? as u64;
                return Ok(Value::Float(f64::from_bits(raw)));
            }
            Ok(Value::Float(bits::decode_float(fp, bytes)?))
        }
        DatatypeClass::Time(t) => {
            if dt.size == 0 || dt.size > 8 {
                return Err(Error::unsupported(format!("{}-byte time values", dt.size)));
            }
            let raw = bits::uint_from_bytes(bytes, t.byte_order)?;
            // Interpret as a two's complement integer of the full size.
            let shift = 128 - dt.size * 8;
            Ok(Value::Time((((raw << shift) as i128) >> shift) as i64))
        }
        DatatypeClass::String(_) => Ok(string_value(string_bytes(dt, bytes, ctx)?)),
        DatatypeClass::Bitfield(b) => {
            let fp = crate::format::messages::datatype::FixedPoint {
                byte_order: b.byte_order,
                lo_pad: b.lo_pad,
                hi_pad: b.hi_pad,
                signed: false,
                bit_offset: b.bit_offset,
                bit_precision: b.bit_precision,
            };
            bits::validate_fixed(&fp, dt.size)?;
            match bits::decode_fixed(&fp, bytes)? {
                Int::Unsigned(u) => u64::try_from(u)
                    .map(Value::Bitfield)
                    .map_err(|_| Error::unsupported("bitfields wider than 64 bits")),
                Int::Signed(_) => Err(Error::invalid("signed bitfield")),
            }
        }
        DatatypeClass::Opaque { .. } => Ok(Value::Opaque(bytes.to_vec())),
        DatatypeClass::Compound { members } => {
            let mut out = Vec::with_capacity(members.len());
            for m in members {
                let t = member_type(m)?;
                let off = m.offset as usize;
                let b = off
                    .checked_add(t.size as usize)
                    .and_then(|end| bytes.get(off..end))
                    .ok_or_else(|| {
                        Error::invalid(format!("compound member {:?} exceeds its compound", m.name))
                    })?;
                out.push((m.name.clone(), decode(&t, b, ctx, depth + 1)?));
            }
            Ok(Value::Compound(out))
        }
        DatatypeClass::Reference(_) => Ok(Value::Reference(decode_reference(dt, bytes, ctx)?)),
        DatatypeClass::Enumeration { base, members } => {
            let value = int_value(dt, bytes)?
                .to_i128()
                .ok_or_else(|| Error::unsupported("enumeration value wider than 127 bits"))?;
            let is_bool = members.len() == 2
                && members[0].name == "FALSE"
                && members[1].name == "TRUE"
                && enum_member_value(base, &members[0].value) == Some(0)
                && enum_member_value(base, &members[1].value) == Some(1);
            if is_bool && (value == 0 || value == 1) {
                return Ok(Value::Bool(value == 1));
            }
            let name = members
                .iter()
                .find(|m| enum_member_value(base, &m.value) == Some(value))
                .map(|m| m.name.clone());
            Ok(Value::Enum { name, value })
        }
        DatatypeClass::VariableLength { kind, base } => match kind {
            VariableLengthKind::String { .. } => Ok(string_value(string_bytes(dt, bytes, ctx)?)),
            VariableLengthKind::Sequence => {
                let data = fetch_vlen(bytes, base.size, ctx)?;
                // fetch_vlen guarantees a non-zero base size when non-empty.
                let size = (base.size as usize).max(1);
                let mut out = Vec::with_capacity(data.len() / size);
                for c in data.chunks_exact(size) {
                    out.push(decode(base, c, ctx, depth + 1)?);
                }
                Ok(Value::VarLen(out))
            }
        },
        DatatypeClass::Array { dims, base, .. } => {
            array_parts(dt)?;
            decode_array(dims, base, bytes, ctx, depth)
        }
        DatatypeClass::Complex { form, base, .. } => {
            if *form != ComplexForm::Rectangular {
                return Err(Error::unsupported(
                    "complex numbers in polar/exponential form",
                ));
            }
            let half = base.size as usize;
            if half.checked_mul(2) != Some(bytes.len()) {
                return Err(Error::invalid("complex datatype size mismatch"));
            }
            let part = |b: &[u8]| -> Result<f64> {
                decode(base, b, ctx, depth + 1)?
                    .as_f64()
                    .ok_or_else(|| Error::invalid("complex part is not numeric"))
            };
            Ok(Value::Complex {
                re: part(&bytes[..half])?,
                im: part(&bytes[half..])?,
            })
        }
    }
}

/// Nested array decoding, slowest dimension first.
fn decode_array(
    dims: &[u32],
    base: &Datatype,
    bytes: &[u8],
    ctx: &DecodeContext<'_>,
    depth: usize,
) -> Result<Value> {
    let Some((&first, rest)) = dims.split_first() else {
        return decode(base, bytes, ctx, depth + 1);
    };
    let n = first as usize;
    if n == 0 {
        return Ok(Value::Array(Vec::new()));
    }
    let chunk = bytes.len() / n;
    if chunk == 0 || chunk * n != bytes.len() {
        return Err(Error::invalid("array datatype size mismatch"));
    }
    let mut out = Vec::with_capacity(n);
    for c in bytes.chunks_exact(chunk) {
        out.push(decode_array(rest, base, c, ctx, depth + 1)?);
    }
    Ok(Value::Array(out))
}

fn enum_member_value(base: &Datatype, raw: &[u8]) -> Option<i128> {
    int_value(base, raw).ok()?.to_i128()
}

fn string_value(b: Vec<u8>) -> Value {
    match String::from_utf8(b) {
        Ok(s) => Value::String(s),
        Err(e) => Value::Bytes(e.into_bytes()),
    }
}

fn write_list<'a>(
    f: &mut fmt::Formatter<'_>,
    open: &str,
    close: &str,
    items: impl Iterator<Item = &'a Value>,
) -> fmt::Result {
    f.write_str(open)?;
    for (i, v) in items.enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{v}")?;
    }
    f.write_str(close)
}

fn write_hex(f: &mut fmt::Formatter<'_>, b: &[u8]) -> fmt::Result {
    f.write_str("0x")?;
    for x in b {
        write!(f, "{x:02x}")?;
    }
    Ok(())
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(v) | Value::Time(v) => write!(f, "{v}"),
            Value::UInt(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v:?}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::String(s) => write!(f, "{s:?}"),
            Value::Bytes(b) | Value::Opaque(b) => write_hex(f, b),
            Value::Enum { name: Some(n), .. } => f.write_str(n),
            Value::Enum { name: None, value } => write!(f, "{value}"),
            Value::Bitfield(v) => write!(f, "{v:#x}"),
            Value::Complex { re, im } => {
                if *im < 0.0 {
                    write!(f, "{re:?}-{:?}i", -im)
                } else {
                    write!(f, "{re:?}+{im:?}i")
                }
            }
            Value::Array(v) | Value::VarLen(v) => write_list(f, "[", "]", v.iter()),
            Value::Compound(m) => {
                f.write_str("{")?;
                for (i, (n, v)) in m.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{n}: {v}")?;
                }
                f.write_str("}")
            }
            Value::Reference(r) => match r {
                Reference::Null => f.write_str("null reference"),
                Reference::Object(a) => write!(f, "object@{a:#x}"),
                Reference::Region(r) => write!(f, "region@{:#x}", r.object),
                Reference::Revised(r) => {
                    match r.object_address() {
                        Some(a) => write!(f, "{:?}@{a:#x}", r.kind)?,
                        None => write!(f, "{:?}", r.kind)?,
                    }
                    if let Some(file) = &r.external_file {
                        write!(f, " in {file:?}")?;
                    }
                    if let Some(a) = &r.attribute {
                        write!(f, " attribute {a:?}")?;
                    }
                    Ok(())
                }
            },
        }
    }
}
