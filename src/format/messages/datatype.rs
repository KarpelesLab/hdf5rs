//! Datatype message (type 0x0003).
//!
//! Spec: "IV.A.2.d. The Datatype Message". A datatype is encoded as a
//! 1-byte class+version, a 24-bit class bit field, a 4-byte element size and
//! class-specific properties. Compound, enumeration, variable-length, array
//! and complex types embed further datatype encodings recursively.

use crate::error::{Error, Result};
use crate::format::ByteReader;

/// Maximum nesting depth of embedded datatypes accepted by the parser.
pub const MAX_DATATYPE_DEPTH: usize = 32;

/// Byte order of an atomic datatype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Least significant byte first.
    LittleEndian,
    /// Most significant byte first.
    BigEndian,
    /// VAX mixed-endian order (floating-point only, datatype version 3+).
    Vax,
}

impl ByteOrder {
    /// The byte order of the running machine.
    pub fn native() -> ByteOrder {
        if cfg!(target_endian = "big") {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        }
    }

    fn from_bit(bit: bool) -> ByteOrder {
        if bit {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        }
    }
}

/// Character set of a string, variable-length string or name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacterSet {
    /// US-ASCII.
    Ascii,
    /// UTF-8.
    Utf8,
}

impl CharacterSet {
    /// Decode a 4-bit (or 8-bit) character set value.
    pub fn from_raw(v: u32) -> Result<CharacterSet> {
        match v {
            0 => Ok(CharacterSet::Ascii),
            1 => Ok(CharacterSet::Utf8),
            other => Err(Error::invalid(format!("reserved character set {other}"))),
        }
    }
}

/// Padding convention of fixed-length and variable-length strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPadding {
    /// NUL terminated (and NUL padded).
    NullTerminate,
    /// NUL padded, not necessarily terminated.
    NullPad,
    /// Space padded (Fortran style).
    SpacePad,
}

impl StringPadding {
    fn from_raw(v: u32) -> Result<StringPadding> {
        match v {
            0 => Ok(StringPadding::NullTerminate),
            1 => Ok(StringPadding::NullPad),
            2 => Ok(StringPadding::SpacePad),
            other => Err(Error::invalid(format!("reserved string padding {other}"))),
        }
    }
}

/// Fixed-point (integer) properties, class 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPoint {
    /// Byte order.
    pub byte_order: ByteOrder,
    /// Value of unused low bits.
    pub lo_pad: bool,
    /// Value of unused high bits.
    pub hi_pad: bool,
    /// Two's complement signed when true.
    pub signed: bool,
    /// Offset of the first significant bit.
    pub bit_offset: u16,
    /// Number of significant bits.
    pub bit_precision: u16,
}

/// Mantissa normalization of a floating-point type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MantissaNormalization {
    /// No normalization.
    None,
    /// The most significant mantissa bit is always set (except for 0.0).
    MsbSet,
    /// The most significant mantissa bit is not stored but implied (IEEE).
    Implied,
}

/// Floating-point properties, class 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatingPoint {
    /// Byte order (may be VAX).
    pub byte_order: ByteOrder,
    /// Value of unused low bits.
    pub lo_pad: bool,
    /// Value of unused high bits.
    pub hi_pad: bool,
    /// Value of unused internal bits.
    pub internal_pad: bool,
    /// Mantissa normalization.
    pub normalization: MantissaNormalization,
    /// Bit position of the sign bit.
    pub sign_location: u8,
    /// Offset of the first significant bit.
    pub bit_offset: u16,
    /// Number of significant bits.
    pub bit_precision: u16,
    /// Bit position of the exponent field.
    pub exponent_location: u8,
    /// Exponent size in bits.
    pub exponent_size: u8,
    /// Bit position of the mantissa field.
    pub mantissa_location: u8,
    /// Mantissa size in bits.
    pub mantissa_size: u8,
    /// Exponent bias.
    pub exponent_bias: u32,
}

impl FloatingPoint {
    /// Whether this is a standard IEEE 754 binary layout for `size` bytes
    /// (2, 4 or 8), ignoring byte order.
    pub fn is_ieee(&self, size: u32) -> bool {
        let (prec, eloc, esize, msize, bias) = match size {
            2 => (16, 10, 5, 10, 15),
            4 => (32, 23, 8, 23, 127),
            8 => (64, 52, 11, 52, 1023),
            _ => return false,
        };
        self.bit_offset == 0
            && self.bit_precision == prec
            && self.sign_location as u16 == prec - 1
            && self.exponent_location == eloc
            && self.exponent_size == esize
            && self.mantissa_location == 0
            && self.mantissa_size == msize
            && self.exponent_bias == bias
            && self.normalization == MantissaNormalization::Implied
    }
}

/// Time properties, class 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Time {
    /// Byte order.
    pub byte_order: ByteOrder,
    /// Number of significant bits.
    pub bit_precision: u16,
}

/// Fixed-length string properties, class 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringType {
    /// Padding convention.
    pub padding: StringPadding,
    /// Character set.
    pub charset: CharacterSet,
}

/// Bitfield properties, class 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bitfield {
    /// Byte order.
    pub byte_order: ByteOrder,
    /// Value of unused low bits.
    pub lo_pad: bool,
    /// Value of unused high bits.
    pub hi_pad: bool,
    /// Offset of the first significant bit.
    pub bit_offset: u16,
    /// Number of significant bits.
    pub bit_precision: u16,
}

/// One member of a compound datatype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompoundMember {
    /// Member name.
    pub name: String,
    /// Byte offset of the member within the compound element.
    pub offset: u32,
    /// Array dimensions of the member (datatype version 1 only; `None` for
    /// scalar members and for later versions, which use the array class).
    pub array_dims: Option<Vec<u32>>,
    /// Member datatype.
    pub datatype: Datatype,
}

/// Kind of a reference datatype, class 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    /// Object reference (`H5R_OBJECT1`): an object header address.
    Object,
    /// Dataset region reference (`H5R_DATASET_REGION1`): a global heap ID.
    DatasetRegion,
    /// Revised object reference (`H5R_OBJECT2`, datatype version 4+).
    Object2,
    /// Revised dataset region reference (`H5R_DATASET_REGION2`).
    DatasetRegion2,
    /// Attribute reference (`H5R_ATTR`).
    Attribute,
}

/// Reference datatype properties, class 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reference {
    /// Kind of reference.
    pub kind: ReferenceKind,
    /// Encoding version of revised references (bits 4-7; 0 for the
    /// backward-compatible kinds).
    pub encoding_version: u8,
}

/// One name/value pair of an enumeration datatype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumMember {
    /// Symbolic name.
    pub name: String,
    /// Raw value, encoded with the enumeration's base type.
    pub value: Vec<u8>,
}

/// Kind of a variable-length datatype, class 9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableLengthKind {
    /// A sequence of base-type elements.
    Sequence,
    /// A string.
    String {
        /// Padding convention.
        padding: StringPadding,
        /// Character set.
        charset: CharacterSet,
    },
}

/// Representation form of a complex number datatype, class 11.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexForm {
    /// Real and imaginary parts.
    Rectangular,
    /// Magnitude and angle.
    Polar,
    /// Exponential form.
    Exponential,
}

/// Class-specific description of a datatype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatatypeClass {
    /// Class 0: integers.
    FixedPoint(FixedPoint),
    /// Class 1: floating-point numbers.
    FloatingPoint(FloatingPoint),
    /// Class 2: time values.
    Time(Time),
    /// Class 3: fixed-length strings.
    String(StringType),
    /// Class 4: bit fields.
    Bitfield(Bitfield),
    /// Class 5: opaque bytes with a descriptive tag.
    Opaque {
        /// The ASCII tag (trailing NUL padding removed).
        tag: String,
    },
    /// Class 6: a structure of named members.
    Compound {
        /// Members in encoded order.
        members: Vec<CompoundMember>,
    },
    /// Class 7: object or region references.
    Reference(Reference),
    /// Class 8: named values of a base integer type.
    Enumeration {
        /// Base type of the values.
        base: Box<Datatype>,
        /// Name/value pairs in encoded order.
        members: Vec<EnumMember>,
    },
    /// Class 9: variable-length sequence or string.
    VariableLength {
        /// Sequence or string.
        kind: VariableLengthKind,
        /// Element type (an unsigned byte for strings).
        base: Box<Datatype>,
    },
    /// Class 10: fixed-size multidimensional array of a base type.
    Array {
        /// Array dimensions, slowest-changing first.
        dims: Vec<u32>,
        /// Permutation indices (datatype version 2 only).
        permutation: Option<Vec<u32>>,
        /// Element type.
        base: Box<Datatype>,
    },
    /// Class 11: complex numbers built from a floating-point base type.
    Complex {
        /// Both parts share the same base type.
        homogeneous: bool,
        /// Representation form.
        form: ComplexForm,
        /// Floating-point type of the parts.
        base: Box<Datatype>,
    },
}

/// A decoded datatype message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Datatype {
    /// Encoding version (1-5).
    pub version: u8,
    /// Size of one element in bytes.
    pub size: u32,
    /// Class-specific description.
    pub class: DatatypeClass,
}

impl Datatype {
    /// Parse a datatype message from the start of `data`.
    ///
    /// Trailing bytes (e.g. object header message padding) are ignored.
    pub fn parse(data: &[u8]) -> Result<Datatype> {
        Self::decode(&mut ByteReader::new(data))
    }

    /// Parse a datatype from `r`, leaving it positioned just past the
    /// encoding.
    pub fn decode(r: &mut ByteReader<'_>) -> Result<Datatype> {
        decode(r, 0)
    }

    /// Byte order of this type, if it has one. Enumerations, arrays and
    /// complex types report the byte order of their base type.
    pub fn byte_order(&self) -> Option<ByteOrder> {
        match &self.class {
            DatatypeClass::FixedPoint(f) => Some(f.byte_order),
            DatatypeClass::FloatingPoint(f) => Some(f.byte_order),
            DatatypeClass::Time(t) => Some(t.byte_order),
            DatatypeClass::Bitfield(b) => Some(b.byte_order),
            DatatypeClass::Enumeration { base, .. }
            | DatatypeClass::Array { base, .. }
            | DatatypeClass::Complex { base, .. } => base.byte_order(),
            _ => None,
        }
    }

    /// Whether this is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(self.class, DatatypeClass::FixedPoint(_))
    }

    /// Whether this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self.class, DatatypeClass::FloatingPoint(_))
    }

    /// Whether this is a full-width integer of `size` bytes and the given
    /// signedness in the machine's native byte order.
    pub fn is_native_int(&self, size: u32, signed: bool) -> bool {
        match &self.class {
            DatatypeClass::FixedPoint(f) => {
                self.size == size
                    && f.signed == signed
                    && f.byte_order == ByteOrder::native()
                    && f.bit_offset == 0
                    && Some(u32::from(f.bit_precision)) == size.checked_mul(8)
            }
            _ => false,
        }
    }

    /// Whether this is an IEEE binary32 in native byte order.
    pub fn is_native_f32(&self) -> bool {
        self.is_native_ieee(4)
    }

    /// Whether this is an IEEE binary64 in native byte order.
    pub fn is_native_f64(&self) -> bool {
        self.is_native_ieee(8)
    }

    fn is_native_ieee(&self, size: u32) -> bool {
        match &self.class {
            DatatypeClass::FloatingPoint(f) => {
                self.size == size && f.byte_order == ByteOrder::native() && f.is_ieee(size)
            }
            _ => false,
        }
    }
}

/// Read a NUL-terminated name; when `pad8` is set, skip NUL padding up to a
/// multiple of eight bytes measured from the start of the name.
fn read_name(r: &mut ByteReader<'_>, pad8: bool) -> Result<String> {
    let rest = r.rest();
    let len = rest
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| Error::invalid("unterminated name in datatype"))?;
    let name = String::from_utf8_lossy(&rest[..len]).into_owned();
    let mut total = len + 1;
    if pad8 {
        total = total
            .checked_next_multiple_of(8)
            .ok_or(Error::UnexpectedEof)?;
    }
    r.skip(total)?;
    Ok(name)
}

fn decode(r: &mut ByteReader<'_>, depth: usize) -> Result<Datatype> {
    if depth > MAX_DATATYPE_DEPTH {
        return Err(Error::LimitExceeded("datatype nesting depth"));
    }
    let class_version = r.read_u8()?;
    let version = class_version >> 4;
    let class_id = class_version & 0x0f;
    if !(1..=5).contains(&version) {
        return Err(Error::unsupported(format!("datatype version {version}")));
    }
    let b = r.read_array::<3>()?;
    let bits = u32::from(b[0]) | u32::from(b[1]) << 8 | u32::from(b[2]) << 16;
    let size = r.read_u32()?;
    let bit = |n: u32| bits & (1 << n) != 0;

    let class = match class_id {
        0 => DatatypeClass::FixedPoint(FixedPoint {
            byte_order: ByteOrder::from_bit(bit(0)),
            lo_pad: bit(1),
            hi_pad: bit(2),
            signed: bit(3),
            bit_offset: r.read_u16()?,
            bit_precision: r.read_u16()?,
        }),
        1 => {
            let byte_order = match (bit(6), bit(0)) {
                (false, false) => ByteOrder::LittleEndian,
                (false, true) => ByteOrder::BigEndian,
                (true, true) => ByteOrder::Vax,
                (true, false) => {
                    return Err(Error::invalid("reserved floating-point byte order"));
                }
            };
            let normalization = match (bits >> 4) & 3 {
                0 => MantissaNormalization::None,
                1 => MantissaNormalization::MsbSet,
                2 => MantissaNormalization::Implied,
                _ => return Err(Error::invalid("reserved mantissa normalization")),
            };
            DatatypeClass::FloatingPoint(FloatingPoint {
                byte_order,
                lo_pad: bit(1),
                hi_pad: bit(2),
                internal_pad: bit(3),
                normalization,
                sign_location: b[1],
                bit_offset: r.read_u16()?,
                bit_precision: r.read_u16()?,
                exponent_location: r.read_u8()?,
                exponent_size: r.read_u8()?,
                mantissa_location: r.read_u8()?,
                mantissa_size: r.read_u8()?,
                exponent_bias: r.read_u32()?,
            })
        }
        2 => DatatypeClass::Time(Time {
            byte_order: ByteOrder::from_bit(bit(0)),
            bit_precision: r.read_u16()?,
        }),
        3 => DatatypeClass::String(StringType {
            padding: StringPadding::from_raw(bits & 0x0f)?,
            charset: CharacterSet::from_raw((bits >> 4) & 0x0f)?,
        }),
        4 => DatatypeClass::Bitfield(Bitfield {
            byte_order: ByteOrder::from_bit(bit(0)),
            lo_pad: bit(1),
            hi_pad: bit(2),
            bit_offset: r.read_u16()?,
            bit_precision: r.read_u16()?,
        }),
        5 => {
            let tag = r.read_bytes((bits & 0xff) as usize)?;
            let end = tag.iter().position(|&c| c == 0).unwrap_or(tag.len());
            DatatypeClass::Opaque {
                tag: String::from_utf8_lossy(&tag[..end]).into_owned(),
            }
        }
        6 => decode_compound(r, version, bits & 0xffff, size, depth)?,
        7 => {
            let kind = match bits & 0x0f {
                0 => ReferenceKind::Object,
                1 => ReferenceKind::DatasetRegion,
                2 if version >= 4 => ReferenceKind::Object2,
                3 if version >= 4 => ReferenceKind::DatasetRegion2,
                4 if version >= 4 => ReferenceKind::Attribute,
                other => {
                    return Err(Error::invalid(format!("reserved reference type {other}")));
                }
            };
            DatatypeClass::Reference(Reference {
                kind,
                encoding_version: ((bits >> 4) & 0x0f) as u8,
            })
        }
        8 => {
            let count = (bits & 0xffff) as usize;
            let base = decode(r, depth + 1)?;
            let mut names = Vec::with_capacity(count.min(r.remaining()));
            for _ in 0..count {
                names.push(read_name(r, version < 3)?);
            }
            let vsize = base.size as usize;
            let total = vsize.checked_mul(count).ok_or(Error::UnexpectedEof)?;
            let values = r.read_bytes(total)?;
            let members = names
                .into_iter()
                .enumerate()
                .map(|(i, name)| EnumMember {
                    name,
                    value: values[i * vsize..(i + 1) * vsize].to_vec(),
                })
                .collect();
            DatatypeClass::Enumeration {
                base: Box::new(base),
                members,
            }
        }
        9 => {
            let kind = match bits & 0x0f {
                0 => VariableLengthKind::Sequence,
                1 => VariableLengthKind::String {
                    padding: StringPadding::from_raw((bits >> 4) & 0x0f)?,
                    charset: CharacterSet::from_raw((bits >> 8) & 0x0f)?,
                },
                other => {
                    return Err(Error::invalid(format!(
                        "reserved variable-length type {other}"
                    )));
                }
            };
            DatatypeClass::VariableLength {
                kind,
                base: Box::new(decode(r, depth + 1)?),
            }
        }
        10 => {
            if version < 2 {
                return Err(Error::invalid("array datatype requires version 2+"));
            }
            let ndims = r.read_u8()? as usize;
            if ndims == 0 {
                return Err(Error::invalid("array datatype with zero dimensions"));
            }
            if version == 2 {
                r.skip(3)?;
            }
            let mut dims = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                dims.push(r.read_u32()?);
            }
            let permutation = if version == 2 {
                let mut p = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    p.push(r.read_u32()?);
                }
                Some(p)
            } else {
                None
            };
            DatatypeClass::Array {
                dims,
                permutation,
                base: Box::new(decode(r, depth + 1)?),
            }
        }
        11 => {
            let form = match (bits >> 1) & 3 {
                0 => ComplexForm::Rectangular,
                1 => ComplexForm::Polar,
                2 => ComplexForm::Exponential,
                _ => return Err(Error::invalid("reserved complex number form")),
            };
            DatatypeClass::Complex {
                homogeneous: bit(0),
                form,
                base: Box::new(decode(r, depth + 1)?),
            }
        }
        other => {
            return Err(Error::unsupported(format!("datatype class {other}")));
        }
    };
    Ok(Datatype {
        version,
        size,
        class,
    })
}

fn decode_compound(
    r: &mut ByteReader<'_>,
    version: u8,
    count: u32,
    size: u32,
    depth: usize,
) -> Result<DatatypeClass> {
    // Version 3+: member offsets use the fewest bytes able to hold `size`.
    let offset_width = match size {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xff_ffff => 3,
        _ => 4,
    };
    let count = count as usize;
    let mut members = Vec::with_capacity(count.min(r.remaining() / 8));
    for _ in 0..count {
        let name = read_name(r, version < 3)?;
        let (offset, array_dims) = if version >= 3 {
            (r.read_uint(offset_width)? as u32, None)
        } else {
            let offset = r.read_u32()?;
            if version == 1 {
                let ndims = r.read_u8()? as usize;
                r.skip(3)?; // reserved
                let _permutation = r.read_u32()?;
                r.skip(4)?; // reserved
                let mut dims = [0u32; 4];
                for d in &mut dims {
                    *d = r.read_u32()?;
                }
                if ndims > 4 {
                    return Err(Error::invalid("compound member with more than 4 dims"));
                }
                (offset, (ndims > 0).then(|| dims[..ndims].to_vec()))
            } else {
                (offset, None)
            }
        };
        let datatype = decode(r, depth + 1)?;
        members.push(CompoundMember {
            name,
            offset,
            array_dims,
            datatype,
        });
    }
    Ok(DatatypeClass::Compound { members })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Datatype header: class/version, 24-bit bit field, size.
    fn header(version: u8, class: u8, bits: u32, size: u32) -> Vec<u8> {
        let mut v = vec![(version << 4) | class];
        v.extend_from_slice(&bits.to_le_bytes()[..3]);
        v.extend_from_slice(&size.to_le_bytes());
        v
    }

    fn int_type(version: u8, size: u32, signed: bool) -> Vec<u8> {
        let mut v = header(version, 0, if signed { 0x08 } else { 0 }, size);
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&(size.wrapping_mul(8) as u16).to_le_bytes());
        v
    }

    fn f64_le() -> Vec<u8> {
        let mut v = header(1, 1, 0x20 | (63 << 8), 8);
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&64u16.to_le_bytes());
        v.extend_from_slice(&[52, 11, 0, 52]);
        v.extend_from_slice(&1023u32.to_le_bytes());
        v
    }

    fn name8(name: &str) -> Vec<u8> {
        let mut v = name.as_bytes().to_vec();
        v.push(0);
        while !v.len().is_multiple_of(8) {
            v.push(0);
        }
        v
    }

    fn cname(name: &str) -> Vec<u8> {
        let mut v = name.as_bytes().to_vec();
        v.push(0);
        v
    }

    #[test]
    fn ieee_f64() {
        let t = Datatype::parse(&f64_le()).unwrap();
        assert_eq!(t.size, 8);
        assert_eq!(t.version, 1);
        let DatatypeClass::FloatingPoint(f) = &t.class else {
            panic!()
        };
        assert_eq!(f.sign_location, 63);
        assert_eq!(f.normalization, MantissaNormalization::Implied);
        assert!(f.is_ieee(8));
        assert_eq!(t.byte_order(), Some(ByteOrder::LittleEndian));
        assert_eq!(t.is_native_f64(), cfg!(target_endian = "little"));
        assert!(!t.is_native_f32());
    }

    #[test]
    fn vax_float_and_reserved_order() {
        let mut v = f64_le();
        v[1] |= 0x41;
        v[0] = 0x31;
        let t = Datatype::parse(&v).unwrap();
        assert_eq!(t.byte_order(), Some(ByteOrder::Vax));
        v[1] &= !0x01;
        assert!(Datatype::parse(&v).is_err());
    }

    #[test]
    fn integers() {
        let t = Datatype::parse(&int_type(1, 4, true)).unwrap();
        assert!(t.is_integer());
        assert_eq!(t.is_native_int(4, true), cfg!(target_endian = "little"));
        assert!(!t.is_native_int(4, false));
        let mut be = int_type(1, 2, false);
        be[1] |= 1;
        let t = Datatype::parse(&be).unwrap();
        assert_eq!(t.byte_order(), Some(ByteOrder::BigEndian));
    }

    #[test]
    fn time_string_bitfield_opaque() {
        let mut v = header(1, 2, 1, 4);
        v.extend_from_slice(&32u16.to_le_bytes());
        let t = Datatype::parse(&v).unwrap();
        assert_eq!(
            t.class,
            DatatypeClass::Time(Time {
                byte_order: ByteOrder::BigEndian,
                bit_precision: 32
            })
        );

        let t = Datatype::parse(&header(1, 3, 0x12, 10)).unwrap();
        assert_eq!(
            t.class,
            DatatypeClass::String(StringType {
                padding: StringPadding::SpacePad,
                charset: CharacterSet::Utf8
            })
        );
        assert!(Datatype::parse(&header(1, 3, 0x03, 10)).is_err());

        let mut v = header(1, 4, 0x06, 1);
        v.extend_from_slice(&[0, 0, 8, 0]);
        let DatatypeClass::Bitfield(b) = Datatype::parse(&v).unwrap().class else {
            panic!()
        };
        assert!(b.lo_pad && b.hi_pad);
        assert_eq!(b.bit_precision, 8);

        let mut v = header(1, 5, 8, 16);
        v.extend_from_slice(&name8("tag"));
        let t = Datatype::parse(&v).unwrap();
        assert_eq!(
            t.class,
            DatatypeClass::Opaque {
                tag: "tag".to_string()
            }
        );
    }

    #[test]
    fn compound_v1_with_array_member() {
        let mut v = header(1, 6, 2, 24);
        v.extend_from_slice(&name8("a"));
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        v.extend_from_slice(&[0; 16]);
        v.extend_from_slice(&f64_le());
        v.extend_from_slice(&name8("longer_name"));
        v.extend_from_slice(&8u32.to_le_bytes());
        v.extend_from_slice(&[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        for d in [2u32, 2, 0, 0] {
            v.extend_from_slice(&d.to_le_bytes());
        }
        v.extend_from_slice(&int_type(1, 1, false));
        let t = Datatype::parse(&v).unwrap();
        let DatatypeClass::Compound { members } = &t.class else {
            panic!()
        };
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].name, "a");
        assert!(members[0].array_dims.is_none());
        assert!(members[0].datatype.is_float());
        assert_eq!(members[1].name, "longer_name");
        assert_eq!(members[1].offset, 8);
        assert_eq!(members[1].array_dims.as_deref(), Some(&[2, 2][..]));
    }

    #[test]
    fn compound_v3_offset_width() {
        // size 300 -> 2-byte member offsets
        let mut v = header(3, 6, 2, 300);
        v.extend_from_slice(&cname("x"));
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&int_type(3, 4, true));
        v.extend_from_slice(&cname("yy"));
        v.extend_from_slice(&292u16.to_le_bytes());
        v.extend_from_slice(&f64_le());
        let mut r = ByteReader::new(&v);
        let t = Datatype::decode(&mut r).unwrap();
        assert!(r.is_empty());
        let DatatypeClass::Compound { members } = &t.class else {
            panic!()
        };
        assert_eq!(members[1].name, "yy");
        assert_eq!(members[1].offset, 292);
    }

    #[test]
    fn compound_v2_nested_array_and_vlen() {
        // compound { arr: array[3] of f64 (v2 array), s: vlen string }
        let mut arr = header(2, 10, 0, 24);
        arr.extend_from_slice(&[1, 0, 0, 0]);
        arr.extend_from_slice(&3u32.to_le_bytes());
        arr.extend_from_slice(&0u32.to_le_bytes());
        arr.extend_from_slice(&f64_le());
        let mut vl = header(1, 9, 0x101, 16);
        vl.extend_from_slice(&int_type(1, 1, false));
        let mut v = header(2, 6, 2, 40);
        v.extend_from_slice(&name8("arr"));
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&arr);
        v.extend_from_slice(&name8("s"));
        v.extend_from_slice(&24u32.to_le_bytes());
        v.extend_from_slice(&vl);
        let t = Datatype::parse(&v).unwrap();
        let DatatypeClass::Compound { members } = &t.class else {
            panic!()
        };
        let DatatypeClass::Array {
            dims,
            permutation,
            base,
        } = &members[0].datatype.class
        else {
            panic!()
        };
        assert_eq!(dims, &[3]);
        assert_eq!(permutation.as_deref(), Some(&[0][..]));
        assert!(base.is_float());
        let DatatypeClass::VariableLength { kind, base } = &members[1].datatype.class else {
            panic!()
        };
        assert_eq!(
            *kind,
            VariableLengthKind::String {
                padding: StringPadding::NullTerminate,
                charset: CharacterSet::Utf8
            }
        );
        assert_eq!(base.size, 1);
    }

    #[test]
    fn array_v3_no_permutation() {
        let mut v = header(3, 10, 0, 16);
        v.push(2);
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&int_type(3, 4, true));
        let t = Datatype::parse(&v).unwrap();
        let DatatypeClass::Array {
            dims, permutation, ..
        } = &t.class
        else {
            panic!()
        };
        assert_eq!(dims, &[2, 2]);
        assert!(permutation.is_none());
        // Array class in version 1 is invalid.
        v[0] = 0x1a;
        assert!(Datatype::parse(&v).is_err());
    }

    #[test]
    fn enumeration_versions() {
        for version in [1u8, 3] {
            let mut v = header(version, 8, 2, 1);
            v.extend_from_slice(&int_type(version, 1, false));
            for n in ["RED", "GREEN"] {
                if version < 3 {
                    v.extend_from_slice(&name8(n));
                } else {
                    v.extend_from_slice(&cname(n));
                }
            }
            v.extend_from_slice(&[0, 1]);
            let mut r = ByteReader::new(&v);
            let t = Datatype::decode(&mut r).unwrap();
            assert!(r.is_empty());
            let DatatypeClass::Enumeration { members, .. } = &t.class else {
                panic!()
            };
            assert_eq!(members[1].name, "GREEN");
            assert_eq!(members[1].value, vec![1]);
            assert_eq!(t.byte_order(), Some(ByteOrder::LittleEndian));
        }
    }

    #[test]
    fn references() {
        let t = Datatype::parse(&header(1, 7, 0, 8)).unwrap();
        assert_eq!(
            t.class,
            DatatypeClass::Reference(Reference {
                kind: ReferenceKind::Object,
                encoding_version: 0
            })
        );
        let t = Datatype::parse(&header(4, 7, 0x14, 64)).unwrap();
        assert_eq!(
            t.class,
            DatatypeClass::Reference(Reference {
                kind: ReferenceKind::Attribute,
                encoding_version: 1
            })
        );
        assert!(Datatype::parse(&header(3, 7, 0x02, 8)).is_err());
    }

    #[test]
    fn complex() {
        let mut v = header(5, 11, 1, 16);
        v.extend_from_slice(&f64_le());
        let t = Datatype::parse(&v).unwrap();
        let DatatypeClass::Complex {
            homogeneous, form, ..
        } = t.class
        else {
            panic!()
        };
        assert!(homogeneous);
        assert_eq!(form, ComplexForm::Rectangular);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(Datatype::parse(&[]).is_err());
        assert!(Datatype::parse(&f64_le()[..15]).is_err());
        assert!(Datatype::parse(&header(0, 0, 0, 4)).is_err());
        assert!(Datatype::parse(&header(1, 12, 0, 4)).is_err());
        // Compound claiming 65535 members with no data.
        assert!(Datatype::parse(&header(1, 6, 0xffff, 4)).is_err());
        // Enumeration with huge values.
        let mut v = header(1, 8, 1, 4);
        v.extend_from_slice(&int_type(1, 0xffff_ffff, false));
        v.extend_from_slice(&name8("A"));
        assert!(Datatype::parse(&v).is_err());
        // Unterminated compound member name.
        let mut v = header(3, 6, 1, 4);
        v.extend_from_slice(b"abc");
        assert!(Datatype::parse(&v).is_err());
    }

    #[test]
    fn recursion_bomb() {
        // 1000 nested vlen sequences.
        let mut v = Vec::new();
        for _ in 0..1000 {
            v.extend_from_slice(&header(1, 9, 0, 16));
        }
        v.extend_from_slice(&int_type(1, 1, false));
        assert!(matches!(Datatype::parse(&v), Err(Error::LimitExceeded(_))));
        // Within the limit it parses.
        let mut v = Vec::new();
        for _ in 0..10 {
            v.extend_from_slice(&header(1, 9, 0, 16));
        }
        v.extend_from_slice(&int_type(1, 1, false));
        assert!(Datatype::parse(&v).is_ok());
    }

    #[test]
    fn mutation_never_panics() {
        let mut arr = header(2, 10, 0, 24);
        arr.extend_from_slice(&[1, 0, 0, 0]);
        arr.extend_from_slice(&3u32.to_le_bytes());
        arr.extend_from_slice(&0u32.to_le_bytes());
        arr.extend_from_slice(&f64_le());
        let mut comp = header(3, 6, 2, 300);
        comp.extend_from_slice(&cname("x"));
        comp.extend_from_slice(&0u16.to_le_bytes());
        comp.extend_from_slice(&arr);
        comp.extend_from_slice(&cname("yy"));
        comp.extend_from_slice(&292u16.to_le_bytes());
        comp.extend_from_slice(&f64_le());
        let mut en = header(1, 8, 2, 1);
        en.extend_from_slice(&int_type(1, 1, false));
        en.extend_from_slice(&name8("A"));
        en.extend_from_slice(&name8("B"));
        en.extend_from_slice(&[0, 1]);
        let samples = [f64_le(), arr, comp, en];
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let mut v = samples[(rnd() % samples.len() as u64) as usize].clone();
            for _ in 0..(rnd() % 4 + 1) {
                let i = (rnd() % v.len() as u64) as usize;
                v[i] = rnd() as u8;
            }
            if rnd() % 4 == 0 {
                let n = (rnd() % v.len() as u64) as usize;
                v.truncate(n);
            }
            let _ = Datatype::parse(&v);
        }
    }
}
