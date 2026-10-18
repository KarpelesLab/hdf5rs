//! Datatype message encoding and convenience constructors.
//!
//! Spec: "IV.A.2.d. The Datatype Message". The encoder honours
//! [`Datatype::version`]: version 1/2 compound members and enumeration names
//! are padded to multiples of eight bytes, version 3+ use the compact
//! encodings; array types need version 2+ and revised references version 4.

use super::datatype::{
    Bitfield, ByteOrder, CharacterSet, CompoundMember, Datatype, DatatypeClass, EnumMember,
    FixedPoint, FloatingPoint, MAX_DATATYPE_DEPTH, MantissaNormalization, ReferenceKind,
    StringPadding, StringType, Time, VariableLengthKind,
};
use crate::error::{Error, Result};
use crate::format::{ByteWriter, Sizes};

fn order_bit(o: ByteOrder) -> Result<u32> {
    match o {
        ByteOrder::LittleEndian => Ok(0),
        ByteOrder::BigEndian => Ok(1),
        ByteOrder::Vax => Err(Error::invalid("VAX byte order is only valid for floats")),
    }
}

fn padding_bits(p: StringPadding) -> u32 {
    match p {
        StringPadding::NullTerminate => 0,
        StringPadding::NullPad => 1,
        StringPadding::SpacePad => 2,
    }
}

fn charset_bits(c: CharacterSet) -> u32 {
    match c {
        CharacterSet::Ascii => 0,
        CharacterSet::Utf8 => 1,
    }
}

/// Write a NUL-terminated name, padded to a multiple of 8 when `pad8`.
fn write_name(w: &mut ByteWriter, name: &str, pad8: bool) -> Result<()> {
    if name.as_bytes().contains(&0) {
        return Err(Error::invalid("datatype member name contains NUL"));
    }
    let start = w.len();
    w.bytes(name.as_bytes()).u8(0);
    if pad8 {
        let len = w.len() - start;
        w.zeros(len.next_multiple_of(8) - len);
    }
    Ok(())
}

/// Encoded length of an opaque tag: NUL-terminated and NUL-padded to a
/// multiple of eight bytes (an empty tag takes no space).
fn opaque_tag_len(tag: &str) -> usize {
    if tag.is_empty() {
        0
    } else {
        (tag.len() + 1).next_multiple_of(8)
    }
}

/// Width of version 3+ compound member offsets for a compound of `size`.
fn member_offset_width(size: u32) -> usize {
    match size {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xff_ffff => 3,
        _ => 4,
    }
}

fn encode(dt: &Datatype, w: &mut ByteWriter, depth: usize) -> Result<()> {
    if depth > MAX_DATATYPE_DEPTH {
        return Err(Error::LimitExceeded("datatype nesting depth"));
    }
    let version = dt.version;
    if !(1..=5).contains(&version) {
        return Err(Error::unsupported(format!("datatype version {version}")));
    }
    let (class_id, bits): (u8, u32) = match &dt.class {
        DatatypeClass::FixedPoint(f) => (
            0,
            order_bit(f.byte_order)?
                | u32::from(f.lo_pad) << 1
                | u32::from(f.hi_pad) << 2
                | u32::from(f.signed) << 3,
        ),
        DatatypeClass::FloatingPoint(f) => {
            let order = match f.byte_order {
                ByteOrder::LittleEndian => 0,
                ByteOrder::BigEndian => 1,
                ByteOrder::Vax if version >= 3 => 0x41,
                ByteOrder::Vax => {
                    return Err(Error::invalid("VAX byte order requires datatype version 3"));
                }
            };
            let norm = match f.normalization {
                MantissaNormalization::None => 0,
                MantissaNormalization::MsbSet => 1,
                MantissaNormalization::Implied => 2,
            };
            (
                1,
                order
                    | u32::from(f.lo_pad) << 1
                    | u32::from(f.hi_pad) << 2
                    | u32::from(f.internal_pad) << 3
                    | norm << 4
                    | u32::from(f.sign_location) << 8,
            )
        }
        DatatypeClass::Time(t) => (2, order_bit(t.byte_order)?),
        DatatypeClass::String(s) => (3, padding_bits(s.padding) | charset_bits(s.charset) << 4),
        DatatypeClass::Bitfield(b) => (
            4,
            order_bit(b.byte_order)? | u32::from(b.lo_pad) << 1 | u32::from(b.hi_pad) << 2,
        ),
        DatatypeClass::Opaque { tag } => {
            if tag.as_bytes().contains(&0) {
                return Err(Error::invalid("opaque tag contains NUL"));
            }
            let padded = opaque_tag_len(tag);
            if padded > 0xff {
                return Err(Error::LimitExceeded("opaque tag longer than 255 bytes"));
            }
            (5, padded as u32)
        }
        DatatypeClass::Compound { members } => {
            let n = u16::try_from(members.len())
                .map_err(|_| Error::LimitExceeded("compound with more than 65535 members"))?;
            (6, u32::from(n))
        }
        DatatypeClass::Reference(r) => {
            let kind = match r.kind {
                ReferenceKind::Object => 0,
                ReferenceKind::DatasetRegion => 1,
                ReferenceKind::Object2 => 2,
                ReferenceKind::DatasetRegion2 => 3,
                ReferenceKind::Attribute => 4,
            };
            if kind >= 2 && version < 4 {
                return Err(Error::invalid(
                    "revised references require datatype version 4",
                ));
            }
            (7, kind | u32::from(r.encoding_version & 0x0f) << 4)
        }
        DatatypeClass::Enumeration { members, .. } => {
            let n = u16::try_from(members.len())
                .map_err(|_| Error::LimitExceeded("enumeration with more than 65535 members"))?;
            (8, u32::from(n))
        }
        DatatypeClass::VariableLength { kind, .. } => (
            9,
            match kind {
                VariableLengthKind::Sequence => 0,
                VariableLengthKind::String { padding, charset } => {
                    1 | padding_bits(*padding) << 4 | charset_bits(*charset) << 8
                }
            },
        ),
        DatatypeClass::Array { .. } => {
            if version < 2 {
                return Err(Error::invalid("array datatype requires version 2+"));
            }
            (10, 0)
        }
        DatatypeClass::Complex {
            homogeneous, form, ..
        } => {
            let form = match form {
                super::datatype::ComplexForm::Rectangular => 0,
                super::datatype::ComplexForm::Polar => 1,
                super::datatype::ComplexForm::Exponential => 2,
            };
            (11, u32::from(*homogeneous) | form << 1)
        }
    };
    w.u8(version << 4 | class_id);
    w.bytes(&bits.to_le_bytes()[..3]);
    w.u32(dt.size);
    match &dt.class {
        DatatypeClass::FixedPoint(FixedPoint {
            bit_offset,
            bit_precision,
            ..
        })
        | DatatypeClass::Bitfield(Bitfield {
            bit_offset,
            bit_precision,
            ..
        }) => {
            w.u16(*bit_offset).u16(*bit_precision);
        }
        DatatypeClass::FloatingPoint(f) => {
            w.u16(f.bit_offset)
                .u16(f.bit_precision)
                .u8(f.exponent_location)
                .u8(f.exponent_size)
                .u8(f.mantissa_location)
                .u8(f.mantissa_size)
                .u32(f.exponent_bias);
        }
        DatatypeClass::Time(t) => {
            w.u16(t.bit_precision);
        }
        DatatypeClass::String(_) | DatatypeClass::Reference(_) => {}
        DatatypeClass::Opaque { tag } => {
            w.bytes(tag.as_bytes());
            w.zeros(opaque_tag_len(tag) - tag.len());
        }
        DatatypeClass::Compound { members } => {
            let width = member_offset_width(dt.size);
            for m in members {
                write_name(w, &m.name, version < 3)?;
                if version >= 3 {
                    w.write_uint(u64::from(m.offset), width)?;
                } else {
                    w.u32(m.offset);
                    if version == 1 {
                        let dims = m.array_dims.as_deref().unwrap_or(&[]);
                        if dims.len() > 4 {
                            return Err(Error::invalid("compound member with more than 4 dims"));
                        }
                        w.u8(dims.len() as u8).zeros(3).u32(0).zeros(4);
                        for i in 0..4 {
                            w.u32(dims.get(i).copied().unwrap_or(0));
                        }
                    } else if m.array_dims.is_some() {
                        return Err(Error::invalid(
                            "compound member array dims require datatype version 1",
                        ));
                    }
                }
                encode(&m.datatype, w, depth + 1)?;
            }
        }
        DatatypeClass::Enumeration { base, members } => {
            encode(base, w, depth + 1)?;
            for m in members {
                write_name(w, &m.name, version < 3)?;
            }
            for m in members {
                if m.value.len() != base.size as usize {
                    return Err(Error::invalid("enumeration value size differs from base"));
                }
                w.bytes(&m.value);
            }
        }
        DatatypeClass::VariableLength { base, .. } | DatatypeClass::Complex { base, .. } => {
            encode(base, w, depth + 1)?;
        }
        DatatypeClass::Array {
            dims,
            permutation,
            base,
        } => {
            let n = u8::try_from(dims.len())
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| Error::invalid("array datatype rank must be 1..=255"))?;
            w.u8(n);
            if version == 2 {
                w.zeros(3);
            }
            for &d in dims {
                w.u32(d);
            }
            if version == 2 {
                match permutation {
                    Some(p) if p.len() == dims.len() => p.iter().for_each(|&x| {
                        w.u32(x);
                    }),
                    Some(_) => return Err(Error::invalid("array permutation rank mismatch")),
                    None => (0..dims.len() as u32).for_each(|x| {
                        w.u32(x);
                    }),
                }
            }
            encode(base, w, depth + 1)?;
        }
    }
    Ok(())
}

impl Datatype {
    /// Encode the datatype message body.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut w = ByteWriter::new();
        self.encode_into(&mut w)?;
        Ok(w.into_vec())
    }

    /// Append the encoding to `w` (used for embedded datatypes, e.g. in
    /// attribute messages).
    pub fn encode_into(&self, w: &mut ByteWriter) -> Result<()> {
        encode(self, w, 0)
    }

    /// A full-width integer of `size` bytes (1, 2, 4 or 8, or any other
    /// non-zero size up to 8192), version 1.
    pub fn integer(size: u32, signed: bool, byte_order: ByteOrder) -> Datatype {
        Datatype {
            version: 1,
            size,
            class: DatatypeClass::FixedPoint(FixedPoint {
                byte_order,
                lo_pad: false,
                hi_pad: false,
                signed,
                bit_offset: 0,
                bit_precision: size.saturating_mul(8).min(u32::from(u16::MAX)) as u16,
            }),
        }
    }

    /// An IEEE 754 binary float of 2, 4 or 8 bytes, version 1.
    pub fn float(size: u32, byte_order: ByteOrder) -> Result<Datatype> {
        let (eloc, esize, msize, bias) = match size {
            2 => (10, 5, 10, 15),
            4 => (23, 8, 23, 127),
            8 => (52, 11, 52, 1023),
            _ => return Err(Error::unsupported(format!("IEEE float of {size} bytes"))),
        };
        let prec = (size * 8) as u16;
        Ok(Datatype {
            version: 1,
            size,
            class: DatatypeClass::FloatingPoint(FloatingPoint {
                byte_order,
                lo_pad: false,
                hi_pad: false,
                internal_pad: false,
                normalization: MantissaNormalization::Implied,
                sign_location: (prec - 1) as u8,
                bit_offset: 0,
                bit_precision: prec,
                exponent_location: eloc,
                exponent_size: esize,
                mantissa_location: 0,
                mantissa_size: msize,
                exponent_bias: bias,
            }),
        })
    }

    /// A time value of `size` bytes (class 2), version 1.
    pub fn time(size: u32, byte_order: ByteOrder) -> Datatype {
        Datatype {
            version: 1,
            size,
            class: DatatypeClass::Time(Time {
                byte_order,
                bit_precision: size.saturating_mul(8).min(u32::from(u16::MAX)) as u16,
            }),
        }
    }

    /// A fixed-length string of `size` bytes, version 1.
    pub fn fixed_string(size: u32, padding: StringPadding, charset: CharacterSet) -> Datatype {
        Datatype {
            version: 1,
            size,
            class: DatatypeClass::String(StringType { padding, charset }),
        }
    }

    /// A bitfield of `size` bytes, version 1.
    pub fn bitfield(size: u32, byte_order: ByteOrder) -> Datatype {
        Datatype {
            version: 1,
            size,
            class: DatatypeClass::Bitfield(Bitfield {
                byte_order,
                lo_pad: false,
                hi_pad: false,
                bit_offset: 0,
                bit_precision: size.saturating_mul(8).min(u32::from(u16::MAX)) as u16,
            }),
        }
    }

    /// An opaque type of `size` bytes with an ASCII `tag`, version 1.
    pub fn opaque(size: u32, tag: &str) -> Datatype {
        Datatype {
            version: 1,
            size,
            class: DatatypeClass::Opaque {
                tag: tag.to_owned(),
            },
        }
    }

    /// An object reference (`H5R_OBJECT1`): an object header address of
    /// "size of offsets" bytes.
    pub fn object_reference(sizes: Sizes) -> Datatype {
        Datatype {
            version: 1,
            size: u32::from(sizes.offset),
            class: DatatypeClass::Reference(super::datatype::Reference {
                kind: ReferenceKind::Object,
                encoding_version: 0,
            }),
        }
    }

    /// Size of a variable-length element: a 4-byte length plus a global
    /// heap ID.
    pub fn vlen_element_size(sizes: Sizes) -> u32 {
        4 + u32::from(sizes.offset) + 4
    }

    /// A variable-length, NUL-terminated string (base type: unsigned
    /// byte), as written by the reference library.
    pub fn vlen_string(charset: CharacterSet, sizes: Sizes) -> Datatype {
        Datatype {
            version: 1,
            size: Self::vlen_element_size(sizes),
            class: DatatypeClass::VariableLength {
                kind: VariableLengthKind::String {
                    padding: StringPadding::NullTerminate,
                    charset,
                },
                base: Box::new(Datatype::integer(1, false, ByteOrder::LittleEndian)),
            },
        }
    }

    /// A variable-length sequence of `base` elements.
    pub fn vlen_sequence(base: Datatype, sizes: Sizes) -> Datatype {
        Datatype {
            version: base.version.max(1),
            size: Self::vlen_element_size(sizes),
            class: DatatypeClass::VariableLength {
                kind: VariableLengthKind::Sequence,
                base: Box::new(base),
            },
        }
    }

    /// A fixed-size array of `base` elements (version 3 or the base's
    /// version if higher). The size saturates on overflow, which the
    /// encoder's consumers reject.
    pub fn array(base: Datatype, dims: &[u32]) -> Datatype {
        let count = dims
            .iter()
            .try_fold(1u32, |acc, &d| acc.checked_mul(d))
            .and_then(|n| n.checked_mul(base.size))
            .unwrap_or(u32::MAX);
        Datatype {
            version: base.version.max(3),
            size: count,
            class: DatatypeClass::Array {
                dims: dims.to_vec(),
                permutation: None,
                base: Box::new(base),
            },
        }
    }

    /// A compound type of `size` bytes with explicit member offsets
    /// (version 3 or the highest member version).
    pub fn compound(size: u32, members: Vec<CompoundMember>) -> Result<Datatype> {
        for m in &members {
            let end = m.offset.checked_add(m.datatype.size);
            if end.is_none_or(|e| e > size) {
                return Err(Error::invalid(format!(
                    "compound member {:?} extends past the compound size",
                    m.name
                )));
            }
        }
        let version = members
            .iter()
            .map(|m| m.datatype.version)
            .max()
            .unwrap_or(3)
            .max(3);
        Ok(Datatype {
            version,
            size,
            class: DatatypeClass::Compound { members },
        })
    }

    /// A packed compound: members laid out back to back in order.
    pub fn packed_compound(members: Vec<(String, Datatype)>) -> Result<Datatype> {
        let mut offset = 0u32;
        let mut out = Vec::with_capacity(members.len());
        for (name, datatype) in members {
            let size = datatype.size;
            out.push(CompoundMember {
                name,
                offset,
                array_dims: None,
                datatype,
            });
            offset = offset
                .checked_add(size)
                .ok_or(Error::LimitExceeded("compound size overflows"))?;
        }
        Self::compound(offset, out)
    }

    /// An enumeration over an integer `base` type; values are given as
    /// signed integers and encoded with the base type.
    pub fn enumeration(base: Datatype, members: &[(&str, i64)]) -> Result<Datatype> {
        let DatatypeClass::FixedPoint(f) = &base.class else {
            return Err(Error::invalid("enumeration base must be an integer type"));
        };
        let size = base.size as usize;
        if !(1..=8).contains(&size) {
            return Err(Error::unsupported("enumeration base wider than 8 bytes"));
        }
        let mut out = Vec::with_capacity(members.len());
        for &(name, v) in members {
            let fits = if f.signed {
                size == 8 || (v >> (size * 8 - 1) == 0 || v >> (size * 8 - 1) == -1)
            } else {
                v >= 0 && (size == 8 || (v as u64) >> (size * 8) == 0)
            };
            if !fits {
                return Err(Error::invalid(format!(
                    "enumeration value {v} does not fit the base type"
                )));
            }
            let le = v.to_le_bytes();
            let mut value = le[..size].to_vec();
            if f.byte_order == ByteOrder::BigEndian {
                value.reverse();
            }
            out.push(EnumMember {
                name: name.to_owned(),
                value,
            });
        }
        Ok(Datatype {
            version: base.version.max(3),
            size: base.size,
            class: DatatypeClass::Enumeration {
                base: Box::new(base),
                members: out,
            },
        })
    }

    /// The boolean type used by h5py: an `int8` enumeration with
    /// `FALSE = 0` and `TRUE = 1`.
    pub fn bool_enum() -> Datatype {
        Datatype {
            version: 3,
            size: 1,
            class: DatatypeClass::Enumeration {
                base: Box::new(Datatype::integer(1, true, ByteOrder::LittleEndian)),
                members: vec![
                    EnumMember {
                        name: "FALSE".into(),
                        value: vec![0],
                    },
                    EnumMember {
                        name: "TRUE".into(),
                        value: vec![1],
                    },
                ],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::messages::datatype::{ComplexForm, Reference};

    fn round_trip(dt: &Datatype) {
        let enc = dt.encode().unwrap();
        let back = Datatype::parse(&enc).unwrap();
        assert_eq!(&back, dt, "encoding {enc:02x?}");
    }

    fn all_types() -> Vec<Datatype> {
        let s = Sizes::DEFAULT;
        let i32le = Datatype::integer(4, true, ByteOrder::LittleEndian);
        let f64be = Datatype::float(8, ByteOrder::BigEndian).unwrap();
        let mut v = vec![
            i32le.clone(),
            Datatype::integer(2, false, ByteOrder::BigEndian),
            Datatype::float(2, ByteOrder::LittleEndian).unwrap(),
            Datatype::float(4, ByteOrder::LittleEndian).unwrap(),
            f64be.clone(),
            Datatype::time(8, ByteOrder::BigEndian),
            Datatype::fixed_string(10, StringPadding::NullPad, CharacterSet::Ascii),
            Datatype::fixed_string(3, StringPadding::SpacePad, CharacterSet::Utf8),
            Datatype::bitfield(2, ByteOrder::BigEndian),
            Datatype::opaque(7, "tag"),
            Datatype::opaque(4, ""),
            Datatype::opaque(4, "exactly8"),
            Datatype::object_reference(s),
            Datatype::vlen_string(CharacterSet::Utf8, s),
            Datatype::vlen_sequence(i32le.clone(), s),
            Datatype::array(f64be.clone(), &[2, 3]),
            Datatype::bool_enum(),
            Datatype::enumeration(
                Datatype::integer(2, true, ByteOrder::BigEndian),
                &[("LOW", -100), ("HIGH", 1000)],
            )
            .unwrap(),
            Datatype::packed_compound(vec![
                ("a".into(), i32le.clone()),
                ("b".into(), Datatype::array(i32le.clone(), &[3])),
                ("name".into(), Datatype::vlen_string(CharacterSet::Ascii, s)),
            ])
            .unwrap(),
            Datatype {
                version: 3,
                size: 16,
                class: DatatypeClass::Complex {
                    homogeneous: true,
                    form: ComplexForm::Rectangular,
                    base: Box::new(f64be.clone()),
                },
            },
            Datatype {
                version: 4,
                size: 64,
                class: DatatypeClass::Reference(Reference {
                    kind: ReferenceKind::Object2,
                    encoding_version: 0,
                }),
            },
        ];
        // A big compound (2-byte member offsets) nested in an array.
        let big = Datatype::compound(
            300,
            vec![
                CompoundMember {
                    name: "x".into(),
                    offset: 0,
                    array_dims: None,
                    datatype: i32le.clone(),
                },
                CompoundMember {
                    name: "y".into(),
                    offset: 292,
                    array_dims: None,
                    datatype: f64be.clone(),
                },
            ],
        )
        .unwrap();
        v.push(Datatype::array(big.clone(), &[2]));
        v.push(big);
        v
    }

    #[test]
    fn encode_round_trip_v3() {
        for dt in all_types() {
            round_trip(&dt);
        }
    }

    #[test]
    fn encode_round_trip_old_versions() {
        let i32le = Datatype::integer(4, true, ByteOrder::LittleEndian);
        for version in [1, 2] {
            let mut c = Datatype::packed_compound(vec![
                ("first".into(), i32le.clone()),
                ("second_member".into(), i32le.clone()),
            ])
            .unwrap();
            c.version = version;
            round_trip(&c);
            let mut e = Datatype::bool_enum();
            e.version = version;
            round_trip(&e);
        }
        let mut c = Datatype::packed_compound(vec![("a".into(), i32le.clone())]).unwrap();
        c.version = 1;
        if let DatatypeClass::Compound { members } = &mut c.class {
            members[0].array_dims = Some(vec![1]);
        }
        round_trip(&c);
        let mut a = Datatype::array(i32le.clone(), &[4, 5]);
        a.version = 2;
        let enc = a.encode().unwrap();
        let back = Datatype::parse(&enc).unwrap();
        let DatatypeClass::Array { permutation, .. } = back.class else {
            panic!()
        };
        assert_eq!(permutation, Some(vec![0, 1]));
    }

    #[test]
    fn encode_errors() {
        let i32le = Datatype::integer(4, true, ByteOrder::LittleEndian);
        let mut a = Datatype::array(i32le.clone(), &[4]);
        a.version = 1;
        assert!(a.encode().is_err());
        assert!(Datatype::array(i32le.clone(), &[]).encode().is_err());
        assert!(Datatype::float(3, ByteOrder::LittleEndian).is_err());
        assert!(Datatype::enumeration(i32le.clone(), &[("x", 1 << 40)]).is_err());
        assert!(
            Datatype::enumeration(
                Datatype::integer(1, false, ByteOrder::LittleEndian),
                &[("x", -1)]
            )
            .is_err()
        );
        assert!(Datatype::opaque(1, "a\0b").encode().is_err());
        assert!(
            Datatype::compound(
                2,
                vec![CompoundMember {
                    name: "x".into(),
                    offset: 0,
                    array_dims: None,
                    datatype: i32le,
                }]
            )
            .is_err()
        );
        let mut bad = Datatype::bool_enum();
        bad.version = 9;
        assert!(bad.encode().is_err());
    }
}
