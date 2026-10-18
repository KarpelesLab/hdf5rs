//! Human-readable descriptions of datatypes.

use std::fmt;

use super::h5type::int_value_opt;
use crate::format::messages::datatype::{
    ByteOrder, CharacterSet, ComplexForm, Datatype, DatatypeClass, ReferenceKind, StringPadding,
    VariableLengthKind,
};

fn order(o: ByteOrder) -> &'static str {
    match o {
        ByteOrder::LittleEndian => "LE",
        ByteOrder::BigEndian => "BE",
        ByteOrder::Vax => "VAX",
    }
}

fn charset(c: CharacterSet) -> &'static str {
    match c {
        CharacterSet::Ascii => "ASCII",
        CharacterSet::Utf8 => "UTF-8",
    }
}

fn padding(p: StringPadding) -> &'static str {
    match p {
        StringPadding::NullTerminate => "null-terminated",
        StringPadding::NullPad => "null-padded",
        StringPadding::SpacePad => "space-padded",
    }
}

/// Formats e.g. `i32 LE`, `f64 BE`, `string[8] ASCII null-terminated`,
/// `compound[16] { a: i32 LE @ 0, b: f64 LE @ 8 }`, `array[2x3] of u8`,
/// `vlen string UTF-8`, `enum i8 { FALSE = 0, TRUE = 1 }`.
impl fmt::Display for Datatype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bits = u64::from(self.size) * 8;
        match &self.class {
            DatatypeClass::FixedPoint(fp) => {
                let s = if fp.signed { 'i' } else { 'u' };
                if self.size == 1 {
                    write!(f, "{s}8")?;
                } else {
                    write!(f, "{s}{bits} {}", order(fp.byte_order))?;
                }
                if u64::from(fp.bit_precision) != bits || fp.bit_offset != 0 {
                    write!(
                        f,
                        " (precision {}, offset {})",
                        fp.bit_precision, fp.bit_offset
                    )?;
                }
                Ok(())
            }
            DatatypeClass::FloatingPoint(fp) => {
                if fp.is_ieee(self.size) {
                    write!(f, "f{bits} {}", order(fp.byte_order))
                } else {
                    write!(
                        f,
                        "float{bits} {} (sign {}, exponent {}@{} bias {}, mantissa {}@{})",
                        order(fp.byte_order),
                        fp.sign_location,
                        fp.exponent_size,
                        fp.exponent_location,
                        fp.exponent_bias,
                        fp.mantissa_size,
                        fp.mantissa_location
                    )
                }
            }
            DatatypeClass::Time(t) => write!(f, "time{bits} {}", order(t.byte_order)),
            DatatypeClass::String(s) => write!(
                f,
                "string[{}] {} {}",
                self.size,
                charset(s.charset),
                padding(s.padding)
            ),
            DatatypeClass::Bitfield(b) => write!(f, "bitfield{bits} {}", order(b.byte_order)),
            DatatypeClass::Opaque { tag } => {
                write!(f, "opaque[{}]", self.size)?;
                if !tag.is_empty() {
                    write!(f, " {tag:?}")?;
                }
                Ok(())
            }
            DatatypeClass::Compound { members } => {
                write!(f, "compound[{}] {{ ", self.size)?;
                for (i, m) in members.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}: ", m.name)?;
                    if let Some(dims) = &m.array_dims {
                        write_dims(f, dims)?;
                        f.write_str(" of ")?;
                    }
                    write!(f, "{} @ {}", m.datatype, m.offset)?;
                }
                f.write_str(" }")
            }
            DatatypeClass::Reference(r) => f.write_str(match r.kind {
                ReferenceKind::Object => "object reference",
                ReferenceKind::DatasetRegion => "dataset region reference",
                ReferenceKind::Object2 => "object reference (revised)",
                ReferenceKind::DatasetRegion2 => "dataset region reference (revised)",
                ReferenceKind::Attribute => "attribute reference",
            }),
            DatatypeClass::Enumeration { base, members } => {
                write!(f, "enum {base} {{ ")?;
                for (i, m) in members.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    match int_value_opt(base, &m.value).and_then(|v| v.to_i128()) {
                        Some(v) => write!(f, "{} = {v}", m.name)?,
                        None => write!(f, "{} = ?", m.name)?,
                    }
                }
                f.write_str(" }")
            }
            DatatypeClass::VariableLength { kind, base } => match kind {
                VariableLengthKind::String {
                    padding: p,
                    charset: c,
                } => {
                    write!(f, "vlen string {}", charset(*c))?;
                    if *p != StringPadding::NullTerminate {
                        write!(f, " {}", padding(*p))?;
                    }
                    Ok(())
                }
                VariableLengthKind::Sequence => write!(f, "vlen sequence of {base}"),
            },
            DatatypeClass::Array { dims, base, .. } => {
                f.write_str("array")?;
                write_dims(f, dims)?;
                write!(f, " of {base}")
            }
            DatatypeClass::Complex { form, base, .. } => {
                f.write_str("complex")?;
                match form {
                    ComplexForm::Rectangular => {}
                    ComplexForm::Polar => f.write_str(" (polar)")?,
                    ComplexForm::Exponential => f.write_str(" (exponential)")?,
                }
                write!(f, " of {base}")
            }
        }
    }
}

fn write_dims(f: &mut fmt::Formatter<'_>, dims: &[u32]) -> fmt::Result {
    f.write_str("[")?;
    for (i, d) in dims.iter().enumerate() {
        if i > 0 {
            f.write_str("x")?;
        }
        write!(f, "{d}")?;
    }
    f.write_str("]")
}
