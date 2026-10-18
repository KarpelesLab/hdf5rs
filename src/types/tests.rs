use super::*;
use super::{Complex, Reference};
use crate::error::Error;
use crate::format::Sizes;
use crate::format::global_heap::GlobalHeapId;
use crate::format::messages::datatype::*;

// ------------------------------------------------------------ builders

const LE: ByteOrder = ByteOrder::LittleEndian;
const BE: ByteOrder = ByteOrder::BigEndian;

fn intp(size: u32, signed: bool, order: ByteOrder, off: u16, prec: u16) -> Datatype {
    Datatype {
        version: 1,
        size,
        class: DatatypeClass::FixedPoint(FixedPoint {
            byte_order: order,
            lo_pad: false,
            hi_pad: false,
            signed,
            bit_offset: off,
            bit_precision: prec,
        }),
    }
}

fn int(size: u32, signed: bool, order: ByteOrder) -> Datatype {
    intp(size, signed, order, 0, (size * 8) as u16)
}

fn float(size: u32, order: ByteOrder) -> Datatype {
    let (eloc, esize, msize, bias) = match size {
        2 => (10, 5, 10, 15),
        4 => (23, 8, 23, 127),
        _ => (52, 11, 52, 1023),
    };
    Datatype {
        version: 1,
        size,
        class: DatatypeClass::FloatingPoint(FloatingPoint {
            byte_order: order,
            lo_pad: false,
            hi_pad: false,
            internal_pad: false,
            normalization: MantissaNormalization::Implied,
            sign_location: (size * 8 - 1) as u8,
            bit_offset: 0,
            bit_precision: (size * 8) as u16,
            exponent_location: eloc,
            exponent_size: esize,
            mantissa_location: 0,
            mantissa_size: msize,
            exponent_bias: bias,
        }),
    }
}

/// x87 80-bit extended precision padded to 16 bytes (numpy longdouble).
fn x87() -> Datatype {
    Datatype {
        version: 1,
        size: 16,
        class: DatatypeClass::FloatingPoint(FloatingPoint {
            byte_order: LE,
            lo_pad: false,
            hi_pad: false,
            internal_pad: false,
            normalization: MantissaNormalization::MsbSet,
            sign_location: 79,
            bit_offset: 0,
            bit_precision: 80,
            exponent_location: 64,
            exponent_size: 15,
            mantissa_location: 0,
            mantissa_size: 64,
            exponent_bias: 16383,
        }),
    }
}

fn fstr(size: u32, padding: StringPadding, charset: CharacterSet) -> Datatype {
    Datatype {
        version: 1,
        size,
        class: DatatypeClass::String(StringType { padding, charset }),
    }
}

fn vstr(charset: CharacterSet) -> Datatype {
    Datatype {
        version: 1,
        size: 16,
        class: DatatypeClass::VariableLength {
            kind: VariableLengthKind::String {
                padding: StringPadding::NullTerminate,
                charset,
            },
            base: Box::new(int(1, false, LE)),
        },
    }
}

fn vseq(base: Datatype) -> Datatype {
    Datatype {
        version: 1,
        size: 16,
        class: DatatypeClass::VariableLength {
            kind: VariableLengthKind::Sequence,
            base: Box::new(base),
        },
    }
}

fn array(dims: &[u32], base: Datatype) -> Datatype {
    let n: u32 = dims.iter().product();
    Datatype {
        version: 3,
        size: n * base.size,
        class: DatatypeClass::Array {
            dims: dims.to_vec(),
            permutation: None,
            base: Box::new(base),
        },
    }
}

fn compound(size: u32, members: Vec<(&str, u32, Datatype)>) -> Datatype {
    Datatype {
        version: 3,
        size,
        class: DatatypeClass::Compound {
            members: members
                .into_iter()
                .map(|(name, offset, datatype)| CompoundMember {
                    name: name.to_string(),
                    offset,
                    array_dims: None,
                    datatype,
                })
                .collect(),
        },
    }
}

fn enumeration(base: Datatype, members: &[(&str, i64)]) -> Datatype {
    let n = base.size as usize;
    let be = base.byte_order() == Some(BE);
    Datatype {
        version: 3,
        size: base.size,
        class: DatatypeClass::Enumeration {
            members: members
                .iter()
                .map(|(name, v)| {
                    let mut b = v.to_le_bytes()[..n].to_vec();
                    if be {
                        b.reverse();
                    }
                    EnumMember {
                        name: name.to_string(),
                        value: b,
                    }
                })
                .collect(),
            base: Box::new(base),
        },
    }
}

fn h5py_bool() -> Datatype {
    enumeration(int(1, true, LE), &[("FALSE", 0), ("TRUE", 1)])
}

fn reference(kind: ReferenceKind, size: u32) -> Datatype {
    Datatype {
        version: if matches!(kind, ReferenceKind::Object | ReferenceKind::DatasetRegion) {
            1
        } else {
            4
        },
        size,
        class: DatatypeClass::Reference(crate::format::messages::datatype::Reference {
            kind,
            encoding_version: 0,
        }),
    }
}

fn ctx() -> DecodeContext<'static> {
    DecodeContext::default()
}

/// Build a global heap collection (8-byte lengths) at `addr` of a buffer.
fn heap_file(addr: u64, objects: &[(u16, &[u8])]) -> Vec<u8> {
    let mut c = b"GCOL".to_vec();
    c.extend([1, 0, 0, 0]);
    c.extend(0u64.to_le_bytes()); // size, patched below
    for (i, d) in objects {
        c.extend(i.to_le_bytes());
        c.extend(1u16.to_le_bytes());
        c.extend([0; 4]);
        c.extend((d.len() as u64).to_le_bytes());
        c.extend(*d);
        c.resize(c.len().div_ceil(8) * 8, 0);
    }
    let size = c.len().max(4096 - 64) as u64;
    c.resize(size as usize, 0);
    c[8..16].copy_from_slice(&size.to_le_bytes());
    let mut file = vec![0u8; addr as usize];
    file.extend(c);
    file
}

fn vlen_el(len: u32, addr: u64, index: u32) -> Vec<u8> {
    let mut v = len.to_le_bytes().to_vec();
    v.extend(addr.to_le_bytes());
    v.extend(index.to_le_bytes());
    v
}

fn heap_id_bytes(addr: u64, index: u32) -> Vec<u8> {
    let mut v = addr.to_le_bytes().to_vec();
    v.extend(index.to_le_bytes());
    v
}

// --------------------------------------------------------------- integers

#[test]
fn integers_exact_both_orders() {
    let vals: [i32; 4] = [0, 1, -2, i32::MAX];
    let le: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let be: Vec<u8> = vals.iter().flat_map(|v| v.to_be_bytes()).collect();
    assert_eq!(
        decode_all::<i32>(&int(4, true, LE), &le, &ctx()).unwrap(),
        vals
    );
    assert_eq!(
        decode_all::<i32>(&int(4, true, BE), &be, &ctx()).unwrap(),
        vals
    );
    assert_eq!(
        i32::decode(&int(4, true, BE), &be[8..12], &ctx()).unwrap(),
        -2
    );
    let u: Vec<u8> = [1u64, u64::MAX]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    assert_eq!(
        decode_all::<u64>(&int(8, false, BE), &u, &ctx()).unwrap(),
        [1, u64::MAX]
    );
    assert_eq!(
        decode_all::<u8>(&int(1, false, LE), &[0, 255], &ctx()).unwrap(),
        [0, 255]
    );
    assert_eq!(
        decode_all::<i8>(&int(1, true, BE), &[0x80], &ctx()).unwrap(),
        [-128]
    );
    assert_eq!(
        decode_all::<i16>(&int(2, true, BE), &[0xff, 0xfe], &ctx()).unwrap(),
        [-2]
    );
    assert_eq!(
        decode_all::<u16>(&int(2, false, LE), &[0x34, 0x12], &ctx()).unwrap(),
        [0x1234]
    );
    assert_eq!(
        decode_all::<u32>(&int(4, false, LE), &[1, 0, 0, 0x80], &ctx()).unwrap(),
        [0x8000_0001]
    );
    assert_eq!(
        decode_all::<i64>(&int(8, true, LE), &(-5i64).to_le_bytes(), &ctx()).unwrap(),
        [-5]
    );
}

#[test]
fn integers_widening_and_rejections() {
    // i8 BE -> i64, u16 -> u32 / i32, u32 -> i64.
    assert_eq!(
        decode_all::<i64>(&int(1, true, BE), &[0xfe, 0x05], &ctx()).unwrap(),
        [-2, 5]
    );
    assert_eq!(
        decode_all::<u32>(&int(2, false, BE), &[0xff, 0xff], &ctx()).unwrap(),
        [65535]
    );
    assert_eq!(
        decode_all::<i32>(&int(2, false, LE), &[0xff, 0xff], &ctx()).unwrap(),
        [65535]
    );
    assert_eq!(
        decode_all::<i64>(&int(4, false, LE), &[0xff; 4], &ctx()).unwrap(),
        [u32::MAX as i64]
    );
    // Lossy or sign-changing conversions are rejected.
    assert!(u8::check(&int(1, true, LE)).is_err());
    assert!(i8::check(&int(1, false, LE)).is_err());
    assert!(i32::check(&int(8, true, LE)).is_err());
    assert!(u16::check(&int(4, false, LE)).is_err());
    assert!(u64::check(&int(8, true, LE)).is_err());
    assert!(i32::check(&float(4, LE)).is_err());
    assert!(i32::check(&fstr(4, StringPadding::NullPad, CharacterSet::Ascii)).is_err());
    // Errors, not panics, for type mismatches on decode.
    assert!(i32::decode(&float(4, LE), &[0; 4], &ctx()).is_err());
    assert!(decode_all::<u8>(&int(1, true, LE), &[1], &ctx()).is_err());
}

#[test]
fn odd_sized_and_bit_offset_integers() {
    // 3-byte big-endian signed.
    let dt = int(3, true, BE);
    assert_eq!(
        decode_all::<i32>(&dt, &[0xff, 0xff, 0xfe, 0x01, 0x00, 0x00], &ctx()).unwrap(),
        [-2, 0x10000]
    );
    assert!(i16::check(&dt).is_err());
    assert_eq!(
        decode_value(&dt, &[0x80, 0, 0], &ctx()).unwrap(),
        Value::Int(-(1 << 23))
    );
    // 12-bit unsigned at bit offset 4 in a 16-bit LE word (N-bit style).
    let dt = intp(2, false, LE, 4, 12);
    let raw = [0xf5u8, 0xab, 0x0f, 0x00];
    assert_eq!(
        decode_all::<u16>(&dt, &raw, &ctx()).unwrap(),
        [0xabf, 0x000]
    );
    // A 12-bit unsigned value fits i16 and even u16.
    assert!(i16::check(&dt).is_ok());
    assert_eq!(
        decode_value(&dt, &raw[..2], &ctx()).unwrap(),
        Value::UInt(0xabf)
    );
    // 12-bit signed at offset 4: 0xabf -> negative.
    let dt = intp(2, true, LE, 4, 12);
    assert_eq!(
        decode_all::<i16>(&dt, &raw[..2], &ctx()).unwrap(),
        [0xabf - 0x1000]
    );
    // 20-bit at offset 4 in a 4-byte BE word.
    let dt = intp(4, false, BE, 4, 20);
    let w: u32 = 0xdead_beef;
    assert_eq!(
        decode_all::<u32>(&dt, &w.to_be_bytes(), &ctx()).unwrap(),
        [(w >> 4) & 0xfffff]
    );
    // Precision beyond the size is invalid.
    assert!(i32::check(&intp(2, true, LE, 4, 16)).is_err());
    // 16-byte integers through Value.
    let dt = int(16, true, LE);
    assert_eq!(
        decode_value(&dt, &(-3i128).to_le_bytes(), &ctx()).unwrap(),
        Value::Int(-3)
    );
    assert!(decode_value(&dt, &i128::MAX.to_le_bytes(), &ctx()).is_err());
    assert!(i64::check(&dt).is_err());
}

#[test]
fn bulk_matches_per_element() {
    let n = 1000;
    let src: Vec<u8> = (0..n * 8).map(|i| (i * 37 % 251) as u8).collect();
    for order in [LE, BE] {
        for (size, signed) in [(1, true), (2, false), (4, true), (8, false)] {
            let dt = int(size, signed, order);
            let all = decode_all::<i64>(&dt, &src[..n * size as usize], &ctx());
            let all = match all {
                Ok(v) => v,
                Err(_) => {
                    assert!(!signed && size == 8);
                    continue;
                }
            };
            for (i, chunk) in src[..n * size as usize]
                .chunks_exact(size as usize)
                .enumerate()
            {
                let v = decode_value(&dt, chunk, &ctx()).unwrap();
                assert_eq!(v.as_i64(), Some(all[i]));
            }
        }
        let dt = float(8, order);
        let all = decode_all::<f64>(&dt, &src, &ctx()).unwrap();
        for (i, chunk) in src.as_chunks::<8>().0.iter().enumerate() {
            let v = decode_value(&dt, chunk, &ctx()).unwrap().as_f64().unwrap();
            assert!(v.to_bits() == all[i].to_bits() || (v.is_nan() && all[i].is_nan()));
        }
    }
}

// ----------------------------------------------------------------- floats

#[test]
fn floats_all_orders_and_widths() {
    let v = [1.5f64, -0.0, f64::INFINITY, 1e-310, f64::MAX];
    let le: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let be: Vec<u8> = v.iter().flat_map(|x| x.to_be_bytes()).collect();
    assert_eq!(decode_all::<f64>(&float(8, LE), &le, &ctx()).unwrap(), v);
    assert_eq!(decode_all::<f64>(&float(8, BE), &be, &ctx()).unwrap(), v);
    assert!(f32::check(&float(8, LE)).is_err());

    let v = [2.5f32, -1.0, f32::NEG_INFINITY, f32::MIN_POSITIVE];
    let be: Vec<u8> = v.iter().flat_map(|x| x.to_be_bytes()).collect();
    assert_eq!(decode_all::<f32>(&float(4, BE), &be, &ctx()).unwrap(), v);
    let wide: Vec<f64> = v.iter().map(|&x| f64::from(x)).collect();
    assert_eq!(decode_all::<f64>(&float(4, BE), &be, &ctx()).unwrap(), wide);
    let nan = f32::NAN.to_le_bytes();
    assert!(f32::decode(&float(4, LE), &nan, &ctx()).unwrap().is_nan());

    // f16, both orders, to f32 and f64 (values from h5py output).
    let h = [0x00, 0x3e, 0x00, 0xc0, 0xff, 0x7b];
    assert_eq!(
        decode_all::<f32>(&float(2, LE), &h, &ctx()).unwrap(),
        [1.5, -2.0, 65504.0]
    );
    let hb = [0x3e, 0x00, 0xc0, 0x00, 0x7b, 0xff];
    assert_eq!(
        decode_all::<f64>(&float(2, BE), &hb, &ctx()).unwrap(),
        [1.5, -2.0, 65504.0]
    );
    assert_eq!(
        decode_value(&float(2, BE), &hb[..2], &ctx()).unwrap(),
        Value::Float(1.5)
    );
    // Single element decode.
    assert_eq!(f64::decode(&float(2, LE), &h[2..4], &ctx()).unwrap(), -2.0);

    // Non-IEEE layouts: x87 extended via Value; not losslessly into f64.
    let one = ((16383u128 << 64) | (1u128 << 63)).to_le_bytes();
    assert_eq!(
        decode_value(&x87(), &one, &ctx()).unwrap(),
        Value::Float(1.0)
    );
    assert!(f64::check(&x87()).is_err());

    // A custom 3-byte float (bfloat16-like layout + padding) works via the
    // generic path: 1 sign, 8 exponent, 7 mantissa bits in the low 16 bits.
    let bf = Datatype {
        version: 1,
        size: 3,
        class: DatatypeClass::FloatingPoint(FloatingPoint {
            byte_order: LE,
            lo_pad: false,
            hi_pad: false,
            internal_pad: false,
            normalization: MantissaNormalization::Implied,
            sign_location: 15,
            bit_offset: 0,
            bit_precision: 16,
            exponent_location: 7,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 7,
            exponent_bias: 127,
        }),
    };
    let x = (-3.25f32).to_bits() >> 16;
    let b = x.to_le_bytes();
    assert_eq!(decode_all::<f32>(&bf, &b[..3], &ctx()).unwrap(), [-3.25]);
    // VAX order is unsupported, not a panic.
    let mut vax = float(4, LE);
    if let DatatypeClass::FloatingPoint(fp) = &mut vax.class {
        fp.byte_order = ByteOrder::Vax;
    }
    assert!(decode_all::<f32>(&vax, &[0; 4], &ctx()).is_err());
    assert!(decode_value(&vax, &[0; 4], &ctx()).is_err());
}

#[test]
fn native_detection() {
    let native = ByteOrder::native();
    let other = if native == LE { BE } else { LE };
    assert!(is_native::<i32>(&int(4, true, native)));
    assert!(!is_native::<i32>(&int(4, true, other)));
    assert!(!is_native::<u32>(&int(4, true, native)));
    assert!(!is_native::<i64>(&int(4, true, native)));
    assert!(is_native::<f64>(&float(8, native)));
    assert!(!is_native::<f32>(&float(8, native)));
    assert!(is_native::<[f32; 3]>(&array(&[3], float(4, native))));
    assert!(!is_native::<String>(&fstr(
        4,
        StringPadding::NullPad,
        CharacterSet::Ascii
    )));
    assert!(!is_native::<i8>(&h5py_bool()));
}

// ------------------------------------------------------------------- bool

#[test]
fn booleans() {
    let dt = h5py_bool();
    assert_eq!(
        decode_all::<bool>(&dt, &[1, 0, 1], &ctx()).unwrap(),
        [true, false, true]
    );
    assert_eq!(decode_value(&dt, &[1], &ctx()).unwrap(), Value::Bool(true));
    // A value outside the bool members is reported as an enum.
    assert_eq!(
        decode_value(&dt, &[7], &ctx()).unwrap(),
        Value::Enum {
            name: None,
            value: 7
        }
    );
    // Integers read as bool too; enum reads as its base integer.
    assert_eq!(
        decode_all::<bool>(&int(2, false, BE), &[0, 0, 0, 2], &ctx()).unwrap(),
        [false, true]
    );
    assert_eq!(decode_all::<i8>(&dt, &[1, 0], &ctx()).unwrap(), [1, 0]);
    assert!(bool::check(&float(4, LE)).is_err());
    assert!(!bool::decode(&dt, &[0], &ctx()).unwrap());
}

// ---------------------------------------------------------------- strings

#[test]
fn fixed_strings() {
    use CharacterSet::*;
    use StringPadding::*;
    // Bytes as stored by h5py for ["alpha", "be", "", "gammadel"].
    let nul = b"alpha\0\0\0be\0\0\0\0\0\0\0\0\0\0\0\0\0\0gammadel";
    let spc = b"alpha   be              gammadel";
    let want = ["alpha", "be", "", "gammadel"];
    for (dt, raw) in [
        (fstr(8, NullTerminate, Ascii), &nul[..]),
        (fstr(8, NullPad, Ascii), &nul[..]),
        (fstr(8, SpacePad, Utf8), &spc[..]),
    ] {
        assert_eq!(decode_all::<String>(&dt, raw, &ctx()).unwrap(), want);
        assert_eq!(
            decode_value(&dt, &raw[..8], &ctx()).unwrap(),
            Value::String("alpha".into())
        );
    }
    // Null-terminated stops at the first NUL; null-padded keeps embedded
    // NULs and strips only trailing ones.
    let raw = b"ab\0cd\0\0\0";
    assert_eq!(
        String::decode(&fstr(8, NullTerminate, Ascii), raw, &ctx()).unwrap(),
        "ab"
    );
    assert_eq!(
        String::decode(&fstr(8, NullPad, Ascii), raw, &ctx()).unwrap(),
        "ab\0cd"
    );
    // Multibyte UTF-8.
    let raw = "日本\0\0\0\0".as_bytes();
    assert_eq!(
        String::decode(&fstr(10, NullPad, Utf8), raw, &ctx()).unwrap(),
        "日本"
    );
    // Invalid UTF-8: error for String, raw bytes otherwise.
    let bad = [b'a', 0xff, 0, 0];
    let dt = fstr(4, NullPad, Ascii);
    assert!(String::decode(&dt, &bad, &ctx()).is_err());
    assert_eq!(
        StringBytes::decode(&dt, &bad, &ctx()).unwrap(),
        StringBytes(vec![b'a', 0xff])
    );
    assert_eq!(
        StringBytes::decode(&dt, &bad, &ctx())
            .unwrap()
            .to_string_lossy(),
        "a\u{fffd}"
    );
    assert_eq!(
        decode_value(&dt, &bad, &ctx()).unwrap(),
        Value::Bytes(vec![b'a', 0xff])
    );
    // Wrong size / wrong type.
    assert!(String::decode(&dt, &bad[..3], &ctx()).is_err());
    assert!(String::check(&int(4, true, LE)).is_err());
    assert!(decode_all::<String>(&dt, &[0; 6], &ctx()).is_err());
}

// --------------------------------------------------------- variable-length

const HEAP: u64 = 0x833;

fn vlen_heap() -> Vec<u8> {
    heap_file(
        HEAP,
        &[
            (1, "héllo wörld".as_bytes()),
            (2, b""),
            (3, b"hello"),
            (4, b"xyz"),
            (6, &[1, 0, 2, 0, 3, 0]),
            (7, &[0xf9, 0xff]),
            (8, b"ab\0\0"),
        ],
    )
}

#[test]
fn vlen_strings() {
    let file = vlen_heap();
    let res = GlobalHeapResolver::new(&file[..], Sizes::DEFAULT);
    let ctx = DecodeContext::with_resolver(Sizes::DEFAULT, &res);
    let dt = vstr(CharacterSet::Utf8);
    // Element layout as written by h5py: u32 byte count + heap ID.
    let mut raw = vlen_el(5, HEAP, 3);
    raw.extend(vlen_el(0, HEAP, 2));
    raw.extend(vlen_el(13, HEAP, 1));
    raw.extend(vlen_el(0, 0, 0));
    raw.extend(vlen_el(4, HEAP, 8));
    assert_eq!(
        decode_all::<String>(&dt, &raw, &ctx).unwrap(),
        ["hello", "", "héllo wörld", "", "ab"]
    );
    assert_eq!(
        decode_value(&dt, &raw[..16], &ctx).unwrap(),
        Value::String("hello".into())
    );
    // Count shorter than the object: only `count` bytes are used.
    assert_eq!(
        String::decode(&dt, &vlen_el(2, HEAP, 3), &ctx).unwrap(),
        "he"
    );
    assert_eq!(
        StringBytes::decode(&vstr(CharacterSet::Ascii), &vlen_el(3, HEAP, 4), &ctx).unwrap(),
        StringBytes(b"xyz".to_vec())
    );
    // Errors: count beyond the object, null ID with a count, missing
    // object, missing resolver, wrong element size.
    assert!(String::decode(&dt, &vlen_el(6, HEAP, 3), &ctx).is_err());
    assert!(String::decode(&dt, &vlen_el(1, 0, 0), &ctx).is_err());
    assert!(String::decode(&dt, &vlen_el(1, HEAP, 5), &ctx).is_err());
    assert!(String::decode(&dt, &vlen_el(1, 0x10, 1), &ctx).is_err());
    assert!(matches!(
        String::decode(&dt, &vlen_el(5, HEAP, 3), &DecodeContext::default()),
        Err(Error::Unsupported(_))
    ));
    // Empty values need no resolver.
    assert_eq!(
        String::decode(&dt, &vlen_el(0, 0, 0), &DecodeContext::default()).unwrap(),
        ""
    );
    assert!(String::decode(&dt, &raw[..15], &ctx).is_err());
}

#[test]
fn vlen_sequences() {
    let file = vlen_heap();
    let res = GlobalHeapResolver::new(&file[..], Sizes::DEFAULT);
    let ctx = DecodeContext::with_resolver(Sizes::DEFAULT, &res);
    let dt = vseq(int(2, true, LE));
    // h5py: [1, 2, 3], [], [-7] (empty uses a null heap ID).
    let mut raw = vlen_el(3, HEAP, 6);
    raw.extend(vlen_el(0, 0, 0));
    raw.extend(vlen_el(1, HEAP, 7));
    let got = decode_all::<Vec<i16>>(&dt, &raw, &ctx).unwrap();
    assert_eq!(got, vec![vec![1, 2, 3], vec![], vec![-7]]);
    let got = decode_all::<Vec<i64>>(&dt, &raw, &ctx).unwrap();
    assert_eq!(got[0], [1, 2, 3]);
    assert_eq!(
        decode_value(&dt, &raw[..16], &ctx).unwrap(),
        Value::VarLen(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
    // Count beyond the object (3 x i16 = 6 bytes available).
    assert!(Vec::<i16>::decode(&dt, &vlen_el(4, HEAP, 6), &ctx).is_err());
    // Wrong element type.
    assert!(Vec::<f32>::check(&dt).is_err());
    assert!(Vec::<i16>::check(&vstr(CharacterSet::Ascii)).is_err());
    // 4-byte offsets: element is 12 bytes.
    let sizes = Sizes {
        offset: 4,
        length: 8,
    };
    let dt4 = Datatype {
        size: 12,
        ..dt.clone()
    };
    let mut el = 3u32.to_le_bytes().to_vec();
    el.extend((HEAP as u32).to_le_bytes());
    el.extend(6u32.to_le_bytes());
    let res4 = GlobalHeapResolver::new(&file[..], Sizes::DEFAULT);
    let ctx4 = DecodeContext::with_resolver(sizes, &res4);
    assert_eq!(Vec::<i16>::decode(&dt4, &el, &ctx4).unwrap(), [1, 2, 3]);
    // The same element with the wrong offset size is an error.
    assert!(Vec::<i16>::decode(&dt4, &el, &ctx).is_err());
}

#[test]
fn resolver_cache_and_errors() {
    let file = vlen_heap();
    let res = GlobalHeapResolver::new(&file[..], Sizes::DEFAULT);
    let id = GlobalHeapId {
        collection_address: HEAP,
        index: 3,
    };
    assert_eq!(res.global_heap_object(&id).unwrap(), b"hello");
    let a = res.collection(HEAP).unwrap();
    let b = res.collection(HEAP).unwrap();
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    assert!(
        res.global_heap_object(&GlobalHeapId {
            collection_address: 0,
            index: 1
        })
        .is_err()
    );
    assert!(
        res.global_heap_object(&GlobalHeapId {
            collection_address: 8,
            index: 1
        })
        .is_err()
    );
    // Through a trait object as well.
    let dynres: &dyn VlenResolver = &res;
    assert_eq!((&dynres).global_heap_object(&id).unwrap(), b"hello");
    assert_eq!(res.sizes(), Sizes::DEFAULT);
    let el = VlenElement::parse(&vlen_el(5, HEAP, 3), Sizes::DEFAULT).unwrap();
    assert_eq!(el.length, 5);
    assert_eq!(el.heap_id, id);
    assert!(VlenElement::parse(&[0; 15], Sizes::DEFAULT).is_err());
}

// --------------------------------------------- compounds, arrays, enums

fn nested_compound() -> Datatype {
    // { a: i32, b: { x: f64 BE, y: [u16; 2x2] }, s: string[4] }
    let inner = compound(
        16,
        vec![
            ("x", 0, float(8, BE)),
            ("y", 8, array(&[2, 2], int(2, false, LE))),
        ],
    );
    compound(
        28,
        vec![
            ("a", 0, int(4, true, LE)),
            ("b", 4, inner),
            (
                "s",
                20,
                fstr(8, StringPadding::NullTerminate, CharacterSet::Ascii),
            ),
        ],
    )
}

fn nested_bytes() -> Vec<u8> {
    let mut v = (-7i32).to_le_bytes().to_vec();
    v.extend(2.5f64.to_be_bytes());
    for x in [1u16, 2, 3, 4] {
        v.extend(x.to_le_bytes());
    }
    v.extend(b"hi\0\0\0\0\0\0");
    v
}

#[test]
fn nested_compound_value() {
    let dt = nested_compound();
    let v = decode_value(&dt, &nested_bytes(), &ctx()).unwrap();
    let want = Value::Compound(vec![
        ("a".into(), Value::Int(-7)),
        (
            "b".into(),
            Value::Compound(vec![
                ("x".into(), Value::Float(2.5)),
                (
                    "y".into(),
                    Value::Array(vec![
                        Value::Array(vec![Value::UInt(1), Value::UInt(2)]),
                        Value::Array(vec![Value::UInt(3), Value::UInt(4)]),
                    ]),
                ),
            ]),
        ),
        ("s".into(), Value::String("hi".into())),
    ]);
    assert_eq!(v, want);
    assert_eq!(v.field("a"), Some(&Value::Int(-7)));
    assert_eq!(
        v.to_string(),
        "{a: -7, b: {x: 2.5, y: [[1, 2], [3, 4]]}, s: \"hi\"}"
    );
    // Member exceeding the compound size is an error.
    let mut bad = dt.clone();
    if let DatatypeClass::Compound { members } = &mut bad.class {
        members[2].offset = 24;
    }
    assert!(decode_value(&bad, &nested_bytes(), &ctx()).is_err());
}

#[derive(Debug, PartialEq)]
struct Rec {
    a: i64,
    x: f64,
    y: [u16; 4],
    s: String,
}

impl H5Type for Rec {
    fn check(dt: &Datatype) -> crate::Result<()> {
        check_member::<i64>(dt, "a")?;
        check_member::<String>(dt, "s")
    }

    fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> crate::Result<Self> {
        let (bt, off) = compound_member(dt, "b")?;
        let b = &bytes[off..off + bt.size as usize];
        Ok(Rec {
            a: decode_member(dt, bytes, ctx, "a")?,
            x: decode_member(&bt, b, ctx, "x")?,
            y: decode_member(&bt, b, ctx, "y")?,
            s: decode_member(dt, bytes, ctx, "s")?,
        })
    }
}

#[test]
fn user_struct_via_members() {
    let dt = nested_compound();
    let mut raw = nested_bytes();
    raw.extend(nested_bytes());
    let recs = decode_all::<Rec>(&dt, &raw, &ctx()).unwrap();
    assert_eq!(recs.len(), 2);
    assert_eq!(
        recs[1],
        Rec {
            a: -7,
            x: 2.5,
            y: [1, 2, 3, 4],
            s: "hi".into()
        }
    );
    assert!(matches!(
        compound_member(&dt, "nope"),
        Err(Error::NotFound(_))
    ));
    assert!(check_member::<f32>(&dt, "a").is_err());
    assert!(compound_member(&int(4, true, LE), "a").is_err());
}

#[test]
fn v1_compound_array_member() {
    let dt = Datatype {
        version: 1,
        size: 8,
        class: DatatypeClass::Compound {
            members: vec![CompoundMember {
                name: "m".into(),
                offset: 0,
                array_dims: Some(vec![2, 2]),
                datatype: int(2, true, BE),
            }],
        },
    };
    let raw = [0, 1, 0, 2, 0, 3, 0xff, 0xff];
    assert_eq!(
        decode_member::<[i16; 4]>(&dt, &raw, &ctx(), "m").unwrap(),
        [1, 2, 3, -1]
    );
    assert_eq!(
        decode_value(&dt, &raw, &ctx()).unwrap().field("m"),
        Some(&Value::Array(vec![
            Value::Array(vec![Value::Int(1), Value::Int(2)]),
            Value::Array(vec![Value::Int(3), Value::Int(-1)]),
        ]))
    );
    assert_eq!(dt.to_string(), "compound[8] { m: [2x2] of i16 BE @ 0 }");
}

#[test]
fn arrays() {
    let dt = array(&[2, 3], int(4, true, LE));
    let raw: Vec<u8> = (0..6i32).flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        <[i32; 6]>::decode(&dt, &raw, &ctx()).unwrap(),
        [0, 1, 2, 3, 4, 5]
    );
    assert_eq!(
        Vec::<i64>::decode(&dt, &raw, &ctx()).unwrap(),
        [0, 1, 2, 3, 4, 5]
    );
    assert!(<[i32; 5]>::check(&dt).is_err());
    assert!(<[i32; 6]>::check(&int(4, true, LE)).is_err());
    let v = decode_value(&dt, &raw, &ctx()).unwrap();
    assert_eq!(v.to_string(), "[[0, 1, 2], [3, 4, 5]]");
    assert_eq!(v.as_slice().map(<[Value]>::len), Some(2));
    // Inconsistent array size.
    let mut bad = dt.clone();
    bad.size = 20;
    assert!(<[i32; 6]>::check(&bad).is_err());
    assert!(decode_value(&bad, &raw[..20], &ctx()).is_err());
    // Zero-sized base with huge dims does not allocate.
    let zero = Datatype {
        version: 3,
        size: 0,
        class: DatatypeClass::Array {
            dims: vec![u32::MAX, u32::MAX],
            permutation: None,
            base: Box::new(Datatype {
                size: 0,
                ..int(1, false, LE)
            }),
        },
    };
    assert!(decode_value(&zero, &[], &ctx()).is_err());
    assert!(Vec::<u8>::decode(&zero, &[], &ctx()).is_err());
}

#[test]
fn enums() {
    let dt = enumeration(int(1, true, LE), &[("RED", 0), ("GREEN", 1), ("BLUE", 42)]);
    assert_eq!(
        decode_value(&dt, &[42], &ctx()).unwrap(),
        Value::Enum {
            name: Some("BLUE".into()),
            value: 42
        }
    );
    assert_eq!(
        decode_value(&dt, &[3], &ctx()).unwrap(),
        Value::Enum {
            name: None,
            value: 3
        }
    );
    let dt = enumeration(int(2, true, BE), &[("HIGH", 1000), ("LOW", -100)]);
    let raw = [0x03, 0xe8, 0xff, 0x9c];
    assert_eq!(decode_all::<i16>(&dt, &raw, &ctx()).unwrap(), [1000, -100]);
    let v = decode_value(&dt, &raw[2..], &ctx()).unwrap();
    assert_eq!(v.to_string(), "LOW");
    assert_eq!(v.as_i64(), Some(-100));
    assert_eq!(dt.to_string(), "enum i16 BE { HIGH = 1000, LOW = -100 }");
}

#[test]
fn complex_numbers() {
    let cls = Datatype {
        version: 5,
        size: 16,
        class: DatatypeClass::Complex {
            homogeneous: true,
            form: ComplexForm::Rectangular,
            base: Box::new(float(8, LE)),
        },
    };
    let mut raw = 1.5f64.to_le_bytes().to_vec();
    raw.extend((-2.0f64).to_le_bytes());
    assert_eq!(
        Complex::<f64>::decode(&cls, &raw, &ctx()).unwrap(),
        Complex { re: 1.5, im: -2.0 }
    );
    assert_eq!(
        decode_value(&cls, &raw, &ctx()).unwrap(),
        Value::Complex { re: 1.5, im: -2.0 }
    );
    assert_eq!(
        decode_value(&cls, &raw, &ctx()).unwrap().to_string(),
        "1.5-2.0i"
    );
    // h5py's compound {r, i} of f32.
    let cmp = compound(8, vec![("r", 0, float(4, LE)), ("i", 4, float(4, LE))]);
    let mut raw = 0.5f32.to_le_bytes().to_vec();
    raw.extend(4.0f32.to_le_bytes());
    assert_eq!(
        decode_all::<Complex<f64>>(&cmp, &raw, &ctx()).unwrap(),
        [Complex { re: 0.5, im: 4.0 }]
    );
    assert!(Complex::<f32>::check(&nested_compound()).is_err());
    let polar = Datatype {
        class: DatatypeClass::Complex {
            homogeneous: true,
            form: ComplexForm::Polar,
            base: Box::new(float(8, LE)),
        },
        ..cls.clone()
    };
    assert!(Complex::<f64>::check(&polar).is_err());
    assert!(decode_value(&polar, &[0; 16], &ctx()).is_err());
}

#[test]
fn bitfield_opaque_time() {
    let bf = Datatype {
        version: 1,
        size: 2,
        class: DatatypeClass::Bitfield(Bitfield {
            byte_order: BE,
            lo_pad: false,
            hi_pad: false,
            bit_offset: 0,
            bit_precision: 16,
        }),
    };
    assert_eq!(
        decode_value(&bf, &[0x12, 0x34], &ctx()).unwrap(),
        Value::Bitfield(0x1234)
    );
    assert_eq!(bf.to_string(), "bitfield16 BE");
    let op = Datatype {
        version: 1,
        size: 3,
        class: DatatypeClass::Opaque { tag: "tag".into() },
    };
    assert_eq!(
        decode_value(&op, &[1, 2, 3], &ctx()).unwrap(),
        Value::Opaque(vec![1, 2, 3])
    );
    assert_eq!(op.to_string(), "opaque[3] \"tag\"");
    let tm = Datatype {
        version: 1,
        size: 4,
        class: DatatypeClass::Time(Time {
            byte_order: BE,
            bit_precision: 32,
        }),
    };
    assert_eq!(
        decode_value(&tm, &[0xff, 0xff, 0xff, 0xfe], &ctx()).unwrap(),
        Value::Time(-2)
    );
}

// ------------------------------------------------------------- references

#[test]
fn object_references() {
    let dt = reference(ReferenceKind::Object, 8);
    let mut raw = 0x1a73u64.to_le_bytes().to_vec();
    raw.extend([0; 8]);
    raw.extend([0xff; 8]);
    assert_eq!(
        decode_all::<Reference>(&dt, &raw, &ctx()).unwrap(),
        [Reference::Object(0x1a73), Reference::Null, Reference::Null]
    );
    assert_eq!(Reference::Object(5).object_address(), Some(5));
    assert_eq!(Reference::Null.object_address(), None);
    // 4-byte offsets.
    let dt4 = reference(ReferenceKind::Object, 4);
    assert_eq!(
        Reference::decode(&dt4, &[0x10, 0, 0, 0], &ctx()).unwrap(),
        Reference::Object(0x10)
    );
    assert!(Reference::decode(&dt, &raw[..4], &ctx()).is_err());
    assert!(Reference::check(&int(8, false, LE)).is_err());
    assert_eq!(dt.to_string(), "object reference");
}

/// Region reference heap objects exactly as written by h5py/libhdf5.
fn region_heap() -> Vec<u8> {
    let hex = |s: &str| -> Vec<u8> {
        s.split_whitespace()
            .map(|h| u8::from_str_radix(h, 16).unwrap())
            .collect()
    };
    // data[1:3, 2:4] (hyperslab v1), data[0, 0], data[...] (all),
    // a 3-point selection, and a regular v3 hyperslab with 2-byte fields
    // and a v2 point selection (libver="latest").
    let objs = [
        hex(
            "73 1a 00 00 00 00 00 00 02 00 00 00 01 00 00 00 00 00 00 00 18 00 00 00 \
             02 00 00 00 01 00 00 00 01 00 00 00 02 00 00 00 02 00 00 00 03 00 00 00",
        ),
        hex("73 1a 00 00 00 00 00 00 03 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00"),
        hex(
            "73 1a 00 00 00 00 00 00 01 00 00 00 01 00 00 00 00 00 00 00 20 00 00 00 \
             02 00 00 00 03 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 00 00 00 00 \
             03 00 00 00 04 00 00 00",
        ),
        hex(
            "ef 05 00 00 00 00 00 00 02 00 00 00 03 00 00 00 01 02 02 00 00 00 00 00 \
             02 00 02 00 01 00 01 00 02 00 02 00 01 00",
        ),
        hex(
            "ef 05 00 00 00 00 00 00 01 00 00 00 02 00 00 00 02 02 00 00 00 03 00 00 \
             00 01 00 02 00 00 00 03 00 04 00",
        ),
    ];
    let refs: Vec<(u16, &[u8])> = objs
        .iter()
        .enumerate()
        .map(|(i, o)| (i as u16 + 1, o.as_slice()))
        .collect();
    heap_file(HEAP, &refs)
}

#[test]
fn region_references() {
    let file = region_heap();
    let res = GlobalHeapResolver::new(&file[..], Sizes::DEFAULT);
    let ctx = DecodeContext::with_resolver(Sizes::DEFAULT, &res);
    let dt = reference(ReferenceKind::DatasetRegion, 12);
    let mut raw = Vec::new();
    for i in 1..=5 {
        raw.extend(heap_id_bytes(HEAP, i));
    }
    raw.extend(heap_id_bytes(0, 0));
    let refs = decode_all::<Reference>(&dt, &raw, &ctx).unwrap();
    let region = |object: u64, selection: Selection| {
        Reference::Region(RegionReference { object, selection })
    };
    assert_eq!(
        refs,
        [
            region(
                0x1a73,
                Selection::Hyperslab {
                    rank: 2,
                    blocks: vec![HyperslabBlock {
                        start: vec![1, 2],
                        end: vec![2, 3]
                    }]
                }
            ),
            region(0x1a73, Selection::All),
            region(
                0x1a73,
                Selection::Points {
                    rank: 2,
                    points: vec![vec![0, 1], vec![2, 0], vec![3, 4]]
                }
            ),
            region(
                0x5ef,
                Selection::RegularHyperslab {
                    start: vec![0, 1],
                    stride: vec![2, 2],
                    count: vec![2, 2],
                    block: vec![1, 1]
                }
            ),
            region(
                0x5ef,
                Selection::Points {
                    rank: 2,
                    points: vec![vec![0, 1], vec![2, 0], vec![3, 4]]
                }
            ),
            Reference::Null,
        ]
    );
    assert_eq!(refs[0].object_address(), Some(0x1a73));
    // Without a resolver: unsupported error.
    assert!(Reference::decode(&dt, &raw[..12], &DecodeContext::default()).is_err());
    // Value wrapper.
    assert_eq!(
        decode_value(&dt, &raw[12..24], &ctx).unwrap().to_string(),
        "region@0x1a73"
    );
}

#[test]
fn selection_encodings() {
    // None.
    let mut v = 0u32.to_le_bytes().to_vec();
    v.extend(1u32.to_le_bytes());
    v.extend([0; 8]);
    assert_eq!(Selection::parse(&v).unwrap(), Selection::None);
    // Irregular v3 hyperslab with 4-byte fields: 2 blocks of rank 1.
    let mut v = 2u32.to_le_bytes().to_vec();
    v.extend(3u32.to_le_bytes());
    v.extend([0, 4]);
    v.extend(1u32.to_le_bytes());
    v.extend(2u32.to_le_bytes());
    for c in [0u32, 1, 5, 7] {
        v.extend(c.to_le_bytes());
    }
    assert_eq!(
        Selection::parse(&v).unwrap(),
        Selection::Hyperslab {
            rank: 1,
            blocks: vec![
                HyperslabBlock {
                    start: vec![0],
                    end: vec![1]
                },
                HyperslabBlock {
                    start: vec![5],
                    end: vec![7]
                }
            ]
        }
    );
    // Regular v2 hyperslab (8-byte fields).
    let mut v = 2u32.to_le_bytes().to_vec();
    v.extend(2u32.to_le_bytes());
    v.push(1);
    v.extend(36u32.to_le_bytes());
    v.extend(1u32.to_le_bytes());
    for c in [3u64, 1, 4, 1] {
        v.extend(c.to_le_bytes());
    }
    assert_eq!(
        Selection::parse(&v).unwrap(),
        Selection::RegularHyperslab {
            start: vec![3],
            stride: vec![1],
            count: vec![4],
            block: vec![1]
        }
    );
    // Huge counts are rejected before allocating.
    let mut v = 1u32.to_le_bytes().to_vec();
    v.extend(1u32.to_le_bytes());
    v.extend([0; 4]);
    v.extend(0u32.to_le_bytes());
    v.extend(2u32.to_le_bytes());
    v.extend(u32::MAX.to_le_bytes());
    assert!(Selection::parse(&v).is_err());
    // Bad type, rank, encode size.
    assert!(Selection::parse(&[9, 0, 0, 0, 1, 0, 0, 0]).is_err());
    assert!(Selection::parse(&[2, 0, 0, 0, 3, 0, 0, 0, 1, 3, 1, 0, 0, 0]).is_err());
    assert!(Selection::parse(&[2, 0, 0, 0, 3, 0, 0, 0, 1, 2, 99, 0, 0, 0]).is_err());
    assert!(Selection::parse(&[]).is_err());
}

#[test]
fn revised_references() {
    // Elements (18 bytes each, 8-byte offsets) and heap objects exactly as
    // written by libhdf5 for H5T_STD_REF data.
    let ext_name = b"/tmp/other.h5\0";
    let mut ext_obj = vec![8u8];
    ext_obj.extend(0x320u64.to_le_bytes());
    ext_obj.extend((ext_name.len() as u16).to_le_bytes());
    ext_obj.extend(ext_name);
    let mut region_obj = vec![8u8];
    region_obj.extend(0xb3u64.to_le_bytes());
    region_obj.extend(40u32.to_le_bytes());
    region_obj.extend(2u32.to_le_bytes());
    for w in [2u32, 1, 0, 24, 2, 1, 1, 2, 2, 3] {
        region_obj.extend(w.to_le_bytes());
    }
    let mut attr_obj = vec![8u8];
    attr_obj.extend(0xb3u64.to_le_bytes());
    attr_obj.extend(5u16.to_le_bytes());
    attr_obj.extend(b"units");
    let file = heap_file(HEAP, &[(1, &region_obj), (2, &attr_obj), (3, &ext_obj)]);
    let res = GlobalHeapResolver::new(&file[..], Sizes::DEFAULT);
    let ctx = DecodeContext::with_resolver(Sizes::DEFAULT, &res);
    let dt = reference(ReferenceKind::Object2, 18);

    let pad = |mut v: Vec<u8>| {
        v.resize(18, 0);
        v
    };
    let mut inline = vec![2u8, 0, 8];
    inline.extend(0xb3u64.to_le_bytes());
    let heap_el = |t: u8, flags: u8, len: u32, idx: u32| {
        let mut v = vec![t, flags];
        v.extend(len.to_le_bytes());
        v.extend(heap_id_bytes(HEAP, idx));
        v
    };
    let mut raw = pad(inline);
    raw.extend(heap_el(3, 0, region_obj.len() as u32, 1));
    raw.extend(heap_el(4, 0, attr_obj.len() as u32, 2));
    raw.extend(heap_el(2, 1, ext_obj.len() as u32, 3));
    raw.extend([0; 18]);
    let refs = decode_all::<Reference>(&dt, &raw, &ctx).unwrap();
    let Reference::Revised(r0) = &refs[0] else {
        panic!("{:?}", refs[0])
    };
    assert_eq!(r0.kind, ReferenceKind::Object2);
    assert_eq!(r0.object_address(), Some(0xb3));
    assert_eq!(refs[0].object_address(), Some(0xb3));
    let Reference::Revised(r1) = &refs[1] else {
        panic!()
    };
    assert_eq!(r1.kind, ReferenceKind::DatasetRegion2);
    assert_eq!(
        r1.selection,
        Some(Selection::Hyperslab {
            rank: 2,
            blocks: vec![HyperslabBlock {
                start: vec![1, 2],
                end: vec![2, 3]
            }]
        })
    );
    let Reference::Revised(r2) = &refs[2] else {
        panic!()
    };
    assert_eq!(r2.attribute.as_deref(), Some("units"));
    let Reference::Revised(r3) = &refs[3] else {
        panic!()
    };
    assert_eq!(r3.external_file.as_deref(), Some("/tmp/other.h5"));
    assert_eq!(r3.object_address(), Some(0x320));
    assert_eq!(refs[3].object_address(), None);
    assert_eq!(refs[4], Reference::Null);
    assert_eq!(
        decode_value(&dt, &raw[54..72], &ctx).unwrap().to_string(),
        "Object2@0x320 in \"/tmp/other.h5\""
    );
    // Block length beyond the heap object; unknown type byte.
    assert!(Reference::decode(&dt, &heap_el(4, 0, 200, 2), &ctx).is_err());
    assert!(Reference::decode(&dt, &pad(vec![9, 0, 1, 1]), &ctx).is_err());
    assert_eq!(dt.to_string(), "object reference (revised)");
}

// ---------------------------------------------------------------- display

#[test]
fn display() {
    use CharacterSet::*;
    use StringPadding::*;
    assert_eq!(int(1, false, LE).to_string(), "u8");
    assert_eq!(int(8, true, BE).to_string(), "i64 BE");
    assert_eq!(
        intp(2, false, LE, 4, 12).to_string(),
        "u16 LE (precision 12, offset 4)"
    );
    assert_eq!(float(2, LE).to_string(), "f16 LE");
    assert_eq!(float(8, BE).to_string(), "f64 BE");
    assert!(x87().to_string().starts_with("float128 LE (sign 79"));
    assert_eq!(
        fstr(8, SpacePad, Utf8).to_string(),
        "string[8] UTF-8 space-padded"
    );
    assert_eq!(vstr(Utf8).to_string(), "vlen string UTF-8");
    assert_eq!(
        vseq(int(2, true, LE)).to_string(),
        "vlen sequence of i16 LE"
    );
    assert_eq!(
        array(&[2, 3], float(4, LE)).to_string(),
        "array[2x3] of f32 LE"
    );
    assert_eq!(h5py_bool().to_string(), "enum i8 { FALSE = 0, TRUE = 1 }");
    assert_eq!(
        nested_compound().to_string(),
        "compound[28] { a: i32 LE @ 0, b: compound[16] { x: f64 BE @ 0, \
         y: array[2x2] of u16 LE @ 8 } @ 4, s: string[8] ASCII null-terminated @ 20 }"
    );
    assert_eq!(
        reference(ReferenceKind::DatasetRegion, 12).to_string(),
        "dataset region reference"
    );
    // Parsed from message bytes (datatype.rs encoding).
    let dt = Datatype::parse(&[
        0x11, 0x20, 0x3f, 0x00, 8, 0, 0, 0, 0, 0, 64, 0, 52, 11, 0, 52, 0xff, 3, 0, 0,
    ])
    .unwrap();
    assert_eq!(dt.to_string(), "f64 LE");
}

// ----------------------------------------------------------- error cases

#[test]
fn size_errors() {
    let dt = int(4, true, LE);
    assert!(i32::decode(&dt, &[0; 3], &ctx()).is_err());
    assert!(i32::decode(&dt, &[0; 5], &ctx()).is_err());
    assert!(decode_all::<i32>(&dt, &[0; 7], &ctx()).is_err());
    assert!(decode_all::<f64>(&float(8, LE), &[0; 9], &ctx()).is_err());
    assert!(
        decode_all::<bool>(&h5py_bool(), &[], &ctx())
            .unwrap()
            .is_empty()
    );
    assert!(decode_value(&dt, &[0; 3], &ctx()).is_err());
    let zero = Datatype {
        size: 0,
        ..dt.clone()
    };
    assert!(decode_all::<i32>(&zero, &[], &ctx()).is_err());
    assert!(decode_all::<String>(&zero, &[], &ctx()).is_err());
    assert_eq!(Value::Float(2.0).as_i64(), None);
    assert_eq!(Value::UInt(3).as_f64(), Some(3.0));
    assert_eq!(Value::Int(-1).as_u64(), None);
    assert_eq!(Value::Int(2).as_bool(), Some(true));
    assert_eq!(Value::String("x".into()).as_str(), Some("x"));
}

// ------------------------------------------------------- never panic

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn mutate_dt(dt: &mut Datatype, rng: &mut Rng) {
    match rng.next() % 6 {
        0 => dt.size = (rng.next() % 40) as u32,
        1 => {
            if let DatatypeClass::FixedPoint(fp) = &mut dt.class {
                fp.bit_offset = (rng.next() % 140) as u16;
                fp.bit_precision = (rng.next() % 140) as u16;
            }
        }
        2 => {
            if let DatatypeClass::FloatingPoint(fp) = &mut dt.class {
                fp.exponent_location = rng.next() as u8;
                fp.exponent_size = rng.next() as u8;
                fp.mantissa_location = rng.next() as u8;
                fp.mantissa_size = rng.next() as u8;
                fp.sign_location = rng.next() as u8;
                fp.exponent_bias = rng.next() as u32;
            }
        }
        3 => match &mut dt.class {
            DatatypeClass::Compound { members } => {
                if let Some(m) = members.first_mut() {
                    m.offset = (rng.next() % 64) as u32;
                    if rng.next().is_multiple_of(2) {
                        mutate_dt(&mut m.datatype, rng);
                    }
                }
            }
            DatatypeClass::Array { dims, base, .. } => {
                if rng.next().is_multiple_of(2) {
                    dims[0] = rng.next() as u32;
                } else {
                    mutate_dt(base, rng);
                }
            }
            DatatypeClass::VariableLength { base, .. }
            | DatatypeClass::Enumeration { base, .. } => mutate_dt(base, rng),
            _ => {}
        },
        _ => {}
    }
}

fn try_all(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) {
    let _ = decode_value(dt, bytes, ctx);
    let _ = decode_all::<i8>(dt, bytes, ctx);
    let _ = decode_all::<u16>(dt, bytes, ctx);
    let _ = decode_all::<i32>(dt, bytes, ctx);
    let _ = decode_all::<i64>(dt, bytes, ctx);
    let _ = decode_all::<u64>(dt, bytes, ctx);
    let _ = decode_all::<f32>(dt, bytes, ctx);
    let _ = decode_all::<f64>(dt, bytes, ctx);
    let _ = decode_all::<bool>(dt, bytes, ctx);
    let _ = decode_all::<String>(dt, bytes, ctx);
    let _ = decode_all::<StringBytes>(dt, bytes, ctx);
    let _ = decode_all::<Vec<i16>>(dt, bytes, ctx);
    let _ = decode_all::<[u16; 4]>(dt, bytes, ctx);
    let _ = decode_all::<Complex<f64>>(dt, bytes, ctx);
    let _ = decode_all::<Reference>(dt, bytes, ctx);
    let _ = decode_all::<Value>(dt, bytes, ctx);
    let _ = dt.to_string();
}

#[test]
fn mutations_never_panic() {
    let samples = vec![
        int(4, true, BE),
        intp(2, false, LE, 4, 12),
        float(2, BE),
        float(8, LE),
        x87(),
        h5py_bool(),
        fstr(8, StringPadding::NullTerminate, CharacterSet::Utf8),
        vstr(CharacterSet::Utf8),
        vseq(int(2, true, LE)),
        nested_compound(),
        array(&[2, 2], int(2, false, LE)),
        reference(ReferenceKind::Object, 8),
        reference(ReferenceKind::DatasetRegion, 12),
        reference(ReferenceKind::Object2, 18),
        compound(8, vec![("r", 0, float(4, LE)), ("i", 4, float(4, LE))]),
    ];
    let regions = region_heap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for round in 0..4000 {
        let mut dt = samples[(rng.next() % samples.len() as u64) as usize].clone();
        if round % 2 == 0 {
            mutate_dt(&mut dt, &mut rng);
        }
        // Random heap file: either valid or mutated.
        let mut file = if rng.next().is_multiple_of(2) {
            vlen_heap()
        } else {
            regions.clone()
        };
        if rng.next().is_multiple_of(3) {
            for _ in 0..4 {
                let i = (rng.next() % file.len() as u64) as usize;
                file[i] = rng.next() as u8;
            }
        }
        let res = GlobalHeapResolver::new(&file[..], Sizes::DEFAULT);
        let ctx = DecodeContext::with_resolver(Sizes::DEFAULT, &res);
        let n = (dt.size as usize).max(1) * (1 + (rng.next() % 3) as usize);
        let mut bytes: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
        // Often point vlen/region elements at the heap.
        if rng.next().is_multiple_of(2) && bytes.len() >= 16 {
            bytes[..16].copy_from_slice(&vlen_el(
                (rng.next() % 20) as u32,
                HEAP,
                (rng.next() % 9) as u32,
            ));
        }
        try_all(&dt, &bytes, &ctx);
        let cut = (rng.next() % (bytes.len() as u64 + 1)) as usize;
        try_all(&dt, &bytes[..cut], &ctx);
    }
}

// ------------------------------------------- selections -> read selections

#[test]
fn selection_conversion() {
    use crate::read::{HyperslabDim as H, Selection as S};
    let blk = |s: &[u64], e: &[u64]| HyperslabBlock {
        start: s.to_vec(),
        end: e.to_vec(),
    };
    assert_eq!(Selection::None.to_read_selection().unwrap(), None);
    assert_eq!(Selection::All.to_read_selection().unwrap(), Some(S::All));
    let reg = Selection::RegularHyperslab {
        start: vec![1, 0],
        stride: vec![3, 1],
        count: vec![2, 1],
        block: vec![2, 4],
    };
    assert_eq!(
        reg.to_read_selection().unwrap(),
        Some(S::Hyperslab(vec![H::new(1, 3, 2, 2), H::new(0, 1, 1, 4)]))
    );
    // A cartesian product of blocks, listed in any order.
    let grid = Selection::Hyperslab {
        rank: 2,
        blocks: vec![
            blk(&[4, 0], &[4, 1]),
            blk(&[0, 0], &[0, 1]),
            blk(&[0, 5], &[0, 6]),
            blk(&[4, 5], &[4, 6]),
        ],
    };
    assert_eq!(
        grid.to_read_selection().unwrap(),
        Some(S::Hyperslab(vec![H::new(0, 4, 2, 1), H::new(0, 5, 2, 2)]))
    );
    // Irregular: a missing corner, unequal blocks, uneven spacing.
    for blocks in [
        vec![blk(&[0, 0], &[0, 0]), blk(&[2, 2], &[2, 2])],
        vec![blk(&[0], &[1]), blk(&[4], &[4])],
        vec![blk(&[0], &[0]), blk(&[2], &[2]), blk(&[5], &[5])],
        vec![blk(&[0], &[3]), blk(&[2], &[5])],
    ] {
        let rank = blocks[0].start.len() as u32;
        assert!(matches!(
            Selection::Hyperslab { rank, blocks }.to_read_selection(),
            Err(Error::Unsupported(_))
        ));
    }
    let pts = Selection::Points {
        rank: 1,
        points: vec![vec![1]],
    };
    assert!(matches!(
        pts.to_read_selection(),
        Err(Error::Unsupported(_))
    ));
    // parse_prefix reports the encoded length of "all" (8 + 8 bytes).
    let mut enc = vec![3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    enc.extend([0xaa; 4]);
    assert_eq!(Selection::parse_prefix(&enc).unwrap(), (Selection::All, 16));
}
