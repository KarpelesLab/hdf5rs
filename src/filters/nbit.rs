//! The N-bit filter (filter id 5).
//!
//! N-bit packing keeps only the *significant* bits of every value — the
//! `precision` bits starting at bit `offset` of an integer or
//! floating-point datatype — and concatenates them into an MSB-first bit
//! stream. Padding bits are dropped on write and restored as zero on read.
//!
//! # Client data
//!
//! The library fills the client data in when the dataset is created (the
//! user passes none), describing the chunk and its datatype:
//!
//! ```text
//! [0] number of client data values
//! [1] 1 if the data needs no packing (stored unchanged), else 0
//! [2] number of elements in a chunk
//! [3..] datatype parameters:
//!   atomic   (integer, float):  1, size, order (0 LE / 1 BE), precision, offset
//!   array:                      2, size, <base type parameters>
//!   compound:                   3, size, member count,
//!                               then per member: offset, <member parameters>
//!   no-op    (anything else):   4, size
//! ```
//!
//! For every element the datatype tree is walked in order: atomic values
//! contribute their significant bits (most significant first), no-op values
//! contribute all of their bytes (8 bits each, still unaligned), arrays
//! repeat their base type. The packed buffer is `floor(bits / 8) + 1` bytes
//! long. Bytes of a compound not covered by any member decode as zero.

use super::bits::{BitReader, BitWriter};
use super::{Filter, FilterContext, NBIT};
use crate::format::messages::datatype::{ByteOrder, Datatype, DatatypeClass};
use crate::{Error, Result};

/// Maximum nesting depth of compound/array parameters.
const MAX_DEPTH: usize = 32;

const CLASS_ATOMIC: u32 = 1;
const CLASS_ARRAY: u32 = 2;
const CLASS_COMPOUND: u32 = 3;
const CLASS_NOOP: u32 = 4;

/// The N-bit filter, id [`NBIT`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NBit;

impl Filter for NBit {
    fn id(&self) -> u16 {
        NBIT
    }

    fn name(&self) -> &str {
        "nbit"
    }

    fn decode(&self, client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        let p = Params::parse(client_data)?;
        if p.no_op {
            return Ok(input);
        }
        let size = p.node.size();
        let out_len = p
            .nelem
            .checked_mul(size)
            .filter(|&n| n <= ctx.max_output)
            .ok_or(Error::LimitExceeded("nbit output size"))?;
        let mut out = vec![0u8; out_len];
        let mut r = BitReader::new(&input);
        if size > 0 {
            for elem in out.chunks_exact_mut(size) {
                p.node.unpack(&mut r, elem)?;
            }
        }
        Ok(out)
    }

    fn encode(&self, client_data: &[u32], input: Vec<u8>, _ctx: &FilterContext) -> Result<Vec<u8>> {
        let p = Params::parse(client_data)?;
        if p.no_op {
            return Ok(input);
        }
        let size = p.node.size();
        if p.nelem.checked_mul(size) != Some(input.len()) {
            return Err(Error::invalid(
                "nbit: chunk size does not match client data",
            ));
        }
        let bits = p.node.packed_bits()?;
        let total = (bits as u64).saturating_mul(p.nelem as u64);
        let mut w = BitWriter::with_capacity(usize::try_from(total / 8 + 1).unwrap_or(0));
        if size > 0 {
            for elem in input.chunks_exact(size) {
                p.node.pack(&mut w, elem);
            }
        }
        Ok(w.finish())
    }
}

/// Parsed client data.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Params {
    no_op: bool,
    nelem: usize,
    node: Node,
}

impl Params {
    fn parse(cd: &[u32]) -> Result<Params> {
        let bad = || Error::invalid("nbit: malformed client data");
        let count = *cd.first().ok_or_else(bad)? as usize;
        let cd = cd.get(..count).filter(|c| c.len() >= 3).ok_or_else(bad)?;
        let no_op = cd[1] != 0;
        let nelem = cd[2] as usize;
        if no_op {
            // The type description is irrelevant (and may be absent).
            return Ok(Params {
                no_op,
                nelem,
                node: Node::NoOp { size: 0 },
            });
        }
        let mut pos = 3;
        let node = Node::parse(cd, &mut pos, 0)?;
        Ok(Params { no_op, nelem, node })
    }
}

/// One node of the datatype description.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Atomic {
        size: usize,
        big_endian: bool,
        precision: usize,
        offset: usize,
    },
    NoOp {
        size: usize,
    },
    Array {
        size: usize,
        base: Box<Node>,
    },
    Compound {
        size: usize,
        members: Vec<(usize, Node)>,
    },
}

fn take(cd: &[u32], pos: &mut usize) -> Result<usize> {
    let v = cd
        .get(*pos)
        .ok_or_else(|| Error::invalid("nbit: client data is truncated"))?;
    *pos += 1;
    Ok(*v as usize)
}

impl Node {
    fn parse(cd: &[u32], pos: &mut usize, depth: usize) -> Result<Node> {
        if depth > MAX_DEPTH {
            return Err(Error::invalid("nbit: datatype nesting too deep"));
        }
        let class = take(cd, pos)? as u32;
        let size = take(cd, pos)?;
        match class {
            CLASS_ATOMIC => {
                let order = take(cd, pos)?;
                let precision = take(cd, pos)?;
                let offset = take(cd, pos)?;
                let ok = order <= 1
                    && precision > 0
                    && size.checked_mul(8).is_some_and(|bits| {
                        precision.checked_add(offset).is_some_and(|e| e <= bits)
                    });
                if !ok {
                    return Err(Error::invalid("nbit: bad atomic datatype parameters"));
                }
                Ok(Node::Atomic {
                    size,
                    big_endian: order == 1,
                    precision,
                    offset,
                })
            }
            CLASS_NOOP => Ok(Node::NoOp { size }),
            CLASS_ARRAY => {
                let base = Node::parse(cd, pos, depth + 1)?;
                let bs = base.size();
                if bs == 0 || size % bs != 0 {
                    return Err(Error::invalid("nbit: bad array datatype parameters"));
                }
                Ok(Node::Array {
                    size,
                    base: Box::new(base),
                })
            }
            CLASS_COMPOUND => {
                let n = take(cd, pos)?;
                // Every member needs at least 3 values; bounds the allocation.
                if n > cd.len() {
                    return Err(Error::invalid("nbit: bad compound member count"));
                }
                let mut members = Vec::with_capacity(n);
                for _ in 0..n {
                    let off = take(cd, pos)?;
                    let m = Node::parse(cd, pos, depth + 1)?;
                    if off.checked_add(m.size()).is_none_or(|e| e > size) {
                        return Err(Error::invalid("nbit: compound member out of bounds"));
                    }
                    members.push((off, m));
                }
                Ok(Node::Compound { size, members })
            }
            _ => Err(Error::invalid("nbit: unknown datatype class")),
        }
    }

    fn size(&self) -> usize {
        match self {
            Node::Atomic { size, .. }
            | Node::NoOp { size }
            | Node::Array { size, .. }
            | Node::Compound { size, .. } => *size,
        }
    }

    /// Packed bits per value of this node.
    fn packed_bits(&self) -> Result<usize> {
        let overflow = || Error::LimitExceeded("nbit packed size");
        Ok(match self {
            Node::Atomic { precision, .. } => *precision,
            Node::NoOp { size } => size.checked_mul(8).ok_or_else(overflow)?,
            Node::Array { size, base } => (size / base.size())
                .checked_mul(base.packed_bits()?)
                .ok_or_else(overflow)?,
            Node::Compound { members, .. } => {
                let mut t = 0usize;
                for (_, m) in members {
                    t = t.checked_add(m.packed_bits()?).ok_or_else(overflow)?;
                }
                t
            }
        })
    }

    /// Unpacks one value into `out` (exactly `self.size()` bytes, zeroed).
    fn unpack(&self, r: &mut BitReader<'_>, out: &mut [u8]) -> Result<()> {
        match self {
            Node::Atomic {
                size,
                big_endian,
                precision,
                offset,
            } => {
                for_each_field(*size, *big_endian, *precision, *offset, |k, shift, n| {
                    out[k] |= (r.read(n)? as u8) << shift;
                    Ok(())
                })?;
            }
            Node::NoOp { .. } => {
                for b in out.iter_mut() {
                    *b = r.read(8)? as u8;
                }
            }
            Node::Array { base, .. } => {
                for part in out.chunks_exact_mut(base.size()) {
                    base.unpack(r, part)?;
                }
            }
            Node::Compound { members, .. } => {
                for (off, m) in members {
                    m.unpack(r, &mut out[*off..*off + m.size()])?;
                }
            }
        }
        Ok(())
    }

    /// Packs one value (`self.size()` bytes).
    fn pack(&self, w: &mut BitWriter, value: &[u8]) {
        match self {
            Node::Atomic {
                size,
                big_endian,
                precision,
                offset,
            } => {
                let _ = for_each_field(*size, *big_endian, *precision, *offset, |k, shift, n| {
                    w.write(u64::from(value[k] >> shift), n);
                    Ok(())
                });
            }
            Node::NoOp { .. } => value.iter().for_each(|&b| w.write(u64::from(b), 8)),
            Node::Array { base, .. } => {
                for part in value.chunks_exact(base.size()) {
                    base.pack(w, part);
                }
            }
            Node::Compound { members, .. } => {
                for (off, m) in members {
                    m.pack(w, &value[*off..*off + m.size()]);
                }
            }
        }
    }
}

/// Visits the significant bit fields of an atomic value, most significant
/// byte first: `f(byte index in memory, bit shift within the byte, bits)`.
fn for_each_field(
    size: usize,
    big_endian: bool,
    precision: usize,
    offset: usize,
    mut f: impl FnMut(usize, u32, u32) -> Result<()>,
) -> Result<()> {
    let end = offset + precision;
    for i in 0..size {
        // `i`-th most significant byte; it holds value bits [base, base + 8).
        let base = (size - 1 - i) * 8;
        let lo = offset.max(base);
        let hi = end.min(base + 8);
        if lo >= hi {
            continue;
        }
        let k = if big_endian { i } else { size - 1 - i };
        f(k, (lo - base) as u32, (hi - lo) as u32)?;
    }
    Ok(())
}

/// Computes the client data the N-bit filter needs for a dataset of
/// `dtype` with `chunk_elements` elements per chunk (what the reference
/// library's "set local" step stores).
///
/// Integers and floating-point numbers are packed to their precision;
/// compounds and arrays are described recursively; every other class is
/// copied unchanged. Variable-length types are rejected.
pub fn client_data(dtype: &Datatype, chunk_elements: u32) -> Result<Vec<u32>> {
    let mut params = Vec::new();
    let reduced = describe(dtype, &mut params, 0)?;
    let mut cd = Vec::with_capacity(params.len() + 3);
    cd.push(0);
    cd.push(u32::from(!reduced));
    cd.push(chunk_elements);
    if reduced {
        cd.extend_from_slice(&params);
    }
    cd[0] = u32::try_from(cd.len()).map_err(|_| Error::LimitExceeded("nbit client data"))?;
    Ok(cd)
}

/// Appends the parameters of `dtype`; returns whether any value has fewer
/// significant bits than its size.
fn describe(dtype: &Datatype, out: &mut Vec<u32>, depth: usize) -> Result<bool> {
    if depth > MAX_DEPTH {
        return Err(Error::invalid("nbit: datatype nesting too deep"));
    }
    let atomic = |out: &mut Vec<u32>, order: ByteOrder, precision: u16, offset: u16| {
        let order = match order {
            ByteOrder::LittleEndian => 0,
            ByteOrder::BigEndian => 1,
            ByteOrder::Vax => return Err(Error::unsupported("nbit: VAX byte order")),
        };
        out.extend_from_slice(&[
            CLASS_ATOMIC,
            dtype.size,
            order,
            u32::from(precision),
            u32::from(offset),
        ]);
        Ok(u32::from(precision) < dtype.size.saturating_mul(8))
    };
    match &dtype.class {
        DatatypeClass::FixedPoint(f) => atomic(out, f.byte_order, f.bit_precision, f.bit_offset),
        DatatypeClass::FloatingPoint(f) => atomic(out, f.byte_order, f.bit_precision, f.bit_offset),
        DatatypeClass::Array { base, .. } => {
            out.extend_from_slice(&[CLASS_ARRAY, dtype.size]);
            describe(base, out, depth + 1)
        }
        DatatypeClass::Compound { members } => {
            let n = u32::try_from(members.len())
                .map_err(|_| Error::LimitExceeded("nbit: compound members"))?;
            out.extend_from_slice(&[CLASS_COMPOUND, dtype.size, n]);
            let mut reduced = false;
            for m in members {
                out.push(m.offset);
                if m.array_dims.is_some() {
                    // Version 1 array members: the member size covers the
                    // whole array, so describe it as an array.
                    let count: u32 = m.array_dims.iter().flatten().product();
                    out.extend_from_slice(&[CLASS_ARRAY, m.datatype.size.saturating_mul(count)]);
                }
                reduced |= describe(&m.datatype, out, depth + 1)?;
            }
            Ok(reduced)
        }
        DatatypeClass::VariableLength { .. } => Err(Error::unsupported(
            "nbit: variable-length data cannot be filtered",
        )),
        _ => {
            out.extend_from_slice(&[CLASS_NOOP, dtype.size]);
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(cd: &[u32], input: &[u8]) -> Result<Vec<u8>> {
        NBit.decode(cd, input.to_vec(), &FilterContext::default())
    }

    fn enc(cd: &[u32], input: &[u8]) -> Vec<u8> {
        NBit.encode(cd, input.to_vec(), &FilterContext::default())
            .unwrap()
    }

    fn check(cd: &[u32], plain: &[u8], packed: &[u8]) {
        assert_eq!(dec(cd, packed).unwrap(), plain, "decode {cd:?}");
        assert_eq!(enc(cd, plain), packed, "encode {cd:?}");
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // Expected bytes below were produced by the reference library.

    #[test]
    fn u16_precision_12() {
        let vals: [u16; 7] = [1, 2, 0xfff, 0x800, 5, 6, 7];
        let plain: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        check(
            &[8, 0, 7, 1, 2, 0, 12, 0],
            &plain,
            &hex("001002fff8000050060070"),
        );
    }

    #[test]
    fn big_endian_with_offset() {
        let vals: [u16; 4] = [1, 2, 0xfff, 0x800];
        let plain: Vec<u8> = vals.iter().flat_map(|v| (v << 2).to_be_bytes()).collect();
        check(&[8, 0, 4, 1, 2, 1, 12, 2], &plain, &hex("001002fff80000"));
        let vals: [i64; 4] = [-5, 5, (1 << 39) - 1, -(1 << 39)];
        let mask = (1i64 << 50) - 1;
        let plain: Vec<u8> = vals
            .iter()
            .flat_map(|v| ((v << 10) & mask).to_be_bytes())
            .collect();
        check(
            &[8, 0, 4, 1, 8, 1, 40, 10],
            &plain,
            &hex("fffffffffb00000000057fffffffff800000000000"),
        );
    }

    #[test]
    fn signed_with_offset() {
        let vals: [i32; 4] = [-1, 1, -524288, 524287];
        let plain: Vec<u8> = vals
            .iter()
            .flat_map(|v| ((v << 4) & 0x00ff_fff0).to_le_bytes())
            .collect();
        check(
            &[8, 0, 4, 1, 4, 0, 20, 4],
            &plain,
            &hex("fffff00001800007ffff00"),
        );
    }

    #[test]
    fn no_op_flag_passes_through() {
        let plain = hex("00000000010000000200000003000000");
        check(&[8, 1, 4, 1, 4, 0, 32, 0], &plain, &plain);
        check(&[3, 1, 2], b"abcdef\0\0", b"abcdef\0\0");
    }

    #[test]
    fn compound_with_noop_and_array_members() {
        // {a: u16 LE prec 12 @0, b: i32 BE @2, c: 3-byte string @6,
        //  d: u8[2] prec 4 offset 2 @9}, size 12.
        let cd = [
            29, 0, 3, 3, 12, 4, 0, 1, 2, 0, 12, 0, 2, 1, 4, 1, 32, 0, 6, 4, 3, 9, 2, 2, 1, 1, 0, 4,
            2,
        ];
        let mut plain = Vec::new();
        for (a, b, c, d) in [
            (1u16, -1i32, *b"ab\0", [1u8, 2]),
            (0xfff, 2, *b"de\0", [3, 15]),
            (0x123, 0x0102_0304, [0; 3], [0, 7]),
        ] {
            plain.extend_from_slice(&a.to_le_bytes());
            plain.extend_from_slice(&b.to_be_bytes());
            plain.extend_from_slice(&c);
            plain.extend_from_slice(&[d[0] << 2, d[1] << 2, 0]);
        }
        check(
            &cd,
            &plain,
            &hex("001ffffffff61620012fff000000026465003f12301020304000000070"),
        );
    }

    #[test]
    fn nested_compound() {
        // {n: {p: u8 prec 5 @0, q: u16 prec 7 off 3 @2} @0, m: u8 prec 2 off 6 @4}
        let cd = [
            28, 0, 2, 3, 6, 2, 0, 3, 4, 2, 0, 1, 1, 0, 5, 0, 2, 1, 2, 0, 7, 3, 4, 1, 1, 0, 2, 6,
        ];
        let mut plain = Vec::new();
        for (p, q, m) in [(31u8, 127u16, 3u8), (1, 3, 1)] {
            plain.push(p);
            plain.push(0);
            plain.extend_from_slice(&(q << 3).to_le_bytes());
            plain.extend_from_slice(&[m << 6, 0]);
        }
        check(&cd, &plain, &hex("fffc20d0"));
    }

    #[test]
    fn array_type() {
        let cd = [10, 0, 2, 2, 6, 1, 2, 1, 10, 1];
        let plain: Vec<u8> = [1u16, 2, 3, 1023, 0, 512]
            .iter()
            .flat_map(|v| (v << 1).to_be_bytes())
            .collect();
        check(&cd, &plain, &hex("0040200fff002000"));
    }

    #[test]
    fn malformed_client_data() {
        for cd in [
            &[][..],
            &[2, 0],
            &[9, 0, 4, 1, 2, 0, 12, 0],  // count too large
            &[8, 0, 4, 1, 2, 0, 17, 0],  // precision > size
            &[8, 0, 4, 1, 2, 0, 12, 5],  // precision + offset > size
            &[8, 0, 4, 1, 2, 2, 12, 0],  // bad order
            &[8, 0, 4, 9, 2, 0, 12, 0],  // bad class
            &[6, 0, 4, 2, 3, 4],         // truncated array
            &[7, 0, 4, 3, 4, 1, 3],      // member out of bounds
            &[8, 0, 4, 3, 4, 100, 0, 0], // member count too large
            &[8, 0, 4, 2, 3, 4, 2, 0],   // array not a multiple of base
        ] {
            assert!(dec(cd, &[0; 16]).is_err(), "{cd:?}");
        }
        // Truncated packed data.
        assert!(dec(&[8, 0, 7, 1, 2, 0, 12, 0], &[0; 10]).is_err());
        // Output cap.
        let ctx = FilterContext::default().with_max_output(10);
        assert!(matches!(
            NBit.decode(&[8, 0, 100, 1, 2, 0, 12, 0], vec![0; 200], &ctx),
            Err(Error::LimitExceeded(_))
        ));
        // Encode with mismatched size.
        assert!(
            NBit.encode(
                &[8, 0, 7, 1, 2, 0, 12, 0],
                vec![0; 13],
                &FilterContext::default()
            )
            .is_err()
        );
    }

    #[test]
    fn deep_nesting_rejected() {
        let mut cd = vec![0, 0, 1];
        for _ in 0..100 {
            cd.extend_from_slice(&[2, 2]);
        }
        cd.extend_from_slice(&[1, 2, 0, 12, 0]);
        cd[0] = cd.len() as u32;
        assert!(dec(&cd, &[0; 8]).is_err());
    }

    #[test]
    fn client_data_from_datatype() {
        use crate::format::messages::datatype::{
            CharacterSet, CompoundMember, FixedPoint, StringPadding, StringType,
        };
        let int = |size: u32, order, precision, offset| Datatype {
            version: 1,
            size,
            class: DatatypeClass::FixedPoint(FixedPoint {
                byte_order: order,
                lo_pad: false,
                hi_pad: false,
                signed: false,
                bit_offset: offset,
                bit_precision: precision,
            }),
        };
        assert_eq!(
            client_data(&int(2, ByteOrder::LittleEndian, 12, 0), 50).unwrap(),
            [8, 0, 50, 1, 2, 0, 12, 0]
        );
        assert_eq!(
            client_data(&int(4, ByteOrder::LittleEndian, 32, 0), 4).unwrap(),
            [3, 1, 4]
        );
        let s = Datatype {
            version: 1,
            size: 3,
            class: DatatypeClass::String(StringType {
                padding: StringPadding::NullTerminate,
                charset: CharacterSet::Ascii,
            }),
        };
        let member = |name: &str, offset, datatype| CompoundMember {
            name: name.into(),
            offset,
            array_dims: None,
            datatype,
        };
        let arr = Datatype {
            version: 2,
            size: 2,
            class: DatatypeClass::Array {
                dims: vec![2],
                permutation: None,
                base: Box::new(int(1, ByteOrder::LittleEndian, 4, 2)),
            },
        };
        let cmp = Datatype {
            version: 1,
            size: 12,
            class: DatatypeClass::Compound {
                members: vec![
                    member("a", 0, int(2, ByteOrder::LittleEndian, 12, 0)),
                    member("b", 2, int(4, ByteOrder::BigEndian, 32, 0)),
                    member("c", 6, s),
                    member("d", 9, arr),
                ],
            },
        };
        assert_eq!(
            client_data(&cmp, 3).unwrap(),
            [
                29, 0, 3, 3, 12, 4, 0, 1, 2, 0, 12, 0, 2, 1, 4, 1, 32, 0, 6, 4, 3, 9, 2, 2, 1, 1,
                0, 4, 2
            ]
        );
    }
}
