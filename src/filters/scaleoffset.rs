//! The scale-offset filter (filter id 6).
//!
//! Scale-offset stores every value of a chunk as an unsigned offset from
//! the chunk minimum, using only as many bits (`minbits`) as the value
//! range needs. Floating-point data is first scaled by `10^D` and rounded
//! ("D-scaling", lossy). The codes are packed into an MSB-first bit stream
//! after a fixed 21-byte header.
//!
//! # Client data
//!
//! The user supplies `[scale type, scale factor]`; the library completes
//! the rest when the dataset is created:
//!
//! ```text
//! [0] scale type: 0 = float D-scale, 1 = float E-scale, 2 = integer
//! [1] scale factor: D for D-scaling, minbits for integers (0 = automatic)
//! [2] number of elements in a chunk
//! [3] datatype class: 0 = integer, 1 = float
//! [4] datatype size in bytes
//! [5] 1 if the integer type is signed
//! [6] byte order: 0 = little-endian, 1 = big-endian
//! [7] 1 if a fill value is defined
//! [8..] the fill value bytes, little-endian, packed four per value
//!       (byte k in bits 8*(k%4) of value 8 + k/4)
//! ```
//!
//! # Chunk format
//!
//! ```text
//! bytes 0..4    minbits (u32 little-endian)
//! byte  4       size of the minimum value field (8)
//! bytes 5..13   minimum (little-endian; integers sign-extended to 64 bits,
//!               floats as their IEEE bits in the low 4 or 8 bytes)
//! bytes 13..21  zero
//! bytes 21..    minbits-bit codes, MSB first, floor(bits / 8) + 1 bytes
//! ```
//!
//! When a fill value is defined, the all-ones code stands for the fill
//! value (elements equal to it — for floats, within `10^-D` of it — are
//! excluded from the range). When the range needs the full width of the
//! type, `minbits` equals the type's bit size and the elements follow the
//! header in little-endian order whatever the dataset's byte order. An
//! integer chunk whose scale factor equals the type's bit size is stored
//! unchanged, without any header.
//!
//! Floating-point values encode as `round(x·10^D − min·10^D)` and decode
//! as `code / 10^D + minimum`, single-precision data in single precision
//! (behaviour matched bit for bit against the reference implementation).
//! E-scaling is not supported (the reference library does not implement it
//! either).

use super::bits::{BitReader, BitWriter};
use super::{Filter, FilterContext, SCALEOFFSET};
use crate::format::messages::datatype::{ByteOrder, Datatype, DatatypeClass};
use crate::{Error, Result};

/// Size of the chunk header.
const HEADER_LEN: usize = 21;

/// Scale type: floating-point D-scaling.
pub const SCALE_FLOAT_DSCALE: u32 = 0;
/// Scale type: floating-point E-scaling (not supported).
pub const SCALE_FLOAT_ESCALE: u32 = 1;
/// Scale type: integer.
pub const SCALE_INT: u32 = 2;

/// The scale-offset filter, id [`SCALEOFFSET`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ScaleOffset;

impl Filter for ScaleOffset {
    fn id(&self) -> u16 {
        SCALEOFFSET
    }

    fn name(&self) -> &str {
        "scaleoffset"
    }

    fn decode(&self, client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        let p = Params::parse(client_data)?;
        if p.passthrough()? {
            return Ok(input);
        }
        let out_len = p
            .nelem
            .checked_mul(p.size)
            .filter(|&n| n <= ctx.max_output)
            .ok_or(Error::LimitExceeded("scaleoffset output size"))?;
        let header = input
            .get(..HEADER_LEN)
            .ok_or_else(|| Error::invalid("scaleoffset: chunk shorter than its header"))?;
        let minbits = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let min_len = usize::from(header[4]);
        if min_len > 8 {
            return Err(Error::invalid("scaleoffset: bad minimum value size"));
        }
        let mut min_bytes = [0u8; 8];
        min_bytes[..min_len].copy_from_slice(&header[5..5 + min_len]);
        let min_raw = u64::from_le_bytes(min_bytes);
        let full_bits = p.size as u32 * 8;
        let data = &input[HEADER_LEN..];
        if minbits > full_bits {
            return Err(Error::invalid("scaleoffset: minbits exceeds the type size"));
        }
        if minbits == full_bits {
            // Stored unconverted, in little-endian order.
            let data = data
                .get(..out_len)
                .ok_or_else(|| Error::invalid("scaleoffset: chunk data is truncated"))?;
            let mut out = Vec::with_capacity(out_len);
            for c in data.chunks_exact(p.size) {
                push_value(&mut out, read_value(c, false), p.size, p.big_endian);
            }
            return Ok(out);
        }
        let mut out = Vec::with_capacity(out_len);
        let mut r = BitReader::new(data);
        let all_ones = if minbits == 0 {
            0
        } else {
            u64::MAX >> (64 - minbits)
        };
        // Fill value as a little-endian integer of `size` bytes.
        let fill = p.fill.map(u64::from_le_bytes);
        let scale = 10f64.powi(p.factor);
        for _ in 0..p.nelem {
            let code = r.read(minbits)?;
            let value = match fill {
                Some(fill) if minbits > 0 && code == all_ones => fill,
                // Single precision adds in single precision.
                _ if p.float && p.size == 4 => {
                    let x = (code as f64 / scale) as f32 + f32::from_bits(min_raw as u32);
                    u64::from(x.to_bits())
                }
                _ if p.float => (code as f64 / scale + f64::from_bits(min_raw)).to_bits(),
                _ => min_raw.wrapping_add(code),
            };
            push_value(&mut out, value, p.size, p.big_endian);
        }
        Ok(out)
    }

    fn encode(&self, client_data: &[u32], input: Vec<u8>, _ctx: &FilterContext) -> Result<Vec<u8>> {
        let p = Params::parse(client_data)?;
        if p.passthrough()? {
            return Ok(input);
        }
        if p.nelem.checked_mul(p.size) != Some(input.len()) {
            return Err(Error::invalid(
                "scaleoffset: chunk size does not match client data",
            ));
        }
        let values: Vec<u64> = input
            .chunks_exact(p.size)
            .map(|c| read_value(c, p.big_endian))
            .collect();
        let full_bits = p.size as u32 * 8;
        let fill = p.fill.map(u64::from_le_bytes);
        let plan = if p.float {
            plan_float(&p, &values, fill)
        } else {
            plan_int(&p, &values, fill)
        };
        let mut out = Vec::with_capacity(HEADER_LEN + input.len() + 1);
        let (minbits, min_raw) = match &plan {
            Plan::Raw { min_raw } => (full_bits, *min_raw),
            Plan::Packed {
                minbits, min_raw, ..
            } => (*minbits, *min_raw),
        };
        out.extend_from_slice(&minbits.to_le_bytes());
        out.push(8);
        out.extend_from_slice(&min_raw.to_le_bytes());
        out.extend_from_slice(&[0; 8]);
        match plan {
            Plan::Raw { .. } => {
                for v in values {
                    push_value(&mut out, v, p.size, false);
                }
            }
            Plan::Packed { minbits, codes, .. } => {
                let mut w = BitWriter::with_capacity(codes.len() * minbits as usize / 8 + 1);
                for c in codes {
                    w.write(c, minbits);
                }
                out.extend_from_slice(&w.finish());
            }
        }
        Ok(out)
    }
}

/// How a chunk is encoded.
enum Plan {
    /// Values stored unchanged (little-endian) after the header. The
    /// reference keeps the minimum in the header, except when the range
    /// (plus the fill code) reaches the type's all-ones value.
    Raw { min_raw: u64 },
    Packed {
        minbits: u32,
        min_raw: u64,
        codes: Vec<u64>,
    },
}

/// Number of bits needed to represent `v`.
fn bit_len(v: u128) -> u32 {
    128 - v.leading_zeros()
}

fn plan_int(p: &Params, values: &[u64], fill: Option<u64>) -> Plan {
    let full_bits = p.size as u32 * 8;
    // Values as i128 (sign-extended when signed) so ranges cannot overflow.
    let widen = |v: u64| -> i128 {
        if p.signed && full_bits < 64 {
            let shift = 64 - full_bits;
            i128::from(((v << shift) as i64) >> shift)
        } else if p.signed {
            i128::from(v as i64)
        } else {
            i128::from(v)
        }
    };
    let mut range: Option<(i128, i128)> = None;
    for &v in values {
        if Some(v) == fill {
            continue;
        }
        let w = widen(v);
        range = Some(match range {
            None => (w, w),
            Some((lo, hi)) => (lo.min(w), hi.max(w)),
        });
    }
    let (lo, hi) = range.unwrap_or((0, 0));
    let needed = (hi - lo) as u128 + u128::from(fill.is_some());
    if p.factor <= 0 && needed >= (1u128 << full_bits) - 1 {
        return Plan::Raw { min_raw: 0 };
    }
    let minbits = if p.factor > 0 {
        p.factor as u32
    } else {
        bit_len(needed)
    };
    if minbits >= full_bits {
        return Plan::Raw { min_raw: lo as u64 };
    }
    let mask = if minbits == 0 {
        0
    } else {
        u64::MAX >> (64 - minbits)
    };
    let codes = values
        .iter()
        .map(|&v| match fill {
            Some(f) if v == f => mask,
            _ => ((widen(v) - lo) as u64) & mask,
        })
        .collect();
    Plan::Packed {
        minbits,
        min_raw: lo as u64,
        codes,
    }
}

fn plan_float(p: &Params, values: &[u64], fill: Option<u64>) -> Plan {
    let full_bits = p.size as u32 * 8;
    let single = p.size == 4;
    let to_f64 = |v: u64| -> f64 {
        if single {
            f64::from(f32::from_bits(v as u32))
        } else {
            f64::from_bits(v)
        }
    };
    let scale = 10f64.powi(p.factor);
    let tolerance = 10f64.powi(-p.factor);
    let fill_f = fill.map(to_f64);
    let is_fill = |x: f64| fill_f.is_some_and(|f| (x - f).abs() < tolerance);
    let mut range: Option<(f64, f64)> = None;
    for &v in values {
        let x = to_f64(v);
        if is_fill(x) {
            continue;
        }
        if !x.is_finite() {
            return Plan::Raw { min_raw: 0 };
        }
        range = Some(match range {
            None => (x, x),
            Some((lo, hi)) => (lo.min(x), hi.max(x)),
        });
    }
    let (lo, hi) = range.unwrap_or((0.0, 0.0));
    let min_raw = if single {
        u64::from((lo as f32).to_bits())
    } else {
        lo.to_bits()
    };
    // Scaled offset from the minimum, `round(x·10^D − min·10^D)`, computed
    // in the type's own precision like the reference.
    let scaled = |x: f64| -> f64 {
        if single {
            let s = scale as f32;
            f64::from((x as f32 * s - lo as f32 * s).round())
        } else {
            (x * scale - lo * scale).round()
        }
    };
    let span = scaled(hi);
    if !span.is_finite() || span >= 2f64.powi(full_bits as i32) - 1.0 {
        return Plan::Raw { min_raw: 0 };
    }
    let needed = span as u128 + u128::from(fill_f.is_some());
    if needed >= (1u128 << full_bits) - 1 {
        return Plan::Raw { min_raw: 0 };
    }
    let minbits = bit_len(needed);
    if minbits >= full_bits {
        return Plan::Raw { min_raw };
    }
    let mask = if minbits == 0 {
        0
    } else {
        u64::MAX >> (64 - minbits)
    };
    let codes = values
        .iter()
        .map(|&v| {
            let x = to_f64(v);
            if is_fill(x) {
                mask
            } else {
                // `as` saturates; the range check above keeps codes in range.
                (scaled(x).max(0.0) as u64).min(mask)
            }
        })
        .collect();
    Plan::Packed {
        minbits,
        min_raw,
        codes,
    }
}

/// Reads a value of `c.len()` (<= 8) bytes as a little-endian-ordered u64.
fn read_value(c: &[u8], big_endian: bool) -> u64 {
    let mut b = [0u8; 8];
    if big_endian {
        for (dst, src) in b.iter_mut().zip(c.iter().rev()) {
            *dst = *src;
        }
    } else {
        b[..c.len()].copy_from_slice(c);
    }
    u64::from_le_bytes(b)
}

fn push_value(out: &mut Vec<u8>, value: u64, size: usize, big_endian: bool) {
    let le = value.to_le_bytes();
    if big_endian {
        out.extend(le[..size].iter().rev());
    } else {
        out.extend_from_slice(&le[..size]);
    }
}

/// Parsed client data.
#[derive(Debug, Clone, Copy)]
struct Params {
    scale_type: u32,
    factor: i32,
    nelem: usize,
    float: bool,
    size: usize,
    signed: bool,
    big_endian: bool,
    /// Fill value bytes (little-endian, zero-extended to 8 bytes).
    fill: Option<[u8; 8]>,
}

impl Params {
    fn parse(cd: &[u32]) -> Result<Params> {
        let bad = |m: &str| Error::invalid(format!("scaleoffset: {m}"));
        if cd.len() < 8 {
            return Err(bad("client data is too short"));
        }
        let float = match cd[3] {
            0 => false,
            1 => true,
            _ => return Err(bad("unknown datatype class")),
        };
        let size = cd[4] as usize;
        let size_ok = if float {
            matches!(size, 4 | 8)
        } else {
            matches!(size, 1 | 2 | 4 | 8)
        };
        if !size_ok {
            return Err(Error::unsupported(format!(
                "scaleoffset: datatype size {size}"
            )));
        }
        let fill = if cd[7] != 0 {
            let words = cd
                .get(8..8 + size.div_ceil(4))
                .ok_or_else(|| bad("fill value missing from client data"))?;
            let mut f = [0u8; 8];
            for (k, b) in f.iter_mut().enumerate().take(size) {
                *b = (words[k / 4] >> (8 * (k % 4))) as u8;
            }
            Some(f)
        } else {
            None
        };
        let p = Params {
            scale_type: cd[0],
            factor: cd[1] as i32,
            nelem: cd[2] as usize,
            float,
            size,
            signed: cd[5] != 0,
            big_endian: match cd[6] {
                0 => false,
                1 => true,
                _ => return Err(bad("unknown byte order")),
            },
            fill,
        };
        match (p.scale_type, p.float) {
            (SCALE_INT, false) => {}
            (SCALE_FLOAT_DSCALE, true) => {
                if !(-300..=300).contains(&p.factor) {
                    return Err(bad("scale factor out of range"));
                }
            }
            (SCALE_FLOAT_ESCALE, true) => {
                return Err(Error::unsupported("scaleoffset: E-scaling"));
            }
            _ => return Err(bad("scale type does not match the datatype class")),
        }
        Ok(p)
    }

    /// Whether the chunk is stored unchanged (integer scale factor equal to
    /// the full bit size).
    fn passthrough(&self) -> Result<bool> {
        if self.float || self.factor <= 0 {
            return Ok(false);
        }
        let full = self.size as i32 * 8;
        match self.factor.cmp(&full) {
            std::cmp::Ordering::Less => Ok(false),
            std::cmp::Ordering::Equal => Ok(true),
            std::cmp::Ordering::Greater => {
                Err(Error::invalid("scaleoffset: minbits exceeds the type size"))
            }
        }
    }
}

/// Computes the client data the scale-offset filter needs (what the
/// reference library's "set local" step stores).
///
/// `scale_type` is [`SCALE_INT`] (with `scale_factor` = minbits, 0 for
/// automatic) or [`SCALE_FLOAT_DSCALE`] (with `scale_factor` = D).
/// `fill` is the dataset's fill value in the file's byte order, if one is
/// defined.
pub fn client_data(
    dtype: &Datatype,
    chunk_elements: u32,
    scale_type: u32,
    scale_factor: i32,
    fill: Option<&[u8]>,
) -> Result<Vec<u32>> {
    let (class, signed, order) = match &dtype.class {
        DatatypeClass::FixedPoint(f) => (0, f.signed, f.byte_order),
        DatatypeClass::FloatingPoint(f) => (1, false, f.byte_order),
        _ => {
            return Err(Error::unsupported(
                "scaleoffset: only integer and floating-point data",
            ));
        }
    };
    let big_endian = match order {
        ByteOrder::LittleEndian => false,
        ByteOrder::BigEndian => true,
        ByteOrder::Vax => return Err(Error::unsupported("scaleoffset: VAX byte order")),
    };
    let mut cd = vec![
        scale_type,
        scale_factor as u32,
        chunk_elements,
        class,
        dtype.size,
        u32::from(signed),
        u32::from(big_endian),
        u32::from(fill.is_some()),
    ];
    cd.extend_from_slice(&[0; 12]);
    if let Some(fill) = fill {
        if fill.len() != dtype.size as usize || fill.len() > 8 {
            return Err(Error::invalid("scaleoffset: fill value size mismatch"));
        }
        let n = fill.len();
        for k in 0..n {
            let b = if big_endian { fill[n - 1 - k] } else { fill[k] };
            cd[8 + k / 4] |= u32::from(b) << (8 * (k % 4));
        }
    }
    // Validate the combination.
    Params::parse(&cd)?;
    Ok(cd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn dec(cd: &[u32], input: &[u8]) -> Result<Vec<u8>> {
        ScaleOffset.decode(cd, input.to_vec(), &FilterContext::default())
    }

    fn enc(cd: &[u32], input: &[u8]) -> Vec<u8> {
        ScaleOffset
            .encode(cd, input.to_vec(), &FilterContext::default())
            .unwrap()
    }

    /// Client data for an integer chunk.
    fn icd(n: u32, size: u32, signed: bool, be: bool, minbits: u32, fill: Option<u32>) -> Vec<u32> {
        let mut cd = vec![2, minbits, n, 0, size, signed as u32, be as u32, 1];
        cd.extend_from_slice(&[0; 12]);
        cd[8] = fill.unwrap_or(0);
        cd
    }

    fn fcd(n: u32, size: u32, be: bool, d: u32, fill: Option<f64>) -> Vec<u32> {
        let mut cd = vec![0, d, n, 1, size, 0, be as u32, 1];
        cd.extend_from_slice(&[0; 12]);
        if let Some(f) = fill {
            let b = if size == 4 {
                u64::from((f as f32).to_bits())
            } else {
                f.to_bits()
            };
            cd[8] = b as u32;
            cd[9] = (b >> 32) as u32;
        }
        cd
    }

    fn ints<const N: usize>(vals: &[i64], be: bool) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| {
                let le = v.to_le_bytes();
                let mut b = le[..N].to_vec();
                if be {
                    b.reverse();
                }
                b
            })
            .collect()
    }

    /// Decode must reproduce `plain` from the reference encoding, and our
    /// encoder must produce the same bytes as the reference library.
    fn check(cd: &[u32], plain: &[u8], packed: &str) {
        let packed = hex(packed);
        assert_eq!(dec(cd, &packed).unwrap(), plain, "decode {cd:?}");
        assert_eq!(enc(cd, plain), packed, "encode {cd:?}");
    }

    // Expected encodings below were produced by the reference library.

    #[test]
    fn integers() {
        check(
            &icd(5, 4, true, false, 0, None),
            &ints::<4>(&[100, 101, 105, 110, 100], false),
            "040000000864000000000000000000000000000000015a00",
        );
        check(
            &icd(5, 4, true, false, 0, None),
            &ints::<4>(&[-100, 101, -105, 110, 100], false),
            "080000000897ffffffffffffff000000000000000005ce00d7cd00",
        );
        check(
            &icd(5, 4, true, true, 0, None),
            &ints::<4>(&[100, 101, 105, 110, 100], true),
            "040000000864000000000000000000000000000000015a00",
        );
        check(
            &icd(4, 1, false, false, 0, None),
            &ints::<1>(&[200, 201, 255, 250], false),
            "0600000008c8000000000000000000000000000000001df200",
        );
        check(
            &icd(3, 2, false, true, 0, None),
            &ints::<2>(&[1000, 2000, 3000], true),
            "0b00000008e8030000000000000000000000000000000fa3e800",
        );
        check(
            &icd(3, 8, true, false, 0, None),
            &ints::<8>(&[-(1 << 40), 1 << 40, 5], false),
            "2a000000080000000000ffffff000000000000000000000000002000000000040000000014",
        );
        // Constant data (fill defined): minbits 1.
        check(
            &icd(4, 4, true, false, 0, None),
            &ints::<4>(&[7, 7, 7, 7], false),
            "01000000080700000000000000000000000000000000",
        );
        // Range needs the full width: stored unchanged, little-endian even
        // for big-endian data.
        let raw = ints::<2>(&[-32768, 32767], false);
        check(
            &icd(2, 2, true, false, 0, None),
            &raw,
            "1000000008000000000000000000000000000000000080ff7f",
        );
        check(
            &icd(2, 2, true, true, 0, None),
            &ints::<2>(&[-32768, 32767], true),
            "1000000008000000000000000000000000000000000080ff7f",
        );
        // Full width but below the all-ones limit: the minimum is kept.
        check(
            &icd(3, 1, false, false, 0, None),
            &ints::<1>(&[0, 255, 5], false),
            "08000000080500000000000000000000000000000000ff05",
        );
    }

    #[test]
    fn integer_fill_values() {
        // Elements equal to the fill value use the all-ones code.
        check(
            &icd(4, 4, true, false, 0, Some(0xffff_ffff)),
            &ints::<4>(&[5, 6, -1, 8], false),
            "03000000080500000000000000000000000000000007b0",
        );
        check(
            &icd(4, 2, true, true, 0, Some(99)),
            &ints::<2>(&[5, 6, 99, 8], true),
            "03000000080500000000000000000000000000000007b0",
        );
        check(
            &icd(4, 1, true, false, 0, Some(0xff)),
            &ints::<1>(&[5, 6, -1, 8], false),
            "03000000080500000000000000000000000000000007b0",
        );
        check(
            &icd(4, 2, false, false, 0, Some(99)),
            &ints::<2>(&[5, 99, 5, 5], false),
            "01000000080500000000000000000000000000000040",
        );
        check(
            &icd(2, 2, false, false, 0, Some(99)),
            &ints::<2>(&[99, 99], false),
            "010000000800000000000000000000000000000000c0",
        );
        // Unsigned values near the top of the range.
        check(
            &icd(2, 4, false, false, 0, None),
            &ints::<4>(&[0, 0xffff_fffe], false),
            "0100000008feffffff00000000000000000000000080",
        );
    }

    #[test]
    fn integer_explicit_minbits() {
        // minbits 3: out-of-range values wrap (as the reference does).
        let cd = icd(4, 4, true, false, 3, None);
        let packed = hex("030000000801000000000000000000000000000000e300");
        assert_eq!(dec(&cd, &packed).unwrap(), ints::<4>(&[0, 1, 7, 1], false));
        assert_eq!(enc(&cd, &ints::<4>(&[0, 1, 7, 9], false)), packed);
        // minbits == full size: no header at all.
        let cd = icd(3, 4, true, false, 32, None);
        let raw = ints::<4>(&[1, 2, 3], false);
        check(&cd, &raw, "010000000200000003000000");
        // minbits > full size is invalid.
        assert!(dec(&icd(3, 4, true, false, 40, None), &raw).is_err());
    }

    #[test]
    fn floats() {
        check(
            &fcd(4, 8, false, 2, Some(0.0)),
            &[1.5f64, -2.25, 3.13, 10.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            "0b0000000800000000000002c000000000000000002ee0010d4c90",
        );
        let f32s = |be: bool| -> Vec<u8> {
            [1.5f32, -2.25, 3.13, 10.0]
                .iter()
                .flat_map(|v| if be { v.to_be_bytes() } else { v.to_le_bytes() })
                .collect()
        };
        check(
            &fcd(4, 4, false, 2, Some(0.0)),
            &f32s(false),
            "0b00000008000010c00000000000000000000000002ee0010d4c90",
        );
        check(
            &fcd(4, 4, true, 2, Some(0.0)),
            &f32s(true),
            "0b00000008000010c00000000000000000000000002ee0010d4c90",
        );
        // Fill value 3.125 is excluded from the range.
        let packed = hex("070000000800000000000002c000000000000000004c03ffb0");
        let cd = fcd(4, 8, false, 1, Some(3.125));
        let got: Vec<f64> = dec(&cd, &packed)
            .unwrap()
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, [1.5499999999999998, -2.25, 3.125, 10.05]);
        let input: Vec<u8> = [1.5f64, -2.25, 3.125, 10.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(enc(&cd, &input), packed);
        // Near-fill values (within 10^-D) are replaced by the fill value.
        let cd = fcd(5, 8, false, 0, Some(0.0));
        let packed = hex("0200000008000000000000f83f0000000000000000f1c0");
        let input: Vec<u8> = [0.4f64, 0.6, 1.5, 2.5, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(enc(&cd, &input), packed);
        let got: Vec<f64> = dec(&cd, &packed)
            .unwrap()
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, [0.0, 0.0, 1.5, 2.5, 0.0]);
        // Range too large for the type: stored unchanged.
        let input: Vec<u8> = [1e300f64, -1e300]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let packed =
            hex("4000000008000000000000000000000000000000009c7500883ce4377e9c7500883ce437fe");
        assert_eq!(enc(&fcd(2, 8, false, 2, Some(0.0)), &input), packed);
        assert_eq!(
            dec(&fcd(2, 8, false, 2, Some(0.0)), &packed).unwrap(),
            input
        );
    }

    #[test]
    fn malformed() {
        let cd = icd(5, 4, true, false, 0, None);
        assert!(dec(&cd, &[0; 20]).is_err()); // short header
        assert!(dec(&cd, &hex("040000000864000000000000000000000000000000")).is_err()); // no data
        assert!(dec(&cd, &hex("210000000864000000000000000000000000000000")).is_err()); // minbits > 32
        assert!(
            dec(
                &cd,
                &hex("040000000964000000000000000000000000000000015a00")
            )
            .is_err()
        );
        assert!(dec(&cd[..7], &[0; 30]).is_err());
        let mut bad = cd.clone();
        bad[4] = 3;
        assert!(dec(&bad, &[0; 30]).is_err());
        bad = cd.clone();
        bad[0] = 0; // float scale on integer data
        assert!(dec(&bad, &[0; 30]).is_err());
        let escale = {
            let mut c = fcd(4, 8, false, 2, None);
            c[0] = 1;
            c
        };
        assert!(matches!(dec(&escale, &[0; 30]), Err(Error::Unsupported(_))));
        let ctx = FilterContext::default().with_max_output(10);
        assert!(matches!(
            ScaleOffset.decode(&cd, vec![0; 30], &ctx),
            Err(Error::LimitExceeded(_))
        ));
    }

    #[test]
    fn random_round_trips() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for size in [1usize, 2, 4, 8] {
            for signed in [false, true] {
                for be in [false, true] {
                    for _ in 0..20 {
                        let n = 1 + (next() % 50) as usize;
                        let range = next() % 64;
                        let vals: Vec<u64> = (0..n).map(|_| next() >> range).collect();
                        let fill = (next() % 2 == 0).then(|| vals[0] as u32);
                        let mut cd = icd(n as u32, size as u32, signed, be, 0, fill);
                        if size < 4 {
                            cd[8] &= (1u32 << (8 * size)) - 1;
                        }
                        let plain: Vec<u8> = vals
                            .iter()
                            .flat_map(|v| {
                                let mut b = v.to_le_bytes()[..size].to_vec();
                                if be {
                                    b.reverse();
                                }
                                b
                            })
                            .collect();
                        let e = enc(&cd, &plain);
                        assert_eq!(dec(&cd, &e).unwrap(), plain);
                    }
                }
            }
        }
    }

    #[test]
    fn client_data_from_datatype() {
        use crate::format::messages::datatype::FixedPoint;
        let dt = Datatype {
            version: 1,
            size: 2,
            class: DatatypeClass::FixedPoint(FixedPoint {
                byte_order: ByteOrder::BigEndian,
                lo_pad: false,
                hi_pad: false,
                signed: true,
                bit_offset: 0,
                bit_precision: 16,
            }),
        };
        let cd = client_data(&dt, 4, SCALE_INT, 0, Some(&[0, 99])).unwrap();
        assert_eq!(cd[..9], [2, 0, 4, 0, 2, 1, 1, 1, 99]);
        assert_eq!(cd.len(), 20);
        assert!(client_data(&dt, 4, SCALE_FLOAT_DSCALE, 2, None).is_err());
    }
}
