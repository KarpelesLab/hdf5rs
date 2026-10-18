//! Bit-level helpers shared by the typed and dynamic decoders: byte-order
//! normalisation, fixed-point extraction with bit offset / precision and a
//! generic floating-point decoder driven by the datatype's field layout.

use crate::error::{Error, Result};
use crate::format::messages::datatype::{
    ByteOrder, FixedPoint, FloatingPoint, MantissaNormalization,
};

/// Widest atomic value (in bytes) handled by the generic paths.
pub(crate) const MAX_ATOMIC_SIZE: usize = 16;

/// Assemble up to 16 bytes stored in `order` into an unsigned integer.
pub(crate) fn uint_from_bytes(bytes: &[u8], order: ByteOrder) -> Result<u128> {
    if bytes.len() > MAX_ATOMIC_SIZE {
        return Err(Error::unsupported(format!(
            "{}-byte atomic values (at most {MAX_ATOMIC_SIZE} supported)",
            bytes.len()
        )));
    }
    let mut v: u128 = 0;
    match order {
        ByteOrder::LittleEndian => {
            for &b in bytes.iter().rev() {
                v = (v << 8) | u128::from(b);
            }
        }
        ByteOrder::BigEndian => {
            for &b in bytes {
                v = (v << 8) | u128::from(b);
            }
        }
        ByteOrder::Vax => return Err(Error::unsupported("VAX byte order")),
    }
    Ok(v)
}

/// A mask of the low `bits` bits (`bits` ≤ 128).
pub(crate) fn mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// A decoded fixed-point value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Int {
    /// Value of a signed (two's complement) type.
    Signed(i128),
    /// Value of an unsigned type.
    Unsigned(u128),
}

impl Int {
    /// Whether the value is zero.
    pub(crate) fn is_zero(self) -> bool {
        matches!(self, Int::Signed(0) | Int::Unsigned(0))
    }

    /// The value as `i128`, if representable.
    pub(crate) fn to_i128(self) -> Option<i128> {
        match self {
            Int::Signed(v) => Some(v),
            Int::Unsigned(v) => i128::try_from(v).ok(),
        }
    }
}

/// Check that a fixed-point description is self-consistent for `size`
/// bytes and supported by the generic decoder.
pub(crate) fn validate_fixed(fp: &FixedPoint, size: u32) -> Result<()> {
    if size as usize > MAX_ATOMIC_SIZE {
        return Err(Error::unsupported(format!("{size}-byte integers")));
    }
    if fp.byte_order == ByteOrder::Vax {
        return Err(Error::unsupported("VAX byte order for integers"));
    }
    let end = u32::from(fp.bit_offset) + u32::from(fp.bit_precision);
    if fp.bit_precision == 0 || end > size * 8 {
        return Err(Error::invalid(format!(
            "integer bit offset {} / precision {} do not fit {size} bytes",
            fp.bit_offset, fp.bit_precision
        )));
    }
    Ok(())
}

/// Decode one fixed-point value. `bytes` must be exactly `size` bytes and
/// the description must have passed [`validate_fixed`].
pub(crate) fn decode_fixed(fp: &FixedPoint, bytes: &[u8]) -> Result<Int> {
    let raw = uint_from_bytes(bytes, fp.byte_order)?;
    let prec = u32::from(fp.bit_precision);
    let v = raw.checked_shr(u32::from(fp.bit_offset)).unwrap_or(0) & mask(prec);
    if fp.signed {
        // Sign-extend from bit `prec - 1`.
        let shift = 128u32.saturating_sub(prec);
        Ok(Int::Signed(((v << shift) as i128) >> shift))
    } else {
        Ok(Int::Unsigned(v))
    }
}

/// Check a floating-point description for the generic decoder.
pub(crate) fn validate_float(fp: &FloatingPoint, size: u32) -> Result<()> {
    if size as usize > MAX_ATOMIC_SIZE {
        return Err(Error::unsupported(format!("{size}-byte floating point")));
    }
    if fp.byte_order == ByteOrder::Vax {
        return Err(Error::unsupported("VAX floating-point byte order"));
    }
    let bits = size * 8;
    let fits = |loc: u32, len: u32| loc + len <= bits;
    if u32::from(fp.sign_location) >= bits
        || !fits(u32::from(fp.exponent_location), u32::from(fp.exponent_size))
        || !fits(u32::from(fp.mantissa_location), u32::from(fp.mantissa_size))
        || fp.exponent_size == 0
        || fp.exponent_size > 32
        || fp.mantissa_size > 120
    {
        return Err(Error::invalid("inconsistent floating-point field layout"));
    }
    Ok(())
}

/// `x * 2^e` without intermediate overflow of the power of two.
fn ldexp(mut x: f64, e: i64) -> f64 {
    // Anything beyond this range saturates to 0 / infinity anyway.
    let mut e = e.clamp(-20_000, 20_000) as i32;
    while e > 1000 && x.is_finite() && x != 0.0 {
        x *= 2f64.powi(1000);
        e -= 1000;
    }
    while e < -1000 && x != 0.0 {
        x *= 2f64.powi(-1000);
        e += 1000;
    }
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    x * 2f64.powi(e)
}

/// Decode one floating-point value of any layout into the nearest `f64`.
/// `bytes` must be exactly the element size; the description must have
/// passed [`validate_float`].
pub(crate) fn decode_float(fp: &FloatingPoint, bytes: &[u8]) -> Result<f64> {
    let raw = uint_from_bytes(bytes, fp.byte_order)?;
    let field =
        |loc: u8, len: u8| raw.checked_shr(u32::from(loc)).unwrap_or(0) & mask(u32::from(len));
    let negative = (raw >> fp.sign_location) & 1 == 1;
    let msize = u32::from(fp.mantissa_size);
    let exp = field(fp.exponent_location, fp.exponent_size) as i64;
    let emax = mask(u32::from(fp.exponent_size)) as i64;
    let mant = field(fp.mantissa_location, fp.mantissa_size);
    let bias = i64::from(fp.exponent_bias);
    let msize_i = i64::from(msize);
    let magnitude = match fp.normalization {
        MantissaNormalization::Implied => {
            if exp == emax {
                if mant == 0 { f64::INFINITY } else { f64::NAN }
            } else if exp == 0 {
                ldexp(mant as f64, 1 - bias - msize_i)
            } else {
                ldexp((mant | (1u128 << msize)) as f64, exp - bias - msize_i)
            }
        }
        MantissaNormalization::MsbSet => {
            // Explicit leading bit (e.g. x87 extended precision).
            let fraction = mant & mask(msize.saturating_sub(1));
            if exp == emax {
                if fraction == 0 {
                    f64::INFINITY
                } else {
                    f64::NAN
                }
            } else {
                ldexp(mant as f64, exp.max(1) - bias - (msize_i - 1))
            }
        }
        MantissaNormalization::None => ldexp(mant as f64, exp - bias - (msize_i - 1)),
    };
    Ok(if negative { -magnitude } else { magnitude })
}

/// Convert IEEE binary16 bits to `f32` (exact).
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match exp {
        0 => {
            // Zero or subnormal: man * 2^-24, exactly representable.
            let v = man as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -v } else { v };
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(order: ByteOrder, signed: bool, off: u16, prec: u16) -> FixedPoint {
        FixedPoint {
            byte_order: order,
            lo_pad: false,
            hi_pad: false,
            signed,
            bit_offset: off,
            bit_precision: prec,
        }
    }

    #[test]
    fn uint_orders() {
        assert_eq!(
            uint_from_bytes(&[1, 2, 3], ByteOrder::LittleEndian).unwrap(),
            0x030201
        );
        assert_eq!(
            uint_from_bytes(&[1, 2, 3], ByteOrder::BigEndian).unwrap(),
            0x010203
        );
        assert!(uint_from_bytes(&[0; 17], ByteOrder::LittleEndian).is_err());
        assert!(uint_from_bytes(&[0; 2], ByteOrder::Vax).is_err());
    }

    #[test]
    fn fixed_bits() {
        // 12-bit signed value at offset 4 in a 2-byte LE word: -1.
        let fp = fixed(ByteOrder::LittleEndian, true, 4, 12);
        validate_fixed(&fp, 2).unwrap();
        assert_eq!(decode_fixed(&fp, &[0xf0, 0xff]).unwrap(), Int::Signed(-1));
        assert_eq!(decode_fixed(&fp, &[0x7f, 0x00]).unwrap(), Int::Signed(7));
        let fp = fixed(ByteOrder::BigEndian, false, 0, 128);
        assert_eq!(
            decode_fixed(&fp, &[0xff; 16]).unwrap(),
            Int::Unsigned(u128::MAX)
        );
        let fp = fixed(ByteOrder::LittleEndian, true, 0, 128);
        assert_eq!(decode_fixed(&fp, &[0xff; 16]).unwrap(), Int::Signed(-1));
        assert!(validate_fixed(&fixed(ByteOrder::LittleEndian, true, 4, 13), 2).is_err());
        assert!(validate_fixed(&fixed(ByteOrder::LittleEndian, true, 0, 0), 2).is_err());
        assert!(validate_fixed(&fixed(ByteOrder::LittleEndian, true, 0, 8), 17).is_err());
    }

    #[test]
    fn half() {
        assert_eq!(f16_to_f32(0x3e00), 1.5);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
    }

    #[test]
    fn generic_float_matches_ieee() {
        let f64le = FloatingPoint {
            byte_order: ByteOrder::LittleEndian,
            lo_pad: false,
            hi_pad: false,
            internal_pad: false,
            normalization: MantissaNormalization::Implied,
            sign_location: 63,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        };
        validate_float(&f64le, 8).unwrap();
        for v in [
            0.0,
            -0.0,
            1.0,
            -2.5,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
            1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let got = decode_float(&f64le, &v.to_le_bytes()).unwrap();
            assert_eq!(got.to_bits(), v.to_bits(), "{v}");
        }
        assert!(
            decode_float(&f64le, &f64::NAN.to_le_bytes())
                .unwrap()
                .is_nan()
        );

        // x87 80-bit extended stored in 16 bytes.
        let x87 = FloatingPoint {
            normalization: MantissaNormalization::MsbSet,
            sign_location: 79,
            bit_precision: 80,
            exponent_location: 64,
            exponent_size: 15,
            mantissa_size: 64,
            exponent_bias: 16383,
            ..f64le
        };
        validate_float(&x87, 16).unwrap();
        let enc =
            |sign: u128, exp: u128, mant: u128| ((sign << 79) | (exp << 64) | mant).to_le_bytes();
        assert_eq!(decode_float(&x87, &enc(0, 16383, 1 << 63)).unwrap(), 1.0);
        assert_eq!(decode_float(&x87, &enc(1, 16384, 3 << 62)).unwrap(), -3.0);
        assert_eq!(
            decode_float(&x87, &enc(0, 0x7fff, 1 << 63)).unwrap(),
            f64::INFINITY
        );
        assert_eq!(decode_float(&x87, &enc(0, 0, 0)).unwrap(), 0.0);
        // Out of f64 range saturates.
        assert_eq!(
            decode_float(&x87, &enc(0, 0x7ffe, 1 << 63)).unwrap(),
            f64::INFINITY
        );
        assert_eq!(decode_float(&x87, &enc(0, 1, 1 << 63)).unwrap(), 0.0);

        let bad = FloatingPoint {
            exponent_location: 60,
            ..f64le
        };
        assert!(validate_float(&bad, 8).is_err());
    }
}
