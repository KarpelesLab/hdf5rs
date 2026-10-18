//! Statically typed decoding: the [`H5Type`] trait and its implementations.

use std::borrow::Cow;

use super::DecodeContext;
use super::bits::{self, Int};
use super::reference::{Reference, decode_reference};
use super::value::{Value, decode_value};
use super::vlen::fetch_vlen;
use crate::error::{Error, Result};
use crate::format::messages::datatype::{
    ByteOrder, ComplexForm, Datatype, DatatypeClass, FixedPoint, FloatingPoint, StringPadding,
    VariableLengthKind,
};

/// A Rust type that elements of (some) HDF5 datatypes decode into.
///
/// Conversion rules of the provided implementations:
///
/// - **Integers** (`u8` … `i64`, `usize`/`isize` excluded): from any
///   fixed-point (or enumeration of fixed-point) type of either byte order,
///   any size up to 16 bytes and any bit offset/precision, as long as every
///   value of the source type fits the target losslessly (e.g. `i64` reads
///   `i8`/`i16`/`i32`/`u32`; `u8` does not read `i8`). Use [`Value`] for
///   other conversions.
/// - **Floats**: `f64` reads IEEE half/single/double and other float
///   layouts whose exponent and mantissa fit binary64; `f32` reads half and
///   single precision. Either byte order; VAX order is unsupported.
/// - **`bool`**: from integers and enumerations (h5py stores `bool` as an
///   `int8` enumeration `FALSE = 0, TRUE = 1`); zero is `false`, anything
///   else `true`.
/// - **`String`**: fixed-length and variable-length strings. Padding is
///   removed (null-terminated: up to the first NUL; null-padded: trailing
///   NULs; space-padded: trailing spaces). Invalid UTF-8 is an error (also
///   for ASCII strings, which are accepted when they are valid UTF-8); use
///   [`StringBytes`] for the raw bytes.
/// - **`Vec<T>`**: variable-length sequences and array datatypes
///   (flattened in row-major order).
/// - **`[T; N]`**: array datatypes with `N` elements in total.
/// - **[`Complex<T>`]**: the complex class (rectangular form) and two-member
///   compounds such as h5py's `{r, i}`.
/// - **[`Reference`]** and **[`Value`]**.
///
/// Implement this trait for your own structs to read compound data; see
/// [`decode_member`].
pub trait H5Type: Sized {
    /// Check whether elements of `dt` can be decoded into `Self`.
    fn check(dt: &Datatype) -> Result<()>;

    /// Decode one element; `bytes` must be exactly `dt.size` bytes.
    fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self>;

    /// Decode a packed buffer of elements, appending them to `out`.
    ///
    /// `bytes.len()` must be a multiple of `dt.size`. Numeric
    /// implementations override this with a loop free of per-element
    /// dispatch.
    fn decode_slice(
        dt: &Datatype,
        bytes: &[u8],
        ctx: &DecodeContext<'_>,
        out: &mut Vec<Self>,
    ) -> Result<()> {
        Self::check(dt)?;
        let size = element_count_check(dt, bytes)?;
        out.reserve(bytes.len() / size);
        for chunk in bytes.chunks_exact(size) {
            out.push(Self::decode(dt, chunk, ctx)?);
        }
        Ok(())
    }

    /// Whether elements of `dt` have exactly the in-memory representation
    /// of `Self` on this machine (so a buffer could be reinterpreted
    /// without conversion).
    fn is_native(dt: &Datatype) -> bool {
        let _ = dt;
        false
    }
}

/// Decode a packed buffer of elements into a new vector.
pub fn decode_all<T: H5Type>(
    dt: &Datatype,
    bytes: &[u8],
    ctx: &DecodeContext<'_>,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    T::decode_slice(dt, bytes, ctx, &mut out)?;
    Ok(out)
}

/// Whether elements of `dt` are bit-for-bit native values of `T`.
pub fn is_native<T: H5Type>(dt: &Datatype) -> bool {
    T::is_native(dt)
}

/// Validate a buffer length against the element size; returns the size.
fn element_count_check(dt: &Datatype, bytes: &[u8]) -> Result<usize> {
    let size = dt.size as usize;
    if size == 0 {
        return Err(Error::invalid("datatype of size zero"));
    }
    if !bytes.len().is_multiple_of(size) {
        return Err(Error::invalid(format!(
            "buffer of {} bytes is not a multiple of the {size}-byte element size",
            bytes.len()
        )));
    }
    Ok(size)
}

fn check_len(dt: &Datatype, bytes: &[u8]) -> Result<()> {
    if bytes.len() != dt.size as usize {
        return Err(Error::invalid(format!(
            "element of {} bytes for a {}-byte datatype",
            bytes.len(),
            dt.size
        )));
    }
    Ok(())
}

fn mismatch<T>(dt: &Datatype, target: &str) -> Result<T> {
    Err(Error::invalid(format!("cannot read {dt} as {target}")))
}

// ---------------------------------------------------------------- integers

/// How to decode integers of a given datatype.
enum IntPlan<'a> {
    /// Same size and signedness, full precision: plain byte swap / copy.
    Exact(ByteOrder),
    /// Whole-byte (1, 2, 4 or 8 bytes) integer, losslessly widened.
    Widen {
        fp: &'a FixedPoint,
        size: u32,
        signed: bool,
        big_endian: bool,
    },
    /// Bit extraction through [`bits::decode_fixed`].
    Generic(&'a FixedPoint),
}

/// The fixed-point description behind an integer or enumeration type.
fn fixed_source(dt: &Datatype) -> Option<(&FixedPoint, u32)> {
    match &dt.class {
        DatatypeClass::FixedPoint(fp) => Some((fp, dt.size)),
        DatatypeClass::Enumeration { base, .. } if base.size == dt.size => match &base.class {
            DatatypeClass::FixedPoint(fp) => Some((fp, base.size)),
            _ => None,
        },
        _ => None,
    }
}

fn int_plan<'a>(dt: &'a Datatype, bits: u32, signed: bool, target: &str) -> Result<IntPlan<'a>> {
    let Some((fp, size)) = fixed_source(dt) else {
        return mismatch(dt, target);
    };
    bits::validate_fixed(fp, size)?;
    let p = u32::from(fp.bit_precision);
    let lossless = match (fp.signed, signed) {
        (true, true) | (false, false) => p <= bits,
        (false, true) => p < bits,
        (true, false) => false,
    };
    if !lossless {
        return mismatch(dt, target);
    }
    if size * 8 == bits && fp.bit_offset == 0 && p == bits && fp.signed == signed {
        Ok(IntPlan::Exact(fp.byte_order))
    } else if fp.bit_offset == 0
        && p == size * 8
        && matches!(size, 1 | 2 | 4 | 8)
        && fp.byte_order != ByteOrder::Vax
    {
        Ok(IntPlan::Widen {
            fp,
            size,
            signed: fp.signed,
            big_endian: fp.byte_order == ByteOrder::BigEndian,
        })
    } else {
        Ok(IntPlan::Generic(fp))
    }
}

macro_rules! impl_int {
    ($($t:ty),*) => {$(
        impl H5Type for $t {
            fn check(dt: &Datatype) -> Result<()> {
                int_plan(dt, <$t>::BITS, <$t>::MIN != 0, stringify!($t)).map(|_| ())
            }

            fn decode(dt: &Datatype, bytes: &[u8], _ctx: &DecodeContext<'_>) -> Result<Self> {
                let plan = int_plan(dt, <$t>::BITS, <$t>::MIN != 0, stringify!($t))?;
                check_len(dt, bytes)?;
                match plan {
                    IntPlan::Exact(order) => {
                        let arr = bytes
                            .try_into()
                            .map_err(|_| Error::invalid("element size mismatch"))?;
                        Ok(match order {
                            ByteOrder::BigEndian => <$t>::from_be_bytes(arr),
                            _ => <$t>::from_le_bytes(arr),
                        })
                    }
                    IntPlan::Widen { fp, .. } | IntPlan::Generic(fp) => {
                        int_from(bits::decode_fixed(fp, bytes)?)
                    }
                }
            }

            fn decode_slice(
                dt: &Datatype,
                bytes: &[u8],
                _ctx: &DecodeContext<'_>,
                out: &mut Vec<Self>,
            ) -> Result<()> {
                let plan = int_plan(dt, <$t>::BITS, <$t>::MIN != 0, stringify!($t))?;
                let size = element_count_check(dt, bytes)?;
                match plan {
                    IntPlan::Exact(order) => {
                        const N: usize = size_of::<$t>();
                        let (chunks, _) = bytes.as_chunks::<N>();
                        if order == ByteOrder::BigEndian {
                            out.extend(chunks.iter().map(|c| <$t>::from_be_bytes(*c)));
                        } else {
                            out.extend(chunks.iter().map(|c| <$t>::from_le_bytes(*c)));
                        }
                    }
                    IntPlan::Widen { size: 1, signed, .. } => {
                        // Lossless by `int_plan`: the casts below only
                        // sign- or zero-extend.
                        if signed {
                            out.extend(bytes.iter().map(|&b| b as i8 as i128 as $t));
                        } else {
                            out.extend(bytes.iter().map(|&b| b as i128 as $t));
                        }
                    }
                    IntPlan::Widen { size, signed, big_endian, .. } => {
                        macro_rules! widen {
                            ($s:ty) => {{
                                let (chunks, _) = bytes.as_chunks::<{ size_of::<$s>() }>();
                                if big_endian {
                                    out.extend(
                                        chunks.iter().map(|c| <$s>::from_be_bytes(*c) as i128 as $t),
                                    );
                                } else {
                                    out.extend(
                                        chunks.iter().map(|c| <$s>::from_le_bytes(*c) as i128 as $t),
                                    );
                                }
                            }};
                        }
                        match (size, signed) {
                            (2, false) => widen!(u16),
                            (2, true) => widen!(i16),
                            (4, false) => widen!(u32),
                            (4, true) => widen!(i32),
                            (8, false) => widen!(u64),
                            _ => widen!(i64),
                        }
                    }
                    IntPlan::Generic(fp) => {
                        out.reserve(bytes.len() / size);
                        for c in bytes.chunks_exact(size) {
                            out.push(int_from(bits::decode_fixed(fp, c)?)?);
                        }
                    }
                }
                Ok(())
            }

            fn is_native(dt: &Datatype) -> bool {
                matches!(dt.class, DatatypeClass::FixedPoint(_))
                    && dt.is_native_int(<$t>::BITS / 8, <$t>::MIN != 0)
            }
        }
    )*};
}

/// Convert an extracted value into the target integer type.
fn int_from<T: TryFrom<i128> + TryFrom<u128>>(v: Int) -> Result<T> {
    let r = match v {
        Int::Signed(s) => T::try_from(s).ok(),
        Int::Unsigned(u) => T::try_from(u).ok(),
    };
    r.ok_or_else(|| Error::invalid("integer value out of range"))
}

impl_int!(u8, i8, u16, i16, u32, i32, u64, i64);

// ------------------------------------------------------------------ floats

/// How to decode floats of a given datatype.
enum FloatPlan<'a> {
    /// IEEE half precision.
    Half(ByteOrder),
    /// IEEE single precision.
    Single(ByteOrder),
    /// IEEE double precision.
    Double(ByteOrder),
    /// Any other layout, through [`bits::decode_float`].
    Generic(&'a FloatingPoint),
}

fn float_plan<'a>(
    dt: &'a Datatype,
    max_ieee: u32,
    max_exp: u8,
    max_mant: u8,
    target: &str,
) -> Result<FloatPlan<'a>> {
    let DatatypeClass::FloatingPoint(fp) = &dt.class else {
        return mismatch(dt, target);
    };
    bits::validate_float(fp, dt.size)?;
    if fp.is_ieee(dt.size) && dt.size <= max_ieee {
        let o = fp.byte_order;
        return Ok(match dt.size {
            2 => FloatPlan::Half(o),
            4 => FloatPlan::Single(o),
            _ => FloatPlan::Double(o),
        });
    }
    if fp.exponent_size <= max_exp && fp.mantissa_size <= max_mant {
        Ok(FloatPlan::Generic(fp))
    } else {
        mismatch(dt, target)
    }
}

fn half_bits(c: [u8; 2], order: ByteOrder) -> u16 {
    if order == ByteOrder::BigEndian {
        u16::from_be_bytes(c)
    } else {
        u16::from_le_bytes(c)
    }
}

macro_rules! impl_float {
    ($t:ty, $max_ieee:expr, $max_exp:expr, $max_mant:expr, $native:ident) => {
        impl H5Type for $t {
            fn check(dt: &Datatype) -> Result<()> {
                float_plan(dt, $max_ieee, $max_exp, $max_mant, stringify!($t)).map(|_| ())
            }

            fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self> {
                check_len(dt, bytes)?;
                let mut out = Vec::with_capacity(1);
                Self::decode_slice(dt, bytes, ctx, &mut out)?;
                out.pop().ok_or(Error::UnexpectedEof)
            }

            #[allow(clippy::unnecessary_cast)]
            fn decode_slice(
                dt: &Datatype,
                bytes: &[u8],
                _ctx: &DecodeContext<'_>,
                out: &mut Vec<Self>,
            ) -> Result<()> {
                let plan = float_plan(dt, $max_ieee, $max_exp, $max_mant, stringify!($t))?;
                let size = element_count_check(dt, bytes)?;
                let be = |o: ByteOrder| o == ByteOrder::BigEndian;
                match plan {
                    FloatPlan::Half(o) => {
                        let (c, _) = bytes.as_chunks::<2>();
                        out.extend(c.iter().map(|c| bits::f16_to_f32(half_bits(*c, o)) as $t));
                    }
                    FloatPlan::Single(o) => {
                        let (c, _) = bytes.as_chunks::<4>();
                        if be(o) {
                            out.extend(c.iter().map(|c| f32::from_be_bytes(*c) as $t));
                        } else {
                            out.extend(c.iter().map(|c| f32::from_le_bytes(*c) as $t));
                        }
                    }
                    FloatPlan::Double(o) => {
                        let (c, _) = bytes.as_chunks::<8>();
                        if be(o) {
                            out.extend(c.iter().map(|c| f64::from_be_bytes(*c) as $t));
                        } else {
                            out.extend(c.iter().map(|c| f64::from_le_bytes(*c) as $t));
                        }
                    }
                    FloatPlan::Generic(fp) => {
                        out.reserve(bytes.len() / size);
                        for c in bytes.chunks_exact(size) {
                            out.push(bits::decode_float(fp, c)? as $t);
                        }
                    }
                }
                Ok(())
            }

            fn is_native(dt: &Datatype) -> bool {
                dt.$native()
            }
        }
    };
}

impl_float!(f32, 4, 8, 23, is_native_f32);
impl_float!(f64, 8, 11, 52, is_native_f64);

// -------------------------------------------------------------------- bool

impl H5Type for bool {
    fn check(dt: &Datatype) -> Result<()> {
        let Some((fp, size)) = fixed_source(dt) else {
            return mismatch(dt, "bool");
        };
        bits::validate_fixed(fp, size)
    }

    fn decode(dt: &Datatype, bytes: &[u8], _ctx: &DecodeContext<'_>) -> Result<Self> {
        Self::check(dt)?;
        check_len(dt, bytes)?;
        let Some((fp, _)) = fixed_source(dt) else {
            return mismatch(dt, "bool");
        };
        Ok(!bits::decode_fixed(fp, bytes)?.is_zero())
    }

    fn decode_slice(
        dt: &Datatype,
        bytes: &[u8],
        ctx: &DecodeContext<'_>,
        out: &mut Vec<Self>,
    ) -> Result<()> {
        Self::check(dt)?;
        let size = element_count_check(dt, bytes)?;
        match fixed_source(dt) {
            Some((fp, 1)) if fp.bit_offset == 0 && fp.bit_precision == 8 => {
                out.extend(bytes.iter().map(|&b| b != 0));
            }
            _ => {
                out.reserve(bytes.len() / size);
                for c in bytes.chunks_exact(size) {
                    out.push(Self::decode(dt, c, ctx)?);
                }
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------- strings

/// Remove string padding according to the padding convention.
pub(crate) fn strip_padding(bytes: &[u8], padding: StringPadding) -> &[u8] {
    match padding {
        StringPadding::NullTerminate => {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            &bytes[..end]
        }
        StringPadding::NullPad => {
            let end = bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
            &bytes[..end]
        }
        StringPadding::SpacePad => {
            let end = bytes.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
            &bytes[..end]
        }
    }
}

/// The bytes of a fixed or variable-length string element, padding
/// removed.
pub(crate) fn string_bytes(
    dt: &Datatype,
    bytes: &[u8],
    ctx: &DecodeContext<'_>,
) -> Result<Vec<u8>> {
    check_len(dt, bytes)?;
    match &dt.class {
        DatatypeClass::String(s) => Ok(strip_padding(bytes, s.padding).to_vec()),
        DatatypeClass::VariableLength {
            kind: VariableLengthKind::String { padding, .. },
            base,
        } => {
            let mut data = fetch_vlen(bytes, base.size, ctx)?;
            let len = strip_padding(&data, *padding).len();
            data.truncate(len);
            Ok(data)
        }
        _ => mismatch(dt, "string"),
    }
}

fn check_string(dt: &Datatype, target: &str) -> Result<()> {
    match &dt.class {
        DatatypeClass::String(_)
        | DatatypeClass::VariableLength {
            kind: VariableLengthKind::String { .. },
            ..
        } => Ok(()),
        _ => mismatch(dt, target),
    }
}

impl H5Type for String {
    fn check(dt: &Datatype) -> Result<()> {
        check_string(dt, "String")
    }

    fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self> {
        String::from_utf8(string_bytes(dt, bytes, ctx)?)
            .map_err(|_| Error::invalid("string is not valid UTF-8"))
    }
}

/// The raw bytes of a fixed or variable-length string (padding removed,
/// no character set validation).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct StringBytes(pub Vec<u8>);

impl StringBytes {
    /// Decode as UTF-8, replacing invalid sequences.
    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(&self.0).into_owned()
    }
}

impl H5Type for StringBytes {
    fn check(dt: &Datatype) -> Result<()> {
        check_string(dt, "StringBytes")
    }

    fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self> {
        string_bytes(dt, bytes, ctx).map(StringBytes)
    }
}

// -------------------------------------------------------- arrays and vlens

/// Element count and base type of an array datatype.
pub(crate) fn array_parts(dt: &Datatype) -> Result<(usize, &Datatype)> {
    let DatatypeClass::Array { dims, base, .. } = &dt.class else {
        return Err(Error::invalid("not an array datatype"));
    };
    let count = dims_product(dims)?;
    check_array_size(count, base.size, dt.size)?;
    Ok((count, base))
}

/// Product of array dimensions.
pub(crate) fn dims_product(dims: &[u32]) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or(Error::LimitExceeded("array element count"))
}

/// Check that `count` elements of `base_size` bytes fill `size` bytes.
pub(crate) fn check_array_size(count: usize, base_size: u32, size: u32) -> Result<()> {
    if base_size == 0 && count > 0 {
        return Err(Error::invalid("array of zero-sized elements"));
    }
    if count.checked_mul(base_size as usize) != Some(size as usize) {
        return Err(Error::invalid(format!(
            "array of {count} x {base_size} bytes in a {size}-byte datatype"
        )));
    }
    Ok(())
}

impl<T: H5Type> H5Type for Vec<T> {
    fn check(dt: &Datatype) -> Result<()> {
        match &dt.class {
            DatatypeClass::VariableLength {
                kind: VariableLengthKind::Sequence,
                base,
            } => T::check(base),
            DatatypeClass::Array { base, .. } => {
                array_parts(dt)?;
                T::check(base)
            }
            _ => mismatch(dt, "Vec"),
        }
    }

    fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self> {
        Self::check(dt)?;
        check_len(dt, bytes)?;
        match &dt.class {
            DatatypeClass::VariableLength { base, .. } => {
                let data = fetch_vlen(bytes, base.size, ctx)?;
                decode_all(base, &data, ctx)
            }
            DatatypeClass::Array { base, .. } => decode_all(base, bytes, ctx),
            _ => mismatch(dt, "Vec"),
        }
    }
}

impl<T: H5Type, const N: usize> H5Type for [T; N] {
    fn check(dt: &Datatype) -> Result<()> {
        let (count, base) = array_parts(dt).or_else(|_| mismatch(dt, "array"))?;
        if count != N {
            return Err(Error::invalid(format!(
                "array datatype has {count} elements, expected {N}"
            )));
        }
        T::check(base)
    }

    fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self> {
        Self::check(dt)?;
        check_len(dt, bytes)?;
        let (_, base) = array_parts(dt)?;
        let v: Vec<T> = decode_all(base, bytes, ctx)?;
        v.try_into()
            .map_err(|_| Error::invalid("array element count mismatch"))
    }

    fn is_native(dt: &Datatype) -> bool {
        match array_parts(dt) {
            Ok((count, base)) => count == N && T::is_native(base),
            Err(_) => false,
        }
    }
}

// ----------------------------------------------------------------- complex

/// A complex number with parts of type `T`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

/// Locate the two parts of a complex-like datatype: (type, offset) pairs.
fn complex_parts(dt: &Datatype) -> Result<[(&Datatype, usize); 2]> {
    match &dt.class {
        DatatypeClass::Complex {
            homogeneous: true,
            form: ComplexForm::Rectangular,
            base,
        } if base.size.checked_mul(2) == Some(dt.size) => {
            Ok([(base, 0), (base, base.size as usize)])
        }
        DatatypeClass::Compound { members }
            if members.len() == 2 && members.iter().all(|m| m.array_dims.is_none()) =>
        {
            let part = |i: usize| -> Result<(&Datatype, usize)> {
                let m = &members[i];
                let off = m.offset as usize;
                let end = off.checked_add(m.datatype.size as usize);
                if end.is_none_or(|e| e > dt.size as usize) {
                    return Err(Error::invalid("compound member exceeds its compound"));
                }
                Ok((&m.datatype, off))
            };
            Ok([part(0)?, part(1)?])
        }
        _ => mismatch(dt, "Complex"),
    }
}

impl<T: H5Type> H5Type for Complex<T> {
    fn check(dt: &Datatype) -> Result<()> {
        for (t, _) in complex_parts(dt)? {
            T::check(t)?;
        }
        Ok(())
    }

    fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self> {
        Self::check(dt)?;
        check_len(dt, bytes)?;
        let [(rt, ro), (it, io)] = complex_parts(dt)?;
        let part = |t: &Datatype, o: usize| {
            let b = bytes
                .get(o..o + t.size as usize)
                .ok_or(Error::UnexpectedEof)?;
            T::decode(t, b, ctx)
        };
        Ok(Complex {
            re: part(rt, ro)?,
            im: part(it, io)?,
        })
    }
}

// ----------------------------------------------------- references, values

impl H5Type for Reference {
    fn check(dt: &Datatype) -> Result<()> {
        match dt.class {
            DatatypeClass::Reference(_) => Ok(()),
            _ => mismatch(dt, "Reference"),
        }
    }

    fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self> {
        decode_reference(dt, bytes, ctx)
    }
}

impl H5Type for Value {
    fn check(_dt: &Datatype) -> Result<()> {
        Ok(())
    }

    fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self> {
        decode_value(dt, bytes, ctx)
    }
}

// --------------------------------------------------------- compound helpers

/// Find member `name` of a compound datatype: its datatype and byte
/// offset. Version 1 array members are returned as array datatypes.
pub fn compound_member<'d>(dt: &'d Datatype, name: &str) -> Result<(Cow<'d, Datatype>, usize)> {
    let DatatypeClass::Compound { members } = &dt.class else {
        return mismatch(dt, "compound");
    };
    let m = members
        .iter()
        .find(|m| m.name == name)
        .ok_or_else(|| Error::NotFound(format!("compound member {name:?}")))?;
    let t = member_type(m)?;
    let end = (m.offset as usize).checked_add(t.size as usize);
    if end.is_none_or(|e| e > dt.size as usize) {
        return Err(Error::invalid(format!(
            "compound member {name:?} exceeds its compound"
        )));
    }
    Ok((t, m.offset as usize))
}

/// The effective datatype of a compound member (version 1 array members
/// are wrapped in a synthesized array datatype).
pub(crate) fn member_type(
    m: &crate::format::messages::datatype::CompoundMember,
) -> Result<Cow<'_, Datatype>> {
    match &m.array_dims {
        None => Ok(Cow::Borrowed(&m.datatype)),
        Some(dims) => {
            let count = dims_product(dims)?;
            let size = count
                .checked_mul(m.datatype.size as usize)
                .and_then(|s| u32::try_from(s).ok())
                .ok_or(Error::LimitExceeded("compound array member size"))?;
            Ok(Cow::Owned(Datatype {
                version: 2,
                size,
                class: DatatypeClass::Array {
                    dims: dims.clone(),
                    permutation: None,
                    base: Box::new(m.datatype.clone()),
                },
            }))
        }
    }
}

/// Check that compound `dt` has a member `name` readable as `T`.
pub fn check_member<T: H5Type>(dt: &Datatype, name: &str) -> Result<()> {
    let (t, _) = compound_member(dt, name)?;
    T::check(&t)
}

/// Decode member `name` of the compound element `bytes` as `T`.
///
/// ```
/// use hdf5rs::Result;
/// use hdf5rs::format::messages::datatype::Datatype;
/// use hdf5rs::types::{DecodeContext, H5Type, check_member, decode_member};
///
/// struct Point { x: f64, id: u32 }
///
/// impl H5Type for Point {
///     fn check(dt: &Datatype) -> Result<()> {
///         check_member::<f64>(dt, "x")?;
///         check_member::<u32>(dt, "id")
///     }
///     fn decode(dt: &Datatype, bytes: &[u8], ctx: &DecodeContext<'_>) -> Result<Self> {
///         Ok(Point {
///             x: decode_member(dt, bytes, ctx, "x")?,
///             id: decode_member(dt, bytes, ctx, "id")?,
///         })
///     }
/// }
/// ```
pub fn decode_member<T: H5Type>(
    dt: &Datatype,
    bytes: &[u8],
    ctx: &DecodeContext<'_>,
    name: &str,
) -> Result<T> {
    check_len(dt, bytes)?;
    let (t, off) = compound_member(dt, name)?;
    let b = bytes
        .get(off..off + t.size as usize)
        .ok_or(Error::UnexpectedEof)?;
    T::decode(&t, b, ctx)
}

/// Decode an integer element of `dt` (integer or enumeration) as `i128`
/// when representable (used by [`Value`] and `Display`).
/// Like [`int_value`], but `None` instead of an error. Never formats
/// `dt` (formatting an enumeration decodes its member values, so error
/// messages built from nested enumerations would cost exponential time).
pub(crate) fn int_value_opt(dt: &Datatype, bytes: &[u8]) -> Option<Int> {
    let (fp, size) = fixed_source(dt)?;
    bits::validate_fixed(fp, size).ok()?;
    if bytes.len() != dt.size as usize {
        return None;
    }
    bits::decode_fixed(fp, bytes).ok()
}

pub(crate) fn int_value(dt: &Datatype, bytes: &[u8]) -> Result<Int> {
    let Some((fp, size)) = fixed_source(dt) else {
        return mismatch(dt, "integer");
    };
    bits::validate_fixed(fp, size)?;
    check_len(dt, bytes)?;
    bits::decode_fixed(fp, bytes)
}
