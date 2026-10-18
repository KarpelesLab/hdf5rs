//! Mapping Rust values to HDF5 datatypes and little-endian element bytes.

use super::ObjectId;
use super::heap::HeapWriter;
use super::storage::{Allocator, WriteAt};
use crate::error::{Error, Result};
use crate::format::Sizes;
use crate::format::messages::datatype::{ByteOrder, CharacterSet, Datatype, StringPadding};

/// State available while encoding element values: global heap storage
/// for variable-length data and deferred object references.
pub struct EncodeContext<'a> {
    pub(crate) sizes: Sizes,
    pub(crate) heap: &'a mut HeapWriter,
    pub(crate) alloc: &'a mut Allocator,
    pub(crate) storage: &'a mut dyn WriteAt,
    /// Object references to patch: (position in the output buffer, target).
    pub(crate) refs: Vec<(usize, ObjectId)>,
}

impl std::fmt::Debug for EncodeContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodeContext")
            .field("sizes", &self.sizes)
            .field("pending_refs", &self.refs.len())
            .finish_non_exhaustive()
    }
}

impl EncodeContext<'_> {
    /// Offset/length widths of the file being written.
    pub fn sizes(&self) -> Sizes {
        self.sizes
    }

    /// Append a variable-length element: the element count, then the
    /// global heap ID of `payload` (the encoded elements). An empty
    /// payload is stored without a heap object (null heap ID).
    pub fn write_vlen(&mut self, out: &mut Vec<u8>, count: usize, payload: &[u8]) -> Result<()> {
        let count = u32::try_from(count)
            .map_err(|_| Error::LimitExceeded("variable-length element count"))?;
        out.extend_from_slice(&count.to_le_bytes());
        let width = self.sizes.offset as usize;
        if payload.is_empty() {
            out.resize(out.len() + width + 4, 0);
            return Ok(());
        }
        let id = self
            .heap
            .add(payload, self.sizes, self.alloc, &mut *self.storage)?;
        let addr = id.collection_address.to_le_bytes();
        out.extend((0..width).map(|i| addr.get(i).copied().unwrap_or(0)));
        out.extend_from_slice(&id.index.to_le_bytes());
        Ok(())
    }

    /// Append an object reference to `target` (an address patched in when
    /// the file is finished), or a null reference for `None`.
    pub fn write_object_ref(&mut self, out: &mut Vec<u8>, target: Option<ObjectId>) {
        if let Some(t) = target {
            self.refs.push((out.len(), t));
        }
        out.resize(out.len() + self.sizes.offset as usize, 0);
    }

    /// Encode `values` into a temporary buffer that will be stored in the
    /// global heap (object references are not allowed there).
    fn encode_payload<T: H5Write>(&mut self, values: &[T]) -> Result<Vec<u8>> {
        let mark = self.refs.len();
        let mut buf = Vec::new();
        T::encode_slice(values, &mut buf, self)?;
        if self.refs.len() != mark {
            self.refs.truncate(mark);
            return Err(Error::unsupported(
                "object references inside variable-length data",
            ));
        }
        Ok(buf)
    }
}

/// A Rust type that can be written as HDF5 elements.
///
/// Implemented for the primitive numeric types (stored little-endian),
/// `bool` (h5py-style `FALSE`/`TRUE` enumeration), `String`/`str`
/// (variable-length UTF-8 strings), [`FixedAscii`]/[`FixedUtf8`]
/// (fixed-length strings), `Vec<T>` (variable-length sequences), `[T; N]`
/// (array type) and [`ObjectRef`] (object references).
///
/// Implement it for your own structs to write compound data:
///
/// ```
/// use hdf5rs::format::Sizes;
/// use hdf5rs::format::messages::datatype::Datatype;
/// use hdf5rs::write::{EncodeContext, H5Write};
///
/// struct Point { x: f64, y: i32 }
///
/// impl H5Write for Point {
///     fn datatype(sizes: Sizes) -> Datatype {
///         Datatype::packed_compound(vec![
///             ("x".into(), f64::datatype(sizes)),
///             ("y".into(), i32::datatype(sizes)),
///         ])
///         .expect("valid compound")
///     }
///     fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> hdf5rs::Result<()> {
///         self.x.encode(out, ctx)?;
///         self.y.encode(out, ctx)
///     }
/// }
/// ```
pub trait H5Write {
    /// The datatype of one element.
    fn datatype(sizes: Sizes) -> Datatype;

    /// Append the encoding of one element (exactly `datatype().size` bytes).
    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()>;

    /// Append the encodings of several elements.
    fn encode_slice(values: &[Self], out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()>
    where
        Self: Sized,
    {
        for v in values {
            v.encode(out, ctx)?;
        }
        Ok(())
    }
}

macro_rules! impl_int {
    ($($t:ty => $signed:expr),* $(,)?) => {$(
        impl H5Write for $t {
            fn datatype(_: Sizes) -> Datatype {
                Datatype::integer(size_of::<$t>() as u32, $signed, ByteOrder::LittleEndian)
            }
            fn encode(&self, out: &mut Vec<u8>, _: &mut EncodeContext<'_>) -> Result<()> {
                out.extend_from_slice(&self.to_le_bytes());
                Ok(())
            }
            fn encode_slice(values: &[Self], out: &mut Vec<u8>, _: &mut EncodeContext<'_>) -> Result<()> {
                out.reserve(values.len() * size_of::<$t>());
                for v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Ok(())
            }
        }
    )*};
}

impl_int!(u8 => false, u16 => false, u32 => false, u64 => false,
          i8 => true, i16 => true, i32 => true, i64 => true);

macro_rules! impl_float {
    ($($t:ty),*) => {$(
        impl H5Write for $t {
            fn datatype(_: Sizes) -> Datatype {
                match Datatype::float(size_of::<$t>() as u32, ByteOrder::LittleEndian) {
                    Ok(dt) => dt,
                    // Unreachable: 4 and 8 are valid IEEE sizes.
                    Err(_) => Datatype::integer(size_of::<$t>() as u32, false, ByteOrder::LittleEndian),
                }
            }
            fn encode(&self, out: &mut Vec<u8>, _: &mut EncodeContext<'_>) -> Result<()> {
                out.extend_from_slice(&self.to_le_bytes());
                Ok(())
            }
            fn encode_slice(values: &[Self], out: &mut Vec<u8>, _: &mut EncodeContext<'_>) -> Result<()> {
                out.reserve(values.len() * size_of::<$t>());
                for v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Ok(())
            }
        }
    )*};
}

impl_float!(f32, f64);

impl H5Write for bool {
    fn datatype(_: Sizes) -> Datatype {
        Datatype::bool_enum()
    }
    fn encode(&self, out: &mut Vec<u8>, _: &mut EncodeContext<'_>) -> Result<()> {
        out.push(u8::from(*self));
        Ok(())
    }
}

impl H5Write for str {
    fn datatype(sizes: Sizes) -> Datatype {
        Datatype::vlen_string(CharacterSet::Utf8, sizes)
    }
    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        ctx.write_vlen(out, self.len(), self.as_bytes())
    }
}

impl H5Write for String {
    fn datatype(sizes: Sizes) -> Datatype {
        <str as H5Write>::datatype(sizes)
    }
    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        H5Write::encode(self.as_str(), out, ctx)
    }
}

impl<T: H5Write + ?Sized> H5Write for &T {
    fn datatype(sizes: Sizes) -> Datatype {
        T::datatype(sizes)
    }
    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        (**self).encode(out, ctx)
    }
}

impl<T: H5Write> H5Write for Vec<T> {
    fn datatype(sizes: Sizes) -> Datatype {
        Datatype::vlen_sequence(T::datatype(sizes), sizes)
    }
    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        let payload = ctx.encode_payload(self)?;
        ctx.write_vlen(out, self.len(), &payload)
    }
}

impl<T: H5Write, const N: usize> H5Write for [T; N] {
    fn datatype(sizes: Sizes) -> Datatype {
        Datatype::array(T::datatype(sizes), &[u32::try_from(N).unwrap_or(u32::MAX)])
    }
    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        T::encode_slice(self, out, ctx)
    }
}

macro_rules! fixed_string {
    ($name:ident, $charset:expr, $what:literal) => {
        #[doc = concat!("A fixed-length ", $what, " string of at most `N` bytes, stored")]
        /// NUL-padded (like NumPy's `S` dtype).
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name<const N: usize>(Vec<u8>);

        impl<const N: usize> $name<N> {
            /// Wrap `s`; fails if it is longer than `N` bytes, contains NUL
            #[doc = concat!("or is not ", $what, ".")]
            pub fn new(s: &str) -> Result<Self> {
                if s.len() > N || s.as_bytes().contains(&0) {
                    return Err(Error::invalid(format!(
                        "string {s:?} does not fit a {N}-byte fixed-length string"
                    )));
                }
                if $charset == CharacterSet::Ascii && !s.is_ascii() {
                    return Err(Error::invalid(format!("string {s:?} is not ASCII")));
                }
                Ok($name(s.as_bytes().to_vec()))
            }

            /// The string bytes (without padding).
            pub fn as_bytes(&self) -> &[u8] {
                &self.0
            }
        }

        impl<const N: usize> H5Write for $name<N> {
            fn datatype(_: Sizes) -> Datatype {
                Datatype::fixed_string(
                    u32::try_from(N).unwrap_or(u32::MAX),
                    StringPadding::NullPad,
                    $charset,
                )
            }
            fn encode(&self, out: &mut Vec<u8>, _: &mut EncodeContext<'_>) -> Result<()> {
                out.extend_from_slice(&self.0);
                out.resize(out.len() + (N - self.0.len()), 0);
                Ok(())
            }
        }
    };
}

fixed_string!(FixedAscii, CharacterSet::Ascii, "ASCII");
fixed_string!(FixedUtf8, CharacterSet::Utf8, "UTF-8");

/// An object reference (`H5R_OBJECT1`) to an object created by the same
/// writer, or a null reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ObjectRef(pub Option<ObjectId>);

impl ObjectRef {
    /// The null reference.
    pub const NULL: ObjectRef = ObjectRef(None);

    /// A reference to `target`.
    pub fn to(target: ObjectId) -> Self {
        ObjectRef(Some(target))
    }
}

impl From<ObjectId> for ObjectRef {
    fn from(id: ObjectId) -> Self {
        ObjectRef::to(id)
    }
}

impl H5Write for ObjectRef {
    fn datatype(sizes: Sizes) -> Datatype {
        Datatype::object_reference(sizes)
    }
    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        ctx.write_object_ref(out, self.0);
        Ok(())
    }
}

/// A value that can be stored as an attribute: a scalar (any [`H5Write`]
/// scalar type, including `str`) or a one-dimensional array (`[T]`,
/// `Vec<T>`, `[T; N]`).
pub trait AttrValue {
    /// Dimensions of the attribute's dataspace; `None` for a scalar.
    fn attr_shape(&self) -> Option<Vec<u64>>;
    /// The element datatype.
    fn attr_datatype(&self, sizes: Sizes) -> Datatype;
    /// Append the encoded elements.
    fn attr_encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()>;
}

macro_rules! scalar_attr {
    ($($t:ty),*) => {$(
        impl AttrValue for $t {
            fn attr_shape(&self) -> Option<Vec<u64>> {
                None
            }
            fn attr_datatype(&self, sizes: Sizes) -> Datatype {
                <$t as H5Write>::datatype(sizes)
            }
            fn attr_encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
                H5Write::encode(self, out, ctx)
            }
        }
    )*};
}

scalar_attr!(
    u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, bool, str, String, ObjectRef
);

impl<const N: usize> AttrValue for FixedAscii<N> {
    fn attr_shape(&self) -> Option<Vec<u64>> {
        None
    }
    fn attr_datatype(&self, sizes: Sizes) -> Datatype {
        <Self as H5Write>::datatype(sizes)
    }
    fn attr_encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        H5Write::encode(self, out, ctx)
    }
}

impl<const N: usize> AttrValue for FixedUtf8<N> {
    fn attr_shape(&self) -> Option<Vec<u64>> {
        None
    }
    fn attr_datatype(&self, sizes: Sizes) -> Datatype {
        <Self as H5Write>::datatype(sizes)
    }
    fn attr_encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        H5Write::encode(self, out, ctx)
    }
}

impl<T: H5Write> AttrValue for [T] {
    fn attr_shape(&self) -> Option<Vec<u64>> {
        Some(vec![self.len() as u64])
    }
    fn attr_datatype(&self, sizes: Sizes) -> Datatype {
        T::datatype(sizes)
    }
    fn attr_encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        T::encode_slice(self, out, ctx)
    }
}

impl<T: H5Write> AttrValue for Vec<T> {
    fn attr_shape(&self) -> Option<Vec<u64>> {
        self.as_slice().attr_shape()
    }
    fn attr_datatype(&self, sizes: Sizes) -> Datatype {
        T::datatype(sizes)
    }
    fn attr_encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        T::encode_slice(self, out, ctx)
    }
}

impl<T: H5Write, const N: usize> AttrValue for [T; N] {
    fn attr_shape(&self) -> Option<Vec<u64>> {
        self.as_slice().attr_shape()
    }
    fn attr_datatype(&self, sizes: Sizes) -> Datatype {
        T::datatype(sizes)
    }
    fn attr_encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        T::encode_slice(self, out, ctx)
    }
}
