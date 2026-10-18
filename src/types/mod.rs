//! Conversion of raw element bytes into Rust values.
//!
//! Dataset and attribute data is stored as packed elements laid out as
//! described by a [`Datatype`](crate::format::messages::datatype::Datatype). This module turns such bytes into values:
//!
//! - [`H5Type`] — statically typed decoding (`i32`, `f64`, `String`,
//!   `[T; N]`, `Vec<T>`, …) with fast bulk paths for numeric data;
//! - [`Value`] / [`decode_value`] — a dynamic model covering every
//!   datatype class;
//! - [`VlenResolver`] / [`GlobalHeapResolver`] — access to the global heap
//!   for variable-length data and region references;
//! - [`Reference`] — object, region and revised references;
//! - a human-readable [`Display`](std::fmt::Display) for datatypes.
//!
//! ```
//! use hdf5rs::format::messages::datatype::Datatype;
//! use hdf5rs::types::{DecodeContext, decode_all};
//!
//! // Datatype message for a big-endian signed 32-bit integer.
//! let msg = [0x10, 0x09, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
//! let dt = Datatype::parse(&msg)?;
//! let raw = [0, 0, 0, 1, 0xff, 0xff, 0xff, 0xfe];
//! let values: Vec<i64> = decode_all(&dt, &raw, &DecodeContext::default())?;
//! assert_eq!(values, [1, -2]);
//! assert_eq!(dt.to_string(), "i32 BE");
//! # Ok::<(), hdf5rs::Error>(())
//! ```

mod bits;
mod display;
mod h5type;
mod reference;
mod value;
mod vlen;

pub use h5type::{
    Complex, H5Type, StringBytes, check_member, compound_member, decode_all, decode_member,
    is_native,
};
pub use reference::{HyperslabBlock, Reference, RegionReference, RevisedReference, Selection};
pub use value::{Value, decode_value};
pub use vlen::{GlobalHeapResolver, VlenElement, VlenResolver};

use crate::format::Sizes;

/// File-level information needed while decoding elements.
#[derive(Clone, Copy)]
pub struct DecodeContext<'a> {
    /// Offset/length widths of the file (global heap IDs and object
    /// references are "Size of Offsets" wide).
    pub sizes: Sizes,
    /// Global heap access for variable-length data and region references;
    /// decoding such data without a resolver is an error.
    pub resolver: Option<&'a dyn VlenResolver>,
}

impl<'a> DecodeContext<'a> {
    /// A context without global heap access.
    pub fn new(sizes: Sizes) -> Self {
        DecodeContext {
            sizes,
            resolver: None,
        }
    }

    /// A context resolving global heap objects through `resolver`.
    pub fn with_resolver(sizes: Sizes, resolver: &'a dyn VlenResolver) -> Self {
        DecodeContext {
            sizes,
            resolver: Some(resolver),
        }
    }
}

impl Default for DecodeContext<'_> {
    /// 8-byte offsets and lengths, no resolver.
    fn default() -> Self {
        DecodeContext::new(Sizes::DEFAULT)
    }
}

impl std::fmt::Debug for DecodeContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeContext")
            .field("sizes", &self.sizes)
            .field("resolver", &self.resolver.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests;
