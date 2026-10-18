//! # hdf5rs
//!
//! A pure Rust, minimal-dependency library for reading and writing
//! [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files.
//!
//! This crate is a clean-room implementation written from the public HDF5
//! File Format Specification. It does not link to, wrap, or derive from the
//! reference C library.
//!
//! The crate is at an early stage; see `ROADMAP.md` in the repository for the
//! planned feature set and milestones.
//!
//! # Navigating a file
//!
//! ```no_run
//! use hdf5rs::{File, Object};
//!
//! let file = File::open("data.h5")?;
//! for entry in file.root()?.members()? {
//!     println!("{}: {:?}", entry.name, entry.link);
//! }
//! let ds = file.dataset("/group/temperature")?;
//! println!("shape {:?}, chunks {:?}", ds.shape(), ds.chunk_shape());
//! for attr in ds.attributes()? {
//!     println!("@{} = {} raw bytes", attr.name(), attr.raw_value().len());
//! }
//! if let Object::Group(g) = file.object("/group")? {
//!     println!("{} links", g.len()?);
//! }
//! # Ok::<(), hdf5rs::Error>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod attribute;
pub mod checksum;
mod dataset;
mod error;
mod file;
pub mod filters;
pub mod format;
mod group;
pub mod io;
mod link;
mod object;
pub mod read;
pub mod types;

pub use attribute::Attribute;
pub use dataset::{Dataset, ExternalFile, NamedDatatype};
pub use error::{Error, Result};
pub use file::{File, SharedStore};
pub use group::{Group, MAX_SOFT_LINK_HOPS};
pub use io::ReadAt;
pub use link::{Link, LinkEntry, LinkStorage};
pub use object::{Object, ObjectKind};

/// The 8-byte signature that marks the start of an HDF5 superblock.
///
/// The superblock is located at file offset 0, or at 512, 1024, 2048, …
/// (successive powers of two) when the file carries a user block.
pub const SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_matches_spec() {
        assert_eq!(&SIGNATURE, b"\x89HDF\r\n\x1a\n");
    }
}
