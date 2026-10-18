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
//!     println!("@{} = {:?}", attr.name(), attr.value()?);
//! }
//! if let Object::Group(g) = file.object("/group")? {
//!     println!("{} links", g.len()?);
//! }
//! # Ok::<(), hdf5rs::Error>(())
//! ```
//!
//! # Reading data
//!
//! [`Dataset::read`] decodes elements into any [`H5Type`]: integers,
//! floats, `bool`, `String`, arrays (`[T; N]`, `Vec<T>`), variable-length
//! sequences, [`Complex`], [`Reference`] and the dynamic [`Value`]. Shapes
//! come with [`Dataset::read_nd`]; [`Selection`]s read hyperslabs. Chunked
//! and compressed data, variable-length data (global heap), virtual
//! datasets and external raw data files are handled transparently;
//! [`Dataset::reader`] adjusts filters and resource limits, and
//! [`FileAccess`] controls whether other files may be opened.
//!
//! This example builds a tiny file in memory (one contiguous dataset of
//! three `i32`) and reads it back:
//!
//! ```
//! use hdf5rs::{File, Selection, Value};
//!
//! # fn tiny_file() -> Vec<u8> {
//! #     fn put(f: &mut Vec<u8>, at: usize, b: &[u8]) {
//! #         if f.len() < at + b.len() { f.resize(at + b.len(), 0); }
//! #         f[at..at + b.len()].copy_from_slice(b);
//! #     }
//! #     fn ohdr(msgs: &[(u8, Vec<u8>)]) -> Vec<u8> {
//! #         let mut body = Vec::new();
//! #         for (t, d) in msgs {
//! #             body.push(*t);
//! #             body.extend((d.len() as u16).to_le_bytes());
//! #             body.push(0);
//! #             body.extend(d);
//! #         }
//! #         let mut h = b"OHDR".to_vec();
//! #         h.extend([2, 0, body.len() as u8]);
//! #         h.extend(body);
//! #         let c = hdf5rs::checksum::metadata_checksum(&h);
//! #         h.extend(c.to_le_bytes());
//! #         h
//! #     }
//! #     let mut f = Vec::new();
//! #     let mut sb = hdf5rs::SIGNATURE.to_vec();
//! #     sb.extend([2, 8, 8, 0]);
//! #     for v in [0u64, u64::MAX, 256, 48] { sb.extend(v.to_le_bytes()); }
//! #     let c = hdf5rs::checksum::metadata_checksum(&sb);
//! #     sb.extend(c.to_le_bytes());
//! #     put(&mut f, 0, &sb);
//! #     let mut link = vec![1, 0, 1, b'x'];
//! #     link.extend(160u64.to_le_bytes());
//! #     put(&mut f, 48, &ohdr(&[(0x06, link)]));
//! #     let mut space = vec![2, 1, 0, 1];
//! #     space.extend(3u64.to_le_bytes());
//! #     let dtype = vec![0x10, 0x08, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
//! #     let mut layout = vec![3, 1];
//! #     layout.extend(240u64.to_le_bytes());
//! #     layout.extend(12u64.to_le_bytes());
//! #     put(&mut f, 160, &ohdr(&[(0x01, space), (0x03, dtype), (0x08, layout)]));
//! #     for (i, v) in [10i32, -20, 30].iter().enumerate() {
//! #         put(&mut f, 240 + 4 * i, &v.to_le_bytes());
//! #     }
//! #     f.resize(256, 0);
//! #     f
//! # }
//! let file = File::from_bytes(tiny_file())?;
//! let ds = file.dataset("/x")?;
//! assert_eq!(ds.read::<i32>()?, [10, -20, 30]);
//! assert_eq!(ds.read_selection::<i64>(&Selection::range(1..3))?, [-20, 30]);
//! let arr = ds.read_nd::<i32>()?;
//! assert_eq!((arr.shape.as_slice(), arr.get(&[2])), (&[3][..], Some(&30)));
//! assert_eq!(ds.read_values()?[1], Value::Int(-20));
//! # Ok::<(), hdf5rs::Error>(())
//! ```
//!
//! # Caching, performance and untrusted input
//!
//! Each [`Dataset`] keeps a [`ChunkCache`](read::ChunkCache) of decoded
//! chunks (16 MiB by default, see [`FileAccess::chunk_cache_bytes`]) and
//! its chunk index, so repeated small reads do not decompress the same
//! chunks again. The `parallel` Cargo feature decodes the chunks of one
//! read on several threads.
//!
//! Every size and count read from a file is validated before use: hostile
//! files produce errors, not panics, hangs or unbounded allocations.
//! [`ReadLimits`] (per file through [`FileAccess::read_limits`], or per
//! read) bounds output and decoded chunk sizes and lists every other bound
//! applied. The parsers are exercised by a mutation stress test and
//! cargo-fuzz targets (`tests/stress.rs`, `fuzz/`).
//!
//! # Writing a file
//!
//! See the [`write`](mod@write) module for details and supported features.
//!
//! ```no_run
//! use hdf5rs::write::FileWriter;
//!
//! let mut f = FileWriter::create("out.h5")?;
//! let mut root = f.root();
//! root.set_attr("title", "example")?;
//! root.new_dataset::<f32>("values").shape(&[2, 2]).write(&[1.0, 2.0, 3.0, 4.0])?;
//! f.finish()?;
//! # Ok::<(), hdf5rs::Error>(())
//! ```
//!
//! # Modifying a file
//!
//! Existing files (written by this crate or by the reference library) can
//! be opened read-write; see
//! [Modifying existing files](mod@write#modifying-existing-files).
//!
//! ```no_run
//! use hdf5rs::write::FileWriter;
//! use hdf5rs::Selection;
//!
//! let mut f = FileWriter::open_rw("data.h5")?;
//! f.group("/results")?.set_attr("checked", &true)?;
//! f.dataset("/results/log")?.append(&[1.5f64, 2.5])?;
//! f.dataset("/results/grid")?.write_selection(&Selection::from_ranges(&[0..1, 0..2]), &[7i32, 8])?;
//! f.unlink("/scratch")?;
//! f.finish()?;
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
mod reader;
pub mod types;
pub mod write;

pub use attribute::Attribute;
pub use dataset::{Dataset, ExternalFile, NamedDatatype};
pub use error::{Error, Result};
pub use file::{File, FileAccess, SharedStore};
pub use filters::FilterRegistry;
pub use format::messages::dataspace::{Dataspace, DataspaceKind};
pub use format::messages::datatype::Datatype;
pub use group::{Group, MAX_SOFT_LINK_HOPS};
pub use io::ReadAt;
pub use link::{Link, LinkEntry, LinkStorage};
pub use object::{Object, ObjectKind};
pub use read::{ChunkInfo, ExternalFileResolver, HyperslabDim, ReadLimits, Selection};
pub use reader::{Array, DatasetReader, MAX_VIRTUAL_DEPTH};
pub use types::{Complex, H5Type, Reference, StringBytes, Value};

/// The 8-byte signature that marks the start of an HDF5 superblock.
///
/// The superblock is located at file offset 0, or at 512, 1024, 2048, …
/// (successive powers of two) when the file carries a user block.
pub const SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];

/// Compiles the README examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_matches_spec() {
        assert_eq!(&SIGNATURE, b"\x89HDF\r\n\x1a\n");
    }
}
