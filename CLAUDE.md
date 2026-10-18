# hdf5rs — working notes

Pure Rust, minimal-dependency HDF5 reader/writer. MIT, edition 2024, MSRV 1.95.
Default branch: `master`. See ROADMAP.md for milestones.

## Hard rules

- **Clean room.** Only use the public HDF5 File Format Specification
  (https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html and
  related spec pages), RFCs (e.g. RFC 1950/1951 for zlib/deflate), and
  black-box observation of files. Never read, copy or translate source code
  of libhdf5, h5py internals, or other HDF5/codec implementations
  (jhdf, hdf5-rust, pyfive, zlib, miniz, …).
- **No dependencies** in `[dependencies]` without explicit approval.
- `#![forbid(unsafe_code)]`. No panics on malformed input: no `unwrap`,
  unchecked indexing or unchecked arithmetic on file-derived values outside
  tests. Cap allocations derived from file data (validate against file size
  or sane limits; return `Error::LimitExceeded`).
- Must pass: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, `RUSTDOCFLAGS=-D warnings cargo doc --no-deps`.

## Layout & conventions

- `src/error.rs` — `Error` / `Result` (`Error::invalid(..)`, `Error::unsupported(..)`).
- `src/io.rs` — `ReadAt` positional read trait (slices, Vec, File, …).
- `src/checksum.rs` — `lookup3` / `metadata_checksum` (Jenkins, init 0).
- `src/format/` — low-level on-disk structures, one file per structure.
  Parse from bytes with `format::ByteReader` (bounds-checked LE cursor,
  `read_offset()` / `read_length()` honour `format::Sizes`; undefined
  address → `format::UNDEFINED_ADDRESS`).
- `src/filters/` — filter codecs (deflate, shuffle, fletcher32, lzf, …).
- Unit tests live next to the code; hand-built byte vectors are fine.
  Integration tests against real files live in `tests/` using
  `tests/fixtures/*.h5`.
- Comments: concise, cite spec section names where helpful.
