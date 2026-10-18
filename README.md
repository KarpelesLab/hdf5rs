# hdf5rs

A pure Rust, minimal-dependency library for reading and writing
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) files.

> **Status:** early development, not usable yet. See [ROADMAP.md](ROADMAP.md).

## Goals

- **Pure Rust** — no C toolchain, no `libhdf5`, no `unsafe` (`#![forbid(unsafe_code)]`).
- **Minimal dependencies** — zero required dependencies; anything extra sits
  behind an opt-in Cargo feature.
- **Read and write** — open files produced by the reference library and other
  tools, and produce files they can read.
- **Clean room** — implemented from the public HDF5 File Format Specification
  only (see below).

## Clean-room policy

This crate is written from the published HDF5 File Format Specification and
from observing the bytes of HDF5 files. Contributors must not consult, copy,
or translate source code from the HDF5 reference C library or from other HDF5
implementations under incompatible licenses. Using the reference tools
(`h5dump`, `h5py`, …) as black boxes to generate test files and check
interoperability is fine.

## Minimum supported Rust version

Rust **1.95** (edition 2024). Raising the MSRV is treated as a minor-version
change.

## License

[MIT](LICENSE)
