# hdf5rs roadmap

This is the plan for taking `hdf5rs` from an empty crate to a usable,
interoperable HDF5 reader and writer. Milestones are roughly ordered by
dependency; each one should end with a tagged `0.x` release.

Section references (§) point to the public *HDF5 File Format Specification,
Version 3.0*, which is the only normative source for this project (see the
clean-room policy in the README).

## Guiding principles

- **Minimal dependencies.** The only dependency is our own pure-Rust
  [`compcol`](https://crates.io/crates/compcol) compression crate, optional
  behind per-codec features (`deflate` on by default). Format-specific
  filters (shuffle, Fletcher-32, LZF, N-bit, scale-offset) live in-crate.
- **No `unsafe`.** Enforced with `#![forbid(unsafe_code)]`.
- **Never panic on bad input.** Every parser returns `Result`; malformed or
  hostile files must produce errors, not panics, infinite loops, or unbounded
  allocations. Fuzzing is part of the plan, not an afterthought.
- **Read broadly, write conservatively.** The reader should accept every
  format version found in the wild. The writer emits a small, well-tested
  subset of format versions (configurable between "maximum compatibility"
  and "latest format").
- **I/O agnostic core.** Parsing works over a small `ReadAt`/`WriteAt`-style
  trait so it can run over files, memory buffers, or custom backends.
  `std::fs::File` is the default backend.
- **Interop tested.** Fixtures are generated with the reference tools
  (h5py / `h5dump` / `h5repack`) used as black boxes, and files we write are
  validated by reading them back with those tools in CI.

## Milestone 0 — Foundations (`0.0.x`)

Low-level parsers marked [x] below exist with unit tests; they are being
validated end-to-end against the fixture corpus as the high-level API lands.

- [x] Crate scaffold, CI (stable + MSRV, fmt, clippy, docs), license
- [x] Error type (`Error`, `Result`), no panics on malformed input
- [x] Storage abstraction: positional read trait, `File` and in-memory
      (`Vec<u8>` / `&[u8]`) implementations (write side: Milestone 4)
- [x] Little-endian byte cursor with variable-width *offset* and *length*
      fields (sizes from the superblock), "undefined address" handling
- [x] Jenkins `lookup3` checksum (§ checksum used by v2 structures)
- [x] Test fixture corpus: script (outside the crate build) that generates a
      set of `.h5` files with h5py, plus checked-in small fixtures
- [ ] Basic fuzz harness (`cargo fuzz`, kept out of the published crate)

## Milestone 1 — Read: file structure (`0.1`)

Goal: open a file and walk its group hierarchy, list objects and attributes.

- [x] Superblock discovery (offset 0, 512, 1024, … for user blocks)
- [x] Superblock versions 0, 1, 2, 3 (§ Level 0A); driver info block (read &
      ignore for non-default drivers)
- [x] Object headers version 1 and version 2 (`OHDR`), continuation blocks
      (`OCHK`), header message iteration
- [x] Old-style groups: symbol table message, v1 B-tree (group nodes),
      symbol table nodes (`SNOD`), local heap (`HEAP`)
- [ ] New-style groups: link info, link messages (hard, soft, external),
      group info message
- [x] Dense link storage: fractal heap (`FRHP`, `FHDB`, `FHIB`, huge/tiny
      objects) and v2 B-tree (`BTHD`, `BTIN`, `BTLF`) name/creation-order
      indexes
- [ ] Path resolution (`/a/b/c`), soft-link following with cycle limits;
      external links reported but not followed by default
- [ ] Public API sketch: `File::open`, `Group`, `Dataset`, `Attribute`,
      iteration over members

## Milestone 2 — Read: datatypes, dataspaces and data (`0.2`)

Goal: read the raw contents of uncompressed datasets and attributes.

- [x] Dataspace message: scalar, simple (with max dims), null
- [x] Datatype message, all classes (§ IV.A.2.d):
  - [ ] fixed-point (integers, any size/endianness/sign)
  - [ ] floating-point (IEEE f16/f32/f64, non-native layouts best effort)
  - [ ] strings (fixed-length; null-terminated / null-padded / space-padded;
        ASCII / UTF-8)
  - [ ] bitfield, opaque, enum
  - [ ] compound (member offsets, nested types)
  - [ ] array
  - [ ] variable-length sequences and strings (global heap `GCOL`)
  - [ ] references (object, region; revised reference format)
  - [ ] time (rare; parse, expose raw)
- [x] Data layout message versions 1–4: compact, contiguous
- [x] Fill value messages (old and new)
- [ ] Attribute message (versions 1–3), attribute info message, dense
      attribute storage (fractal heap + v2 B-tree)
- [ ] Shared object header messages (committed datatypes, SOHM table
      `SMTB` / `SMLI`) — read side
- [ ] Typed read API: `dataset.read::<T>()` into `Vec<T>` for primitive
      types, plus a raw-bytes + datatype-descriptor path for everything else;
      a small `H5Type` trait users can implement (derive macro later, as a
      separate optional crate)
- [ ] Hyperslab selection on read (start/stride/count/block)

## Milestone 3 — Read: chunked storage and filters (`0.3`)

Goal: read the vast majority of real-world files, including compressed ones.

- [ ] Chunked layout with every chunk index type:
  - [ ] v1 B-tree (raw data chunk nodes) — layout v1–3
  - [ ] single chunk, implicit, fixed array (`FAHD`/`FADB`), extensible array
        (`EAHD`/`EAIB`/`EASB`/`EADB`), v2 B-tree — layout v4
- [x] Filter pipeline message (versions 1 and 2), per-chunk filter masks
- [ ] Filters implemented in-crate:
  - [x] deflate (zlib) — via `compcol`
  - [x] shuffle
  - [x] Fletcher-32 (to be verified against fixtures)
  - [ ] N-bit and scale-offset
  - [x] LZF (common in h5py output; to be verified against fixtures)
- [x] Pluggable filters: a `Filter` trait plus registry so users can add
      codecs without the core depending on them
- [ ] Optional `compcol`-backed features for Zstd, LZ4, bzip2 (and Blosc
      framing on top of them)
- [ ] Chunk cache with configurable size
- [ ] Virtual dataset layout (layout v4, class 3) — read, same-file and
      external sources

## Milestone 4 — Write: basic files (`0.4`)

Goal: create new files that the reference tools read without complaint.

- [ ] File creation; superblock v2/v3 (latest format) and v0 (compat) modes
- [ ] Object header v2 writing (with v1 for compat mode)
- [ ] Free-space management for the writer (simple append allocator first,
      then reuse of freed space)
- [ ] Groups: compact link storage, then dense storage (fractal heap +
      v2 B-tree writing) for large groups; old-style symbol-table groups in
      compat mode
- [ ] Datasets: contiguous and compact layouts, all fixed-size datatypes,
      fill values
- [ ] Attributes: compact and dense storage
- [ ] Variable-length strings and sequences (global heap writing)
- [ ] Committed (named) datatypes, hard and soft links
- [ ] Correct metadata checksums everywhere
- [ ] CI step: round-trip every written fixture through `h5dump` / h5py

## Milestone 5 — Write: chunking, compression, modification (`0.5`)

- [ ] Chunked datasets with fixed-array, extensible-array, v2 B-tree and
      single-chunk indexes (and v1 B-tree in compat mode)
- [ ] Resizable datasets (unlimited dimensions), append workflows
- [ ] Filters on write: deflate, shuffle, Fletcher-32, LZF
- [ ] Open existing files read-write: add/delete objects and attributes,
      overwrite data in place, extend datasets
- [ ] Partial (hyperslab) writes
- [ ] Crash-consistency considerations; optional SWMR-compatible write order
      (documented, best effort)

## Milestone 6 — Hardening and ergonomics (`0.6` → `1.0`)

- [ ] Fuzzing of all parsers in CI (time-boxed), with corpus seeded from
      fixtures
- [ ] Resource limits for untrusted input (max allocation, max recursion
      depth, max chunk size)
- [ ] Performance: avoid per-element allocations, vectorizable byte-order
      conversion, parallel chunk decoding behind an optional feature
- [ ] Optional `ndarray` integration feature
- [ ] Optional derive macro crate (`hdf5rs-derive`) for compound types
- [ ] `no_std` + `alloc` support for the parsing core (stretch goal)
- [ ] Documentation: guide, examples (`examples/`), format notes
- [ ] API review and stabilization → `1.0`

## Explicitly out of scope (for now)

- Parallel HDF5 / MPI-IO
- Non-default file drivers as writers (family, multi/split, direct, ROS3);
  reading family/split files may be revisited later
- Bit-for-bit reproduction of files written by the reference library
- The HDF4 format
