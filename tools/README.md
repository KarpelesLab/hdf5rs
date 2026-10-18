# Tools

Helper scripts that are not part of the crate build (this directory is
excluded from the published package).

## `gen_fixtures.py` — interoperability fixtures

Generates the HDF5 files in `tests/fixtures/` with the reference
implementation (libhdf5 via h5py), used strictly as a black box, and writes
`tests/fixtures/manifest.json` (what the reference reader reports for each
file) and `tests/fixtures/files.txt`. The manifest schema is documented in
[`tests/fixtures/README.md`](../tests/fixtures/README.md).

Regenerate from the repository root, with a virtualenv kept **outside** the
repository:

```sh
python3 -m venv /tmp/hdf5rs-venv
/tmp/hdf5rs-venv/bin/pip install h5py numpy hdf5plugin   # hdf5plugin is optional
/tmp/hdf5rs-venv/bin/python tools/gen_fixtures.py         # --out DIR to write elsewhere
```

Without `hdf5plugin`, `filters_plugins.h5` is not produced and the old copy
is deleted.

Output is deterministic for a given h5py / HDF5 version (object timestamps
disabled, fixed random seed), with two exceptions where h5py offers no way
to turn timestamps off: the committed datatypes in `dtypes_latest.h5` and
the superblock-extension object header in `fsm_paged.h5` carry creation
times (useful as test cases for the object-header "times" field). So a
re-run changes those two files and their `sha256` in the manifest; commit
such a regeneration only together with a real change. The versions
used are recorded at the top of the manifest; the current corpus was built
with h5py 3.16.0 / HDF5 2.0.0.

Regenerated files **must be committed** (the `.h5` files, `manifest.json`
and `files.txt` together): tests run against the checked-in corpus and do not
need Python. Keep each fixture small (ideally < 64 KiB) and the whole corpus
well under 3 MiB.

## `check_written.py` — files written by hdf5rs, read by h5py

Black-box interoperability check of the writer: `examples/write_demo.rs`
writes a file exercising every supported feature, and the script asserts
its structure and contents through h5py (the reference library), then
modifies a copy with h5py (adding/deleting links and attributes, crossing
the compact/dense thresholds) and checks the result.

```sh
cargo run --example write_demo -- /tmp/demo.h5 v3     # or v2 / v0; add o4 / times
/tmp/hdf5rs-venv/bin/python tools/check_written.py /tmp/demo.h5
```

The CI `interop` job runs it for superblock v0 (compatibility format), v2,
v3 and 4-byte offsets.

## `check_modified.py` — files modified by hdf5rs, checked by h5py

Black-box check of the read-write mode (`FileWriter::open_rw`): h5py
creates files in several modes (libver earliest / v108 / latest, with and
without creation-order tracking, a non-default file space strategy),
`examples/modify_demo.rs` modifies them (`scenario`: dense/compact
transitions, symbol-table growth, deletions, attribute overwrite, writes
into every chunk index, resize, append; `touch`: changes every object and
rewrites every dataset), and h5py checks the expected changes and that
everything else is unchanged, then modifies the result itself. Files with
persistent free-space managers / paged allocation must be refused. The
reverse direction is covered too: a `write_demo` file modified by h5py,
then by hdf5rs, then by h5py again, read back by hdf5rs.

```sh
cargo build --examples
/tmp/hdf5rs-venv/bin/python tools/check_modified.py   # --keep DIR keeps the files
```

Clean-room reminder: only use h5py's public API to create and inspect files;
never read libhdf5 or h5py source code.
