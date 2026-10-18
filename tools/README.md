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

Clean-room reminder: only use h5py's public API to create and inspect files;
never read libhdf5 or h5py source code.
