# hdf5rs

A pure Rust, minimal-dependency library for reading and writing
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) files.

> **Status:** early development. Reading works for the files in the test
> corpus (all datatypes, every layout and chunk index, the common filters,
> virtual datasets); writing is not implemented yet. See
> [ROADMAP.md](ROADMAP.md).

## Reading a file

```rust,no_run
use hdf5rs::{File, Selection, Value};

fn main() -> hdf5rs::Result<()> {
    let file = File::open("data.h5")?;

    // Walk the hierarchy.
    for entry in file.root()?.members()? {
        println!("{}: {:?}", entry.name, entry.link);
    }

    // Typed reads: the whole dataset, with its shape, or a hyperslab.
    let temps = file.dataset("/weather/temperature")?;
    let all: Vec<f64> = temps.read()?;
    let grid = temps.read_nd::<f32>()?; // grid.data, grid.shape
    let block: Vec<f64> = temps.read_selection(&Selection::from_ranges(&[0..10, 0..4]))?;
    println!("{} values, shape {:?}, block {block:?}", all.len(), grid.shape);

    // Strings, variable-length data and compounds.
    let names: Vec<String> = file.dataset("/stations/names")?.read()?;
    let records: Vec<Value> = file.dataset("/stations/info")?.read_values()?;
    if let Some(id) = records.first().and_then(|r| r.field("id")).and_then(Value::as_i64) {
        println!("first of {} stations has id {id}", names.len());
    }

    // Attributes.
    let units: String = temps.attribute("units")?.read_scalar()?;
    println!("temperatures in {units}");
    Ok(())
}
```

Try `cargo run --example h5ls -- -v -d some_file.h5` for a tree view of a
file.

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
