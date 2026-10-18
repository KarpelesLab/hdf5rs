# hdf5rs

[![CI](https://github.com/KarpelesLab/hdf5rs/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/hdf5rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/hdf5rs.svg)](https://crates.io/crates/hdf5rs)
[![docs.rs](https://img.shields.io/docsrs/hdf5rs)](https://docs.rs/hdf5rs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure Rust, minimal-dependency library for reading and writing
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) files.

> **Status:** pre-1.0, the API may still change. Reading covers all
> datatypes, every layout and chunk index, the common filters and virtual
> datasets. Writing covers groups, attributes, contiguous, compact and
> chunked (filtered, resizable) datasets, and modifying existing files.
> Files written here are checked against h5py in CI. See
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

## Writing a file

```rust,no_run
use hdf5rs::write::FileWriter;

fn main() -> hdf5rs::Result<()> {
    let mut f = FileWriter::create("out.h5")?;
    let mut root = f.root();
    root.set_attr("title", "example")?;
    let mut g = root.create_group("weather")?;
    let mut ds = g
        .new_dataset::<f64>("temperature")
        .shape(&[2, 3])
        .write(&[20.5, 21.0, 21.5, 22.0, 22.5, 23.0])?;
    ds.set_attr("units", "celsius")?;
    // Chunked and compressed.
    g.new_dataset::<i32>("counts")
        .shape(&[1000])
        .chunk(&[100])
        .shuffle()
        .deflate(6)
        .write(&(0..1000).collect::<Vec<i32>>())?;
    f.finish()?;
    Ok(())
}
```

Try `cargo run --example h5ls -- -v -d some_file.h5` for a tree view of a
file.

## Cargo features

| Feature | Default | Enables |
|---|---|---|
| `deflate` | yes | gzip/zlib filter (id 1) |
| `lzf` | yes | LZF filter (id 32000) |
| `zstd` | | Zstandard filter (id 32015) |
| `lz4` | | LZ4 filter (id 32004) |
| `bzip2` | | BZIP2 filter (id 307) |
| `blosc` | | Blosc filter (id 32001; lz4, zlib, zstd and snappy codecs) |
| `bitshuffle` | | Bitshuffle filter (id 32008) |
| `all-filters` | | all of the filters above |
| `parallel` | | decode the chunks of one read on several threads |

Shuffle, Fletcher-32, N-bit and scale-offset are always available. The
compression codecs come from [`compcol`](https://crates.io/crates/compcol),
a pure-Rust compression crate, which is the only (optional) dependency.

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
