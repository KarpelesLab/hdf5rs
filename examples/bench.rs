//! Read-path micro benchmarks (no external benchmark framework).
//!
//! `cargo run --release --example bench [-- filter]`
//!
//! Measures, best of several runs:
//! - `tests/fixtures/big_gzip.h5` (shuffle + gzip chunks): full reads,
//!   type conversion and repeated small selections (chunk cache);
//! - a synthetic 64 MiB contiguous `f64` dataset written in memory: raw,
//!   typed, converting and strided reads;
//! - synthetic 4096×4096 `f32` chunked datasets (256×256 chunks), built
//!   directly over an in-memory store: shuffle+deflate through a version 1
//!   B-tree index, and unfiltered through an implicit index.
//!
//! An optional argument only runs benchmarks whose name contains it.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use hdf5rs::filters::{FilterContext, FilterRegistry, PipelineStep, encode_pipeline};
use hdf5rs::format::Sizes;
use hdf5rs::format::messages::dataspace::{Dataspace, DataspaceKind};
use hdf5rs::format::messages::filter_pipeline::{FilterInfo, FilterPipeline};
use hdf5rs::format::messages::layout::{ChunkIndex, ChunkedLayout, DataLayout};
use hdf5rs::read::{ChunkCache, DatasetStorage, Selection};
use hdf5rs::write::FileWriter;
use hdf5rs::{File, HyperslabDim};

struct Bench {
    filter: Option<String>,
}

impl Bench {
    /// Time `f` (best of a few runs, at least ~0.3 s of total work) and
    /// print the result; `bytes` is the amount of data produced per run.
    fn run(&self, name: &str, bytes: usize, mut f: impl FnMut()) {
        if let Some(flt) = &self.filter
            && !name.contains(flt.as_str())
        {
            return;
        }
        f(); // warm-up
        let mut best = Duration::MAX;
        let mut total = Duration::ZERO;
        let mut runs = 0;
        while runs < 3 || (total < Duration::from_millis(300) && runs < 200) {
            let t = Instant::now();
            f();
            let e = t.elapsed();
            best = best.min(e);
            total += e;
            runs += 1;
        }
        let secs = best.as_secs_f64();
        if bytes > 0 {
            println!(
                "{name:<44} {:>10.3} ms {:>9.1} MiB/s",
                secs * 1e3,
                bytes as f64 / secs / (1 << 20) as f64
            );
        } else {
            println!("{name:<44} {:>10.3} ms", secs * 1e3);
        }
    }
}

/// Tiny deterministic PRNG for selection positions.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn simple_space(dims: &[u64]) -> Dataspace {
    Dataspace {
        version: 2,
        kind: DataspaceKind::Simple,
        dims: dims.to_vec(),
        max_dims: None,
        permutation: None,
    }
}

fn big_gzip(b: &Bench) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/big_gzip.h5");
    let file = File::open(&path).expect("big_gzip.h5");
    let root = file.root().unwrap();
    for entry in root.members().unwrap() {
        let ds = file.dataset(&entry.name).unwrap();
        let n = ds.element_count().unwrap() as usize;
        let esize = ds.datatype().size as usize;
        println!(
            "# big_gzip {}: shape {:?} chunks {:?} {} B elements",
            entry.name,
            ds.shape(),
            ds.chunk_shape(),
            esize
        );
        let tag = format!("big_gzip/{}", entry.name);
        b.run(&format!("{tag} read_raw"), n * esize, || {
            black_box(ds.reader().use_chunk_cache(false).read_raw().unwrap());
        });
        if ds.reader().use_chunk_cache(false).read::<i64>().is_ok() {
            b.run(&format!("{tag} read::<i32>"), n * 4, || {
                black_box(ds.reader().use_chunk_cache(false).read::<i32>().unwrap());
            });
            b.run(&format!("{tag} read::<i64>"), n * 8, || {
                black_box(ds.reader().use_chunk_cache(false).read::<i64>().unwrap());
            });
        } else {
            b.run(&format!("{tag} read::<f64>"), n * 8, || {
                black_box(ds.reader().use_chunk_cache(false).read::<f64>().unwrap());
            });
        }
        // Repeated small reads at random positions (chunk cache).
        let shape = ds.shape().to_vec();
        let mut rng = Rng(0x1234_5678);
        b.run(&format!("{tag} 200 small selections"), 0, || {
            for _ in 0..200 {
                let ranges: Vec<_> = shape
                    .iter()
                    .map(|&d| {
                        let s = rng.next() % d.saturating_sub(4).max(1);
                        s..(s + 4).min(d)
                    })
                    .collect();
                black_box(
                    ds.read_raw_selection(&Selection::from_ranges(&ranges))
                        .unwrap(),
                );
            }
        });
        b.run(&format!("{tag} 200 small selections, no cache"), 0, || {
            for _ in 0..200 {
                let ranges: Vec<_> = shape
                    .iter()
                    .map(|&d| {
                        let s = rng.next() % d.saturating_sub(4).max(1);
                        s..(s + 4).min(d)
                    })
                    .collect();
                black_box(
                    ds.reader()
                        .use_chunk_cache(false)
                        .read_raw_selection(&Selection::from_ranges(&ranges))
                        .unwrap(),
                );
            }
        });
    }
}

fn contiguous(b: &Bench) {
    const N: usize = 8 << 20; // 64 MiB of f64
    let data: Vec<f64> = (0..N).map(|i| i as f64 * 0.5).collect();
    let mut w = FileWriter::in_memory();
    w.root()
        .new_dataset::<f64>("x")
        .shape(&[N as u64])
        .write(&data)
        .unwrap();
    w.root()
        .new_dataset::<f64>("m")
        .shape(&[2048, 4096])
        .write(&data)
        .unwrap();
    let ints: Vec<i32> = (0..2 * N as i32).collect();
    w.root()
        .new_dataset::<i32>("i")
        .shape(&[2 * N as u64])
        .write(&ints)
        .unwrap();
    let bytes = w.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("x").unwrap();
    b.run("contiguous 64MiB read_raw", N * 8, || {
        black_box(ds.read_raw().unwrap());
    });
    b.run("contiguous 64MiB read::<f64>", N * 8, || {
        black_box(ds.read::<f64>().unwrap());
    });
    let ints = file.dataset("i").unwrap();
    b.run("contiguous 64MiB i32 read::<i32>", N * 8, || {
        black_box(ints.read::<i32>().unwrap());
    });
    b.run("contiguous 64MiB i32 read::<i64>", N * 16, || {
        black_box(ints.read::<i64>().unwrap());
    });
    let strided = Selection::hyperslab(vec![HyperslabDim::new(0, 2, N as u64 / 2, 1)]);
    b.run("contiguous 64MiB every 2nd element", N * 4, || {
        black_box(ds.read_raw_selection(&strided).unwrap());
    });
    let m = file.dataset("m").unwrap();
    let col = Selection::from_ranges(&[0..2048, 100..101]);
    b.run("contiguous 2048x4096 one column", 2048 * 8, || {
        black_box(m.read_raw_selection(&col).unwrap());
    });
    let boxsel = Selection::from_ranges(&[512..1536, 1024..3072]);
    b.run(
        "contiguous 2048x4096 1024x2048 box",
        1024 * 2048 * 8,
        || {
            black_box(m.read_raw_selection(&boxsel).unwrap());
        },
    );
}

const SIDE: u64 = 4096;
const CHUNK: u64 = 256;

/// A 4096×4096 f32 dataset in 256×256 chunks: (store, layout, pipeline).
fn synthetic_chunked(filtered: bool) -> (Vec<u8>, DataLayout, Option<FilterPipeline>) {
    let per = SIDE / CHUNK;
    let nchunks = (per * per) as usize;
    let chunk_elems = (CHUNK * CHUNK) as usize;
    let registry = FilterRegistry::with_builtin();
    let pipeline = FilterPipeline {
        filters: vec![
            FilterInfo {
                id: hdf5rs::filters::SHUFFLE,
                name: None,
                optional: false,
                client_data: vec![4],
            },
            FilterInfo {
                id: hdf5rs::filters::DEFLATE,
                name: None,
                optional: false,
                client_data: vec![4],
            },
        ],
    };
    let steps: Vec<PipelineStep<'_>> = pipeline
        .filters
        .iter()
        .map(|f| PipelineStep {
            id: f.id,
            optional: f.optional,
            client_data: &f.client_data,
        })
        .collect();
    let ctx = FilterContext::new(4);
    // Data: a smooth function, so it compresses like real measurements.
    let mut store = vec![0u8; 64];
    let mut entries = Vec::with_capacity(nchunks);
    for cy in 0..per {
        for cx in 0..per {
            let mut raw = Vec::with_capacity(chunk_elems * 4);
            for y in 0..CHUNK {
                for x in 0..CHUNK {
                    let (gy, gx) = ((cy * CHUNK + y) as f32, (cx * CHUNK + x) as f32);
                    let v = (gy * 0.01).sin() * (gx * 0.013).cos() * 100.0;
                    raw.extend_from_slice(&v.to_le_bytes());
                }
            }
            let (stored, mask) = if filtered {
                encode_pipeline(&registry, &steps, raw, &ctx).unwrap()
            } else {
                (raw, 0)
            };
            let addr = store.len() as u64;
            entries.push((cy * CHUNK, cx * CHUNK, addr, stored.len() as u32, mask));
            store.extend_from_slice(&stored);
        }
    }
    if !filtered {
        let layout = DataLayout::Chunked(ChunkedLayout {
            dims: vec![CHUNK, CHUNK],
            element_size: 4,
            flags: 0,
            index: ChunkIndex::Implicit { address: 64 },
            layout_version: 4,
        });
        return (store, layout, None);
    }
    // One leaf node of a version 1 chunk B-tree holding every chunk.
    let root = store.len() as u64;
    let mut node = b"TREE".to_vec();
    node.extend([1u8, 0]);
    node.extend((entries.len() as u16).to_le_bytes());
    node.extend(u64::MAX.to_le_bytes());
    node.extend(u64::MAX.to_le_bytes());
    let key = |node: &mut Vec<u8>, size: u32, mask: u32, y: u64, x: u64| {
        node.extend(size.to_le_bytes());
        node.extend(mask.to_le_bytes());
        for v in [y, x, 0] {
            node.extend(v.to_le_bytes());
        }
    };
    for &(y, x, addr, size, mask) in &entries {
        key(&mut node, size, mask, y, x);
        node.extend(addr.to_le_bytes());
    }
    key(&mut node, 0, 0, SIDE, 0);
    store.extend_from_slice(&node);
    let layout = DataLayout::Chunked(ChunkedLayout {
        dims: vec![CHUNK, CHUNK],
        element_size: 4,
        flags: 0,
        index: ChunkIndex::BTreeV1 { address: root },
        layout_version: 3,
    });
    (store, layout, Some(pipeline))
}

fn chunked(b: &Bench, filtered: bool) {
    let (store, layout, pipeline) = synthetic_chunked(filtered);
    let space = simple_space(&[SIDE, SIDE]);
    let tag = if filtered {
        "chunked shuffle+gzip"
    } else {
        "chunked unfiltered"
    };
    println!(
        "# {tag}: 4096x4096 f32, 256x256 chunks, {} stored bytes",
        store.len()
    );
    let total = (SIDE * SIDE * 4) as usize;
    let storage = DatasetStorage::new(&store, Sizes::DEFAULT, &layout, &space, 4)
        .with_filters(pipeline.as_ref());
    b.run(&format!("{tag} full read"), total, || {
        black_box(storage.read(&Selection::All).unwrap());
    });
    let row = Selection::from_ranges(&[1000..1001, 0..SIDE]);
    b.run(&format!("{tag} one row"), (SIDE * 4) as usize, || {
        black_box(storage.read(&row).unwrap());
    });
    let col = Selection::from_ranges(&[0..SIDE, 1000..1001]);
    b.run(&format!("{tag} one column"), (SIDE * 4) as usize, || {
        black_box(storage.read(&col).unwrap());
    });
    let cache = ChunkCache::new(64 << 20);
    let cached = storage.with_chunk_cache(&cache);
    let mut rng = Rng(42);
    b.run(&format!("{tag} 500 8x8 boxes, cached"), 0, || {
        for _ in 0..500 {
            let y = rng.next() % 1024;
            let x = rng.next() % 1024;
            let sel = Selection::from_ranges(&[y..y + 8, x..x + 8]);
            black_box(cached.read(&sel).unwrap());
        }
    });
    b.run(&format!("{tag} 500 8x8 boxes, uncached"), 0, || {
        for _ in 0..500 {
            let y = rng.next() % 1024;
            let x = rng.next() % 1024;
            let sel = Selection::from_ranges(&[y..y + 8, x..x + 8]);
            black_box(storage.read(&sel).unwrap());
        }
    });
}

fn filters(b: &Bench) {
    let registry = FilterRegistry::with_builtin();
    let data: Vec<u8> = (0..(1u32 << 20))
        .flat_map(|i| ((i as f32 * 0.001).sin() * 1000.0).to_le_bytes())
        .collect();
    let len = data.len();
    for es in [2u32, 4, 8] {
        let f = registry.get(hdf5rs::filters::SHUFFLE).unwrap();
        let ctx = FilterContext::new(es as usize);
        let shuffled = f.encode(&[es], data.clone(), &ctx).unwrap();
        b.run(&format!("filter unshuffle 4MiB, {es}-byte"), len, || {
            black_box(f.decode(&[es], shuffled.clone(), &ctx).unwrap());
        });
    }
    let f = registry.get(hdf5rs::filters::DEFLATE).unwrap();
    let ctx = FilterContext::new(4).with_expected_size(Some(len));
    let z = f.encode(&[4], data.clone(), &ctx).unwrap();
    b.run("filter inflate 4MiB", len, || {
        black_box(f.decode(&[], z.clone(), &ctx).unwrap());
    });
}

fn main() {
    let b = Bench {
        filter: std::env::args().nth(1),
    };
    filters(&b);
    big_gzip(&b);
    contiguous(&b);
    chunked(&b, true);
    chunked(&b, false);
}
