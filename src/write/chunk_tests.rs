//! Round-trip tests of chunked, filtered and resizable datasets.

use super::*;
use crate::format::messages::layout::{ChunkIndex, DataLayout};
use crate::{File, H5Type};

fn opts(v: u8) -> WriteOptions {
    WriteOptions {
        superblock_version: v,
        ..WriteOptions::default()
    }
}

fn writer(v: u8) -> FileWriter<Vec<u8>> {
    FileWriter::new(Vec::new(), opts(v)).unwrap()
}

fn reopen(w: FileWriter<Vec<u8>>) -> File {
    File::from_bytes(w.finish().unwrap()).unwrap()
}

fn read<T: H5Type>(f: &File, path: &str) -> Vec<T> {
    f.dataset(path).unwrap().read::<T>().unwrap()
}

fn index_of(f: &File, path: &str) -> ChunkIndex {
    match f.dataset(path).unwrap().layout() {
        DataLayout::Chunked(c) => c.index,
        other => panic!("{path}: not chunked: {other:?}"),
    }
}

fn kind(i: &ChunkIndex) -> &'static str {
    match i {
        ChunkIndex::BTreeV1 { .. } => "btree1",
        ChunkIndex::SingleChunk { .. } => "single",
        ChunkIndex::Implicit { .. } => "implicit",
        ChunkIndex::FixedArray { .. } => "farray",
        ChunkIndex::ExtensibleArray { .. } => "earray",
        ChunkIndex::BTreeV2 { .. } => "btree2",
    }
}

fn iota(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

#[test]
fn every_index_type_round_trips() {
    for v in [0u8, 2, 3] {
        let mut w = writer(v);
        let mut root = w.root();
        // Single chunk.
        root.new_dataset::<i32>("single")
            .chunk(&[10])
            .write(&iota(10))
            .unwrap();
        // Implicit (early allocation, fixed size, no filters).
        root.new_dataset::<i32>("implicit")
            .shape(&[5, 7])
            .chunk(&[2, 3])
            .alloc_time(AllocTime::Early)
            .write(&iota(35))
            .unwrap();
        // Fixed array, small and paged (> 1024 chunks).
        root.new_dataset::<i32>("farray")
            .shape(&[6, 10])
            .chunk(&[4, 4])
            .write(&iota(60))
            .unwrap();
        root.new_dataset::<i32>("farray_paged")
            .chunk(&[2])
            .write(&iota(5001))
            .unwrap();
        // Extensible array with secondary blocks (unlimited middle dim).
        root.new_dataset::<i32>("earray")
            .shape(&[3, 700, 2])
            .max_shape(&[Some(3), None, Some(4)])
            .chunk(&[2, 1, 2])
            .write(&iota(4200))
            .unwrap();
        // v2 B-tree with several levels.
        root.new_dataset::<i32>("btree2")
            .shape(&[60, 70])
            .max_shape(&[None, None])
            .chunk(&[1, 1])
            .write(&iota(4200))
            .unwrap();
        let f = reopen(w);
        let expect = |name: &str, latest: &str| {
            let k = kind(&index_of(&f, name));
            let want = if v >= 3 { latest } else { "btree1" };
            assert_eq!(k, want, "v{v} {name}");
        };
        expect("single", "single");
        expect("implicit", "implicit");
        expect("farray", "farray");
        expect("farray_paged", "farray");
        expect("earray", "earray");
        expect("btree2", "btree2");
        assert_eq!(read::<i32>(&f, "single"), iota(10));
        assert_eq!(read::<i32>(&f, "implicit"), iota(35));
        assert_eq!(read::<i32>(&f, "farray"), iota(60));
        assert_eq!(read::<i32>(&f, "farray_paged"), iota(5001));
        assert_eq!(read::<i32>(&f, "earray"), iota(4200));
        assert_eq!(read::<i32>(&f, "btree2"), iota(4200));
        let d = f.dataset("earray").unwrap();
        assert_eq!(d.chunk_shape(), Some(&[2u64, 1, 2][..]));
        assert_eq!(d.max_shape(), vec![3, u64::MAX, 4]);
        // Edge chunks are stored at full size.
        for c in f.dataset("farray").unwrap().chunks().unwrap() {
            assert_eq!(c.size, 64);
        }
        // Hyperslab read across chunks.
        let sel = Selection::from_ranges(&[1..3, 300..302, 0..2]);
        let got = d.read_selection::<i32>(&sel).unwrap();
        let want: Vec<i32> = [1u64, 2]
            .iter()
            .flat_map(|&a| [300u64, 301].map(move |b| (a, b)))
            .flat_map(|(a, b)| [0u64, 1].map(move |c| (a * 1400 + b * 2 + c) as i32))
            .collect();
        assert_eq!(got, want);
    }
}

#[cfg(feature = "deflate")]
#[test]
fn small_offsets() {
    let mut w = FileWriter::new(
        Vec::new(),
        WriteOptions {
            superblock_version: 3,
            sizes: Sizes {
                offset: 4,
                length: 4,
            },
            track_times: false,
        },
    )
    .unwrap();
    let mut root = w.root();
    root.new_dataset::<f64>("ea")
        .max_shape(&[None])
        .chunk(&[3])
        .deflate(5)
        .write(&(0..500).map(f64::from).collect::<Vec<_>>())
        .unwrap();
    root.new_dataset::<u16>("bt")
        .shape(&[20, 20])
        .max_shape(&[None, None])
        .chunk(&[3, 3])
        .shuffle()
        .write(&(0..400u16).collect::<Vec<_>>())
        .unwrap();
    let f = reopen(w);
    assert_eq!(
        read::<f64>(&f, "ea"),
        (0..500).map(f64::from).collect::<Vec<_>>()
    );
    assert_eq!(read::<u16>(&f, "bt"), (0..400u16).collect::<Vec<_>>());
}

#[cfg(all(feature = "deflate", feature = "lzf"))]
#[test]
fn filter_combinations() {
    for v in [0u8, 2, 3] {
        let mut w = writer(v);
        let mut root = w.root();
        let data: Vec<i32> = (0..1000).map(|i| (i % 37) * 3).collect();
        root.new_dataset::<i32>("gzip")
            .chunk(&[128])
            .deflate(6)
            .write(&data)
            .unwrap();
        root.new_dataset::<i32>("shuf_gzip_f32")
            .chunk(&[100])
            .shuffle()
            .deflate(9)
            .fletcher32()
            .write(&data)
            .unwrap();
        root.new_dataset::<i32>("lzf")
            .chunk(&[256])
            .lzf()
            .write(&data)
            .unwrap();
        root.new_dataset::<i32>("nbit")
            .chunk(&[64])
            .nbit()
            .write(&data)
            .unwrap();
        root.new_dataset::<i32>("so_int")
            .chunk(&[64])
            .scale_offset(ScaleOffsetMode::Integer { min_bits: 0 })
            .write(&data)
            .unwrap();
        let floats: Vec<f32> = (0..300).map(|i| i as f32 * 0.25).collect();
        root.new_dataset::<f32>("so_float")
            .chunk(&[50])
            .scale_offset(ScaleOffsetMode::FloatDecimal { decimal_digits: 2 })
            .write(&floats)
            .unwrap();
        // Auto chunking from filters alone; single chunk when it fits.
        root.new_dataset::<i32>("auto")
            .deflate(1)
            .write(&data)
            .unwrap();
        // 2-D filtered with edge chunks.
        root.new_dataset::<i32>("edge2d")
            .shape(&[25, 40])
            .chunk(&[7, 9])
            .shuffle()
            .deflate(4)
            .write(&iota(1000))
            .unwrap();
        let f = reopen(w);
        for name in ["gzip", "shuf_gzip_f32", "lzf", "nbit", "so_int", "auto"] {
            assert_eq!(read::<i32>(&f, name), data, "v{v} {name}");
        }
        assert_eq!(read::<f32>(&f, "so_float"), floats);
        assert_eq!(read::<i32>(&f, "edge2d"), iota(1000));
        let d = f.dataset("shuf_gzip_f32").unwrap();
        let ids: Vec<u16> = d.filters().unwrap().filters.iter().map(|f| f.id).collect();
        assert_eq!(ids, [2, 1, 3]);
        // Compressed chunks are smaller than raw ones.
        for c in f.dataset("gzip").unwrap().chunks().unwrap() {
            assert!(c.size < 512, "{c:?}");
        }
        assert_eq!(
            f.dataset("auto").unwrap().chunk_shape(),
            Some(&[1000u64][..])
        );
    }
}

#[test]
#[cfg(feature = "lzf")]
fn optional_filter_failure_sets_mask() {
    for v in [0u8, 3] {
        let mut w = writer(v);
        // Incompressible data: LZF fails and is skipped for those chunks.
        let mut x = 0x1234_5678u32;
        let noise: Vec<u32> = (0..64)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x
            })
            .collect();
        let mut data = noise.clone();
        data.extend(std::iter::repeat_n(7u32, 64));
        w.root()
            .new_dataset::<u32>("mixed")
            .chunk(&[64])
            .lzf()
            .write(&data)
            .unwrap();
        let f = reopen(w);
        let d = f.dataset("mixed").unwrap();
        let masks: Vec<u32> = d.chunks().unwrap().iter().map(|c| c.filter_mask).collect();
        assert_eq!(masks, [1, 0]);
        assert_eq!(d.read::<u32>().unwrap(), data);
    }
}

#[cfg(feature = "deflate")]
#[test]
fn dont_filter_partial_chunks() {
    for v in [0u8, 3] {
        let mut w = writer(v);
        w.root()
            .new_dataset::<i32>("p")
            .chunk(&[4])
            .deflate(5)
            .dont_filter_partial_chunks()
            .write(&iota(10))
            .unwrap();
        let f = reopen(w);
        let d = f.dataset("p").unwrap();
        assert_eq!(d.read::<i32>().unwrap(), iota(10));
        let chunks = d.chunks().unwrap();
        assert_eq!(chunks.last().unwrap().size, 16);
        assert_eq!(chunks.last().unwrap().filter_mask, 1);
    }
}

#[cfg(feature = "zstd")]
#[test]
fn generic_filter_by_id() {
    let mut w = writer(3);
    let data: Vec<i64> = (0..2000).map(|i| i / 10).collect();
    w.root()
        .new_dataset::<i64>("z")
        .chunk(&[500])
        .filter(crate::filters::ZSTD, &[3])
        .write(&data)
        .unwrap();
    let f = reopen(w);
    assert_eq!(read::<i64>(&f, "z"), data);
}

#[test]
fn unknown_filter_is_rejected() {
    let mut w = writer(3);
    let mut root = w.root();
    let r = root.new_dataset::<i32>("x").filter(31999, &[]).write(&[1]);
    assert!(r.is_err());
}

#[test]
fn append_and_resize() {
    for v in [0u8, 2, 3] {
        let mut w = writer(v);
        let mut root = w.root();
        let mut ds = root
            .new_dataset::<i32>("log")
            .shape(&[0])
            .max_shape(&[None])
            .chunk(&[16])
            .fill_value(-1)
            .unwrap()
            .create()
            .unwrap();
        let mut expect = Vec::new();
        for i in 0..100 {
            let piece: Vec<i32> = (0..7).map(|k| i * 7 + k).collect();
            ds.append(&piece).unwrap();
            expect.extend(piece);
        }
        // Grow: new elements read as the fill value.
        ds.resize(&[710]).unwrap();
        expect.extend([-1; 10]);
        // 2-D table appended along rows.
        let mut t = root
            .new_dataset::<f64>("table")
            .shape(&[0, 3])
            .max_shape(&[None, Some(3)])
            .chunk(&[4, 3])
            .create()
            .unwrap();
        for r in 0..10 {
            t.append(&[r as f64, r as f64 + 0.5, -(r as f64)]).unwrap();
        }
        assert_eq!(t.shape().unwrap(), [10, 3]);
        assert_eq!(t.max_shape().unwrap(), [None, Some(3)]);
        let f = reopen(w);
        assert_eq!(read::<i32>(&f, "log"), expect, "v{v}");
        let table = read::<f64>(&f, "table");
        assert_eq!(table.len(), 30);
        assert_eq!(&table[27..], &[9.0, 9.5, -9.0]);
        if v >= 3 {
            assert_eq!(kind(&index_of(&f, "log")), "earray");
        }
    }
}

#[cfg(feature = "deflate")]
#[test]
fn shrink_then_grow_exposes_fill() {
    let mut w = writer(3);
    let mut root = w.root();
    let mut ds = root
        .new_dataset::<i32>("d")
        .shape(&[10, 10])
        .max_shape(&[None, None])
        .chunk(&[4, 4])
        .deflate(3)
        .fill_value(9)
        .unwrap()
        .write(&iota(100))
        .unwrap();
    ds.resize(&[3, 6]).unwrap();
    ds.resize(&[10, 10]).unwrap();
    let f = reopen(w);
    let got = read::<i32>(&f, "d");
    for (i, v) in got.iter().enumerate() {
        let (r, c) = (i / 10, i % 10);
        let want = if r < 3 && c < 6 { i as i32 } else { 9 };
        assert_eq!(*v, want, "({r},{c})");
    }
}

#[cfg(feature = "deflate")]
#[test]
fn hyperslab_writes() {
    for v in [0u8, 3] {
        for cache in [usize::MAX, 0] {
            let mut w = writer(v);
            w.set_chunk_cache_size(cache);
            let mut root = w.root();
            let mut ds = root
                .new_dataset::<u16>("h")
                .shape(&[20, 30])
                .chunk(&[6, 7])
                .deflate(2)
                .create()
                .unwrap();
            let mut expect = vec![0u16; 600];
            // Strided blocks: rows 1,5,9 (blocks of 2), cols 3..13.
            let sel = Selection::hyperslab(vec![
                crate::read::HyperslabDim::new(1, 4, 3, 2),
                crate::read::HyperslabDim::range(3..13),
            ]);
            let vals: Vec<u16> = (0..60).map(|i| 1000 + i).collect();
            ds.write_selection(&sel, &vals).unwrap();
            let mut k = 0;
            for r in [1, 2, 5, 6, 9, 10] {
                for c in 3..13 {
                    expect[r * 30 + c] = vals[k];
                    k += 1;
                }
            }
            // Overwrite a single element twice, then a full row.
            ds.write_selection(&Selection::from_ranges(&[5..6, 4..5]), &[7u16])
                .unwrap();
            ds.write_selection(&Selection::from_ranges(&[5..6, 4..5]), &[8u16])
                .unwrap();
            expect[5 * 30 + 4] = 8;
            let row: Vec<u16> = (0..30).collect();
            ds.write_selection(&Selection::from_ranges(&[19..20, 0..30]), &row)
                .unwrap();
            expect[570..600].copy_from_slice(&row);
            let f = reopen(w);
            assert_eq!(read::<u16>(&f, "h"), expect, "v{v} cache {cache}");
        }
    }
}

#[test]
fn partial_writes_other_layouts() {
    let mut w = writer(2);
    let mut root = w.root();
    let mut c = root
        .new_dataset::<i32>("contig")
        .shape(&[4, 5])
        .fill_value(3)
        .unwrap()
        .create()
        .unwrap();
    c.write_selection(&Selection::from_ranges(&[1..3, 2..4]), &[1, 2, 3, 4])
        .unwrap();
    let mut k = root
        .new_dataset::<i32>("compact")
        .shape(&[3])
        .compact()
        .write(&[1, 2, 3])
        .unwrap();
    k.write_selection(&Selection::range(2..3), &[30]).unwrap();
    assert!(k.resize(&[4]).is_err());
    let f = reopen(w);
    let mut expect = vec![3; 20];
    expect[7] = 1;
    expect[8] = 2;
    expect[12] = 3;
    expect[13] = 4;
    assert_eq!(read::<i32>(&f, "contig"), expect);
    assert_eq!(read::<i32>(&f, "compact"), [1, 2, 30]);
}

#[test]
fn variable_length_chunked() {
    let mut w = writer(3);
    let mut root = w.root();
    let mut ds = root
        .new_dataset::<String>("names")
        .shape(&[0])
        .max_shape(&[None])
        .chunk(&[2])
        .create()
        .unwrap();
    ds.append(&["a".to_string(), "bb".into(), "ccc".into()])
        .unwrap();
    ds.append(&["dddd".to_string()]).unwrap();
    let f = reopen(w);
    assert_eq!(read::<String>(&f, "names"), ["a", "bb", "ccc", "dddd"]);
}

#[test]
fn early_allocation_and_fill() {
    for v in [0u8, 3] {
        let mut w = writer(v);
        let mut root = w.root();
        root.new_dataset::<i16>("e")
            .shape(&[9])
            .chunk(&[4])
            .alloc_time(AllocTime::Early)
            .fill_value(-5)
            .unwrap()
            .create()
            .unwrap();
        root.new_dataset::<i16>("lazy")
            .shape(&[9])
            .chunk(&[4])
            .fill_value(4)
            .unwrap()
            .create()
            .unwrap();
        let f = reopen(w);
        assert_eq!(read::<i16>(&f, "e"), [-5; 9]);
        assert_eq!(f.dataset("e").unwrap().chunks().unwrap().len(), 3);
        assert_eq!(read::<i16>(&f, "lazy"), [4; 9]);
        assert!(f.dataset("lazy").unwrap().chunks().unwrap().is_empty());
    }
}

#[test]
fn invalid_chunk_requests() {
    let mut w = writer(3);
    let mut root = w.root();
    assert!(
        root.new_dataset::<i32>("a")
            .chunk(&[0])
            .write(&[1])
            .is_err()
    );
    assert!(
        root.new_dataset::<i32>("b")
            .chunk(&[2, 2])
            .write(&[1])
            .is_err()
    );
    assert!(
        root.new_dataset::<i32>("c")
            .chunk(&[5])
            .write(&[1, 2])
            .is_err()
    );
    assert!(
        root.new_dataset::<i32>("d")
            .shape(&[])
            .chunk(&[1])
            .write(&[1])
            .is_err()
    );
    assert!(
        root.new_dataset::<i32>("e")
            .compact()
            .deflate(1)
            .write(&[1])
            .is_err()
    );
    let mut ds = root
        .new_dataset::<i32>("f")
        .max_shape(&[Some(4)])
        .write(&[1, 2])
        .unwrap();
    assert!(ds.resize(&[5]).is_err());
    assert!(ds.append(&[3]).is_err());
    ds.resize(&[4]).unwrap();
}

#[test]
fn file_storage_round_trip() {
    let dir = std::env::temp_dir().join(format!("hdf5rs-chunk-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.h5");
    {
        let mut w = FileWriter::create_with(&path, opts(3)).unwrap();
        w.set_chunk_cache_size(0);
        let mut root = w.root();
        let mut ds = root
            .new_dataset::<u8>("bytes")
            .shape(&[0])
            .max_shape(&[None])
            .chunk(&[100])
            .create()
            .unwrap();
        for i in 0..50u8 {
            ds.append(&[i; 3]).unwrap();
        }
        w.finish().unwrap();
    }
    let f = File::open(&path).unwrap();
    let got = read::<u8>(&f, "bytes");
    assert_eq!(got.len(), 150);
    assert!(got.chunks(3).enumerate().all(|(i, c)| c == [i as u8; 3]));
    std::fs::remove_dir_all(&dir).unwrap();
}
