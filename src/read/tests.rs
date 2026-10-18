//! Unit tests of the reading engine on synthetic in-memory files.

use super::*;
use crate::checksum::metadata_checksum;
use crate::filters::deflate::zlib_compress;
use crate::filters::shuffle::shuffle;
use crate::format::UNDEFINED_ADDRESS;
use crate::format::messages::dataspace::{DataspaceKind, UNLIMITED};
use crate::format::messages::external_files::ExternalFileSlot;
use crate::format::messages::fill_value::{AllocTime, FillWriteTime};
use crate::format::messages::filter_pipeline::FilterInfo;
use crate::format::messages::layout::{
    ChunkIndex, ChunkedLayout, DONT_FILTER_PARTIAL_BOUND_CHUNKS, FilteredChunk,
    SINGLE_INDEX_WITH_FILTER,
};
use crate::format::test_util::{FileBuilder, Rng, mutate};

const S: Sizes = Sizes::DEFAULT;
const E: usize = 4;

/// Turns raw chunk bytes (and the chunk number) into stored bytes and a
/// filter mask.
type Encoder = dyn Fn(&[u8], usize) -> (Vec<u8>, u32);

// ------------------------------------------------------------ helpers --

fn space(dims: &[u64]) -> Dataspace {
    Dataspace {
        version: 2,
        kind: DataspaceKind::Simple,
        dims: dims.to_vec(),
        max_dims: None,
        permutation: None,
    }
}

fn space_max(dims: &[u64], max: &[u64]) -> Dataspace {
    Dataspace {
        max_dims: Some(max.to_vec()),
        ..space(dims)
    }
}

fn scalar_space(kind: DataspaceKind) -> Dataspace {
    Dataspace { kind, ..space(&[]) }
}

fn nelem(dims: &[u64]) -> usize {
    dims.iter().product::<u64>() as usize
}

/// Element `i` of the test arrays: `i * 3 + 1` as little-endian u32.
fn full_array(dims: &[u64]) -> Vec<u8> {
    (0..nelem(dims) as u32)
        .flat_map(|i| (i * 3 + 1).to_le_bytes())
        .collect()
}

fn strides(dims: &[u64]) -> Vec<u64> {
    let mut s = vec![1u64; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

/// The full-size on-disk chunk at `scaled`, with garbage (0xEE) in the
/// positions beyond the dataset edge.
fn chunk_of(full: &[u8], dims: &[u64], cd: &[u64], scaled: &[u64]) -> Vec<u8> {
    let n = nelem(cd);
    let ds = strides(dims);
    let cs = strides(cd);
    let mut out = vec![0xee; n * E];
    for k in 0..n as u64 {
        let mut lin = 0u64;
        let mut inside = true;
        for d in 0..dims.len() {
            let c = (k / cs[d]) % cd[d] + scaled[d] * cd[d];
            inside &= c < dims[d];
            lin += c * ds[d];
        }
        if inside {
            let (k, lin) = (k as usize, lin as usize);
            out[k * E..k * E + E].copy_from_slice(&full[lin * E..lin * E + E]);
        }
    }
    out
}

/// Overwrite the in-bounds part of chunk `scaled` in `full` with `fill`.
fn blank_chunk(full: &mut [u8], dims: &[u64], cd: &[u64], scaled: &[u64], fill: [u8; 4]) {
    let ds = strides(dims);
    for lin in 0..nelem(dims) as u64 {
        let inside = (0..dims.len()).all(|d| (lin / ds[d]) % dims[d] / cd[d] == scaled[d]);
        if inside {
            let l = lin as usize;
            full[l * E..l * E + E].copy_from_slice(&fill);
        }
    }
}

fn grid(dims: &[u64], cd: &[u64]) -> Vec<Vec<u64>> {
    let n: Vec<u64> = dims.iter().zip(cd).map(|(d, c)| d.div_ceil(*c)).collect();
    let total: u64 = n.iter().product();
    (0..total)
        .map(|mut i| {
            let mut v = vec![0; n.len()];
            for d in (0..n.len()).rev() {
                v[d] = i % n[d];
                i /= n[d];
            }
            v
        })
        .collect()
}

/// Naive reference extraction of a hyperslab from a full array.
fn reference(full: &[u8], dims: &[u64], sel: &[HyperslabDim]) -> Vec<u8> {
    let coords: Vec<Vec<u64>> = sel
        .iter()
        .map(|h| {
            (0..h.count)
                .flat_map(|k| (0..h.block).map(move |b| h.start + k * h.stride + b))
                .collect()
        })
        .collect();
    let total: usize = coords.iter().map(Vec::len).product();
    let ds = strides(dims);
    let mut out = Vec::with_capacity(total * E);
    for mut i in 0..total {
        let mut lin = 0u64;
        let mut pos = vec![0; dims.len()];
        for d in (0..dims.len()).rev() {
            pos[d] = coords[d][i % coords[d].len()];
            i /= coords[d].len();
        }
        for d in 0..dims.len() {
            lin += pos[d] * ds[d];
        }
        let l = lin as usize;
        out.extend_from_slice(&full[l * E..l * E + E]);
    }
    out
}

fn random_hyperslab(rng: &mut Rng, dims: &[u64]) -> Vec<HyperslabDim> {
    dims.iter()
        .map(|&d| {
            let start = rng.below(d as usize) as u64;
            let block = 1 + rng.below((d - start).min(5) as usize) as u64;
            let stride = block + rng.below(5) as u64;
            let max_count = (d - start - block) / stride + 1;
            let count = 1 + rng.below(max_count as usize) as u64;
            HyperslabDim::new(start, stride, count, block)
        })
        .collect()
}

/// Read everything and many random hyperslabs, comparing with `expected`.
fn check(st: &DatasetStorage<'_>, expected: &[u8]) {
    assert_eq!(st.read(&Selection::All).unwrap(), expected, "full read");
    let dims = st.dataspace().dims.clone();
    let mut rng = Rng(0x5eed_0123_4567_89ab ^ expected.len() as u64);
    for _ in 0..60 {
        let h = random_hyperslab(&mut rng, &dims);
        let got = st.read(&Selection::hyperslab(h.clone())).unwrap();
        assert_eq!(got, reference(expected, &dims, &h), "{h:?}");
    }
}

fn fill_v3(value: &[u8]) -> FillValue {
    FillValue {
        version: 3,
        alloc_time: AllocTime::Late,
        write_time: FillWriteTime::IfSet,
        state: FillValueState::Defined,
        value: Some(value.to_vec()),
    }
}

fn filter(id: u16, cd: &[u32]) -> FilterInfo {
    FilterInfo {
        id,
        name: None,
        optional: false,
        client_data: cd.to_vec(),
    }
}

fn shuffle_deflate() -> FilterPipeline {
    FilterPipeline {
        filters: vec![filter(2, &[E as u32]), filter(1, &[6])],
    }
}

fn encode_sd(raw: &[u8]) -> Vec<u8> {
    zlib_compress(&shuffle(raw.to_vec(), E), 6).unwrap()
}

fn chunked(dims: &[u64], index: ChunkIndex, flags: u8) -> DataLayout {
    DataLayout::Chunked(ChunkedLayout {
        dims: dims.to_vec(),
        element_size: E as u64,
        flags,
        index,
    })
}

fn put_sealed(f: &mut FileBuilder, addr: u64, mut bytes: Vec<u8>) {
    let c = metadata_checksum(&bytes);
    bytes.extend_from_slice(&c.to_le_bytes());
    f.put(addr, &bytes);
}

// ------------------------------------------------- compact/contiguous --

#[test]
fn compact_layouts() {
    let dims = [5, 6];
    let full = full_array(&dims);
    let layout = DataLayout::Compact {
        data: full.clone(),
        dims: None,
    };
    let sp = space(&dims);
    let store: Vec<u8> = Vec::new();
    check(&DatasetStorage::new(&store, S, &layout, &sp, E), &full);

    let one = DataLayout::Compact {
        data: vec![1, 2, 3, 4],
        dims: None,
    };
    let sc = scalar_space(DataspaceKind::Scalar);
    let st = DatasetStorage::new(&store, S, &one, &sc, E);
    assert_eq!(st.read(&Selection::All).unwrap(), [1, 2, 3, 4]);
    assert!(st.read(&Selection::range(0..1)).is_err());
    let null = scalar_space(DataspaceKind::Null);
    let st = DatasetStorage::new(&store, S, &one, &null, E);
    assert!(st.read(&Selection::All).unwrap().is_empty());
    // too little data
    let short = DataLayout::Compact {
        data: vec![0; 10],
        dims: None,
    };
    assert!(
        DatasetStorage::new(&store, S, &short, &sp, E)
            .read(&Selection::All)
            .is_err()
    );
    assert!(
        DatasetStorage::new(&store, S, &layout, &sp, E)
            .chunks()
            .is_err()
    );
}

#[test]
fn contiguous_with_coalescing() {
    // Rows of 16 KiB so that strided reads cross the coalescing gap.
    let dims = [24, 4096];
    let full = full_array(&dims);
    let mut f = FileBuilder::new(S);
    f.put(100, &full);
    let layout = DataLayout::Contiguous {
        address: 100,
        size: Some(full.len() as u64),
        dims: None,
    };
    let sp = space(&dims);
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E);
    check(&st, &full);
    for sel in [
        vec![
            HyperslabDim::new(0, 5, 5, 1),
            HyperslabDim::new(7, 1000, 4, 3),
        ],
        vec![HyperslabDim::range(3..4), HyperslabDim::range(0..4096)],
        vec![
            HyperslabDim::new(1, 2, 10, 1),
            HyperslabDim::range(100..200),
        ],
    ] {
        assert_eq!(
            st.read(&Selection::hyperslab(sel.clone())).unwrap(),
            reference(&full, &dims, &sel)
        );
    }
    // Declared size too small / file too short.
    let small = DataLayout::Contiguous {
        address: 100,
        size: Some(10),
        dims: None,
    };
    assert!(
        DatasetStorage::new(&f.data, S, &small, &sp, E)
            .read(&Selection::All)
            .is_err()
    );
    let past = DataLayout::Contiguous {
        address: 200,
        size: None,
        dims: None,
    };
    assert!(
        DatasetStorage::new(&f.data, S, &past, &sp, E)
            .read(&Selection::All)
            .is_err()
    );
}

#[test]
fn unwritten_contiguous_reads_fill_value() {
    let layout = DataLayout::Contiguous {
        address: UNDEFINED_ADDRESS,
        size: Some(40),
        dims: None,
    };
    let sp = space(&[10]);
    let store: Vec<u8> = Vec::new();
    let st = DatasetStorage::new(&store, S, &layout, &sp, E);
    assert_eq!(st.read(&Selection::All).unwrap(), vec![0; 40]);
    let fill = fill_v3(&(-5i32).to_le_bytes());
    let st = st.with_fill_value(Some(&fill));
    let got = st.read(&Selection::range(2..5)).unwrap();
    assert_eq!(got, [(-5i32).to_le_bytes(); 3].concat());
    // Old-style fill value applies when no new message defines one.
    let old = OldFillValue {
        value: 7u32.to_le_bytes().to_vec(),
    };
    let st = DatasetStorage::new(&store, S, &layout, &sp, E).with_old_fill_value(Some(&old));
    assert_eq!(
        st.read(&Selection::All).unwrap(),
        [7u32.to_le_bytes(); 10].concat()
    );
    let undefined = FillValue {
        state: FillValueState::Undefined,
        value: None,
        ..fill_v3(&[])
    };
    let st = st.with_fill_value(Some(&undefined));
    assert_eq!(st.read(&Selection::All).unwrap(), vec![0; 40]);
    // Fill value of the wrong size.
    let bad = fill_v3(&[1, 2]);
    let st = DatasetStorage::new(&store, S, &layout, &sp, E).with_fill_value(Some(&bad));
    assert!(st.read(&Selection::All).is_err());
}

#[test]
fn external_files() {
    let full = full_array(&[12]);
    let mut f = FileBuilder::new(S);
    f.local_heap(0x10, 0x40, b"\0a.bin\0b.bin\0", u64::MAX);
    let ext = ExternalFiles {
        heap_address: 0x10,
        allocated_slots: 2,
        slots: vec![
            ExternalFileSlot {
                name_offset: 1,
                file_offset: 3,
                size: 20,
            },
            ExternalFileSlot {
                name_offset: 7,
                file_offset: 0,
                size: UNLIMITED,
            },
        ],
    };
    let mut a = vec![9u8; 3];
    a.extend_from_slice(&full[..20]);
    let b = full[20..].to_vec();
    let resolver = move |name: &str| -> Result<Box<dyn ReadAt>> {
        match name {
            "a.bin" => Ok(Box::new(a.clone())),
            "b.bin" => Ok(Box::new(b.clone())),
            _ => Err(Error::NotFound(name.into())),
        }
    };
    let layout = DataLayout::Contiguous {
        address: UNDEFINED_ADDRESS,
        size: Some(48),
        dims: None,
    };
    let sp = space(&[12]);
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E);
    let no_resolver = st.with_external(Some(&ext), None);
    assert!(matches!(
        no_resolver.read(&Selection::All),
        Err(Error::Unsupported(_))
    ));
    let st = st.with_external(Some(&ext), Some(&resolver));
    check(&st, &full);
    // A file shorter than its reserved size reads zeros past its end.
    let mut ext2 = ext.clone();
    ext2.slots[1].size = 40;
    let st2 = DatasetStorage::new(&f.data, S, &layout, &sp, E)
        .with_external(Some(&ext2), Some(&resolver));
    assert_eq!(st2.read(&Selection::All).unwrap(), full);
    let missing = |_: &str| -> Result<Box<dyn ReadAt>> { Err(Error::NotFound("x".into())) };
    let st3 =
        DatasetStorage::new(&f.data, S, &layout, &sp, E).with_external(Some(&ext), Some(&missing));
    assert!(st3.read(&Selection::All).is_err());
}

// ---------------------------------------------------------- B-tree v1 --

/// Build a one-leaf chunk B-tree at 0x40 for the chunks of `dims` / `cd`,
/// skipping `missing`. `encode` turns raw chunk bytes into (stored bytes,
/// filter mask). Returns the file and the expected full array.
fn btree_v1_file(
    dims: &[u64],
    cd: &[u64],
    missing: &[Vec<u64>],
    fill: [u8; 4],
    encode: &Encoder,
) -> (FileBuilder, Vec<u8>) {
    let full = full_array(dims);
    let mut expected = full.clone();
    let mut f = FileBuilder::new(S);
    let mut keys = Vec::new();
    let mut children = Vec::new();
    let mut addr = 0x10000u64;
    for (i, s) in grid(dims, cd).iter().enumerate() {
        if missing.contains(s) {
            blank_chunk(&mut expected, dims, cd, s, fill);
            continue;
        }
        let (bytes, mask) = encode(&chunk_of(&full, dims, cd, s), i);
        let off: Vec<u64> = s.iter().zip(cd).map(|(a, b)| a * b).collect();
        keys.push(FileBuilder::chunk_key(bytes.len() as u32, mask, &off));
        children.push(addr);
        f.put(addr, &bytes);
        addr += bytes.len() as u64 + 16;
    }
    keys.push(FileBuilder::chunk_key(0, 0, dims));
    f.btree(0x40, 1, 0, &keys, &children);
    (f, expected)
}

#[test]
fn btree_v1_edges_and_missing_chunks() {
    let dims = [7, 10];
    let cd = [3, 4];
    let fill = (-1i32).to_le_bytes();
    let missing = [vec![1, 1], vec![2, 2]];
    let (f, expected) = btree_v1_file(&dims, &cd, &missing, fill, &|c, _| (c.to_vec(), 0));
    let layout = chunked(&cd, ChunkIndex::BTreeV1 { address: 0x40 }, 0);
    let sp = space(&dims);
    let fv = fill_v3(&fill);
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E).with_fill_value(Some(&fv));
    check(&st, &expected);
    let chunks = st.chunks().unwrap();
    assert_eq!(chunks.len(), 7);
    assert_eq!(chunks[0].offset, vec![0, 0]);
    assert_eq!(chunks[1].scaled, vec![0, 1]);
    assert_eq!(chunks[1].offset, vec![0, 4]);
    assert!(chunks.iter().all(|c| c.size == 48 && c.filter_mask == 0));
    // Undefined index address: everything reads as fill.
    let empty = chunked(
        &cd,
        ChunkIndex::BTreeV1 {
            address: UNDEFINED_ADDRESS,
        },
        0,
    );
    let st = DatasetStorage::new(&f.data, S, &empty, &sp, E).with_fill_value(Some(&fv));
    assert_eq!(st.read(&Selection::All).unwrap(), [fill; 70].concat());
    assert!(st.chunks().unwrap().is_empty());
}

#[test]
fn btree_v1_many_chunks_uses_full_scan() {
    let dims = [40, 30];
    let cd = [2, 3];
    let missing = [vec![3, 4], vec![19, 9]];
    let (f, expected) = btree_v1_file(&dims, &cd, &missing, [0; 4], &|c, _| (c.to_vec(), 0));
    let layout = chunked(&cd, ChunkIndex::BTreeV1 { address: 0x40 }, 0);
    let sp = space(&dims);
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E);
    check(&st, &expected);
    assert_eq!(st.chunks().unwrap().len(), 198);
}

#[test]
fn btree_v1_shuffle_deflate_with_masks() {
    let dims = [9, 11];
    let cd = [4, 4];
    // Every third chunk skips deflate (bit 1) and is stored shuffled only.
    let encode = |c: &[u8], i: usize| {
        if i.is_multiple_of(3) {
            (shuffle(c.to_vec(), E), 0b10)
        } else {
            (encode_sd(c), 0)
        }
    };
    let (f, expected) = btree_v1_file(&dims, &cd, &[vec![2, 0]], [0; 4], &encode);
    let layout = chunked(&cd, ChunkIndex::BTreeV1 { address: 0x40 }, 0);
    let sp = space(&dims);
    let pl = shuffle_deflate();
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E).with_filters(Some(&pl));
    check(&st, &expected);
    let chunks = st.chunks().unwrap();
    assert_eq!(chunks[0].filter_mask, 0b10);
    assert!(chunks[1].size < 64);
    // Without the pipeline the chunk bytes are taken as-is: wrong data,
    // but no error. With an empty registry the filters are unavailable.
    let empty = FilterRegistry::new();
    let st2 = st.with_registry(&empty);
    assert!(matches!(
        st2.read(&Selection::All),
        Err(Error::Unsupported(_))
    ));
    // A pipeline that does not restore the chunk size is an error.
    let only_deflate = FilterPipeline {
        filters: vec![filter(1, &[])],
    };
    let (f2, _) = btree_v1_file(&dims, &cd, &[], [0; 4], &|c, _| {
        (zlib_compress(&c[..c.len() - 4], 6).unwrap(), 0)
    });
    let st3 = DatasetStorage::new(&f2.data, S, &layout, &sp, E).with_filters(Some(&only_deflate));
    assert!(matches!(
        st3.read(&Selection::All),
        Err(Error::InvalidFormat(_))
    ));
}

// ------------------------------------------------------- v4 indexes --

#[test]
fn implicit_index() {
    let dims = [10, 10];
    let cd = [3, 4];
    let full = full_array(&dims);
    let mut f = FileBuilder::new(S);
    let mut addr = 0x100;
    for s in grid(&dims, &cd) {
        let c = chunk_of(&full, &dims, &cd, &s);
        f.put(addr, &c);
        addr += c.len() as u64;
    }
    let layout = chunked(&cd, ChunkIndex::Implicit { address: 0x100 }, 0);
    let sp = space(&dims);
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E);
    check(&st, &full);
    let chunks = st.chunks().unwrap();
    assert_eq!(chunks.len(), 12);
    assert_eq!(chunks[5].address, 0x100 + 5 * 48);
    assert_eq!(chunks[5].scaled, vec![1, 2]);
}

#[test]
fn single_chunk_index() {
    let dims = [4, 5];
    let full = full_array(&dims);
    let mut f = FileBuilder::new(S);
    f.put(0x80, &full);
    let sp = space(&dims);
    let layout = chunked(
        &dims,
        ChunkIndex::SingleChunk {
            address: 0x80,
            filtered: None,
        },
        0,
    );
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E);
    check(&st, &full);
    assert_eq!(st.chunks().unwrap().len(), 1);

    let z = encode_sd(&full);
    f.put(0x200, &z);
    let layout = chunked(
        &dims,
        ChunkIndex::SingleChunk {
            address: 0x200,
            filtered: Some(FilteredChunk {
                size: z.len() as u64,
                filter_mask: 0,
            }),
        },
        SINGLE_INDEX_WITH_FILTER,
    );
    let pl = shuffle_deflate();
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E).with_filters(Some(&pl));
    check(&st, &full);
    assert_eq!(st.chunks().unwrap()[0].size, z.len() as u64);
    // Filtered dataset whose single chunk carries no size.
    let layout = chunked(
        &dims,
        ChunkIndex::SingleChunk {
            address: 0x200,
            filtered: None,
        },
        0,
    );
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E).with_filters(Some(&pl));
    assert!(st.read(&Selection::All).is_err());
}

/// Write a fixed array (header at 0x40, data block at 0x80) holding
/// `entries` (`None` = unallocated; `Some((addr, size, mask))`).
fn fixed_array(f: &mut FileBuilder, filtered: bool, entries: &[Option<(u64, u64, u32)>]) {
    let esize: u8 = if filtered { 8 + 4 + 4 } else { 8 };
    let mut h = b"FAHD".to_vec();
    h.extend([0, u8::from(filtered), esize, 10]);
    h.extend((entries.len() as u64).to_le_bytes());
    h.extend(0x80u64.to_le_bytes());
    put_sealed(f, 0x40, h);
    let mut d = b"FADB".to_vec();
    d.extend([0, u8::from(filtered)]);
    d.extend(0x40u64.to_le_bytes());
    for e in entries {
        let (a, s, m) = e.unwrap_or((u64::MAX, 0, 0));
        d.extend(a.to_le_bytes());
        if filtered {
            d.extend((s as u32).to_le_bytes());
            d.extend(m.to_le_bytes());
        }
    }
    put_sealed(f, 0x80, d);
}

#[test]
fn fixed_array_index() {
    let dims = [10, 10];
    let cd = [3, 4];
    let full = full_array(&dims);
    let mut expected = full.clone();
    let fill = 0x0102_0304u32.to_le_bytes();
    let mut f = FileBuilder::new(S);
    let mut entries = Vec::new();
    let mut entries_z = Vec::new();
    let mut addr = 0x1000;
    for s in grid(&dims, &cd) {
        if s == [2, 1] {
            entries.push(None);
            entries_z.push(None);
            blank_chunk(&mut expected, &dims, &cd, &s, fill);
            continue;
        }
        let c = chunk_of(&full, &dims, &cd, &s);
        f.put(addr, &c);
        entries.push(Some((addr, 0, 0)));
        addr += c.len() as u64;
        let z = encode_sd(&c);
        f.put(addr, &z);
        entries_z.push(Some((addr, z.len() as u64, 0)));
        addr += z.len() as u64;
    }
    let sp = space(&dims);
    let fv = fill_v3(&fill);
    let layout = chunked(
        &cd,
        ChunkIndex::FixedArray {
            address: 0x40,
            page_bits: 10,
        },
        0,
    );
    let mut plain = f.clone();
    fixed_array(&mut plain, false, &entries);
    let st = DatasetStorage::new(&plain.data, S, &layout, &sp, E).with_fill_value(Some(&fv));
    check(&st, &expected);
    assert_eq!(st.chunks().unwrap().len(), 11);
    // The filtered variant.
    let mut z = f.clone();
    fixed_array(&mut z, true, &entries_z);
    let pl = shuffle_deflate();
    let st = DatasetStorage::new(&z.data, S, &layout, &sp, E)
        .with_fill_value(Some(&fv))
        .with_filters(Some(&pl));
    check(&st, &expected);
    // Filter pipeline / client ID disagreement.
    let st = DatasetStorage::new(&plain.data, S, &layout, &sp, E).with_filters(Some(&pl));
    assert!(st.read(&Selection::All).is_err());
    // Array size must match the chunk grid of the maximum dimensions.
    let grown = space_max(&dims, &[10, 20]);
    let st = DatasetStorage::new(&plain.data, S, &layout, &grown, E);
    assert!(st.read(&Selection::All).is_err());
}

#[test]
fn fixed_array_uses_maximum_dimensions() {
    // dims [4, 5] with max [6, 9], chunks [2, 3]: max grid 3 x 3.
    let dims = [4, 5];
    let cd = [2, 3];
    let full = full_array(&dims);
    let mut f = FileBuilder::new(S);
    let mut entries = vec![None; 9];
    let mut addr = 0x1000;
    for s in grid(&dims, &cd) {
        let c = chunk_of(&full, &dims, &cd, &s);
        f.put(addr, &c);
        entries[(s[0] * 3 + s[1]) as usize] = Some((addr, 0, 0));
        addr += c.len() as u64;
    }
    fixed_array(&mut f, false, &entries);
    let sp = space_max(&dims, &[6, 9]);
    let layout = chunked(
        &cd,
        ChunkIndex::FixedArray {
            address: 0x40,
            page_bits: 10,
        },
        0,
    );
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E);
    check(&st, &full);
    let scaled: Vec<_> = st.chunks().unwrap().into_iter().map(|c| c.scaled).collect();
    assert_eq!(scaled, grid(&dims, &cd));
}

#[test]
fn dont_filter_partial_bound_chunks() {
    let dims = [5, 7];
    let cd = [2, 3];
    let full = full_array(&dims);
    let mut f = FileBuilder::new(S);
    let mut entries = Vec::new();
    let mut addr = 0x1000;
    for s in grid(&dims, &cd) {
        let c = chunk_of(&full, &dims, &cd, &s);
        let partial = (s[0] + 1) * cd[0] > dims[0] || (s[1] + 1) * cd[1] > dims[1];
        let stored = if partial { c } else { encode_sd(&c) };
        f.put(addr, &stored);
        entries.push(Some((addr, stored.len() as u64, 0)));
        addr += stored.len() as u64;
    }
    fixed_array(&mut f, true, &entries);
    let sp = space(&dims);
    let pl = shuffle_deflate();
    let layout = chunked(
        &cd,
        ChunkIndex::FixedArray {
            address: 0x40,
            page_bits: 10,
        },
        DONT_FILTER_PARTIAL_BOUND_CHUNKS,
    );
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E).with_filters(Some(&pl));
    check(&st, &full);
    let layout = chunked(
        &cd,
        ChunkIndex::FixedArray {
            address: 0x40,
            page_bits: 10,
        },
        0,
    );
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E).with_filters(Some(&pl));
    assert!(st.read(&Selection::All).is_err());
}

/// Extensible array with all elements inline in the index block (8
/// slots): header at 0x40, index block at 0x100.
fn extensible_array(f: &mut FileBuilder, records: &[Option<u64>]) {
    assert!(records.len() <= 8);
    let mut h = b"EAHD".to_vec();
    // version, client, element size, max bits, index elems, dblk min
    // elems, sblk min ptrs, page bits
    h.extend([0, 0, 8, 32, 8, 16, 4, 10]);
    for v in [0u64, 0, 0, 0, records.len() as u64, records.len() as u64] {
        h.extend(v.to_le_bytes());
    }
    h.extend(0x100u64.to_le_bytes());
    put_sealed(f, 0x40, h);
    let mut b = b"EAIB".to_vec();
    b.extend([0, 0]);
    b.extend(0x40u64.to_le_bytes());
    for i in 0..8 {
        let a = records.get(i).copied().flatten().unwrap_or(u64::MAX);
        b.extend(a.to_le_bytes());
    }
    // 2 * (4 - 1) data block addresses, 1 + 32 - 4 - 4 secondary blocks.
    for _ in 0..(6 + 25) {
        b.extend(u64::MAX.to_le_bytes());
    }
    put_sealed(f, 0x100, b);
}

fn ea_index() -> ChunkIndex {
    ChunkIndex::ExtensibleArray {
        address: 0x40,
        max_bits: 32,
        index_elements: 8,
        min_pointers: 4,
        min_elements: 16,
        page_bits: 10,
    }
}

#[test]
fn extensible_array_index_1d_and_2d() {
    // 1-D: dims [9], chunks [2] -> 5 chunks; chunk 3 missing.
    let dims = [9];
    let cd = [2];
    let full = full_array(&dims);
    let mut expected = full.clone();
    let mut f = FileBuilder::new(S);
    let mut recs = Vec::new();
    for s in grid(&dims, &cd) {
        if s == [3] {
            recs.push(None);
            blank_chunk(&mut expected, &dims, &cd, &s, [0; 4]);
            continue;
        }
        let a = 0x1000 + s[0] * 8;
        f.put(a, &chunk_of(&full, &dims, &cd, &s));
        recs.push(Some(a));
    }
    extensible_array(&mut f, &recs);
    let sp = space_max(&dims, &[UNLIMITED]);
    let layout = chunked(&cd, ea_index(), 0);
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E);
    check(&st, &expected);
    assert_eq!(st.chunks().unwrap().len(), 4);

    // 2-D with the unlimited dimension last: dims [3, 5], max [4, unl],
    // chunks [2, 2]. Fixed grid (from the maximum) has 2 rows; the linear
    // index is s1 * 2 + s0.
    let dims = [3, 5];
    let cd = [2, 2];
    let full = full_array(&dims);
    let mut f = FileBuilder::new(S);
    let mut recs = vec![None; 6];
    for s in grid(&dims, &cd) {
        let li = (s[1] * 2 + s[0]) as usize;
        let a = 0x1000 + li as u64 * 16;
        f.put(a, &chunk_of(&full, &dims, &cd, &s));
        recs[li] = Some(a);
    }
    extensible_array(&mut f, &recs);
    let layout = chunked(&cd, ea_index(), 0);
    let sp = space_max(&dims, &[4, UNLIMITED]);
    let st = DatasetStorage::new(&f.data, S, &layout, &sp, E);
    check(&st, &full);
    let chunks = st.chunks().unwrap();
    assert_eq!(chunks.len(), 6);
    assert_eq!(chunks[1].scaled, vec![1, 0]);
    // Exactly one unlimited dimension is required.
    let bad = space_max(&dims, &[UNLIMITED, UNLIMITED]);
    assert!(
        DatasetStorage::new(&f.data, S, &layout, &bad, E)
            .read(&Selection::All)
            .is_err()
    );
}

/// v2 B-tree (header at 0x40, single leaf at 0x100) with chunk records.
fn btree_v2(f: &mut FileBuilder, filtered: bool, rank: usize, recs: &[(Vec<u64>, u64, u64, u32)]) {
    let rsize = 8 + 8 * rank + if filtered { 4 + 4 } else { 0 };
    let node_size = 512u32;
    let mut h = b"BTHD".to_vec();
    h.push(0);
    h.push(if filtered { 11 } else { 10 });
    h.extend(node_size.to_le_bytes());
    h.extend((rsize as u16).to_le_bytes());
    h.extend(0u16.to_le_bytes()); // depth
    h.extend([100, 40]);
    h.extend(0x100u64.to_le_bytes());
    h.extend((recs.len() as u16).to_le_bytes());
    h.extend((recs.len() as u64).to_le_bytes());
    put_sealed(f, 0x40, h);
    let mut n = b"BTLF".to_vec();
    n.push(0);
    n.push(if filtered { 11 } else { 10 });
    for (scaled, addr, size, mask) in recs {
        n.extend(addr.to_le_bytes());
        if filtered {
            n.extend((*size as u32).to_le_bytes());
            n.extend(mask.to_le_bytes());
        }
        for s in scaled {
            n.extend(s.to_le_bytes());
        }
    }
    let c = metadata_checksum(&n);
    n.extend(c.to_le_bytes());
    n.resize(node_size as usize, 0);
    f.put(0x100, &n);
}

#[test]
fn btree_v2_index() {
    let dims = [6, 7];
    let cd = [2, 2];
    let full = full_array(&dims);
    let mut expected = full.clone();
    let fill = 0xdead_beefu32.to_le_bytes();
    let mut f = FileBuilder::new(S);
    let mut recs = Vec::new();
    let mut recs_z = Vec::new();
    let mut addr = 0x1000;
    for s in grid(&dims, &cd) {
        if s == [1, 2] || s == [2, 3] {
            blank_chunk(&mut expected, &dims, &cd, &s, fill);
            continue;
        }
        let c = chunk_of(&full, &dims, &cd, &s);
        f.put(addr, &c);
        recs.push((s.clone(), addr, 0, 0));
        addr += c.len() as u64;
        let z = encode_sd(&c);
        f.put(addr, &z);
        recs_z.push((s.clone(), addr, z.len() as u64, 0));
        addr += z.len() as u64;
    }
    let sp = space_max(&dims, &[UNLIMITED, UNLIMITED]);
    let fv = fill_v3(&fill);
    let index = ChunkIndex::BTreeV2 {
        address: 0x40,
        node_size: 512,
        split_percent: 100,
        merge_percent: 40,
    };
    let layout = chunked(&cd, index, 0);
    let mut plain = f.clone();
    btree_v2(&mut plain, false, 2, &recs);
    let st = DatasetStorage::new(&plain.data, S, &layout, &sp, E).with_fill_value(Some(&fv));
    check(&st, &expected);
    assert_eq!(st.chunks().unwrap().len(), recs.len());
    let mut z = f.clone();
    btree_v2(&mut z, true, 2, &recs_z);
    let pl = shuffle_deflate();
    let st = DatasetStorage::new(&z.data, S, &layout, &sp, E)
        .with_fill_value(Some(&fv))
        .with_filters(Some(&pl));
    check(&st, &expected);
    // Tree type must match the presence of filters.
    let st = DatasetStorage::new(&plain.data, S, &layout, &sp, E).with_filters(Some(&pl));
    assert!(st.read(&Selection::All).is_err());
}

// ------------------------------------------------- limits and errors --

#[test]
fn limits_and_unsupported() {
    let store: Vec<u8> = Vec::new();
    let sp = space(&[1000]);
    let layout = DataLayout::Contiguous {
        address: UNDEFINED_ADDRESS,
        size: None,
        dims: None,
    };
    let st = DatasetStorage::new(&store, S, &layout, &sp, E)
        .with_limits(ReadLimits::with_max_output(100));
    assert!(matches!(
        st.read(&Selection::All),
        Err(Error::LimitExceeded(_))
    ));
    assert_eq!(st.read(&Selection::range(0..25)).unwrap().len(), 100);
    let virt = DataLayout::Virtual {
        heap_address: 0,
        heap_index: 0,
    };
    let st = DatasetStorage::new(&store, S, &virt, &sp, E);
    assert!(matches!(
        st.read(&Selection::All),
        Err(Error::Unsupported(_))
    ));
    // Chunk larger than the chunk limit.
    let layout = chunked(&[1000], ChunkIndex::Implicit { address: 0 }, 0);
    let st = DatasetStorage::new(&store, S, &layout, &sp, E)
        .with_limits(ReadLimits::default().max_chunk_bytes(1000));
    assert!(matches!(
        st.read(&Selection::All),
        Err(Error::LimitExceeded(_))
    ));
    // Rank / element size mismatches.
    let layout = chunked(&[10, 10], ChunkIndex::Implicit { address: 0 }, 0);
    assert!(
        DatasetStorage::new(&store, S, &layout, &sp, E)
            .read(&Selection::All)
            .is_err()
    );
    let layout = chunked(&[10], ChunkIndex::Implicit { address: 0 }, 0);
    assert!(
        DatasetStorage::new(&store, S, &layout, &sp, 8)
            .read(&Selection::All)
            .is_err()
    );
    assert!(
        DatasetStorage::new(&store, S, &layout, &sp, 0)
            .read(&Selection::All)
            .is_err()
    );
    // Huge dataspace: the element count overflows, still no panic.
    let huge = space(&[u64::MAX, u64::MAX]);
    let st = DatasetStorage::new(&store, S, &layout, &huge, E);
    assert!(st.read(&Selection::All).is_err());
    let one = Selection::from_ranges(&[5..6, 7..8]);
    let layout = chunked(&[1 << 20, 1 << 20], ChunkIndex::Implicit { address: 0 }, 0);
    let st = DatasetStorage::new(&store, S, &layout, &huge, E);
    assert!(st.read(&one).is_err());
}

// ------------------------------------------------------ never panic --

#[test]
fn mutated_inputs_never_panic() {
    // A handful of valid configurations, then random corruption of the
    // file bytes and of the layout / dataspace descriptions.
    let dims = [7, 10];
    let cd = [3, 4];
    let enc = |c: &[u8], i: usize| {
        if i.is_multiple_of(2) {
            (encode_sd(c), 0)
        } else {
            (shuffle(c.to_vec(), E), 2)
        }
    };
    let (bt, _) = btree_v1_file(&dims, &cd, &[vec![1, 1]], [0; 4], &enc);
    let full = full_array(&dims);
    let mut fa = FileBuilder::new(S);
    let mut entries = Vec::new();
    let mut ea = FileBuilder::new(S);
    let mut v2 = FileBuilder::new(S);
    let mut v2recs = Vec::new();
    let mut addr = 0x1000;
    for s in grid(&dims, &cd) {
        let z = encode_sd(&chunk_of(&full, &dims, &cd, &s));
        fa.put(addr, &z);
        v2.put(addr, &z);
        entries.push(Some((addr, z.len() as u64, 0)));
        v2recs.push((s.clone(), addr, z.len() as u64, 0));
        addr += z.len() as u64;
    }
    fixed_array(&mut fa, true, &entries);
    btree_v2(&mut v2, true, 2, &v2recs);
    let full1 = full_array(&[9]);
    let mut recs = Vec::new();
    for s in grid(&[9], &[2]) {
        let a = 0x1000 + s[0] * 8;
        ea.put(a, &chunk_of(&full1, &[9], &[2], &s));
        recs.push(Some(a));
    }
    extensible_array(&mut ea, &recs);

    let v2index = ChunkIndex::BTreeV2 {
        address: 0x40,
        node_size: 512,
        split_percent: 100,
        merge_percent: 40,
    };
    let cases: Vec<(Vec<u8>, DataLayout, Dataspace, bool)> = vec![
        (
            bt.data.clone(),
            chunked(&cd, ChunkIndex::BTreeV1 { address: 0x40 }, 0),
            space(&dims),
            true,
        ),
        (
            fa.data.clone(),
            chunked(
                &cd,
                ChunkIndex::FixedArray {
                    address: 0x40,
                    page_bits: 10,
                },
                0,
            ),
            space(&dims),
            true,
        ),
        (
            v2.data.clone(),
            chunked(&cd, v2index, 0),
            space_max(&dims, &[UNLIMITED, 20]),
            true,
        ),
        (
            ea.data.clone(),
            chunked(&[2], ea_index(), 0),
            space_max(&[9], &[UNLIMITED]),
            false,
        ),
        (
            full.clone(),
            DataLayout::Contiguous {
                address: 0,
                size: None,
                dims: None,
            },
            space(&dims),
            false,
        ),
    ];
    let pl = shuffle_deflate();
    let mut rng = Rng(0x0bad_cafe_1234_5678);
    let limits = ReadLimits::with_max_output(1 << 16).max_chunk_bytes(1 << 16);
    for _ in 0..3000 {
        let (data, layout, sp, filtered) = &cases[rng.below(cases.len())];
        let data = if rng.below(4) == 0 {
            data.clone()
        } else {
            mutate(&mut rng, data)
        };
        let mut layout = layout.clone();
        let mut sp = sp.clone();
        match rng.below(6) {
            0 => {
                if let DataLayout::Chunked(c) = &mut layout {
                    let i = rng.below(c.dims.len());
                    c.dims[i] = [0, 1, 2, 5, 1 << 20, u64::MAX][rng.below(6)];
                }
            }
            1 => {
                let i = rng.below(sp.dims.len());
                sp.dims[i] = [0, 1, 3, 100, 1 << 40, u64::MAX - 1][rng.below(6)];
            }
            2 => {
                if let DataLayout::Chunked(c) = &mut layout {
                    c.flags = rng.next() as u8;
                    c.index = match c.index {
                        ChunkIndex::SingleChunk { .. } => c.index,
                        other => ChunkIndex::SingleChunk {
                            address: other.address(),
                            filtered: Some(FilteredChunk {
                                size: rng.next() % 4096,
                                filter_mask: rng.next() as u32,
                            }),
                        },
                    };
                }
            }
            3 => sp.max_dims = Some(vec![rng.next() % 50; sp.dims.len()]),
            _ => {}
        }
        let st = DatasetStorage::new(&data, S, &layout, &sp, E)
            .with_filters(filtered.then_some(&pl))
            .with_limits(limits);
        let _ = st.read(&Selection::All);
        let _ = st.chunks();
        if sp.dims.iter().all(|&d| d > 0 && d < 1000) {
            let h = random_hyperslab(&mut rng, &sp.dims);
            let _ = st.read(&Selection::hyperslab(h));
        }
    }
}
