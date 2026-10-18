//! The chunk cache and the chunk-visiting strategies of chunked reads give
//! the same bytes as uncached reads, for every chunked fixture dataset;
//! file-level resource limits apply.

use std::path::PathBuf;

use hdf5rs::{Dataset, File, FileAccess, HyperslabDim, Link, Selection};

fn fixture_with(name: &str, access: FileAccess) -> File {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    File::open_with(p, access).unwrap()
}

/// Every chunked dataset reachable from the root (two levels deep).
fn chunked_datasets(file: &File) -> Vec<Dataset> {
    let mut out = Vec::new();
    let mut groups = vec![file.root().unwrap()];
    while let Some(g) = groups.pop() {
        for e in g.members().unwrap() {
            if !matches!(e.link, Link::Hard { .. }) {
                continue;
            }
            match g.object(&e.name) {
                Ok(hdf5rs::Object::Group(sub)) if groups.len() < 64 => groups.push(sub),
                Ok(hdf5rs::Object::Dataset(d)) if d.is_chunked() => out.push(d),
                _ => {}
            }
        }
    }
    out
}

/// The row-major bytes of `sel` (one block per dimension) cut out of a
/// full raw read.
fn cut(full: &[u8], shape: &[u64], esize: usize, sel: &[(u64, u64)]) -> Vec<u8> {
    let mut out = Vec::new();
    let n: u64 = sel.iter().map(|&(_, len)| len).product();
    for i in 0..n {
        // Unravel `i` within the selection, then ravel within the shape.
        let mut rem = i;
        let mut lin = 0u64;
        let mut coords = vec![0u64; sel.len()];
        for (k, &(_, len)) in sel.iter().enumerate().rev() {
            coords[k] = rem % len;
            rem /= len;
        }
        for (k, &d) in shape.iter().enumerate() {
            lin = lin * d + sel[k].0 + coords[k];
        }
        let at = lin as usize * esize;
        out.extend_from_slice(&full[at..at + esize]);
    }
    out
}

const FILES: &[&str] = &[
    "layouts_latest.h5",
    "layouts_earliest.h5",
    "filters.h5",
    "filters_earliest.h5",
    "resized.h5",
    "resized_earliest.h5",
    "big_gzip.h5",
];

#[test]
fn cached_reads_match_uncached_reads() {
    let mut checked = 0;
    for name in FILES {
        let file = fixture_with(name, FileAccess::default());
        for ds in chunked_datasets(&file) {
            let Ok(reference) = ds.reader().use_chunk_cache(false).read_raw() else {
                continue; // e.g. a filter that is not built in
            };
            let shape = ds.shape().to_vec();
            if shape.contains(&0) {
                continue;
            }
            let esize = ds.datatype().size as usize;
            // Small reads first (individual index lookups), then the
            // whole dataset (whole index), then small reads again (cache).
            let mut seed = 0x9e37_79b9_7f4a_7c15u64;
            let mut boxes = Vec::new();
            for _ in 0..12 {
                let sel: Vec<(u64, u64)> = shape
                    .iter()
                    .map(|&d| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let start = seed % d;
                        let len = 1 + (seed >> 32) % (d - start).min(5);
                        (start, len)
                    })
                    .collect();
                boxes.push(sel);
            }
            let read_box = |sel: &[(u64, u64)]| {
                let ranges: Vec<_> = sel.iter().map(|&(s, l)| s..s + l).collect();
                ds.read_raw_selection(&Selection::from_ranges(&ranges))
                    .unwrap()
            };
            for b in &boxes[..6] {
                assert_eq!(
                    read_box(b),
                    cut(&reference, &shape, esize, b),
                    "{}",
                    ds.path()
                );
            }
            assert_eq!(ds.read_raw().unwrap(), reference, "{}", ds.path());
            for b in &boxes {
                assert_eq!(
                    read_box(b),
                    cut(&reference, &shape, esize, b),
                    "{}",
                    ds.path()
                );
            }
            // A strided selection along every dimension.
            let strided: Vec<HyperslabDim> = shape
                .iter()
                .map(|&d| HyperslabDim::new(0, 2, d.div_ceil(2), 1))
                .collect();
            let got = ds
                .read_raw_selection(&Selection::hyperslab(strided.clone()))
                .unwrap();
            let uncached = ds
                .reader()
                .use_chunk_cache(false)
                .read_raw_selection(&Selection::hyperslab(strided))
                .unwrap();
            assert_eq!(got, uncached, "{}", ds.path());
            // Chunk listing through the cached index agrees.
            assert_eq!(
                ds.chunks().unwrap(),
                ds.reader().use_chunk_cache(false).chunks().unwrap()
            );
            checked += 1;
        }
    }
    let min = if cfg!(feature = "deflate") { 40 } else { 10 };
    assert!(checked > min, "only {checked} chunked datasets checked");
}

#[test]
#[cfg(feature = "deflate")]
fn cache_hits_and_capacity() {
    let file = fixture_with("big_gzip.h5", FileAccess::default());
    let ds = file.dataset("/arange_i32").unwrap();
    let sel = Selection::range(100..110);
    ds.read_raw_selection(&sel).unwrap();
    let before = ds.chunk_cache().stats();
    assert_eq!(before.chunks, 1);
    for _ in 0..10 {
        ds.read_raw_selection(&sel).unwrap();
    }
    let after = ds.chunk_cache().stats();
    assert_eq!(after.hits, before.hits + 10);
    assert_eq!(after.misses, before.misses);
    // Clones share the cache.
    let clone = ds.clone();
    clone.read_raw_selection(&sel).unwrap();
    assert_eq!(ds.chunk_cache().stats().hits, after.hits + 1);
    // The whole dataset (4 MiB) fits in the default 16 MiB.
    ds.read_raw().unwrap();
    let full = ds.chunk_cache().stats();
    assert!(full.index_loaded);
    assert_eq!(full.bytes, 4 << 20);
    // Shrinking evicts; clearing empties.
    ds.chunk_cache().set_capacity(1 << 20);
    assert!(ds.chunk_cache().stats().bytes <= 1 << 20);
    ds.chunk_cache().clear();
    assert_eq!(ds.chunk_cache().stats().chunks, 0);

    // Disabled through the file access policy.
    let file = fixture_with("big_gzip.h5", FileAccess::default().chunk_cache_bytes(0));
    let ds = file.dataset("/arange_i32").unwrap();
    assert_eq!(ds.read::<i32>().unwrap()[12345], 12345);
    assert_eq!(ds.chunk_cache().stats().chunks, 0);
}

#[test]
#[cfg(feature = "deflate")]
fn file_level_limits() {
    use hdf5rs::read::ReadLimits;
    let access = FileAccess::default().read_limits(ReadLimits::with_max_output(1 << 16));
    let file = fixture_with("big_gzip.h5", access);
    let ds = file.dataset("/arange_i32").unwrap();
    assert!(matches!(
        ds.read::<i32>(),
        Err(hdf5rs::Error::LimitExceeded(_))
    ));
    assert_eq!(
        ds.read_selection::<i32>(&Selection::range(0..3)).unwrap(),
        [0, 1, 2]
    );
    // A per-read override wins.
    let all = ds
        .reader()
        .limits(ReadLimits::default())
        .read::<i32>()
        .unwrap();
    assert_eq!(all.len(), 1 << 20);
}

#[test]
fn link_hop_limit() {
    use hdf5rs::read::ReadLimits;
    let file = fixture_with("groups_latest.h5", FileAccess::default());
    assert!(file.object("/soft_to_target").is_ok());
    let strict = FileAccess::default().read_limits(ReadLimits::default().max_link_hops(0));
    let file = fixture_with("groups_latest.h5", strict);
    assert!(matches!(
        file.object("/soft_to_target"),
        Err(hdf5rs::Error::LimitExceeded(_))
    ));
    assert!(file.object("/target").is_ok());
}
