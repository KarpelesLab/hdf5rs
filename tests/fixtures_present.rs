//! Checks that every fixture listed in `tests/fixtures/files.txt` exists and
//! carries the HDF5 signature at the recorded superblock offset.

use std::fs;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn all_listed_fixtures_have_signature() {
    let dir = fixtures_dir();
    if !dir.exists() {
        // Fixtures are excluded from the published package.
        eprintln!("tests/fixtures not present; skipping");
        return;
    }
    let list = fs::read_to_string(dir.join("files.txt")).expect("read files.txt");
    let mut count = 0;
    for line in list.lines().filter(|l| !l.trim().is_empty()) {
        let mut parts = line.split_whitespace();
        let name = parts.next().expect("file name");
        let offset: usize = parts
            .next()
            .expect("superblock offset")
            .parse()
            .expect("numeric offset");
        let data = fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            data.len() >= offset + hdf5rs::SIGNATURE.len(),
            "{name}: file too short"
        );
        assert_eq!(
            data[offset..offset + hdf5rs::SIGNATURE.len()],
            hdf5rs::SIGNATURE,
            "{name}: no HDF5 signature at offset {offset}"
        );
        count += 1;
    }
    assert!(count > 0, "files.txt lists no fixtures");
}

#[test]
fn manifest_present() {
    let dir = fixtures_dir();
    if !dir.exists() {
        return;
    }
    let manifest = fs::read_to_string(dir.join("manifest.json")).expect("manifest");
    assert!(manifest.trim_start().starts_with('{'));
}
