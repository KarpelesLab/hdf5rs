//! Structure-level reading of real files: open every fixture, walk the
//! whole hierarchy, load every attribute and dataset description.
//!
//! Fixtures live in `tests/fixtures/`; set `HDF5RS_EXTRA_FIXTURES` to a
//! directory of additional `.h5` files to include them too.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use hdf5rs::{Error, File, Group, Link, LinkStorage, Object};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn h5_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "h5"))
        .collect();
    v.sort();
    v
}

fn all_files() -> Vec<PathBuf> {
    let mut v = h5_files(&fixture_dir());
    if let Some(extra) = std::env::var_os("HDF5RS_EXTRA_FIXTURES") {
        v.extend(h5_files(Path::new(&extra)));
    }
    v
}

fn fixture(name: &str) -> Option<File> {
    let p = fixture_dir().join(name);
    if !p.exists() {
        eprintln!("note: fixture {name} missing, skipping");
        return None;
    }
    Some(File::open(&p).unwrap_or_else(|e| panic!("{name}: {e}")))
}

/// Walk `group` recursively, returning a list of `(path, kind)` lines.
fn walk(group: &Group, seen: &mut HashSet<u64>, out: &mut Vec<String>) -> hdf5rs::Result<()> {
    for a in group.attributes()? {
        out.push(format!("{}@{}", group.path(), a.name()));
    }
    for entry in group.members()? {
        let path = if group.path() == "/" {
            format!("/{}", entry.name)
        } else {
            format!("{}/{}", group.path(), entry.name)
        };
        let obj = match &entry.link {
            Link::Hard { .. } | Link::Soft { .. } => match group.object(&entry.name) {
                Ok(o) => o,
                Err(Error::NotFound(_)) if matches!(entry.link, Link::Soft { .. }) => {
                    out.push(format!("{path} -> dangling"));
                    continue;
                }
                Err(e) => return Err(e),
            },
            Link::External { file, path: p } => {
                out.push(format!("{path} -> external {file}:{p}"));
                assert!(matches!(
                    group.object(&entry.name),
                    Err(Error::Unsupported(_))
                ));
                continue;
            }
            Link::UserDefined { .. } => continue,
        };
        assert_eq!(obj.path(), path);
        for a in obj.attributes()? {
            out.push(format!("{path}@{}", a.name()));
            assert_eq!(obj.attribute(a.name())?.raw_value(), a.raw_value());
        }
        match &obj {
            Object::Group(g) => {
                out.push(format!("{path}/"));
                if seen.insert(g.address()) {
                    walk(g, seen, out)?;
                }
            }
            Object::Dataset(d) => {
                out.push(format!("{path} dataset {:?}", d.shape()));
                let _ = (d.datatype(), d.layout(), d.filters(), d.fill_value_bytes());
            }
            Object::NamedDatatype(t) => {
                out.push(format!("{path} datatype size {}", t.datatype().size));
            }
        }
    }
    Ok(())
}

#[test]
fn walk_all_fixtures() {
    let files = all_files();
    if files.is_empty() {
        eprintln!("note: no fixtures found in {}", fixture_dir().display());
        return;
    }
    let mut failures = Vec::new();
    for path in &files {
        let res = File::open(path).and_then(|f| {
            let root = f.root()?;
            let mut seen = HashSet::from([root.address()]);
            let mut out = Vec::new();
            walk(&root, &mut seen, &mut out)?;
            Ok(out)
        });
        match res {
            Ok(out) => eprintln!("{}: {} entries", path.display(), out.len()),
            Err(e) => failures.push(format!("{}: {e}", path.display())),
        }
    }
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
}

#[path = "support/json.rs"]
mod json;

use hdf5rs::format::messages::MessageType;
use hdf5rs::format::messages::attribute_info::AttributeInfo;
use hdf5rs::format::messages::dataspace::{DataspaceKind, UNLIMITED};
use hdf5rs::format::messages::datatype::{ByteOrder, Datatype, DatatypeClass};
use hdf5rs::format::messages::layout::{ChunkIndex, DataLayout};
use json::Json;

/// Decode a little-endian full-width integer value (first element).
fn scalar_int(dt: &Datatype, raw: &[u8]) -> Option<i128> {
    let DatatypeClass::FixedPoint(fp) = &dt.class else {
        return None;
    };
    let size = dt.size as usize;
    if fp.byte_order != ByteOrder::LittleEndian || size == 0 || size > 8 || raw.len() < size {
        return None;
    }
    let mut buf = [0u8; 8];
    buf[..size].copy_from_slice(&raw[..size]);
    let u = u64::from_le_bytes(buf);
    Some(if fp.signed {
        let shift = 64 - 8 * size as u32;
        i128::from(((u << shift) as i64) >> shift)
    } else {
        i128::from(u)
    })
}

fn has_vlen(dt: &Datatype) -> bool {
    match &dt.class {
        DatatypeClass::VariableLength { .. } => true,
        DatatypeClass::Compound { members } => members.iter().any(|m| has_vlen(&m.datatype)),
        DatatypeClass::Array { base, .. } => has_vlen(base),
        _ => false,
    }
}

fn check_dtype(ctx: &str, expect: &Json, dt: &Datatype) {
    match expect.get("size") {
        Some(Json::Str(s)) if s == "variable" => assert!(
            matches!(dt.class, DatatypeClass::VariableLength { .. }),
            "{ctx}: expected variable-length type, got {dt:?}"
        ),
        // The manifest records NumPy's in-memory size, which differs from
        // the file's for types containing variable-length members.
        Some(Json::Num(n)) if !has_vlen(dt) => {
            assert_eq!(u64::from(dt.size), *n as u64, "{ctx}: datatype size")
        }
        _ => {}
    }
    let class = expect.get("class").and_then(Json::str).unwrap_or("");
    let ok = match (class, &dt.class) {
        ("integer", DatatypeClass::FixedPoint(_))
        | ("float", DatatypeClass::FloatingPoint(_))
        | ("compound", DatatypeClass::Compound { .. })
        | ("enum", DatatypeClass::Enumeration { .. })
        | ("array", DatatypeClass::Array { .. })
        | ("opaque", DatatypeClass::Opaque { .. })
        | ("string", DatatypeClass::String(_) | DatatypeClass::VariableLength { .. })
        | ("vlen", DatatypeClass::VariableLength { .. }) => true,
        ("integer" | "float" | "compound" | "enum" | "array" | "opaque" | "string" | "vlen", _) => {
            false
        }
        _ => true, // a class this test does not check
    };
    assert!(ok, "{ctx}: expected class {class}, got {:?}", dt.class);
}

fn check_dataspace(ctx: &str, expect: &Json, kind: DataspaceKind, shape: &[u64]) {
    match expect.get("class").and_then(Json::str) {
        Some("null") => assert_eq!(kind, DataspaceKind::Null, "{ctx}"),
        Some("scalar") => assert_eq!(kind, DataspaceKind::Scalar, "{ctx}"),
        Some("simple") => {
            assert_eq!(kind, DataspaceKind::Simple, "{ctx}");
            let want: Vec<u64> = expect
                .get("shape")
                .map(|s| s.arr().iter().filter_map(Json::u64).collect())
                .unwrap_or_default();
            assert_eq!(shape, want.as_slice(), "{ctx}: shape");
        }
        _ => {}
    }
}

fn check_attributes(ctx: &str, expect: &Json, obj: &Object) {
    let want: Vec<&str> = expect
        .get("attributes")
        .map(|a| {
            a.arr()
                .iter()
                .filter_map(|x| x.get("name")?.str())
                .collect()
        })
        .unwrap_or_default();
    let mut sorted = want.clone();
    sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    let got = obj.attribute_names().unwrap();
    assert_eq!(got, sorted, "{ctx}: attribute names");
    let exp = expect.get("expect");
    if exp
        .and_then(|e| e.get("attr_creation_order_tracked"))
        .and_then(Json::bool)
        == Some(true)
    {
        let names: Vec<String> = match obj {
            Object::Group(g) => g.attributes_by_creation_order(),
            Object::Dataset(d) => d.attributes_by_creation_order(),
            Object::NamedDatatype(t) => t.attributes_by_creation_order(),
        }
        .unwrap()
        .into_iter()
        .map(|a| a.name().to_owned())
        .collect();
        assert_eq!(names, want, "{ctx}: attribute creation order");
    }
    if let Some(storage) = exp
        .and_then(|e| e.get("attribute_storage"))
        .and_then(Json::str)
    {
        let dense = obj
            .header()
            .find_message(MessageType::ATTRIBUTE_INFO)
            .map(|m| AttributeInfo::parse(&m.data, hdf5rs::format::Sizes::DEFAULT).unwrap())
            .is_some_and(|i| i.fractal_heap_address != hdf5rs::format::UNDEFINED_ADDRESS);
        assert_eq!(dense, storage == "dense", "{ctx}: attribute storage");
    }
    for a in expect.get("attributes").map(Json::arr).unwrap_or_default() {
        let name = a.get("name").and_then(Json::str).unwrap();
        let actx = format!("{ctx}@{name}");
        let attr = obj
            .attribute(name)
            .unwrap_or_else(|e| panic!("{actx}: {e}"));
        if let Some(dt) = a.get("dtype") {
            check_dtype(&actx, dt, attr.datatype());
        }
        if let Some(ds) = a.get("dataspace") {
            check_dataspace(&actx, ds, attr.dataspace().kind, attr.shape());
        }
        if let Some(v) = scalar_int(attr.datatype(), attr.raw_value())
            && attr.dataspace().kind == DataspaceKind::Scalar
            && let Some(Json::Num(want)) = a.get("data")
        {
            assert_eq!(v as f64, *want, "{actx}: value");
        }
        let n = attr.element_count().unwrap();
        assert_eq!(
            attr.raw_value().len() as u64,
            n * u64::from(attr.datatype().size),
            "{actx}: raw value size"
        );
    }
}

fn check_group(ctx: &str, f: &File, expect: &Json, g: &Group) {
    let links = expect.get("links").map(Json::arr).unwrap_or_default();
    let exp = expect.get("expect");
    // h5py lists tracked groups in creation order.
    let tracked = g.creation_order_tracked().unwrap();
    if exp
        .and_then(|e| e.get("creation_order_tracked"))
        .and_then(Json::bool)
        == Some(true)
    {
        assert!(tracked, "{ctx}: creation order should be tracked");
    }
    let members = if tracked {
        g.members_by_creation_order().unwrap()
    } else {
        g.members().unwrap()
    };
    let got: Vec<&str> = members.iter().map(|m| m.name.as_str()).collect();
    let want: Vec<&str> = links.iter().filter_map(|l| l.get("name")?.str()).collect();
    assert_eq!(got, want, "{ctx}: link names/order");
    if let Some(s) = exp.and_then(|e| e.get("link_storage")).and_then(Json::str) {
        let want = match s {
            "compact" => LinkStorage::Compact,
            "dense" => LinkStorage::Dense,
            _ => LinkStorage::SymbolTable,
        };
        assert_eq!(g.link_storage().unwrap(), want, "{ctx}: link storage");
    }
    for (l, m) in links.iter().zip(&members) {
        let target = l.get("target").and_then(Json::str).unwrap_or("");
        match (l.get("type").and_then(Json::str), &m.link) {
            (Some("hard"), Link::Hard { address }) => {
                let t = f.object(target).unwrap();
                assert_eq!(t.address(), *address, "{ctx}/{}: hard link target", m.name);
                assert_eq!(g.link(&m.name).unwrap(), m.link);
            }
            (Some("soft"), Link::Soft { path }) => assert_eq!(path, target, "{ctx}"),
            (Some("external"), Link::External { file, path }) => {
                assert_eq!(path, target, "{ctx}");
                assert_eq!(Some(file.as_str()), l.get("file").and_then(Json::str));
            }
            (t, link) => panic!("{ctx}/{}: expected {t:?} link, got {link:?}", m.name),
        }
    }
}

fn check_dataset(ctx: &str, expect: &Json, d: &hdf5rs::Dataset) {
    if let Some(dt) = expect.get("dtype") {
        check_dtype(ctx, dt, d.datatype());
    }
    if let Some(ds) = expect.get("dataspace") {
        check_dataspace(ctx, ds, d.dataspace().kind, d.shape());
        if let Some(max) = ds.get("maxshape") {
            let want: Vec<u64> = max
                .arr()
                .iter()
                .map(|v| v.u64().unwrap_or(UNLIMITED))
                .collect();
            assert_eq!(d.max_shape(), want, "{ctx}: maxshape");
        }
    }
    let layout_ok = match expect.get("layout").and_then(Json::str) {
        Some("compact") => d.is_compact(),
        Some("contiguous") => d.is_contiguous(),
        Some("chunked") => d.is_chunked(),
        Some("virtual") => d.is_virtual(),
        _ => true,
    };
    assert!(layout_ok, "{ctx}: layout {:?}", d.layout());
    if let Some(cs) = expect.get("chunk_shape") {
        let want: Vec<u64> = cs.arr().iter().filter_map(Json::u64).collect();
        if !want.is_empty() {
            assert_eq!(d.chunk_shape(), Some(want.as_slice()), "{ctx}: chunk shape");
        }
    }
    if let Some(ci) = expect
        .get("expect")
        .and_then(|e| e.get("chunk_index"))
        .and_then(Json::str)
        && let DataLayout::Chunked(c) = d.layout()
    {
        let ok = matches!(
            (ci, c.index),
            ("btree_v1", ChunkIndex::BTreeV1 { .. })
                | ("btree_v2", ChunkIndex::BTreeV2 { .. })
                | ("fixed_array", ChunkIndex::FixedArray { .. })
                | ("extensible_array", ChunkIndex::ExtensibleArray { .. })
                | ("implicit", ChunkIndex::Implicit { .. })
                | ("single" | "single_chunk", ChunkIndex::SingleChunk { .. })
        );
        assert!(ok, "{ctx}: chunk index {ci} vs {:?}", c.index);
    }
    let want_filters: Vec<u64> = expect
        .get("filters")
        .map(|f| f.arr().iter().filter_map(|x| x.get("id")?.u64()).collect())
        .unwrap_or_default();
    let got_filters: Vec<u64> = d
        .filters()
        .map(|p| p.filters.iter().map(|f| u64::from(f.id)).collect())
        .unwrap_or_default();
    assert_eq!(got_filters, want_filters, "{ctx}: filters");
}

#[test]
fn manifest_structure() {
    let path = fixture_dir().join("manifest.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("note: {} missing, skipping", path.display());
        return;
    };
    let manifest = Json::parse(&text);
    for (name, info) in manifest.get("files").map(Json::entries).unwrap_or_default() {
        let Some(f) = fixture(name) else { continue };
        assert_eq!(
            Some(u64::from(f.superblock().version)),
            info.get("superblock_version").and_then(Json::u64),
            "{name}: superblock version"
        );
        assert_eq!(
            Some(f.superblock().superblock_offset),
            info.get("superblock_offset").and_then(Json::u64),
            "{name}: superblock offset"
        );
        for (path, expect) in info.get("objects").map(Json::entries).unwrap_or_default() {
            let ctx = format!("{name}:{path}");
            let obj = f.object(path).unwrap_or_else(|e| panic!("{ctx}: {e}"));
            assert_eq!(obj.path(), path, "{ctx}");
            check_attributes(&ctx, expect, &obj);
            match (expect.get("kind").and_then(Json::str), &obj) {
                (Some("group"), Object::Group(g)) => check_group(&ctx, &f, expect, g),
                (Some("dataset"), Object::Dataset(d)) => check_dataset(&ctx, expect, d),
                (Some("datatype"), Object::NamedDatatype(t)) => {
                    if let Some(dt) = expect.get("dtype") {
                        check_dtype(&ctx, dt, t.datatype());
                    }
                }
                (k, o) => panic!("{ctx}: expected {k:?}, got {:?}", o.kind()),
            }
        }
    }
}

/// Files with shared object header messages (SOHM). The reference
/// generator does not produce these, so they are only checked when
/// `HDF5RS_EXTRA_FIXTURES` points at a directory holding `sohm_list.h5`
/// and/or `sohm_btree.h5` (4 datasets `d0..d3` with gzip, fill value 7 and
/// attributes `units`/`scale`/`n`, plus `grp/x`; one index for
/// dataspace+datatype messages, one for fill+pipeline+attribute messages).
#[test]
fn sohm_extra_fixtures() {
    use hdf5rs::format::sohm::SohmIndexType;
    let Some(dir) = std::env::var_os("HDF5RS_EXTRA_FIXTURES") else {
        eprintln!("note: HDF5RS_EXTRA_FIXTURES not set, skipping SOHM checks");
        return;
    };
    for (name, kind) in [
        ("sohm_list.h5", SohmIndexType::List),
        ("sohm_btree.h5", SohmIndexType::BTree),
    ] {
        let p = Path::new(&dir).join(name);
        if !p.exists() {
            continue;
        }
        let f = File::open(&p).unwrap();
        let table = f.shared_message_table().unwrap().expect("SOHM table");
        assert_eq!(table.indexes.len(), 2);
        assert!(table.indexes.iter().all(|i| i.index_type == kind), "{name}");
        for i in 0..4u8 {
            let d = f.dataset(&format!("d{i}")).unwrap();
            assert_eq!(d.shape(), &[10]);
            assert!(d.datatype().is_integer());
            assert_eq!(d.fill_value_bytes(), Some(&7i32.to_le_bytes()[..]));
            assert_eq!(d.filters().unwrap().filters[0].id, 1);
            let n = d.attribute("n").unwrap();
            assert_eq!(n.raw_value(), &i32::from(i).to_le_bytes());
            let scale = d.attribute("scale").unwrap();
            assert_eq!(scale.raw_value(), &1.5f64.to_le_bytes());
            let units = d.attribute("units").unwrap();
            assert!(matches!(
                units.datatype().class,
                DatatypeClass::VariableLength { .. }
            ));
            assert_eq!(d.attribute_names().unwrap(), ["n", "scale", "units"]);
        }
        let g = f.group("grp").unwrap();
        assert_eq!(g.attribute_names().unwrap(), ["units"]);
        assert_eq!(f.dataset("/grp/x").unwrap().shape(), &[3, 4]);
    }
}

/// Randomly corrupted copies of real files must never panic.
#[test]
fn mutated_fixtures_do_not_panic() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for path in h5_files(&fixture_dir()) {
        let Ok(base) = std::fs::read(&path) else {
            continue;
        };
        if base.len() > 200_000 {
            continue;
        }
        for _ in 0..150 {
            let mut data = base.clone();
            for _ in 0..1 + next() % 4 {
                let i = (next() % data.len() as u64) as usize;
                data[i] ^= 1 << (next() % 8);
            }
            let Ok(f) = File::from_bytes(data) else {
                continue;
            };
            if let Ok(root) = f.root() {
                let mut seen = HashSet::from([root.address()]);
                let _ = walk_lenient(&root, &mut seen, 0);
            }
        }
    }
}

/// Like `walk`, but ignores errors and never asserts.
fn walk_lenient(group: &Group, seen: &mut HashSet<u64>, depth: usize) -> hdf5rs::Result<()> {
    if depth > 16 || seen.len() > 1000 {
        return Ok(());
    }
    let _ = group.attributes();
    let _ = group.members_by_creation_order();
    for entry in group.members()? {
        let _ = group.link(&entry.name);
        let Ok(obj) = group.object(&entry.name) else {
            continue;
        };
        let _ = obj.attributes();
        let _ = obj.modification_time();
        if let Object::Group(g) = obj
            && seen.insert(g.address())
        {
            let _ = walk_lenient(&g, seen, depth + 1);
        }
    }
    Ok(())
}
