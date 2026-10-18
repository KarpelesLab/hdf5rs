//! Read-write modification of existing files: every fixture (written by
//! the reference library) and files written by this crate are copied,
//! opened with `FileWriter::open_rw`, modified, closed and read back. The
//! untouched content must be exactly what it was (compared through the
//! reader, which `read_values.rs` checks against the manifest), and the
//! changes must be visible.

#[path = "support/json.rs"]
mod json;

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use hdf5rs::write::{FileWriter, WriteOptions};
use hdf5rs::{Error, File, Group, Link, LinkStorage, Object, Selection};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixtures() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(fixtures_dir())
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.ends_with(".h5").then_some(n)
        })
        .collect();
    v.sort();
    v
}

fn temp_path(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("modify");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn temp_copy(fixture: &str, tag: &str) -> PathBuf {
    let p = temp_path(&format!("{tag}-{fixture}"));
    std::fs::copy(fixtures_dir().join(fixture), &p).unwrap();
    p
}

fn digest(bytes: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Whether a name was added by the modifications of these tests.
fn is_new(name: &str) -> bool {
    name.starts_with("rw_")
}

/// A textual description of everything reachable from the root through
/// hard links (keyed by the first path each object is reached by),
/// leaving out names starting with `rw_`.
fn dump(f: &File) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut seen = HashMap::new();
    walk(&Object::Group(f.root().unwrap()), "/", &mut seen, &mut out);
    out
}

fn describe_attrs(o: &Object, out: &mut String) {
    let mut attrs = match o.attributes() {
        Ok(a) => a,
        Err(e) => {
            out.push_str(&format!(" attrs-error({e})"));
            return;
        }
    };
    attrs.retain(|a| !is_new(a.name()));
    for a in attrs {
        let v = match a.values() {
            Ok(v) => format!("{v:?}"),
            Err(e) => format!("error({e})"),
        };
        out.push_str(&format!(
            "\n  @{} {:?} {:?} = {v}",
            a.name(),
            a.datatype(),
            a.shape()
        ));
    }
}

fn walk(
    o: &Object,
    path: &str,
    seen: &mut HashMap<u64, String>,
    out: &mut BTreeMap<String, String>,
) {
    if let Some(first) = seen.get(&o.address()) {
        out.insert(path.to_owned(), format!("alias of {first}"));
        return;
    }
    seen.insert(o.address(), path.to_owned());
    let mut s = String::new();
    match o {
        Object::Group(g) => {
            s.push_str("group");
            let members = g.members().unwrap();
            for m in &members {
                if is_new(&m.name) {
                    continue;
                }
                let l = match &m.link {
                    Link::Hard { .. } => "hard".to_owned(),
                    other => format!("{other:?}"),
                };
                s.push_str(&format!("\n  {} -> {l}", m.name));
            }
            describe_attrs(o, &mut s);
            out.insert(path.to_owned(), s);
            for m in members {
                if is_new(&m.name) {
                    continue;
                }
                if let Link::Hard { address } = m.link {
                    let child = child_path(path, &m.name);
                    let obj = o.file().object_at(address, &child).unwrap();
                    walk(&obj, &child, seen, out);
                }
            }
            return;
        }
        Object::Dataset(d) => {
            s.push_str(&format!(
                "dataset {:?} shape {:?} max {:?}",
                d.datatype(),
                d.shape(),
                d.max_shape()
            ));
            let n = d.element_count().unwrap_or(0);
            if n <= 4096 {
                match d.read_values() {
                    Ok(v) => s.push_str(&format!(" = {v:?}")),
                    Err(e) => s.push_str(&format!(" error({e})")),
                }
            } else {
                match d.read_raw() {
                    Ok(v) => s.push_str(&format!(" raw#{:x}", digest(&v))),
                    Err(e) => s.push_str(&format!(" error({e})")),
                }
            }
        }
        Object::NamedDatatype(t) => s.push_str(&format!("datatype {:?}", t.datatype())),
    }
    describe_attrs(o, &mut s);
    out.insert(path.to_owned(), s);
}

fn child_path(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

/// Every object reachable through hard links: (first path, kind).
fn objects(f: &File) -> Vec<(String, Object)> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    fn rec(
        g: &Group,
        path: &str,
        seen: &mut std::collections::HashSet<u64>,
        out: &mut Vec<(String, Object)>,
    ) {
        for m in g.members().unwrap() {
            if let Link::Hard { address } = m.link {
                if !seen.insert(address) {
                    continue;
                }
                let child = child_path(path, &m.name);
                let obj = g.file().object_at(address, &child).unwrap();
                out.push((child.clone(), obj.clone()));
                if let Object::Group(sub) = &obj {
                    rec(sub, &child, seen, out);
                }
            }
        }
    }
    let root = f.root().unwrap();
    seen.insert(root.address());
    out.push(("/".to_owned(), Object::Group(root.clone())));
    rec(&root, "/", &mut seen, &mut out);
    out
}

/// Touch every object of the file at `path`: add an attribute to each,
/// a soft link to each group, rewrite every dataset with its own values
/// (rebuilding chunk indexes), and add a new group with a dataset.
/// Returns the number of datasets rewritten.
fn modify_everything(path: &Path) -> hdf5rs::Result<usize> {
    let original = File::open(path)?;
    let objs = objects(&original);
    let mut w = FileWriter::open_rw(path)?;
    {
        let mut root = w.root();
        root.set_attr("rw_attr", &42i32)?;
        let mut g = root.create_group("rw_group")?;
        g.new_dataset::<i32>("d").write(&[7, 8, 9])?;
        g.set_attr("rw_note", "added")?;
    }
    let mut rewritten = 0;
    for (p, obj) in &objs {
        w.set_attr(p, "rw_a", &(p.len() as u64))?;
        match obj {
            Object::Group(_) => {
                w.group(p)?.link_soft("rw_soft", "/rw_group/d")?;
            }
            Object::Dataset(d) => {
                // Scale-offset is lossy: re-encoding could change values.
                let lossy = d
                    .filters()
                    .is_some_and(|f| f.filters.iter().any(|x| x.id == 6));
                if lossy || d.is_virtual() || d.external_files().is_some() {
                    continue;
                }
                let Ok(raw) = d.read_raw() else { continue };
                let mut dw = w.dataset(p)?;
                match dw.write_selection_raw(&Selection::All, &raw) {
                    Ok(()) => rewritten += 1,
                    Err(Error::Unsupported(_)) => {}
                    Err(e) => return Err(e),
                }
            }
            Object::NamedDatatype(_) => {}
        }
    }
    w.finish()?;
    Ok(rewritten)
}

fn check_new_content(f: &File, objs: usize) {
    assert_eq!(
        f.root()
            .unwrap()
            .attribute("rw_attr")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        42
    );
    assert_eq!(
        f.dataset("/rw_group/d").unwrap().read::<i32>().unwrap(),
        [7, 8, 9]
    );
    assert_eq!(
        f.group("/rw_group")
            .unwrap()
            .attribute("rw_note")
            .unwrap()
            .read_scalar::<String>()
            .unwrap(),
        "added"
    );
    let all = objects(f);
    let mut touched = 0;
    for (p, o) in &all {
        if p.starts_with("/rw_group") {
            continue;
        }
        let a = o.attribute("rw_a").unwrap_or_else(|e| panic!("{p}: {e}"));
        assert_eq!(a.read_scalar::<u64>().unwrap(), p.len() as u64, "{p}");
        if let Object::Group(g) = o {
            match g.link("rw_soft").unwrap() {
                Link::Soft { path } => assert_eq!(path, "/rw_group/d"),
                other => panic!("{p}: {other:?}"),
            }
            assert_eq!(
                g.dataset("rw_soft").unwrap().read::<i32>().unwrap(),
                [7, 8, 9]
            );
        }
        touched += 1;
    }
    assert_eq!(touched, objs, "objects");
}

/// Compare the structure of a modified fixture with what the manifest
/// says about the original: object kinds, links, attribute names and
/// dataset shapes (values are compared with the original file by `dump`;
/// `read_values.rs` checks the original values against the manifest).
fn check_manifest(name: &str, f: &File, manifest: &json::Json) {
    let info = manifest
        .get("files")
        .and_then(|m| m.get(name))
        .unwrap_or_else(|| panic!("{name} not in the manifest"));
    for (path, obj) in info.get("objects").unwrap().entries() {
        let o = f
            .object(path)
            .unwrap_or_else(|e| panic!("{name}:{path}: {e}"));
        let kind = obj.get("kind").and_then(json::Json::str).unwrap();
        let what = format!("{name}:{path}");
        match (kind, &o) {
            ("group", Object::Group(g)) => {
                let mut want: Vec<(String, String)> = obj
                    .get("links")
                    .map(json::Json::arr)
                    .unwrap_or_default()
                    .iter()
                    .map(|l| {
                        (
                            l.get("name").and_then(json::Json::str).unwrap().to_owned(),
                            l.get("type").and_then(json::Json::str).unwrap().to_owned(),
                        )
                    })
                    .collect();
                want.sort();
                let got: Vec<(String, String)> = g
                    .members()
                    .unwrap()
                    .into_iter()
                    .filter(|m| !is_new(&m.name))
                    .map(|m| {
                        let t = match m.link {
                            Link::Hard { .. } => "hard",
                            Link::Soft { .. } => "soft",
                            Link::External { .. } => "external",
                            _ => "other",
                        };
                        (m.name, t.to_owned())
                    })
                    .collect();
                assert_eq!(got, want, "{what}: links");
            }
            ("dataset", Object::Dataset(d)) => {
                let shape: Vec<u64> = obj
                    .get("dataspace")
                    .and_then(|s| s.get("shape"))
                    .map(json::Json::arr)
                    .unwrap_or_default()
                    .iter()
                    .map(|v| v.u64().unwrap())
                    .collect();
                assert_eq!(d.shape(), &shape[..], "{what}: shape");
            }
            ("datatype", Object::NamedDatatype(_)) => {}
            (k, o) => panic!("{what}: expected {k}, got {:?}", o.kind()),
        }
        let mut want: Vec<String> = obj
            .get("attributes")
            .map(json::Json::arr)
            .unwrap_or_default()
            .iter()
            .map(|a| a.get("name").and_then(json::Json::str).unwrap().to_owned())
            .collect();
        want.sort();
        let mut got: Vec<String> = o
            .attribute_names()
            .unwrap()
            .into_iter()
            .filter(|n| !is_new(n))
            .collect();
        got.sort();
        assert_eq!(got, want, "{what}: attributes");
    }
}

#[test]
fn modify_every_fixture() {
    let manifest =
        json::Json::parse(&std::fs::read_to_string(fixtures_dir().join("manifest.json")).unwrap());
    for name in fixtures() {
        let path = temp_copy(&name, "all");
        let before = File::open(&path).unwrap();
        let expect = dump(&before);
        let nobjs = objects(&before).len();
        drop(before);
        match modify_everything(&path) {
            Ok(n) => eprintln!("{name}: {nobjs} objects, {n} datasets rewritten"),
            Err(Error::Unsupported(m)) if name == "fsm_paged.h5" => {
                eprintln!("{name}: refused ({m})");
                // Nothing was written.
                let orig = std::fs::read(fixtures_dir().join(&name)).unwrap();
                assert_eq!(std::fs::read(&path).unwrap(), orig);
                continue;
            }
            Err(e) => panic!("{name}: {e}"),
        }
        let after = File::open(&path).unwrap_or_else(|e| panic!("{name}: reopen: {e}"));
        let got = dump(&after);
        for (k, v) in &expect {
            assert_eq!(got.get(k), Some(v), "{name}: {k} changed");
        }
        for k in got.keys() {
            assert!(expect.contains_key(k), "{name}: unexpected {k}");
        }
        check_new_content(&after, nobjs);
        check_manifest(&name, &after, &manifest);
        // The end-of-file address (absolute, like the reference library
        // writes it) matches the file size.
        assert_eq!(
            after.superblock().eof_address,
            std::fs::metadata(&path).unwrap().len(),
            "{name}: end of file"
        );
        if name == "userblock.h5" {
            let orig = std::fs::read(fixtures_dir().join(&name)).unwrap();
            let now = std::fs::read(&path).unwrap();
            assert_eq!(now[..512], orig[..512], "user block preserved");
        }
    }
}

/// Files written by this crate, modified in every format.
#[test]
fn modify_own_files() {
    for version in [0u8, 2, 3] {
        let path = temp_path(&format!("own-v{version}.h5"));
        let opts = WriteOptions {
            superblock_version: version,
            ..WriteOptions::default()
        };
        let mut w = FileWriter::create_with(&path, opts).unwrap();
        {
            let mut root = w.root();
            root.set_attr("title", "own").unwrap();
            let mut g = root.create_group("g").unwrap();
            for i in 0..5 {
                g.new_dataset::<i32>(&format!("d{i}"))
                    .write(&[i, i + 1])
                    .unwrap();
            }
            let b = g
                .new_dataset::<f64>("chunked")
                .shape(&[0])
                .max_shape(&[None])
                .chunk(&[4])
                .shuffle();
            let b = if cfg!(feature = "deflate") {
                b.deflate(4)
            } else {
                b.fletcher32()
            };
            b.create()
                .unwrap()
                .append(&[1.0, 2.0, 3.0, 4.0, 5.0])
                .unwrap();
        }
        w.finish().unwrap();
        let before = dump(&File::open(&path).unwrap());
        let nobjs = objects(&File::open(&path).unwrap()).len();
        modify_everything(&path).unwrap();
        let after = File::open(&path).unwrap();
        let got = dump(&after);
        for (k, v) in &before {
            assert_eq!(got.get(k), Some(v), "v{version}: {k} changed");
        }
        check_new_content(&after, nobjs);
        // A second round on the modified file.
        let mut w = FileWriter::open_rw(&path).unwrap();
        w.dataset("/g/chunked")
            .unwrap()
            .append(&[6.0, 7.0])
            .unwrap();
        w.unlink("/g/d0").unwrap();
        w.finish().unwrap();
        let f = File::open(&path).unwrap();
        assert_eq!(
            f.dataset("/g/chunked").unwrap().read::<f64>().unwrap(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert!(!f.group("/g").unwrap().contains("d0").unwrap());
        assert_eq!(f.dataset("/g/d1").unwrap().read::<i32>().unwrap(), [1, 2]);
    }
}

#[test]
fn storage_kinds_after_growth() {
    let path = temp_copy("groups_latest.h5", "kinds");
    let mut w = FileWriter::open_rw(&path).unwrap();
    {
        let mut a = w.group("/a").unwrap();
        for i in 0..20 {
            a.link_soft(&format!("rw_{i:02}"), "/target").unwrap();
        }
    }
    w.finish().unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.group("/a").unwrap().link_storage().unwrap(),
        LinkStorage::Dense
    );
    assert_eq!(f.group("/a").unwrap().len().unwrap(), 22);
}

fn names(g: &Group) -> Vec<String> {
    g.link_names().unwrap()
}

#[test]
fn new_style_groups() {
    let path = temp_copy("groups_latest.h5", "groups");
    let before = File::open(&path).unwrap();
    let many: Vec<String> = names(&before.group("/many").unwrap());
    let tracked_many = before
        .group("/tracked_many")
        .unwrap()
        .members_by_creation_order()
        .unwrap();
    let tracked = before
        .group("/tracked")
        .unwrap()
        .members_by_creation_order()
        .unwrap();
    let alias_rc = before.dataset("/target").unwrap().header().ref_count;
    drop(before);

    let mut w = FileWriter::open_rw(&path).unwrap();
    // Dense -> compact: keep 4 of 100 links.
    for n in many.iter().skip(4) {
        w.unlink(&format!("/many/{n}")).unwrap();
    }
    // Compact tracked group grows past the phase change.
    {
        let mut g = w.group("/tracked").unwrap();
        for i in 0..10 {
            g.create_group(&format!("rw_t{i}")).unwrap();
        }
    }
    // Dense tracked group: remove some, add some.
    {
        let mut g = w.group("/tracked_many").unwrap();
        for e in tracked_many.iter().take(5) {
            g.unlink(&e.name).unwrap();
        }
        g.new_dataset::<u8>("rw_x").write(&[1, 2]).unwrap();
        g.link_soft("rw_y", "/target").unwrap();
    }
    // One less hard link to /target.
    w.unlink("/hard_alias").unwrap();
    // Remove and re-create under the same name.
    w.unlink("/a/b/c/leaf").unwrap();
    w.group("/a/b/c")
        .unwrap()
        .new_dataset::<i64>("leaf")
        .write(&[-1, -2])
        .unwrap();
    w.finish().unwrap();

    let f = File::open(&path).unwrap();
    let g = f.group("/many").unwrap();
    assert_eq!(g.link_storage().unwrap(), LinkStorage::Compact);
    assert_eq!(names(&g), many[..4]);
    for n in &many[..4] {
        g.dataset(n).unwrap().read_values().unwrap();
    }
    let g = f.group("/tracked").unwrap();
    assert_eq!(g.link_storage().unwrap(), LinkStorage::Dense);
    let order: Vec<String> = g
        .members_by_creation_order()
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    let mut expect: Vec<String> = tracked.iter().map(|e| e.name.clone()).collect();
    expect.extend((0..10).map(|i| format!("rw_t{i}")));
    assert_eq!(order, expect);
    let g = f.group("/tracked_many").unwrap();
    let order: Vec<String> = g
        .members_by_creation_order()
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    let mut expect: Vec<String> = tracked_many
        .iter()
        .skip(5)
        .map(|e| e.name.clone())
        .collect();
    expect.extend(["rw_x".to_owned(), "rw_y".to_owned()]);
    assert_eq!(order, expect);
    assert_eq!(
        g.dataset("rw_y").unwrap().address(),
        f.dataset("/target").unwrap().address()
    );
    assert!(!f.root().unwrap().contains("hard_alias").unwrap());
    assert_eq!(
        f.dataset("/target").unwrap().header().ref_count,
        alias_rc - 1
    );
    assert_eq!(
        f.dataset("/a/b/c/leaf").unwrap().read::<i64>().unwrap(),
        [-1, -2]
    );
}

#[test]
fn symbol_table_groups() {
    let path = temp_copy("groups_earliest.h5", "symtab");
    let before = File::open(&path).unwrap();
    let many = names(&before.group("/many").unwrap());
    drop(before);
    let mut w = FileWriter::open_rw(&path).unwrap();
    {
        let mut g = w.group("/many").unwrap();
        // 400 entries: several B-tree nodes below a new root level.
        for i in 0..300 {
            g.new_dataset::<i32>(&format!("rw_{i:03}"))
                .write(&[i])
                .unwrap();
        }
        g.link_soft("rw_soft", "/many/d000").unwrap();
        for n in many.iter().step_by(3) {
            g.unlink(n).unwrap();
        }
    }
    {
        let mut e = w.group("/empty").unwrap();
        e.create_group("rw_sub")
            .unwrap()
            .set_attr("x", &1u8)
            .unwrap();
    }
    w.finish().unwrap();
    let f = File::open(&path).unwrap();
    let g = f.group("/many").unwrap();
    let mut expect: Vec<String> = many
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 != 0)
        .map(|(_, n)| n.clone())
        .collect();
    expect.extend((0..300).map(|i| format!("rw_{i:03}")));
    expect.push("rw_soft".into());
    expect.sort();
    assert_eq!(names(&g), expect);
    for i in [0, 150, 299] {
        assert_eq!(
            g.dataset(&format!("rw_{i:03}"))
                .unwrap()
                .read::<i32>()
                .unwrap(),
            [i]
        );
    }
    assert!(matches!(g.link("rw_soft").unwrap(), Link::Soft { path } if path == "/many/d000"));
    let sub = f.group("/empty/rw_sub").unwrap();
    assert_eq!(sub.attribute("x").unwrap().read_scalar::<u8>().unwrap(), 1);
    assert_eq!(g.link_storage().unwrap(), LinkStorage::SymbolTable);
}

#[test]
fn attributes() {
    for name in ["attributes_latest.h5", "attributes_earliest.h5"] {
        let path = temp_copy(name, "attrs");
        let latest = name.contains("latest");
        let mut w = FileWriter::open_rw(&path).unwrap();
        w.set_attr("/", "root_int", &5i32).unwrap();
        w.delete_attr("/", "root_str").unwrap();
        assert!(w.delete_attr("/", "nope").is_err());
        {
            let mut g = w.group("/group").unwrap();
            for i in 0..12 {
                g.set_attr(&format!("rw_{i}"), &(i as f32)).unwrap();
            }
        }
        {
            let mut d = w.dataset("/dataset").unwrap();
            let names = d.attr_names().unwrap();
            for n in names.iter().skip(2) {
                d.delete_attr(n).unwrap();
            }
            d.set_attr("rw_vec", &[1u16, 2, 3][..]).unwrap();
        }
        if latest {
            let mut g = w.group("/tracked").unwrap();
            for i in 0..10 {
                g.set_attr(&format!("rw_{i}"), &i).unwrap();
            }
            w.set_attr("/dense", "rw_big", &vec![7u8; 70_000][..])
                .unwrap();
        }
        w.finish().unwrap();
        let f = File::open(&path).unwrap();
        let root = f.root().unwrap();
        assert_eq!(
            root.attribute("root_int")
                .unwrap()
                .read_scalar::<i32>()
                .unwrap(),
            5
        );
        assert!(root.attribute("root_str").is_err());
        let g = f.group("/group").unwrap();
        for i in 0..12 {
            assert_eq!(
                g.attribute(&format!("rw_{i}"))
                    .unwrap()
                    .read_scalar::<f32>()
                    .unwrap(),
                i as f32
            );
        }
        let d = f.dataset("/dataset").unwrap();
        assert_eq!(d.attribute_names().unwrap().len(), 3, "{name}");
        assert_eq!(
            d.attribute("rw_vec").unwrap().read::<u16>().unwrap(),
            [1, 2, 3]
        );
        if latest {
            let g = f.group("/tracked").unwrap();
            let order: Vec<String> = g
                .attributes_by_creation_order()
                .unwrap()
                .iter()
                .map(|a| a.name().to_owned())
                .collect();
            let new: Vec<String> = (0..10).map(|i| format!("rw_{i}")).collect();
            assert_eq!(order[order.len() - 10..], new[..]);
            let big = f.group("/dense").unwrap().attribute("rw_big").unwrap();
            assert_eq!(big.read::<u8>().unwrap(), vec![7u8; 70_000]);
        }
    }
}

/// Row-major positions (element indexes) of a box selection.
fn box_positions(dims: &[u64], ranges: &[std::ops::Range<u64>]) -> Vec<u64> {
    let mut out = Vec::new();
    if ranges.iter().any(|r| r.is_empty()) {
        return out;
    }
    let mut idx: Vec<u64> = ranges.iter().map(|r| r.start).collect();
    loop {
        let mut lin = 0u64;
        for (i, &d) in dims.iter().enumerate() {
            lin = lin * d + idx[i];
        }
        out.push(lin);
        let mut k = idx.len();
        loop {
            if k == 0 {
                return out;
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] < ranges[k].end {
                break;
            }
            idx[k] = ranges[k].start;
        }
    }
}

struct WritePlan {
    path: String,
    ranges: Vec<std::ops::Range<u64>>,
    patch: Vec<u8>,
    expect: Vec<u8>,
    dims: Vec<u64>,
    grown: Vec<u64>,
    fill: Vec<u8>,
    chunked: bool,
}

/// Partial writes into, then growth of, every non-scalar dataset of the
/// layout fixtures (every chunk index type, filters, compact, contiguous).
#[test]
fn dataset_writes_and_resize() {
    for name in [
        "layouts_latest.h5",
        "layouts_earliest.h5",
        "resized.h5",
        "resized_earliest.h5",
        "filters.h5",
        "filters_earliest.h5",
        "sizes_4byte_latest.h5",
    ] {
        let path = temp_copy(name, "data");
        let before = File::open(&path).unwrap();
        let mut plans = Vec::new();
        for (p, o) in objects(&before) {
            let Object::Dataset(d) = o else { continue };
            if d.shape().is_empty() || d.datatype().size == 0 {
                continue;
            }
            // Lossy (scale-offset) and precision-reducing (N-bit) filters
            // would not give the patch back exactly.
            let lossy = d
                .filters()
                .is_some_and(|f| f.filters.iter().any(|x| x.id == 6 || x.id == 5));
            // Raw writes of variable-length / reference data are refused.
            let dt = format!("{:?}", d.datatype());
            if lossy || dt.contains("VariableLength") || dt.contains("Reference") {
                continue;
            }
            let Ok(mut expect) = d.read_raw() else {
                continue;
            };
            let dims = d.shape().to_vec();
            let esize = d.datatype().size as usize;
            let ranges: Vec<std::ops::Range<u64>> = dims
                .iter()
                .map(|&n| n / 3..(2 * n / 3 + 1).min(n))
                .collect();
            let pos = box_positions(&dims, &ranges);
            let mut patch = Vec::with_capacity(pos.len() * esize);
            for (k, &lin) in pos.iter().enumerate() {
                for b in 0..esize {
                    let v = (k * 31 + b * 7 + 1) as u8;
                    patch.push(v);
                    expect[lin as usize * esize + b] = v;
                }
            }
            // Grow every dimension that allows it by 3.
            let max = d.max_shape();
            let grown: Vec<u64> = dims
                .iter()
                .zip(&max)
                .map(|(&n, &m)| (n + 3).min(m))
                .collect();
            let fill = d
                .fill_value_bytes()
                .map(<[u8]>::to_vec)
                .unwrap_or_else(|| vec![0; esize]);
            plans.push(WritePlan {
                path: p,
                ranges,
                patch,
                expect,
                dims,
                grown,
                fill,
                chunked: d.is_chunked(),
            });
        }
        drop(before);
        let mut w = FileWriter::open_rw(&path).unwrap();
        for pl in &plans {
            let mut d = w.dataset(&pl.path).unwrap();
            d.write_selection_raw(&Selection::from_ranges(&pl.ranges), &pl.patch)
                .unwrap_or_else(|e| panic!("{name}:{}: {e}", pl.path));
            if pl.chunked && pl.grown != pl.dims {
                d.resize(&pl.grown)
                    .unwrap_or_else(|e| panic!("{name}:{}: {e}", pl.path));
            }
        }
        w.finish().unwrap();
        let f = File::open(&path).unwrap();
        for pl in &plans {
            let p = &pl.path;
            let d = f.dataset(p).unwrap();
            let got = d.read_raw().unwrap();
            if !pl.chunked || pl.grown == pl.dims {
                assert_eq!(got, pl.expect, "{name}:{p}");
                continue;
            }
            assert_eq!(d.shape(), &pl.grown[..], "{name}:{p}");
            let esize = pl.fill.len();
            let n: u64 = pl.grown.iter().product();
            let old: std::collections::HashSet<u64> = box_positions(
                &pl.grown,
                &pl.dims.iter().map(|&n| 0..n).collect::<Vec<_>>(),
            )
            .into_iter()
            .collect();
            let mut k = 0usize;
            for lin in 0..n {
                let e = &got[lin as usize * esize..(lin as usize + 1) * esize];
                if old.contains(&lin) {
                    assert_eq!(
                        e,
                        &pl.expect[k * esize..(k + 1) * esize],
                        "{name}:{p}[{lin}]"
                    );
                    k += 1;
                } else {
                    assert_eq!(e, &pl.fill[..], "{name}:{p}[{lin}] fill");
                }
            }
        }
    }
}

#[test]
fn append_and_shrink() {
    let path = temp_copy("resized.h5", "append");
    let before = File::open(&path).unwrap();
    let grown = before.dataset("/grown").unwrap().read::<i32>().unwrap();
    drop(before);
    let mut w = FileWriter::open_rw(&path).unwrap();
    w.dataset("/grown")
        .unwrap()
        .append(&[100, 101, 102, 103, 104])
        .unwrap();
    w.dataset("/shrunk").unwrap().resize(&[2]).unwrap();
    w.finish().unwrap();
    let f = File::open(&path).unwrap();
    let mut expect = grown.clone();
    expect.extend([100, 101, 102, 103, 104]);
    assert_eq!(f.dataset("/grown").unwrap().read::<i32>().unwrap(), expect);
    assert_eq!(f.dataset("/shrunk").unwrap().shape(), &[2]);
}

#[test]
fn committed_types_refcounts_times_vlen() {
    let start = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;
    for name in ["dtypes_latest.h5", "dtypes_earliest.h5"] {
        let path = temp_copy(name, "misc");
        let before = File::open(&path).unwrap();
        let point = before.object("/named/point").unwrap();
        let point_rc = point.header().ref_count;
        let point_addr = point.address();
        let u16_rc = before.dataset("/named_use/u16").unwrap().header().ref_count;
        let vlen = before
            .dataset("/string/vlen_utf8")
            .unwrap()
            .read::<String>()
            .unwrap();
        drop(before);
        let mut w = FileWriter::open_rw(&path).unwrap();
        w.set_attr("/named/point", "rw_seen", &1u8).unwrap();
        let pid = w.object_id("/named/point").unwrap();
        let uid = w.object_id("/named/u16be").ok();
        let target = w.object_id("/named_use/u16").unwrap();
        {
            let mut g = w.group("/named_use").unwrap();
            let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            g.new_dataset_committed("rw_pts", pid)
                .unwrap()
                .write_raw(&raw)
                .unwrap();
            g.link_hard("rw_alias", target).unwrap();
        }
        let _ = uid;
        w.dataset("/string/vlen_utf8")
            .unwrap()
            .write_selection(&Selection::range(1..2), &["changed ✓".to_owned()])
            .unwrap();
        w.finish().unwrap();
        let f = File::open(&path).unwrap();
        let point = f.object("/named/point").unwrap();
        assert_eq!(point.address(), point_addr);
        assert_eq!(point.header().ref_count, point_rc + 1, "{name}");
        if let Some(t) = point.times() {
            assert!(
                t.modification >= start && t.change >= start,
                "{name}: times"
            );
        }
        let pts = f.dataset("/named_use/rw_pts").unwrap();
        assert_eq!(pts.committed_datatype_address(), Some(point_addr));
        assert_eq!(pts.read_raw().unwrap().len(), 16);
        assert_eq!(
            f.dataset("/named_use/u16").unwrap().header().ref_count,
            u16_rc + 1
        );
        assert_eq!(
            f.dataset("/named_use/rw_alias").unwrap().address(),
            f.dataset("/named_use/u16").unwrap().address()
        );
        let mut expect = vlen.clone();
        expect[1] = "changed ✓".into();
        assert_eq!(
            f.dataset("/string/vlen_utf8")
                .unwrap()
                .read::<String>()
                .unwrap(),
            expect
        );
    }
}

/// Files with shared object header messages, when available (see
/// `read_structure.rs`).
#[test]
fn sohm_extra_fixtures() {
    let Some(dir) = std::env::var_os("HDF5RS_EXTRA_FIXTURES") else {
        eprintln!("note: HDF5RS_EXTRA_FIXTURES not set, skipping SOHM modification");
        return;
    };
    for name in ["sohm_list.h5", "sohm_btree.h5"] {
        let src = Path::new(&dir).join(name);
        if !src.exists() {
            continue;
        }
        let path = temp_path(&format!("sohm-{name}"));
        std::fs::copy(&src, &path).unwrap();
        let before = File::open(&path).unwrap();
        let expect = dump(&before);
        let n = objects(&before).len();
        drop(before);
        modify_everything(&path).unwrap();
        let mut w = FileWriter::open_rw(&path).unwrap();
        w.delete_attr("/d0", "units").unwrap();
        w.set_attr("/d1", "scale", &2.5f64).unwrap();
        w.finish().unwrap();
        let after = File::open(&path).unwrap();
        let got = dump(&after);
        for (k, v) in &expect {
            if k == "/d0" || k == "/d1" {
                continue;
            }
            assert_eq!(got.get(k), Some(v), "{name}: {k}");
        }
        check_new_content(&after, n);
        assert!(after.dataset("/d0").unwrap().attribute("units").is_err());
        assert_eq!(
            after
                .dataset("/d1")
                .unwrap()
                .attribute("scale")
                .unwrap()
                .read_scalar::<f64>()
                .unwrap(),
            2.5
        );
    }
}

#[test]
fn in_memory_and_refusals() {
    let bytes = std::fs::read(fixtures_dir().join("superblock_v3.h5")).unwrap();
    let mut w = FileWriter::open_rw_bytes(bytes).unwrap();
    w.root().set_attr("rw", &1i8).unwrap();
    let out = w.finish().unwrap();
    let f = File::from_bytes(out).unwrap();
    assert_eq!(
        f.root()
            .unwrap()
            .attribute("rw")
            .unwrap()
            .read_scalar::<i8>()
            .unwrap(),
        1
    );
    assert_eq!(f.superblock().consistency_flags, 0);

    // A writer that was never finished leaves the "open for writing" flag
    // of a version 3 superblock set: opening again is refused.
    let p = temp_copy("superblock_v3.h5", "locked");
    let w = FileWriter::open_rw(&p).unwrap();
    drop(w);
    assert!(matches!(
        FileWriter::open_rw(&p),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(File::open(&p).unwrap().superblock().consistency_flags, 1);

    let p = temp_copy("fsm_paged.h5", "fsm");
    assert!(matches!(
        FileWriter::open_rw(&p),
        Err(Error::Unsupported(_))
    ));
}
