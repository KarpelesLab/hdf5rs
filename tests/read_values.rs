//! Every dataset and attribute of the fixture corpus, read through the
//! high-level API (typed where the datatype maps to a Rust primitive or
//! string, [`Value`] always) and compared with `tests/fixtures/manifest.json`.

#[path = "support/json.rs"]
mod json;
#[path = "support/sha256.rs"]
mod sha256;

use std::collections::BTreeSet;
use std::path::PathBuf;

use hdf5rs::types::Selection as RegionSelection;
use hdf5rs::{
    Array, Attribute, Dataset, Error, File, FileAccess, H5Type, HyperslabDim, Reference, Selection,
    Value,
};
use json::Json;
use sha256::sha256_hex;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn manifest() -> Json {
    Json::parse(&std::fs::read_to_string(fixtures_dir().join("manifest.json")).unwrap())
}

/// Flatten nested JSON lists to `depth` levels (the dataspace rank).
fn flatten<'a>(j: &'a Json, depth: usize, out: &mut Vec<&'a Json>) {
    match j {
        Json::Arr(v) if depth > 0 => v.iter().for_each(|x| flatten(x, depth - 1, out)),
        other => out.push(other),
    }
}

/// The expected elements of an object with `data`, flattened over its
/// dataspace (`None` for a null dataspace).
fn expected_elements(obj: &Json) -> Option<Vec<&Json>> {
    let data = obj.get("data")?;
    let space = obj.get("dataspace").unwrap();
    match space.get("class").and_then(Json::str) {
        Some("null") => {
            assert_eq!(*data, Json::Null);
            Some(Vec::new())
        }
        Some("scalar") => Some(vec![data]),
        _ => {
            let rank = space.get("shape").unwrap().arr().len();
            let mut out = Vec::new();
            flatten(data, rank, &mut out);
            Some(out)
        }
    }
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn want_float(j: &Json) -> f64 {
    match j {
        Json::Str(s) if s == "nan" => f64::NAN,
        Json::Str(s) if s == "inf" => f64::INFINITY,
        Json::Str(s) if s == "-inf" => f64::NEG_INFINITY,
        other => other.num().unwrap_or_else(|| panic!("float {other:?}")),
    }
}

fn float_eq(got: f64, want: f64, size: u64) -> bool {
    if got.is_nan() || want.is_nan() {
        return got.is_nan() && want.is_nan();
    }
    if size <= 4 {
        got as f32 == want as f32
    } else {
        got == want
    }
}

fn value_int(v: &Value) -> Option<i128> {
    match v {
        Value::Int(i) => Some(i128::from(*i)),
        Value::UInt(u) | Value::Bitfield(u) => Some(i128::from(*u)),
        Value::Enum { value, .. } => Some(*value),
        _ => None,
    }
}

/// All coordinates selected by a read selection in a dataspace of `dims`.
fn coords_of(sel: &Selection, dims: &[u64]) -> BTreeSet<Vec<u64>> {
    let per: Vec<Vec<u64>> = match sel {
        Selection::All => dims.iter().map(|&d| (0..d).collect()).collect(),
        Selection::Hyperslab(h) => h
            .iter()
            .map(|h| {
                (0..h.count)
                    .flat_map(|k| (0..h.block).map(move |b| h.start + k * h.stride + b))
                    .collect()
            })
            .collect(),
    };
    let mut out = BTreeSet::new();
    let mut cur = vec![Vec::new()];
    for axis in per {
        cur = cur
            .into_iter()
            .flat_map(|p| {
                axis.iter().map(move |&c| {
                    let mut q = p.clone();
                    q.push(c);
                    q
                })
            })
            .collect();
    }
    out.extend(cur);
    out
}

/// Coordinates of a manifest selection.
fn json_coords(j: &Json, dims: &[u64]) -> BTreeSet<Vec<u64>> {
    match j.get("selection").and_then(Json::str) {
        Some("all") => coords_of(&Selection::All, dims),
        Some("none") => BTreeSet::new(),
        Some("hyperslab") => {
            let mut out = BTreeSet::new();
            for b in j.get("blocks").unwrap().arr() {
                let s: Vec<u64> = b
                    .get("start")
                    .unwrap()
                    .arr()
                    .iter()
                    .filter_map(Json::u64)
                    .collect();
                let e: Vec<u64> = b
                    .get("end")
                    .unwrap()
                    .arr()
                    .iter()
                    .filter_map(Json::u64)
                    .collect();
                let h: Vec<HyperslabDim> = s
                    .iter()
                    .zip(&e)
                    .map(|(&s, &e)| HyperslabDim::range(s..e + 1))
                    .collect();
                out.extend(coords_of(&Selection::Hyperslab(h), dims));
            }
            out
        }
        Some("points") => j
            .get("points")
            .unwrap()
            .arr()
            .iter()
            .map(|p| p.arr().iter().filter_map(Json::u64).collect())
            .collect(),
        other => panic!("selection {other:?}"),
    }
}

struct Ctx<'a> {
    file: &'a File,
    what: String,
}

impl Ctx<'_> {
    fn address_of(&self, path: &str) -> u64 {
        self.file
            .object(path)
            .unwrap_or_else(|e| panic!("{}: {path}: {e}", self.what))
            .address()
    }

    fn check_region(&self, sel: &RegionSelection, j: &Json, target: u64) {
        let dataset = self
            .file
            .object(j.get("region_ref").and_then(Json::str).unwrap())
            .unwrap();
        assert_eq!(dataset.address(), target, "{}: region target", self.what);
        let dims = dataset.as_dataset().unwrap().shape().to_vec();
        let want = json_coords(j, &dims);
        let got = match sel {
            RegionSelection::Points { points, .. } => points.iter().cloned().collect(),
            other => match other.to_read_selection().unwrap() {
                None => BTreeSet::new(),
                Some(s) => coords_of(&s, &dims),
            },
        };
        assert_eq!(got, want, "{}: region selection", self.what);
        if let Some(n) = j.get("npoints").and_then(Json::u64) {
            assert_eq!(got.len() as u64, n, "{}: region npoints", self.what);
        }
    }

    /// Compare one decoded element with its manifest value.
    fn cmp(&self, v: &Value, j: &Json, dt: &Json) {
        let what = &self.what;
        let class = dt.get("class").and_then(Json::str).unwrap();
        let size = dt.get("size").and_then(Json::u64).unwrap_or(0);
        match class {
            "integer" | "bitfield" => {
                assert_eq!(value_int(v), j.int(), "{what}: {v:?} vs {j:?}")
            }
            "float" => {
                let got = v.as_f64().unwrap_or_else(|| panic!("{what}: {v:?}"));
                let want = want_float(j);
                assert!(float_eq(got, want, size), "{what}: {got} != {want}");
            }
            "enum" => match j {
                Json::Bool(b) => assert_eq!(v, &Value::Bool(*b), "{what}"),
                _ => assert_eq!(value_int(v), j.int(), "{what}: {v:?} vs {j:?}"),
            },
            "string" => match (v, j) {
                (Value::String(s), Json::Str(w)) => assert_eq!(s, w, "{what}"),
                (Value::Bytes(b), Json::Obj(_)) => {
                    assert_eq!(*b, hex(j.get("hex").and_then(Json::str).unwrap()), "{what}")
                }
                _ => panic!("{what}: {v:?} vs {j:?}"),
            },
            "opaque" => {
                let Value::Opaque(b) = v else {
                    panic!("{what}: {v:?}")
                };
                assert_eq!(*b, hex(j.get("hex").and_then(Json::str).unwrap()), "{what}");
            }
            "compound" => {
                let members = dt.get("members").unwrap().arr();
                if let Json::Arr(parts) = j {
                    // Complex numbers: [re, im].
                    let (re, im) = match v {
                        Value::Complex { re, im } => (*re, *im),
                        Value::Compound(_) => (
                            v.field("r").and_then(Value::as_f64).unwrap(),
                            v.field("i").and_then(Value::as_f64).unwrap(),
                        ),
                        _ => panic!("{what}: {v:?}"),
                    };
                    let msize = members[0]
                        .get("type")
                        .and_then(|t| t.get("size"))
                        .and_then(Json::u64)
                        .unwrap();
                    assert!(float_eq(re, want_float(&parts[0]), msize), "{what}: re");
                    assert!(float_eq(im, want_float(&parts[1]), msize), "{what}: im");
                    return;
                }
                let Value::Compound(fields) = v else {
                    panic!("{what}: {v:?}")
                };
                assert_eq!(fields.len(), members.len(), "{what}: member count");
                for m in members {
                    let name = m.get("name").and_then(Json::str).unwrap();
                    let fv = v.field(name).unwrap_or_else(|| panic!("{what}: no {name}"));
                    let sub = Ctx {
                        file: self.file,
                        what: format!("{what}.{name}"),
                    };
                    sub.cmp(fv, j.get(name).unwrap(), m.get("type").unwrap());
                }
            }
            "array" => {
                let dims: Vec<u64> = dt
                    .get("dims")
                    .unwrap()
                    .arr()
                    .iter()
                    .filter_map(Json::u64)
                    .collect();
                self.cmp_array(v, j, &dims, dt.get("base").unwrap());
            }
            "vlen" => {
                let Value::VarLen(items) = v else {
                    panic!("{what}: {v:?}")
                };
                let want = j.arr();
                assert_eq!(items.len(), want.len(), "{what}: vlen length");
                for (x, w) in items.iter().zip(want) {
                    self.cmp(x, w, dt.get("base").unwrap());
                }
            }
            "reference" => {
                let Value::Reference(r) = v else {
                    panic!("{what}: {v:?}")
                };
                match (r, j) {
                    (Reference::Null, Json::Null) => {}
                    (Reference::Object(addr), Json::Obj(_)) => {
                        let path = j.get("ref").and_then(Json::str).unwrap();
                        assert_eq!(*addr, self.address_of(path), "{what}: object ref");
                    }
                    (Reference::Region(reg), Json::Obj(_)) => {
                        self.check_region(&reg.selection, j, reg.object)
                    }
                    _ => panic!("{what}: {r:?} vs {j:?}"),
                }
            }
            other => panic!("{what}: unexpected class {other}"),
        }
    }

    fn cmp_array(&self, v: &Value, j: &Json, dims: &[u64], base: &Json) {
        let Some((&n, rest)) = dims.split_first() else {
            return self.cmp(v, j, base);
        };
        let Value::Array(items) = v else {
            panic!("{}: {v:?}", self.what)
        };
        assert_eq!(items.len() as u64, n, "{}: array length", self.what);
        assert_eq!(
            j.arr().len() as u64,
            n,
            "{}: manifest array length",
            self.what
        );
        for (x, w) in items.iter().zip(j.arr()) {
            self.cmp_array(x, w, rest, base);
        }
    }
}

/// Source of elements: a dataset or an attribute.
enum Item<'a> {
    D(&'a Dataset),
    A(&'a Attribute),
}

impl Item<'_> {
    fn read<T: H5Type>(&self) -> hdf5rs::Result<Array<T>> {
        match self {
            Item::D(d) => d.read_nd(),
            Item::A(a) => a.read_nd(),
        }
    }

    fn shape(&self) -> Vec<u64> {
        match self {
            Item::D(d) => d.shape().to_vec(),
            Item::A(a) => a.shape().to_vec(),
        }
    }
}

/// Typed read of an integer item into `T`, widened to i128.
fn typed_ints<T: H5Type + Into<i128>>(item: &Item<'_>) -> hdf5rs::Result<Vec<i128>> {
    Ok(item.read::<T>()?.data.into_iter().map(Into::into).collect())
}

/// Check one dataset or attribute. Returns Err only for unsupported
/// features.
fn check_item(ctx: &Ctx<'_>, item: &Item<'_>, obj: &Json) -> hdf5rs::Result<()> {
    let what = &ctx.what;
    let dt = obj.get("dtype").unwrap();
    let class = dt.get("class").and_then(Json::str).unwrap();
    let size = dt.get("size").and_then(Json::u64).unwrap_or(0);
    let signed = dt.get("signed").and_then(Json::bool) == Some(true);

    // Dynamic values: always.
    let values = item.read::<Value>()?;
    assert_eq!(values.shape, item.shape(), "{what}: shape");
    let count: u64 = values.shape.iter().product();
    let is_null = obj
        .get("dataspace")
        .and_then(|s| s.get("class"))
        .and_then(Json::str)
        == Some("null");
    assert_eq!(
        values.len() as u64,
        if is_null { 0 } else { count },
        "{what}: element count"
    );

    if let Some(summary) = obj.get("data_summary") {
        let n = summary.get("count").and_then(Json::u64).unwrap();
        assert_eq!(values.len() as u64, n, "{what}: summary count");
        // Typed read, re-encoded as little-endian bytes of the element size.
        let le: Vec<u8> = match (class, size, signed) {
            ("integer", 1, false) => item.read::<u8>()?.data,
            ("integer", 1, true) => item.read::<i8>()?.data.iter().map(|&v| v as u8).collect(),
            ("integer", 2, _) => typed_ints::<i32>(item)?
                .iter()
                .flat_map(|&v| (v as u16).to_le_bytes())
                .collect(),
            ("integer", 4, true) => item
                .read::<i32>()?
                .data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            ("integer", 4, false) => item
                .read::<u32>()?
                .data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            ("integer", 8, true) => item
                .read::<i64>()?
                .data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            ("integer", 8, false) => item
                .read::<u64>()?
                .data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            ("float", 4, _) => item
                .read::<f32>()?
                .data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            ("float", 8, _) => item
                .read::<f64>()?
                .data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            other => panic!("{what}: no summary check for {other:?}"),
        };
        let want = summary.get("sha256_le").and_then(Json::str).unwrap();
        assert_eq!(sha256_hex(&le), want, "{what}: sha256 of values");
        return Ok(());
    }

    let elements = expected_elements(obj).unwrap_or_else(|| panic!("{what}: no data"));
    assert_eq!(
        values.len(),
        elements.len(),
        "{what}: manifest element count"
    );

    if obj.get("data_encoding").and_then(Json::str) == Some("raw_hex") {
        for (i, (v, j)) in values.data.iter().zip(&elements).enumerate() {
            let want = hex(j.get("hex").and_then(Json::str).unwrap());
            match v {
                Value::Opaque(b) => assert_eq!(*b, want, "{what}[{i}]"),
                other => panic!("{what}[{i}]: {other:?}"),
            }
        }
        return Ok(());
    }

    for (i, (v, j)) in values.data.iter().zip(&elements).enumerate() {
        let sub = Ctx {
            file: ctx.file,
            what: format!("{what}[{i}]"),
        };
        sub.cmp(v, j, dt);
    }

    // Typed reads.
    match class {
        "integer" => {
            let got = match (size, signed) {
                (1, true) => typed_ints::<i8>(item)?,
                (1, false) => typed_ints::<u8>(item)?,
                (2, true) => typed_ints::<i16>(item)?,
                (2, false) => typed_ints::<u16>(item)?,
                (4, true) => typed_ints::<i32>(item)?,
                (4, false) => typed_ints::<u32>(item)?,
                (8, true) => typed_ints::<i64>(item)?,
                (8, false) => typed_ints::<u64>(item)?,
                _ => panic!("{what}: integer size {size}"),
            };
            let want: Vec<i128> = elements.iter().map(|j| j.int().unwrap()).collect();
            assert_eq!(got, want, "{what}: typed integers");
            // Widening reads.
            if size < 8 {
                let wide = typed_ints::<i64>(item)?;
                assert_eq!(wide, want, "{what}: widened to i64");
            }
        }
        "float" => {
            let got = item.read::<f64>()?.data;
            for (i, (g, j)) in got.iter().zip(&elements).enumerate() {
                assert!(float_eq(*g, want_float(j), size), "{what}[{i}]: f64");
            }
            if size <= 4 {
                let got = item.read::<f32>()?.data;
                for (i, (g, j)) in got.iter().zip(&elements).enumerate() {
                    assert!(
                        float_eq(f64::from(*g), want_float(j), size),
                        "{what}[{i}]: f32"
                    );
                }
            }
        }
        "string" => {
            if elements.iter().all(|j| j.str().is_some()) {
                let got = item.read::<String>()?.data;
                let want: Vec<&str> = elements.iter().map(|j| j.str().unwrap()).collect();
                assert_eq!(got, want, "{what}: typed strings");
            }
            if let (Some(hexes), Item::D(d)) = (obj.get("raw_elements_hex"), item) {
                let want: String = hexes.arr().iter().filter_map(Json::str).collect();
                assert_eq!(d.read_raw()?, hex(&want), "{what}: raw string bytes");
            }
        }
        "enum" if elements.iter().all(|j| matches!(j, Json::Bool(_))) => {
            let got = item.read::<bool>()?.data;
            let want: Vec<bool> = elements.iter().filter_map(|j| j.bool()).collect();
            assert_eq!(got, want, "{what}: typed bools");
        }
        _ => {}
    }
    Ok(())
}

#[test]
fn read_values_of_every_fixture() {
    let manifest = manifest();
    let mut checked = 0;
    let mut skipped = Vec::new();
    for (fname, info) in manifest.get("files").unwrap().entries() {
        let file = File::open(fixtures_dir().join(fname)).unwrap();
        for (path, obj) in info.get("objects").unwrap().entries() {
            let object = file
                .object(path)
                .unwrap_or_else(|e| panic!("{fname}:{path}: {e}"));
            if obj.get("kind").and_then(Json::str) == Some("dataset") {
                let ds = object.as_dataset().unwrap();
                let ctx = Ctx {
                    file: &file,
                    what: format!("{fname}:{path}"),
                };
                match check_item(&ctx, &Item::D(ds), obj) {
                    Ok(()) => checked += 1,
                    Err(Error::Unsupported(m)) => skipped.push(format!("{}: {m}", ctx.what)),
                    Err(e) => panic!("{}: {e}", ctx.what),
                }
            }
            for a in obj.get("attributes").map(Json::arr).unwrap_or_default() {
                let name = a.get("name").and_then(Json::str).unwrap();
                let attr = object.attribute(name).unwrap();
                let ctx = Ctx {
                    file: &file,
                    what: format!("{fname}:{path}@{name}"),
                };
                match check_item(&ctx, &Item::A(&attr), a) {
                    Ok(()) => checked += 1,
                    Err(Error::Unsupported(m)) => skipped.push(format!("{}: {m}", ctx.what)),
                    Err(e) => panic!("{}: {e}", ctx.what),
                }
            }
        }
    }
    eprintln!("checked {checked} datasets/attributes; skipped:");
    for s in &skipped {
        eprintln!("  {s}");
    }
    assert!(checked > 550, "only {checked} checked");
    // Only filters that are not implemented may be skipped.
    for s in &skipped {
        assert!(s.contains("filter"), "unexpected unsupported: {s}");
    }
}

// ------------------------------------------------------- focused tests --

fn fixture(name: &str) -> File {
    File::open(fixtures_dir().join(name)).unwrap()
}

#[test]
fn virtual_datasets() {
    let f = fixture("vds.h5");
    let v = f.dataset("/virtual").unwrap();
    assert!(v.is_virtual());
    let m = v.virtual_mappings().unwrap();
    assert_eq!(m.len(), 2);
    assert_eq!(m[1].source_file, "vds_source.h5");
    let all = v.read::<i32>().unwrap();
    let mut want: Vec<i32> = (0..20).collect();
    want.extend([-1; 5]);
    assert_eq!(all, want);
    // Selections crossing mappings and the unmapped tail.
    let part = v
        .read_selection::<i32>(&Selection::hyperslab(vec![HyperslabDim::new(3, 4, 6, 2)]))
        .unwrap();
    assert_eq!(part, [3, 4, 7, 8, 11, 12, 15, 16, 19, -1, -1, -1]);
    let s = f.dataset("/virtual_strided").unwrap();
    let arr = s
        .read_selection_nd::<i64>(&Selection::from_ranges(&[0..2, 1..4]))
        .unwrap();
    assert_eq!(arr.shape, [2, 3]);
    assert_eq!(arr.data, [2, 4, 6, 3, 5, 7]);
    assert_eq!(arr.get(&[1, 2]), Some(&7));

    // From memory: same-file sources work, other files cannot be opened.
    let bytes = std::fs::read(fixtures_dir().join("vds.h5")).unwrap();
    let mem = File::from_bytes(bytes).unwrap();
    assert_eq!(
        mem.dataset("/virtual_strided")
            .unwrap()
            .read::<i32>()
            .unwrap(),
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    );
    assert!(matches!(
        mem.dataset("/virtual").unwrap().read::<i32>(),
        Err(Error::Unsupported(_))
    ));
    // Access to other files disabled.
    let locked = File::open_with(
        fixtures_dir().join("vds.h5"),
        FileAccess::default().external_files(false),
    )
    .unwrap();
    assert!(matches!(
        locked.dataset("/virtual").unwrap().read::<i32>(),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn scalar_and_attribute_helpers() {
    let f = fixture("attributes_latest.h5");
    let root = f.root().unwrap();
    assert_eq!(
        root.attribute("root_int")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        1
    );
    let s = root.attribute("root_str").unwrap();
    assert_eq!(s.read_scalar::<String>().unwrap(), "root attribute");
    assert_eq!(s.value().unwrap(), Value::String("root attribute".into()));
    let ds = f.dataset("/dataset").unwrap();
    let arr = ds.attribute("array_type").unwrap();
    assert_eq!(arr.read::<[i32; 3]>().unwrap(), [[0, 1, 2], [3, 4, 5]]);
    assert_eq!(
        arr.read::<Vec<i64>>().unwrap(),
        [vec![0, 1, 2], vec![3, 4, 5]]
    );
    assert!(matches!(arr.value().unwrap(), Value::Array(v) if v.len() == 2));
    assert!(matches!(
        arr.read_scalar::<Vec<i32>>(),
        Err(Error::WrongObjectType(_))
    ));
    let vl = ds.attribute("vlen_int").unwrap();
    assert_eq!(vl.read::<Vec<i32>>().unwrap(), [vec![1, 2], vec![3]]);
    assert!(ds.attribute("bool").unwrap().read_scalar::<bool>().unwrap());
    // Type mismatches are errors, not panics.
    assert!(ds.read::<String>().is_err());
    assert!(root.attribute("root_str").unwrap().read::<i32>().is_err());

    let d = fixture("dataspaces.h5");
    for o in d.root().unwrap().objects().unwrap() {
        let Some(ds) = o.as_dataset() else { continue };
        let n = ds.element_count().unwrap();
        let r = ds.read_scalar::<Value>();
        if n == 1 {
            r.unwrap();
        } else {
            assert!(matches!(r, Err(Error::WrongObjectType(_))), "{}", ds.path());
        }
    }
}

#[test]
#[cfg(feature = "deflate")]
fn reader_options() {
    let f = fixture("big_gzip.h5");
    let ds = f.dataset("/arange_i32").unwrap();
    let r = ds
        .reader()
        .limits(hdf5rs::ReadLimits::with_max_output(1 << 16));
    assert!(matches!(r.read::<i32>(), Err(Error::LimitExceeded(_))));
    assert_eq!(
        r.read_selection::<i64>(&Selection::range(5..8)).unwrap(),
        [5, 6, 7]
    );
    // An empty registry cannot decode the chunks.
    let empty = hdf5rs::FilterRegistry::new();
    assert!(matches!(
        ds.reader()
            .filters(&empty)
            .read_selection::<i32>(&Selection::range(0..1)),
        Err(Error::Unsupported(_))
    ));
    assert!(!ds.chunks().unwrap().is_empty());
    assert!(ds.reader().filters(&empty).chunks().is_ok());
}

#[test]
fn external_links() {
    for name in ["groups_earliest.h5", "groups_latest.h5"] {
        let f = fixture(name);
        let root = f.root().unwrap();
        let ext = root
            .members()
            .unwrap()
            .into_iter()
            .find_map(|e| match e.link {
                hdf5rs::Link::External { .. } => Some(e.name),
                _ => None,
            });
        let Some(ext) = ext else { continue };
        assert!(matches!(f.object(&ext), Err(Error::Unsupported(_))));
        let target = root.follow_external(&ext).unwrap();
        assert!(target.file().path().unwrap().ends_with("superblock_v3.h5"));
        let followed = File::open_with(
            fixtures_dir().join(name),
            FileAccess::default().follow_external_links(true),
        )
        .unwrap();
        assert_eq!(followed.object(&ext).unwrap().address(), target.address());
        let locked = File::open_with(
            fixtures_dir().join(name),
            FileAccess::default()
                .follow_external_links(true)
                .external_files(false),
        )
        .unwrap();
        assert!(matches!(locked.object(&ext), Err(Error::Unsupported(_))));
        assert!(matches!(
            root.follow_external("nonexistent"),
            Err(Error::NotFound(_))
        ));
        return;
    }
    panic!("no external link found in the group fixtures");
}

#[test]
fn mutated_virtual_dataset_does_not_panic() {
    let base = std::fs::read(fixtures_dir().join("vds.h5")).unwrap();
    for i in 0..base.len() {
        let mut data = base.clone();
        data[i] ^= 0x5a;
        let Ok(f) = File::from_bytes(data) else {
            continue;
        };
        for p in ["/virtual", "/virtual_strided", "/local_src"] {
            if let Ok(d) = f.dataset(p) {
                let _ = d.virtual_mappings();
                let _ = d.read_values();
                let _ = d.read_selection::<i32>(&Selection::range(1..2));
            }
        }
    }
}
