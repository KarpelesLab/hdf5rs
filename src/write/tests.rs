//! Round-trip tests: files produced by the writer, read back with this
//! crate's reader.

use super::*;
use crate::format::messages::datatype::{ByteOrder, CharacterSet, StringPadding};
use crate::format::messages::layout::DataLayout;
use crate::read::{DatasetStorage, Selection};
use crate::types::{
    DecodeContext, GlobalHeapResolver, H5Type, Reference, Value, decode_all, decode_value,
};
use crate::{Attribute, Dataset, File, Link, LinkStorage, Object};

fn raw_data(file: &File, ds: &Dataset) -> Vec<u8> {
    let storage = DatasetStorage::new(
        file.store().as_ref(),
        file.sizes(),
        ds.layout(),
        ds.dataspace(),
        ds.datatype().size as usize,
    )
    .with_fill_value(ds.fill_value());
    storage.read(&Selection::All).unwrap()
}

fn read<T: H5Type>(file: &File, path: &str) -> Vec<T> {
    let ds = file.dataset(path).unwrap();
    let raw = raw_data(file, &ds);
    let resolver = GlobalHeapResolver::new(file.store().clone(), file.sizes());
    let ctx = DecodeContext::with_resolver(file.sizes(), &resolver);
    decode_all(ds.datatype(), &raw, &ctx).unwrap()
}

fn read_attr<T: H5Type>(file: &File, a: &Attribute) -> Vec<T> {
    let resolver = GlobalHeapResolver::new(file.store().clone(), file.sizes());
    let ctx = DecodeContext::with_resolver(file.sizes(), &resolver);
    decode_all(a.datatype(), a.raw_value(), &ctx).unwrap()
}

fn value(file: &File, path: &str) -> Vec<Value> {
    read::<Value>(file, path)
}

fn reopen(w: FileWriter<Vec<u8>>) -> File {
    File::from_bytes(w.finish().unwrap()).unwrap()
}

#[test]
fn empty_file() {
    let f = reopen(FileWriter::in_memory());
    assert_eq!(f.superblock().version, 2);
    let root = f.root().unwrap();
    assert!(root.is_empty().unwrap());
    assert_eq!(root.link_storage().unwrap(), LinkStorage::Compact);
    assert_eq!(f.superblock().eof_address, f.store().size().unwrap());
}

#[test]
fn numeric_datasets() {
    let mut w = FileWriter::in_memory();
    let mut root = w.root();
    root.new_dataset::<i8>("i8").write(&[-1i8, 2]).unwrap();
    root.new_dataset::<u16>("u16")
        .write(&[1u16, 65535])
        .unwrap();
    root.new_dataset::<i32>("i32")
        .write(&[i32::MIN, 5])
        .unwrap();
    root.new_dataset::<u64>("u64").write(&[u64::MAX]).unwrap();
    root.new_dataset::<f32>("f32")
        .write(&[0.5f32, -2.0])
        .unwrap();
    let grid: Vec<f64> = (0..6).map(f64::from).collect();
    root.new_dataset::<f64>("grid")
        .shape(&[2, 3])
        .write(&grid)
        .unwrap();
    root.new_dataset::<f64>("scalar")
        .shape(&[])
        .write(&[1.25f64])
        .unwrap();
    root.new_dataset::<bool>("b").write(&[true, false]).unwrap();
    root.new_dataset::<u8>("compact")
        .compact()
        .write(&[9u8, 8, 7])
        .unwrap();
    root.new_dataset::<i16>("filled")
        .shape(&[4])
        .fill_value(-3i16)
        .unwrap()
        .create()
        .unwrap();
    root.new_dataset::<i16>("filled_compact")
        .shape(&[2])
        .compact()
        .fill_value(11i16)
        .unwrap()
        .create()
        .unwrap();
    root.new_dataset::<u32>("unwritten")
        .shape(&[3])
        .create()
        .unwrap();
    root.new_dataset::<f32>("empty").write(&[]).unwrap();
    let f = reopen(w);

    assert_eq!(read::<i8>(&f, "/i8"), [-1, 2]);
    assert_eq!(read::<u16>(&f, "/u16"), [1, 65535]);
    assert_eq!(read::<i32>(&f, "/i32"), [i32::MIN, 5]);
    assert_eq!(read::<u64>(&f, "/u64"), [u64::MAX]);
    assert_eq!(read::<f32>(&f, "/f32"), [0.5, -2.0]);
    let g = f.dataset("/grid").unwrap();
    assert_eq!(g.shape(), &[2, 3]);
    assert!(g.is_contiguous());
    assert!(g.datatype().is_native_f64() || cfg!(target_endian = "big"));
    assert_eq!(read::<f64>(&f, "/grid"), grid);
    let s = f.dataset("/scalar").unwrap();
    assert!(s.shape().is_empty());
    assert_eq!(read::<f64>(&f, "/scalar"), [1.25]);
    assert_eq!(read::<bool>(&f, "/b"), [true, false]);
    let c = f.dataset("/compact").unwrap();
    assert!(c.is_compact());
    assert_eq!(read::<u8>(&f, "/compact"), [9, 8, 7]);
    assert_eq!(read::<i16>(&f, "/filled"), [-3; 4]);
    assert_eq!(read::<i16>(&f, "/filled_compact"), [11, 11]);
    let un = f.dataset("/unwritten").unwrap();
    assert!(matches!(
        un.layout(),
        DataLayout::Contiguous {
            address: UNDEFINED_ADDRESS,
            size: Some(12),
            ..
        }
    ));
    assert_eq!(read::<u32>(&f, "/unwritten"), [0; 3]);
    assert_eq!(f.dataset("/empty").unwrap().shape(), &[0]);
    assert!(read::<f32>(&f, "/empty").is_empty());
}

#[test]
fn strings_and_attributes() {
    let mut w = FileWriter::in_memory();
    let mut root = w.root();
    root.set_attr("title", "hello").unwrap();
    root.set_attr("n", &7u32).unwrap();
    root.set_attr("v", &[1.0f32, 2.0]).unwrap();
    root.set_attr("list", &vec!["a".to_string(), "bc".into()])
        .unwrap();
    root.set_attr("\u{3bb}", "utf-8 name").unwrap();
    root.set_attr("fixed", &FixedAscii::<4>::new("ab").unwrap())
        .unwrap();
    root.set_attr("n", &8u32).unwrap(); // replaces
    root.new_dataset::<String>("vlen")
        .write(&["x".to_string(), String::new(), "\u{e9}t\u{e9}".into()])
        .unwrap();
    root.new_dataset::<&str>("strs").write(&["p", "q"]).unwrap();
    root.new_dataset::<FixedUtf8<5>>("fixed")
        .write(&[FixedUtf8::<5>::new("caf\u{e9}").unwrap()])
        .unwrap();
    let f = reopen(w);
    let root = f.root().unwrap();
    let names = root.attribute_names().unwrap();
    assert_eq!(names.len(), 6);
    let attr = |n: &str| root.attribute(n).unwrap();
    assert_eq!(read_attr::<String>(&f, &attr("title")), ["hello"]);
    assert!(attr("title").shape().is_empty());
    assert_eq!(read_attr::<u32>(&f, &attr("n")), [8]);
    assert_eq!(read_attr::<f32>(&f, &attr("v")), [1.0, 2.0]);
    assert_eq!(attr("v").shape(), &[2]);
    assert_eq!(read_attr::<String>(&f, &attr("list")), ["a", "bc"]);
    let utf = attr("\u{3bb}");
    assert_eq!(utf.name_charset(), CharacterSet::Utf8);
    assert_eq!(attr("title").name_charset(), CharacterSet::Ascii);
    assert_eq!(read_attr::<String>(&f, &utf), ["utf-8 name"]);
    assert_eq!(read_attr::<String>(&f, &attr("fixed")), ["ab"]);
    assert_eq!(read::<String>(&f, "/vlen"), ["x", "", "\u{e9}t\u{e9}"]);
    assert_eq!(read::<String>(&f, "/strs"), ["p", "q"]);
    assert_eq!(read::<String>(&f, "/fixed"), ["caf\u{e9}"]);
}

struct Pt {
    a: u8,
    b: f64,
}

impl H5Write for Pt {
    fn datatype(sizes: Sizes) -> Datatype {
        Datatype::packed_compound(vec![
            ("a".into(), u8::datatype(sizes)),
            ("b".into(), f64::datatype(sizes)),
        ])
        .unwrap()
    }
    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> Result<()> {
        self.a.encode(out, ctx)?;
        self.b.encode(out, ctx)
    }
}

#[test]
fn compound_enum_array_vlen() {
    let mut w = FileWriter::in_memory();
    let mut root = w.root();
    root.new_dataset::<Pt>("pts")
        .write(&[Pt { a: 1, b: 2.5 }, Pt { a: 3, b: -1.0 }])
        .unwrap();
    let e = Datatype::enumeration(
        Datatype::integer(2, true, ByteOrder::BigEndian),
        &[("LO", -1), ("HI", 300)],
    )
    .unwrap();
    root.new_dataset_with_type("enum", e)
        .write_raw(&[0x01, 0x2c, 0xff, 0xff])
        .unwrap();
    root.new_dataset::<[u16; 2]>("arr")
        .write(&[[1, 2], [3, 4]])
        .unwrap();
    root.new_dataset::<Vec<i64>>("seq")
        .write(&[vec![1, 2, 3], vec![], vec![-4]])
        .unwrap();
    root.new_dataset::<Vec<String>>("nested")
        .write(&[vec!["a".to_string(), "b".to_string()]])
        .unwrap();
    let f = reopen(w);
    let pts = value(&f, "/pts");
    assert_eq!(pts[0].field("a").and_then(Value::as_u64), Some(1));
    assert_eq!(pts[1].field("b").and_then(Value::as_f64), Some(-1.0));
    let en = value(&f, "/enum");
    assert_eq!(en[0].to_string(), "HI");
    assert_eq!(en[1].to_string(), "LO");
    assert_eq!(read::<[u16; 2]>(&f, "/arr"), [[1, 2], [3, 4]]);
    assert_eq!(
        read::<Vec<i64>>(&f, "/seq"),
        [vec![1, 2, 3], vec![], vec![-4]]
    );
    assert_eq!(read::<Vec<String>>(&f, "/nested"), [vec!["a", "b"]]);
}

#[test]
fn groups_links_and_references() {
    let mut w = FileWriter::in_memory();
    let mut root = w.root();
    let mut g = root.create_group("g").unwrap();
    let mut sub = g.create_group("sub").unwrap();
    let ds = sub.new_dataset::<i32>("d").write(&[1, 2]).unwrap().id();
    let gid = g.id();
    let mut root = w.root();
    root.link_hard("alias", ds).unwrap();
    root.link_soft("soft", "/g/sub/d").unwrap();
    root.link_external("ext", "other.h5", "/x").unwrap();
    root.new_dataset::<ObjectRef>("refs")
        .write(&[ObjectRef::to(ds), ObjectRef::to(gid), ObjectRef::NULL])
        .unwrap();
    root.new_dataset::<ObjectRef>("cref")
        .compact()
        .write(&[ObjectRef::to(ds)])
        .unwrap();
    root.set_attr("aref", &ObjectRef::to(gid)).unwrap();
    // A cycle: the subgroup links back to the root.
    let rid = w.root().id();
    w.group("/g/sub").unwrap().link_hard("up", rid).unwrap();
    assert!(w.group("/g/sub/d").is_err());
    assert!(w.dataset("/g/sub/d").is_ok());
    let f = reopen(w);

    let d = f.dataset("/g/sub/d").unwrap();
    assert_eq!(d.header().ref_count, 2);
    assert_eq!(f.dataset("/alias").unwrap().address(), d.address());
    assert_eq!(f.dataset("/soft").unwrap().address(), d.address());
    assert_eq!(f.root().unwrap().header().ref_count, 2);
    assert_eq!(
        f.group("/g/sub/up").unwrap().address(),
        f.root().unwrap().address()
    );
    let root = f.root().unwrap();
    assert_eq!(
        root.link("soft").unwrap(),
        Link::Soft {
            path: "/g/sub/d".into()
        }
    );
    assert_eq!(
        root.link("ext").unwrap(),
        Link::External {
            file: "other.h5".into(),
            path: "/x".into()
        }
    );
    let g = f.group("/g").unwrap();
    let refs = read::<Reference>(&f, "/refs");
    assert_eq!(refs[0], Reference::Object(d.address()));
    assert_eq!(refs[1], Reference::Object(g.address()));
    assert!(matches!(refs[2], Reference::Object(0) | Reference::Null));
    assert_eq!(
        read::<Reference>(&f, "/cref"),
        [Reference::Object(d.address())]
    );
    let a = root.attribute("aref").unwrap();
    assert_eq!(
        read_attr::<Reference>(&f, &a),
        [Reference::Object(g.address())]
    );
}

#[test]
fn committed_datatype() {
    let mut w = FileWriter::in_memory();
    let mut root = w.root();
    let t = root
        .commit_datatype("t", Pt::datatype(Sizes::DEFAULT))
        .unwrap();
    root.new_dataset_committed("d", t)
        .unwrap()
        .with_element_type::<Pt>()
        .write(&[Pt { a: 5, b: 0.5 }])
        .unwrap();
    assert!(root.new_dataset_committed("bad", root.id()).is_err());
    let f = reopen(w);
    let Object::NamedDatatype(nt) = f.object("/t").unwrap() else {
        panic!("expected a named datatype");
    };
    assert_eq!(nt.header().ref_count, 2);
    let d = f.dataset("/d").unwrap();
    assert_eq!(d.committed_datatype_address(), Some(nt.address()));
    assert_eq!(d.datatype(), nt.datatype());
    assert_eq!(
        value(&f, "/d")[0].field("a").and_then(Value::as_u64),
        Some(5)
    );
}

#[test]
fn many_links_and_attributes() {
    let mut w = FileWriter::in_memory();
    let mut root = w.root();
    for i in 0..50 {
        root.create_group(&format!("g{i}")).unwrap();
        root.set_attr(&format!("a{i}"), &i).unwrap();
    }
    let f = reopen(w);
    let root = f.root().unwrap();
    assert_eq!(root.len().unwrap(), 50);
    assert_eq!(root.attribute_names().unwrap().len(), 50);
    let a = root.attribute("a42").unwrap();
    assert_eq!(read_attr::<i32>(&f, &a), [42]);
    assert_eq!(root.link_storage().unwrap(), LinkStorage::Dense);
    assert!(root.header().attribute_phase_change.is_none());
}

#[test]
fn options_superblock_and_sizes() {
    for (version, sizes) in [
        (3, Sizes::DEFAULT),
        (
            2,
            Sizes {
                offset: 4,
                length: 4,
            },
        ),
        (
            3,
            Sizes {
                offset: 4,
                length: 8,
            },
        ),
        (
            2,
            Sizes {
                offset: 2,
                length: 2,
            },
        ),
        (
            0,
            Sizes {
                offset: 4,
                length: 4,
            },
        ),
    ] {
        let opts = WriteOptions {
            superblock_version: version,
            sizes,
            track_times: true,
        };
        let mut w = FileWriter::new(Vec::new(), opts).unwrap();
        let mut root = w.root();
        root.new_dataset::<&str>("s").write(&["abc"]).unwrap();
        root.new_dataset::<u8>("d").write(&[1u8, 2]).unwrap();
        root.set_attr("r", &ObjectRef::to(root.id())).unwrap();
        let f = reopen(w);
        assert_eq!(f.superblock().version, version);
        assert_eq!(f.sizes(), sizes);
        assert_eq!(read::<String>(&f, "/s"), ["abc"]);
        assert_eq!(read::<u8>(&f, "/d"), [1, 2]);
        if version == 0 {
            assert!(f.root().unwrap().modification_time().unwrap().is_some());
        } else {
            assert!(f.root().unwrap().times().is_some());
        }
        let a = f.root().unwrap().attribute("r").unwrap();
        assert_eq!(
            read_attr::<Reference>(&f, &a),
            [Reference::Object(f.root().unwrap().address())]
        );
    }
    assert!(
        FileWriter::new(
            Vec::new(),
            WriteOptions {
                superblock_version: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    let bad = WriteOptions {
        sizes: Sizes {
            offset: 3,
            length: 8,
        },
        ..Default::default()
    };
    assert!(FileWriter::new(Vec::new(), bad).is_err());
}

#[test]
fn small_offsets_overflow_is_an_error() {
    let opts = WriteOptions {
        sizes: Sizes {
            offset: 2,
            length: 2,
        },
        ..Default::default()
    };
    let mut w = FileWriter::new(Vec::new(), opts).unwrap();
    let r = w
        .root()
        .new_dataset::<u8>("big")
        .write(&vec![0u8; 70000])
        .map(|_| ());
    assert!(matches!(r, Err(Error::LimitExceeded(_))));
}

#[test]
fn deterministic_output() {
    let build = || {
        let mut w = FileWriter::in_memory();
        let mut root = w.root();
        root.set_attr("x", "y").unwrap();
        root.create_group("a")
            .unwrap()
            .new_dataset::<i32>("d")
            .write(&[1])
            .unwrap();
        w.finish().unwrap()
    };
    assert_eq!(build(), build());
}

#[test]
fn errors() {
    let mut w = FileWriter::in_memory();
    let mut root = w.root();
    root.create_group("g").unwrap();
    assert!(root.create_group("g").is_err());
    assert!(root.create_group("").is_err());
    assert!(root.create_group("a/b").is_err());
    assert!(root.create_group(".").is_err());
    assert!(root.new_dataset::<i32>("g").write(&[1]).is_err());
    assert!(
        root.new_dataset::<i32>("x")
            .shape(&[3])
            .write(&[1, 2])
            .is_err()
    );
    assert!(root.new_dataset::<i32>("x").create().is_err());
    let r = root
        .new_dataset_with_type("x", Datatype::integer(4, true, ByteOrder::BigEndian))
        .with_element_type::<i32>()
        .write(&[1i32]);
    assert!(r.is_err());
    assert!(root.new_dataset::<i32>("x").write_raw(&[1, 2, 3]).is_err());
    assert!(root.new_dataset::<String>("x").write_raw(&[0; 16]).is_err());
    assert!(
        root.new_dataset::<u8>("x")
            .compact()
            .write(&vec![0u8; 70000])
            .is_err()
    );
    assert!(
        root.new_dataset::<String>("x")
            .fill_value(String::new())
            .is_err()
    );
    assert!(root.set_attr("", &1).is_err());
    assert!(
        root.set_attr_raw(
            "r",
            Datatype::integer(4, true, ByteOrder::LittleEndian),
            None,
            &[1]
        )
        .is_err()
    );
    assert!(root.link_soft("s", "").is_err());
    assert!(root.link_hard("h", ObjectId(999)).is_err());
    assert!(w.group("/missing").is_err());
    assert!(w.dataset("/g").is_err());
    w.root().link_soft("soft", "/g").unwrap();
    assert!(w.group("/soft").is_err());
    // The writer is still usable after errors.
    w.root().new_dataset::<i32>("x").write(&[1]).unwrap();
    let f = reopen(w);
    assert_eq!(f.root().unwrap().link_names().unwrap(), ["g", "soft", "x"]);
}

#[test]
fn raw_attributes_and_paths() {
    let mut w = FileWriter::in_memory();
    w.root().create_group("g").unwrap();
    let dt = Datatype::fixed_string(3, StringPadding::SpacePad, CharacterSet::Ascii);
    w.group("/g")
        .unwrap()
        .set_attr_raw("s", dt, Some(&[2]), b"ab cd ")
        .unwrap();
    w.set_attr("/g", "n", &1.5f64).unwrap();
    let mut g = w.group("g").unwrap();
    let mut d = g.new_dataset::<u8>("d").write(&[1u8]).unwrap();
    d.set_attr("k", &2u8).unwrap();
    d.set_attr_raw("o", Datatype::opaque(2, "t"), None, &[1, 2])
        .unwrap();
    let f = reopen(w);
    let g = f.group("/g").unwrap();
    assert_eq!(
        read_attr::<String>(&f, &g.attribute("s").unwrap()),
        ["ab", "cd"]
    );
    assert_eq!(read_attr::<f64>(&f, &g.attribute("n").unwrap()), [1.5]);
    let d = f.dataset("/g/d").unwrap();
    assert_eq!(d.attribute("o").unwrap().raw_value(), &[1, 2]);
    let ctx = DecodeContext::default();
    let k = d.attribute("k").unwrap();
    assert_eq!(
        decode_value(k.datatype(), k.raw_value(), &ctx)
            .unwrap()
            .as_u64(),
        Some(2)
    );
}

#[test]
fn file_and_seek_storage() {
    let dir = std::env::temp_dir().join(format!("hdf5rs-write-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.h5");
    let mut w = FileWriter::create(&path).unwrap();
    w.root().new_dataset::<f64>("x").write(&[1.0, 2.0]).unwrap();
    w.finish().unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read::<f64>(&f, "/x"), [1.0, 2.0]);
    std::fs::remove_dir_all(&dir).unwrap();

    let mut w = FileWriter::new(
        SeekWriter(std::io::Cursor::new(Vec::new())),
        WriteOptions::default(),
    )
    .unwrap();
    w.root().new_dataset::<u8>("x").write(&[3u8]).unwrap();
    let bytes = w.finish().unwrap().0.into_inner();
    let f = File::from_bytes(bytes).unwrap();
    assert_eq!(read::<u8>(&f, "/x"), [3]);
}

#[test]
fn large_vlen_data_spans_collections() {
    let mut w = FileWriter::in_memory();
    let strings: Vec<String> = (0..500).map(|i| format!("string number {i:04}")).collect();
    let big = "z".repeat(20000);
    let mut root = w.root();
    root.new_dataset::<String>("many").write(&strings).unwrap();
    root.new_dataset::<&str>("big")
        .write(&[big.as_str()])
        .unwrap();
    let f = reopen(w);
    assert_eq!(read::<String>(&f, "/many"), strings);
    assert_eq!(read::<String>(&f, "/big"), [big]);
}

#[test]
fn reads_reference_vlen_with_4_byte_lengths() {
    // Global heap objects written by the reference library with 4-byte
    // lengths have 8-byte aligned headers.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sizes_4byte_latest.h5"
    );
    let f = File::open(path).unwrap();
    let a = f.root().unwrap().attribute("note").unwrap();
    assert_eq!(
        read_attr::<String>(&f, &a),
        ["custom file creation properties"]
    );
}

#[test]
fn dense_links_and_attributes() {
    let mut w = FileWriter::in_memory();
    let mut root = w.root();
    let mut g = root.create_group("big").unwrap();
    for i in 0..2000u32 {
        g.new_dataset::<u32>(&format!("d{i:04}"))
            .shape(&[])
            .write(&[i])
            .unwrap();
    }
    g.link_soft("soft", "/big/d0007").unwrap();
    let mut ds = root.new_dataset::<u8>("with_attrs").write(&[1u8]).unwrap();
    for i in 0..12 {
        ds.set_attr(&format!("attr{i}"), &format!("value {i}"))
            .unwrap();
    }
    let big: Vec<u16> = (0..40000).map(|i| i as u16).collect();
    ds.set_attr("huge", &big).unwrap();
    let mut root = w.root();
    let mut one = root.create_group("one_big_attr").unwrap();
    one.set_attr("blob", &vec![7u8; 100_000]).unwrap();
    let f = reopen(w);

    let g = f.group("/big").unwrap();
    assert_eq!(g.link_storage().unwrap(), LinkStorage::Dense);
    assert_eq!(g.len().unwrap(), 2001);
    assert_eq!(read::<u32>(&f, "/big/d1234"), [1234]);
    assert_eq!(read::<u32>(&f, "/big/soft"), [7]);
    assert!(!g.contains("d2000").unwrap());
    let ds = f.dataset("/with_attrs").unwrap();
    assert_eq!(ds.attribute_names().unwrap().len(), 13);
    assert_eq!(
        read_attr::<String>(&f, &ds.attribute("attr11").unwrap()),
        ["value 11"]
    );
    assert_eq!(read_attr::<u16>(&f, &ds.attribute("huge").unwrap()), big);
    let one = f.group("/one_big_attr").unwrap();
    assert_eq!(one.link_storage().unwrap(), LinkStorage::Compact);
    assert_eq!(
        one.attribute("blob").unwrap().raw_value(),
        vec![7u8; 100_000]
    );
}

fn compat_writer() -> FileWriter<Vec<u8>> {
    FileWriter::new(
        Vec::new(),
        WriteOptions {
            superblock_version: 0,
            ..WriteOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn compat_format_round_trip() {
    let mut w = compat_writer();
    let mut root = w.root();
    root.set_attr("title", "old style").unwrap();
    root.set_attr("v", &[1i16, 2, 3]).unwrap();
    let mut g = root.create_group("g").unwrap();
    for i in 0..300u32 {
        g.new_dataset::<u32>(&format!("d{i:03}"))
            .shape(&[])
            .write(&[i])
            .unwrap();
    }
    for i in 0..20 {
        g.set_attr(&format!("a{i}"), &i).unwrap();
    }
    let gid = g.id();
    g.link_soft("soft", "/g/d042").unwrap();
    assert!(g.link_external("ext", "x.h5", "/y").is_err());
    let mut root = w.root();
    root.new_dataset::<Pt>("pts")
        .compact()
        .write(&[Pt { a: 1, b: 2.0 }])
        .unwrap();
    root.new_dataset::<[i32; 2]>("arr")
        .write(&[[1, 2], [3, 4]])
        .unwrap();
    root.new_dataset::<String>("s")
        .write(&["x".to_string(), "yz".into()])
        .unwrap();
    root.new_dataset::<bool>("b")
        .fill_value(true)
        .unwrap()
        .shape(&[3])
        .create()
        .unwrap();
    root.new_dataset::<ObjectRef>("r")
        .write(&[ObjectRef::to(gid)])
        .unwrap();
    let t = root
        .commit_datatype("t", Datatype::integer(2, false, ByteOrder::BigEndian))
        .unwrap();
    root.new_dataset_committed("c", t)
        .unwrap()
        .write_raw(&[0, 1, 0, 2])
        .unwrap();
    root.create_group("empty").unwrap();
    root.link_hard("g_again", gid).unwrap();
    assert!(root.set_attr("big", &vec![0u8; 70000]).is_err());
    let f = reopen(w);

    assert_eq!(f.superblock().version, 0);
    let root = f.root().unwrap();
    assert_eq!(root.link_storage().unwrap(), LinkStorage::SymbolTable);
    assert_eq!(root.header().version, 1);
    assert_eq!(
        read_attr::<String>(&f, &root.attribute("title").unwrap()),
        ["old style"]
    );
    assert_eq!(
        read_attr::<i16>(&f, &root.attribute("v").unwrap()),
        [1, 2, 3]
    );
    let g = f.group("/g").unwrap();
    assert_eq!(g.len().unwrap(), 301);
    assert_eq!(g.header().ref_count, 2);
    assert_eq!(g.attribute_names().unwrap().len(), 20);
    assert_eq!(read::<u32>(&f, "/g/d299"), [299]);
    assert_eq!(read::<u32>(&f, "/g/soft"), [42]);
    assert_eq!(read::<u32>(&f, "/g_again/d007"), [7]);
    assert_eq!(
        value(&f, "/pts")[0].field("b").and_then(Value::as_f64),
        Some(2.0)
    );
    assert_eq!(read::<[i32; 2]>(&f, "/arr"), [[1, 2], [3, 4]]);
    assert_eq!(read::<String>(&f, "/s"), ["x", "yz"]);
    assert_eq!(read::<bool>(&f, "/b"), [true; 3]);
    assert_eq!(
        read::<Reference>(&f, "/r"),
        [Reference::Object(g.address())]
    );
    assert_eq!(value(&f, "/c")[1].as_u64(), Some(2));
    assert!(f.group("/empty").unwrap().is_empty().unwrap());
}
