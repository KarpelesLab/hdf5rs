//! Write a demo HDF5 file exercising the writer.
//!
//! `cargo run --example write_demo -- out.h5 [v0|v2|v3] [o4] [times]`
//!
//! `o4` uses 4-byte offsets and lengths; `times` stores object timestamps.
//!
//! `tools/check_written.py` verifies the result with h5py.

use hdf5rs::format::Sizes;
use hdf5rs::format::messages::datatype::{
    ByteOrder, CharacterSet, CompoundMember, Datatype, StringPadding,
};
use hdf5rs::write::{
    EncodeContext, FileWriter, FixedAscii, H5Write, Layout, ObjectRef, WriteOptions,
};

/// A compound element.
struct Point {
    x: f64,
    y: f64,
    id: i32,
}

impl H5Write for Point {
    fn datatype(sizes: Sizes) -> Datatype {
        Datatype::packed_compound(vec![
            ("x".into(), f64::datatype(sizes)),
            ("y".into(), f64::datatype(sizes)),
            ("id".into(), i32::datatype(sizes)),
        ])
        .expect("valid compound")
    }

    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> hdf5rs::Result<()> {
        self.x.encode(out, ctx)?;
        self.y.encode(out, ctx)?;
        self.id.encode(out, ctx)
    }
}

fn main() -> hdf5rs::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: write_demo OUT.h5 [v0|v2|v3] [o4] [times]");
        std::process::exit(2);
    };
    let mut options = WriteOptions::default();
    for arg in args {
        match arg.as_str() {
            "v2" => options.superblock_version = 2,
            "v3" => options.superblock_version = 3,
            "v0" => options.superblock_version = 0,
            "o4" => {
                options.sizes = Sizes {
                    offset: 4,
                    length: 4,
                }
            }
            "times" => options.track_times = true,
            other => {
                eprintln!("unknown option {other:?}");
                std::process::exit(2);
            }
        }
    }
    let compat = options.superblock_version == 0;
    let mut f = FileWriter::create_with(&path, options)?;
    let sizes = f.sizes();
    let mut root = f.root();
    root.set_attr("title", "hdf5rs write demo")?;
    root.set_attr("version", &3i32)?;
    root.set_attr("scale", &[0.5f64, 1.5, 2.5])?;
    root.set_attr("flag", &true)?;
    root.set_attr(
        "names",
        &vec!["alpha".to_string(), "beta".into(), "\u{3b3}".into()],
    )?;
    root.set_attr("fixed", &FixedAscii::<8>::new("abc")?)?;

    // Numeric datasets of every primitive type.
    {
        let mut g = root.create_group("numbers")?;
        g.new_dataset::<i8>("i8").write(&[-128i8, -1, 0, 1, 127])?;
        g.new_dataset::<u8>("u8").write(&[0u8, 1, 255])?;
        g.new_dataset::<i16>("i16").write(&[-32768i16, 0, 32767])?;
        g.new_dataset::<u16>("u16").write(&[0u16, 65535])?;
        g.new_dataset::<i32>("i32")
            .write(&[i32::MIN, 0, i32::MAX])?;
        g.new_dataset::<u32>("u32").write(&[0u32, u32::MAX])?;
        g.new_dataset::<i64>("i64")
            .write(&[i64::MIN, 0, i64::MAX])?;
        g.new_dataset::<u64>("u64").write(&[0u64, u64::MAX])?;
        g.new_dataset::<f32>("f32")
            .write(&[1.5f32, -0.25, f32::INFINITY])?;
        let grid: Vec<f64> = (0..12).map(|i| i as f64 * 0.5).collect();
        let mut ds = g.new_dataset::<f64>("grid").shape(&[3, 4]).write(&grid)?;
        ds.set_attr("units", "m")?;
        ds.set_attr("valid_range", &[0.0f64, 10.0])?;
        g.new_dataset::<f64>("scalar")
            .shape(&[])
            .write(&[42.0f64])?;
        g.new_dataset::<bool>("bools").write(&[true, false, true])?;
        g.new_dataset::<u16>("compact")
            .layout(Layout::Compact)
            .write(&[1u16, 2, 3, 4])?;
        g.new_dataset::<i32>("filled")
            .shape(&[2, 5])
            .fill_value(-7i32)?
            .create()?;
        g.new_dataset::<f32>("empty").shape(&[0]).write(&[])?;
        g.new_dataset_with_type("be_u32", Datatype::integer(4, false, ByteOrder::BigEndian))
            .write_raw(&[0, 0, 1, 0, 0, 0, 0, 2])?;
    }

    // Strings.
    {
        let mut g = root.create_group("strings")?;
        g.new_dataset::<String>("vlen").write(&[
            "hello".to_string(),
            "".into(),
            "w\u{f6}rld".into(),
        ])?;
        g.new_dataset::<&str>("vlen_scalar")
            .shape(&[])
            .write(&["just one"])?;
        let fixed: Vec<FixedAscii<6>> = ["ab", "cdef", "ghijkl"]
            .iter()
            .map(|s| FixedAscii::new(s))
            .collect::<hdf5rs::Result<_>>()?;
        g.new_dataset::<FixedAscii<6>>("fixed").write(&fixed)?;
        let utf8 = Datatype::fixed_string(8, StringPadding::NullTerminate, CharacterSet::Utf8);
        g.new_dataset_with_type("fixed_utf8", utf8)
            .write_raw(b"caf\xc3\xa9\0\0\0")?;
    }

    // Compound, enum, array, vlen sequences, opaque, bitfield.
    {
        let mut g = root.create_group("types")?;
        g.new_dataset::<Point>("points").write(&[
            Point {
                x: 1.0,
                y: 2.0,
                id: 7,
            },
            Point {
                x: -1.0,
                y: 0.5,
                id: 8,
            },
        ])?;
        let color = Datatype::enumeration(
            Datatype::integer(1, false, ByteOrder::LittleEndian),
            &[("RED", 0), ("GREEN", 1), ("BLUE", 2)],
        )?;
        g.new_dataset_with_type("colors", color)
            .write_raw(&[0, 2, 1, 1])?;
        g.new_dataset::<[i32; 3]>("arrays")
            .write(&[[1, 2, 3], [4, 5, 6]])?;
        g.new_dataset::<Vec<i32>>("ragged")
            .write(&[vec![1], vec![], vec![2, 3, 4]])?;
        g.new_dataset_with_type("opaque", Datatype::opaque(4, "demo-tag"))
            .write_raw(&[1, 2, 3, 4, 5, 6, 7, 8])?;
        g.new_dataset_with_type("bits", Datatype::bitfield(1, ByteOrder::LittleEndian))
            .write_raw(&[0b1010_0101, 0xff])?;
        let padded = Datatype::compound(
            16,
            vec![
                CompoundMember {
                    name: "a".into(),
                    offset: 0,
                    array_dims: None,
                    datatype: Datatype::integer(1, false, ByteOrder::LittleEndian),
                },
                CompoundMember {
                    name: "b".into(),
                    offset: 8,
                    array_dims: None,
                    datatype: Datatype::float(8, ByteOrder::LittleEndian)?,
                },
            ],
        )?;
        let mut raw = vec![0u8; 16];
        raw[0] = 9;
        raw[8..].copy_from_slice(&2.25f64.to_le_bytes());
        g.new_dataset_with_type("padded", padded).write_raw(&raw)?;
        let named = g.commit_datatype("point_t", Point::datatype(sizes))?;
        let mut raw = Vec::new();
        for v in [3.0f64, 4.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        raw.extend_from_slice(&5i32.to_le_bytes());
        g.new_dataset_committed("named_points", named)?
            .write_raw(&raw)?;
    }

    // Links and references.
    let target = f.object_id("/numbers/grid")?;
    let group = f.object_id("/strings")?;
    let mut root = f.root();
    {
        let mut links = root.create_group("links")?;
        links.link_hard("grid_again", target)?;
        links.link_soft("soft_grid", "/numbers/grid")?;
        links.link_soft("dangling", "/does/not/exist")?;
        if !compat {
            // Not representable in old-style (symbol table) groups.
            links.link_external("ext", "other.h5", "/some/object")?;
        }
        links.new_dataset::<ObjectRef>("refs").write(&[
            ObjectRef::to(target),
            ObjectRef::to(group),
            ObjectRef::NULL,
        ])?;
        links.set_attr("grid_ref", &ObjectRef::to(target))?;
    }
    {
        let mut many = root.create_group("many")?;
        for i in 0..20 {
            many.create_group(&format!("g{i:02}"))?;
            many.set_attr(&format!("a{i:02}"), &(i as i64))?;
        }
        // Dense storage: 1500 links, plus huge attributes.
        let mut huge = many.create_group("huge")?;
        for i in 0..1500u32 {
            huge.new_dataset::<u32>(&format!("item{i:04}"))
                .shape(&[])
                .write(&[i])?;
        }
        // Compatibility-format attributes must fit a 64 KiB header message.
        let n = if compat { 2000 } else { 20000 };
        let blob: Vec<f64> = (0..n).map(f64::from).collect();
        huge.set_attr("blob", &blob)?;
        huge.set_attr("note", "a dense group")?;
    }
    {
        let mut deep = root.create_group("a")?;
        let mut b = deep.create_group("b")?;
        let mut c = b.create_group("c")?;
        c.set_attr("depth", &3u8)?;
        c.set_attr("na\u{ef}ve \u{1f600}", "utf-8 attribute name")?;
    }
    f.finish()?;
    println!("wrote {path}");
    Ok(())
}
