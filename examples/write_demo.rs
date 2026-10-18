//! Write a demo HDF5 file exercising the writer.
//!
//! `cargo run --example write_demo -- out.h5 [v0|v2|v3] [o4] [times]`
//!
//! `o4` uses 4-byte offsets and lengths; `times` stores object timestamps.
//! The `/chunked` group holds chunked, filtered and resizable datasets
//! (chunk indexes: v1 B-tree for superblocks 0 and 2, the version 4
//! indexes for superblock 3).
//!
//! `tools/check_written.py` verifies the result with h5py.

use hdf5rs::format::Sizes;
use hdf5rs::format::messages::datatype::{
    ByteOrder, CharacterSet, CompoundMember, Datatype, DatatypeClass, StringPadding,
};
use hdf5rs::write::{
    AllocTime, EncodeContext, FileWriter, FixedAscii, GroupWriter, H5Write, Layout, ObjectRef,
    ScaleOffsetMode, WriteOptions,
};
use hdf5rs::{HyperslabDim, Selection};

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
    write_chunked(&mut root)?;
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

/// Chunked, filtered and resizable datasets (checked by
/// `tools/check_written.py`).
fn write_chunked(root: &mut GroupWriter<'_, std::fs::File>) -> hdf5rs::Result<()> {
    let mut g = root.create_group("chunked")?;
    let iota = |n: i32| (0..n).collect::<Vec<i32>>();
    // One per chunk index type (superblock 3).
    g.new_dataset::<i32>("single")
        .chunk(&[10])
        .write(&iota(10))?;
    g.new_dataset::<i32>("farray")
        .shape(&[6, 10])
        .chunk(&[4, 4])
        .write(&iota(60))?;
    g.new_dataset::<i32>("farray_paged")
        .chunk(&[2])
        .write(&iota(5001))?;
    g.new_dataset::<i32>("implicit")
        .shape(&[5, 7])
        .chunk(&[2, 3])
        .alloc_time(AllocTime::Early)
        .write(&iota(35))?;
    g.new_dataset::<i32>("earray")
        .shape(&[3, 700, 2])
        .max_shape(&[Some(3), None, Some(4)])
        .chunk(&[2, 1, 2])
        .write(&iota(4200))?;
    g.new_dataset::<i32>("btree2")
        .shape(&[60, 70])
        .max_shape(&[None, None])
        .chunk(&[1, 1])
        .write(&iota(4200))?;
    // The same indexes with filtered chunks.
    g.new_dataset::<i32>("single_gz")
        .chunk(&[10])
        .deflate(4)
        .write(&iota(10))?;
    g.new_dataset::<i32>("farray_gz")
        .chunk(&[2])
        .deflate(1)
        .write(&iota(5001))?;
    g.new_dataset::<i32>("earray_gz")
        .max_shape(&[None])
        .chunk(&[4])
        .shuffle()
        .deflate(1)
        .write(&iota(20000))?;
    g.new_dataset::<i32>("btree2_gz")
        .shape(&[40, 50])
        .max_shape(&[None, None])
        .chunk(&[1, 1])
        .deflate(1)
        .fletcher32()
        .write(&iota(2000))?;
    // Filters.
    let data: Vec<i32> = (0..1000).map(|i| (i % 37) * 3).collect();
    g.new_dataset::<i32>("gzip")
        .chunk(&[128])
        .deflate(6)
        .write(&data)?;
    g.new_dataset::<i32>("shuf_gzip_f32")
        .chunk(&[100])
        .shuffle()
        .deflate(9)
        .fletcher32()
        .write(&data)?;
    g.new_dataset::<i32>("lzf")
        .chunk(&[256])
        .lzf()
        .write(&data)?;
    let mut x = 0x1234_5678u32;
    let mut mixed: Vec<u32> = (0..64)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        })
        .collect();
    mixed.extend(std::iter::repeat_n(7u32, 64));
    g.new_dataset::<u32>("lzf_mixed")
        .chunk(&[64])
        .lzf()
        .write(&mixed)?;
    let mut i12 = Datatype::integer(2, true, ByteOrder::LittleEndian);
    if let DatatypeClass::FixedPoint(fp) = &mut i12.class {
        fp.bit_precision = 12;
    }
    let raw: Vec<u8> = (-500i16..500).flat_map(|v| v.to_le_bytes()).collect();
    g.new_dataset_with_type("nbit", i12)
        .chunk(&[100])
        .nbit()
        .write_raw(&raw)?;
    g.new_dataset::<i32>("so_int")
        .chunk(&[64])
        .scale_offset(ScaleOffsetMode::Integer { min_bits: 0 })
        .write(&data)?;
    let floats: Vec<f32> = (0..300).map(|i| i as f32 * 0.25).collect();
    g.new_dataset::<f32>("so_float")
        .chunk(&[50])
        .scale_offset(ScaleOffsetMode::FloatDecimal { decimal_digits: 2 })
        .write(&floats)?;
    g.new_dataset::<i32>("partial")
        .chunk(&[4])
        .deflate(5)
        .dont_filter_partial_chunks()
        .write(&iota(10))?;
    g.new_dataset::<i32>("edge2d")
        .shape(&[25, 40])
        .chunk(&[7, 9])
        .shuffle()
        .deflate(4)
        .write(&iota(1000))?;
    g.new_dataset::<f64>("auto")
        .deflate(1)
        .max_shape(&[None])
        .write(&[1.0, 2.0])?;
    // Resizable datasets.
    {
        let mut log = g
            .new_dataset::<i32>("log")
            .shape(&[0])
            .max_shape(&[None])
            .chunk(&[16])
            .deflate(1)
            .fill_value(-1)?
            .create()?;
        for i in 0..100 {
            log.append(&(0..7).map(|k| i * 7 + k).collect::<Vec<_>>())?;
        }
        log.resize(&[710])?;
    }
    {
        let mut t = g
            .new_dataset::<f64>("table")
            .shape(&[0, 3])
            .max_shape(&[None, Some(3)])
            .chunk(&[4, 3])
            .create()?;
        for r in 0..10 {
            let r = f64::from(r);
            t.append(&[r, r + 0.5, -r])?;
        }
    }
    g.new_dataset::<i32>("empty_ext")
        .shape(&[0, 4])
        .max_shape(&[None, None])
        .chunk(&[2, 2])
        .create()?;
    g.new_dataset::<i16>("fill_early")
        .shape(&[9])
        .chunk(&[4])
        .alloc_time(AllocTime::Early)
        .fill_value(-5)?
        .create()?;
    {
        let mut h = g
            .new_dataset::<u16>("hyper")
            .shape(&[20, 30])
            .chunk(&[6, 7])
            .deflate(2)
            .create()?;
        let sel = Selection::hyperslab(vec![
            HyperslabDim::new(1, 4, 3, 2),
            HyperslabDim::range(3..13),
        ]);
        h.write_selection(&sel, &(0..60).map(|i| 1000 + i).collect::<Vec<u16>>())?;
        let row: Vec<u16> = (0..30).collect();
        h.write_selection(&Selection::from_ranges(&[19..20, 0..30]), &row)?;
    }
    {
        let mut names = g
            .new_dataset::<String>("names")
            .shape(&[0])
            .max_shape(&[None])
            .chunk(&[2])
            .create()?;
        names.append(&["a".to_string(), "bb".into(), "ccc".into()])?;
        names.append(&["dddd".to_string()])?;
    }
    Ok(())
}
