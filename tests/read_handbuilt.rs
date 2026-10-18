//! High-level API tests against tiny hand-assembled files (one old-style
//! v0 file and one v2 file with compact links).

use hdf5rs::checksum::metadata_checksum;
use hdf5rs::{Error, File, Link, LinkStorage, Object, ObjectKind};

const UNDEF: u64 = u64::MAX;

#[derive(Default)]
struct B {
    data: Vec<u8>,
}

impl B {
    fn put(&mut self, addr: usize, bytes: &[u8]) {
        if self.data.len() < addr + bytes.len() {
            self.data.resize(addr + bytes.len(), 0);
        }
        self.data[addr..addr + bytes.len()].copy_from_slice(bytes);
    }
}

fn u16le(v: u16) -> [u8; 2] {
    v.to_le_bytes()
}
fn u32le(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}
fn u64le(v: u64) -> [u8; 8] {
    v.to_le_bytes()
}

fn pad8(v: &mut Vec<u8>) {
    while !v.len().is_multiple_of(8) {
        v.push(0);
    }
}

/// Version 1 object header with the given `(type, flags, body)` messages.
fn ohdr_v1(msgs: &[(u16, u8, Vec<u8>)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (t, flags, data) in msgs {
        let mut d = data.clone();
        pad8(&mut d);
        body.extend(u16le(*t));
        body.extend(u16le(d.len() as u16));
        body.push(*flags);
        body.extend([0; 3]);
        body.extend(d);
    }
    let mut h = vec![1, 0];
    h.extend(u16le(msgs.len() as u16));
    h.extend(u32le(1));
    h.extend(u32le(body.len() as u32));
    h.extend([0; 4]);
    h.extend(body);
    h
}

/// Version 2 object header (no times, 1-byte chunk size).
fn ohdr_v2(msgs: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (t, data) in msgs {
        body.push(*t);
        body.extend(u16le(data.len() as u16));
        body.push(0);
        body.extend(data);
    }
    let mut h = b"OHDR".to_vec();
    h.push(2);
    h.push(0);
    h.push(body.len() as u8);
    h.extend(body);
    let sum = metadata_checksum(&h);
    h.extend(u32le(sum));
    h
}

fn symbol_table_msg(btree: u64, heap: u64) -> Vec<u8> {
    let mut v = u64le(btree).to_vec();
    v.extend(u64le(heap));
    v
}

fn entry(name_off: u64, header: u64, cache: u32, scratch: [u64; 2]) -> Vec<u8> {
    let mut v = u64le(name_off).to_vec();
    v.extend(u64le(header));
    v.extend(u32le(cache));
    v.extend(u32le(0));
    v.extend(u64le(scratch[0]));
    v.extend(u64le(scratch[1]));
    v
}

const I32: [u8; 12] = [0x10, 0x08, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];

/// Old-style file: root group with a sub-group `child` (with one integer
/// attribute) and a soft link `soft -> /child`.
fn v0_file() -> Vec<u8> {
    let mut b = B::default();
    // Superblock v0.
    let mut sb = hdf5rs::SIGNATURE.to_vec();
    sb.extend([0, 0, 0, 0, 0, 8, 8, 0]);
    sb.extend(u16le(4));
    sb.extend(u16le(16));
    sb.extend(u32le(0));
    sb.extend(u64le(0));
    sb.extend(u64le(UNDEF));
    sb.extend(u64le(700));
    sb.extend(u64le(UNDEF));
    sb.extend(entry(0, 96, 1, [300, 200]));
    b.put(0, &sb);
    // Root object header.
    b.put(
        96,
        &ohdr_v1(&[
            (0x11, 0, symbol_table_msg(300, 200)),
            (0x0d, 0, b"root group\0".to_vec()),
        ]),
    );
    // Local heap (shared by both groups) with its data segment at 240.
    let mut heap = b"HEAP".to_vec();
    heap.extend([0; 4]);
    heap.extend(u64le(40));
    heap.extend(u64le(UNDEF));
    heap.extend(u64le(240));
    b.put(200, &heap);
    let mut seg = vec![0u8; 8];
    seg.extend(b"child\0\0\0soft\0\0\0\0/child\0\0");
    seg.resize(40, 0);
    b.put(240, &seg);
    // Root B-tree: one SNOD.
    let mut bt = b"TREE".to_vec();
    bt.extend([0, 0]);
    bt.extend(u16le(1));
    bt.extend(u64le(UNDEF));
    bt.extend(u64le(UNDEF));
    bt.extend(u64le(0));
    bt.extend(u64le(400));
    bt.extend(u64le(16));
    b.put(300, &bt);
    // SNOD with the two entries.
    let mut snod = b"SNOD".to_vec();
    snod.extend([1, 0]);
    snod.extend(u16le(2));
    snod.extend(entry(8, 500, 1, [600, 200]));
    snod.extend(entry(16, UNDEF, 2, [24, 0]));
    b.put(400, &snod);
    // Child group header with an attribute "a" = 42i32 (attribute v1).
    let mut attr = vec![1, 0];
    attr.extend(u16le(2));
    attr.extend(u16le(12));
    attr.extend(u16le(8));
    attr.extend(b"a\0");
    pad8(&mut attr);
    attr.extend(I32);
    pad8(&mut attr);
    attr.extend([1, 0, 0, 0, 0, 0, 0, 0]); // scalar dataspace v1
    attr.extend(42i32.to_le_bytes());
    b.put(
        500,
        &ohdr_v1(&[(0x11, 0, symbol_table_msg(600, 200)), (0x0c, 0, attr)]),
    );
    // Empty B-tree for the child group.
    let mut bt = b"TREE".to_vec();
    bt.extend([0, 0]);
    bt.extend(u16le(0));
    bt.extend(u64le(UNDEF));
    bt.extend(u64le(UNDEF));
    bt.extend(u64le(0));
    b.put(600, &bt);
    b.data.resize(700, 0);
    b.data
}

fn link_msg(name: &str, target: &Link) -> Vec<u8> {
    let mut v = vec![1];
    let (ty, value) = match target {
        Link::Hard { address } => (None, u64le(*address).to_vec()),
        Link::Soft { path } => {
            let mut v = u16le(path.len() as u16).to_vec();
            v.extend(path.as_bytes());
            (Some(1u8), v)
        }
        Link::External { file, path } => {
            let mut val = vec![0];
            val.extend(file.as_bytes());
            val.push(0);
            val.extend(path.as_bytes());
            val.push(0);
            let mut v = u16le(val.len() as u16).to_vec();
            v.extend(val);
            (Some(64u8), v)
        }
        Link::UserDefined { .. } => unreachable!(),
    };
    match ty {
        Some(t) => v.extend([0x08, t]),
        None => v.push(0),
    }
    v.push(name.len() as u8);
    v.extend(name.as_bytes());
    v.extend(value);
    v
}

/// New-style file (superblock v2): root group with compact links to a
/// dataset `data`, soft links `alias -> /data` and `loop -> loop`, and an
/// external link.
fn v2_file() -> Vec<u8> {
    let mut b = B::default();
    let mut sb = hdf5rs::SIGNATURE.to_vec();
    sb.extend([2, 8, 8, 0]);
    sb.extend(u64le(0));
    sb.extend(u64le(UNDEF));
    sb.extend(u64le(400));
    sb.extend(u64le(48));
    let sum = metadata_checksum(&sb);
    sb.extend(u32le(sum));
    b.put(0, &sb);
    let mut link_info = vec![0, 0];
    link_info.extend(u64le(UNDEF));
    link_info.extend(u64le(UNDEF));
    b.put(
        48,
        &ohdr_v2(&[
            (0x02, link_info),
            (0x06, link_msg("data", &Link::Hard { address: 300 })),
            (
                0x06,
                link_msg(
                    "alias",
                    &Link::Soft {
                        path: "/data".into(),
                    },
                ),
            ),
            (
                0x06,
                link_msg(
                    "loop",
                    &Link::Soft {
                        path: "loop".into(),
                    },
                ),
            ),
            (
                0x06,
                link_msg(
                    "ext",
                    &Link::External {
                        file: "other.h5".into(),
                        path: "/x".into(),
                    },
                ),
            ),
        ]),
    );
    let mut space = vec![2, 1, 0, 1];
    space.extend(u64le(3));
    let mut layout = vec![3, 1];
    layout.extend(u64le(380));
    layout.extend(u64le(12));
    b.put(
        300,
        &ohdr_v2(&[(0x01, space), (0x03, I32.to_vec()), (0x08, layout)]),
    );
    b.data.resize(400, 0);
    b.data
}

#[test]
fn old_style_groups() {
    let f = File::from_bytes(v0_file()).unwrap();
    assert_eq!(f.superblock().version, 0);
    let root = f.root().unwrap();
    assert_eq!(root.path(), "/");
    assert_eq!(root.name(), "/");
    assert_eq!(root.link_storage().unwrap(), LinkStorage::SymbolTable);
    assert_eq!(root.link_names().unwrap(), ["child", "soft"]);
    assert_eq!(root.comment().unwrap().as_deref(), Some("root group"));
    assert_eq!(
        root.link("soft").unwrap(),
        Link::Soft {
            path: "/child".into()
        }
    );
    assert!(matches!(
        root.members_by_creation_order(),
        Err(Error::Unsupported(_))
    ));
    let child = f.group("/child").unwrap();
    assert_eq!(child.name(), "child");
    assert!(child.is_empty().unwrap());
    let a = child.attribute("a").unwrap();
    assert_eq!(a.raw_value(), &42i32.to_le_bytes());
    assert_eq!(a.shape(), &[] as &[u64]);
    assert_eq!(child.attribute_names().unwrap(), ["a"]);
    assert!(matches!(child.attribute("b"), Err(Error::NotFound(_))));
    let via_soft = f.group("soft").unwrap();
    assert_eq!(via_soft.path(), "/soft");
    assert_eq!(via_soft.address(), child.address());
    assert!(matches!(f.object("/missing"), Err(Error::NotFound(_))));
    assert!(matches!(
        f.dataset("/child"),
        Err(Error::WrongObjectType(_))
    ));
    assert_eq!(child.object("/child").unwrap().kind(), ObjectKind::Group);
    let objs = root.objects().unwrap();
    assert_eq!(objs.len(), 2);
}

#[test]
fn new_style_compact_links() {
    let f = File::from_bytes(v2_file()).unwrap();
    let root = f.root().unwrap();
    assert_eq!(root.link_storage().unwrap(), LinkStorage::Compact);
    assert_eq!(root.link_names().unwrap(), ["alias", "data", "ext", "loop"]);
    let d = f.dataset("/data").unwrap();
    assert_eq!(d.shape(), &[3]);
    assert!(d.is_contiguous());
    assert!(d.datatype().is_integer());
    assert_eq!(d.chunk_shape(), None);
    let alias = f.dataset("alias").unwrap();
    assert_eq!(alias.path(), "/alias");
    assert_eq!(alias.address(), d.address());
    assert!(matches!(f.object("loop"), Err(Error::LimitExceeded(_))));
    assert!(matches!(f.object("ext"), Err(Error::Unsupported(_))));
    assert!(matches!(f.object("data/x"), Err(Error::NotFound(_))));
    assert!(matches!(f.group("data"), Err(Error::WrongObjectType(_))));
    match f.object("/data").unwrap() {
        Object::Dataset(d) => assert!(d.attributes().unwrap().is_empty()),
        o => panic!("{:?}", o.kind()),
    }
    assert!(!root.contains("nope").unwrap());
    assert!(root.contains("ext").unwrap());
}

#[test]
fn corrupted_files_do_not_panic() {
    for base in [v0_file(), v2_file()] {
        for i in 0..base.len() {
            for x in [0x01u8, 0x80, 0xff] {
                let mut data = base.clone();
                data[i] ^= x;
                let Ok(f) = File::from_bytes(data) else {
                    continue;
                };
                let Ok(root) = f.root() else { continue };
                let _ = root.attributes();
                if let Ok(objs) = root.objects() {
                    for o in objs {
                        let _ = o.attributes();
                    }
                }
                for p in ["child", "soft", "data", "alias", "loop", "ext"] {
                    let _ = f.object(p);
                }
                if let Ok(d) = f.dataset("data") {
                    let _ = d.read_values();
                }
            }
        }
    }
}

/// Superblock v2 file whose root links `ext` to a 3-element i32 dataset
/// stored in the external raw data file `raw_name` (bytes 4..16).
fn external_data_file(raw_name: &str) -> Vec<u8> {
    let mut b = B::default();
    let mut sb = hdf5rs::SIGNATURE.to_vec();
    sb.extend([2, 8, 8, 0]);
    sb.extend(u64le(0));
    sb.extend(u64le(UNDEF));
    sb.extend(u64le(700));
    sb.extend(u64le(48));
    let sum = metadata_checksum(&sb);
    sb.extend(u32le(sum));
    b.put(0, &sb);
    let mut link_info = vec![0, 0];
    link_info.extend(u64le(UNDEF));
    link_info.extend(u64le(UNDEF));
    b.put(
        48,
        &ohdr_v2(&[
            (0x02, link_info),
            (0x06, link_msg("ext", &Link::Hard { address: 300 })),
        ]),
    );
    let mut space = vec![2, 1, 0, 1];
    space.extend(u64le(3));
    let mut layout = vec![3, 1];
    layout.extend(u64le(UNDEF));
    layout.extend(u64le(12));
    let mut efl = vec![1, 0, 0, 0];
    efl.extend(u16le(1));
    efl.extend(u16le(1));
    efl.extend(u64le(500));
    for v in [8u64, 4, 12] {
        efl.extend(u64le(v));
    }
    b.put(
        300,
        &ohdr_v2(&[
            (0x01, space),
            (0x03, I32.to_vec()),
            (0x08, layout),
            (0x07, efl),
        ]),
    );
    // Local heap with the file name at offset 8.
    let mut seg = vec![0u8; 8];
    seg.extend(raw_name.as_bytes());
    seg.push(0);
    pad8(&mut seg);
    let mut heap = b"HEAP".to_vec();
    heap.extend([0, 0, 0, 0]);
    heap.extend(u64le(seg.len() as u64));
    heap.extend(u64le(UNDEF));
    heap.extend(u64le(540));
    b.put(500, &heap);
    b.put(540, &seg);
    b.data.resize(700, 0);
    b.data
}

#[test]
fn external_raw_data_files() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("hdf5rs_external_{}", std::process::id()));
    let sub = dir.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    let mut raw = vec![0xee; 4];
    for v in [7i32, -8, 9] {
        raw.extend(v.to_le_bytes());
    }
    std::fs::write(dir.join("raw.bin"), &raw).unwrap();
    std::fs::write(dir.join("plain.h5"), external_data_file("raw.bin")).unwrap();
    std::fs::write(sub.join("up.h5"), external_data_file("../raw.bin")).unwrap();

    let f = File::open(dir.join("plain.h5")).unwrap();
    let d = f.dataset("ext").unwrap();
    assert_eq!(d.external_files().unwrap()[0].name, "raw.bin");
    assert_eq!(d.read::<i32>().unwrap(), [7, -8, 9]);
    assert_eq!(
        d.read_selection::<i64>(&hdf5rs::Selection::range(1..3))
            .unwrap(),
        [-8, 9]
    );

    // Custom resolver.
    let resolver = |name: &str| -> hdf5rs::Result<Box<dyn hdf5rs::ReadAt>> {
        assert_eq!(name, "raw.bin");
        let mut v = vec![0u8; 4];
        v.extend([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        Ok(Box::new(v))
    };
    assert_eq!(
        d.reader()
            .external_resolver(&resolver)
            .read::<i32>()
            .unwrap(),
        [1, 2, 3]
    );

    // From memory: no directory to resolve against.
    let mem = File::from_bytes(std::fs::read(dir.join("plain.h5")).unwrap()).unwrap();
    assert!(matches!(
        mem.dataset("ext").unwrap().read::<i32>(),
        Err(Error::Unsupported(_))
    ));
    // Disabled.
    let locked = File::open_with(
        dir.join("plain.h5"),
        hdf5rs::FileAccess::default().external_files(false),
    )
    .unwrap();
    assert!(matches!(
        locked.dataset("ext").unwrap().read::<i32>(),
        Err(Error::Unsupported(_))
    ));
    // `..` components: honoured by default, rejected when confined.
    let up = File::open(sub.join("up.h5")).unwrap();
    assert_eq!(
        up.dataset("ext").unwrap().read::<i32>().unwrap(),
        [7, -8, 9]
    );
    let confined = File::open_with(
        sub.join("up.h5"),
        hdf5rs::FileAccess::default().confine_to_directory(true),
    )
    .unwrap();
    assert!(matches!(
        confined.dataset("ext").unwrap().read::<i32>(),
        Err(Error::InvalidFormat(_))
    ));
    let _ = std::fs::remove_dir_all(&dir);
}
