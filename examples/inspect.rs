//! Dump the object headers of an HDF5 file (debugging aid).
//!
//! `cargo run --example inspect -- file.h5`

use hdf5rs::format::messages::MessageType;
use hdf5rs::{File, Link, Object};

fn hex(b: &[u8]) -> String {
    b.iter()
        .map(|x| format!("{x:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn dump(file: &File, obj: &Object, depth: usize, seen: &mut Vec<u64>) -> hdf5rs::Result<()> {
    let pad = "  ".repeat(depth);
    let h = obj.header();
    println!(
        "{pad}{} @{} kind={:?} v{} flags={:#04x} chunks={:?}",
        obj.path(),
        h.address,
        obj.kind(),
        h.version,
        h.flags,
        h.chunks
    );
    for m in &h.messages {
        let n = m.data.len().min(96);
        println!(
            "{pad}  [{:#06x} {}] flags={:#04x} len={} {}",
            m.msg_type,
            MessageType::name(m.msg_type),
            m.flags,
            m.data.len(),
            hex(&m.data[..n])
        );
    }
    if seen.contains(&h.address) {
        return Ok(());
    }
    seen.push(h.address);
    if let Object::Group(g) = obj {
        for e in g.members()? {
            match e.link {
                Link::Hard { address } => {
                    let path = if obj.path() == "/" {
                        format!("/{}", e.name)
                    } else {
                        format!("{}/{}", obj.path(), e.name)
                    };
                    let child = file.object_at(address, &path)?;
                    dump(file, &child, depth + 1, seen)?;
                }
                other => println!("{pad}  {} -> {:?}", e.name, other),
            }
        }
    }
    Ok(())
}

fn main() -> hdf5rs::Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: inspect FILE");
        std::process::exit(2);
    };
    let file = File::open(&path)?;
    println!("superblock: {:?}", file.superblock());
    let root = file.object("/")?;
    dump(&file, &root, 0, &mut Vec::new())
}
