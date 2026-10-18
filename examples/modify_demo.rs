//! Modify an existing HDF5 file in place (read-write mode).
//!
//! `cargo run --example modify_demo -- <touch|scenario|read> file.h5`
//!
//! - `touch`: generic changes to any file: an attribute `rw_a` on every
//!   object, a soft link `rw_soft` in every group, every dataset rewritten
//!   with its own values (rebuilding chunk indexes), a new group
//!   `/rw_group` with a dataset `d` = [7, 8, 9] and a root attribute
//!   `rw_attr` = 42.
//! - `scenario`: the changes `tools/check_modified.py` expects on the base
//!   file it creates with h5py.
//! - `read`: read every object, attribute and dataset (fails on error).
//!
//! `tools/check_modified.py` drives this with files created by h5py and
//! verifies the results with h5py.

use std::collections::HashSet;

use hdf5rs::write::FileWriter;
use hdf5rs::{Error, File, Group, Link, Object, Selection};

/// Every object reachable through hard links, by first path.
fn objects(f: &File) -> hdf5rs::Result<Vec<(String, Object)>> {
    fn rec(
        g: &Group,
        path: &str,
        seen: &mut HashSet<u64>,
        out: &mut Vec<(String, Object)>,
    ) -> hdf5rs::Result<()> {
        for m in g.members()? {
            if let Link::Hard { address } = m.link {
                if !seen.insert(address) {
                    continue;
                }
                let child = if path == "/" {
                    format!("/{}", m.name)
                } else {
                    format!("{path}/{}", m.name)
                };
                let obj = g.file().object_at(address, &child)?;
                out.push((child.clone(), obj.clone()));
                if let Object::Group(sub) = &obj {
                    rec(sub, &child, seen, out)?;
                }
            }
        }
        Ok(())
    }
    let root = f.root()?;
    let mut seen = HashSet::from([root.address()]);
    let mut out = vec![("/".to_owned(), Object::Group(root.clone()))];
    rec(&root, "/", &mut seen, &mut out)?;
    Ok(out)
}

fn touch(path: &str) -> hdf5rs::Result<()> {
    let objs = objects(&File::open(path)?)?;
    let mut w = FileWriter::open_rw(path)?;
    {
        let mut root = w.root();
        root.set_attr("rw_attr", &42i32)?;
        let mut g = root.create_group("rw_group")?;
        g.new_dataset::<i32>("d").write(&[7, 8, 9])?;
    }
    let mut rewritten = 0;
    for (p, obj) in &objs {
        w.set_attr(p, "rw_a", &(p.len() as u64))?;
        match obj {
            Object::Group(_) => w.group(p)?.link_soft("rw_soft", "/rw_group/d")?,
            Object::Dataset(d) => {
                let lossy = d
                    .filters()
                    .is_some_and(|f| f.filters.iter().any(|x| x.id == 6));
                if lossy || d.is_virtual() || d.external_files().is_some() {
                    continue;
                }
                let Ok(raw) = d.read_raw() else { continue };
                match w.dataset(p)?.write_selection_raw(&Selection::All, &raw) {
                    Ok(()) => rewritten += 1,
                    Err(Error::Unsupported(_)) => {}
                    Err(e) => return Err(e),
                }
            }
            Object::NamedDatatype(_) => {}
        }
    }
    w.finish()?;
    println!(
        "{path}: touched {} objects, rewrote {rewritten} datasets",
        objs.len()
    );
    Ok(())
}

fn scenario(path: &str) -> hdf5rs::Result<()> {
    let mut w = FileWriter::open_rw(path)?;
    w.root().set_attr("modified_by", "hdf5rs")?;
    {
        let mut g = w.group("/grp")?;
        for i in 0..10 {
            g.create_group(&format!("rw_g{i}"))?;
        }
    }
    for i in 0..15 {
        w.unlink(&format!("/dense_grp/x{i:02}"))?;
    }
    {
        let mut g = w.group("/attrs")?;
        g.set_attr("a0", &100i64)?;
        g.delete_attr("a1")?;
        for i in 0..15 {
            g.set_attr(&format!("rw_b{i:02}"), &(i as i64))?;
        }
    }
    for i in 0..18 {
        w.delete_attr("/dense_attrs", &format!("c{i:02}"))?;
    }
    w.dataset("/data/fixed")?
        .write_selection(&Selection::from_ranges(&[1..3, 2..4]), &[99i32; 4])?;
    let more: Vec<f64> = (0..10).map(|i| 1000.0 + f64::from(i)).collect();
    w.dataset("/data/ea")?.append(&more)?;
    {
        let mut d = w.dataset("/data/bt2")?;
        d.resize(&[8, 9])?;
        d.write_selection(&Selection::from_ranges(&[6..8, 0..9]), &[-1i32; 18])?;
    }
    w.dataset("/data/gz")?
        .write_selection(&Selection::range(5..25), &[-5i32; 20])?;
    w.dataset("/data/contig")?
        .write_selection(&Selection::range(3..4), &[3.5f32])?;
    w.dataset("/data/compact")?
        .write_selection(&Selection::range(0..2), &[7i16, 8])?;
    w.dataset("/data/single")?
        .write_selection(&Selection::from_ranges(&[0..1, 0..1]), &[42i32])?;
    w.unlink("/to_delete")?;
    w.unlink("/hard_b")?;
    {
        let mut root = w.root();
        let mut g = root.create_group("new")?;
        let mut d = g.new_dataset::<u8>("ds").write(&[1, 2, 3])?;
        d.set_attr("by", "hdf5rs")?;
    }
    w.finish()?;
    println!("{path}: scenario applied");
    Ok(())
}

fn read_all(path: &str) -> hdf5rs::Result<()> {
    let f = File::open(path)?;
    let objs = objects(&f)?;
    for (p, o) in &objs {
        for a in o.attributes()? {
            a.values()
                .map_err(|e| Error::invalid(format!("{p}@{}: {e}", a.name())))?;
        }
        if let Object::Dataset(d) = o {
            match d.read_values() {
                Ok(_) | Err(Error::Unsupported(_)) => {}
                Err(e) => return Err(Error::invalid(format!("{p}: {e}"))),
            }
        }
    }
    println!("{path}: read {} objects", objs.len());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(mode), Some(path)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: modify_demo <touch|scenario|read> file.h5");
        std::process::exit(2);
    };
    let r = match mode.as_str() {
        "touch" => touch(path),
        "scenario" => scenario(path),
        "read" => read_all(path),
        _ => {
            eprintln!("unknown mode {mode}");
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    }
}
