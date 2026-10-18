//! List the contents of an HDF5 file as a tree, with datatypes, shapes,
//! layouts and attributes.
//!
//! ```text
//! cargo run --example h5ls -- [-v] [-d] [--follow-external] FILE [PATH]
//! ```
//!
//! `-v` prints attribute values, `-d` prints (the start of) dataset values.

use std::collections::HashSet;
use std::process::ExitCode;

use hdf5rs::format::messages::layout::DataLayout;
use hdf5rs::{Attribute, Dataset, File, FileAccess, Group, Link, Object, Value};

/// Maximum number of elements printed per value.
const MAX_SHOWN: usize = 8;

struct Options {
    values: bool,
    data: bool,
}

fn shape(dims: &[u64], kind: hdf5rs::DataspaceKind) -> String {
    match kind {
        hdf5rs::DataspaceKind::Null => "null".into(),
        hdf5rs::DataspaceKind::Scalar => "scalar".into(),
        hdf5rs::DataspaceKind::Simple => format!(
            "[{}]",
            dims.iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn preview(values: &[Value]) -> String {
    let mut s = values
        .iter()
        .take(MAX_SHOWN)
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if values.len() > MAX_SHOWN {
        s.push_str(&format!(", … ({} total)", values.len()));
    }
    s
}

fn print_attributes(attrs: hdf5rs::Result<Vec<Attribute>>, indent: &str, opts: &Options) {
    let attrs = match attrs {
        Ok(a) => a,
        Err(e) => {
            println!("{indent}  @<error: {e}>");
            return;
        }
    };
    for a in attrs {
        let mut line = format!(
            "{indent}  @{}: {} {}",
            a.name(),
            a.datatype(),
            shape(a.shape(), a.dataspace().kind)
        );
        if opts.values {
            match a.values() {
                Ok(v) => line.push_str(&format!(" = {}", preview(&v))),
                Err(e) => line.push_str(&format!(" = <{e}>")),
            }
        }
        println!("{line}");
    }
}

fn describe_dataset(d: &Dataset) -> String {
    let layout = match d.layout() {
        DataLayout::Compact { .. } => "compact".to_string(),
        DataLayout::Contiguous { .. } if d.external_files().is_some() => "external".to_string(),
        DataLayout::Contiguous { .. } => "contiguous".to_string(),
        DataLayout::Chunked(c) => format!("chunked {:?}", c.dims),
        DataLayout::Virtual { .. } => "virtual".to_string(),
    };
    let filters = d
        .filters()
        .map(|p| {
            let names: Vec<String> = p
                .filters
                .iter()
                .map(|f| match (&f.name, f.id) {
                    (Some(n), _) => n.clone(),
                    (None, 1) => "deflate".into(),
                    (None, 2) => "shuffle".into(),
                    (None, 3) => "fletcher32".into(),
                    (None, 4) => "szip".into(),
                    (None, 5) => "nbit".into(),
                    (None, 6) => "scaleoffset".into(),
                    (None, 32000) => "lzf".into(),
                    (None, id) => format!("#{id}"),
                })
                .collect();
            format!(" filters=[{}]", names.join(", "))
        })
        .unwrap_or_default();
    format!(
        "dataset {} {} {layout}{filters}",
        d.datatype(),
        shape(d.shape(), d.dataspace().kind)
    )
}

fn walk(group: &Group, indent: &str, seen: &mut HashSet<u64>, opts: &Options) {
    let members = match group.members() {
        Ok(m) => m,
        Err(e) => {
            println!("{indent}<error listing {}: {e}>", group.path());
            return;
        }
    };
    for entry in members {
        let name = &entry.name;
        match &entry.link {
            Link::Soft { path } => {
                println!("{indent}{name} -> {path}");
                continue;
            }
            Link::External { file, path } => {
                print!("{indent}{name} -> {file}:{path}");
                if group.file().access().follow_external_links {
                    match group.follow_external(name) {
                        // Objects of another file: addresses are not comparable.
                        Ok(o) => show(&o, indent, &mut HashSet::new(), opts),
                        Err(e) => println!(" <error: {e}>"),
                    }
                } else {
                    println!();
                }
                continue;
            }
            Link::UserDefined { link_type, .. } => {
                println!("{indent}{name} (user-defined link type {link_type})");
                continue;
            }
            Link::Hard { .. } => {}
        }
        match group.object(name) {
            Ok(o) => {
                print!("{indent}{name}");
                show(&o, indent, seen, opts);
            }
            Err(e) => println!("{indent}{name} <error: {e}>"),
        }
    }
}

fn show(o: &Object, indent: &str, seen: &mut HashSet<u64>, opts: &Options) {
    let first = seen.insert(o.address());
    match o {
        Object::Group(g) => {
            println!("/ (group)");
            if !first {
                println!("{indent}  (already listed)");
                return;
            }
            print_attributes(g.attributes(), indent, opts);
            walk(g, &format!("{indent}  "), seen, opts);
        }
        Object::Dataset(d) => {
            println!(": {}", describe_dataset(d));
            if first {
                print_attributes(d.attributes(), indent, opts);
                if opts.data {
                    match d.read_values() {
                        Ok(v) => println!("{indent}  = {}", preview(&v)),
                        Err(e) => println!("{indent}  = <{e}>"),
                    }
                }
            }
        }
        Object::NamedDatatype(t) => {
            println!(": datatype {}", t.datatype());
            if first {
                print_attributes(t.attributes(), indent, opts);
            }
        }
    }
}

fn main() -> ExitCode {
    let mut opts = Options {
        values: false,
        data: false,
    };
    let mut access = FileAccess::default();
    let mut positional = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-v" => opts.values = true,
            "-d" => opts.data = true,
            "--follow-external" => access = access.follow_external_links(true),
            _ => positional.push(arg),
        }
    }
    let Some(path) = positional.first() else {
        eprintln!("usage: h5ls [-v] [-d] [--follow-external] FILE [PATH]");
        return ExitCode::from(2);
    };
    let file = match File::open_with(path, access) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let start = positional.get(1).map(String::as_str).unwrap_or("/");
    match file.object(start) {
        Ok(o) => {
            print!("{}", start.trim_end_matches('/'));
            let mut seen = HashSet::new();
            show(&o, "", &mut seen, &opts);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{start}: {e}");
            ExitCode::FAILURE
        }
    }
}
