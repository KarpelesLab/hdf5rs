//! Old-style (symbol table) groups: walking a group's v1 B-tree and symbol
//! table nodes, resolving names through its local heap.

use super::btree_v1::{MAX_ENTRIES, group_btree_children};
use super::local_heap::LocalHeap;
use super::symbol_table_node::SymbolTableNode;
use super::{Sizes, SymbolTableEntry};
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// List the members of an old-style group, given the addresses from its
/// Symbol Table message (or a cached [`super::ScratchPad::Group`]).
///
/// Names are returned as raw bytes (without the NUL terminator), in stored
/// order (which, for well-formed files, is sorted by name).
pub fn read_group_v1_raw(
    store: &(impl ReadAt + ?Sized),
    btree_address: u64,
    heap_address: u64,
    sizes: Sizes,
) -> Result<Vec<(Vec<u8>, SymbolTableEntry)>> {
    let heap = LocalHeap::read(store, heap_address, sizes)?;
    let mut out = Vec::new();
    for snod in group_btree_children(store, btree_address, sizes)? {
        let node = SymbolTableNode::read(store, snod, sizes)?;
        if out.len().saturating_add(node.entries.len()) > MAX_ENTRIES {
            return Err(Error::LimitExceeded("group member count"));
        }
        for entry in node.entries {
            let name = heap.get_bytes(entry.link_name_offset)?.to_vec();
            out.push((name, entry));
        }
    }
    Ok(out)
}

/// Like [`read_group_v1_raw`], but with names decoded as UTF-8; invalid
/// sequences are replaced with U+FFFD.
pub fn read_group_v1(
    store: &(impl ReadAt + ?Sized),
    btree_address: u64,
    heap_address: u64,
    sizes: Sizes,
) -> Result<Vec<(String, SymbolTableEntry)>> {
    Ok(
        read_group_v1_raw(store, btree_address, heap_address, sizes)?
            .into_iter()
            .map(|(name, e)| {
                let name = match String::from_utf8(name) {
                    Ok(s) => s,
                    Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
                };
                (name, e)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::{FileBuilder, Rng};

    /// Group with 5 members spread over 3 SNODs under a 2-level B-tree.
    fn group_file(sizes: Sizes) -> FileBuilder {
        let mut f = FileBuilder::new(sizes);
        let names = ["alpha", "beta", "delta", "gamma", "zeta"];
        let mut seg = vec![0u8; 8];
        let mut offs = Vec::new();
        for n in names {
            offs.push(seg.len() as u64);
            seg.extend(n.as_bytes());
            seg.push(0);
            seg.resize(seg.len().div_ceil(8) * 8, 0);
        }
        f.local_heap(0, 64, &seg, u64::MAX);
        let e: Vec<_> = (0..5).map(|i| f.entry(offs[i], 5000 + i as u64)).collect();
        f.snod(200, &e[0..2]);
        f.snod(400, &e[2..4]);
        f.snod(600, &e[4..5]);
        let k = |o: u64| FileBuilder::uint(o, sizes.length);
        f.btree(800, 0, 0, &[k(0), k(offs[1]), k(offs[3])], &[200, 400]);
        f.btree(900, 0, 0, &[k(offs[3]), k(offs[4])], &[600]);
        f.btree(1000, 0, 1, &[k(0), k(offs[3]), k(offs[4])], &[800, 900]);
        f
    }

    #[test]
    fn lists_members() {
        for sizes in [
            Sizes::DEFAULT,
            Sizes {
                offset: 4,
                length: 4,
            },
        ] {
            let f = group_file(sizes);
            let m = read_group_v1(&f.data, 1000, 0, sizes).unwrap();
            let names: Vec<_> = m.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, ["alpha", "beta", "delta", "gamma", "zeta"]);
            assert_eq!(m[3].1.object_header_address, 5003);
        }
    }

    #[test]
    fn bad_name_offset() {
        let mut f = group_file(Sizes::DEFAULT);
        let e = f.entry(10_000, 1);
        f.snod(600, &[e]);
        assert!(read_group_v1(&f.data, 1000, 0, Sizes::DEFAULT).is_err());
    }

    #[test]
    fn mutations_never_panic() {
        let f = group_file(Sizes::DEFAULT);
        let mut rng = Rng(99);
        for _ in 0..5000 {
            let d = rng.mutate(&f.data);
            let _ = read_group_v1(&d, 1000, 0, Sizes::DEFAULT);
        }
    }
}
