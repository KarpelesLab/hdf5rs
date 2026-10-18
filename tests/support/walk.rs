//! Walk everything reachable in an (untrusted) HDF5 image: groups, links,
//! attributes, datasets (metadata, chunk lists, raw reads, decoded values).
//!
//! Shared by the stress test (`tests/stress.rs`) and the cargo-fuzz targets
//! (`fuzz/`). Every error is ignored; the point is that nothing panics,
//! hangs or allocates without bound.

#![allow(dead_code)]

use std::collections::HashSet;

use hdf5rs::read::ReadLimits;
use hdf5rs::{Dataset, File, FileAccess, Group, HyperslabDim, Link, Object, Selection};

/// Bounds of one walk.
#[derive(Debug, Clone, Copy)]
pub struct WalkLimits {
    /// Most objects opened.
    pub max_objects: usize,
    /// Deepest group nesting followed.
    pub max_depth: usize,
    /// Most links examined per group.
    pub max_links: usize,
    /// Read limits for dataset reads.
    pub read: ReadLimits,
    /// Datasets/attributes with more elements are not decoded into
    /// `Value`s (raw reads still happen).
    pub max_value_elements: u64,
}

impl Default for WalkLimits {
    fn default() -> Self {
        WalkLimits {
            max_objects: 512,
            max_depth: 16,
            max_links: 512,
            read: ReadLimits::with_max_output(8 << 20).max_chunk_bytes(8 << 20),
            max_value_elements: 1 << 14,
        }
    }
}

/// What a walk touched (useful to check coverage on valid files).
#[derive(Debug, Default, Clone, Copy)]
pub struct WalkStats {
    pub objects: usize,
    pub attributes: usize,
    pub datasets_read: usize,
    pub bytes_read: usize,
}

struct Walker {
    limits: WalkLimits,
    visited: HashSet<u64>,
    stats: WalkStats,
}

/// Open `data` as an HDF5 file and walk it.
pub fn walk_bytes(data: Vec<u8>, limits: WalkLimits) -> WalkStats {
    let access = FileAccess::default()
        .read_limits(limits.read)
        .chunk_cache_bytes(4 << 20);
    match File::from_bytes_with(data, access) {
        Ok(f) => walk_file(&f, limits),
        Err(_) => WalkStats::default(),
    }
}

/// Walk an opened file.
pub fn walk_file(file: &File, limits: WalkLimits) -> WalkStats {
    let mut w = Walker {
        limits,
        visited: HashSet::new(),
        stats: WalkStats::default(),
    };
    let _ = file.superblock_extension();
    let _ = file.shared_message_table();
    if let Ok(root) = file.root() {
        w.visited.insert(root.address());
        w.group(&root, 0);
    }
    w.stats
}

impl Walker {
    fn budget_left(&self) -> bool {
        self.stats.objects < self.limits.max_objects
    }

    fn object(&mut self, obj: &Object, depth: usize) {
        self.stats.objects += 1;
        let _ = obj.modification_time();
        let _ = obj.comment();
        let _ = obj.times();
        let _ = obj.attribute_names();
        if let Ok(attrs) = obj.attributes() {
            for a in attrs.iter().take(256) {
                self.stats.attributes += 1;
                let _ = a.datatype().to_string();
                let _ = obj.attribute(a.name());
                if a.element_count()
                    .is_ok_and(|n| n <= self.limits.max_value_elements)
                {
                    if let Ok(v) = a.value() {
                        let _ = v.to_string();
                    }
                    let _ = a.read::<f64>();
                    let _ = a.read::<String>();
                }
            }
        }
        if let Some(g) = obj.as_group() {
            let _ = g.attributes_by_creation_order();
            if depth < self.limits.max_depth {
                self.group(g, depth + 1);
            }
        } else if let Some(d) = obj.as_dataset() {
            self.dataset(d);
        } else if let Some(t) = obj.as_named_datatype() {
            let _ = t.datatype().to_string();
        }
    }

    fn group(&mut self, g: &Group, depth: usize) {
        let _ = g.link_storage();
        let _ = g.creation_order_tracked();
        let _ = g.len();
        let _ = g.members_by_creation_order();
        let Ok(members) = g.members() else { return };
        for e in members.iter().take(self.limits.max_links) {
            if !self.budget_left() {
                return;
            }
            let _ = g.contains(&e.name);
            let _ = g.link(&e.name);
            match &e.link {
                Link::Hard { address } => {
                    if !self.visited.insert(*address) {
                        continue;
                    }
                    if let Ok(o) = g.file().object_at(*address, &e.name) {
                        self.object(&o, depth);
                    }
                }
                Link::Soft { .. } => {
                    // Resolve (exercises hop limits) but do not descend.
                    if let Ok(o) = g.object(&e.name) {
                        self.stats.objects += 1;
                        let _ = o.attribute_names();
                    }
                }
                Link::External { .. } => {
                    let _ = g.follow_external(&e.name);
                }
                Link::UserDefined { .. } => {}
            }
        }
    }

    fn dataset(&mut self, d: &Dataset) {
        let _ = d.datatype().to_string();
        let _ = d.max_shape();
        let _ = d.fill_value_bytes();
        let _ = d.external_files();
        let _ = d.virtual_mappings();
        let reader = d.reader().limits(self.limits.read);
        // A corner first (individual chunk index lookups), then everything
        // (whole index), then a strided selection (chunk cache).
        let shape = d.shape();
        let corner: Vec<_> = shape.iter().map(|&n| n.saturating_sub(2)..n).collect();
        let corner = Selection::from_ranges(&corner);
        if !shape.is_empty() {
            let _ = reader.read_raw_selection(&corner);
        }
        let _ = reader.chunks();
        if let Ok(raw) = reader.read_raw() {
            self.stats.datasets_read += 1;
            self.stats.bytes_read += raw.len();
        }
        if !shape.is_empty() {
            let strided: Vec<HyperslabDim> = shape
                .iter()
                .map(|&n| HyperslabDim::new(n / 3, 3, (n - n / 3) / 3, 1))
                .collect();
            let _ = reader.read_raw_selection(&Selection::hyperslab(strided));
            let _ = reader.use_chunk_cache(false).read_raw_selection(&corner);
        }
        if d.element_count()
            .is_ok_and(|n| n <= self.limits.max_value_elements)
        {
            if let Ok(v) = reader.read_values() {
                for x in v.iter().take(16) {
                    let _ = x.to_string();
                }
            }
            let _ = reader.read::<f64>();
            let _ = reader.read::<i64>();
            let _ = reader.read::<String>();
        }
    }
}
