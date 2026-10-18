#![no_main]

//! Open arbitrary bytes as an HDF5 file and walk everything: groups, links,
//! attributes and datasets, bounded by `WalkLimits` (see
//! `tests/support/walk.rs`).

use libfuzzer_sys::fuzz_target;

#[path = "../../tests/support/walk.rs"]
mod walk;

fuzz_target!(|data: &[u8]| {
    walk::walk_bytes(data.to_vec(), walk::WalkLimits::default());
});
