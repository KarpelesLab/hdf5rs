#![no_main]

//! Fuzz the `object_header` parser harness (see `tests/support/parsers.rs`).

use libfuzzer_sys::fuzz_target;

#[path = "../../tests/support/parsers.rs"]
mod parsers;

fuzz_target!(|data: &[u8]| parsers::object_header(data));
