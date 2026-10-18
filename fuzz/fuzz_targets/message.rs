#![no_main]

//! Fuzz the `message` parser harness (see `tests/support/parsers.rs`).

use libfuzzer_sys::fuzz_target;

#[path = "../../tests/support/parsers.rs"]
mod parsers;

fuzz_target!(|data: &[u8]| parsers::message(data));
