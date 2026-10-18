//! Zlib / deflate codec (RFC 1950 / RFC 1951).
//!
//! **PLACEHOLDER.** This file is a stub so the filter framework compiles;
//! the real implementation is developed separately and replaces it.
//! [`zlib_decompress`] always fails with [`Error::Unsupported`], and
//! [`zlib_compress`] emits uncompressed ("stored") deflate blocks only.

use crate::{Error, Result};

/// Decompresses a zlib stream.
///
/// `size_hint` is a pre-allocation hint; output larger than `max_output`
/// must fail with [`Error::LimitExceeded`].
///
/// **Stub:** always returns [`Error::Unsupported`].
pub fn zlib_decompress(input: &[u8], size_hint: usize, max_output: usize) -> Result<Vec<u8>> {
    let _ = (input, size_hint, max_output);
    Err(Error::unsupported("deflate"))
}

/// Compresses `input` into a zlib stream at `level` (0–9).
///
/// **Stub:** ignores `level` and writes stored blocks (valid zlib, no
/// compression).
pub fn zlib_compress(input: &[u8], level: u32) -> Vec<u8> {
    let _ = level;
    let mut out = Vec::with_capacity(input.len() + input.len() / 65535 * 5 + 11);
    out.extend_from_slice(&[0x78, 0x01]);
    let mut blocks = input.chunks(65535).peekable();
    if blocks.peek().is_none() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
    }
    while let Some(block) = blocks.next() {
        let last = u8::from(blocks.peek().is_none());
        let len = block.len() as u16;
        out.push(last);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in input {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}
