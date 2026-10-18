//! The Zstandard filter (registered filter id 32015, feature `zstd`).
//!
//! Each chunk is a single Zstandard frame (RFC 8878). Client data:
//! `[level]` for encoding (default 3); ignored when decoding.

use super::codec::{zstd_compress, zstd_decompress};
use super::{Filter, FilterContext, ZSTD};
use crate::Result;

/// The Zstandard filter, id [`ZSTD`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Zstd;

impl Filter for Zstd {
    fn id(&self) -> u16 {
        ZSTD
    }

    fn name(&self) -> &str {
        "zstd"
    }

    fn decode(&self, _client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        zstd_decompress(&input, ctx.max_output)
    }

    fn encode(&self, client_data: &[u32], input: Vec<u8>, _ctx: &FilterContext) -> Result<Vec<u8>> {
        let level = client_data.first().map_or(3, |&l| l as i32);
        zstd_compress(&input, level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let data: Vec<u8> = (0..50_000u32).flat_map(|i| (i / 9).to_le_bytes()).collect();
        let ctx = FilterContext::new(4);
        let c = Zstd.encode(&[3], data.clone(), &ctx).unwrap();
        assert!(c.len() < data.len() / 10);
        assert_eq!(Zstd.decode(&[3], c, &ctx).unwrap(), data);
        assert!(Zstd.decode(&[], vec![1, 2, 3], &ctx).is_err());
    }
}
