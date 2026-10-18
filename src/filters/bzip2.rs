//! The BZIP2 filter (registered filter id 307, feature `bzip2`).
//!
//! Each chunk is a single bzip2 stream. Client data: `[block size]`
//! (1–9, in units of 100 kB) for encoding (default 9); ignored when
//! decoding.

use super::codec::{bzip2_compress, bzip2_decompress};
use super::{BZIP2, Filter, FilterContext};
use crate::Result;

/// The BZIP2 filter, id [`BZIP2`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Bzip2;

impl Filter for Bzip2 {
    fn id(&self) -> u16 {
        BZIP2
    }

    fn name(&self) -> &str {
        "bzip2"
    }

    fn decode(&self, _client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        bzip2_decompress(&input, ctx.max_output)
    }

    fn encode(&self, client_data: &[u32], input: Vec<u8>, _ctx: &FilterContext) -> Result<Vec<u8>> {
        bzip2_compress(&input, client_data.first().copied().unwrap_or(9))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let data: Vec<u8> = (0..20_000u32).flat_map(|i| (i / 9).to_le_bytes()).collect();
        let ctx = FilterContext::new(4);
        let c = Bzip2.encode(&[9], data.clone(), &ctx).unwrap();
        assert!(c.starts_with(b"BZh9"));
        assert!(c.len() < data.len() / 10);
        assert_eq!(Bzip2.decode(&[9], c, &ctx).unwrap(), data);
        assert!(Bzip2.decode(&[], b"BZh9garbage".to_vec(), &ctx).is_err());
    }
}
