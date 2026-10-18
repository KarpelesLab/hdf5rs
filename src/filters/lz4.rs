//! The LZ4 filter (registered filter id 32004, feature `lz4`).
//!
//! The HDF5 LZ4 filter splits a chunk into blocks and stores each one as a
//! raw LZ4 block (no LZ4 frame format):
//!
//! ```text
//! u64 BE   total uncompressed size
//! u32 BE   block size (uncompressed bytes per block; the last may be short)
//! per block:
//!   u32 BE   stored size
//!   ...      LZ4 block, or the block itself when the stored size equals
//!            the block's uncompressed size (incompressible data)
//! ```
//!
//! Client data: `[block size]` for encoding (0 = one block per chunk, up
//! to 1 GiB); ignored when decoding.

use super::codec::{lz4_block_compress, lz4_block_decompress};
use super::{Filter, FilterContext, LZ4};
use crate::{Error, Result};

/// Default (and maximum) block size: 1 GiB.
const DEFAULT_BLOCK: usize = 1 << 30;

/// The LZ4 filter, id [`LZ4`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Lz4;

fn be32(input: &[u8], pos: usize) -> Result<usize> {
    input
        .get(pos..pos + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .ok_or_else(|| Error::invalid("lz4: chunk is truncated"))
}

impl Filter for Lz4 {
    fn id(&self) -> u16 {
        LZ4
    }

    fn name(&self) -> &str {
        "lz4"
    }

    fn decode(&self, _client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        let header = input
            .get(..12)
            .ok_or_else(|| Error::invalid("lz4: chunk is truncated"))?;
        let total = u64::from_be_bytes(header[..8].try_into().unwrap_or([0xff; 8]));
        let total = usize::try_from(total)
            .ok()
            .filter(|&t| t <= ctx.max_output)
            .ok_or(Error::LimitExceeded("lz4 output size"))?;
        let block = be32(&input, 8)?;
        if block == 0 && total > 0 {
            return Err(Error::invalid("lz4: zero block size"));
        }
        let mut out = Vec::with_capacity(ctx.capacity_hint(Some(total)));
        let mut pos = 12;
        while out.len() < total {
            let this = block.min(total - out.len());
            let stored = be32(&input, pos)?;
            pos += 4;
            let data = input
                .get(pos..pos.saturating_add(stored))
                .ok_or_else(|| Error::invalid("lz4: chunk is truncated"))?;
            pos += stored;
            if stored == this {
                out.extend_from_slice(data);
            } else if stored > this {
                return Err(Error::invalid(
                    "lz4: block larger than its uncompressed size",
                ));
            } else {
                out.extend_from_slice(&lz4_block_decompress(data, this)?);
            }
        }
        Ok(out)
    }

    fn encode(&self, client_data: &[u32], input: Vec<u8>, _ctx: &FilterContext) -> Result<Vec<u8>> {
        let requested = client_data.first().copied().unwrap_or(0) as usize;
        let block = match requested {
            0 => DEFAULT_BLOCK,
            n => n.min(DEFAULT_BLOCK),
        }
        .min(input.len());
        let block_u32 = u32::try_from(block).map_err(|_| Error::LimitExceeded("lz4 block size"))?;
        let mut out = Vec::with_capacity(input.len() / 2 + 16);
        out.extend_from_slice(&(input.len() as u64).to_be_bytes());
        out.extend_from_slice(&block_u32.to_be_bytes());
        if block > 0 {
            for part in input.chunks(block) {
                let c = lz4_block_compress(part);
                if c.len() < part.len() {
                    out.extend_from_slice(&(c.len() as u32).to_be_bytes());
                    out.extend_from_slice(&c);
                } else {
                    out.extend_from_slice(&(part.len() as u32).to_be_bytes());
                    out.extend_from_slice(part);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_blocks() {
        let ctx = FilterContext::new(4);
        let data: Vec<u8> = (0..5000u32).flat_map(|i| (i / 7).to_le_bytes()).collect();
        for block in [0u32, 1024, 999, 20_000, 1 << 31] {
            let c = Lz4.encode(&[block], data.clone(), &ctx).unwrap();
            assert!(c.len() < data.len() / 2, "{block}");
            assert_eq!(Lz4.decode(&[], c, &ctx).unwrap(), data);
        }
        // Empty chunk.
        let c = Lz4.encode(&[], Vec::new(), &ctx).unwrap();
        assert_eq!(Lz4.decode(&[], c, &ctx).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn raw_blocks() {
        // Layout written by the reference filter for 200 incompressible
        // bytes: total, block size, stored size == block size, raw bytes.
        let data: Vec<u8> = (0..200u8).collect();
        let mut chunk = vec![0, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 200, 0, 0, 0, 200];
        chunk.extend_from_slice(&data);
        let ctx = FilterContext::new(1);
        assert_eq!(Lz4.decode(&[0], chunk.clone(), &ctx).unwrap(), data);
        assert_eq!(Lz4.encode(&[0], data, &ctx).unwrap(), chunk);
    }

    #[test]
    fn malformed() {
        let ctx = FilterContext::new(1).with_max_output(1000);
        let mut huge = vec![0xff; 8];
        huge.extend_from_slice(&[0, 0, 1, 0]);
        assert!(matches!(
            Lz4.decode(&[], huge, &ctx),
            Err(Error::LimitExceeded(_))
        ));
        let zero_block = vec![0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0];
        assert!(Lz4.decode(&[], zero_block, &ctx).is_err());
        let truncated = vec![0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10, 1, 2];
        assert!(Lz4.decode(&[], truncated, &ctx).is_err());
        let too_big = vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 1, 2];
        assert!(Lz4.decode(&[], too_big, &ctx).is_err());
        assert!(Lz4.decode(&[], vec![0; 5], &ctx).is_err());
    }
}
