//! Bitshuffle: the bit-transpose transform and the Bitshuffle filter
//! (registered filter id 32008, feature `bitshuffle`).
//!
//! # Transform
//!
//! A block of `n` elements of `s` bytes (`n` a multiple of 8) is viewed as
//! an `n × 8s` bit matrix and transposed: bit `k` of byte `j` of element
//! `i` moves to bit `i % 8` of output byte `(8j + k) · n/8 + i/8`. So the
//! output holds, for every bit position, one bit of each element, packed
//! LSB first. Trailing elements that do not fill a multiple of 8 (and any
//! trailing partial element) are copied unchanged.
//!
//! # Filter chunk format
//!
//! Client data: `[major, minor, element size, block size (elements, 0 =
//! automatic), compression (0 none, 2 LZ4, 3 Zstandard), zstd level]`.
//!
//! Without compression the chunk is transformed in blocks of `block size`
//! elements (the final partial block as described above) and stored as is.
//! With compression:
//!
//! ```text
//! u64 BE   uncompressed size in bytes
//! u32 BE   block size in bytes
//! per block (full blocks, then the final partial block truncated to a
//! multiple of 8 elements):
//!   u32 BE   compressed size
//!   ...      LZ4 block / Zstandard frame of the transformed block
//! then the leftover (< 8) elements, unchanged and uncompressed
//! ```

/// Bit-transposes `input` as elements of `elem_size` bytes (see the module
/// documentation). Trailing bytes not forming a multiple of 8 elements are
/// copied unchanged.
pub fn bitshuffle(input: &[u8], elem_size: usize) -> Vec<u8> {
    transpose(input, elem_size, true)
}

/// Inverse of [`bitshuffle`].
pub fn bitunshuffle(input: &[u8], elem_size: usize) -> Vec<u8> {
    transpose(input, elem_size, false)
}

fn transpose(input: &[u8], elem_size: usize, forward: bool) -> Vec<u8> {
    let mut out = vec![0u8; input.len()];
    if elem_size == 0 {
        out.copy_from_slice(input);
        return out;
    }
    let n = input.len() / elem_size / 8 * 8;
    let main = n * elem_size;
    let row = n / 8;
    for i in 0..n {
        for j in 0..elem_size {
            for k in 0..8 {
                // Element-major position and bit-major position of one bit.
                let (eb, ebit) = (i * elem_size + j, k);
                let (bb, bbit) = ((j * 8 + k) * row + i / 8, i % 8);
                let (src, sbit, dst, dbit) = if forward {
                    (eb, ebit, bb, bbit)
                } else {
                    (bb, bbit, eb, ebit)
                };
                out[dst] |= ((input[src] >> sbit) & 1) << dbit;
            }
        }
    }
    out[main..].copy_from_slice(&input[main..]);
    out
}

/// Default block size (elements) for `elem_size`-byte elements: about
/// 8 KiB, a multiple of 8, at least 128 elements.
#[cfg(feature = "bitshuffle")]
fn default_block(elem_size: usize) -> usize {
    (8192 / elem_size.max(1) / 8 * 8).max(128)
}

#[cfg(feature = "bitshuffle")]
pub use filter::Bitshuffle;

#[cfg(feature = "bitshuffle")]
mod filter {
    use super::*;
    use crate::filters::codec::{
        lz4_block_compress, lz4_block_decompress, zstd_compress, zstd_decompress,
    };
    use crate::filters::{BITSHUFFLE, Filter, FilterContext};
    use crate::{Error, Result};

    const COMPRESS_NONE: u32 = 0;
    const COMPRESS_LZ4: u32 = 2;
    const COMPRESS_ZSTD: u32 = 3;

    /// The Bitshuffle filter, id [`BITSHUFFLE`].
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Bitshuffle;

    struct Params {
        elem_size: usize,
        block: usize,
        compression: u32,
        level: i32,
    }

    impl Params {
        fn parse(cd: &[u32], ctx: &FilterContext) -> Result<Params> {
            let elem_size = match cd.get(2) {
                Some(&s) if s > 0 => s as usize,
                _ => ctx.element_size,
            };
            // Bounded so that `8 * elem_size` cannot overflow.
            if elem_size == 0 || elem_size > (u32::MAX / 8) as usize {
                return Err(Error::invalid("bitshuffle: bad element size"));
            }
            let block = match cd.get(3) {
                Some(&b) if b > 0 => b as usize,
                _ => default_block(elem_size),
            };
            if block % 8 != 0 {
                return Err(Error::invalid("bitshuffle: block size not a multiple of 8"));
            }
            let compression = cd.get(4).copied().unwrap_or(COMPRESS_NONE);
            if !matches!(compression, COMPRESS_NONE | COMPRESS_LZ4 | COMPRESS_ZSTD) {
                return Err(Error::unsupported(format!(
                    "bitshuffle: compression {compression}"
                )));
            }
            Ok(Params {
                elem_size,
                block,
                compression,
                level: cd.get(5).map_or(3, |&l| l as i32),
            })
        }
    }

    fn be32(input: &[u8], pos: usize) -> Result<usize> {
        input
            .get(pos..pos + 4)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize)
            .ok_or_else(|| Error::invalid("bitshuffle: chunk is truncated"))
    }

    /// Applies `f` to every block (`block_bytes` long, the last one shorter)
    /// of `input`; each call receives the block and returns its transform.
    fn blocks(input: &[u8], block_bytes: usize, mut f: impl FnMut(&[u8]) -> Vec<u8>) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len());
        for part in input.chunks(block_bytes.max(1)) {
            out.extend_from_slice(&f(part));
        }
        out
    }

    impl Filter for Bitshuffle {
        fn id(&self) -> u16 {
            BITSHUFFLE
        }

        fn name(&self) -> &str {
            "bitshuffle"
        }

        fn decode(
            &self,
            client_data: &[u32],
            input: Vec<u8>,
            ctx: &FilterContext,
        ) -> Result<Vec<u8>> {
            let p = Params::parse(client_data, ctx)?;
            let es = p.elem_size;
            if p.compression == COMPRESS_NONE {
                let bb = p.block.saturating_mul(es);
                return Ok(blocks(&input, bb, |b| bitunshuffle(b, es)));
            }
            let total = input
                .get(..8)
                .map(|b| u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .ok_or_else(|| Error::invalid("bitshuffle: chunk is truncated"))?;
            let total = usize::try_from(total)
                .ok()
                .filter(|&t| t <= ctx.max_output)
                .ok_or(Error::LimitExceeded("bitshuffle output size"))?;
            let block_bytes = be32(&input, 8)?;
            if block_bytes == 0 || block_bytes % (8 * es) != 0 {
                return Err(Error::invalid("bitshuffle: bad block size"));
            }
            let mut out = Vec::with_capacity(ctx.capacity_hint(Some(total)));
            let mut pos = 12;
            // Bytes covered by compressed blocks: whole groups of 8 elements.
            let compressed_end = total / (8 * es) * (8 * es);
            while out.len() < compressed_end {
                let this = block_bytes.min(compressed_end - out.len());
                let csize = be32(&input, pos)?;
                pos += 4;
                let data = input
                    .get(pos..pos.saturating_add(csize))
                    .ok_or_else(|| Error::invalid("bitshuffle: chunk is truncated"))?;
                pos += csize;
                let plain = if p.compression == COMPRESS_LZ4 {
                    lz4_block_decompress(data, this)?
                } else {
                    let d = zstd_decompress(data, this)?;
                    if d.len() != this {
                        return Err(Error::invalid("bitshuffle: block has the wrong size"));
                    }
                    d
                };
                out.extend_from_slice(&bitunshuffle(&plain, es));
            }
            let rest = input
                .get(pos..pos.saturating_add(total - out.len()))
                .ok_or_else(|| Error::invalid("bitshuffle: chunk is truncated"))?;
            out.extend_from_slice(rest);
            Ok(out)
        }

        fn encode(
            &self,
            client_data: &[u32],
            input: Vec<u8>,
            ctx: &FilterContext,
        ) -> Result<Vec<u8>> {
            let p = Params::parse(client_data, ctx)?;
            let es = p.elem_size;
            let block_bytes = p.block.saturating_mul(es);
            if p.compression == COMPRESS_NONE {
                return Ok(blocks(&input, block_bytes, |b| bitshuffle(b, es)));
            }
            let block_u32 = u32::try_from(block_bytes)
                .map_err(|_| Error::LimitExceeded("bitshuffle block size"))?;
            let mut out = Vec::with_capacity(input.len() / 2 + 16);
            out.extend_from_slice(&(input.len() as u64).to_be_bytes());
            out.extend_from_slice(&block_u32.to_be_bytes());
            let compressed_end = input.len() / (8 * es) * (8 * es);
            for part in input[..compressed_end].chunks(block_bytes) {
                let t = bitshuffle(part, es);
                let c = if p.compression == COMPRESS_LZ4 {
                    lz4_block_compress(&t)
                } else {
                    zstd_compress(&t, p.level)?
                };
                let len = u32::try_from(c.len())
                    .map_err(|_| Error::LimitExceeded("bitshuffle block size"))?;
                out.extend_from_slice(&len.to_be_bytes());
                out.extend_from_slice(&c);
            }
            out.extend_from_slice(&input[compressed_end..]);
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        // 50 int32 values 0..50: 48 transposed elements, 2 copied.
        let data: Vec<u8> = (0..50u32).flat_map(|i| i.to_le_bytes()).collect();
        let t = bitshuffle(&data, 4);
        assert_eq!(&t[..6], &[0xaa; 6]);
        assert_eq!(&t[6..12], &[0xcc; 6]);
        assert_eq!(&t[192..], &[48, 0, 0, 0, 49, 0, 0, 0]);
        assert_eq!(bitunshuffle(&t, 4), data);
    }

    #[test]
    fn round_trips() {
        for es in [1usize, 2, 3, 4, 8, 16] {
            for n in [0usize, 1, 7, 8, 9, 64, 100] {
                let data: Vec<u8> = (0..n * es + es / 2).map(|i| (i * 37 + 5) as u8).collect();
                assert_eq!(bitunshuffle(&bitshuffle(&data, es), es), data, "{es} {n}");
            }
        }
        assert_eq!(bitshuffle(&[1, 2, 3], 0), [1, 2, 3]);
    }

    #[cfg(feature = "bitshuffle")]
    #[test]
    fn filter_round_trips() {
        use crate::Error;
        use crate::filters::{Filter, FilterContext};
        let ctx = FilterContext::new(4);
        let data: Vec<u8> = (0..5003u32).flat_map(|i| (i / 7).to_le_bytes()).collect();
        for cd in [
            &[0u32, 4, 4, 0, 0][..],
            &[0, 4, 4, 0, 2],
            &[0, 4, 4, 256, 2],
            &[0, 4, 4, 0, 3, 3],
            &[0, 4, 4, 64, 3, 1],
            &[],
        ] {
            let c = Bitshuffle.encode(cd, data.clone(), &ctx).unwrap();
            if cd.get(4).is_some_and(|&c| c != 0) {
                assert!(c.len() < data.len() / 2, "{cd:?}");
            }
            assert_eq!(Bitshuffle.decode(cd, c, &ctx).unwrap(), data, "{cd:?}");
        }
        assert!(
            Bitshuffle
                .decode(&[0, 4, 4, 0, 9], data.clone(), &ctx)
                .is_err()
        );
        assert!(
            Bitshuffle
                .decode(&[0, 4, 4, 7, 0], data.clone(), &ctx)
                .is_err()
        );
        assert!(
            Bitshuffle
                .decode(&[0, 4, 4, 0, 2], vec![0; 11], &ctx)
                .is_err()
        );
        let small = ctx.with_max_output(100);
        let c = Bitshuffle.encode(&[0, 4, 4, 0, 2], data, &ctx).unwrap();
        assert!(matches!(
            Bitshuffle.decode(&[0, 4, 4, 0, 2], c, &small),
            Err(Error::LimitExceeded(_))
        ));
    }
}
