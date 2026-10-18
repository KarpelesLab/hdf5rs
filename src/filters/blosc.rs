//! The Blosc filter (registered filter id 32001, feature `blosc`).
//!
//! Chunks hold one Blosc (version 1 format) buffer, as described in the
//! public Blosc header format documentation:
//!
//! ```text
//! byte 0      format version (2)
//! byte 1      codec format version
//! byte 2      flags: 0x01 byte shuffle, 0x02 stored uncompressed
//!             ("memcpyed"), 0x04 bit shuffle, 0x10 blocks not split,
//!             bits 5-7 codec (0 blosclz, 1 lz4/lz4hc, 2 snappy, 3 zlib,
//!             4 zstd)
//! byte 3      type size
//! bytes 4..8  uncompressed size (u32 LE)
//! bytes 8..12 block size (u32 LE)
//! bytes 12..16 compressed size of the whole buffer (u32 LE)
//! ```
//!
//! A "memcpyed" buffer has the data right after the header. Otherwise the
//! header is followed by one u32 LE start offset per block, and every block
//! (the last one may be shorter) consists of one or — when split, which
//! the last, shorter block never is — `type size` streams, each a u32 LE
//! compressed size followed by the compressed bytes (or the raw bytes when
//! the size equals the stream's uncompressed size). After decompression a
//! block is un-shuffled (byte or bit shuffle over `type size` elements;
//! blocks that do not hold a multiple of 8 elements are never bit-shuffled).
//!
//! The blosclz codec is not supported (there is no public specification of
//! its format); lz4, snappy, zlib and zstd are.
//!
//! Client data: `[filter revision, blosc version, type size, chunk size,
//! level, shuffle (0 none, 1 byte, 2 bit), codec (0 blosclz, 1 lz4,
//! 2 lz4hc, 3 snappy, 4 zlib, 5 zstd)]`. Only the last four matter for
//! encoding (decoding is driven by the header). Encoding with blosclz uses
//! lz4 instead (any Blosc reader handles it); level 0 stores the data
//! uncompressed.

use super::bitshuffle::{bitshuffle, bitunshuffle};
use super::codec;
use super::deflate::{zlib_compress, zlib_decompress};
use super::shuffle::{shuffle, unshuffle};
use super::{BLOSC, Filter, FilterContext};
use crate::{Error, Result};

const HEADER_LEN: usize = 16;
const FLAG_SHUFFLE: u8 = 0x01;
const FLAG_MEMCPYED: u8 = 0x02;
const FLAG_BITSHUFFLE: u8 = 0x04;
const FLAG_NOSPLIT: u8 = 0x10;

const CODEC_BLOSCLZ: u8 = 0;
const CODEC_LZ4: u8 = 1;
const CODEC_SNAPPY: u8 = 2;
const CODEC_ZLIB: u8 = 3;
const CODEC_ZSTD: u8 = 4;

/// Block size used when encoding chunks larger than one block.
const ENCODE_BLOCK: usize = 256 << 10;

/// The Blosc filter, id [`BLOSC`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Blosc;

fn le32(input: &[u8], pos: usize) -> Result<usize> {
    input
        .get(pos..pos.saturating_add(4))
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .ok_or_else(|| Error::invalid("blosc: buffer is truncated"))
}

/// Whether a block of `bsize` bytes is bit-shuffled when the bit-shuffle
/// flag is set: only blocks holding a whole multiple of 8 elements are;
/// others are stored unshuffled.
fn bitshuffled(bsize: usize, typesize: usize) -> bool {
    bsize.is_multiple_of(8 * typesize.max(1))
}

fn decompress(codec: u8, data: &[u8], len: usize) -> Result<Vec<u8>> {
    let out = match codec {
        CODEC_LZ4 => return codec::lz4_block_decompress(data, len),
        CODEC_SNAPPY => codec::snappy_decompress(data, len)?,
        CODEC_ZLIB => zlib_decompress(data, len, len)?,
        CODEC_ZSTD => codec::zstd_decompress(data, len)?,
        CODEC_BLOSCLZ => return Err(Error::unsupported("blosc: blosclz codec")),
        c => return Err(Error::unsupported(format!("blosc: codec {c}"))),
    };
    if out.len() != len {
        return Err(Error::invalid(
            "blosc: stream decoded to an unexpected size",
        ));
    }
    Ok(out)
}

fn compress(codec: u8, data: &[u8], level: u32) -> Result<Vec<u8>> {
    match codec {
        CODEC_LZ4 => Ok(codec::lz4_block_compress(data)),
        CODEC_SNAPPY => codec::snappy_compress(data),
        CODEC_ZLIB => zlib_compress(data, level),
        CODEC_ZSTD => codec::zstd_compress(data, level as i32),
        _ => Err(Error::unsupported("blosc: codec")),
    }
}

impl Filter for Blosc {
    fn id(&self) -> u16 {
        BLOSC
    }

    fn name(&self) -> &str {
        "blosc"
    }

    fn decode(&self, _client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        let h = input
            .get(..HEADER_LEN)
            .ok_or_else(|| Error::invalid("blosc: buffer is truncated"))?;
        let (version, flags, typesize) = (h[0], h[2], usize::from(h[3]));
        if version == 0 || version > 2 {
            return Err(Error::unsupported(format!(
                "blosc: format version {version}"
            )));
        }
        let nbytes = le32(&input, 4)?;
        let blocksize = le32(&input, 8)?;
        let cbytes = le32(&input, 12)?;
        if nbytes > ctx.max_output {
            return Err(Error::LimitExceeded("blosc output size"));
        }
        let buf = input
            .get(..cbytes)
            .filter(|_| cbytes >= HEADER_LEN)
            .ok_or_else(|| Error::invalid("blosc: bad compressed size"))?;
        if flags & FLAG_MEMCPYED != 0 {
            return buf
                .get(HEADER_LEN..HEADER_LEN + nbytes)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| Error::invalid("blosc: buffer is truncated"));
        }
        if nbytes == 0 {
            return Ok(Vec::new());
        }
        if blocksize == 0 {
            return Err(Error::invalid("blosc: zero block size"));
        }
        let nblocks = nbytes.div_ceil(blocksize);
        if nblocks.saturating_mul(4) > buf.len() - HEADER_LEN {
            return Err(Error::invalid("blosc: too many blocks"));
        }
        let codec = flags >> 5;
        let mut out = Vec::with_capacity(ctx.capacity_hint(Some(nbytes)));
        for b in 0..nblocks {
            let bsize = blocksize.min(nbytes - b * blocksize);
            let leftover = bsize < blocksize;
            let split = flags & FLAG_NOSPLIT == 0 && !leftover && typesize > 1;
            let nstreams = if split { typesize } else { 1 };
            if bsize % nstreams != 0 {
                return Err(Error::invalid(
                    "blosc: block not a multiple of the type size",
                ));
            }
            let neblock = bsize / nstreams;
            let mut pos = le32(buf, HEADER_LEN + 4 * b)?;
            let mut block = Vec::with_capacity(bsize);
            for _ in 0..nstreams {
                let csize = le32(buf, pos)?;
                pos += 4;
                let data = buf
                    .get(pos..pos.saturating_add(csize))
                    .ok_or_else(|| Error::invalid("blosc: stream is truncated"))?;
                pos += csize;
                if csize == neblock {
                    block.extend_from_slice(data);
                } else if csize > neblock {
                    return Err(Error::invalid("blosc: stream larger than its block"));
                } else {
                    block.extend_from_slice(&decompress(codec, data, neblock)?);
                }
            }
            let block = if flags & FLAG_SHUFFLE != 0 {
                unshuffle(block, typesize)
            } else if flags & FLAG_BITSHUFFLE != 0 && bitshuffled(bsize, typesize) {
                bitunshuffle(&block, typesize)
            } else {
                block
            };
            out.extend_from_slice(&block);
        }
        Ok(out)
    }

    fn encode(&self, client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        let typesize = match client_data.get(2) {
            Some(&t) if t > 0 => t as usize,
            _ => ctx.element_size.max(1),
        };
        // Blosc stores the type size in one byte.
        let typesize = if typesize > 255 { 1 } else { typesize };
        let level = client_data.get(4).copied().unwrap_or(5).min(9);
        let shuffle_mode = client_data.get(5).copied().unwrap_or(1);
        let codec = match client_data.get(6).copied().unwrap_or(0) {
            0..=2 => CODEC_LZ4, // blosclz is replaced by lz4
            3 => CODEC_SNAPPY,
            4 => CODEC_ZLIB,
            5 => CODEC_ZSTD,
            c => return Err(Error::unsupported(format!("blosc: compressor {c}"))),
        };
        let nbytes = u32::try_from(input.len())
            .ok()
            .filter(|&n| n <= i32::MAX as u32 - HEADER_LEN as u32)
            .ok_or(Error::LimitExceeded("blosc input size"))?;
        let shuffle_flag = match shuffle_mode {
            0 => 0,
            1 => FLAG_SHUFFLE,
            2 => FLAG_BITSHUFFLE,
            s => return Err(Error::unsupported(format!("blosc: shuffle mode {s}"))),
        };
        let flags = shuffle_flag | FLAG_NOSPLIT | (codec << 5);
        let header = |flags: u8, blocksize: usize, cbytes: usize| {
            let mut h = vec![2, 1, flags, typesize as u8];
            h.extend_from_slice(&nbytes.to_le_bytes());
            h.extend_from_slice(&(blocksize as u32).to_le_bytes());
            h.extend_from_slice(&(cbytes as u32).to_le_bytes());
            h
        };
        let memcpyed = |input: &[u8]| {
            let mut out = header(flags | FLAG_MEMCPYED, input.len(), HEADER_LEN + input.len());
            out.extend_from_slice(input);
            out
        };
        if level == 0 || input.is_empty() {
            return Ok(memcpyed(&input));
        }
        let blocksize = if input.len() <= ENCODE_BLOCK {
            input.len()
        } else {
            ENCODE_BLOCK / typesize * typesize
        };
        let nblocks = input.len().div_ceil(blocksize);
        let mut body = Vec::with_capacity(input.len() / 2);
        let mut starts = Vec::with_capacity(nblocks);
        let base = HEADER_LEN + 4 * nblocks;
        for part in input.chunks(blocksize) {
            starts.push(base + body.len());
            let shuffled = match shuffle_flag {
                FLAG_SHUFFLE => shuffle(part.to_vec(), typesize),
                FLAG_BITSHUFFLE if bitshuffled(part.len(), typesize) => bitshuffle(part, typesize),
                _ => part.to_vec(),
            };
            let c = compress(codec, &shuffled, level)?;
            let stream = if c.len() < shuffled.len() {
                &c
            } else {
                &shuffled
            };
            body.extend_from_slice(&(stream.len() as u32).to_le_bytes());
            body.extend_from_slice(stream);
        }
        let cbytes = base + body.len();
        if cbytes >= HEADER_LEN + input.len() {
            return Ok(memcpyed(&input));
        }
        let mut out = header(flags, blocksize, cbytes);
        for s in starts {
            out.extend_from_slice(&(s as u32).to_le_bytes());
        }
        out.extend_from_slice(&body);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let ctx = FilterContext::new(4);
        let data: Vec<u8> = (0..300_001u32)
            .flat_map(|i| (i / 7).to_le_bytes())
            .collect();
        for codec in 0..=5u32 {
            for sh in 0..=2u32 {
                let cd = [2, 2, 4, 0, 5, sh, codec];
                let c = Blosc.encode(&cd, data.clone(), &ctx).unwrap();
                assert!(c.len() < data.len() / 2, "{codec} {sh}");
                assert_eq!(Blosc.decode(&cd, c, &ctx).unwrap(), data, "{codec} {sh}");
            }
        }
        // Level 0 and incompressible data are stored "memcpyed".
        let c = Blosc
            .encode(&[2, 2, 4, 0, 0, 1, 1], data[..1000].to_vec(), &ctx)
            .unwrap();
        assert_eq!(c[2] & FLAG_MEMCPYED, FLAG_MEMCPYED);
        assert_eq!(Blosc.decode(&[], c, &ctx).unwrap(), &data[..1000]);
        let c = Blosc.encode(&[], Vec::new(), &ctx).unwrap();
        assert_eq!(Blosc.decode(&[], c, &ctx).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn reference_chunk() {
        // A 20-byte chunk (5 int32: 0..5) with byte shuffle, laid out as
        // the reference implementation does for incompressible streams.
        let data: Vec<u8> = (0..5u32).flat_map(|i| i.to_le_bytes()).collect();
        let shuffled = shuffle(data.clone(), 4);
        // Hand-assembled: header, one block start, one raw stream.
        let mut chunk = vec![
            2, 1, 0x31, 4, 20, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0,
        ];
        chunk.extend_from_slice(&20u32.to_le_bytes());
        chunk.extend_from_slice(&shuffled);
        assert_eq!(
            Blosc.decode(&[], chunk, &FilterContext::new(4)).unwrap(),
            data
        );
        // Split streams: 4 raw streams of 5 bytes.
        let mut chunk = vec![
            2, 1, 0x21, 4, 20, 0, 0, 0, 20, 0, 0, 0, 56, 0, 0, 0, 20, 0, 0, 0,
        ];
        for s in shuffled.chunks(5) {
            chunk.extend_from_slice(&5u32.to_le_bytes());
            chunk.extend_from_slice(s);
        }
        assert_eq!(
            Blosc.decode(&[], chunk, &FilterContext::new(4)).unwrap(),
            data
        );
    }

    #[test]
    fn malformed() {
        let ctx = FilterContext::new(4).with_max_output(1 << 20);
        let ok = Blosc
            .encode(
                &[2, 2, 4, 0, 5, 1, 1],
                (0..4000u32).flat_map(|i| (i / 3).to_le_bytes()).collect(),
                &ctx,
            )
            .unwrap();
        assert!(Blosc.decode(&[], ok[..10].to_vec(), &ctx).is_err());
        assert!(
            Blosc
                .decode(&[], ok[..ok.len() - 1].to_vec(), &ctx)
                .is_err()
        );
        let mut bad = ok.clone();
        bad[2] &= 0x1f; // blosclz
        assert!(matches!(
            Blosc.decode(&[], bad, &ctx),
            Err(Error::Unsupported(_))
        ));
        let mut bad = ok.clone();
        bad[8..12].copy_from_slice(&[0; 4]); // zero block size
        assert!(Blosc.decode(&[], bad, &ctx).is_err());
        let mut bad = ok.clone();
        bad[8..12].copy_from_slice(&[1, 0, 0, 0]); // too many blocks
        assert!(Blosc.decode(&[], bad, &ctx).is_err());
        let mut bad = ok.clone();
        bad[4..8].copy_from_slice(&[0, 0, 0, 0x10]);
        assert!(matches!(
            Blosc.decode(&[], bad, &ctx),
            Err(Error::LimitExceeded(_))
        ));
        let mut bad = ok.clone();
        bad[0] = 9;
        assert!(Blosc.decode(&[], bad, &ctx).is_err());
    }
}
