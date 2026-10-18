//! Thin wrappers around the `compcol` codecs used by the optional filters.
//!
//! Every decompressor is capped (decompression-bomb protection) and maps
//! `compcol` errors onto [`Error`].

#![allow(dead_code)] // which helpers are used depends on the enabled features

use crate::{Error, Result};

pub(crate) fn map_err(name: &'static str) -> impl Fn(compcol::Error) -> Error {
    move |e| match e {
        compcol::Error::OutputLimitExceeded => Error::LimitExceeded(name),
        e => Error::invalid(format!("{name}: {e}")),
    }
}

/// Runs `A`'s decoder over `input`, producing at most `max_output` bytes.
///
/// A local loop rather than `compcol::vec::decompress_to_vec_capped`: that
/// helper stops feeding the decoder once all input is consumed, even when
/// the decoder still has buffered output to deliver (seen with large bzip2
/// blocks), and then fails in `finish`.
///
/// `capacity` is a pre-allocation hint (0: twice the input size).
#[cfg(any(
    feature = "zstd",
    feature = "bzip2",
    feature = "blosc",
    feature = "bitshuffle",
    feature = "lzf"
))]
pub(crate) fn decode_all<A: compcol::Algorithm>(
    input: &[u8],
    capacity: usize,
    max_output: usize,
    name: &'static str,
) -> Result<Vec<u8>> {
    use compcol::{Decoder, Status};
    let err = map_err(name);
    let mut dec = A::decoder_with(A::DecoderConfig::default());
    let capacity = match capacity {
        0 => input.len().saturating_mul(2),
        n => n,
    };
    let mut out: Vec<u8> = Vec::with_capacity(capacity.min(max_output));
    let mut scratch = vec![0u8; 64 << 10];
    let push = |out: &mut Vec<u8>, data: &[u8]| {
        if out.len().saturating_add(data.len()) > max_output {
            return Err(Error::LimitExceeded(name));
        }
        out.extend_from_slice(data);
        Ok(())
    };
    let mut consumed = 0usize;
    loop {
        let rest = input.get(consumed..).unwrap_or(&[]);
        let (p, status) = dec.decode(rest, &mut scratch).map_err(&err)?;
        push(&mut out, scratch.get(..p.written).unwrap_or(&[]))?;
        consumed = consumed.saturating_add(p.consumed);
        let stalled = p.consumed == 0 && p.written == 0;
        match status {
            Status::StreamEnd => return Ok(out),
            Status::OutputFull if !stalled => {}
            Status::InputEmpty if !stalled && consumed < input.len() => {}
            _ => break,
        }
    }
    loop {
        let (p, status) = dec.finish(&mut scratch).map_err(&err)?;
        push(&mut out, scratch.get(..p.written).unwrap_or(&[]))?;
        if matches!(status, Status::StreamEnd) {
            return Ok(out);
        }
        if p.written == 0 {
            return Err(Error::invalid(format!("{name}: truncated stream")));
        }
    }
}

/// Decompresses one Zstandard frame, producing at most `max_output` bytes.
#[cfg(any(feature = "zstd", feature = "blosc", feature = "bitshuffle"))]
pub(crate) fn zstd_decompress(input: &[u8], max_output: usize) -> Result<Vec<u8>> {
    decode_all::<compcol::zstd::Zstd>(input, 0, max_output, "zstd")
}

/// Compresses `input` into one Zstandard frame. `level` is clamped to 1–22.
#[cfg(any(feature = "zstd", feature = "blosc", feature = "bitshuffle"))]
pub(crate) fn zstd_compress(input: &[u8], level: i32) -> Result<Vec<u8>> {
    let config = compcol::zstd::EncoderConfig {
        level: level.clamp(1, 22) as u8,
    };
    let frame = compcol::vec::compress_to_vec_with::<compcol::zstd::Zstd>(input, config)
        .map_err(map_err("zstd"))?;
    Ok(with_content_size(frame, input.len() as u64))
}

/// Adds the Frame_Content_Size field (RFC 8878 §3.1.1.1) to a single
/// Zstandard frame that lacks it: the HDF5 zstd filter sizes its output
/// buffer from that field and rejects frames without it.
#[cfg(any(feature = "zstd", feature = "blosc", feature = "bitshuffle"))]
fn with_content_size(frame: Vec<u8>, size: u64) -> Vec<u8> {
    const MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];
    let Some(&fhd) = frame.get(4) else {
        return frame;
    };
    let single_segment = fhd & 0x20 != 0;
    if !frame.starts_with(&MAGIC) || fhd >> 6 != 0 || single_segment {
        return frame; // already has a content size (or is not a frame)
    }
    let dict_len = [0, 1, 2, 4][usize::from(fhd & 3)];
    // Magic, descriptor, window descriptor, dictionary id.
    let header_len = 6 + dict_len;
    if frame.len() < header_len {
        return frame;
    }
    let mut out = Vec::with_capacity(frame.len() + 8);
    out.extend_from_slice(&MAGIC);
    out.push(fhd | 0xc0); // FCS_Field_Size = 8 bytes
    out.extend_from_slice(&frame[5..header_len]);
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&frame[header_len..]);
    out
}

/// Decodes one raw LZ4 block that must expand to exactly `len` bytes.
#[cfg(any(feature = "lz4", feature = "blosc", feature = "bitshuffle"))]
pub(crate) fn lz4_block_decompress(input: &[u8], len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(len.min(64 << 20));
    compcol::lz4::block::decode_block(input, &mut out, len).map_err(map_err("lz4"))?;
    if out.len() != len {
        return Err(Error::invalid("lz4: block decoded to an unexpected size"));
    }
    Ok(out)
}

/// Encodes `input` as one raw LZ4 block.
#[cfg(any(feature = "lz4", feature = "blosc", feature = "bitshuffle"))]
pub(crate) fn lz4_block_compress(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    compcol::lz4::block::encode_block(input, &mut out);
    out
}

/// Decompresses a bzip2 stream, producing at most `max_output` bytes.
#[cfg(feature = "bzip2")]
pub(crate) fn bzip2_decompress(input: &[u8], max_output: usize) -> Result<Vec<u8>> {
    decode_all::<compcol::bzip2::Bzip2>(input, 0, max_output, "bzip2")
}

/// Compresses `input` into a bzip2 stream with block size `level` (1–9).
#[cfg(feature = "bzip2")]
pub(crate) fn bzip2_compress(input: &[u8], level: u32) -> Result<Vec<u8>> {
    let config = compcol::bzip2::EncoderConfig {
        level: level.clamp(1, 9) as u8,
    };
    compcol::vec::compress_to_vec_with::<compcol::bzip2::Bzip2>(input, config)
        .map_err(map_err("bzip2"))
}

/// Decompresses a raw Snappy block, producing at most `max_output` bytes.
#[cfg(feature = "blosc")]
pub(crate) fn snappy_decompress(input: &[u8], max_output: usize) -> Result<Vec<u8>> {
    decode_all::<compcol::snappy::Snappy>(input, 0, max_output, "snappy")
}

/// Compresses `input` into a raw Snappy block.
#[cfg(feature = "blosc")]
pub(crate) fn snappy_compress(input: &[u8]) -> Result<Vec<u8>> {
    compcol::vec::compress_to_vec::<compcol::snappy::Snappy>(input).map_err(map_err("snappy"))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(any(feature = "lz4", feature = "blosc", feature = "bitshuffle"))]
    fn lz4_round_trip_and_size_check() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i / 7) as u8).collect();
        let c = lz4_block_compress(&data);
        assert!(c.len() < data.len() / 4);
        assert_eq!(lz4_block_decompress(&c, data.len()).unwrap(), data);
        assert!(lz4_block_decompress(&c, data.len() - 1).is_err());
        assert!(lz4_block_decompress(&c, data.len() + 1).is_err());
    }

    #[test]
    #[cfg(any(feature = "zstd", feature = "blosc", feature = "bitshuffle"))]
    fn zstd_round_trip_and_cap() {
        let data = vec![7u8; 100_000];
        let c = zstd_compress(&data, 3).unwrap();
        // Frame_Content_Size present (8 bytes) and correct.
        assert_eq!(c[4] >> 6, 3);
        assert_eq!(zstd_decompress(&c, data.len()).unwrap(), data);
        assert!(matches!(
            zstd_decompress(&c, 1000),
            Err(Error::LimitExceeded(_))
        ));
    }

    #[test]
    #[cfg(feature = "bzip2")]
    fn bzip2_round_trip() {
        let data: Vec<u8> = (0..20_000u32).map(|i| (i % 13) as u8).collect();
        let c = bzip2_compress(&data, 9).unwrap();
        assert_eq!(bzip2_decompress(&c, data.len()).unwrap(), data);
        assert!(bzip2_decompress(&c, 100).is_err());
    }

    #[test]
    #[cfg(feature = "blosc")]
    fn snappy_round_trip() {
        let data: Vec<u8> = (0..20_000u32).map(|i| (i % 13) as u8).collect();
        let c = snappy_compress(&data).unwrap();
        assert_eq!(snappy_decompress(&c, data.len()).unwrap(), data);
        assert!(snappy_decompress(&c, 100).is_err());
    }
}
