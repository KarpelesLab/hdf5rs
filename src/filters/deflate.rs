//! Zlib / deflate codec (RFC 1950 / RFC 1951), backed by `compcol`.

use crate::{Error, Result};

/// Decompresses a zlib stream.
///
/// `size_hint` is only a pre-allocation hint; output larger than
/// `max_output` fails with [`Error::LimitExceeded`].
pub fn zlib_decompress(input: &[u8], size_hint: usize, max_output: usize) -> Result<Vec<u8>> {
    use compcol::{Algorithm, Decoder, Status};
    const LIMIT: Error = Error::LimitExceeded("deflate output exceeds limit");
    let err = |e: compcol::Error| match e {
        compcol::Error::OutputLimitExceeded => LIMIT,
        e => Error::invalid(format!("deflate: {e}")),
    };
    let mut dec = compcol::zlib::Zlib::decoder();
    // Decode straight into the output buffer (no staging copy), sized from
    // the hint (the expected chunk size) and grown geometrically.
    let first = if size_hint > 0 {
        size_hint
    } else {
        input.len().saturating_mul(4)
    };
    let mut out = vec![0u8; first.max(1024).min(max_output)];
    let mut written = 0usize;
    let mut consumed = 0usize;
    let mut finishing = false;
    let mut probe = [0u8; 1];
    loop {
        if written == out.len() && out.len() < max_output {
            let grown = out.len().saturating_mul(2).max(4096).min(max_output);
            out.resize(grown, 0);
        }
        // At the cap, offer a one-byte buffer: any output is too much.
        let at_cap = written >= out.len();
        let buf: &mut [u8] = match out.get_mut(written..) {
            Some(b) if !at_cap => b,
            _ => &mut probe,
        };
        let (p, status) = if finishing {
            dec.finish(buf)
        } else {
            dec.decode(input.get(consumed..).unwrap_or(&[]), buf)
        }
        .map_err(err)?;
        if at_cap && p.written > 0 {
            return Err(LIMIT);
        }
        written = written.saturating_add(p.written);
        consumed = consumed.saturating_add(p.consumed);
        match status {
            Status::StreamEnd => break,
            _ if p.consumed == 0 && p.written == 0 => {
                if finishing {
                    return Err(Error::invalid("deflate: truncated stream"));
                }
                finishing = true;
            }
            Status::InputEmpty if consumed >= input.len() => finishing = true,
            _ => {}
        }
    }
    out.truncate(written);
    Ok(out)
}

/// Compresses `input` into a zlib stream at `level` (0–9).
///
/// Level 0 is treated as level 1, the fastest compressing level.
pub fn zlib_compress(input: &[u8], level: u32) -> Result<Vec<u8>> {
    let config = compcol::zlib::EncoderConfig {
        level: level.clamp(1, 9) as u8,
    };
    compcol::vec::compress_to_vec_with::<compcol::zlib::Zlib>(input, config)
        .map_err(|e| Error::invalid(format!("deflate: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let data: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        for level in [0, 1, 6, 9] {
            let z = zlib_compress(&data, level).unwrap();
            assert!(z.len() < data.len() / 10);
            assert_eq!(zlib_decompress(&z, data.len(), data.len()).unwrap(), data);
        }
        // Any hint; the cap is exact.
        let z = zlib_compress(&data, 6).unwrap();
        let n = data.len();
        for hint in [0, 1, 5000, n - 1, n, 10 * n] {
            for max in [n, n + 1, 4 * n] {
                assert_eq!(zlib_decompress(&z, hint, max).unwrap(), data);
            }
            for max in [0, 1, 1024, n - 1] {
                assert!(matches!(
                    zlib_decompress(&z, hint, max),
                    Err(Error::LimitExceeded(_))
                ));
            }
        }
        let empty = zlib_compress(&[], 6).unwrap();
        assert_eq!(zlib_decompress(&empty, 0, 0).unwrap(), b"");
    }

    #[test]
    fn known_stream() {
        // zlib.compress(b"hello hello hello")
        let z = [
            0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x90, 0x00, 0x3a, 0x2e,
            0x06, 0x7d,
        ];
        assert_eq!(zlib_decompress(&z, 0, 1024).unwrap(), b"hello hello hello");
    }

    #[test]
    fn limits_and_corruption() {
        let data = vec![0u8; 10_000];
        let z = zlib_compress(&data, 6).unwrap();
        assert!(matches!(
            zlib_decompress(&z, 0, 100),
            Err(Error::LimitExceeded(_))
        ));
        let mut bad = z.clone();
        let n = bad.len();
        bad[n - 1] ^= 1;
        assert!(zlib_decompress(&bad, 0, 20_000).is_err());
        assert!(zlib_decompress(&z[..z.len() / 2], 0, 20_000).is_err());
    }
}
