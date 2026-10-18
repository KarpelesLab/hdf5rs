//! Zlib / deflate codec (RFC 1950 / RFC 1951), backed by `compcol`.

use crate::{Error, Result};

/// Decompresses a zlib stream.
///
/// `size_hint` is only a pre-allocation hint; output larger than
/// `max_output` fails with [`Error::LimitExceeded`].
pub fn zlib_decompress(input: &[u8], size_hint: usize, max_output: usize) -> Result<Vec<u8>> {
    let _ = size_hint;
    compcol::vec::decompress_to_vec_capped::<compcol::zlib::Zlib>(input, max_output as u64).map_err(
        |e| match e {
            compcol::Error::OutputLimitExceeded => {
                Error::LimitExceeded("deflate output exceeds limit")
            }
            e => Error::invalid(format!("deflate: {e}")),
        },
    )
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
