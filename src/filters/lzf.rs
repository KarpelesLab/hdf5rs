//! The LZF filter (id 32000), as registered by h5py.
//!
//! Chunks hold a raw LZF stream (no header). The stream is a sequence of
//! commands, each introduced by a control byte `c`:
//!
//! - `c < 32`: a literal run; the next `c + 1` bytes are copied to the
//!   output.
//! - `c >= 32`: a back-reference. The length field is `c >> 5` (1–6); if it
//!   is 7 an extra byte follows and is added to it. Then one more byte
//!   follows; the distance is `((c & 0x1f) << 8 | byte) + 1` and
//!   `length + 2` bytes are copied from that far back in the output
//!   (copies may overlap the bytes being produced).
//!
//! The codec itself is `compcol`'s LZF implementation (feature `lzf`).
//!
//! Client data (as written by h5py): `[filter version, LZF API version,
//! uncompressed chunk size]`. Decoding uses the chunk size only as a
//! pre-allocation hint. Encoding fails when the compressed form would not
//! be smaller than the input, so an optional LZF filter is skipped for
//! incompressible chunks.

use super::{Filter, FilterContext, LZF};
use crate::{Error, Result};

/// The LZF filter.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lzf;

impl Filter for Lzf {
    fn id(&self) -> u16 {
        LZF
    }

    fn name(&self) -> &str {
        "lzf"
    }

    fn decode(&self, client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        let hint = client_data
            .get(2)
            .and_then(|&n| usize::try_from(n).ok())
            .filter(|&n| n > 0);
        decompress(&input, ctx.capacity_hint(hint), ctx.max_output)
    }

    fn encode(&self, _client_data: &[u32], input: Vec<u8>, _: &FilterContext) -> Result<Vec<u8>> {
        compress(&input).ok_or_else(|| Error::unsupported("lzf: data is not compressible"))
    }
}

/// Decompresses a raw LZF stream.
///
/// `capacity` is a pre-allocation hint; producing more than `max_output`
/// bytes fails with [`Error::LimitExceeded`]. Truncated input and
/// back-references before the start of the output are errors.
pub fn decompress(input: &[u8], capacity: usize, max_output: usize) -> Result<Vec<u8>> {
    super::codec::decode_all::<compcol::lzf::Lzf>(input, capacity.max(1), max_output, "lzf")
}

/// Compresses `input` into a raw LZF stream.
///
/// Returns `None` when the result would not be strictly smaller than the
/// input (including for empty input).
pub fn compress(input: &[u8]) -> Option<Vec<u8>> {
    let out = compcol::vec::compress_to_vec::<compcol::lzf::Lzf>(input).ok()?;
    (out.len() < input.len()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX: usize = 1 << 20;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    fn hand_built_stream() {
        // Literal "abc", then back-ref length 6 distance 3 → "abcabcabc".
        let stream = [2, b'a', b'b', b'c', (4 << 5), 2];
        assert_eq!(decompress(&stream, 0, MAX).unwrap(), b"abcabcabc");
        // Long back-ref: run of 'x' with length 7 + 10 + 2 = 19, distance 1.
        let stream = [0, b'x', (7 << 5), 10, 0];
        assert_eq!(decompress(&stream, 0, MAX).unwrap(), vec![b'x'; 20]);
        // Distance with high bits: literal of 300 bytes, then copy from 300 back.
        let lit: Vec<u8> = (0..300u32).map(|i| i as u8 ^ 0x5a).collect();
        let mut stream = Vec::new();
        for run in lit.chunks(32) {
            stream.push(run.len() as u8 - 1);
            stream.extend_from_slice(run);
        }
        let off = 299usize;
        stream.extend_from_slice(&[(1 << 5) | (off >> 8) as u8, off as u8]);
        let mut want = lit.clone();
        want.extend_from_slice(&lit[..3]);
        assert_eq!(decompress(&stream, 0, MAX).unwrap(), want);
        assert_eq!(decompress(&[], 0, MAX).unwrap(), b"");
    }

    #[test]
    fn corrupt_streams() {
        // Truncated literal, missing offset byte, missing length byte.
        for s in [
            &[3, 1, 2][..],
            &[0, 1, 0x20],
            &[0, 1, 0xe0],
            &[0, 1, 0xe0, 3],
        ] {
            assert!(
                matches!(decompress(s, 0, MAX), Err(Error::InvalidFormat(_))),
                "{s:?}"
            );
        }
        // Back-reference before the start.
        assert!(decompress(&[0x20, 0], 0, MAX).is_err());
        assert!(decompress(&[0, 1, 0x20, 1], 0, MAX).is_err());
        // Output limit.
        assert!(matches!(
            decompress(&[0, b'x', 0xe0, 255, 0], 0, 100),
            Err(Error::LimitExceeded(_))
        ));
        assert!(matches!(
            decompress(&[3, 1, 2, 3, 4], 0, 3),
            Err(Error::LimitExceeded(_))
        ));
        assert_eq!(decompress(&[3, 1, 2, 3, 4], 0, 4).unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn roundtrip() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut cases: Vec<Vec<u8>> = vec![
            vec![0; 100_000],
            b"abcabcabcabcabcabcabcabcabcabcabcabcabc".to_vec(),
            (0..50_000u32).flat_map(|i| (i / 7).to_le_bytes()).collect(),
            b"the quick brown fox jumps over the lazy dog. ".repeat(500),
        ];
        // Low-entropy random data of various lengths.
        for len in [1000, 9000, 20_000] {
            cases.push((0..len).map(|_| b"aab"[rng.below(3)]).collect());
        }
        // Short inputs may or may not compress, but must round-trip if they do.
        for len in 0..300 {
            let d: Vec<u8> = (0..len).map(|_| b"aab"[rng.below(3)]).collect();
            if let Some(c) = compress(&d) {
                assert!(c.len() < d.len());
                assert_eq!(decompress(&c, 0, MAX).unwrap(), d);
            }
        }
        for d in cases {
            let c = compress(&d).expect("compressible");
            assert!(c.len() < d.len());
            assert_eq!(decompress(&c, d.len(), MAX).unwrap(), d);
            // Wrong / missing size hints do not matter.
            assert_eq!(decompress(&c, 0, MAX).unwrap(), d);
            assert_eq!(decompress(&c, 7, MAX).unwrap(), d);
        }
    }

    #[test]
    fn incompressible() {
        let mut rng = Rng(42);
        let d: Vec<u8> = (0..10_000).map(|_| rng.next() as u8).collect();
        assert!(compress(&d).is_none());
        assert!(compress(&[]).is_none());
        assert!(compress(b"ab").is_none());
        assert!(compress(b"aaaa").is_none()); // 4 bytes -> 4 bytes
        let ctx = FilterContext::default();
        assert!(Lzf.encode(&[], d.clone(), &ctx).is_err());
    }

    #[test]
    fn filter_client_data_hint() {
        let ctx = FilterContext::default().with_max_output(5000);
        let d = vec![7u8; 4000];
        let c = Lzf.encode(&[4, 0, 4000], d.clone(), &ctx).unwrap();
        assert_eq!(Lzf.decode(&[4, 0, 4000], c.clone(), &ctx).unwrap(), d);
        // Lying hints are harmless.
        assert_eq!(Lzf.decode(&[4, 0, u32::MAX], c.clone(), &ctx).unwrap(), d);
        assert_eq!(Lzf.decode(&[4, 0, 1], c.clone(), &ctx).unwrap(), d);
        assert_eq!(Lzf.decode(&[], c.clone(), &ctx).unwrap(), d);
        let small = FilterContext::default().with_max_output(3999);
        assert!(Lzf.decode(&[], c, &small).is_err());
    }

    #[test]
    fn mutation_never_panics() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let base: Vec<u8> = b"hello hello hello world, world, world! ".repeat(40);
        let good = compress(&base).unwrap();
        for _ in 0..20_000 {
            let mut m = good.clone();
            for _ in 0..1 + rng.below(4) {
                match rng.below(3) {
                    0 => {
                        let i = rng.below(m.len());
                        m[i] = rng.next() as u8;
                    }
                    1 => m.truncate(rng.below(m.len() + 1)),
                    _ => {
                        let i = rng.below(m.len() + 1);
                        m.insert(i, rng.next() as u8);
                    }
                }
                if m.is_empty() {
                    break;
                }
            }
            let _ = decompress(&m, rng.below(5000), 4096);
        }
        // Pure random streams.
        for _ in 0..5_000 {
            let len = rng.below(64);
            let m: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            if let Ok(out) = decompress(&m, 0, 4096) {
                assert!(out.len() <= 4096);
            }
        }
    }
}
