//! The Fletcher-32 checksum filter (id 3).
//!
//! Encoding appends a 4-byte checksum of the chunk; decoding verifies and
//! strips it. The filter takes no client data.
//!
//! # Checksum definition
//!
//! Fletcher-32 (public definition) sums 16-bit words modulo 65535:
//! `s1 += w; s2 += s1`, and the result is `s2 << 16 | s1`.
//!
//! Details chosen for HDF5 (to be confirmed against real files):
//!
//! - Words are formed **big-endian** from consecutive byte pairs
//!   (`data[2k] << 8 | data[2k+1]`), independent of host byte order; an odd
//!   final byte forms a word on its own as its high byte (i.e. the data is
//!   conceptually zero-padded).
//! - Both sums start at `0xffff` and are reduced by end-around-carry folding
//!   (the classic optimised formulation). This is congruent to the plain
//!   modulo formulation but represents a sum ≡ 0 (mod 65535) as `0xffff`
//!   rather than `0`. Verification therefore compares each half modulo
//!   65535, accepting either representation.
//! - The 32-bit checksum is stored **little-endian** after the data. As a
//!   fallback, verification also accepts the byte-reversed (big-endian)
//!   encoding, which some older writers are believed to have produced.

use super::{FLETCHER32, Filter, FilterContext};
use crate::{Error, Result};

/// The Fletcher-32 checksum filter.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fletcher32;

impl Filter for Fletcher32 {
    fn id(&self) -> u16 {
        FLETCHER32
    }

    fn name(&self) -> &str {
        "fletcher32"
    }

    fn decode(
        &self,
        _client_data: &[u32],
        mut input: Vec<u8>,
        _: &FilterContext,
    ) -> Result<Vec<u8>> {
        let body = input
            .len()
            .checked_sub(4)
            .ok_or_else(|| Error::invalid("fletcher32: chunk shorter than its checksum"))?;
        let mut stored = [0u8; 4];
        stored.copy_from_slice(&input[body..]);
        let actual = checksum(&input[..body]);
        let le = u32::from_le_bytes(stored);
        let be = u32::from_be_bytes(stored);
        if !equivalent(le, actual) && !equivalent(be, actual) {
            return Err(Error::ChecksumMismatch {
                what: "fletcher32 filter",
                expected: le,
                actual,
            });
        }
        input.truncate(body);
        Ok(input)
    }

    fn encode(
        &self,
        _client_data: &[u32],
        mut input: Vec<u8>,
        _: &FilterContext,
    ) -> Result<Vec<u8>> {
        let sum = checksum(&input);
        input.extend_from_slice(&sum.to_le_bytes());
        Ok(input)
    }
}

/// Whether two checksums agree with each 16-bit half compared mod 65535.
fn equivalent(a: u32, b: u32) -> bool {
    let norm = |x: u32| ((x >> 16) % 0xffff, (x & 0xffff) % 0xffff);
    norm(a) == norm(b)
}

/// Computes the Fletcher-32 checksum of `data` as described in the module
/// documentation.
pub fn checksum(data: &[u8]) -> u32 {
    // With both sums at most 0x10000 at the start of a block, 360 words
    // keep the accumulators below 2^32 (s2 ≤ 0x10000 * 361 + 64980 * 0xffff).
    const BLOCK_WORDS: usize = 360;
    fn fold(x: u32) -> u32 {
        (x & 0xffff) + (x >> 16)
    }
    let (mut s1, mut s2) = (0xffffu32, 0xffffu32);
    let (words, rem) = data.as_chunks::<2>();
    for block in words.chunks(BLOCK_WORDS) {
        for pair in block {
            s1 += u32::from(u16::from_be_bytes(*pair));
            s2 += s1;
        }
        s1 = fold(fold(s1));
        s2 = fold(fold(s2));
    }
    if let [last] = rem {
        s1 += u32::from(*last) << 8;
        s2 += s1;
    }
    s1 = fold(fold(s1));
    s2 = fold(fold(s2));
    (s2 << 16) | s1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward reference: modulo after every word.
    fn naive(data: &[u8]) -> u32 {
        let (mut s1, mut s2) = (0u64, 0u64);
        for pair in data.chunks(2) {
            let w = u64::from(pair[0]) << 8 | pair.get(1).map_or(0, |&b| u64::from(b));
            s1 = (s1 + w) % 65535;
            s2 = (s2 + s1) % 65535;
        }
        ((s2 as u32) << 16) | s1 as u32
    }

    /// Swaps each byte pair so big-endian words read like little-endian ones.
    fn swap_pairs(d: &[u8]) -> Vec<u8> {
        d.chunks(2)
            .flat_map(|p| {
                if p.len() == 2 {
                    vec![p[1], p[0]]
                } else {
                    vec![p[0]]
                }
            })
            .collect()
    }

    #[test]
    fn public_vectors() {
        // Widely published Fletcher-32 values (little-endian 16-bit words).
        // With pairs swapped they exercise our big-endian word formation.
        assert_eq!(checksum(&swap_pairs(b"abcdef")), 0x5650_2D2A);
        assert_eq!(checksum(&swap_pairs(b"abcdefgh")), 0xEBE1_9591);
    }

    #[test]
    fn matches_naive_up_to_representation() {
        let mut x = 0x1234_5678u32;
        for len in (0..50).chain([719, 720, 721, 1440, 5000, 70_000]) {
            let d: Vec<u8> = (0..len)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    x as u8
                })
                .collect();
            assert!(equivalent(checksum(&d), naive(&d)), "len {len}");
        }
        // All-0xff data drives the sums to large values without overflow.
        let d = vec![0xffu8; 100_000];
        assert!(equivalent(checksum(&d), naive(&d)));
        // Zero data: folded representation.
        assert_eq!(checksum(&[0; 8]), 0xffff_ffff);
        assert_eq!(checksum(&[]), 0xffff_ffff);
        assert_eq!(checksum(&[1]), naive(&[1]));
        assert_eq!(checksum(&[1]), 0x0100_0100);
    }

    #[test]
    fn roundtrip_and_corruption() {
        let ctx = FilterContext::default();
        let data: Vec<u8> = (0..1001u32).map(|i| (i * 31) as u8).collect();
        let enc = Fletcher32.encode(&[], data.clone(), &ctx).unwrap();
        assert_eq!(enc.len(), data.len() + 4);
        assert_eq!(Fletcher32.decode(&[], enc.clone(), &ctx).unwrap(), data);
        for i in [0, 500, 1000, 1002] {
            let mut bad = enc.clone();
            bad[i] ^= 0x10;
            assert!(matches!(
                Fletcher32.decode(&[], bad, &ctx),
                Err(Error::ChecksumMismatch { .. })
            ));
        }
        assert!(Fletcher32.decode(&[], vec![1, 2, 3], &ctx).is_err());
        assert_eq!(Fletcher32.decode(&[], vec![0xff; 4], &ctx).unwrap(), []);
    }

    #[test]
    fn accepts_alternate_encodings() {
        let ctx = FilterContext::default();
        let data = vec![0u8; 16];
        // Zero sums stored as 0 instead of 0xffff.
        let mut d = data.clone();
        d.extend_from_slice(&[0, 0, 0, 0]);
        assert_eq!(Fletcher32.decode(&[], d, &ctx).unwrap(), data);
        // Big-endian stored checksum.
        let data = b"hello world".to_vec();
        let mut d = data.clone();
        d.extend_from_slice(&checksum(&data).to_be_bytes());
        assert_eq!(Fletcher32.decode(&[], d, &ctx).unwrap(), data);
    }
}
