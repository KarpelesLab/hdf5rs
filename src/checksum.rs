//! Checksums used by the HDF5 format.
//!
//! Version 2 metadata structures are protected by Bob Jenkins' `lookup3`
//! hash (`hashlittle`), a public-domain algorithm.

/// Whether stored checksums are verified when reading.
///
/// Always `true`, except in builds with `--cfg fuzzing` (set automatically
/// by `cargo fuzz`, and used by the mutation stress test) where checksum
/// mismatches are ignored so that mutated inputs reach the deeper parsers.
pub(crate) const VERIFY: bool = !cfg!(fuzzing);

#[inline]
fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(4);
    *b = b.wrapping_add(*a);
}

#[inline]
fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(14));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(11));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(25));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(16));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(4));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(14));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(24));
}

#[inline]
fn le_word(bytes: &[u8]) -> u32 {
    // Zero-pads short tails, matching the byte-wise tail handling of lookup3.
    let mut w = [0u8; 4];
    w[..bytes.len()].copy_from_slice(bytes);
    u32::from_le_bytes(w)
}

/// Jenkins `lookup3` `hashlittle` over `data` with the given initial value.
pub fn lookup3(data: &[u8], init: u32) -> u32 {
    let base = 0xdead_beef_u32
        .wrapping_add(data.len() as u32)
        .wrapping_add(init);
    let (mut a, mut b, mut c) = (base, base, base);

    let mut rest = data;
    while rest.len() > 12 {
        a = a.wrapping_add(le_word(&rest[0..4]));
        b = b.wrapping_add(le_word(&rest[4..8]));
        c = c.wrapping_add(le_word(&rest[8..12]));
        mix(&mut a, &mut b, &mut c);
        rest = &rest[12..];
    }

    if rest.is_empty() {
        return c;
    }
    a = a.wrapping_add(le_word(&rest[..rest.len().min(4)]));
    if rest.len() > 4 {
        b = b.wrapping_add(le_word(&rest[4..rest.len().min(8)]));
    }
    if rest.len() > 8 {
        c = c.wrapping_add(le_word(&rest[8..]));
    }
    final_mix(&mut a, &mut b, &mut c);
    c
}

/// Checksum of HDF5 metadata: `lookup3` with an initial value of 0.
pub fn metadata_checksum(data: &[u8]) -> u32 {
    lookup3(data, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup3_known_vectors() {
        assert_eq!(lookup3(b"", 0), 0xdeadbeef);
        assert_eq!(lookup3(b"Four score and seven years ago", 0), 0x17770551);
        assert_eq!(lookup3(b"Four score and seven years ago", 1), 0xcd628161);
    }
}
