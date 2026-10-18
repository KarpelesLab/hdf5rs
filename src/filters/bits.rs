//! MSB-first bit streams shared by the N-bit and scale-offset filters.
//!
//! Both filters pack values into a contiguous bit stream starting at the
//! most significant bit of the first byte, without any per-value alignment.

use crate::{Error, Result};

/// Appends values to an MSB-first bit stream.
#[derive(Debug, Default)]
pub(crate) struct BitWriter {
    out: Vec<u8>,
    /// Pending bits, right-aligned (at most 7 between calls).
    acc: u64,
    nacc: u32,
}

impl BitWriter {
    pub(crate) fn with_capacity(bytes: usize) -> Self {
        BitWriter {
            out: Vec::with_capacity(bytes),
            acc: 0,
            nacc: 0,
        }
    }

    /// Appends the low `n` bits of `value` (`n <= 64`), most significant
    /// first.
    pub(crate) fn write(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 64);
        if n > 32 {
            self.write_small(value >> 32, n - 32);
            self.write_small(value & 0xffff_ffff, 32);
        } else {
            self.write_small(value, n);
        }
    }

    fn write_small(&mut self, value: u64, n: u32) {
        if n == 0 {
            return;
        }
        let mask = (1u64 << n) - 1;
        self.acc = (self.acc << n) | (value & mask);
        self.nacc += n;
        while self.nacc >= 8 {
            self.nacc -= 8;
            self.out.push((self.acc >> self.nacc) as u8);
        }
        self.acc &= (1u64 << self.nacc) - 1;
    }

    /// Total number of bits written so far.
    #[cfg(test)]
    fn bit_len(&self) -> u64 {
        self.out.len() as u64 * 8 + u64::from(self.nacc)
    }

    /// Flushes the stream. Like the reference implementation, the result
    /// always has `floor(bits / 8) + 1` bytes (one spare byte when the bit
    /// count is a multiple of 8).
    pub(crate) fn finish(mut self) -> Vec<u8> {
        let pad = 8 - self.nacc;
        self.write_small(0, pad);
        self.out
    }
}

/// Reads values from an MSB-first bit stream.
#[derive(Debug)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits.
    pos: u64,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// Reads `n` bits (`n <= 64`), most significant first.
    pub(crate) fn read(&mut self, n: u32) -> Result<u64> {
        debug_assert!(n <= 64);
        let end = self
            .pos
            .checked_add(u64::from(n))
            .filter(|&e| e <= self.data.len() as u64 * 8)
            .ok_or_else(|| Error::invalid("packed data is truncated"))?;
        let mut v = 0u64;
        let mut pos = self.pos;
        while pos < end {
            let byte = self.data[(pos / 8) as usize];
            let bit_in_byte = (pos % 8) as u32;
            let avail = 8 - bit_in_byte;
            let take = avail.min((end - pos) as u32);
            let bits = (u32::from(byte) >> (avail - take)) & ((1u32 << take) - 1);
            // `take <= 8`, and `v` holds fewer than `n - take` bits.
            v = (v << take) | u64::from(bits);
            pos += u64::from(take);
        }
        self.pos = end;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_padding() {
        let mut w = BitWriter::default();
        w.write(1, 12);
        w.write(0xfff, 12);
        w.write(0x5, 3);
        w.write(u64::MAX, 64);
        w.write(0x1234_5678_9abc, 48);
        assert_eq!(w.bit_len(), 139);
        let out = w.finish();
        assert_eq!(out.len(), 139 / 8 + 1);
        assert_eq!(&out[..3], &[0x00, 0x1f, 0xff]);
        let mut r = BitReader::new(&out);
        assert_eq!(r.read(12).unwrap(), 1);
        assert_eq!(r.read(12).unwrap(), 0xfff);
        assert_eq!(r.read(3).unwrap(), 5);
        assert_eq!(r.read(64).unwrap(), u64::MAX);
        assert_eq!(r.read(48).unwrap(), 0x1234_5678_9abc);
        assert_eq!(r.read(5).unwrap(), 0);
        assert!(r.read(1).is_err());
        // A multiple of 8 bits still gets a spare byte.
        let mut w = BitWriter::default();
        w.write(0xab, 8);
        assert_eq!(w.finish(), [0xab, 0]);
        assert_eq!(BitWriter::default().finish(), [0]);
    }
}
