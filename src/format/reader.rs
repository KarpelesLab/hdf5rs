use super::{Sizes, UNDEFINED_ADDRESS};
use crate::error::{Error, Result};

/// A bounds-checked little-endian cursor over an in-memory byte slice.
///
/// All HDF5 metadata is little-endian. Every read returns
/// [`Error::UnexpectedEof`] instead of panicking when data runs out.
#[derive(Debug, Clone)]
pub struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
    sizes: Sizes,
}

impl<'a> ByteReader<'a> {
    /// Create a reader using [`Sizes::DEFAULT`].
    pub fn new(data: &'a [u8]) -> Self {
        Self::with_sizes(data, Sizes::DEFAULT)
    }

    /// Create a reader with explicit offset/length widths.
    pub fn with_sizes(data: &'a [u8], sizes: Sizes) -> Self {
        ByteReader {
            data,
            pos: 0,
            sizes,
        }
    }

    /// The offset/length widths used by [`read_offset`](Self::read_offset)
    /// and [`read_length`](Self::read_length).
    pub fn sizes(&self) -> Sizes {
        self.sizes
    }

    /// Current position within the underlying slice.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Move to an absolute position (may be at most the slice length).
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        self.pos = pos;
        Ok(())
    }

    /// Number of unread bytes.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// Whether all bytes have been consumed.
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// The whole underlying slice.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The unread part of the slice, without consuming it.
    pub fn rest(&self) -> &'a [u8] {
        &self.data[self.pos..]
    }

    /// Consume and return the next `n` bytes.
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::UnexpectedEof)?;
        let out = self.data.get(self.pos..end).ok_or(Error::UnexpectedEof)?;
        self.pos = end;
        Ok(out)
    }

    /// Consume exactly `N` bytes into an array.
    pub fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0; N];
        out.copy_from_slice(self.read_bytes(N)?);
        Ok(out)
    }

    /// Skip `n` bytes.
    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.read_bytes(n).map(|_| ())
    }

    /// Read one byte.
    pub fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_array::<1>()?[0])
    }

    /// Read a little-endian `u16`.
    pub fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    /// Read a little-endian `u32`.
    pub fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    /// Read a little-endian `u64`.
    pub fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    /// Read a little-endian unsigned integer of `width` bytes (0..=8).
    ///
    /// Widths up to 16 are accepted if the high bytes are zero.
    pub fn read_uint(&mut self, width: usize) -> Result<u64> {
        let bytes = self.read_bytes(width)?;
        let (low, high) = bytes.split_at(width.min(8));
        if high.iter().any(|&b| b != 0) {
            return Err(Error::LimitExceeded("integer wider than 64 bits"));
        }
        let mut buf = [0u8; 8];
        buf[..low.len()].copy_from_slice(low);
        Ok(u64::from_le_bytes(buf))
    }

    /// Read a file address of the configured offset width.
    ///
    /// The all-ones "undefined address" maps to [`UNDEFINED_ADDRESS`].
    pub fn read_offset(&mut self) -> Result<u64> {
        let width = self.sizes.offset as usize;
        let bytes = self.read_bytes(width)?;
        if bytes.iter().all(|&b| b == 0xff) {
            return Ok(UNDEFINED_ADDRESS);
        }
        ByteReader::new(bytes).read_uint(width)
    }

    /// Read a length of the configured length width.
    pub fn read_length(&mut self) -> Result<u64> {
        self.read_uint(self.sizes.length as usize)
    }

    /// Check for an expected 4-byte signature, e.g. `b"OHDR"`.
    pub fn expect_signature(&mut self, sig: &[u8; 4]) -> Result<()> {
        if &self.read_array::<4>()? != sig {
            return Err(Error::invalid(format!(
                "expected signature {:?}",
                String::from_utf8_lossy(sig)
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_little_endian() {
        let data = [1, 0, 2, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 5];
        let mut r = ByteReader::with_sizes(
            &data,
            Sizes {
                offset: 4,
                length: 2,
            },
        );
        assert_eq!(r.read_length().unwrap(), 1);
        assert_eq!(r.read_u32().unwrap(), 2);
        assert_eq!(r.read_offset().unwrap(), UNDEFINED_ADDRESS);
        assert_eq!(r.read_u8().unwrap(), 5);
        assert!(r.is_empty());
        assert!(matches!(r.read_u8(), Err(Error::UnexpectedEof)));
    }

    #[test]
    fn wide_uint() {
        let mut data = [0u8; 16];
        data[0] = 7;
        assert_eq!(ByteReader::new(&data).read_uint(16).unwrap(), 7);
        data[9] = 1;
        assert!(ByteReader::new(&data).read_uint(16).is_err());
    }
}
