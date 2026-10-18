use super::{Sizes, UNDEFINED_ADDRESS};
use crate::error::{Error, Result};

/// A little-endian byte sink, the encoding counterpart of
/// [`ByteReader`](super::ByteReader).
///
/// Fixed-width writers are infallible; variable-width ones
/// ([`write_uint`](Self::write_uint), [`write_offset`](Self::write_offset),
/// [`write_length`](Self::write_length)) fail with
/// [`Error::LimitExceeded`] when the value does not fit the width.
#[derive(Debug, Clone, Default)]
pub struct ByteWriter {
    buf: Vec<u8>,
    sizes: Sizes,
}

impl ByteWriter {
    /// An empty writer using [`Sizes::DEFAULT`].
    pub fn new() -> Self {
        Self::with_sizes(Sizes::DEFAULT)
    }

    /// An empty writer with explicit offset/length widths.
    pub fn with_sizes(sizes: Sizes) -> Self {
        ByteWriter {
            buf: Vec::new(),
            sizes,
        }
    }

    /// The offset/length widths in use.
    pub fn sizes(&self) -> Sizes {
        self.sizes
    }

    /// Number of bytes written so far.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether nothing has been written yet.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// The bytes written so far.
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    /// Mutable access to the bytes written so far (for patching fields).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    /// Consume the writer, returning the encoded bytes.
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    /// Append one byte.
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }

    /// Append a little-endian `u16`.
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// Append a little-endian `u32`.
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// Append a little-endian `u64`.
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// Append raw bytes.
    pub fn bytes(&mut self, v: &[u8]) -> &mut Self {
        self.buf.extend_from_slice(v);
        self
    }

    /// Append `n` zero bytes.
    pub fn zeros(&mut self, n: usize) -> &mut Self {
        self.buf.resize(self.buf.len().saturating_add(n), 0);
        self
    }

    /// Pad with zeros up to a multiple of `align` bytes (measured from the
    /// start of the buffer).
    pub fn align(&mut self, align: usize) -> &mut Self {
        let target = self.buf.len().next_multiple_of(align.max(1));
        self.buf.resize(target, 0);
        self
    }

    /// Append an unsigned integer in `width` bytes (0..=16).
    pub fn write_uint(&mut self, v: u64, width: usize) -> Result<&mut Self> {
        if width < 8 && v >> (width * 8) != 0 {
            return Err(Error::LimitExceeded("value does not fit its encoded width"));
        }
        if width > 16 {
            return Err(Error::invalid("integer field wider than 16 bytes"));
        }
        let b = v.to_le_bytes();
        for i in 0..width {
            self.buf.push(b.get(i).copied().unwrap_or(0));
        }
        Ok(self)
    }

    /// Append a file address in the configured offset width;
    /// [`UNDEFINED_ADDRESS`] is written as all ones.
    pub fn write_offset(&mut self, v: u64) -> Result<&mut Self> {
        let width = self.sizes.offset as usize;
        if v == UNDEFINED_ADDRESS {
            self.buf.resize(self.buf.len() + width, 0xff);
            return Ok(self);
        }
        if width < 8 && v == (1u64 << (width * 8)) - 1 {
            // Would collide with the undefined-address pattern.
            return Err(Error::LimitExceeded("address does not fit size of offsets"));
        }
        self.write_uint(v, width)
    }

    /// Append a length in the configured length width.
    pub fn write_length(&mut self, v: u64) -> Result<&mut Self> {
        self.write_uint(v, self.sizes.length as usize)
    }

    /// Append a length-width value where all ones means "unlimited"
    /// (`u64::MAX`), as used by dataspace maximum dimensions.
    pub fn write_length_or_unlimited(&mut self, v: u64) -> Result<&mut Self> {
        if v == u64::MAX {
            let width = self.sizes.length as usize;
            self.buf.resize(self.buf.len() + width, 0xff);
            return Ok(self);
        }
        self.write_length(v)
    }

    /// Overwrite bytes at `pos` (which must already have been written).
    pub fn patch(&mut self, pos: usize, bytes: &[u8]) -> Result<()> {
        let end = pos.checked_add(bytes.len()).ok_or(Error::UnexpectedEof)?;
        self.buf
            .get_mut(pos..end)
            .ok_or(Error::UnexpectedEof)?
            .copy_from_slice(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::ByteReader;

    #[test]
    fn round_trip_fields() {
        let sizes = Sizes {
            offset: 4,
            length: 2,
        };
        let mut w = ByteWriter::with_sizes(sizes);
        w.u8(1).u16(2).u32(3).u64(4);
        w.write_offset(UNDEFINED_ADDRESS).unwrap();
        w.write_offset(77).unwrap();
        w.write_length(300).unwrap();
        w.write_length_or_unlimited(u64::MAX).unwrap();
        assert!(w.write_length(70000).is_err());
        assert!(w.write_offset(0xffff_ffff).is_err());
        let data = w.into_vec();
        let mut r = ByteReader::with_sizes(&data, sizes);
        assert_eq!(r.read_u8().unwrap(), 1);
        assert_eq!(r.read_u16().unwrap(), 2);
        assert_eq!(r.read_u32().unwrap(), 3);
        assert_eq!(r.read_u64().unwrap(), 4);
        assert_eq!(r.read_offset().unwrap(), UNDEFINED_ADDRESS);
        assert_eq!(r.read_offset().unwrap(), 77);
        assert_eq!(r.read_length().unwrap(), 300);
        assert_eq!(r.read_u16().unwrap(), 0xffff);
        assert!(r.is_empty());
    }

    #[test]
    fn align_and_patch() {
        let mut w = ByteWriter::new();
        w.u8(1).align(8);
        assert_eq!(w.len(), 8);
        w.patch(2, &[9, 9]).unwrap();
        assert!(w.patch(7, &[1, 2]).is_err());
        assert_eq!(w.as_slice(), &[1, 0, 9, 9, 0, 0, 0, 0]);
    }
}
