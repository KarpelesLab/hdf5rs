//! Storage abstraction: positional reads over files, buffers or custom
//! backends.

use crate::error::{Error, Result};

/// Positional, shared-reference read access to a byte store.
///
/// Implementations must be safe to call from `&self`; parsers never rely on
/// an implicit cursor.
pub trait ReadAt {
    /// Fill `buf` entirely with the bytes starting at `pos`.
    ///
    /// Returns [`Error::UnexpectedEof`] if the store ends first.
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> Result<()>;

    /// Total size of the store in bytes.
    fn size(&self) -> Result<u64>;

    /// Read `len` bytes starting at `pos` into a new vector.
    fn read_vec_at(&self, pos: u64, len: usize) -> Result<Vec<u8>> {
        let end = pos.checked_add(len as u64).ok_or(Error::UnexpectedEof)?;
        if end > self.size()? {
            return Err(Error::UnexpectedEof);
        }
        let mut buf = vec![0; len];
        self.read_exact_at(pos, &mut buf)?;
        Ok(buf)
    }
}

impl ReadAt for [u8] {
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        let start = usize::try_from(pos).map_err(|_| Error::UnexpectedEof)?;
        let end = start.checked_add(buf.len()).ok_or(Error::UnexpectedEof)?;
        let src = self.get(start..end).ok_or(Error::UnexpectedEof)?;
        buf.copy_from_slice(src);
        Ok(())
    }

    fn size(&self) -> Result<u64> {
        Ok(self.len() as u64)
    }
}

impl ReadAt for Vec<u8> {
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        self.as_slice().read_exact_at(pos, buf)
    }

    fn size(&self) -> Result<u64> {
        Ok(self.len() as u64)
    }
}

impl<T: ReadAt + ?Sized> ReadAt for &T {
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_exact_at(pos, buf)
    }

    fn size(&self) -> Result<u64> {
        (**self).size()
    }
}

impl<T: ReadAt + ?Sized> ReadAt for Box<T> {
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_exact_at(pos, buf)
    }

    fn size(&self) -> Result<u64> {
        (**self).size()
    }
}

impl<T: ReadAt + ?Sized> ReadAt for std::sync::Arc<T> {
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_exact_at(pos, buf)
    }

    fn size(&self) -> Result<u64> {
        (**self).size()
    }
}

impl ReadAt for std::fs::File {
    #[cfg(unix)]
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, pos)?;
        Ok(())
    }

    #[cfg(windows)]
    fn read_exact_at(&self, mut pos: u64, mut buf: &mut [u8]) -> Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            match self.seek_read(buf, pos) {
                Ok(0) => return Err(Error::UnexpectedEof),
                Ok(n) => {
                    buf = &mut buf[n..];
                    pos += n as u64;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    #[cfg(not(any(unix, windows)))]
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = self;
        f.seek(SeekFrom::Start(pos))?;
        f.read_exact(buf)?;
        Ok(())
    }

    fn size(&self) -> Result<u64> {
        Ok(self.metadata()?.len())
    }
}

/// A view of another store shifted by a fixed base offset.
///
/// HDF5 addresses are relative to the superblock's base address; wrapping
/// the underlying file in an `OffsetReader` lets parsers use file addresses
/// directly. Position `p` of the view maps to `base + p` of the inner store.
#[derive(Debug, Clone)]
pub struct OffsetReader<R> {
    inner: R,
    base: u64,
}

impl<R: ReadAt> OffsetReader<R> {
    /// Create a view of `inner` starting at absolute offset `base`.
    pub fn new(inner: R, base: u64) -> Self {
        OffsetReader { inner, base }
    }

    /// The absolute offset of position 0 of this view.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// The wrapped store.
    pub fn inner(&self) -> &R {
        &self.inner
    }

    /// Unwrap, returning the inner store.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: ReadAt> ReadAt for OffsetReader<R> {
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        let abs = self.base.checked_add(pos).ok_or(Error::UnexpectedEof)?;
        self.inner.read_exact_at(abs, buf)
    }

    fn size(&self) -> Result<u64> {
        Ok(self.inner.size()?.saturating_sub(self.base))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_reads() {
        let data = [1u8, 2, 3, 4, 5];
        let mut buf = [0; 2];
        data[..].read_exact_at(3, &mut buf).unwrap();
        assert_eq!(buf, [4, 5]);
        assert!(matches!(
            data[..].read_exact_at(4, &mut buf),
            Err(Error::UnexpectedEof)
        ));
        assert!(matches!(
            data[..].read_exact_at(u64::MAX, &mut buf),
            Err(Error::UnexpectedEof)
        ));
        assert!(data[..].read_vec_at(1, 100).is_err());
    }

    #[test]
    fn offset_reader() {
        let data = [1u8, 2, 3, 4, 5];
        let r = OffsetReader::new(&data[..], 2);
        assert_eq!(r.size().unwrap(), 3);
        assert_eq!(r.read_vec_at(0, 3).unwrap(), vec![3, 4, 5]);
        assert!(r.read_vec_at(1, 3).is_err());
        assert!(r.read_vec_at(u64::MAX, 1).is_err());
        let r = OffsetReader::new(&data[..], 10);
        assert_eq!(r.size().unwrap(), 0);
    }
}
