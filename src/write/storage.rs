//! Output storage and space allocation for the writer.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::Sizes;

/// Positional output: the write-side counterpart of [`ReadAt`](crate::ReadAt).
///
/// The writer appends raw data and metadata at increasing addresses and
/// finally rewrites the superblock at offset 0, so implementations only
/// need to support writes at arbitrary positions (gaps, if any, read back
/// as zeros).
pub trait WriteAt {
    /// Write all of `data` starting at absolute position `pos`.
    fn write_all_at(&mut self, pos: u64, data: &[u8]) -> Result<()>;

    /// Called once when the file is complete; `len` is the final file size
    /// (the end-of-file address). Implementations should make the output
    /// exactly `len` bytes long when they can and flush buffers.
    fn finish(&mut self, len: u64) -> Result<()>;

    /// Whether [`read_exact_at`](Self::read_exact_at) works. Storage that
    /// can read back lets the writer evict partially written chunks from
    /// memory and rewrite stored chunks; without it they stay in memory
    /// until the file is finished. Defaults to `false`.
    fn supports_read_back(&self) -> bool {
        false
    }

    /// Read exactly `buf.len()` previously written bytes at `pos`. The
    /// default implementation reports [`Error::Unsupported`].
    fn read_exact_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<()> {
        let _ = (pos, buf);
        Err(Error::unsupported("reading back from this storage"))
    }
}

impl WriteAt for Vec<u8> {
    fn write_all_at(&mut self, pos: u64, data: &[u8]) -> Result<()> {
        let start =
            usize::try_from(pos).map_err(|_| Error::LimitExceeded("in-memory file size"))?;
        let end = start
            .checked_add(data.len())
            .ok_or(Error::LimitExceeded("in-memory file size"))?;
        if self.len() < end {
            self.resize(end, 0);
        }
        self[start..end].copy_from_slice(data);
        Ok(())
    }

    fn finish(&mut self, len: u64) -> Result<()> {
        let len = usize::try_from(len).map_err(|_| Error::LimitExceeded("in-memory file size"))?;
        self.resize(len, 0);
        Ok(())
    }

    fn supports_read_back(&self) -> bool {
        true
    }

    fn read_exact_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<()> {
        let src = usize::try_from(pos)
            .ok()
            .and_then(|p| Some(p..p.checked_add(buf.len())?))
            .and_then(|r| self.get(r))
            .ok_or(Error::UnexpectedEof)?;
        buf.copy_from_slice(src);
        Ok(())
    }
}

impl WriteAt for std::fs::File {
    fn write_all_at(&mut self, pos: u64, data: &[u8]) -> Result<()> {
        self.seek(SeekFrom::Start(pos))?;
        self.write_all(data)?;
        Ok(())
    }

    fn finish(&mut self, len: u64) -> Result<()> {
        self.set_len(len)?;
        self.flush()?;
        Ok(())
    }

    fn supports_read_back(&self) -> bool {
        true
    }

    fn read_exact_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<()> {
        self.seek(SeekFrom::Start(pos))?;
        self.read_exact(buf)?;
        Ok(())
    }
}

/// Adapter making any `Write + Seek` sink (e.g. a `BufWriter<File>` or a
/// `Cursor<Vec<u8>>`) usable as [`WriteAt`]. The output is not truncated
/// by [`WriteAt::finish`]; start from an empty sink.
#[derive(Debug)]
pub struct SeekWriter<W>(pub W);

impl<W: Write + Seek> WriteAt for SeekWriter<W> {
    fn write_all_at(&mut self, pos: u64, data: &[u8]) -> Result<()> {
        self.0.seek(SeekFrom::Start(pos))?;
        self.0.write_all(data)?;
        Ok(())
    }

    fn finish(&mut self, _len: u64) -> Result<()> {
        self.0.flush()?;
        Ok(())
    }
}

/// Append-only space allocator: hands out consecutive address ranges
/// starting after the superblock. Freed space is never reused (yet).
#[derive(Debug, Clone)]
pub(crate) struct Allocator {
    /// End of allocated space ("end of allocation").
    eoa: u64,
    /// Largest representable address + 1 for the file's offset width
    /// (the all-ones pattern is reserved for "undefined").
    limit: u64,
}

impl Allocator {
    pub(crate) fn new(start: u64, sizes: Sizes) -> Self {
        let limit = if sizes.offset >= 8 {
            u64::MAX
        } else {
            (1u64 << (u32::from(sizes.offset) * 8)) - 1
        };
        Allocator { eoa: start, limit }
    }

    /// Reserve `size` bytes, returning their address.
    pub(crate) fn alloc(&mut self, size: u64) -> Result<u64> {
        let addr = self.eoa;
        let end =
            addr.checked_add(size)
                .filter(|&e| e < self.limit)
                .ok_or(Error::LimitExceeded(
                    "file size exceeds the size of offsets",
                ))?;
        self.eoa = end;
        Ok(addr)
    }

    /// Current end of allocated space.
    pub(crate) fn eoa(&self) -> u64 {
        self.eoa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_storage() {
        let mut v = Vec::new();
        v.write_all_at(4, &[1, 2]).unwrap();
        v.write_all_at(0, &[9]).unwrap();
        assert_eq!(v, [9, 0, 0, 0, 1, 2]);
        v.finish(8).unwrap();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn seek_writer() {
        let mut w = SeekWriter(std::io::Cursor::new(Vec::new()));
        w.write_all_at(2, &[5]).unwrap();
        w.write_all_at(0, &[7]).unwrap();
        w.finish(3).unwrap();
        assert_eq!(w.0.into_inner(), [7, 0, 5]);
    }

    #[test]
    fn allocator_limits() {
        let mut a = Allocator::new(
            48,
            Sizes {
                offset: 2,
                length: 2,
            },
        );
        assert_eq!(a.alloc(10).unwrap(), 48);
        assert_eq!(a.eoa(), 58);
        assert!(a.alloc(70000).is_err());
        assert!(a.alloc(0xffff - 58).is_err());
        assert_eq!(a.alloc(0xffff - 59).unwrap(), 58);
    }
}
