//! External data files message (type 0x0007).
//!
//! Spec: "IV.A.2.h. The Data Storage - External Data Files Message".

use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};

/// One slot of the external file list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalFileSlot {
    /// Offset of the file name in the local heap.
    pub name_offset: u64,
    /// Byte offset of the data within the external file.
    pub file_offset: u64,
    /// Number of bytes reserved in the external file.
    pub size: u64,
}

/// A decoded external data files message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalFiles {
    /// Address of the local heap holding the file names.
    pub heap_address: u64,
    /// Number of allocated slots (at least `slots.len()`).
    pub allocated_slots: u16,
    /// The used slots, in dataset address order.
    pub slots: Vec<ExternalFileSlot>,
}

impl ExternalFiles {
    /// Parse an external data files message.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<ExternalFiles> {
        let mut r = ByteReader::with_sizes(data, sizes);
        let version = r.read_u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!(
                "external data files version {version}"
            )));
        }
        r.skip(3)?;
        let allocated_slots = r.read_u16()?;
        let used = r.read_u16()?;
        if used > allocated_slots {
            return Err(Error::invalid("external file list uses unallocated slots"));
        }
        let heap_address = r.read_offset()?;
        let slot_size = 3 * sizes.length as usize;
        let mut slots = Vec::with_capacity((used as usize).min(r.remaining() / slot_size.max(1)));
        for _ in 0..used {
            slots.push(ExternalFileSlot {
                name_offset: r.read_length()?,
                file_offset: r.read_length()?,
                size: r.read_length()?,
            });
        }
        Ok(ExternalFiles {
            heap_address,
            allocated_slots,
            slots,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_two_slots() {
        let mut v = vec![1, 0, 0, 0, 2, 0, 2, 0];
        v.extend_from_slice(&0x500u64.to_le_bytes());
        for x in [8u64, 0, 400, 16, 400, 400] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        let e = ExternalFiles::parse(&v, Sizes::DEFAULT).unwrap();
        assert_eq!(e.heap_address, 0x500);
        assert_eq!(e.slots.len(), 2);
        assert_eq!(
            e.slots[1],
            ExternalFileSlot {
                name_offset: 16,
                file_offset: 400,
                size: 400
            }
        );
        assert!(ExternalFiles::parse(&v[..v.len() - 1], Sizes::DEFAULT).is_err());
    }

    #[test]
    fn rejects_bad_input() {
        assert!(ExternalFiles::parse(&[], Sizes::DEFAULT).is_err());
        assert!(ExternalFiles::parse(&[2, 0, 0, 0, 0, 0, 0, 0], Sizes::DEFAULT).is_err());
        let mut v = vec![1, 0, 0, 0, 1, 0, 2, 0];
        v.extend_from_slice(&[0; 8]);
        assert!(ExternalFiles::parse(&v, Sizes::DEFAULT).is_err());
        let mut v = vec![1, 0, 0, 0, 0xff, 0xff, 0xff, 0xff];
        v.extend_from_slice(&[0; 8]);
        assert!(ExternalFiles::parse(&v, Sizes::DEFAULT).is_err());
    }
}
