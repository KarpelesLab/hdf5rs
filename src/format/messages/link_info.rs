//! Link Info message (type 0x0002, spec IV.A.2.c).

use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes, UNDEFINED_ADDRESS};

/// Variable state of a new-style group's links.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkInfoMessage {
    /// Whether link creation order is tracked (flag bit 0).
    pub creation_order_tracked: bool,
    /// Whether link creation order is indexed (flag bit 1).
    pub creation_order_indexed: bool,
    /// Maximum creation index, present when creation order is tracked.
    pub max_creation_index: Option<u64>,
    /// Fractal heap address for dense link storage (undefined if compact).
    pub fractal_heap_address: u64,
    /// v2 B-tree address of the name index (undefined if compact).
    pub name_index_address: u64,
    /// v2 B-tree address of the creation order index, when indexed.
    pub creation_order_index_address: Option<u64>,
}

impl LinkInfoMessage {
    /// Decode the message body.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "link info message version {version}"
            )));
        }
        let flags = r.read_u8()?;
        if flags & !0x03 != 0 {
            return Err(Error::invalid("reserved link info flags set"));
        }
        let tracked = flags & 0x01 != 0;
        let indexed = flags & 0x02 != 0;
        let max_creation_index = if tracked { Some(r.read_u64()?) } else { None };
        let fractal_heap_address = r.read_offset()?;
        let name_index_address = r.read_offset()?;
        let creation_order_index_address = if indexed {
            Some(r.read_offset()?)
        } else {
            None
        };
        Ok(LinkInfoMessage {
            creation_order_tracked: tracked,
            creation_order_indexed: indexed,
            max_creation_index,
            fractal_heap_address,
            name_index_address,
            creation_order_index_address,
        })
    }

    /// Whether links are stored densely (in a fractal heap) rather than as
    /// Link messages in the object header.
    pub fn is_dense(&self) -> bool {
        self.fractal_heap_address != UNDEFINED_ADDRESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::W;

    #[test]
    fn parse_compact() {
        let mut w = W::new();
        w.u8(0).u8(0).offset(u64::MAX).offset(u64::MAX);
        let m = LinkInfoMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert!(!m.is_dense());
        assert_eq!(m.max_creation_index, None);
        assert_eq!(m.creation_order_index_address, None);
    }

    #[test]
    fn parse_dense_indexed() {
        let mut w = W::new();
        w.u8(0).u8(3).u64(17).offset(100).offset(200).offset(300);
        let m = LinkInfoMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert!(m.is_dense());
        assert_eq!(m.max_creation_index, Some(17));
        assert_eq!(m.name_index_address, 200);
        assert_eq!(m.creation_order_index_address, Some(300));
        assert!(LinkInfoMessage::parse(&w.buf[..w.len() - 1], Sizes::DEFAULT).is_err());
    }
}
