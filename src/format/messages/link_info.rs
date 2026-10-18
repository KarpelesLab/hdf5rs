//! Link Info message (type 0x0002, spec IV.A.2.c).

use crate::error::{Error, Result};
use crate::format::{ByteReader, ByteWriter, Sizes, UNDEFINED_ADDRESS};

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

impl LinkInfoMessage {
    /// Link info for a group with compact storage (no dense structures).
    pub fn compact(creation_order_tracked: bool) -> Self {
        LinkInfoMessage {
            creation_order_tracked,
            creation_order_indexed: false,
            max_creation_index: creation_order_tracked.then_some(0),
            fractal_heap_address: UNDEFINED_ADDRESS,
            name_index_address: UNDEFINED_ADDRESS,
            creation_order_index_address: None,
        }
    }

    /// Encode the message body.
    pub fn encode(&self, sizes: Sizes) -> Result<Vec<u8>> {
        if self.creation_order_indexed && !self.creation_order_tracked {
            return Err(Error::invalid(
                "link creation order indexed but not tracked",
            ));
        }
        if self.creation_order_indexed != self.creation_order_index_address.is_some() {
            return Err(Error::invalid("creation order index address mismatch"));
        }
        let mut w = ByteWriter::with_sizes(sizes);
        w.u8(0)
            .u8(u8::from(self.creation_order_tracked) | u8::from(self.creation_order_indexed) << 1);
        if self.creation_order_tracked {
            w.u64(self.max_creation_index.unwrap_or(0));
        }
        w.write_offset(self.fractal_heap_address)?;
        w.write_offset(self.name_index_address)?;
        if let Some(a) = self.creation_order_index_address {
            w.write_offset(a)?;
        }
        Ok(w.into_vec())
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

    #[test]
    fn encode_round_trip() {
        let s4 = Sizes {
            offset: 4,
            length: 4,
        };
        let cases = [
            LinkInfoMessage::compact(false),
            LinkInfoMessage::compact(true),
            LinkInfoMessage {
                creation_order_tracked: true,
                creation_order_indexed: true,
                max_creation_index: Some(77),
                fractal_heap_address: 100,
                name_index_address: 200,
                creation_order_index_address: Some(300),
            },
        ];
        for sizes in [Sizes::DEFAULT, s4] {
            for m in &cases {
                let enc = m.encode(sizes).unwrap();
                assert_eq!(&LinkInfoMessage::parse(&enc, sizes).unwrap(), m);
            }
        }
        let mut bad = cases[2];
        bad.creation_order_index_address = None;
        assert!(bad.encode(s4).is_err());
    }
}
