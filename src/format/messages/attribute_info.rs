//! Attribute info message (type 0x0015).
//!
//! Spec: "IV.A.2.v. The Attribute Info Message", version 0.

use crate::error::{Error, Result};
use crate::format::{ByteReader, ByteWriter, Sizes};

/// A decoded attribute info message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttributeInfo {
    /// Maximum creation index, when creation order is tracked.
    pub max_creation_index: Option<u16>,
    /// Address of the fractal heap holding dense attributes (may be
    /// undefined when attributes are stored compactly).
    pub fractal_heap_address: u64,
    /// Address of the v2 B-tree indexing attribute names.
    pub name_btree_address: u64,
    /// Address of the v2 B-tree indexing creation order, when indexed.
    pub creation_order_btree_address: Option<u64>,
}

impl AttributeInfo {
    /// Parse an attribute info message.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<AttributeInfo> {
        let mut r = ByteReader::with_sizes(data, sizes);
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "attribute info version {version}"
            )));
        }
        let flags = r.read_u8()?;
        if flags & !0x03 != 0 {
            return Err(Error::invalid("reserved attribute info flags set"));
        }
        let max_creation_index = if flags & 1 != 0 {
            Some(r.read_u16()?)
        } else {
            None
        };
        let fractal_heap_address = r.read_offset()?;
        let name_btree_address = r.read_offset()?;
        let creation_order_btree_address = if flags & 2 != 0 {
            Some(r.read_offset()?)
        } else {
            None
        };
        Ok(AttributeInfo {
            max_creation_index,
            fractal_heap_address,
            name_btree_address,
            creation_order_btree_address,
        })
    }
}

impl AttributeInfo {
    /// Attribute info for compact attribute storage.
    pub fn compact(creation_order_tracked: bool) -> AttributeInfo {
        AttributeInfo {
            max_creation_index: creation_order_tracked.then_some(0),
            fractal_heap_address: crate::format::UNDEFINED_ADDRESS,
            name_btree_address: crate::format::UNDEFINED_ADDRESS,
            creation_order_btree_address: None,
        }
    }

    /// Encode the message body.
    pub fn encode(&self, sizes: Sizes) -> Result<Vec<u8>> {
        let mut w = ByteWriter::with_sizes(sizes);
        w.u8(0).u8(u8::from(self.max_creation_index.is_some())
            | u8::from(self.creation_order_btree_address.is_some()) << 1);
        if let Some(m) = self.max_creation_index {
            w.u16(m);
        }
        w.write_offset(self.fractal_heap_address)?;
        w.write_offset(self.name_btree_address)?;
        if let Some(a) = self.creation_order_btree_address {
            w.write_offset(a)?;
        }
        Ok(w.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::UNDEFINED_ADDRESS;

    #[test]
    fn parse_variants() {
        let mut v = vec![0, 0];
        v.extend_from_slice(&[0xff; 16]);
        let a = AttributeInfo::parse(&v, Sizes::DEFAULT).unwrap();
        assert_eq!(a.max_creation_index, None);
        assert_eq!(a.fractal_heap_address, UNDEFINED_ADDRESS);
        assert_eq!(a.creation_order_btree_address, None);

        let s4 = Sizes {
            offset: 4,
            length: 4,
        };
        let mut v = vec![0, 3, 5, 0];
        for x in [0x100u32, 0x200, 0x300] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        let a = AttributeInfo::parse(&v, s4).unwrap();
        assert_eq!(a.max_creation_index, Some(5));
        assert_eq!(a.fractal_heap_address, 0x100);
        assert_eq!(a.name_btree_address, 0x200);
        assert_eq!(a.creation_order_btree_address, Some(0x300));
        assert!(AttributeInfo::parse(&v[..v.len() - 1], s4).is_err());
    }

    #[test]
    fn rejects_bad_input() {
        assert!(AttributeInfo::parse(&[], Sizes::DEFAULT).is_err());
        assert!(AttributeInfo::parse(&[1, 0], Sizes::DEFAULT).is_err());
        assert!(AttributeInfo::parse(&[0, 4], Sizes::DEFAULT).is_err());
    }

    #[test]
    fn encode_round_trip() {
        let cases = [
            AttributeInfo::compact(false),
            AttributeInfo::compact(true),
            AttributeInfo {
                max_creation_index: Some(3),
                fractal_heap_address: 10,
                name_btree_address: 20,
                creation_order_btree_address: Some(30),
            },
        ];
        for m in &cases {
            let enc = m.encode(Sizes::DEFAULT).unwrap();
            assert_eq!(&AttributeInfo::parse(&enc, Sizes::DEFAULT).unwrap(), m);
        }
    }
}
