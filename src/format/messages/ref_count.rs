//! Object Reference Count message (type 0x0016, spec IV.A.2.w).

use crate::error::{Error, Result};
use crate::format::{ByteReader, ByteWriter, Sizes};

/// Hard-link reference count of an object with a version 2 header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefCountMessage {
    /// The reference count.
    pub ref_count: u32,
}

impl RefCountMessage {
    /// Decode the message body.
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "reference count message version {version}"
            )));
        }
        Ok(RefCountMessage {
            ref_count: r.read_u32()?,
        })
    }
}

impl RefCountMessage {
    /// Encode the message body (version 0).
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut w = ByteWriter::new();
        w.u8(0).u32(self.ref_count);
        Ok(w.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        let m = RefCountMessage::parse(&[0, 3, 0, 0, 0], Sizes::DEFAULT).unwrap();
        assert_eq!(m.ref_count, 3);
        assert!(RefCountMessage::parse(&[1, 3, 0, 0, 0], Sizes::DEFAULT).is_err());
        assert!(RefCountMessage::parse(&[0, 3, 0], Sizes::DEFAULT).is_err());
    }

    #[test]
    fn encode_round_trip() {
        let m = RefCountMessage { ref_count: 3 };
        assert_eq!(
            RefCountMessage::parse(&m.encode().unwrap(), Sizes::DEFAULT).unwrap(),
            m
        );
    }
}
