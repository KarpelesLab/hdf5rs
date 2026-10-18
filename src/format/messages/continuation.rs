//! Object Header Continuation message (type 0x0010, spec IV.A.2.q).

use crate::error::Result;
use crate::format::{ByteReader, Sizes};

/// Points at a further block of object header messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinuationMessage {
    /// Address of the continuation block.
    pub offset: u64,
    /// Length of the continuation block in bytes (for version 2 headers
    /// this includes the `OCHK` signature and the checksum).
    pub length: u64,
}

impl ContinuationMessage {
    /// Decode the message body.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        Ok(ContinuationMessage {
            offset: r.read_offset()?,
            length: r.read_length()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::W;

    #[test]
    fn parse() {
        let s = Sizes {
            offset: 4,
            length: 2,
        };
        let mut w = W::with_sizes(s);
        w.offset(1000).length(200);
        let m = ContinuationMessage::parse(&w.buf, s).unwrap();
        assert_eq!(m.offset, 1000);
        assert_eq!(m.length, 200);
        assert!(ContinuationMessage::parse(&w.buf[..5], s).is_err());
    }
}
