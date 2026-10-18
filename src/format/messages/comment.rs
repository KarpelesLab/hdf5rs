//! Object Comment message (type 0x000D, spec IV.A.2.n).

use super::until_nul;
use crate::error::Result;
use crate::format::Sizes;

/// A short free-text description of an object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommentMessage {
    /// The comment text, up to (excluding) the NUL terminator. The spec
    /// says ASCII; other bytes are replaced lossily.
    pub comment: String,
}

impl CommentMessage {
    /// Decode the message body. A missing terminator is tolerated.
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<Self> {
        Ok(CommentMessage {
            comment: String::from_utf8_lossy(until_nul(data)).into_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        let m = CommentMessage::parse(b"hello\0\0\0", Sizes::DEFAULT).unwrap();
        assert_eq!(m.comment, "hello");
        let m = CommentMessage::parse(b"", Sizes::DEFAULT).unwrap();
        assert_eq!(m.comment, "");
    }
}
