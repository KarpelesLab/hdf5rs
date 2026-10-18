//! B-tree 'K' Values message (type 0x0013, spec IV.A.2.t).

use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};

/// Non-default v1 B-tree 'K' values (superblock extension only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BtreeKMessage {
    /// Indexed storage internal node K.
    pub indexed_storage_internal_k: u16,
    /// Group internal node K.
    pub group_internal_k: u16,
    /// Group leaf node K.
    pub group_leaf_k: u16,
}

impl BtreeKMessage {
    /// Decode the message body.
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "B-tree 'K' values message version {version}"
            )));
        }
        let m = BtreeKMessage {
            indexed_storage_internal_k: r.read_u16()?,
            group_internal_k: r.read_u16()?,
            group_leaf_k: r.read_u16()?,
        };
        if m.indexed_storage_internal_k == 0 || m.group_internal_k == 0 || m.group_leaf_k == 0 {
            return Err(Error::invalid("B-tree K values must be non-zero"));
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        let d = [0, 32, 0, 16, 0, 4, 0];
        let m = BtreeKMessage::parse(&d, Sizes::DEFAULT).unwrap();
        assert_eq!(m.indexed_storage_internal_k, 32);
        assert_eq!(m.group_internal_k, 16);
        assert_eq!(m.group_leaf_k, 4);
        assert!(BtreeKMessage::parse(&d[..6], Sizes::DEFAULT).is_err());
        assert!(BtreeKMessage::parse(&[0, 0, 0, 16, 0, 4, 0], Sizes::DEFAULT).is_err());
    }
}
