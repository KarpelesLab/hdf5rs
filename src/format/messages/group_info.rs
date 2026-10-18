//! Group Info message (type 0x000A, spec IV.A.2.k).

use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};

/// Constant settings of a new-style group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupInfoMessage {
    /// Link phase change values `(max_compact, min_dense)`, if stored.
    pub link_phase_change: Option<(u16, u16)>,
    /// Estimated `(number of entries, link name length)`, if stored.
    pub estimated_entries: Option<(u16, u16)>,
}

impl GroupInfoMessage {
    /// Default estimated number of entries when not stored.
    pub const DEFAULT_ESTIMATED_ENTRIES: u16 = 4;
    /// Default estimated link name length when not stored.
    pub const DEFAULT_ESTIMATED_NAME_LENGTH: u16 = 8;

    /// Decode the message body.
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "group info message version {version}"
            )));
        }
        let flags = r.read_u8()?;
        if flags & !0x03 != 0 {
            return Err(Error::invalid("reserved group info flags set"));
        }
        let link_phase_change = if flags & 0x01 != 0 {
            Some((r.read_u16()?, r.read_u16()?))
        } else {
            None
        };
        let estimated_entries = if flags & 0x02 != 0 {
            Some((r.read_u16()?, r.read_u16()?))
        } else {
            None
        };
        Ok(GroupInfoMessage {
            link_phase_change,
            estimated_entries,
        })
    }

    /// Estimated number of entries (stored or default).
    pub fn estimated_num_entries(&self) -> u16 {
        self.estimated_entries
            .map_or(Self::DEFAULT_ESTIMATED_ENTRIES, |e| e.0)
    }

    /// Estimated link name length (stored or default).
    pub fn estimated_name_length(&self) -> u16 {
        self.estimated_entries
            .map_or(Self::DEFAULT_ESTIMATED_NAME_LENGTH, |e| e.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        let m = GroupInfoMessage::parse(&[0, 0], Sizes::DEFAULT).unwrap();
        assert_eq!(m.link_phase_change, None);
        assert_eq!(m.estimated_num_entries(), 4);
        assert_eq!(m.estimated_name_length(), 8);
        let d = [0, 3, 8, 0, 6, 0, 10, 0, 20, 0];
        let m = GroupInfoMessage::parse(&d, Sizes::DEFAULT).unwrap();
        assert_eq!(m.link_phase_change, Some((8, 6)));
        assert_eq!(m.estimated_entries, Some((10, 20)));
        assert!(GroupInfoMessage::parse(&d[..9], Sizes::DEFAULT).is_err());
        assert!(GroupInfoMessage::parse(&[1, 0], Sizes::DEFAULT).is_err());
        assert!(GroupInfoMessage::parse(&[0, 4], Sizes::DEFAULT).is_err());
    }
}
