//! Group Info message (type 0x000A, spec IV.A.2.k).

use crate::error::{Error, Result};
use crate::format::{ByteReader, ByteWriter, Sizes};

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

impl GroupInfoMessage {
    /// Encode the message body.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut w = ByteWriter::new();
        w.u8(0).u8(u8::from(self.link_phase_change.is_some())
            | u8::from(self.estimated_entries.is_some()) << 1);
        if let Some((max_compact, min_dense)) = self.link_phase_change {
            w.u16(max_compact).u16(min_dense);
        }
        if let Some((entries, name_len)) = self.estimated_entries {
            w.u16(entries).u16(name_len);
        }
        Ok(w.into_vec())
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

    #[test]
    fn encode_round_trip() {
        for m in [
            GroupInfoMessage {
                link_phase_change: None,
                estimated_entries: None,
            },
            GroupInfoMessage {
                link_phase_change: Some((100, 50)),
                estimated_entries: None,
            },
            GroupInfoMessage {
                link_phase_change: Some((8, 6)),
                estimated_entries: Some((4, 16)),
            },
        ] {
            let enc = m.encode().unwrap();
            assert_eq!(GroupInfoMessage::parse(&enc, Sizes::DEFAULT).unwrap(), m);
        }
        let empty = GroupInfoMessage {
            link_phase_change: None,
            estimated_entries: None,
        };
        assert_eq!(empty.encode().unwrap(), [0, 0]);
    }
}
