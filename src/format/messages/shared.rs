//! Shared message encoding (spec IV.A.2, "Shared Message" versions 1–3).
//!
//! When bit 1 of a header message's flags is set, its body is not the
//! message itself but one of these records, pointing at where the real
//! message lives.

use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};

/// Where the actual message body is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedLocation {
    /// The message is in another object's header (a "committed" message,
    /// e.g. a named datatype). Holds that object header's address.
    ObjectHeader(u64),
    /// The message is in the file's shared object header message (SOHM)
    /// fractal heap. Holds the 8-byte heap ID.
    SharedHeap([u8; 8]),
    /// Version 3 type 0: not shared and not shareable (no location).
    NotShared,
    /// Version 3 type 3: not shared, but shareable (no location).
    Shareable,
}

/// A decoded shared message record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMessage {
    /// Encoding version (1, 2 or 3).
    pub version: u8,
    /// Where the message body lives.
    pub location: SharedLocation,
}

impl SharedMessage {
    /// Decode a shared message record.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        let version = r.read_u8()?;
        let kind = r.read_u8()?;
        let location = match version {
            1 | 2 => {
                if version == 1 {
                    // Reserved: 2 bytes + 4 bytes.
                    r.skip(6)?;
                }
                // The spec documents type 0 for versions 1 and 2, but files
                // written by the reference library use 2 ("committed in an
                // object header", as in version 3); accept both.
                if kind != 0 && kind != 2 {
                    return Err(Error::invalid(format!(
                        "shared message v{version} with unknown type {kind}"
                    )));
                }
                SharedLocation::ObjectHeader(r.read_offset()?)
            }
            3 => match kind {
                0 => SharedLocation::NotShared,
                1 => SharedLocation::SharedHeap(r.read_array::<8>()?),
                2 => SharedLocation::ObjectHeader(r.read_offset()?),
                3 => SharedLocation::Shareable,
                k => {
                    return Err(Error::invalid(format!(
                        "shared message v3 with unknown type {k}"
                    )));
                }
            },
            v => return Err(Error::unsupported(format!("shared message version {v}"))),
        };
        Ok(SharedMessage { version, location })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::W;

    #[test]
    fn versions() {
        let mut w = W::new();
        w.u8(1).u8(0).u16(0).u32(0).offset(800);
        let m = SharedMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(m.location, SharedLocation::ObjectHeader(800));

        let mut w = W::new();
        w.u8(2).u8(0).offset(900);
        let m = SharedMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(m.version, 2);
        assert_eq!(m.location, SharedLocation::ObjectHeader(900));

        let mut w = W::new();
        w.u8(3).u8(2).offset(1000);
        let m = SharedMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(m.location, SharedLocation::ObjectHeader(1000));

        let mut w = W::new();
        w.u8(3).u8(1).bytes(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let m = SharedMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(
            m.location,
            SharedLocation::SharedHeap([1, 2, 3, 4, 5, 6, 7, 8])
        );
        assert!(SharedMessage::parse(&w.buf[..9], Sizes::DEFAULT).is_err());
    }

    #[test]
    fn bad() {
        assert!(SharedMessage::parse(&[4, 0, 0, 0], Sizes::DEFAULT).is_err());
        assert!(SharedMessage::parse(&[3, 9], Sizes::DEFAULT).is_err());
        assert!(SharedMessage::parse(&[2, 1, 0, 0, 0, 0, 0, 0, 0, 0], Sizes::DEFAULT).is_err());
        assert_eq!(
            SharedMessage::parse(&[2, 2, 9, 0, 0, 0, 0, 0, 0, 0], Sizes::DEFAULT)
                .unwrap()
                .location,
            SharedLocation::ObjectHeader(9)
        );
        assert!(SharedMessage::parse(&[1, 0, 0, 0, 0, 0, 0], Sizes::DEFAULT).is_err());
    }
}
