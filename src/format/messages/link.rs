//! Link message (type 0x0006, spec IV.A.2.g).

use super::until_nul;
use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};

/// Character set of a link name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharSet {
    /// ASCII (0).
    Ascii,
    /// UTF-8 (1).
    Utf8,
}

/// Link type value of a soft link.
pub const LINK_TYPE_SOFT: u8 = 1;
/// Link type value of an external link.
pub const LINK_TYPE_EXTERNAL: u8 = 64;

/// What a link points to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkTarget {
    /// Hard link: address of the target object header.
    Hard {
        /// Object header address.
        address: u64,
    },
    /// Soft link: a path within this file.
    Soft {
        /// Link value (path); not NUL-terminated on disk.
        path: String,
    },
    /// External link: an object in another HDF5 file.
    External {
        /// Version (upper 4 bits of the first value byte; 0 defined).
        version: u8,
        /// Flags (lower 4 bits of the first value byte; 0 defined).
        flags: u8,
        /// Name of the target HDF5 file.
        file: String,
        /// Absolute path of the object within that file.
        path: String,
    },
    /// User-defined link type (65–255) or reserved type (2–63).
    UserDefined {
        /// Link type value.
        link_type: u8,
        /// Opaque link data.
        data: Vec<u8>,
    },
}

/// A decoded Link message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkMessage {
    /// Creation order index, if stored.
    pub creation_order: Option<u64>,
    /// Character set of the name.
    pub char_set: CharSet,
    /// Link name. Non-UTF-8 names are rejected as invalid.
    pub name: String,
    /// Link target.
    pub target: LinkTarget,
}

fn utf8(bytes: &[u8], what: &'static str) -> Result<String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| Error::invalid(what))
}

impl LinkMessage {
    /// Decode the message body (version 1).
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        let version = r.read_u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!(
                "link message version {version}"
            )));
        }
        let flags = r.read_u8()?;
        if flags & 0xe0 != 0 {
            return Err(Error::invalid("reserved link message flags set"));
        }
        // An explicit type of 0 (hard) "should never be stored" but is
        // harmless, so it is accepted.
        let link_type = if flags & 0x08 != 0 { r.read_u8()? } else { 0 };
        let creation_order = if flags & 0x04 != 0 {
            Some(r.read_u64()?)
        } else {
            None
        };
        let char_set = if flags & 0x10 != 0 {
            match r.read_u8()? {
                0 => CharSet::Ascii,
                1 => CharSet::Utf8,
                c => return Err(Error::invalid(format!("unknown link name charset {c}"))),
            }
        } else {
            CharSet::Ascii
        };
        let name_len = r.read_uint(1 << (flags & 0x03))?;
        let name_len = usize::try_from(name_len).map_err(|_| Error::UnexpectedEof)?;
        if name_len == 0 {
            return Err(Error::invalid("empty link name"));
        }
        let name = utf8(r.read_bytes(name_len)?, "link name is not valid UTF-8")?;
        let target = match link_type {
            0 => LinkTarget::Hard {
                address: r.read_offset()?,
            },
            _ => {
                let len = r.read_u16()? as usize;
                let value = r.read_bytes(len)?;
                match link_type {
                    LINK_TYPE_SOFT => LinkTarget::Soft {
                        path: utf8(value, "soft link value is not valid UTF-8")?,
                    },
                    LINK_TYPE_EXTERNAL => parse_external(value)?,
                    t => LinkTarget::UserDefined {
                        link_type: t,
                        data: value.to_vec(),
                    },
                }
            }
        };
        Ok(LinkMessage {
            creation_order,
            char_set,
            name,
            target,
        })
    }
}

fn parse_external(value: &[u8]) -> Result<LinkTarget> {
    let (&first, rest) = value
        .split_first()
        .ok_or(Error::invalid("empty external link value"))?;
    let file = until_nul(rest);
    let after = rest
        .get(file.len() + 1..)
        .ok_or(Error::invalid("external link missing object path"))?;
    // The final terminator is tolerated if missing.
    let path = until_nul(after);
    Ok(LinkTarget::External {
        version: first >> 4,
        flags: first & 0x0f,
        file: utf8(file, "external link file name is not valid UTF-8")?,
        path: utf8(path, "external link path is not valid UTF-8")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::W;

    #[test]
    fn hard_link() {
        let mut w = W::new();
        w.u8(1).u8(0).u8(4).bytes(b"data").offset(1234);
        let m = LinkMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(m.name, "data");
        assert_eq!(m.char_set, CharSet::Ascii);
        assert_eq!(m.creation_order, None);
        assert_eq!(m.target, LinkTarget::Hard { address: 1234 });
        for n in 0..w.len() {
            assert!(LinkMessage::parse(&w.buf[..n], Sizes::DEFAULT).is_err());
        }
    }

    #[test]
    fn soft_link_with_order_and_utf8() {
        let name = "caf\u{e9}";
        let mut w = W::new();
        // flags: 2-byte name length, creation order, link type, charset
        w.u8(1)
            .u8(0x01 | 0x04 | 0x08 | 0x10)
            .u8(LINK_TYPE_SOFT)
            .u64(7)
            .u8(1)
            .u16(name.len() as u16)
            .bytes(name.as_bytes())
            .u16(5)
            .bytes(b"/a/b/");
        let m = LinkMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(m.name, name);
        assert_eq!(m.char_set, CharSet::Utf8);
        assert_eq!(m.creation_order, Some(7));
        assert_eq!(
            m.target,
            LinkTarget::Soft {
                path: "/a/b/".into()
            }
        );
    }

    #[test]
    fn external_and_user_defined() {
        let value = b"\x00other.h5\0/grp/ds\0";
        let mut w = W::new();
        w.u8(1)
            .u8(0x08)
            .u8(LINK_TYPE_EXTERNAL)
            .u8(3)
            .bytes(b"ext")
            .u16(value.len() as u16)
            .bytes(value);
        let m = LinkMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(
            m.target,
            LinkTarget::External {
                version: 0,
                flags: 0,
                file: "other.h5".into(),
                path: "/grp/ds".into()
            }
        );

        let mut w = W::new();
        w.u8(1)
            .u8(0x08)
            .u8(200)
            .u8(1)
            .bytes(b"u")
            .u16(2)
            .bytes(&[9, 8]);
        let m = LinkMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(
            m.target,
            LinkTarget::UserDefined {
                link_type: 200,
                data: vec![9, 8]
            }
        );
    }

    #[test]
    fn bad() {
        // Version 2.
        assert!(
            LinkMessage::parse(&[2, 0, 1, b'a', 0, 0, 0, 0, 0, 0, 0, 0], Sizes::DEFAULT).is_err()
        );
        // External link with no path.
        let mut w = W::new();
        w.u8(1)
            .u8(0x08)
            .u8(64)
            .u8(1)
            .bytes(b"e")
            .u16(3)
            .bytes(b"\0ab");
        assert!(LinkMessage::parse(&w.buf, Sizes::DEFAULT).is_err());
        // Huge name length (8-byte field).
        let mut w = W::new();
        w.u8(1).u8(0x03).u64(u64::MAX);
        assert!(LinkMessage::parse(&w.buf, Sizes::DEFAULT).is_err());
    }
}
