//! Attribute message (type 0x000C).
//!
//! Spec: "IV.A.2.m. The Attribute Message", versions 1-3.

use super::dataspace::Dataspace;
use super::datatype::{CharacterSet, Datatype};
use crate::error::{Error, Result};
use crate::format::{ByteReader, ByteWriter, Sizes};

/// A message field that is either encoded inline or as a shared-message
/// reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaybeShared<T> {
    /// The message is stored inline and has been decoded.
    Inline(T),
    /// The field holds a shared message encoding (raw bytes, undecoded).
    Shared(Vec<u8>),
}

impl<T> MaybeShared<T> {
    /// The inline value, if not shared.
    pub fn inline(&self) -> Option<&T> {
        match self {
            MaybeShared::Inline(t) => Some(t),
            MaybeShared::Shared(_) => None,
        }
    }
}

/// A decoded attribute message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribute {
    /// Encoding version (1-3).
    pub version: u8,
    /// Attribute name (terminating NUL removed).
    pub name: String,
    /// Character set of the name (always ASCII before version 3).
    pub name_charset: CharacterSet,
    /// The attribute's datatype.
    pub datatype: MaybeShared<Datatype>,
    /// The attribute's dataspace.
    pub dataspace: MaybeShared<Dataspace>,
    /// Raw attribute data. When both datatype and dataspace are inline this
    /// is exactly `element_count * size` bytes; otherwise it is the whole
    /// remainder of the message.
    pub data: Vec<u8>,
}

/// Read a field of `len` bytes, padded to 8 bytes in version 1.
fn field<'a>(r: &mut ByteReader<'a>, len: usize, version: u8) -> Result<&'a [u8]> {
    let bytes = r.read_bytes(len)?;
    if version == 1 {
        let padded = len
            .checked_next_multiple_of(8)
            .ok_or(Error::UnexpectedEof)?;
        r.skip(padded - len)?;
    }
    Ok(bytes)
}

impl Attribute {
    /// Parse an attribute message.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Attribute> {
        let mut r = ByteReader::with_sizes(data, sizes);
        let version = r.read_u8()?;
        if !(1..=3).contains(&version) {
            return Err(Error::unsupported(format!("attribute version {version}")));
        }
        let flags = r.read_u8()?;
        if version == 1 {
            // Reserved in version 1: shared types are not supported.
        } else if flags & !0x03 != 0 {
            return Err(Error::invalid("reserved attribute flags set"));
        }
        let flags = if version == 1 { 0 } else { flags };
        let name_size = r.read_u16()? as usize;
        let datatype_size = r.read_u16()? as usize;
        let dataspace_size = r.read_u16()? as usize;
        let name_charset = if version >= 3 {
            CharacterSet::from_raw(u32::from(r.read_u8()?))?
        } else {
            CharacterSet::Ascii
        };

        let raw_name = field(&mut r, name_size, version)?;
        let end = raw_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(raw_name.len());
        let name = String::from_utf8_lossy(&raw_name[..end]).into_owned();

        let dt_bytes = field(&mut r, datatype_size, version)?;
        let datatype = if flags & 1 != 0 {
            MaybeShared::Shared(dt_bytes.to_vec())
        } else {
            MaybeShared::Inline(Datatype::parse(dt_bytes)?)
        };
        let ds_bytes = field(&mut r, dataspace_size, version)?;
        let dataspace = if flags & 2 != 0 {
            MaybeShared::Shared(ds_bytes.to_vec())
        } else {
            MaybeShared::Inline(Dataspace::parse(ds_bytes, sizes)?)
        };

        let rest = r.rest();
        let data = match (&datatype, &dataspace) {
            (MaybeShared::Inline(dt), MaybeShared::Inline(ds)) => {
                let len = ds
                    .element_count()?
                    .checked_mul(u64::from(dt.size))
                    .ok_or(Error::LimitExceeded("attribute data size"))?;
                let len = usize::try_from(len).map_err(|_| Error::UnexpectedEof)?;
                rest.get(..len).ok_or(Error::UnexpectedEof)?.to_vec()
            }
            _ => rest.to_vec(),
        };
        Ok(Attribute {
            version,
            name,
            name_charset,
            datatype,
            dataspace,
            data,
        })
    }
}

/// Write a field, padded to 8 bytes in version 1.
fn put_field(w: &mut ByteWriter, bytes: &[u8], version: u8) {
    w.bytes(bytes);
    if version == 1 {
        w.zeros(bytes.len().next_multiple_of(8) - bytes.len());
    }
}

impl Attribute {
    /// A version 3 attribute with an inline datatype and dataspace; the
    /// name character set is ASCII for ASCII names, UTF-8 otherwise.
    pub fn new_v3(name: &str, datatype: Datatype, dataspace: Dataspace, data: Vec<u8>) -> Self {
        Attribute {
            version: 3,
            name: name.to_owned(),
            name_charset: if name.is_ascii() {
                CharacterSet::Ascii
            } else {
                CharacterSet::Utf8
            },
            datatype: MaybeShared::Inline(datatype),
            dataspace: MaybeShared::Inline(dataspace),
            data,
        }
    }

    /// Encode the message body (versions 1-3, as given by `version`).
    ///
    /// Shared fields are written verbatim with the matching flag bit.
    /// For inline fields the data must be exactly `element count * size`
    /// bytes.
    pub fn encode(&self, sizes: Sizes) -> Result<Vec<u8>> {
        let version = self.version;
        if !(1..=3).contains(&version) {
            return Err(Error::unsupported(format!("attribute version {version}")));
        }
        if self.name.is_empty() || self.name.as_bytes().contains(&0) {
            return Err(Error::invalid("attribute name is empty or contains NUL"));
        }
        if version < 3 && self.name_charset != CharacterSet::Ascii {
            return Err(Error::invalid("UTF-8 attribute names need version 3"));
        }
        let mut name = self.name.as_bytes().to_vec();
        name.push(0);
        let mut flags = 0u8;
        let dt = match &self.datatype {
            MaybeShared::Inline(dt) => dt.encode()?,
            MaybeShared::Shared(raw) => {
                flags |= 1;
                raw.clone()
            }
        };
        let ds = match &self.dataspace {
            MaybeShared::Inline(ds) => ds.encode(sizes)?,
            MaybeShared::Shared(raw) => {
                flags |= 2;
                raw.clone()
            }
        };
        if version == 1 && flags != 0 {
            return Err(Error::invalid("shared attribute fields need version 2+"));
        }
        if let (MaybeShared::Inline(dt), MaybeShared::Inline(ds)) =
            (&self.datatype, &self.dataspace)
        {
            let expect = ds.element_count()?.checked_mul(u64::from(dt.size));
            if expect != Some(self.data.len() as u64) {
                return Err(Error::invalid(
                    "attribute data size does not match its type",
                ));
            }
        }
        let len16 = |n: usize| {
            u16::try_from(n).map_err(|_| Error::LimitExceeded("attribute field over 65535 bytes"))
        };
        let mut w = ByteWriter::with_sizes(sizes);
        w.u8(version)
            .u8(flags)
            .u16(len16(name.len())?)
            .u16(len16(dt.len())?)
            .u16(len16(ds.len())?);
        if version >= 3 {
            w.u8(match self.name_charset {
                CharacterSet::Ascii => 0,
                CharacterSet::Utf8 => 1,
            });
        }
        put_field(&mut w, &name, version);
        put_field(&mut w, &dt, version);
        put_field(&mut w, &ds, version);
        w.bytes(&self.data);
        Ok(w.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::messages::datatype::DatatypeClass;

    const S: Sizes = Sizes::DEFAULT;

    fn i32_type() -> Vec<u8> {
        vec![0x10, 0x08, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0]
    }

    fn simple_space_v1(n: u64) -> Vec<u8> {
        let mut v = vec![1, 1, 0, 0, 0, 0, 0, 0];
        v.extend_from_slice(&n.to_le_bytes());
        v
    }

    fn pad8(v: &mut Vec<u8>) {
        while !v.len().is_multiple_of(8) {
            v.push(0);
        }
    }

    #[test]
    fn version_1_padded() {
        let dt = i32_type(); // 12 bytes -> padded to 16
        let ds = simple_space_v1(2); // 16 bytes
        let mut v = vec![1, 0];
        v.extend_from_slice(&6u16.to_le_bytes()); // "hello\0"
        v.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        v.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        v.extend_from_slice(b"hello\0");
        pad8(&mut v);
        v.extend_from_slice(&dt);
        pad8(&mut v);
        v.extend_from_slice(&ds);
        v.extend_from_slice(&1i32.to_le_bytes());
        v.extend_from_slice(&2i32.to_le_bytes());
        v.extend_from_slice(&[0; 4]); // message padding
        let a = Attribute::parse(&v, S).unwrap();
        assert_eq!(a.name, "hello");
        assert_eq!(a.name_charset, CharacterSet::Ascii);
        assert!(a.datatype.inline().unwrap().is_integer());
        assert_eq!(a.dataspace.inline().unwrap().element_count().unwrap(), 2);
        assert_eq!(a.data, vec![1, 0, 0, 0, 2, 0, 0, 0]);
    }

    #[test]
    fn version_3_scalar_utf8() {
        let dt = i32_type();
        let ds = vec![2, 0, 0, 0];
        let mut v = vec![3, 0];
        v.extend_from_slice(&3u16.to_le_bytes());
        v.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        v.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        v.push(1);
        v.extend_from_slice("é\0".as_bytes());
        v.extend_from_slice(&dt);
        v.extend_from_slice(&ds);
        v.extend_from_slice(&7i32.to_le_bytes());
        let a = Attribute::parse(&v, S).unwrap();
        assert_eq!(a.name, "é");
        assert_eq!(a.name_charset, CharacterSet::Utf8);
        assert_eq!(a.data, vec![7, 0, 0, 0]);
        assert!(matches!(
            a.datatype.inline().unwrap().class,
            DatatypeClass::FixedPoint(_)
        ));
        // Truncated data.
        assert!(Attribute::parse(&v[..v.len() - 1], S).is_err());
    }

    #[test]
    fn version_2_shared_datatype() {
        let shared = vec![3, 2, 0x10, 0, 0, 0, 0, 0, 0, 0];
        let ds = vec![2, 0, 0, 0];
        let mut v = vec![2, 1];
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&(shared.len() as u16).to_le_bytes());
        v.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        v.extend_from_slice(b"x\0");
        v.extend_from_slice(&shared);
        v.extend_from_slice(&ds);
        v.extend_from_slice(&[9, 9]);
        let a = Attribute::parse(&v, S).unwrap();
        assert_eq!(a.datatype, MaybeShared::Shared(shared));
        assert!(a.dataspace.inline().is_some());
        assert_eq!(a.data, vec![9, 9]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(Attribute::parse(&[], S).is_err());
        assert!(Attribute::parse(&[4, 0, 0, 0, 0, 0, 0, 0], S).is_err());
        assert!(Attribute::parse(&[2, 4, 0, 0, 0, 0, 0, 0], S).is_err());
        // Name larger than the message.
        assert!(Attribute::parse(&[2, 0, 0xff, 0, 0, 0, 0, 0, b'a'], S).is_err());
        // Garbage datatype.
        let mut v = vec![2, 0, 2, 0, 8, 0, 4, 0, b'a', 0];
        v.extend_from_slice(&[0xff; 8]);
        v.extend_from_slice(&[2, 0, 0, 0]);
        assert!(Attribute::parse(&v, S).is_err());
    }

    #[test]
    fn mutation_never_panics() {
        let dt = i32_type();
        let ds = simple_space_v1(2);
        let mut v = vec![1, 0];
        v.extend_from_slice(&6u16.to_le_bytes());
        v.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        v.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        v.extend_from_slice(b"hello\0");
        pad8(&mut v);
        v.extend_from_slice(&dt);
        pad8(&mut v);
        v.extend_from_slice(&ds);
        v.extend_from_slice(&[1, 0, 0, 0, 2, 0, 0, 0]);
        let mut state = 0x7777_1111_3333_9999u64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..10_000 {
            let mut m = v.clone();
            for _ in 0..(rnd() % 3 + 1) {
                let i = (rnd() % m.len() as u64) as usize;
                m[i] = rnd() as u8;
            }
            m.truncate((rnd() % (m.len() as u64 + 1)) as usize);
            let _ = Attribute::parse(&m, S);
        }
    }

    #[test]
    fn encode_round_trip() {
        use crate::format::messages::datatype::ByteOrder;
        let i32le = Datatype::integer(4, true, ByteOrder::LittleEndian);
        let mut cases = vec![
            Attribute::new_v3(
                "temp\u{e9}rature",
                i32le.clone(),
                Dataspace::simple(&[2], None),
                vec![1, 0, 0, 0, 2, 0, 0, 0],
            ),
            Attribute::new_v3("s", i32le.clone(), Dataspace::scalar(), vec![7, 0, 0, 0]),
            Attribute::new_v3("n", i32le.clone(), Dataspace::null(), vec![]),
        ];
        for version in [1, 2] {
            let mut a = Attribute::new_v3(
                "units",
                i32le.clone(),
                Dataspace::simple(&[1], None),
                vec![1, 2, 3, 4],
            );
            a.version = version;
            a.name_charset = CharacterSet::Ascii;
            if version == 1 {
                a.dataspace = MaybeShared::Inline(Dataspace {
                    version: 1,
                    ..Dataspace::simple(&[1], None)
                });
            }
            cases.push(a);
        }
        let mut shared = Attribute::new_v3("c", i32le.clone(), Dataspace::scalar(), vec![]);
        shared.datatype = MaybeShared::Shared(vec![3, 2, 0, 1, 0, 0, 0, 0, 0, 0]);
        shared.data = vec![9, 9, 9, 9];
        cases.push(shared);
        for a in &cases {
            let enc = a.encode(Sizes::DEFAULT).unwrap();
            assert_eq!(&Attribute::parse(&enc, Sizes::DEFAULT).unwrap(), a);
        }
        let mut bad = cases[0].clone();
        bad.data.pop();
        assert!(bad.encode(Sizes::DEFAULT).is_err());
        let mut bad = cases[0].clone();
        bad.version = 2;
        assert!(bad.encode(Sizes::DEFAULT).is_err());
    }
}
