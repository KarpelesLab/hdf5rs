//! Filter pipeline message (type 0x000B).
//!
//! Spec: "IV.A.2.l. The Data Storage - Filter Pipeline Message", versions 1
//! and 2.

use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};

/// Maximum number of filters in a pipeline, per the specification.
pub const MAX_FILTERS: usize = 32;

/// Filter ID of gzip/deflate compression.
pub const FILTER_DEFLATE: u16 = 1;
/// Filter ID of byte shuffling.
pub const FILTER_SHUFFLE: u16 = 2;
/// Filter ID of the Fletcher-32 checksum.
pub const FILTER_FLETCHER32: u16 = 3;
/// Filter ID of SZIP compression.
pub const FILTER_SZIP: u16 = 4;
/// Filter ID of N-bit packing.
pub const FILTER_NBIT: u16 = 5;
/// Filter ID of scale-offset encoding.
pub const FILTER_SCALEOFFSET: u16 = 6;

/// One filter of a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterInfo {
    /// Filter identification value.
    pub id: u16,
    /// Optional filter name (NUL padding removed).
    pub name: Option<String>,
    /// Optional filters may be skipped silently when they fail on output.
    pub optional: bool,
    /// Filter parameters.
    pub client_data: Vec<u32>,
}

/// A decoded filter pipeline message.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FilterPipeline {
    /// Filters in application order (for writing; reverse to read).
    pub filters: Vec<FilterInfo>,
}

impl FilterPipeline {
    /// Parse a filter pipeline message.
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<FilterPipeline> {
        let mut r = ByteReader::new(data);
        let version = r.read_u8()?;
        if !(1..=2).contains(&version) {
            return Err(Error::unsupported(format!(
                "filter pipeline version {version}"
            )));
        }
        let count = r.read_u8()? as usize;
        if count > MAX_FILTERS {
            return Err(Error::invalid("more than 32 filters in pipeline"));
        }
        if version == 1 {
            r.skip(6)?; // reserved
        }
        let mut filters = Vec::with_capacity(count);
        for _ in 0..count {
            let id = r.read_u16()?;
            let name_len = if version == 1 || id >= 256 {
                r.read_u16()? as usize
            } else {
                0
            };
            let flags = r.read_u16()?;
            let nvalues = r.read_u16()? as usize;
            let name = if name_len > 0 {
                let raw = r.read_bytes(name_len)?;
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                Some(String::from_utf8_lossy(&raw[..end]).into_owned())
            } else {
                None
            };
            let mut client_data = Vec::with_capacity(nvalues.min(r.remaining() / 4));
            for _ in 0..nvalues {
                client_data.push(r.read_u32()?);
            }
            if version == 1 && nvalues % 2 == 1 {
                r.skip(4)?;
            }
            filters.push(FilterInfo {
                id,
                name,
                optional: flags & 1 != 0,
                client_data,
            });
        }
        Ok(FilterPipeline { filters })
    }

    /// Encode the message body as version 1 or 2.
    ///
    /// Version 1 stores every filter's name (when present), NUL-terminated
    /// and padded to a multiple of eight bytes, and pads odd client data
    /// counts; version 2 stores names only for filter IDs of 256 and above,
    /// without padding.
    pub fn encode(&self, version: u8) -> Result<Vec<u8>> {
        if !(1..=2).contains(&version) {
            return Err(Error::unsupported(format!(
                "encoding filter pipeline version {version}"
            )));
        }
        if self.filters.len() > MAX_FILTERS {
            return Err(Error::invalid("more than 32 filters in pipeline"));
        }
        let mut w = crate::format::ByteWriter::new();
        w.u8(version).u8(self.filters.len() as u8);
        if version == 1 {
            w.zeros(6);
        }
        for f in &self.filters {
            let nvalues = u16::try_from(f.client_data.len())
                .map_err(|_| Error::LimitExceeded("filter client data"))?;
            let name: Vec<u8> = match &f.name {
                Some(n) if version == 1 || f.id >= 256 => {
                    if n.contains('\0') {
                        return Err(Error::invalid("filter name contains NUL"));
                    }
                    let mut b = n.as_bytes().to_vec();
                    b.push(0);
                    if version == 1 {
                        b.resize(b.len().next_multiple_of(8), 0);
                    }
                    b
                }
                _ => Vec::new(),
            };
            let name_len =
                u16::try_from(name.len()).map_err(|_| Error::LimitExceeded("filter name"))?;
            w.u16(f.id);
            if version == 1 || f.id >= 256 {
                w.u16(name_len);
            }
            w.u16(u16::from(f.optional)).u16(nvalues).bytes(&name);
            for &v in &f.client_data {
                w.u32(v);
            }
            if version == 1 && nvalues % 2 == 1 {
                w.zeros(4);
            }
        }
        Ok(w.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Sizes = Sizes::DEFAULT;

    fn filter_v1(v: &mut Vec<u8>, id: u16, name: &[u8], flags: u16, cd: &[u32]) {
        v.extend_from_slice(&id.to_le_bytes());
        v.extend_from_slice(&(name.len() as u16).to_le_bytes());
        v.extend_from_slice(&flags.to_le_bytes());
        v.extend_from_slice(&(cd.len() as u16).to_le_bytes());
        v.extend_from_slice(name);
        for c in cd {
            v.extend_from_slice(&c.to_le_bytes());
        }
        if cd.len() % 2 == 1 {
            v.extend_from_slice(&[0; 4]);
        }
    }

    #[test]
    fn version_1() {
        let mut v = vec![1, 2, 0, 0, 0, 0, 0, 0];
        filter_v1(&mut v, 2, b"", 0, &[8]);
        filter_v1(&mut v, 1, b"deflate\0", 1, &[6]);
        let p = FilterPipeline::parse(&v, S).unwrap();
        assert_eq!(
            p.filters,
            vec![
                FilterInfo {
                    id: FILTER_SHUFFLE,
                    name: None,
                    optional: false,
                    client_data: vec![8]
                },
                FilterInfo {
                    id: FILTER_DEFLATE,
                    name: Some("deflate".into()),
                    optional: true,
                    client_data: vec![6]
                }
            ]
        );
    }

    #[test]
    fn version_2() {
        let mut v = vec![2, 2];
        // deflate: id < 256, no name length
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&4u32.to_le_bytes());
        // third-party filter with name, no padding
        v.extend_from_slice(&32000u16.to_le_bytes());
        v.extend_from_slice(&4u16.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(b"lzf\0");
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&2u32.to_le_bytes());
        let p = FilterPipeline::parse(&v, S).unwrap();
        assert_eq!(p.filters[0].client_data, vec![4]);
        assert_eq!(p.filters[0].name, None);
        assert_eq!(p.filters[1].id, 32000);
        assert_eq!(p.filters[1].name.as_deref(), Some("lzf"));
        assert!(p.filters[1].optional);
        assert_eq!(p.filters[1].client_data, vec![1, 2]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(FilterPipeline::parse(&[], S).is_err());
        assert!(FilterPipeline::parse(&[3, 0], S).is_err());
        assert!(FilterPipeline::parse(&[2, 33], S).is_err());
        assert!(FilterPipeline::parse(&[2, 1, 1, 0, 0, 0, 0xff, 0xff], S).is_err());
        // v1 with odd client data but missing padding
        let mut v = vec![1, 1, 0, 0, 0, 0, 0, 0];
        filter_v1(&mut v, 1, b"", 0, &[6]);
        v.truncate(v.len() - 4);
        assert!(FilterPipeline::parse(&v, S).is_err());
    }

    #[test]
    fn mutation_never_panics() {
        let mut v = vec![1, 2, 0, 0, 0, 0, 0, 0];
        filter_v1(&mut v, 2, b"", 0, &[8]);
        filter_v1(&mut v, 1, b"deflate\0", 1, &[6]);
        let mut state = 0x5555_aaaa_1234_9876u64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..10_000 {
            let mut m = v.clone();
            let i = (rnd() % m.len() as u64) as usize;
            m[i] = rnd() as u8;
            m.truncate((rnd() % (m.len() as u64 + 1)) as usize);
            let _ = FilterPipeline::parse(&m, S);
        }
    }
}

#[cfg(test)]
mod encode_tests {
    use super::*;

    fn sample() -> FilterPipeline {
        FilterPipeline {
            filters: vec![
                FilterInfo {
                    id: FILTER_SHUFFLE,
                    name: Some("shuffle".into()),
                    optional: true,
                    client_data: vec![4],
                },
                FilterInfo {
                    id: FILTER_DEFLATE,
                    name: Some("deflate".into()),
                    optional: true,
                    client_data: vec![4],
                },
                FilterInfo {
                    id: FILTER_FLETCHER32,
                    name: Some("fletcher32".into()),
                    optional: false,
                    client_data: vec![],
                },
                FilterInfo {
                    id: 32000,
                    name: Some("lzf".into()),
                    optional: true,
                    client_data: vec![4, 261, 40],
                },
            ],
        }
    }

    #[test]
    fn matches_reference_bytes() {
        // Bytes as written by the reference library (v2: names only for
        // IDs >= 256; v1: names padded to 8 bytes).
        let p = FilterPipeline {
            filters: sample().filters[3..].to_vec(),
        };
        let mut v2 = vec![2, 1, 0, 0x7d, 4, 0, 1, 0, 3, 0];
        v2.extend_from_slice(b"lzf\0");
        v2.extend_from_slice(&[4, 0, 0, 0, 5, 1, 0, 0, 40, 0, 0, 0]);
        assert_eq!(p.encode(2).unwrap(), v2);
        let p = FilterPipeline {
            filters: sample().filters[1..2].to_vec(),
        };
        let mut v1 = vec![1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 1, 0, 1, 0];
        v1.extend_from_slice(b"deflate\0");
        v1.extend_from_slice(&[4, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(p.encode(1).unwrap(), v1);
    }

    #[test]
    fn round_trip() {
        let p = sample();
        for v in [1, 2] {
            let mut got = FilterPipeline::parse(&p.encode(v).unwrap(), Sizes::DEFAULT).unwrap();
            if v == 2 {
                for (g, e) in got.filters.iter_mut().zip(&p.filters) {
                    if e.id < 256 {
                        g.name = e.name.clone();
                    }
                }
            }
            assert_eq!(got, p);
        }
        assert!(p.encode(3).is_err());
    }
}
