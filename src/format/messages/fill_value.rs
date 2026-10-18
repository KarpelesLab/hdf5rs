//! Fill value messages: old (type 0x0004) and new (type 0x0005).
//!
//! Spec: "IV.A.2.e. Data Storage - Fill Value (Old) Message" and
//! "IV.A.2.f. The Data Storage - Fill Value Message", versions 1-3.

use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};

/// The deprecated fill value message (type 0x0004).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OldFillValue {
    /// Fill value bytes, interpreted with the dataset's datatype.
    pub value: Vec<u8>,
}

impl OldFillValue {
    /// Parse an old-style fill value message.
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<OldFillValue> {
        let mut r = ByteReader::new(data);
        let size = r.read_u32()? as usize;
        Ok(OldFillValue {
            value: r.read_bytes(size)?.to_vec(),
        })
    }
}

/// When storage space for raw data is allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocTime {
    /// Value 0 ("not used"): no preference recorded.
    Unspecified,
    /// Allocate everything at dataset creation.
    Early,
    /// Allocate when the dataset is first written.
    Late,
    /// Allocate chunks as they are written.
    Incremental,
}

/// When the fill value is written to newly allocated storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillWriteTime {
    /// Always written on allocation.
    OnAlloc,
    /// Never written.
    Never,
    /// Written only if set explicitly by the user.
    IfSet,
}

/// Definition state of the fill value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillValueState {
    /// No fill value is defined (version 1/2 "defined" = 0, or version 3
    /// with neither the defined nor the undefined flag): readers use the
    /// library default of all zero bytes.
    Default,
    /// The fill value has explicitly been marked undefined (version 3).
    Undefined,
    /// A fill value is defined (it may still be empty).
    Defined,
}

/// The fill value message (type 0x0005).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FillValue {
    /// Encoding version (1-3).
    pub version: u8,
    /// Space allocation time.
    pub alloc_time: AllocTime,
    /// Fill value write time.
    pub write_time: FillWriteTime,
    /// Whether a fill value is defined.
    pub state: FillValueState,
    /// Fill value bytes, when stored with a non-zero size.
    pub value: Option<Vec<u8>>,
}

fn alloc_time(v: u8) -> Result<AllocTime> {
    Ok(match v {
        0 => AllocTime::Unspecified,
        1 => AllocTime::Early,
        2 => AllocTime::Late,
        3 => AllocTime::Incremental,
        other => return Err(Error::invalid(format!("unknown allocation time {other}"))),
    })
}

fn write_time(v: u8) -> Result<FillWriteTime> {
    Ok(match v {
        0 => FillWriteTime::OnAlloc,
        1 => FillWriteTime::Never,
        2 => FillWriteTime::IfSet,
        other => {
            return Err(Error::invalid(format!(
                "unknown fill value write time {other}"
            )));
        }
    })
}

fn read_value(r: &mut ByteReader<'_>) -> Result<Option<Vec<u8>>> {
    let size = r.read_u32()? as usize;
    let bytes = r.read_bytes(size)?;
    Ok((size > 0).then(|| bytes.to_vec()))
}

impl FillValue {
    /// Parse a fill value message.
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<FillValue> {
        let mut r = ByteReader::new(data);
        let version = r.read_u8()?;
        match version {
            1 | 2 => {
                let alloc_time = alloc_time(r.read_u8()?)?;
                let write_time = write_time(r.read_u8()?)?;
                let defined = match r.read_u8()? {
                    0 => false,
                    1 => true,
                    other => {
                        return Err(Error::invalid(format!(
                            "invalid fill value defined flag {other}"
                        )));
                    }
                };
                let value = if version == 1 || defined {
                    read_value(&mut r)?
                } else {
                    None
                };
                Ok(FillValue {
                    version,
                    alloc_time,
                    write_time,
                    state: if defined {
                        FillValueState::Defined
                    } else {
                        FillValueState::Default
                    },
                    value,
                })
            }
            3 => {
                let flags = r.read_u8()?;
                if flags & 0xc0 != 0 {
                    return Err(Error::invalid("reserved fill value flags set"));
                }
                let state = match (flags & 0x10 != 0, flags & 0x20 != 0) {
                    (false, false) => FillValueState::Default,
                    (true, false) => FillValueState::Undefined,
                    (false, true) => FillValueState::Defined,
                    (true, true) => {
                        return Err(Error::invalid("fill value both defined and undefined"));
                    }
                };
                let value = if state == FillValueState::Defined {
                    read_value(&mut r)?
                } else {
                    None
                };
                Ok(FillValue {
                    version,
                    alloc_time: alloc_time(flags & 3)?,
                    write_time: write_time((flags >> 2) & 3)?,
                    state,
                    value,
                })
            }
            v => Err(Error::unsupported(format!("fill value version {v}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Sizes = Sizes::DEFAULT;

    #[test]
    fn old_message() {
        let v = [4, 0, 0, 0, 1, 2, 3, 4];
        assert_eq!(OldFillValue::parse(&v, S).unwrap().value, vec![1, 2, 3, 4]);
        assert!(OldFillValue::parse(&v[..6], S).is_err());
    }

    #[test]
    fn versions_1_and_2() {
        let v = [1, 1, 0, 0, 0, 0, 0, 0];
        let f = FillValue::parse(&v, S).unwrap();
        assert_eq!(f.alloc_time, AllocTime::Early);
        assert_eq!(f.state, FillValueState::Default);
        assert_eq!(f.value, None);

        let v = [2, 2, 2, 0];
        let f = FillValue::parse(&v, S).unwrap();
        assert_eq!(f.write_time, FillWriteTime::IfSet);
        assert_eq!(f.value, None);

        let v = [2, 3, 1, 1, 2, 0, 0, 0, 0xab, 0xcd];
        let f = FillValue::parse(&v, S).unwrap();
        assert_eq!(f.alloc_time, AllocTime::Incremental);
        assert_eq!(f.write_time, FillWriteTime::Never);
        assert_eq!(f.state, FillValueState::Defined);
        assert_eq!(f.value, Some(vec![0xab, 0xcd]));
    }

    #[test]
    fn version_3_flags() {
        // late alloc (2), write if set (2 << 2), undefined (bit 4)
        let f = FillValue::parse(&[3, 0x1a], S).unwrap();
        assert_eq!(f.alloc_time, AllocTime::Late);
        assert_eq!(f.write_time, FillWriteTime::IfSet);
        assert_eq!(f.state, FillValueState::Undefined);
        // defined with 4-byte value
        let f = FillValue::parse(&[3, 0x21, 4, 0, 0, 0, 1, 0, 0, 0], S).unwrap();
        assert_eq!(f.state, FillValueState::Defined);
        assert_eq!(f.value, Some(vec![1, 0, 0, 0]));
        let f = FillValue::parse(&[3, 0x02], S).unwrap();
        assert_eq!(f.state, FillValueState::Default);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(FillValue::parse(&[], S).is_err());
        assert!(FillValue::parse(&[4, 0], S).is_err());
        assert!(FillValue::parse(&[3, 0x30], S).is_err());
        assert!(FillValue::parse(&[3, 0x40], S).is_err());
        assert!(FillValue::parse(&[3, 0x0c], S).is_err()); // write time 3
        assert!(FillValue::parse(&[2, 4, 0, 0], S).is_err());
        assert!(FillValue::parse(&[2, 1, 0, 1, 0xff, 0xff, 0xff, 0xff], S).is_err());
        assert!(FillValue::parse(&[1, 1, 0, 0], S).is_err());
    }

    #[test]
    fn mutation_never_panics() {
        let samples: [&[u8]; 3] = [
            &[3, 0x21, 4, 0, 0, 0, 1, 0, 0, 0],
            &[2, 3, 1, 1, 2, 0, 0, 0, 0xab, 0xcd],
            &[1, 1, 0, 0, 0, 0, 0, 0],
        ];
        let mut state = 0x0123_4567_89ab_cdefu64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..10_000 {
            let mut v = samples[(rnd() % 3) as usize].to_vec();
            let i = (rnd() % v.len() as u64) as usize;
            v[i] = rnd() as u8;
            v.truncate((rnd() % (v.len() as u64 + 1)) as usize);
            let _ = FillValue::parse(&v, S);
            let _ = OldFillValue::parse(&v, S);
        }
    }
}
