//! Dataspace message (type 0x0001).
//!
//! Spec: "IV.A.2.b. The Dataspace Message", versions 1 and 2.

use crate::error::{Error, Result};
use crate::format::{ByteReader, ByteWriter, Sizes};

/// Maximum dimension size meaning "unlimited" (all ones on disk, in any
/// length width).
pub const UNLIMITED: u64 = u64::MAX;

/// Shape class of a dataspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataspaceKind {
    /// A single, dimensionless element.
    Scalar,
    /// A rectilinear array of rank >= 1.
    Simple,
    /// No elements at all.
    Null,
}

/// A decoded dataspace message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dataspace {
    /// Encoding version (1 or 2).
    pub version: u8,
    /// Scalar, simple or null.
    pub kind: DataspaceKind,
    /// Current dimension sizes, slowest-changing first (empty unless simple).
    pub dims: Vec<u64>,
    /// Maximum dimension sizes if stored; [`UNLIMITED`] marks an unlimited
    /// dimension. When absent the maximum equals the current size.
    pub max_dims: Option<Vec<u64>>,
    /// Permutation indices (version 1 only, never used in practice).
    pub permutation: Option<Vec<u64>>,
}

/// Read one length-sized value, mapping the all-ones pattern to
/// [`UNLIMITED`].
fn read_size(r: &mut ByteReader<'_>) -> Result<u64> {
    let width = r.sizes().length as usize;
    let bytes = r.read_bytes(width)?;
    if !bytes.is_empty() && bytes.iter().all(|&b| b == 0xff) {
        return Ok(UNLIMITED);
    }
    ByteReader::new(bytes).read_uint(width)
}

impl Dataspace {
    /// Parse a dataspace message.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Dataspace> {
        let mut r = ByteReader::with_sizes(data, sizes);
        let version = r.read_u8()?;
        let rank = r.read_u8()? as usize;
        let flags = r.read_u8()?;
        let kind = match version {
            1 => {
                r.skip(5)?; // reserved byte + 4 reserved bytes
                if rank == 0 {
                    DataspaceKind::Scalar
                } else {
                    DataspaceKind::Simple
                }
            }
            2 => match r.read_u8()? {
                0 => DataspaceKind::Scalar,
                1 => DataspaceKind::Simple,
                2 => DataspaceKind::Null,
                t => return Err(Error::invalid(format!("unknown dataspace type {t}"))),
            },
            v => return Err(Error::unsupported(format!("dataspace version {v}"))),
        };
        if (kind == DataspaceKind::Simple) != (rank > 0) {
            return Err(Error::invalid("dataspace rank does not match its type"));
        }
        let read_list = |r: &mut ByteReader<'_>| -> Result<Vec<u64>> {
            let mut v = Vec::with_capacity(rank);
            for _ in 0..rank {
                v.push(read_size(r)?);
            }
            Ok(v)
        };
        let dims = read_list(&mut r)?;
        if dims.contains(&UNLIMITED) {
            return Err(Error::invalid("dataspace current size is unlimited"));
        }
        let max_dims = if flags & 1 != 0 {
            Some(read_list(&mut r)?)
        } else {
            None
        };
        let permutation = if version == 1 && flags & 2 != 0 {
            Some(read_list(&mut r)?)
        } else {
            None
        };
        Ok(Dataspace {
            version,
            kind,
            dims,
            max_dims,
            permutation,
        })
    }

    /// A version 2 scalar dataspace.
    pub fn scalar() -> Dataspace {
        Dataspace {
            version: 2,
            kind: DataspaceKind::Scalar,
            dims: Vec::new(),
            max_dims: None,
            permutation: None,
        }
    }

    /// A version 2 null (empty) dataspace.
    pub fn null() -> Dataspace {
        Dataspace {
            kind: DataspaceKind::Null,
            ..Dataspace::scalar()
        }
    }

    /// A version 2 simple dataspace; an empty `dims` gives a scalar.
    /// `max_dims` of `None` means fixed size.
    pub fn simple(dims: &[u64], max_dims: Option<&[u64]>) -> Dataspace {
        if dims.is_empty() {
            return Dataspace::scalar();
        }
        Dataspace {
            version: 2,
            kind: DataspaceKind::Simple,
            dims: dims.to_vec(),
            max_dims: max_dims.map(<[u64]>::to_vec),
            permutation: None,
        }
    }

    /// Encode the message body (version 1 or 2, as given by `version`).
    pub fn encode(&self, sizes: Sizes) -> Result<Vec<u8>> {
        let rank = u8::try_from(self.dims.len())
            .ok()
            .filter(|&r| r <= 32)
            .ok_or(Error::LimitExceeded("dataspace rank above 32"))?;
        if (self.kind == DataspaceKind::Simple) != (rank > 0) {
            return Err(Error::invalid("dataspace rank does not match its type"));
        }
        let check_len = |v: &Option<Vec<u64>>| match v {
            Some(m) if m.len() != self.dims.len() => Err(Error::invalid(
                "dataspace maximum/permutation rank mismatch",
            )),
            _ => Ok(()),
        };
        check_len(&self.max_dims)?;
        check_len(&self.permutation)?;
        if let Some(m) = &self.max_dims
            && m.iter().zip(&self.dims).any(|(&mx, &d)| mx < d)
        {
            return Err(Error::invalid("dataspace maximum below current size"));
        }
        let mut flags = u8::from(self.max_dims.is_some());
        let mut w = ByteWriter::with_sizes(sizes);
        match self.version {
            1 => {
                if self.kind == DataspaceKind::Null {
                    return Err(Error::invalid("null dataspace needs version 2"));
                }
                if self.permutation.is_some() {
                    flags |= 2;
                }
                w.u8(1).u8(rank).u8(flags).zeros(5);
            }
            2 => {
                if self.permutation.is_some() {
                    return Err(Error::invalid("dataspace version 2 has no permutation"));
                }
                let ty = match self.kind {
                    DataspaceKind::Scalar => 0,
                    DataspaceKind::Simple => 1,
                    DataspaceKind::Null => 2,
                };
                w.u8(2).u8(rank).u8(flags).u8(ty);
            }
            v => return Err(Error::unsupported(format!("dataspace version {v}"))),
        }
        for &d in &self.dims {
            if d == UNLIMITED {
                return Err(Error::invalid("dataspace current size is unlimited"));
            }
            w.write_length(d)?;
        }
        for &d in self.max_dims.iter().flatten() {
            w.write_length_or_unlimited(d)?;
        }
        for &d in self.permutation.iter().flatten() {
            w.write_length(d)?;
        }
        Ok(w.into_vec())
    }

    /// Number of dimensions (0 for scalar and null dataspaces).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements: 1 for scalar, 0 for null, the product of
    /// the dimensions otherwise. Overflow yields [`Error::LimitExceeded`].
    pub fn element_count(&self) -> Result<u64> {
        match self.kind {
            DataspaceKind::Null => Ok(0),
            DataspaceKind::Scalar => Ok(1),
            DataspaceKind::Simple => self.dims.iter().try_fold(1u64, |acc, &d| {
                acc.checked_mul(d)
                    .ok_or(Error::LimitExceeded("dataspace element count overflows"))
            }),
        }
    }

    /// Maximum size of dimension `i`: the stored maximum, or the current
    /// size when no maxima are stored. `None` if `i` is out of range.
    pub fn max_dim(&self, i: usize) -> Option<u64> {
        match &self.max_dims {
            Some(m) => m.get(i).copied(),
            None => self.dims.get(i).copied(),
        }
    }

    /// Whether dimension `i` is unlimited.
    pub fn is_unlimited(&self, i: usize) -> bool {
        self.max_dim(i) == Some(UNLIMITED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S4: Sizes = Sizes {
        offset: 4,
        length: 4,
    };

    #[test]
    fn v1_simple_with_max_and_permutation() {
        let mut v = vec![1, 2, 3, 0, 0, 0, 0, 0];
        for x in [10u64, 20, u64::MAX, 20, 0, 1] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        let ds = Dataspace::parse(&v, Sizes::DEFAULT).unwrap();
        assert_eq!(ds.kind, DataspaceKind::Simple);
        assert_eq!(ds.dims, vec![10, 20]);
        assert!(ds.is_unlimited(0));
        assert!(!ds.is_unlimited(1));
        assert_eq!(ds.permutation, Some(vec![0, 1]));
        assert_eq!(ds.element_count().unwrap(), 200);
    }

    #[test]
    fn v1_scalar() {
        let ds = Dataspace::parse(&[1, 0, 0, 0, 0, 0, 0, 0], Sizes::DEFAULT).unwrap();
        assert_eq!(ds.kind, DataspaceKind::Scalar);
        assert_eq!(ds.rank(), 0);
        assert_eq!(ds.element_count().unwrap(), 1);
    }

    #[test]
    fn v2_kinds_and_narrow_lengths() {
        let mut v = vec![2, 1, 1, 1];
        v.extend_from_slice(&5u32.to_le_bytes());
        v.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
        let ds = Dataspace::parse(&v, S4).unwrap();
        assert_eq!(ds.dims, vec![5]);
        assert_eq!(ds.max_dims, Some(vec![UNLIMITED]));
        assert_eq!(ds.max_dim(0), Some(UNLIMITED));
        assert_eq!(ds.max_dim(1), None);

        let ds = Dataspace::parse(&[2, 0, 0, 2], S4).unwrap();
        assert_eq!(ds.kind, DataspaceKind::Null);
        assert_eq!(ds.element_count().unwrap(), 0);
        let ds = Dataspace::parse(&[2, 0, 0, 0], S4).unwrap();
        assert_eq!(ds.kind, DataspaceKind::Scalar);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(Dataspace::parse(&[], S4).is_err());
        assert!(Dataspace::parse(&[3, 0, 0, 0], S4).is_err());
        assert!(Dataspace::parse(&[2, 0, 0, 3], S4).is_err());
        assert!(Dataspace::parse(&[2, 1, 0, 0, 1, 0, 0, 0], S4).is_err()); // scalar with rank
        assert!(Dataspace::parse(&[2, 0, 0, 1], S4).is_err()); // simple without rank
        assert!(Dataspace::parse(&[2, 2, 0, 1, 1, 0, 0, 0], S4).is_err()); // truncated
        let mut v = vec![2, 1, 0, 1];
        v.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
        assert!(Dataspace::parse(&v, S4).is_err());
    }

    #[test]
    fn element_count_overflow() {
        let mut v = vec![2, 3, 0, 1];
        for _ in 0..3 {
            v.extend_from_slice(&(1u64 << 30).to_le_bytes());
        }
        let ds = Dataspace::parse(&v, Sizes::DEFAULT).unwrap();
        assert!(matches!(ds.element_count(), Err(Error::LimitExceeded(_))));
    }

    #[test]
    fn mutation_never_panics() {
        let mut a = vec![1, 2, 3, 0, 0, 0, 0, 0];
        for x in [10u64, 20, u64::MAX, 20, 0, 1] {
            a.extend_from_slice(&x.to_le_bytes());
        }
        let mut b = vec![2, 2, 1, 1];
        for x in [3u64, 4, 5, u64::MAX] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        let samples = [a, b];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..10_000 {
            let mut v = samples[(rnd() % 2) as usize].clone();
            let i = (rnd() % v.len() as u64) as usize;
            v[i] = rnd() as u8;
            v.truncate((rnd() % (v.len() as u64 + 1)) as usize);
            if let Ok(ds) = Dataspace::parse(&v, Sizes::DEFAULT) {
                let _ = ds.element_count();
            }
        }
    }

    #[test]
    fn encode_round_trip() {
        let cases = [
            Dataspace::scalar(),
            Dataspace::null(),
            Dataspace::simple(&[3, 4], None),
            Dataspace::simple(&[3, 0], Some(&[UNLIMITED, 10])),
            Dataspace {
                version: 1,
                kind: DataspaceKind::Simple,
                dims: vec![5],
                max_dims: Some(vec![7]),
                permutation: Some(vec![0]),
            },
            Dataspace {
                version: 1,
                ..Dataspace::scalar()
            },
        ];
        for sizes in [Sizes::DEFAULT, S4] {
            for ds in &cases {
                let enc = ds.encode(sizes).unwrap();
                assert_eq!(&Dataspace::parse(&enc, sizes).unwrap(), ds);
            }
        }
        assert!(Dataspace::simple(&[5], Some(&[4])).encode(S4).is_err());
        assert!(Dataspace::simple(&[1 << 40], None).encode(S4).is_err());
        let null_v1 = Dataspace {
            version: 1,
            ..Dataspace::null()
        };
        assert!(null_v1.encode(S4).is_err());
    }
}
