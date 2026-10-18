//! Virtual dataset mappings: the "Virtual Dataset Global Heap Block"
//! referenced by a version 4 data layout message of class 3.
//!
//! Layout (spec, Disk Format Level 1F): version (1 byte), number of entries
//! ("Size of Lengths" bytes), then per entry the source file name and
//! source dataset name (null-terminated strings), the source selection and
//! the virtual selection (dataspace selection encodings), and finally a
//! 4-byte Jenkins lookup3 checksum of everything before it.
//!
//! Version 0 is the documented layout. Version 1 (written by recent
//! versions of the reference library, not described by the public
//! specification) adds a flags byte before each entry. Its meaning was
//! inferred from black-box observation of files:
//!
//! - `0x04`: the source is in the same file (`"."`); no file name stored;
//! - `0x01`: instead of the file name, a "Size of Lengths" index of an
//!   earlier entry whose file name is reused;
//! - `0x02`: likewise for the source dataset name.
//!
//! Other flags are rejected as unsupported.

use crate::checksum::lookup3;
use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};
use crate::types::Selection;

/// Entry flag (version 1): the file name is shared with an earlier entry.
const FLAG_SHARED_FILE: u8 = 0x01;
/// Entry flag (version 1): the dataset name is shared with an earlier entry.
const FLAG_SHARED_DATASET: u8 = 0x02;
/// Entry flag (version 1): the source dataset is in the same file.
const FLAG_SAME_FILE: u8 = 0x04;

/// Source file name meaning "the file holding the virtual dataset".
pub const SAME_FILE: &str = ".";

/// One mapping of a virtual dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualMapping {
    /// Name of the file holding the source dataset ([`SAME_FILE`] for the
    /// virtual dataset's own file).
    pub source_file: String,
    /// Path of the source dataset within that file.
    pub source_dataset: String,
    /// Selection within the source dataset.
    pub source_selection: Selection,
    /// Selection within the virtual dataset receiving the source elements.
    pub virtual_selection: Selection,
}

/// Parse a virtual dataset global heap block.
pub fn parse_virtual_mappings(data: &[u8], sizes: Sizes) -> Result<Vec<VirtualMapping>> {
    let mut r = ByteReader::with_sizes(data, sizes);
    let version = r.read_u8()?;
    if version > 1 {
        return Err(Error::unsupported(format!(
            "virtual dataset global heap block version {version}"
        )));
    }
    let count = r.read_length()?;
    // Each entry takes at least a few bytes: reject absurd counts early.
    if count > r.remaining() as u64 {
        return Err(Error::invalid(
            "virtual dataset mapping count exceeds block",
        ));
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let flags = if version >= 1 { r.read_u8()? } else { 0 };
        if flags & !(FLAG_SAME_FILE | FLAG_SHARED_FILE | FLAG_SHARED_DATASET) != 0 {
            return Err(Error::unsupported(format!(
                "virtual dataset mapping flags {flags:#04x}"
            )));
        }
        let source_file = if flags & FLAG_SAME_FILE != 0 {
            SAME_FILE.to_owned()
        } else if flags & FLAG_SHARED_FILE != 0 {
            shared(&mut r, &out)?.source_file.clone()
        } else {
            cstr(&mut r)?
        };
        let source_dataset = if flags & FLAG_SHARED_DATASET != 0 {
            shared(&mut r, &out)?.source_dataset.clone()
        } else {
            cstr(&mut r)?
        };
        let source_selection = selection(&mut r)?;
        let virtual_selection = selection(&mut r)?;
        out.push(VirtualMapping {
            source_file,
            source_dataset,
            source_selection,
            virtual_selection,
        });
    }
    let end = r.position();
    let stored = r.read_u32()?;
    let computed = lookup3(data.get(..end).ok_or(Error::UnexpectedEof)?, 0);
    if crate::checksum::VERIFY && stored != computed {
        return Err(Error::ChecksumMismatch {
            what: "virtual dataset global heap block",
            expected: stored,
            actual: computed,
        });
    }
    Ok(out)
}

/// The earlier entry referenced by a shared-name index.
fn shared<'m>(r: &mut ByteReader<'_>, earlier: &'m [VirtualMapping]) -> Result<&'m VirtualMapping> {
    let index = r.read_length()?;
    usize::try_from(index)
        .ok()
        .and_then(|i| earlier.get(i))
        .ok_or_else(|| {
            Error::invalid(format!(
                "virtual dataset mapping refers to entry {index}, not an earlier one"
            ))
        })
}

fn cstr(r: &mut ByteReader<'_>) -> Result<String> {
    let rest = r.rest();
    let len = rest
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| Error::invalid("unterminated name in virtual dataset mapping"))?;
    let s = String::from_utf8_lossy(&rest[..len]).into_owned();
    r.skip(len + 1)?;
    Ok(s)
}

fn selection(r: &mut ByteReader<'_>) -> Result<Selection> {
    let (sel, used) = Selection::parse_prefix(r.rest())?;
    r.skip(used)?;
    Ok(sel)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(version: u8, entries: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut v = vec![version];
        v.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (head, sels) in entries {
            v.extend_from_slice(head);
            v.extend_from_slice(sels);
        }
        let c = lookup3(&v, 0);
        v.extend_from_slice(&c.to_le_bytes());
        v
    }

    const ALL: [u8; 16] = [3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    fn regular_1d(start: u16, block_len: u16) -> Vec<u8> {
        let mut v = vec![2, 0, 0, 0, 3, 0, 0, 0, 1, 2, 1, 0, 0, 0];
        for x in [start, 1, 1, block_len] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    #[test]
    fn parses_version_1() {
        let mut sels = ALL.to_vec();
        sels.extend(regular_1d(10, 10));
        let data = block(
            1,
            &[(b"\x04local_src\0", &sels), (b"\0other.h5\0src\0", &sels)],
        );
        let m = parse_virtual_mappings(&data, Sizes::DEFAULT).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].source_file, ".");
        assert_eq!(m[0].source_dataset, "local_src");
        assert_eq!(m[0].source_selection, Selection::All);
        assert_eq!(
            m[0].virtual_selection,
            Selection::RegularHyperslab {
                start: vec![10],
                stride: vec![1],
                count: vec![1],
                block: vec![10],
            }
        );
        assert_eq!(m[1].source_file, "other.h5");
        assert_eq!(m[1].source_dataset, "src");
    }

    #[test]
    fn shared_names() {
        let mut sels = ALL.to_vec();
        sels.extend_from_slice(&ALL);
        let data = block(
            1,
            &[
                (b"\0f.h5\0d1\0", &sels),
                (b"\x01\0\0\0\0\0\0\0\0d2\0", &sels),
                (b"\x06\0\0\0\0\0\0\0\0", &sels),
            ],
        );
        let m = parse_virtual_mappings(&data, Sizes::DEFAULT).unwrap();
        assert_eq!(
            (m[1].source_file.as_str(), m[1].source_dataset.as_str()),
            ("f.h5", "d2")
        );
        assert_eq!(
            (m[2].source_file.as_str(), m[2].source_dataset.as_str()),
            (".", "d1")
        );
    }

    #[test]
    fn parses_version_0() {
        let mut sels = ALL.to_vec();
        sels.extend_from_slice(&ALL);
        let data = block(0, &[(b"f.h5\0/d\0", &sels)]);
        let m = parse_virtual_mappings(&data, Sizes::DEFAULT).unwrap();
        assert_eq!(m[0].source_file, "f.h5");
        assert_eq!(m[0].source_dataset, "/d");
    }

    #[test]
    fn rejects_bad_blocks() {
        let mut sels = ALL.to_vec();
        sels.extend_from_slice(&ALL);
        let mut data = block(1, &[(b"\x04d\0", &sels)]);
        let n = data.len();
        data[n - 1] ^= 1;
        assert!(parse_virtual_mappings(&data, Sizes::DEFAULT).is_err());
        let data = block(1, &[(b"\x08d\0", &sels)]);
        assert!(matches!(
            parse_virtual_mappings(&data, Sizes::DEFAULT),
            Err(Error::Unsupported(_))
        ));
        // Shared name referring to a later entry.
        let data = block(1, &[(b"\x06\0\0\0\0\0\0\0\0", &sels)]);
        assert!(parse_virtual_mappings(&data, Sizes::DEFAULT).is_err());
        let data = block(2, &[]);
        assert!(parse_virtual_mappings(&data, Sizes::DEFAULT).is_err());
        // Truncations never panic.
        let good = block(1, &[(b"\x04d\0", &sels)]);
        for i in 0..good.len() {
            assert!(parse_virtual_mappings(&good[..i], Sizes::DEFAULT).is_err());
        }
        // Huge entry count.
        let mut v = vec![0];
        v.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse_virtual_mappings(&v, Sizes::DEFAULT).is_err());
    }
}
