//! The superblock (spec: "Disk Format: Level 0A - Format Signature and
//! Superblock") and the optional driver information block ("Level 0B").

use super::{ByteReader, ByteWriter, Sizes, SymbolTableEntry, UNDEFINED_ADDRESS};
use crate::checksum::metadata_checksum;
use crate::error::{Error, Result};
use crate::io::{OffsetReader, ReadAt};

/// First non-zero offset searched for the signature; later candidates
/// double each time (512, 1024, 2048, …).
const FIRST_USER_BLOCK_OFFSET: u64 = 512;

/// Upper bound on the bytes a v0/v1 superblock can occupy (16-byte offsets).
const MAX_V01_SIZE: u64 = 8 + 16 + 4 + 4 * 16 + (2 * 16 + 24);

/// Upper bound on the driver information payload we are willing to load.
const MAX_DRIVER_INFO: u64 = 1 << 20;

/// Where the root group is described.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootGroup {
    /// Versions 0/1: the root group's symbol table entry.
    SymbolTableEntry(SymbolTableEntry),
    /// Versions 2/3: the (relative) address of the root group object header.
    ObjectHeader(u64),
}

/// A version 0 driver information block (spec: "Level 0B - File Driver
/// Info"). Parsed for completeness; the reader does not act on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverInfoBlock {
    /// Block version (only 0 is defined).
    pub version: u8,
    /// Eight-byte ASCII driver identification, e.g. `NCSAmult`/`NCSAfami`.
    pub driver_id: [u8; 8],
    /// Driver-defined payload.
    pub info: Vec<u8>,
}

impl DriverInfoBlock {
    /// The driver identification as text (lossy).
    pub fn driver_id_str(&self) -> String {
        String::from_utf8_lossy(&self.driver_id).into_owned()
    }

    /// Parse a driver information block from the start of `data`.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "driver information block version {version}"
            )));
        }
        r.skip(3)?;
        let size = r.read_u32()? as usize;
        let driver_id = r.read_array::<8>()?;
        let info = r.read_bytes(size)?.to_vec();
        Ok(DriverInfoBlock {
            version,
            driver_id,
            info,
        })
    }
}

/// A parsed superblock, any version 0–3.
///
/// Addresses are stored exactly as encoded (relative to the base address,
/// undefined = [`UNDEFINED_ADDRESS`]). Use [`Superblock::absolute`] or
/// [`Superblock::relative_store`] to resolve them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Superblock {
    /// Superblock version (0, 1, 2 or 3).
    pub version: u8,
    /// Absolute offset of the signature within the store.
    pub superblock_offset: u64,
    /// Offset/length widths used throughout the file.
    pub sizes: Sizes,
    /// Base address as encoded in the superblock.
    pub base_address: u64,
    /// Version of the free-space storage format (v0/1 only).
    pub free_space_version: Option<u8>,
    /// Version of the root group symbol table entry format (v0/1 only).
    pub root_entry_version: Option<u8>,
    /// Version of the shared header message format (v0/1 only).
    pub shared_header_version: Option<u8>,
    /// Group leaf node K (v0/1 only; v2+ uses the B-tree 'K' message).
    pub group_leaf_k: Option<u16>,
    /// Group internal node K (v0/1 only).
    pub group_internal_k: Option<u16>,
    /// Indexed storage internal node K (v1 only).
    pub indexed_storage_k: Option<u16>,
    /// File consistency flags (4 bytes in v0/1, 1 byte in v2/3).
    pub consistency_flags: u32,
    /// Address of global free-space info (v0/1; normally undefined).
    pub free_space_address: u64,
    /// Superblock extension object header address (v2/3), else undefined.
    pub extension_address: u64,
    /// End-of-file address (relative to the base address, see
    /// [`Superblock::absolute`]).
    pub eof_address: u64,
    /// Driver information block address (v0/1), else undefined.
    pub driver_info_address: u64,
    /// The parsed driver information block, if present (v0/1).
    pub driver_info: Option<DriverInfoBlock>,
    /// Root group location.
    pub root: RootGroup,
}

impl Superblock {
    /// Find the superblock signature: offset 0, then 512, 1024, 2048, …
    /// while the signature still fits in the store.
    pub fn locate<R: ReadAt + ?Sized>(store: &R) -> Result<u64> {
        let size = store.size()?;
        let mut next = Some(0u64);
        while let Some(pos) = next {
            if pos.checked_add(8).is_none_or(|end| end > size) {
                break;
            }
            let mut sig = [0u8; 8];
            store.read_exact_at(pos, &mut sig)?;
            if sig == crate::SIGNATURE {
                return Ok(pos);
            }
            next = if pos == 0 {
                Some(FIRST_USER_BLOCK_OFFSET)
            } else {
                pos.checked_mul(2)
            };
        }
        Err(Error::invalid("HDF5 signature not found"))
    }

    /// Locate and parse the superblock (and v0/1 driver information block).
    pub fn read<R: ReadAt + ?Sized>(store: &R) -> Result<Self> {
        let offset = Self::locate(store)?;
        Self::read_at(store, offset)
    }

    /// Parse the superblock whose signature is at absolute `offset`.
    pub fn read_at<R: ReadAt + ?Sized>(store: &R, offset: u64) -> Result<Self> {
        let size = store.size()?;
        let avail = size.checked_sub(offset).ok_or(Error::UnexpectedEof)?;
        let len = avail.min(MAX_V01_SIZE) as usize;
        let data = store.read_vec_at(offset, len)?;
        let mut sb = Self::parse(&data, offset)?;
        if sb.driver_info_address != UNDEFINED_ADDRESS {
            let abs = sb
                .absolute(sb.driver_info_address)
                .ok_or(Error::invalid("driver info block address overflows"))?;
            let avail = size.checked_sub(abs).ok_or(Error::UnexpectedEof)?;
            let len = avail.min(16 + MAX_DRIVER_INFO) as usize;
            let block = store.read_vec_at(abs, len)?;
            sb.driver_info = Some(DriverInfoBlock::parse(&block)?);
        }
        Ok(sb)
    }

    /// Parse a superblock from `data`, which starts at the signature located
    /// at absolute offset `superblock_offset`. The driver information block
    /// is not loaded (see [`Superblock::read_at`]).
    pub fn parse(data: &[u8], superblock_offset: u64) -> Result<Self> {
        let mut r = ByteReader::new(data);
        if r.read_array::<8>()? != crate::SIGNATURE {
            return Err(Error::invalid("bad HDF5 signature"));
        }
        let version = r.read_u8()?;
        match version {
            0 | 1 => Self::parse_v01(r, version, superblock_offset),
            2 | 3 => Self::parse_v23(r, version, superblock_offset),
            v => Err(Error::unsupported(format!("superblock version {v}"))),
        }
    }

    fn parse_v01(mut r: ByteReader<'_>, version: u8, superblock_offset: u64) -> Result<Self> {
        let free_space_version = r.read_u8()?;
        let root_entry_version = r.read_u8()?;
        r.skip(1)?;
        let shared_header_version = r.read_u8()?;
        let sizes = parse_sizes(r.read_u8()?, r.read_u8()?)?;
        r.skip(1)?;
        let group_leaf_k = r.read_u16()?;
        let group_internal_k = r.read_u16()?;
        if group_leaf_k == 0 || group_internal_k == 0 {
            return Err(Error::invalid("group B-tree K must be non-zero"));
        }
        let consistency_flags = r.read_u32()?;
        let indexed_storage_k = if version == 1 {
            let k = r.read_u16()?;
            r.skip(2)?;
            if k == 0 {
                return Err(Error::invalid("indexed storage K must be non-zero"));
            }
            Some(k)
        } else {
            None
        };
        let mut r = ByteReader::with_sizes(r.rest(), sizes);
        let base_address = r.read_offset()?;
        let free_space_address = r.read_offset()?;
        let eof_address = r.read_offset()?;
        let driver_info_address = r.read_offset()?;
        let root = SymbolTableEntry::parse(&mut r)?;
        Ok(Superblock {
            version,
            superblock_offset,
            sizes,
            base_address,
            free_space_version: Some(free_space_version),
            root_entry_version: Some(root_entry_version),
            shared_header_version: Some(shared_header_version),
            group_leaf_k: Some(group_leaf_k),
            group_internal_k: Some(group_internal_k),
            indexed_storage_k,
            consistency_flags,
            free_space_address,
            extension_address: UNDEFINED_ADDRESS,
            eof_address,
            driver_info_address,
            driver_info: None,
            root: RootGroup::SymbolTableEntry(root),
        })
    }

    fn parse_v23(mut r: ByteReader<'_>, version: u8, superblock_offset: u64) -> Result<Self> {
        let sizes = parse_sizes(r.read_u8()?, r.read_u8()?)?;
        let consistency_flags = u32::from(r.read_u8()?);
        let start = r.position();
        let mut r = ByteReader::with_sizes(r.data(), sizes);
        r.seek(start)?;
        let base_address = r.read_offset()?;
        let extension_address = r.read_offset()?;
        let eof_address = r.read_offset()?;
        let root_address = r.read_offset()?;
        let covered = r.position();
        let stored = r.read_u32()?;
        let computed = metadata_checksum(r.data().get(..covered).ok_or(Error::UnexpectedEof)?);
        if stored != computed {
            return Err(Error::ChecksumMismatch {
                what: "superblock",
                expected: stored,
                actual: computed,
            });
        }
        Ok(Superblock {
            version,
            superblock_offset,
            sizes,
            base_address,
            free_space_version: None,
            root_entry_version: None,
            shared_header_version: None,
            group_leaf_k: None,
            group_internal_k: None,
            indexed_storage_k: None,
            consistency_flags,
            free_space_address: UNDEFINED_ADDRESS,
            extension_address,
            eof_address,
            driver_info_address: UNDEFINED_ADDRESS,
            driver_info: None,
            root: RootGroup::ObjectHeader(root_address),
        })
    }

    /// The effective base address: the absolute offset that relative file
    /// addresses are measured from.
    ///
    /// The spec says that when the encoded base address differs from the
    /// superblock's location, the contents are assumed to have been moved
    /// and the base is adjusted accordingly, so this is always the
    /// signature's location.
    pub fn base(&self) -> u64 {
        self.superblock_offset
    }

    /// Convert a relative file address to an absolute store offset.
    ///
    /// Returns `None` for [`UNDEFINED_ADDRESS`] or on overflow.
    pub fn absolute(&self, relative: u64) -> Option<u64> {
        if relative == UNDEFINED_ADDRESS {
            return None;
        }
        self.base().checked_add(relative)
    }

    /// Wrap `store` so that positions are relative file addresses, which is
    /// what [`ObjectHeader::read`](super::object_header::ObjectHeader::read)
    /// and other structure parsers expect.
    pub fn relative_store<R: ReadAt>(&self, store: R) -> OffsetReader<R> {
        OffsetReader::new(store, self.base())
    }

    /// The (relative) address of the root group's object header.
    pub fn root_object_header_address(&self) -> u64 {
        match self.root {
            RootGroup::SymbolTableEntry(e) => e.object_header_address,
            RootGroup::ObjectHeader(a) => a,
        }
    }
}

impl Superblock {
    /// A version 2 or 3 superblock at offset 0 with no extension, as
    /// written by a fresh file writer.
    pub fn new_v23(version: u8, sizes: Sizes, root_address: u64, eof_address: u64) -> Superblock {
        Superblock {
            version,
            superblock_offset: 0,
            sizes,
            base_address: 0,
            free_space_version: None,
            root_entry_version: None,
            shared_header_version: None,
            group_leaf_k: None,
            group_internal_k: None,
            indexed_storage_k: None,
            consistency_flags: 0,
            free_space_address: UNDEFINED_ADDRESS,
            extension_address: UNDEFINED_ADDRESS,
            eof_address,
            driver_info_address: UNDEFINED_ADDRESS,
            driver_info: None,
            root: RootGroup::ObjectHeader(root_address),
        }
    }

    /// Encoded size of a version 2/3 superblock with the given widths.
    pub fn encoded_size_v23(sizes: Sizes) -> usize {
        8 + 1 + 2 + 1 + 4 * sizes.offset as usize + 4
    }

    /// A version 0 superblock at offset 0 with the default B-tree K values
    /// (group leaf 4, group internal 16) and no driver information.
    pub fn new_v0(sizes: Sizes, root: SymbolTableEntry, eof_address: u64) -> Superblock {
        Superblock {
            version: 0,
            superblock_offset: 0,
            sizes,
            base_address: 0,
            free_space_version: Some(0),
            root_entry_version: Some(0),
            shared_header_version: Some(0),
            group_leaf_k: Some(4),
            group_internal_k: Some(16),
            indexed_storage_k: None,
            consistency_flags: 0,
            free_space_address: UNDEFINED_ADDRESS,
            extension_address: UNDEFINED_ADDRESS,
            eof_address,
            driver_info_address: UNDEFINED_ADDRESS,
            driver_info: None,
            root: RootGroup::SymbolTableEntry(root),
        }
    }

    /// Encoded size of a version 0 superblock with the given widths.
    pub fn encoded_size_v0(sizes: Sizes) -> usize {
        8 + 16 + 4 * sizes.offset as usize + SymbolTableEntry::encoded_size(sizes.offset)
    }

    fn encode_v01(&self) -> Result<Vec<u8>> {
        let RootGroup::SymbolTableEntry(root) = self.root else {
            return Err(Error::invalid(
                "version 0/1 superblock needs a root symbol table entry",
            ));
        };
        let leaf_k = self.group_leaf_k.unwrap_or(4);
        let internal_k = self.group_internal_k.unwrap_or(16);
        if leaf_k == 0 || internal_k == 0 {
            return Err(Error::invalid("group B-tree K must be non-zero"));
        }
        let mut w = ByteWriter::with_sizes(self.sizes);
        w.bytes(&crate::SIGNATURE)
            .u8(self.version)
            .u8(self.free_space_version.unwrap_or(0))
            .u8(self.root_entry_version.unwrap_or(0))
            .u8(0)
            .u8(self.shared_header_version.unwrap_or(0))
            .u8(self.sizes.offset)
            .u8(self.sizes.length)
            .u8(0)
            .u16(leaf_k)
            .u16(internal_k)
            .u32(self.consistency_flags);
        if self.version == 1 {
            let k = self
                .indexed_storage_k
                .filter(|&k| k > 0)
                .ok_or_else(|| Error::invalid("superblock v1 needs indexed storage K"))?;
            w.u16(k).u16(0);
        }
        w.write_offset(self.base_address)?;
        w.write_offset(self.free_space_address)?;
        w.write_offset(self.eof_address)?;
        w.write_offset(self.driver_info_address)?;
        root.encode(&mut w)?;
        Ok(w.into_vec())
    }

    /// Encode the superblock, signature through checksum (versions 2/3)
    /// or root symbol table entry (versions 0/1). A driver information
    /// block is not written.
    pub fn encode(&self) -> Result<Vec<u8>> {
        parse_sizes(self.sizes.offset, self.sizes.length)?;
        match self.version {
            0 | 1 => return self.encode_v01(),
            2 | 3 => {}
            v => {
                return Err(Error::unsupported(format!(
                    "encoding superblock version {v}"
                )));
            }
        }
        let RootGroup::ObjectHeader(root) = self.root else {
            return Err(Error::invalid(
                "version 2/3 superblock needs a root object header address",
            ));
        };
        let flags = u8::try_from(self.consistency_flags)
            .map_err(|_| Error::invalid("consistency flags do not fit one byte"))?;
        let mut w = ByteWriter::with_sizes(self.sizes);
        w.bytes(&crate::SIGNATURE)
            .u8(self.version)
            .u8(self.sizes.offset)
            .u8(self.sizes.length)
            .u8(flags);
        w.write_offset(self.base_address)?;
        w.write_offset(self.extension_address)?;
        w.write_offset(self.eof_address)?;
        w.write_offset(root)?;
        let sum = metadata_checksum(w.as_slice());
        w.u32(sum);
        Ok(w.into_vec())
    }
}

fn parse_sizes(offset: u8, length: u8) -> Result<Sizes> {
    let ok = |v: u8| matches!(v, 2 | 4 | 8 | 16);
    if !ok(offset) {
        return Err(Error::invalid(format!(
            "unsupported size of offsets {offset}"
        )));
    }
    if !ok(length) {
        return Err(Error::invalid(format!(
            "unsupported size of lengths {length}"
        )));
    }
    Ok(Sizes { offset, length })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::ScratchPad;
    use crate::format::test_util::{Rng, W, mutate};

    fn v0_bytes(version: u8, driver_info: u64) -> Vec<u8> {
        let mut w = W::new();
        w.bytes(&crate::SIGNATURE)
            .u8(version)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(8)
            .u8(8)
            .u8(0)
            .u16(4)
            .u16(16)
            .u32(0);
        if version == 1 {
            w.u16(32).u16(0);
        }
        w.offset(0) // base
            .offset(u64::MAX) // free space
            .offset(4096) // eof
            .offset(driver_info);
        // root symbol table entry
        w.offset(0).offset(96).u32(1).u32(0).offset(136).offset(680);
        w.buf
    }

    fn v2_bytes(version: u8) -> Vec<u8> {
        let mut w = W::new();
        w.bytes(&crate::SIGNATURE)
            .u8(version)
            .u8(8)
            .u8(8)
            .u8(0)
            .offset(0)
            .offset(u64::MAX)
            .offset(2048)
            .offset(48);
        let c = metadata_checksum(&w.buf);
        w.u32(c);
        w.buf
    }

    #[test]
    fn parse_v0() {
        let data = v0_bytes(0, u64::MAX);
        let sb = Superblock::read(&data).unwrap();
        assert_eq!(sb.version, 0);
        assert_eq!(sb.sizes, Sizes::DEFAULT);
        assert_eq!(sb.group_leaf_k, Some(4));
        assert_eq!(sb.group_internal_k, Some(16));
        assert_eq!(sb.indexed_storage_k, None);
        assert_eq!(sb.eof_address, 4096);
        assert_eq!(sb.free_space_address, UNDEFINED_ADDRESS);
        assert_eq!(sb.root_object_header_address(), 96);
        match sb.root {
            RootGroup::SymbolTableEntry(e) => assert_eq!(
                e.scratch,
                ScratchPad::Group {
                    btree_address: 136,
                    heap_address: 680
                }
            ),
            _ => panic!("expected symbol table entry"),
        }
        assert!(sb.driver_info.is_none());
    }

    #[test]
    fn parse_v1_with_driver_info() {
        let mut data = v0_bytes(1, 0);
        let sb_len = data.len() as u64;
        // Patch driver info address to point just past the superblock.
        let pos = 8 + 16 + 4 + 3 * 8;
        data[pos..pos + 8].copy_from_slice(&sb_len.to_le_bytes());
        let mut w = W::new();
        w.u8(0).zeros(3).u32(8).bytes(b"NCSAfami").u64(1 << 30);
        data.extend_from_slice(&w.buf);
        let sb = Superblock::read(&data).unwrap();
        assert_eq!(sb.version, 1);
        assert_eq!(sb.indexed_storage_k, Some(32));
        let di = sb.driver_info.unwrap();
        assert_eq!(di.driver_id_str(), "NCSAfami");
        assert_eq!(di.info, (1u64 << 30).to_le_bytes());
    }

    #[test]
    fn parse_v2_v3_and_checksum() {
        for v in [2, 3] {
            let data = v2_bytes(v);
            let sb = Superblock::read(&data).unwrap();
            assert_eq!(sb.version, v);
            assert_eq!(sb.root, RootGroup::ObjectHeader(48));
            assert_eq!(sb.extension_address, UNDEFINED_ADDRESS);
            assert_eq!(sb.eof_address, 2048);
        }
        let mut data = v2_bytes(2);
        data[20] ^= 1;
        assert!(matches!(
            Superblock::read(&data),
            Err(Error::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn user_block() {
        let mut data = vec![0u8; 1024];
        data.extend_from_slice(&v2_bytes(2));
        let sb = Superblock::read(&data).unwrap();
        assert_eq!(sb.superblock_offset, 1024);
        assert_eq!(sb.base(), 1024);
        assert_eq!(sb.absolute(48), Some(1072));
        assert_eq!(sb.absolute(UNDEFINED_ADDRESS), None);
        let rel = sb.relative_store(&data);
        assert_eq!(rel.size().unwrap(), data.len() as u64 - 1024);
        // Signature at a non-candidate offset is not found.
        let mut data = vec![0u8; 700];
        data.extend_from_slice(&v2_bytes(2));
        assert!(Superblock::read(&data).is_err());
    }

    #[test]
    fn truncated_and_bad() {
        let data = v0_bytes(0, u64::MAX);
        for n in 0..data.len() {
            assert!(Superblock::read(&data[..n]).is_err(), "len {n}");
        }
        let data = v2_bytes(3);
        for n in 0..data.len() {
            assert!(Superblock::read(&data[..n]).is_err(), "len {n}");
        }
        let mut data = v0_bytes(0, u64::MAX);
        data[8] = 7;
        assert!(matches!(
            Superblock::read(&data),
            Err(Error::Unsupported(_))
        ));
        let mut data = v0_bytes(0, u64::MAX);
        data[13] = 3; // size of offsets
        assert!(Superblock::read(&data).is_err());
        let mut data = v0_bytes(0, u64::MAX);
        data[16] = 0; // group leaf K = 0
        data[17] = 0;
        assert!(Superblock::read(&data).is_err());
    }

    #[test]
    fn small_offsets() {
        let mut w = W::with_sizes(Sizes {
            offset: 4,
            length: 2,
        });
        w.bytes(&crate::SIGNATURE)
            .u8(0)
            .zeros(4)
            .u8(4)
            .u8(2)
            .u8(0)
            .u16(4)
            .u16(16)
            .u32(0)
            .offset(0)
            .offset(u64::MAX)
            .offset(500)
            .offset(u64::MAX)
            .offset(0)
            .offset(60)
            .u32(0)
            .u32(0)
            .zeros(16);
        let sb = Superblock::read(&w.buf).unwrap();
        assert_eq!(sb.sizes.offset, 4);
        assert_eq!(sb.root_object_header_address(), 60);
    }

    #[test]
    fn encode_round_trip() {
        for v in [2, 3] {
            for sizes in [
                Sizes::DEFAULT,
                Sizes {
                    offset: 4,
                    length: 2,
                },
            ] {
                let sb = Superblock::new_v23(v, sizes, 48, 4096);
                let bytes = sb.encode().unwrap();
                assert_eq!(bytes.len(), Superblock::encoded_size_v23(sizes));
                assert_eq!(Superblock::read(&bytes).unwrap(), sb);
            }
        }
        assert_eq!(
            v2_bytes(2),
            Superblock::read(&v2_bytes(2)).unwrap().encode().unwrap()
        );
        for v in [0, 1] {
            let bytes = v0_bytes(v, u64::MAX);
            let sb = Superblock::read(&bytes).unwrap();
            assert_eq!(sb.encode().unwrap(), bytes);
        }
        let root = SymbolTableEntry {
            link_name_offset: 0,
            object_header_address: 96,
            scratch: ScratchPad::Group {
                btree_address: 136,
                heap_address: 680,
            },
        };
        let sb = Superblock::new_v0(Sizes::DEFAULT, root, 1000);
        let enc = sb.encode().unwrap();
        assert_eq!(enc.len(), Superblock::encoded_size_v0(Sizes::DEFAULT));
        assert_eq!(Superblock::read(&enc).unwrap(), sb);
        let mut bad = sb.clone();
        bad.root = RootGroup::ObjectHeader(5);
        assert!(bad.encode().is_err());
    }

    #[test]
    fn mutation_never_panics() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let seeds = [v0_bytes(0, u64::MAX), v0_bytes(1, u64::MAX), v2_bytes(2)];
        for seed in &seeds {
            for _ in 0..5000 {
                let m = mutate(&mut rng, seed);
                let _ = Superblock::read(&m);
            }
        }
    }
}
