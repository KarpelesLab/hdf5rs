//! Data layout message (type 0x0008).
//!
//! Spec: "IV.A.2.i. The Data Layout Message", versions 1-5. Version 5
//! (file format specification 4.0) is encoded exactly like version 4; it
//! only changes the width of the "Chunk Size" field of filtered chunk
//! records in fixed array, extensible array and v2 B-tree indexes (see
//! [`ChunkedLayout::filtered_chunk_size_width`]).

use crate::error::{Error, Result};
use crate::format::{ByteReader, ByteWriter, Sizes};

/// Chunked layout flag (version 4): do not filter partial edge chunks.
pub const DONT_FILTER_PARTIAL_BOUND_CHUNKS: u8 = 0x01;
/// Chunked layout flag (version 4): the single chunk is filtered.
pub const SINGLE_INDEX_WITH_FILTER: u8 = 0x02;

/// Filter information of a filtered single-chunk index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilteredChunk {
    /// Size of the chunk after filtering, in bytes.
    pub size: u64,
    /// Filter mask: bit `n` set means filter `n` was skipped.
    pub filter_mask: u32,
}

/// How the chunks of a chunked dataset are located.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkIndex {
    /// Version 1 B-tree (layout versions 1-3).
    BTreeV1 {
        /// Address of the B-tree root.
        address: u64,
    },
    /// Single chunk index: the whole dataset is one chunk.
    SingleChunk {
        /// Address of the chunk.
        address: u64,
        /// Present when the chunk is filtered.
        filtered: Option<FilteredChunk>,
    },
    /// Implicit index: chunks stored contiguously in order.
    Implicit {
        /// Address of the first chunk.
        address: u64,
    },
    /// Fixed array index.
    FixedArray {
        /// Address of the fixed array header.
        address: u64,
        /// Bits needed to store the maximum number of elements in a page.
        page_bits: u8,
    },
    /// Extensible array index.
    ExtensibleArray {
        /// Address of the extensible array header.
        address: u64,
        /// Bits needed to store the maximum number of elements.
        max_bits: u8,
        /// Number of elements stored in the index block.
        index_elements: u8,
        /// Minimum number of data block pointers for a super block.
        min_pointers: u8,
        /// Minimum number of elements per data block.
        min_elements: u8,
        /// Bits needed to store the maximum number of elements in a page.
        page_bits: u8,
    },
    /// Version 2 B-tree index.
    BTreeV2 {
        /// Address of the B-tree header.
        address: u64,
        /// Node size in bytes.
        node_size: u32,
        /// Split percentage.
        split_percent: u8,
        /// Merge percentage.
        merge_percent: u8,
    },
}

impl ChunkIndex {
    /// Address of the index structure (or of the chunk data for the single
    /// chunk and implicit indexes). May be [`UNDEFINED_ADDRESS`].
    ///
    /// [`UNDEFINED_ADDRESS`]: crate::format::UNDEFINED_ADDRESS
    pub fn address(&self) -> u64 {
        match *self {
            ChunkIndex::BTreeV1 { address }
            | ChunkIndex::SingleChunk { address, .. }
            | ChunkIndex::Implicit { address }
            | ChunkIndex::FixedArray { address, .. }
            | ChunkIndex::ExtensibleArray { address, .. }
            | ChunkIndex::BTreeV2 { address, .. } => address,
        }
    }
}

/// Chunked storage description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkedLayout {
    /// Chunk dimensions in elements, one per dataspace dimension. The extra
    /// trailing on-disk "dimension" holding the element size is split off
    /// into [`element_size`](Self::element_size).
    pub dims: Vec<u64>,
    /// Size of one dataset element in bytes, as recorded in the layout.
    pub element_size: u64,
    /// Version 4 flags ([`DONT_FILTER_PARTIAL_BOUND_CHUNKS`],
    /// [`SINGLE_INDEX_WITH_FILTER`]); 0 for older versions.
    pub flags: u8,
    /// Chunk index.
    pub index: ChunkIndex,
    /// Version of the layout message this was decoded from (1-5).
    pub layout_version: u8,
}

impl ChunkedLayout {
    /// Width in bytes of the "Chunk Size" field of filtered chunk records
    /// in fixed array, extensible array and v2 B-tree indexes, when the
    /// layout version fixes it: the size of offsets for version 5 and
    /// later. `None` for version 4, where the width is derived from the
    /// index structure (one byte more than needed for an unfiltered chunk,
    /// at most 8).
    pub fn filtered_chunk_size_width(&self, sizes: Sizes) -> Option<u8> {
        (self.layout_version >= 5).then_some(sizes.offset)
    }
}

/// A decoded data layout message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataLayout {
    /// Raw data stored inside the message.
    Compact {
        /// The raw data.
        data: Vec<u8>,
        /// Array dimensions (versions 1-2 only).
        dims: Option<Vec<u32>>,
    },
    /// Raw data stored in one contiguous block of the file.
    Contiguous {
        /// Address of the data; may be [`UNDEFINED_ADDRESS`] when not yet
        /// allocated.
        ///
        /// [`UNDEFINED_ADDRESS`]: crate::format::UNDEFINED_ADDRESS
        address: u64,
        /// Allocated size in bytes (versions 3+; versions 1-2 do not store it).
        size: Option<u64>,
        /// Array dimensions (versions 1-2 only).
        dims: Option<Vec<u32>>,
    },
    /// Raw data split into separately stored chunks.
    Chunked(ChunkedLayout),
    /// Virtual dataset: mapping stored in the global heap (version 4).
    Virtual {
        /// Address of the global heap collection.
        heap_address: u64,
        /// Index of the object within the collection.
        heap_index: u32,
    },
}

/// Split the on-disk chunk dimension list into (chunk dims, element size).
fn split_chunk_dims(mut dims: Vec<u64>) -> Result<(Vec<u64>, u64)> {
    if dims.len() < 2 {
        return Err(Error::invalid("chunked layout needs at least 2 dimensions"));
    }
    let element_size = dims.pop().unwrap_or(0);
    if dims.contains(&0) {
        return Err(Error::invalid("chunk dimension of size zero"));
    }
    Ok((dims, element_size))
}

impl DataLayout {
    /// Parse a data layout message.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<DataLayout> {
        let mut r = ByteReader::with_sizes(data, sizes);
        match r.read_u8()? {
            v @ (1 | 2) => parse_v1_v2(&mut r, v),
            v @ (3..=5) => parse_v3_v4(&mut r, v),
            v => Err(Error::unsupported(format!("data layout version {v}"))),
        }
    }
}

fn parse_v1_v2(r: &mut ByteReader<'_>, version: u8) -> Result<DataLayout> {
    let ndims = r.read_u8()? as usize;
    let class = r.read_u8()?;
    r.skip(5)?; // reserved
    let address = if class == 0 {
        None
    } else {
        Some(r.read_offset()?)
    };
    let mut dims = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        dims.push(r.read_u32()?);
    }
    match (class, address) {
        (0, _) => {
            let size = r.read_u32()? as usize;
            Ok(DataLayout::Compact {
                data: r.read_bytes(size)?.to_vec(),
                dims: Some(dims),
            })
        }
        (1, Some(address)) => Ok(DataLayout::Contiguous {
            address,
            size: None,
            dims: Some(dims),
        }),
        (2, Some(address)) => {
            // The final dimension is the dataset element size.
            let (dims, element_size) = split_chunk_dims(dims.into_iter().map(u64::from).collect())?;
            Ok(DataLayout::Chunked(ChunkedLayout {
                dims,
                element_size,
                flags: 0,
                index: ChunkIndex::BTreeV1 { address },
                layout_version: version,
            }))
        }
        (c, _) => Err(Error::invalid(format!("unknown layout class {c}"))),
    }
}

fn parse_v3_v4(r: &mut ByteReader<'_>, version: u8) -> Result<DataLayout> {
    match r.read_u8()? {
        0 => {
            let size = r.read_u16()? as usize;
            Ok(DataLayout::Compact {
                data: r.read_bytes(size)?.to_vec(),
                dims: None,
            })
        }
        1 => Ok(DataLayout::Contiguous {
            address: r.read_offset()?,
            size: Some(r.read_length()?),
            dims: None,
        }),
        2 if version == 3 => {
            let ndims = r.read_u8()? as usize;
            let address = r.read_offset()?;
            let mut dims = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                dims.push(u64::from(r.read_u32()?));
            }
            let (dims, element_size) = split_chunk_dims(dims)?;
            Ok(DataLayout::Chunked(ChunkedLayout {
                dims,
                element_size,
                flags: 0,
                index: ChunkIndex::BTreeV1 { address },
                layout_version: version,
            }))
        }
        2 => parse_v4_chunked(r, version).map(DataLayout::Chunked),
        3 if version >= 4 => Ok(DataLayout::Virtual {
            heap_address: r.read_offset()?,
            heap_index: r.read_u32()?,
        }),
        c => Err(Error::invalid(format!("unknown layout class {c}"))),
    }
}

fn parse_v4_chunked(r: &mut ByteReader<'_>, layout_version: u8) -> Result<ChunkedLayout> {
    let flags = r.read_u8()?;
    let ndims = r.read_u8()? as usize;
    let width = r.read_u8()? as usize;
    if !(1..=8).contains(&width) {
        return Err(Error::invalid(format!(
            "chunk dimension encoded length {width}"
        )));
    }
    let mut dims = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        dims.push(r.read_uint(width)?);
    }
    let (dims, element_size) = split_chunk_dims(dims)?;
    let index = match r.read_u8()? {
        1 => {
            let filtered = if flags & SINGLE_INDEX_WITH_FILTER != 0 {
                Some(FilteredChunk {
                    size: r.read_length()?,
                    filter_mask: r.read_u32()?,
                })
            } else {
                None
            };
            ChunkIndex::SingleChunk {
                address: r.read_offset()?,
                filtered,
            }
        }
        2 => ChunkIndex::Implicit {
            address: r.read_offset()?,
        },
        3 => {
            let page_bits = r.read_u8()?;
            ChunkIndex::FixedArray {
                page_bits,
                address: r.read_offset()?,
            }
        }
        4 => {
            let [
                max_bits,
                index_elements,
                min_pointers,
                min_elements,
                page_bits,
            ] = r.read_array::<5>()?;
            ChunkIndex::ExtensibleArray {
                max_bits,
                index_elements,
                min_pointers,
                min_elements,
                page_bits,
                address: r.read_offset()?,
            }
        }
        5 => {
            let node_size = r.read_u32()?;
            let split_percent = r.read_u8()?;
            let merge_percent = r.read_u8()?;
            ChunkIndex::BTreeV2 {
                node_size,
                split_percent,
                merge_percent,
                address: r.read_offset()?,
            }
        }
        t => return Err(Error::invalid(format!("unknown chunk index type {t}"))),
    };
    Ok(ChunkedLayout {
        dims,
        element_size,
        flags,
        index,
        layout_version,
    })
}

impl DataLayout {
    /// Encode the message body as layout version `version` (3 or 4; 5 is
    /// accepted for chunked layouts and encodes like 4).
    ///
    /// Compact and contiguous layouts are supported for versions 3-5,
    /// chunked layouts with a v1 B-tree index for version 3 and with the
    /// version 4 indexes for versions 4/5, virtual layouts for 4/5.
    pub fn encode(&self, version: u8, sizes: Sizes) -> Result<Vec<u8>> {
        if !(3..=5).contains(&version) {
            return Err(Error::unsupported(format!(
                "encoding data layout version {version}"
            )));
        }
        let mut w = ByteWriter::with_sizes(sizes);
        w.u8(version);
        match self {
            DataLayout::Compact { data, dims } => {
                if dims.is_some() {
                    return Err(Error::invalid("layout dims require version 1/2"));
                }
                let len = u16::try_from(data.len())
                    .map_err(|_| Error::LimitExceeded("compact data larger than 65535 bytes"))?;
                w.u8(0).u16(len).bytes(data);
            }
            DataLayout::Contiguous {
                address,
                size,
                dims,
            } => {
                if dims.is_some() {
                    return Err(Error::invalid("layout dims require version 1/2"));
                }
                let size = size.ok_or_else(|| Error::invalid("contiguous layout needs a size"))?;
                w.u8(1);
                w.write_offset(*address)?;
                w.write_length(size)?;
            }
            DataLayout::Chunked(c) => encode_chunked(&mut w, c, version)?,
            DataLayout::Virtual {
                heap_address,
                heap_index,
            } => {
                if version < 4 {
                    return Err(Error::invalid("virtual layout requires version 4"));
                }
                w.u8(3);
                w.write_offset(*heap_address)?;
                w.u32(*heap_index);
            }
        }
        Ok(w.into_vec())
    }
}

fn encode_chunked(w: &mut ByteWriter, c: &ChunkedLayout, version: u8) -> Result<()> {
    let rank = c
        .dims
        .len()
        .checked_add(1)
        .and_then(|n| u8::try_from(n).ok())
        .ok_or(Error::LimitExceeded("chunk rank"))?;
    if c.dims.contains(&0) {
        return Err(Error::invalid("chunk dimension of size zero"));
    }
    w.u8(2);
    if version == 3 {
        let ChunkIndex::BTreeV1 { address } = c.index else {
            return Err(Error::invalid("layout version 3 needs a v1 B-tree index"));
        };
        w.u8(rank);
        w.write_offset(address)?;
        for &d in c.dims.iter().chain(std::iter::once(&c.element_size)) {
            let d = u32::try_from(d).map_err(|_| Error::LimitExceeded("chunk dimension"))?;
            w.u32(d);
        }
        return Ok(());
    }
    let max = c
        .dims
        .iter()
        .copied()
        .chain([c.element_size])
        .max()
        .unwrap_or(1);
    let width = ((64 - max.leading_zeros()) as usize).div_ceil(8).max(1);
    w.u8(c.flags).u8(rank).u8(width as u8);
    for &d in c.dims.iter().chain(std::iter::once(&c.element_size)) {
        w.write_uint(d, width)?;
    }
    match c.index {
        ChunkIndex::BTreeV1 { .. } => {
            return Err(Error::invalid("v1 B-tree index requires layout version 3"));
        }
        ChunkIndex::SingleChunk { address, filtered } => {
            w.u8(1);
            match (filtered, c.flags & SINGLE_INDEX_WITH_FILTER != 0) {
                (Some(f), true) => {
                    w.write_length(f.size)?;
                    w.u32(f.filter_mask);
                }
                (None, false) => {}
                _ => return Err(Error::invalid("single chunk filter flag mismatch")),
            }
            w.write_offset(address)?;
        }
        ChunkIndex::Implicit { address } => {
            w.u8(2);
            w.write_offset(address)?;
        }
        ChunkIndex::FixedArray { address, page_bits } => {
            w.u8(3).u8(page_bits);
            w.write_offset(address)?;
        }
        ChunkIndex::ExtensibleArray {
            address,
            max_bits,
            index_elements,
            min_pointers,
            min_elements,
            page_bits,
        } => {
            w.u8(4)
                .u8(max_bits)
                .u8(index_elements)
                .u8(min_pointers)
                .u8(min_elements)
                .u8(page_bits);
            w.write_offset(address)?;
        }
        ChunkIndex::BTreeV2 {
            address,
            node_size,
            split_percent,
            merge_percent,
        } => {
            w.u8(5).u32(node_size).u8(split_percent).u8(merge_percent);
            w.write_offset(address)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::UNDEFINED_ADDRESS;

    const S: Sizes = Sizes::DEFAULT;

    #[test]
    fn v1_v2_classes() {
        // contiguous, 2 dims
        let mut v = vec![1, 2, 1, 0, 0, 0, 0, 0];
        v.extend_from_slice(&0x800u64.to_le_bytes());
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(&5u32.to_le_bytes());
        assert_eq!(
            DataLayout::parse(&v, S).unwrap(),
            DataLayout::Contiguous {
                address: 0x800,
                size: None,
                dims: Some(vec![4, 5])
            }
        );
        // chunked: dims 2x3, element size 8
        let mut v = vec![2, 3, 2, 0, 0, 0, 0, 0];
        v.extend_from_slice(&0x400u64.to_le_bytes());
        for d in [2u32, 3, 8] {
            v.extend_from_slice(&d.to_le_bytes());
        }
        let DataLayout::Chunked(c) = DataLayout::parse(&v, S).unwrap() else {
            panic!()
        };
        assert_eq!(c.dims, vec![2, 3]);
        assert_eq!(c.element_size, 8);
        assert_eq!(c.index, ChunkIndex::BTreeV1 { address: 0x400 });
        // compact
        let mut v = vec![2, 1, 0, 0, 0, 0, 0, 0];
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&[7, 8, 9]);
        assert_eq!(
            DataLayout::parse(&v, S).unwrap(),
            DataLayout::Compact {
                data: vec![7, 8, 9],
                dims: Some(vec![3])
            }
        );
    }

    #[test]
    fn v3_classes() {
        let v = [3, 0, 2, 0, 0xaa, 0xbb];
        assert_eq!(
            DataLayout::parse(&v, S).unwrap(),
            DataLayout::Compact {
                data: vec![0xaa, 0xbb],
                dims: None
            }
        );
        let mut v = vec![3, 1];
        v.extend_from_slice(&[0xff; 8]);
        v.extend_from_slice(&80u64.to_le_bytes());
        assert_eq!(
            DataLayout::parse(&v, S).unwrap(),
            DataLayout::Contiguous {
                address: UNDEFINED_ADDRESS,
                size: Some(80),
                dims: None
            }
        );
        let mut v = vec![3, 2, 3];
        v.extend_from_slice(&0x1000u64.to_le_bytes());
        for d in [10u32, 20, 4] {
            v.extend_from_slice(&d.to_le_bytes());
        }
        let DataLayout::Chunked(c) = DataLayout::parse(&v, S).unwrap() else {
            panic!()
        };
        assert_eq!((c.dims, c.element_size), (vec![10, 20], 4));
        // virtual is v4 only
        assert!(DataLayout::parse(&[3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], S).is_err());
    }

    fn v4_chunked(flags: u8, index_type: u8, info: &[u8]) -> Vec<u8> {
        let mut v = vec![4, 2, flags, 3, 2];
        for d in [100u16, 50, 8] {
            v.extend_from_slice(&d.to_le_bytes());
        }
        v.push(index_type);
        v.extend_from_slice(info);
        v.extend_from_slice(&0x2000u64.to_le_bytes());
        v
    }

    #[test]
    fn v4_chunk_indexes() {
        let parse = |v: Vec<u8>| match DataLayout::parse(&v, S).unwrap() {
            DataLayout::Chunked(c) => c,
            other => panic!("{other:?}"),
        };
        let c = parse(v4_chunked(0, 1, &[]));
        assert_eq!(c.dims, vec![100, 50]);
        assert_eq!(c.element_size, 8);
        assert_eq!(
            c.index,
            ChunkIndex::SingleChunk {
                address: 0x2000,
                filtered: None
            }
        );
        let mut info = 77u64.to_le_bytes().to_vec();
        info.extend_from_slice(&1u32.to_le_bytes());
        let c = parse(v4_chunked(SINGLE_INDEX_WITH_FILTER, 1, &info));
        assert_eq!(
            c.index,
            ChunkIndex::SingleChunk {
                address: 0x2000,
                filtered: Some(FilteredChunk {
                    size: 77,
                    filter_mask: 1
                })
            }
        );
        assert_eq!(
            parse(v4_chunked(0, 2, &[])).index,
            ChunkIndex::Implicit { address: 0x2000 }
        );
        assert_eq!(
            parse(v4_chunked(1, 3, &[10])).index,
            ChunkIndex::FixedArray {
                address: 0x2000,
                page_bits: 10
            }
        );
        assert_eq!(
            parse(v4_chunked(0, 4, &[32, 4, 4, 16, 10])).index,
            ChunkIndex::ExtensibleArray {
                address: 0x2000,
                max_bits: 32,
                index_elements: 4,
                min_pointers: 4,
                min_elements: 16,
                page_bits: 10
            }
        );
        let mut info = 4096u32.to_le_bytes().to_vec();
        info.extend_from_slice(&[100, 40]);
        let c = parse(v4_chunked(0, 5, &info));
        assert_eq!(
            c.index,
            ChunkIndex::BTreeV2 {
                address: 0x2000,
                node_size: 4096,
                split_percent: 100,
                merge_percent: 40
            }
        );
        assert_eq!(c.index.address(), 0x2000);
        assert!(DataLayout::parse(&v4_chunked(0, 6, &[]), S).is_err());
    }

    #[test]
    fn v5_matches_v4_single_chunk() {
        // HDF5 2.0 writes version 5 for filtered chunked datasets (seen in
        // the h5py fixtures: filtered single chunk below), encoded exactly
        // like version 4.
        let v = [
            5, 2, 2, 3, 1, 4, 5, 4, 1, 0x2d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xe0, 9, 0, 0, 0, 0,
            0, 0,
        ];
        let DataLayout::Chunked(c) = DataLayout::parse(&v, S).unwrap() else {
            panic!()
        };
        assert_eq!((c.dims, c.element_size), (vec![4, 5], 4));
        assert_eq!(
            c.index,
            ChunkIndex::SingleChunk {
                address: 0x9e0,
                filtered: Some(FilteredChunk {
                    size: 0x2d,
                    filter_mask: 0
                })
            }
        );
    }

    #[test]
    fn v5_like_v4() {
        let mut v = v4_chunked(0, 3, &[10]);
        v[0] = 5;
        let DataLayout::Chunked(c) = DataLayout::parse(&v, S).unwrap() else {
            panic!("not chunked");
        };
        assert_eq!(c.layout_version, 5);
        assert_eq!(c.filtered_chunk_size_width(S), Some(8));
        let DataLayout::Chunked(c) = DataLayout::parse(&v4_chunked(0, 3, &[10]), S).unwrap() else {
            panic!("not chunked");
        };
        assert_eq!(c.filtered_chunk_size_width(S), None);
        assert!(DataLayout::parse(&[6, 1], S).is_err());
    }

    #[test]
    fn v4_virtual() {
        let mut v = vec![4, 3];
        v.extend_from_slice(&0x3000u64.to_le_bytes());
        v.extend_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            DataLayout::parse(&v, S).unwrap(),
            DataLayout::Virtual {
                heap_address: 0x3000,
                heap_index: 2
            }
        );
    }

    #[test]
    fn rejects_bad_input() {
        assert!(DataLayout::parse(&[], S).is_err());
        assert!(DataLayout::parse(&[5, 1], S).is_err());
        assert!(DataLayout::parse(&[3, 0, 10, 0, 1], S).is_err());
        // zero / too wide dimension encoding
        assert!(DataLayout::parse(&[4, 2, 0, 1, 0], S).is_err());
        assert!(DataLayout::parse(&[4, 2, 0, 1, 9], S).is_err());
        // chunked with a single (element size) dimension
        let mut v = vec![3, 2, 1];
        v.extend_from_slice(&0u64.to_le_bytes());
        v.extend_from_slice(&4u32.to_le_bytes());
        assert!(DataLayout::parse(&v, S).is_err());
        // zero chunk dimension
        let mut v = vec![3, 2, 2];
        v.extend_from_slice(&0u64.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&4u32.to_le_bytes());
        assert!(DataLayout::parse(&v, S).is_err());
    }

    #[test]
    fn mutation_never_panics() {
        let mut info = 4096u32.to_le_bytes().to_vec();
        info.extend_from_slice(&[100, 40]);
        let samples = [
            v4_chunked(0, 5, &info),
            v4_chunked(2, 1, &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            vec![3, 0, 2, 0, 0xaa, 0xbb],
        ];
        let mut state = 0xdead_beef_cafe_f00du64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..10_000 {
            let mut v = samples[(rnd() % 3) as usize].clone();
            let i = (rnd() % v.len() as u64) as usize;
            v[i] = rnd() as u8;
            v.truncate((rnd() % (v.len() as u64 + 1)) as usize);
            let _ = DataLayout::parse(&v, S);
        }
    }

    #[test]
    fn encode_round_trip() {
        let chunked = |index, flags, v| {
            DataLayout::Chunked(ChunkedLayout {
                dims: vec![4, 300],
                element_size: 8,
                flags,
                index,
                layout_version: v,
            })
        };
        let cases = [
            (
                3,
                DataLayout::Compact {
                    data: vec![1, 2, 3],
                    dims: None,
                },
            ),
            (
                4,
                DataLayout::Compact {
                    data: vec![],
                    dims: None,
                },
            ),
            (
                3,
                DataLayout::Contiguous {
                    address: 4096,
                    size: Some(80),
                    dims: None,
                },
            ),
            (
                3,
                DataLayout::Contiguous {
                    address: crate::format::UNDEFINED_ADDRESS,
                    size: Some(80),
                    dims: None,
                },
            ),
            (3, chunked(ChunkIndex::BTreeV1 { address: 99 }, 0, 3)),
            (
                4,
                chunked(
                    ChunkIndex::SingleChunk {
                        address: 7,
                        filtered: Some(FilteredChunk {
                            size: 10,
                            filter_mask: 1,
                        }),
                    },
                    SINGLE_INDEX_WITH_FILTER,
                    4,
                ),
            ),
            (4, chunked(ChunkIndex::Implicit { address: 7 }, 0, 4)),
            (
                4,
                chunked(
                    ChunkIndex::FixedArray {
                        address: 7,
                        page_bits: 10,
                    },
                    1,
                    4,
                ),
            ),
            (
                5,
                chunked(
                    ChunkIndex::ExtensibleArray {
                        address: 7,
                        max_bits: 32,
                        index_elements: 4,
                        min_pointers: 4,
                        min_elements: 16,
                        page_bits: 10,
                    },
                    0,
                    5,
                ),
            ),
            (
                4,
                chunked(
                    ChunkIndex::BTreeV2 {
                        address: 7,
                        node_size: 2048,
                        split_percent: 100,
                        merge_percent: 40,
                    },
                    0,
                    4,
                ),
            ),
            (
                4,
                DataLayout::Virtual {
                    heap_address: 5,
                    heap_index: 2,
                },
            ),
        ];
        for (v, l) in &cases {
            let enc = l.encode(*v, Sizes::DEFAULT).unwrap();
            assert_eq!(
                &DataLayout::parse(&enc, Sizes::DEFAULT).unwrap(),
                l,
                "{l:?}"
            );
        }
        let big = DataLayout::Compact {
            data: vec![0; 70000],
            dims: None,
        };
        assert!(big.encode(3, Sizes::DEFAULT).is_err());
        assert!(cases[0].1.encode(2, Sizes::DEFAULT).is_err());
        assert!(cases[10].1.encode(3, Sizes::DEFAULT).is_err());
    }
}
