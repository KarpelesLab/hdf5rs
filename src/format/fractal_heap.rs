//! Fractal heaps (spec: "Fractal Heap", level 1G).
//!
//! A fractal heap stores variable-size objects addressed by heap IDs. Small
//! ("tiny") objects live inside the ID itself, large ("huge") objects are
//! stored separately in the file (optionally indexed by a v2 B-tree), and
//! everything else ("managed" objects) lives in direct blocks arranged in a
//! doubling table rooted at either a direct or an indirect block.
//!
//! # Doubling table geometry
//!
//! With table width `W`, starting block size `S` and maximum direct block
//! size `M` (all powers of two), row `r` of any indirect block holds `W`
//! blocks of size `S` for `r <= 1` and `S * 2^(r-1)` otherwise. Row `r`
//! therefore starts at offset `0` for `r = 0` and `W * S * 2^(r-1)` for
//! `r >= 1`, relative to the indirect block's own offset in the heap's
//! linear address space; an indirect block with `n` rows spans
//! `W * S * 2^(n-1)` bytes.
//!
//! Rows `0 .. max_dblock_rows` (`max_dblock_rows = log2(M) - log2(S) + 2`)
//! point to direct blocks; later rows point to child indirect blocks. A child
//! indirect block in a row of block size `B` must span exactly `B` bytes, so
//! it has `log2(B) - log2(S * W) + 1` rows. (The spec phrases this as
//! `log2(block_size) - log2(start block size) + 1` where `block_size` is the
//! size the indirect block "represents"; the form above follows from the
//! span equation and is what keeps child offsets consistent.)
//!
//! Managed object offsets are byte offsets in the heap's linear address
//! space; an object at heap offset `o` lies in the direct block whose range
//! contains `o`, at `o - block_offset` from the start of that block
//! (i.e. counting the block's own header).

use super::btree_v2::{BTreeV2, Record, RecordContext};
use super::{ByteReader, Sizes, UNDEFINED_ADDRESS};
use crate::checksum::metadata_checksum;
use crate::error::{Error, Result};
use crate::io::ReadAt;

/// Callback undoing a heap's I/O filter pipeline: receives the stored bytes
/// and the filter mask (bit `i` set: filter `i` was skipped) and returns the
/// de-filtered bytes.
pub type FilterDecode<'a> = &'a dyn Fn(&[u8], u32) -> Result<Vec<u8>>;

/// Maximum nesting of indirect blocks.
const MAX_INDIRECT_DEPTH: usize = 64;
/// Maximum number of rows in any indirect block.
const MAX_ROWS: u32 = 64;

/// Header flag bit 1: direct blocks carry a checksum.
const FLAG_CHECKSUM_DIRECT: u8 = 0x02;

/// The fractal heap header (`FRHP`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FractalHeapHeader {
    /// Length in bytes of heap IDs.
    pub heap_id_len: u16,
    /// Size of the encoded I/O filter information (0: no filters).
    pub io_filter_len: u16,
    /// Status flags (bit 0: huge IDs wrapped; bit 1: checksummed direct
    /// blocks).
    pub flags: u8,
    /// Objects larger than this are stored as huge objects.
    pub max_managed_object_size: u32,
    /// Next huge object ID.
    pub next_huge_id: u64,
    /// Address of the v2 B-tree indexing huge objects.
    pub huge_btree_address: u64,
    /// Free space in managed blocks.
    pub free_space: u64,
    /// Address of the managed-block free-space manager.
    pub free_space_manager_address: u64,
    /// Total managed space (upper bound of the linear address space).
    pub managed_space: u64,
    /// Allocated managed space.
    pub allocated_managed_space: u64,
    /// Offset of the direct block allocation iterator.
    pub direct_block_iterator_offset: u64,
    /// Number of managed objects.
    pub managed_object_count: u64,
    /// Total size of huge objects.
    pub huge_object_size: u64,
    /// Number of huge objects.
    pub huge_object_count: u64,
    /// Total size of tiny objects.
    pub tiny_object_size: u64,
    /// Number of tiny objects.
    pub tiny_object_count: u64,
    /// Number of columns of the doubling table.
    pub table_width: u16,
    /// Size of blocks in the first two rows.
    pub starting_block_size: u64,
    /// Largest direct block size.
    pub max_direct_block_size: u64,
    /// log2 of the maximum size of the linear address space.
    pub max_heap_size_bits: u16,
    /// Starting number of rows of the root indirect block.
    pub starting_root_rows: u16,
    /// Address of the root block ([`UNDEFINED_ADDRESS`] if empty).
    pub root_address: u64,
    /// Current rows of the root indirect block (0: the root is a direct
    /// block).
    pub current_root_rows: u16,
    /// Stored size of the filtered root direct block (filtered heaps only).
    pub filtered_root_size: Option<u64>,
    /// Filter mask of the root direct block (filtered heaps only).
    pub root_filter_mask: Option<u32>,
    /// Raw encoded filter pipeline message (empty when unfiltered).
    pub filter_info: Vec<u8>,
}

impl FractalHeapHeader {
    /// Size of the header up to (excluding) the optional filter fields.
    fn fixed_len(sizes: Sizes) -> usize {
        22 + 12 * sizes.length as usize + 3 * sizes.offset as usize
    }

    /// Parse and checksum-verify a header from `data`.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::with_sizes(data, sizes);
        r.expect_signature(b"FRHP")?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "fractal heap version {version}"
            )));
        }
        let heap_id_len = r.read_u16()?;
        let io_filter_len = r.read_u16()?;
        let flags = r.read_u8()?;
        let max_managed_object_size = r.read_u32()?;
        let next_huge_id = r.read_length()?;
        let huge_btree_address = r.read_offset()?;
        let free_space = r.read_length()?;
        let free_space_manager_address = r.read_offset()?;
        let managed_space = r.read_length()?;
        let allocated_managed_space = r.read_length()?;
        let direct_block_iterator_offset = r.read_length()?;
        let managed_object_count = r.read_length()?;
        let huge_object_size = r.read_length()?;
        let huge_object_count = r.read_length()?;
        let tiny_object_size = r.read_length()?;
        let tiny_object_count = r.read_length()?;
        let table_width = r.read_u16()?;
        let starting_block_size = r.read_length()?;
        let max_direct_block_size = r.read_length()?;
        let max_heap_size_bits = r.read_u16()?;
        let starting_root_rows = r.read_u16()?;
        let root_address = r.read_offset()?;
        let current_root_rows = r.read_u16()?;
        let (filtered_root_size, root_filter_mask, filter_info) = if io_filter_len > 0 {
            (
                Some(r.read_length()?),
                Some(r.read_u32()?),
                r.read_bytes(usize::from(io_filter_len))?.to_vec(),
            )
        } else {
            (None, None, Vec::new())
        };
        let end = r.position();
        let actual = metadata_checksum(data.get(..end).ok_or(Error::UnexpectedEof)?);
        let expected = r.read_u32()?;
        if expected != actual {
            return Err(Error::ChecksumMismatch {
                what: "fractal heap header",
                expected,
                actual,
            });
        }
        Ok(FractalHeapHeader {
            heap_id_len,
            io_filter_len,
            flags,
            max_managed_object_size,
            next_huge_id,
            huge_btree_address,
            free_space,
            free_space_manager_address,
            managed_space,
            allocated_managed_space,
            direct_block_iterator_offset,
            managed_object_count,
            huge_object_size,
            huge_object_count,
            tiny_object_size,
            tiny_object_count,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size_bits,
            starting_root_rows,
            root_address,
            current_root_rows,
            filtered_root_size,
            root_filter_mask,
            filter_info,
        })
    }

    /// Whether objects in this heap pass through an I/O filter pipeline.
    pub fn is_filtered(&self) -> bool {
        self.io_filter_len > 0
    }
}

/// A decoded heap ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeapId {
    /// An object in the managed (doubling table) space.
    Managed {
        /// Offset in the heap's linear address space.
        offset: u64,
        /// Length of the object.
        length: u64,
    },
    /// An object stored inside the ID itself.
    Tiny(Vec<u8>),
    /// A huge object whose location is stored in the ID.
    HugeDirect {
        /// Address of the (possibly filtered) object.
        address: u64,
        /// Stored length of the object.
        length: u64,
        /// `(filter mask, de-filtered size)` for filtered heaps.
        filter: Option<(u32, u64)>,
    },
    /// A huge object located through the huge-object v2 B-tree.
    HugeIndirect {
        /// Key into the B-tree (the huge object ID).
        key: u64,
    },
}

/// A direct block reference: where it is and how it is stored.
#[derive(Debug, Clone, Copy)]
struct DirectRef {
    address: u64,
    /// Offset of the block in the heap's linear address space.
    block_offset: u64,
    /// Size of the block in the doubling table (= de-filtered size).
    size: u64,
    /// `(stored size, filter mask)` when filtered.
    filtered: Option<(u64, u32)>,
}

/// A decoded indirect block.
#[derive(Debug)]
struct IndirectBlock {
    /// `(address, filtered (size, mask))` per direct entry.
    direct: Vec<(u64, Option<(u64, u32)>)>,
    indirect: Vec<u64>,
}

/// Least number of bytes able to encode `v` (1..=8).
fn bytes_for(v: u64) -> usize {
    (64 - v.leading_zeros() as usize).div_ceil(8).max(1)
}

/// An opened fractal heap.
#[derive(Debug, Clone)]
pub struct FractalHeap {
    /// Address of the heap header.
    pub address: u64,
    /// The decoded header.
    pub header: FractalHeapHeader,
    sizes: Sizes,
    width: u64,
    log2_width: u32,
    log2_start: u32,
    max_dblock_rows: u32,
    /// Width of "Block Offset" fields and of managed ID offsets.
    offset_width: usize,
    /// Width of managed ID lengths.
    length_width: usize,
}

impl FractalHeap {
    /// Read and validate the heap header at `addr`.
    pub fn open<R: ReadAt + ?Sized>(store: &R, addr: u64, sizes: Sizes) -> Result<Self> {
        let fixed = FractalHeapHeader::fixed_len(sizes);
        let head = store.read_vec_at(addr, fixed)?;
        let filter_len = usize::from(u16::from_le_bytes([
            *head.get(7).ok_or(Error::UnexpectedEof)?,
            *head.get(8).ok_or(Error::UnexpectedEof)?,
        ]));
        let total = fixed
            + if filter_len > 0 {
                sizes.length as usize + 4 + filter_len
            } else {
                0
            }
            + 4;
        let buf = store.read_vec_at(addr, total)?;
        let header = FractalHeapHeader::parse(&buf, sizes)?;

        let pow2 = |v: u64, what: &'static str| -> Result<u32> {
            if v == 0 || !v.is_power_of_two() {
                return Err(Error::invalid(what));
            }
            Ok(v.trailing_zeros())
        };
        let log2_width = pow2(
            u64::from(header.table_width),
            "fractal heap table width not a power of two",
        )?;
        let log2_start = pow2(
            header.starting_block_size,
            "fractal heap starting block size not a power of two",
        )?;
        let log2_max_direct = pow2(
            header.max_direct_block_size,
            "fractal heap max direct block size not a power of two",
        )?;
        if log2_max_direct < log2_start {
            return Err(Error::invalid(
                "fractal heap max direct block size below starting size",
            ));
        }
        let bits = u32::from(header.max_heap_size_bits);
        if bits == 0 || bits > 64 {
            return Err(Error::invalid(
                "fractal heap maximum heap size out of range",
            ));
        }
        if log2_width + log2_start >= 64 {
            return Err(Error::LimitExceeded(
                "fractal heap doubling table too large",
            ));
        }
        if u32::from(header.current_root_rows) > MAX_ROWS {
            return Err(Error::LimitExceeded("fractal heap root rows"));
        }
        if usize::from(header.heap_id_len) < 1 {
            return Err(Error::invalid("fractal heap ID length is zero"));
        }
        let max_dblock_rows = log2_max_direct - log2_start + 2;
        let length_width = bytes_for(
            header
                .max_direct_block_size
                .min(u64::from(header.max_managed_object_size)),
        );
        Ok(FractalHeap {
            address: addr,
            sizes,
            width: u64::from(header.table_width),
            log2_width,
            log2_start,
            max_dblock_rows,
            offset_width: (bits as usize).div_ceil(8),
            length_width,
            header,
        })
    }

    /// Width in bytes of the offset field of managed heap IDs (and of the
    /// "Block Offset" field of heap blocks).
    pub fn managed_offset_width(&self) -> usize {
        self.offset_width
    }

    /// Width in bytes of the length field of managed heap IDs.
    pub fn managed_length_width(&self) -> usize {
        self.length_width
    }

    /// Maximum number of direct block rows in any indirect block.
    pub fn max_direct_rows(&self) -> u32 {
        self.max_dblock_rows
    }

    /// Size of blocks in row `row` of the doubling table.
    fn row_block_size(&self, row: u32) -> Result<u64> {
        let shift = row.saturating_sub(1);
        if shift >= 64 - self.log2_start {
            return Err(Error::LimitExceeded("fractal heap row too large"));
        }
        Ok(self.header.starting_block_size << shift)
    }

    /// Offset of row `row` relative to the start of its indirect block.
    fn row_start(&self, row: u32) -> Result<u64> {
        if row == 0 {
            return Ok(0);
        }
        let shift = row - 1;
        if shift >= 64 - self.log2_start - self.log2_width {
            return Err(Error::LimitExceeded("fractal heap row too large"));
        }
        Ok((self.width << self.log2_start) << shift)
    }

    /// Map an offset relative to an indirect block to `(row, column)`.
    fn locate(&self, rel: u64) -> (u32, u64) {
        let row_bytes = self.width << self.log2_start;
        if rel < row_bytes {
            return (0, rel >> self.log2_start);
        }
        // rel / row_bytes >= 1; row = floor(log2(rel / row_bytes)) + 1.
        let k = 63 - (rel >> (self.log2_start + self.log2_width)).leading_zeros();
        let row_start = row_bytes << k;
        let col = (rel - row_start) >> (self.log2_start + k);
        (k + 1, col)
    }

    /// Number of rows of a child indirect block found in row `row`.
    fn child_rows(&self, row: u32) -> Result<u32> {
        let log2_block = self.row_block_size(row)?.trailing_zeros();
        let base = self.log2_start + self.log2_width;
        if log2_block < base {
            return Err(Error::invalid(
                "fractal heap indirect block smaller than one row",
            ));
        }
        Ok(log2_block - base + 1)
    }

    /// Size of the fixed prefix of direct blocks (including any checksum).
    fn direct_header_len(&self) -> u64 {
        let checksum = if self.header.flags & FLAG_CHECKSUM_DIRECT != 0 {
            4
        } else {
            0
        };
        5 + u64::from(self.sizes.offset) + self.offset_width as u64 + checksum
    }

    /// Decode a heap ID without resolving it.
    pub fn parse_id(&self, id: &[u8]) -> Result<HeapId> {
        let mut r = ByteReader::with_sizes(id, self.sizes);
        let flags = r.read_u8()?;
        if flags >> 6 != 0 {
            return Err(Error::unsupported(format!(
                "fractal heap ID version {}",
                flags >> 6
            )));
        }
        match (flags >> 4) & 3 {
            0 => Ok(HeapId::Managed {
                offset: r.read_uint(self.offset_width)?,
                length: r.read_uint(self.length_width)?,
            }),
            1 => {
                let o = usize::from(self.sizes.offset);
                let l = usize::from(self.sizes.length);
                let id_len = usize::from(self.header.heap_id_len);
                let filtered = self.header.is_filtered();
                let direct_len = if filtered {
                    1 + o + 2 * l + 4
                } else {
                    1 + o + l
                };
                if id_len >= direct_len {
                    let address = r.read_offset()?;
                    let length = r.read_length()?;
                    let filter = if filtered {
                        Some((r.read_u32()?, r.read_length()?))
                    } else {
                        None
                    };
                    Ok(HeapId::HugeDirect {
                        address,
                        length,
                        filter,
                    })
                } else {
                    // The key is "size of lengths" wide but truncated to the
                    // space available in the ID.
                    let key_len = l.min(id_len.saturating_sub(1));
                    Ok(HeapId::HugeIndirect {
                        key: r.read_uint(key_len)?,
                    })
                }
            }
            2 => {
                let len = if self.header.heap_id_len <= 18 {
                    usize::from(flags & 0x0f) + 1
                } else {
                    let low = r.read_u8()?;
                    ((usize::from(flags & 0x0f) << 8) | usize::from(low)) + 1
                };
                Ok(HeapId::Tiny(r.read_bytes(len)?.to_vec()))
            }
            _ => Err(Error::invalid("reserved fractal heap ID type")),
        }
    }

    /// Fetch the object identified by `heap_id`. Filtered heaps require
    /// [`get_with_filter`](Self::get_with_filter).
    pub fn get<R: ReadAt + ?Sized>(&self, store: &R, heap_id: &[u8]) -> Result<Vec<u8>> {
        self.get_with_filter(store, heap_id, None)
    }

    /// Fetch the object identified by `heap_id`, using `decode` to undo the
    /// heap's I/O filters where needed.
    pub fn get_with_filter<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        heap_id: &[u8],
        decode: Option<FilterDecode<'_>>,
    ) -> Result<Vec<u8>> {
        match self.parse_id(heap_id)? {
            HeapId::Tiny(data) => Ok(data),
            HeapId::Managed { offset, length } => self.read_managed(store, offset, length, decode),
            HeapId::HugeDirect {
                address,
                length,
                filter,
            } => self.read_huge(store, address, length, filter, decode),
            HeapId::HugeIndirect { key } => match self.lookup_huge(store, key)? {
                HeapId::HugeDirect {
                    address,
                    length,
                    filter,
                } => self.read_huge(store, address, length, filter, decode),
                _ => Err(Error::invalid("unexpected huge object lookup result")),
            },
        }
    }

    /// Resolve an indirect huge object ID through the huge-object B-tree,
    /// returning the equivalent [`HeapId::HugeDirect`].
    fn lookup_huge<R: ReadAt + ?Sized>(&self, store: &R, key: u64) -> Result<HeapId> {
        if self.header.huge_btree_address == UNDEFINED_ADDRESS {
            return Err(Error::invalid("huge heap ID but no huge object B-tree"));
        }
        let tree = BTreeV2::open(
            store,
            self.header.huge_btree_address,
            self.sizes,
            RecordContext::default(),
        )?;
        let expected = if self.header.is_filtered() { 2 } else { 1 };
        if tree.header.tree_type != expected {
            return Err(Error::invalid("unexpected huge object B-tree type"));
        }
        let id_of = |r: &Record| match r {
            Record::HugeIndirect { id, .. } | Record::HugeIndirectFiltered { id, .. } => *id,
            _ => 0,
        };
        let mut hit = tree.find(store, |r| id_of(r).cmp(&key))?;
        if hit.is_none() {
            // Fall back to a scan in case records are not ordered by ID.
            let mut found = None;
            tree.visit(store, &mut |r| {
                if id_of(r) == key {
                    found = Some(r.clone());
                    return Ok(false);
                }
                Ok(true)
            })?;
            hit = found;
        }
        match hit {
            Some(Record::HugeIndirect {
                address, length, ..
            }) => Ok(HeapId::HugeDirect {
                address,
                length,
                filter: None,
            }),
            Some(Record::HugeIndirectFiltered {
                address,
                length,
                filter_mask,
                memory_size,
                ..
            }) => Ok(HeapId::HugeDirect {
                address,
                length,
                filter: Some((filter_mask, memory_size)),
            }),
            _ => Err(Error::NotFound(format!("huge heap object {key}"))),
        }
    }

    fn read_huge<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        address: u64,
        length: u64,
        filter: Option<(u32, u64)>,
        decode: Option<FilterDecode<'_>>,
    ) -> Result<Vec<u8>> {
        let len = usize::try_from(length).map_err(|_| Error::LimitExceeded("huge object"))?;
        let raw = store.read_vec_at(address, len)?;
        match filter {
            None => Ok(raw),
            Some((mask, size)) => {
                let decode =
                    decode.ok_or_else(|| Error::unsupported("filtered fractal heap object"))?;
                let out = decode(&raw, mask)?;
                if out.len() as u64 != size {
                    return Err(Error::invalid("de-filtered huge object size mismatch"));
                }
                Ok(out)
            }
        }
    }

    fn read_managed<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        offset: u64,
        length: u64,
        decode: Option<FilterDecode<'_>>,
    ) -> Result<Vec<u8>> {
        if length == 0 {
            return Err(Error::invalid("zero-length managed heap object"));
        }
        if self.header.root_address == UNDEFINED_ADDRESS {
            return Err(Error::invalid("managed heap ID in an empty heap"));
        }
        let block = self.find_direct_block(store, offset)?;
        let rel = offset - block.block_offset;
        self.read_from_direct(store, &block, rel, length, decode)
    }

    /// Locate the direct block containing heap offset `offset`.
    fn find_direct_block<R: ReadAt + ?Sized>(&self, store: &R, offset: u64) -> Result<DirectRef> {
        let filtered = self.header.is_filtered();
        if self.header.current_root_rows == 0 {
            let size = self.header.starting_block_size;
            if offset >= size {
                return Err(Error::invalid("heap offset beyond root direct block"));
            }
            let filt = match (self.header.filtered_root_size, self.header.root_filter_mask) {
                (Some(s), Some(m)) => Some((s, m)),
                _ => None,
            };
            return Ok(DirectRef {
                address: self.header.root_address,
                block_offset: 0,
                size,
                filtered: if filtered { filt } else { None },
            });
        }
        let mut addr = self.header.root_address;
        let mut nrows = u32::from(self.header.current_root_rows);
        let mut base = 0u64;
        for _ in 0..MAX_INDIRECT_DEPTH {
            let ib = self.read_indirect(store, addr, nrows, base)?;
            let rel = offset
                .checked_sub(base)
                .ok_or_else(|| Error::invalid("heap offset before indirect block"))?;
            let (row, col) = self.locate(rel);
            if row >= nrows {
                return Err(Error::invalid("heap offset beyond indirect block"));
            }
            let size = self.row_block_size(row)?;
            let child_base = base
                .checked_add(self.row_start(row)?)
                .and_then(|b| b.checked_add(col.checked_mul(size)?))
                .ok_or(Error::LimitExceeded("fractal heap offset overflow"))?;
            let w = self.width;
            if row < self.max_dblock_rows {
                let idx = usize::try_from(u64::from(row) * w + col)
                    .map_err(|_| Error::LimitExceeded("fractal heap entry index"))?;
                let &(address, filt) = ib
                    .direct
                    .get(idx)
                    .ok_or_else(|| Error::invalid("fractal heap direct entry out of range"))?;
                if address == UNDEFINED_ADDRESS {
                    return Err(Error::invalid("heap offset in unallocated direct block"));
                }
                return Ok(DirectRef {
                    address,
                    block_offset: child_base,
                    size,
                    filtered: filt,
                });
            }
            let idx = usize::try_from(u64::from(row - self.max_dblock_rows) * w + col)
                .map_err(|_| Error::LimitExceeded("fractal heap entry index"))?;
            let child = *ib
                .indirect
                .get(idx)
                .ok_or_else(|| Error::invalid("fractal heap indirect entry out of range"))?;
            if child == UNDEFINED_ADDRESS {
                return Err(Error::invalid("heap offset in unallocated indirect block"));
            }
            nrows = self.child_rows(row)?;
            addr = child;
            base = child_base;
        }
        Err(Error::LimitExceeded("fractal heap indirect block nesting"))
    }

    fn read_indirect<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        addr: u64,
        nrows: u32,
        expected_offset: u64,
    ) -> Result<IndirectBlock> {
        if nrows == 0 || nrows > MAX_ROWS {
            return Err(Error::invalid("fractal heap indirect block row count"));
        }
        let drows = u64::from(nrows.min(self.max_dblock_rows));
        let irows = u64::from(nrows.saturating_sub(self.max_dblock_rows));
        let filtered = self.header.is_filtered();
        let o = u64::from(self.sizes.offset);
        let entry = if filtered {
            o + u64::from(self.sizes.length) + 4
        } else {
            o
        };
        let k = drows * self.width;
        let n = irows * self.width;
        let size = (5 + o + self.offset_width as u64)
            .checked_add(k.checked_mul(entry).ok_or(Error::UnexpectedEof)?)
            .and_then(|s| s.checked_add(n.checked_mul(o)?))
            .and_then(|s| s.checked_add(4))
            .ok_or(Error::UnexpectedEof)?;
        let size = usize::try_from(size).map_err(|_| Error::UnexpectedEof)?;
        let buf = store.read_vec_at(addr, size)?;
        let mut r = ByteReader::with_sizes(&buf, self.sizes);
        r.expect_signature(b"FHIB")?;
        self.check_block_prefix(&mut r, expected_offset, "indirect")?;
        let mut direct = Vec::with_capacity(k as usize);
        for _ in 0..k {
            let address = r.read_offset()?;
            let filt = if filtered {
                Some((r.read_length()?, r.read_u32()?))
            } else {
                None
            };
            direct.push((address, filt));
        }
        let mut indirect = Vec::with_capacity(n as usize);
        for _ in 0..n {
            indirect.push(r.read_offset()?);
        }
        let end = r.position();
        let actual = metadata_checksum(buf.get(..end).ok_or(Error::UnexpectedEof)?);
        let expected = r.read_u32()?;
        if expected != actual {
            return Err(Error::ChecksumMismatch {
                what: "fractal heap indirect block",
                expected,
                actual,
            });
        }
        Ok(IndirectBlock { direct, indirect })
    }

    /// Check version, heap header address and block offset of a block whose
    /// signature has been consumed.
    fn check_block_prefix(
        &self,
        r: &mut ByteReader<'_>,
        expected_offset: u64,
        what: &str,
    ) -> Result<()> {
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "fractal heap {what} block version {version}"
            )));
        }
        if r.read_offset()? != self.address {
            return Err(Error::invalid(format!(
                "fractal heap {what} block belongs to another heap"
            )));
        }
        if r.read_uint(self.offset_width)? != expected_offset {
            return Err(Error::invalid(format!(
                "fractal heap {what} block has unexpected offset"
            )));
        }
        Ok(())
    }

    fn read_from_direct<R: ReadAt + ?Sized>(
        &self,
        store: &R,
        block: &DirectRef,
        rel: u64,
        length: u64,
        decode: Option<FilterDecode<'_>>,
    ) -> Result<Vec<u8>> {
        let hdr_len = self.direct_header_len();
        let end = rel
            .checked_add(length)
            .ok_or_else(|| Error::invalid("heap object length overflow"))?;
        if rel < hdr_len || end > block.size {
            return Err(Error::invalid("heap object outside its direct block"));
        }
        let checksummed = self.header.flags & FLAG_CHECKSUM_DIRECT != 0;
        let size = usize::try_from(block.size).map_err(|_| Error::UnexpectedEof)?;
        // `rel < end <= block.size`, which fits in usize.
        let (rel, end) = (rel as usize, end as usize);

        if block.filtered.is_none() && !checksummed {
            // Fast path: read only the block prefix and the object.
            let head = store.read_vec_at(block.address, hdr_len as usize)?;
            self.check_direct_prefix(&head, block.block_offset)?;
            let pos = block
                .address
                .checked_add(rel as u64)
                .ok_or(Error::UnexpectedEof)?;
            return store.read_vec_at(pos, end - rel);
        }

        let data = match block.filtered {
            Some((stored, mask)) => {
                let decode =
                    decode.ok_or_else(|| Error::unsupported("filtered fractal heap block"))?;
                let stored =
                    usize::try_from(stored).map_err(|_| Error::LimitExceeded("filtered block"))?;
                let raw = store.read_vec_at(block.address, stored)?;
                let out = decode(&raw, mask)?;
                if out.len() != size {
                    return Err(Error::invalid("de-filtered direct block size mismatch"));
                }
                out
            }
            None => store.read_vec_at(block.address, size)?,
        };
        self.check_direct_prefix(&data, block.block_offset)?;
        if checksummed {
            // The checksum covers the whole block with the checksum field
            // itself taken as zero.
            let pos = hdr_len as usize - 4;
            let stored =
                ByteReader::new(data.get(pos..).ok_or(Error::UnexpectedEof)?).read_u32()?;
            let mut copy = data.clone();
            copy[pos..pos + 4].fill(0);
            let actual = metadata_checksum(&copy);
            if stored != actual {
                return Err(Error::ChecksumMismatch {
                    what: "fractal heap direct block",
                    expected: stored,
                    actual,
                });
            }
        }
        data.get(rel..end)
            .map(<[u8]>::to_vec)
            .ok_or(Error::UnexpectedEof)
    }

    fn check_direct_prefix(&self, data: &[u8], block_offset: u64) -> Result<()> {
        let mut r = ByteReader::with_sizes(data, self.sizes);
        r.expect_signature(b"FHDB")?;
        self.check_block_prefix(&mut r, block_offset, "direct")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::btree_v2::tests::{
        FileBuilder, HeaderSpec, Rng, header_bytes, leaf_bytes, push_uint, seal,
    };

    struct HeapSpec {
        heap_id_len: u16,
        flags: u8,
        max_managed: u32,
        huge_btree: u64,
        width: u16,
        start: u64,
        max_direct: u64,
        max_bits: u16,
        root: u64,
        cur_rows: u16,
        /// (filtered root size, root mask, filter info)
        filter: Option<(u64, u32, Vec<u8>)>,
    }

    impl HeapSpec {
        fn basic() -> Self {
            HeapSpec {
                heap_id_len: 7,
                flags: 0,
                max_managed: 100,
                huge_btree: UNDEFINED_ADDRESS,
                width: 2,
                start: 64,
                max_direct: 128,
                max_bits: 16,
                root: UNDEFINED_ADDRESS,
                cur_rows: 0,
                filter: None,
            }
        }

        fn header_len(&self) -> usize {
            let f = self.filter.as_ref().map_or(0, |(_, _, i)| 12 + i.len());
            146 + f
        }

        fn bytes(&self) -> Vec<u8> {
            let mut v = b"FRHP".to_vec();
            v.push(0);
            v.extend_from_slice(&self.heap_id_len.to_le_bytes());
            let flen = self.filter.as_ref().map_or(0, |(_, _, i)| i.len() as u16);
            v.extend_from_slice(&flen.to_le_bytes());
            v.push(self.flags);
            v.extend_from_slice(&self.max_managed.to_le_bytes());
            push_uint(&mut v, 1, 8); // next huge id
            push_uint(&mut v, self.huge_btree, 8);
            push_uint(&mut v, 0, 8); // free space
            push_uint(&mut v, UNDEFINED_ADDRESS, 8); // fsm
            for _ in 0..8 {
                push_uint(&mut v, 0, 8); // statistics
            }
            v.extend_from_slice(&self.width.to_le_bytes());
            push_uint(&mut v, self.start, 8);
            push_uint(&mut v, self.max_direct, 8);
            v.extend_from_slice(&self.max_bits.to_le_bytes());
            v.extend_from_slice(&0u16.to_le_bytes());
            push_uint(&mut v, self.root, 8);
            v.extend_from_slice(&self.cur_rows.to_le_bytes());
            if let Some((size, mask, info)) = &self.filter {
                push_uint(&mut v, *size, 8);
                v.extend_from_slice(&mask.to_le_bytes());
                v.extend_from_slice(info);
            }
            let v = seal(v);
            assert_eq!(v.len(), self.header_len());
            v
        }
    }

    /// Build a direct block; `bow` is the block offset width.
    fn direct_block(
        heap: u64,
        block_offset: u64,
        size: usize,
        bow: usize,
        checksum: bool,
        objects: &[(usize, &[u8])],
    ) -> Vec<u8> {
        let mut v = b"FHDB".to_vec();
        v.push(0);
        push_uint(&mut v, heap, 8);
        push_uint(&mut v, block_offset, bow);
        let cpos = v.len();
        if checksum {
            v.extend_from_slice(&[0; 4]);
        }
        v.resize(size, 0);
        for (off, data) in objects {
            v[*off..*off + data.len()].copy_from_slice(data);
        }
        if checksum {
            let c = metadata_checksum(&v);
            v[cpos..cpos + 4].copy_from_slice(&c.to_le_bytes());
        }
        v
    }

    fn indirect_block(
        heap: u64,
        block_offset: u64,
        bow: usize,
        direct: &[(u64, Option<(u64, u32)>)],
        indirect: &[u64],
    ) -> Vec<u8> {
        let mut v = b"FHIB".to_vec();
        v.push(0);
        push_uint(&mut v, heap, 8);
        push_uint(&mut v, block_offset, bow);
        for (a, f) in direct {
            push_uint(&mut v, *a, 8);
            if let Some((s, m)) = f {
                push_uint(&mut v, *s, 8);
                v.extend_from_slice(&m.to_le_bytes());
            }
        }
        for a in indirect {
            push_uint(&mut v, *a, 8);
        }
        seal(v)
    }

    /// Managed ID with 2-byte offset and 1-byte length, padded to 7 bytes.
    fn managed_id(offset: u64, len: u8) -> Vec<u8> {
        let mut v = vec![0x00];
        push_uint(&mut v, offset, 2);
        v.push(len);
        v.resize(7, 0);
        v
    }

    fn xor(data: &[u8], _mask: u32) -> Result<Vec<u8>> {
        Ok(data.iter().map(|b| b ^ 0x5a).collect())
    }

    #[test]
    fn root_direct_block_with_checksum() {
        let mut f = FileBuilder::new();
        let mut spec = HeapSpec {
            flags: FLAG_CHECKSUM_DIRECT,
            width: 4,
            start: 256,
            max_direct: 1024,
            max_managed: 200,
            cur_rows: 0,
            ..HeapSpec::basic()
        };
        let heap = f.reserve(spec.header_len());
        let block = direct_block(heap, 0, 256, 2, true, &[(19, b"hello"), (40, b"world!")]);
        spec.root = f.place(&block);
        f.write_at(heap, &spec.bytes());

        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        assert_eq!(h.managed_offset_width(), 2);
        assert_eq!(h.managed_length_width(), 1);
        assert_eq!(h.get(&f.data, &managed_id(19, 5)).unwrap(), b"hello");
        assert_eq!(h.get(&f.data, &managed_id(40, 6)).unwrap(), b"world!");
        // Inside the block header, past the block end, zero length.
        assert!(h.get(&f.data, &managed_id(10, 5)).is_err());
        assert!(h.get(&f.data, &managed_id(250, 10)).is_err());
        assert!(h.get(&f.data, &managed_id(300, 1)).is_err());
        assert!(h.get(&f.data, &managed_id(19, 0)).is_err());
        // Checksum mismatch in the block body.
        let mut bad = f.data.clone();
        bad[spec.root as usize + 100] ^= 1;
        assert!(matches!(
            h.get(&bad, &managed_id(19, 5)),
            Err(Error::ChecksumMismatch { .. })
        ));
        // Header checksum mismatch.
        let mut bad = f.data.clone();
        bad[heap as usize + 20] ^= 1;
        assert!(matches!(
            FractalHeap::open(&bad, heap, Sizes::DEFAULT),
            Err(Error::ChecksumMismatch { .. })
        ));
        // Truncation.
        assert!(FractalHeap::open(&f.data[..heap as usize + 100], heap, Sizes::DEFAULT).is_err());
        assert!(
            h.get(&f.data[..spec.root as usize + 30], &managed_id(19, 5))
                .is_err()
        );
    }

    /// (heap ID, expected object) pairs.
    type Objects = Vec<(Vec<u8>, &'static [u8])>;

    /// Heap with W=2, S=64, M=128, 16-bit space and a 4-row root indirect
    /// block, whose row 3 holds 2-row child indirect blocks.
    ///
    /// Root rows: 0: [0, 64), [64, 128); 1: [128, 192), [192, 256);
    /// 2: [256, 384), [384, 512); 3 (indirect): [512, 768), [768, 1024).
    fn indirect_heap() -> (FileBuilder, u64, Objects) {
        let mut f = FileBuilder::new();
        let mut spec = HeapSpec {
            cur_rows: 4,
            ..HeapSpec::basic()
        };
        let heap = f.reserve(spec.header_len());
        let d0 = f.place(&direct_block(heap, 0, 64, 2, false, &[(15, b"alpha")]));
        let d3 = f.place(&direct_block(heap, 192, 64, 2, false, &[(20, b"bravo")]));
        let d5 = f.place(&direct_block(heap, 384, 128, 2, false, &[(65, b"charlie")]));
        let c0 = f.place(&direct_block(heap, 512, 64, 2, false, &[(15, b"delta")]));
        let c3 = f.place(&direct_block(heap, 704, 64, 2, false, &[(30, b"echo")]));
        let u = UNDEFINED_ADDRESS;
        let child = f.place(&indirect_block(
            heap,
            512,
            2,
            &[(c0, None), (u, None), (u, None), (c3, None)],
            &[],
        ));
        spec.root = f.place(&indirect_block(
            heap,
            0,
            2,
            &[
                (d0, None),
                (u, None),
                (u, None),
                (d3, None),
                (u, None),
                (d5, None),
            ],
            &[child, u],
        ));
        f.write_at(heap, &spec.bytes());
        let objects: Objects = vec![
            (managed_id(15, 5), b"alpha"),
            (managed_id(212, 5), b"bravo"),
            (managed_id(449, 7), b"charlie"),
            (managed_id(527, 5), b"delta"),
            (managed_id(734, 4), b"echo"),
        ];
        (f, heap, objects)
    }

    #[test]
    fn geometry() {
        let (f, heap, _) = indirect_heap();
        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        assert_eq!(h.max_direct_rows(), 3);
        let rows: Vec<_> = [0u64, 63, 64, 127, 128, 255, 256, 383, 384, 511, 512, 1023]
            .iter()
            .map(|&o| h.locate(o))
            .collect();
        assert_eq!(
            rows,
            [
                (0, 0),
                (0, 0),
                (0, 1),
                (0, 1),
                (1, 0),
                (1, 1),
                (2, 0),
                (2, 0),
                (2, 1),
                (2, 1),
                (3, 0),
                (3, 1)
            ]
        );
        assert_eq!(h.row_start(3).unwrap(), 512);
        assert_eq!(h.row_block_size(3).unwrap(), 256);
        assert_eq!(h.child_rows(3).unwrap(), 2);
        assert_eq!(h.child_rows(4).unwrap(), 3);
    }

    #[test]
    fn indirect_root_and_nested_blocks() {
        let (f, heap, objects) = indirect_heap();
        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        for (id, want) in &objects {
            assert_eq!(&h.get(&f.data, id).unwrap(), want);
        }
        // Unallocated direct block, unallocated indirect block, beyond root.
        assert!(h.get(&f.data, &managed_id(64 + 15, 3)).is_err());
        assert!(h.get(&f.data, &managed_id(512 + 64 + 20, 3)).is_err());
        assert!(h.get(&f.data, &managed_id(768 + 15, 3)).is_err());
        assert!(h.get(&f.data, &managed_id(1024 + 15, 3)).is_err());
        // Object crossing the end of its block.
        assert!(h.get(&f.data, &managed_id(60, 10)).is_err());
        // Mismatched block offset in a direct block header.
        let mut bad = f.data.clone();
        let d3 = f
            .data
            .windows(4)
            .enumerate()
            .filter(|(_, w)| w == b"FHDB")
            .nth(1);
        let d3 = d3.unwrap().0;
        bad[d3 + 13] ^= 1;
        assert!(h.get(&bad, &objects[1].0).is_err());
        assert_eq!(h.get(&bad, &objects[0].0).unwrap(), b"alpha");
    }

    #[test]
    fn indirect_cycle_is_rejected() {
        let mut f = FileBuilder::new();
        let mut spec = HeapSpec {
            cur_rows: 4,
            ..HeapSpec::basic()
        };
        let heap = f.reserve(spec.header_len());
        let root = f.reserve(5 + 8 + 2 + 8 * 8 + 4);
        let u = UNDEFINED_ADDRESS;
        f.write_at(
            root,
            &indirect_block(heap, 0, 2, &[(u, None); 6], &[root, root]),
        );
        spec.root = root;
        f.write_at(heap, &spec.bytes());
        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        assert!(h.get(&f.data, &managed_id(512 + 15, 3)).is_err());
    }

    #[test]
    fn tiny_objects() {
        let mut f = FileBuilder::new();
        let spec = HeapSpec::basic();
        let heap = f.place(&spec.bytes());
        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        let id = [0x20 | 4, b'h', b'e', b'l', b'l', b'o', 0];
        assert_eq!(h.get(&f.data, &id).unwrap(), b"hello");
        assert_eq!(h.parse_id(&id).unwrap(), HeapId::Tiny(b"hello".to_vec()));
        // Length beyond the ID.
        assert!(h.get(&f.data, &[0x20 | 9, 1, 2, 3, 4, 5, 6]).is_err());
        // Bad version and reserved type.
        assert!(h.get(&f.data, &[0x40, 0, 0, 0, 0, 0, 0]).is_err());
        assert!(h.get(&f.data, &[0x30, 0, 0, 0, 0, 0, 0]).is_err());

        // Extended form with 20-byte IDs.
        let mut f = FileBuilder::new();
        let spec = HeapSpec {
            heap_id_len: 20,
            ..HeapSpec::basic()
        };
        let heap = f.place(&spec.bytes());
        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        let mut id = vec![0x20, 17];
        id.extend(0..18u8);
        assert_eq!(h.get(&f.data, &id).unwrap(), (0..18u8).collect::<Vec<_>>());
    }

    #[test]
    fn huge_direct_objects() {
        let mut f = FileBuilder::new();
        let payload: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
        let obj = f.place(&payload);
        let spec = HeapSpec {
            heap_id_len: 17,
            ..HeapSpec::basic()
        };
        let heap = f.place(&spec.bytes());
        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        let mut id = vec![0x10];
        push_uint(&mut id, obj, 8);
        push_uint(&mut id, 300, 8);
        assert_eq!(h.get(&f.data, &id).unwrap(), payload);
        // Length past end of file.
        let mut id = vec![0x10];
        push_uint(&mut id, obj, 8);
        push_uint(&mut id, 1 << 40, 8);
        assert!(h.get(&f.data, &id).is_err());
    }

    #[test]
    fn huge_indirect_objects() {
        let mut f = FileBuilder::new();
        let a = f.place(b"first huge object");
        let b = f.place(b"second huge object, longer");
        let rec = |addr: u64, len: u64, id: u64| {
            let mut v = Vec::new();
            push_uint(&mut v, addr, 8);
            push_uint(&mut v, len, 8);
            push_uint(&mut v, id, 8);
            v
        };
        let leaf = f.place(&leaf_bytes(1, 512, &[rec(a, 17, 1), rec(b, 26, 2)]));
        let tree = f.place(&header_bytes(&HeaderSpec {
            tree_type: 1,
            node_size: 512,
            record_size: 24,
            depth: 0,
            root: leaf,
            root_n: 2,
            total: 2,
        }));
        let spec = HeapSpec {
            huge_btree: tree,
            ..HeapSpec::basic()
        };
        let heap = f.place(&spec.bytes());
        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        let id = |k: u8| [0x10, k, 0, 0, 0, 0, 0];
        assert_eq!(h.parse_id(&id(2)).unwrap(), HeapId::HugeIndirect { key: 2 });
        assert_eq!(h.get(&f.data, &id(1)).unwrap(), b"first huge object");
        assert_eq!(
            h.get(&f.data, &id(2)).unwrap(),
            b"second huge object, longer"
        );
        assert!(matches!(h.get(&f.data, &id(3)), Err(Error::NotFound(_))));
    }

    #[test]
    fn filtered_blocks_and_huge_objects() {
        let mut f = FileBuilder::new();
        let mut spec = HeapSpec {
            heap_id_len: 29,
            cur_rows: 1,
            filter: Some((0, 0, vec![2, 1, 0, 0, 1, 0, 0, 0])),
            ..HeapSpec::basic()
        };
        let heap = f.reserve(spec.header_len());
        let enc = |v: Vec<u8>| -> Vec<u8> { v.iter().map(|b| b ^ 0x5a).collect() };
        let b1 = enc(direct_block(heap, 64, 64, 2, false, &[(20, b"filtered!")]));
        let b1_addr = f.place(&b1);
        let huge = enc(b"a filtered huge object".to_vec());
        let huge_addr = f.place(&huge);
        spec.root = f.place(&indirect_block(
            heap,
            0,
            2,
            &[(UNDEFINED_ADDRESS, Some((0, 0))), (b1_addr, Some((64, 0)))],
            &[],
        ));
        f.write_at(heap, &spec.bytes());
        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        assert_eq!(h.header.filter_info, [2, 1, 0, 0, 1, 0, 0, 0]);
        let id = managed_id(84, 9);
        assert!(matches!(h.get(&f.data, &id), Err(Error::Unsupported(_))));
        assert_eq!(
            h.get_with_filter(&f.data, &id, Some(&xor)).unwrap(),
            b"filtered!"
        );
        let mut hid = vec![0x10];
        push_uint(&mut hid, huge_addr, 8);
        push_uint(&mut hid, 22, 8);
        hid.extend_from_slice(&0u32.to_le_bytes());
        push_uint(&mut hid, 22, 8);
        assert_eq!(
            h.get_with_filter(&f.data, &hid, Some(&xor)).unwrap(),
            b"a filtered huge object"
        );
        // Decoded size mismatch.
        let bad_size = |d: &[u8], _m: u32| -> Result<Vec<u8>> { Ok(d[1..].to_vec()) };
        assert!(h.get_with_filter(&f.data, &id, Some(&bad_size)).is_err());
        assert!(h.get_with_filter(&f.data, &hid, Some(&bad_size)).is_err());
    }

    #[test]
    fn filtered_root_direct_block() {
        let mut f = FileBuilder::new();
        let mut spec = HeapSpec {
            flags: FLAG_CHECKSUM_DIRECT,
            filter: Some((64, 0, vec![9; 4])),
            ..HeapSpec::basic()
        };
        let heap = f.reserve(spec.header_len());
        let block = direct_block(heap, 0, 64, 2, true, &[(30, b"rooted")]);
        spec.root = f.place(&block.iter().map(|b| b ^ 0x5a).collect::<Vec<_>>());
        f.write_at(heap, &spec.bytes());
        let h = FractalHeap::open(&f.data, heap, Sizes::DEFAULT).unwrap();
        assert_eq!(
            h.get_with_filter(&f.data, &managed_id(30, 6), Some(&xor))
                .unwrap(),
            b"rooted"
        );
    }

    #[test]
    fn mutations_never_panic() {
        let (f, heap, objects) = indirect_heap();
        let mut rng = Rng(0x0bad_5eed_1234_9876);
        for round in 0..4000u32 {
            let mut data = f.data.clone();
            let n = 1 + rng.next() % 4;
            for _ in 0..n {
                let i = (rng.next() % data.len() as u64) as usize;
                data[i] = rng.next() as u8;
            }
            if rng.next().is_multiple_of(8) {
                data.truncate((rng.next() % data.len() as u64) as usize);
            }
            // Occasionally re-seal the header so later structures get
            // exercised with a corrupted (but checksum-valid) header.
            if round.is_multiple_of(3) && data.len() >= heap as usize + 146 {
                let h = heap as usize;
                let c = metadata_checksum(&data[h..h + 142]);
                data[h + 142..h + 146].copy_from_slice(&c.to_le_bytes());
            }
            if let Ok(h) = FractalHeap::open(&data, heap, Sizes::DEFAULT) {
                for (id, _) in &objects {
                    let _ = h.get_with_filter(&data, id, Some(&xor));
                }
                let mut id = objects[(rng.next() % 5) as usize].0.clone();
                let j = (rng.next() % id.len() as u64) as usize;
                id[j] = rng.next() as u8;
                let _ = h.get_with_filter(&data, &id, Some(&xor));
                let _ = h.get(&data, &id[..(rng.next() % 8) as usize]);
            }
        }
    }
}
