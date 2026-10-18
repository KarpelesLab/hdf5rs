# Interoperability fixtures

Small HDF5 files written by the reference implementation (libhdf5 through
h5py, used strictly as a black box) plus a machine-readable description of
what the reference reader reports for each of them.

**Do not edit these files by hand.** They are produced by
[`tools/gen_fixtures.py`](../../tools/gen_fixtures.py); see
[`tools/README.md`](../../tools/README.md) for how to regenerate them.
The directory is excluded from the published crate.

- `*.h5` — the fixtures.
- `files.txt` — one line per fixture: `<file name> <superblock offset>`
  (easy to consume from Rust tests without a JSON parser).
- `manifest.json` — expectations (schema below).

## Files

| File | Superblock | Contents |
|------|-----------:|----------|
| `attributes_earliest.h5` | 0 | Attributes of every type category on root/group/dataset; 20 attributes on one object header |
| `attributes_latest.h5` | 3 | Same attributes, plus dense attribute storage (20 attrs), an 80 000-byte attribute (> 64 KiB, dense), creation-order tracked attributes (compact and dense) |
| `big_gzip.h5` | 3 | 1 Mi-element `arange` int32 and a 512×512 f64 array, shuffle+gzip — for performance tests |
| `dataspaces.h5` | 3 | Scalar, null (`h5py.Empty`), 1-D … 4-D, zero-size dims, unlimited and bounded max dims |
| `dataspaces_earliest.h5` | 0 | Scalar, null, simple, zero-size and unlimited dataspaces |
| `dtypes_earliest.h5` | 0 | Every datatype class (see below), earliest format |
| `dtypes_latest.h5` | 3 | Every datatype class, latest format |
| `filters.h5` | 3 | deflate levels 1/4/9, shuffle, shuffle+gzip, fletcher32, shuffle+gzip+fletcher32, lzf, scale-offset (int and float), N-bit (12-bit and 20-bit@offset 4), gzip on big-endian, random and vlen-string data |
| `filters_earliest.h5` | 0 | gzip, shuffle+gzip+fletcher32, lzf with v1 B-tree chunk index |
| `filters_plugins.h5` | 3 | Third-party filters (blosc/lz4, blosc/zstd, zstd, lz4, bzip2, bitshuffle) — *unsupported* by default, decoded when the matching Cargo features (or `all-filters`) are enabled |
| `fsm_paged.h5` | 3 | Paged file-space strategy with persistent free-space managers, after deletions |
| `groups_earliest.h5` | 0 | Symbol-table groups: nested, empty, 100 entries (multi-node B-tree), soft (absolute, relative, dangling), external and repeated hard links |
| `groups_latest.h5` | 3 | Link-message groups: same as above with dense link storage for 100 links, creation-order tracked groups (compact and dense) |
| `layouts_earliest.h5` | 0 | Compact, contiguous (written and unwritten), chunked with v1 B-tree (incl. multi-level, partial, gzip, unlimited) |
| `layouts_latest.h5` | 3 | Compact, contiguous, and every layout-v4 chunk index: single chunk, implicit, fixed array (incl. paged), extensible array (incl. 400 chunks), v2 B-tree (incl. 400 chunks); partial and never-written chunked datasets with fill values; edge chunks |
| `resized.h5` | 3 | Chunked datasets grown (1-D, 2-D) and shrunk after creation |
| `resized_earliest.h5` | 0 | Chunked dataset grown after creation |
| `sizes_4byte_earliest.h5` | 0 | 4-byte offsets and lengths |
| `sizes_4byte_latest.h5` | 3 | 4-byte offsets and lengths |
| `superblock_v0.h5` | 0 | `libver="earliest"` minimal file |
| `superblock_v2.h5` | 2 | `libver=("v108","v108")` minimal file |
| `superblock_v3.h5` | 3 | `libver="latest"` minimal file (also the target of the external links) |
| `userblock.h5` | 0 | 512-byte user block (starts with `hdf5rs userblock test\n`), superblock at offset 512 |
| `vds.h5` | 3 | Virtual datasets mapping a same-file source and `vds_source.h5:/src`, plus a strided mapping |
| `vds_source.h5` | 3 | Source for `vds.h5` |

Datatypes covered by `dtypes_*.h5`: integers 8/16/32/64-bit signed and
unsigned in both byte orders; f16/f32/f64 in both byte orders (incl. ±inf,
NaN, max, smallest normal); bool (h5py enum `FALSE`/`TRUE`); complex
(stored by h5py as a compound `{r, i}`); fixed-length strings
(ASCII/UTF-8 × null-terminated/null-padded/space-padded, multibyte UTF-8);
variable-length strings (ASCII/UTF-8, scalar); vlen int32/float64 sequences;
compounds (nested, array member, padded/explicit offsets, vlen members);
enums (int8, big-endian int16); array types; opaque (with and without
tag); bitfields (8/16/32-bit, BE); committed datatypes (used by a dataset
and an attribute); object references (incl. null) and dataset-region
references (hyperslab and all).

Not generated (not reachable through h5py): superblock version 1 (needs
non-default B-tree K values), shared object header messages (SOHM tables),
the time datatype, szip, external raw-data storage.

## `manifest.json` schema

```text
{
  "schema_version": 1,
  "generator": "tools/gen_fixtures.py",
  "h5py_version", "hdf5_version", "numpy_version", "hdf5plugin_version": string|null,
  "files": { "<file name>": File }
}
```

### File

| Key | Meaning |
|-----|---------|
| `description` | Human-readable summary |
| `libver` | libver bounds passed when creating (`"earliest"`, `"latest"` or `[low, high]`) |
| `size`, `sha256` | File size in bytes and SHA-256 of the whole file |
| `superblock_offset` | Byte offset of the signature (0, or 512 for `userblock.h5`) |
| `superblock_version` | Superblock version byte as found in the file |
| `userblock_size`, `userblock_prefix` | Only for files with a user block |
| `sizeof_offsets`, `sizeof_lengths` | Only for files with non-default sizes |
| `objects` | Map of absolute object path → Object. Every object appears once, under the first path it is reached by (depth-first, in the group's iteration order). |

### Object

Common keys: `kind` (`"group"`, `"dataset"`, `"datatype"`), `attributes`
(list of Attribute, in h5py iteration order: creation order when tracked,
otherwise by name) and optionally `expect` (see below).

**group**: `links` — list, in iteration order (creation order when tracked,
otherwise name order), of:

- `{"name", "type": "hard", "target": "/abs/path"}` — `target` is the path
  under which the object is listed in `objects` (so a second hard link to the
  same object points at the first path);
- `{"name", "type": "soft", "target": "<link value>"}` (may dangle);
- `{"name", "type": "external", "file", "target"}`.

**datatype** (committed): `dtype`.

**dataset**:

| Key | Meaning |
|-----|---------|
| `dtype` | Datatype (below) |
| `dataspace` | Dataspace (below) |
| `layout` | `"compact"`, `"contiguous"`, `"chunked"`, `"virtual"` |
| `chunk_shape` | Chunked only |
| `filters` | Pipeline order: `{"id", "name", "flags", "client_data": [ints]}` (client data as reported after the library's `set_local` step) |
| `storage_size` | Bytes of raw data allocated (after filters) |
| `data_offset` | Contiguous only: absolute file address of the data, `null` if never allocated |
| `num_chunks_allocated` | Chunked only |
| `chunks` | Chunked with ≤ 64 allocated chunks: `{"offset": [logical element coords], "filter_mask", "address", "size"}` in the library's iteration order |
| `virtual_mappings` | Virtual only: `{"file", "dataset", "virtual_selection", "source_selection"}` (Selection below) |
| `fill_value` | When user-defined |
| `alloc_time` | `"early"`, `"late"`, `"incremental"` |
| `data` | Full contents (≤ 1024 elements), nested lists in row-major order; `null` for a null dataspace; for scalars the bare value |
| `data_summary` | Instead of `data` for larger datasets: `{"count", "sha256_le", "sum", "min", "max", "first": [8 values], "last": [8 values]}` — `sha256_le` is the SHA-256 of the flattened values encoded as little-endian bytes of the dataset's element size |
| `data_encoding` | `"raw_hex"` when the reference library offered no conversion (opaque with a tag): each element is `{"hex": ...}` of its stored bytes |
| `raw_elements_hex` | Fixed-length strings only: exact stored bytes of each element (padding included), flattened row-major |

### Attribute

`{"name", "dtype", "dataspace", "data"}` (or `data_summary` instead of
`data` for > 1024 elements). Same value encoding as datasets.

### Datatype

Always `class` and (except variable-length types) `size` in bytes.

| class | Extra keys |
|-------|-----------|
| `integer` | `order` (`"le"`/`"be"`), `signed`; `precision` and `bit_offset` if not the full width |
| `float` | `order`, `fields`: `{sign_pos, exp_pos, exp_size, mant_pos, mant_size, exp_bias}` |
| `string` | `size` (int or `"variable"`), `charset` (`"ascii"`/`"utf8"`), `padding` (`"nullterm"`/`"nullpad"`/`"spacepad"`) |
| `bitfield` | `order` |
| `opaque` | `tag` |
| `compound` | `members`: `[{"name", "offset", "type"}]` |
| `enum` | `base` (Datatype), `members`: `{name: value}` |
| `vlen` | `base` (sequence; no `size`) |
| `array` | `dims`, `base` |
| `reference` | `kind`: `"object"` or `"region"` |

`committed: true` and `committed_path` are added when the type is a
committed (named) datatype or refers to one.

### Dataspace

`{"class": "null"}`, `{"class": "scalar", "shape": []}` or
`{"class": "simple", "shape": [..], "maxshape": [.. | null]}` (`null` =
unlimited).

### Value encoding

| Type | JSON |
|------|------|
| integer, enum | number (enum: the underlying integer) |
| bool (enum FALSE/TRUE) | `true` / `false` |
| bitfield | unsigned integer (bytes interpreted in the type's byte order) |
| float | number, or the strings `"nan"`, `"inf"`, `"-inf"` |
| string | JSON string as returned by the reference reader (padding/terminator stripped); `{"hex": ..}` if not valid UTF-8 |
| opaque | `{"hex": "..."}` |
| compound | object keyed by member name |
| array, vlen sequence | list |
| complex (on disk: compound `{r, i}`) | `[re, im]` (h5py reads it back as a complex number) |
| object reference | `{"ref": "/target/path"}`, `null` for a null reference |
| region reference | `{"region_ref": "/dataset", ...Selection}` |

### Selection

`{"dataspace", "selection": "all"|"none"|"hyperslab"|"points", "npoints"}`
plus, for hyperslabs, `blocks: [{"start", "end"}]` (inclusive corners) and,
when regular, `regular: {start, stride, count, block}`; for points,
`points: [[coords]]`.

### `expect`

Extra expectations recorded by the generator rather than read back from
h5py (h5py does not expose them). They follow from the documented library
rules and were sanity-checked by looking for the corresponding structure
signatures in the file:

- `chunk_index`: `"single_chunk"`, `"implicit"`, `"fixed_array"`,
  `"extensible_array"`, `"btree_v2"`, `"btree_v1"`;
- `link_storage` / `attribute_storage`: `"compact"` or `"dense"`;
- `creation_order_tracked`, `attr_creation_order_tracked`: `true`;
- `storage_allocated: false`;
- `note`: free text.
