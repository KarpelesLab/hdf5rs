//! End-to-end check of the raw data reading engine against the h5py
//! fixture corpus: superblock -> groups -> dataset object header -> header
//! messages -> `hdf5rs::read`, compared with `tests/fixtures/manifest.json`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use hdf5rs::Error;
use hdf5rs::format::Sizes;
use hdf5rs::format::btree_v2::{BTreeV2, Record, RecordContext};
use hdf5rs::format::fractal_heap::FractalHeap;
use hdf5rs::format::group_v1::read_group_v1;
use hdf5rs::format::messages::MessageType;
use hdf5rs::format::messages::dataspace::{Dataspace, DataspaceKind};
use hdf5rs::format::messages::datatype::Datatype;
use hdf5rs::format::messages::external_files::ExternalFiles;
use hdf5rs::format::messages::fill_value::{FillValue, OldFillValue};
use hdf5rs::format::messages::filter_pipeline::FilterPipeline;
use hdf5rs::format::messages::layout::DataLayout;
use hdf5rs::format::messages::link::{LinkMessage, LinkTarget};
use hdf5rs::format::messages::link_info::LinkInfoMessage;
use hdf5rs::format::messages::symbol_table::SymbolTableMessage;
use hdf5rs::format::object_header::ObjectHeader;
use hdf5rs::format::superblock::Superblock;
use hdf5rs::io::OffsetReader;
use hdf5rs::read::{DatasetStorage, HyperslabDim, Selection};

// ---------------------------------------------------------------- JSON --

#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) => n.parse().ok(),
            _ => None,
        }
    }
    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => &[],
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Json {
        self.ws();
        match self.s[self.i] {
            b'{' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b'}' {
                        self.i += 1;
                        break;
                    }
                    let Json::Str(k) = self.value() else {
                        panic!("key")
                    };
                    self.ws();
                    assert_eq!(self.s[self.i], b':');
                    self.i += 1;
                    let val = self.value();
                    v.push((k, val));
                    self.ws();
                    if self.s[self.i] == b',' {
                        self.i += 1;
                    }
                }
                Json::Obj(v)
            }
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b']' {
                        self.i += 1;
                        break;
                    }
                    v.push(self.value());
                    self.ws();
                    if self.s[self.i] == b',' {
                        self.i += 1;
                    }
                }
                Json::Arr(v)
            }
            b'"' => {
                self.i += 1;
                let mut out = String::new();
                loop {
                    let c = self.s[self.i];
                    self.i += 1;
                    match c {
                        b'"' => break,
                        b'\\' => {
                            let e = self.s[self.i];
                            self.i += 1;
                            match e {
                                b'n' => out.push('\n'),
                                b't' => out.push('\t'),
                                b'r' => out.push('\r'),
                                b'b' => out.push('\u{8}'),
                                b'f' => out.push('\u{c}'),
                                b'u' => {
                                    let h =
                                        std::str::from_utf8(&self.s[self.i..self.i + 4]).unwrap();
                                    self.i += 4;
                                    let mut cp = u32::from_str_radix(h, 16).unwrap();
                                    if (0xd800..0xdc00).contains(&cp) {
                                        let h2 =
                                            std::str::from_utf8(&self.s[self.i + 2..self.i + 6])
                                                .unwrap();
                                        self.i += 6;
                                        let lo = u32::from_str_radix(h2, 16).unwrap();
                                        cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                                    }
                                    out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                                }
                                other => out.push(other as char),
                            }
                        }
                        _ => {
                            // Re-assemble multi-byte UTF-8.
                            let start = self.i - 1;
                            let len = match c {
                                0..=0x7f => 1,
                                0xc0..=0xdf => 2,
                                0xe0..=0xef => 3,
                                _ => 4,
                            };
                            self.i = start + len;
                            out.push_str(std::str::from_utf8(&self.s[start..self.i]).unwrap());
                        }
                    }
                }
                Json::Str(out)
            }
            b't' => {
                self.i += 4;
                Json::Bool(true)
            }
            b'f' => {
                self.i += 5;
                Json::Bool(false)
            }
            b'n' => {
                self.i += 4;
                Json::Null
            }
            _ => {
                let start = self.i;
                while self.i < self.s.len()
                    && matches!(
                        self.s[self.i],
                        b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
                    )
                {
                    self.i += 1;
                }
                Json::Num(String::from_utf8(self.s[start..self.i].to_vec()).unwrap())
            }
        }
    }
}

fn parse_json(s: &str) -> Json {
    Parser {
        s: s.as_bytes(),
        i: 0,
    }
    .value()
}

// -------------------------------------------------------------- SHA-256 --

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[test]
fn sha256_known_answer() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

// ----------------------------------------------------------- navigation --

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

struct File {
    store: OffsetReader<Vec<u8>>,
    sizes: Sizes,
    root: u64,
}

fn open(name: &str) -> File {
    let bytes = std::fs::read(fixtures_dir().join(name)).unwrap();
    let sb = Superblock::read(&bytes[..]).unwrap();
    let root = sb.root_object_header_address();
    File {
        sizes: sb.sizes,
        store: sb.relative_store(bytes),
        root,
    }
}

/// Members (name -> object header address) of the group at `addr`.
fn members(f: &File, addr: u64) -> BTreeMap<String, u64> {
    let oh = ObjectHeader::read(&f.store, addr, f.sizes).unwrap();
    let mut out = BTreeMap::new();
    if let Some(m) = oh.find_message(MessageType::SYMBOL_TABLE) {
        let st = SymbolTableMessage::parse(&m.data, f.sizes).unwrap();
        for (name, e) in
            read_group_v1(&f.store, st.btree_address, st.local_heap_address, f.sizes).unwrap()
        {
            out.insert(name, e.object_header_address);
        }
        return out;
    }
    let mut add = |lm: LinkMessage| {
        if let LinkTarget::Hard { address } = lm.target {
            out.insert(lm.name, address);
        }
    };
    for m in oh.messages_of_type(MessageType::LINK) {
        add(LinkMessage::parse(&m.data, f.sizes).unwrap());
    }
    if let Some(m) = oh.find_message(MessageType::LINK_INFO) {
        let li = LinkInfoMessage::parse(&m.data, f.sizes).unwrap();
        if li.is_dense() {
            let heap = FractalHeap::open(&f.store, li.fractal_heap_address, f.sizes).unwrap();
            let ctx = RecordContext {
                heap_id_len: Some(usize::from(heap.header.heap_id_len)),
                ..RecordContext::default()
            };
            let tree = BTreeV2::open(&f.store, li.name_index_address, f.sizes, ctx).unwrap();
            for r in tree.records(&f.store).unwrap() {
                if let Record::LinkName { heap_id, .. } = r {
                    let raw = heap.get(&f.store, &heap_id).unwrap();
                    add(LinkMessage::parse(&raw, f.sizes).unwrap());
                }
            }
        }
    }
    out
}

fn resolve(f: &File, path: &str) -> u64 {
    let mut addr = f.root;
    for part in path.split('/').filter(|p| !p.is_empty()) {
        addr = *members(f, addr)
            .get(part)
            .unwrap_or_else(|| panic!("{path}: no member {part}"));
    }
    addr
}

struct Messages {
    space: Dataspace,
    layout: DataLayout,
    elem_size: Option<usize>,
    pipeline: Option<FilterPipeline>,
    fill: Option<FillValue>,
    old_fill: Option<OldFillValue>,
    external: Option<ExternalFiles>,
}

fn dataset_messages(f: &File, addr: u64) -> Messages {
    let oh = ObjectHeader::read(&f.store, addr, f.sizes).unwrap();
    let msg = |t: u16| oh.find_message(t).filter(|m| !m.is_shared());
    Messages {
        space: Dataspace::parse(&msg(MessageType::DATASPACE).unwrap().data, f.sizes).unwrap(),
        layout: DataLayout::parse(&msg(MessageType::LAYOUT).unwrap().data, f.sizes).unwrap(),
        elem_size: msg(MessageType::DATATYPE)
            .map(|m| Datatype::parse(&m.data).unwrap().size as usize),
        pipeline: msg(MessageType::FILTER_PIPELINE)
            .map(|m| FilterPipeline::parse(&m.data, f.sizes).unwrap()),
        fill: msg(MessageType::FILL_VALUE).map(|m| FillValue::parse(&m.data, f.sizes).unwrap()),
        old_fill: msg(MessageType::FILL_VALUE_OLD)
            .map(|m| OldFillValue::parse(&m.data, f.sizes).unwrap()),
        external: msg(MessageType::EXTERNAL_FILES)
            .map(|m| ExternalFiles::parse(&m.data, f.sizes).unwrap()),
    }
}

// -------------------------------------------------------------- checks --

fn flatten<'a>(j: &'a Json, out: &mut Vec<&'a Json>) {
    match j {
        Json::Arr(v) => v.iter().for_each(|x| flatten(x, out)),
        other => out.push(other),
    }
}

/// Element bytes converted to little-endian order.
fn to_le(raw: &[u8], size: usize, big_endian: bool) -> Vec<u8> {
    if !big_endian {
        return raw.to_vec();
    }
    raw.chunks(size)
        .flat_map(|c| c.iter().rev().copied().collect::<Vec<_>>())
        .collect()
}

fn decode_float(le: &[u8]) -> f64 {
    match le.len() {
        4 => f32::from_le_bytes(le.try_into().unwrap()) as f64,
        8 => f64::from_le_bytes(le.try_into().unwrap()),
        2 => {
            let h = u16::from_le_bytes(le.try_into().unwrap());
            let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            let e = ((h >> 10) & 0x1f) as i32;
            let m = (h & 0x3ff) as f64;
            sign * match e {
                0 => m * 2f64.powi(-24),
                31 if m == 0.0 => f64::INFINITY,
                31 => f64::NAN,
                _ => (1.0 + m / 1024.0) * 2f64.powi(e - 15),
            }
        }
        _ => panic!("float size"),
    }
}

/// Compare the raw bytes of a numeric dataset with the manifest values.
/// Returns false when the dtype is not checked.
fn check_values(path: &str, dtype: &Json, obj: &Json, raw: &[u8]) -> bool {
    let class = dtype.get("class").and_then(Json::str).unwrap_or("");
    let Some(size) = dtype.get("size").and_then(Json::u64).map(|s| s as usize) else {
        return false;
    };
    if let Some(hex) = obj.get("raw_elements_hex") {
        let want: String = hex.arr().iter().filter_map(Json::str).collect();
        let got: String = raw.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, want, "{path}: raw string bytes");
        return true;
    }
    if class != "integer" && class != "float" {
        return false;
    }
    let be = dtype.get("order").and_then(Json::str) == Some("be");
    let le = to_le(raw, size, be);
    if let Some(summary) = obj.get("data_summary") {
        let want = summary.get("sha256_le").and_then(Json::str).unwrap();
        assert_eq!(sha256_hex(&le), want, "{path}: sha256 of values");
        return true;
    }
    let Some(data) = obj.get("data") else {
        return false;
    };
    if *data == Json::Null {
        assert!(raw.is_empty(), "{path}: null dataspace");
        return true;
    }
    let mut vals = Vec::new();
    flatten(data, &mut vals);
    assert_eq!(vals.len() * size, raw.len(), "{path}: element count");
    let signed = dtype.get("signed") == Some(&Json::Bool(true));
    for (i, (v, el)) in vals.iter().zip(le.chunks(size)).enumerate() {
        if class == "integer" {
            let mut buf = [0u8; 16];
            buf[..size].copy_from_slice(el);
            if signed && el[size - 1] & 0x80 != 0 {
                buf[size..].fill(0xff);
            }
            let got = i128::from_le_bytes(buf);
            let want: i128 = match v {
                Json::Num(n) => n.parse().unwrap(),
                Json::Bool(b) => *b as i128,
                other => panic!("{path}: {other:?}"),
            };
            assert_eq!(got, want, "{path}[{i}]");
        } else {
            let got = decode_float(el);
            let want = match v {
                Json::Num(n) => n.parse::<f64>().unwrap(),
                Json::Str(s) if s == "nan" => f64::NAN,
                Json::Str(s) if s == "inf" => f64::INFINITY,
                Json::Str(s) if s == "-inf" => f64::NEG_INFINITY,
                other => panic!("{path}: {other:?}"),
            };
            assert!(
                got == want || (got.is_nan() && want.is_nan()),
                "{path}[{i}]: {got} != {want}"
            );
        }
    }
    true
}

/// Naive reference: extract a hyperslab from a full row-major array.
fn reference(full: &[u8], dims: &[u64], sel: &[HyperslabDim], esize: usize) -> Vec<u8> {
    let coords: Vec<Vec<u64>> = sel
        .iter()
        .map(|h| {
            (0..h.count)
                .flat_map(|k| (0..h.block).map(move |b| h.start + k * h.stride + b))
                .collect()
        })
        .collect();
    let mut out = Vec::new();
    let mut idx = vec![0usize; dims.len()];
    if coords.iter().any(Vec::is_empty) {
        return out;
    }
    loop {
        let mut lin = 0u64;
        for (d, &i) in idx.iter().enumerate() {
            lin = lin * dims[d] + coords[d][i];
        }
        let p = lin as usize * esize;
        out.extend_from_slice(&full[p..p + esize]);
        let mut k = dims.len();
        loop {
            if k == 0 {
                return out;
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] < coords[k].len() {
                break;
            }
            idx[k] = 0;
        }
    }
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n.max(1)
    }
}

fn random_hyperslab(rng: &mut Rng, dims: &[u64]) -> Vec<HyperslabDim> {
    dims.iter()
        .map(|&d| {
            if d == 0 {
                return HyperslabDim::new(0, 1, 0, 1);
            }
            let start = rng.below(d);
            let block = 1 + rng.below((d - start).min(4));
            let stride = block + rng.below(4);
            let max_count = (d - start - block) / stride + 1;
            let count = 1 + rng.below(max_count);
            HyperslabDim::new(start, stride, count, block)
        })
        .collect()
}

#[test]
fn fixtures_read_raw() {
    let manifest_text = std::fs::read_to_string(fixtures_dir().join("manifest.json")).unwrap();
    let manifest = parse_json(&manifest_text);
    let Json::Obj(files) = manifest.get("files").unwrap() else {
        panic!()
    };
    let mut checked = 0;
    let mut unsupported = Vec::new();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for (fname, fobj) in files {
        let f = open(fname);
        let Json::Obj(objects) = fobj.get("objects").unwrap() else {
            panic!()
        };
        for (path, obj) in objects {
            if obj.get("kind").and_then(Json::str) != Some("dataset") {
                continue;
            }
            let what = format!("{fname}:{path}");
            let addr = resolve(&f, path);
            let m = dataset_messages(&f, addr);
            let dtype = obj.get("dtype").unwrap();
            let elem_size = m
                .elem_size
                .or_else(|| dtype.get("size").and_then(Json::u64).map(|s| s as usize))
                .unwrap();
            let storage = DatasetStorage::new(&f.store, f.sizes, &m.layout, &m.space, elem_size)
                .with_filters(m.pipeline.as_ref())
                .with_fill_value(m.fill.as_ref())
                .with_old_fill_value(m.old_fill.as_ref())
                .with_external(m.external.as_ref(), None);
            let raw = match storage.read(&Selection::All) {
                Ok(raw) => raw,
                Err(Error::Unsupported(msg)) => {
                    unsupported.push(format!("{what}: {msg}"));
                    continue;
                }
                Err(e) => panic!("{what}: {e}"),
            };
            let count = m.space.element_count().unwrap() as usize;
            assert_eq!(raw.len(), count * elem_size, "{what}: output size");
            if check_values(&what, dtype, obj, &raw) {
                checked += 1;
            }

            if let DataLayout::Chunked(_) = m.layout {
                let chunks = storage.chunks().unwrap_or_else(|e| panic!("{what}: {e}"));
                if let Some(n) = obj.get("num_chunks_allocated").and_then(Json::u64) {
                    assert_eq!(chunks.len() as u64, n, "{what}: allocated chunk count");
                }
                if let Some(expected) = obj.get("chunks") {
                    let mut want: Vec<(Vec<u64>, u64, u64, u64)> = expected
                        .arr()
                        .iter()
                        .map(|c| {
                            (
                                c.get("offset")
                                    .unwrap()
                                    .arr()
                                    .iter()
                                    .map(|x| x.u64().unwrap())
                                    .collect(),
                                c.get("address").unwrap().u64().unwrap(),
                                c.get("size").unwrap().u64().unwrap(),
                                c.get("filter_mask").unwrap().u64().unwrap(),
                            )
                        })
                        .collect();
                    let mut got: Vec<_> = chunks
                        .iter()
                        .map(|c| {
                            (
                                c.offset.clone(),
                                c.address,
                                c.size,
                                u64::from(c.filter_mask),
                            )
                        })
                        .collect();
                    want.sort();
                    got.sort();
                    assert_eq!(got, want, "{what}: chunk list");
                }
            }

            // Hyperslabs against a naive extraction from the full read.
            if m.space.kind == DataspaceKind::Simple && count > 0 {
                for _ in 0..8 {
                    let h = random_hyperslab(&mut rng, &m.space.dims);
                    let got = storage
                        .read(&Selection::hyperslab(h.clone()))
                        .unwrap_or_else(|e| panic!("{what}: {h:?}: {e}"));
                    let want = reference(&raw, &m.space.dims, &h, elem_size);
                    assert!(got == want, "{what}: hyperslab {h:?}");
                }
            }
        }
    }
    eprintln!("checked {checked} datasets; unsupported:");
    for u in &unsupported {
        eprintln!("  {u}");
    }
    assert!(checked > 60, "only {checked} datasets checked");
    // Only filters we do not implement (and virtual datasets) may be
    // unsupported.
    for u in &unsupported {
        assert!(
            u.contains("filter id") || u.contains("virtual"),
            "unexpected unsupported: {u}"
        );
    }
}

#[test]
#[cfg(feature = "deflate")]
fn bounded_limits_refuse_large_reads() {
    let f = open("big_gzip.h5");
    let addr = resolve(&f, "/arange_i32");
    let m = dataset_messages(&f, addr);
    let storage = DatasetStorage::new(&f.store, f.sizes, &m.layout, &m.space, 4)
        .with_filters(m.pipeline.as_ref())
        .with_limits(hdf5rs::read::ReadLimits::with_max_output(1 << 20));
    assert!(matches!(
        storage.read(&Selection::All),
        Err(Error::LimitExceeded(_))
    ));
    let part = storage.read(&Selection::range(100_000..100_010)).unwrap();
    let vals: Vec<i32> = part
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals, (100_000..100_010).collect::<Vec<i32>>());
}
