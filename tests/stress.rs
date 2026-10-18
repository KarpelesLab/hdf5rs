//! Deterministic mutation stress test (a stable-Rust stand-in for fuzzing).
//!
//! Every fixture in `tests/fixtures` is mutated with a seeded PRNG (bit
//! flips, byte and word overwrites with interesting values, truncations,
//! insertions/deletions, splices from other fixtures) and then opened and
//! walked completely (`support/walk.rs`): groups, links, attributes,
//! datasets, chunk lists, raw reads and decoded values.
//!
//! Failures detected:
//! - panics (caught per input, reported with the input saved to disk);
//! - hangs (a watchdog aborts the process when one input takes too long);
//! - excessive allocation (a counting global allocator flags inputs whose
//!   single largest allocation or peak live memory exceeds a budget).
//!
//! Environment variables:
//! - `HDF5RS_FUZZ_ITERS`: number of mutated inputs (default 400);
//! - `HDF5RS_FUZZ_SEED`: PRNG seed (default fixed);
//! - `HDF5RS_FUZZ_DUMP`: directory where failing inputs are written
//!   (default: the system temporary directory);
//! - `HDF5RS_FUZZ_REPRO`: path of one input to replay instead of mutating;
//! - `HDF5RS_FUZZ_TIMEOUT`: per-input watchdog timeout in seconds
//!   (default 120).
//!
//! Most version 2 structures carry checksums, which reject almost every
//! mutation early. Building with `RUSTFLAGS="--cfg fuzzing"` (as cargo-fuzz
//! does) disables metadata checksum verification so mutations reach the
//! deeper parsers.
//!
//! Run long campaigns in release mode, e.g.
//! `HDF5RS_FUZZ_ITERS=1000000 cargo test --release --test stress`.

#[path = "support/parsers.rs"]
mod parsers;
#[path = "support/walk.rs"]
mod walk;

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hdf5rs::filters::{FilterContext, FilterRegistry};
use hdf5rs::format::Sizes;
use hdf5rs::{Dataset, File, Link, Object};
use walk::{WalkLimits, WalkStats, walk_bytes};

// ------------------------------------------------- counting allocator --

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

fn note_alloc(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
    LARGEST.fetch_max(size, Ordering::Relaxed);
}

// SAFETY: forwards to the system allocator, only adding bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        note_alloc(new_size);
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Largest single allocation tolerated while walking one input.
const MAX_SINGLE_ALLOC: usize = 96 << 20;
/// Largest peak live memory (above the baseline) tolerated per input.
const MAX_PEAK: usize = 192 << 20;

// ------------------------------------------------------------- PRNG --

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

const INTERESTING: [u64; 14] = [
    0,
    1,
    2,
    7,
    8,
    0x10,
    0x7f,
    0x80,
    0xff,
    0x100,
    0xffff,
    0x7fff_ffff,
    0xffff_ffff,
    u64::MAX,
];

/// Apply one random mutation to `data`.
fn mutate(rng: &mut Rng, data: &mut Vec<u8>, corpus: &[Vec<u8>]) {
    if data.is_empty() {
        data.push(rng.next() as u8);
        return;
    }
    let len = data.len();
    let pos = rng.below(len);
    match rng.below(10) {
        0 | 1 => data[pos] ^= 1 << rng.below(8),
        2 => data[pos] = rng.next() as u8,
        3 => data[pos] = INTERESTING[rng.below(INTERESTING.len())] as u8,
        4 | 5 => {
            // Overwrite a little-endian word with an interesting value,
            // occasionally related to the file size.
            let width = [2usize, 4, 8][rng.below(3)];
            let v = match rng.below(4) {
                0 => len as u64 + rng.below(64) as u64,
                1 => rng.next(),
                _ => INTERESTING[rng.below(INTERESTING.len())],
            };
            let bytes = v.to_le_bytes();
            for (i, b) in bytes.iter().take(width).enumerate() {
                if let Some(slot) = data.get_mut(pos + i) {
                    *slot = *b;
                }
            }
        }
        6 => data.truncate(pos),
        7 => {
            // Splice a block from another fixture (or this one).
            let src = &corpus[rng.below(corpus.len())];
            if src.is_empty() {
                return;
            }
            let from = rng.below(src.len());
            let n = (1 + rng.below(256)).min(src.len() - from);
            for i in 0..n {
                if let Some(slot) = data.get_mut(pos + i) {
                    *slot = src[from + i];
                }
            }
        }
        8 => {
            // Copy a block within the file.
            let from = rng.below(len);
            let n = (1 + rng.below(64)).min(len - from);
            let block = data[from..from + n].to_vec();
            for (i, b) in block.into_iter().enumerate() {
                if let Some(slot) = data.get_mut(pos + i) {
                    *slot = b;
                }
            }
        }
        _ => {
            if rng.below(2) == 0 {
                let n = 1 + rng.below(16);
                for _ in 0..n {
                    data.insert(pos, rng.next() as u8);
                }
            } else {
                let n = (1 + rng.below(16)).min(len - pos);
                data.drain(pos..pos + n);
            }
        }
    }
}

// ---------------------------------------------------------- harness --

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_corpus() -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(fixtures_dir())
        .expect("fixtures directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "h5"))
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            (name, std::fs::read(&p).unwrap())
        })
        .collect();
    v.sort();
    v
}

fn env_u64(name: &str) -> Option<u64> {
    let s = std::env::var(name).ok()?;
    let s = s.trim();
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

fn dump_dir() -> PathBuf {
    std::env::var_os("HDF5RS_FUZZ_DUMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

fn save(input: &[u8], tag: &str) -> PathBuf {
    let dir = dump_dir();
    let _ = std::fs::create_dir_all(&dir);
    let p = dir.join(format!("hdf5rs-stress-{tag}.h5"));
    let _ = std::fs::write(&p, input);
    p
}

/// Shared with the watchdog: the input being processed and when it began.
struct Current {
    input: Mutex<Vec<u8>>,
    tag: Mutex<String>,
    /// Milliseconds since `epoch` at which the current input started
    /// (`u64::MAX` = idle).
    started: AtomicU64,
}

fn spawn_watchdog(cur: Arc<Current>, epoch: Instant, timeout: Duration) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let s = cur.started.load(Ordering::Relaxed);
            if s == u64::MAX {
                continue;
            }
            let now = epoch.elapsed().as_millis() as u64;
            if now.saturating_sub(s) > timeout.as_millis() as u64 {
                let input = cur.input.lock().unwrap_or_else(|e| e.into_inner()).clone();
                let tag = cur.tag.lock().unwrap_or_else(|e| e.into_inner()).clone();
                let p = save(&input, &format!("hang-{tag}"));
                eprintln!(
                    "stress: input {tag} ran longer than {timeout:?}; saved to {}",
                    p.display()
                );
                std::process::abort();
            }
        }
    });
}

#[derive(Default)]
struct Failures(Vec<String>);

/// Run harness `target` ("walk" or a `parsers::TARGETS` name) on `input`.
fn run_target(target: &str, input: Vec<u8>) -> WalkStats {
    if target == "walk" {
        return walk_bytes(input, WalkLimits::default());
    }
    if let Some((_, f)) = parsers::TARGETS.iter().find(|(n, _)| *n == target) {
        f(&input);
    }
    WalkStats::default()
}

/// Run one input, checking for panics and allocation blow-ups.
fn run_one(
    target: &str,
    input: &[u8],
    tag: &str,
    cur: &Current,
    epoch: Instant,
    fails: &mut Failures,
) -> WalkStats {
    let tag = &format!("{target}-{tag}");
    *cur.input.lock().unwrap() = input.to_vec();
    *cur.tag.lock().unwrap() = tag.to_owned();
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    LARGEST.store(0, Ordering::Relaxed);
    let t0 = Instant::now();
    cur.started
        .store(epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
    let owned = input.to_vec();
    let r = catch_unwind(AssertUnwindSafe(|| run_target(target, owned)));
    cur.started.store(u64::MAX, Ordering::Relaxed);
    let elapsed = t0.elapsed();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    let largest = LARGEST.load(Ordering::Relaxed);
    if r.is_err() {
        let p = save(input, &format!("panic-{tag}"));
        fails
            .0
            .push(format!("panic on {tag} (saved to {})", p.display()));
    }
    if largest > MAX_SINGLE_ALLOC || peak > MAX_PEAK {
        let p = save(input, &format!("alloc-{tag}"));
        fails.0.push(format!(
            "{tag}: largest allocation {largest} B, peak {peak} B (saved to {})",
            p.display()
        ));
    }
    if elapsed > Duration::from_secs(20) {
        eprintln!("stress: slow input {tag}: {elapsed:?}");
    }
    r.unwrap_or_default()
}

#[test]
fn mutated_fixtures_never_panic() {
    let epoch = Instant::now();
    let cur = Arc::new(Current {
        input: Mutex::new(Vec::new()),
        tag: Mutex::new(String::new()),
        started: AtomicU64::new(u64::MAX),
    });
    let timeout = Duration::from_secs(env_u64("HDF5RS_FUZZ_TIMEOUT").unwrap_or(120));
    spawn_watchdog(cur.clone(), epoch, timeout);
    let mut fails = Failures::default();

    if let Some(p) = std::env::var_os("HDF5RS_FUZZ_REPRO") {
        let input = std::fs::read(Path::new(&p)).expect("repro input");
        let target = std::env::var("HDF5RS_FUZZ_TARGET").unwrap_or_else(|_| "walk".into());
        run_one(&target, &input, "repro", &cur, epoch, &mut fails);
        assert!(fails.0.is_empty(), "{}", fails.0.join("\n"));
        return;
    }

    let corpus = load_corpus();
    assert!(!corpus.is_empty());
    let bytes: Vec<Vec<u8>> = corpus.iter().map(|(_, b)| b.clone()).collect();

    // Inputs that once failed (minimized crash corpus).
    let regressions = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/regressions");
    if let Ok(dir) = std::fs::read_dir(&regressions) {
        for p in dir.filter_map(|e| e.ok()).map(|e| e.path()) {
            if p.extension().is_some_and(|x| x == "h5") {
                let data = std::fs::read(&p).unwrap();
                let name = p.file_name().unwrap().to_string_lossy().into_owned();
                // `<target>-<anything>.h5`, or a plain file name for walks.
                let target = parsers::TARGETS
                    .iter()
                    .map(|(n, _)| *n)
                    .find(|n| name.starts_with(&format!("{n}-")))
                    .unwrap_or("walk");
                run_one(target, &data, &name, &cur, epoch, &mut fails);
            }
        }
    }

    // Unmutated fixtures first: they must walk cleanly and within budget.
    let mut total = WalkStats::default();
    for (name, data) in &corpus {
        let s = run_one("walk", data, name, &cur, epoch, &mut fails);
        total.objects += s.objects;
        total.attributes += s.attributes;
        total.datasets_read += s.datasets_read;
        total.bytes_read += s.bytes_read;
    }
    eprintln!("stress: unmutated corpus walk: {total:?}");
    assert!(
        total.objects > 300 && total.attributes > 100 && total.datasets_read > 200,
        "walker coverage regressed: {total:?}"
    );

    let seeds = parser_seeds(&corpus);
    if let Some(dir) = std::env::var_os("HDF5RS_FUZZ_WRITE_CORPUS") {
        write_corpus(Path::new(&dir), &corpus, &seeds);
        return;
    }
    // Unmutated parser seeds.
    for (i, sd) in seeds.iter().enumerate() {
        run_one(
            sd.target(),
            &sd.build(&bytes),
            &format!("seed{i}"),
            &cur,
            epoch,
            &mut fails,
        );
    }
    let mut targets: Vec<&str> = seeds.iter().map(Seed::target).collect();
    targets.sort();
    targets.dedup();
    eprintln!("stress: {} parser seeds for {targets:?}", seeds.len());
    assert!(
        targets.len() >= 9,
        "parser seed coverage regressed: {targets:?}"
    );

    let iters = env_u64("HDF5RS_FUZZ_ITERS").unwrap_or(400);
    let seed = env_u64("HDF5RS_FUZZ_SEED").unwrap_or(0x5eed_1234_abcd_0001) | 1;
    let mut rng = Rng(seed);
    let report_every = (iters / 20).max(1);
    let start = Instant::now();
    for i in 0..iters {
        // Alternate whole-file walks and individual parsers.
        let (target, mut data, name) = if i % 2 == 0 {
            let which = (i as usize / 2) % corpus.len();
            let name = corpus[which].0.trim_end_matches(".h5").to_owned();
            ("walk", bytes[which].clone(), name)
        } else {
            let k = rng.below(seeds.len());
            (
                seeds[k].target(),
                seeds[k].build(&bytes),
                format!("seed{k}"),
            )
        };
        let max_mutations = if rng.below(4) == 0 { 16 } else { 4 };
        let n = 1 + rng.below(max_mutations);
        for _ in 0..n {
            mutate(&mut rng, &mut data, &bytes);
        }
        let tag = format!("{name}-{seed:x}-{i}");
        run_one(target, &data, &tag, &cur, epoch, &mut fails);
        if iters >= 10_000 && (i + 1) % report_every == 0 {
            eprintln!(
                "stress: {}/{iters} inputs, {:?}, {} failures",
                i + 1,
                start.elapsed(),
                fails.0.len()
            );
        }
        if fails.0.len() >= 50 {
            break;
        }
    }
    assert!(
        fails.0.is_empty(),
        "{} failing inputs:\n{}",
        fails.0.len(),
        fails.0.join("\n")
    );
}

// ------------------------------------------------------ parser seeds --

/// A starting input for one parser harness.
enum Seed {
    /// A structure inside fixture `fixture` at `addr`, after `params`.
    At {
        target: &'static str,
        params: Vec<u8>,
        addr: u32,
        fixture: usize,
        /// Bytes of user block to drop from the fixture.
        skip: usize,
    },
    /// A self-contained input.
    Bytes { target: &'static str, data: Vec<u8> },
}

impl Seed {
    fn target(&self) -> &'static str {
        match self {
            Seed::At { target, .. } | Seed::Bytes { target, .. } => target,
        }
    }

    fn build(&self, corpus: &[Vec<u8>]) -> Vec<u8> {
        match self {
            Seed::At {
                params,
                addr,
                fixture,
                skip,
                ..
            } => {
                let image = corpus[*fixture].get(*skip..).unwrap_or(&[]);
                parsers::with_address(params, *addr, image)
            }
            Seed::Bytes { data, .. } => data.clone(),
        }
    }
}

/// Parameter byte selecting a file's offset/length widths (see
/// `parsers::sizes`).
fn sizes_param(s: Sizes) -> u8 {
    let w = |x: u8| match x {
        4 => 1,
        2 => 2,
        _ => 0,
    };
    w(s.offset) | (w(s.length) << 2)
}

/// Every object reachable from the root through hard links.
fn all_objects(file: &File) -> Vec<Object> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut todo = Vec::new();
    if let Ok(root) = file.root() {
        seen.insert(root.address());
        todo.push(Object::Group(root));
    }
    while let Some(o) = todo.pop() {
        if let Some(g) = o.as_group() {
            for e in g.members().unwrap_or_default() {
                if let Link::Hard { address } = e.link
                    && seen.insert(address)
                    && let Ok(child) = file.object_at(address, &e.name)
                {
                    todo.push(child);
                }
            }
        }
        out.push(o);
    }
    out
}

/// Seeds for the parser harnesses, extracted from the fixtures.
fn parser_seeds(corpus: &[(String, Vec<u8>)]) -> Vec<Seed> {
    const PER_SIGNATURE: usize = 12;
    let registry = FilterRegistry::with_builtin();
    let filter_ids: Vec<u16> = registry.ids().collect();
    let mut seeds = Vec::new();
    for (fi, (_, data)) in corpus.iter().enumerate() {
        seeds.push(Seed::Bytes {
            target: "superblock",
            data: data.clone(),
        });
        let Ok(file) = File::from_bytes(data.clone()) else {
            continue;
        };
        let p = sizes_param(file.sizes());
        let base = file.superblock().base();
        // Structures found by signature (addresses relative to the base).
        for &(target, sig, nparams) in parsers::SIGNATURES {
            let hits = data
                .windows(sig.len())
                .enumerate()
                .filter(|(_, w)| *w == sig)
                .map(|(i, _)| i as u64)
                .filter_map(|i| i.checked_sub(base))
                .take(PER_SIGNATURE);
            for off in hits {
                let Ok(addr) = u32::try_from(off) else {
                    continue;
                };
                let mut variants = vec![vec![p], vec![p | 0x80]];
                if nparams == 2 {
                    variants = vec![vec![p, 0x00], vec![p, 0x20], vec![p | 0x60, 0x40]];
                }
                for params in variants {
                    seeds.push(Seed::At {
                        target,
                        params,
                        addr,
                        fixture: fi,
                        skip: base as usize,
                    });
                }
            }
        }
        for obj in all_objects(&file) {
            // Header messages.
            for m in obj.header().messages.iter().take(64) {
                let mut v = vec![p, m.msg_type as u8];
                v.extend_from_slice(&m.data);
                seeds.push(Seed::Bytes {
                    target: "message",
                    data: v,
                });
            }
            // Datatypes with sample elements.
            let mut typed: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            for a in obj.attributes().unwrap_or_default() {
                let raw = a.raw_value();
                typed.push((
                    encode_datatype(a.datatype()),
                    raw[..raw.len().min(255)].to_vec(),
                ));
            }
            if let Some(d) = obj.as_dataset() {
                let raw = d.read_raw().unwrap_or_default();
                typed.push((
                    encode_datatype(d.datatype()),
                    raw[..raw.len().min(255)].to_vec(),
                ));
                filter_seeds(d, &registry, &filter_ids, &mut seeds);
            }
            for (desc, payload) in typed {
                if desc.is_empty() {
                    continue;
                }
                let mut v = vec![payload.len() as u8];
                v.extend(desc);
                v.extend(payload);
                seeds.push(Seed::Bytes {
                    target: "datatype",
                    data: v,
                });
            }
        }
    }
    seeds
}

/// The datatype message of `dt` (empty when it cannot be encoded).
fn encode_datatype(dt: &hdf5rs::Datatype) -> Vec<u8> {
    dt.encode().unwrap_or_default()
}

/// Inputs for every filter of a chunked dataset's pipeline, obtained by
/// undoing the pipeline step by step on a few stored chunks.
fn filter_seeds(d: &Dataset, registry: &FilterRegistry, ids: &[u16], seeds: &mut Vec<Seed>) {
    let Some(pipeline) = d.filters() else { return };
    let Ok(chunks) = d.chunks() else { return };
    let store = d.file().store();
    let esize = d.datatype().size as usize;
    for c in chunks.iter().take(3) {
        let Ok(len) = usize::try_from(c.size) else {
            continue;
        };
        let Ok(mut data) = store.read_vec_at(c.address, len) else {
            continue;
        };
        for (i, f) in pipeline.filters.iter().enumerate().rev() {
            if c.filter_mask & (1 << i) != 0 {
                continue;
            }
            let Some(sel) = ids.iter().position(|&id| id == f.id) else {
                break;
            };
            let mut v = vec![sel as u8, f.client_data.len().min(23) as u8];
            for cd in f.client_data.iter().take(23) {
                v.extend(cd.to_le_bytes());
            }
            v.push(esize.min(255) as u8);
            v.extend_from_slice(&data[..data.len().min(64 << 10)]);
            seeds.push(Seed::Bytes {
                target: "filters",
                data: v,
            });
            let Some(filter) = registry.get(f.id) else {
                break;
            };
            let ctx = FilterContext::new(esize).with_max_output(64 << 20);
            match filter.decode(&f.client_data, data, &ctx) {
                Ok(next) => data = next,
                Err(_) => break,
            }
        }
    }
}

/// Write the seeds as a cargo-fuzz corpus: `dir/<target>/<n>`.
fn write_corpus(dir: &Path, corpus: &[(String, Vec<u8>)], seeds: &[Seed]) {
    let bytes: Vec<Vec<u8>> = corpus.iter().map(|(_, b)| b.clone()).collect();
    let walk = dir.join("open_walk");
    std::fs::create_dir_all(&walk).unwrap();
    for (name, data) in corpus {
        std::fs::write(walk.join(name), data).unwrap();
    }
    for (i, s) in seeds.iter().enumerate() {
        let d = dir.join(s.target());
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(format!("seed-{i:05}")), s.build(&bytes)).unwrap();
    }
}
