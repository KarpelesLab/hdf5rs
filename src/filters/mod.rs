//! Filter pipeline framework and filter codecs.
//!
//! HDF5 stores chunked dataset data through a *filter pipeline* (spec:
//! "Data Storage - Filter Pipeline" message, type 0x000B). On write each
//! filter of the pipeline is applied in order; on read the filters are
//! undone in reverse order. Every chunk carries a 32-bit *filter mask*
//! in its index entry: bit `i` set means filter `i` of the pipeline was
//! **not** applied to that chunk (typically because it was marked optional
//! and failed, e.g. a compressor that could not shrink the data).
//!
//! This module provides:
//!
//! - the [`Filter`] trait implemented by every codec,
//! - a [`FilterRegistry`] mapping filter ids to implementations,
//! - [`decode_pipeline`] / [`encode_pipeline`] which run a pipeline
//!   description ([`PipelineStep`]s) over one chunk,
//! - built-in codecs: `deflate` (feature `deflate`), [`shuffle`],
//!   [`fletcher32`], [`lzf`], [`nbit`], [`scaleoffset`],
//! - optional codecs backed by `compcol`, each behind its own Cargo feature
//!   (all enabled by `all-filters`): `zstd`, `lz4`, `bzip2`, `blosc` and
//!   `bitshuffle`.

use std::collections::BTreeMap;
use std::fmt;

use crate::{Error, Result};

mod bits;
#[cfg(any(feature = "bitshuffle", feature = "blosc"))]
pub mod bitshuffle;
#[cfg(feature = "blosc")]
pub mod blosc;
#[cfg(feature = "bzip2")]
pub mod bzip2;
#[cfg(any(
    feature = "zstd",
    feature = "lz4",
    feature = "bzip2",
    feature = "blosc",
    feature = "bitshuffle"
))]
mod codec;
#[cfg(any(feature = "deflate", feature = "blosc"))]
pub mod deflate;
pub mod fletcher32;
#[cfg(feature = "lz4")]
pub mod lz4;
pub mod lzf;
pub mod nbit;
pub mod scaleoffset;
pub mod shuffle;
#[cfg(feature = "zstd")]
pub mod zstd;

/// Filter id: gzip / zlib deflate compression (spec-defined).
pub const DEFLATE: u16 = 1;
/// Filter id: byte shuffle (spec-defined).
pub const SHUFFLE: u16 = 2;
/// Filter id: Fletcher-32 checksum (spec-defined).
pub const FLETCHER32: u16 = 3;
/// Filter id: SZIP compression (spec-defined; not built in).
pub const SZIP: u16 = 4;
/// Filter id: N-bit packing (spec-defined).
pub const NBIT: u16 = 5;
/// Filter id: scale + offset (spec-defined).
pub const SCALEOFFSET: u16 = 6;
/// Filter id: BZIP2 compression (registered third-party filter).
pub const BZIP2: u16 = 307;
/// Filter id: LZF compression as used by h5py (registered third-party filter).
pub const LZF: u16 = 32000;
/// Filter id: Blosc meta-compressor (registered third-party filter).
pub const BLOSC: u16 = 32001;
/// Filter id: MAFISC (registered third-party filter).
pub const MAFISC: u16 = 32002;
/// Filter id: Snappy (registered third-party filter).
pub const SNAPPY: u16 = 32003;
/// Filter id: LZ4 (registered third-party filter).
pub const LZ4: u16 = 32004;
/// Filter id: APAX (registered third-party filter).
pub const APAX: u16 = 32005;
/// Filter id: CBF (registered third-party filter).
pub const CBF: u16 = 32006;
/// Filter id: JPEG-XR (registered third-party filter).
pub const JPEG_XR: u16 = 32007;
/// Filter id: Bitshuffle (registered third-party filter).
pub const BITSHUFFLE: u16 = 32008;
/// Filter id: SPDP (registered third-party filter).
pub const SPDP: u16 = 32009;
/// Filter id: LPC-Rice (registered third-party filter).
pub const LPC_RICE: u16 = 32010;
/// Filter id: CCSDS-123 (registered third-party filter).
pub const CCSDS_123: u16 = 32011;
/// Filter id: JPEG-LS (registered third-party filter).
pub const JPEG_LS: u16 = 32012;
/// Filter id: ZFP (registered third-party filter).
pub const ZFP: u16 = 32013;
/// Filter id: FPZip (registered third-party filter).
pub const FPZIP: u16 = 32014;
/// Filter id: Zstandard (registered third-party filter).
pub const ZSTD: u16 = 32015;
/// Filter id: B³D (registered third-party filter).
pub const B3D: u16 = 32016;
/// Filter id: SZ (registered third-party filter).
pub const SZ: u16 = 32017;
/// Filter id: FCIDECOMP (registered third-party filter).
pub const FCIDECOMP: u16 = 32018;
/// Filter id: JPEG (registered third-party filter).
pub const JPEG: u16 = 32019;
/// Filter id: VBZ (registered third-party filter).
pub const VBZ: u16 = 32020;
/// Filter id: Blosc2 (registered third-party filter).
pub const BLOSC2: u16 = 32026;

/// Maximum number of filters in one pipeline (spec: the filter pipeline
/// message allows at most 32 filters, matching the 32-bit chunk filter mask).
pub const MAX_PIPELINE_LEN: usize = 32;

/// Default cap on the output of a single filter (1 GiB).
pub const DEFAULT_MAX_OUTPUT: usize = 1 << 30;

/// Per-chunk information passed to filters.
///
/// Construct with [`FilterContext::new`] and refine with the `with_*`
/// methods; new fields may be added in future releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct FilterContext {
    /// Size in bytes of one dataset element (used e.g. by shuffle when the
    /// filter has no client data).
    pub element_size: usize,
    /// Upper bound on the size of any filter's output. Decompressors must
    /// fail with [`Error::LimitExceeded`] rather than produce more
    /// (decompression-bomb protection).
    pub max_output: usize,
    /// Expected size of the fully decoded chunk, if known. Only a hint for
    /// buffer pre-allocation; never trusted for correctness.
    pub expected_size: Option<usize>,
}

impl FilterContext {
    /// Context for elements of `element_size` bytes, with
    /// [`DEFAULT_MAX_OUTPUT`] and no size hint.
    pub fn new(element_size: usize) -> Self {
        FilterContext {
            element_size,
            max_output: DEFAULT_MAX_OUTPUT,
            expected_size: None,
        }
    }

    /// Sets [`FilterContext::max_output`].
    pub fn with_max_output(mut self, max_output: usize) -> Self {
        self.max_output = max_output;
        self
    }

    /// Sets [`FilterContext::expected_size`].
    pub fn with_expected_size(mut self, expected_size: Option<usize>) -> Self {
        self.expected_size = expected_size;
        self
    }

    /// A pre-allocation size derived from `hint` (and the expected size),
    /// clamped to [`FilterContext::max_output`] and a modest absolute bound
    /// so untrusted hints cannot force huge allocations up front.
    pub(crate) fn capacity_hint(&self, hint: Option<usize>) -> usize {
        const MAX_PREALLOC: usize = 64 << 20;
        hint.or(self.expected_size)
            .unwrap_or(0)
            .min(self.max_output)
            .min(MAX_PREALLOC)
    }
}

impl Default for FilterContext {
    fn default() -> Self {
        FilterContext::new(1)
    }
}

/// A filter codec.
///
/// `client_data` is the filter's client data array from the filter
/// pipeline message. Both directions take ownership of the input so that
/// pass-through filters can return it without copying.
pub trait Filter: Send + Sync {
    /// The filter identification value.
    fn id(&self) -> u16;
    /// A human-readable name.
    fn name(&self) -> &str;
    /// Undoes the filter (read direction).
    fn decode(&self, client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>>;
    /// Applies the filter (write direction).
    fn encode(&self, client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>>;
}

/// The deflate (gzip/zlib) filter, id [`DEFLATE`] (requires the `deflate`
/// feature).
///
/// Client data: `[level]` (0–9, default 6) for encoding; ignored when
/// decoding. Chunks hold a zlib stream (RFC 1950). The codec itself lives in
/// [`deflate`]; this adapter is kept here so the codec module only needs to
/// provide `zlib_decompress` and `zlib_compress`.
#[cfg(feature = "deflate")]
#[derive(Debug, Clone, Copy, Default)]
pub struct Deflate;

#[cfg(feature = "deflate")]
impl Filter for Deflate {
    fn id(&self) -> u16 {
        DEFLATE
    }

    fn name(&self) -> &str {
        "deflate"
    }

    fn decode(&self, _client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        deflate::zlib_decompress(&input, ctx.capacity_hint(None), ctx.max_output)
    }

    fn encode(&self, client_data: &[u32], input: Vec<u8>, _ctx: &FilterContext) -> Result<Vec<u8>> {
        let level = client_data.first().copied().unwrap_or(6).min(9);
        deflate::zlib_compress(&input, level)
    }
}

/// A set of filter implementations, keyed by filter id.
#[derive(Default)]
pub struct FilterRegistry {
    filters: BTreeMap<u16, Box<dyn Filter>>,
}

impl FilterRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        FilterRegistry::default()
    }

    /// A registry with the built-in filters: deflate (feature `deflate`),
    /// shuffle, fletcher32, N-bit, scale-offset and lzf, plus the optional
    /// filters whose features are enabled: zstd, lz4, bzip2, blosc and
    /// bitshuffle.
    pub fn with_builtin() -> Self {
        let mut r = FilterRegistry::new();
        #[cfg(feature = "deflate")]
        r.register(Box::new(Deflate));
        r.register(Box::new(shuffle::Shuffle));
        r.register(Box::new(fletcher32::Fletcher32));
        r.register(Box::new(lzf::Lzf));
        r.register(Box::new(nbit::NBit));
        r.register(Box::new(scaleoffset::ScaleOffset));
        #[cfg(feature = "zstd")]
        r.register(Box::new(zstd::Zstd));
        #[cfg(feature = "lz4")]
        r.register(Box::new(lz4::Lz4));
        #[cfg(feature = "bzip2")]
        r.register(Box::new(bzip2::Bzip2));
        #[cfg(feature = "blosc")]
        r.register(Box::new(blosc::Blosc));
        #[cfg(feature = "bitshuffle")]
        r.register(Box::new(bitshuffle::Bitshuffle));
        r
    }

    /// Adds `filter`, replacing any filter previously registered with the
    /// same id.
    pub fn register(&mut self, filter: Box<dyn Filter>) {
        self.filters.insert(filter.id(), filter);
    }

    /// The filter registered for `id`, if any.
    pub fn get(&self, id: u16) -> Option<&dyn Filter> {
        self.filters.get(&id).map(|f| f.as_ref())
    }

    /// Ids of all registered filters, ascending.
    pub fn ids(&self) -> impl Iterator<Item = u16> + '_ {
        self.filters.keys().copied()
    }
}

impl fmt::Debug for FilterRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.filters.iter().map(|(id, flt)| (id, flt.name())))
            .finish()
    }
}

/// One filter of a pipeline, as described by the filter pipeline message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineStep<'a> {
    /// Filter identification value.
    pub id: u16,
    /// The "optional" flag: if the filter fails while writing, the chunk is
    /// stored without it and its filter-mask bit is set.
    pub optional: bool,
    /// The filter's client data values.
    pub client_data: &'a [u32],
}

fn check_len(steps: &[PipelineStep<'_>]) -> Result<()> {
    if steps.len() > MAX_PIPELINE_LEN {
        return Err(Error::invalid("filter pipeline has more than 32 filters"));
    }
    Ok(())
}

fn check_output(out: &[u8], ctx: &FilterContext) -> Result<()> {
    if out.len() > ctx.max_output {
        return Err(Error::LimitExceeded("filter output size"));
    }
    Ok(())
}

/// Undoes a filter pipeline on one chunk (read direction).
///
/// Filters are applied in reverse pipeline order; filter `i` is skipped
/// when bit `i` of `filter_mask` is set. A filter that is not skipped but
/// is missing from `registry` is an error even when marked optional: the
/// mask says it was applied, so the data cannot be decoded without it.
pub fn decode_pipeline(
    registry: &FilterRegistry,
    steps: &[PipelineStep<'_>],
    filter_mask: u32,
    mut data: Vec<u8>,
    ctx: &FilterContext,
) -> Result<Vec<u8>> {
    check_len(steps)?;
    for (i, step) in steps.iter().enumerate().rev() {
        if filter_mask & (1u32 << i) != 0 {
            continue;
        }
        let filter = registry
            .get(step.id)
            .ok_or_else(|| Error::unsupported(format!("filter id {} is not available", step.id)))?;
        data = filter.decode(step.client_data, data, ctx)?;
        check_output(&data, ctx)?;
    }
    Ok(data)
}

/// Applies a filter pipeline to one chunk (write direction).
///
/// Filters run in pipeline order. Returns the filtered data and the chunk
/// filter mask to store alongside it. An optional filter that fails or is
/// not registered is skipped and its mask bit set; a failing or missing
/// mandatory filter is an error.
pub fn encode_pipeline(
    registry: &FilterRegistry,
    steps: &[PipelineStep<'_>],
    mut data: Vec<u8>,
    ctx: &FilterContext,
) -> Result<(Vec<u8>, u32)> {
    check_len(steps)?;
    let mut mask = 0u32;
    for (i, step) in steps.iter().enumerate() {
        let bit = 1u32 << i;
        let Some(filter) = registry.get(step.id) else {
            if step.optional {
                mask |= bit;
                continue;
            }
            return Err(Error::unsupported(format!(
                "filter id {} is not available",
                step.id
            )));
        };
        if step.optional {
            // Keep the input in case the filter fails.
            match filter.encode(step.client_data, data.clone(), ctx) {
                Ok(out) if out.len() <= ctx.max_output => data = out,
                _ => mask |= bit,
            }
        } else {
            data = filter.encode(step.client_data, data, ctx)?;
            check_output(&data, ctx)?;
        }
    }
    Ok((data, mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test filter: XORs every byte with a key; encode fails on demand.
    struct Xor {
        id: u16,
        key: u8,
        fail_encode: bool,
    }

    impl Filter for Xor {
        fn id(&self) -> u16 {
            self.id
        }
        fn name(&self) -> &str {
            "xor"
        }
        fn decode(&self, _: &[u32], mut input: Vec<u8>, _: &FilterContext) -> Result<Vec<u8>> {
            input.iter_mut().for_each(|b| *b ^= self.key);
            Ok(input)
        }
        fn encode(&self, cd: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
            if self.fail_encode {
                return Err(Error::invalid("xor refuses"));
            }
            self.decode(cd, input, ctx)
        }
    }

    /// Test filter: appends its id byte on encode, checks & strips on decode.
    /// Makes ordering visible.
    struct Tag(u16);

    impl Filter for Tag {
        fn id(&self) -> u16 {
            self.0
        }
        fn name(&self) -> &str {
            "tag"
        }
        fn decode(&self, _: &[u32], mut input: Vec<u8>, _: &FilterContext) -> Result<Vec<u8>> {
            match input.pop() {
                Some(b) if b == self.0 as u8 => Ok(input),
                _ => Err(Error::invalid("tag mismatch")),
            }
        }
        fn encode(&self, _: &[u32], mut input: Vec<u8>, _: &FilterContext) -> Result<Vec<u8>> {
            input.push(self.0 as u8);
            Ok(input)
        }
    }

    fn step(id: u16, optional: bool) -> PipelineStep<'static> {
        PipelineStep {
            id,
            optional,
            client_data: &[],
        }
    }

    fn tag_registry() -> FilterRegistry {
        let mut r = FilterRegistry::new();
        for id in [300, 301, 302] {
            r.register(Box::new(Tag(id)));
        }
        r
    }

    #[test]
    fn registry_basics() {
        let r = FilterRegistry::with_builtin();
        let mut expected = vec![SHUFFLE, FLETCHER32, NBIT, SCALEOFFSET];
        if cfg!(feature = "deflate") {
            expected.insert(0, DEFLATE);
        }
        for (id, on) in [
            (BZIP2, cfg!(feature = "bzip2")),
            (LZF, true),
            (BLOSC, cfg!(feature = "blosc")),
            (LZ4, cfg!(feature = "lz4")),
            (BITSHUFFLE, cfg!(feature = "bitshuffle")),
            (ZSTD, cfg!(feature = "zstd")),
        ] {
            if on {
                expected.push(id);
            }
        }
        assert_eq!(r.ids().collect::<Vec<_>>(), expected);
        assert_eq!(r.get(SHUFFLE).unwrap().name(), "shuffle");
        assert!(r.get(SZIP).is_none());
        assert!(FilterRegistry::new().get(DEFLATE).is_none());
        let mut r = FilterRegistry::new();
        r.register(Box::new(Tag(5)));
        r.register(Box::new(Xor {
            id: 5,
            key: 1,
            fail_encode: false,
        }));
        assert_eq!(r.get(5).unwrap().name(), "xor");
        assert!(format!("{r:?}").contains("xor"));
    }

    #[test]
    fn pipeline_order() {
        let r = tag_registry();
        let steps = [step(300, false), step(301, false), step(302, false)];
        let ctx = FilterContext::default();
        let (enc, mask) = encode_pipeline(&r, &steps, vec![9], &ctx).unwrap();
        assert_eq!(mask, 0);
        assert_eq!(enc, [9, 44, 45, 46]);
        assert_eq!(decode_pipeline(&r, &steps, 0, enc, &ctx).unwrap(), [9]);
        // Wrong order would hit a tag mismatch.
        let rev = [step(302, false), step(301, false), step(300, false)];
        assert!(decode_pipeline(&r, &rev, 0, vec![9, 44, 45, 46], &ctx).is_err());
    }

    #[test]
    fn mask_skips_filters_on_decode() {
        let r = tag_registry();
        let steps = [step(300, false), step(301, false), step(302, false)];
        let ctx = FilterContext::default();
        // Filter 1 was not applied.
        assert_eq!(
            decode_pipeline(&r, &steps, 0b010, vec![9, 44, 46], &ctx).unwrap(),
            [9]
        );
        // All skipped: pass-through, even for unknown filters.
        let unknown = [step(999, false)];
        assert_eq!(
            decode_pipeline(&r, &unknown, 1, vec![1, 2], &ctx).unwrap(),
            [1, 2]
        );
    }

    #[test]
    fn unknown_filters() {
        let r = tag_registry();
        let ctx = FilterContext::default();
        let mandatory = [step(300, false), step(999, false)];
        assert!(matches!(
            decode_pipeline(&r, &mandatory, 0, vec![1], &ctx),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            encode_pipeline(&r, &mandatory, vec![1], &ctx),
            Err(Error::Unsupported(_))
        ));
        // Unknown optional: skipped on write (mask bit set) ...
        let optional = [step(300, false), step(999, true), step(301, false)];
        let (enc, mask) = encode_pipeline(&r, &optional, vec![1], &ctx).unwrap();
        assert_eq!((enc.as_slice(), mask), (&[1, 44, 45][..], 0b010));
        assert_eq!(
            decode_pipeline(&r, &optional, mask, enc, &ctx).unwrap(),
            [1]
        );
        // ... but required on read if the mask says it was applied.
        assert!(decode_pipeline(&r, &optional, 0, vec![1, 44, 45], &ctx).is_err());
    }

    #[test]
    fn optional_failure_preserves_data() {
        let mut r = tag_registry();
        r.register(Box::new(Xor {
            id: 400,
            key: 0xff,
            fail_encode: true,
        }));
        let ctx = FilterContext::default();
        let steps = [step(300, false), step(400, true), step(301, false)];
        let (enc, mask) = encode_pipeline(&r, &steps, vec![7, 8], &ctx).unwrap();
        assert_eq!(mask, 0b010);
        assert_eq!(enc, [7, 8, 44, 45]);
        assert_eq!(
            decode_pipeline(&r, &steps, mask, enc, &ctx).unwrap(),
            [7, 8]
        );
        // Mandatory failure is an error.
        let steps = [step(400, false)];
        assert!(encode_pipeline(&r, &steps, vec![1], &ctx).is_err());
    }

    #[test]
    fn too_many_filters() {
        let r = tag_registry();
        let steps = vec![step(300, true); 33];
        let ctx = FilterContext::default();
        assert!(decode_pipeline(&r, &steps, u32::MAX, vec![], &ctx).is_err());
        assert!(encode_pipeline(&r, &steps, vec![], &ctx).is_err());
        // Exactly 32 is fine.
        let steps = vec![step(999, true); 32];
        let (_, mask) = encode_pipeline(&r, &steps, vec![], &ctx).unwrap();
        assert_eq!(mask, u32::MAX);
    }

    #[test]
    fn output_limit_enforced() {
        let r = tag_registry();
        let ctx = FilterContext::default().with_max_output(2);
        let steps = [step(300, false)];
        assert!(matches!(
            encode_pipeline(&r, &steps, vec![1, 2], &ctx),
            Err(Error::LimitExceeded(_))
        ));
        let opt = [step(300, true)];
        let (out, mask) = encode_pipeline(&r, &opt, vec![1, 2], &ctx).unwrap();
        assert_eq!((out.as_slice(), mask), (&[1, 2][..], 1));
    }

    #[test]
    fn builtin_pipeline_roundtrip() {
        // shuffle -> fletcher32 -> lzf(optional), as h5py might configure.
        let r = FilterRegistry::with_builtin();
        let data: Vec<u8> = (0..4000u32).flat_map(|i| (i / 3).to_le_bytes()).collect();
        let ctx = FilterContext::new(4).with_expected_size(Some(data.len()));
        let cd = [0u32, 0, 0];
        let steps = [
            step(SHUFFLE, false),
            step(FLETCHER32, false),
            PipelineStep {
                id: LZF,
                optional: true,
                client_data: &cd,
            },
        ];
        let (enc, mask) = encode_pipeline(&r, &steps, data.clone(), &ctx).unwrap();
        assert_eq!(mask, 0);
        assert!(enc.len() < data.len() / 4);
        assert_eq!(decode_pipeline(&r, &steps, mask, enc, &ctx).unwrap(), data);
    }

    #[test]
    fn mutated_streams_never_panic() {
        // Valid encodings from every built-in filter, then corrupted.
        let r = FilterRegistry::with_builtin();
        let data: Vec<u8> = (0..4000u32).flat_map(|i| (i / 5).to_le_bytes()).collect();
        let n = (data.len() / 4) as u32;
        let cds: Vec<(u16, Vec<u32>)> = vec![
            (DEFLATE, vec![6]),
            (SHUFFLE, vec![4]),
            (FLETCHER32, vec![]),
            (NBIT, vec![8, 0, n, 1, 4, 0, 20, 0]),
            (SCALEOFFSET, {
                let mut cd = vec![2, 0, n, 0, 4, 1, 0, 1];
                cd.resize(20, 0);
                cd
            }),
            (SCALEOFFSET, {
                let mut cd = vec![0, 2, n / 2, 1, 8, 0, 0, 1];
                cd.resize(20, 0);
                cd
            }),
            (LZF, vec![]),
            (ZSTD, vec![3]),
            (LZ4, vec![1000]),
            (BZIP2, vec![9]),
            (BLOSC, vec![2, 2, 4, 0, 5, 1, 1]),
            (BLOSC, vec![2, 2, 4, 0, 5, 2, 5]),
            (BITSHUFFLE, vec![0, 4, 4, 0, 2]),
            (BITSHUFFLE, vec![0, 4, 4, 64, 3, 3]),
        ];
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let ctx = FilterContext::new(4).with_max_output(1 << 20);
        for (id, cd) in &cds {
            let Some(f) = r.get(*id) else { continue };
            let enc = f.encode(cd, data.clone(), &ctx).unwrap();
            assert!(f.decode(cd, enc.clone(), &ctx).is_ok(), "{id}");
            for _ in 0..300 {
                let mut m = enc.clone();
                match next() % 3 {
                    0 => m.truncate((next() as usize) % (m.len() + 1)),
                    1 => {
                        for _ in 0..1 + next() % 4 {
                            let i = (next() as usize) % m.len().max(1);
                            if let Some(b) = m.get_mut(i) {
                                *b ^= 1 << (next() % 8);
                            }
                        }
                    }
                    _ => {
                        let i = (next() as usize) % m.len().max(1);
                        if let Some(b) = m.get_mut(i) {
                            *b = next() as u8;
                        }
                    }
                }
                let _ = f.decode(cd, m, &ctx);
            }
        }
    }

    #[test]
    fn builtin_decoders_never_panic() {
        let r = FilterRegistry::with_builtin();
        let mut x = 0x0123_4567_89ab_cdefu64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..5_000 {
            let len = (next() % 200) as usize;
            let data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let cd: Vec<u32> = (0..next() % 4).map(|_| next() as u32).collect();
            let ctx = FilterContext::new((next() % 17) as usize).with_max_output(1000);
            for id in r.ids() {
                let _ = r.get(id).unwrap().decode(&cd, data.clone(), &ctx);
            }
        }
    }
}
