//! The shuffle filter (id 2).
//!
//! Shuffling regroups the bytes of fixed-size elements by significance: all
//! first bytes of every element, then all second bytes, and so on. This
//! makes the data more compressible for a following compressor. For `n`
//! whole elements of size `s`, shuffled byte `b * n + i` is byte `b` of
//! element `i`. Trailing bytes that do not fill a whole element are copied
//! unchanged to the end.
//!
//! Client data: `[element_size]`; when absent the context's element size is
//! used. Element sizes 0 and 1 make the filter a no-op.

use super::{Filter, FilterContext, SHUFFLE};
use crate::Result;

/// The shuffle filter.
#[derive(Debug, Clone, Copy, Default)]
pub struct Shuffle;

fn element_size(client_data: &[u32], ctx: &FilterContext) -> usize {
    client_data.first().map_or(ctx.element_size, |&s| {
        usize::try_from(s).unwrap_or(usize::MAX)
    })
}

impl Filter for Shuffle {
    fn id(&self) -> u16 {
        SHUFFLE
    }

    fn name(&self) -> &str {
        "shuffle"
    }

    fn decode(&self, client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        Ok(unshuffle(input, element_size(client_data, ctx)))
    }

    fn encode(&self, client_data: &[u32], input: Vec<u8>, ctx: &FilterContext) -> Result<Vec<u8>> {
        Ok(shuffle(input, element_size(client_data, ctx)))
    }
}

/// Shuffles `data` for elements of `elem_size` bytes.
pub fn shuffle(data: Vec<u8>, elem_size: usize) -> Vec<u8> {
    if elem_size <= 1 || data.len() / 2 < elem_size {
        // A single element (or none) shuffles to itself.
        return data;
    }
    let n = data.len() / elem_size;
    let body = n * elem_size;
    let mut out = vec![0u8; data.len()];
    let (out_body, out_tail) = out.split_at_mut(body);
    for (b, plane) in out_body.chunks_exact_mut(n).enumerate() {
        for (o, elem) in plane.iter_mut().zip(data.chunks_exact(elem_size)) {
            *o = elem[b];
        }
    }
    out_tail.copy_from_slice(&data[body..]);
    out
}

/// Reverses [`shuffle`] for elements of `elem_size` bytes.
pub fn unshuffle(data: Vec<u8>, elem_size: usize) -> Vec<u8> {
    if elem_size <= 1 || data.len() / 2 < elem_size {
        return data;
    }
    let n = data.len() / elem_size;
    let body = n * elem_size;
    let mut out = vec![0u8; data.len()];
    let (out_body, out_tail) = out.split_at_mut(body);
    for (b, plane) in data[..body].chunks_exact(n).enumerate() {
        for (elem, &v) in out_body.chunks_exact_mut(elem_size).zip(plane) {
            elem[b] = v;
        }
    }
    out_tail.copy_from_slice(&data[body..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        let data: Vec<u8> = (0u8..10).collect();
        // 3 elements of 3 bytes, 1 trailing byte.
        let s = shuffle(data.clone(), 3);
        assert_eq!(s, [0, 3, 6, 1, 4, 7, 2, 5, 8, 9]);
        assert_eq!(unshuffle(s, 3), data);
        // u16 little-endian values 0x0100, 0x0302.
        assert_eq!(shuffle(vec![0, 1, 2, 3], 2), [0, 2, 1, 3]);
    }

    #[test]
    fn roundtrip_many_sizes() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + i / 13) as u8).collect();
        for es in 0..=20 {
            for len in [0, 1, 2, 5, 17, 64, 999, 1000] {
                let d = data[..len].to_vec();
                let s = shuffle(d.clone(), es);
                assert_eq!(s.len(), d.len());
                assert_eq!(unshuffle(s, es), d, "es={es} len={len}");
            }
        }
    }

    #[test]
    fn filter_uses_client_data_then_ctx() {
        let ctx = FilterContext::new(2);
        let d = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(
            Shuffle.encode(&[], d.clone(), &ctx).unwrap(),
            [0, 2, 4, 1, 3, 5]
        );
        assert_eq!(
            Shuffle.encode(&[3], d.clone(), &ctx).unwrap(),
            [0, 3, 1, 4, 2, 5]
        );
        assert_eq!(Shuffle.decode(&[u32::MAX], d.clone(), &ctx).unwrap(), d);
        let enc = Shuffle.encode(&[3], d.clone(), &ctx).unwrap();
        assert_eq!(Shuffle.decode(&[3], enc, &ctx).unwrap(), d);
    }
}
