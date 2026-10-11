//! All the vectors of one `(model_id, dim)` embedding space, in one contiguous slab.
//!
//! A cosine is only meaningful between two vectors the same model produced, and the noise cutoff's
//! background is a property of one such population, so the space — not the engine — owns the model
//! id and the dimension: nothing here can score a vector against one from another model. The engine
//! keeps one [`Space`] per embedding model and decides what goes in and when it is saved.
//!
//! Vectors are L2-normalised on the way in, so a cosine is a plain dot product, and they sit
//! back to back in a single `Vec<f32>` so a brute-force search is one linear, cache-friendly pass
//! (100k vectors of 768 numbers is 300 MB of reads: tens of milliseconds, no index to build or keep
//! consistent). Removal swaps the last vector into the hole so the slab never has gaps.
//!
//! Deterministic: equal scores order by key, and [`Space::seed`] depends on which keys are present,
//! not on the order they arrived in.

use std::collections::HashMap;

use crate::cutoff::{self, Background};
use crate::thresholds::BACKGROUND_REFRESH_FRACTION;

/// The largest dimension a space accepts. Real embedding models sit between 256 and 4096; this
/// bounds what a hostile sidecar can make the slab allocate per vector.
const MAX_DIM: usize = 8192;

/// One embedding space: every vector of one model at one dimension.
#[derive(Clone, Debug)]
pub struct Space {
    model_id: String,
    dim: usize,
    /// The key of each vector, in slab order.
    keys: Vec<u64>,
    /// The vectors, L2-normalised, back to back; `keys.len() * dim` numbers.
    data: Vec<f32>,
    /// Key -> position in `keys` (the vector is at `position * dim`).
    index: HashMap<u64, usize>,
}

impl Space {
    /// A space for `model_id` with `dim` numbers per vector; `None` when `dim` is 0 or over 8192
    /// or `model_id` is empty.
    pub fn new(model_id: &str, dim: usize) -> Option<Space> {
        if dim == 0 || dim > MAX_DIM || model_id.is_empty() {
            return None;
        }
        Some(Space { model_id: model_id.to_owned(), dim, keys: Vec::new(), data: Vec::new(), index: HashMap::new() })
    }

    /// The model whose vectors this space holds.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Numbers per vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Vectors in the space.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Insert or replace `key`'s vector, L2-normalised on the way in. Returns `false` (and changes
    /// nothing) when the length is not `dim`, any value is not finite, or the norm is 0 (a zero
    /// vector has no direction, so it has no cosine with anything).
    pub fn insert(&mut self, key: u64, vector: &[f32]) -> bool {
        let Some(scale) = unit_scale(vector, self.dim) else {
            return false;
        };
        let normalised = vector.iter().map(|x| (f64::from(*x) * scale) as f32);
        if let Some(&slot) = self.index.get(&key) {
            let Some(start) = slot.checked_mul(self.dim) else {
                return false;
            };
            let Some(row) = start.checked_add(self.dim).and_then(|end| self.data.get_mut(start..end)) else {
                return false;
            };
            for (dst, src) in row.iter_mut().zip(normalised) {
                *dst = src;
            }
            return true;
        }
        self.index.insert(key, self.keys.len());
        self.keys.push(key);
        self.data.extend(normalised);
        true
    }

    /// Forget `key` (swap-remove: O(`dim`), the slab stays dense). Returns whether it was there.
    pub fn remove(&mut self, key: u64) -> bool {
        let Some(slot) = self.index.remove(&key) else {
            return false;
        };
        let dim = self.dim;
        let last = self.keys.len().saturating_sub(1);
        let (Some(start), Some(last_start)) = (slot.checked_mul(dim), last.checked_mul(dim)) else {
            return true;
        };
        if slot != last {
            // The last vector moves into the hole.
            if last_start <= self.data.len() {
                let (head, tail) = self.data.split_at_mut(last_start);
                if let (Some(dst), Some(src)) = (head.get_mut(start..start.saturating_add(dim)), tail.get(..dim)) {
                    dst.copy_from_slice(src);
                }
            }
        }
        self.data.truncate(last_start);
        if slot < self.keys.len() {
            self.keys.swap_remove(slot);
        }
        if slot != last
            && let Some(moved) = self.keys.get(slot)
        {
            self.index.insert(*moved, slot);
        }
        true
    }

    /// Whether `key` has a vector here.
    pub fn contains(&self, key: u64) -> bool {
        self.index.contains_key(&key)
    }

    /// `key`'s normalised vector.
    pub fn get(&self, key: u64) -> Option<&[f32]> {
        let start = self.index.get(&key)?.checked_mul(self.dim)?;
        self.data.get(start..start.checked_add(self.dim)?)
    }

    /// Every `(key, vector)`, in slab order (for the engine to write a sidecar).
    pub fn iter(&self) -> impl Iterator<Item = (u64, &[f32])> + '_ {
        self.keys.iter().copied().zip(self.data.chunks_exact(self.dim))
    }

    /// The `limit` nearest vectors to `query` by cosine (the query is normalised first; a zero,
    /// non-finite or wrong-length query gives nothing), most similar first, ties by key. `exclude`
    /// leaves one key out (the photo the query came from).
    pub fn nearest(&self, query: &[f32], limit: usize, exclude: Option<u64>) -> Vec<(u64, f32)> {
        if limit == 0 {
            return Vec::new();
        }
        match self.score_all(query, exclude) {
            Some(scored) => top(scored, limit),
            None => Vec::new(),
        }
    }

    /// A seed that names this space's contents: `cutoff::mix64` folded over the keys in sorted
    /// order (so insertion order does not matter), then mixed with the count and the dimension.
    pub fn seed(&self) -> u64 {
        let mut keys = self.keys.clone();
        keys.sort_unstable();
        let folded = keys.iter().fold(0_u64, |acc, key| cutoff::mix64(acc ^ key));
        let folded = cutoff::mix64(folded ^ self.keys.len() as u64);
        cutoff::mix64(folded ^ self.dim as u64)
    }

    /// The background statistics of this space ([`cutoff::background`] over its vectors, seeded
    /// by [`Space::seed`]); `None` under two vectors.
    pub fn background(&self) -> Option<Background> {
        let dim = self.dim;
        cutoff::background(self.len(), self.seed(), |i| {
            let start = i.checked_mul(dim)?;
            self.data.get(start..start.checked_add(dim)?).map(<[f32]>::to_vec)
        })
    }

    /// [`Space::nearest`], then the noise cutoff: scores under the ceiling for `bg` (the candidates
    /// being the vectors actually scored, i.e. `len` minus the excluded one) are dropped. Returns
    /// the kept list and the ceiling applied, if any. With `bg` `None` there is no cutoff.
    pub fn search(&self, query: &[f32], limit: usize, exclude: Option<u64>, bg: Option<&Background>) -> (Vec<(u64, f32)>, Option<f32>) {
        if limit == 0 {
            return (Vec::new(), None);
        }
        let Some(scored) = self.score_all(query, exclude) else {
            return (Vec::new(), None);
        };
        let candidates = scored.len();
        let mut hits = top(scored, limit);
        let ceiling = bg.and_then(|bg| cutoff::apply(&mut hits, bg, candidates));
        (hits, ceiling)
    }

    /// Whether `bg` is stale for this space: the space has grown or shrunk by more than
    /// [`BACKGROUND_REFRESH_FRACTION`] of the size it was sampled at **and** its seed has changed.
    /// Both, because one insert changes the seed (it names the contents) and that alone must not
    /// force a resample of 2000 pairs.
    pub fn background_is_stale(&self, bg: &Background) -> bool {
        let moved = self.len().abs_diff(bg.n_at_compute) as f64;
        let allowed = f64::from(BACKGROUND_REFRESH_FRACTION) * bg.n_at_compute as f64;
        bg.seed != self.seed() && moved > allowed
    }

    /// The cosine of `query` to every vector but `exclude`, in slab order. `None` for a query that
    /// cannot be scored (wrong length, non-finite, zero).
    fn score_all(&self, query: &[f32], exclude: Option<u64>) -> Option<Vec<(u64, f32)>> {
        let scale = unit_scale(query, self.dim)?;
        let q: Vec<f32> = query.iter().map(|x| (f64::from(*x) * scale) as f32).collect();
        let mut scored = Vec::with_capacity(self.keys.len());
        for (key, row) in self.keys.iter().zip(self.data.chunks_exact(self.dim)) {
            if exclude == Some(*key) {
                continue;
            }
            // Both sides are unit length, so the dot product is the cosine; rounding can leave it
            // a hair outside -1..1. Adding 0.0 turns -0.0 into 0.0 so ties compare equal.
            let cosine = dot(&q, row).clamp(-1.0, 1.0) + 0.0;
            if cosine.is_finite() {
                scored.push((*key, cosine));
            }
        }
        Some(scored)
    }
}

/// The factor that makes `vector` unit length, or `None` when it is the wrong length, has a
/// non-finite value, or has no length to speak of. Accumulated in f64 so large and small values
/// neither overflow nor underflow the sum of squares.
fn unit_scale(vector: &[f32], dim: usize) -> Option<f64> {
    if vector.len() != dim || vector.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let norm = vector.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    (norm.is_finite() && norm > 0.0).then(|| 1.0 / norm)
}

/// A dot product with eight independent accumulators and no per-element bounds checks. The
/// accumulators let the compiler keep the adds in vector registers (a single running sum must stay
/// in order, so it cannot); destructuring fixed-size chunks keeps the loop body straight-line even
/// at the dev profile's opt-level, where the auto-vectoriser is off.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let tail: f32 = a_chunks.remainder().iter().zip(b_chunks.remainder()).map(|(x, y)| x * y).sum();
    let mut acc = [0.0_f32; 8];
    for (ca, cb) in a_chunks.zip(b_chunks) {
        if let (Ok(&[a0, a1, a2, a3, a4, a5, a6, a7]), Ok(&[b0, b1, b2, b3, b4, b5, b6, b7])) = (<&[f32; 8]>::try_from(ca), <&[f32; 8]>::try_from(cb))
        {
            let [c0, c1, c2, c3, c4, c5, c6, c7] = &mut acc;
            *c0 += a0 * b0;
            *c1 += a1 * b1;
            *c2 += a2 * b2;
            *c3 += a3 * b3;
            *c4 += a4 * b4;
            *c5 += a5 * b5;
            *c6 += a6 * b6;
            *c7 += a7 * b7;
        }
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// The best `limit` of `scored`, most similar first, ties by key. Selects before sorting so a
/// large space pays O(n) plus a sort of `limit`, not a sort of n.
fn top(mut scored: Vec<(u64, f32)>, limit: usize) -> Vec<(u64, f32)> {
    let order = |a: &(u64, f32), b: &(u64, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if limit == 0 {
        return Vec::new();
    }
    if scored.len() > limit {
        scored.select_nth_unstable_by(limit - 1, order);
        scored.truncate(limit);
    }
    scored.sort_unstable_by(order);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::time::Instant;

    /// A deterministic pseudo-random (not yet normalised) vector for `index`.
    fn random(index: u64, salt: u64, dim: usize) -> Vec<f32> {
        (0..dim as u64)
            .map(|d| {
                let h = cutoff::mix64(salt ^ cutoff::mix64(index.wrapping_mul(10_007).wrapping_add(d)));
                (h >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn random_space(n: u64, dim: usize) -> Space {
        let mut space = Space::new("m", dim).expect("space");
        for i in 0..n {
            assert!(space.insert(i, &random(i, 7, dim)));
        }
        space
    }

    fn assert_dense(space: &Space) {
        assert_eq!(space.data.len(), space.len() * space.dim());
        assert_eq!(space.keys.len(), space.index.len());
        for (slot, key) in space.keys.iter().enumerate() {
            assert_eq!(space.index.get(key), Some(&slot));
        }
    }

    #[test]
    fn new_validates_its_arguments() {
        assert!(Space::new("m", 0).is_none());
        assert!(Space::new("m", 8193).is_none());
        assert!(Space::new("", 4).is_none());
        let space = Space::new("siglip2", 8192).expect("space");
        assert_eq!(space.model_id(), "siglip2");
        assert_eq!(space.dim(), 8192);
        assert!(space.is_empty());
        assert_eq!(space.len(), 0);
    }

    #[test]
    fn insert_replace_remove_contains_get_iter() {
        let mut space = Space::new("m", 2).expect("space");
        assert!(space.insert(10, &[1.0, 0.0]));
        assert!(space.insert(20, &[0.0, 2.0]));
        assert!(space.insert(30, &[-3.0, 0.0]));
        assert_eq!(space.len(), 3);
        assert!(space.contains(20) && !space.contains(99));
        assert_eq!(space.get(30), Some(&[-1.0, 0.0][..]));
        assert_eq!(space.get(99), None);

        // Replacing keeps one vector under the key.
        assert!(space.insert(20, &[5.0, 0.0]));
        assert_eq!(space.len(), 3);
        assert_eq!(space.get(20), Some(&[1.0, 0.0][..]));

        // Removing the first moves the last into its place.
        assert!(space.remove(10));
        assert!(!space.remove(10));
        assert_eq!(space.len(), 2);
        assert!(!space.contains(10));
        let all: Vec<(u64, Vec<f32>)> = space.iter().map(|(k, v)| (k, v.to_vec())).collect();
        assert_eq!(all, vec![(30, vec![-1.0, 0.0]), (20, vec![1.0, 0.0])]);
        assert_eq!(space.get(30), Some(&[-1.0, 0.0][..]));
        assert_dense(&space);

        // Removing the last one, then the only one.
        assert!(space.remove(20));
        assert!(space.remove(30));
        assert!(space.is_empty());
        assert_eq!(space.iter().count(), 0);
        assert_dense(&space);
    }

    #[test]
    fn vectors_are_normalised_on_the_way_in() {
        let mut space = Space::new("m", 2).expect("space");
        assert!(space.insert(1, &[3.0, 4.0]));
        let v = space.get(1).expect("vector");
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6, "{v:?}");
        // Huge and tiny values normalise too, without overflowing or vanishing.
        assert!(space.insert(2, &[3.0e30, 4.0e30]));
        assert!(space.insert(3, &[3.0e-30, 4.0e-30]));
        for key in [2, 3] {
            let v = space.get(key).expect("vector");
            assert!((v[0] - 0.6).abs() < 1e-5 && (v[1] - 0.8).abs() < 1e-5, "{v:?}");
        }
    }

    #[test]
    fn bad_vectors_change_nothing() {
        let mut space = Space::new("m", 3).expect("space");
        assert!(space.insert(1, &[1.0, 0.0, 0.0]));
        assert!(!space.insert(2, &[1.0, 0.0]));
        assert!(!space.insert(2, &[1.0, 0.0, 0.0, 0.0]));
        assert!(!space.insert(2, &[]));
        assert!(!space.insert(2, &[1.0, f32::NAN, 0.0]));
        assert!(!space.insert(2, &[1.0, f32::INFINITY, 0.0]));
        assert!(!space.insert(2, &[0.0, 0.0, 0.0]));
        // A bad replacement leaves the old vector alone.
        assert!(!space.insert(1, &[0.0, 0.0, 0.0]));
        assert_eq!(space.len(), 1);
        assert!(!space.contains(2));
        assert_eq!(space.get(1), Some(&[1.0, 0.0, 0.0][..]));
        assert_dense(&space);
    }

    #[test]
    fn nearest_finds_the_planted_neighbour_first() {
        let mut space = random_space(200, 32);
        let query = random(5_000, 99, 32);
        // A near copy of the query, planted under a key that sorts last.
        let near: Vec<f32> = query.iter().zip(random(1, 3, 32)).map(|(q, n)| q + 0.05 * n).collect();
        assert!(space.insert(9_999, &near));
        let hits = space.nearest(&query, 5, None);
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].0, 9_999);
        assert!(hits[0].1 > 0.95, "{}", hits[0].1);
        assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn nearest_excludes_the_given_key() {
        let space = random_space(50, 16);
        let own = space.get(7).expect("vector").to_vec();
        assert_eq!(space.nearest(&own, 1, None)[0].0, 7);
        let hits = space.nearest(&own, 100, Some(7));
        assert_eq!(hits.len(), 49);
        assert!(hits.iter().all(|(k, _)| *k != 7));
    }

    #[test]
    fn nearest_breaks_ties_by_key_and_respects_limit() {
        let mut space = Space::new("m", 2).expect("space");
        for key in [5, 1, 9, 3] {
            assert!(space.insert(key, &[1.0, 0.0]));
        }
        assert!(space.insert(7, &[0.0, 1.0]));
        let hits = space.nearest(&[1.0, 0.0], 10, None);
        let keys: Vec<u64> = hits.iter().map(|h| h.0).collect();
        assert_eq!(keys, vec![1, 3, 5, 9, 7]);
        // The cut falls inside the tie: the lowest keys survive.
        let keys: Vec<u64> = space.nearest(&[1.0, 0.0], 2, None).iter().map(|h| h.0).collect();
        assert_eq!(keys, vec![1, 3]);
        assert!(space.nearest(&[1.0, 0.0], 0, None).is_empty());
        assert_eq!(space.nearest(&[1.0, 0.0], usize::MAX, None).len(), 5);
        // Scores are cosines, and an exact match is exactly 1.
        assert_eq!(hits[0].1, 1.0);
        assert_eq!(hits[4].1, 0.0);
    }

    #[test]
    fn nearest_on_nothing_gives_nothing() {
        let empty = Space::new("m", 4).expect("space");
        assert!(empty.nearest(&[1.0, 0.0, 0.0, 0.0], 5, None).is_empty());
        assert_eq!(empty.search(&[1.0, 0.0, 0.0, 0.0], 5, None, None), (Vec::new(), None));
        let space = random_space(10, 4);
        assert!(space.nearest(&[1.0, 0.0, 0.0], 5, None).is_empty());
        assert!(space.nearest(&[1.0, 0.0, 0.0, 0.0, 0.0], 5, None).is_empty());
        assert!(space.nearest(&[0.0; 4], 5, None).is_empty());
        assert!(space.nearest(&[f32::NAN, 0.0, 0.0, 1.0], 5, None).is_empty());
        assert!(space.nearest(&[], 5, None).is_empty());
    }

    #[test]
    fn the_query_is_normalised_first() {
        let space = random_space(20, 8);
        let q = random(777, 1, 8);
        let scaled: Vec<f32> = q.iter().map(|x| x * 40.0).collect();
        let (a, b) = (space.nearest(&q, 20, None), space.nearest(&scaled, 20, None));
        assert_eq!(a.len(), b.len());
        // Scaling rounds differently in the last bit, which may swap two near-equal scores.
        assert!(a.iter().all(|(k, s)| b.iter().any(|(k2, s2)| k == k2 && (s - s2).abs() < 1e-5)));
    }

    #[test]
    fn seed_ignores_insertion_order_but_not_contents() {
        let mut a = Space::new("m", 4).expect("space");
        let mut b = Space::new("m", 4).expect("space");
        for key in [1, 2, 3, 4] {
            assert!(a.insert(key, &random(key, 1, 4)));
        }
        for key in [4, 2, 1, 3] {
            assert!(b.insert(key, &random(key, 1, 4)));
        }
        assert_eq!(a.seed(), b.seed());
        // Different vectors under the same keys are the same names; a different key is not.
        assert!(a.insert(5, &random(5, 1, 4)));
        assert_ne!(a.seed(), b.seed());
        assert!(a.remove(5));
        assert_eq!(a.seed(), b.seed());
        assert!(a.remove(1));
        assert_ne!(a.seed(), b.seed());
        // The dimension is part of the name.
        let mut c = Space::new("m", 5).expect("space");
        for key in [1, 2, 3, 4] {
            assert!(c.insert(key, &random(key, 1, 5)));
        }
        assert_ne!(c.seed(), b.seed());
    }

    #[test]
    fn background_needs_two_vectors_and_is_deterministic() {
        let mut space = Space::new("m", 16).expect("space");
        assert!(space.background().is_none());
        assert!(space.insert(1, &random(1, 7, 16)));
        assert!(space.background().is_none());
        assert!(space.insert(2, &random(2, 7, 16)));
        assert!(space.background().is_some());
        let space = random_space(300, 32);
        let bg = space.background().expect("background");
        assert_eq!(Some(bg), space.background());
        assert_eq!(bg.n_at_compute, 300);
        assert_eq!(bg.seed, space.seed());
        assert!(bg.mean < 0.1 && bg.sigma > 0.0, "{bg:?}");
    }

    #[test]
    fn search_drops_noise_and_keeps_a_real_hit() {
        let dim = 64;
        let mut space = random_space(400, dim);
        let query = random(10_000, 99, dim);
        // Three near copies of the query: cosine about 0.95 with it.
        for (i, key) in [9_001_u64, 9_002, 9_003].into_iter().enumerate() {
            let noise = random(i as u64, 5, dim);
            let near: Vec<f32> = query.iter().zip(&noise).map(|(q, n)| q + 0.1 * n).collect();
            assert!(space.insert(key, &near));
        }
        let bg = space.background().expect("background");

        // Without a background nothing is cut.
        let (all, none) = space.search(&query, 50, None, None);
        assert_eq!((all.len(), none), (50, None));

        let (hits, ceiling) = space.search(&query, 50, None, Some(&bg));
        let ceiling = ceiling.expect("the cutoff engages");
        let keys: Vec<u64> = hits.iter().map(|h| h.0).collect();
        assert_eq!(keys.len(), 3, "{hits:?} under {ceiling}");
        assert!(keys.iter().all(|k| (9_001..=9_003).contains(k)));
        assert!(hits.iter().all(|h| h.1 >= ceiling && h.1 > 0.9), "{hits:?}");

        // A pure-noise query keeps nothing, and the excluded vector is not a candidate.
        let (hits, ceiling) = space.search(&random(20_000, 4, dim), 50, None, Some(&bg));
        assert!(hits.is_empty(), "{hits:?}");
        assert!(ceiling.is_some());
        let a = space.search(&query, 5, Some(9_001), Some(&bg)).1.expect("ceiling");
        let b = space.search(&query, 5, None, Some(&bg)).1.expect("ceiling");
        assert!(a < b, "one fewer candidate lowers the ceiling");
    }

    #[test]
    fn a_short_list_is_shown_whole() {
        let space = random_space(5, 8);
        let bg = Background { mean: 0.0, sigma: 0.1, n_at_compute: 5, seed: 0 };
        let (hits, ceiling) = space.search(&random(9, 9, 8), 10, None, Some(&bg));
        assert_eq!((hits.len(), ceiling), (5, None));
    }

    #[test]
    fn background_staleness_needs_both_a_new_seed_and_real_growth() {
        let mut space = random_space(100, 8);
        let bg = space.background().expect("background");
        assert!(!space.background_is_stale(&bg));
        // One insert changes the seed but is not growth.
        assert!(space.insert(1_000, &random(1_000, 7, 8)));
        assert_ne!(space.seed(), bg.seed);
        assert!(!space.background_is_stale(&bg));
        // Ten more is exactly 10%: not more than the fraction.
        for key in 1_001..1_010 {
            assert!(space.insert(key, &random(key, 7, 8)));
        }
        assert_eq!(space.len(), 110);
        assert!(!space.background_is_stale(&bg));
        assert!(space.insert(1_010, &random(1_010, 7, 8)));
        assert!(space.background_is_stale(&bg));
        // Shrinking counts the same way.
        let mut small = random_space(100, 8);
        let bg = small.background().expect("background");
        for key in 0..11 {
            assert!(small.remove(key));
        }
        assert!(small.background_is_stale(&bg));
        // Same size, different contents: the seed moved but the size did not.
        let mut same = random_space(100, 8);
        assert!(same.remove(0));
        assert!(same.insert(5_000, &random(5_000, 7, 8)));
        assert_ne!(same.seed(), bg.seed);
        assert!(!same.background_is_stale(&bg));
        // Large change but identical seed cannot happen by growth; a fresh background is fresh.
        let fresh = small.background().expect("background");
        assert!(!small.background_is_stale(&fresh));
    }

    #[test]
    fn dot_matches_the_obvious_sum() {
        for n in [0, 1, 7, 8, 9, 31, 33] {
            let a = random(1, 1, n);
            let b = random(2, 2, n);
            let plain: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - plain).abs() < 1e-4, "{n}");
        }
    }

    #[test]
    fn brute_force_is_fast_enough() {
        let space = random_space(10_000, 256);
        let query = random(77_777, 5, 256);
        let start = Instant::now();
        let hits = space.nearest(&query, 50, None);
        let elapsed = start.elapsed();
        assert_eq!(hits.len(), 50);
        eprintln!("nearest over 10k x 256: {elapsed:?}");
        assert!(elapsed.as_secs_f32() < 1.0, "{elapsed:?}");
    }

    /// The design target: 100k x 768 in well under 100 ms. Slow to build, so run on request:
    /// `cargo test -p lightcraft-vera space -- --ignored --nocapture`.
    #[test]
    #[ignore = "builds a 300 MB space; run on request"]
    fn brute_force_at_the_design_size() {
        let space = random_space(100_000, 768);
        let query = random(77_777, 5, 768);
        let start = Instant::now();
        let hits = space.nearest(&query, 50, None);
        let elapsed = start.elapsed();
        assert_eq!(hits.len(), 50);
        eprintln!("nearest over 100k x 768: {elapsed:?}");
        assert!(elapsed.as_millis() < 100, "{elapsed:?}");
    }

    proptest! {
        #[test]
        fn random_operations_keep_the_slab_dense(
            ops in proptest::collection::vec(
                (0_u8..4, 0_u64..12, proptest::collection::vec(-5.0_f32..5.0, 0..5)),
                0..60,
            ),
        ) {
            let mut space = Space::new("m", 3).expect("space");
            for (op, key, vector) in ops {
                match op {
                    0 | 1 => { let _ = space.insert(key, &vector); }
                    2 => { let _ = space.remove(key); }
                    _ => {
                        let _ = space.nearest(&vector, 4, Some(key));
                        let _ = space.search(&vector, 4, None, space.background().as_ref());
                        let _ = space.get(key);
                    }
                }
                prop_assert_eq!(space.data.len(), space.len() * space.dim());
                prop_assert_eq!(space.iter().count(), space.len());
                prop_assert_eq!(space.index.len(), space.len());
                for (slot, key) in space.keys.iter().enumerate() {
                    prop_assert_eq!(space.index.get(key), Some(&slot));
                }
            }
        }
    }
}
