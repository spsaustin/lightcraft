//! Where meaning ends and noise begins in a list of cosine scores.
//!
//! A cosine's "unrelated" floor is not zero, and a nonsense query produces a list bunched just
//! above that floor, so neither a fixed threshold nor a ratio to the top can tell "this whole list
//! is background" from "these are good hits". The studio's answer, ported here constant for
//! constant:
//!
//! 1. **Background.** Sample [`BACKGROUND_PAIRS`] pairs of vectors from the space (deterministic:
//!    a splitmix64 stream from a seed derived from the space, so the same space gives the same
//!    sample) and take their cosines. The *lower half* of that sample is a one-sided view of the
//!    unrelated population; its median is the background mean and its median absolute deviation
//!    times [`TAIL_MAD_TO_SIGMA`] is the background sigma.
//! 2. **Ceiling.** For a query scored against `n` candidates, the largest of `n` background
//!    cosines is expected about `sqrt(2 ln n)` sigmas above the mean; `+ 0.674` is the median of
//!    the maximum's own spread; `+ 1` is one more sigma of margin. Anything under
//!    `mean + sigma · (sqrt(2 ln n) + 0.674 + 1)` is what noise alone would have produced.
//! 3. **Engage** only once at least [`MIN_SCORED_FOR_CUTOFF`] candidates were scored; a shorter
//!    list is shown whole.
//!
//! The vector lane applies this to raw cosines, before fusion; the fused list then keeps whatever
//! the keyword lane also returned, so a frame found by its words is never cut by the model's doubt.

use crate::thresholds::{BACKGROUND_PAIRS, MIN_SCORED_FOR_CUTOFF, TAIL_MAD_TO_SIGMA};

/// A space's background cosine statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Background {
    pub mean: f32,
    pub sigma: f32,
    /// How many vectors the space held when this was sampled (recompute when it has grown by
    /// [`crate::thresholds::BACKGROUND_REFRESH_FRACTION`]).
    pub n_at_compute: usize,
    pub seed: u64,
}

/// Cosine of two vectors assumed L2-normalised (a dot product); shorter slices are zero-padded by
/// stopping at the shorter length.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// splitmix64: a stateless, platform-independent mix, so a seed names the same sample everywhere.
pub fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sample the background of a space of `n` vectors read through `vector(i)` (which returns `None`
/// for an index it cannot serve), from `seed`. `None` when the space has fewer than two vectors.
pub fn background(n: usize, seed: u64, vector: impl Fn(usize) -> Option<Vec<f32>>) -> Option<Background> {
    if n < 2 {
        return None;
    }
    let modulus = n as u64;
    let mut cosines: Vec<f32> = Vec::with_capacity(BACKGROUND_PAIRS);
    for p in 0..BACKGROUND_PAIRS as u64 {
        let i = (mix64(seed ^ (2 * p)) % modulus) as usize;
        let j = (mix64(seed ^ (2 * p + 1)) % modulus) as usize;
        if i == j {
            continue;
        }
        let (Some(a), Some(b)) = (vector(i), vector(j)) else {
            continue;
        };
        cosines.push(cosine(&a, &b));
    }
    background_from_scores(&cosines, n, seed)
}

/// The background statistic over any population of cosines: the lower half's median is the mean,
/// its median absolute deviation times [`TAIL_MAD_TO_SIGMA`] the sigma. [`background`] feeds it
/// random pairs of a space; a query feeds it its own cosines against every vector, which is the
/// right population when the query lives on another scale than the space's pairs (a text query
/// against image vectors). `None` under two finite scores.
pub fn background_from_scores(scores: &[f32], n_at_compute: usize, seed: u64) -> Option<Background> {
    let mut cosines: Vec<f32> = scores.iter().copied().filter(|c| c.is_finite()).collect();
    if cosines.len() < 2 {
        return None;
    }
    cosines.sort_by(f32::total_cmp);
    let lower = cosines.get(..cosines.len().div_ceil(2))?;
    let mean = median_sorted(lower)?;
    let mut deviations: Vec<f32> = lower.iter().map(|x| (x - mean).abs()).collect();
    deviations.sort_by(f32::total_cmp);
    let mad = median_sorted(&deviations)?;
    let mut sigma = mad * TAIL_MAD_TO_SIGMA;
    if !sigma.is_finite() || sigma <= 0.0 {
        sigma = SIGMA_FLOOR;
    }
    Some(Background { mean, sigma, n_at_compute, seed })
}

/// Smallest sigma a background may have, so the ceiling is still defined for a space of identical
/// (or constant-cosine) vectors.
const SIGMA_FLOOR: f32 = 1e-6;

/// Median of an already sorted slice (`None` when empty).
fn median_sorted(sorted: &[f32]) -> Option<f32> {
    let len = sorted.len();
    if len == 0 {
        return None;
    }
    let hi = *sorted.get(len / 2)?;
    if len % 2 == 1 {
        Some(hi)
    } else {
        let lo = *sorted.get(len / 2 - 1)?;
        Some((lo + hi) / 2.0)
    }
}

/// The score under which a result among `candidates` scored ones is noise for this background.
pub fn ceiling(bg: &Background, candidates: usize) -> f32 {
    let n = candidates.max(2) as f32;
    bg.mean + bg.sigma * ((2.0 * n.ln()).sqrt() + 0.674 + 1.0)
}

/// Drop everything under the ceiling from `scored` (document, cosine), when at least
/// [`MIN_SCORED_FOR_CUTOFF`] were scored; returns the ceiling applied, if any. `scored` keeps its
/// order.
pub fn apply(scored: &mut Vec<(u64, f32)>, bg: &Background, candidates: usize) -> Option<f32> {
    if candidates < MIN_SCORED_FOR_CUTOFF {
        return None;
    }
    let c = ceiling(bg, candidates);
    scored.retain(|(_, s)| *s >= c);
    Some(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_from_scores_reads_the_lower_half() {
        // ten background scores around 0.1 and two hits near 0.9: the hits never move the statistic
        let scores = [0.08, 0.1, 0.12, 0.09, 0.11, 0.1, 0.13, 0.07, 0.1, 0.1, 0.9, 0.88];
        let bg = background_from_scores(&scores, 12, 7).unwrap();
        assert!((bg.mean - 0.09).abs() < 0.02, "{bg:?}");
        assert!(bg.sigma > 0.0 && bg.sigma < 0.1, "{bg:?}");
        assert_eq!((bg.n_at_compute, bg.seed), (12, 7));
        let c = ceiling(&bg, 12);
        assert!(c > 0.13 && c < 0.9, "the hits clear the ceiling, the background does not: {c}");
        assert!(background_from_scores(&[0.5], 1, 0).is_none());
        assert!(background_from_scores(&[f32::NAN, 0.5, f32::INFINITY], 3, 0).is_none());
    }

    const DIMS: usize = 64;

    /// A deterministic pseudo-random unit vector for `index`.
    fn unit(index: u64, salt: u64) -> Vec<f32> {
        let mut raw: Vec<f32> = (0..DIMS as u64)
            .map(|d| {
                let h = mix64(salt ^ mix64(index.wrapping_mul(1_000).wrapping_add(d)));
                (h >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect();
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut raw {
            *x /= norm;
        }
        raw
    }

    fn space(n: usize) -> Vec<Vec<f32>> {
        (0..n as u64).map(|i| unit(i, 7)).collect()
    }

    fn random_background(n: usize, seed: u64) -> Option<Background> {
        let vectors = space(n);
        background(n, seed, |i| vectors.get(i).cloned())
    }

    #[test]
    fn same_seed_gives_identical_background() {
        let a = random_background(300, 42);
        let b = random_background(300, 42);
        assert!(a.is_some());
        assert_eq!(a, b);
        let c = random_background(300, 43);
        assert_ne!(a, c);
    }

    #[test]
    fn random_unit_vectors_have_a_small_background() {
        let bg = random_background(500, 1).expect("background");
        // Cosines of random 64-d vectors have sigma about 1/8; the lower-half median sits about
        // 0.67 sigma below zero.
        assert!(bg.mean > -0.2 && bg.mean < 0.0, "mean {}", bg.mean);
        assert!(bg.sigma > 0.05 && bg.sigma < 0.25, "sigma {}", bg.sigma);
        assert_eq!(bg.n_at_compute, 500);
        assert_eq!(bg.seed, 1);
    }

    #[test]
    fn apply_keeps_a_planted_hit_and_drops_background() {
        let n = 400;
        let bg = random_background(n, 5).expect("background");
        let vectors = space(n);
        let query = unit(10_000, 99);
        let mut scored: Vec<(u64, f32)> = vectors.iter().enumerate().map(|(i, v)| (i as u64, cosine(&query, v))).collect();
        scored.push((9_999, 0.9));
        let candidates = scored.len();
        let applied = apply(&mut scored, &bg, candidates);
        assert!(applied.is_some());
        assert_eq!(scored, vec![(9_999, 0.9)]);
    }

    #[test]
    fn apply_does_nothing_for_short_lists() {
        let bg = Background { mean: 0.0, sigma: 0.1, n_at_compute: 100, seed: 0 };
        let mut scored: Vec<(u64, f32)> = (0..MIN_SCORED_FOR_CUTOFF as u64 - 1).map(|i| (i, 0.0)).collect();
        let before = scored.clone();
        let short = scored.len();
        assert_eq!(apply(&mut scored, &bg, short), None);
        assert_eq!(scored, before);
        // At the minimum it engages.
        let mut scored: Vec<(u64, f32)> = (0..MIN_SCORED_FOR_CUTOFF as u64).map(|i| (i, 0.0)).collect();
        let len = scored.len();
        assert!(apply(&mut scored, &bg, len).is_some());
        assert!(scored.is_empty());
    }

    #[test]
    fn ceiling_grows_with_candidates() {
        let bg = Background { mean: 0.1, sigma: 0.05, n_at_compute: 100, seed: 0 };
        let mut last = ceiling(&bg, 2);
        for n in [10, 100, 1_000, 100_000] {
            let c = ceiling(&bg, n);
            assert!(c > last, "{n}: {c} <= {last}");
            last = c;
        }
        assert!(ceiling(&bg, 0).is_finite());
        assert_eq!(ceiling(&bg, 0), ceiling(&bg, 2));
    }

    #[test]
    fn tiny_or_unreadable_spaces_have_no_background() {
        assert!(background(0, 1, |_| Some(vec![1.0])).is_none());
        assert!(background(1, 1, |_| Some(vec![1.0])).is_none());
        assert!(background(100, 1, |_| None).is_none());
    }

    #[test]
    fn identical_vectors_get_a_sigma_floor() {
        let bg = background(10, 3, |_| Some(vec![1.0, 0.0])).expect("background");
        assert!(bg.sigma > 0.0 && bg.sigma < 1e-3);
        assert!((bg.mean - 1.0).abs() < 1e-6);
        assert!(ceiling(&bg, 100).is_finite());
    }

    #[test]
    fn median_of_small_slices() {
        assert_eq!(median_sorted(&[]), None);
        assert_eq!(median_sorted(&[2.0]), Some(2.0));
        assert_eq!(median_sorted(&[1.0, 3.0]), Some(2.0));
        assert_eq!(median_sorted(&[1.0, 2.0, 9.0]), Some(2.0));
    }

    #[test]
    fn nan_vectors_are_skipped() {
        assert!(background(10, 3, |_| Some(vec![f32::NAN])).is_none());
    }
}
