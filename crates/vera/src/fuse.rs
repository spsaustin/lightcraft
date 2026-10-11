//! Reciprocal rank fusion: several ranked lists into one, by position only.
//!
//! `score(d) = Σ_lists weight · 1 / (k + rank_in_list(d))`, ranks 1-based, `k` = [`RRF_K`] by
//! default. Scores from the lanes are deliberately not read: a cosine and a BM25 are not on one
//! scale, and the studio's measurement showed rank fusion beat every score normalisation it tried.
//! Which is also why the noise cutoff lives *before* fusion ([`crate::cutoff`]): after it, no
//! magnitude is left to threshold.

use std::collections::{BTreeMap, HashSet};

use crate::thresholds::RRF_K;

/// Fuse `lists` (each most-relevant-first, documents by id) with per-list `weights` (a missing
/// weight is `1.0`), with the constant `k`. Most relevant first; ties by document id.
///
/// A document repeated within one list counts once, at its first (best) position; positions are
/// counted over the list as given, so a repeat still occupies a rank.
pub fn rrf_with(lists: &[&[u64]], weights: &[f32], k: f32) -> Vec<(u64, f32)> {
    let mut scores: BTreeMap<u64, f32> = BTreeMap::new();
    for (i, list) in lists.iter().enumerate() {
        let weight = weights.get(i).copied().unwrap_or(1.0);
        let mut seen: HashSet<u64> = HashSet::new();
        for (pos, doc) in list.iter().enumerate() {
            if !seen.insert(*doc) {
                continue;
            }
            let contribution = weight / (k + (pos + 1) as f32);
            if contribution.is_finite() {
                *scores.entry(*doc).or_insert(0.0) += contribution;
            }
        }
    }
    let mut out: Vec<(u64, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// [`rrf_with`] at `k` = [`RRF_K`].
pub fn rrf(lists: &[&[u64]], weights: &[f32]) -> Vec<(u64, f32)> {
    rrf_with(lists, weights, RRF_K)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn hand_computed_two_lists() {
        let a: &[u64] = &[10, 20, 30];
        let b: &[u64] = &[20, 40, 10];
        let out = rrf_with(&[a, b], &[], 60.0);
        // 20: 1/62 + 1/61; 10: 1/61 + 1/63; 40: 1/62; 30: 1/63.
        let expect = [(20, 1.0 / 62.0 + 1.0 / 61.0), (10, 1.0 / 61.0 + 1.0 / 63.0), (40, 1.0 / 62.0), (30, 1.0 / 63.0)];
        assert_eq!(out.len(), expect.len());
        for ((doc, score), (edoc, escore)) in out.iter().zip(expect) {
            assert_eq!(*doc, edoc);
            assert!(close(*score, escore), "{doc}: {score} vs {escore}");
        }
    }

    #[test]
    fn both_lists_at_rank_one_beats_one_list_at_rank_one() {
        let a: &[u64] = &[1, 2];
        let b: &[u64] = &[1, 3];
        let c: &[u64] = &[9];
        let out = rrf(&[a, b, c], &[]);
        assert_eq!(out.first().map(|x| x.0), Some(1));
        let one = out.iter().find(|x| x.0 == 9).map(|x| x.1);
        assert!(out.first().map(|x| x.1) > one);
    }

    #[test]
    fn weights_scale_a_lane() {
        let a: &[u64] = &[1];
        let b: &[u64] = &[2];
        let even = rrf_with(&[a, b], &[1.0, 1.0], 60.0);
        assert_eq!(even.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 2]);
        let weighted = rrf_with(&[a, b], &[1.0, 3.0], 60.0);
        assert_eq!(weighted.iter().map(|x| x.0).collect::<Vec<_>>(), vec![2, 1]);
        assert!(close(weighted.first().map(|x| x.1).unwrap_or(0.0), 3.0 / 61.0));
        // A missing weight is 1.0.
        let missing = rrf_with(&[a, b], &[2.0], 60.0);
        assert_eq!(missing.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 2]);
        assert!(close(missing.get(1).map(|x| x.1).unwrap_or(0.0), 1.0 / 61.0));
    }

    #[test]
    fn empty_inputs_return_empty() {
        assert!(rrf(&[], &[]).is_empty());
        let empty: &[u64] = &[];
        assert!(rrf(&[empty, empty], &[1.0]).is_empty());
    }

    #[test]
    fn duplicate_within_a_list_counts_once() {
        let a: &[u64] = &[7, 7, 7];
        let out = rrf_with(&[a], &[], 60.0);
        assert_eq!(out.len(), 1);
        assert!(close(out.first().map(|x| x.1).unwrap_or(0.0), 1.0 / 61.0));
    }

    #[test]
    fn ties_order_by_document_id() {
        let a: &[u64] = &[5];
        let b: &[u64] = &[3];
        let out = rrf_with(&[a, b], &[], 60.0);
        // 5 at rank 1 in one list and 3 at rank 1 in the other: equal scores.
        assert_eq!(out.iter().map(|x| x.0).collect::<Vec<_>>(), vec![3, 5]);
    }

    #[test]
    fn rrf_uses_the_default_constant() {
        let a: &[u64] = &[1, 2];
        assert_eq!(rrf(&[a], &[]), rrf_with(&[a], &[], RRF_K));
        let out = rrf(&[a], &[]);
        assert!(close(out.first().map(|x| x.1).unwrap_or(0.0), 1.0 / 61.0));
    }
}
