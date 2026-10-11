//! The keyword lane: a small inverted index with BM25 ranking over a photo's words.
//!
//! The studio's keyword lane quoted every query word and OR-ed them into SQLite's FTS5, ranked by
//! BM25 with the title weighted above the keywords, the keywords above the caption. LightCraft has
//! no SQLite, so the same lane is an in-memory index: documents are `u64` ids (the engine passes
//! photo ids), each with a fixed number of **fields** (title, keywords, caption, alt text, file
//! stem, camera/lens…) whose weights the caller sets once. A query is tokenised the same way as a
//! document; each query token matches every indexed term it is a **prefix** of (so `moun` finds
//! `mountains`), tokens are OR-ed, and a document matching more tokens ranks above one matching
//! fewer because its BM25 sum is larger.
//!
//! Tokens: Unicode letters and digits, lower-cased (`char::to_lowercase`), split on everything
//! else; no stemming, no stop words (a photographer's words are short and specific). NFKC folding is
//! deliberately left out: the standard library has no table for it and a crate for it is a
//! dependency decision the engine's owner makes (`docs/vera.md` → Open questions).
//!
//! Deterministic: equal scores order by document id, so the same catalog gives the same list.

use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;

use crate::thresholds::{BM25_B, BM25_K1};

/// One ranked result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub doc: u64,
    pub score: f32,
}

/// The index. Cheap to build (a 100k-photo catalog in well under the two seconds `docs/vera.md`
/// budgets), rebuilt from a catalog snapshot and updated in place as metadata changes.
#[derive(Clone, Debug, Default)]
pub struct Index {
    /// One weight per field, in the order documents give their fields.
    weights: Vec<f32>,
    /// Term -> (document -> weighted term frequency: the sum over fields of weight x count). A
    /// sorted map so a prefix query is a range scan.
    postings: BTreeMap<String, BTreeMap<u64, f32>>,
    /// Document -> weighted length (the sum of its weighted term frequencies). Its key set is the
    /// set of indexed documents, so `len()` and the BM25 average length read it.
    doc_len: HashMap<u64, f32>,
    /// Document -> the terms it posted under, so removal touches only those terms.
    doc_terms: HashMap<u64, Vec<String>>,
}

impl Index {
    /// An empty index whose documents have `weights.len()` fields, weighted as given (a weight of
    /// `0.0` means the field is stored for nothing; `1.0` is neutral).
    pub fn new(weights: &[f32]) -> Index {
        Index { weights: weights.to_vec(), ..Index::default() }
    }

    /// The field weights this index was made with.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Documents in the index.
    pub fn len(&self) -> usize {
        self.doc_len.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_len.is_empty()
    }

    /// Index (or re-index) `doc` with one string per field, in weight order. Fewer strings than
    /// weights means the missing fields are empty; extra strings are ignored.
    pub fn upsert(&mut self, doc: u64, fields: &[&str]) {
        self.remove(doc);
        let mut weighted: BTreeMap<String, f32> = BTreeMap::new();
        for (weight, text) in self.weights.iter().zip(fields) {
            // A zero, negative or non-finite weight stores the field for nothing.
            if !weight.is_finite() || *weight <= 0.0 {
                continue;
            }
            for token in tokenize(text) {
                *weighted.entry(token).or_insert(0.0) += *weight;
            }
        }
        let mut length = 0.0_f32;
        let mut terms = Vec::with_capacity(weighted.len());
        for (term, tf) in weighted {
            length += tf;
            self.postings.entry(term.clone()).or_default().insert(doc, tf);
            terms.push(term);
        }
        if !length.is_finite() {
            length = f32::MAX;
        }
        self.doc_len.insert(doc, length);
        self.doc_terms.insert(doc, terms);
    }

    /// Forget `doc`. A document that was never indexed is a no-op.
    pub fn remove(&mut self, doc: u64) {
        self.doc_len.remove(&doc);
        let Some(terms) = self.doc_terms.remove(&doc) else {
            return;
        };
        for term in terms {
            let now_empty = match self.postings.get_mut(&term) {
                Some(docs) => {
                    docs.remove(&doc);
                    docs.is_empty()
                }
                None => false,
            };
            if now_empty {
                self.postings.remove(&term);
            }
        }
    }

    /// Whether `doc` is in the index.
    pub fn contains(&self, doc: u64) -> bool {
        self.doc_len.contains_key(&doc)
    }

    /// The best `limit` documents for `query`, most relevant first, ties by document id. A query
    /// with no tokens, or whose tokens match no term, returns nothing — never everything.
    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        if limit == 0 || self.doc_len.is_empty() {
            return Vec::new();
        }
        let mut tokens = tokenize(query);
        tokens.sort();
        tokens.dedup();
        if tokens.is_empty() {
            return Vec::new();
        }

        let n_docs = self.doc_len.len();
        let n = n_docs as f32;
        // f64 so the sum does not depend on the hash map's iteration order.
        let total: f64 = self.doc_len.values().map(|l| f64::from(*l)).sum();
        let avg_len = (total / n_docs as f64) as f32;

        let mut scores: HashMap<u64, f32> = HashMap::new();
        for token in &tokens {
            for (term, docs) in self.postings.range::<str, _>((Bound::Included(token.as_str()), Bound::Unbounded)) {
                if !term.starts_with(token.as_str()) {
                    break;
                }
                let df = docs.len() as f32;
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                for (doc, tf) in docs {
                    let len = self.doc_len.get(doc).copied().unwrap_or(avg_len);
                    let ratio = if avg_len > 0.0 { len / avg_len } else { 1.0 };
                    let part = tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * ratio));
                    let contribution = idf * part;
                    if contribution.is_finite() {
                        *scores.entry(*doc).or_insert(0.0) += contribution;
                    }
                }
            }
        }

        let mut hits: Vec<Hit> = scores.into_iter().map(|(doc, score)| Hit { doc, score }).collect();
        let order = |a: &Hit, b: &Hit| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc));
        if hits.len() > limit {
            hits.select_nth_unstable_by(limit - 1, order);
            hits.truncate(limit);
        }
        hits.sort_by(order);
        hits
    }
}

/// The tokens of `text`: runs of Unicode alphanumerics, lower-cased. Public so the engine can show
/// a user what their query became.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for l in ch.to_lowercase() {
                cur.push(l);
            }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutoff::mix64;
    use proptest::prelude::*;

    fn ids(hits: &[Hit]) -> Vec<u64> {
        hits.iter().map(|h| h.doc).collect()
    }

    #[test]
    fn title_outranks_caption() {
        let mut ix = Index::new(&[3.0, 1.0]);
        // The caption-only document has the lower id, so a tie-break could not explain the order.
        ix.upsert(1, &["lake", "a mountain at dawn"]);
        ix.upsert(2, &["mountain", "a lake at dawn"]);
        let hits = ix.search("mountain", 10);
        assert_eq!(ids(&hits), vec![2, 1]);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn more_matching_tokens_rank_higher() {
        let mut ix = Index::new(&[1.0]);
        ix.upsert(1, &["red"]);
        ix.upsert(2, &["red barn"]);
        ix.upsert(3, &["blue sky"]);
        let hits = ix.search("red barn", 10);
        assert_eq!(ids(&hits), vec![2, 1]);
    }

    #[test]
    fn prefix_matches_and_unrelated_matches_nothing() {
        let mut ix = Index::new(&[1.0]);
        ix.upsert(1, &["mountains"]);
        ix.upsert(2, &["mount everest"]);
        ix.upsert(3, &["ocean"]);
        let mut found = ids(&ix.search("moun", 10));
        found.sort_unstable();
        assert_eq!(found, vec![1, 2]);
        assert!(ix.search("zzz", 10).is_empty());
        assert!(ix.search("", 10).is_empty());
        assert!(ix.search("  !!! ", 10).is_empty());
        // A prefix is not a substring.
        assert!(ix.search("ountains", 10).is_empty());
    }

    #[test]
    fn prefix_terms_each_contribute() {
        let mut ix = Index::new(&[1.0]);
        ix.upsert(1, &["mountain mount"]);
        ix.upsert(2, &["mountain"]);
        ix.upsert(3, &["ocean"]);
        let hits = ix.search("moun", 10);
        assert_eq!(ids(&hits), vec![1, 2]);
    }

    #[test]
    fn zero_limit_returns_nothing() {
        let mut ix = Index::new(&[1.0]);
        ix.upsert(1, &["tree"]);
        assert!(ix.search("tree", 0).is_empty());
        assert_eq!(ix.search("tree", 1).len(), 1);
    }

    #[test]
    fn upsert_replaces_old_words() {
        let mut ix = Index::new(&[1.0]);
        ix.upsert(1, &["harbour boats"]);
        ix.upsert(2, &["forest"]);
        assert_eq!(ids(&ix.search("harbour", 10)), vec![1]);
        ix.upsert(1, &["desert"]);
        assert_eq!(ix.len(), 2);
        assert!(ix.search("harbour", 10).is_empty());
        assert!(ix.search("boats", 10).is_empty());
        assert_eq!(ids(&ix.search("desert", 10)), vec![1]);
    }

    #[test]
    fn remove_forgets_the_document() {
        let mut ix = Index::new(&[1.0]);
        ix.upsert(1, &["harbour"]);
        ix.upsert(2, &["harbour"]);
        assert!(ix.contains(1));
        ix.remove(1);
        assert!(!ix.contains(1));
        assert_eq!(ix.len(), 1);
        assert_eq!(ids(&ix.search("harbour", 10)), vec![2]);
        ix.remove(1);
        ix.remove(99);
        ix.remove(2);
        assert!(ix.is_empty());
        assert!(ix.search("harbour", 10).is_empty());
    }

    #[test]
    fn empty_and_zero_weight_documents_are_indexed_but_match_nothing() {
        let mut ix = Index::new(&[0.0, 1.0]);
        ix.upsert(1, &["hidden", ""]);
        ix.upsert(2, &[]);
        assert_eq!(ix.len(), 2);
        assert!(ix.search("hidden", 10).is_empty());
        let empty = Index::new(&[]);
        assert!(empty.search("anything", 10).is_empty());
        assert_eq!(empty.weights(), &[] as &[f32]);
    }

    #[test]
    fn ties_order_by_document_id() {
        let mut ix = Index::new(&[1.0]);
        for doc in [9_u64, 3, 5, 7] {
            ix.upsert(doc, &["same words here"]);
        }
        assert_eq!(ids(&ix.search("words", 10)), vec![3, 5, 7, 9]);
        assert_eq!(ids(&ix.search("words", 2)), vec![3, 5]);
    }

    #[test]
    fn unicode_is_lower_cased() {
        let mut ix = Index::new(&[1.0]);
        ix.upsert(1, &["Ärger im Büro"]);
        ix.upsert(2, &["Straße"]);
        assert_eq!(ids(&ix.search("ärger", 10)), vec![1]);
        assert_eq!(ids(&ix.search("ÄRGER", 10)), vec![1]);
        assert_eq!(ids(&ix.search("BÜRO", 10)), vec![1]);
        assert_eq!(tokenize("Ärger, im Büro!"), vec!["ärger", "im", "büro"]);
    }

    #[test]
    fn twenty_thousand_documents_index_and_search_quickly() {
        let vocab: Vec<String> = (0_u64..4000)
            .map(|i| {
                let mut h = mix64(i);
                let len = 4 + (h % 5) as usize;
                let mut w = String::new();
                for _ in 0..len {
                    h = mix64(h);
                    w.push(char::from(b'a' + (h % 26) as u8));
                }
                w
            })
            .collect();
        let word = |h: u64| vocab.get((h % vocab.len() as u64) as usize).map(String::as_str).unwrap_or("x");
        let start = std::time::Instant::now();
        let mut ix = Index::new(&[3.0, 1.0]);
        for doc in 0_u64..20_000 {
            let mut h = mix64(doc ^ 0xABCD);
            let mut title = String::new();
            let mut caption = String::new();
            for k in 0..10 {
                h = mix64(h);
                let target = if k < 3 { &mut title } else { &mut caption };
                target.push_str(word(h));
                target.push(' ');
            }
            ix.upsert(doc, &[&title, &caption]);
        }
        let built = start.elapsed();
        let mut total_hits = 0;
        for q in 0_u64..50 {
            let query = format!("{} {}", word(mix64(q)), word(mix64(q + 1000)));
            let hits = ix.search(&query, 50);
            assert!(hits.len() <= 50);
            total_hits += hits.len();
        }
        let all = start.elapsed();
        eprintln!("20k docs: build {built:?}, 50 searches done at {all:?} ({total_hits} hits)");
        assert!(total_hits > 0);
        assert!(all < std::time::Duration::from_secs(2), "indexing and searching 20k documents took {all:?}");
    }

    proptest! {
        #[test]
        fn random_documents_and_queries_never_panic(
            docs in proptest::collection::vec(
                (0_u64..40, proptest::collection::vec("[a-zA-Z0-9äöüß ,.]{0,30}", 0..4)),
                0..30,
            ),
            removed in proptest::collection::vec(0_u64..40, 0..10),
            query in "\\PC{0,24}",
            limit in 0_usize..20,
        ) {
            let mut ix = Index::new(&[3.0, 2.0, 1.0]);
            for (doc, fields) in &docs {
                let refs: Vec<&str> = fields.iter().map(String::as_str).collect();
                ix.upsert(*doc, &refs);
            }
            for doc in &removed {
                ix.remove(*doc);
            }
            let hits = ix.search(&query, limit);
            prop_assert!(hits.len() <= limit);
            for pair in hits.windows(2) {
                if let [a, b] = pair {
                    prop_assert!(a.score > b.score || (a.score == b.score && a.doc < b.doc));
                }
            }
            for hit in &hits {
                prop_assert!(ix.contains(hit.doc));
                prop_assert!(hit.score.is_finite() && hit.score > 0.0);
            }
        }
    }
}
