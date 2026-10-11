//! Zero-shot keywords (`docs/vera.md` → Phase 3, `docs/vera-describe.md` → "Keywords by zero-shot"): every prompt of
//! the shipped vocabulary ([`super::vocabulary`]) embedded once by SigLIP 2's text tower, and each photo's canonical
//! image vector scored against them. No new model; scoring a photo is a few hundred dot products.
//!
//! - **The matrix** ([`Matrix`]): one unit vector per prompt (a term has one prompt per phrasing), cached beside the
//!   library as `vera-keywords-<model id>-<vocabulary hash>.bin`. Little-endian: `"LCVK"`, version `1`, the model id
//!   (`u16` length + UTF-8), the vocabulary hash `u64`, `dim u32`, then one record per prompt in the vocabulary's order:
//!   `facet u16`, `term u16`, `dim × f32`. A file of another model, vocabulary, dimension or prompt list is ignored and
//!   rebuilt. In memory it outlives a library change (it depends on the model and the vocabulary only).
//! - **A term's cosine** is the best of its phrasings' cosines with the photo's vector.
//! - **Calibration**, three layers: (1) per keyword, once the library has [`CALIBRATE_FROM`] vectors or more, each
//!   term's median and MAD over a deterministic sample of up to [`CALIBRATION_SAMPLE`] library vectors ([`Calibration`],
//!   recomputed when the space has moved as its background is), so a photo's score for a term is
//!   `z = (cos − median) / (MAD · 2.5)`: a "hub" term that every picture is close to stops winning everywhere; below
//!   that many vectors raw cosines are used. (2) A per-photo ceiling: this photo's scores against every term are the
//!   population of [`lightcraft_vera::cutoff::background_from_scores`], and terms under `mean + 2.25 sigma` of it
//!   ([`CEILING_SIGMAS`], measured) are left out. On its own this ceiling ranks rather than separates: a photo of
//!   nothing in the vocabulary still has a top tail. Per-keyword calibration is what tells the two apart. (3) The vocabulary's facet rules: a `single` facet (time of day, season) keeps
//!   its best term only when it beats the runner-up by the facet's margin (a cosine; in z units, divided by the
//!   median term spread), a `multi` facet at most its `max`. At most [`MAX_KEYWORDS`] in all, the best first; the list
//!   is given in the vocabulary's facet order, then by score.

// the web build has no model to build the matrix with
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_variables, unused_mut))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lightcraft_vera::cutoff::{self, Background, mix64};
use lightcraft_vera::thresholds;
use serde_json::{Value, json};

use super::vocabulary::{Mode, Vocabulary};
use crate::Session;

const MAGIC: &[u8; 4] = b"LCVK";
const VERSION: u8 = 1;
/// Largest matrix file read (2000 terms × 8 phrasings × 768 numbers is 49 MB).
const MAX_FILE: u64 = 64 << 20;
const MAX_MODEL_ID: usize = 256;
/// The library needs this many vectors before keywords are scored per keyword (z-scores) instead of by raw cosine.
pub(crate) const CALIBRATE_FROM: usize = 200;
/// Most library vectors a calibration reads.
pub(crate) const CALIBRATION_SAMPLE: usize = 2000;
/// MAD → sigma for the per-keyword spread (the same factor as the cutoff's background).
const MAD_TO_SIGMA: f32 = 2.5;
/// The smallest spread a term may have (a constant term's z would divide by zero).
const SIGMA_FLOOR: f32 = 1e-6;
/// After a failed vocabulary build, how long before it is tried again.
const RETRY_BUILD: std::time::Duration = std::time::Duration::from_secs(60);
/// Most keywords proposed for one photo.
pub(crate) const MAX_KEYWORDS: usize = 15;
/// The per-photo ceiling, in sigmas of the photo's own background over every term (its lower half's median and MAD
/// × 2.5, [`cutoff::background_from_scores`]). Measured on the demo library (`describe_real_keywords_recover_the_demo_
/// scenes`, raw cosines, 398 terms): 2.0 → recall@10 of the scenes' own keywords 0.66 at 14.4 proposals a photo (15
/// cap), **2.25 → 0.59 at 11.5**, 2.5 → 0.50 at 7.7, 3.0 → 0.34 at 3.2; the search lane's maximum-of-n ceiling
/// (`cutoff::ceiling`, 5.1 sigmas at 398 candidates) let 1 keyword through in 24 scenes, because a scene's best term
/// sits only 2.5–4.5 sigmas over its background. 2.25 keeps the proposals in the PRD's 5–12 band.
pub(crate) const CEILING_SIGMAS: f32 = 2.25;

/// The file name of the matrix for `model_id` and a vocabulary hash, in the library folder.
pub(crate) fn file_name(model_id: &str, vocab_hash: u64) -> String {
    format!("vera-keywords-{model_id}-{vocab_hash:016x}.bin")
}

/// The vocabulary's prompts embedded: one unit vector per prompt, with the facet and term it stands for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Matrix {
    pub model_id: String,
    pub vocab_hash: u64,
    pub dim: usize,
    /// (facet, term) of each prompt, in the vocabulary's prompt order.
    pub rows: Vec<(u16, u16)>,
    /// `rows.len() × dim` numbers.
    pub data: Vec<f32>,
}

impl Matrix {
    /// A matrix from one vector per prompt of `vocab` (each normalised); `None` when the count, a length or a value is
    /// wrong.
    pub fn from_vectors(model_id: &str, vocab: &Vocabulary, dim: usize, vectors: &[Vec<f32>]) -> Option<Matrix> {
        let prompts = vocab.prompts();
        if prompts.len() != vectors.len() || dim == 0 {
            return None;
        }
        let mut rows = Vec::with_capacity(prompts.len());
        let mut data = Vec::with_capacity(prompts.len().checked_mul(dim)?);
        for ((f, t, _), v) in prompts.iter().zip(vectors) {
            if v.len() != dim {
                return None;
            }
            rows.push((u16::try_from(*f).ok()?, u16::try_from(*t).ok()?));
            data.extend(super::embeddings::unit(v)?);
        }
        Some(Matrix { model_id: model_id.to_string(), vocab_hash: vocab.hash(), dim, rows, data })
    }

    /// Whether this matrix was made for `vocab` with `model_id` (same hash, same prompt list).
    pub fn fits(&self, model_id: &str, vocab: &Vocabulary) -> bool {
        if self.model_id != model_id || self.vocab_hash != vocab.hash() {
            return false;
        }
        let prompts = vocab.prompts();
        prompts.len() == self.rows.len()
            && prompts.iter().zip(&self.rows).all(|((f, t, _), (rf, rt))| usize::from(*rf) == *f && usize::from(*rt) == *t)
            && self.data.len() == self.rows.len().saturating_mul(self.dim)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.data.len() * 4 + self.rows.len() * 4);
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        let id = self.model_id.as_bytes().get(..self.model_id.len().min(MAX_MODEL_ID)).unwrap_or_default();
        out.extend_from_slice(&(id.len() as u16).to_le_bytes());
        out.extend_from_slice(id);
        out.extend_from_slice(&self.vocab_hash.to_le_bytes());
        out.extend_from_slice(&u32::try_from(self.dim).unwrap_or(0).to_le_bytes());
        for (i, (f, t)) in self.rows.iter().enumerate() {
            out.extend_from_slice(&f.to_le_bytes());
            out.extend_from_slice(&t.to_le_bytes());
            for x in self.data.get(i * self.dim..(i + 1) * self.dim).unwrap_or_default() {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// The matrix in `bytes` when it is one for `model_id`, `vocab` and `dim`; `None` otherwise (rebuilt).
    pub fn from_bytes(bytes: &[u8], model_id: &str, vocab: &Vocabulary, dim: usize) -> Option<Matrix> {
        let rest = bytes.strip_prefix(MAGIC)?;
        let (version, rest) = rest.split_first()?;
        if *version != VERSION {
            return None;
        }
        let (len, rest) = rest.split_first_chunk::<2>()?;
        let len = usize::from(u16::from_le_bytes(*len));
        let id = rest.get(..len)?;
        if id != model_id.as_bytes() {
            return None;
        }
        let rest = rest.get(len..)?;
        let (hash, rest) = rest.split_first_chunk::<8>()?;
        let (d, mut rest) = rest.split_first_chunk::<4>()?;
        if usize::try_from(u32::from_le_bytes(*d)).ok()? != dim || dim == 0 {
            return None;
        }
        let record = dim.checked_mul(4)?.checked_add(4)?;
        if rest.len() % record != 0 {
            return None;
        }
        let n = rest.len() / record;
        let mut rows = Vec::with_capacity(n);
        let mut data = Vec::with_capacity(n.checked_mul(dim)?);
        while !rest.is_empty() {
            let (f, r) = rest.split_first_chunk::<2>()?;
            let (t, r) = r.split_first_chunk::<2>()?;
            let body = r.get(..dim * 4)?;
            rows.push((u16::from_le_bytes(*f), u16::from_le_bytes(*t)));
            for c in body.chunks_exact(4) {
                let x = f32::from_le_bytes(c.try_into().ok()?);
                if !x.is_finite() {
                    return None;
                }
                data.push(x);
            }
            rest = r.get(dim * 4..)?;
        }
        let m = Matrix { model_id: model_id.to_string(), vocab_hash: u64::from_le_bytes(*hash), dim, rows, data };
        m.fits(model_id, vocab).then_some(m)
    }
}

/// Embed every prompt of `vocab` with the model's text tower (a one-off of a minute or so on a CPU).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn build(model: &lightcraft_embed::Model, vocab: &Vocabulary) -> Result<Matrix, String> {
    let started = web_time::Instant::now();
    let prompts = vocab.prompts();
    let mut vectors = Vec::with_capacity(prompts.len());
    for (_, _, text) in &prompts {
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.embed_query(text)));
        match run {
            Ok(Ok(v)) => vectors.push(v),
            Ok(Err(e)) => return Err(e.to_string()),
            Err(_) => return Err("the text tower stopped unexpectedly".into()),
        }
    }
    let m = Matrix::from_vectors(lightcraft_embed::MODEL_ID, vocab, lightcraft_embed::DIM, &vectors)
        .ok_or_else(|| "the model returned unusable keyword embeddings".to_string())?;
    log::info!("vera: {} keyword prompts embedded in {:.1} s", prompts.len(), started.elapsed().as_secs_f64());
    Ok(m)
}

/// Each term's place in a flat list (facet by facet, terms in order), and the facet each belongs to.
fn term_offsets(vocab: &Vocabulary) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(vocab.facets.len());
    let mut n = 0usize;
    for f in &vocab.facets {
        offsets.push(n);
        n = n.saturating_add(f.terms.len());
    }
    offsets
}

/// Every term's cosine with `v` (the best of its phrasings'), flat in [`term_offsets`] order. `None` for a vector of
/// another length.
pub(crate) fn term_cosines(vocab: &Vocabulary, matrix: &Matrix, v: &[f32]) -> Option<Vec<f32>> {
    if v.len() != matrix.dim {
        return None;
    }
    let offsets = term_offsets(vocab);
    let mut out = vec![f32::NEG_INFINITY; vocab.terms()];
    for (i, (f, t)) in matrix.rows.iter().enumerate() {
        let row = matrix.data.get(i * matrix.dim..(i + 1) * matrix.dim)?;
        let c = cutoff::cosine(row, v);
        let at = offsets.get(usize::from(*f))?.checked_add(usize::from(*t))?;
        let slot = out.get_mut(at)?;
        if c > *slot {
            *slot = c;
        }
    }
    Some(out)
}

/// Per-keyword normalisation over the library: each term's median cosine and spread (MAD × 2.5).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Calibration {
    /// (median, sigma) per term, flat in [`term_offsets`] order.
    pub terms: Vec<(f32, f32)>,
    /// The median of the terms' sigmas: turns a facet margin (a cosine) into z units.
    pub sigma_median: f32,
    /// The space it was computed on (for the staleness rule of the space's background).
    pub n_at_compute: usize,
    pub seed: u64,
    pub vocab_hash: u64,
    pub sampled: usize,
}

impl Calibration {
    /// Whether the space has moved enough since (the space background's rule: grown or shrunk by a tenth and changed).
    pub fn is_stale(&self, space: &lightcraft_vera::Space) -> bool {
        // the size first: the seed walks every key, and this is asked on every describe job
        let moved = space.len().abs_diff(self.n_at_compute) as f64;
        if moved <= f64::from(thresholds::BACKGROUND_REFRESH_FRACTION) * self.n_at_compute as f64 {
            return false;
        }
        let bg = Background { mean: 0.0, sigma: 0.0, n_at_compute: self.n_at_compute, seed: self.seed };
        space.background_is_stale(&bg)
    }
}

fn median(v: &mut [f32]) -> Option<f32> {
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n == 0 {
        return None;
    }
    let hi = *v.get(n / 2)?;
    if n % 2 == 1 { Some(hi) } else { Some((hi + *v.get(n / 2 - 1)?) / 2.0) }
}

/// The calibration of `vocab` over `space`: a deterministic sample of up to [`CALIBRATION_SAMPLE`] of its vectors
/// (ordered by a mix of their key and the space's seed). `None` under [`CALIBRATE_FROM`] vectors.
pub(crate) fn calibrate(vocab: &Vocabulary, matrix: &Matrix, space: &lightcraft_vera::Space) -> Option<Calibration> {
    if space.len() < CALIBRATE_FROM {
        return None;
    }
    let seed = space.seed();
    let mut keys: Vec<(u64, u64)> = space.iter().map(|(k, _)| (mix64(k ^ seed), k)).collect();
    keys.sort_unstable();
    keys.truncate(CALIBRATION_SAMPLE);
    let terms = vocab.terms();
    let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(keys.len()); terms];
    for (_, k) in &keys {
        let Some(v) = space.get(*k) else { continue };
        let Some(c) = term_cosines(vocab, matrix, v) else { continue };
        for (col, x) in columns.iter_mut().zip(c) {
            col.push(x);
        }
    }
    let mut out = Vec::with_capacity(terms);
    for col in &mut columns {
        let med = median(col)?;
        let mut dev: Vec<f32> = col.iter().map(|x| (x - med).abs()).collect();
        let mad = median(&mut dev)?;
        let sigma = mad * MAD_TO_SIGMA;
        out.push((med, if sigma.is_finite() && sigma > 0.0 { sigma } else { SIGMA_FLOOR }));
    }
    let mut sigmas: Vec<f32> = out.iter().map(|(_, s)| *s).collect();
    let sigma_median = median(&mut sigmas).unwrap_or(SIGMA_FLOOR).max(SIGMA_FLOOR);
    Some(Calibration { terms: out, sigma_median, n_at_compute: space.len(), seed, vocab_hash: vocab.hash(), sampled: keys.len() })
}

/// One proposed keyword.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Keyword {
    pub keyword: String,
    pub facet: String,
    pub cosine: f32,
    /// The per-keyword z-score, when calibrated.
    pub z: Option<f32>,
}

/// A photo's zero-shot keywords.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Scored {
    pub keywords: Vec<Keyword>,
    /// The per-photo ceiling applied (in the score's units: z when calibrated, else cosine).
    pub ceiling: Option<f32>,
    pub calibrated: bool,
}

impl Scored {
    pub fn to_json(&self) -> Value {
        let round = |x: f32| (f64::from(x) * 1e4).round() / 1e4;
        let keywords: Vec<Value> = self
            .keywords
            .iter()
            .map(|k| {
                let mut row = json!({"keyword": k.keyword, "facet": k.facet, "cosine": round(k.cosine)});
                if let Some(z) = k.z {
                    row["z"] = json!(round(z));
                }
                row
            })
            .collect();
        json!({"keywords": keywords, "ceiling": self.ceiling.map(round), "calibrated": self.calibrated})
    }
}

/// The keywords for the picture whose vector is `v`: the three layers of the module doc, at most `limit` (capped at
/// [`MAX_KEYWORDS`]). `None` for a vector of another length.
pub(crate) fn score(vocab: &Vocabulary, matrix: &Matrix, calibration: Option<&Calibration>, v: &[f32], limit: usize) -> Option<Scored> {
    let cos = term_cosines(vocab, matrix, v)?;
    let calibration = calibration.filter(|c| c.terms.len() == cos.len() && c.vocab_hash == vocab.hash());
    let scores: Vec<f32> = match calibration {
        Some(c) => cos.iter().zip(&c.terms).map(|(x, (m, s))| (x - m) / s).collect(),
        None => cos.clone(),
    };
    // (2) the per-photo ceiling: this photo's scores against every term are its own background
    let n = scores.len();
    let bg = cutoff::background_from_scores(&scores, n, 0);
    let ceiling = bg.filter(|_| n >= thresholds::MIN_SCORED_FOR_CUTOFF).map(|b| b.mean + CEILING_SIGMAS * b.sigma);
    let above: std::collections::HashSet<usize> =
        scores.iter().enumerate().filter(|(_, s)| s.is_finite() && ceiling.is_none_or(|c| **s >= c)).map(|(i, _)| i).collect();
    // (3) facet rules
    let margin_scale = calibration.map_or(1.0, |c| 1.0 / c.sigma_median);
    let offsets = term_offsets(vocab);
    let mut picked: Vec<(usize, usize, f32)> = Vec::new(); // (facet, flat term, score)
    for (fi, facet) in vocab.facets.iter().enumerate() {
        let Some(&start) = offsets.get(fi) else { continue };
        let mut terms: Vec<(usize, f32)> = (start..start.saturating_add(facet.terms.len())).filter_map(|t| scores.get(t).map(|s| (t, *s))).collect();
        terms.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        match facet.mode {
            Mode::Single { margin } => {
                let Some(&(best, s)) = terms.first() else { continue };
                let clear = terms.get(1).is_none_or(|(_, r)| s - r >= margin * margin_scale);
                if above.contains(&best) && clear {
                    picked.push((fi, best, s));
                }
            }
            Mode::Multi { max } => {
                picked.extend(terms.iter().filter(|(t, _)| above.contains(t)).take(max).map(|(t, s)| (fi, *t, *s)));
            }
        }
    }
    picked.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.1.cmp(&b.1)));
    picked.truncate(limit.clamp(1, MAX_KEYWORDS));
    picked.sort_by(|a, b| a.0.cmp(&b.0).then(b.2.total_cmp(&a.2)).then(a.1.cmp(&b.1)));
    let keywords = picked
        .into_iter()
        .filter_map(|(fi, t, s)| {
            let facet = vocab.facets.get(fi)?;
            let term = facet.terms.get(t.checked_sub(*offsets.get(fi)?)?)?;
            Some(Keyword { keyword: term.keyword.clone(), facet: facet.name.clone(), cosine: *cos.get(t)?, z: calibration.map(|_| s) })
        })
        .collect();
    Some(Scored { keywords, ceiling, calibrated: calibration.is_some() })
}

/// What lasts across libraries (the vocabulary and its matrix) and what does not (the calibration).
pub(crate) struct KeywordState {
    pub vocab: Arc<Vocabulary>,
    pub matrix: Option<Arc<Matrix>>,
    /// The library folder the matrix was last written to.
    saved_in: Option<PathBuf>,
    pub calibration: Option<Arc<Calibration>>,
    /// A matrix being built off the session thread (`vera.pump`'s describe work).
    #[cfg(not(target_arch = "wasm32"))]
    building: Option<std::sync::mpsc::Receiver<Result<Matrix, String>>>,
    /// Why the last build failed, and when.
    pub error: Option<(String, web_time::Instant)>,
}

impl Default for KeywordState {
    fn default() -> Self {
        KeywordState {
            vocab: Arc::new(Vocabulary::builtin()),
            matrix: None,
            saved_in: None,
            calibration: None,
            #[cfg(not(target_arch = "wasm32"))]
            building: None,
            error: None,
        }
    }
}

impl KeywordState {
    /// Another library: the calibration is its library's; the matrix stays.
    pub fn library_changed(&mut self) -> KeywordState {
        KeywordState {
            vocab: self.vocab.clone(),
            matrix: self.matrix.take(),
            #[cfg(not(target_arch = "wasm32"))]
            building: self.building.take(),
            ..KeywordState::default()
        }
    }

    pub fn building(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return self.building.is_some();
        #[cfg(target_arch = "wasm32")]
        false
    }
}

/// The matrix file in library folder `dir`.
fn matrix_path(dir: &Path, vocab: &Vocabulary) -> PathBuf {
    dir.join(file_name(super::lane::MODEL_ID, vocab.hash()))
}

/// Keep `m` and write it beside the library (once per library folder).
fn adopt(s: &mut Session, m: Arc<Matrix>) {
    s.vera.describe.keywords.matrix = Some(m);
    save_matrix(s);
}

fn save_matrix(s: &mut Session) {
    let dir = super::library_dir(s);
    let k = &mut s.vera.describe.keywords;
    let (Some(dir), Some(m)) = (dir, &k.matrix) else { return };
    if k.saved_in.as_ref() == Some(&dir) {
        return;
    }
    let path = matrix_path(&dir, &k.vocab);
    if !path.is_file()
        && let Err(e) = crate::denoise::write_atomic(&path, &m.to_bytes())
    {
        log::warn!("vera: could not save the keyword vectors: {e}");
    }
    k.saved_in = Some(dir);
}

/// The vocabulary's matrix: in memory, else from the library's file, else built with the model's text tower on a thread
/// of its own (`Ok(None)` until it is done: a few minutes, once). `Ok(None)` when there is none and the vector lane
/// cannot make one.
pub(crate) fn matrix(s: &mut Session) -> Result<Option<Arc<Matrix>>, String> {
    let vocab = s.vera.describe.keywords.vocab.clone();
    // made here for this vocabulary (or read and checked whole): the id and hash are enough
    if let Some(m) = s.vera.describe.keywords.matrix.clone()
        && m.model_id == super::lane::MODEL_ID
        && m.vocab_hash == vocab.hash()
        && m.dim == lightcraft_embed::DIM
    {
        save_matrix(s);
        return Ok(Some(m));
    }
    #[cfg(not(target_arch = "wasm32"))]
    // a build already under way (polled before the file is looked for: this is asked every frame while it runs)
    if let Some(rx) = &s.vera.describe.keywords.building {
        match rx.try_recv() {
            Ok(Ok(m)) => {
                s.vera.describe.keywords.building = None;
                let m = Arc::new(m);
                adopt(s, m.clone());
                return Ok(Some(m));
            }
            Ok(Err(e)) => {
                s.vera.describe.keywords.building = None;
                s.vera.describe.keywords.error = Some((e.clone(), web_time::Instant::now()));
                return Err(e);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => s.vera.describe.keywords.building = None,
        }
    }
    if let Some(dir) = super::library_dir(s) {
        let path = matrix_path(&dir, &vocab);
        if let Ok(meta) = std::fs::metadata(&path)
            && meta.len() <= MAX_FILE
            && let Ok(bytes) = std::fs::read(&path)
        {
            match Matrix::from_bytes(&bytes, super::lane::MODEL_ID, &vocab, lightcraft_embed::DIM) {
                Some(m) => {
                    let m = Arc::new(m);
                    s.vera.describe.keywords.matrix = Some(m.clone());
                    s.vera.describe.keywords.saved_in = Some(dir);
                    return Ok(Some(m));
                }
                None => log::debug!("vera: {} is not this vocabulary's keyword cache; it will be rebuilt", path.display()),
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if super::lane::state(s) != super::lane::LaneState::Live {
            return Ok(None);
        }
        // a build that failed is tried again after a while, not on every frame
        if let Some((e, at)) = &s.vera.describe.keywords.error
            && at.elapsed() < RETRY_BUILD
        {
            return Err(e.clone());
        }
        let model = super::lane::model(s)?;
        s.vera.describe.keywords.error = None;
        let (tx, rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new().name("lightcraft-vera-keywords".into()).spawn(move || {
            let _ = tx.send(build(&model, &vocab));
        });
        match spawned {
            Ok(_) => s.vera.describe.keywords.building = Some(rx),
            Err(e) => return Err(format!("could not start the keyword thread: {e}")),
        }
        Ok(None)
    }
    #[cfg(target_arch = "wasm32")]
    Ok(None)
}

/// The library's per-keyword calibration (recomputed when the space has moved); `None` under [`CALIBRATE_FROM`]
/// vectors.
pub(crate) fn calibration(s: &mut Session, matrix: &Matrix) -> Option<Arc<Calibration>> {
    let space = s.vera.lane.store.space()?;
    if space.len() < CALIBRATE_FROM {
        return None;
    }
    let k = &s.vera.describe.keywords;
    if let Some(c) = &k.calibration
        && c.vocab_hash == k.vocab.hash()
        && !c.is_stale(space)
    {
        return Some(c.clone());
    }
    let started = web_time::Instant::now();
    let c = Arc::new(calibrate(&k.vocab, matrix, space)?);
    log::debug!("vera: keyword calibration over {} pictures in {} ms", c.sampled, started.elapsed().as_millis());
    s.vera.describe.keywords.calibration = Some(c.clone());
    Some(c)
}

/// What `photo.keywords` says while the vocabulary is being embedded for the first time.
pub(crate) const BUILDING: &str =
    "the keyword vocabulary is being embedded for the first time (a few minutes on a CPU, once; kept beside the library): try again shortly";

/// `photo.keywords {id?, limit?}`: the zero-shot keywords for one photo (read-only: nothing is written to it).
pub(crate) fn photo_keywords(s: &mut Session, p: &Value) -> crate::Result<Value> {
    const C: &str = "photo.keywords";
    use crate::cmd::bad;
    let id = match p.get("id") {
        None | Some(Value::Null) => s.active().ok_or_else(|| bad(C, "no photo"))?,
        Some(v) => {
            let id = v.as_u64().map(lightcraft_catalog::PhotoId).ok_or_else(|| bad(C, "`id` must be a photo id"))?;
            if s.catalog.photo(id).is_none() {
                return Err(bad(C, format!("no photo {}", id.0)));
            }
            id
        }
    };
    let limit = match p.get("limit") {
        None | Some(Value::Null) => MAX_KEYWORDS,
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| bad(C, format!("`limit` must be a whole number from 1 to {MAX_KEYWORDS}")))?;
            usize::try_from(n).unwrap_or(MAX_KEYWORDS).clamp(1, MAX_KEYWORDS)
        }
    };
    super::ensure_open(s);
    let not_ready = |s: &Session| {
        let st = super::lane::state(s);
        if st == super::lane::LaneState::Live { "this photo could not be embedded".to_string() } else { st.reason() }
    };
    let v = match super::lane::vector_of(s, id, true) {
        Ok(Some(v)) => v,
        Ok(None) => return Err(bad(C, format!("keywords need the photo's embedding: {}", not_ready(s)))),
        Err(e) => return Err(bad(C, format!("keywords need the photo's embedding: {e}"))),
    };
    // the vocabulary is embedded once, off this thread (a few minutes on a CPU): until then, a clear error
    let m = match matrix(s) {
        Ok(Some(m)) => m,
        Ok(None) if s.vera.describe.keywords.building() => return Err(bad(C, BUILDING)),
        Ok(None) => return Err(bad(C, format!("keywords need the SigLIP 2 text tower: {}", not_ready(s)))),
        Err(e) => return Err(bad(C, format!("the keyword vocabulary could not be embedded: {e}"))),
    };
    let cal = calibration(s, &m);
    let vocab = s.vera.describe.keywords.vocab.clone();
    let scored = score(&vocab, &m, cal.as_deref(), &v, limit).ok_or_else(|| bad(C, "the photo's embedding does not fit the keyword vectors"))?;
    let mut out = scored.to_json();
    out["id"] = json!(id.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 16;

    fn tiny() -> Vocabulary {
        Vocabulary::parse(
            "[things] multi max=2\nlake | a lake\nrock\ntree\nsand\nsnow\nice\nfog\nsky\ncloud\nfield\n[time] single margin=0.05 prompt=\"at {}\"\ndawn\ndusk\nnoon\n",
        )
        .unwrap()
    }

    fn basis(i: usize) -> Vec<f32> {
        (0..DIM).map(|j| if j == i { 1.0 } else { 0.0 }).collect()
    }

    /// One vector per prompt: term `t` (flat) along axis `t`, both phrasings of `lake` alike.
    fn matrix(v: &Vocabulary) -> Matrix {
        let offsets = term_offsets(v);
        let vectors: Vec<Vec<f32>> = v.prompts().iter().map(|(f, t, _)| basis(offsets[*f] + t)).collect();
        Matrix::from_vectors("m", v, DIM, &vectors).unwrap()
    }

    fn mix(parts: &[(usize, f32)]) -> Vec<f32> {
        let mut v = vec![0.01f32; DIM];
        for (i, w) in parts {
            v[*i] += w;
        }
        super::super::embeddings::unit(&v).unwrap()
    }

    #[test]
    fn the_matrix_round_trips_and_refuses_foreign_files() {
        let v = tiny();
        let m = matrix(&v);
        assert!(m.fits("m", &v));
        let bytes = m.to_bytes();
        assert_eq!(Matrix::from_bytes(&bytes, "m", &v, DIM), Some(m.clone()));
        assert!(Matrix::from_bytes(&bytes, "other", &v, DIM).is_none(), "another model");
        assert!(Matrix::from_bytes(&bytes, "m", &v, DIM * 2).is_none(), "another dimension");
        let other = Vocabulary::parse("[things] multi\nlake\n").unwrap();
        assert!(Matrix::from_bytes(&bytes, "m", &other, DIM).is_none(), "another vocabulary");
        assert!(Matrix::from_bytes(&bytes[..bytes.len() - 3], "m", &v, DIM).is_none(), "cut short");
        assert!(Matrix::from_bytes(b"LCVK", "m", &v, DIM).is_none());
        assert!(Matrix::from_bytes(b"", "m", &v, DIM).is_none());
        let mut bad = bytes.clone();
        bad[4] = 9;
        assert!(Matrix::from_bytes(&bad, "m", &v, DIM).is_none(), "another version");
        assert!(file_name("siglip2", 0xabc).ends_with("-0000000000000abc.bin"));
    }

    #[test]
    fn scores_take_the_best_phrasing_and_apply_the_ceiling_and_facets() {
        let v = tiny();
        let m = matrix(&v);
        // strongly lake and rock, a little tree; dawn clearly over dusk
        let photo = mix(&[(0, 1.0), (1, 0.8), (2, 0.15), (10, 0.7), (11, 0.2)]);
        let s = score(&v, &m, None, &photo, 15).unwrap();
        let words: Vec<&str> = s.keywords.iter().map(|k| k.keyword.as_str()).collect();
        assert_eq!(words, ["lake", "rock", "dawn"], "{s:?}");
        assert!(!s.calibrated && s.ceiling.is_some());
        assert!(s.keywords.iter().all(|k| k.z.is_none()));
        // facet order first: things before time, and by score inside a facet
        assert!(s.keywords[0].cosine > s.keywords[1].cosine);
        // a single-answer facet without a clear winner says nothing
        let unclear = mix(&[(0, 1.0), (1, 0.8), (10, 0.5), (11, 0.49)]);
        let s = score(&v, &m, None, &unclear, 15).unwrap();
        assert!(!s.keywords.iter().any(|k| k.facet == "time"), "{s:?}");
        // the multi facet's max and the global limit
        let many = mix(&[(0, 1.0), (1, 0.9), (2, 0.85), (3, 0.8)]);
        assert_eq!(score(&v, &m, None, &many, 15).unwrap().keywords.len(), 2, "max=2");
        assert_eq!(score(&v, &m, None, &photo, 1).unwrap().keywords.len(), 1);
        // a flat photo (all terms alike) proposes nothing
        let flat = vec![0.25f32; DIM];
        assert!(score(&v, &m, None, &flat, 15).unwrap().keywords.is_empty());
        // a vector of another length is refused
        assert!(score(&v, &m, None, &[1.0; 3], 15).is_none());
    }

    #[test]
    fn calibration_turns_hub_terms_into_z_scores() {
        let v = tiny();
        let m = matrix(&v);
        // a library where every picture leans towards "sky" (a hub), and a few are lakes
        let mut space = lightcraft_vera::Space::new("m", DIM).unwrap();
        for i in 0..250u64 {
            let lake = if i % 10 == 0 { 0.9 } else { 0.0 };
            let jitter = (mix64(i) % 100) as f32 / 1000.0;
            assert!(space.insert(i, &mix(&[(7, 0.8 + jitter), (0, lake), ((i % 5) as usize + 2, 0.2)])));
        }
        let cal = calibrate(&v, &m, &space).unwrap();
        assert_eq!((cal.sampled, cal.n_at_compute, cal.terms.len()), (250, 250, v.terms()));
        assert!(!cal.is_stale(&space));
        // a photo near both sky and lake: raw cosines rank the hub first, z-scores the lake
        let photo = mix(&[(7, 0.85), (0, 0.6)]);
        let raw = score(&v, &m, None, &photo, 15).unwrap();
        let raw_top = raw.keywords.iter().max_by(|a, b| a.cosine.total_cmp(&b.cosine)).unwrap();
        assert_eq!(raw_top.keyword, "sky");
        let z = score(&v, &m, Some(&cal), &photo, 15).unwrap();
        assert!(z.calibrated);
        let z_top = z.keywords.iter().max_by(|a, b| a.z.unwrap().total_cmp(&b.z.unwrap())).unwrap();
        assert_eq!(z_top.keyword, "lake", "{z:?}");
        // a small space is not calibrated
        let mut small = lightcraft_vera::Space::new("m", DIM).unwrap();
        small.insert(1, &photo);
        assert!(calibrate(&v, &m, &small).is_none());
    }
}
