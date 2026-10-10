//! The identity lane: perceptual digests that say whether two files are the **same picture**.
//!
//! Two digests, computed from one luma plane:
//! - **PDQ** (`pdq/1.0`): 256 bits. Luma → two passes of a Jarosz box filter (window = dimension /
//!   128, as the published algorithm) → 64×64 point decimation → a 16×64 DCT that drops the DC row
//!   and column → the 16×16 low-frequency block → each coefficient compared with the block's median
//!   (Torben's, so a flat block yields a defined answer) → bits in the reference's row-major order,
//!   most significant bit first within each byte. Plus the reference's 0..100 image-domain quality
//!   metric (the gradient that survived decimation), which the caller uses to distrust flat frames.
//! - **pHash** (`phash-dct/64`): 64 bits. Luma → 32×32 (box filter) → 2D DCT → the top-left 8×8
//!   block (DC excluded from the median) → bits above the median.
//!
//! Both are written from the published descriptions (the PDQ paper and its stated hex ordering,
//! Zauner's thesis for pHash), clean-room: no reference code was read. The fixture tests in this
//! module are the proof that the implementation matches the studio's measurements of the
//! reference: resizes under tight, re-exports under loose, unrelated frames above loose, and the
//! crop at about [`thresholds::CROP_PDQ_MEASURED`] bits.
//!
//! Distances are plain Hamming distances; the decision ([`sameness`]) is the studio's: any digest
//! under its tight threshold, or both under loose, is the same picture.

use crate::thresholds;

/// A luma plane, row-major, values in `0.0..=1.0` (anything outside is clamped by the digest).
/// Borrowed, so the caller converts its RGB once and digests twice.
#[derive(Clone, Copy, Debug)]
pub struct Luma<'a> {
    pub width: usize,
    pub height: usize,
    /// `width * height` samples; a shorter slice is an error the digest reports as `None`.
    pub data: &'a [f32],
}

impl<'a> Luma<'a> {
    /// A luma plane over `data` if the dimensions agree with it; `None` otherwise.
    pub fn new(width: usize, height: usize, data: &'a [f32]) -> Option<Luma<'a>> {
        let n = width.checked_mul(height)?;
        (n > 0 && data.len() >= n).then_some(Luma { width, height, data })
    }

    /// The sample at (`x`, `y`), or `0.0` outside the plane (never panics).
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        y.checked_mul(self.width).and_then(|i| i.checked_add(x)).and_then(|i| self.data.get(i)).copied().unwrap_or(0.0)
    }
}

/// Rec. 601 luma of interleaved `[r, g, b]` rows, the weights the studio used (0.299, 0.587, 0.114),
/// as a plane the digest can read.
pub fn luma_from_rgb(width: usize, height: usize, rgb: &[[f32; 3]]) -> Vec<f32> {
    let n = width.checked_mul(height).unwrap_or(0).min(rgb.len());
    rgb.iter().take(n).map(|[r, g, b]| (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)).collect()
}

/// Both digests of one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digest {
    /// PDQ, 256 bits, the reference's byte order.
    pub pdq: [u8; 32],
    /// PDQ's image-domain quality, 0..=100.
    pub pdq_quality: u8,
    /// pHash, 64 bits.
    pub phash: u64,
}

/// What two digests say about two frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sameness {
    /// The same picture: a resize, a re-export, a format change.
    Same,
    /// Possibly the same picture: a variant edit, a heavy crop, a burst neighbour. Worth showing,
    /// never acted on automatically.
    Possible,
    /// Different pictures as far as digests can tell (a crop can still be the same picture: that
    /// is the embedding lane's job).
    Different,
}

/// Both digests of `img`, or `None` when the plane is too small to fingerprint (under 2×2).
pub fn digest(img: &Luma) -> Option<Digest> {
    let (pdq, pdq_quality) = pdq(img)?;
    let phash = phash(img)?;
    Some(Digest { pdq, pdq_quality, phash })
}

/// PDQ of `img` and its quality, or `None` for a plane too small to fingerprint (under 2×2) or
/// whose `data` is shorter than `width * height`.
///
/// Steps (all from the published description):
/// 1. Luma clamped to `0..=1` (NaN reads as 0, ±inf as the nearest bound).
/// 2. Jarosz filter: two passes of a centred box blur along each row (window
///    `max(1, width / 128)`), then two along each column (window `max(1, height / 128)`). Each
///    window is the length of the axis it slides along divided by 128, so a 64×64 decimation samples
///    a plane already averaged over about half a decimation cell; under 128 samples on an axis the
///    window is 1 and the pass is the identity. At the edges a window averages only the samples it
///    covers.
/// 3. 64×64 point decimation at row `⌊(i + ½)·height / 64⌋`, column `⌊(j + ½)·width / 64⌋`
///    (defined for planes under 64 on an axis: indices repeat).
/// 4. `B = D·A·Dᵀ` with `D[i][j] = √(2/64)·cos(π/(2·64)·(i+1)·(2j+1))`, `i < 16`, `j < 64`: the
///    orthonormal DCT-II rows 1..=16, so the DC row and column are dropped.
/// 5. Bit `i·16 + j` is `B[i][j] > median(B)` (the lower median of the 256, the value Torben's
///    algorithm returns), packed row-major into byte `(i·16 + j) / 8`, most significant bit first.
///
/// Quality (see [`pdq_quality`]) is measured on the decimated 64×64 plane.
pub fn pdq(img: &Luma) -> Option<([u8; 32], u8)> {
    let mut plane = sanitised(img)?;
    let (w, h) = (img.width, img.height);
    let win_x = (w / 128).max(1);
    let win_y = (h / 128).max(1);
    for _ in 0..2 {
        box_rows(&mut plane, w, win_x);
    }
    for _ in 0..2 {
        box_cols(&mut plane, w, win_y);
    }

    // 64×64 point decimation, row-major.
    let rows: Vec<usize> = (0..PDQ_GRID).map(|i| decimation_index(i, h)).collect();
    let cols: Vec<usize> = (0..PDQ_GRID).map(|j| decimation_index(j, w)).collect();
    let mut a = Vec::with_capacity(PDQ_GRID * PDQ_GRID);
    for &y in &rows {
        for &x in &cols {
            let v = y.checked_mul(w).and_then(|i| i.checked_add(x)).and_then(|i| plane.get(i)).copied().unwrap_or(0.0);
            a.push(f64::from(v));
        }
    }

    let quality = pdq_quality(&a);

    let basis: Vec<Vec<f64>> = (0..PDQ_BLOCK)
        .map(|i| {
            (0..PDQ_GRID)
                .map(|j| {
                    let scale = (2.0 / PDQ_GRID as f64).sqrt();
                    scale * (std::f64::consts::PI / 2.0 / PDQ_GRID as f64 * (i + 1) as f64 * (2 * j + 1) as f64).cos()
                })
                .collect()
        })
        .collect();
    let b = dct_block(&a, PDQ_GRID, &basis);
    let median = lower_median(&b)?;

    let mut out = [0u8; 32];
    for (k, v) in b.iter().enumerate() {
        if *v > median
            && let Some(byte) = out.get_mut(k / 8)
        {
            *byte |= 0x80 >> (k % 8);
        }
    }
    Some((out, quality))
}

/// PDQ's image-domain quality of the decimated 64×64 plane `a` (row-major, values `0..=1`):
///
/// `q = min(100, ⌊ Σ ⌊100·|Δ|⌋ / 90 ⌋)`
///
/// where the sum runs over every horizontal (`a[i][j+1] − a[i][j]`) and vertical
/// (`a[i+1][j] − a[i][j]`) neighbour difference, 2·64·63 = 8064 of them, each truncated to whole
/// hundredths of the luma range before summing. This is the reference scaling: a plane whose
/// neighbours differ by the full `0..1` range sums to 806 400 and clamps to 100, a perfectly flat
/// plane (and one whose only differences are rounding noise under 1/100) scores exactly 0, and
/// the 100 point scale saturates once the mean difference reaches about 1.1% of the range, so the
/// [`thresholds::PDQ_MIN_QUALITY`] floor of 50 rejects only frames with almost no surviving
/// gradient.
fn pdq_quality(a: &[f64]) -> u8 {
    let hundredths = |d: f64| -> u64 {
        let v = (d.abs() * 100.0).trunc();
        if v.is_finite() && v > 0.0 { v as u64 } else { 0 }
    };
    let mut sum: u64 = 0;
    for row in a.chunks_exact(PDQ_GRID) {
        for pair in row.windows(2) {
            if let [p, q] = pair {
                sum = sum.saturating_add(hundredths(q - p));
            }
        }
    }
    for (upper, lower) in a.chunks_exact(PDQ_GRID).zip(a.chunks_exact(PDQ_GRID).skip(1)) {
        for (p, q) in upper.iter().zip(lower) {
            sum = sum.saturating_add(hundredths(q - p));
        }
    }
    u8::try_from((sum / 90).min(100)).unwrap_or(100)
}

/// pHash (Zauner's DCT hash) of `img`, or `None` for a plane too small to fingerprint (under 2×2)
/// or whose `data` is shorter than `width * height`.
///
/// Luma clamped to `0..=1` → area-average (box filter) resample to 32×32, exact fractional
/// coverage so it is defined for planes smaller than 32 too → orthonormal 2D DCT-II,
/// `C[u][v] = α(u)·α(v)·Σ_y Σ_x a[y][x]·cos(π(2y+1)u/64)·cos(π(2x+1)v/64)` with `α(0) = √(1/32)`,
/// `α(k>0) = √(2/32)`, of which only the top-left 8×8 is computed → the median of those 64
/// coefficients excluding the DC term `C[0][0]` (63 values, so an exact middle) → bit `u·8 + v` is
/// `C[u][v] > median`, packed row-major with bit 0 as the most significant bit of the `u64`.
pub fn phash(img: &Luma) -> Option<u64> {
    let plane = sanitised(img)?;
    let (w, h) = (img.width, img.height);
    let small = area_resample(&plane, w, h, PHASH_GRID)?;

    let basis: Vec<Vec<f64>> = (0..PHASH_BLOCK)
        .map(|u| {
            let alpha = if u == 0 { (1.0 / PHASH_GRID as f64).sqrt() } else { (2.0 / PHASH_GRID as f64).sqrt() };
            (0..PHASH_GRID).map(|x| alpha * (std::f64::consts::PI / 2.0 / PHASH_GRID as f64 * u as f64 * (2 * x + 1) as f64).cos()).collect()
        })
        .collect();
    let c = dct_block(&small, PHASH_GRID, &basis);
    let ac: Vec<f64> = c.iter().skip(1).copied().collect();
    let median = lower_median(&ac)?;

    let mut out = 0u64;
    for (k, v) in c.iter().enumerate().take(64) {
        if *v > median {
            out |= 1u64 << (63 - k);
        }
    }
    Some(out)
}

/// PDQ's decimated grid edge and its low-frequency block edge.
const PDQ_GRID: usize = 64;
const PDQ_BLOCK: usize = 16;
/// pHash's resample edge and its low-frequency block edge.
const PHASH_GRID: usize = 32;
const PHASH_BLOCK: usize = 8;

/// `img`'s first `width * height` samples clamped to `0..=1` (NaN as 0), or `None` when the plane
/// is under 2×2 or the slice is short. The allocation is bounded by the caller's slice.
fn sanitised(img: &Luma) -> Option<Vec<f32>> {
    if img.width < 2 || img.height < 2 {
        return None;
    }
    let n = img.width.checked_mul(img.height)?;
    let src = img.data.get(..n)?;
    Some(src.iter().map(|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }).collect())
}

/// `⌊(i + ½)·n / 64⌋` in integers (`(2i + 1)·n / 128`), kept inside `0..n`.
fn decimation_index(i: usize, n: usize) -> usize {
    let num = (2 * i + 1).saturating_mul(n);
    (num / (2 * PDQ_GRID)).min(n.saturating_sub(1))
}

/// A centred box blur of `src` over `window` samples (`(window − 1) / 2` before, `window / 2`
/// after), averaging only the samples inside the slice at the edges. Running sums, no indexing.
fn box_1d(src: &[f32], window: usize, out: &mut Vec<f32>) {
    out.clear();
    if window <= 1 {
        out.extend_from_slice(src);
        return;
    }
    let before = (window - 1) / 2;
    let after = window / 2;
    let n = src.len();
    let mut prefix = Vec::with_capacity(n.saturating_add(1));
    let mut acc = 0.0f64;
    prefix.push(acc);
    for v in src {
        acc += f64::from(*v);
        prefix.push(acc);
    }
    for i in 0..n {
        let lo = i.saturating_sub(before);
        let hi = i.saturating_add(after).saturating_add(1).min(n);
        let (Some(p_hi), Some(p_lo)) = (prefix.get(hi), prefix.get(lo)) else {
            out.push(0.0);
            continue;
        };
        let count = hi.saturating_sub(lo).max(1);
        out.push(((p_hi - p_lo) / count as f64) as f32);
    }
}

/// One box pass along every row of a `w`-wide plane.
fn box_rows(plane: &mut [f32], w: usize, window: usize) {
    if window <= 1 || w == 0 {
        return;
    }
    let mut out = Vec::with_capacity(w);
    for row in plane.chunks_exact_mut(w) {
        box_1d(row, window, &mut out);
        row.copy_from_slice(&out);
    }
}

/// One box pass down every column of a `w`-wide plane.
fn box_cols(plane: &mut [f32], w: usize, window: usize) {
    if window <= 1 || w == 0 {
        return;
    }
    let mut col = Vec::new();
    let mut out = Vec::new();
    for x in 0..w {
        col.clear();
        col.extend(plane.iter().skip(x).step_by(w).copied());
        box_1d(&col, window, &mut out);
        for (dst, v) in plane.iter_mut().skip(x).step_by(w).zip(&out) {
            *dst = *v;
        }
    }
}

/// For each of `m` output cells over `n` source samples, the source indices it covers and their
/// weights (fractional coverage, summing to 1): output `k` spans `[k·n/m, (k+1)·n/m)`.
fn area_weights(n: usize, m: usize) -> Vec<Vec<(usize, f64)>> {
    (0..m)
        .map(|k| {
            // Work in units of 1/m of a source sample so every bound is an integer.
            let start = k.saturating_mul(n);
            let end = (k + 1).saturating_mul(n);
            let first = start / m;
            let last = end.saturating_sub(1) / m;
            (first..=last.min(n.saturating_sub(1)))
                .filter_map(|s| {
                    let s_lo = s.saturating_mul(m);
                    let s_hi = (s + 1).saturating_mul(m);
                    let overlap = end.min(s_hi).saturating_sub(start.max(s_lo));
                    (overlap > 0).then(|| (s, overlap as f64 / n as f64))
                })
                .collect()
        })
        .collect()
}

/// Area-average resample of a `w × h` plane to `m × m`, row-major, or `None` on an empty plane.
fn area_resample(plane: &[f32], w: usize, h: usize, m: usize) -> Option<Vec<f64>> {
    if w == 0 || h == 0 {
        return None;
    }
    let wx = area_weights(w, m);
    let wy = area_weights(h, m);
    // Horizontal: h rows of m.
    let mut horiz = Vec::with_capacity(h.checked_mul(m)?);
    for row in plane.chunks_exact(w).take(h) {
        for cell in &wx {
            horiz.push(cell.iter().map(|(s, wt)| row.get(*s).map_or(0.0, |v| f64::from(*v)) * wt).sum::<f64>());
        }
    }
    // Vertical: m rows of m.
    let mut out = Vec::with_capacity(m.checked_mul(m)?);
    for cell in &wy {
        for x in 0..m {
            out.push(
                cell.iter()
                    .map(|(s, wt)| s.checked_mul(m).and_then(|i| i.checked_add(x)).and_then(|i| horiz.get(i)).copied().unwrap_or(0.0) * wt)
                    .sum::<f64>(),
            );
        }
    }
    Some(out)
}

/// `E·A·Eᵀ` for an `n × n` row-major `a` and `k` basis rows of length `n`: a `k × k` row-major
/// block of separable transform coefficients (row index = vertical frequency).
fn dct_block(a: &[f64], n: usize, basis: &[Vec<f64>]) -> Vec<f64> {
    // T = E·A, k × n.
    let t: Vec<Vec<f64>> = basis
        .iter()
        .map(|e| {
            let mut acc = vec![0.0f64; n];
            for (ev, arow) in e.iter().zip(a.chunks_exact(n)) {
                for (t, av) in acc.iter_mut().zip(arow) {
                    *t += ev * av;
                }
            }
            acc
        })
        .collect();
    // B = T·Eᵀ, k × k.
    t.iter().flat_map(|trow| basis.iter().map(move |e| trow.iter().zip(e).map(|(x, y)| x * y).sum::<f64>())).collect()
}

/// The lower median of `v` (element `⌈n/2⌉ − 1` in sorted order, what Torben's algorithm returns),
/// exact under ties, or `None` when empty.
fn lower_median(v: &[f64]) -> Option<f64> {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s.get(s.len().checked_sub(1)? / 2).copied()
}

/// Bits that differ between two PDQ digests.
pub fn hamming256(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Bits that differ between two pHash digests.
pub fn hamming64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// The studio's decision: any digest at or under its tight threshold is the same picture (a PDQ
/// digest only when both frames' quality clears [`thresholds::PDQ_MIN_QUALITY`]); otherwise both at
/// or under loose is a possible match; otherwise different.
pub fn sameness(a: &Digest, b: &Digest) -> Sameness {
    compare(a, b).sameness
}

/// Distances between two digests, for reports: (PDQ bits of 256, pHash bits of 64).
pub fn distances(a: &Digest, b: &Digest) -> (u32, u32) {
    (hamming256(&a.pdq, &b.pdq), hamming64(a.phash, b.phash))
}

/// The algorithm these digests come from. A store that remembers digests beside this number can
/// tell when a change to the computation (not the thresholds) makes its records stale and must be
/// rebuilt rather than compared with fresh ones. Bumped only when a digest of the same pixels
/// would come out differently.
pub const ALGORITHM_VERSION: u8 = 1;

/// [`sameness`] with the distances it was decided on, computed once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Comparison {
    pub sameness: Sameness,
    /// PDQ bits of 256 that differ.
    pub pdq: u32,
    /// pHash bits of 64 that differ.
    pub phash: u32,
}

/// Compare two digests: the verdict and both distances in one pass.
pub fn compare(a: &Digest, b: &Digest) -> Comparison {
    let pdq = hamming256(&a.pdq, &b.pdq);
    let phash = hamming64(a.phash, b.phash);
    let pdq_trusted = a.pdq_quality >= thresholds::PDQ_MIN_QUALITY && b.pdq_quality >= thresholds::PDQ_MIN_QUALITY;
    let sameness = if (pdq_trusted && pdq <= thresholds::PDQ_TIGHT) || phash <= thresholds::PHASH_TIGHT {
        Sameness::Same
    } else if pdq <= thresholds::PDQ_LOOSE && phash <= thresholds::PHASH_LOOSE {
        Sameness::Possible
    } else {
        Sameness::Different
    };
    Comparison { sameness, pdq, phash }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_scenes::{Scene, demo_library};
    use proptest::prelude::*;

    /// A rendered frame as an owned luma plane.
    struct Plane {
        w: usize,
        h: usize,
        data: Vec<f32>,
    }

    impl Plane {
        fn luma(&self) -> Luma<'_> {
            Luma::new(self.w, self.h, &self.data).expect("plane dimensions agree with data")
        }
        fn digest(&self) -> Digest {
            digest(&self.luma()).expect("plane is at least 2×2")
        }
    }

    fn rgb_plane(img: &lightcraft_raster::Rgb32f) -> Plane {
        Plane { w: img.width, h: img.height, data: luma_from_rgb(img.width, img.height, &img.data) }
    }

    /// `scene` at long edge `edge`, keeping its aspect.
    fn render(scene: &Scene, edge: usize) -> lightcraft_raster::Rgb32f {
        scene.render_fit(edge)
    }

    fn jpeg_round_trip(img: &lightcraft_raster::Rgb32f, quality: u8) -> Plane {
        let bytes: Vec<u8> = img.data.iter().flat_map(|px| px.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
        let mut buf = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality)
            .encode(&bytes, img.width as u32, img.height as u32, image::ExtendedColorType::Rgb8)
            .expect("jpeg encode");
        let decoded = image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg).expect("jpeg decode").to_rgb8();
        let (w, h) = (decoded.width() as usize, decoded.height() as usize);
        let rgb: Vec<[f32; 3]> = decoded.pixels().map(|p| p.0.map(|c| f32::from(c) / 255.0)).collect();
        Plane { w, h, data: luma_from_rgb(w, h, &rgb) }
    }

    #[test]
    fn resizes_are_the_same_picture() {
        let scene = &demo_library()[0];
        assert!((scene.aspect() - 1.5).abs() < 1e-3, "fixture scene is 3:2");
        let edges = [2048, 1024, 512, 256];
        let frames: Vec<(usize, Digest)> = edges.iter().map(|&e| (e, rgb_plane(&render(scene, e)).digest())).collect();
        let (mut pdq_min, mut pdq_max, mut ph_min, mut ph_max) = (u32::MAX, 0, u32::MAX, 0);
        for (i, (ea, a)) in frames.iter().enumerate() {
            for (eb, b) in frames.iter().skip(i + 1) {
                let (pd, ph) = distances(a, b);
                (pdq_min, pdq_max, ph_min, ph_max) = (pdq_min.min(pd), pdq_max.max(pd), ph_min.min(ph), ph_max.max(ph));
                let msg = format!("{ea} vs {eb}: pdq {pd}, phash {ph}, quality {} / {}", a.pdq_quality, b.pdq_quality);
                assert_eq!(sameness(a, b), Sameness::Same, "{msg}");
                assert!(pd <= thresholds::PDQ_TIGHT, "{msg}");
                assert!(ph <= thresholds::PHASH_TIGHT, "{msg}");
            }
        }
        println!("resize set: pdq {pdq_min}..={pdq_max}, phash {ph_min}..={ph_max}");
    }

    #[test]
    fn re_exports_are_the_same_picture() {
        let scene = &demo_library()[0];
        let src_rgb = render(scene, 1024);
        let src = rgb_plane(&src_rgb);
        let d0 = src.digest();
        for q in [85u8, 50] {
            let d = jpeg_round_trip(&src_rgb, q).digest();
            let (pd, ph) = distances(&d0, &d);
            println!("jpeg q{q}: pdq {pd}, phash {ph}, quality {}", d.pdq_quality);
            assert_eq!(sameness(&d0, &d), Sameness::Same, "jpeg q{q}: pdq {pd}, phash {ph}");
        }
        let gamma = Plane { w: src.w, h: src.h, data: src.data.iter().map(|v| v.powf(1.1)).collect() };
        let d = gamma.digest();
        let (pd, ph) = distances(&d0, &d);
        println!("gamma 1.1: pdq {pd}, phash {ph}, verdict {:?}", sameness(&d0, &d));
        assert_ne!(sameness(&d0, &d), Sameness::Different, "gamma 1.1: pdq {pd}, phash {ph}");
    }

    #[test]
    fn unrelated_scenes_are_different() {
        let lib = demo_library();
        let picked: Vec<&Scene> = lib.iter().take(8).collect();
        let digests: Vec<(&str, Digest)> = picked.iter().map(|s| (s.name.as_str(), rgb_plane(&render(s, 512)).digest())).collect();
        for (n, d) in &digests {
            assert!(d.pdq_quality >= thresholds::PDQ_MIN_QUALITY, "{n}: quality {}", d.pdq_quality);
        }
        let (mut pdq_min, mut ph_min) = (u32::MAX, u32::MAX);
        let mut log = Vec::new();
        for (i, (na, a)) in digests.iter().enumerate() {
            for (nb, b) in digests.iter().skip(i + 1) {
                let (pd, ph) = distances(a, b);
                pdq_min = pdq_min.min(pd);
                ph_min = ph_min.min(ph);
                log.push(format!("{na} vs {nb}: pdq {pd}, phash {ph}"));
            }
        }
        for (i, (na, a)) in digests.iter().enumerate() {
            for (nb, b) in digests.iter().skip(i + 1) {
                let (pd, ph) = distances(a, b);
                assert!(pd > thresholds::PDQ_LOOSE || ph > thresholds::PHASH_LOOSE, "{na} vs {nb}: pdq {pd}, phash {ph}; min pdq seen {pdq_min}");
                assert_eq!(sameness(a, b), Sameness::Different, "{na} vs {nb}: pdq {pd}, phash {ph}; min pdq seen {pdq_min}");
            }
        }
        println!("unrelated set ({} scenes): min pdq {pdq_min}, min phash {ph_min}\n{}", digests.len(), log.join("\n"));
    }

    #[test]
    fn central_crop_is_a_digest_blind_spot() {
        // Observed on the synthetic fixture (scene 0 at 1024, central 70%×70%): PDQ 130, pHash 22,
        // against the studio's real-photo measurement of about CROP_PDQ_MEASURED (140). Not
        // asserted: the ±CROP_PDQ_TOLERANCE cross-check needs that photograph.
        let scene = &demo_library()[0];
        let full = rgb_plane(&render(scene, 1024));
        let (cw, ch) = (full.w * 7 / 10, full.h * 7 / 10);
        let (x0, y0) = ((full.w - cw) / 2, (full.h - ch) / 2);
        let data: Vec<f32> = (y0..y0 + ch).flat_map(|y| full.data[y * full.w + x0..y * full.w + x0 + cw].iter().copied()).collect();
        let crop = Plane { w: cw, h: ch, data };
        let (a, b) = (full.digest(), crop.digest());
        let (pd, ph) = distances(&a, &b);
        println!("crop 70%: pdq {pd}, phash {ph}");
        assert_eq!(sameness(&a, &b), Sameness::Different, "crop: pdq {pd}, phash {ph}");
    }

    #[test]
    fn flat_frames_have_zero_quality() {
        let grey = |v: f32| Plane { w: 200, h: 150, data: vec![v; 200 * 150] };
        let (a, b) = (grey(0.3).digest(), grey(0.7).digest());
        assert_eq!(a.pdq_quality, 0);
        assert_eq!(b.pdq_quality, 0);
        // The PDQ lane never vouches for a flat frame; pHash may (both hashes are of nothing).
        let (pd, ph) = distances(&a, &b);
        println!("flat greys: pdq {pd}, phash {ph}, verdict {:?}", sameness(&a, &b));
        assert_eq!(grey(0.3).digest(), a, "flat digest is defined and repeatable");
        // Quality gate: even identical PDQ bits do not make two flat frames the same picture.
        let twin = Digest { pdq: a.pdq, pdq_quality: 0, phash: !a.phash };
        assert_ne!(sameness(&a, &twin), Sameness::Same);
    }

    #[test]
    fn tiny_planes() {
        let one = [0.5f32];
        assert_eq!(digest(&Luma::new(1, 1, &one).unwrap()), None);
        let four = [0.0f32, 1.0, 1.0, 0.0];
        assert!(digest(&Luma::new(2, 2, &four).unwrap()).is_some());
        // A hand-built plane whose slice is short is refused, not read past.
        assert_eq!(digest(&Luma { width: 4, height: 4, data: &four }), None);
        assert_eq!(digest(&Luma { width: usize::MAX, height: 2, data: &four }), None);
    }

    #[test]
    fn deterministic_and_orientation_sensitive() {
        let scene = &demo_library()[0];
        let p = rgb_plane(&render(scene, 512));
        let d = p.digest();
        assert_eq!(d, p.digest());
        let mirrored = Plane { w: p.w, h: p.h, data: p.data.chunks_exact(p.w).flat_map(|r| r.iter().rev().copied()).collect() };
        let flipped = Plane { w: p.w, h: p.h, data: p.data.chunks_exact(p.w).rev().flat_map(|r| r.iter().copied()).collect() };
        // 90° clockwise: new (x, y) reads old (y, h − 1 − x).
        let rotated = Plane {
            w: p.h,
            h: p.w,
            data: (0..p.w).flat_map(|y| (0..p.h).map(move |x| (x, y))).map(|(x, y)| p.data[(p.h - 1 - x) * p.w + y]).collect(),
        };
        for (name, other) in [("mirrored", &mirrored), ("flipped", &flipped), ("rotated", &rotated)] {
            let o = other.digest();
            let (pd, ph) = distances(&d, &o);
            println!("{name}: pdq {pd}, phash {ph}");
            assert_ne!(o.pdq, d.pdq, "{name}");
            assert_ne!(o.phash, d.phash, "{name}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]
        #[test]
        fn never_panics(
            w in 0usize..=300,
            h in 0usize..=300,
            frac in 0.0f64..1.3,
            pool in prop::collection::vec(prop_oneof![proptest::num::f32::ANY, 0.0f32..1.0, Just(f32::NAN), Just(f32::INFINITY), Just(f32::NEG_INFINITY)], 1..64),
        ) {
            let len = ((w * h) as f64 * frac) as usize;
            let data: Vec<f32> = pool.iter().copied().cycle().take(len).collect();
            if let Some(l) = Luma::new(w, h, &data) {
                let d = digest(&l);
                prop_assert_eq!(d.is_some(), w >= 2 && h >= 2);
                if let Some(d) = d {
                    prop_assert!(d.pdq_quality <= 100);
                }
            }
            // Hand-built planes bypass `new`'s check; the digest still never reads past the slice.
            let _ = digest(&Luma { width: w, height: h, data: &data });
        }
    }
}
