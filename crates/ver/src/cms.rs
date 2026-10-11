//! ICC → sRGB for matrix/TRC profiles, evaluated in `f64` and rounded once.
//!
//! The profile pins littleCMS 2.19 (relative colorimetric, black point compensation, the lcms
//! built-in sRGB as destination) and measures its sRGB → sRGB transform as exact identity on
//! 8-bit input. `moxcms`'s 8-bit transform does not reproduce that: over the full 2²⁴ RGB cube
//! with the corpus's own embedded profile (lcms's "sRGB built-in"), it moves 15 351 466 colours
//! (91.5 %; 15 282 346 by one level, 69 120 by two), the grey axis included (1 → 0, 2 → 1, …),
//! because its 8-bit path truncates. Its `f32` transform with correct rounding still moves
//! 347 807 (2.1 %): its tone curves are table-interpolated.
//!
//! For matrix/TRC RGB profiles this module therefore evaluates the transform the way littleCMS
//! itself evaluates an 8-bit matrix/TRC → matrix/TRC transform (see [`rgb_matrix_shaper`]),
//! against lcms's built-in sRGB (Rec. 709 primaries, D65 white, Bradford-adapted to lcms's D50;
//! its parametric curve and analytic inverse). For a source colorimetrically equal to sRGB (up
//! to the s15Fixed16 quantisation of its tags) every input maps to itself, which is what lcms
//! measured. Gray TRC profiles follow lcms's resampled 33-node evaluation (see [`gray_trc`]). Black point compensation is the identity for
//! these profiles when the source TRCs map 0 to 0 (both black points are then zero), which is
//! every common RGB working space; a source TRC with a non-zero floor is the one matrix/TRC case
//! where lcms's BPC would move black and this path does not.
//!
//! Every other profile (LUT-based, CMYK, Lab) goes through `moxcms`'s `f32` transform with the
//! same single rounding. Neither path is pinned by corpus release 1 (`PROFILE.md` §6: the one
//! ICC entry is sRGB), so outputs for non-sRGB profiles are this implementation's own and are
//! not claimed to match lcms bit for bit.

use moxcms::{ColorProfile, DataColorSpace, ToneReprCurve, Xyzd};

type M3 = [[f64; 3]; 3];

fn mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

fn apply(m: &M3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn inverse(m: &M3) -> Option<M3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv],
        [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv],
        [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv],
    ])
}

/// lcms's D50 (`cmsD50_XYZ`).
const D50: [f64; 3] = [0.9642, 1.0, 0.8249];

/// `_cmsBuildRGB2XYZtransferMatrix` for Rec. 709 primaries and D65, adapted to D50 by Bradford
/// (`_cmsAdaptMatrixToD50`): the lcms built-in sRGB's RGB → PCS matrix.
fn srgb_to_pcs() -> Option<M3> {
    let (xn, yn) = (0.3127, 0.3290);
    let p = [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)];
    let prim: M3 = [[p[0].0, p[1].0, p[2].0], [p[0].1, p[1].1, p[2].1], [1.0 - p[0].0 - p[0].1, 1.0 - p[1].0 - p[1].1, 1.0 - p[2].0 - p[2].1]];
    let white = [xn / yn, 1.0, (1.0 - xn - yn) / yn];
    let coef = apply(&inverse(&prim)?, white);
    let mut m = prim;
    for row in &mut m {
        for (j, v) in row.iter_mut().enumerate() {
            *v *= coef[j];
        }
    }
    let bradford: M3 = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];
    let src = apply(&bradford, white);
    let dst = apply(&bradford, D50);
    let cone: M3 = [[dst[0] / src[0], 0.0, 0.0], [0.0, dst[1] / src[1], 0.0], [0.0, 0.0, dst[2] / src[2]]];
    let adapt = mul(&inverse(&bradford)?, &mul(&cone, &bradford));
    Some(mul(&adapt, &m))
}

fn eval_curve(c: &ToneReprCurve, x: f64) -> Option<f64> {
    let x = x.clamp(0.0, 1.0);
    let y = match c {
        ToneReprCurve::Lut(t) => match t.len() {
            0 => x,
            1 => x.powf(f64::from(t[0]) / 256.0),
            n => {
                let pos = x * (n - 1) as f64;
                let i = (pos.floor() as usize).min(n - 2);
                let f = pos - i as f64;
                let a = f64::from(*t.get(i)?) / 65535.0;
                let b = f64::from(*t.get(i + 1)?) / 65535.0;
                a + (b - a) * f
            }
        },
        ToneReprCurve::Parametric(p) => {
            let p: Vec<f64> = p.iter().map(|&v| f64::from(v)).collect();
            let pw = |base: f64, g: f64| if base > 0.0 { base.powf(g) } else { 0.0 };
            match p.as_slice() {
                [g] => pw(x, *g),
                [g, a, b] => {
                    if x >= -b / a {
                        pw(a * x + b, *g)
                    } else {
                        0.0
                    }
                }
                [g, a, b, c] => {
                    if x >= -b / a {
                        pw(a * x + b, *g) + c
                    } else {
                        *c
                    }
                }
                [g, a, b, c, d] => {
                    if x >= *d {
                        pw(a * x + b, *g)
                    } else {
                        c * x
                    }
                }
                [g, a, b, c, d, e, f] => {
                    if x >= *d {
                        pw(a * x + b, *g) + e
                    } else {
                        c * x + f
                    }
                }
                _ => return None,
            }
        }
    };
    y.is_finite().then_some(y)
}

fn has_lut(p: &ColorProfile) -> bool {
    p.lut_a_to_b_colorimetric.is_some() || p.lut_a_to_b_perceptual.is_some()
}

fn col(x: &Xyzd) -> [f64; 3] {
    [x.x, x.y, x.z]
}

/// lcms's `cmsEvalToneCurveFloat` for one curve at an `f32` input: parametric curves in `f64`,
/// sampled curves through the 16-bit table interpolation (`LinLerp1D`).
fn lcms_curve_f32(c: &ToneReprCurve, r: f32) -> Option<f32> {
    match c {
        ToneReprCurve::Lut(t) if t.len() >= 2 => {
            let w = quick_saturate_word(f64::from(r) * 65535.0);
            Some(f32::from(lin_lerp_16(t, w)) / 65535.0)
        }
        _ => eval_curve(c, f64::from(r)).map(|v| v as f32),
    }
}

/// `_cmsQuickSaturateWord`.
fn quick_saturate_word(d: f64) -> u16 {
    let d = d + 0.5;
    if d <= 0.0 {
        0
    } else if d >= 65535.0 {
        65535
    } else {
        d.floor() as u16
    }
}

/// `LinLerp1D` over a 16-bit table (`Domain = n − 1`).
fn lin_lerp_16(t: &[u16], v: u16) -> u16 {
    let domain = (t.len() - 1) as i64;
    if v == 0xFFFF || domain == 0 {
        return t.last().copied().unwrap_or(0);
    }
    let val3 = domain * i64::from(v);
    let val3 = val3 + ((val3 + 0x7FFF) / 0xFFFF); // _cmsToFixedDomain
    let cell0 = (val3 >> 16) as usize;
    let rest = val3 & 0xFFFF;
    let y0 = i64::from(t.get(cell0).copied().unwrap_or(0));
    let y1 = i64::from(t.get(cell0 + 1).copied().unwrap_or(y0 as u16));
    let dif = (y1 - y0) * rest + 0x8000;
    ((dif >> 16) + y0).clamp(0, 65535) as u16
}

/// The lcms built-in sRGB's reverse curve (parametric type −4) at `r`.
fn srgb_reverse(r: f64) -> f64 {
    let p: [f64; 5] = [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045];
    let e = p[1] * p[4] + p[2];
    let disc = if e < 0.0 { 0.0 } else { e.powf(p[0]) };
    if r >= disc {
        let e = if r > 0.0 { r.powf(1.0 / p[0]) } else { 0.0 };
        (e - p[2]) / p[1]
    } else {
        r / p[3]
    }
}

fn from_16_to_8(w: u16) -> u8 {
    ((u32::from(w) * 65281 + 8388608) >> 24) as u8
}

/// RGB samples through a matrix/TRC profile, or `None` when the profile is not one.
///
/// This follows littleCMS's own 8-bit evaluation of a matrix/TRC → matrix/TRC transform:
/// when the joined matrix is the identity to within 1/65535 (`_cmsMAT3isIdentity`) the matrices
/// drop out and the curves are joined into a 4096-point table (`OptimizeByJoiningCurves`), which
/// is the identity when every entry is within 0x0f of linear; otherwise `OptimizeMatrixShaper`:
/// 256-entry input shapers in 1.14 fixed point, the joined matrix in 1.14 fixed point, and a
/// 16385-entry output shaper rounded to 8 bits.
pub(crate) fn rgb_matrix_shaper(p: &ColorProfile, rgb: &[u8]) -> Option<Vec<u8>> {
    if p.color_space != DataColorSpace::Rgb || p.pcs != DataColorSpace::Xyz || has_lut(p) {
        return None;
    }
    let curves = [p.red_trc.as_ref()?, p.green_trc.as_ref()?, p.blue_trc.as_ref()?];
    let (r, g, b) = (col(&p.red_colorant), col(&p.green_colorant), col(&p.blue_colorant));
    let src: M3 = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
    let m = mul(&inverse(&srgb_to_pcs()?)?, &src);
    let identity = (0..3).all(|i| (0..3).all(|j| (m[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1.0 / 65535.0));
    let mut out = Vec::with_capacity(rgb.len());
    if identity {
        // OptimizeByJoiningCurves: 4096 points of reverse(trc(x)), 16-bit.
        let mut tables: Vec<Vec<u16>> = Vec::with_capacity(3);
        for c in curves {
            let mut t = Vec::with_capacity(4096);
            for i in 0..4096 {
                let x = (i as f64 / 4095.0) as f32;
                let y = lcms_curve_f32(c, x)?;
                let o = srgb_reverse(f64::from(y)) as f32;
                t.push(quick_saturate_word(f64::from(o) * 65535.0));
            }
            tables.push(t);
        }
        let linear = tables.iter().all(|t| {
            t.iter().enumerate().all(|(i, &v)| {
                let q = quick_saturate_word(i as f64 * 65535.0 / 4095.0);
                (i32::from(v) - i32::from(q)).abs() <= 0x0f
            })
        });
        if linear {
            return Some(rgb.to_vec());
        }
        let luts: Vec<[u8; 256]> = tables
            .iter()
            .map(|t| {
                let mut l = [0u8; 256];
                for (i, slot) in l.iter_mut().enumerate() {
                    *slot = from_16_to_8(lin_lerp_16(t, (i as u16) * 257));
                }
                l
            })
            .collect();
        for px in rgb.chunks_exact(3) {
            out.extend_from_slice(&[luts[0][usize::from(px[0])], luts[1][usize::from(px[1])], luts[2][usize::from(px[2])]]);
        }
        return Some(out);
    }
    // OptimizeMatrixShaper.
    let fix14 = |x: f64| (x * 16384.0 + 0.5).floor() as i64;
    let mut shaper1 = [[0i64; 256]; 3];
    for (ci, c) in curves.iter().enumerate() {
        for i in 0..256 {
            let y = lcms_curve_f32(c, (i as f64 / 255.0) as f32)?;
            shaper1[ci][i] = if f64::from(y) < 131072.0 { fix14(f64::from(y)) } else { 0x7fff_ffff };
        }
    }
    let mut shaper2 = vec![0u8; 16385];
    for (i, slot) in shaper2.iter_mut().enumerate() {
        let r = (i as f64 / 16384.0) as f32;
        let v = (srgb_reverse(f64::from(r)) as f32).clamp(0.0, 1.0);
        *slot = from_16_to_8(quick_saturate_word(f64::from(v) * 65535.0));
    }
    let mf: [[i64; 3]; 3] = [
        [fix14(m[0][0]), fix14(m[0][1]), fix14(m[0][2])],
        [fix14(m[1][0]), fix14(m[1][1]), fix14(m[1][2])],
        [fix14(m[2][0]), fix14(m[2][1]), fix14(m[2][2])],
    ];
    for px in rgb.chunks_exact(3) {
        let v = [shaper1[0][usize::from(px[0])], shaper1[1][usize::from(px[1])], shaper1[2][usize::from(px[2])]];
        for row in &mf {
            let l = (row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + 0x2000) >> 14;
            out.push(shaper2.get(l.clamp(0, 16384) as usize).copied().unwrap_or(0));
        }
    }
    Some(out)
}

/// Gray samples through a gray TRC profile, or `None`.
///
/// littleCMS evaluates an 8-bit gray → RGB transform through `OptimizeByResampling`: the
/// pipeline (gray TRC → Y on the D50 white → the sRGB matrix inverse → the sRGB reverse curve)
/// sampled at 33 nodes into a 16-bit table, read back by 1-D linear interpolation of the
/// 16-bit-expanded input, then rounded to 8 bits. Every output channel carries the same value,
/// because D50 maps to sRGB white.
pub(crate) fn gray_trc(p: &ColorProfile, gray: &[u8]) -> Option<Vec<u8>> {
    if p.color_space != DataColorSpace::Gray || has_lut(p) {
        return None;
    }
    let c = p.gray_trc.as_ref()?;
    const NODES: usize = 33;
    let mut table = Vec::with_capacity(NODES);
    for i in 0..NODES {
        let q = quick_saturate_word(i as f64 * 65535.0 / (NODES - 1) as f64);
        let y = lcms_curve_f32(c, (f64::from(q) / 65535.0) as f32)?;
        let v = (srgb_reverse(f64::from(y).clamp(0.0, 1.0)) as f32).clamp(0.0, 1.0);
        table.push(quick_saturate_word(f64::from(v) * 65535.0));
    }
    let mut lut = [0u8; 256];
    for (v, slot) in lut.iter_mut().enumerate() {
        *slot = from_16_to_8(lin_lerp_16(&table, (v as u16) * 257));
    }
    Some(gray.iter().flat_map(|&g| [lut[usize::from(g)]; 3]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_matrix_is_lcms_shaped() {
        let m = srgb_to_pcs().expect("matrix");
        // White (1,1,1) maps to D50.
        let w = apply(&m, [1.0, 1.0, 1.0]);
        for (a, b) in w.iter().zip(D50) {
            assert!((a - b).abs() < 1e-9, "{w:?}");
        }
        // lcms's red colorant, as its built-in sRGB writes rXYZ (s15Fixed16 0x6fa0 0x38f5 0x0390).
        assert!((m[0][0] - f64::from(0x6fa0) / 65536.0).abs() < 1.0 / 65536.0);
        assert!((m[1][0] - f64::from(0x38f5) / 65536.0).abs() < 1.0 / 65536.0);
        assert!((m[2][0] - f64::from(0x0390) / 65536.0).abs() < 1.0 / 65536.0);
    }
}
