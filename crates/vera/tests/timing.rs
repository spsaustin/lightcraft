//! Per-step timing of the perceptual digests on a 24 MP (6000×4000) RGB8 picture. Ignored by
//! default: `cargo test -p lightcraft-vera --test timing -- --ignored --nocapture`.

use lightcraft_vera::digest::{self, Luma};
use std::time::Instant;

fn picture(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    let mut s = 0x2545_f491_4f6c_dd1du64;
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v.extend_from_slice(&[((x * 255) / w) as u8, ((y * 255) / h) as u8, ((((x / 40) ^ (y / 40)) * 37) as u8).wrapping_add((s & 15) as u8)]);
        }
    }
    v
}

#[test]
#[ignore = "timing; run with --ignored --nocapture"]
fn digests_at_24_mp() {
    let (w, h) = (6000usize, 4000usize);
    let rgb = picture(w, h);
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;

    // The luma-plane API (one f32 plane in 0..=1), as `lightcraft-ver` used it before.
    let t = Instant::now();
    let f: Vec<[f32; 3]> = rgb.chunks_exact(3).map(|p| [f32::from(p[0]) / 255.0, f32::from(p[1]) / 255.0, f32::from(p[2]) / 255.0]).collect();
    let luma = digest::luma_from_rgb(w, h, &f);
    let t_luma = ms(t);
    let plane = Luma::new(w, h, &luma).expect("plane");
    let t = Instant::now();
    let _ = digest::pdq(&plane).expect("pdq");
    let t_pdq = ms(t);
    let t = Instant::now();
    let _ = digest::phash(&plane).expect("phash");
    let t_phash = ms(t);
    println!("luma-plane API: luma {t_luma:.0} ms, pdq {t_pdq:.0} ms, phash {t_phash:.0} ms");
    timing_rgb8(w, h, &rgb);
}

fn timing_rgb8(w: usize, h: usize, rgb: &[u8]) {
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let planes = digest::Planes::from_rgb8(w, h, rgb).expect("planes");
    let t_planes = ms(t);
    let t = Instant::now();
    let _ = digest::pdq_planes(&planes).expect("pdq");
    let t_pdq = ms(t);
    let t = Instant::now();
    let _ = digest::phash_planes(&planes).expect("phash");
    let t_phash = ms(t);
    println!("reference RGB8 API: planes {t_planes:.0} ms, pdq + quality {t_pdq:.0} ms, phash {t_phash:.0} ms");
}
