//! Making fingerprints ([`super::digests::Fingerprint`]): over the full-size CPNP-1 Canonical Buffer for a file with a
//! CPNP-1 path, provisionally from the thumbnail-level picture otherwise; and the thread that makes them in the
//! background, so a full-size decode never stalls the frame that pumps.
//!
//! The pump hands the worker at most the pace's number of pictures per call and keeps at most [`MAX_IN_FLIGHT`] out at
//! once (one being decoded, one waiting), so its memory is one Canonical Buffer (3 bytes a pixel: 72 MB for 24 MP) and
//! one file's bytes at a time. A demo scene has no file and is fingerprinted on the spot, as before.

use lightcraft_catalog::PhotoId;
use lightcraft_raster::Rgb32f;
use lightcraft_vera::Digest;

use super::digests::Fingerprint;
use crate::media::SourceRef;

/// Most pictures handed to the worker and not back yet.
pub(crate) const MAX_IN_FLIGHT: usize = 2;
/// The sRGB transfer function (IEC 61966-2-1), linear light → encoded.
fn srgb_encode(c: f32) -> f32 {
    let c = if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 };
    if c <= 0.003_130_8 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Provisional digests of a decoded (linear-light) picture: its Rec. 601 luma after the sRGB transfer function, the
/// domain of the Canonical Buffer the canonical digests are computed in, so the two compare. `None` for a picture too
/// small to fingerprint.
pub(crate) fn digest_linear(img: &Rgb32f) -> Option<Digest> {
    let q = |v: f32| (srgb_encode(v) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    let rgb8: Vec<u8> = img.data.iter().flat_map(|p| [q(p[0]), q(p[1]), q(p[2])]).collect();
    lightcraft_vera::digest_rgb8(img.width, img.height, &rgb8)
}

/// The canonical fingerprint of a file's bytes, or why it has none (no CPNP-1 path, undecodable).
pub(crate) fn canonical(bytes: &[u8]) -> Result<Fingerprint, String> {
    crate::ver::cpnp_format(bytes).map_err(|e| e.to_string())?;
    let (_buf, id, quality) = lightcraft_ver::hashing::identity_with_quality(bytes).map_err(|e| e.to_string())?;
    Fingerprint::from_identity(&id, Some(quality)).ok_or_else(|| "the identity values are not well formed".into())
}

/// What one picture is fingerprinted from: its file (when it has one) and its thumbnail-level source.
#[derive(Clone)]
pub(crate) struct Input {
    pub path: Option<String>,
    pub read: Option<crate::merge::ByteReader>,
    pub thumb: SourceRef,
}

/// Fingerprint one picture: canonically when its file has a CPNP-1 path (and this build canonicalises: not the browser,
/// where a full-size decode would stall the page), else provisionally from its thumbnail-level source, with the file's
/// content hash when it was read.
pub(crate) fn make(input: &Input) -> Result<Fingerprint, String> {
    let mut content = None;
    if let Some(path) = &input.path {
        let bytes = match &input.read {
            Some(r) => r(path),
            None => std::fs::read(path).map_err(|e| format!("{path}: {e}")),
        };
        match bytes {
            Ok(b) if !cfg!(target_arch = "wasm32") => match canonical(&b) {
                Ok(fp) => return Ok(fp),
                Err(e) => {
                    log::debug!("vera: {path}: provisional digests ({e})");
                    content = crate::ver::util::unhex(&crate::ver::util::sha256_hex(&b)).and_then(|v| v.try_into().ok());
                }
            },
            Ok(_) => {}
            // an offline original: its smart preview may still serve the thumbnail
            Err(e) => log::debug!("vera: {path}: {e}"),
        }
    }
    let img = input.thumb.load()?;
    let digest = digest_linear(&img).ok_or("too small to fingerprint")?;
    Ok(Fingerprint { digest, pixel: None, content })
}

/// One picture for the worker.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) struct Job {
    pub key: String,
    pub photos: usize,
    /// The photos that show the picture, in the order to try.
    pub inputs: Vec<(PhotoId, Input)>,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) struct Done {
    pub key: String,
    pub photos: usize,
    pub result: Result<Fingerprint, String>,
}

/// The fingerprinting thread (native builds).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct Worker {
    jobs: std::sync::mpsc::Sender<Job>,
    done: std::sync::mpsc::Receiver<Done>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Worker {
    pub fn start() -> Option<Worker> {
        let (jobs, rx) = std::sync::mpsc::channel::<Job>();
        let (tx, done) = std::sync::mpsc::channel::<Done>();
        let spawned = std::thread::Builder::new().name("lightcraft-vera-digest".into()).spawn(move || {
            while let Ok(job) = rx.recv() {
                let mut result = Err("no photo to fingerprint".to_string());
                for (id, input) in &job.inputs {
                    let r = crate::guard::catch(&format!("fingerprinting photo {}", id.0), || make(input)).and_then(|r| r);
                    match r {
                        Ok(fp) => {
                            result = Ok(fp);
                            break;
                        }
                        Err(e) => result = Err(e),
                    }
                }
                if tx.send(Done { key: job.key, photos: job.photos, result }).is_err() {
                    return;
                }
            }
        });
        spawned.ok().map(|_| Worker { jobs, done })
    }

    pub fn submit(&self, job: Job) -> bool {
        self.jobs.send(job).is_ok()
    }

    pub fn finished(&self) -> Vec<Done> {
        self.done.try_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_encoding_and_quality() {
        assert_eq!(srgb_encode(0.0), 0.0);
        assert!((srgb_encode(1.0) - 1.0).abs() < 1e-6);
        assert!((srgb_encode(0.214) - 0.5).abs() < 0.01);
        assert_eq!(srgb_encode(f32::NAN), 0.0);
        assert!(canonical(b"not a picture").is_err());
    }
}
