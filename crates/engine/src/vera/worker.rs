//! The vector lane's heavy part, off the session thread: render each photo to a small upright sRGB picture, squash it
//! to the model's 256 × 256 input and embed a batch at a time.
//!
//! One thread, sharing the loaded [`Model`] with the session (it is `Sync`; the session embeds queries and the odd
//! single picture with it). The session thread only prepares the jobs (which needs the catalog: [`prepare`]) and
//! later takes in the vectors, so the window never waits for the model. Rendering is the face scan's path
//! (`Session::preview_job` with default settings, uncropped; a raw's embedded preview first when it is large enough),
//! and the render and the model pass both run on the face scan's pool for the pace, never on rayon's global pool,
//! which the loupe and exports use.
//!
//! Memory: the model (about 420 MB with the image tower, once), one render at a time at [`RENDER_EDGE`], and one batch
//! of [`MAX_BATCH`] 256 × 256 pictures (12 bytes a pixel: 12.6 MB).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use lightcraft_catalog::PhotoId;
use lightcraft_develop::DevelopSettings;
use lightcraft_embed::{INPUT, MAX_BATCH, Model};
use lightcraft_raster::Rgba8;

use crate::Session;
use crate::faces_worker::{Pace as ScanPace, scan_pool};
use crate::media::{PreviewLoader, RenderJob};

/// Long edge photos are rendered at before being squashed to [`INPUT`]: twice the input, so the squash averages a few
/// pixels into each (the thumbnail-level source serves it, the cheapest decode).
pub(crate) const RENDER_EDGE: usize = 512;

/// One picture to embed.
pub(crate) struct Job {
    /// The session's library epoch when this was prepared (results for another library are dropped).
    pub epoch: u64,
    pub key: String,
    pub photo: PhotoId,
    pub render: RenderJob,
    /// A raw's embedded preview, tried before the render job (much faster than decoding the raw).
    pub preview: Option<(String, PreviewLoader)>,
    pub pace: ScanPace,
}

/// Why a job has no vector.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Failure {
    /// The photo could not be rendered: skipped for the session.
    Picture(String),
    /// The model could not run (a damaged checkpoint, out of memory): the lane stops with this error; the photo is
    /// not skipped.
    Model(String),
}

/// A finished job.
pub(crate) struct Done {
    pub epoch: u64,
    pub key: String,
    pub photo: PhotoId,
    pub result: Result<Vec<f32>, Failure>,
}

/// The render job for photo `id`'s picture (upright, uncropped, default settings) and its embedded preview; `None` for
/// a photo that cannot be rendered.
pub(crate) fn prepare(s: &mut Session, id: PhotoId) -> Option<(RenderJob, Option<(String, PreviewLoader)>)> {
    let photo = s.catalog.photo(id)?.clone();
    let render = s.preview_job(id, RENDER_EDGE, RENDER_EDGE, false, &DevelopSettings::default())?;
    Some((render, s.scan_preview_of(&photo)))
}

/// The model input for a job: its embedded preview when that is at least [`INPUT`] on its long edge, else the render.
pub(crate) fn picture(render: RenderJob, preview: Option<(String, PreviewLoader)>) -> Result<Vec<[f32; 3]>, String> {
    let from_preview = preview.and_then(|(path, load)| load(&path, RENDER_EDGE)).filter(|img| img.width.max(img.height) >= INPUT);
    let image = match from_preview {
        Some(img) => img,
        None => render.run().rendered?.image,
    };
    squash(&image, INPUT).ok_or_else(|| "the picture is empty".to_string())
}

/// For each of `n` output cells along an axis of `len` source pixels, the source pixels it covers and how much of each
/// (the weights of one cell sum to 1).
fn coverage(len: usize, n: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = len as f64 / n as f64;
    (0..n)
        .map(|i| {
            let (a, b) = (i as f64 * scale, (i + 1) as f64 * scale);
            let mut cell = Vec::new();
            let mut x = a.floor() as usize;
            while (x as f64) < b && x < len {
                let w = (b.min(x as f64 + 1.0) - a.max(x as f64)) / scale;
                if w > 0.0 {
                    cell.push((x, w as f32));
                }
                x += 1;
            }
            cell
        })
        .collect()
}

/// `img` resized to exactly `side × side` by area averaging, without keeping its aspect (as the SigLIP processor
/// does), as display-encoded RGB in 0..1. `None` for an empty picture.
pub(crate) fn squash(img: &Rgba8, side: usize) -> Option<Vec<[f32; 3]>> {
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 || side == 0 || img.data.len() < w.checked_mul(h)? {
        return None;
    }
    let (cols, rows) = (coverage(w, side), coverage(h, side));
    // horizontal pass: h rows of `side`
    let mut mid = vec![[0f32; 3]; h.checked_mul(side)?];
    for y in 0..h {
        let src = img.data.get(y * w..(y + 1) * w)?;
        let dst = mid.get_mut(y * side..(y + 1) * side)?;
        for (out, cell) in dst.iter_mut().zip(&cols) {
            for &(x, wt) in cell {
                let p = src.get(x)?;
                for c in 0..3 {
                    out[c] += f32::from(p[c]) / 255.0 * wt;
                }
            }
        }
    }
    // vertical pass
    let mut out = vec![[0f32; 3]; side * side];
    for (oy, cell) in rows.iter().enumerate() {
        let dst = out.get_mut(oy * side..(oy + 1) * side)?;
        for &(y, wt) in cell {
            let src = mid.get(y * side..(y + 1) * side)?;
            for (o, p) in dst.iter_mut().zip(src) {
                for c in 0..3 {
                    o[c] += p[c] * wt;
                }
            }
        }
    }
    for p in &mut out {
        for c in p.iter_mut() {
            *c = c.clamp(0.0, 1.0);
        }
    }
    Some(out)
}

/// Run `f` on the face scan's pool for `pace` (so its parallel parts stay off rayon's global pool).
fn on_pool<R: Send>(pace: ScanPace, f: impl FnOnce() -> R + Send) -> R {
    match scan_pool(pace) {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// Embed one batch of pictures; a panic or an error is the model's failure.
fn embed_batch(model: &Model, pace: ScanPace, pictures: &[&[[f32; 3]]]) -> Result<Vec<Vec<f32>>, String> {
    if pictures.is_empty() {
        return Ok(Vec::new());
    }
    let run = catch_unwind(AssertUnwindSafe(|| on_pool(pace, || model.embed_images(pictures))));
    match run {
        Ok(Ok(v)) if v.len() == pictures.len() => Ok(v),
        Ok(Ok(v)) => Err(format!("the model returned {} embeddings for {} pictures", v.len(), pictures.len())),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("the model stopped unexpectedly".into()),
    }
}

/// The thread that embeds what it is given, [`MAX_BATCH`] pictures per model pass.
pub(crate) struct Worker {
    jobs: Sender<Job>,
    done: Receiver<Done>,
}

impl Worker {
    pub fn start(model: Arc<Model>) -> Option<Worker> {
        let (jobs, rx) = channel::<Job>();
        let (tx, done) = channel::<Done>();
        let spawned = std::thread::Builder::new().name("lightcraft-vera-embed".into()).spawn(move || {
            // ends when the session (the only sender) drops the worker
            while let Ok(first) = rx.recv() {
                let mut batch: Vec<(Job, Result<Vec<[f32; 3]>, String>)> = Vec::with_capacity(MAX_BATCH);
                let mut next = Some(first);
                let mut pace = ScanPace::Light;
                // render what has arrived, taking more as it comes, until a batch is full or nothing is waiting
                while let Some(mut job) = next.take() {
                    pace = job.pace;
                    let render = job.render.clone();
                    let preview = job.preview.take();
                    let pic = catch_unwind(AssertUnwindSafe(|| on_pool(pace, || picture(render, preview))))
                        .unwrap_or_else(|_| Err("rendering stopped unexpectedly".into()));
                    batch.push((job, pic));
                    if batch.len() >= MAX_BATCH {
                        break;
                    }
                    next = rx.try_recv().ok();
                }
                let ok: Vec<&[[f32; 3]]> = batch.iter().filter_map(|(_, p)| p.as_ref().ok().map(Vec::as_slice)).collect();
                let mut vectors = match embed_batch(&model, pace, &ok) {
                    Ok(v) => Ok(v.into_iter()),
                    Err(e) => Err(e),
                };
                for (job, pic) in batch {
                    let result = match (pic, &mut vectors) {
                        (Err(e), _) => Err(Failure::Picture(e)),
                        (Ok(_), Ok(it)) => it.next().ok_or_else(|| Failure::Model("no embedding".into())),
                        (Ok(_), Err(e)) => Err(Failure::Model(e.clone())),
                    };
                    if tx.send(Done { epoch: job.epoch, key: job.key, photo: job.photo, result }).is_err() {
                        return;
                    }
                }
            }
        });
        spawned.ok().map(|_| Worker { jobs, done })
    }

    pub fn submit(&self, job: Job) -> bool {
        self.jobs.send(job).is_ok()
    }

    /// Everything finished since the last call.
    pub fn finished(&self) -> Vec<Done> {
        self.done.try_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_weights_sum_to_one_and_stay_inside() {
        for (len, n) in [(512, 256), (300, 256), (256, 256), (100, 256), (1, 256), (2048, 256), (7, 3)] {
            let cells = coverage(len, n);
            assert_eq!(cells.len(), n);
            for c in &cells {
                let sum: f32 = c.iter().map(|(_, w)| w).sum();
                assert!((sum - 1.0).abs() < 1e-4, "{len}→{n}: {sum}");
                assert!(c.iter().all(|(x, _)| *x < len));
            }
        }
    }

    #[test]
    fn squash_averages_areas_and_ignores_aspect() {
        // a 4 × 2 picture: left half black, right half white → 2 × 2 keeps the halves; 1 × 1 is mid-grey
        let img = Rgba8::from_fn(4, 2, |x, _| if x < 2 { [0, 0, 0, 255] } else { [255, 255, 255, 255] });
        let out = squash(&img, 2).unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!((out[0], out[1], out[2], out[3]), ([0.0; 3], [1.0; 3], [0.0; 3], [1.0; 3]));
        let one = squash(&img, 1).unwrap();
        assert!((one[0][0] - 0.5).abs() < 1e-6);
        // a non-square photo still fills the square, and an upscale is defined
        let wide = Rgba8::from_fn(600, 200, |x, y| [(x % 256) as u8, (y % 256) as u8, 7, 255]);
        let sq = squash(&wide, INPUT).unwrap();
        assert_eq!(sq.len(), INPUT * INPUT);
        assert!(sq.iter().all(|p| p.iter().all(|c| (0.0..=1.0).contains(c))));
        let tiny = Rgba8::from_fn(3, 5, |_, _| [128, 64, 32, 255]);
        let up = squash(&tiny, 16).unwrap();
        assert!(up.iter().all(|p| (p[0] - 128.0 / 255.0).abs() < 1e-5));
        // empty or inconsistent pictures are refused, not read past
        let empty = Rgba8 { width: 0, height: 0, data: Vec::new() };
        assert!(squash(&empty, 4).is_none());
        let short = Rgba8 { width: 10, height: 10, data: vec![[0; 4]; 5] };
        assert!(squash(&short, 4).is_none());
    }
}
