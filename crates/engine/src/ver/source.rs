//! Where a Record's `canonical_visual` vector comes from: an [`EmbeddingSource`] that names its space and embeds a
//! Canonical Buffer. VER 1.0 §6: the canonical visual embedding is "computed from the Canonical Buffer and nothing
//! else", so a source is handed the buffer itself, never a render or a preview.
//!
//! [`LaneSource`] is the production one: the vector lane's SigLIP 2 model (the same weights the lane searches with),
//! fed by [`super::space::model_input`]. Tests use a deterministic source in a test space of their own.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use lightcraft_ver::cpnp::CanonicalBuffer;
use serde_json::Value;

use super::VerError;

/// What a source says about the space its vectors live in.
#[derive(Clone, Debug)]
pub(crate) struct SpaceInfo {
    pub space_id: String,
    pub dim: usize,
    pub dtype: String,
    /// The full space descriptor (`spaces[]` entry).
    pub descriptor: Value,
    /// `embeddings[].producer`.
    pub producer: String,
    /// `embeddings[].compute` (VER 1.1 §9).
    pub compute: Value,
}

/// A way to embed a Canonical Buffer. `Send + Sync`: exports embed on their own threads.
pub(crate) trait EmbeddingSource: Send + Sync {
    /// The space, or why there is none ([`VerError::NeedsModel`] when the model is not installed). Cheap: asked
    /// before any picture is decoded.
    fn space(&self) -> Result<SpaceInfo, VerError>;
    /// The unit vector of `buf` in [`EmbeddingSource::space`].
    fn embed(&self, buf: &CanonicalBuffer) -> Result<Vec<f32>, VerError>;
}

/// A source that cannot embed, saying why (the model is missing, the web build).
pub(crate) struct Unavailable(pub VerError);

impl EmbeddingSource for Unavailable {
    fn space(&self) -> Result<SpaceInfo, VerError> {
        Err(self.0.clone())
    }
    fn embed(&self, _: &CanonicalBuffer) -> Result<Vec<f32>, VerError> {
        Err(self.0.clone())
    }
}

/// The vector lane's model as a VER embedding source.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct LaneSource {
    pub model: Arc<lightcraft_embed::Model>,
    pub authority: String,
    /// SHA-256 of the weights file, computed on first use ([`super::space::weights_sha256`]) and shared.
    pub weights: Arc<std::sync::OnceLock<Result<String, String>>>,
    pub weights_path: std::path::PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl LaneSource {
    fn device_class(&self) -> &'static str {
        let d = self.model.device();
        if d.is_cpu() { "cpu" } else { "gpu" }
    }

    fn device_name(&self) -> &'static str {
        let d = self.model.device();
        if d.is_metal() {
            "metal"
        } else if d.is_cuda() {
            "cuda"
        } else {
            "cpu"
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl EmbeddingSource for LaneSource {
    fn space(&self) -> Result<SpaceInfo, VerError> {
        let weights = self.weights.get_or_init(|| super::space::weights_sha256(&self.weights_path)).clone().map_err(VerError::Model)?;
        Ok(SpaceInfo {
            space_id: super::space::space_id(&self.authority),
            dim: super::space::DIM,
            dtype: super::space::DTYPE.into(),
            descriptor: super::space::descriptor(&self.authority, &weights),
            producer: format!("lightcraft-embed/{} (candle {CANDLE_VERSION}, {}, fp32)", env!("CARGO_PKG_VERSION"), self.device_name()),
            compute: serde_json::json!({"runtime": "candle", "runtime_version": CANDLE_VERSION, "precision": "fp32", "device_class": self.device_class()}),
        })
    }

    fn embed(&self, buf: &CanonicalBuffer) -> Result<Vec<f32>, VerError> {
        let input = super::space::model_input(&buf.rgb, buf.width as usize, buf.height as usize)
            .ok_or_else(|| VerError::Model("the Canonical Buffer is empty".into()))?;
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.model.embed_image(&input)));
        match run {
            Ok(Ok(v)) if v.len() == super::space::DIM && v.iter().all(|x| x.is_finite()) => Ok(v),
            Ok(Ok(v)) => Err(VerError::Model(format!("the model returned an unusable embedding ({} numbers)", v.len()))),
            Ok(Err(e)) => Err(VerError::Model(e.to_string())),
            Err(_) => Err(VerError::Model("the image tower stopped unexpectedly".into())),
        }
    }
}

/// The candle version the model runs on (the workspace pins it exactly).
#[cfg(not(target_arch = "wasm32"))]
const CANDLE_VERSION: &str = "0.9.2";

/// A deterministic source for tests: 8 numbers from the Canonical Buffer's mean colour and its gradients, in the
/// test space `org.verstandard.space.test-visual/0.1` (fp16, no model). Two renditions of one picture land close;
/// unrelated pictures do not.
#[cfg(test)]
pub(crate) struct TestSource;

#[cfg(test)]
pub(crate) const TEST_SPACE: &str = "org.verstandard.space.test-visual/0.1";

#[cfg(test)]
impl EmbeddingSource for TestSource {
    fn space(&self) -> Result<SpaceInfo, VerError> {
        Ok(SpaceInfo {
            space_id: TEST_SPACE.into(),
            dim: 8,
            dtype: "fp16".into(),
            descriptor: serde_json::json!({
                "space_id": TEST_SPACE,
                "kind": "visual",
                "model": {"family": "test", "architecture": "mean-colour-and-gradients", "checkpoint": "lightcraft-test-visual", "weights_sha256": "0".repeat(64)},
                "dim": 8,
                "dtype": "fp16",
                "normalization": "l2",
                "metric": "cosine",
                "preprocessing": {"resize": "none", "interpolation": "none", "antialias": false, "crop": "none", "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "channel_order": "rgb"},
                "provenance": {"publisher": "verstandard.org", "published_at": "2026-10-11T00:00:00Z", "license": "Apache-2.0"},
            }),
            producer: "lightcraft-test/0".into(),
            compute: serde_json::json!({"runtime": "lightcraft-test", "runtime_version": "0", "precision": "fp32", "device_class": "cpu"}),
        })
    }

    fn embed(&self, buf: &CanonicalBuffer) -> Result<Vec<f32>, VerError> {
        let (w, h) = (buf.width as usize, buf.height as usize);
        let px = |x: usize, y: usize, c: usize| f32::from(buf.rgb.get((y * w + x) * 3 + c).copied().unwrap_or(0)) / 255.0;
        let mut v = [0f32; 8];
        let n = (w * h).max(1) as f32;
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    v[c] += px(x, y, c) / n;
                }
                let l = px(x, y, 1);
                if x + 1 < w {
                    v[3] += (px(x + 1, y, 1) - l).abs() / n;
                }
                if y + 1 < h {
                    v[4] += (px(x, y + 1, 1) - l).abs() / n;
                }
                // which quarter is brighter: coarse layout
                v[5 + usize::from(x * 2 >= w) + usize::from(y * 2 >= h).min(1)] += l / n;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        Ok(v.iter().map(|x| x / norm).collect())
    }
}
