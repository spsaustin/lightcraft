//! SigLIP 2 (Google, 2025) image and text embeddings in pure Rust on candle: the model behind
//! VERA's vector lane (`docs/vera.md`, the choice in `docs/vera-models.md`).
//!
//! One image tower ([`Model::embed_image`]) turns a 256 × 256 picture into a 768-d unit vector; one
//! text tower ([`Model::embed_text`]) turns a query into a vector in the same space. Cosine between
//! them is what *find by meaning*, *similar picture* and the embedding tier of *same picture* read.
//!
//! On macOS the model runs on the GPU through Metal; elsewhere on the CPU. The weights are not part
//! of LightCraft: they are the `google/siglip2-base-patch32-256` checkpoint (Apache-2.0), downloaded
//! only when the user asks for it and accepts its licence ([`fetch`]), verified before use.
//!
//! Ported from the Hugging Face `transformers` implementation (Apache-2.0); see `NOTICE`. This crate
//! is therefore licensed under the Apache License 2.0 only (not the MIT option of the rest of
//! LightCraft). No UI dependencies (L3).
//!
//! Modified work (Apache License 2.0, §4(b)): ported by the LightCraft contributors in 2026 from the
//! Python/PyTorch SigLIP code of Hugging Face Transformers (`src/transformers/models/siglip`),
//! Copyright The HuggingFace Team and Google LLC; translated to Rust on candle and restructured.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(not(target_arch = "wasm32"))]
pub mod fetch;
#[cfg(not(target_arch = "wasm32"))]
mod nn;
#[cfg(not(target_arch = "wasm32"))]
mod siglip;
pub mod tokenizer;
#[cfg(not(target_arch = "wasm32"))]
mod weights;

use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("SigLIP 2 model: {0}")]
    Model(String),
    #[error("SigLIP 2 model files not found in {0} (expected model.safetensors, config.json, tokenizer.model)")]
    Missing(PathBuf),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[cfg(not(target_arch = "wasm32"))]
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// The model this crate implements, as the fetch manifest and the embedding space name it.
pub const MODEL_ID: &str = "siglip2-base-patch32-256";
/// Side of the square model input.
pub const INPUT: usize = 256;
/// Numbers per embedding.
pub const DIM: usize = 768;
/// Tokens per text input (shorter texts are padded, longer ones cut).
pub const CONTEXT: usize = 64;
/// The checkpoint's weights file.
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// The checkpoint's architecture description (read for its sizes, checked against this crate's constants).
pub const CONFIG_FILE: &str = "config.json";
/// The SentencePiece model the text tower's tokenizer reads.
pub const TOKENIZER_FILE: &str = "tokenizer.model";

/// Whether `dir` holds a usable checkpoint.
pub fn is_model_dir(dir: &Path) -> bool {
    dir.join(WEIGHTS_FILE).is_file() && dir.join(CONFIG_FILE).is_file() && dir.join(TOKENIZER_FILE).is_file()
}

#[cfg(not(target_arch = "wasm32"))]
pub use siglip::{MAX_BATCH, Model};

#[cfg(not(target_arch = "wasm32"))]
/// The best device here: Metal on macOS (when a GPU is available), else the CPU.
pub fn best_device() -> candle_core::Device {
    #[cfg(target_os = "macos")]
    if let Ok(d) = candle_core::Device::new_metal(0) {
        return d;
    }
    candle_core::Device::Cpu
}
