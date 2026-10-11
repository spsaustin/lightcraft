//! Downloading the SigLIP 2 model, only when the user asked for it. The downloader (mirrors,
//! resuming, checks, the pure-Rust HTTPS client) is the `lightcraft-fetch` crate; this module says
//! what SigLIP 2 needs and where it may come from. Mirrors the shape of `lightcraft_segment::fetch`.
//!
//! The weights are never part of LightCraft (see docs/vera-models.md); the user downloads them, after
//! accepting [`LICENSE_NOTICE`], from [`DEFAULT_MIRRORS`] or their own list.

pub use lightcraft_fetch::*;

use std::path::Path;

/// The `google/siglip2-base-patch32-256` files [`crate::Model::load`] needs, every one pinned by
/// size and SHA-256 (all verified by downloading the revision in [`DEFAULT_MIRRORS`]).
pub const SIGLIP2_FILES: &[FileSpec] = &[
    FileSpec {
        name: crate::CONFIG_FILE,
        size: Some(298),
        sha256: Some("52deee7a2d6fb2e73c45720d9c4056f0e4dc582ef84524aac229c6d5aa348dee"),
        max: 298,
    },
    FileSpec {
        name: crate::TOKENIZER_FILE,
        size: Some(4_241_003),
        sha256: Some("61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"),
        max: 4_241_003,
    },
    FileSpec {
        name: crate::WEIGHTS_FILE,
        size: Some(SIGLIP2_WEIGHTS_SIZE),
        sha256: Some("7d241bb3becad218f211f480487f491df4f8c0a472ecf7afdec5615815a301f1"),
        max: SIGLIP2_WEIGHTS_SIZE,
    },
];

/// Size of `model.safetensors` (1.51 GB, 787 MB of it the text tower's token embedding).
pub const SIGLIP2_WEIGHTS_SIZE: u64 = 1_507_473_288;

/// Where LightCraft downloads the model from, in order: each is a base URL, and file `f` is at
/// `<base>/<f>`. The Hugging Face repository is not gated, so its pinned revision can be a default
/// (unlike SAM 3's).
// TODO(maintainer): add a LightCraft mirror first once one exists (the files there must match
// `SIGLIP2_FILES`).
pub const DEFAULT_MIRRORS: &[&str] = &["https://huggingface.co/google/siglip2-base-patch32-256/resolve/94dffa8cb1179de3e03f091dbc3917e5d5a9ae84"];

/// Environment variable with extra mirrors (base URLs separated by commas, spaces or newlines),
/// tried before the defaults.
pub const MIRRORS_ENV: &str = "LIGHTCRAFT_SIGLIP2_MIRRORS";

/// Environment variable with a bearer token for a configured mirror's own host.
pub const TOKEN_ENV: &str = "LIGHTCRAFT_SIGLIP2_TOKEN";

/// The mirrors to try for SigLIP 2, in order: the environment variable's, then the mirrors file's
/// (one base URL per line, `#` comments), then [`DEFAULT_MIRRORS`].
pub fn mirrors(env: Option<&str>, file: Option<&Path>) -> Vec<String> {
    lightcraft_fetch::mirrors(env, file, DEFAULT_MIRRORS)
}

/// Download settings for SigLIP 2: the defaults, plus [`TOKEN_ENV`].
pub fn options() -> Options {
    Options { token_env: Some(TOKEN_ENV), ..Options::default() }
}

/// What to tell the user when there is no mirror for the model.
pub fn no_mirrors_message() -> String {
    format!(
        "no download location is configured for the SigLIP 2 model (set {MIRRORS_ENV}, or put the files in the model folder yourself; see docs/vera-models.md)"
    )
}

/// The model's licence, for the engine's licence dialog.
pub const LICENSE_NAME: &str = "Apache License 2.0";
/// The model card, which states the licence.
pub const LICENSE_URL: &str = "https://huggingface.co/google/siglip2-base-patch32-256";
/// The in-app notice shown before the download (followed by the full Apache-2.0 text).
pub const LICENSE_NOTICE: &str = "SigLIP 2 Base (google/siglip2-base-patch32-256), Copyright 2025 Google LLC. Licensed under the \
Apache License, Version 2.0 (full text follows) — commercial use permitted. Google's big_vision \
release states model materials are also available under CC-BY 4.0. Trained by Google on the WebLI \
dataset (not public).";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn siglip2_mirrors_come_from_the_environment_the_file_and_the_defaults() {
        let m = mirrors(Some("https://a.example/x/"), Some(Path::new("/nonexistent/mirrors.txt")));
        assert_eq!(m.len(), 1 + DEFAULT_MIRRORS.len());
        assert!(m.iter().any(|u| u.contains("huggingface.co/google/siglip2-base-patch32-256")));
        assert!(no_mirrors_message().contains(MIRRORS_ENV));
        assert_eq!(options().token_env, Some(TOKEN_ENV));
    }

    #[test]
    fn every_file_is_pinned() {
        for f in SIGLIP2_FILES {
            assert!(f.size.is_some() && f.sha256.is_some_and(|h| h.len() == 64), "{}", f.name);
            assert!(f.size.is_some_and(|s| s <= f.max));
        }
        let names: Vec<&str> = SIGLIP2_FILES.iter().map(|f| f.name).collect();
        assert_eq!(names, [crate::CONFIG_FILE, crate::TOKENIZER_FILE, crate::WEIGHTS_FILE]);
        assert!(LICENSE_NOTICE.contains("Apache License, Version 2.0"));
    }
}
