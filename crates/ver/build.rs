//! Records the toolchain an evaluation must publish (decoder-corpus README, "Publishing an
//! evaluation"): the versions of the crates on the CPNP-1 path, read from the workspace
//! `Cargo.lock`, and the compiler's own version string.

use std::path::Path;
use std::process::Command;

const CRATES: &[&str] = &["png", "image-webp", "tiff", "gif", "image", "moxcms", "miniz_oxide", "zune-jpeg", "jpeg-decoder"];

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let lock_path = Path::new(&manifest).join("../../Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../../Cargo.toml");
    let lock = std::fs::read_to_string(&lock_path).unwrap_or_default();
    // The version requirement each crate is used under (this crate's manifest, then the
    // workspace's), so a lock file holding two semver-incompatible copies reports ours.
    let manifests = [Path::new(&manifest).join("Cargo.toml"), Path::new(&manifest).join("../../Cargo.toml")];
    let texts: Vec<String> = manifests.iter().map(|p| std::fs::read_to_string(p).unwrap_or_default()).collect();
    let requirement = |name: &str| -> Option<String> {
        texts.iter().flat_map(|t| t.lines()).find_map(|l| {
            let rest = l.trim().strip_prefix(name)?.trim_start().strip_prefix('=')?;
            let start = rest.find('"')? + 1;
            let end = rest[start..].find('"')? + start;
            Some(rest[start..end].to_string())
        })
    };
    let mut out = Vec::new();
    for name in CRATES {
        let mut versions: Vec<String> = Vec::new();
        let mut lines = lock.lines();
        while let Some(line) = lines.next() {
            if line.trim() == format!("name = \"{name}\"")
                && let Some(v) = lines.next()
                && let Some(v) = v.trim().strip_prefix("version = \"")
            {
                versions.push(v.trim_end_matches('"').to_string());
            }
        }
        if let Some(req) = requirement(name) {
            let matching: Vec<String> = versions.iter().filter(|v| v.starts_with(&format!("{req}.")) || **v == req).cloned().collect();
            if !matching.is_empty() {
                versions = matching;
            }
        }
        let v = if versions.is_empty() { "unknown".to_string() } else { versions.join(" / ") };
        out.push(format!("{name}={v}"));
    }
    println!("cargo:rustc-env=VER_TOOLCHAIN_CRATES={}", out.join(";"));
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=VER_RUSTC_VERSION={version}");
}
