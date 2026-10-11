//! `cargo xtask ver …`: the VER standard's published test vectors and record validation, over the
//! copies vendored in `crates/ver/standard/` (no network).

use lightcraft_ver::corpus;
use lightcraft_ver::record::{Record, Schema};
use lightcraft_ver::validate::{self, Severity};
use std::path::{Path, PathBuf};
use std::process::Command;

const USAGE: &str =
    "usage: cargo xtask ver <corpus | refsets | validate [--schema S] [--oracle auto|required|off] FILE... | oracle | evaluation [--write [PATH]]>";

/// The upstream Python validator: (interpreter, `PYTHONPATH`), found in this order:
/// 1. `VER_ORACLE_PY` (an interpreter) and `VER_ORACLE_SRC` (`packages/ver-validator/src`);
/// 2. `VERA_CHECKOUT`, a vera checkout: its `packages/ver-validator/src`, run by its
///    `.venv/bin/python`, else by `python3` on `PATH` if that can import `ver_validator`;
/// 3. a sibling checkout `../vera` next to the workspace, derived the same way.
pub fn oracle(root: &Path) -> Option<(PathBuf, PathBuf)> {
    if let (Ok(py), Ok(src)) = (std::env::var("VER_ORACLE_PY"), std::env::var("VER_ORACLE_SRC"))
        && !py.is_empty()
        && !src.is_empty()
    {
        return Some((PathBuf::from(py), PathBuf::from(src)));
    }
    let checkouts = std::env::var_os("VERA_CHECKOUT").map(PathBuf::from).into_iter().chain(std::iter::once(root.join("../vera")));
    checkouts.into_iter().find_map(|c| from_checkout(&c))
}

/// The validator inside a vera checkout, if it has one and some Python can run it.
fn from_checkout(checkout: &Path) -> Option<(PathBuf, PathBuf)> {
    let src = checkout.join("packages/ver-validator/src");
    if !src.join("ver_validator").is_dir() {
        return None;
    }
    let venv = checkout.join(".venv/bin/python");
    if venv.is_file() {
        return Some((venv, src));
    }
    let imports = Command::new("python3").args(["-c", "import ver_validator"]).env("PYTHONPATH", &src).output().is_ok_and(|o| o.status.success());
    imports.then(|| (PathBuf::from("python3"), src))
}

pub fn run(root: &Path, args: &[&str]) -> Result<(), String> {
    match args.first().copied() {
        Some("corpus") => corpus_cmd(root),
        Some("refsets") => refsets(root),
        Some("validate") => validate_cmd(root, &args[1..]),
        Some("oracle") => oracle_tests(root),
        Some("evaluation") => evaluation(root, &args[1..]),
        _ => Err(USAGE.into()),
    }
}

fn corpus_cmd(root: &Path) -> Result<(), String> {
    let report = corpus::run_decoder_corpus(&root.join(corpus::DECODER_CORPUS_DIR))?;
    println!("{}", corpus::render_evaluation(&report));
    if report.reproduced < corpus::PINNED_DECODER_CORPUS {
        return Err(format!(
            "decoder corpus: {} of {} reproduced, below the pinned {}",
            report.reproduced,
            report.total,
            corpus::PINNED_DECODER_CORPUS
        ));
    }
    eprintln!("ver corpus: {} of {} reproduced ({})", report.reproduced, report.total, report.profile);
    Ok(())
}

fn refsets(root: &Path) -> Result<(), String> {
    let report = corpus::run_reference_set_pixels(&root.join(corpus::REFERENCE_SETS_DIR))?;
    println!("{}", corpus::render_evaluation(&report));
    if report.reproduced < corpus::PINNED_REFERENCE_SET_PIXELS || report.reproduced < report.total {
        return Err(format!("reference sets: {} of {} pixel_sha256 reproduced", report.reproduced, report.total));
    }
    eprintln!("ver refsets: {} of {} reproduced", report.reproduced, report.total);
    Ok(())
}

fn validate_cmd(root: &Path, args: &[&str]) -> Result<(), String> {
    let mut schema_flag: Option<&str> = None;
    let mut oracle_mode = "auto";
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match *a {
            "--schema" => schema_flag = Some(it.next().copied().ok_or("--schema needs 1.0.1 or 1.1-draft")?),
            "--oracle" => oracle_mode = it.next().copied().ok_or("--oracle needs auto, required or off")?,
            f => files.push(f),
        }
    }
    if files.is_empty() {
        return Err(USAGE.into());
    }
    let oracle = match oracle_mode {
        "off" => None,
        "auto" => oracle(root),
        "required" => Some(
            oracle(root).ok_or("the upstream validator is required (--oracle required) but was not found: set VER_ORACLE_PY and VER_ORACLE_SRC")?,
        ),
        other => return Err(format!("unknown --oracle mode `{other}`")),
    };
    let mut failed = 0;
    for f in files {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        println!("== {f}");
        let record = match Record::parse(&text) {
            Ok(r) => r,
            Err(e) => {
                println!("  lightcraft-ver: error {e}");
                failed += 1;
                continue;
            }
        };
        let schema = match schema_flag {
            Some("1.1-draft") => Schema::V1_1Draft,
            Some("1.0.1") => Schema::V1_0_1,
            Some(other) => return Err(format!("unknown schema `{other}`")),
            None => record.schema().unwrap_or(Schema::V1_0_1),
        };
        let issues = validate::all(&record.0, schema);
        let ok = validate::is_conformant(&issues);
        println!("  lightcraft-ver ({}): {}", schema_name(schema), if ok { "conformant" } else { "nonconformant" });
        for i in &issues {
            println!("    {} {:<8} {:<40} {}", if i.severity == Severity::Error { "error  " } else { "warning" }, i.code, i.pointer, i.message);
        }
        if !ok {
            failed += 1;
        }
        if let Some((py, src)) = &oracle {
            let out = Command::new(py)
                .args(["-m", "ver_validator.cli", "--schema", schema_name(schema)])
                .arg(f)
                .env("PYTHONPATH", src)
                .output()
                .map_err(|e| format!("{}: {e}", py.display()))?;
            println!("  upstream ver_validator:");
            for line in String::from_utf8_lossy(&out.stdout).lines().chain(String::from_utf8_lossy(&out.stderr).lines()) {
                println!("    {line}");
            }
            if !out.status.success() {
                failed += 1;
            }
        }
    }
    if oracle.is_none() && oracle_mode == "auto" {
        eprintln!("(upstream validator not found: set VER_ORACLE_PY + VER_ORACLE_SRC, or VERA_CHECKOUT, or check out vera as ../vera)");
    }
    if failed > 0 { Err(format!("{failed} validation failure(s)")) } else { Ok(()) }
}

fn schema_name(s: Schema) -> &'static str {
    match s {
        Schema::V1_0_1 => "1.0.1",
        Schema::V1_1Draft => "1.1-draft",
    }
}

fn oracle_tests(root: &Path) -> Result<(), String> {
    let (py, src) = oracle(root)
        .ok_or("the upstream validator was not found: set VER_ORACLE_PY + VER_ORACLE_SRC, or VERA_CHECKOUT, or check out vera as ../vera")?;
    let mut c = crate::cargo();
    c.args(["test", "-p", "lightcraft-ver", "--test", "integration", "oracle", "--", "--nocapture"])
        .env("VER_ORACLE_PY", &py)
        .env("VER_ORACLE_SRC", &src);
    crate::run(c, "cargo test -p lightcraft-ver --test integration oracle (with the upstream validator)")
}

const BEGIN: &str = "<!-- ver-evaluation:begin (generated by `cargo xtask ver evaluation --write`; do not edit) -->";
const END: &str = "<!-- ver-evaluation:end -->";

fn evaluation(root: &Path, args: &[&str]) -> Result<(), String> {
    let report = corpus::run_decoder_corpus(&root.join(corpus::DECODER_CORPUS_DIR))?;
    let generated = corpus::render_evaluation(&report);
    let Some(pos) = args.iter().position(|a| *a == "--write") else {
        println!("{generated}");
        return Ok(());
    };
    let target = root.join(args.get(pos + 1).copied().unwrap_or("docs/ver-cpnp.md"));
    let doc = std::fs::read_to_string(&target).unwrap_or_default();
    let section = format!("{BEGIN}\n\n{generated}\n{END}");
    let updated = match (doc.find(BEGIN), doc.find(END)) {
        (Some(b), Some(e)) if e > b => format!("{}{section}{}", &doc[..b], &doc[e + END.len()..]),
        _ if doc.is_empty() => format!("# VER decoder evaluation\n\n{section}\n"),
        _ => format!("{}\n\n{section}\n", doc.trim_end()),
    };
    std::fs::write(&target, updated).map_err(|e| format!("{}: {e}", target.display()))?;
    eprintln!("wrote the evaluation into {}", target.display());
    Ok(())
}
