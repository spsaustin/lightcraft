# VERA embedding model: the Phase 2 decision

> **Last updated:** 2026-10-10 · **Status:** decided for Phase 2 of `docs/vera.md`; SHA-256 values are re-computed at implementation time.

The vector lane (`docs/vera.md` → Phase 2) needs one image+text embedding model that passes three tests
from the PRD: a licence that can be shown in-app and permits the user's commercial use of their own
catalog; it runs on the pinned candle 0.9.2 (or the in-tree ONNX interpreter) with no C dependency;
and a 50k-frame catalog embeds overnight (about 8 h, 1.74 images/s) at a background pace on a
2020-class laptop, CPU only. This is the comparison and the choice.

## Decision

- **Primary: `google/siglip2-base-patch32-256`** — Apache-2.0, 94.6M-parameter image tower, 256 px
  input, 64 patch tokens, 768-d embeddings, about 5.5 GMACs per image. Same family and tokenizer as
  the reference model the studio's thresholds were measured on. Estimated 3–4 h for 50k frames on two
  CPU threads. 74.0 % ImageNet zero-shot.
- **Fallback: `laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K`** — MIT, 88M image tower, 224 px, 512-d,
  4.3 GMACs. Its byte-level BPE tokenizer is the one `crates/segment/src/tokenizer.rs` already
  implements (77-token context instead of 32). 72.7 % ImageNet zero-shot. Used if the SigLIP 2
  tokenizer cannot be made to match the reference encodings, or if the 1.5 GB download is judged
  unacceptable.
- **The reference `siglip2-so400m-patch14-384` is not shipped**: it fails the overnight test by a
  factor of about 20 (roughly 333 GMACs per image, 7–11 days for 50k frames on CPU). It may return as
  an opt-in "quality" model for machines with Metal; the code is the same, only the config differs.
- **The thresholds are re-measured.** `EMBED_DUPLICATE` 0.82 and `EMBED_NEAR` 0.78 were measured on
  so400m; cosine scales do not carry across models. Gate 2 re-measures them on the fixture set
  against the shipped model and records the new values and the model id in `thresholds.rs`.

## Comparison

| Model | Licence | Params (image / total) | Input | Dim | Tokenizer | candle 0.9.2 | GMACs / image | Overnight (50k, CPU) |
|---|---|---|---|---|---|---|---|---|
| siglip2-so400m-patch14-384 | Apache-2.0 | 428M / 1.14B (4.54 GB f32) | 384, 729 tokens | 1152 | Gemma SentencePiece-BPE, 256k | yes | ~333 | **fails** (170–260 h) |
| siglip2-base-patch16-256 | Apache-2.0 | 95M / 375M (1.50 GB) | 256, 256 tokens | 768 | Gemma SP-BPE | yes | ~23 | marginal (6–17 h) |
| siglip2-base-patch16-224 | Apache-2.0 | 95M / 375M | 224 | 768 | Gemma SP-BPE | yes | ~18 | marginal (5–14 h) |
| **siglip2-base-patch32-256** | **Apache-2.0** | **94.6M / 377M (1.51 GB)** | 256, 64 tokens | 768 | Gemma SP-BPE | yes | **~5.5** | **passes (3–4 h)** |
| siglip-base-patch16-224 (v1) | Apache-2.0 | 93M / 203M (813 MB) | 224 | 768 | SentencePiece **unigram** + precompiled charsmap | yes | ~18 | marginal |
| openai/clip-vit-base-patch32 | MIT (no safetensors; pickle only) | 88M / 151M | 224 | 512 | byte BPE | yes | 4.3 | passes |
| laion CLIP-ViT-B-32-laion2B-s34B-b79K | MIT | 87.5M / 151M (605 MB) | 224 | 512 | byte BPE | yes | 4.3 | passes (66.6 % zero-shot) |
| **laion CLIP-ViT-B-32-DataComp.XL** | **MIT** | 88M / 151M (605 MB) | 224 | 512 | byte BPE | open_clip key names (needs a key map) | 4.3 | passes |
| Apple MobileCLIP S0/S1/S2, MobileCLIP2 | apple-amlr: **research only** | — | 256 | 512 | BPE | yes | — | **fails licence** |
| jina-clip-v2 | **CC-BY-NC-4.0** | — | — | — | — | — | — | **fails licence** |
| jina-clip-v1 | Apache-2.0 | EVA02-B/16 + JinaBERT | 224 | 768 | WordPiece | no (remote code) | ~18 | not worth porting |
| nomic-embed-vision-v1.5 | Apache-2.0 | ~92M | 224 | 768 | WordPiece, separate text model | no (custom BERT) | ~18 | not worth porting |

Licence notes: Apple's licence reads "exclusively for Research Purposes … does not include any
commercial exploitation, product development or use in any commercial product or service". The
OpenAI and LAION model cards say "any deployed use case of the model – whether commercial or not – is
currently out of scope"; that is a model-card caveat, not a licence term, and it is one reason CLIP
is the fallback rather than the primary. No Hugging Face repo in the table carries a LICENSE file;
the licence is the README's `license:` tag, and for SigLIP 2 the big_vision release states software is
Apache-2.0 and all other materials CC-BY 4.0. The maintainer signs off on the in-app wording.

## How it runs

- **Where:** a new crate on candle-core + candle-nn 0.9.2 only, in-tree like SAM 3 in
  `crates/segment` (its `nn.rs` Attention / Mlp / layer norm / scaled-dot-product attention, `vit.rs`
  and `weights.rs` are the references; `Mlp` needs a GELU-tanh option, `weights.rs` reads f32 only,
  which SigLIP 2 is). About 180 lines for the image tower (patch conv, learned position embedding, 12
  layers, post-LN, MAP head) and 100 for the text tower. `candle-transformers = "=0.9.2"` was checked
  and is clean (no `tokenizers`, no C build, default features empty) and has a `siglip.rs`, but it is
  77k lines of compile time for 300 lines of use, so the port stays in-tree.
- **Not the faces ONNX interpreter:** it lacks MatMul, Softmax, LayerNormalization, Erf, Gather,
  Concat, Slice, caps tensors at 64 Mi elements (the 256k×768 token embedding is 196.6M) and files at
  512 MiB.
- **Device:** as `segment::best_device()`: Metal on macOS, CPU elsewhere. Both towers f32.
- **Memory:** read the token-embedding rows positionally instead of loading the 787 MB table, so the
  text tower stays near 340 MB resident; the image tower is about 380 MB. Both are counted into the
  per-worker estimate `faces_worker` uses for the Pace model.
- **Preprocessing:** resize to 256×256 bilinear, then `(x / 255 − 0.5) / 0.5`, from the preview
  crate's thumbnail-level pixels (no second decode).

## Tokenizer (SigLIP 2)

`tokenizer.model` is **SentencePiece BPE, not unigram** (TrainerSpec `model_type = 2`,
`byte_fallback = 1`, `split_digits = 1`, identity normaliser, 256,000 pieces of which 256 are byte
pieces and 245 user-defined). The in-tree encoder: lower-case the text; replace spaces with `▁`;
match user-defined pieces greedily; split digits; merge by piece score (SentencePiece-style BPE);
fall back to `<0xXX>` byte pieces; append `<eos>` (id 1); pad with `<pad>` (id 0) to 64 tokens. The
protobuf reader needs only fields 1.1 / 1.2 / 1.3 (piece, score, type); the repo already has a
bounded hand-written protobuf reader in `crates/faces/src/onnx.rs`. About 250 lines plus golden tests
against the Hugging Face `tokenizer.json` encodings. (SigLIP v1 would have needed a unigram Viterbi
decoder and the precompiled-charsmap normaliser; another reason v2 is the choice.)

## Files to pin

Primary, base `https://huggingface.co/google/siglip2-base-patch32-256/resolve/94dffa8cb1179de3e03f091dbc3917e5d5a9ae84/`
(not gated, so it can be the default mirror):

| File | Bytes | SHA-256 |
|---|---|---|
| `model.safetensors` | 1,507,473,288 | `7d241bb3becad218f211f480487f491df4f8c0a472ecf7afdec5615815a301f1` (Hugging Face LFS value; verify) |
| `tokenizer.model` | 4,241,003 | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` (verified with sha256sum) |
| `config.json` | 298 | compute at implementation |
| `preprocessor_config.json` | 394 | compute at implementation |

Fallback, commit `f0e2ffa09cbadab3db6a261ec1ec56407ce42912` of the DataComp repo:
`open_clip_model.safetensors` 605,143,284 bytes, SHA-256
`3c00043509d2e3f35ec62bd85a643a393883dcda624d8208d034cc390c708297`; `vocab.json` 862,328 bytes;
`merges.txt` 524,657 bytes (hashes computed at implementation).

## In-app licence text (draft for the maintainer)

> SigLIP 2 Base (google/siglip2-base-patch32-256), Copyright 2025 Google LLC. Licensed under the
> Apache License, Version 2.0 (full text follows) — commercial use permitted. Google's big_vision
> release states model materials are also available under CC-BY 4.0. Trained by Google on the WebLI
> dataset (not public).

followed by the full Apache-2.0 text (`LICENSE-APACHE` can be reused).

## Throughput evidence

Target 1.74 images/s. Published points: PyTorch CLIP B/32 on a 2-core 2.3 GHz Xeon, 64 images in
8.07 s (about 7.9 images/s, about 35 GMAC/s); an i7-12700K runs B/32-256 at 114 ms per image.
Assuming candle's `gemm` reaches 50–70 % of PyTorch (18–25 GMAC/s on two threads, 35–50 on four):
SigLIP 2 B/32-256 at 3.3–4.5 images/s, 3.1–4.2 h for 50k; B/16-224 at 10–14 h on two threads;
so400m at 7–11 days. **These are estimates**: the first Phase 2 PR benchmarks candle 0.9.2 on a real
2020-class laptop before the numbers are trusted.

## Risks

1. The 1.5 GB download, 787 MB of it the 256k-token text embedding. Mitigation: a LightCraft mirror
   re-packed as separate vision and text files (Apache-2.0 §4 allows it with the modification stated).
2. Tokenizer fidelity (BPE by score, user-defined pieces, byte fallback): golden tests are mandatory.
3. Retrieval quality below the reference (74.0 % vs 84.1 % zero-shot): B/16-224 as an opt-in quality
   model on Metal machines, same code.
4. candle 0.9.2 CPU speed unmeasured until the benchmark.
5. Fallback caveats: the CLIP model cards' deployment caveat; DataComp's uncurated Common Crawl
   provenance.

## Revision history

- 2026-10-10: created from the Phase 2 model research (Hugging Face API blob listings, safetensors
  headers read by range request, crates.io tarballs for candle 0.9.2, model cards and licences).
