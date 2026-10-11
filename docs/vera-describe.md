# VERA Describe: the Phase 3 decision

> **Last updated:** 2026-10-11 · **Status:** decided for Phase 3 of `docs/vera.md`; throughput figures are estimates until the first PR benchmarks them; SHA-256 values are re-computed at implementation time.

Describe proposes a **title, caption, alt text and keywords** for a photograph. The user reviews every proposal and
accepts, edits or rejects it; nothing is written to a photo without an accept (`describe.accept` is an ordinary
`photo.setMeta`, undoable). The PRD prefers a local path and allows a remote endpoint the user configures. This is the
comparison and the choice, under the same three tests as the embedding model (`vera-models.md`): a licence that can be
shown in-app and permits the user's commercial use of their own catalog; runs on candle 0.9.2 with no C dependency and
no `tokenizers` crate; and background CPU speed on a 2020-class laptop (the PRD's archivist scenario: 2,000 frames in
an evening).

## Decision

**Design A now, Design B later, remote path first.**

1. **Remote path first** (the smallest change that delivers the whole review loop): an OpenAI-compatible
   `POST {base}/v1/chat/completions` with the image as a base64 data URL, which llama.cpp, Ollama, vLLM and LM Studio
   all serve on the user's own machine. LightCraft ships no default host. Plain `http` only to a loopback address;
   `https` for anything else (the segment crate's precedent: "bind to loopback, tunnel over SSH").
2. **Keywords locally, from the model already shipped:** SigLIP 2 B/32 zero-shot over an original vocabulary of about
   400 photographic terms (ours; no protected or personal attributes), scored against each photo's existing image
   embedding with the lane's own background-based cutoff. No new model, about 0.1 ms per photo.
3. **Captions locally, from `microsoft/Florence-2-base`** (MIT, 232M parameters, f16 safetensors 463 MB): its
   `<CAPTION>` prompt gives the caption (and the title, by rule), `<DETAILED_CAPTION>` the alt text,
   `<MORE_DETAILED_CAPTION>` an opt-in extended description. An in-tree port of about 1,000 lines (DaViT image encoder,
   BART encoder-decoder, 3-beam search with no-repeat-3-gram, a byte-level BPE *decoder* only: the three task prompts
   are fixed id arrays).
4. **Fallback captioner: BLIP base** (BSD-3-Clause, 247M; candle-transformers has the architecture): cheaper on CPU
   but its model card carries a "research purposes only" line (not a licence term) and its weights exist only as a
   pickle, so it would need a LightCraft safetensors mirror.
5. **Design B later** (one vision-language model writing all four fields): first through the remote path, and a local
   SmolVLM only after a measured comparison. On CPU it is not better: SmolVLM-256M costs about what Florence-2 does
   per photo, has no captioning benchmark, and replaces a controlled vocabulary with free text.

## Comparison

CPU basis: 4 threads at 25–45 GMAC/s (from the measured SigLIP throughput), 2 threads at 12–22; token decoding is
memory-bound (f32 weights at about 15 GB/s). "Port" counts non-test lines; `siglip.rs` is about 485.

| Model | Licence | Params | Input | Quality | Tokenizer | In candle 0.9.2? | Port | Weights | GMAC | s / image, 4 t (2 t) | 2,000 frames, 4 t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BLIP base | BSD-3-Clause (+ a "research only" model-card line) | 247M | 384 | COCO CIDEr 133.3 | WordPiece | yes (`blip.rs`, `blip_text.rs`) | ~650 | **pickle / .h5 only**, 990 MB | ~63 | 2.1–3.2 (3.7–6.1) | 1.2–1.8 h |
| BLIP large | BSD-3-Clause, same caveat | 470M | 384 | 136.7; "arafed" artefact reported | WordPiece | yes | ~650 | f32, 1.88 GB | ~200 | 5–9 (10–18) | 3–5 h |
| GIT-base-coco | MIT | 129M | 224 | 131.4 (paper); checkpoint saw 10M pairs | WordPiece | no | ~500 | pickle only | ~27 | 1–1.5 | < 1 h |
| **Florence-2-base** | **MIT** | 232M | 768 | 133.0 zero-shot / 140.0 fine-tuned; three caption lengths | BART byte BPE | no | ~1,000 | **f16 safetensors, 463 MB** | 182 + 28 per prompt | 7–11 (13–22) | **3.9–6.1 h** |
| Florence-2-large | MIT | 0.77B | 768 | 135.6 / 143.3 | BART byte BPE | no | same code | f16, 1.55 GB | ~820 | 18–33 | 10–18 h ✗ |
| SmolVLM-256M-Instruct | Apache-2.0 | 256M | 512, untiled | none published | GPT-2 byte BPE | no | ~750 | bf16, 513 MB | ~131 | 9.5–12 (14–19) | 5.3–6.7 h |
| SmolVLM-500M-Instruct | Apache-2.0 | 507M | 512 | none | byte BPE | no | ~750 | bf16, 1.02 GB | ~170 | 20–23 | 11–13 h ✗ |
| Moondream2 (2025-06) | Apache-2.0 | 1.93B | 378, multi-crop | none | own | `moondream.rs` targets moondream1 | ~900+ | bf16, 3.9 GB | 330–1,000 | 26–96 | ✗ |
| Qwen2-VL-2B | Apache-2.0 | 2.2B | dynamic | — | byte BPE 151k | no | ~1,200 | bf16, 4.4 GB | ~1,200 | 90–110 | ✗ |
| Qwen2.5-VL-3B | **non-commercial only** ✗ | | | | | | | | | | |
| PaliGemma 2 3B | **Gemma Terms** (pass-through restrictions, remote restriction right) ✗ | | | | | | | | | | |
| LLaVA-1.5-7B | **Llama 2 Community** ✗, too large | | | | | | | | | | |
| LFM2.5-VL-450M | **LFM 1.0** (revenue cap) ✗ | | | | | | | | | | |

## Keywords by zero-shot: what to expect and how to keep it honest

SigLIP 2 B/32 scores 74.0 % ImageNet zero-shot; no multi-label tagging evaluation of it was found. The nearest
evidence is CLIP ViT-B zero-shot tagging (TagCLIP: mAP 63.3 on COCO-80, 85.8 on VOC; RAM: precision / recall 77.9 /
52.9 on OpenImages-common). Expectation: **65–80 % precision at 40–55 % recall**, good on scene-level facets (setting,
genre, time of day, weather, light, dominant subject), weak on small or secondary objects (64 image tokens), counting
and fine-grained species. Enough for proposals a person reviews; not enough to write keywords unattended, which the
PRD forbids anyway.

- **Prompt:** the documented `"This is a photo of {term}."`, lower-cased, padded to 64 tokens; facet-specific variants
  ("This is a photo taken at dusk.", "This is a macro photograph."); template ensembling only if validation shows a
  gain.
- **Calibration, three layers:** per-keyword normalisation (each term's median and MAD over the library's embeddings
  once there are 200 or more, scored as a z-score, which removes "hub" terms); a per-photo ceiling (`cutoff.rs`
  transposed: the photo is the query, the vocabulary the candidates; a sigmoid cross-check with the model's own logit
  scale 111.8 and bias −16.77 is optional); facet rules (single-answer facets such as time of day take the top term
  only by a clear margin; open facets capped at 8 subjects and 15 keywords in all). A term that also appears in the
  caption moves up.
- **Vocabulary:** about 400 original terms in a tree: subjects ~180, settings ~80, genre / technique ~25, light ~20,
  colour ~15, mood ~15, weather ~12, time of day 6, season 4. No gender, ethnicity, age, religion or body terms; people
  stay generic (people, person, crowd, portrait). It includes every keyword the procedural scenes carry. Embedding the
  vocabulary is a one-off (50–90 s), cached beside the library keyed by model id and vocabulary hash.
- **Validation:** the 24 demo scenes as a smoke test (recall@10 of each scene's own keywords; never "dog", "car" or
  "people"); then about 300 CC0 photographs in the gitignored corpus, labelled by two people, tuned on half and
  reported on the other half per facet, targets precision ≥ 0.8 on scene facets and ≥ 0.7 on subjects at 5–12
  proposals per photo. Thresholds recorded with the model id as `thresholds.rs` does.

## Title and alt text from a caption (deterministic, unit-tested, always reviewed)

- **Title** from `<CAPTION>`: drop a leading article and the trailing period; keep it whole at 8 words or fewer, else
  cut at the last clause boundary before word 8 (", ", " while ", " with ", " and "); Chicago title case (articles,
  coordinating conjunctions and prepositions lower-case unless first or last).
- **Caption:** the sentence as written.
- **Alt text** from `<DETAILED_CAPTION>`: strip openings that talk about the image ("The image shows", "In this
  image", "This is an image of"); capitalise; end with a period; at most 250 characters, cut at a sentence boundary.
- WCAG 1.1.1 wants an alternative that serves the same purpose as the image, which depends on context no model knows:
  a reviewed draft is the honest framing. A single BLIP-length caption identifies a photo; Florence-2's detailed
  caption fits the case where the photograph is the content. A vision-language model is not required for this.

## The remote protocol

`crates/fetch/src/http.rs` can only GET today; a `request(method, url, headers, body)` with `get` as a wrapper is
about 30 lines plus a loopback test. Policy: plain `http` only to loopback, `https` elsewhere.

- Request: `POST {base}/v1/chat/completions`, `Authorization: Bearer <token>`, `model` the user's, temperature 0.2,
  `max_tokens` 400, `response_format` a strict JSON schema `{title ≤ 80, caption ≤ 300, alt ≤ 300, keywords ≤ 20 ×
  ≤ 40}`, a system message ("You write catalog metadata for a photographer…"), and a user message with the
  instructions (title ≤ 8 words in Title Case without a period; one-sentence caption; alt 1–2 sentences ≤ 250
  characters; 5–15 lower-case keywords; JSON only) and the image as `data:image/jpeg;base64,…`.
- Image: a fresh 1024 px JPEG at quality 85 rendered with default settings, so no EXIF or GPS leaves the machine.
- Response: `choices[0].message.content`, code fences stripped, types and lengths checked; on 400 retry with
  `{"type": "json_object"}`, then with no format and the first `{…}` extracted. Body capped at 64 KiB; 120 s without
  data is a stall. Batches of four, three attempts per frame, 429 + `Retry-After` honoured without spending an
  attempt (the studio's protocol).
- Server support: llama.cpp (data URLs, schema), Ollama (base64 data URLs only; JSON mode), vLLM (OpenAI vision
  format), LM Studio (schema; data-URL bugs reported in some versions).
- Token: `<config>/models/describe-remote.token`, created mode 0600 and replaced atomically; refused if group- or
  world-readable (as ssh does); `LIGHTCRAFT_DESCRIBE_TOKEN` overrides. Endpoint URL and model name in prefs. Never in
  the catalog, the journal, logs or command output; MCP sees only `tokenSet: true`.

## Files to pin

Florence-2-base, commit `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac`, base
`https://huggingface.co/microsoft/Florence-2-base/resolve/<commit>/` (not gated):

| File | Bytes |
|---|---|
| `model.safetensors` | 463,221,266 (Hugging Face LFS SHA-256 `03075d2d2d2bbd3e180b9ba0afae4aa8563226e2d32911656966e05b2f2ee060`; verify) |
| `vocab.json` | 1,099,884 |
| `config.json` | 2,430 |
| `preprocessor_config.json` | 806 (768 × 768 bicubic, ImageNet mean / std, 577 image tokens) |
| `LICENSE` | 1,141 |

(`tokenizer.json`, 1,355,863 bytes, is used only to generate golden tests.) BLIP base fallback at commit
`82a37760796d32b1411fe092ab5d4e227313294b`: `pytorch_model.bin` 989,820,849 bytes, `vocab.txt` 231,508 bytes.

## In-app licence text (draft for the maintainer)

> Florence-2 base (microsoft/Florence-2-base), Copyright (c) Microsoft Corporation. Licensed under the MIT License
> (full text follows); commercial use permitted. Trained by Microsoft on FLD-5B (not public). Captions are
> machine-written and can be wrong; LightCraft keeps nothing until you accept it. Keywords come from SigLIP 2 (see its
> notice).

## Risks

1. **Florence-2 CPU speed** is an estimate (7–11 s per image on 4 threads; depthwise convolutions need an in-tree
   helper because candle's grouped convolution loops per group). Gate: over 12 s per image on 4 threads, try a 512–576
   px input (the 2D position embedding allows it; quality unmeasured), else BLIP base.
2. **Port effort** of about 1,000 lines with no candle reference; `weights.rs` needs an f16 → f32 path.
3. **Keyword recall near 50 %** and the dominant-subject bias; per-keyword normalisation is mandatory.
4. **Captions may guess gender or age** ("a man"): the maintainer decides whether proposals are rewritten neutrally.
5. **Remote:** LM Studio's data-URL bugs; weak JSON adherence on small models; pixels leave the machine, so consent per
   endpoint and the host shown wherever a describe task is listed.

## Not verified

A multi-label tagging evaluation of SigLIP 2; measured CPU timings for any candidate; vLLM's `json_schema`
support; whether LM Studio's data-URL bugs are fixed.

## Revision history

- 2026-10-11: created from the Phase 3 model research (model cards, Hugging Face API blob listings, candle-transformers
  0.9.2 sources, the TagCLIP and RAM papers, server documentation for llama.cpp, Ollama, vLLM and LM Studio).
