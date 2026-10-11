# VERA: finding photos by words, and the same picture twice

VERA is how LightCraft answers two questions about a library: *which photos are about this?* (search) and *is this the
same picture as one I already have?* (duplicates). The design comes from the VERA studio, a search system built for one
photographer's archive: its lanes, its measured thresholds and the fixtures that chose them. Here it is three parts:

- **`lightcraft-vera`** (`crates/vera`, layer 2): the algorithms, as pure functions over slices. A BM25 keyword index,
  the PDQ and pHash digests with the same-picture decision, reciprocal rank fusion and the noise cutoff. No files, no
  threads, no models, no knowledge of the catalog.
- **`lightcraft-embed`** (`crates/embed`, layer 3): the SigLIP 2 image and text towers on candle, in pure Rust (the
  model choice is [vera-models.md](vera-models.md)). Turns a 256 × 256 picture or a query into a 768-number unit vector.
- **The engine** (`crates/engine/src/vera/`): what to digest and embed and when, where the results are kept, and the
  commands that show them. Everything is a command, so the UI, the CLI, the control channel and MCP all see the same
  thing.

## Lanes

A search runs on *lanes*, each a different way of finding a photo; their ranked lists are fused into one. Phase 2 has
three lanes, the third once its model is downloaded.

| Lane | What it finds | State |
|---|---|---|
| **keyword** | Photos whose words match: title, keywords (every level of `travel\|Italy\|Rome`), caption, alt text, extended description, file name, camera, lens and place | **live** |
| **digest** | The same picture under other bytes: a resize, a re-export, a format change | **live** |
| **vector** | Photos by meaning ("a dog on a beach"), look-alikes (`photo.similar`), and crops the digests cannot catch | **live** on the desktop once the user downloads SigLIP 2 (Phase 2); `needs_desktop` on the web |
| **describe** | Words written for photos that have none, so the keyword lane can find them | disabled (Phase 3) |

`vera.status` reports these on every build, with a sentence for each lane that is not running, so nothing claims a lane
it does not have.

### The keyword lane

Each photo in the library is one document with seven fields, weighted: title 3, keywords 2.5, caption 1.5, alt text 1,
extended description 0.5, file name without its extension 1, and camera + lens + location, city, state and country 0.5.
Words are runs of letters and digits, lower-cased; a query word matches every indexed word it is a prefix of (`moun`
finds `mountains`); any word of the query may match, and a photo that matches more of them ranks higher. Equal scores
are ordered by photo id, so the same catalog always gives the same list.

The index is built from the catalog when a search needs it and built again whenever the catalog has changed since (a
full rebuild; the build time and the photo count are logged at debug level so the cost can be watched as libraries
grow). It is not saved: it is cheaper to build than to read back.

### The digest lane

Each picture gets two fingerprints from its thumbnail-level picture (at most 512 pixels, turned into Rec. 601 luma):
**PDQ** (256 bits, with a 0–100 quality figure that says how much detail survived) and **pHash** (64 bits). Two
pictures are compared by how many bits differ:

- **the same picture**: PDQ within 31 bits (when both pictures have a quality of 50 or more), or pHash within 6. Resizes,
  re-exports and format changes land here.
- **possibly the same**: both within their loose bounds (PDQ 63, pHash 12): a variant edit, a heavy crop, a burst
  neighbour. Only ever shown, never acted on.
- **different**: anything else. A real crop is usually here; finding crops is the vector lane's job.

The numbers and the measurement behind each are in `crates/vera/src/thresholds.rs`.

Fingerprints are kept per *picture*, not per photo: by the photo's content key (its file's content hash), so a virtual
copy shares its original's fingerprint and a re-linked file keeps it.

## Fingerprinting in the background

Every photo in the library (videos aside) is fingerprinted once, in the background, by `vera.pump`, which the app calls
every frame. Each call is cheap: it digests at most a few pictures and returns.

- **Pace.** The app says how hard it may work, as for the face scan: `pause` does nothing but report (dragging, typing,
  scrolling); `light` digests one picture per call; `normal` two; `full` four.
- **Newest photos first**, so a fresh import is covered soonest. The catalog is looked through again for new photos at
  most once a second, and only once the current list is done.
- **Unreadable pictures** (an offline original with no smart preview, a file that will not decode) are remembered for
  the session and not tried again every frame; `vera.status` counts them as `unreadable`.
- **The activity stack** shows "Fingerprinting photos" with a count while there is work; it cannot be stopped from there
  (pause is the app's job) and goes away when the work is done.
- The pictures are read without being kept in the session's picture cache, so a pass over the library does not push
  out what you are working on.

## Duplicates

`photo.duplicates` lists the photos that are the same picture as one photo, in three tiers: `bytes` (the same file
content, by hash), `picture` (the same picture by fingerprint) and `possible`. Each row carries the PDQ and pHash
distances. Virtual copies are not duplicates and are left out. The photo's own fingerprint is made on the spot when the
background pass has not got to it yet; other photos are compared once they have theirs (`pendingDigests` says how many
have not).

Two photos you say are not the same picture (`photo.ignorePair`) are no longer paired, by `photo.duplicates` or by the
import check, until `photo.unignorePair`. `photo.stackDuplicate` puts a duplicate in a collapsed stack under the photo
it copies, as one undo step.

## The import check

When an import is reviewed (`library.importPreview`, the Import dialog) and the library already has fingerprints, each
file that is not already a duplicate by path or by bytes is decoded at thumbnail level, fingerprinted and compared with
every fingerprint in the library (the first 2000 files of a review; brute force is fast at this scale). This runs in the
review's own scan (`import::scan_with`, on the app's scan thread), so the window stays responsive. A match is shown on
the candidate as `nearDuplicate: {existing, tier, pdq, phash}`: the first same picture found, else the closest possible
one. A pair set apart with `photo.ignorePair` is not matched.

The import then does what `onNearDuplicate` says with files that are the **same picture** (a possible match is only
shown):

- `stack` (the default): import it and stack it under the library photo, collapsed with the library's photo on top, in
  the import's one undo step. The report lists `stacked: [{photo, under}]`.
- `import`: import it like any other file.
- `skip`: leave it out; the report lists it in `duplicates` with `reason: "picture"`.

The match travels with the review's probe of the file (`Session::import_probes`), so the import does not decode
anything again; a file that has changed since the review is probed afresh and is not treated as a match. An import
that was not reviewed first is not checked, and neither is Synchronize Folder.

## The vector lane (Phase 2)

Finding photos by meaning, and pictures that look alike, by comparing embeddings: each picture becomes a 768-number unit
vector, a query in words becomes one in the same space, and the cosine between two says how alike they are.

### The model

**SigLIP 2 Base, patch 32, 256 px** (`google/siglip2-base-patch32-256`, Apache-2.0), chosen in
[vera-models.md](vera-models.md) for its licence, its size and its speed on a CPU. `lightcraft-embed` runs both towers on
candle (Metal on macOS, the CPU elsewhere), each loaded the first time it is used: the image tower by the background
worker, the text tower by the first search. The model is not part of LightCraft and nothing requires it: without it
the keyword and digest lanes work as before and `vera.status` says what is missing.

The app keeps the model in `<config>/models/siglip2-base-patch32-256` (`LIGHTCRAFT_SIGLIP2_DIR` overrides it), set on the
session as `Session::vera_model_dir` the way the SAM 3 folder is (`apps/lightcraft/src/main.rs`). A session without a
folder (the CLI, MCP, tests) reports `needs_model` with "nowhere to keep it".

### The download and the licence

`vera.model.status` gives what a licence dialog needs: `licence: {name, url, notice}` (the notice is the text drafted in
vera-models.md), the size (`bytes`, 1.51 GB) and the model id. `vera.model.download {acceptLicense: true}` starts the
download only when the user has accepted it; without `acceptLicense` it is an error that names the licence and its URL.
The files are the three pinned in `lightcraft_embed::fetch::SIGLIP2_FILES` (`config.json`, `tokenizer.model`,
`model.safetensors`), each checked by size and SHA-256, fetched with `lightcraft-fetch` (resumable `.part` files, the
next mirror when one fails) from `LIGHTCRAFT_SIGLIP2_MIRRORS`, then `siglip2-mirrors.txt` beside the model folder, then
the pinned Hugging Face revision. It runs on a background thread through the same downloader as SAM 3's
(`segment/download.rs`), with an activity-stack row (`kind "download"`, "Downloading model", detail "SigLIP 2") whose ✕
cancels it like `vera.model.cancel`; what arrived is kept and a later download resumes. Once the three files are in
place the next `vera.pump` opens the model.

`vera.model.remove` deletes the model's files (and their partial downloads) from the model folder, the folder itself
when nothing else is in it, and the library's embedding sidecar; the lane goes back to `needs_model`.

### Switching it off

`vera.config {vectorLane: false}` switches the lane off for the library (saved in `prefs.json` as
`"vera": {"vectorLane": false}`); the model's memory is given back at once and the vectors made so far are kept.
`vera.config {}` reads the setting; it is on by default once the model is installed.

### Embedding in the background

`vera.pump` (the same per-frame call as the fingerprints) also runs the vector lane when it is live:

- **The worker.** One background thread (`vera/worker.rs`) shares the loaded model with the session. For each photo it
  renders the picture the way the face scan does: upright, uncropped, default settings (`Session::preview_job`), at
  512 px on the long edge (the thumbnail-level source serves it), or a raw's embedded preview when that is at least
  256 px. It squashes the result to exactly 256 × 256 by area averaging, without keeping the aspect (as the SigLIP
  processor resizes), and embeds up to 16 pictures (`MAX_BATCH`) per model pass. The render and the model pass run on
  the face scan's thread pool for the pace, never on rayon's global pool, which the loupe and exports use.
- **Pace.** The session thread prepares at most the pace's number of render jobs per call (the same table as the
  fingerprints: `pause` none, `light` 1, `normal` 2, `full` 4), newest photos first, with at most 32 handed out at once.
  Pause feeds nothing; what the worker holds finishes.
- **Memory.** The model with its image tower is about 420 MB, counted once (`vera.status` → `model.memoryBytes` while it
  is loaded), plus one batch of inputs (16 × 256 × 256 × 3 floats, 12.6 MB) and one 512 px render at a time.
- **Failures.** A photo that cannot be rendered is skipped for the session (`vera.status` → `unembeddable`). A model that
  fails to load or to run (a damaged download, out of memory) stops the lane in the `error` state with its message, and
  is tried again after 30 seconds; its photos are not skipped.
- **The activity stack** shows "Embedding photos" (`kind "vera"`) with a count while photos are pending; it cannot be
  stopped from there.
- Results are taken in by the pump on the session thread, inserted in the space and appended to the sidecar every 3
  seconds and when the work runs out (and when the library closes).
- **Measured** (the demo library, 24 photos, CPU, four threads shared with other builds): every photo embedded in 10–17 s
  (1.4–2.4 photos a second, render included); the first query, which loads the text tower, 0.75–0.9 s; a repeated query
  0.1–0.2 ms (a per-process cache of 256 query vectors).

The web build does not start the worker: the lane says `needs_desktop`.

### The embedding sidecar

`vera-embeddings-siglip2-base-patch32-256.bin` in the library folder, one canonical vector per picture (by content key,
like the fingerprints, so a virtual copy shares its original's). Little-endian: `"LCVE"`, version `1`, the model id
(`u16` length + UTF-8), `dim u32` (768), then records of `key_len u16`, the content key (UTF-8), `role u8` and `dim × f32`.
Role `0` is the picture as it is (canonical); `1` (contextual) and `2` (fused) are reserved for Phase 3 and read past by
this build. A file of another model or dimension, another version, or damaged is ignored and rewritten; a torn last
record makes the file be written whole before anything is appended; records of photos no longer in the catalog are
dropped when the library opens (and the file rewritten without them); a repeated key keeps its last vector. The file is
read as a stream (up to 4 GiB) and rewritten through a `.part` file and a rename. In memory the vectors are one
`lightcraft_vera::Space`, keyed by the low 64 bits of the content key's 128-bit hash, with maps both ways.

### How search fuses, and where the cutoff applies

`search.text` runs the keyword lane as before and, when the vector lane is live, embeds the query (once per process
for a given query) and asks the space for its nearest pictures. Each picture's photos join the vector list in rank
order. The two lists are fused by reciprocal rank (`lightcraft_vera::rrf`, k = 60, weight 1 each), which reads ranks,
never scores. Each row carries `lane` (`keyword`, `vector` or `both`), the fused `score`, and `cosine` / `keywordScore`
when it has them; the result carries `lanes: ["keyword", "vector"]` and `cutoff`, the noise ceiling applied.

The **noise cutoff** applies to the vector lane alone, before fusion: vector hits under the ceiling are dropped, so a
query about nothing in the library adds nothing; keyword hits are never cut. The ceiling is
`mean + sigma · (sqrt(2 ln n) + 1.674)` over a background (`crates/vera/src/cutoff.rs`), and the background depends
on what is compared:

- **A text query** uses the lower half of *this query's* cosines with every picture (median, and MAD × 2.5). A
  text–image cosine lives on another scale than an image–image one: on the demo library the space's image pairs have a
  background mean of 0.675 (sigma 0.078), which puts the ceiling at 1.002, while every text–image cosine is between 0.00
  and 0.14. Measured with the per-query background: "the northern lights", "sand dunes in the desert", "a pink flower
  close up" and "a field of lavender" keep exactly their scenes (ceilings 0.044–0.098), "a dog" and "a red sports car"
  keep nothing. "lake at dawn" keeps nothing either: the model ranks the three lake scenes first (0.138, 0.130, 0.129),
  but the eight water-and-sky scenes sit together at 0.11–0.14 and the ceiling is 0.184, so the lake comes first by its
  words alone. SigLIP's own calibration agrees that these synthetic scenes are weak matches: with its logit scale
  (e^4.717 = 111.8) and bias (−16.77), cosine 0.138 is a probability of 0.21.
- **An image query** (`photo.similar`) uses the same per-query statistic, with one difference: its ceiling never rises
  above `EMBED_NEAR`. A picture that near is a near duplicate by measurement and is never noise, however alike the
  rest of the library is. (The space's own background, 2000 random pairs of its pictures, is still computed and
  reported for diagnostics; on a small or uniform library such as the demo it would put the ceiling above 1.)

When the vector lane should have run and could not (the text tower failed to load), the search is the keyword lane
alone and the result says so in `warning`.

### Similar photos

`photo.similar {id?, limit?: 50, filter?: true}` lists the pictures nearest to one photo's, best first, with their
cosines, above this query's noise ceiling (capped at `EMBED_NEAR`, above). A photo the background pass has not embedded yet is embedded on the
spot (one picture, on the session thread) when the model is installed; otherwise the command fails with "the vector
lane is not ready" and the lane's reason. With `filter`, the view shows the photo and the results in rank order
(relevance sort). `pendingEmbeddings` counts the photos not embedded yet.

On the demo library, 24 synthetic landscapes that are all alike to the model (image pairs 0.56–0.97), the lake scene's
per-query ceiling lands at 0.90 (the cap) and keeps the other lake scene (0.972) and nothing else.

### The `similar` tier of duplicates

`photo.duplicates` adds a fourth tier, `similar` (`lane: "vector"`), for photos the fingerprints do not match (or have
not fingerprinted yet) whose cosine with this one is at least `EMBED_NEAR`; at `EMBED_DUPLICATE` or over the row is
flagged `strong: true`. A crop, which the fingerprints miss, lands here: that is what the tier is for. Rows carry the
cosine, and the fingerprint distances when both photos have them (`null` otherwise). The tier comes after `possible`,
best cosine first; ignored pairs and virtual copies are left out as in the other tiers. The photo's own vector is made
on the spot like its fingerprint.

Both thresholds are measured on the shipped model (Gate 2, below): `EMBED_DUPLICATE` 0.94, `EMBED_NEAR` 0.90. The
studio's so400m values (0.82, 0.78) do not transfer: on this model they list nine different landscapes as similar to the
lake scene, six of them `strong`.

### Gate 2: the cosines of the shipped model

`crates/embed/tests/thresholds.rs` (gated on `LIGHTCRAFT_SIGLIP2_DIR`) embeds the digest tests' fixtures, every picture
squashed to 256 × 256 by area averaging, and prints every cosine; `vera_gate2_jpeg_re_exports` in the engine adds the
JPEG pairs. Measured on `siglip2-base-patch32-256`, CPU:

| Pair | Cosine |
|---|---|
| The fixture scene at 2048 / 1024 / 512 / 256 px (the six pairs) | 0.9760 – 0.9989 (2048–1024 0.9989, 2048–512 0.9978, 1024–512 0.9985, 512–256 0.9814, 1024–256 0.9795, 2048–256 0.9760) |
| 1024 px vs its 8-bit copy | 0.9997 |
| 1024 px vs its JPEG at quality 85 / 50 (4:2:0) | 0.9850 / 0.9475 |
| 1024 px vs its 70 % centre crop | 0.9677 |
| Eight unrelated scenes (28 pairs) | 0.5553 (Cosmos bloom – Canyon walls) to 0.8527 (Lake at first light – Morning fog) |

From these numbers `crates/vera/src/thresholds.rs` sets `EMBED_DUPLICATE` = 0.94 (under the weakest true pair, the
quality-50 JPEG at 0.9475; the crop at 0.9677 is caught, which is the tier's purpose) and `EMBED_NEAR` = 0.90 (above the
most alike unrelated pair, 0.853, with margin), recorded against `EMBED_THRESHOLDS_MODEL`. Procedural scenes are more
distinct than one photographer's real work, so `EMBED_NEAR` is the value to re-check first on a real catalog.

### `vera.status`

The vector lane's `state` is one of:

| State | When |
|---|---|
| `live` | The model is installed and the lane is on |
| `needs_model` | No model in the model folder (or no folder at all); the reason says where the files go and how to download them |
| `downloading` | The download is running (the reason gives the percentage) |
| `disabled_by_user` | Switched off with `vera.config` |
| `needs_desktop` | The web build |
| `error` | The model failed to load or to run; the reason carries the message |

with a sentence in `reason` for each. `vera.status` also counts `embedded`, `pendingEmbeddings`, `unembeddable` and
`embeddings` (pictures in the space), reports `model: {id, installed, loaded, memoryBytes}`, and lists both sidecar
files in `sidecars` (`[{name, path?, bytes?}]`; `sidecar` still names the digest file alone).

## In the app and over MCP

- **Search:** typing in the search field filters live as before; **Enter** runs `search.text` and the grid shows the
  results in rank order under a chip reading Ranked: “…”. The chip's ×, Clear Filters or typing again clear the ranking
  and put the earlier sort back. A search that finds nothing says so and leaves the view alone.
- **Duplicates:** Photo ▸ Show Duplicates (`photo.showDuplicates`, also on the grid's context menu) shows a photo and
  its duplicates under a "Duplicates of …" chip, or says none were found and how many photos are not fingerprinted yet.
- **Import:** the review badges "Same picture" and "Possibly the same" (the tooltip names the library photo and gives
  both distances) and offers "Same pictures: Stack under the original / Import anyway / Skip", remembered for the
  session; the finish message counts what was stacked.
- **Background work:** the app pumps `vera.pump` every frame with the face scan's pace; `ui.inspect` reports it as
  `vera {pending, digested, pace}`.
- **MCP:** helper tools `search_text`, `photo_duplicates` and `vera_status`, and `import.onNearDuplicate`; everything
  else through `run_command` (see [mcp.md](mcp.md)).

## Commands

| Command | Params → result |
|---|---|
| `vera.status` | `{}` → `{digested, photos, pendingDigests, unreadable, digests, embedded, pendingEmbeddings, unembeddable, embeddings, model: {id, installed, loaded, memoryBytes}, indexDocs, ignoredPairs, lanes: {keyword, digest, vector: {state, reason}, describe: {state, reason}}, sidecar: {path?, bytes?}, sidecars: [{name, path?, bytes?}]}` |
| `vera.pump` | `{pace?: pause\|light\|normal\|full}` → `{active, pending, digested, photos, pace, embedding: {state, pending, embedded, photos, inFlight}}`; not journaled |
| `vera.digest` | `{ids?}` (the selection) → `{digested, already, failed: [{id, error}]}`; fingerprint these photos now; not journaled |
| `search.text` | `{query, limit?: 200, filter?: true}` → `{photos: [{id, score, lane: keyword\|vector\|both, cosine?, keywordScore?}], cutoff, lanes, filtered, warning?}`. Without the vector lane: `lanes: ["keyword"]`, `cutoff: null`, `score` the BM25 score (as in Phase 1); with it: `lanes: ["keyword", "vector"]`, `score` the fused one, `cutoff` the vector lane's ceiling. With `filter` the grid shows the results in rank order (sort key `relevance`); a search that finds nothing leaves the view alone |
| `photo.similar` | `{id?, limit?: 50, filter?: true}` → `{id, photos: [{id, cosine, lane: "vector"}], cutoff, embedded, pendingEmbeddings, filtered}`; an error when the vector lane is not ready |
| `photo.duplicates` | `{id?, limit?: 50, filter?: false}` → `{id, photos: [{id, tier: bytes\|picture\|possible\|similar, pdq, phash, lane?, cosine?, strong?}], ignored, digested, pendingDigests, embedded, pendingEmbeddings}`; rows by tier, then cosine (similar) or PDQ distance |
| `vera.model.status` | `{}` → `{installed, dir?, downloading?: {done, total, file}, downloadError?, licence: {name, url, notice}, modelId, bytes, loaded, state}`; not journaled |
| `vera.model.download` | `{acceptLicense: true}` → `{started, installed, downloading}`; an error naming the licence without `acceptLicense`; not journaled |
| `vera.model.cancel` | `{}` → `{cancelled}`; not journaled |
| `vera.model.remove` | `{}` → `{removed, files, sidecar, installed}`: the model files and this library's embeddings; not journaled |
| `vera.config` | `{vectorLane?: bool}` → `{vectorLane, state}`; saved in the library's `prefs.json`; not journaled |
| `photo.ignorePair` / `photo.unignorePair` | `{a, b}` → `{changed, ignoredPairs}` |
| `photo.stackDuplicate` | `{id, under}` → `{stack, count}`; undoable |
| `library.import` | adds `onNearDuplicate?: stack\|import\|skip`; the report adds `stacked?` and the `picture` reason |
| `library.importPreview` | candidates add `nearDuplicate?` |

Every parameter is checked: wrong types are errors, limits and id lists are capped at 10,000, a query at 2,000
characters.

The `relevance` sort (`SortKey::Relevance` in `crates/catalog/src/query.rs`) orders the grid by the photo's place in the
view's `only` list (the search's ranking), photos it does not list last; `library.sort {key: "relevance"}` selects it
like any other key.

## Files beside the library

All are rebuildable or small, and all live in the library folder; an in-memory library keeps them for the session only.

- **`vera-digests.bin`**: the fingerprints. Little-endian: `"LCVD"`, version `1`, then records of `key_len u16`, the
  content key (UTF-8), `pdq [u8; 32]`, `pdq_quality u8`, `phash u64`, `source u8`. `source` names what the fingerprint
  was made from (`1`: thumbnail-level luma); a change of rendition bumps it, and records made from another one are
  dropped. New records are appended (every three seconds while the pump works, and when it finishes). A file that is
  damaged, cut short, of another format or over 1 GiB is ignored and written again whole; when a library opens, records
  of photos no longer in its catalog are dropped and the file is rewritten without them.
- **`vera-embeddings-siglip2-base-patch32-256.bin`**: the vector lane's embeddings, one per picture (format above, under
  *The embedding sidecar*); rebuildable like the fingerprints, deleted by `vera.model.remove`.
- **`vera-ignored.json`**: the pairs set apart with `photo.ignorePair`, as pairs of content keys
  (`{"version": 1, "pairs": [[a, b], …]}`, read up to 1 MiB, at most 10,000 pairs, written atomically).

### What is not in the op log, and why

None of these files is part of the catalog or its journal. Fingerprints and embeddings are derived data: rebuilt from the pictures at any
time, and tied to a rendition that may change. The ignored pairs are a user decision, and so arguably belong in the
catalog, but they are kept outside the op log for now: `photo.ignorePair` is not an undo step and the pairs do not
travel with a catalog snapshot. Stacking a duplicate (`photo.stackDuplicate`, the import's `stack`) does go through the
op log, like any stack.

## Plan

The phases of the PRD this work follows, each gated on a measured proof rather than a date:

| Phase | What | State |
|---|---|---|
| 1 | Keyword lane ranked (`search.text`, relevance sort); digest lane: background fingerprints, `photo.duplicates`, ignored pairs, stacking; the import's same-picture check | done: engine, app and MCP (below) |
| 2 | Vector lane: SigLIP 2 Base patch32-256 as an opt-in download (`crates/embed`, the choice in [vera-models.md](vera-models.md)), vectors beside the library, the noise cutoff and rank fusion with the keyword lane, `photo.similar`, crops found as duplicates | engine done (this page); Gate 2 measured and the thresholds set; UI and MCP surfaces following |
| 3 | Describe: words proposed for photos that have none, by a captioner the user points at (local when a licensed one fits, else an endpoint they configure), reviewed before anything is kept | planned |
| 4 | The web build: keyword and digest lanes in the browser; the vector lane reports `needs_desktop` until a browser-runnable model is proven | planned |

## Open questions

- **The embedding thresholds on real photographs.** 0.94 / 0.90 come from procedural fixtures; one photographer's real
  catalog sits closer together. Re-check `EMBED_NEAR` there before trusting the `similar` tier's lower edge.
- **A calibrated text cutoff.** SigLIP's logit scale and bias (a probability per query–picture pair) are a model-native
  alternative for text queries; `lightcraft-embed` does not expose them yet.

- **NFKC.** The keyword lane folds case but not compatibility forms (full-width letters, ligatures): the standard
  library has no table for it, and a crate for it is a dependency decision.
- **An op for ignored pairs.** Should `photo.ignorePair` be a catalog op (undoable, journaled, part of a snapshot)? It
  would cost a catalog format change for a small, rarely changed set.
- **A worker thread for fingerprinting.** The pump decodes on the session thread, a picture or a few per frame. A worker
  (like the face scan's) would keep frames smooth on slow files (large raws, network shares) at the cost of the
  plumbing the face scan has.
- **Incremental index updates.** The keyword index is rebuilt whole after any catalog change; `Index::upsert` and
  `remove` could follow the ops instead once a library is large enough for the rebuild to show.
