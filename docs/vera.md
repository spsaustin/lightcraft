# VERA: finding photos by words, and the same picture twice

VERA is how LightCraft answers two questions about a library: *which photos are about this?* (search) and *is this the
same picture as one I already have?* (duplicates). The design comes from the VERA studio, a search system built for one
photographer's archive: its lanes, its measured thresholds and the fixtures that chose them. Here it is two halves:

- **`lightcraft-vera`** (`crates/vera`, layer 2): the algorithms, as pure functions over slices. A BM25 keyword index,
  the PDQ and pHash digests with the same-picture decision, reciprocal rank fusion and the noise cutoff. No files, no
  threads, no models, no knowledge of the catalog.
- **The engine** (`crates/engine/src/vera/`): what to digest and when, where the results are kept, and the commands that
  show them. Everything is a command, so the UI, the CLI, the control channel and MCP all see the same thing.

## Lanes

A search runs on *lanes*, each a different way of finding a photo; their ranked lists are fused into one. Phase 1 has
two live lanes.

| Lane | What it finds | State in Phase 1 |
|---|---|---|
| **keyword** | Photos whose words match: title, keywords (every level of `travel\|Italy\|Rome`), caption, alt text, extended description, file name, camera, lens and place | **live** |
| **digest** | The same picture under other bytes: a resize, a re-export, a format change | **live** |
| **vector** | Photos by meaning ("a dog on a beach"), and crops the digests cannot catch | needs a model (Phase 2) |
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

## Commands

| Command | Params → result |
|---|---|
| `vera.status` | `{}` → `{digested, photos, pendingDigests, unreadable, digests, indexDocs, ignoredPairs, lanes: {keyword, digest, vector: {state, reason}, describe: {state, reason}}, sidecar: {path?, bytes?}}` |
| `vera.pump` | `{pace?: pause\|light\|normal\|full}` → `{active, pending, digested, photos, pace}`; not journaled |
| `vera.digest` | `{ids?}` (the selection) → `{digested, already, failed: [{id, error}]}`; fingerprint these photos now; not journaled |
| `search.text` | `{query, limit?: 200, filter?: true}` → `{photos: [{id, score, lane: "keyword"}], cutoff: null, lanes: ["keyword"], filtered}`. With `filter` the grid shows the results in rank order (sort key `relevance`); a search that finds nothing leaves the view alone |
| `photo.duplicates` | `{id?, limit?: 50, filter?: false}` → `{id, photos: [{id, tier: bytes\|picture\|possible, pdq, phash}], ignored, digested, pendingDigests}`; rows by tier, then PDQ distance |
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

Both are rebuildable or small, and both live in the library folder; an in-memory library keeps them for the session only.

- **`vera-digests.bin`**: the fingerprints. Little-endian: `"LCVD"`, version `1`, then records of `key_len u16`, the
  content key (UTF-8), `pdq [u8; 32]`, `pdq_quality u8`, `phash u64`, `source u8`. `source` names what the fingerprint
  was made from (`1`: thumbnail-level luma); a change of rendition bumps it, and records made from another one are
  dropped. New records are appended (every three seconds while the pump works, and when it finishes). A file that is
  damaged, cut short, of another format or over 1 GiB is ignored and written again whole; when a library opens, records
  of photos no longer in its catalog are dropped and the file is rewritten without them.
- **`vera-ignored.json`**: the pairs set apart with `photo.ignorePair`, as pairs of content keys
  (`{"version": 1, "pairs": [[a, b], …]}`, read up to 1 MiB, at most 10,000 pairs, written atomically).

### What is not in the op log, and why

Neither file is part of the catalog or its journal. Fingerprints are derived data: rebuilt from the pictures at any
time, and tied to a rendition that may change. The ignored pairs are a user decision, and so arguably belong in the
catalog, but they are kept outside the op log for now: `photo.ignorePair` is not an undo step and the pairs do not
travel with a catalog snapshot. Stacking a duplicate (`photo.stackDuplicate`, the import's `stack`) does go through the
op log, like any stack.

## Plan

The phases of the PRD this work follows, each gated on a measured proof rather than a date:

| Phase | What | State |
|---|---|---|
| 1 | Keyword lane ranked (`search.text`, relevance sort); digest lane: background fingerprints, `photo.duplicates`, ignored pairs, stacking; the import's same-picture check | this page (engine); UI and MCP surfaces following |
| 2 | Vector lane: SigLIP 2 Base patch32-256 as an opt-in download (`crates/embed`, the choice in [vera-models.md](vera-models.md)), vectors beside the library, the noise cutoff and rank fusion with the keyword lane, `photo.similar`, crops found as duplicates | model and tokenizer ported and proven against the reference; engine worker next |
| 3 | Describe: words proposed for photos that have none, by a captioner the user points at (local when a licensed one fits, else an endpoint they configure), reviewed before anything is kept | planned |
| 4 | The web build: keyword and digest lanes in the browser; the vector lane reports `needs_desktop` until a browser-runnable model is proven | planned |

## Open questions

- **NFKC.** The keyword lane folds case but not compatibility forms (full-width letters, ligatures): the standard
  library has no table for it, and a crate for it is a dependency decision.
- **An op for ignored pairs.** Should `photo.ignorePair` be a catalog op (undoable, journaled, part of a snapshot)? It
  would cost a catalog format change for a small, rarely changed set.
- **A worker thread for fingerprinting.** The pump decodes on the session thread, a picture or a few per frame. A worker
  (like the face scan's) would keep frames smooth on slow files (large raws, network shares) at the cost of the
  plumbing the face scan has.
- **Incremental index updates.** The keyword index is rebuilt whole after any catalog change; `Index::upsert` and
  `remove` could follow the ops instead once a library is large enough for the rebuild to show.
