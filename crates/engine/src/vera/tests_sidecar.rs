//! VERA's sidecar files through a library's file store (the browser build's storage), and the web pump's time budget.

use std::collections::HashSet;
use std::time::Duration;

use lightcraft_vera::Digest;
use serde_json::json;
use web_time::Instant;

use super::sidecar::fake::FakeFiles;
use super::sidecar::{Loaded, Sidecars};
use super::*;
use crate::library::LibraryStores;

fn d(n: u8) -> Digest {
    Digest { pdq: [n; 32], pdq_quality: 80, phash: u64::from(n) * 0x0101_0101 }
}

/// A session over a library whose files (catalog and settings) live in `files`, as in the browser.
fn session_in(files: &FakeFiles) -> Session {
    let mut s = Session::new();
    let stores = LibraryStores {
        dir: "browser:fake/library".into(),
        catalog: Box::new(lightcraft_catalog::MemStore::default()),
        files: Box::new(files.clone()),
        on_disk: false,
    };
    s.open_library_in(stores, true).unwrap();
    s
}

#[test]
fn digests_are_written_appended_reloaded_and_compacted_through_the_store() {
    let files = FakeFiles::default();
    let mut st = digests::DigestStore::default();
    st.reset(Sidecars::Store(files.shared()), None);
    for (k, n) in [("keep", 1), ("gone", 2)] {
        st.insert(k.into(), d(n));
    }
    st.save().unwrap();
    // the first save writes the file whole (there was none), with its header
    assert_eq!((files.rewrites(digests::FILE_NAME), files.appends(digests::FILE_NAME)), (1, 0));
    let first = files.get(digests::FILE_NAME).unwrap();
    assert!(first.starts_with(b"LCVD\x01"), "{first:?}");
    assert_eq!(st.file_len(), Some(first.len() as u64));
    // later ones append only the new records: never the whole file again
    st.insert("also".into(), d(3));
    st.save().unwrap();
    assert_eq!((files.rewrites(digests::FILE_NAME), files.appends(digests::FILE_NAME)), (1, 1));
    let second = files.get(digests::FILE_NAME).unwrap();
    assert_eq!(second.len() - first.len(), 2 + 4 + 42, "one record appended");
    assert_eq!(second.get(..first.len()), Some(&first[..]));
    drop(st);
    // a reload (a new page over the same storage) reads every record back, and appends again
    let reloaded = files.reloaded();
    let mut st = digests::DigestStore::default();
    st.reset(Sidecars::Store(reloaded.shared()), None);
    assert_eq!((st.len(), st.get("also"), st.has_unsaved()), (3, Some(&d(3)), false));
    st.insert("more".into(), d(4));
    st.save().unwrap();
    assert_eq!((reloaded.rewrites(digests::FILE_NAME), reloaded.appends(digests::FILE_NAME)), (0, 1));
    drop(st);
    // a photo removed since: its record is dropped, and the file compacted once (a whole rewrite)
    let live: HashSet<String> = ["keep", "also", "more"].iter().map(|k| k.to_string()).collect();
    let compacted = reloaded.reloaded();
    let mut st = digests::DigestStore::default();
    st.reset(Sidecars::Store(compacted.shared()), Some(&live));
    assert_eq!(st.len(), 3);
    st.save().unwrap();
    assert_eq!((compacted.rewrites(digests::FILE_NAME), compacted.appends(digests::FILE_NAME)), (1, 0));
    assert!(compacted.get(digests::FILE_NAME).unwrap().len() < reloaded.get(digests::FILE_NAME).unwrap().len());
    drop(st);
    let mut st = digests::DigestStore::default();
    st.reset(Sidecars::Store(compacted.reloaded().shared()), None);
    assert_eq!(st.len(), 3);
    assert!(!st.contains("gone"));
}

#[test]
fn a_failing_store_keeps_digests_pending_until_it_works() {
    let files = FakeFiles::default();
    let mut st = digests::DigestStore::default();
    st.reset(Sidecars::Store(files.shared()), None);
    st.insert("a".into(), d(1));
    st.save().unwrap();
    files.set_fail(true);
    st.insert("b".into(), d(2));
    assert!(st.save().is_err(), "a full quota is reported");
    assert!(st.has_unsaved(), "the record stays pending");
    files.set_fail(false);
    st.save().unwrap();
    assert!(!st.has_unsaved());
    let mut again = digests::DigestStore::default();
    again.reset(Sidecars::Store(files.reloaded().shared()), None);
    assert_eq!(again.get("b"), Some(&d(2)));
}

#[test]
fn embeddings_append_through_the_store() {
    let files = FakeFiles::default();
    let name = embeddings::file_name("test-model");
    let v = |n: usize| -> Vec<f32> { (0..8).map(|i| if i == n { 1.0 } else { 0.1 }).collect() };
    let mut st = embeddings::EmbeddingStore::new("test-model", 8);
    st.reset(Sidecars::Store(files.shared()), None);
    assert!(st.insert("a".into(), &v(0)));
    st.save().unwrap();
    assert!(st.insert("b".into(), &v(1)));
    assert!(st.insert_role("b".into(), embeddings::ROLE_CONTEXTUAL, &v(2)));
    st.save().unwrap();
    assert_eq!((files.rewrites(&name), files.appends(&name)), (1, 1), "written once, then appended");
    assert_eq!(st.file_len(), files.get(&name).map(|b| b.len() as u64));
    drop(st);
    let reloaded = files.reloaded();
    let mut st = embeddings::EmbeddingStore::new("test-model", 8);
    st.reset(Sidecars::Store(reloaded.shared()), None);
    assert_eq!((st.len(), st.role_count(embeddings::ROLE_CONTEXTUAL), st.has_unsaved()), (2, 1, false));
    // vera.model.remove: the store has no delete, so the file is emptied (and taken for a foreign one next time)
    assert!(st.delete_file().unwrap());
    assert_eq!(reloaded.get(&name), Some(Vec::new()));
    assert!(!st.delete_file().unwrap(), "nothing left to remove");
    let mut st = embeddings::EmbeddingStore::new("test-model", 8);
    st.reset(Sidecars::Store(reloaded.reloaded().shared()), None);
    assert!(st.is_empty());
}

#[test]
fn ignored_pairs_and_proposals_go_through_the_store() {
    let files = FakeFiles::default();
    let mut pairs = IgnoredPairs::load(Sidecars::Store(files.shared()));
    assert_eq!(pairs.len(), 0);
    assert!(pairs.insert("k1", "k2").unwrap());
    pairs.save().unwrap();
    let again = IgnoredPairs::load(Sidecars::Store(files.reloaded().shared()));
    assert!(again.contains("k2", "k1"));
    // a damaged file: no pairs, and no panic
    let damaged = files.reloaded();
    {
        let mut st = damaged.clone();
        lightcraft_catalog::Store::write_atomic(&mut st, IGNORED_FILE, b"{not json").unwrap();
    }
    assert_eq!(IgnoredPairs::load(Sidecars::Store(damaged.shared())).len(), 0);

    let live: HashSet<u64> = [1, 2].into_iter().collect();
    let mut queue = review::Review::load(Sidecars::Store(files.shared()), &live);
    for photo in [1, 2] {
        assert!(queue.insert(review::Proposal { photo, at: "2026-10-11T00:00:00Z".into(), ..Default::default() }));
    }
    queue.save().unwrap();
    assert_eq!(files.rewrites(review::FILE_NAME), 1);
    drop(queue);
    let queue = review::Review::load(Sidecars::Store(files.reloaded().shared()), &live);
    assert_eq!(queue.len(), 2);
    drop(queue);
    // photo 2 is gone: its proposal is dropped, and the queue rewritten without it when it is next saved
    let fewer: HashSet<u64> = [1].into_iter().collect();
    let store = files.reloaded();
    let mut queue = review::Review::load(Sidecars::Store(store.shared()), &fewer);
    assert_eq!(queue.len(), 1);
    queue.save().unwrap();
    assert_eq!(store.rewrites(review::FILE_NAME), 1);
    let queue = review::Review::load(Sidecars::Store(store.reloaded().shared()), &live);
    assert_eq!((queue.len(), queue.get(1).is_some(), queue.get(2).is_some()), (1, true, false));
}

#[test]
fn reading_a_store_file_respects_the_cap() {
    let files = FakeFiles::default();
    let sc = Sidecars::Store(files.shared());
    assert!(matches!(sc.read("x.bin", 4), Loaded::Missing));
    sc.write_atomic("x.bin", b"12345").unwrap();
    assert!(matches!(sc.read("x.bin", 4), Loaded::TooLarge));
    assert!(matches!(sc.read("x.bin", 5), Loaded::Bytes(b) if b == b"12345"));
    assert_eq!((sc.len("x.bin"), sc.exists("x.bin"), sc.path_of("x.bin")), (Some(5), true, None));
    assert!(sc.describe("x.bin").contains("fake browser storage"));
    // the same store is the same place; another store, a folder or memory is not
    assert!(sc.same_place(&sc.clone()));
    assert!(!sc.same_place(&Sidecars::Store(files.shared())));
    assert!(!sc.same_place(&Sidecars::Memory) && Sidecars::Memory.same_place(&Sidecars::Memory));
    assert!(!Sidecars::Memory.persists() && sc.persists());
}

/// A library that is not a folder (the browser's) keeps VERA's files in its own store: a new session over the same
/// storage has every digest and decision without fingerprinting again.
#[test]
fn a_store_library_keeps_vera_files_across_sessions() {
    let files = FakeFiles::default();
    let mut s = session_in(&files);
    let photos = s.catalog.photos().filter(|p| eligible(p)).count();
    assert!(photos >= 3);
    for _ in 0..1000 {
        let r = s.execute("vera.pump", &json!({"pace": "full"})).unwrap();
        if r["pending"] == 0 {
            break;
        }
    }
    let st = s.execute("vera.status", &json!({})).unwrap();
    let digested = st["digested"].as_u64().unwrap();
    assert!(digested > 0 && st["pendingDigests"] == 0, "{st}");
    // the status says where the file is: in the library's store, not a path
    assert_eq!(st["sidecar"], json!({}), "{st}");
    assert!(st["sidecars"][0]["store"].as_str().is_some_and(|w| w.contains("vera-digests.bin")), "{st}");
    assert!(st["sidecars"][0]["bytes"].as_u64().is_some_and(|n| n > 6), "{st}");
    let ids: Vec<u64> = s.catalog.photos().filter(|p| eligible(p)).map(|p| p.id.0).take(2).collect();
    s.execute("photo.ignorePair", &json!({"a": ids[0], "b": ids[1]})).unwrap();
    assert!(files.get(digests::FILE_NAME).is_some() && files.get(IGNORED_FILE).is_some());
    s.close_library().unwrap();
    drop(s);

    // a fresh session (its catalog seeded again with the same demo photos: the same content keys)
    let reloaded = files.reloaded();
    let mut s = session_in(&reloaded);
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((st["digested"].as_u64(), st["ignoredPairs"].as_u64()), (Some(digested), Some(1)), "{st}");
    assert_eq!(st["pendingDigests"], 0, "nothing to fingerprint again: {st}");
    // reading them back wrote nothing
    assert_eq!((reloaded.rewrites(digests::FILE_NAME), reloaded.appends(digests::FILE_NAME)), (0, 0));
}

#[test]
fn a_library_on_disk_still_uses_its_folder() {
    let dir = std::env::temp_dir().join(format!("lc-vera-sidecar-dir-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut s = Session::new();
    s.open_library(&dir, true).unwrap();
    assert!(matches!(sidecar::of(&s), Sidecars::Dir(ref d) if *d == dir));
    s.close_library().unwrap();
    drop(s);
    // no library: memory only
    assert!(matches!(sidecar::of(&Session::new()), Sidecars::Memory));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_web_step_budget_stops_starting_pictures_after_8_ms() {
    let t0 = Instant::now();
    let ms = |n: u64| t0 + Duration::from_millis(n);
    // the browser: the pace's count, cut at the budget
    let mut step = Step::new(4, Some(WEB_STEP_BUDGET), t0);
    assert!(step.may_start(ms(0)));
    step.started_one();
    assert!(step.may_start(ms(3)), "under the budget");
    step.started_one();
    assert!(!step.may_start(ms(8)), "the budget is spent");
    assert!(!step.may_start(ms(50)));
    // one slow picture never blocks the first of a call
    let step = Step::new(4, Some(WEB_STEP_BUDGET), t0);
    assert!(step.may_start(ms(100)));
    // the count still bounds a fast step
    let mut step = Step::new(2, Some(WEB_STEP_BUDGET), t0);
    step.started_one();
    step.started_one();
    assert!(!step.may_start(ms(0)));
    // the desktop: no budget, the count alone
    let mut step = Step::new(2, None, t0);
    step.started_one();
    assert!(step.may_start(ms(10_000)));
    step.started_one();
    assert!(!step.may_start(ms(0)));
    // pause starts nothing
    assert!(!Step::new(0, None, t0).may_start(ms(0)));
    assert_eq!(step_budget(), if cfg!(target_arch = "wasm32") { Some(WEB_STEP_BUDGET) } else { None });
}
