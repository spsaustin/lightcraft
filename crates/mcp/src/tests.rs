use serde_json::{Value, json};

use crate::{Backend, Headless, PROTOCOL_VERSION, Server, call_tool, command_tool_name};

fn server() -> Server {
    Server::new(Box::new(Headless::demo()))
}

fn rpc(s: &mut Server, id: u64, method: &str, params: Value) -> Value {
    let line = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string();
    let reply = s.handle_line(&line).expect("reply");
    let v: Value = serde_json::from_str(&reply).unwrap();
    assert_eq!(v["jsonrpc"], "2.0");
    assert_eq!(v["id"], id);
    v
}

#[test]
fn initialize_negotiates_version() {
    let mut s = server();
    let r = rpc(&mut s, 1, "initialize", json!({"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}));
    assert_eq!(r["result"]["protocolVersion"], "2025-03-26");
    assert_eq!(r["result"]["serverInfo"]["name"], "lightcraft");
    assert!(r["result"]["capabilities"]["tools"].is_object());
    let r = rpc(&mut s, 2, "initialize", json!({"protocolVersion": "1999-01-01"}));
    assert_eq!(r["result"]["protocolVersion"], PROTOCOL_VERSION);
    assert!(s.handle_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());
    assert!(s.is_initialized());
}

#[test]
fn errors() {
    let mut s = server();
    let v: Value = serde_json::from_str(&s.handle_line("{not json").unwrap()).unwrap();
    assert_eq!(v["error"]["code"], -32700);
    assert_eq!(rpc(&mut s, 1, "nope", json!({}))["error"]["code"], -32601);
    assert_eq!(rpc(&mut s, 2, "tools/call", json!({}))["error"]["code"], -32602);
    assert!(s.handle_line("   ").is_none());
    // Unknown tools and failing commands are tool errors, not protocol errors.
    let r = rpc(&mut s, 3, "tools/call", json!({"name": "nope", "arguments": {}}));
    assert_eq!(r["result"]["isError"], true);
    let r = rpc(&mut s, 4, "tools/call", json!({"name": "run_command", "arguments": {"command": "no.such"}}));
    assert_eq!(r["result"]["isError"], true);
    assert!(r["result"]["content"][0]["text"].as_str().unwrap().contains("unknown command"));
}

#[test]
fn tools_list_has_helpers_and_every_command() {
    let mut s = server();
    let r = rpc(&mut s, 1, "tools/list", json!({}));
    let tools = r["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    for n in ["list_commands", "run_command", "import", "set_develop", "render_photo", "export"] {
        assert!(names.contains(&n), "{n}");
    }
    // Headless: no UI tools.
    assert!(!names.contains(&"screenshot"));
    let hl = Headless::demo();
    for c in hl.session.commands() {
        assert!(!c.id.contains('_'), "command ids must not contain `_` ({})", c.id);
        let n = command_tool_name(c.id);
        assert!(names.contains(&n.as_str()), "{n}");
        assert!(n.len() <= 64 && n.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_'), "{n}");
    }
    assert!(names.contains(&"cmd_app_export"));
    for t in tools {
        assert_eq!(t["inputSchema"]["type"], "object");
    }
    // Compact mode keeps only the helpers.
    let mut s = Server::new(Box::new(Headless::demo())).with_command_tools(false);
    let r = rpc(&mut s, 1, "tools/list", json!({}));
    assert!(r["result"]["tools"].as_array().unwrap().iter().all(|t| !t["name"].as_str().unwrap().starts_with("cmd_")));
}

/// The `import` helper passes the copy options through: a folder template files the copy.
#[test]
fn import_tool_copies_with_a_folder_template() {
    let base = std::env::temp_dir().join(format!("lc-mcp-import-tpl-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (src, dest) = (base.join("card"), base.join("out"));
    std::fs::create_dir_all(&src).unwrap();
    let img = lightcraft_raster::Rgba8 { width: 8, height: 8, data: vec![[20, 3, 9, 255]; 64] };
    let png = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &lightcraft_codecs::EncodeMeta::default()).unwrap();
    std::fs::write(src.join("a.png"), png).unwrap();
    let mut b = Headless::demo();
    b.session.clock = Box::new(|| "2026-01-14T05:58:48".to_string());
    let args =
        json!({"paths": [src.to_string_lossy()], "mode": "copy", "destination": dest.to_string_lossy(), "organize": "{date:%Y}/{date:%Y%m%d}"});
    let r = call_tool(&mut b, "import", &args);
    assert!(!r.is_error, "{r:?}");
    assert!(dest.join("2026").join("20260114").join("a.png").is_file());
    let _ = std::fs::remove_dir_all(&base);
}

/// Issue #93: `render_photo`, `ui.render` and `app.export` with an exact `path` must not replace
/// a photo's original.
#[test]
fn path_writes_never_replace_an_original() {
    let base = std::env::temp_dir().join(format!("lc-mcp-guard-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let img = lightcraft_raster::Rgba8 { width: 8, height: 8, data: vec![[90, 30, 9, 255]; 64] };
    let png = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &lightcraft_codecs::EncodeMeta::default()).unwrap();
    let orig = base.join("a.png");
    std::fs::write(&orig, &png).unwrap();
    let mut b = Headless::demo();
    let r = call_tool(&mut b, "import", &json!({"paths": [orig.to_string_lossy()]}));
    assert!(!r.is_error, "{r:?}");
    let id = b.session.catalog.photos().find(|p| p.file_name == "a.png").unwrap().id.0;
    let path = orig.to_string_lossy().to_string();
    let r = call_tool(&mut b, "render_photo", &json!({"id": id, "path": path}));
    assert!(r.is_error, "{r:?}");
    assert!(b.call("ui.render", json!({"id": id, "path": path})).unwrap_err().contains("original"));
    let e = b.call("app.export", json!({"ids": [id], "path": path})).unwrap_err();
    assert!(e.contains("never writes over an original"), "{e}");
    assert_eq!(std::fs::read(&orig).unwrap(), png, "the original is untouched");
    // another path still works
    let r = call_tool(&mut b, "render_photo", &json!({"id": id, "path": base.join("render.png").to_string_lossy()}));
    assert!(!r.is_error, "{r:?}");
    let _ = std::fs::remove_dir_all(&base);
}

/// Issue #181: a misspelled or unreadable `app.export` param is an error through the headless
/// backend (what `lightcraft-cli run` and the MCP `export` tool use), not a silent default.
#[test]
fn export_refuses_unknown_params_and_bad_values() {
    let mut b = Headless::demo();
    let out = "/nonexistent-lc-test/x.jpg";
    let e = b.call("app.export", json!({"path": out, "longEdgee": 400})).unwrap_err();
    assert!(e.contains("unknown parameter `longEdgee` (did you mean `longEdge`?)"), "{e}");
    let e = b.call("app.export", json!({"path": out, "longEdge": "banana"})).unwrap_err();
    assert!(e.contains("`longEdge` must be a number"), "{e}");
    let e = b.call("app.export", json!({"path": out, "watermark": {"text": "x", "size": 3}})).unwrap_err();
    assert!(e.contains("`watermark.size` must be a number 0.005..0.5"), "{e}");
    // `export` tool and preset expansion go through the same check
    let r = call_tool(&mut b, "export", &json!({"path": out, "quality": 101}));
    assert!(r.is_error, "{r:?}");
    let e = b.session.execute("export.savePreset", &json!({"name": "Typo", "params": {"qualty": 5}})).unwrap_err().to_string();
    assert!(e.contains("did you mean `quality`"), "{e}");
}

/// Every option the `export` tool advertises reaches `app.export` (they were accepted, then dropped).
#[test]
fn export_tool_forwards_its_advertised_options() {
    let mut b = Headless::demo();
    let base = std::env::temp_dir().join(format!("lc-mcp-export-fwd-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let r = call_tool(&mut b, "export", &json!({"dir": base.to_string_lossy(), "subfolder": "sub", "longEdge": 64, "format": "png"}));
    assert!(!r.is_error, "{r:?}");
    let files: Vec<_> = std::fs::read_dir(base.join("sub")).unwrap().collect();
    assert_eq!(files.len(), 1, "written into the subfolder");
    let _ = std::fs::remove_dir_all(&base);
}

/// The `import` helper moves: renamed into the folder template, the source removed.
#[test]
fn import_tool_moves_with_a_folder_template() {
    let base = std::env::temp_dir().join(format!("lc-mcp-import-move-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (src, dest) = (base.join("card"), base.join("out"));
    std::fs::create_dir_all(&src).unwrap();
    let img = lightcraft_raster::Rgba8 { width: 8, height: 8, data: vec![[21, 3, 9, 255]; 64] };
    let png = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &lightcraft_codecs::EncodeMeta::default()).unwrap();
    std::fs::write(src.join("sample.png"), png).unwrap();
    let mut b = Headless::demo();
    b.session.clock = Box::new(|| "2026-01-14T05:58:48".to_string());
    let args = json!({"paths": [src.to_string_lossy()], "mode": "move", "destination": dest.to_string_lossy(),
        "organize": "{date:%Y}/{date:%Y%m%d}", "rename": "{date:%Y%m%d}_{seq:3}"});
    let r = call_tool(&mut b, "import", &args);
    assert!(!r.is_error, "{r:?}");
    assert!(dest.join("2026").join("20260114").join("20260114_001.png").is_file());
    assert!(!src.join("sample.png").exists(), "moved");
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn command_tools_run_commands() {
    let mut b = Headless::demo();
    let first = b.session.visible_cloned()[0];
    let r = call_tool(&mut b, "cmd_library_select", &json!({"ids": [first.0]}));
    assert!(!r.is_error, "{r:?}");
    let r = call_tool(&mut b, "cmd_photo_rate", &json!({"rating": 4}));
    assert!(!r.is_error, "{r:?}");
    assert_eq!(b.session.catalog.photo(first).unwrap().rating, 4);
    let r = call_tool(&mut b, "cmd_edit_undo", &json!({}));
    assert!(!r.is_error);
    assert_eq!(b.session.catalog.photo(first).unwrap().rating, 0);
}

#[test]
fn set_develop_values_and_settings() {
    let mut b = Headless::demo();
    let first = b.session.visible_cloned()[0];
    let r = call_tool(&mut b, "set_develop", &json!({"id": first.0, "values": {"light.exposure": 1.25}}));
    assert!(!r.is_error, "{r:?}");
    assert_eq!(r.structured.as_ref().unwrap()["controls"][0]["value"], 1.25);
    assert_eq!(b.session.develop_of(first).unwrap().light.exposure, 1.25);
    let r = call_tool(&mut b, "set_develop", &json!({"settings": {"light": {"contrast": 30.0}}}));
    assert!(!r.is_error, "{r:?}");
    let d = b.session.develop_of(first).unwrap();
    assert_eq!((d.light.exposure, d.light.contrast), (1.25, 30.0));
    assert!(call_tool(&mut b, "set_develop", &json!({})).is_error);
    assert!(call_tool(&mut b, "set_develop", &json!({"values": {"no.such": 1}})).is_error);
}

#[test]
fn crop_needs_a_rect_angle_or_reset() {
    let mut b = Headless::demo();
    let first = b.session.visible_cloned()[0];
    // A guessed parameter must not succeed silently.
    let r = call_tool(&mut b, "crop", &json!({"id": first.0, "aspect": "1:1"}));
    assert!(r.is_error, "{r:?}");
    assert!(call_tool(&mut b, "crop", &json!({"id": first.0, "reset": false})).is_error);
    assert!(!call_tool(&mut b, "crop", &json!({"id": first.0, "rect": [0.1, 0.0, 0.9, 1.0]})).is_error);
    assert!(!call_tool(&mut b, "crop", &json!({"id": first.0, "reset": true})).is_error);
}

#[test]
fn ui_tools_need_the_app() {
    let mut b = Headless::demo();
    let r = call_tool(&mut b, "screenshot", &json!({}));
    assert!(r.is_error);
    assert!(r.content[0]["text"].as_str().unwrap().contains("--connect"));
}

#[test]
fn resources() {
    let mut s = server();
    let r = rpc(&mut s, 1, "resources/list", json!({}));
    let list = r["result"]["resources"].as_array().unwrap();
    assert_eq!(list.len(), 7);
    for (i, res) in list.iter().enumerate() {
        let uri = res["uri"].as_str().unwrap();
        let r = rpc(&mut s, 10 + i as u64, "resources/read", json!({"uri": uri}));
        let text = r["result"]["contents"][0]["text"].as_str().unwrap_or_else(|| panic!("{uri}: {r}"));
        serde_json::from_str::<Value>(text).unwrap();
    }
    assert_eq!(rpc(&mut s, 99, "resources/read", json!({"uri": "lightcraft://nope"}))["error"]["code"], -32002);
}

/// `select_photos` with an id that is not in the library is a tool error, and the photo that was
/// active stays active (#182).
#[test]
fn select_photos_rejects_unknown_ids_and_keeps_the_active_photo() {
    let mut b = Headless::demo();
    let first = b.session.visible_cloned()[0];
    let r = call_tool(&mut b, "select_photos", &json!({"ids": [first.0]}));
    assert!(!r.is_error, "{r:?}");
    let r = call_tool(&mut b, "select_photos", &json!({"ids": [9999]}));
    assert!(r.is_error, "{r:?}");
    assert!(r.content[0]["text"].as_str().unwrap().contains("no such photo 9999"), "{r:?}");
    assert_eq!(b.session.active(), Some(first));
    let r = call_tool(&mut b, "get_develop", &json!({}));
    assert!(!r.is_error, "{r:?}");
    let r = call_tool(&mut b, "get_develop", &json!({"id": 9999}));
    assert!(r.is_error, "{r:?}");
    let r = call_tool(&mut b, "set_develop", &json!({"id": 9999, "values": {"light.exposure": 1.0}}));
    assert!(r.is_error, "{r:?}");
    assert_eq!(b.session.active(), Some(first));
}

#[test]
fn core_tools_have_hints_and_reject_unknown_keys() {
    for compact in [false, true] {
        let mut s = server().with_command_tools(!compact);
        let list = rpc(&mut s, 1, "tools/list", json!({}));
        let tools = list["result"]["tools"].as_array().unwrap();
        for name in ["command_list", "command_run", "command_batch", "doc_inspect", "render_preview", "list_commands", "run_command", "render_photo"]
        {
            assert!(tools.iter().any(|t| t["name"] == name), "missing {name}");
        }
        for t in tools {
            assert!(!t["title"].as_str().unwrap().is_empty());
            for hint in ["readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"] {
                assert!(t["annotations"][hint].is_boolean(), "{t}");
            }
        }
        for name in ["command_list", "command_run", "command_batch", "doc_inspect", "render_preview", "query_photos", "run_command"] {
            let r = rpc(&mut s, 2, "tools/call", json!({"name":name,"arguments":{"misspelled":true}}));
            assert_eq!(r["error"]["code"], -32602, "{r}");
            assert!(r["error"]["message"].as_str().unwrap().contains("misspelled"), "{r}");
        }
        let preview = tools.iter().find(|t| t["name"] == "render_preview").unwrap();
        assert_eq!(preview["annotations"]["readOnlyHint"], true);
        assert!(preview["inputSchema"]["properties"].get("path").is_none());
        let writer = tools.iter().find(|t| t["name"] == "render_photo").unwrap();
        assert_eq!(writer["annotations"]["readOnlyHint"], false);
        let export = tools.iter().find(|t| t["name"] == "export").unwrap();
        for key in lightcraft_engine::export::OPTION_PARAMS.iter().chain(lightcraft_engine::export::TARGET_PARAMS) {
            assert!(export["inputSchema"]["properties"].get(*key).is_some(), "missing export argument {key}");
        }
        let before = rpc(&mut s, 2, "resources/read", json!({"uri":"lightcraft://library"}));
        let image = rpc(&mut s, 2, "tools/call", json!({"name":"render_preview","arguments":{"max_side":64}}));
        assert_eq!(image["result"]["isError"], false, "{image}");
        assert!(image["result"]["content"].as_array().unwrap().iter().any(|c| c["type"] == "image"));
        assert_eq!(before, rpc(&mut s, 2, "resources/read", json!({"uri":"lightcraft://library"})));
        let batch = json!({"steps":[{"id":"catalog.stats"},{"id":"no.such"},{"id":"catalog.stats"}],"stop_on_error":false});
        let r = rpc(&mut s, 3, "tools/call", json!({"name":"command_batch","arguments":batch}));
        assert_eq!(r["result"]["isError"], true);
        assert_eq!(r["result"]["structuredContent"]["completed"], 2);
        assert_eq!(r["result"]["structuredContent"]["failed"], 1);
        let r = rpc(&mut s, 4, "tools/call", json!({"name":"command_batch","arguments":{"steps":[{"id":"no.such"},{"id":"catalog.stats"}]}}));
        assert_eq!(r["result"]["structuredContent"]["completed"], 0);
        assert_eq!(r["result"]["structuredContent"]["results"].as_array().unwrap().len(), 1);
    }
}

#[test]
fn core_resources_match_tools_after_parse_error() {
    let mut s = server();
    let bad: Value = serde_json::from_str(&s.handle_line("{bad").unwrap()).unwrap();
    assert_eq!(bad["error"]["code"], -32700);
    assert!(bad["id"].is_null());
    let list = rpc(&mut s, 1, "resources/list", json!({}));
    for (uri, tool) in [("lightcraft://document", "doc_inspect"), ("lightcraft://commands", "command_list")] {
        assert!(list["result"]["resources"].as_array().unwrap().iter().any(|r| r["uri"] == uri));
        let resource = rpc(&mut s, 2, "resources/read", json!({"uri":uri}));
        let call = rpc(&mut s, 3, "tools/call", json!({"name":tool,"arguments":{}}));
        let a: Value = serde_json::from_str(resource["result"]["contents"][0]["text"].as_str().unwrap()).unwrap();
        let b: Value = serde_json::from_str(call["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(a, b);
    }
    let old = rpc(&mut s, 4, "resources/read", json!({"uri":"lightcraft://library"}));
    assert!(old["result"]["contents"].is_array(), "{old}");
}

#[test]
fn escaped_tool_panic_is_an_error_and_session_survives() {
    struct Faulty {
        inner: Headless,
        fail: bool,
    }
    impl Backend for Faulty {
        fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
            if method == "engine.execute" && params["command"] == "catalog.stats" && std::mem::take(&mut self.fail) {
                panic!("synthetic backend panic");
            }
            self.inner.call(method, params)
        }
        fn has_ui(&self) -> bool {
            false
        }
        fn describe(&self) -> String {
            "test".into()
        }
    }
    let mut s = Server::new(Box::new(Faulty { inner: Headless::demo(), fail: true }));
    let params = json!({"name":"run_command","arguments":{"command":"catalog.stats"}});
    let r = rpc(&mut s, 1, "tools/call", params.clone());
    assert_eq!(r["result"]["isError"], true, "{r}");
    assert!(r.to_string().contains("synthetic backend panic"));
    assert_eq!(rpc(&mut s, 2, "tools/call", params)["result"]["isError"], false);
    assert!(rpc(&mut s, 3, "ping", json!({}))["result"].is_object());
}

#[test]
fn modern_requests_receive_result_and_cache_fields() {
    let mut s = server();
    let meta = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28"});
    for (method, mut params) in [
        ("tools/list", json!({})),
        ("resources/list", json!({})),
        ("resources/templates/list", json!({})),
        ("resources/read", json!({"uri":"lightcraft://library"})),
    ] {
        params["_meta"] = meta.clone();
        let r = rpc(&mut s, 1, method, params);
        assert_eq!(r["result"]["resultType"], "complete", "{r}");
        assert_eq!(r["result"]["cacheScope"], "private");
        assert_eq!(r["result"]["ttlMs"], if method == "resources/read" { 0 } else { 600_000 });
    }
    assert!(rpc(&mut s, 2, "tools/list", json!({}))["result"].get("resultType").is_none());
}

/// A running app's control channel, as far as the MCP server can tell: it records each call.
struct Recorder(Vec<(String, Value)>);

impl Backend for Recorder {
    fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
        self.0.push((method.to_string(), params));
        Ok(Value::Null)
    }
    fn has_ui(&self) -> bool {
        true
    }
    fn describe(&self) -> String {
        "recorder".into()
    }
}

/// With the app attached, agents cut, copy and paste in its text fields (`clipboard`).
#[test]
fn the_clipboard_tool_cuts_copies_and_pastes_in_the_app() {
    let names: Vec<String> = crate::tools::helper_tools(true).iter().filter_map(|t| t["name"].as_str().map(str::to_string)).collect();
    assert!(names.iter().any(|n| n == "clipboard"), "{names:?}");
    assert!(!crate::tools::helper_tools(false).iter().any(|t| t["name"] == "clipboard"), "not without the app");
    let mut b = Recorder(vec![]);
    let r = call_tool(&mut b, "clipboard", &json!({"action": "paste", "text": "travel"}));
    assert!(!r.is_error, "{r:?}");
    assert_eq!(b.0, vec![("ui.clipboard".to_string(), json!({"action": "paste", "text": "travel"}))]);
}

/// The search and duplicate helpers are listed, `import` forwards `onNearDuplicate`, and the
/// helpers reject keys they do not have.
#[test]
fn vera_tools_are_listed_with_their_arguments() {
    let mut s = server();
    let list = rpc(&mut s, 1, "tools/list", json!({}));
    let tools = list["result"]["tools"].as_array().unwrap();
    for name in ["search_text", "photo_duplicates", "vera_status", "photo_similar", "vera_model"] {
        assert!(tools.iter().any(|t| t["name"] == name), "missing {name}");
    }
    let tool = |n: &str| tools.iter().find(|t| t["name"] == n).unwrap();
    assert_eq!(tool("search_text")["inputSchema"]["required"], json!(["query"]));
    assert_eq!(tool("vera_status")["annotations"]["readOnlyHint"], true);
    assert_eq!(tool("search_text")["annotations"]["readOnlyHint"], false, "filter changes the view, as select_photos does");
    assert_eq!(tool("photo_duplicates")["annotations"]["readOnlyHint"], false);
    // photo_similar: a view tool like the other two
    let similar = tool("photo_similar");
    assert_eq!(similar["inputSchema"]["required"], json!([]));
    for k in ["id", "limit", "filter"] {
        assert!(similar["inputSchema"]["properties"].get(k).is_some(), "{k}");
    }
    assert_eq!(similar["annotations"]["readOnlyHint"], false, "filter changes the view");
    assert_eq!(similar["annotations"]["destructiveHint"], false);
    assert_eq!(similar["annotations"]["idempotentHint"], true);
    // vera_model: one tool with a read-only action and writing ones is annotated for the worst
    let model = tool("vera_model");
    assert_eq!(model["inputSchema"]["properties"]["action"]["enum"], json!(["status", "download", "cancel", "remove", "config"]));
    for k in ["acceptLicense", "vectorLane"] {
        assert_eq!(model["inputSchema"]["properties"][k]["type"], "boolean", "{k}");
    }
    assert_eq!(model["inputSchema"]["required"], json!([]));
    assert_eq!(model["annotations"]["readOnlyHint"], false);
    assert_eq!(model["annotations"]["destructiveHint"], true);
    assert_eq!(model["annotations"]["idempotentHint"], false);
    let d = model["description"].as_str().unwrap();
    for needle in ["1.5 GB", "Hugging Face", "Apache License 2.0", "acceptLicense: true", "licence.notice", "licence.url"] {
        assert!(d.contains(needle), "description lacks {needle}");
    }
    let d = tool("search_text")["description"].as_str().unwrap();
    assert!(["both", "cosine", "keywordScore", "warning"].iter().all(|w| d.contains(w)), "{d}");
    let d = tool("vera_status")["description"].as_str().unwrap();
    assert!(["needs_model", "downloading", "disabled_by_user", "needs_desktop", "spaceBackground"].iter().all(|w| d.contains(w)), "{d}");
    let n = &tool("import")["inputSchema"]["properties"]["onNearDuplicate"];
    assert_eq!(n["enum"], json!(["stack", "import", "skip"]));
    for (name, args) in [
        ("search_text", json!({"query": "x", "misspelled": 1})),
        ("photo_duplicates", json!({"misspelled": 1})),
        ("vera_status", json!({"misspelled": 1})),
        ("photo_similar", json!({"misspelled": 1})),
        ("vera_model", json!({"misspelled": 1})),
    ] {
        let r = rpc(&mut s, 2, "tools/call", json!({"name": name, "arguments": args}));
        assert_eq!(r["error"]["code"], -32602, "{r}");
        assert!(r["error"]["message"].as_str().unwrap().contains("misspelled"), "{r}");
    }
}

#[test]
fn vera_status_reports_the_lanes() {
    let mut b = Headless::demo();
    let r = call_tool(&mut b, "vera_status", &json!({}));
    assert!(!r.is_error, "{r:?}");
    let v = r.structured.unwrap();
    assert_eq!(v["lanes"]["keyword"], "live");
    assert!(v["photos"].as_u64().unwrap() > 0, "{v}");
    assert!(v["lanes"]["vector"]["state"].is_string(), "{v}");
}

#[test]
fn search_text_finds_a_titled_photo_first() {
    let mut b = Headless::demo();
    let first = b.session.visible_cloned()[0];
    let r = call_tool(&mut b, "run_command", &json!({"command": "library.select", "params": {"ids": [first.0]}}));
    assert!(!r.is_error, "{r:?}");
    let r = call_tool(&mut b, "run_command", &json!({"command": "photo.setMeta", "params": {"title": "Zanzibar harbour"}}));
    assert!(!r.is_error, "{r:?}");
    let r = call_tool(&mut b, "search_text", &json!({"query": "zanzi"}));
    assert!(!r.is_error, "{r:?}");
    let v = r.structured.unwrap();
    assert_eq!(v["photos"][0]["id"], first.0, "{v}");
    assert_eq!(v["photos"][0]["lane"], "keyword");
    assert!(v["cutoff"].is_null());
    assert_eq!(v["filtered"], true);
    assert_eq!(b.session.visible_cloned(), vec![first], "the view shows the results");
    // `filter: false` leaves the view alone
    let _ = call_tool(&mut b, "run_command", &json!({"command": "library.clearFilter"}));
    let r = call_tool(&mut b, "search_text", &json!({"query": "zanzi", "limit": 1, "filter": false}));
    assert_eq!(r.structured.unwrap()["filtered"], false);
    assert!(b.session.visible_cloned().len() > 1);
    assert!(call_tool(&mut b, "search_text", &json!({})).is_error);
    assert!(call_tool(&mut b, "search_text", &json!({"query": 5})).is_error);
}

#[test]
fn photo_duplicates_returns_the_report_shape() {
    let mut b = Headless::demo();
    let first = b.session.visible_cloned()[0];
    let r = call_tool(&mut b, "photo_duplicates", &json!({"id": first.0}));
    assert!(!r.is_error, "{r:?}");
    let v = r.structured.unwrap();
    assert_eq!(v["id"], first.0);
    assert!(v["photos"].is_array(), "{v}");
    for k in ["digested", "pendingDigests", "ignored"] {
        assert!(v.get(k).is_some(), "missing {k}: {v}");
    }
    // the default is the active photo; an unknown one is an error
    assert!(!call_tool(&mut b, "photo_duplicates", &json!({})).is_error);
    assert!(call_tool(&mut b, "photo_duplicates", &json!({"id": 9999})).is_error);
}

/// Without the model the look-alike search is a tool error that says the lane is not ready.
#[test]
fn photo_similar_without_the_model_is_not_ready() {
    let mut b = Headless::demo();
    let first = b.session.visible_cloned()[0];
    for args in [json!({}), json!({"id": first.0, "limit": 5, "filter": false})] {
        let r = call_tool(&mut b, "photo_similar", &args);
        assert!(r.is_error, "{r:?}");
        assert!(r.content[0]["text"].as_str().unwrap().contains("not ready"), "{r:?}");
    }
    assert!(call_tool(&mut b, "photo_similar", &json!({"id": 9999})).is_error);
    assert!(call_tool(&mut b, "photo_similar", &json!({"limit": -1})).is_error);
}

#[test]
fn vera_model_status_is_the_default_and_carries_the_licence() {
    let mut b = Headless::demo();
    for args in [json!({}), json!({"action": "status"})] {
        let r = call_tool(&mut b, "vera_model", &args);
        assert!(!r.is_error, "{r:?}");
        let v = r.structured.unwrap();
        assert_eq!(v["installed"], false, "{v}");
        assert!(v["licence"]["notice"].as_str().unwrap().contains("Apache"), "{v}");
        assert!(v["licence"]["url"].is_string(), "{v}");
    }
}

/// The 1.5 GB download is refused without `acceptLicense: true` (no test passes it: no network).
#[test]
fn vera_model_download_needs_the_licence_accepted() {
    let mut b = Headless::demo();
    for args in [json!({"action": "download"}), json!({"action": "download", "acceptLicense": false})] {
        let r = call_tool(&mut b, "vera_model", &args);
        assert!(r.is_error, "{r:?}");
        let t = r.content[0]["text"].as_str().unwrap();
        assert!(t.contains("License") && t.contains("Apache"), "{t}");
        assert!(t.contains("acceptLicense"), "{t}");
    }
    let r = call_tool(&mut b, "vera_model", &json!({"action": "cancel"}));
    assert!(!r.is_error, "{r:?}");
    assert_eq!(r.structured.unwrap()["cancelled"], false);
}

#[test]
fn vera_model_config_round_trips_and_status_follows() {
    let mut b = Headless::demo();
    let r = call_tool(&mut b, "vera_model", &json!({"action": "config"}));
    assert!(!r.is_error, "{r:?}");
    assert_eq!(r.structured.unwrap()["vectorLane"], true);
    let r = call_tool(&mut b, "vera_model", &json!({"action": "config", "vectorLane": false}));
    assert!(!r.is_error, "{r:?}");
    let v = r.structured.unwrap();
    assert_eq!(v["vectorLane"], false, "{v}");
    // no model folder on this session: the lane is still `needs_model`, and the switch is remembered
    let v = call_tool(&mut b, "vera_status", &json!({})).structured.unwrap();
    assert_eq!(v["lanes"]["vector"]["state"], "needs_model", "{v}");
    // a folder that looks like an installed model (empty files, never loaded): now the lane says it is off
    let dir = std::env::temp_dir().join(format!("lc-mcp-model-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["model.safetensors", "config.json", "tokenizer.model"] {
        std::fs::write(dir.join(f), b"not a model").unwrap();
    }
    b.session.vera_model_dir = Some(dir.clone());
    let v = call_tool(&mut b, "vera_status", &json!({})).structured.unwrap();
    assert_eq!((v["model"]["installed"].clone(), v["lanes"]["vector"]["state"].clone()), (json!(true), json!("disabled_by_user")), "{v}");
    let v = call_tool(&mut b, "vera_model", &json!({})).structured.unwrap();
    assert_eq!((v["installed"].clone(), v["state"].clone()), (json!(true), json!("disabled_by_user")), "{v}");
    let r = call_tool(&mut b, "photo_similar", &json!({}));
    assert!(r.is_error && r.content[0]["text"].as_str().unwrap().contains("not ready"), "{r:?}");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(call_tool(&mut b, "vera_model", &json!({"action": "config", "vectorLane": "off"})).is_error);
    let r = call_tool(&mut b, "vera_model", &json!({"action": "config", "vectorLane": true}));
    assert_eq!(r.structured.unwrap()["vectorLane"], true);
}

/// `remove` with nothing installed removes nothing, and says so.
#[test]
fn vera_model_remove_with_nothing_installed_removes_nothing() {
    let mut b = Headless::demo();
    let r = call_tool(&mut b, "vera_model", &json!({"action": "remove"}));
    assert!(!r.is_error, "{r:?}");
    let v = r.structured.unwrap();
    assert_eq!((v["files"].clone(), v["installed"].clone()), (json!(0), json!(false)), "{v}");
}

/// An unknown or mistyped action is refused before the engine is called.
#[test]
fn vera_model_rejects_unknown_actions_before_the_engine() {
    let mut rec = Recorder(vec![]);
    for args in [json!({"action": "delete"}), json!({"action": "Status"}), json!({"action": 5}), json!({"action": ""})] {
        let r = call_tool(&mut rec, "vera_model", &args);
        assert!(r.is_error, "{args}: {r:?}");
        assert!(r.content[0]["text"].as_str().unwrap().contains("action"), "{r:?}");
    }
    assert!(rec.0.is_empty(), "the engine was not called: {:?}", rec.0);
    // `acceptLicense` goes to download only, `vectorLane` to config only (a recorder: nothing is fetched)
    let r = call_tool(&mut rec, "vera_model", &json!({"action": "download", "acceptLicense": false, "vectorLane": false}));
    assert!(!r.is_error);
    let r = call_tool(&mut rec, "vera_model", &json!({"action": "config", "acceptLicense": false, "vectorLane": false}));
    assert!(!r.is_error);
    assert_eq!(
        rec.0,
        vec![
            ("engine.execute".to_string(), json!({"command": "vera.model.download", "params": {"acceptLicense": false}})),
            ("engine.execute".to_string(), json!({"command": "vera.config", "params": {"vectorLane": false}})),
        ]
    );
}

fn text_of(r: &crate::ToolResult) -> String {
    r.content[0]["text"].as_str().unwrap().to_string()
}

/// The Describe helpers are listed with their arguments and hints, and reject keys they do not have.
#[test]
fn describe_tools_are_listed_with_schemas_and_hints() {
    let mut s = server();
    let list = rpc(&mut s, 1, "tools/list", json!({}));
    let tools = list["result"]["tools"].as_array().unwrap();
    let tool = |n: &str| tools.iter().find(|t| t["name"] == n).unwrap_or_else(|| panic!("missing {n}"));
    let hints = |n: &str| {
        let a = &tool(n)["annotations"];
        (a["readOnlyHint"].clone(), a["destructiveHint"].clone(), a["idempotentHint"].clone())
    };
    let props = |n: &str, ks: &[&str]| {
        for k in ks {
            assert!(tool(n)["inputSchema"]["properties"].get(*k).is_some(), "{n}.{k}");
        }
        assert_eq!(tool(n)["inputSchema"]["required"], json!([]), "{n}");
    };
    props("vera_describe_config", &["endpoint", "model", "allowRemoteHost", "token"]);
    props("photo_describe", &["ids", "keywordsOnly"]);
    props("describe_review", &["action", "id", "fields", "edits", "replaceKeywords", "limit"]);
    props("photo_keywords", &["id", "limit"]);
    assert_eq!(tool("describe_review")["inputSchema"]["properties"]["action"]["enum"], json!(["list", "accept", "reject", "clear"]));
    assert_eq!(tool("photo_describe")["inputSchema"]["properties"]["ids"]["type"], "array");
    assert_eq!(hints("vera_describe_config"), (json!(false), json!(false), json!(true)));
    assert_eq!(hints("photo_describe"), (json!(false), json!(false), json!(false)));
    assert_eq!(hints("describe_review"), (json!(false), json!(false), json!(false)));
    assert_eq!(hints("photo_keywords"), (json!(true), json!(false), json!(true)));
    for n in ["vera_describe_config", "photo_describe", "describe_review", "photo_keywords"] {
        assert!(tool(n)["annotations"]["title"].is_string(), "{n}");
        assert_eq!(tool(n)["annotations"]["openWorldHint"], false, "{n}");
    }
    let d = tool("vera_describe_config")["description"].as_str().unwrap();
    for needle in ["OpenAI-compatible", "127.0.0.1", "https", "allowRemoteHost: true", "0600", "tokenSet", "never"] {
        assert!(d.contains(needle), "vera_describe_config lacks {needle}");
    }
    let d = tool("vera_status")["description"].as_str().unwrap();
    for needle in ["needs_endpoint", "needs_token", "disabled", "host"] {
        assert!(d.contains(needle), "vera_status lacks {needle}");
    }
    for (name, args) in [
        ("vera_describe_config", json!({"misspelled": 1})),
        ("photo_describe", json!({"misspelled": 1})),
        ("describe_review", json!({"misspelled": 1})),
        ("photo_keywords", json!({"misspelled": 1})),
    ] {
        let r = rpc(&mut s, 2, "tools/call", json!({"name": name, "arguments": args}));
        assert_eq!(r["error"]["code"], -32602, "{r}");
        assert!(r["error"]["message"].as_str().unwrap().contains("misspelled"), "{r}");
    }
}

/// The endpoint is set without a token; a token is kept (needs a model folder to live beside) and never shown.
#[test]
fn vera_describe_config_sets_the_endpoint_and_never_shows_the_token() {
    const SECRET: &str = "sk-secret-xyz";
    let mut b = Headless::demo();
    let r = call_tool(&mut b, "vera_describe_config", &json!({"endpoint": "http://127.0.0.1:1"}));
    assert!(!r.is_error, "{r:?}");
    let v = r.structured.unwrap();
    assert_eq!((v["endpoint"].clone(), v["tokenSet"].clone()), (json!("http://127.0.0.1:1"), json!(false)), "{v}");
    // an empty call reads
    let v = call_tool(&mut b, "vera_describe_config", &json!({})).structured.unwrap();
    assert_eq!(v["endpoint"], "http://127.0.0.1:1");

    let dir = std::env::temp_dir().join(format!("lc-mcp-describe-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    b.session.vera_model_dir = Some(dir.join("models"));
    let r = call_tool(&mut b, "vera_describe_config", &json!({"model": "stub-vlm", "token": SECRET}));
    let all = format!("{r:?}{}", serde_json::to_string(&r.to_value()).unwrap());
    assert!(!r.is_error, "{r:?}");
    assert!(!all.contains(SECRET), "the token is not in the result: {all}");
    let v = r.structured.unwrap();
    assert_eq!((v["tokenSet"].clone(), v["model"].clone()), (json!(true), json!("stub-vlm")), "{v}");
    // nor does a later read, or the status, show it
    for (tool, args) in [("vera_describe_config", json!({})), ("vera_status", json!({}))] {
        let r = call_tool(&mut b, tool, &args);
        assert!(!serde_json::to_string(&r.to_value()).unwrap().contains(SECRET), "{tool}");
    }
    // null clears it
    let v = call_tool(&mut b, "vera_describe_config", &json!({"token": null})).structured.unwrap();
    assert_eq!(v["tokenSet"], false, "{v}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A refused endpoint is a tool error that says why and never repeats the token given with it.
#[test]
fn vera_describe_config_errors_never_repeat_the_token() {
    const SECRET: &str = "sk-secret-xyz";
    let mut b = Headless::demo();
    let r = call_tool(&mut b, "vera_describe_config", &json!({"endpoint": "http://example.com", "token": SECRET}));
    assert!(r.is_error, "{r:?}");
    let t = text_of(&r);
    assert!(t.contains("https"), "{t}");
    assert!(!t.contains(SECRET), "{t}");
    assert!(!serde_json::to_string(&r.to_value()).unwrap().contains(SECRET));
    // a token alone with nowhere to keep it, or of the wrong type, is refused without repeating it either
    for args in [json!({"token": SECRET}), json!({"token": SECRET, "allowRemoteHost": "yes"}), json!({"endpoint": 5, "token": SECRET})] {
        let r = call_tool(&mut b, "vera_describe_config", &args);
        assert!(r.is_error, "{args}: {r:?}");
        assert!(!text_of(&r).contains(SECRET), "{args}");
    }
    // whatever the engine says, the token is scrubbed from it
    struct Echo;
    impl Backend for Echo {
        fn call(&mut self, _: &str, params: Value) -> Result<Value, String> {
            Err(format!("bad params {}", params["params"]))
        }
        fn has_ui(&self) -> bool {
            true
        }
        fn describe(&self) -> String {
            "echo".into()
        }
    }
    let r = call_tool(&mut Echo, "vera_describe_config", &json!({"endpoint": "x", "token": SECRET}));
    assert!(r.is_error && !text_of(&r).contains(SECRET), "{r:?}");
}

/// Without an endpoint Describe is a tool error naming it; keywords alone need the model, and say so.
#[test]
fn photo_describe_says_what_is_missing() {
    let mut b = Headless::demo();
    let first = b.session.visible_cloned()[0];
    let r = call_tool(&mut b, "photo_describe", &json!({"ids": [first.0]}));
    assert!(r.is_error, "{r:?}");
    assert!(text_of(&r).contains("endpoint"), "{r:?}");
    for args in [json!({"keywordsOnly": true}), json!({"ids": [first.0], "keywordsOnly": true})] {
        let r = call_tool(&mut b, "photo_describe", &args);
        assert!(r.is_error, "{args}: {r:?}");
        assert!(text_of(&r).contains("model"), "{r:?}");
    }
    assert!(call_tool(&mut b, "photo_describe", &json!({"keywordsOnly": "yes"})).is_error);
    // nothing was queued
    let v = call_tool(&mut b, "describe_review", &json!({})).structured.unwrap();
    assert_eq!((v["pending"].clone(), v["inFlight"].clone()), (json!(0), json!(0)), "{v}");
}

#[test]
fn describe_review_lists_an_empty_queue_and_clear_and_reject_are_harmless() {
    let mut b = Headless::demo();
    for args in [json!({}), json!({"action": "list"}), json!({"action": "list", "limit": 5})] {
        let r = call_tool(&mut b, "describe_review", &args);
        assert!(!r.is_error, "{args}: {r:?}");
        let v = r.structured.unwrap();
        assert_eq!((v["proposals"].clone(), v["total"].clone()), (json!([]), json!(0)), "{v}");
    }
    assert!(!call_tool(&mut b, "describe_review", &json!({"action": "clear"})).is_error);
    let first = b.session.visible_cloned()[0];
    assert!(!call_tool(&mut b, "describe_review", &json!({"action": "reject", "id": first.0})).is_error);
    // accepting what was never proposed is the engine's error
    let r = call_tool(&mut b, "describe_review", &json!({"action": "accept", "id": first.0}));
    assert!(r.is_error && text_of(&r).contains("no proposal"), "{r:?}");
}

/// accept / reject need an `id`, an unknown action is refused, and the engine is not called for either; the rest is forwarded as given.
#[test]
fn describe_review_validates_before_the_engine() {
    let mut rec = Recorder(vec![]);
    for args in [
        json!({"action": "accept"}),
        json!({"action": "reject"}),
        json!({"action": "accept", "id": null}),
        json!({"action": "accept", "id": "7"}),
        json!({"action": "accept", "id": -1}),
        json!({"action": "approve", "id": 1}),
        json!({"action": "List"}),
        json!({"action": 5}),
        json!({"action": ""}),
    ] {
        let r = call_tool(&mut rec, "describe_review", &args);
        assert!(r.is_error, "{args}: {r:?}");
    }
    assert!(rec.0.is_empty(), "the engine was not called: {:?}", rec.0);
    let t = text_of(&call_tool(&mut rec, "describe_review", &json!({"action": "accept"})));
    assert!(t.contains("`id`"), "{t}");
    let t = text_of(&call_tool(&mut rec, "describe_review", &json!({"action": "approve"})));
    assert!(t.contains("approve") && t.contains("accept"), "{t}");
    assert!(rec.0.is_empty());

    for args in [
        json!({}),
        json!({"action": "list", "limit": 3, "id": 9}),
        json!({"action": "accept", "id": 4, "fields": ["title"], "edits": {"title": "T"}, "replaceKeywords": true, "limit": 2}),
        json!({"action": "reject", "id": 4, "fields": ["title"]}),
        json!({"action": "clear", "id": 4}),
    ] {
        assert!(!call_tool(&mut rec, "describe_review", &args).is_error);
    }
    let call = |c: &str, p: Value| ("engine.execute".to_string(), json!({"command": c, "params": p}));
    assert_eq!(
        rec.0,
        vec![
            call("describe.list", json!({})),
            call("describe.list", json!({"limit": 3})),
            call("describe.accept", json!({"id": 4, "fields": ["title"], "edits": {"title": "T"}, "replaceKeywords": true})),
            call("describe.reject", json!({"id": 4})),
            call("describe.clear", json!({})),
        ]
    );
}

/// The other helpers forward exactly their own arguments.
#[test]
fn describe_helpers_forward_their_arguments() {
    let mut rec = Recorder(vec![]);
    call_tool(&mut rec, "photo_describe", &json!({"ids": [1, 2], "keywordsOnly": true}));
    call_tool(&mut rec, "photo_describe", &json!({}));
    call_tool(&mut rec, "photo_keywords", &json!({"id": 3, "limit": 5}));
    call_tool(&mut rec, "vera_describe_config", &json!({"endpoint": null, "model": "m", "allowRemoteHost": false}));
    let call = |c: &str, p: Value| ("engine.execute".to_string(), json!({"command": c, "params": p}));
    assert_eq!(
        rec.0,
        vec![
            call("photo.describe", json!({"ids": [1, 2], "keywordsOnly": true})),
            call("photo.describe", json!({})),
            call("photo.keywords", json!({"id": 3, "limit": 5})),
            call("vera.describe.config", json!({"endpoint": null, "model": "m", "allowRemoteHost": false})),
        ]
    );
}

/// Without the model there are no keywords to suggest, and the error names the model.
#[test]
fn photo_keywords_without_the_model_is_an_error_naming_it() {
    let mut b = Headless::demo();
    let first = b.session.visible_cloned()[0];
    for args in [json!({}), json!({"id": first.0, "limit": 5})] {
        let r = call_tool(&mut b, "photo_keywords", &args);
        assert!(r.is_error, "{args}: {r:?}");
        assert!(text_of(&r).contains("model"), "{r:?}");
    }
    assert!(call_tool(&mut b, "photo_keywords", &json!({"id": 9999})).is_error);
    assert!(call_tool(&mut b, "photo_keywords", &json!({"limit": 0})).is_error);
}
