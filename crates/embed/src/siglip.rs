//! The SigLIP 2 Base patch32-256 towers (`SiglipModel` of Hugging Face Transformers, the
//! fixed-resolution variant the `google/siglip2-base-patch32-256` checkpoint uses).
//!
//! - **Image tower** (`SiglipVisionTransformer`): a 32 × 32 patch embedding (a strided conv,
//!   computed here as a matmul over patches), a learned position embedding of 64 entries, 12
//!   pre-LN encoder layers (bidirectional attention, GELU-tanh MLP), `post_layernorm`, then the
//!   multihead-attention pooling head (a learned probe attends over the 64 tokens, then a residual
//!   MLP). Its output is the image embedding; there is no projection after it.
//! - **Text tower** (`SiglipTextTransformer`): token + learned position embeddings over 64 tokens, the
//!   same encoder layers with **no** attention mask (SigLIP attends to padding, as it was trained),
//!   `final_layer_norm`, then the hidden state of the **last** position (index 63, usually padding)
//!   through `head`, a 768 → 768 linear layer.
//!
//! Both outputs are L2-normalised here. The towers load on first use, so an image-only worker never
//! holds the text tower and a query never holds the image tower; the 256,000-row token embedding is
//! never loaded at all (each query reads its 64 rows from the file, [`Weights::read_rows`]).
//!
//! Modified work (Apache License 2.0, §4(b)): ported by the LightCraft contributors in 2026 from the
//! Python/PyTorch SigLIP code of Hugging Face Transformers (`models/siglip/modeling_siglip.py`),
//! Copyright The HuggingFace Team and Google LLC; translated to Rust on candle and restructured. See
//! NOTICE.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use serde_json::Value;

use crate::nn::{Attention, Mlp, linear, ln, to_vec};
use crate::tokenizer::Tokenizer;
use crate::weights::{Backend, Weights};
use crate::{CONFIG_FILE, CONTEXT, DIM, Error, INPUT, Result, TOKENIZER_FILE, WEIGHTS_FILE};

/// Most images run through the image tower in one forward pass ([`Model::embed_images`] splits
/// longer batches); bounds the activations at about 16 × 64 × 3072 floats per layer.
pub const MAX_BATCH: usize = 16;
/// Largest `config.json` read (the real one is 298 bytes).
const MAX_CONFIG_BYTES: u64 = 1 << 20;
/// The token-embedding table, read row by row.
const TOKEN_EMBEDDING: &str = "text_model.embeddings.token_embedding.weight";

/// One transformer stack's sizes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TowerConfig {
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub eps: f64,
}

/// The architecture, as `config.json` (with Transformers' `SiglipConfig` defaults) describes it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Config {
    pub vision: TowerConfig,
    pub image: usize,
    pub patch: usize,
    pub text: TowerConfig,
    pub vocab: usize,
    pub context: usize,
    pub projection: usize,
}

impl Config {
    /// `google/siglip2-base-patch32-256`.
    #[cfg(test)]
    pub const BASE: Config = Config {
        vision: TowerConfig { hidden: 768, inter: 3072, layers: 12, heads: 12, eps: 1e-6 },
        image: 256,
        patch: 32,
        text: TowerConfig { hidden: 768, inter: 3072, layers: 12, heads: 12, eps: 1e-6 },
        vocab: 256_000,
        context: 64,
        projection: 768,
    };

    /// Patches per side.
    pub fn grid(&self) -> usize {
        self.image.checked_div(self.patch).unwrap_or(0)
    }

    /// Read a `config.json`. Fields it leaves out take Transformers' defaults (which differ from
    /// this checkpoint for `image_size` and `patch_size`, so those must be present). The result is
    /// checked against what this crate's API fixes: [`INPUT`], [`DIM`] and [`CONTEXT`].
    pub fn from_json(text: &str) -> Result<Config> {
        let bad = |m: String| Error::Model(format!("{CONFIG_FILE}: {m}"));
        let root: Value = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
        let empty = Value::Object(serde_json::Map::new());
        let v = root.get("vision_config").unwrap_or(&empty);
        let t = root.get("text_config").unwrap_or(&empty);
        let int = |o: &Value, k: &str, d: usize| -> Result<usize> {
            match o.get(k) {
                None | Some(Value::Null) => Ok(d),
                Some(x) => x.as_u64().and_then(|x| usize::try_from(x).ok()).filter(|x| *x > 0).ok_or_else(|| bad(format!("{k} = {x} is not a size"))),
            }
        };
        let float = |o: &Value, k: &str, d: f64| -> Result<f64> {
            match o.get(k) {
                None | Some(Value::Null) => Ok(d),
                Some(x) => x.as_f64().filter(|x| x.is_finite() && *x > 0.0).ok_or_else(|| bad(format!("{k} = {x} is not an epsilon"))),
            }
        };
        let act = |o: &Value| -> Result<()> {
            match o.get("hidden_act").and_then(Value::as_str) {
                None | Some("gelu_pytorch_tanh") => Ok(()),
                Some(a) => Err(bad(format!("activation {a} (this crate implements gelu_pytorch_tanh)"))),
            }
        };
        act(v)?;
        act(t)?;
        if v.get("vision_use_head").and_then(Value::as_bool) == Some(false) {
            return Err(bad("vision_use_head = false (this crate implements the attention-pooling head)".into()));
        }
        if int(v, "num_channels", 3)? != 3 {
            return Err(bad("num_channels is not 3".into()));
        }
        let tower = |o: &Value| -> Result<TowerConfig> {
            Ok(TowerConfig {
                hidden: int(o, "hidden_size", 768)?,
                inter: int(o, "intermediate_size", 3072)?,
                layers: int(o, "num_hidden_layers", 12)?,
                heads: int(o, "num_attention_heads", 12)?,
                eps: float(o, "layer_norm_eps", 1e-6)?,
            })
        };
        let text_tower = tower(t)?;
        let c = Config {
            vision: tower(v)?,
            image: int(v, "image_size", 224)?,
            patch: int(v, "patch_size", 16)?,
            vocab: int(t, "vocab_size", 32000)?,
            context: int(t, "max_position_embeddings", 64)?,
            projection: int(t, "projection_size", text_tower.hidden)?,
            text: text_tower,
        };
        if c.image != INPUT || c.vision.hidden != DIM || c.projection != DIM || c.context != CONTEXT {
            return Err(bad(format!(
                "image {} / width {} / projection {} / context {}; this build expects {INPUT} / {DIM} / {DIM} / {CONTEXT}",
                c.image, c.vision.hidden, c.projection, c.context
            )));
        }
        if !c.image.is_multiple_of(c.patch) {
            return Err(bad(format!("patch {} does not tile image {}", c.patch, c.image)));
        }
        Ok(c)
    }
}

struct Layer {
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    mlp: Mlp,
}

impl Layer {
    /// `SiglipEncoderLayer`: `x + attn(ln1(x))`, then `x + mlp(ln2(x))`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let x = (x + self.attn.forward(&h, &h)?)?;
        let h = self.ln2.forward(&x)?;
        Ok((&x + self.mlp.forward(&h)?)?)
    }
}

/// `SiglipEncoder` at `vb` (`…encoder`).
fn encoder(c: &TowerConfig, vb: VarBuilder) -> Result<Vec<Layer>> {
    (0..c.layers)
        .map(|i| {
            let l = vb.pp(format!("layers.{i}"));
            Ok(Layer {
                ln1: ln(c.hidden, c.eps, l.pp("layer_norm1"))?,
                attn: Attention::new(c.hidden, c.heads, l.pp("self_attn"))?,
                ln2: ln(c.hidden, c.eps, l.pp("layer_norm2"))?,
                mlp: Mlp::new(c.hidden, c.inter, l.pp("mlp"))?,
            })
        })
        .collect()
}

fn run(layers: &[Layer], x: Tensor) -> Result<Tensor> {
    layers.iter().try_fold(x, |x, l| l.forward(&x))
}

/// `SiglipMultiheadAttentionPoolingHead`: the probe attends over the tokens (PyTorch's packed
/// `nn.MultiheadAttention`), then `h + mlp(layernorm(h))`.
pub(crate) struct PoolingHead {
    probe: Tensor,
    attn: Attention,
    norm: LayerNorm,
    mlp: Mlp,
}

impl PoolingHead {
    /// `vb` at `vision_model.head`.
    pub fn new(c: &TowerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            probe: vb.get((1, 1, c.hidden), "probe")?,
            attn: Attention::packed(c.hidden, c.heads, vb.pp("attention"))?,
            norm: ln(c.hidden, c.eps, vb.pp("layernorm"))?,
            mlp: Mlp::new(c.hidden, c.inter, vb.pp("mlp"))?,
        })
    }

    /// `[B, N, hidden]` → `[B, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _, h) = x.dims3()?;
        let probe = self.probe.broadcast_as((b, 1, h))?.contiguous()?;
        let y = self.attn.forward(&probe, x)?;
        let y = (&y + self.mlp.forward(&self.norm.forward(&y)?)?)?;
        Ok(y.squeeze(1)?)
    }
}

/// `SiglipVisionTransformer`.
pub(crate) struct Vision {
    /// The patch conv as a linear map over flattened `(channel, y, x)` patches.
    patch: Linear,
    pos: Tensor,
    layers: Vec<Layer>,
    post_ln: LayerNorm,
    head: PoolingHead,
    side: usize,
    patch_side: usize,
}

impl Vision {
    /// `vb` at `vision_model`.
    pub fn new(c: &Config, vb: VarBuilder) -> Result<Self> {
        let (h, p, g) = (c.vision.hidden, c.patch, c.grid());
        let e = vb.pp("embeddings");
        let w = e.get((h, 3, p, p), "patch_embedding.weight")?.reshape((h, 3 * p * p))?;
        let b = e.get(h, "patch_embedding.bias")?;
        Ok(Self {
            patch: Linear::new(w, Some(b)),
            pos: e.get((g * g, h), "position_embedding.weight")?.unsqueeze(0)?,
            layers: encoder(&c.vision, vb.pp("encoder"))?,
            post_ln: ln(h, c.vision.eps, vb.pp("post_layernorm"))?,
            head: PoolingHead::new(&c.vision, vb.pp("head"))?,
            side: c.image,
            patch_side: p,
        })
    }

    /// `pixels`: `[B, 3, side, side]` normalised to −1..1. Returns `[B, hidden]` (not normalised).
    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let (b, ch, hh, ww) = pixels.dims4()?;
        let (p, g) = (self.patch_side, self.side / self.patch_side.max(1));
        if ch != 3 || hh != self.side || ww != self.side {
            return Err(Error::Model(format!("image tower takes [B, 3, {s}, {s}], got {:?}", pixels.dims(), s = self.side)));
        }
        // stride = kernel, padding "valid": the conv is a matmul over non-overlapping patches,
        // flattened in the conv weight's (channel, y, x) order, patches row-major
        let patches = pixels.reshape((b, 3, g, p, g, p))?.permute((0, 2, 4, 1, 3, 5))?.contiguous()?.reshape((b, g * g, 3 * p * p))?;
        let x = self.patch.forward(&patches)?.broadcast_add(&self.pos)?;
        let x = self.post_ln.forward(&run(&self.layers, x)?)?;
        self.head.forward(&x)
    }
}

/// `SiglipTextTransformer` without its token embedding (read row by row by the caller).
pub(crate) struct Text {
    pos: Tensor,
    layers: Vec<Layer>,
    final_ln: LayerNorm,
    head: Linear,
}

impl Text {
    /// `vb` at `text_model`.
    pub fn new(c: &Config, vb: VarBuilder) -> Result<Self> {
        let h = c.text.hidden;
        Ok(Self {
            pos: vb.get((c.context, h), "embeddings.position_embedding.weight")?.unsqueeze(0)?,
            layers: encoder(&c.text, vb.pp("encoder"))?,
            final_ln: ln(h, c.text.eps, vb.pp("final_layer_norm"))?,
            head: linear(h, c.projection, vb.pp("head"))?,
        })
    }

    /// `tokens`: the token embeddings `[B, context, hidden]`. Returns `[B, projection]` (not
    /// normalised): the last position's state through `head`.
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let x = tokens.broadcast_add(&self.pos)?;
        let x = self.final_ln.forward(&run(&self.layers, x)?)?;
        let last = x.dim(1)?.checked_sub(1).ok_or_else(|| Error::Model("empty text".into()))?;
        Ok(self.head.forward(&x.narrow(1, last, 1)?.squeeze(1)?)?)
    }
}

/// Each row of `[B, dim]` scaled to unit length.
fn unit_rows(t: &Tensor) -> Result<Vec<Vec<f32>>> {
    let (_, dim) = t.dims2()?;
    to_vec(t)?
        .chunks(dim.max(1))
        .map(|r| {
            let n = r.iter().map(|v| v * v).sum::<f32>().sqrt();
            if n.is_finite() && n > 0.0 {
                Ok(r.iter().map(|v| v / n).collect())
            } else {
                Err(Error::Model(format!("degenerate embedding (norm {n})")))
            }
        })
        .collect()
}

/// A loaded checkpoint. The towers and the tokenizer load on first use; `Model` is `Sync`, so one
/// instance can serve several threads.
pub struct Model {
    dir: PathBuf,
    device: Device,
    config: Config,
    weights: Arc<Weights>,
    vision: Mutex<Option<Arc<Vision>>>,
    text: Mutex<Option<Arc<Text>>>,
    tokenizer: Mutex<Option<Arc<Tokenizer>>>,
}

impl Model {
    /// Open the checkpoint in `dir` (`model.safetensors`, `config.json`, `tokenizer.model`): read
    /// and check the config, and check the weights header against it (a damaged or different
    /// checkpoint fails here). No tower is loaded yet.
    pub fn load(dir: &Path, device: Device) -> Result<Model> {
        if !crate::is_model_dir(dir) {
            return Err(Error::Missing(dir.to_path_buf()));
        }
        let config = read_config(&dir.join(CONFIG_FILE))?;
        let weights = Weights::open(&dir.join(WEIGHTS_FILE))?;
        check_header(&weights, &config)?;
        Ok(Model { dir: dir.to_path_buf(), device, config, weights, vision: Mutex::new(None), text: Mutex::new(None), tokenizer: Mutex::new(None) })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn vb(&self) -> VarBuilder<'static> {
        VarBuilder::from_backend(Box::new(Backend(self.weights.clone())), DType::F32, self.device.clone())
    }

    fn vision(&self) -> Result<Arc<Vision>> {
        let mut slot = self.vision.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(v) = slot.as_ref() {
            return Ok(v.clone());
        }
        let v = Arc::new(Vision::new(&self.config, self.vb().pp("vision_model"))?);
        *slot = Some(v.clone());
        Ok(v)
    }

    fn text(&self) -> Result<Arc<Text>> {
        let mut slot = self.text.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(t) = slot.as_ref() {
            return Ok(t.clone());
        }
        let t = Arc::new(Text::new(&self.config, self.vb().pp("text_model"))?);
        *slot = Some(t.clone());
        Ok(t)
    }

    /// The text tower's tokenizer (`tokenizer.model`), read on first use.
    pub fn tokenizer(&self) -> Result<Arc<Tokenizer>> {
        let mut slot = self.tokenizer.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(t) = slot.as_ref() {
            return Ok(t.clone());
        }
        let t = Arc::new(Tokenizer::load(&self.dir.join(TOKENIZER_FILE))?);
        *slot = Some(t.clone());
        Ok(t)
    }

    /// Load the image tower now (else it loads on the first image).
    pub fn load_image_tower(&self) -> Result<()> {
        self.vision().map(|_| ())
    }

    /// Load the text tower now (else it loads on the first text).
    pub fn load_text_tower(&self) -> Result<()> {
        self.text().map(|_| ())
    }

    /// The unit embedding of one [`INPUT`] × [`INPUT`] picture: row-major RGB in 0..1 (sRGB-encoded,
    /// already resized). Values outside 0..1 are clamped, NaN reads as 0.
    pub fn embed_image(&self, rgb: &[[f32; 3]]) -> Result<Vec<f32>> {
        self.embed_images(&[rgb])?.pop().ok_or_else(|| Error::Model("no embedding".into()))
    }

    /// [`Model::embed_image`] for several pictures, [`MAX_BATCH`] per forward pass.
    pub fn embed_images(&self, batch: &[&[[f32; 3]]]) -> Result<Vec<Vec<f32>>> {
        let vision = self.vision()?;
        let mut out = Vec::with_capacity(batch.len());
        for chunk in batch.chunks(MAX_BATCH) {
            let pixels = pixels(chunk, &self.device)?;
            out.extend(unit_rows(&vision.forward(&pixels)?)?);
        }
        Ok(out)
    }

    /// The unit embedding of a tokenised text: [`CONTEXT`] ids (`<eos>`-terminated, `<pad>`-padded,
    /// as [`Tokenizer::encode_padded`] makes them). An id past the vocabulary is an error.
    pub fn embed_text(&self, ids: &[u32; CONTEXT]) -> Result<Vec<f32>> {
        let text = self.text()?;
        let tokens = self.weights.read_rows(TOKEN_EMBEDDING, ids)?.to_device(&self.device)?.unsqueeze(0)?;
        unit_rows(&text.forward(&tokens)?)?.pop().ok_or_else(|| Error::Model("no embedding".into()))
    }

    /// Tokenise `text` and embed it ([`Model::embed_text`]).
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenizer()?.encode_padded(text);
        self.embed_text(&ids)
    }
}

fn read_config(path: &Path) -> Result<Config> {
    use std::io::Read;
    let io = |e: std::io::Error| Error::Model(format!("{}: {e}", path.display()));
    let mut text = String::new();
    std::fs::File::open(path).map_err(io)?.take(MAX_CONFIG_BYTES).read_to_string(&mut text).map_err(io)?;
    Config::from_json(&text)
}

/// The header holds the tensors `c` describes, at their sizes (the rest is checked as each tower
/// loads), and no more encoder layers than the config says.
fn check_header(w: &Weights, c: &Config) -> Result<()> {
    let (vh, th, g, p) = (c.vision.hidden, c.text.hidden, c.grid(), c.patch);
    let want: [(&str, Vec<usize>); 6] = [
        ("vision_model.embeddings.patch_embedding.weight", vec![vh, 3, p, p]),
        ("vision_model.embeddings.position_embedding.weight", vec![g * g, vh]),
        ("vision_model.head.attention.in_proj_weight", vec![3 * vh, vh]),
        (TOKEN_EMBEDDING, vec![c.vocab, th]),
        ("text_model.embeddings.position_embedding.weight", vec![c.context, th]),
        ("text_model.head.weight", vec![c.projection, th]),
    ];
    for (name, shape) in want {
        match w.shape(name) {
            Some(s) if s == shape.as_slice() => {}
            Some(s) => return Err(Error::Model(format!("{name} is {s:?}, the config says {shape:?} (a different checkpoint?)"))),
            None => return Err(Error::Model(format!("{WEIGHTS_FILE} has no {name} (not a SigLIP checkpoint?)"))),
        }
    }
    for (tower, layers) in [("vision_model", c.vision.layers), ("text_model", c.text.layers)] {
        let has = |i: usize| w.contains(&format!("{tower}.encoder.layers.{i}.layer_norm1.weight"));
        if layers == 0 || !has(layers - 1) || has(layers) {
            return Err(Error::Model(format!("{tower}: the weights do not have the config's {layers} layers")));
        }
    }
    Ok(())
}

/// `[B, 3, INPUT, INPUT]` from row-major RGB in 0..1, normalised as the checkpoint's processor does:
/// `(x − 0.5) / 0.5` per channel.
fn pixels(batch: &[&[[f32; 3]]], device: &Device) -> Result<Tensor> {
    let n = INPUT * INPUT;
    let norm = |v: f32| (if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) } - 0.5) / 0.5;
    let mut out = Vec::with_capacity(batch.len().saturating_mul(3 * n));
    for img in batch {
        if img.len() != n {
            return Err(Error::Model(format!("image of {} pixels; the model takes {INPUT} × {INPUT} = {n}", img.len())));
        }
        out.extend(img.iter().map(|p| norm(p[0])));
        out.extend(img.iter().map(|p| norm(p[1])));
        out.extend(img.iter().map(|p| norm(p[2])));
    }
    Ok(Tensor::from_vec(out, (batch.len(), 3, INPUT, INPUT), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// The checkpoint's `config.json`, verbatim.
    const REAL_CONFIG: &str = r#"{
  "initializer_factor": 1.0,
  "model_type": "siglip",
  "text_config": {
    "model_type": "siglip_text_model",
    "vocab_size": 256000
  },
  "transformers_version": "4.49.0.dev0",
  "vision_config": {
    "image_size": 256,
    "model_type": "siglip_vision_model",
    "patch_size": 32
  }
}"#;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("lc-embed-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn is_model_dir_needs_all_three_files() {
        let d = tmp("dir");
        assert!(!crate::is_model_dir(&d));
        std::fs::write(d.join(WEIGHTS_FILE), b"x").unwrap();
        std::fs::write(d.join(CONFIG_FILE), b"{}").unwrap();
        assert!(!crate::is_model_dir(&d));
        std::fs::write(d.join(TOKENIZER_FILE), b"x").unwrap();
        assert!(crate::is_model_dir(&d));
        // a directory where a file should be does not count
        std::fs::remove_file(d.join(TOKENIZER_FILE)).unwrap();
        std::fs::create_dir(d.join(TOKENIZER_FILE)).unwrap();
        assert!(!crate::is_model_dir(&d));
        // and loading it is an error that names the folder, not a panic
        assert!(matches!(Model::load(&d, Device::Cpu), Err(Error::Missing(_))));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn config_reads_the_checkpoint_and_rejects_others() {
        assert_eq!(Config::from_json(REAL_CONFIG).unwrap(), Config::BASE);
        assert_eq!(Config::BASE.grid(), 8);
        // without image/patch size the Transformers defaults (224, 16) apply: not this model
        assert!(Config::from_json(r#"{"text_config":{"vocab_size":256000}}"#).is_err());
        for bad in [
            r#"{"vision_config":{"image_size":256,"patch_size":32,"hidden_act":"gelu"}}"#,
            r#"{"vision_config":{"image_size":256,"patch_size":32,"hidden_size":1152}}"#,
            r#"{"vision_config":{"image_size":256,"patch_size":30}}"#,
            r#"{"vision_config":{"image_size":256,"patch_size":0}}"#,
            r#"{"vision_config":{"image_size":256,"patch_size":32},"text_config":{"max_position_embeddings":-1}}"#,
            r#"{"vision_config":{"image_size":256,"patch_size":32,"layer_norm_eps":"x"}}"#,
            r#"{"vision_config":{"image_size":256,"patch_size":32,"vision_use_head":false}}"#,
            "[1, 2",
        ] {
            assert!(Config::from_json(bad).is_err(), "{bad}");
        }
        // a B/16 config reads (same code, more patches)
        let b16 = Config::from_json(r#"{"vision_config":{"image_size":256,"patch_size":16}}"#).unwrap();
        assert_eq!((b16.grid(), b16.vocab), (16, 32000));
    }

    /// A tiny random-weight SigLIP: 8 × 8 images in 4 × 4 patches, width 8, 2 heads, 2 layers.
    fn tiny() -> (Config, VarMap) {
        let t = TowerConfig { hidden: 8, inter: 16, layers: 2, heads: 2, eps: 1e-6 };
        let c = Config { vision: t.clone(), image: 8, patch: 4, text: t, vocab: 10, context: 5, projection: 8 };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        // create every variable, then give them all random values (layer norms included)
        Vision::new(&c, vb.pp("vision_model")).unwrap();
        Text::new(&c, vb.pp("text_model")).unwrap();
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.5, v.shape(), &Device::Cpu).unwrap()).unwrap();
        }
        (c, vm)
    }

    #[test]
    fn pooling_head_keeps_shapes_and_batch_rows_independent() {
        let (c, vm) = tiny();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let head = PoolingHead::new(&c.vision, vb.pp("vision_model.head")).unwrap();
        let x = Tensor::randn(0f32, 1.0, (3, 4, 8), &Device::Cpu).unwrap();
        let y = head.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 8]);
        let one = head.forward(&x.narrow(0, 1, 1).unwrap()).unwrap();
        let d = to_vec(&(y.narrow(0, 1, 1).unwrap() - one).unwrap().abs().unwrap()).unwrap();
        assert!(d.iter().all(|v| *v < 1e-5), "{d:?}");
    }

    #[test]
    fn tiny_towers_give_unit_embeddings_of_their_width() {
        let (c, vm) = tiny();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let vision = Vision::new(&c, vb.pp("vision_model")).unwrap();
        let px = Tensor::randn(0f32, 0.5, (2, 3, 8, 8), &Device::Cpu).unwrap();
        let e = unit_rows(&vision.forward(&px).unwrap()).unwrap();
        assert_eq!(e.len(), 2);
        for r in &e {
            assert_eq!(r.len(), 8);
            assert!((r.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert!(e[0] != e[1]);
        // the wrong image size is an error
        assert!(vision.forward(&Tensor::zeros((1, 3, 16, 16), DType::F32, &Device::Cpu).unwrap()).is_err());

        let text = Text::new(&c, vb.pp("text_model")).unwrap();
        let tok = Tensor::randn(0f32, 1.0, (1, 5, 8), &Device::Cpu).unwrap();
        let t = unit_rows(&text.forward(&tok).unwrap()).unwrap();
        assert_eq!((t.len(), t[0].len()), (1, 8));
        // bidirectional: the first token changes the last position's state
        let tok2 = Tensor::cat(&[Tensor::zeros((1, 1, 8), DType::F32, &Device::Cpu).unwrap(), tok.narrow(1, 1, 4).unwrap()], 1).unwrap();
        assert!(unit_rows(&text.forward(&tok2).unwrap()).unwrap()[0] != t[0]);
        // an all-zero output cannot be normalised: an error, not NaNs
        assert!(unit_rows(&Tensor::zeros((1, 8), DType::F32, &Device::Cpu).unwrap()).is_err());
    }

    #[test]
    fn pixels_are_normalised_channel_planes() {
        let mut img = vec![[0.0f32, 0.5, 1.0]; INPUT * INPUT];
        img[1] = [f32::NAN, 2.0, -1.0];
        let t = pixels(&[&img], &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, INPUT, INPUT]);
        let v = to_vec(&t).unwrap();
        let n = INPUT * INPUT;
        assert_eq!((v[0], v[n], v[2 * n]), (-1.0, 0.0, 1.0));
        assert_eq!((v[1], v[n + 1], v[2 * n + 1]), (-1.0, 1.0, -1.0));
        assert!(pixels(&[&img[..10]], &Device::Cpu).is_err());
    }

    // ---- integration tests on the real checkpoint (LIGHTCRAFT_SIGLIP2_DIR) ----

    fn model_dir() -> Option<PathBuf> {
        match std::env::var_os("LIGHTCRAFT_SIGLIP2_DIR") {
            Some(d) => Some(PathBuf::from(d)),
            None => {
                eprintln!("skipped: set LIGHTCRAFT_SIGLIP2_DIR to a google/siglip2-base-patch32-256 checkpoint");
                None
            }
        }
    }

    fn srgb(v: f32) -> f32 {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
    }

    /// A demo scene at `side × side`, display-encoded (linear sRGB from the scene, clipped, sRGB curve).
    fn scene_rgb(s: &lightcraft_scenes::Scene, side: usize) -> Vec<[f32; 3]> {
        lightcraft_scenes::render_srgb_linear(s.kind, s.seed, side, side).data.iter().map(|p| p.map(srgb)).collect()
    }

    /// Halve a `side × side` image by 2 × 2 area averaging.
    fn halve(img: &[[f32; 3]], side: usize) -> Vec<[f32; 3]> {
        let h = side / 2;
        let mut out = vec![[0f32; 3]; h * h];
        for y in 0..h {
            for x in 0..h {
                for c in 0..3 {
                    let s = img[2 * y * side + 2 * x][c]
                        + img[2 * y * side + 2 * x + 1][c]
                        + img[(2 * y + 1) * side + 2 * x][c]
                        + img[(2 * y + 1) * side + 2 * x + 1][c];
                    out[y * h + x][c] = s / 4.0;
                }
            }
        }
        out
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn norm(a: &[f32]) -> f32 {
        a.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Resident memory of this process (Linux), for the memory plan.
    fn rss_mb() -> String {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).map(|l| l.trim_start_matches("VmRSS:").trim().to_string()))
            .unwrap_or_else(|| "n/a".into())
    }

    fn scene(name: &str) -> lightcraft_scenes::Scene {
        lightcraft_scenes::demo_library().into_iter().find(|s| s.name == name).unwrap()
    }

    #[test]
    fn checkpoint_image_embeddings_are_stable_and_discriminate() {
        let Some(dir) = model_dir() else { return };
        let m = Model::load(&dir, Device::Cpu).unwrap();
        let lake = scene("Lake at first light");
        let a = m.embed_image(&scene_rgb(&lake, 256)).unwrap();
        let b = m.embed_image(&halve(&scene_rgb(&lake, 512), 512)).unwrap();
        let dunes = m.embed_image(&scene_rgb(&scene("Dune ridges"), 256)).unwrap();
        let aurora = m.embed_image(&scene_rgb(&scene("Aurora over the ridge"), 256)).unwrap();
        for e in [&a, &b, &dunes, &aurora] {
            assert_eq!(e.len(), DIM);
            assert!((norm(e) - 1.0).abs() < 1e-4);
        }
        let (same, ld, la, da) = (cos(&a, &b), cos(&a, &dunes), cos(&a, &aurora), cos(&dunes, &aurora));
        eprintln!("cosine: lake 256 vs lake 512→256 {same:.4}; lake–dunes {ld:.4}; lake–aurora {la:.4}; dunes–aurora {da:.4}");
        assert!(same > 0.9, "{same}");
        assert!(ld < same && la < same && da < same);
    }

    #[test]
    fn checkpoint_text_tower_plumbing_with_hand_written_ids() {
        let Some(dir) = model_dir() else { return };
        let m = Model::load(&dir, Device::Cpu).unwrap();
        // arbitrary in-vocabulary ids, `<eos>` (1), then `<pad>` (0)
        let mut a = [0u32; CONTEXT];
        a[..4].copy_from_slice(&[235250, 13940, 696, 1]);
        let mut b = [0u32; CONTEXT];
        b[..4].copy_from_slice(&[3413, 7432, 3354, 1]);
        let (ea, eb) = (m.embed_text(&a).unwrap(), m.embed_text(&b).unwrap());
        assert_eq!(ea.len(), DIM);
        assert!((norm(&ea) - 1.0).abs() < 1e-4 && (norm(&eb) - 1.0).abs() < 1e-4);
        assert_eq!(m.embed_text(&a).unwrap(), ea, "deterministic");
        assert!(cos(&ea, &eb) < 0.999, "different ids, different embeddings");
        // an id past the 256,000-entry vocabulary is an error
        let mut bad = a;
        bad[0] = 256_000;
        assert!(m.embed_text(&bad).is_err());
    }

    #[test]
    fn checkpoint_zero_shot_lake_at_dawn() {
        let Some(dir) = model_dir() else { return };
        let m = Model::load(&dir, Device::Cpu).unwrap();
        match m.tokenizer() {
            Ok(t) if !t.is_empty() => {}
            _ => {
                eprintln!("skipped: the tokenizer is not functional yet");
                return;
            }
        }
        let img = m.embed_image(&scene_rgb(&scene("Lake at first light"), 256)).unwrap();
        let (good, bad) = (m.embed_query("a lake at dawn").unwrap(), m.embed_query("a city street at night").unwrap());
        let (g, b) = (cos(&img, &good), cos(&img, &bad));
        eprintln!("zero-shot: 'a lake at dawn' {g:.4}, 'a city street at night' {b:.4}");
        assert!(g > b);
    }

    /// Run alone for clean memory numbers:
    /// `cargo test -p lightcraft-embed --release checkpoint_memory_and_throughput -- --nocapture`.
    #[test]
    fn checkpoint_memory_and_throughput() {
        let Some(dir) = model_dir() else { return };
        eprintln!("RSS before load: {}", rss_mb());
        let m = Model::load(&dir, Device::Cpu).unwrap();
        m.load_image_tower().unwrap();
        eprintln!("RSS with the image tower: {}", rss_mb());
        let imgs: Vec<Vec<[f32; 3]>> = lightcraft_scenes::demo_library().iter().take(MAX_BATCH).map(|s| scene_rgb(s, INPUT)).collect();
        let refs: Vec<&[[f32; 3]]> = imgs.iter().map(|v| v.as_slice()).collect();
        let single = m.embed_image(refs[3]).unwrap();
        let t = std::time::Instant::now();
        let out = m.embed_images(&refs).unwrap();
        let s = t.elapsed().as_secs_f64();
        eprintln!(
            "CPU throughput: {} images in {s:.2} s = {:.2} images/s (batch of {MAX_BATCH}, {} threads)",
            out.len(),
            out.len() as f64 / s,
            candle_core::utils::get_num_threads()
        );
        eprintln!("RSS after a batch of {MAX_BATCH}: {}", rss_mb());
        // batching does not change the result
        assert!(cos(&out[3], &single) > 0.9999);
        m.load_text_tower().unwrap();
        let mut ids = [0u32; CONTEXT];
        ids[0] = 1;
        let t = std::time::Instant::now();
        m.embed_text(&ids).unwrap();
        eprintln!("RSS with both towers: {} (one text in {:.0} ms)", rss_mb(), t.elapsed().as_secs_f64() * 1e3);
    }

    /// Against Hugging Face Transformers: with `LIGHTCRAFT_SIGLIP2_REF` set to a folder, the first
    /// run writes the inputs (`scene{i}.rgb.f32`, 256 × 256 × 3 row-major); the reference script
    /// writes `scene{i}.emb.f32` (and `text{j}.ids.u32` / `text{j}.emb.f32`); later runs compare.
    #[test]
    fn checkpoint_matches_transformers_reference() {
        let (Some(dir), Some(rd)) = (model_dir(), std::env::var_os("LIGHTCRAFT_SIGLIP2_REF").map(PathBuf::from)) else {
            eprintln!("skipped: set LIGHTCRAFT_SIGLIP2_REF (and LIGHTCRAFT_SIGLIP2_DIR) for the reference comparison");
            return;
        };
        let f32s = |p: &Path| -> Option<Vec<f32>> {
            std::fs::read(p).ok().map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        };
        std::fs::create_dir_all(&rd).unwrap();
        let names = ["Lake at first light", "Dune ridges", "Aurora over the ridge"];
        let m = Model::load(&dir, Device::Cpu).unwrap();
        let mut compared = 0;
        for (i, n) in names.iter().enumerate() {
            let input = rd.join(format!("scene{i}.rgb.f32"));
            let img = scene_rgb(&scene(n), INPUT);
            if !input.exists() {
                std::fs::write(&input, img.iter().flatten().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            }
            if let Some(r) = f32s(&rd.join(format!("scene{i}.emb.f32"))) {
                let ours = m.embed_image(&img).unwrap();
                let c = cos(&ours, &r) / norm(&r);
                eprintln!("reference image cosine, {n}: {c:.6}");
                assert!(c > 0.999, "{n}: {c}");
                compared += 1;
            }
        }
        for j in 0.. {
            let Ok(b) = std::fs::read(rd.join(format!("text{j}.ids.u32"))) else { break };
            let ids: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let ids: [u32; CONTEXT] = ids.try_into().unwrap();
            let r = f32s(&rd.join(format!("text{j}.emb.f32"))).unwrap();
            let c = cos(&m.embed_text(&ids).unwrap(), &r) / norm(&r);
            eprintln!("reference text cosine, text{j}: {c:.6}");
            assert!(c > 0.999, "text{j}: {c}");
            compared += 1;
        }
        eprintln!("compared {compared} embeddings against the reference in {}", rd.display());
    }
}
