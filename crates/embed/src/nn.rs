//! The small neural-network pieces the towers share: layer norm, linear layers, scaled-dot-product
//! attention, multi-head attention (separate q/k/v projections, and PyTorch's packed
//! `nn.MultiheadAttention`), and the MLP with GELU-tanh.
//!
//! Adapted from `crates/segment/src/nn.rs` (same licence and provenance; this crate does not depend
//! on the segment crate).
//!
//! Modified work (Apache License 2.0, §4(b)): ported by the LightCraft contributors in 2026 from the
//! Python/PyTorch SigLIP code of Hugging Face Transformers (`models/siglip/modeling_siglip.py`),
//! Copyright The HuggingFace Team and Google LLC; translated to Rust on candle and restructured. See
//! NOTICE.

use candle_core::{DType, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use crate::{Error, Result};

pub fn ln(d: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(candle_nn::layer_norm(d, eps, vb)?)
}

pub fn linear(i: usize, o: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(candle_nn::linear(i, o, vb)?)
}

/// Scaled dot-product attention on `[B, H, L, d]` tensors (no mask). SigLIP's sequences are short
/// (64 tokens), so the scores are computed in one block.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let att = (q.matmul(&k.t()?)? * scale)?;
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    Ok(att.matmul(v)?)
}

/// Multi-head attention from projections: `[B, L, hidden]` in, `[B, Lq, hidden]` out.
pub struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    /// `SiglipAttention`: `q_proj`, `k_proj`, `v_proj`, `out_proj` under `vb`.
    pub fn new(hidden: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = head_dim(hidden, heads)?;
        Ok(Self {
            q: linear(hidden, hidden, vb.pp("q_proj"))?,
            k: linear(hidden, hidden, vb.pp("k_proj"))?,
            v: linear(hidden, hidden, vb.pp("v_proj"))?,
            o: linear(hidden, hidden, vb.pp("out_proj"))?,
            heads,
            head_dim,
        })
    }

    /// PyTorch's `nn.MultiheadAttention` (batch first): the q, k and v projections packed in
    /// `in_proj_weight` `[3·hidden, hidden]` / `in_proj_bias` `[3·hidden]` (q rows first, then k,
    /// then v), and `out_proj` under `vb`.
    pub fn packed(hidden: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = head_dim(hidden, heads)?;
        let three = hidden.checked_mul(3).ok_or_else(|| Error::Model("attention width overflows".into()))?;
        let w = vb.get((three, hidden), "in_proj_weight")?;
        let b = vb.get(three, "in_proj_bias")?;
        let part = |i: usize| -> Result<Linear> {
            let wi = w.narrow(0, i * hidden, hidden)?.contiguous()?;
            let bi = b.narrow(0, i * hidden, hidden)?.contiguous()?;
            Ok(Linear::new(wi, Some(bi)))
        };
        Ok(Self { q: part(0)?, k: part(1)?, v: part(2)?, o: linear(hidden, hidden, vb.pp("out_proj"))?, heads, head_dim })
    }

    fn split(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        Ok(x.reshape((b, l, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?)
    }

    /// `q`: `[B, Lq, hidden]`, `kv`: `[B, Lk, hidden]`. No mask: SigLIP attends to every token,
    /// padding included (as it was trained).
    pub fn forward(&self, q: &Tensor, kv: &Tensor) -> Result<Tensor> {
        let (b, lq, _) = q.dims3()?;
        let q = self.split(&self.q.forward(q)?)?;
        let k = self.split(&self.k.forward(kv)?)?;
        let v = self.split(&self.v.forward(kv)?)?;
        let o = sdpa(&q, &k, &v, (self.head_dim as f64).powf(-0.5))?;
        let o = o.transpose(1, 2)?.reshape((b, lq, self.heads * self.head_dim))?;
        Ok(self.o.forward(&o)?)
    }
}

fn head_dim(hidden: usize, heads: usize) -> Result<usize> {
    match hidden.checked_div(heads) {
        Some(d) if d > 0 && d * heads == hidden => Ok(d),
        _ => Err(Error::Model(format!("{hidden} channels do not split into {heads} heads"))),
    }
}

/// `SiglipMLP`: fc1 → GELU, tanh approximation (`gelu_pytorch_tanh`) → fc2.
///
/// candle's `Tensor::gelu` is the tanh approximation; `gelu_erf` would be the exact GELU.
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn new(hidden: usize, inter: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { fc1: linear(hidden, inter, vb.pp("fc1"))?, fc2: linear(inter, hidden, vb.pp("fc2"))? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.gelu()?;
        Ok(self.fc2.forward(&x)?)
    }
}

/// A tensor's values as `f32` on the CPU.
pub fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    #[test]
    fn mlp_gelu_is_the_tanh_approximation() {
        let dev = Device::Cpu;
        let mut t = HashMap::new();
        // identity fc1/fc2 on one channel: the MLP is the activation itself
        t.insert("fc1.weight".to_string(), Tensor::ones((1, 1), DType::F32, &dev).unwrap());
        t.insert("fc1.bias".to_string(), Tensor::zeros(1, DType::F32, &dev).unwrap());
        t.insert("fc2.weight".to_string(), Tensor::ones((1, 1), DType::F32, &dev).unwrap());
        t.insert("fc2.bias".to_string(), Tensor::zeros(1, DType::F32, &dev).unwrap());
        let mlp = Mlp::new(1, 1, VarBuilder::from_tensors(t, DType::F32, &dev)).unwrap();
        let x = -1.5f32;
        let y = to_vec(&mlp.forward(&Tensor::new(&[[x]], &dev).unwrap()).unwrap()).unwrap()[0];
        let tanh = 0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
        assert!((y - tanh).abs() < 1e-6, "{y} vs {tanh}");
        // and not the exact (erf) GELU, −1.5·Φ(−1.5) = −0.1002108, which differs here by ~2e-4
        assert!((y - (-0.100_210_8)).abs() > 1e-4, "{y}");
    }

    #[test]
    fn packed_attention_matches_separate_projections() {
        let dev = Device::Cpu;
        let (h, heads) = (8, 2);
        let w = Tensor::randn(0f32, 1.0, (3 * h, h), &dev).unwrap();
        let b = Tensor::randn(0f32, 1.0, 3 * h, &dev).unwrap();
        let ow = Tensor::randn(0f32, 1.0, (h, h), &dev).unwrap();
        let ob = Tensor::randn(0f32, 1.0, h, &dev).unwrap();
        let mut packed = HashMap::new();
        packed.insert("in_proj_weight".to_string(), w.clone());
        packed.insert("in_proj_bias".to_string(), b.clone());
        packed.insert("out_proj.weight".to_string(), ow.clone());
        packed.insert("out_proj.bias".to_string(), ob.clone());
        let mut split = HashMap::new();
        for (i, n) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
            split.insert(format!("{n}.weight"), w.narrow(0, i * h, h).unwrap().contiguous().unwrap());
            split.insert(format!("{n}.bias"), b.narrow(0, i * h, h).unwrap().contiguous().unwrap());
        }
        split.insert("out_proj.weight".to_string(), ow);
        split.insert("out_proj.bias".to_string(), ob);
        let a = Attention::packed(h, heads, VarBuilder::from_tensors(packed, DType::F32, &dev)).unwrap();
        let s = Attention::new(h, heads, VarBuilder::from_tensors(split, DType::F32, &dev)).unwrap();
        let q = Tensor::randn(0f32, 1.0, (2, 1, h), &dev).unwrap();
        let kv = Tensor::randn(0f32, 1.0, (2, 5, h), &dev).unwrap();
        let (ya, ys) = (a.forward(&q, &kv).unwrap(), s.forward(&q, &kv).unwrap());
        assert_eq!(ya.dims(), &[2, 1, h]);
        let d = to_vec(&(ya - ys).unwrap().abs().unwrap()).unwrap();
        assert!(d.iter().all(|v| *v < 1e-5));
        // a width that does not split into the heads is an error, not a panic
        assert!(head_dim(10, 3).is_err() && head_dim(8, 0).is_err());
    }
}
