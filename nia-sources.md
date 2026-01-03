# Nia Sources Tracking

## Research Papers

| Title | arXiv ID | Source ID | Status | Notes |
|-------|----------|-----------|--------|-------|
| mHC: Manifold-Constrained Hyper-Connections | 2512.24880 | 820ff393-cc17-41e7-9e22-b2df85e7dd92 | ✅ Indexed | DeepSeek's constrained multi-stream - adds Sinkhorn normalization to fix HC instability |
| Hyper-Connections (original HC) | 2409.19606 | 74ac30e3-e390-47af-8857-a2f9db589690 | ✅ Indexed | Original HC paper - expansion rate ablations, n=4 optimal |

## Repositories

| Name | Source ID | Status | Notes |
|------|-----------|--------|-------|
| tokenbender/mHC-manifold-constrained-hyper-connections | indexed | ✅ Indexed | nanoGPT + mHC implementation, has HC and mHC configs |

## Documentation

| Name | URL | Source ID | Status | Notes |
|------|-----|-----------|--------|-------|
| HuggingFace Transformers | huggingface.co/docs | 01a925fc-baac-43b2-9265-df37bf9cb355 | ✅ Indexed | General HF docs |
| HF Model Customization | how_to_hack_models | 1638a3b2-61ad-43d1-b1fd-34c8b5d2ca35 | ⏳ Processing | Custom forward pass hooks |
| CleanRL PPO | docs.cleanrl.dev/ppo | a79ecbc2-d817-4319-813c-f12a0212eb8b | ✅ Indexed | PPO implementation reference |

---

## Quick Reference

### Key Insight: How mHC Replaces Residual Connections

**Standard Residual:**
```
x_{l+1} = x_l + F(x_l, W_l)  # single stream, identity shortcut
```

**Hyper-Connections (HC):**
```
x_{l+1} = H^res_l · x_l + H^post_l · F(H^pre_l · x_l, W_l)
```
- Expands single stream to n parallel hidden vectors
- H^res, H^pre, H^post are learnable matrices
- **Problem**: Loses identity mapping property → training instability

**mHC (Manifold-Constrained HC):**
- Same structure as HC but projects H^res onto **doubly-stochastic manifold** via Sinkhorn
- Doubly stochastic = all rows sum to 1 AND all columns sum to 1
- Restores stable gradient flow like residual connections

### Tunable Parameters in HC/mHC

| Parameter | Description | Values Tested | Optimal |
|-----------|-------------|---------------|---------|
| **Expansion Rate (n)** | Number of parallel streams | 1, 2, 4, 8 | **n=4** |
| **Gating Factor Init (α)** | Initial mixing strength | 0.01 | 0.01 |
| **Sinkhorn ε_max** | Iterations for doubly-stochastic projection | 20 | 20 |
| **Static vs Dynamic** | Fixed weights vs input-dependent | SHC, DHC | DHC (dynamic) |
| **tanh activation** | Non-linearity on dynamic weights | with/without | with tanh |

#### Expansion Rate Ablation (from HC paper, Table 1):

| n | V2 Eval Loss | V3 Eval Loss | Downstream Avg |
|---|--------------|--------------|----------------|
| 1 | 2.822 (+0.011) | 2.556 (+0.012) | 62.3% (-0.2%) |
| 2 | 2.792 (-0.019) | 2.537 (-0.007) | 63.8% (+1.3%) |
| 4 | 2.779 (-0.032) | 2.516 (-0.028) | **64.4% (+1.9%)** |
| 8 | 2.777 (-0.034) | 2.514 (-0.030) | 63.8% (+1.3%) |

**Key findings:**
- n=1 is **worse** than baseline (seesaw effect)
- n=4 is optimal (best downstream accuracy)
- n=8 provides minimal additional benefits

### The Mixing Matrix (ℋ𝒞)

The (n+1) × (n+1) hyper-connection matrix:
```
ℋ𝒞 = | 0      β₁    β₂    ...  βₙ   |
      | α₁,₀   α₁,₁  α₁,₂  ...  α₁,ₙ |
      | α₂,₀   α₂,₁  α₂,₂  ...  α₂,ₙ |
      | ⋮      ⋮     ⋮     ⋱    ⋮    |
      | αₙ,₀   αₙ,₁  αₙ,₂  ...  αₙ,ₙ |

- β parameters: Depth-connections (vertical, n params)
- α parameters: Width-connections (lateral, n×(n+1) params)
- Total per layer: n×(n+2) params per HC module × 2 modules (attn + FFN)
```

### Implementation Notes

1. **Static HC (SHC)**: β, α are fixed learnable params
2. **Dynamic HC (DHC)**: β, α are predicted from input via small projections
3. **Sinkhorn-Knopp (mHC only)**: Iteratively normalize rows/columns to make doubly-stochastic
4. **Two HC modules per layer**: One after attention, one after FFN

### PPO Hyperparameters for Sparse Rewards

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Learning rate | 1e-4 to 3e-4 | Start lower for stability |
| Entropy coef | 0.05-0.1 | Higher for exploration |
| Batch size | 1024-2048 | Larger helps with rare rewards |
| GAE λ | 0.95-0.99 | Credit assignment |
| Discount γ | 0.99+ | Long-term planning |
| Clip range | 0.1-0.2 | Standard PPO |
