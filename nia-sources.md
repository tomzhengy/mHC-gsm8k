# Nia Sources Tracking

## Research Papers

| Title | arXiv ID | Source ID | Status | Notes |
|-------|----------|-----------|--------|-------|
| mHC: Manifold-Constrained Hyper-Connections | 2512.24880 | 820ff393-cc17-41e7-9e22-b2df85e7dd92 | ✅ Indexed | Core paper - DeepSeek's constrained multi-stream architecture |
| Hyper-Connections (original HC) | 2409.19606 | 74ac30e3-e390-47af-8857-a2f9db589690 | ⏳ Processing | Original HC paper showing expanded residual streams |

## Repositories

_None indexed yet for this project_

## Documentation

| Name | URL | Source ID | Status | Notes |
|------|-----|-----------|--------|-------|
| HuggingFace Transformers | huggingface.co/docs | 01a925fc-baac-43b2-9265-df37bf9cb355 | ✅ Indexed | General HF docs |
| HF Model Customization | how_to_hack_models | 1638a3b2-61ad-43d1-b1fd-34c8b5d2ca35 | ⏳ Processing | Custom forward pass hooks |
| CleanRL PPO | docs.cleanrl.dev/ppo | a79ecbc2-d817-4319-813c-f12a0212eb8b | ✅ Indexed | PPO implementation reference |

---

## Quick Reference

### mHC Paper Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Expansion Rate λ | 4 | Number of streams |
| Gating Factor Init | 0.01 | Small to start near identity |
| Sinkhorn ε_max | 20 | Iterations for doubly-stochastic projection |
| Benchmarks | GSM8K, BBH, DROP, MMLU | Standard evaluation |

### Key Equations

**HC multi-stream propagation:**
```
x_{l+1} = H^res_l · x_l + H^post_l · F(H^pre_l · x_l, W_l)
```

**Your simplified MVP:**
```
S = (1-g) * S + g * (M @ S)   # where M is row-stochastic mixing matrix
```

### Implementation Notes from Research

1. **Mixing matrix init**: Start with strong diagonal bias (identity-like) - paper uses 0.01 gating factor
2. **Row-stochastic constraint**: `softmax(logits, dim=-1)` for MVP, Sinkhorn for full mHC
3. **Only main stream through blocks**: Other streams are "latent scratchpads" - reduces compute
4. **Per-layer hooks**: Use PyTorch `register_forward_hook` or subclass decoder layers

### PPO Hyperparameters for Sparse Rewards

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Learning rate | 1e-4 to 3e-4 | Start lower for stability |
| Entropy coef | 0.05-0.1 | Higher for exploration |
| Batch size | 1024-2048 | Larger helps with rare rewards |
| GAE λ | 0.95-0.99 | Credit assignment |
| Discount γ | 0.99+ | Long-term planning |
| Clip range | 0.1-0.2 | Standard PPO |
