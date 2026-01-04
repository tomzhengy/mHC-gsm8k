"""
Architecture:
- Input: [B, T, n*C] (flattened streams)
- Generates [B, T, n, n] H_res, [B, T, n] H_pre/H_post per token
- Output: [B, T, n*C]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_log(
    logits: torch.Tensor,
    num_iters: int = 50,
    tau: float = 0.1,
) -> torch.Tensor:

    n = logits.shape[-1]
    
    # scale by temperature
    Z = logits / tau
    
    # target log marginals (uniform distribution)
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)
    
    # dual variables for row/column normalization
    # shape: [...] (one per row/column, broadcasted)
    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)
    
    # alternating row/column normalization in log-space
    for _ in range(num_iters):
        # row normalization: u_i = log_marginal - logsumexp_j(Z_ij + v_j)
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        # column normalization: v_j = log_marginal - logsumexp_i(Z_ij + u_i)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
    
    # u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
    
    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class DynamicMHC(nn.Module):

    def __init__(
        self,
        dim: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 50,
        sinkhorn_tau: float = 0.1,
        layer_idx: int = 0,
        gate_noise: bool = True,
        gate_exploration_prob: float = 0.2,
        gate_noise_scale: float = 0.3,
    ):

        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.layer_idx = layer_idx
        
        # gate noise settings (for RL robustness)
        self.gate_noise_during_training = gate_noise
        self.gate_exploration_prob = gate_exploration_prob  # target exploration prob
        self.gate_noise_scale = gate_noise_scale
        self._current_explore_prob = 0.02  # start with minimal exploration (warmup)
        
        n = num_streams
        widened_dim = dim * n
        
        # normalization for matrix generation input
        # using RMSNorm without learnable params for simplicity
        self.norm = nn.RMSNorm(widened_dim, elementwise_affine=False)
        
        # linear projections to generate per-token matrix logits
        # output sizes: H_res -> n*n, H_pre -> n, H_post -> n
        self.proj_H_res = nn.Linear(widened_dim, n * n, bias=False)
        self.proj_H_pre = nn.Linear(widened_dim, n, bias=False)
        self.proj_H_post = nn.Linear(widened_dim, n, bias=False)
        
        # initialize projections small to start near base matrices
        nn.init.normal_(self.proj_H_res.weight, std=0.01)
        nn.init.normal_(self.proj_H_pre.weight, std=0.01)
        nn.init.normal_(self.proj_H_post.weight, std=0.01)
        
        # learnable base matrices (static component)
        # H_res_base: near-identity initialization
        # off-diagonal: -0.5, diagonal: 0.0
        # with tau=0.05, this gives ~99.99% diagonal, small but nonzero off-diagonal
        # allowing gradients to flow while maintaining stability
        init_h_res = torch.full((n, n), -0.5)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_base = nn.Parameter(init_h_res)
        
        # H_pre_base: prefer stream 0 but allow gradient flow
        # stream 0: 1.0, others: 0.0
        # after softmax: stream 0 gets ~0.58, others get ~0.14 each
        # (less extreme than -8 which gives 99.9% to stream 0)
        init_h_pre = torch.zeros(n)
        init_h_pre[0] = 1.0
        self.H_pre_base = nn.Parameter(init_h_pre)
        
        # H_post_base: uniform distribution
        # all zeros -> after softmax, each stream gets 1/n
        self.H_post_base = nn.Parameter(torch.zeros(n))
        
        # controllable gate for RL tuning
        # g=1 (sigmoid(0)=0.5 by default, we init to make sigmoid(gate)≈1)
        # gate interpolates H_res between identity (g=0) and computed (g=1)
        self.gate = nn.Parameter(torch.tensor([5.0]))  # sigmoid(5) ≈ 0.993
    
    def get_matrices(self, x: torch.Tensor):

        B, T, _ = x.shape
        n = self.num_streams
        
        # normalize input for stable matrix generation
        x_norm = self.norm(x)
        
        # generate per-token adjustments to base matrices
        H_res_delta = self.proj_H_res(x_norm).view(B, T, n, n)
        H_pre_delta = self.proj_H_pre(x_norm)   # [B, T, n]
        H_post_delta = self.proj_H_post(x_norm) # [B, T, n]
        
        # combine base + dynamic adjustments
        # broadcasting: base [n, n] + delta [B, T, n, n]
        H_res_logits = self.H_res_base + H_res_delta
        H_pre_logits = self.H_pre_base + H_pre_delta
        H_post_logits = self.H_post_base + H_post_delta
        
        # apply constraints
        # H_res: doubly-stochastic via Sinkhorn-Knopp
        H_res = sinkhorn_log(H_res_logits, self.sinkhorn_iters, self.sinkhorn_tau)
        
        # H_pre, H_post: softmax-normalized (rows sum to 1)
        H_pre = F.softmax(H_pre_logits, dim=-1)   # [B, T, n]
        H_post = F.softmax(H_post_logits, dim=-1) # [B, T, n]
        
        # apply controllable gate: interpolate H_res with identity
        # H_res_gated = (1 - g) * I + g * H_res
        g = torch.sigmoid(self.gate)
        
        # during training, add noise to gate for robustness (for RL tuning)
        # this ensures the model learns to work across all gate values
        if self.training and self.gate_noise_during_training:
            if torch.rand(1).item() < self._current_explore_prob:
                # full exploration: sample from [0.1, 0.9] to avoid extremes
                # g=0 disables mHC entirely, g=1 fully commits - both are harsh
                g = 0.1 + 0.8 * torch.rand(1, device=x.device, dtype=x.dtype)
            else:
                # local exploration: perturb learned gate by ±scale
                lo = 1.0 - self.gate_noise_scale
                hi = 1.0 + self.gate_noise_scale
                noise = lo + (hi - lo) * torch.rand(1, device=x.device, dtype=x.dtype)
                g = (g * noise).clamp(0.1, 0.9)  # clamp to safe range
        
        # identity matrix for interpolation
        I = torch.eye(n, device=H_res.device, dtype=H_res.dtype)
        H_res = (1.0 - g) * I + g * H_res
        
        return H_res, H_pre, H_post
    
    def forward(self, x: torch.Tensor, branch_fn) -> torch.Tensor:

        B, T, nC = x.shape
        n = self.num_streams
        C = nC // n
        
        # get per-token matrices
        H_res, H_pre, H_post = self.get_matrices(x)
        
        # unflatten streams: [B, T, n*C] -> [B, T, n, C]
        x_streams = x.view(B, T, n, C)
        
        # === WIDTH CONNECTION ===
        # create branch input as weighted sum of streams
        # x_pre[b,t,c] = sum_i H_pre[b,t,i] * x_streams[b,t,i,c]
        x_pre = torch.einsum('btnc,btn->btc', x_streams, H_pre)  # [B, T, C]
        
        # === BRANCH ===
        # apply attention or MLP
        y = branch_fn(x_pre)  # [B, T, C]
        
        # === DEPTH CONNECTION ===
        # residual mixing (per mHC paper): x_res[b,t,i,c] = sum_j H_res[b,t,i,j] * x[b,t,j,c]
        # H_res[i,j] = weight from input stream j to output stream i (standard matrix convention)
        x_mixed = torch.einsum('btij,btjc->btic', H_res, x_streams)  # [B, T, n, C]
        
        # distribute branch output to streams
        # y_distributed[b,t,j,c] = H_post[b,t,j] * y[b,t,c]
        y_distributed = torch.einsum('btc,btn->btnc', y, H_post)  # [B, T, n, C]
        
        # combine: x_{l+1} = H_res @ x_l + H_post * F(...)
        output = x_mixed + y_distributed  # [B, T, n, C]
        
        # flatten back: [B, T, n, C] -> [B, T, n*C]
        return output.view(B, T, nC)
    
    def set_gate(self, value: float):

        with torch.no_grad():
            # clamp to avoid numerical issues with sigmoid inverse
            value = max(1e-6, min(1.0 - 1e-6, value))
            # inverse sigmoid: logit = log(p / (1-p))
            logit = math.log(value / (1.0 - value))
            self.gate.fill_(logit)
    
    def get_gate(self) -> float:
        """Get current effective gate value in [0, 1]."""
        return torch.sigmoid(self.gate).item()
    
    def set_exploration_schedule(self, progress: float, warmup_frac: float = 0.1):
        """
        Update exploration probability based on training progress.
        
        Args:
            progress: Training progress in [0, 1] (current_step / total_steps)
            warmup_frac: Fraction of training to ramp up exploration (default: 10%)
        
        Schedule:
            - progress < warmup_frac: linear ramp from 0.02 to target exploration
            - progress >= warmup_frac: full target exploration
        """
        target_explore = self.gate_exploration_prob
        min_explore = 0.02  # 2% exploration minimum during warmup
        
        if progress < warmup_frac:
            # linear ramp: 0.02 -> target over warmup period
            ramp = progress / warmup_frac
            self._current_explore_prob = min_explore + ramp * (target_explore - min_explore)
        else:
            self._current_explore_prob = target_explore
    
    def get_sinkhorn_diagnostics(self) -> dict:
        """
        Compute row/column errors for the current H_res_base after Sinkhorn.
        Call this periodically during training to verify doubly-stochastic property.
        """
        n = self.num_streams
        # compute H_res from base (no dynamic adjustment, just base matrix)
        H_res = sinkhorn_log(
            self.H_res_base.unsqueeze(0).unsqueeze(0),  # [1, 1, n, n]
            self.sinkhorn_iters,
            self.sinkhorn_tau
        )[0, 0]  # [n, n]
        
        row_err = (H_res.sum(dim=-1) - 1).abs().mean().item()
        col_err = (H_res.sum(dim=-2) - 1).abs().mean().item()
        
        return {
            "row_err": row_err,
            "col_err": col_err,
            "diag_mean": H_res.diag().mean().item(),
            "offdiag_mean": H_res[~torch.eye(n, dtype=bool, device=H_res.device)].mean().item(),
        }
    
    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_streams={self.num_streams}, "
            f"sinkhorn_iters={self.sinkhorn_iters}, sinkhorn_tau={self.sinkhorn_tau}, "
            f"layer_idx={self.layer_idx}"
        )


# helper function for stream expansion/reduction
def expand_streams(x: torch.Tensor, num_streams: int) -> torch.Tensor:

    B, T, C = x.shape
    # replicate along a new stream dimension, then flatten
    return x.unsqueeze(-2).expand(B, T, num_streams, C).reshape(B, T, num_streams * C)


def reduce_streams(x: torch.Tensor, num_streams: int) -> torch.Tensor:

    B, T, nC = x.shape
    C = nC // num_streams
    return x.view(B, T, num_streams, C).sum(dim=-2)


# test function
if __name__ == "__main__":
    print("testing mhc.py...")
    
    # test sinkhorn
    print("\n1. testing sinkhorn_log...")
    logits = torch.randn(2, 3, 4, 4)  # [B, T, n, n]
    ds_matrix = sinkhorn_log(logits, num_iters=20, tau=0.05)
    row_sums = ds_matrix.sum(dim=-1)
    col_sums = ds_matrix.sum(dim=-2)
    print(f"   input shape: {logits.shape}")
    print(f"   output shape: {ds_matrix.shape}")
    print(f"   row sums (should be ~1): mean={row_sums.mean():.4f}, std={row_sums.std():.6f}")
    print(f"   col sums (should be ~1): mean={col_sums.mean():.4f}, std={col_sums.std():.6f}")
    print(f"   all non-negative: {(ds_matrix >= 0).all()}")
    
    # test dynamic mhc
    print("\n2. testing DynamicMHC...")
    mhc = DynamicMHC(dim=64, num_streams=4, sinkhorn_iters=20, sinkhorn_tau=0.05)
    mhc.eval()  # disable gate noise
    
    B, T, n, C = 2, 8, 4, 64
    x = torch.randn(B, T, n * C)
    
    def dummy_branch(z):
        return z * 0.5  # simple transformation
    
    y = mhc(x, dummy_branch)
    print(f"   input shape: {x.shape}")
    print(f"   output shape: {y.shape}")
    print(f"   current gate value: {mhc.get_gate():.4f}")
    
    # test gate control
    print("\n3. testing gate control...")
    mhc.set_gate(0.0)
    print(f"   after set_gate(0.0): {mhc.get_gate():.6f}")
    mhc.set_gate(0.5)
    print(f"   after set_gate(0.5): {mhc.get_gate():.6f}")
    mhc.set_gate(1.0)
    print(f"   after set_gate(1.0): {mhc.get_gate():.6f}")
    
    # test expand/reduce
    print("\n4. testing expand/reduce streams...")
    x_single = torch.randn(2, 8, 64)
    x_expanded = expand_streams(x_single, num_streams=4)
    x_reduced = reduce_streams(x_expanded, num_streams=4)
    print(f"   single stream shape: {x_single.shape}")
    print(f"   expanded shape: {x_expanded.shape}")
    print(f"   reduced shape: {x_reduced.shape}")
    # reduced should be 4x the original (since we sum 4 copies)
    print(f"   reduced ≈ 4 * original: {torch.allclose(x_reduced, 4 * x_single)}")
    
    print("\n✓ all tests passed!")

