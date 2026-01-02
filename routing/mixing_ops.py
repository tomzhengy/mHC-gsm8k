"""
Mixing matrix operations for multi-stream residual connections.

Provides both row-stochastic (MVP) and doubly-stochastic (Sinkhorn) 
mixing matrices for stream interpolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


def row_stochastic(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to row-stochastic matrix via softmax.
    
    Each row sums to 1 (convex combination per output stream).
    
    Args:
        logits: (n, n) matrix of unnormalized logits
        
    Returns:
        (n, n) row-stochastic matrix
    """
    return F.softmax(logits, dim=-1)


def sinkhorn(
    logits: torch.Tensor,
    n_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Project logits to doubly-stochastic matrix via Sinkhorn-Knopp.
    
    Doubly-stochastic: rows AND columns sum to 1.
    This provides stronger stability guarantees:
    - No amplification (spectral radius ≤ 1)
    - No collapse (all streams receive input)
    - Closure under composition
    
    Args:
        logits: (n, n) matrix of unnormalized logits
        n_iters: Number of Sinkhorn iterations (mHC uses ~20)
        eps: Small constant for numerical stability
        
    Returns:
        (n, n) doubly-stochastic matrix
    """
    # Start with exp(logits) to ensure positivity
    M = torch.exp(logits)
    
    for _ in range(n_iters):
        # Row normalization
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        # Column normalization  
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    
    return M


class MixingMatrix(nn.Module):
    """
    Learnable mixing matrix for multi-stream residual connections.
    
    Supports row-stochastic (MVP) and doubly-stochastic (Sinkhorn) modes.
    Initialized near-identity for stable training.
    """
    
    def __init__(
        self,
        n_streams: int = 4,
        mode: Literal["row_stochastic", "sinkhorn"] = "row_stochastic",
        init_scale: float = 4.0,
        sinkhorn_iters: int = 20,
    ):
        """
        Args:
            n_streams: Number of residual streams
            mode: "row_stochastic" (softmax) or "sinkhorn" (doubly-stochastic)
            init_scale: Scale for diagonal initialization (higher = more identity-like)
            sinkhorn_iters: Iterations for Sinkhorn projection
        """
        super().__init__()
        self.n_streams = n_streams
        self.mode = mode
        self.sinkhorn_iters = sinkhorn_iters
        
        # Initialize with strong diagonal bias → near-identity behavior
        # When init_scale=4.0: diagonal ~97% after softmax
        logits = torch.eye(n_streams) * init_scale
        self.logits = nn.Parameter(logits)
    
    def forward(self) -> torch.Tensor:
        """
        Returns:
            (n_streams, n_streams) stochastic mixing matrix
        """
        if self.mode == "row_stochastic":
            return row_stochastic(self.logits)
        elif self.mode == "sinkhorn":
            return sinkhorn(self.logits, n_iters=self.sinkhorn_iters)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    @property
    def matrix(self) -> torch.Tensor:
        """Convenience property to get current mixing matrix."""
        return self.forward()
    
    def get_diagnostics(self) -> dict:
        """
        Return diagnostic info about the mixing matrix.
        
        Useful for logging and debugging stability issues.
        """
        with torch.no_grad():
            M = self.forward()
            return {
                "mix_diagonal_mean": M.diagonal().mean().item(),
                "mix_diagonal_min": M.diagonal().min().item(),
                "mix_off_diagonal_max": (M - torch.diag(M.diagonal())).max().item(),
                "mix_row_sums_std": M.sum(dim=-1).std().item(),
                "mix_col_sums_std": M.sum(dim=-2).std().item(),
            }


def apply_mixing(
    streams: torch.Tensor,
    mixing_matrix: torch.Tensor,
    gate: float | torch.Tensor,
) -> torch.Tensor:
    """
    Apply gated mixing across streams.
    
    S_new = (1-g) * S + g * (M @ S)
    
    When g=0: identity (no mixing)
    When g=1: full mixing according to M
    
    Args:
        streams: (n_streams, batch, seq_len, hidden_dim) tensor
        mixing_matrix: (n_streams, n_streams) stochastic matrix
        gate: scalar or tensor in [0, 1] controlling mixing strength
        
    Returns:
        Mixed streams with same shape as input
    """
    # Ensure gate is a tensor with correct dtype/device
    if not isinstance(gate, torch.Tensor):
        gate = torch.tensor(gate, device=streams.device, dtype=streams.dtype)
    
    # Apply mixing: einsum over stream dimension
    # M[i,j] * S[j] summed over j gives new S[i]
    streams_mixed = torch.einsum("ij,jbth->ibth", mixing_matrix, streams)
    
    # Interpolate between identity and mixed
    return (1.0 - gate) * streams + gate * streams_mixed


def compute_stream_diagnostics(streams: torch.Tensor) -> dict:
    """
    Compute diagnostic metrics for stream health.
    
    Call this periodically during forward pass to monitor stability.
    
    Args:
        streams: (n_streams, batch, seq_len, hidden_dim)
        
    Returns:
        Dict of diagnostic metrics
    """
    with torch.no_grad():
        # Per-stream L2 norms (averaged over batch and sequence)
        stream_norms = streams.norm(dim=-1).mean(dim=(1, 2))  # (n_streams,)
        
        # Stream diversity: how different are the streams from each other?
        # Use variance across stream dimension
        stream_mean = streams.mean(dim=0, keepdim=True)
        diversity = ((streams - stream_mean) ** 2).mean()
        
        return {
            "stream_norm_mean": stream_norms.mean().item(),
            "stream_norm_std": stream_norms.std().item(),
            "stream_norm_max": stream_norms.max().item(),
            "stream_norm_min": stream_norms.min().item(),
            "stream_norm_ratio": (stream_norms.max() / (stream_norms.min() + 1e-8)).item(),
            "stream_diversity": diversity.item(),
        }

