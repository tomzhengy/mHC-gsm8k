"""
Routing module for multi-stream residual connections.

Provides:
- MultiStreamDecoder: Wrapper for HF causal LMs with controllable mixing
- MixingMatrix: Learnable row/doubly-stochastic mixing matrices
- greedy_decode/sample_decode: Generation utilities that accept mixing gate g
"""

from .mixing_ops import (
    MixingMatrix,
    row_stochastic,
    sinkhorn,
    apply_mixing,
    compute_stream_diagnostics,
)
from .multistream_wrapper import (
    MultiStreamDecoder,
    MultiStreamOutput,
    greedy_decode,
    sample_decode,
)

__all__ = [
    # Main wrapper
    "MultiStreamDecoder",
    "MultiStreamOutput",
    # Mixing operations
    "MixingMatrix",
    "row_stochastic",
    "sinkhorn",
    "apply_mixing",
    "compute_stream_diagnostics",
    # Generation utilities
    "greedy_decode",
    "sample_decode",
]

