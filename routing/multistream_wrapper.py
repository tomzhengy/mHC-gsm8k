"""
Multi-stream decoder wrapper for HuggingFace causal LM models.

Wraps a decoder-only transformer (Qwen, Llama, etc.) with n residual streams.
Only stream 0 is forwarded through each block; other streams are mixed between layers.

This design:
- Keeps base model weights frozen
- Adds minimal overhead (mixing is cheap)
- Allows RL to control mixing gate g
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Literal
from dataclasses import dataclass

from .mixing_ops import MixingMatrix, apply_mixing, compute_stream_diagnostics


@dataclass
class MultiStreamOutput:
    """Output from MultiStreamDecoder forward pass."""
    logits: torch.Tensor
    # Diagnostics for logging/debugging
    stream_diagnostics: Optional[dict] = None
    mixing_diagnostics: Optional[dict] = None


class MultiStreamDecoder(nn.Module):
    """
    Wraps a decoder-only transformer with n residual streams.
    
    Architecture:
        - Maintain n streams, each (batch, seq, hidden)
        - Only stream 0 goes through each transformer block
        - Between layers: apply gated mixing across all streams
        - Output: stream 0 after final norm → lm_head
    
    The mixing allows information to flow between streams,
    controlled by gate g ∈ [0, 1]:
        - g=0: no mixing (identity, baseline-equivalent)
        - g=1: full mixing according to learned matrix M
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> wrapped = MultiStreamDecoder(base, n_streams=4)
        >>> wrapped.freeze_base()
        >>> out = wrapped(input_ids, g=0.5)
    """
    
    # Supported model architectures (attribute paths may differ)
    SUPPORTED_ARCHS = {
        "qwen2": {"layers": "model.layers", "embed": "model.embed_tokens", "norm": "model.norm", "head": "lm_head"},
        "llama": {"layers": "model.layers", "embed": "model.embed_tokens", "norm": "model.norm", "head": "lm_head"},
        "mistral": {"layers": "model.layers", "embed": "model.embed_tokens", "norm": "model.norm", "head": "lm_head"},
        "gemma": {"layers": "model.layers", "embed": "model.embed_tokens", "norm": "model.norm", "head": "lm_head"},
        "gemma2": {"layers": "model.layers", "embed": "model.embed_tokens", "norm": "model.norm", "head": "lm_head"},
    }
    
    def __init__(
        self,
        base_model: nn.Module,
        n_streams: int = 4,
        mixing_mode: Literal["row_stochastic", "sinkhorn"] = "row_stochastic",
        mixing_init_scale: float = 4.0,
        sinkhorn_iters: int = 20,
        collect_diagnostics: bool = False,
    ):
        """
        Args:
            base_model: HuggingFace causal LM (e.g., Qwen2ForCausalLM)
            n_streams: Number of residual streams (mHC uses 4)
            mixing_mode: "row_stochastic" (MVP) or "sinkhorn" (full mHC)
            mixing_init_scale: Diagonal bias for near-identity init
            sinkhorn_iters: Iterations for Sinkhorn projection
            collect_diagnostics: Whether to compute stream diagnostics each forward
        """
        super().__init__()
        self.base = base_model
        self.n_streams = n_streams
        self.collect_diagnostics = collect_diagnostics
        
        # Detect architecture and get attribute paths
        self._arch_config = self._detect_architecture()
        
        # Learnable mixing matrix
        self.mixing = MixingMatrix(
            n_streams=n_streams,
            mode=mixing_mode,
            init_scale=mixing_init_scale,
            sinkhorn_iters=sinkhorn_iters,
        )
        
        # Cache commonly accessed modules
        self._layers = self._get_attr(self._arch_config["layers"])
        self._embed = self._get_attr(self._arch_config["embed"])
        self._norm = self._get_attr(self._arch_config["norm"])
        self._head = self._get_attr(self._arch_config["head"])
        
        # Get rotary embedding module if available (for newer HF models)
        self._rotary_emb = None
        if hasattr(self.base.model, "rotary_emb"):
            self._rotary_emb = self.base.model.rotary_emb
        
        self.num_layers = len(self._layers)
    
    def _detect_architecture(self) -> dict:
        """Detect model architecture from config."""
        model_type = getattr(self.base.config, "model_type", "").lower()
        
        if model_type in self.SUPPORTED_ARCHS:
            return self.SUPPORTED_ARCHS[model_type]
        
        # Try common patterns
        if hasattr(self.base, "model") and hasattr(self.base.model, "layers"):
            return self.SUPPORTED_ARCHS["llama"]  # Default Llama-like structure
        
        raise ValueError(
            f"Unsupported architecture: {model_type}. "
            f"Supported: {list(self.SUPPORTED_ARCHS.keys())}"
        )
    
    def _get_attr(self, attr_path: str):
        """Get nested attribute from base model."""
        obj = self.base
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        return obj
    
    @torch.no_grad()
    def freeze_base(self):
        """Freeze all base model parameters."""
        self.base.requires_grad_(False)
    
    @torch.no_grad()  
    def freeze_mixing(self):
        """Freeze mixing matrix (for pure RL control experiments)."""
        self.mixing.requires_grad_(False)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        g: Union[float, torch.Tensor, list[float]] = 0.0,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> MultiStreamOutput:
        """
        Forward pass with multi-stream routing.
        
        Args:
            input_ids: (batch, seq_len) input token IDs
            attention_mask: (batch, seq_len) attention mask
            position_ids: (batch, seq_len) position IDs (optional)
            inputs_embeds: (batch, seq_len, hidden) pre-computed embeddings
            g: Mixing gate - scalar, tensor, or list of per-layer floats
            past_key_values: KV cache for generation (experimental)
            use_cache: Whether to return updated KV cache
            **kwargs: Additional args passed to transformer blocks
            
        Returns:
            MultiStreamOutput with logits and optional diagnostics
        """
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self._embed(input_ids)
        
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        
        # Initialize all streams with the same embedding
        # Shape: (n_streams, batch, seq, hidden)
        streams = inputs_embeds.unsqueeze(0).expand(self.n_streams, -1, -1, -1).clone()
        
        # Get mixing matrix once (it's the same for all layers in MVP)
        # Ensure mixing matrix is on the same device as the streams
        M = self.mixing().to(device=device, dtype=dtype)
        
        # Normalize gate format
        if isinstance(g, list):
            assert len(g) == self.num_layers, f"g list must have {self.num_layers} elements"
            g_per_layer = g
        else:
            g_per_layer = None
        
        # Setup for KV cache if needed
        new_past_key_values = [] if use_cache else None
        
        # Compute position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask in the format expected by the model
        # HF models expect either None or a properly formatted causal mask
        # We'll let the model handle causal masking internally by passing None
        # when we have a simple "all ones" mask
        prepared_attention_mask = None
        if attention_mask is not None:
            # Check if it's a simple "all ones" mask - if so, pass None
            if not (attention_mask == 1).all():
                # Convert to float and create proper 4D mask
                # Shape: (batch, 1, seq, seq) for causal attention
                prepared_attention_mask = attention_mask[:, None, None, :].to(dtype)
                # Create causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device, dtype=dtype) * float("-inf"),
                    diagonal=1
                )
                # Combine with padding mask
                prepared_attention_mask = causal_mask.unsqueeze(0).unsqueeze(0) + (1 - prepared_attention_mask) * float("-inf")
        
        # Compute rotary embeddings (required by newer HF models like Qwen2)
        position_embeddings = None
        if self._rotary_emb is not None:
            position_embeddings = self._rotary_emb(streams[0], position_ids)
        
        # Process through each layer
        for layer_idx, layer in enumerate(self._layers):
            # Get gate for this layer
            g_l = g_per_layer[layer_idx] if g_per_layer is not None else g
            if not isinstance(g_l, torch.Tensor):
                g_l = torch.tensor(g_l, device=device, dtype=dtype)
            
            # === Step 1: Run transformer block on main stream only ===
            x = streams[0]  # (batch, seq, hidden)
            
            # Handle KV cache
            past_kv = past_key_values[layer_idx] if past_key_values else None
            
            # Forward through layer
            layer_kwargs = {
                "attention_mask": prepared_attention_mask,
                "position_ids": position_ids,
                "past_key_value": past_kv,
                "use_cache": use_cache,
                **kwargs,
            }
            # Add position embeddings if available (required by Qwen2, Llama3, etc.)
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            
            layer_outputs = layer(x, **layer_kwargs)
            
            # HF layers return (hidden_states, ..., optional_kv_cache)
            y = layer_outputs[0]
            
            if use_cache:
                new_past_key_values.append(layer_outputs[-1])
            
            # === Step 2: Write back to main stream ===
            streams = streams.clone()  # For autograd if training mixing params
            streams[0] = y
            
            # === Step 3: Apply gated mixing across all streams ===
            streams = apply_mixing(streams, M, g_l)
        
        # Final norm and projection (on main stream only)
        hidden = self._norm(streams[0])
        logits = self._head(hidden)
        
        # Collect diagnostics if requested
        stream_diag = None
        mixing_diag = None
        if self.collect_diagnostics:
            stream_diag = compute_stream_diagnostics(streams)
            mixing_diag = self.mixing.get_diagnostics()
        
        output = MultiStreamOutput(
            logits=logits,
            stream_diagnostics=stream_diag,
            mixing_diagnostics=mixing_diag,
        )
        
        # Handle cache for generation
        if use_cache:
            # Attach cache to output for compatibility
            output.past_key_values = tuple(new_past_key_values)
        
        return output


def greedy_decode(
    model: MultiStreamDecoder,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    g: Union[float, list[float]] = 0.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simple greedy decoding with controllable mixing gate.
    
    Use this for MVP instead of HF generate() to easily pass g.
    
    Args:
        model: MultiStreamDecoder instance
        input_ids: (batch, seq) prompt token IDs
        max_new_tokens: Maximum tokens to generate
        g: Mixing gate (fixed for entire generation)
        eos_token_id: Stop generation on this token
        pad_token_id: Padding token ID
        attention_mask: Initial attention mask
        
    Returns:
        (batch, seq + new_tokens) generated token IDs
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Initialize
    generated = input_ids.clone()
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    # Track which sequences are done
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                g=g,
            )
            
            # Get next token (greedy)
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1)
            
            # Handle padding for done sequences
            if pad_token_id is not None:
                next_tokens = torch.where(done, pad_token_id, next_tokens)
            
            # Append to generated
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask, 
                (~done).long().unsqueeze(-1)
            ], dim=-1)
            
            # Check for EOS
            if eos_token_id is not None:
                done = done | (next_tokens == eos_token_id)
                if done.all():
                    break
    
    return generated


def sample_decode(
    model: MultiStreamDecoder,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    g: Union[float, list[float]] = 0.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sampling-based decoding with temperature and top-p.
    
    For evaluation, use greedy_decode. This is for exploration/diversity.
    
    Args:
        model: MultiStreamDecoder instance
        input_ids: (batch, seq) prompt token IDs
        max_new_tokens: Maximum tokens to generate
        g: Mixing gate
        temperature: Sampling temperature (1.0 = neutral)
        top_p: Nucleus sampling threshold (1.0 = no filtering)
        eos_token_id: Stop on this token
        pad_token_id: Padding token
        attention_mask: Initial attention mask
        
    Returns:
        (batch, seq + new_tokens) generated token IDs
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    generated = input_ids.clone()
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                g=g,
            )
            
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            if pad_token_id is not None:
                next_tokens = torch.where(done, pad_token_id, next_tokens)
            
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                (~done).long().unsqueeze(-1)
            ], dim=-1)
            
            if eos_token_id is not None:
                done = done | (next_tokens == eos_token_id)
                if done.all():
                    break
    
    return generated

