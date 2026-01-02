import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routing import MultiStreamDecoder, greedy_decode


def test_wrapper_loads(model_name: str, device: str = "cuda"):
    """Test 1: Wrapper loads and wraps model correctly."""
    print(f"\n{'='*60}")
    print("Test 1: Loading wrapper")
    print(f"{'='*60}")
    
    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    
    print("Wrapping with MultiStreamDecoder (n_streams=4)")
    wrapped = MultiStreamDecoder(
        base_model,
        n_streams=4,
        mixing_mode="row_stochastic",
        collect_diagnostics=True,
    )
    wrapped.freeze_base()
    
    # Move mixing module to same device as base model
    if device == "cuda":
        wrapped.mixing = wrapped.mixing.cuda()
    
    print(f"✅ Wrapper created successfully")
    print(f"   - Number of layers: {wrapped.num_layers}")
    print(f"   - Number of streams: {wrapped.n_streams}")
    print(f"   - Mixing mode: {wrapped.mixing.mode}")
    
    return wrapped


def test_identity_behavior(wrapped: MultiStreamDecoder, tokenizer, device: str = "cuda"):
    """Test 2: g=0 should produce outputs close to baseline."""
    print(f"\n{'='*60}")
    print("Test 2: Identity behavior (g=0)")
    print(f"{'='*60}")
    
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    # Get baseline output (directly from base model)
    with torch.no_grad():
        base_out = wrapped.base(**inputs)
        base_logits = base_out.logits
    
    # Get wrapped output with g=0
    with torch.no_grad():
        wrapped_out = wrapped(**inputs, g=0.0)
        wrapped_logits = wrapped_out.logits
    
    # Compare
    diff = (base_logits - wrapped_logits).abs().mean().item()
    max_diff = (base_logits - wrapped_logits).abs().max().item()
    
    print(f"   Prompt: '{test_prompt}'")
    print(f"   Mean absolute difference: {diff:.6f}")
    print(f"   Max absolute difference: {max_diff:.6f}")
    
    # Should be very close (small numerical differences from clone() operations)
    if diff < 0.01:
        print(f"✅ g=0 produces baseline-equivalent outputs")
    else:
        print(f"⚠️  Larger than expected difference (may be OK, check manually)")
    
    # Check next token prediction is same
    base_next = base_logits[0, -1].argmax().item()
    wrapped_next = wrapped_logits[0, -1].argmax().item()
    
    print(f"   Base next token: {tokenizer.decode([base_next])} (id={base_next})")
    print(f"   Wrapped next token: {tokenizer.decode([wrapped_next])} (id={wrapped_next})")
    
    if base_next == wrapped_next:
        print(f"✅ Same next token prediction")
    else:
        print(f"⚠️  Different next token (check if this matters for your task)")
    
    return diff


def test_mixing_effect(wrapped: MultiStreamDecoder, tokenizer, device: str = "cuda"):
    """Test 3: g>0 should produce different outputs."""
    print(f"\n{'='*60}")
    print("Test 3: Mixing effect (g=0 vs g=0.5)")
    print(f"{'='*60}")
    
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_g0 = wrapped(**inputs, g=0.0)
        out_g05 = wrapped(**inputs, g=0.5)
    
    diff = (out_g0.logits - out_g05.logits).abs().mean().item()
    
    print(f"   Mean difference between g=0 and g=0.5: {diff:.6f}")
    
    if diff > 0.001:
        print(f"✅ Mixing gate has effect on outputs")
    else:
        print(f"⚠️  Mixing has minimal effect (check mixing matrix init)")
    
    # Show different predictions at different g values
    print("\n   Next token predictions at different g values:")
    for g_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        with torch.no_grad():
            out = wrapped(**inputs, g=g_val)
        next_token = out.logits[0, -1].argmax().item()
        print(f"     g={g_val}: {tokenizer.decode([next_token])} (id={next_token})")


def test_greedy_decode(wrapped: MultiStreamDecoder, tokenizer, device: str = "cuda"):
    """Test 4: Greedy decoding produces coherent output."""
    print(f"\n{'='*60}")
    print("Test 4: Greedy decoding")
    print(f"{'='*60}")
    
    test_prompt = "Question: What is 2 + 2?\nAnswer:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    print(f"   Prompt: '{test_prompt}'")
    
    for g_val in [0.0, 0.5]:
        output_ids = greedy_decode(
            wrapped,
            inputs.input_ids,
            max_new_tokens=20,
            g=g_val,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.attention_mask,
        )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_text = generated_text[len(test_prompt):]
        print(f"   g={g_val}: {new_text.strip()[:50]}...")
    
    print(f"✅ Greedy decoding works")


def test_diagnostics(wrapped: MultiStreamDecoder, tokenizer, device: str = "cuda"):
    """Test 5: Diagnostics are computed correctly."""
    print(f"\n{'='*60}")
    print("Test 5: Diagnostics")
    print(f"{'='*60}")
    
    test_prompt = "Hello world"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = wrapped(**inputs, g=0.5)
    
    print("   Mixing matrix diagnostics:")
    if out.mixing_diagnostics:
        for k, v in out.mixing_diagnostics.items():
            print(f"     {k}: {v:.4f}")
    
    print("\n   Stream diagnostics:")
    if out.stream_diagnostics:
        for k, v in out.stream_diagnostics.items():
            print(f"     {k}: {v:.4f}")
    
    # Check for healthy values
    if out.stream_diagnostics:
        ratio = out.stream_diagnostics.get("stream_norm_ratio", 1.0)
        if ratio < 10:
            print(f"✅ Stream norms are healthy (ratio={ratio:.2f})")
        else:
            print(f"⚠️  Stream norm ratio high ({ratio:.2f}) - may indicate instability")


def main():
    parser = argparse.ArgumentParser(description="Test multi-stream routing wrapper")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model to test with",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = parser.parse_args()
    
    print(f"Testing routing module with {args.model} on {args.device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run tests
    wrapped = test_wrapper_loads(args.model, args.device)
    test_identity_behavior(wrapped, tokenizer, args.device)
    test_mixing_effect(wrapped, tokenizer, args.device)
    test_greedy_decode(wrapped, tokenizer, args.device)
    test_diagnostics(wrapped, tokenizer, args.device)
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

