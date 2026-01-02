# RL-Controlled Routing for LLMs: Complete Implementation Guide

A detailed, build-from-zero outline for implementing RL-controlled internal routing using **Lambda GPUs**, **Qwen3-4B-Instruct**, **Gymnasium**, **CleanRL PPO**, **lm-eval-harness**, with optional **vLLM** and **Judgment Labs**.

---

## 0. Ground Rules and Research Constraints

### What You're Proving (and What You're Not)

**Claim:** An RL controller that chooses internal routing (mixing vs identity) improves reasoning accuracy **without changing decoding** and ideally without touching base weights.

You must keep:
- Decoding fixed (temperature/top-p/max tokens fixed)
- Dataset splits fixed
- Baseline parity strong

### Why This Stack

- **Accelerate** handles single→multi-GPU launch and mixed precision cleanly (`accelerate config` then `accelerate launch ...`)
- **CleanRL PPO** is a simple PPO reference that's easy to adapt for custom action spaces and logs useful metrics by default
- **lm-evaluation-harness** is the standard benchmark harness across tasks like GSM8K
- **SVAMP** is a standard OOD math word problem set; use it as a generalization check

---

## 1. Repo + Environment Setup (Lambda)

### 1.1 Create the Repo Skeleton

Keep strict separations (you already started this):

```
routing/        # multistream wrapper + mixing operators
controller/     # PPO policy/value net + feature encoding
envs/           # Gymnasium environment (GSM8KEnv)
train/          # PPO trainer entrypoint(s)
configs/        # accelerate configs, experiment configs
scripts/        # eval scripts, data prep, plotting
```

### 1.2 Install + Verify Dependencies

- Use your `pyproject.toml`
- Install in editable mode
- Verify imports **before** building routing

---

## 2. Define the Core Architecture You Will Control

### 2.1 Choose the "Routing Knob" (Start MVP-Simple)

You want a **small** action space so PPO works.

**MVP Action:** One scalar gate per problem:
- `g ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`
- Interprets as "how much mixing vs identity"

Later upgrade:
- Per-layer gates: `g_l` (vector action)

### 2.2 Define Multi-Stream Residual in a Way You Can Actually Implement

You do **not** need full mHC to start. You need "HC-like" multi-streaming with stable mixing.

**Minimal multistream representation:**
- Each layer maintains `n` residual streams (e.g., `n=4`)
- Layer computes its normal update once
- You apply:
  - Identity carry
  - A mixing step across streams
- Controller outputs `g` to interpolate between identity and mixing

**Keep Qwen3 base weights frozen** initially.

### 2.3 Decide Where to Hook It in the Transformer Block

Pick a consistent location:
- After attention residual add
- After MLP residual add (or just once per block at the end for MVP)

Goal: One stable intervention point per block.

---

## 3. Implement Routing Module (routing/)

### 3.1 Build It in 3 Milestones

**Milestone A: "No-op hook"**
- Insert a wrapper that runs and produces identical outputs to baseline when `g=0`

**Milestone B: "Single-stream gate"**
- Start with `n=1` but still apply a controlled residual scaling to ensure your hook works

**Milestone C: "Multi-stream"**
- Turn on `n=4`
- Add mixing operator

### 3.2 Mixing Operator Choice (MVP)

Start with something stable and easy:
- `mix = softmax(W)` row-normalized mixing matrix
- Apply `streams = (1-g)*streams + g*(streams @ mix)` (conceptually)

Later, you can add stronger stability constraints (mHC uses constrained mixing for stability; that's phase 2).

### 3.3 Instrumentation (You'll Need These Stats)

Return per-forward-pass metrics:
- Stream norms
- Stream diversity (variance across streams)
- "Dominance" (max stream contribution)

These become:
- Logging
- Optional RL observations
- Optional stability penalties

---

## 4. Build the RL Environment (envs/)

### 4.1 Gymnasium Environment Definition

Use Gymnasium's standard API:

**`reset()`:**
- Sample a GSM8K training example
- Produce prompt
- Return observation

**`step(action)`:**
- Action → `g` (or `g_l`)
- Run **one generation** with fixed decoding settings
- Parse final numeric answer
- Reward = exact match (plus light shaping if needed)
- Done = True (one-step episode)

Gymnasium is the maintained successor to Gym—use it as the standard env interface.

### 4.2 Observation Design (Start Tiny)

Start with:
- Prompt token length
- Maybe an easy difficulty proxy (length bucket)

Then add internal metrics only if PPO needs them.

### 4.3 Reward Function (Practical)

Start:
- `+1` correct numeric answer
- `0` otherwise

If learning stalls early, add small shaping:
- `+0.05` if parseable number
- `-0.01` length penalty (avoid verbosity)

Keep it minimal so your claim stays clean.

---

## 5. RL Training Loop (train/ with CleanRL PPO)

### 5.1 Why PPO Here

PPO's clipped updates are robust when you have:
- Sparse rewards
- Small discrete action space
- High-variance rollouts

CleanRL provides a single-file PPO reference with logging and research-friendly defaults.

### 5.2 What You Train

**Only train the controller policy/value nets:**
- Base Qwen3 weights frozen
- Routing module deterministic given `g`

### 5.3 PPO Details That Matter for Your Setting

- Batch collection: many 1-step episodes
- Entropy bonus: important so it explores gates
- Normalize rewards or advantages

### 5.4 Launch on Lambda with Accelerate

Use `accelerate config` once, then launch (multi-GPU if needed).

---

## 6. Evaluation Pipeline (scripts/ + lm-eval-harness)

### 6.1 Make Your Benchmark "Official"

Use lm-eval-harness as the canonical evaluator.

You'll run:
- GSM8K test (in-domain)
- SVAMP (OOD)

SVAMP is a standard dataset for testing math-word-problem robustness.

### 6.2 How to Integrate Your Model into lm-eval

Two options:
1. Wrap your model as a HuggingFace model callable (simpler)
2. Use lm-eval's "model interface" hook

Goal: Evaluate:
- Baseline Qwen3-4B
- Multi-stream fixed g
- RL-controlled g

Keep prompts + decoding identical.

---

## 7. Baselines & Ablations (Non-Negotiable)

Run these five:

1. **Baseline**: Qwen3-4B-Instruct, no routing
2. **Fixed mixing**: Multi-stream with fixed `g=0.5`
3. **Random mixing**: Sample `g` randomly per example
4. **Supervised controller**: Tiny MLP predicts `g` from prompt length
5. **RL controller**: PPO policy outputs `g`

Your novelty is beating (2) and (4), not just baseline.

---

## 8. What to Log (So You Can Interpret and Defend Results)

### Training Curves (W&B)

- Episode reward / accuracy
- Gate distribution over time (histogram of chosen g)
- Entropy, policy loss, value loss (CleanRL logs these)

### Behavioral + Structural Diagnostics

- `g` vs difficulty buckets (e.g., by answer length / prompt length)
- If per-layer: heatmap of `g_l`
- Stream dominance/diversity metrics

### Benchmark Metrics

- GSM8K accuracy
- SVAMP accuracy

---

## 9. Speed / Scaling Plan (Only When Needed)

### 9.1 If Rollouts Are Slow

Add **vLLM** for faster batched generation, but only after correctness is validated.

### 9.2 If You Scale Up Later

Consider DeepSpeed, but don't reach for it until you hit memory limits.

---

## 10. Optional: Judgment Labs Observability (Where It Helps)

Use it for:
- Tracing failures (prompt → g → answer → reward)
- Regression monitoring
- Debugging "why did it pick high mixing here?"

But keep lm-eval as the benchmark source of truth.

---

## 11. Execution Checklist (The Build Order That Actually Works)

1. **Run Qwen3-4B baseline** on 50 GSM8K examples (sanity)
2. Implement routing **no-op** (`g=0` identical outputs)
3. Turn on multi-stream with fixed `g=0.5` and confirm it runs
4. Build GSM8KEnv with fixed action and verify reward works
5. Run PPO for a tiny number of steps; verify it learns *anything*
6. Add baselines (random + supervised)
7. Full training run
8. Run lm-eval on GSM8K test + SVAMP
9. Plot `g` vs difficulty + ablations
10. Only then optimize speed

---

## Key Concepts to Research/Understand

- **Residual connections / identity mapping**: Why "do nothing" paths stabilize deep nets (the reason mixing needs constraints)
- **Why PPO is stable** for small discrete controls (clipped updates)
- **How lm-eval-harness defines tasks** and ensures reproducible evaluation
- **SVAMP's role** as OOD robustness, not just another split
- **Accelerate launch patterns** for single/multi GPU on Lambda

---

## References

- [Accelerate: Launching scripts](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch)
- [CleanRL PPO Documentation](https://docs.cleanrl.dev/rl-algorithms/ppo/)
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [SVAMP Dataset](https://github.com/arkilpatel/SVAMP)