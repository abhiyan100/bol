# bol-cleanup: Bol's own transcript-cleanup model

A LoRA fine-tune of [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)
(Apache-2.0) that turns raw voice dictation into a clean coding prompt:
fillers and stutters out, "auth dot py" to `auth.py`, "dash dash verbose" to
`--verbose`, meaning untouched. Small enough (~200MB at 4-bit) to sit next to
Parakeet and Kokoro and answer in well under half a second.

Why a fine-tune: off-the-shelf 1B-class models failed our safety bar in live
testing (dropped "don't touch" clauses, parroted examples). A 350M tuned on
exactly this task is faster AND safer. Same recipe FluidVoice uses for its
closed "Fluid Intelligence" model, except ours is open, data pipeline
included.

## Reproduce

```bash
# 1. Data: rule-based spoken corruption + REAL say->Parakeet roundtrip pairs
uv run python training/generate_data.py --out training/data --roundtrip

# 2. Train (M-series Mac, minutes)
uv run python -m mlx_lm lora \
  --model LiquidAI/LFM2.5-350M-MLX-bf16 --train --data training/data \
  --fine-tune-type lora --num-layers 16 --batch-size 8 --iters 600 \
  --adapter-path training/adapters --mask-prompt

# 3. Evaluate (exactness, negation/file/flag preservation, identity)
uv run python training/evaluate.py --adapter training/adapters

# 4. Fuse and quantize to 4-bit
uv run python -m mlx_lm fuse \
  --model LiquidAI/LFM2.5-350M-MLX-bf16 \
  --adapter-path training/adapters --save-path training/bol-cleanup-350m
uv run python -m mlx_lm convert \
  --hf-path training/bol-cleanup-350m -q --q-bits 4 \
  --mlx-path training/bol-cleanup-350m-4bit
```

## Use in Bol

```toml
[cleanup]
model = "abhiyan10/bol-cleanup-350m-4bit"   # the default; or a local path
```

With a cleanup model configured, local-mode "clean it up" uses it instead of
the deterministic rules alone. The deterministic pass still runs first and
remains the fallback on any failure, timeout, or suspicious rewrite.

## Data

- ~150 seed prompts covering files, flags, git, tests, negations, quoted
  strings, numbers, identifiers
- 6 rule-corrupted variants per seed (fillers, stutters, spoken symbols,
  homophones, dropped punctuation)
- ~150 roundtrip pairs: each seed spoken by macOS `say` and transcribed by
  Parakeet, capturing the STT engine's real error patterns
- identity pairs (clean input must come back byte-identical) so the model
  learns to leave clean text alone
