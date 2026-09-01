"""Evaluate bol-cleanup: exactness, meaning safety, identity behavior.

Usage:
  uv run python training/evaluate.py --adapter training/adapters
  uv run python training/evaluate.py --model <fused-or-hf-model>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlx_lm import generate, load

SYSTEM = "Clean this voice dictation for a coding agent. Output only the cleaned text."

NEGATIONS = re.compile(r"\b(don't|dont|not|never|no|without|except)\b", re.IGNORECASE)
TOKENS = re.compile(r"\b[\w-]+\.\w{1,5}\b|--\w[\w-]*")


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s.-]", "", text.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M-MLX-bf16")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="training/data/test.jsonl")
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()]

    exact = near = neg_ok = tok_ok = ident_ok = 0
    n_neg = n_tok = n_ident = 0
    misses = []

    for row in rows:
        noisy = row["messages"][1]["content"]
        clean = row["messages"][2]["content"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": noisy}],
            add_generation_prompt=True,
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=120).strip()

        if out == clean:
            exact += 1
        if normalize(out) == normalize(clean):
            near += 1
        else:
            misses.append((noisy, clean, out))

        ref_negs = NEGATIONS.findall(clean)
        if ref_negs:
            n_neg += 1
            if len(NEGATIONS.findall(out)) >= len(ref_negs):
                neg_ok += 1
        ref_toks = set(TOKENS.findall(clean))
        if ref_toks:
            n_tok += 1
            if ref_toks.issubset(set(TOKENS.findall(out))):
                tok_ok += 1
        if noisy == clean:
            n_ident += 1
            if normalize(out) == normalize(clean):
                ident_ok += 1

    n = len(rows)
    print(f"n={n}")
    print(f"exact match:        {exact}/{n} ({exact/n:.0%})")
    print(f"normalized match:   {near}/{n} ({near/n:.0%})")
    if n_neg:
        print(f"negations kept:     {neg_ok}/{n_neg} ({neg_ok/n_neg:.0%})")
    if n_tok:
        print(f"files/flags kept:   {tok_ok}/{n_tok} ({tok_ok/n_tok:.0%})")
    if n_ident:
        print(f"clean left alone:   {ident_ok}/{n_ident} ({ident_ok/n_ident:.0%})")
    print("\nworst misses:")
    for noisy, clean, out in misses[:5]:
        print(f"  in:   {noisy}\n  want: {clean}\n  got:  {out}\n")


if __name__ == "__main__":
    main()
