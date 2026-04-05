#!/usr/bin/env python3
"""
One-shot demo: load Qwen3.5 (or compatible) GGUF via llama-cpp-python, run the same
judge prompt / inference / parsing as ``resume_qwen35_gguf_judge_metric.py``, and print
aggregate metrics (no W&B).

Run from this directory::

    cd CB-LLMs/generation
    python demo_qwen35_judge_sample.py --gguf_path /path/to/model.gguf

Or HF download spec::

    python demo_qwen35_judge_sample.py \\
      --gguf_path "unsloth/Qwen3.5-27B-GGUF::Qwen3.5-27B-Q8_0.gguf"
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Same module as the full benchmark (judge_one, load_llama_cpp_llm, parse_score_1_to_10).
import resume_qwen35_gguf_judge_metric as judge_mod


# ---------------------------------------------------------------------------
# Hard-coded demo data (steerability-style: concept name + generated snippet)
# ---------------------------------------------------------------------------
DEMO_CONCEPTS = [
    "Engaging plot.",
    "Flat or one-dimensional characters.",
]

DEMO_TEXTS_BY_CONCEPT: list[list[str]] = [
    [
        "The film kept me hooked from the opening scene; every twist felt earned and the pacing never dragged.",
        "I cared about what happened next until the final shot; the story had real momentum.",
    ],
    [
        "The protagonist was nuanced and changed believably over the arc; side characters had distinct voices.",
        "The cast delivered layered performances; I remembered each character's motivation.",
    ],
]


def aggregate_like_resume(
    scores_by_concept: list[list[float | None]],
    concept_names: list[str],
) -> dict:
    """Mirror resume_qwen35_gguf_judge_metric.run_judge_on_texts summary (no wandb)."""
    all_scores: list[float] = []
    per_concept: dict = {}
    parse_ok = 0
    parse_fail = 0

    for concept_name, row in zip(concept_names, scores_by_concept):
        scores_this: list[float] = []
        for s in row:
            if s is None:
                parse_fail += 1
            else:
                parse_ok += 1
                all_scores.append(float(s))
                scores_this.append(float(s))
        if scores_this:
            a = np.array(scores_this, dtype=np.float64)
            per_concept[concept_name] = {
                "n": len(scores_this),
                "judge_mean_1_10": float(a.mean()),
                "judge_std_1_10": float(a.std()) if a.size > 1 else 0.0,
            }
        else:
            per_concept[concept_name] = {"n": len(row), "judge_mean_1_10": float("nan"), "judge_std_1_10": 0.0}

    if not all_scores:
        g_mean, g_std = float("nan"), 0.0
    else:
        a = np.array(all_scores, dtype=np.float64)
        g_mean, g_std = float(a.mean()), float(a.std()) if a.size > 1 else 0.0

    return {
        "judge_mean_1_10": g_mean,
        "judge_std_1_10": g_std,
        "judge_total_n_scored": len(all_scores),
        "judge_parse_ok": parse_ok,
        "judge_parse_fail": parse_fail,
        "per_concept": per_concept,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Demo Qwen GGUF steerability judge (hard-coded samples).")
    p.add_argument(
        "--gguf_path",
        type=str,
        required=True,
        help="Local .gguf path or HF spec repo_id::filename (see resume_qwen35_gguf_judge_metric).",
    )
    p.add_argument(
        "--n_ctx",
        type=int,
        default=8192,
        help="Context for judge + long reasoning (default 8192).",
    )
    p.add_argument("--n_gpu_layers", type=int, default=-1)
    p.add_argument("--n_batch", type=int, default=256)
    p.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Must be large enough for thinking + final SCORE line (default 2048).",
    )
    p.add_argument("--max_chars", type=int, default=1200)
    p.add_argument("--judge_disable_thinking", action="store_true", help="Pass enable_thinking=False to template.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    enable_thinking = not args.judge_disable_thinking
    print("Loading model...")
    llm = judge_mod.load_llama_cpp_llm(
        args.gguf_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        verbose=args.verbose,
    )

    scores_by_concept: list[list[float | None]] = []

    for ci, concept_name in enumerate(DEMO_CONCEPTS):
        texts = DEMO_TEXTS_BY_CONCEPT[ci] if ci < len(DEMO_TEXTS_BY_CONCEPT) else []
        row_scores: list[float | None] = []
        print(f"\n{'='*60}\nConcept [{ci}] {concept_name!r} ({len(texts)} samples)\n{'='*60}")
        for b, t in enumerate(texts):
            print(f"\n--- sample {b} (preview) ---\n{t[:200]}{'...' if len(t) > 200 else ''}\n")
            score, raw = judge_mod.judge_one(
                llm,
                concept_name,
                t,
                args.max_chars,
                args.max_tokens,
                enable_thinking=enable_thinking,
            )
            cleaned = judge_mod._strip_think_blocks(raw or "")
            reparsed = judge_mod.parse_score_1_to_10(cleaned)
            print("--- raw assistant output (first 800 chars) ---")
            print((raw or "")[:800] + ("..." if len(raw or "") > 800 else ""))
            print("\n--- post-process ---")
            print(f"  stripped_think_len={len(cleaned)}  parse_score_1_to_10(cleaned)={reparsed}")
            print(f"  judge_one returned score={score} (should match reparsed)")
            row_scores.append(score)
        scores_by_concept.append(row_scores)

    metrics = aggregate_like_resume(scores_by_concept, DEMO_CONCEPTS)
    print(f"\n{'='*60}\nAggregate (same shape as benchmark log dict)\n{'='*60}")
    print(f"  judge_mean_1_10: {metrics['judge_mean_1_10']}")
    print(f"  judge_std_1_10:  {metrics['judge_std_1_10']}")
    print(f"  n_scored:          {metrics['judge_total_n_scored']}")
    print(f"  parse_ok / fail:   {metrics['judge_parse_ok']} / {metrics['judge_parse_fail']}")
    for cname, row in metrics["per_concept"].items():
        print(f"  per_concept[{cname!r}]: n={row['n']} mean={row['judge_mean_1_10']} std={row['judge_std_1_10']}")

    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
