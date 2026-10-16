"""
judge_calibration.py

The assignment explicitly asks for "evidence of how well your judge agrees
with a human." This script does that: it takes a subset of the golden set
(recommend 30-40 examples) that YOU score by hand using the exact same
rubric as judge.py, then compares your scores to the LLM judge's scores.

WHY THIS MATTERS: an LLM-as-judge number is worthless on its own -- it
could just be agreeing with itself. This produces the correlation numbers
that go in the report's evaluation section, and if agreement is weak on a
dimension, that's a real finding for the failure-analysis section, not
something to hide.

Usage:
  1. Run run_eval.py first to generate judge scores for the golden set.
  2. Hand-score 30-40 of the SAME rows yourself, save as
     eval/human_judge_subset.csv with columns:
     thread_id, human_relevance, human_grounding, human_tone,
     human_actionability, human_hallucination
  3. Run this script.
"""
import argparse
import pandas as pd
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_scores", default="eval/judge_scores.csv")
    ap.add_argument("--human_subset", default="eval/human_judge_subset.csv")
    args = ap.parse_args()

    judge_df = pd.read_csv(args.judge_scores)
    human_df = pd.read_csv(args.human_subset)
    merged = human_df.merge(judge_df, on="thread_id", suffixes=("_human", "_judge"))

    print(f"Comparing on {len(merged)} double-labeled examples\n")

    dims = ["relevance", "grounding", "tone", "actionability"]
    for dim in dims:
        human_col, judge_col = f"human_{dim}", dim
        if human_col not in merged or judge_col not in merged:
            continue
        rho, p = spearmanr(merged[human_col], merged[judge_col])
        mean_abs_diff = (merged[human_col] - merged[judge_col]).abs().mean()
        print(f"{dim:15s}  Spearman rho={rho:.2f} (p={p:.3f})  mean |diff|={mean_abs_diff:.2f}")

    if "human_hallucination" in merged and "hallucination" in merged:
        agree = (merged["human_hallucination"] == merged["hallucination"]).mean()
        print(f"\nhallucination flag agreement: {agree:.1%}")

    print("\nInterpretation guide (put your actual numbers + take in the report):")
    print("  rho > 0.6            : judge is a reasonable proxy for this dimension")
    print("  0.3 < rho <= 0.6     : usable for tracking trends, not for absolute claims")
    print("  rho <= 0.3           : don't trust this dimension's judge score; report it as unreliable")


if __name__ == "__main__":
    main()
