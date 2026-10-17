"""
judge_calibration.py — Human-LLM Judge Calibration for TraceGuard

Compares human evaluation scores with LLM-as-judge (or offline heuristic)
scores across the 4 core dimensions (relevance, grounding, tone, actionability)
and the binary hallucination flag.

WHY CALIBRATION IS ESSENTIAL:
An LLM judge can exhibit systematic biases (verbosity bias, self-enhancement,
lenience). Without measuring correlation against independent human ratings,
automated judge scores cannot be trusted as an objective benchmark.

RUBRIC (1-5 scale for dimensions):
- relevance:
    1 = completely off-topic
    3 = addresses some points but misses key issue
    5 = directly answers customer's specific issue
- grounding:
    1 = contradicts or invents policies/refund amounts out of nowhere
    3 = generally plausible but asserts specifics not in context
    5 = strictly adheres to retrieved examples and Uber policies
- tone:
    1 = rude, aggressive, or robotic dismissiveness
    3 = neutral/bland
    5 = empathetic, professional, brand-appropriate customer support
- actionability:
    1 = leaves customer stranded with no next step
    3 = vague next step ("we'll look into it")
    5 = clear, concrete instructions (check receipt, reply with details)
- hallucination (binary True/False):
    True = response fabricates specific refund amounts, ticket numbers, or false promises
    False = response makes no unsubstantiated factual claims

WORKFLOW:
1. Generate annotation template:
     python eval/judge_calibration.py --generate-template
2. An independent human reviewer fills in:
     eval/human_judge_subset.csv
3. Run calibration:
     python eval/judge_calibration.py
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)


def generate_annotation_template(
    judge_scores_path: str = "eval/judge_scores.csv",
    output_path: str = "eval/human_judge_subset.csv",
):
    """Generate a template CSV for human annotators from the golden dataset / judge scores."""
    from eval.golden_dataset import GOLDEN_CASES

    template_rows = []
    for c in GOLDEN_CASES:
        template_rows.append({
            "case_id": c["case_id"],
            "customer_message": c["customer_message"],
            "expected_intent": c["expected_intent"],
            "expected_escalation": c["expected_escalation"],
            "human_relevance": "",
            "human_grounding": "",
            "human_tone": "",
            "human_actionability": "",
            "human_hallucination": "",
            "annotator_id": "",
            "annotation_notes": "",
        })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(template_rows)
    df.to_csv(out_file, index=False)
    print(f"Generated human annotation template with {len(df)} cases at: {out_file}")
    print("\nNext steps:")
    print("  1. Open the file and fill in scores (1-5) for human_relevance, human_grounding,")
    print("     human_tone, and human_actionability.")
    print("  2. Fill in human_hallucination as True or False.")
    print("  3. Run: python eval/judge_calibration.py")


def run_calibration(
    judge_scores_path: str = "eval/judge_scores.csv",
    human_subset_path: str = "eval/human_judge_subset.csv",
) -> dict:
    """Run calibration comparison between human scores and judge scores."""
    judge_file = Path(judge_scores_path)
    human_file = Path(human_subset_path)

    if not judge_file.exists():
        print(f"Error: Judge scores file not found at '{judge_scores_path}'.")
        print("Run `python eval/run_eval.py --judge` first to generate judge scores.")
        return {"status": "missing_judge_scores"}

    if not human_file.exists():
        print(f"Notice: Human annotation file not found at '{human_subset_path}'.")
        print("Generating a template for you...")
        generate_annotation_template(judge_scores_path, human_subset_path)
        return {"status": "pending_human_annotations"}

    judge_df = pd.read_csv(judge_file)
    human_df = pd.read_csv(human_file)

    # Determine merge key
    merge_key = "case_id" if "case_id" in judge_df.columns and "case_id" in human_df.columns else "thread_id"
    if merge_key not in judge_df.columns or merge_key not in human_df.columns:
        print(f"Error: Could not find matching ID column ('case_id' or 'thread_id') to merge datasets.")
        return {"status": "schema_mismatch"}

    merged = human_df.merge(judge_df, on=merge_key, suffixes=("_human", "_judge"))

    # Check if human scores are present
    dims = ["relevance", "grounding", "tone", "actionability"]
    valid_scored = 0
    for dim in dims:
        h_col = f"human_{dim}"
        if h_col in merged:
            valid_scored += merged[h_col].dropna().count()

    if valid_scored == 0:
        print("=" * 60)
        print("TraceGuard Human-LLM Judge Calibration: PENDING")
        print("=" * 60)
        print(f"The template '{human_subset_path}' exists ({len(merged)} cases),")
        print("but no human numerical scores (1-5) have been entered yet.")
        print("\nTo complete calibration:")
        print("  1. Edit eval/human_judge_subset.csv with 1-5 ratings.")
        print("  2. Re-run: python eval/judge_calibration.py")
        return {"status": "pending_human_annotations", "total_cases": len(merged)}

    print(f"\n{'='*60}")
    print(f"TraceGuard Human–Judge Calibration Report")
    print(f"{'='*60}")
    print(f"Matched cases: {len(merged)}\n")

    report = {"status": "completed", "dimensions": {}}

    for dim in dims:
        human_col, judge_col = f"human_{dim}", dim
        if human_col not in merged or judge_col not in merged:
            continue

        # Filter out rows where human score is missing or NaN
        valid = merged[[human_col, judge_col]].dropna()
        valid[human_col] = pd.to_numeric(valid[human_col], errors="coerce")
        valid[judge_col] = pd.to_numeric(valid[judge_col], errors="coerce")
        valid = valid.dropna()

        if len(valid) < 3:
            print(f"{dim:15s}  Insufficient valid ratings (need >= 3, got {len(valid)})")
            continue

        rho, p_spearman = spearmanr(valid[human_col], valid[judge_col])
        r, p_pearson = pearsonr(valid[human_col], valid[judge_col])
        mae = (valid[human_col] - valid[judge_col]).abs().mean()

        report["dimensions"][dim] = {
            "n_samples": len(valid),
            "spearman_rho": round(float(rho), 3) if not np.isnan(rho) else 0.0,
            "spearman_p": round(float(p_spearman), 4) if not np.isnan(p_spearman) else 1.0,
            "pearson_r": round(float(r), 3) if not np.isnan(r) else 0.0,
            "mae": round(float(mae), 3),
        }

        rho_str = f"{rho:.2f}" if not np.isnan(rho) else "N/A"
        p_str = f"{p_spearman:.3f}" if not np.isnan(p_spearman) else "N/A"
        print(f"{dim:15s}  n={len(valid):2d}  Spearman rho={rho_str} (p={p_str})  MAE={mae:.2f}")

    # Hallucination flag agreement
    if "human_hallucination" in merged and "hallucination" in merged:
        valid_hal = merged[["human_hallucination", "hallucination"]].dropna()
        if len(valid_hal) > 0:
            h_bool = valid_hal["human_hallucination"].astype(str).str.lower().isin(["true", "1", "yes"])
            j_bool = valid_hal["hallucination"].astype(str).str.lower().isin(["true", "1", "yes"])
            exact_match = (h_bool == j_bool).mean()
            print(f"\nHallucination flag exact agreement: {exact_match:.1%} (n={len(valid_hal)})")
            report["hallucination_agreement"] = round(float(exact_match), 4)

    print("\nCalibration Quality Reference:")
    print("  Spearman rho > 0.60       : Strong agreement; judge is a reliable proxy")
    print("  0.30 <= Spearman rho <= 0.60: Moderate; suitable for trend monitoring, not fine-grained gates")
    print("  Spearman rho < 0.30       : Weak/Poor; judge should not be relied upon for this dimension")

    return report


def main():
    parser = argparse.ArgumentParser(description="TraceGuard Human-LLM Judge Calibration")
    parser.add_argument("--judge_scores", default="eval/judge_scores.csv", help="Path to judge scores CSV")
    parser.add_argument("--human_subset", default="eval/human_judge_subset.csv", help="Path to human subset CSV")
    parser.add_argument("--generate-template", action="store_true", help="Generate template CSV for human annotations")
    args = parser.parse_args()

    if args.generate_template:
        generate_annotation_template(args.judge_scores, args.human_subset)
    else:
        run_calibration(args.judge_scores, args.human_subset)


if __name__ == "__main__":
    main()
