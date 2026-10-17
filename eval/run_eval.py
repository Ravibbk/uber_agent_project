"""
run_eval.py — TraceGuard Evaluation Harness

Runs the agent, trivial baseline, and keyword baseline against the versioned
golden dataset. Computes intent accuracy, escalation precision/recall,
and optional LLM-judge reply quality scores.

CHANGES (TraceGuard upgrade):
- Uses versioned golden dataset (Defect B fix).
- Trivial baseline uses predeclared constant (Defect C fix).
- Excludes eval set from retrieval corpus (Defect D fix).
- Imports judge lazily — no crash without API key (Defect A fix).
- Produces machine-readable JSON results.
- Records run metadata for regression comparison.
"""
import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import UberSupportAgent, load_threads
from src.baselines import trivial_predict, simple_predict


def run_evaluation(golden_cases, agent, eval_thread_ids=None):
    """Run the agent on all golden cases, with retrieval leakage control."""
    agent_results = []
    for case in golden_cases:
        msg = case["customer_message"]
        out = agent.handle(
            customer_message=msg,
            thread_context="",
            thread_id=case.get("case_id"),
            exclude_ids=eval_thread_ids,
        )
        agent_results.append(out)
    return agent_results


def compute_metrics(gold_labels, pred_labels, label_name="intent"):
    """Compute accuracy and macro-F1 for classification."""
    acc = accuracy_score(gold_labels, pred_labels)
    f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
    return {f"{label_name}_accuracy": round(acc, 4), f"{label_name}_macro_f1": round(f1, 4)}


def compute_escalation_metrics(gold_escalation, pred_escalation):
    """Compute precision/recall for escalation."""
    prec = precision_score(gold_escalation, pred_escalation, pos_label="ESCALATE", zero_division=0)
    rec = recall_score(gold_escalation, pred_escalation, pos_label="ESCALATE", zero_division=0)
    return {
        "escalation_precision": round(prec, 4),
        "escalation_recall": round(rec, 4),
    }


def main():
    ap = argparse.ArgumentParser(description="TraceGuard Evaluation Harness")
    ap.add_argument("--threads", default="data/sample/twcs_sample.csv",
                     help="Historical threads for the retrieval index")
    ap.add_argument("--output", default="eval/eval_results.json",
                     help="Output path for JSON results")
    ap.add_argument("--judge", action="store_true",
                     help="Run LLM judge scoring (requires API key or uses offline heuristic)")
    args = ap.parse_args()

    # Load golden dataset from the versioned module
    from eval.golden_dataset import GOLDEN_CASES, DATASET_VERSION, compute_dataset_hash, validate_dataset

    # Validate
    validation = validate_dataset(GOLDEN_CASES)
    if not validation["is_valid"]:
        print(f"ERROR: Dataset validation failed: {json.dumps(validation['errors'], indent=2)}")
        sys.exit(1)
    print(f"Dataset validation: PASSED ({validation['total_cases']} cases)")

    dataset_hash = compute_dataset_hash(GOLDEN_CASES)

    # Load historical threads for retrieval
    threads_df = load_threads(args.threads)
    print(f"Loaded {len(threads_df)} historical threads for retrieval index")

    # Collect eval thread IDs for leakage control (Defect D fix)
    eval_thread_ids = {case["case_id"] for case in GOLDEN_CASES}

    agent = UberSupportAgent(historical_threads=threads_df)

    # Run agent
    print("\nRunning agent on golden dataset...")
    agent_results = run_evaluation(GOLDEN_CASES, agent, eval_thread_ids)

    # Run baselines (Defect C fix: trivial_predict takes only messages, not labels)
    messages = [c["customer_message"] for c in GOLDEN_CASES]
    trivial_out = trivial_predict(messages)
    simple_out = simple_predict(messages)

    # Extract predictions
    gold_intents = [c["expected_intent"] for c in GOLDEN_CASES]
    gold_escalation = [c["expected_escalation"] for c in GOLDEN_CASES]
    agent_intents = [r["intent"] for r in agent_results]
    agent_escalation = [r["escalation_decision"] for r in agent_results]

    # Compute metrics
    run_id = str(uuid.uuid4())[:12]
    results = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATASET_VERSION,
        "dataset_hash": dataset_hash,
        "dataset_size": len(GOLDEN_CASES),
        "agent_mode": "offline" if not agent_results[0].get("trace", {}).get("stages") else "offline",
        "metrics": {
            "agent": {
                **compute_metrics(gold_intents, agent_intents),
                **compute_escalation_metrics(gold_escalation, agent_escalation),
            },
            "trivial_baseline": {
                **compute_metrics(gold_intents, [r["intent"] for r in trivial_out]),
                **compute_escalation_metrics(gold_escalation, [r["escalation_decision"] for r in trivial_out]),
            },
            "simple_baseline": {
                **compute_metrics(gold_intents, [r["intent"] for r in simple_out]),
                **compute_escalation_metrics(gold_escalation, [r["escalation_decision"] for r in simple_out]),
            },
        },
        "per_case": [],
    }

    # Per-case details
    for i, case in enumerate(GOLDEN_CASES):
        case_result = {
            "case_id": case["case_id"],
            "customer_message": case["customer_message"][:100],
            "expected_intent": case["expected_intent"],
            "predicted_intent": agent_results[i]["intent"],
            "intent_correct": case["expected_intent"] == agent_results[i]["intent"],
            "expected_escalation": case["expected_escalation"],
            "predicted_escalation": agent_results[i]["escalation_decision"],
            "escalation_correct": case["expected_escalation"] == agent_results[i]["escalation_decision"],
            "risk_category": case["risk_category"],
        }
        results["per_case"].append(case_result)

    # Optional judge scoring
    if args.judge:
        print("\nRunning judge scoring...")
        from eval.judge import judge_reply
        judge_rows = []
        for i, case in enumerate(GOLDEN_CASES):
            cid = case["case_id"]
            msg = case["customer_message"]
            if not msg:
                judge_rows.append({
                    "case_id": cid,
                    "customer_message": msg,
                    "reply": agent_results[i].get("drafted_reply", ""),
                    "relevance": 0, "grounding": 0, "tone": 0, "actionability": 0,
                    "hallucination": False, "judge_mode": "skipped", "notes": "Empty input",
                })
                continue
            j = judge_reply(
                msg,
                agent_results[i]["intent"],
                agent_results[i]["drafted_reply"],
                grounding_examples=[],
            )
            j["case_id"] = cid
            j["customer_message"] = msg
            j["reply"] = agent_results[i].get("drafted_reply", "")
            judge_rows.append(j)

        judge_modes = set(j.get("judge_mode", "unknown") for j in judge_rows)
        scored = [j for j in judge_rows if j.get("judge_mode") != "skipped"]
        if scored:
            results["judge_scores"] = {
                "judge_mode": list(judge_modes),
                "mean_relevance": round(sum(j["relevance"] for j in scored) / len(scored), 2),
                "mean_grounding": round(sum(j["grounding"] for j in scored) / len(scored), 2),
                "mean_tone": round(sum(j["tone"] for j in scored) / len(scored), 2),
                "mean_actionability": round(sum(j["actionability"] for j in scored) / len(scored), 2),
                "hallucination_rate": round(
                    sum(1 for j in scored if j.get("hallucination")) / len(scored), 4
                ),
            }
            # Also write judge_scores.csv for calibration
            try:
                import pandas as pd
                judge_csv_path = Path("eval/judge_scores.csv")
                pd.DataFrame(judge_rows).to_csv(judge_csv_path, index=False)
                print(f"Saved judge scores to {judge_csv_path}")
            except Exception as e:
                logger.warning(f"Failed to save judge_scores.csv: {e}")

    # Print results
    print(f"\n{'='*60}")
    print(f"TraceGuard Evaluation Results — Run {run_id}")
    print(f"Dataset: {DATASET_VERSION} ({len(GOLDEN_CASES)} cases, hash={dataset_hash})")
    print(f"{'='*60}")

    print(f"\nIntent Classification:")
    for name in ["agent", "trivial_baseline", "simple_baseline"]:
        m = results["metrics"][name]
        print(f"  {name:20s}  accuracy={m['intent_accuracy']:.1%}   macro-F1={m['intent_macro_f1']:.3f}")

    print(f"\nEscalation Decision (positive class = ESCALATE):")
    for name in ["agent", "trivial_baseline", "simple_baseline"]:
        m = results["metrics"][name]
        print(f"  {name:20s}  precision={m['escalation_precision']:.1%}   recall={m['escalation_recall']:.1%}")

    # Show failures
    failures = [c for c in results["per_case"] if not c["intent_correct"] or not c["escalation_correct"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  {f['case_id']}: expected={f['expected_intent']}/{f['expected_escalation']} "
                  f"got={f['predicted_intent']}/{f['predicted_escalation']}")
    else:
        print(f"\nAll {len(GOLDEN_CASES)} cases passed intent and escalation checks.")

    if "judge_scores" in results:
        print(f"\nReply Quality (judge mode: {results['judge_scores']['judge_mode']}):")
        for dim in ["mean_relevance", "mean_grounding", "mean_tone", "mean_actionability"]:
            print(f"  {dim:25s} {results['judge_scores'][dim]:.2f}")
        print(f"  hallucination_rate       {results['judge_scores']['hallucination_rate']:.1%}")

    # Save JSON results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
