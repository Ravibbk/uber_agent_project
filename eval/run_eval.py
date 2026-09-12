"""
run_eval.py

The master evaluation script. Runs the full agent AND both baselines
against the hand-labeled golden set, and prints/saves a comparison table
covering:
  - intent classification accuracy + macro-F1 (agent vs simple vs trivial)
  - escalation precision/recall against gold_escalate labels
  - reply quality via LLM-judge (mean score per dimension, for the agent only
    -- baselines' template replies aren't meaningfully judged the same way)

Also writes eval/judge_scores.csv, which judge_calibration.py consumes.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from pipeline import UberSupportAgent
from baselines import trivial_predict, simple_predict
from judge import judge_reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.csv",
                     help="Hand-labeled golden set (after filling gold_intent/gold_escalate)")
    ap.add_argument("--threads", default="data/processed/uber_threads.csv",
                     help="Full processed threads, used as the historical grounding pool")
    ap.add_argument("--judge_out", default="eval/judge_scores.csv")
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    golden = golden[golden["gold_intent"].notna() & (golden["gold_intent"] != "")]
    print(f"Evaluating on {len(golden)} hand-labeled golden examples\n")

    historical = pd.read_csv(args.threads)
    agent = UberSupportAgent(historical_threads=historical)

    agent_results, judge_rows = [], []
    for _, row in golden.iterrows():
        out = agent.handle(row["customer_message"], row.get("thread_context", ""), row["thread_id"])
        agent_results.append(out)

        j = judge_reply(
            row["customer_message"], out["intent"], out["drafted_reply"],
            grounding_examples=[],  # kept simple here; pipeline logs example count only
        )
        j["thread_id"] = row["thread_id"]
        judge_rows.append(j)

    agent_df = pd.DataFrame(agent_results)
    judge_df = pd.DataFrame(judge_rows)
    judge_df.to_csv(args.judge_out, index=False)

    trivial_out = trivial_predict(golden.rename(columns={"gold_intent": "gold_intent"}), golden["customer_message"])
    simple_out = simple_predict(golden["customer_message"].tolist())

    gold_intents = golden["gold_intent"].tolist()
    gold_escalate = golden["gold_escalate"].tolist()

    def report_intent(name, preds):
        acc = accuracy_score(gold_intents, preds)
        f1 = f1_score(gold_intents, preds, average="macro", zero_division=0)
        print(f"  {name:10s}  accuracy={acc:.1%}   macro-F1={f1:.3f}")
        return acc, f1

    print("Intent classification:")
    report_intent("trivial", [r["intent"] for r in trivial_out])
    report_intent("simple", [r["intent"] for r in simple_out])
    report_intent("agent", agent_df["intent"].tolist())

    print("\nEscalation decision (positive class = ESCALATE):")
    for name, preds in [
        ("trivial", [r["escalation_decision"] for r in trivial_out]),
        ("simple", [r["escalation_decision"] for r in simple_out]),
        ("agent", agent_df["escalation_decision"].tolist()),
    ]:
        prec = precision_score(gold_escalate, preds, pos_label="ESCALATE", zero_division=0)
        rec = recall_score(gold_escalate, preds, pos_label="ESCALATE", zero_division=0)
        print(f"  {name:10s}  precision={prec:.1%}   recall={rec:.1%}")

    print("\nReply quality (agent only, LLM-judge, mean scores 1-5):")
    for dim in ["relevance", "grounding", "tone", "actionability"]:
        if dim in judge_df:
            print(f"  {dim:15s} {judge_df[dim].mean():.2f}")
    if "hallucination" in judge_df:
        print(f"  hallucination rate: {judge_df['hallucination'].mean():.1%}")

    print(f"\nJudge scores written to {args.judge_out}")
    print("Run judge_calibration.py next (after hand-scoring a subset) to check judge reliability.")


if __name__ == "__main__":
    main()
