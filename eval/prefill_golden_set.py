"""Create rule-based label suggestions for a human to review."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from escalation import decide
from intents import fallback_intent_from_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval/golden_set_TO_LABEL.csv")
    parser.add_argument("--output", default="eval/golden_set_PREFILLED.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    suggestions = []
    for message in df["customer_message"].fillna(""):
        intent = fallback_intent_from_message(message)
        escalation = decide(message, intent, 0.72)
        suggestions.append(
            {
                "gold_intent": intent,
                "gold_escalate": escalation["decision"],
                "labeling_notes": "Auto-suggested; review before using as evaluation gold.",
            }
        )

    suggested = pd.DataFrame(suggestions)
    df["gold_intent"] = suggested["gold_intent"]
    df["gold_escalate"] = suggested["gold_escalate"]
    df["labeling_notes"] = suggested["labeling_notes"]
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} suggested labels to {args.output}")
    print("Review the suggestions manually, then save the final file as eval/golden_set.csv.")


if __name__ == "__main__":
    main()