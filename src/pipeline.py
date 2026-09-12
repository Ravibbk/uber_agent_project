"""
pipeline.py

End-to-end: customer message -> intent -> retrieved grounding examples ->
drafted reply -> escalation decision. This is the single entry point the
eval harness and the README quickstart both call.
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_prep import build_uber_threads, load_raw
from intents import classify
from retrieval import HistoricalResolutionIndex
from reply_gen import draft_reply
from escalation import decide


def load_threads(path: str) -> pd.DataFrame:
    """Accept either a processed thread CSV or a raw TWCS-like CSV."""
    df = pd.read_csv(path)
    required = {"customer_message", "brand_reply"}

    if required.issubset(df.columns):
        return df

    raw_path = Path(path)
    if raw_path.exists():
        try:
            raw_df = load_raw(str(raw_path))
            processed_df = build_uber_threads(raw_df)
            if not processed_df.empty:
                return processed_df
        except Exception:
            pass

    raise ValueError(
        f"Input file '{path}' is not in the expected format. "
        "Expected either processed columns ['customer_message', 'brand_reply'] "
        "or a raw TWCS-style CSV."
    )


class UberSupportAgent:
    def __init__(self, historical_threads: pd.DataFrame):
        self.index = HistoricalResolutionIndex(historical_threads)

    def handle(self, customer_message: str, thread_context: str = "", thread_id: str = None) -> dict:
        intent_result = classify(customer_message, thread_context)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        examples = self.index.retrieve(customer_message, k=3, exclude_thread_id=thread_id)
        reply = draft_reply(customer_message, intent, examples, thread_context)
        escalation = decide(customer_message, intent, confidence)

        return {
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": confidence,
            "intent_rationale": intent_result.get("rationale", ""),
            "drafted_reply": reply,
            "escalation_decision": escalation["decision"],
            "escalation_reason": escalation["reason"],
            "num_grounding_examples": len(examples),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/processed/uber_threads.csv")
    ap.add_argument("--limit", type=int, default=20, help="How many messages to run through the pipeline")
    ap.add_argument("--out", default="data/processed/pipeline_sample_output.csv")
    args = ap.parse_args()

    df = load_threads(args.threads)
    agent = UberSupportAgent(historical_threads=df)

    sample = df.sample(min(args.limit, len(df)), random_state=1)
    results = []
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        out = agent.handle(row["customer_message"], row.get("thread_context", ""), row["thread_id"])
        results.append(out)

    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"Wrote {len(results)} pipeline outputs to {args.out}")


if __name__ == "__main__":
    main()
