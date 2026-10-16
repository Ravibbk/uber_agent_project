"""
build_golden_set.py

Samples 150-250 examples from the processed Uber threads for YOU to hand-label.
This script does NOT label anything -- it picks a good, defensible sample and
writes a CSV with empty columns for you to fill in by reading each tweet.

SAMPLING STRATEGY (goes in the report's "how sampled/labelled" note):
- Stratify by a cheap proxy for topic (keyword bucket, using the same
  keyword rules as the simple baseline) so the golden set isn't dominated
  by whatever the single most common issue is.
- Within each bucket, sample randomly across the full date range, not just
  recent tweets -- avoids overfitting the eval to one time period's Twitter
  UI/policy quirks (e.g. old tweets mention "Uber" fare disputes differently
  post- vs pre- price transparency changes).
- Deliberately oversample short/ambiguous messages (<40 chars) a little,
  because that's where classifiers tend to fail -- the eval set should
  contain the hard cases, not just easy ones, or the accuracy number will
  be misleadingly high (see report/REPORT.md, "what's misleading" section).
"""
import argparse
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from baselines import KEYWORD_RULES  # reuse buckets as a stratification proxy


def bucket(text: str) -> str:
    text_lower = text.lower()
    for intent, kws in KEYWORD_RULES:
        if any(kw in text_lower for kw in kws):
            return intent
    return "general_inquiry"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/processed/uber_threads.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="eval/golden_set_TO_LABEL.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.threads)
    df["bucket"] = df["customer_message"].apply(bucket)
    df["is_short"] = df["customer_message"].str.len() < 40

    per_bucket = max(1, args.n // df["bucket"].nunique())
    samples = []
    for bucket_name, group in df.groupby("bucket"):
        short = group[group["is_short"]]
        long_ = group[~group["is_short"]]
        n_short = min(len(short), max(1, per_bucket // 4))  # oversample short/ambiguous
        n_long = min(len(long_), per_bucket - n_short)
        samples.append(short.sample(n_short, random_state=7) if n_short else short.iloc[:0])
        samples.append(long_.sample(n_long, random_state=7) if n_long else long_.iloc[:0])

    golden = pd.concat(samples).drop_duplicates(subset="thread_id")
    if len(golden) > args.n:
        golden = golden.sample(args.n, random_state=7)
    golden = golden.reset_index(drop=True)

    # Empty columns for hand-labeling
    golden["gold_intent"] = ""
    golden["gold_escalate"] = ""  # "AUTO_HANDLE" or "ESCALATE"
    golden["labeling_notes"] = ""

    cols = ["thread_id", "customer_message", "thread_context", "brand_reply",
            "gold_intent", "gold_escalate", "labeling_notes"]
    golden[cols].to_csv(args.out, index=False)
    print(f"Wrote {len(golden)} examples to {args.out} for hand-labeling.")
    print("Open it in a spreadsheet, fill gold_intent + gold_escalate for each row, save as golden_set.csv")


if __name__ == "__main__":
    main()
