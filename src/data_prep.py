"""
data_prep.py

Loads the raw Kaggle "Customer Support on Twitter" CSV (twcs.csv), filters
down to conversations involving @Uber_Support, and reconstructs each
conversation into (customer_message, brand_reply, thread_context) rows.

WHY THIS APPROACH (be ready to explain this):
- The raw file is a flat table of ~2.8M individual tweets, each with
  `response_tweet_id` and `in_response_to_tweet_id` columns that link tweets
  into threads. A single row is NOT a conversation -- we have to walk the
  links ourselves.
- We only keep threads where Uber_Support is the *responder* (inbound=False,
  author_id == 'Uber_Support'), and we require a customer tweet that this
  reply is answering (in_response_to_tweet_id is not null).
- We keep the *first customer message* in a thread and Uber_Support's
  *final substantive reply* in that thread as our (query, resolution) pair.
  This is a deliberate simplification -- see decision_log.md, decision #3 --
  because most threads are 2-4 turns and the final brand message is usually
  the resolution or hand-off point.
"""
import re
import argparse
import pandas as pd
from pathlib import Path

BRAND = "Uber_Support"

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip urls and @mentions, collapse whitespace. Keep case/punctuation --
    tone matters for intent + reply quality."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": "Int64",
            "author_id": str,
            "inbound": bool,
            "in_response_to_tweet_id": "Int64",
            "response_tweet_id": str,  # can be comma-separated list of ids
        },
        parse_dates=["created_at"],
    )
    return df


def build_uber_threads(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct (customer_message -> Uber_Support reply) pairs."""
    df = df.set_index("tweet_id", drop=False)

    # All tweets authored by Uber_Support that are replies to someone
    brand_replies = df[
        (df["author_id"] == BRAND) & (df["in_response_to_tweet_id"].notna())
    ]

    rows = []
    for _, reply in brand_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        if parent_id not in df.index:
            continue
        customer_msg = df.loc[parent_id]
        if not isinstance(customer_msg, pd.Series):
            # duplicate index edge case -- skip, keep pipeline simple
            continue
        if not customer_msg["inbound"]:
            continue  # parent wasn't actually a customer message

        # Walk further back to grab up to 2 prior turns of context, if any
        context_chain = [clean_text(customer_msg["text"])]
        cur = customer_msg
        for _ in range(2):
            prev_id = cur.get("in_response_to_tweet_id")
            if pd.isna(prev_id) or prev_id not in df.index:
                break
            cur = df.loc[prev_id]
            if isinstance(cur, pd.Series):
                context_chain.insert(0, clean_text(cur["text"]))

        rows.append(
            {
                "thread_id": f"{customer_msg['tweet_id']}_{reply['tweet_id']}",
                "customer_message": clean_text(customer_msg["text"]),
                "thread_context": " | ".join(context_chain),
                "brand_reply": clean_text(reply["text"]),
                "created_at": customer_msg["created_at"],
            }
        )

    out = pd.DataFrame(rows)
    # Drop obviously empty / too-short rows
    out = out[
        (out["customer_message"].str.len() > 5)
        & (out["brand_reply"].str.len() > 5)
    ].reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/twcs/twcs.csv")
    ap.add_argument("--out", default="data/processed/uber_threads.csv")
    ap.add_argument(
        "--sample", type=int, default=None,
        help="Optional: randomly subsample N threads (useful for fast iteration).",
    )
    args = ap.parse_args()

    print(f"Loading {args.raw} ...")
    df = load_raw(args.raw)
    print(f"Loaded {len(df):,} raw tweets")

    threads = build_uber_threads(df)
    print(f"Reconstructed {len(threads):,} Uber_Support (customer, reply) pairs")

    if args.sample and len(threads) > args.sample:
        threads = threads.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(threads):,} threads")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    threads.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
