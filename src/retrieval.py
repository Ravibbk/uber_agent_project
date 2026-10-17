"""
retrieval.py

Builds a simple TF-IDF index over historical (customer_message -> brand_reply)
pairs, so we can retrieve "here's how Uber_Support handled similar issues
before" as grounding context for the reply generator.

WHY TF-IDF AND NOT EMBEDDINGS (be ready to explain this):
Twitter support messages are short and full of domain-specific vocabulary
("surge", "pool", "UberEATS", trip codes). TF-IDF over the raw text is fast,
free, fully reproducible without another API dependency, and works well for
short, keyword-heavy text -- and it's easy to explain and debug (you can
literally see the shared words). An embeddings-based retriever is a natural
"next week" upgrade -- see report/REPORT.md, "what I'd do next" section.
This is decision #6 in decision_log.md.

CHANGES (TraceGuard upgrade):
- Added exclude_ids parameter for eval leakage control (Defect D fix).
- Added thread_ids tracking for better exclusion.
"""
import logging
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class HistoricalResolutionIndex:
    def __init__(self, threads_df: pd.DataFrame):
        self.df = threads_df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError("Historical thread data must contain at least one row.")

        # Track thread IDs for leakage exclusion
        self.thread_ids = set()
        if "thread_id" in self.df.columns:
            self.thread_ids = set(self.df["thread_id"].dropna().astype(str).tolist())

        self.vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(self.df["customer_message"])

    def retrieve(
        self,
        query: str,
        k: int = 3,
        exclude_thread_id: str = None,
        exclude_ids: set = None,
    ) -> list:
        """Return top-k most similar historical (customer_message, brand_reply)
        pairs to ground the reply generator with real past resolutions.

        Args:
            query: The customer message to find similar threads for.
            k: Number of results to return.
            exclude_thread_id: Single thread ID to exclude (e.g., the thread being evaluated).
            exclude_ids: Set of thread IDs to exclude (Defect D fix: for eval leakage control).
        """
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1]

        # Build exclusion set
        excluded = set()
        if exclude_thread_id:
            excluded.add(str(exclude_thread_id))
        if exclude_ids:
            excluded.update(str(i) for i in exclude_ids)

        results = []
        for idx in top_idx:
            row = self.df.iloc[idx]

            # Check exclusion
            if excluded and "thread_id" in row.index:
                if str(row["thread_id"]) in excluded:
                    continue

            if sims[idx] <= 0:
                break

            results.append(
                {
                    "similarity": float(sims[idx]),
                    "customer_message": row["customer_message"],
                    "brand_reply": row["brand_reply"],
                    "thread_id": row.get("thread_id", "unknown"),
                }
            )
            if len(results) >= k:
                break
        return results
