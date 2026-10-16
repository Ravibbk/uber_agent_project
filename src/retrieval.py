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
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalResolutionIndex:
    def __init__(self, threads_df: pd.DataFrame):
        self.df = threads_df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError("Historical thread data must contain at least one row.")
        self.vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(self.df["customer_message"])

    def retrieve(self, query: str, k: int = 3, exclude_thread_id: str = None):
        """Return top-k most similar historical (customer_message, brand_reply)
        pairs to ground the reply generator with real past resolutions."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1]

        results = []
        for idx in top_idx:
            row = self.df.iloc[idx]
            if exclude_thread_id and row["thread_id"] == exclude_thread_id:
                continue
            if sims[idx] <= 0:
                break
            results.append(
                {
                    "similarity": float(sims[idx]),
                    "customer_message": row["customer_message"],
                    "brand_reply": row["brand_reply"],
                }
            )
            if len(results) >= k:
                break
        return results
