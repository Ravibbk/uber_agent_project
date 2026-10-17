"""
test_retrieval.py — Tests for the TF-IDF retrieval module.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from src.retrieval import HistoricalResolutionIndex


@pytest.fixture
def index():
    df = pd.DataFrame([
        {"thread_id": "t1", "customer_message": "I was overcharged for my ride", "brand_reply": "DM us your trip ID."},
        {"thread_id": "t2", "customer_message": "left my phone in the car", "brand_reply": "Use Lost Item."},
        {"thread_id": "t3", "customer_message": "driver was rude and unhelpful", "brand_reply": "DM us your trip ID for feedback."},
        {"thread_id": "t4", "customer_message": "app keeps crashing on my phone", "brand_reply": "Try reinstalling."},
    ])
    return HistoricalResolutionIndex(df)


class TestRetrieval:
    def test_returns_list(self, index):
        results = index.retrieve("I was charged too much")
        assert isinstance(results, list)

    def test_results_have_required_fields(self, index):
        results = index.retrieve("overcharged ride")
        if results:
            assert "similarity" in results[0]
            assert "customer_message" in results[0]
            assert "brand_reply" in results[0]

    def test_similarity_ordering(self, index):
        results = index.retrieve("overcharged ride", k=3)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_exclude_thread_id(self, index):
        results = index.retrieve("I was overcharged", exclude_thread_id="t1")
        thread_ids = [r.get("thread_id") for r in results]
        assert "t1" not in thread_ids

    def test_exclude_ids_set(self, index):
        """Defect D: test that exclude_ids removes specified threads."""
        results = index.retrieve("overcharged", exclude_ids={"t1", "t2"})
        thread_ids = [r.get("thread_id") for r in results]
        assert "t1" not in thread_ids
        assert "t2" not in thread_ids

    def test_k_limit(self, index):
        results = index.retrieve("test query", k=2)
        assert len(results) <= 2

    def test_empty_query(self, index):
        results = index.retrieve("")
        assert isinstance(results, list)


class TestRetrievalEdgeCases:
    def test_single_row_index(self):
        df = pd.DataFrame([
            {"thread_id": "t1", "customer_message": "test message", "brand_reply": "test reply"},
        ])
        idx = HistoricalResolutionIndex(df)
        results = idx.retrieve("test")
        assert len(results) >= 0

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="at least one row"):
            HistoricalResolutionIndex(pd.DataFrame(columns=["customer_message", "brand_reply"]))
