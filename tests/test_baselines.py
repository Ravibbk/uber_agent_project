"""
test_baselines.py — Tests for baseline models.

Verifies that:
- Trivial baseline uses predeclared constant, not test labels (Defect C).
- Simple baseline produces valid outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import trivial_predict, simple_predict, TRIVIAL_INTENT, KEYWORD_RULES
from src.intents import INTENTS


class TestTrivialBaseline:
    def test_always_predicts_constant(self):
        """Defect C regression: trivial baseline must NOT peek at test labels."""
        results = trivial_predict(["msg1", "msg2", "msg3"])
        for r in results:
            assert r["intent"] == TRIVIAL_INTENT

    def test_always_auto_handles(self):
        results = trivial_predict(["anything"])
        assert results[0]["escalation_decision"] == "AUTO_HANDLE"

    def test_has_canned_reply(self):
        results = trivial_predict(["test"])
        assert results[0]["reply"]
        assert len(results[0]["reply"]) > 10

    def test_correct_count(self):
        results = trivial_predict(["a", "b", "c", "d"])
        assert len(results) == 4

    def test_does_not_accept_labels(self):
        """Defect C: trivial_predict signature must NOT accept train_df."""
        import inspect
        sig = inspect.signature(trivial_predict)
        params = list(sig.parameters.keys())
        assert "train_df" not in params, "trivial_predict must not accept train_df (Defect C)"


class TestSimpleBaseline:
    def test_safety_detection(self):
        results = simple_predict(["I felt unsafe during the ride"])
        assert results[0]["intent"] == "safety_concern"
        assert results[0]["escalation_decision"] == "ESCALATE"

    def test_fare_dispute(self):
        results = simple_predict(["I was overcharged for my ride"])
        assert results[0]["intent"] == "fare_dispute"

    def test_general_fallback(self):
        results = simple_predict(["hello there"])
        assert results[0]["intent"] == "general_inquiry"

    def test_valid_intents(self):
        messages = ["test1", "I was charged wrong", "app crashed", "lost my phone"]
        results = simple_predict(messages)
        for r in results:
            assert r["intent"] in INTENTS

    def test_has_reply_for_all(self):
        messages = ["test"]
        results = simple_predict(messages)
        assert results[0]["reply"]
