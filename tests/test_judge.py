"""
test_judge.py — Tests for the LLM judge module.

Verifies that:
- Importing judge.py does NOT crash without API key (Defect A).
- Offline judge returns valid scores.
- judge_reply() works in offline mode.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Force offline mode BEFORE import
os.environ["JUDGE_OFFLINE"] = "1"
os.environ.pop("OPENAI_API_KEY", None)


class TestJudgeImport:
    """Defect A regression: judge.py must import without API key."""

    def test_import_succeeds_without_key(self):
        """This test would FAIL on the original code."""
        # If this import raises RuntimeError, the test fails
        from eval.judge import judge_reply, offline_judge
        assert callable(judge_reply)
        assert callable(offline_judge)


class TestOfflineJudge:
    def test_returns_all_dimensions(self):
        from eval.judge import offline_judge
        result = offline_judge("I was overcharged", "fare_dispute", "DM us your trip ID")
        assert "relevance" in result
        assert "grounding" in result
        assert "tone" in result
        assert "actionability" in result
        assert "hallucination" in result
        assert "judge_mode" in result

    def test_scores_are_valid_range(self):
        from eval.judge import offline_judge
        result = offline_judge("test", "general_inquiry", "We're here to help")
        for dim in ["relevance", "grounding", "tone", "actionability"]:
            assert 1 <= result[dim] <= 5

    def test_hallucination_is_bool(self):
        from eval.judge import offline_judge
        result = offline_judge("test", "general_inquiry", "We'll help you")
        assert isinstance(result["hallucination"], bool)

    def test_judge_mode_is_offline(self):
        from eval.judge import offline_judge
        result = offline_judge("test", "general_inquiry", "help")
        assert result["judge_mode"] == "offline_heuristic"

    def test_detects_dollar_hallucination(self):
        from eval.judge import offline_judge
        result = offline_judge("test", "general_inquiry", "We'll refund you $50")
        assert result["hallucination"] is True


class TestJudgeReplyOffline:
    def test_judge_reply_works_offline(self):
        from eval.judge import judge_reply
        result = judge_reply("I was overcharged", "fare_dispute", "DM us your trip ID")
        assert result["judge_mode"] == "offline_heuristic"
        assert "relevance" in result

    def test_empty_reply(self):
        from eval.judge import judge_reply
        result = judge_reply("test", "general_inquiry", "")
        assert result["grounding"] == 1  # empty reply gets low grounding
