"""
test_pipeline.py — End-to-end pipeline tests.

Covers: full pipeline flow, tracing, empty input, safety escalation,
offline mode behavior.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["OFFLINE_MODE"] = "1"
os.environ["JUDGE_OFFLINE"] = "1"
os.environ.pop("OPENAI_API_KEY", None)

import pandas as pd
import pytest
from src.pipeline import UberSupportAgent


@pytest.fixture
def agent():
    df = pd.DataFrame([
        {"thread_id": "t1", "customer_message": "I was charged double", "brand_reply": "DM us your trip ID.", "thread_context": ""},
        {"thread_id": "t2", "customer_message": "left my phone in the car", "brand_reply": "Use Lost Item in the app.", "thread_context": ""},
        {"thread_id": "t3", "customer_message": "driver was very rude", "brand_reply": "DM us your trip ID for feedback.", "thread_context": ""},
    ])
    return UberSupportAgent(historical_threads=df)


class TestPipelineBasic:
    def test_returns_required_fields(self, agent):
        result = agent.handle("I was overcharged")
        assert "intent" in result
        assert "intent_confidence" in result
        assert "drafted_reply" in result
        assert "escalation_decision" in result
        assert "escalation_reason" in result
        assert "trace" in result

    def test_intent_is_valid(self, agent):
        from src.intents import INTENTS
        result = agent.handle("test message")
        assert result["intent"] in INTENTS

    def test_escalation_is_valid(self, agent):
        result = agent.handle("test message")
        assert result["escalation_decision"] in {"AUTO_HANDLE", "ESCALATE"}

    def test_reply_is_nonempty(self, agent):
        result = agent.handle("I need help with my fare")
        assert result["drafted_reply"]
        assert len(result["drafted_reply"]) > 10


class TestPipelineTracing:
    def test_trace_present(self, agent):
        result = agent.handle("test message")
        trace = result["trace"]
        assert "run_id" in trace
        assert "total_spans" in trace
        assert trace["total_spans"] >= 4

    def test_trace_has_all_stages(self, agent):
        result = agent.handle("test message")
        stages = [s["stage"] for s in result["trace"]["stages"]]
        assert "input_validation" in stages
        assert "intent_classification" in stages
        assert "retrieval" in stages
        assert "reply_generation" in stages
        assert "escalation" in stages

    def test_trace_timing(self, agent):
        result = agent.handle("test message")
        for stage in result["trace"]["stages"]:
            assert stage["duration_ms"] is not None or stage["status"] == "started"


class TestPipelineEmptyInput:
    def test_empty_string(self, agent):
        result = agent.handle("")
        assert result["escalation_decision"] == "ESCALATE"
        assert result.get("error") == "empty_input"

    def test_whitespace_only(self, agent):
        result = agent.handle("   ")
        assert result["escalation_decision"] == "ESCALATE"


class TestPipelineSafety:
    def test_swerving_escalates(self, agent):
        """Defect E regression: swerving must escalate through the full pipeline."""
        result = agent.handle("My driver was swerving across lanes")
        assert result["escalation_decision"] == "ESCALATE"

    def test_threatened_escalates(self, agent):
        result = agent.handle("The driver threatened me")
        assert result["escalation_decision"] == "ESCALATE"

    def test_felt_unsafe_escalates(self, agent):
        result = agent.handle("I felt unsafe during the ride")
        assert result["escalation_decision"] == "ESCALATE"

    def test_fraud_escalates(self, agent):
        result = agent.handle("I need help with a possible fraud incident")
        assert result["escalation_decision"] == "ESCALATE"
