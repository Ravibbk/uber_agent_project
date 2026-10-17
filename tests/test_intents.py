"""
test_intents.py — Tests for intent classification.

Covers: keyword fallback accuracy, safety-first ordering, edge cases.
All tests run offline without API calls.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["OFFLINE_MODE"] = "1"
os.environ.pop("OPENAI_API_KEY", None)

from src.intents import classify, fallback_intent_from_message, INTENTS


class TestFallbackIntentClassifier:
    """Tests for the rule-based fallback intent classifier."""

    def test_fare_dispute(self):
        assert fallback_intent_from_message("I was overcharged for my ride") == "fare_dispute"

    def test_refund_request(self):
        assert fallback_intent_from_message("I want a refund for the cancellation fee") == "refund_request"

    def test_safety_concern_explicit(self):
        assert fallback_intent_from_message("I felt unsafe during the ride") == "safety_concern"

    def test_safety_concern_swerving(self):
        """Defect E regression: 'swerving' must map to safety_concern, not driver_behavior."""
        assert fallback_intent_from_message("My driver was swerving across lanes") == "safety_concern"

    def test_safety_concern_threatened(self):
        """Defect E regression: 'threatened' must map to safety_concern."""
        assert fallback_intent_from_message("The driver threatened me") == "safety_concern"

    def test_safety_concern_reckless(self):
        assert fallback_intent_from_message("reckless driving the whole way") == "safety_concern"

    def test_safety_concern_dangerous(self):
        assert fallback_intent_from_message("That was a dangerous ride") == "safety_concern"

    def test_driver_behavior_not_safety(self):
        """Rude driver is driver_behavior, not safety_concern."""
        assert fallback_intent_from_message("driver was rude and unhelpful") == "driver_behavior"

    def test_lost_item(self):
        assert fallback_intent_from_message("I left my phone in the car") == "lost_item"

    def test_account_access(self):
        assert fallback_intent_from_message("I can't log into my account") == "account_access"

    def test_app_technical(self):
        assert fallback_intent_from_message("the app keeps crashing") == "app_technical_issue"

    def test_cancellation(self):
        assert fallback_intent_from_message("my ride was cancelled for no reason") == "cancellation_issue"

    def test_general_inquiry(self):
        assert fallback_intent_from_message("do you have student discounts?") == "general_inquiry"

    def test_empty_input(self):
        assert fallback_intent_from_message("") == "general_inquiry"

    def test_none_input(self):
        assert fallback_intent_from_message(None) == "general_inquiry"


class TestClassifyOffline:
    """Tests for the classify() function in offline mode."""

    def test_returns_dict(self):
        result = classify("I was overcharged")
        assert isinstance(result, dict)
        assert "intent" in result
        assert "confidence" in result
        assert "rationale" in result

    def test_intent_is_valid(self):
        result = classify("something happened")
        assert result["intent"] in INTENTS

    def test_offline_has_model_field(self):
        result = classify("test message")
        assert result.get("model") == "offline_rules"

    def test_safety_classified_correctly(self):
        result = classify("my driver was swerving dangerously")
        assert result["intent"] == "safety_concern"

    def test_empty_message(self):
        result = classify("")
        assert result["intent"] == "general_inquiry"
