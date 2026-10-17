"""
test_escalation.py — Tests for escalation logic.

Covers: hard-escalate intents, implicit safety signals, low confidence,
legal/fraud language, angry refunds, and auto-handle cases.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.escalation import decide, HARD_ESCALATE_INTENTS, CONFIDENCE_THRESHOLD


class TestHardEscalation:
    """Safety-critical intents must ALWAYS escalate."""

    def test_safety_concern_escalates(self):
        result = decide("any message", "safety_concern", 0.99)
        assert result["decision"] == "ESCALATE"
        assert result["trigger_type"] == "hard_intent"

    def test_safety_concern_low_confidence_still_escalates(self):
        result = decide("any message", "safety_concern", 0.1)
        assert result["decision"] == "ESCALATE"


class TestImplicitSafety:
    """Implicit safety signals must escalate even with non-safety intent."""

    def test_swerving_escalates(self):
        """Defect E regression: swerving must escalate."""
        result = decide("My driver was swerving across lanes", "driver_behavior", 0.9)
        assert result["decision"] == "ESCALATE"
        assert result["trigger_type"] == "implicit_safety"

    def test_threatened_escalates(self):
        """Defect E regression: threatened must escalate."""
        result = decide("The driver threatened me", "driver_behavior", 0.9)
        assert result["decision"] == "ESCALATE"

    def test_reckless_escalates(self):
        result = decide("reckless driving the whole ride", "driver_behavior", 0.8)
        assert result["decision"] == "ESCALATE"

    def test_felt_unsafe_escalates(self):
        result = decide("I felt unsafe during the ride", "driver_behavior", 0.9)
        assert result["decision"] == "ESCALATE"

    def test_dangerous_escalates(self):
        result = decide("That was extremely dangerous", "general_inquiry", 0.7)
        assert result["decision"] == "ESCALATE"


class TestLowConfidence:
    def test_below_threshold_escalates(self):
        result = decide("something vague", "general_inquiry", 0.3)
        assert result["decision"] == "ESCALATE"
        assert result["trigger_type"] == "low_confidence"

    def test_at_threshold_auto_handles(self):
        result = decide("simple question", "general_inquiry", CONFIDENCE_THRESHOLD)
        assert result["decision"] == "AUTO_HANDLE"


class TestLegalLanguage:
    def test_lawyer_mention_escalates(self):
        result = decide("I'm going to call my lawyer", "fare_dispute", 0.9)
        assert result["decision"] == "ESCALATE"

    def test_fraud_mention_escalates(self):
        result = decide("This is fraud", "account_access", 0.8)
        assert result["decision"] == "ESCALATE"

    def test_police_mention_escalates(self):
        result = decide("I'm calling the police", "driver_behavior", 0.8)
        assert result["decision"] == "ESCALATE"


class TestAutoHandle:
    def test_normal_fare_dispute(self):
        result = decide("I was overcharged for my ride", "fare_dispute", 0.9)
        assert result["decision"] == "AUTO_HANDLE"

    def test_normal_lost_item(self):
        result = decide("I left my phone in the car", "lost_item", 0.8)
        assert result["decision"] == "AUTO_HANDLE"

    def test_general_inquiry(self):
        result = decide("Do you have student discounts?", "general_inquiry", 0.9)
        assert result["decision"] == "AUTO_HANDLE"
