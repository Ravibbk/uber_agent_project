"""
escalation.py

Decides whether a message should be AUTO_HANDLED or ESCALATED to a human,
with a stated reason.

DESIGN: hard rule-based triggers FIRST, then a confidence-based fallback.
WHY NOT "let the LLM decide everything" (be ready to explain this):
Escalation is a safety-critical decision -- a false negative here (auto-
handling something that should go to a human) is much worse than a false
positive. Hard-coding non-negotiable triggers (safety, low classifier
confidence) means these can't be silently reasoned away by the LLM on a
bad day. This is decision #8 in decision_log.md.

CHANGES (TraceGuard upgrade):
- Added implicit safety keyword scanning independent of intent (Defect E fix).
- Expanded anger/legal markers.
- Returns structured reason with trigger type for tracing.
"""
import logging

logger = logging.getLogger(__name__)

# Intents that ALWAYS escalate, no matter what the model thinks -- these carry
# legal/safety/brand risk too high to auto-resolve.
HARD_ESCALATE_INTENTS = {"safety_concern"}

# Below this classifier confidence, we don't trust the intent label enough
# to auto-handle -- better to route to a human than reply confidently to
# the wrong problem.
CONFIDENCE_THRESHOLD = 0.55

# Implicit safety signals -- even if intent was classified as something else,
# these words indicate potential safety concerns that should escalate.
# (Defect E fix: catches "My driver was swerving" even when classified as driver_behavior)
SAFETY_KEYWORDS = [
    "unsafe", "swerving", "swerve", "reckless", "erratic", "erratically",
    "threatened", "threaten", "weapon", "assault", "attacked", "attack",
    "harass", "harassment", "stalking", "followed me", "kidnap",
    "accident", "hit me", "hurt", "injured", "emergency",
    "dangerous", "danger", "violence", "violent",
    "gun", "knife", "scared", "feared for", "felt unsafe",
    "ran a red", "ran red light", "driving on the wrong side",
]

# Simple negative-sentiment / anger signal words -- crude on purpose (see
# decision log #9: we chose a cheap heuristic here rather than a second
# LLM call, and measured that it's good enough in eval/run_eval.py)
ANGER_MARKERS = [
    "unacceptable", "lawsuit", "lawyer", "scam", "worst", "never again",
    "disgusting", "furious", "fraud", "stolen", "police",
    "attorney", "sue", "legal action", "report you", "consumer protection",
]


def decide(customer_message: str, intent: str, confidence: float) -> dict:
    """Decide whether to escalate. Returns dict with decision, reason, and trigger_type."""
    text_lower = (customer_message or "").lower()

    # 1. Hard-escalate by intent
    if intent in HARD_ESCALATE_INTENTS:
        return {
            "decision": "ESCALATE",
            "reason": f"Intent '{intent}' is a hard-escalate category (safety-related).",
            "trigger_type": "hard_intent",
        }

    # 2. Implicit safety keyword scan — even if intent wasn't safety_concern
    # (Defect E fix: catches swerving, threatened, etc.)
    matched_safety = [kw for kw in SAFETY_KEYWORDS if kw in text_lower]
    if matched_safety:
        return {
            "decision": "ESCALATE",
            "reason": f"Implicit safety signal detected (matched: {matched_safety[:3]}), "
                      f"escalating regardless of classified intent '{intent}'.",
            "trigger_type": "implicit_safety",
        }

    # 3. Low confidence
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "decision": "ESCALATE",
            "reason": f"Intent classification confidence ({confidence:.2f}) below threshold "
                      f"({CONFIDENCE_THRESHOLD}); risk of replying to the wrong problem.",
            "trigger_type": "low_confidence",
        }

    # 4. Angry refund request
    if intent == "refund_request" and any(m in text_lower for m in ANGER_MARKERS):
        return {
            "decision": "ESCALATE",
            "reason": "Refund request combined with high-anger language; risk of "
                      "brand/legal escalation if handled wrong by a bot.",
            "trigger_type": "angry_refund",
        }

    # 5. Legal/fraud language regardless of intent
    legal_markers = ["lawyer", "lawsuit", "police", "fraud", "attorney",
                     "sue", "legal action", "consumer protection"]
    if any(m in text_lower for m in legal_markers):
        return {
            "decision": "ESCALATE",
            "reason": "Message contains legal/fraud-risk language regardless of intent.",
            "trigger_type": "legal_language",
        }

    return {
        "decision": "AUTO_HANDLE",
        "reason": f"Intent '{intent}' classified with confidence {confidence:.2f}, "
                  f"no hard-escalate or high-risk-language triggers present.",
        "trigger_type": "none",
    }
