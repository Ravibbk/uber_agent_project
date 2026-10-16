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
"""

# Intents that ALWAYS escalate, no matter what the model thinks -- these carry
# legal/safety/brand risk too high to auto-resolve.
HARD_ESCALATE_INTENTS = {"safety_concern"}

# Below this classifier confidence, we don't trust the intent label enough
# to auto-handle -- better to route to a human than reply confidently to
# the wrong problem.
CONFIDENCE_THRESHOLD = 0.55

# Simple negative-sentiment / anger signal words -- crude on purpose (see
# decision log #9: we chose a cheap heuristic here rather than a second
# LLM call, and measured that it's good enough in eval/run_eval.py)
ANGER_MARKERS = [
    "unacceptable", "lawsuit", "lawyer", "scam", "worst", "never again",
    "disgusting", "furious", "fraud", "stolen", "police",
]


def decide(customer_message: str, intent: str, confidence: float) -> dict:
    text_lower = customer_message.lower()

    if intent in HARD_ESCALATE_INTENTS:
        return {
            "decision": "ESCALATE",
            "reason": f"Intent '{intent}' is a hard-escalate category (safety-related).",
        }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "decision": "ESCALATE",
            "reason": f"Intent classification confidence ({confidence:.2f}) below threshold "
                       f"({CONFIDENCE_THRESHOLD}); risk of replying to the wrong problem.",
        }

    if intent == "refund_request" and any(m in text_lower for m in ANGER_MARKERS):
        return {
            "decision": "ESCALATE",
            "reason": "Refund request combined with high-anger language; risk of "
                      "brand/legal escalation if handled wrong by a bot.",
        }

    if any(m in text_lower for m in ["lawyer", "lawsuit", "police", "fraud"]):
        return {
            "decision": "ESCALATE",
            "reason": "Message contains legal/fraud-risk language regardless of intent.",
        }

    return {
        "decision": "AUTO_HANDLE",
        "reason": f"Intent '{intent}' classified with confidence {confidence:.2f}, "
                  f"no hard-escalate or high-risk-language triggers present.",
    }
