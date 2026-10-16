"""
baselines.py

Two baselines the report compares against, per the assignment's requirement:

1. TRIVIAL baseline: always predicts the single most frequent intent in the
   training threads, and always drafts the same generic canned reply.
   Always escalates nothing (auto-handles everything) -- this is meant to
   look bad on escalation-recall specifically.

2. SIMPLE baseline: keyword/regex-based intent classifier (no LLM) +
   intent-specific template replies (no retrieval, no grounding) +
   rule-only escalation (safety keywords only).

These exist so the headline numbers mean something -- "our agent gets 81%
intent accuracy" is meaningless without knowing the trivial baseline gets,
say, 24% (1/9 intents, roughly) and the simple keyword baseline gets 58%.
"""
from collections import Counter

# --- Trivial baseline ---

CANNED_REPLY = "We're sorry to hear about this. Please DM us your trip details and we'll take a look right away."


def trivial_predict(train_df, test_messages):
    most_common_intent = Counter(train_df.get("gold_intent", [])).most_common(1)
    fallback_intent = most_common_intent[0][0] if most_common_intent else "general_inquiry"
    return [
        {"intent": fallback_intent, "reply": CANNED_REPLY, "escalation_decision": "AUTO_HANDLE"}
        for _ in test_messages
    ]


# --- Simple keyword baseline ---

KEYWORD_RULES = [
    ("safety_concern", ["unsafe", "assault", "harass", "accident", "scared", "threatened"]),
    ("fare_dispute", ["overcharg", "wrong fare", "charged me", "extra charge", "double charge"]),
    ("refund_request", ["refund", "money back", "reimburse"]),
    ("driver_behavior", ["rude", "driver was", "unprofessional", "refused to"]),
    ("lost_item", ["left my", "lost item", "forgot my", "phone in the car", "wallet in"]),
    ("account_access", ["can't log in", "cannot log in", "password", "verify my account", "payment method"]),
    ("cancellation_issue", ["cancel", "cancellation fee"]),
    ("app_technical_issue", ["app crash", "bug", "gps", "map is wrong", "not working"]),
]

TEMPLATE_REPLIES = {
    "safety_concern": "We take this very seriously. Please DM us right away so we can escalate this for you.",
    "fare_dispute": "Sorry about the fare confusion -- please DM your trip ID and we'll review the charge.",
    "refund_request": "We'd like to help. Please DM your trip ID so we can look into a refund.",
    "driver_behavior": "We're sorry to hear that. Please DM your trip ID so we can follow up with the driver.",
    "lost_item": "Please use the 'lost item' option in your trip history in the app, or DM your trip ID.",
    "account_access": "Please DM us the email on your account and we'll help you regain access.",
    "cancellation_issue": "Please DM your trip ID and we'll review the cancellation fee.",
    "app_technical_issue": "Sorry about that! Please try updating the app; if it persists, DM us your device/OS.",
    "general_inquiry": "Thanks for reaching out! Please DM us more details and we'll help.",
}


def simple_predict(test_messages):
    results = []
    for msg in test_messages:
        text_lower = msg.lower()
        matched_intent = "general_inquiry"
        for intent, kws in KEYWORD_RULES:
            if any(kw in text_lower for kw in kws):
                matched_intent = intent
                break
        escalation = "ESCALATE" if matched_intent == "safety_concern" else "AUTO_HANDLE"
        results.append(
            {
                "intent": matched_intent,
                "reply": TEMPLATE_REPLIES[matched_intent],
                "escalation_decision": escalation,
            }
        )
    return results
