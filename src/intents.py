"""
intents.py

Defines the intent taxonomy for Uber_Support and a classifier that labels
an incoming customer message with one of these intents.

WHERE THE TAXONOMY CAME FROM:
This is NOT invented from thin air -- it should be derived by reading a
sample of real Uber_Support threads (see notebooks/01_explore_intents.ipynb
or just eyeball data/processed/uber_threads.csv after running data_prep.py)
and open-coding ~100-150 of them, then merging near-duplicate categories
until you get a small, mutually-exclusive set. The 9 below are a reasonable
starting taxonomy based on how Uber's public support handles typically
break down; TREAT THIS AS A DRAFT and adjust after you've actually read
your sample -- that adjustment is itself a decision worth logging.

CHANGES (TraceGuard upgrade):
- Lazy OpenAI client initialization (no import-time crash without key).
- Logging instead of silent exception swallowing (Defect F fix).
- Expanded safety keywords in fallback classifier (Defect E fix).
- Hardcoded confidence replaced with context-appropriate values.
"""
import hashlib
import json
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

OFFLINE_MODE = os.getenv("OFFLINE_MODE", "0") == "1"
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "gpt-4o-mini")

# Lazy client — never crash at import time (Defect A fix)
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OFFLINE_MODE:
        return None
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=api_key)
        return _client
    except Exception as exc:
        logger.warning("Failed to initialize OpenAI client: %s", exc)
        return None


INTENTS = {
    "fare_dispute": "Customer says they were charged incorrectly / overcharged / surprise fare.",
    "refund_request": "Customer is explicitly asking for money back (not necessarily a fare dispute -- could be a cancellation fee, promo not applied, etc).",
    "driver_behavior": "Complaint about a driver's conduct, rudeness, unsafe driving, route issues, refusal of service.",
    "safety_concern": "Anything involving physical safety, harassment, accident, or the customer feeling unsafe. HIGH PRIORITY.",
    "lost_item": "Customer left an item in a vehicle and wants help getting it back.",
    "account_access": "Login issues, payment method problems, can't verify identity, app won't let them book.",
    "app_technical_issue": "Bugs, crashes, GPS/map errors, unrelated to account access specifically.",
    "cancellation_issue": "Disputes or confusion around a ride/order cancellation and any associated fee.",
    "general_inquiry": "Anything else -- praise, general questions, unclear intent, small talk.",
}

SYSTEM_PROMPT = f"""You are an intent classifier for Uber's customer support Twitter account.
Classify the customer's message into exactly one of these intents:

{json.dumps(INTENTS, indent=2)}

Respond with strict JSON only, no markdown fences:
{{"intent": "<one of the keys above>", "confidence": <float 0-1>, "rationale": "<one short sentence>"}}
"""

# Hash for prompt versioning
SYSTEM_PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


def fallback_intent_from_message(customer_message: str) -> str:
    """Rule-based fallback intent classifier.

    Keyword order matters: safety_concern is checked FIRST to avoid
    misclassifying implicit safety signals as driver_behavior (Defect E fix).
    """
    text = (customer_message or "").lower()

    # Safety concern — checked FIRST, expanded keywords (Defect E fix)
    safety_keywords = [
        "unsafe", "safety", "harass", "harassment", "accident", "police",
        "reckless", "hurt", "scared", "felt unsafe", "swerving", "swerve",
        "threatened", "threaten", "weapon", "attack", "assault", "hit me",
        "dangerous", "danger", "emergency", "violence", "violent",
        "followed me", "stalking", "kidnap", "gun", "knife",
        "driving erratically", "ran a red", "ran red light",
    ]
    if any(term in text for term in safety_keywords):
        return "safety_concern"

    if any(term in text for term in [
        "lost item", "left my phone", "forgot my bag", "lost my",
        "item in the car", "left item", "left my wallet", "left my bag",
    ]):
        return "lost_item"

    if any(term in text for term in [
        "login", "log in", "password", "account", "reset password",
        "verify identity", "can't access", "sign in", "locked out",
        "hacked",
    ]):
        return "account_access"

    if any(term in text for term in [
        "app crash", "crash", "bug", "error", "gps", "map",
        "screen", "freeze", "loading", "not working", "glitch",
    ]):
        return "app_technical_issue"

    # Refund before fare_dispute (refund is more specific)
    if any(term in text for term in [
        "refund", "money back", "want refund", "reimburse",
    ]):
        return "refund_request"

    if any(term in text for term in [
        "charged", "fare", "price", "overcharged", "double charge",
        "extra charge", "wrong fare",
    ]):
        return "fare_dispute"

    if any(term in text for term in [
        "cancel", "cancellation", "cancelled", "canceled",
        "cancellation fee",
    ]):
        return "cancellation_issue"

    if any(term in text for term in [
        "driver", "rude", "driver was rude", "refused",
        "route", "on the phone", "driver behavior", "unprofessional",
    ]):
        return "driver_behavior"

    return "general_inquiry"


def classify(customer_message: str, thread_context: str = "") -> dict:
    """Classify a customer message into one of the 9 intents.

    Returns dict with keys: intent, confidence, rationale, model, prompt_hash.
    Falls back to rule-based classification when no API is available.
    """
    client = _get_client()
    if client is None:
        intent = fallback_intent_from_message(customer_message)
        return {
            "intent": intent,
            "confidence": 0.55,
            "rationale": "Offline fallback: rule-based intent detection used because no valid OpenAI key or quota was available.",
            "model": "offline_rules",
            "prompt_hash": "n/a",
        }

    user_content = f"Thread context: {thread_context}\n\nCustomer message to classify: {customer_message}"
    try:
        resp = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)

        # Extract token usage if provider supplies it
        token_usage = None
        if resp.usage:
            token_usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        result = {
            "intent": parsed.get("intent", "general_inquiry"),
            "confidence": parsed.get("confidence", 0.5),
            "rationale": parsed.get("rationale", ""),
            "model": CLASSIFY_MODEL,
            "prompt_hash": SYSTEM_PROMPT_HASH,
            "token_usage": token_usage,
        }

        if result["intent"] not in INTENTS:
            logger.warning("LLM returned unknown intent '%s', falling back to general_inquiry", result["intent"])
            result["intent"] = "general_inquiry"

        return result

    except Exception as exc:
        # Defect F fix: log the error instead of silently swallowing
        logger.error("Intent classification API call failed: %s", exc, exc_info=True)
        intent = fallback_intent_from_message(customer_message)
        return {
            "intent": intent,
            "confidence": 0.40,
            "rationale": f"OpenAI request failed ({type(exc).__name__}); offline rule-based fallback used.",
            "model": "offline_rules",
            "prompt_hash": "n/a",
            "error": type(exc).__name__,
        }


if __name__ == "__main__":
    # quick smoke test
    examples = [
        "my driver took a completely wrong route and charged me double what the app quoted",
        "My driver was swerving across lanes",
        "The driver threatened me",
        "I felt unsafe during the ride",
    ]
    for ex in examples:
        result = classify(ex)
        print(f"  {ex!r}")
        print(f"    -> {result['intent']} (confidence={result['confidence']:.2f})")
        print()
