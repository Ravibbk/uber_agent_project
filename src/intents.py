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
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "0") == "1"
client = OpenAI(api_key=api_key) if api_key and not OFFLINE_MODE else None
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "gpt-4o-mini")

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


def fallback_intent_from_message(customer_message: str) -> str:
    text = (customer_message or "").lower()
    if any(term in text for term in ["unsafe", "safety", "harass", "harassment", "accident", "police", "reckless", "hurt", "scared", "unsafe driver", "felt unsafe"]):
        return "safety_concern"
    if any(term in text for term in ["lost item", "left my phone", "forgot my bag", "lost my", "item in the car", "left item"]):
        return "lost_item"
    if any(term in text for term in ["login", "log in", "password", "account", "reset password", "verify identity", "can't access", "sign in"]):
        return "account_access"
    if any(term in text for term in ["app", "crash", "bug", "error", "gps", "map", "screen", "freeze", "loading"]):
        return "app_technical_issue"
    if any(term in text for term in ["charged", "fare", "price", "overcharged", "double", "quote", "refund", "cancellation fee"]):
        return "fare_dispute" if "refund" not in text and "cancellation fee" not in text else "refund_request" if "refund" in text else "cancellation_issue"
    if any(term in text for term in ["driver", "rude", "driver was rude", "refused", "route", "on the phone", "driver behavior"]):
        return "driver_behavior"
    if any(term in text for term in ["cancel", "cancellation", "cancelled", "cancelled on me", "driver cancelled"]):
        return "cancellation_issue"
    if any(term in text for term in ["refund", "money back", "want refund", "charged me wrong"]):
        return "refund_request"
    return "general_inquiry"


def classify(customer_message: str, thread_context: str = "") -> dict:
    if client is None:
        intent = fallback_intent_from_message(customer_message)
        return {
            "intent": intent,
            "confidence": 0.72,
            "rationale": "Offline fallback: rule-based intent detection used because no valid OpenAI key or quota was available.",
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
    except Exception:
        intent = fallback_intent_from_message(customer_message)
        return {
            "intent": intent,
            "confidence": 0.65,
            "rationale": "OpenAI request failed; offline rule-based fallback used instead.",
        }
    if parsed.get("intent") not in INTENTS:
        parsed["intent"] = "general_inquiry"
    return parsed


if __name__ == "__main__":
    # quick smoke test
    example = "my driver took a completely wrong route and charged me double what the app quoted"
    print(classify(example))
