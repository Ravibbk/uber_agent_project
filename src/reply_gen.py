"""
reply_gen.py

Drafts a reply to a customer message, grounded in how Uber_Support has
historically handled similar issues (retrieved via retrieval.py).

Grounding is done by literally showing the model 2-3 real past
(customer_message, brand_reply) pairs with similar issues, and instructing
it to match that resolution *pattern* (tone, what info is asked for, what
is/isn't promised) rather than copy specific details like trip IDs or
dollar amounts, which will be wrong for this customer.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

from intents import fallback_intent_from_message

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None
REPLY_MODEL = os.getenv("REPLY_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are drafting a reply as Uber's official Twitter support account (@Uber_Support).

Rules:
- Match the tone and structure of the example past resolutions below -- they show how this brand
  actually handles issues like this one.
- Do NOT invent specific facts (trip IDs, refund amounts, dates) that you don't have.
- If the historical pattern is "ask the customer to DM their trip details", do that -- don't
  try to resolve it yourself in the tweet.
- Keep it under 280 characters, in Twitter support voice: brief, empathetic, action-oriented.
- Output ONLY the reply text, nothing else.
"""


def draft_reply(customer_message: str, intent: str, examples: list, thread_context: str = "") -> str:
    if client is None:
        fallback_intent = fallback_intent_from_message(customer_message)
        if fallback_intent == "safety_concern":
            return "We take safety concerns seriously. Please DM us your trip details and we will review this immediately."
        if fallback_intent == "lost_item":
            return "We’re sorry about that. Please use the Lost Item option in your trip history or DM us your trip details so we can assist."
        if fallback_intent == "account_access":
            return "We’re sorry about the access issue. Please DM the email on your account and we will help you regain access."
        if fallback_intent == "app_technical_issue":
            return "We’re sorry the app is giving you trouble. Please try reinstalling the app and DM us your device details if it keeps happening."
        if fallback_intent == "fare_dispute":
            return "We understand your concern. Please DM your trip details so we can review the fare and look into the charge."
        if fallback_intent == "refund_request":
            return "We’re sorry to hear that. Please DM your trip details so we can review the refund request."
        if fallback_intent == "cancellation_issue":
            return "We understand the frustration. Please DM your trip details and we will review the cancellation issue."
        if fallback_intent == "driver_behavior":
            return "We’re sorry you had that experience. Please DM your trip details so we can look into the driver feedback."
        return "We’re sorry for the inconvenience. Please DM your trip details and we’ll review this with you."

    examples_block = "\n\n".join(
        f"Past similar issue: {ex['customer_message']}\nPast Uber_Support reply: {ex['brand_reply']}"
        for ex in examples
    ) or "(no close historical match found -- use general best-practice support tone)"

    user_content = f"""Customer's intent (pre-classified): {intent}
Thread context: {thread_context}
Customer message: {customer_message}

Examples of how Uber_Support has resolved similar issues before:
{examples_block}

Draft the reply now."""

    try:
        resp = client.chat.completions.create(
            model=REPLY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "We’re sorry for the inconvenience. Please DM your trip details and we’ll review this with you."
