"""
judge.py

LLM-as-judge for reply quality. Scores each drafted reply on 4 dimensions,
each 1-5, plus a pass/fail hallucination flag. Kept multi-dimensional
(rather than one overall 1-5) because a single "quality" number hides
WHICH way a reply is bad -- and the failure analysis in the report needs
that breakdown.

Judge model is deliberately a DIFFERENT, stronger call than the reply
generator would use by default in a cost-optimized setting, and we always
score against the actual customer message + intent, not just "does this
reply sound good in a vacuum" -- an ungrounded judge is a big source of
inflated eval scores (see judge_calibration.py + report, "what's misleading").
"""
import os
import json
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")

JUDGE_SYSTEM_PROMPT = """You are a strict quality reviewer for Uber's Twitter customer support replies.
Score the DRAFTED REPLY against the customer message on these 4 dimensions, each 1-5:

- relevance: does it actually address what the customer said (not just intent label)?
- grounding: is it consistent with how the brand has resolved similar issues (given the examples), without inventing facts (specific $ amounts, dates, promises) that weren't given?
- tone: appropriate, empathetic, on-brand support voice, not robotic or dismissive?
- actionability: does the customer know what happens next?

Also set "hallucination": true if the reply states a specific fact (amount, date, policy) not present in the input.

Respond with strict JSON only:
{"relevance": <1-5>, "grounding": <1-5>, "tone": <1-5>, "actionability": <1-5>, "hallucination": <true/false>, "notes": "<one sentence>"}
"""


def judge_reply(customer_message: str, intent: str, drafted_reply: str, grounding_examples: list) -> dict:
    if client is None:
        return offline_judge(customer_message, intent, drafted_reply, grounding_examples)

    examples_block = "\n".join(
        f"- {ex['customer_message']} -> {ex['brand_reply']}" for ex in grounding_examples
    ) or "(none provided)"

    user_content = f"""Customer message: {customer_message}
Classified intent: {intent}
Grounding examples given to the reply generator:
{examples_block}

Drafted reply to evaluate: {drafted_reply}
"""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except (OpenAIError, json.JSONDecodeError, TypeError, AttributeError):
        # Keep evaluation runnable without converting an API outage into a
        # misleading success-shaped score.
        return offline_judge(customer_message, intent, drafted_reply, grounding_examples)


def offline_judge(customer_message: str, intent: str, drafted_reply: str, grounding_examples: list) -> dict:
    """Deterministic smoke-test rubric used when no judge API is configured."""
    message = (customer_message or "").lower()
    reply = (drafted_reply or "").lower()
    intent_terms = {
        "fare_dispute": ("fare", "charge", "trip"),
        "refund_request": ("refund", "trip"),
        "driver_behavior": ("driver", "trip"),
        "safety_concern": ("safety", "trip"),
        "lost_item": ("lost", "trip"),
        "account_access": ("account", "email"),
        "app_technical_issue": ("app", "device"),
        "cancellation_issue": ("cancel", "trip"),
        "general_inquiry": ("details", "help"),
    }
    terms = intent_terms.get(intent, ("help",))
    overlap = sum(term in reply for term in terms)
    asks_for_next_step = any(
        phrase in reply for phrase in ("dm", "contact", "try ", "please use", "send")
    )
    hallucination = any(token in reply for token in ("$0", "$1", "tomorrow", "guarantee"))
    return {
        "relevance": min(5, 3 + overlap),
        "grounding": 4 if grounding_examples and not hallucination else 3,
        "tone": 4 if any(word in reply for word in ("sorry", "understand", "seriously")) else 3,
        "actionability": 4 if asks_for_next_step else 2,
        "hallucination": hallucination,
        "notes": "Deterministic offline rubric; replace with LLM judge scores for the submission.",
    }
