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

CHANGES (TraceGuard upgrade):
- Defect A fix: NO import-time crash. The OpenAI client is lazy-loaded.
  JUDGE_OFFLINE=1 works without any API key.
- Offline judge is clearly labeled as heuristic.
- judge_mode field always present in output.
"""
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
JUDGE_OFFLINE = os.getenv("JUDGE_OFFLINE", "0") == "1"

# Lazy client — Defect A fix: NEVER crash at import time
_client = None


def _get_judge_client():
    """Lazy-initialize the OpenAI client for the judge.

    Returns None if:
    - JUDGE_OFFLINE=1 (explicitly requesting offline mode)
    - No OPENAI_API_KEY is set
    """
    global _client
    if _client is not None:
        return _client
    if JUDGE_OFFLINE:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=api_key)
        return _client
    except Exception as exc:
        logger.warning("Failed to initialize OpenAI client for judge: %s", exc)
        return None


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


def offline_judge(customer_message: str, intent: str, drafted_reply: str) -> dict:
    """Return a transparent local score when the judge API is unavailable.

    These are HEURISTIC scores based on keyword matching. They are NOT
    equivalent to LLM judge scores and should be clearly labeled as such.
    """
    message = (customer_message or "").lower()
    reply = (drafted_reply or "").lower()
    action_words = ["dm", "direct message", "email", "contact", "send", "help"]
    risk_words = ["unsafe", "assault", "harass", "accident", "threat", "police"]
    intent_words = {
        "safety_concern": risk_words,
        "lost_item": ["lost", "left", "phone", "wallet", "item"],
        "account_access": ["login", "log in", "password", "account", "access"],
        "fare_dispute": ["fare", "charge", "charged", "overcharged", "price"],
        "refund_request": ["refund", "money back", "reimburse"],
        "driver_behavior": ["driver", "rude", "behavior", "route"],
        "app_technical_issue": ["app", "crash", "bug", "error", "working"],
        "cancellation_issue": ["cancel", "cancellation"],
        "general_inquiry": [],
    }
    matches = sum(word in message for word in intent_words.get(intent, []))
    relevance = 4 if matches else 3
    actionability = 5 if any(word in reply for word in action_words) else 3
    tone = 4 if any(word in reply for word in ["sorry", "help", "understand", "please"]) else 3
    hallucination = bool(re.search(r"\$\d+|\b\d+%|\b\d{1,2}/\d{1,2}/\d{2,4}\b", reply))
    grounding = 3 if reply else 1
    return {
        "relevance": relevance,
        "grounding": grounding,
        "tone": tone,
        "actionability": actionability,
        "hallucination": hallucination,
        "notes": "HEURISTIC SCORE: Offline rule-based judge. Not equivalent to LLM judge.",
        "judge_mode": "offline_heuristic",
    }


def judge_reply(customer_message: str, intent: str, drafted_reply: str, grounding_examples: list = None) -> dict:
    """Score a drafted reply on 4 quality dimensions + hallucination.

    Returns a dict with: relevance, grounding, tone, actionability,
    hallucination (bool), notes, judge_mode.

    If no API key or JUDGE_OFFLINE=1, uses offline heuristic judge.
    If API call fails, falls back to offline with a logged warning.
    """
    if grounding_examples is None:
        grounding_examples = []

    client = _get_judge_client()
    if client is None:
        if not JUDGE_OFFLINE:
            logger.info(
                "No OPENAI_API_KEY available for LLM judge. "
                "Using offline heuristic judge. Set JUDGE_OFFLINE=1 to suppress this message, "
                "or provide an API key for LLM-based judging."
            )
        return offline_judge(customer_message, intent, drafted_reply)

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
        result = json.loads(resp.choices[0].message.content)
        result["judge_mode"] = "llm"
        return result
    except Exception as exc:
        logger.error("LLM judge call failed: %s — falling back to offline heuristic", exc)
        result = offline_judge(customer_message, intent, drafted_reply)
        result["notes"] = f"LLM judge failed ({type(exc).__name__}); fell back to heuristic. " + result["notes"]
        return result
