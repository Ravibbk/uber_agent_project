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
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
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
    examples_block = "\n".join(
        f"- {ex['customer_message']} -> {ex['brand_reply']}" for ex in grounding_examples
    ) or "(none provided)"

    user_content = f"""Customer message: {customer_message}
Classified intent: {intent}
Grounding examples given to the reply generator:
{examples_block}

Drafted reply to evaluate: {drafted_reply}
"""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"relevance": None, "grounding": None, "tone": None,
                 "actionability": None, "hallucination": None, "notes": "parse_error"}
