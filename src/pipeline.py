"""
pipeline.py

End-to-end: customer message -> intent -> retrieved grounding examples ->
drafted reply -> escalation decision. This is the single entry point the
eval harness and the README quickstart both call.

CHANGES (TraceGuard upgrade):
- Structured tracing for every pipeline stage.
- Input validation.
- Returns rich metadata including trace summary.
- Uses package imports.

DISCLAIMER: This is an independent portfolio prototype (TraceGuard).
It is NOT affiliated with or endorsed by Uber.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import build_uber_threads, load_raw
from src.intents import classify
from src.retrieval import HistoricalResolutionIndex
from src.reply_gen import draft_reply
from src.escalation import decide
from src.tracing import TraceRecorder

logger = logging.getLogger(__name__)


def load_threads(path: str) -> pd.DataFrame:
    """Accept either a processed thread CSV or a raw TWCS-like CSV."""
    df = pd.read_csv(path)
    required = {"customer_message", "brand_reply"}

    if required.issubset(df.columns):
        return df

    raw_path = Path(path)
    if raw_path.exists():
        try:
            raw_df = load_raw(str(raw_path))
            processed_df = build_uber_threads(raw_df)
            if not processed_df.empty:
                return processed_df
        except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"Could not parse raw TWCS-style CSV '{path}'.") from exc

    raise ValueError(
        f"Input file '{path}' is not in the expected format. "
        "Expected either processed columns ['customer_message', 'brand_reply'] "
        "or a raw TWCS-style CSV."
    )


class UberSupportAgent:
    """End-to-end support agent with structured tracing.

    DISCLAIMER: This is an independent portfolio prototype and is NOT
    affiliated with or endorsed by Uber.
    """

    def __init__(self, historical_threads: Optional[pd.DataFrame] = None, trace_dir: Path = None):
        if historical_threads is None:
            default_csv = Path("data/processed/uber_threads.csv")
            if default_csv.exists():
                try:
                    historical_threads = pd.read_csv(default_csv)
                except Exception as e:
                    logger.warning("Could not read %s: %s", default_csv, e)
                    historical_threads = None
            if historical_threads is None or historical_threads.empty:
                historical_threads = pd.DataFrame([
                    {"thread_id": "default_1", "customer_message": "I was overcharged for my ride", "brand_reply": "Please DM us your trip ID and account details so we can review the fare."},
                    {"thread_id": "default_2", "customer_message": "I left an item in the car", "brand_reply": "Please use the 'Find lost item' feature in your app to reach the driver."},
                ])
        self.index = HistoricalResolutionIndex(historical_threads)
        self.trace_dir = trace_dir or Path("traces")

    def handle(
        self,
        customer_message: str,
        thread_context: str = "",
        thread_id: str = None,
        exclude_ids: set = None,
    ) -> dict:
        """Process a customer message through the full pipeline.

        Args:
            customer_message: The customer's support message.
            thread_context: Prior conversation context.
            thread_id: ID of the thread (for retrieval exclusion).
            exclude_ids: Additional thread IDs to exclude from retrieval (for eval leakage control).

        Returns:
            dict with intent, reply, escalation, and trace summary.
        """
        tracer = TraceRecorder(trace_dir=self.trace_dir)

        # Stage 0: Input validation
        span = tracer.start_span("input_validation")
        if not customer_message or not customer_message.strip():
            span.fail("Empty or missing customer message", error_category="validation")
            tracer.write()
            return {
                "run_id": tracer.run_id,
                "customer_message": customer_message or "",
                "intent": "general_inquiry",
                "intent_confidence": 0.0,
                "intent_method": "deterministic_fallback",
                "intent_rationale": "Empty input — cannot classify.",
                "drafted_reply": "We'd be happy to help! Could you please describe the issue you're experiencing?",
                "escalation_decision": "ESCALATE",
                "escalation_reason": "Empty customer message — cannot auto-handle.",
                "escalation_trigger_type": "empty_input",
                "grounding_examples": [],
                "num_grounding_examples": 0,
                "trace": tracer.summary(),
                "spans": tracer.to_dicts(),
                "error": "empty_input",
            }
        span.succeed(message_length=len(customer_message))

        # Stage 1: Intent classification
        span = tracer.start_span("intent_classification")
        try:
            intent_result = classify(customer_message, thread_context)
            intent = intent_result["intent"]
            confidence = intent_result["confidence"]
            span.set_model_info(
                model=intent_result.get("model"),
                prompt_version=intent_result.get("prompt_hash"),
                token_usage=intent_result.get("token_usage"),
            )
            if intent_result.get("error"):
                span.fallback(intent_result["error"], intent=intent, confidence=confidence)
            else:
                span.succeed(intent=intent, confidence=confidence)
        except Exception as exc:
            logger.error("Classification stage failed: %s", exc, exc_info=True)
            span.fail(str(exc), error_category="classification_error")
            intent = "general_inquiry"
            confidence = 0.0
            intent_result = {"intent": intent, "confidence": confidence, "rationale": f"Error: {exc}"}

        # Stage 2: Retrieval
        span = tracer.start_span("retrieval")
        try:
            examples = self.index.retrieve(
                customer_message, k=3,
                exclude_thread_id=thread_id,
                exclude_ids=exclude_ids,
            )
            span.succeed(num_results=len(examples))
        except Exception as exc:
            logger.error("Retrieval stage failed: %s", exc, exc_info=True)
            span.fail(str(exc), error_category="retrieval_error")
            examples = []

        # Stage 3: Reply generation
        span = tracer.start_span("reply_generation")
        try:
            reply_result = draft_reply(customer_message, intent, examples, thread_context)
            reply = reply_result["reply"]
            span.set_model_info(
                model=reply_result.get("model"),
                prompt_version=reply_result.get("prompt_hash"),
                token_usage=reply_result.get("token_usage"),
            )
            if reply_result.get("is_fallback"):
                span.fallback("Using offline template", reply_length=len(reply))
            else:
                span.succeed(reply_length=len(reply))
        except Exception as exc:
            logger.error("Reply generation stage failed: %s", exc, exc_info=True)
            span.fail(str(exc), error_category="reply_error")
            reply = "We're sorry for the inconvenience. Please DM your trip details and we'll review this with you."

        # Stage 4: Escalation decision
        span = tracer.start_span("escalation")
        try:
            escalation = decide(customer_message, intent, confidence)
            span.succeed(
                decision=escalation["decision"],
                trigger_type=escalation.get("trigger_type", "unknown"),
            )
        except Exception as exc:
            logger.error("Escalation stage failed: %s", exc, exc_info=True)
            span.fail(str(exc), error_category="escalation_error")
            escalation = {"decision": "ESCALATE", "reason": f"Escalation logic error: {exc}"}

        # Write trace (never breaks the agent)
        tracer.write()

        return {
            "run_id": tracer.run_id,
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": confidence,
            "intent_method": intent_result.get("method", "heuristic"),
            "intent_rationale": intent_result.get("rationale", ""),
            "drafted_reply": reply,
            "escalation_decision": escalation["decision"],
            "escalation_reason": escalation["reason"],
            "escalation_trigger_type": escalation.get("trigger_type", "unknown"),
            "grounding_examples": examples,
            "num_grounding_examples": len(examples),
            "trace": tracer.summary(),
            "spans": tracer.to_dicts(),
        }


_default_agent: Optional[UberSupportAgent] = None


def get_agent() -> UberSupportAgent:
    """Retrieve or create the singleton UberSupportAgent instance."""
    global _default_agent
    if _default_agent is None:
        _default_agent = UberSupportAgent()
    return _default_agent


def run_pipeline(
    customer_message: str,
    thread_context: str = "",
    thread_id: str = None,
    exclude_ids: set = None,
) -> dict:
    """Convenience functional wrapper around default UberSupportAgent."""
    agent = get_agent()
    return agent.handle(
        customer_message,
        thread_context=thread_context,
        thread_id=thread_id,
        exclude_ids=exclude_ids,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/processed/uber_threads.csv")
    ap.add_argument("--limit", type=int, default=20, help="How many messages to run through the pipeline")
    ap.add_argument("--out", default="data/processed/pipeline_sample_output.csv")
    args = ap.parse_args()

    df = load_threads(args.threads)
    agent = UberSupportAgent(historical_threads=df)

    sample = df.sample(min(args.limit, len(df)), random_state=1)
    results = []
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        out = agent.handle(row["customer_message"], row.get("thread_context", ""), row.get("thread_id"))
        results.append(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Remove trace dict before CSV export
    for r in results:
        r.pop("trace", None)
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"Wrote {len(results)} pipeline outputs to {args.out}")


if __name__ == "__main__":
    main()
