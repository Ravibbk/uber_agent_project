"""
tracing.py — Structured Tracing for TraceGuard

Lightweight JSONL-based tracing for the agent pipeline. Each pipeline run
gets a unique run_id, and each stage (classification, retrieval, reply
generation, escalation) gets its own span with timing, status, and error
information.

Design decisions:
- JSONL format: one JSON object per line, easy to grep/filter/stream.
- Tracing never breaks the agent: all writes are wrapped in try/except.
- Sensitive data (API keys, auth headers) is never recorded.
- Token usage and cost are recorded ONLY when the provider supplies them.
- Timestamps are always UTC ISO-8601.
"""
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default trace output directory
TRACE_DIR = Path("traces")


class Span:
    """A single stage within a pipeline run."""

    def __init__(
        self,
        run_id: str,
        stage: str,
        parent_span_id: Optional[str] = None,
    ):
        self.run_id = run_id
        self.span_id = str(uuid.uuid4())[:12]
        self.parent_span_id = parent_span_id
        self.stage = stage
        self.start_time = datetime.now(timezone.utc)
        self._start_perf = time.perf_counter()
        self.end_time: Optional[datetime] = None
        self.duration_ms: Optional[float] = None
        self.status: str = "started"
        self.error: Optional[str] = None
        self.error_category: Optional[str] = None
        self.fallback_used: bool = False
        self.model: Optional[str] = None
        self.prompt_version: Optional[str] = None
        self.token_usage: Optional[dict] = None  # Only from provider
        self.estimated_cost_usd: Optional[float] = None
        self.metadata: dict = {}

    def succeed(self, **metadata: Any) -> "Span":
        self.status = "success"
        self._finish()
        self.metadata.update(metadata)
        return self

    def fail(self, error: str, error_category: str = "unknown", **metadata: Any) -> "Span":
        self.status = "error"
        self.error = error
        self.error_category = error_category
        self._finish()
        self.metadata.update(metadata)
        return self

    def fallback(self, reason: str, **metadata: Any) -> "Span":
        self.status = "fallback"
        self.fallback_used = True
        self.error = reason
        self.error_category = "fallback"
        self._finish()
        self.metadata.update(metadata)
        return self

    def set_model_info(
        self,
        model: Optional[str] = None,
        prompt_version: Optional[str] = None,
        token_usage: Optional[dict] = None,
        estimated_cost_usd: Optional[float] = None,
    ) -> "Span":
        if model:
            self.model = model
        if prompt_version:
            self.prompt_version = prompt_version
        if token_usage:
            self.token_usage = token_usage
        if estimated_cost_usd is not None:
            self.estimated_cost_usd = estimated_cost_usd
        return self

    def _finish(self) -> None:
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = round((time.perf_counter() - self._start_perf) * 1000, 2)

    def to_dict(self) -> dict:
        d = {
            "run_id": self.run_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "stage": self.stage,
            "timestamp_utc": self.start_time.isoformat(),
            "end_timestamp_utc": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "error_category": self.error_category,
            "fallback_used": self.fallback_used,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "token_usage": self.token_usage,
            "estimated_cost_usd": self.estimated_cost_usd,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class TraceRecorder:
    """Records spans for a single pipeline run to JSONL."""

    def __init__(self, run_id: Optional[str] = None, trace_dir: Optional[Path] = None):
        self.run_id = run_id or str(uuid.uuid4())[:16]
        self.trace_dir = trace_dir or TRACE_DIR
        self.spans: list[Span] = []

    def start_span(self, stage: str, parent_span_id: Optional[str] = None) -> Span:
        span = Span(run_id=self.run_id, stage=stage, parent_span_id=parent_span_id)
        self.spans.append(span)
        return span

    def to_dicts(self) -> list[dict]:
        return [s.to_dict() for s in self.spans]

    def write(self) -> Optional[Path]:
        """Write all spans to a JSONL file. Returns path or None if write failed."""
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f"trace_{self.run_id}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for span in self.spans:
                    f.write(json.dumps(span.to_dict(), default=str) + "\n")
            return path
        except Exception as exc:
            # Tracing must never break the agent
            logger.warning("Failed to write trace file: %s", exc)
            return None

    def summary(self) -> dict:
        """Return a concise summary of this run's trace."""
        total_ms = sum(s.duration_ms or 0 for s in self.spans)
        errors = [s for s in self.spans if s.status == "error"]
        fallbacks = [s for s in self.spans if s.fallback_used]
        return {
            "run_id": self.run_id,
            "total_spans": len(self.spans),
            "total_duration_ms": round(total_ms, 2),
            "errors": len(errors),
            "fallbacks": len(fallbacks),
            "stages": [
                {
                    "stage": s.stage,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "model": s.model,
                }
                for s in self.spans
            ],
        }


def load_trace(path: Path) -> list[dict]:
    """Load a trace file and return list of span dicts."""
    spans = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans
