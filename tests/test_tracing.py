"""
test_tracing.py — Tests for the structured tracing module.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.tracing import Span, TraceRecorder, load_trace


class TestSpan:
    def test_succeed(self):
        span = Span("run1", "test_stage")
        span.succeed(key="value")
        assert span.status == "success"
        assert span.duration_ms is not None
        assert span.metadata["key"] == "value"

    def test_fail(self):
        span = Span("run1", "test_stage")
        span.fail("something broke", error_category="api_error")
        assert span.status == "error"
        assert span.error == "something broke"
        assert span.error_category == "api_error"
        assert span.duration_ms is not None

    def test_fallback(self):
        span = Span("run1", "test_stage")
        span.fallback("no API key")
        assert span.status == "fallback"
        assert span.fallback_used is True

    def test_model_info(self):
        span = Span("run1", "test_stage")
        span.set_model_info(model="gpt-4o", prompt_version="abc123")
        assert span.model == "gpt-4o"
        assert span.prompt_version == "abc123"

    def test_to_dict(self):
        span = Span("run1", "test_stage")
        span.succeed()
        d = span.to_dict()
        assert d["run_id"] == "run1"
        assert d["stage"] == "test_stage"
        assert "timestamp_utc" in d
        assert "duration_ms" in d


class TestTraceRecorder:
    def test_start_span(self):
        recorder = TraceRecorder()
        span = recorder.start_span("test")
        assert span.run_id == recorder.run_id

    def test_multiple_spans(self):
        recorder = TraceRecorder()
        recorder.start_span("stage1").succeed()
        recorder.start_span("stage2").succeed()
        assert len(recorder.spans) == 2

    def test_summary(self):
        recorder = TraceRecorder()
        recorder.start_span("stage1").succeed()
        recorder.start_span("stage2").fail("error")
        summary = recorder.summary()
        assert summary["total_spans"] == 2
        assert summary["errors"] == 1

    def test_write_and_load(self, tmp_path):
        recorder = TraceRecorder(trace_dir=tmp_path)
        recorder.start_span("stage1").succeed()
        recorder.start_span("stage2").succeed()
        path = recorder.write()
        assert path is not None
        assert path.exists()

        loaded = load_trace(path)
        assert len(loaded) == 2
        assert loaded[0]["stage"] == "stage1"

    def test_write_creates_directory(self, tmp_path):
        trace_dir = tmp_path / "nested" / "traces"
        recorder = TraceRecorder(trace_dir=trace_dir)
        recorder.start_span("test").succeed()
        path = recorder.write()
        assert path is not None

    def test_to_dicts(self):
        recorder = TraceRecorder()
        recorder.start_span("test").succeed()
        dicts = recorder.to_dicts()
        assert len(dicts) == 1
        assert isinstance(dicts[0], dict)
