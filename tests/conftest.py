"""
conftest.py — Shared pytest fixtures for TraceGuard tests.

All tests run WITHOUT paid API calls. External services are mocked.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src and project root are on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Force offline mode for all tests
os.environ["OFFLINE_MODE"] = "1"
os.environ["JUDGE_OFFLINE"] = "1"
# Ensure no real API key is used
os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture
def sample_threads_df():
    """A minimal DataFrame of historical threads for testing."""
    return pd.DataFrame([
        {
            "thread_id": "t1",
            "customer_message": "I was charged $47 for a ride that should have been $22",
            "brand_reply": "Please DM us your trip ID and we'll review the fare.",
            "thread_context": "",
        },
        {
            "thread_id": "t2",
            "customer_message": "left my phone in the car",
            "brand_reply": "Use the Lost Item option in your trip history.",
            "thread_context": "",
        },
        {
            "thread_id": "t3",
            "customer_message": "driver was swerving all over the road, very scary",
            "brand_reply": "We take safety very seriously. Please DM us immediately.",
            "thread_context": "",
        },
        {
            "thread_id": "t4",
            "customer_message": "app keeps crashing on Android",
            "brand_reply": "Please try reinstalling the app.",
            "thread_context": "",
        },
        {
            "thread_id": "t5",
            "customer_message": "can't log into my account",
            "brand_reply": "Please DM the email on your account.",
            "thread_context": "",
        },
    ])


@pytest.fixture
def agent(sample_threads_df):
    """An UberSupportAgent in offline mode for testing."""
    from src.pipeline import UberSupportAgent
    return UberSupportAgent(historical_threads=sample_threads_df)


@pytest.fixture
def golden_cases():
    """The versioned golden dataset cases."""
    from eval.golden_dataset import GOLDEN_CASES
    return GOLDEN_CASES
