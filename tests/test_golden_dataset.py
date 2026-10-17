"""
test_golden_dataset.py — Tests for the versioned golden dataset.

Validates schema, unique IDs, label values, and coverage.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.golden_dataset import (
    GOLDEN_CASES,
    DATASET_VERSION,
    VALID_INTENTS,
    VALID_ESCALATIONS,
    VALID_RISK_CATEGORIES,
    VALID_SOURCES,
    VALID_ANNOTATION_STATUSES,
    validate_dataset,
    validate_case,
    compute_dataset_hash,
)


class TestDatasetValidation:
    def test_dataset_is_valid(self):
        report = validate_dataset(GOLDEN_CASES)
        assert report["is_valid"], f"Validation errors: {report['errors']}"

    def test_no_duplicate_ids(self):
        report = validate_dataset(GOLDEN_CASES)
        assert report["duplicate_ids"] == []

    def test_minimum_size(self):
        assert len(GOLDEN_CASES) >= 30, f"Need at least 30 cases, got {len(GOLDEN_CASES)}"

    def test_all_cases_have_version(self):
        for case in GOLDEN_CASES:
            assert case["dataset_version"] == DATASET_VERSION


class TestCaseSchema:
    def test_all_intents_valid(self):
        for case in GOLDEN_CASES:
            assert case["expected_intent"] in VALID_INTENTS, \
                f"Case {case['case_id']}: invalid intent {case['expected_intent']}"

    def test_all_escalations_valid(self):
        for case in GOLDEN_CASES:
            assert case["expected_escalation"] in VALID_ESCALATIONS, \
                f"Case {case['case_id']}: invalid escalation {case['expected_escalation']}"

    def test_all_risk_categories_valid(self):
        for case in GOLDEN_CASES:
            assert case["risk_category"] in VALID_RISK_CATEGORIES, \
                f"Case {case['case_id']}: invalid risk_category {case['risk_category']}"

    def test_all_sources_valid(self):
        for case in GOLDEN_CASES:
            assert case["source"] in VALID_SOURCES, \
                f"Case {case['case_id']}: invalid source {case['source']}"

    def test_all_annotation_statuses_valid(self):
        for case in GOLDEN_CASES:
            assert case["annotation_status"] in VALID_ANNOTATION_STATUSES


class TestDatasetCoverage:
    def test_all_intents_represented(self):
        intents = {c["expected_intent"] for c in GOLDEN_CASES}
        assert VALID_INTENTS.issubset(intents), \
            f"Missing intents: {VALID_INTENTS - intents}"

    def test_has_safety_cases(self):
        safety = [c for c in GOLDEN_CASES if c["risk_category"] == "safety"]
        assert len(safety) >= 3

    def test_has_adversarial_cases(self):
        adv = [c for c in GOLDEN_CASES if c["risk_category"] == "adversarial"]
        assert len(adv) >= 1

    def test_has_edge_cases(self):
        edges = [c for c in GOLDEN_CASES if c["risk_category"] in ("ambiguous", "edge_case")]
        assert len(edges) >= 2

    def test_has_escalation_cases(self):
        escalate = [c for c in GOLDEN_CASES if c["expected_escalation"] == "ESCALATE"]
        assert len(escalate) >= 5

    def test_has_regression_cases(self):
        regression = [c for c in GOLDEN_CASES if c["source"] == "fault_injection"]
        assert len(regression) >= 1


class TestDatasetHash:
    def test_hash_is_deterministic(self):
        h1 = compute_dataset_hash(GOLDEN_CASES)
        h2 = compute_dataset_hash(GOLDEN_CASES)
        assert h1 == h2

    def test_hash_is_string(self):
        h = compute_dataset_hash(GOLDEN_CASES)
        assert isinstance(h, str)
        assert len(h) == 16


class TestSingleCaseValidation:
    def test_valid_case(self):
        case = GOLDEN_CASES[0]
        errors = validate_case(case)
        assert errors == []

    def test_missing_field(self):
        case = {"case_id": "test"}
        errors = validate_case(case)
        assert len(errors) > 0

    def test_invalid_intent(self):
        case = {**GOLDEN_CASES[0], "expected_intent": "nonexistent_intent"}
        errors = validate_case(case)
        assert any("Invalid intent" in e for e in errors)
