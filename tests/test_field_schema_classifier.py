from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from python.librarian.field_classifier import (
    QwenTeacherFieldClassifier,
    RuleFieldClassifier,
    prediction_cache_key,
)
from python.librarian.field_schema import FieldPrediction, FieldRegistry, default_field_registry


class FieldSchemaClassifierTests(unittest.TestCase):
    def test_registry_round_trip_is_stable(self) -> None:
        registry = default_field_registry()
        registry.record_predictions(
            [
                FieldPrediction("license constraint", "must credit Cameron", 0.91),
                FieldPrediction("license_constraint", "must credit Cameron", 0.89),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            registry.save(path)
            first = path.read_text(encoding="utf-8")
            FieldRegistry.load(path).save(path)
            second = path.read_text(encoding="utf-8")
        self.assertEqual(first, second)
        self.assertIn("license_constraint", registry.fields)
        self.assertEqual(registry.fields["license_constraint"].status, "proposed")

    def test_auto_promotion_gates_are_deterministic(self) -> None:
        registry = default_field_registry()
        predictions = [
            FieldPrediction("license_constraint", f"value_{index % 6}", 0.91)
            for index in range(30)
        ]
        registry.record_predictions(predictions)
        promoted = registry.promote_ready_fields(
            {"license_constraint": 0.05},
            {"license_constraint": 0.0},
            min_count=25,
            min_confidence=0.82,
            min_distinct_values=5,
            min_lift=0.03,
            max_hard_negative_delta=0.01,
        )
        self.assertEqual(promoted, ["license_constraint"])
        self.assertEqual(registry.fields["license_constraint"].status, "promoted")

    def test_rule_classifier_extracts_stress_fields(self) -> None:
        registry = default_field_registry()
        classifier = RuleFieldClassifier()
        predictions = classifier.classify(
            "user_001 asked that project_042 use Qwen for aurora with teal as the current color.",
            registry,
        )
        pairs = {(item.field_name, item.value) for item in predictions}
        self.assertIn(("user_id", "user_001"), pairs)
        self.assertIn(("project", "project_042"), pairs)
        self.assertIn(("brand", "aurora"), pairs)
        self.assertIn(("tool", "qwen"), pairs)
        self.assertIn(("time_scope", "current"), pairs)

    def test_qwen_cache_key_uses_registry_version(self) -> None:
        registry = default_field_registry()
        key_one = prediction_cache_key("hello user_001", registry)
        registry.version += 1
        key_two = prediction_cache_key("hello user_001", registry)
        self.assertNotEqual(key_one, key_two)

    def test_qwen_cache_reads_predictions(self) -> None:
        registry = default_field_registry()
        text = "remember user_001 project_001"
        key = prediction_cache_key(text, registry)
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        key: [
                            {"field_name": "user_id", "value": "user_001", "confidence": 0.95},
                            {"field_name": "project", "value": "project_001", "confidence": 0.94},
                        ]
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            classifier = QwenTeacherFieldClassifier(cache_path)
            predictions = classifier.classify(text, registry)
        pairs = {(item.field_name, item.value) for item in predictions}
        self.assertEqual(pairs, {("project", "project_001"), ("user_id", "user_001")})


if __name__ == "__main__":
    unittest.main()
