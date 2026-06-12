from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from python.librarian.field_schema import FieldPrediction, FieldRegistry, default_field_registry


PROMPT_VERSION = "field-classifier-v1"
USER_RE = re.compile(r"\buser_\d{3}\b", re.IGNORECASE)
PROJECT_RE = re.compile(r"\bproject_\d{3}\b", re.IGNORECASE)
SESSION_RE = re.compile(r"\buser_\d{3}_project_\d{3}_[a-z0-9_:-]+\b", re.IGNORECASE)
BRANDS = ("aurora", "cinder", "northstar", "ember", "harbor", "kinetic", "mosaic", "signal")
TOOLS = ("qwen", "figma", "slack", "jira", "github", "bigquery", "drive", "salesforce")
COLORS = ("teal", "black", "white", "coral", "navy", "silver", "lime", "violet", "amber", "graphite")
TONES = ("concise", "formal", "technical", "plainspoken", "visual", "compliance-first", "executive")


def prediction_cache_key(text: str, registry: FieldRegistry, prompt_version: str = PROMPT_VERSION) -> str:
    payload = {
        "prompt_version": prompt_version,
        "registry_version": registry.version,
        "fields": sorted(registry.fields),
        "text": str(text or ""),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_prediction_json(value: Any, *, source_type: str, teacher_version: str) -> list[FieldPrediction]:
    if isinstance(value, str):
        data = json.loads(value)
    else:
        data = value
    if isinstance(data, dict):
        data = data.get("fields") or data.get("predictions") or []
    if not isinstance(data, list):
        raise ValueError("field prediction payload must be a list or object with fields")
    predictions: list[FieldPrediction] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        field_name = item.get("field") if "field" in item else item.get("field_name")
        prediction = FieldPrediction(
            field_name=str(field_name or ""),
            value=str(item.get("value") or ""),
            confidence=float(item.get("confidence") or 0.0),
            source_span=str(item.get("source_span") or item.get("span") or item.get("value") or ""),
            source_type=source_type,
            teacher_version=teacher_version,
        ).normalized()
        if prediction.field_name and prediction.value and prediction.confidence > 0.0:
            predictions.append(prediction)
    return stable_predictions(predictions)


def stable_predictions(predictions: Iterable[FieldPrediction]) -> list[FieldPrediction]:
    best: dict[tuple[str, str], FieldPrediction] = {}
    for prediction in predictions:
        item = prediction.normalized()
        if not item.field_name or not item.value:
            continue
        key = (item.field_name, item.value)
        current = best.get(key)
        if current is None or item.confidence > current.confidence:
            best[key] = item
    return [best[key] for key in sorted(best)]


class FieldPredictionCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {}

    def get(self, key: str) -> Any | None:
        return self.data.get(key)

    def put(self, key: str, predictions: list[FieldPrediction]) -> None:
        self.data[key] = [
            {
                "field_name": item.field_name,
                "value": item.value,
                "confidence": round(float(item.confidence), 8),
                "source_span": item.source_span,
                "source_type": item.source_type,
                "teacher_version": item.teacher_version,
            }
            for item in stable_predictions(predictions)
        ]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RuleFieldClassifier:
    def __init__(self, teacher_version: str = "rules-v1"):
        self.teacher_version = teacher_version

    def classify(self, text: str, registry: FieldRegistry | None = None) -> list[FieldPrediction]:
        registry = registry or default_field_registry()
        lower = str(text or "").lower()
        predictions: list[FieldPrediction] = []

        def add(field_name: str, value: str, confidence: float, span: str | None = None) -> None:
            resolved = registry.resolve_name(field_name)
            predictions.append(
                FieldPrediction(
                    field_name=resolved,
                    value=value,
                    confidence=confidence,
                    source_span=span or value,
                    source_type="rules",
                    teacher_version=self.teacher_version,
                )
            )

        for user in USER_RE.findall(lower):
            add("user_id", user, 0.96)
            add("entity", user, 0.82)
        for project in PROJECT_RE.findall(lower):
            add("project", project, 0.96)
            add("entity", project, 0.84)
        for session in SESSION_RE.findall(lower):
            add("session_id", session, 0.88)
        for brand in BRANDS:
            if re.search(rf"\b{re.escape(brand)}\b", lower):
                add("brand", brand, 0.92)
                add("entity", brand, 0.80)
        for tool in TOOLS:
            if re.search(rf"\b{re.escape(tool)}\b", lower):
                add("tool", tool, 0.84)
        for color in COLORS:
            if re.search(rf"\b{re.escape(color)}\b", lower):
                add("preference", f"color:{color}", 0.74, color)
        for tone in TONES:
            if re.search(rf"\b{re.escape(tone)}\b", lower):
                add("preference", f"tone:{tone}", 0.72, tone)
        if any(term in lower for term in ("current", "accepted", "now", "latest")):
            add("time_scope", "current", 0.78)
        if any(term in lower for term in ("older", "old", "stale", "previous", "superseded")):
            add("time_scope", "stale", 0.78)
        if any(term in lower for term in ("correction", "corrected", "superseded", "override")):
            add("correction", "has_correction_signal", 0.80)
        return stable_predictions(predictions)


class QwenTeacherFieldClassifier:
    def __init__(
        self,
        cache_path: str | Path,
        *,
        prompt_version: str = PROMPT_VERSION,
        allow_rule_fallback: bool = False,
    ):
        self.cache = FieldPredictionCache(cache_path)
        self.prompt_version = prompt_version
        self.allow_rule_fallback = bool(allow_rule_fallback)
        self.rules = RuleFieldClassifier()

    def classify(self, text: str, registry: FieldRegistry | None = None) -> list[FieldPrediction]:
        registry = registry or default_field_registry()
        key = prediction_cache_key(text, registry, self.prompt_version)
        cached = self.cache.get(key)
        if cached is not None:
            return parse_prediction_json(cached, source_type="qwen_cache", teacher_version=self.prompt_version)
        if not self.allow_rule_fallback:
            return []
        predictions = [
            FieldPrediction(
                field_name=item.field_name,
                value=item.value,
                confidence=item.confidence,
                source_span=item.source_span,
                source_type="qwen_cache_fallback_rules",
                teacher_version=self.prompt_version,
            )
            for item in self.rules.classify(text, registry)
        ]
        self.cache.put(key, predictions)
        return stable_predictions(predictions)

    def save(self) -> None:
        self.cache.save()


def apply_derived_metadata(
    item: dict[str, Any],
    predictions: list[FieldPrediction],
    registry: FieldRegistry,
) -> None:
    promoted = registry.promoted_names()
    fields: dict[str, list[dict[str, Any]]] = {}
    for prediction in stable_predictions(predictions):
        resolved = registry.resolve_name(prediction.field_name)
        if resolved not in promoted:
            continue
        bucket = fields.setdefault(resolved, [])
        bucket.append(
            {
                "value": prediction.value,
                "confidence": round(float(prediction.confidence), 8),
                "source_span": prediction.source_span,
                "source_type": prediction.source_type,
                "teacher_version": prediction.teacher_version,
            }
        )
    item["derived_metadata"] = {"schema_version": registry.version, "fields": dict(sorted(fields.items()))}


def classify_items(
    items: Iterable[dict[str, Any]],
    classifier: Any,
    registry: FieldRegistry,
    *,
    text_key: str = "text",
) -> None:
    all_predictions: list[FieldPrediction] = []
    for item in items:
        predictions = classifier.classify(str(item.get(text_key) or ""), registry)
        registry.record_predictions(predictions)
        apply_derived_metadata(item, predictions, registry)
        all_predictions.extend(predictions)
    if hasattr(classifier, "save"):
        classifier.save()
