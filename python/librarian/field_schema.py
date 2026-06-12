from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


FIELD_NAME_RE = re.compile(r"[^a-z0-9_]+")


def normalize_field_name(value: str) -> str:
    value = FIELD_NAME_RE.sub("_", str(value or "").strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def normalize_field_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


@dataclass
class FieldDefinition:
    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    examples: list[str] = field(default_factory=list)
    status: str = "promoted"
    version: int = 1
    seen_count: int = 0
    confidence_sum: float = 0.0
    distinct_values: list[str] = field(default_factory=list)
    validation_lift: float = 0.0
    hard_negative_delta: float = 0.0

    def normalized(self) -> "FieldDefinition":
        values = sorted({normalize_field_value(item) for item in self.distinct_values if normalize_field_value(item)})
        return FieldDefinition(
            name=normalize_field_name(self.name),
            aliases=sorted({normalize_field_name(item) for item in self.aliases if normalize_field_name(item)}),
            description=str(self.description or ""),
            examples=sorted({normalize_field_value(item) for item in self.examples if normalize_field_value(item)}),
            status=str(self.status or "proposed"),
            version=int(self.version or 1),
            seen_count=int(self.seen_count or 0),
            confidence_sum=round(float(self.confidence_sum or 0.0), 8),
            distinct_values=values,
            validation_lift=round(float(self.validation_lift or 0.0), 8),
            hard_negative_delta=round(float(self.hard_negative_delta or 0.0), 8),
        )

    @property
    def mean_confidence(self) -> float:
        return float(self.confidence_sum) / max(1, int(self.seen_count))


@dataclass(frozen=True)
class FieldPrediction:
    field_name: str
    value: str
    confidence: float
    source_span: str = ""
    source_type: str = "unknown"
    teacher_version: str = ""

    def normalized(self) -> "FieldPrediction":
        return FieldPrediction(
            field_name=normalize_field_name(self.field_name),
            value=normalize_field_value(self.value),
            confidence=max(0.0, min(1.0, float(self.confidence))),
            source_span=normalize_field_value(self.source_span),
            source_type=normalize_field_name(self.source_type or "unknown"),
            teacher_version=normalize_field_value(self.teacher_version),
        )


@dataclass
class FieldRegistry:
    version: int = 1
    fields: dict[str, FieldDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, FieldDefinition] = {}
        for name, definition in self.fields.items():
            item = definition if isinstance(definition, FieldDefinition) else FieldDefinition(**definition)
            item.name = item.name or name
            normalized[item.normalized().name] = item.normalized()
        self.fields = dict(sorted(normalized.items()))

    def promoted_names(self) -> set[str]:
        return {name for name, item in self.fields.items() if item.status == "promoted"}

    def resolve_name(self, name: str) -> str:
        wanted = normalize_field_name(name)
        if wanted in self.fields:
            return wanted
        for field_name, definition in self.fields.items():
            if wanted in {normalize_field_name(alias) for alias in definition.aliases}:
                return field_name
        return wanted

    def record_predictions(self, predictions: list[FieldPrediction]) -> None:
        for prediction in sorted((item.normalized() for item in predictions), key=lambda item: (item.field_name, item.value)):
            if not prediction.field_name or not prediction.value:
                continue
            field_name = self.resolve_name(prediction.field_name)
            definition = self.fields.get(field_name)
            if definition is None:
                definition = FieldDefinition(name=field_name, status="proposed", version=int(self.version))
            definition.seen_count += 1
            definition.confidence_sum += float(prediction.confidence)
            values = set(definition.distinct_values)
            values.add(prediction.value)
            definition.distinct_values = sorted(values)
            self.fields[field_name] = definition.normalized()
        self.fields = dict(sorted(self.fields.items()))

    def promote_ready_fields(
        self,
        validation_lifts: dict[str, float],
        hard_negative_deltas: dict[str, float] | None = None,
        *,
        min_count: int = 25,
        min_confidence: float = 0.82,
        min_distinct_values: int = 5,
        min_lift: float = 0.03,
        max_hard_negative_delta: float = 0.01,
    ) -> list[str]:
        hard_negative_deltas = hard_negative_deltas or {}
        promoted: list[str] = []
        for field_name in sorted(self.fields):
            definition = self.fields[field_name]
            if definition.status == "promoted":
                continue
            lift = float(validation_lifts.get(field_name, 0.0))
            hard_delta = float(hard_negative_deltas.get(field_name, 0.0))
            if definition.seen_count < min_count:
                continue
            if definition.mean_confidence < min_confidence:
                continue
            if len(definition.distinct_values) < min_distinct_values:
                continue
            if lift < min_lift:
                continue
            if hard_delta > max_hard_negative_delta:
                continue
            definition.status = "promoted"
            definition.version = int(self.version) + 1
            definition.validation_lift = lift
            definition.hard_negative_delta = hard_delta
            self.fields[field_name] = definition.normalized()
            promoted.append(field_name)
        if promoted:
            self.version += 1
        return promoted

    def to_json_data(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "fields": {name: asdict(definition.normalized()) for name, definition in sorted(self.fields.items())},
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_json_data(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FieldRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {name: FieldDefinition(**definition) for name, definition in (data.get("fields") or {}).items()}
        return cls(version=int(data.get("version") or 1), fields=fields)


def default_field_registry() -> FieldRegistry:
    fields = {
        "user_id": FieldDefinition("user_id", aliases=["user", "person"], description="Stable user identifier"),
        "project": FieldDefinition("project", aliases=["workstream", "project_id"], description="Project or bounded work objective"),
        "brand": FieldDefinition("brand", aliases=["company", "client_brand"], description="Brand or customer identity"),
        "session_id": FieldDefinition("session_id", aliases=["session"], description="Conversation or event session"),
        "entity": FieldDefinition("entity", aliases=["named_entity"], description="Named entity useful for retrieval"),
        "preference": FieldDefinition("preference", aliases=["constraint", "likes"], description="Preference or constraint"),
        "correction": FieldDefinition("correction", aliases=["supersedes", "override"], description="Correction or supersession signal"),
        "tool": FieldDefinition("tool", aliases=["software", "system"], description="Tool or platform mentioned"),
        "time_scope": FieldDefinition("time_scope", aliases=["recency", "temporal_scope"], description="Temporal scope such as current or stale"),
        "topic": FieldDefinition("topic", aliases=["subject"], description="Main topic or task"),
    }
    return FieldRegistry(version=1, fields=fields)
