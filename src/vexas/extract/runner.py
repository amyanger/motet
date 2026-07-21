"""Extraction orchestration: prompt per category, then gate everything.

The gate runs on raw JSON before pydantic parsing, so an ungrounded field
never becomes a typed object at all. Rejections are counted rather than
repaired -- rejection rate is a quality signal we surface in the UI.
"""

from dataclasses import dataclass, field

from vexas.extract.client import complete_json
from vexas.extract.prompts import CATEGORIES, build_prompt
from vexas.gate import is_grounded
from vexas.schema import (
    GroundedField,
    LabValue,
    PatientRecord,
    TimelineEvent,
    TreatmentCourse,
)
from vexas.segment import PatientUnit

#: Keys that identify a dict as an extracted field rather than a container.
_FIELD_MARKER = "quote"


@dataclass
class GateStats:
    kept: int = 0
    rejected: int = 0
    rejected_quotes: list[str] = field(default_factory=list)

    def merge(self, other: "GateStats") -> None:
        self.kept += other.kept
        self.rejected += other.rejected
        self.rejected_quotes.extend(other.rejected_quotes)

    @property
    def rejection_rate(self) -> float:
        total = self.kept + self.rejected
        return 0.0 if total == 0 else self.rejected / total


def _looks_like_field(value: object) -> bool:
    return isinstance(value, dict) and (
        _FIELD_MARKER in value or "value" in value
    )


def apply_gate(payload: object, source: str, stats: GateStats | None = None):
    """Recursively strip every field whose quote is not verbatim in source.

    Returns (cleaned_payload, stats). Containers are preserved; only field
    dicts are judged.
    """
    stats = stats if stats is not None else GateStats()

    if isinstance(payload, dict):
        if _looks_like_field(payload):
            quote = payload.get(_FIELD_MARKER)
            if isinstance(quote, str) and is_grounded(quote, source):
                stats.kept += 1
                return payload, stats
            stats.rejected += 1
            stats.rejected_quotes.append(quote if isinstance(quote, str) else "")
            return None, stats

        cleaned: dict = {}
        direct_field_count = 0
        rejected_field_count = 0
        for key, value in payload.items():
            is_direct_field = isinstance(value, dict) and _looks_like_field(value)
            if is_direct_field:
                direct_field_count += 1
            result, stats = apply_gate(value, source, stats)
            if is_direct_field and result is None:
                rejected_field_count += 1
            if result is not None:
                cleaned[key] = result
        # If all direct fields were rejected, return empty.
        if direct_field_count > 0 and rejected_field_count == direct_field_count:
            return {}, stats
        return cleaned, stats

    if isinstance(payload, list):
        cleaned_list = []
        for item in payload:
            result, stats = apply_gate(item, source, stats)
            # A list entry whose only field was rejected collapses to {}.
            if result is not None and result != {}:
                cleaned_list.append(result)
        return cleaned_list, stats

    return payload, stats


def _to_field(raw: object) -> GroundedField | None:
    if not isinstance(raw, dict) or "quote" not in raw:
        return None
    return GroundedField(**raw)


def extract_unit(unit: PatientUnit, complete=complete_json):
    """Run all five category prompts over one patient unit.

    `complete` is injected so tests can stub the API.
    """
    stats = GateStats()
    payloads: dict[str, dict] = {}

    for category in CATEGORIES:
        prompt = build_prompt(category, unit.text)
        raw = complete(prompt)
        cleaned, stats_for_category = apply_gate(raw, unit.text)
        stats.merge(stats_for_category)
        payloads[category] = cleaned or {}

    record = PatientRecord(pmid=unit.pmid, patient_id=unit.patient_id)

    for name, raw in payloads.get("demographics", {}).get("fields", {}).items():
        parsed = _to_field(raw)
        if parsed is not None:
            record.demographics[name] = parsed

    for raw in payloads.get("timeline", {}).get("timeline", []):
        symptom = _to_field(raw.get("symptom"))
        if symptom is None:
            continue
        record.timeline.append(
            TimelineEvent(
                symptom=symptom,
                sequence_index=raw.get("sequence_index"),
                interval_value=raw.get("interval_value"),
                interval_unit=raw.get("interval_unit"),
            )
        )

    for raw in payloads.get("timeline", {}).get("prior_diagnoses", []):
        parsed = _to_field(raw)
        if parsed is not None:
            record.prior_diagnoses.append(parsed)

    ttd = _to_field(payloads.get("timeline", {}).get("time_to_diagnosis"))
    if ttd is not None:
        record.demographics["time_to_diagnosis"] = ttd

    for raw in payloads.get("labs", {}).get("labs", []):
        analyte = _to_field(raw.get("analyte"))
        if analyte is None:
            continue
        record.labs.append(
            LabValue(
                analyte=analyte,
                value_as_written=raw.get("value_as_written"),
                unit_as_written=raw.get("unit_as_written"),
            )
        )

    for raw in payloads.get("treatments", {}).get("treatments", []):
        treatment = _to_field(raw.get("treatment"))
        if treatment is None:
            continue
        record.treatments.append(
            TreatmentCourse(
                treatment=treatment,
                line_of_therapy=raw.get("line_of_therapy"),
                dose_as_written=raw.get("dose_as_written"),
                steroid_dose_before=_to_field(raw.get("steroid_dose_before")),
                steroid_dose_after=_to_field(raw.get("steroid_dose_after")),
                crp_before=_to_field(raw.get("crp_before")),
                crp_after=_to_field(raw.get("crp_after")),
                hemoglobin_after=_to_field(raw.get("hemoglobin_after")),
                platelets_after=_to_field(raw.get("platelets_after")),
                symptom_resolution=_to_field(raw.get("symptom_resolution")),
                follow_up_duration=_to_field(raw.get("follow_up_duration")),
                author_response_label=_to_field(
                    raw.get("author_response_label")
                ),
            )
        )

    outcome_payload = payloads.get("outcome", {})
    record.outcome = _to_field(outcome_payload.get("outcome"))
    record.reason_for_publication = _to_field(
        outcome_payload.get("reason_for_publication")
    )

    return record, stats
