"""Blind second pass.

Self-verification improves few-shot clinical IE (arXiv:2306.00024). The
mechanism is the asymmetry between generating and checking, which only
holds if the verifier is genuinely independent. So this prompt contains
the source text and ONE candidate field, and nothing else -- no extraction
prompt, no prior reasoning, no sibling fields.

Fails closed: anything unparseable or errored is treated as unsupported.
A field wrongly dropped costs recall; a field wrongly kept costs
credibility, which is the more expensive failure here.
"""

from dataclasses import dataclass, field

from vexas.extract.client import complete_json
from vexas.schema import GroundedField, PatientRecord

VERIFY_INSTRUCTIONS = """\
You are checking one extracted claim against a source text.

Answer only this: does the source text support this claim exactly as
stated, including its certainty, temporality and experiencer?

Reject the claim if:
- The quote does not appear in the source.
- The certainty is wrong (e.g. marked positive when the source says the
  finding was absent, or when the source only raises it as a possibility
  or something considered and excluded).
- The temporality is wrong (current finding vs past history).
- The experiencer is wrong (the finding belongs to a relative, not the
  patient).

Return ONLY: {"supported": true or false, "reason": "<short reason>"}
"""


@dataclass
class Verdict:
    supported: bool
    reason: str


def build_verify_prompt(field_name: str, field: GroundedField,
                        source: str) -> list[dict]:
    """Build a cache-friendly content block list.

    Same prefix rule as extract/prompts.build_prompt: instructions and
    source text are stable across every verification of a given report,
    the claim varies. This is the highest-volume call in the pipeline --
    roughly 30 per report -- so the cached prefix matters most here.
    """
    return [
        {"type": "text", "text": VERIFY_INSTRUCTIONS},
        {
            "type": "text",
            "text": f"SOURCE TEXT:\n---\n{source}\n---",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"CLAIM:\n"
                f"  field: {field_name}\n"
                f"  value: {field.value}\n"
                f"  quote: {field.quote}\n"
                f"  certainty: {field.certainty.value}\n"
                f"  temporality: {field.temporality.value}\n"
                f"  experiencer: {field.experiencer.value}"
            ),
        },
    ]


def verify_field(field_name: str, field: GroundedField, source: str,
                 complete=complete_json) -> Verdict:
    prompt = build_verify_prompt(field_name, field, source)
    try:
        payload = complete(prompt, max_tokens=512)
    except Exception as exc:
        return Verdict(supported=False, reason=f"verification error: {exc}")

    supported = payload.get("supported")
    if not isinstance(supported, bool):
        return Verdict(supported=False, reason="unparseable verdict")
    return Verdict(supported=supported,
                   reason=str(payload.get("reason", "")))


@dataclass
class VerifyStats:
    verified: int = 0
    rejected: int = 0
    rejected_fields: list[str] = field(default_factory=list)

    def merge(self, other: "VerifyStats") -> None:
        self.verified += other.verified
        self.rejected += other.rejected
        self.rejected_fields.extend(other.rejected_fields)

    @property
    def rejection_rate(self) -> float:
        total = self.verified + self.rejected
        return 0.0 if total == 0 else self.rejected / total


#: Optional GroundedField components of a TreatmentCourse. The treatment
#: itself is required; these are nulled when unsupported rather than
#: dropping the whole course.
_TREATMENT_COMPONENTS = (
    "steroid_dose_before", "steroid_dose_after", "crp_before", "crp_after",
    "hemoglobin_after", "platelets_after", "symptom_resolution",
    "follow_up_duration", "author_response_label",
)


def verify_record(record: PatientRecord, source: str,
                  complete=complete_json) -> tuple[PatientRecord, VerifyStats]:
    """Re-check every field of one record against its source, blind.

    This is the fifth pipeline stage. Each surviving field from extraction
    is verified independently -- the verifier sees only the source and one
    claim, never the sibling fields or the extraction reasoning. A field
    the second pass cannot support is removed: a required field takes its
    container with it (an unsupported symptom drops the timeline event, an
    unsupported treatment drops the course), an optional component is nulled.

    Mutates and returns the record so a caller can prune in place.
    """
    stats = VerifyStats()

    def supported(name: str, gfield: GroundedField) -> bool:
        verdict = verify_field(name, gfield, source, complete=complete)
        if verdict.supported:
            stats.verified += 1
            return True
        stats.rejected += 1
        stats.rejected_fields.append(name)
        return False

    record.demographics = {
        key: value for key, value in record.demographics.items()
        if supported(f"demographics.{key}", value)
    }
    record.timeline = [
        event for event in record.timeline
        if supported("timeline.symptom", event.symptom)
    ]
    record.labs = [
        lab for lab in record.labs
        if supported("labs.analyte", lab.analyte)
    ]

    kept_treatments = []
    for course in record.treatments:
        if not supported("treatments.treatment", course.treatment):
            continue
        for attr in _TREATMENT_COMPONENTS:
            component = getattr(course, attr)
            if component is not None and not supported(
                    f"treatments.{attr}", component):
                setattr(course, attr, None)
        kept_treatments.append(course)
    record.treatments = kept_treatments

    record.prior_diagnoses = [
        diagnosis for diagnosis in record.prior_diagnoses
        if supported("prior_diagnoses", diagnosis)
    ]
    if record.outcome is not None and not supported("outcome", record.outcome):
        record.outcome = None
    if record.reason_for_publication is not None and not supported(
            "reason_for_publication", record.reason_for_publication):
        record.reason_for_publication = None

    return record, stats
