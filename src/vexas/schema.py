"""Extraction schema.

Two design decisions here are load-bearing and must not be simplified away:

1. Three orthogonal assertion axes (certainty, temporality, experiencer)
   rather than a single presence flag. Fine-tuned assertion models beat
   general LLMs by +23.4% on hypotheticals and +8.4% on absent findings
   (arXiv:2503.17425). "We considered lymphoma but ruled it out" recorded
   as "patient had lymphoma" is the failure most likely to discredit the
   tool in front of a clinician.

2. `not_stated` is never `absent`. A missing field means the paper did not
   say. `certainty=NEGATED` means the paper said no. Authors under-report
   common features, so missing-ness carries no information about true
   absence. Collapsing these manufactures false negatives.
"""

from enum import Enum

from pydantic import BaseModel, Field

from vexas.gate import is_grounded

#: Explicit marker for "the source did not state this". Distinct from a
#: negated finding. Prefer leaving the field as None; use this only where a
#: string is structurally required.
NOT_STATED = "not_stated"


class Certainty(str, Enum):
    POSITIVE = "positive"
    NEGATED = "negated"
    POSSIBLE = "possible"
    HYPOTHETICAL = "hypothetical"


class Temporality(str, Enum):
    PRESENT = "present"
    PAST_HISTORY = "past_history"


class Experiencer(str, Enum):
    PATIENT = "patient"
    FAMILY_MEMBER = "family_member"


class GroundedField(BaseModel):
    """A single extracted value with its verbatim source evidence."""

    value: str | None = None
    unit_as_written: str | None = None
    quote: str = Field(..., min_length=1)
    section: str | None = None
    certainty: Certainty = Certainty.POSITIVE
    temporality: Temporality = Temporality.PRESENT
    experiencer: Experiencer = Experiencer.PATIENT

    def is_grounded_in(self, source: str) -> bool:
        return is_grounded(self.quote, source)


class TimelineEvent(BaseModel):
    """One symptom with its position in the onset sequence.

    `sequence_index` is None when the paper describes the symptom but does
    not place it relative to others. Many case reports narrate loosely
    ("months of fevers, then a rash"), so unordered must be representable.
    """

    symptom: GroundedField
    sequence_index: int | None = None
    interval_value: float | None = None
    interval_unit: str | None = None


class LabValue(BaseModel):
    analyte: GroundedField
    #: Never converted by the model. See normalize/units.py.
    value_as_written: str | None = None
    unit_as_written: str | None = None


class TreatmentCourse(BaseModel):
    """One treatment, with the raw components of response.

    No consensus response criteria exist in VEXAS -- every paper defines
    "complete response" ad hoc. So we record the components and apply one
    consistent definition in the aggregation layer. The author's own label
    is retained separately and treated as low-trust.
    """

    treatment: GroundedField
    line_of_therapy: int | None = None
    dose_as_written: str | None = None
    steroid_dose_before: GroundedField | None = None
    steroid_dose_after: GroundedField | None = None
    crp_before: GroundedField | None = None
    crp_after: GroundedField | None = None
    hemoglobin_after: GroundedField | None = None
    platelets_after: GroundedField | None = None
    symptom_resolution: GroundedField | None = None
    #: Required-with-null. Response claims are meaningless without it.
    follow_up_duration: GroundedField | None = None
    #: Low-trust. The paper's own verdict, kept for comparison only.
    author_response_label: GroundedField | None = None


class PatientRecord(BaseModel):
    """One patient. Case series yield several of these per PMID."""

    pmid: str
    patient_id: str
    demographics: dict[str, GroundedField] = Field(default_factory=dict)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    labs: list[LabValue] = Field(default_factory=list)
    treatments: list[TreatmentCourse] = Field(default_factory=list)
    prior_diagnoses: list[GroundedField] = Field(default_factory=list)
    outcome: GroundedField | None = None
    #: Publication bias is the primary confounder. Encoding it per case
    #: makes the bias inspectable rather than invisible.
    reason_for_publication: GroundedField | None = None
