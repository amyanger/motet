"""Split multi-patient papers into per-patient units.

Extraction runs per unit, never per paper. A prompt that sees two patients
and is asked to keep them straight will occasionally merge them, and the
resulting record looks entirely plausible -- there is no downstream check
that catches it. Splitting first makes the failure impossible rather than
unlikely.
"""

from dataclasses import dataclass

from vexas.extract.client import complete_json
from vexas.fetch.pmc import Article

SEGMENT_INSTRUCTIONS = """\
You are given the full text of one published case report. It may describe a
single patient or several (a case series).

Identify how many DISTINCT patients are described. For each patient, copy a
short anchor phrase VERBATIM from the source text below that marks where that
patient's description begins -- normally the sentence that first introduces
them (e.g. "A 68-year-old man presented with..."). Copy the anchor character
for character; do not paraphrase, shorten, or correct it. Each anchor must be
unique and appear exactly once in the text.

Return ONLY: {"patients": [{"anchor": "<verbatim phrase>"}, ...]}
A single-patient report returns exactly one entry.
"""

SEGMENT_TASK = "TASK: return the patients array as specified above."


@dataclass
class PatientUnit:
    patient_id: str
    pmid: str
    text: str


def _segment_prompt(text: str) -> list[dict]:
    """Content blocks with the mandatory stable-first cache ordering."""
    return [
        {"type": "text", "text": SEGMENT_INSTRUCTIONS},
        {
            "type": "text",
            "text": f"SOURCE TEXT:\n---\n{text}\n---",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": SEGMENT_TASK},
    ]


def _patients(payload: object) -> list | None:
    if not isinstance(payload, dict):
        return None
    patients = payload.get("patients")
    return patients if isinstance(patients, list) else None


def detect_patient_count(article: Article, complete=complete_json) -> int:
    """How many distinct patients the report describes. At least one."""
    patients = _patients(complete(_segment_prompt(article.full_text),
                                  max_tokens=1024))
    return max(1, len(patients)) if patients else 1


def segment_texts(article: Article, complete=complete_json) -> list[str]:
    """Split the article's full text into one verbatim block per patient.

    The model proposes verbatim anchor phrases; splitting happens in Python
    at the raw byte offsets of those anchors, so the returned blocks
    reproduce the source exactly (their concatenation equals full_text) and
    the substring gate still works downstream.

    Fails safe: if the report is single-patient, or any anchor does not
    appear verbatim, or two anchors collide, the whole text is returned as
    one unit. A wrong split is worse than no split.
    """
    text = article.full_text
    patients = _patients(complete(_segment_prompt(text), max_tokens=1024))
    if not patients or len(patients) <= 1:
        return [text]

    offsets = []
    for patient in patients:
        anchor = patient.get("anchor") if isinstance(patient, dict) else None
        if not isinstance(anchor, str) or not anchor:
            return [text]
        index = text.find(anchor)
        if index == -1:
            return [text]
        offsets.append(index)

    if len(set(offsets)) != len(offsets):
        return [text]

    # The first block absorbs any preamble before the earliest anchor, so
    # the internal split points are the second anchor onward.
    boundaries = [0] + sorted(offsets)[1:] + [len(text)]
    return [text[boundaries[i]:boundaries[i + 1]]
            for i in range(len(offsets))]


def split_patients(article: Article, unit_texts: list[str]) -> list[PatientUnit]:
    """Wrap per-patient text blocks in units with stable ids.

    Text is preserved byte-for-byte. Any normalization here would desync
    quotes from source and cause spurious gate rejections.
    """
    identifier = article.pmid or article.pmc_id
    units: list[PatientUnit] = []
    for text in unit_texts:
        if not text.strip():
            continue
        units.append(
            PatientUnit(
                patient_id=f"{identifier}-p{len(units) + 1}",
                pmid=identifier,
                text=text,
            )
        )
    return units
