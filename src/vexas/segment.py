"""Split multi-patient papers into per-patient units.

Extraction runs per unit, never per paper. A prompt that sees two patients
and is asked to keep them straight will occasionally merge them, and the
resulting record looks entirely plausible -- there is no downstream check
that catches it. Splitting first makes the failure impossible rather than
unlikely.
"""

from dataclasses import dataclass

from vexas.fetch.pmc import Article


@dataclass
class PatientUnit:
    patient_id: str
    pmid: str
    text: str


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
