"""UBA1 variant nomenclature normalization.

Case reports spell the same variant many ways: p.Met41Thr, p.M41T, M41T,
c.122T>C. Grouping by raw string would scatter one variant across four
buckets. This collapses protein-level spellings to a canonical short form
and labels splice variants distinctly.

cDNA-only notation returns None rather than a guess -- translating it needs
a codon table and transcript coordinates we deliberately do not carry. A
recorded gap beats a wrong genotype.
"""

import re

AMINO_ACIDS: dict[str, str] = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C",
    "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
    "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P",
    "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
}

_SINGLE_LETTER = set(AMINO_ACIDS.values())

_THREE_LETTER = re.compile(
    r"^(?:p\.)?([A-Za-z]{3})(\d+)([A-Za-z]{3})$", re.IGNORECASE
)
_ONE_LETTER = re.compile(
    r"^(?:p\.)?([A-Za-z])(\d+)([A-Za-z])$", re.IGNORECASE
)
_SPLICE = re.compile(r"^c\.\d+[-+]\d+[ACGT]>[ACGT]$", re.IGNORECASE)


def normalize_variant(raw: str) -> str | None:
    """Return canonical short form (e.g. 'M41T'), 'splice:...', or None."""
    if not raw:
        return None
    text = raw.strip()

    if _SPLICE.match(text):
        return f"splice:{text}"

    match = _THREE_LETTER.match(text)
    if match:
        ref, position, alt = match.groups()
        ref_letter = AMINO_ACIDS.get(ref.lower())
        alt_letter = AMINO_ACIDS.get(alt.lower())
        if ref_letter and alt_letter:
            return f"{ref_letter}{position}{alt_letter}"
        return None

    match = _ONE_LETTER.match(text)
    if match:
        ref, position, alt = match.groups()
        ref_letter, alt_letter = ref.upper(), alt.upper()
        if ref_letter in _SINGLE_LETTER and alt_letter in _SINGLE_LETTER:
            return f"{ref_letter}{position}{alt_letter}"

    return None
