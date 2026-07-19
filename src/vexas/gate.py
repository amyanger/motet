"""The substring gate.

Every extracted field carries a verbatim quote. This module mechanically
verifies that quote is a literal substring of the source text. A field whose
quote fails is DISCARDED, never repaired.

This exists because the dominant frontier-model failure on case-report
extraction is merging details and introducing inferences rather than
extracting verbatim text (CaseReportBench, arXiv:2505.17265). A soft prompt
instruction cannot prevent that; a substring assertion can.

Keep this module dependency-free and pure.
"""

import re
import unicodedata

# Unicode punctuation that publishers substitute for ASCII equivalents.
_PUNCT_MAP = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    " ": " ",
}

_WHITESPACE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Canonicalize text for substring comparison.

    Applies NFKC normalization, maps publisher punctuation to ASCII,
    collapses whitespace runs, lowercases, and strips.

    Deliberately permissive about formatting, strict about wording: a
    paraphrase will still fail.
    """
    text = unicodedata.normalize("NFKC", text)
    for source_char, ascii_char in _PUNCT_MAP.items():
        text = text.replace(source_char, ascii_char)
    text = _WHITESPACE.sub(" ", text)
    return text.strip().lower()


def is_grounded(quote: str, source: str) -> bool:
    """True if `quote` appears verbatim in `source` after normalization.

    An empty or whitespace-only quote is never grounded — that is the
    degenerate case where a model returns a field with no evidence.
    """
    normalized_quote = normalize_for_match(quote)
    if not normalized_quote:
        return False
    return normalized_quote in normalize_for_match(source)
