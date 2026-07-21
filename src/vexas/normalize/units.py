"""Unit conversion in code, never by the model.

Lab and imaging values are the weakest extraction category for every model
in CaseReportBench, and LLM arithmetic on clinical values is an unforced
error. The model copies `value_as_written` and `unit_as_written`; this
module converts.

Unrecognized analytes and units return None rather than a guess. A missing
conversion is a visible gap; a wrong one is invisible.
"""

import re

#: analyte (lowercase) -> canonical unit
CANONICAL: dict[str, str] = {
    "crp": "mg/L",
    "esr": "mm/h",
    "hemoglobin": "g/dL",
    "haemoglobin": "g/dL",
    "platelets": "10^9/L",
    "ferritin": "ng/mL",
    "mcv": "fL",
}

#: (analyte, from_unit) -> multiplier to reach the canonical unit
_FACTORS: dict[tuple[str, str], float] = {
    ("crp", "mg/l"): 1.0,
    ("crp", "mg/dl"): 10.0,
    ("esr", "mm/h"): 1.0,
    ("esr", "mm/hr"): 1.0,
    ("hemoglobin", "g/dl"): 1.0,
    ("hemoglobin", "g/l"): 0.1,
    ("haemoglobin", "g/dl"): 1.0,
    ("haemoglobin", "g/l"): 0.1,
    ("platelets", "10^9/l"): 1.0,
    ("platelets", "x10^9/l"): 1.0,
    ("platelets", "/ul"): 0.001,
    ("platelets", "10^3/ul"): 1.0,
    ("ferritin", "ng/ml"): 1.0,
    ("ferritin", "ug/l"): 1.0,
    ("mcv", "fl"): 1.0,
}

_NUMBER = re.compile(r"(?<!\d)-?\d+(?:\.\d+)?")


def parse_numeric(text: str) -> float | None:
    """Extract a number from a clinical value string.

    Handles comparators ('<0.5'), thousands separators ('1,200') and
    ranges ('10-20', taken as the midpoint). Returns None when there is
    no number to find.
    """
    if not text:
        return None
    cleaned = text.replace(",", "").strip()
    numbers = _NUMBER.findall(cleaned)
    if not numbers:
        return None
    if len(numbers) >= 2 and re.search(r"\d\s*[-–]\s*\d", cleaned):
        return (float(numbers[0]) + float(numbers[1])) / 2
    return float(numbers[0])


def convert(analyte: str, value: float, unit: str) -> tuple[float, str] | None:
    """Convert to the canonical unit. None if analyte or unit is unknown."""
    key = analyte.strip().lower()
    if key not in CANONICAL:
        return None
    factor = _FACTORS.get((key, unit.strip().lower()))
    if factor is None:
        return None
    converted = round(value * factor, 6)
    # Trim float noise from decimal multipliers (0.1 * 124 -> 12.4).
    return (round(converted, 4), CANONICAL[key])
