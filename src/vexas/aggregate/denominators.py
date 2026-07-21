"""Claims discipline, enforced structurally.

Case reports have no denominator: we know how many REPORTS describe a
feature, never how many PATIENTS have it. Authors also preferentially
publish extraordinary findings and under-report ordinary ones. So any
statement shaped like "X% of patients" is indefensible regardless of how
carefully the extraction was done.

`Count` deliberately has no `.percentage` property. Anything trivial to
emit eventually gets emitted.
"""

from dataclasses import dataclass, field

STANDING_CAVEAT = (
    "Counts are of published reports, not patients. Absence in reports "
    "does not imply absence in patients: authors under-report common "
    "features and preferentially publish unusual ones."
)

FORBIDDEN_PHRASES: frozenset[str] = frozenset({
    "% of patients",
    "prevalence",
    "incidence",
    "more effective than",
    "less effective than",
    "causes",
    "efficacy",
    "cures",
    "leads to remission",
})


class OverclaimError(ValueError):
    """Raised when output text makes a claim case reports cannot support."""


def assert_no_overclaim(text: str) -> None:
    lowered = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            raise OverclaimError(
                f"output contains forbidden phrase {phrase!r}: {text!r}"
            )
    # Bare percentages are banned outright; counts carry denominators.
    import re

    if re.search(r"\d+\s*%", text):
        raise OverclaimError(f"output contains a bare percentage: {text!r}")


@dataclass(frozen=True)
class Count:
    n: int
    denominator: int
    label: str
    supporting_pmids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("denominator must be positive")
        if self.n > self.denominator:
            raise ValueError(
                f"n ({self.n}) exceeds denominator ({self.denominator})"
            )

    def as_text(self) -> str:
        return f"reported in {self.n} of {self.denominator} reports"


@dataclass
class Pattern:
    title: str
    count: Count
    caveat: str = STANDING_CAVEAT

    def summary(self) -> str:
        return f"{self.title}: {self.count.as_text()}"
