"""Relevance triage -- stage 2.5.

The corpus query returns every paper that mentions VEXAS, not only papers
that report a VEXAS patient. Extracting a paper that only names VEXAS in a
differential yields a full wrong-disease record that inflates every
denominator. This stage classifies each paper before extraction.

It fails closed to UNCERTAIN: an errored, unparseable, or ungrounded verdict
is surfaced for human review, never silently kept and never silently dropped.
A verdict that keeps or excludes a paper must carry a verbatim quote that the
substring gate confirms is in the source, the same discipline extraction uses.
"""

from dataclasses import dataclass
from enum import Enum

from vexas.config import settings
from vexas.extract.client import complete_json
from vexas.fetch.pmc import Article
from vexas.gate import is_grounded


class RelevanceLabel(str, Enum):
    CASE_REPORT = "case_report"
    MENTION_ONLY = "mention_only"
    REVIEW = "review"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class RelevanceVerdict:
    label: RelevanceLabel
    quote: str
    reason: str


RELEVANCE_INSTRUCTIONS = """\
You are triaging a biomedical paper. Decide what kind of paper this is with
respect to VEXAS syndrome, using ONLY the source text below.

Choose exactly one label:
- "case_report": the paper reports at least one patient who was diagnosed with
  VEXAS syndrome (a somatic UBA1 variant, or an explicit VEXAS diagnosis). If
  the paper both reviews the literature AND reports such a patient, it is still
  case_report.
- "mention_only": VEXAS appears only in a differential diagnosis, as background,
  or in an editorial, and no VEXAS patient is actually reported here.
- "review": a review or meta-analysis with no newly reported VEXAS patient.
- "uncertain": you cannot tell from the text.

For case_report, mention_only, and review you MUST include a "quote" copied
VERBATIM, character for character, from the source text that best supports the
label (for case_report, the sentence stating the patient's VEXAS or UBA1
diagnosis; for mention_only, the sentence showing VEXAS is only referenced). Do
not paraphrase the quote. If you cannot find a supporting verbatim quote, use
"uncertain" with an empty quote.

Return ONLY this JSON object, no commentary:
{"label": "<one of the four>", "quote": "<verbatim supporting text>", "reason": "<short reason>"}
"""


def build_relevance_prompt(text: str) -> list[dict]:
    """Two content blocks: instruction, then source text.

    No cache_control: this is a single call per paper, so there is no later
    call to read a cache write back at 0.1x. A cache write would only add its
    1.25x premium. Extraction's cache families are untouched.
    """
    return [
        {"type": "text", "text": RELEVANCE_INSTRUCTIONS},
        {"type": "text", "text": f"SOURCE TEXT:\n---\n{text}\n---"},
    ]


def classify_relevance(article: Article, complete=complete_json) -> RelevanceVerdict:
    source = article.full_text
    prompt = build_relevance_prompt(source)
    try:
        payload = complete(prompt, max_tokens=512, model=settings.relevance_model)
    except Exception as exc:
        return RelevanceVerdict(RelevanceLabel.UNCERTAIN, "", f"gate error: {exc}")

    raw_label = payload.get("label")
    quote = payload.get("quote", "")
    reason = str(payload.get("reason", ""))
    if not isinstance(quote, str):
        quote = ""

    try:
        label = RelevanceLabel(raw_label)
    except ValueError:
        return RelevanceVerdict(
            RelevanceLabel.UNCERTAIN, "", f"unknown label: {raw_label!r}"
        )

    if label is RelevanceLabel.UNCERTAIN:
        return RelevanceVerdict(label, quote, reason)

    # A decisive label must be backed by a grounded quote, or it fails closed.
    if not is_grounded(quote, source):
        return RelevanceVerdict(
            RelevanceLabel.UNCERTAIN, "", "quote not grounded in source"
        )
    return RelevanceVerdict(label, quote, reason)
