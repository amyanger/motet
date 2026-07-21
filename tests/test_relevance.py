from vexas.fetch.pmc import Article
from vexas.relevance import (
    RelevanceLabel,
    RelevanceVerdict,
    build_relevance_prompt,
    classify_relevance,
)

# A source with a clear VEXAS diagnosis sentence used across tests.
CASE_TEXT = (
    "Case\nA 68-year-old man presented with fevers and chondritis. "
    "Genetic testing confirmed a somatic UBA1 p.Met41Thr variant, "
    "and he was diagnosed with VEXAS syndrome."
)
MENTION_TEXT = (
    "Discussion\nThe differential for this patient with Adult-onset Still's "
    "disease included VEXAS syndrome, which was considered and excluded."
)


def _article(text):
    return Article(pmc_id="PMC1", pmid="1", title="t", sections={"S": text})


def _fake(payload):
    def complete(prompt, max_tokens=512, model=None):
        return payload
    return complete


def test_prompt_includes_source_text_and_instruction():
    blocks = build_relevance_prompt("SRC-BODY")
    joined = " ".join(b["text"] for b in blocks)
    assert "SRC-BODY" in joined
    assert "VEXAS" in joined
    # Single call: no cache breakpoint anywhere.
    assert all("cache_control" not in b for b in blocks)


def test_case_report_with_grounded_quote():
    verdict = classify_relevance(
        _article(CASE_TEXT),
        complete=_fake({"label": "case_report",
                        "quote": "he was diagnosed with VEXAS syndrome",
                        "reason": "patient diagnosed with VEXAS"}),
    )
    assert verdict.label is RelevanceLabel.CASE_REPORT
    assert verdict.quote == "he was diagnosed with VEXAS syndrome"


def test_mention_only_with_grounded_quote():
    verdict = classify_relevance(
        _article(MENTION_TEXT),
        complete=_fake({"label": "mention_only",
                        "quote": "included VEXAS syndrome, which was considered and excluded",
                        "reason": "only in the differential"}),
    )
    assert verdict.label is RelevanceLabel.MENTION_ONLY


def test_ungrounded_quote_fails_closed_to_uncertain():
    verdict = classify_relevance(
        _article(CASE_TEXT),
        complete=_fake({"label": "case_report",
                        "quote": "this sentence is not in the source",
                        "reason": "x"}),
    )
    assert verdict.label is RelevanceLabel.UNCERTAIN
    assert verdict.quote == ""


def test_unknown_label_fails_closed_to_uncertain():
    verdict = classify_relevance(
        _article(CASE_TEXT),
        complete=_fake({"label": "definitely", "quote": "", "reason": "x"}),
    )
    assert verdict.label is RelevanceLabel.UNCERTAIN


def test_exception_fails_closed_to_uncertain():
    def boom(prompt, max_tokens=512, model=None):
        raise RuntimeError("api down")

    verdict = classify_relevance(_article(CASE_TEXT), complete=boom)
    assert verdict.label is RelevanceLabel.UNCERTAIN
    assert "api down" in verdict.reason


def test_model_uncertain_passes_through_without_grounding():
    verdict = classify_relevance(
        _article(CASE_TEXT),
        complete=_fake({"label": "uncertain", "quote": "", "reason": "unclear"}),
    )
    assert verdict.label is RelevanceLabel.UNCERTAIN
    assert verdict.reason == "unclear"
