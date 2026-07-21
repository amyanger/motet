from vexas.schema import Certainty, GroundedField
from vexas.verify import Verdict, build_verify_prompt, verify_field

SOURCE = "The patient had no evidence of ear chondritis at presentation."


def _field():
    return GroundedField(value="ear chondritis",
                         quote="no evidence of ear chondritis",
                         certainty=Certainty.NEGATED)


def _flat(field_name="symptom", field=None, source=SOURCE) -> str:
    blocks = build_verify_prompt(field_name, field or _field(), source)
    return "\n".join(b["text"] for b in blocks)


def test_prompt_contains_source_and_candidate():
    prompt = _flat()
    assert SOURCE in prompt
    assert "ear chondritis" in prompt


def test_prompt_does_not_leak_extraction_reasoning():
    """Independence between passes is the mechanism. Guard it."""
    lowered = _flat().lower()
    assert "step by step" not in lowered
    assert "you previously" not in lowered
    assert "extraction prompt" not in lowered


def test_prompt_asks_about_assertion_axes():
    assert "certainty" in _flat().lower()


def test_source_precedes_the_claim_and_carries_the_breakpoint():
    """Caching: ~30 verify calls per report share this prefix."""
    blocks = build_verify_prompt("symptom", _field(), SOURCE)
    source_index = next(i for i, b in enumerate(blocks) if SOURCE in b["text"])
    claim_index = next(
        i for i, b in enumerate(blocks) if b["text"].startswith("CLAIM:")
    )
    assert source_index < claim_index
    cached = [b for b in blocks if "cache_control" in b]
    assert len(cached) == 1 and SOURCE in cached[0]["text"]


def test_cached_prefix_is_identical_across_different_claims():
    a = build_verify_prompt("symptom", _field(), SOURCE)
    b = build_verify_prompt("treatment", _field(), SOURCE)
    assert a[:2] == b[:2]


def test_supported_verdict_is_parsed():
    verdict = verify_field("symptom", _field(), SOURCE,
                           complete=lambda p, **k: {"supported": True,
                                                    "reason": "explicit"})
    assert verdict.supported is True
    assert verdict.reason == "explicit"


def test_unsupported_verdict_is_parsed():
    verdict = verify_field("symptom", _field(), SOURCE,
                           complete=lambda p, **k: {"supported": False,
                                                    "reason": "wrong certainty"})
    assert verdict.supported is False


def test_missing_supported_key_defaults_to_unsupported():
    """Fail closed: an unparseable verdict must not silently pass."""
    verdict = verify_field("symptom", _field(), SOURCE,
                           complete=lambda p, **k: {"reason": "unclear"})
    assert verdict.supported is False


def test_api_failure_defaults_to_unsupported():
    def boom(prompt, **kwargs):
        raise RuntimeError("API down")

    verdict = verify_field("symptom", _field(), SOURCE, complete=boom)
    assert verdict.supported is False
    assert "error" in verdict.reason.lower()
