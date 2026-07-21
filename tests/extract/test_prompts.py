import pytest

from vexas.extract.prompts import CATEGORIES, build_prompt


def _flat(category: str, text: str) -> str:
    """Concatenate all block text for content assertions."""
    return "\n".join(b["text"] for b in build_prompt(category, text))


def test_all_five_categories_exist():
    assert set(CATEGORIES) == {
        "demographics", "timeline", "labs", "treatments", "outcome"
    }


def test_prompt_embeds_the_source_text():
    assert "A 68-year-old man with fevers." in _flat(
        "timeline", "A 68-year-old man with fevers."
    )


def test_prompt_demands_verbatim_quotes():
    prompt = _flat("labs", "CRP 12 mg/dL").lower()
    assert "verbatim" in prompt
    assert "quote" in prompt


def test_prompt_forbids_unit_conversion():
    assert "do not convert" in _flat("labs", "CRP 12 mg/dL").lower()


def test_prompt_explains_not_stated_versus_negated():
    prompt = _flat("timeline", "text").lower()
    assert "not_stated" in prompt
    assert "negated" in prompt


def test_prompt_requires_assertion_axes():
    prompt = _flat("timeline", "text")
    for axis in ("certainty", "temporality", "experiencer"):
        assert axis in prompt


def test_unknown_category_raises():
    with pytest.raises(KeyError):
        build_prompt("nonexistent", "text")


# --- Caching structure. These guard ~$230 per corpus run. ---

def test_source_text_precedes_the_category_instruction():
    """Caching is a prefix match: stable content must come first."""
    blocks = build_prompt("timeline", "SOURCE_MARKER")
    source_index = next(
        i for i, b in enumerate(blocks) if "SOURCE_MARKER" in b["text"]
    )
    task_index = next(
        i for i, b in enumerate(blocks) if b["text"].startswith("TASK:")
    )
    assert source_index < task_index


def test_cache_breakpoint_sits_on_the_source_block():
    blocks = build_prompt("timeline", "SOURCE_MARKER")
    cached = [b for b in blocks if "cache_control" in b]
    assert len(cached) == 1
    assert "SOURCE_MARKER" in cached[0]["text"]


def test_cached_prefix_is_identical_across_categories():
    """The whole saving depends on byte-identical prefixes."""
    a = build_prompt("timeline", "same text")
    b = build_prompt("labs", "same text")
    assert a[:2] == b[:2]
    assert a[2] != b[2]
