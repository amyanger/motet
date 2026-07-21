"""Blind second pass.

Self-verification improves few-shot clinical IE (arXiv:2306.00024). The
mechanism is the asymmetry between generating and checking, which only
holds if the verifier is genuinely independent. So this prompt contains
the source text and ONE candidate field, and nothing else -- no extraction
prompt, no prior reasoning, no sibling fields.

Fails closed: anything unparseable or errored is treated as unsupported.
A field wrongly dropped costs recall; a field wrongly kept costs
credibility, which is the more expensive failure here.
"""

from dataclasses import dataclass

from vexas.extract.client import complete_json
from vexas.schema import GroundedField

VERIFY_INSTRUCTIONS = """\
You are checking one extracted claim against a source text.

Answer only this: does the source text support this claim exactly as
stated, including its certainty, temporality and experiencer?

Reject the claim if:
- The quote does not appear in the source.
- The certainty is wrong (e.g. marked positive when the source says the
  finding was absent, or when the source only raises it as a possibility
  or something considered and excluded).
- The temporality is wrong (current finding vs past history).
- The experiencer is wrong (the finding belongs to a relative, not the
  patient).

Return ONLY: {"supported": true or false, "reason": "<short reason>"}
"""


@dataclass
class Verdict:
    supported: bool
    reason: str


def build_verify_prompt(field_name: str, field: GroundedField,
                        source: str) -> list[dict]:
    """Build a cache-friendly content block list.

    Same prefix rule as extract/prompts.build_prompt: instructions and
    source text are stable across every verification of a given report,
    the claim varies. This is the highest-volume call in the pipeline --
    roughly 30 per report -- so the cached prefix matters most here.
    """
    return [
        {"type": "text", "text": VERIFY_INSTRUCTIONS},
        {
            "type": "text",
            "text": f"SOURCE TEXT:\n---\n{source}\n---",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"CLAIM:\n"
                f"  field: {field_name}\n"
                f"  value: {field.value}\n"
                f"  quote: {field.quote}\n"
                f"  certainty: {field.certainty.value}\n"
                f"  temporality: {field.temporality.value}\n"
                f"  experiencer: {field.experiencer.value}"
            ),
        },
    ]


def verify_field(field_name: str, field: GroundedField, source: str,
                 complete=complete_json) -> Verdict:
    prompt = build_verify_prompt(field_name, field, source)
    try:
        payload = complete(prompt, max_tokens=512)
    except Exception as exc:
        return Verdict(supported=False, reason=f"verification error: {exc}")

    supported = payload.get("supported")
    if not isinstance(supported, bool):
        return Verdict(supported=False, reason="unparseable verdict")
    return Verdict(supported=supported,
                   reason=str(payload.get("reason", "")))
