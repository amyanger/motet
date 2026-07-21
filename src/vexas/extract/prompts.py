"""One prompt per field category.

Category-specific prompting rather than one global prompt is the largest
measured accuracy lever for this task (28.4% -> ~52% token set ratio,
CaseReportBench arXiv:2505.17265). Chain-of-thought is deliberately absent:
the same benchmark found it gave little advantage here.
"""

from dataclasses import dataclass

SHARED_RULES = """\
You are extracting structured data from a single patient's case report.

Absolute rules:
1. Every field you emit MUST include a "quote" that is copied VERBATIM,
   character for character, from the source text below. Do not paraphrase,
   summarize, correct, or normalize the quote. If you cannot find verbatim
   supporting text, omit the field entirely.
2. Do NOT convert units, compute values, or infer anything not stated.
   Copy numbers and units exactly as written.
3. Do NOT emit ontology identifiers (HPO, RxNorm, LOINC, OMIM codes).
   Emit only the surface phrase as written.
4. Distinguish these three cases and never conflate them:
   - The report states the finding is present -> certainty "positive"
   - The report states the finding is ABSENT -> certainty "negated"
   - The report does not mention it -> OMIT the field (not_stated)
   Omission and negation are different facts. Never emit a negated field
   for something simply not mentioned.
5. Every finding carries three assertion axes:
   - certainty: positive | negated | possible | hypothetical
     Use "hypothetical" for conditions merely considered or ruled out
     ("we considered lymphoma", "differential included GCA").
   - temporality: present | past_history
   - experiencer: patient | family_member
     Use "family_member" for family history. Never attribute a relative's
     finding to the patient.

Return ONLY a JSON object. No commentary.
"""


@dataclass(frozen=True)
class CategorySpec:
    name: str
    instruction: str
    json_shape: str


CATEGORIES: dict[str, CategorySpec] = {
    "demographics": CategorySpec(
        name="demographics",
        instruction=(
            "Extract patient age, sex, and country of the reporting centre. "
            "Also extract the UBA1 variant exactly as written (e.g. 'p.Met41Thr', "
            "'M41T', 'c.122T>C') and the variant allele frequency if stated."
        ),
        json_shape="""{
  "fields": {
    "age": {"value": "68", "unit_as_written": "years", "quote": "...",
            "certainty": "positive", "temporality": "present",
            "experiencer": "patient"},
    "sex": {"value": "male", "quote": "...", "certainty": "positive",
            "temporality": "present", "experiencer": "patient"},
    "country": {"value": "Japan", "quote": "...", "certainty": "positive",
                "temporality": "present", "experiencer": "patient"},
    "uba1_variant": {"value": "p.Met41Thr", "quote": "...",
                     "certainty": "positive", "temporality": "present",
                     "experiencer": "patient"},
    "vaf": {"value": "62", "unit_as_written": "%", "quote": "...",
            "certainty": "positive", "temporality": "present",
            "experiencer": "patient"}
  }
}""",
    ),
    "timeline": CategorySpec(
        name="timeline",
        instruction=(
            "Extract each symptom or manifestation AND its position in the "
            "onset sequence. This is the most important category.\n"
            "- sequence_index: 1 for the first manifestation, 2 for the next, "
            "and so on. Use null when the report does not place the symptom "
            "relative to others. Do NOT guess an order that is not stated.\n"
            "- interval_value / interval_unit: time since the PREVIOUS event, "
            "as written ('six months' -> 6, 'months'). Null if not stated.\n"
            "Also extract any diagnoses the patient carried BEFORE VEXAS was "
            "confirmed (e.g. relapsing polychondritis, giant cell arteritis, "
            "Sweet syndrome, polymyalgia rheumatica, MDS) as prior_diagnoses, "
            "and the time from first symptom to UBA1 testing if stated."
        ),
        json_shape="""{
  "timeline": [
    {"symptom": {"value": "recurrent fevers", "quote": "...",
                 "certainty": "positive", "temporality": "present",
                 "experiencer": "patient"},
     "sequence_index": 1, "interval_value": null, "interval_unit": null}
  ],
  "prior_diagnoses": [
    {"value": "relapsing polychondritis", "quote": "...",
     "certainty": "positive", "temporality": "past_history",
     "experiencer": "patient"}
  ],
  "time_to_diagnosis": {"value": "3", "unit_as_written": "years",
                        "quote": "...", "certainty": "positive",
                        "temporality": "present", "experiencer": "patient"}
}""",
    ),
    "labs": CategorySpec(
        name="labs",
        instruction=(
            "Extract laboratory values. Copy the number and the unit exactly "
            "as printed. DO NOT convert between units (mg/L and mg/dL, g/dL "
            "and g/L). DO NOT compute derived values. If a value is given as "
            "a range or with a qualifier, copy it as written."
        ),
        json_shape="""{
  "labs": [
    {"analyte": {"value": "CRP", "quote": "...", "certainty": "positive",
                 "temporality": "present", "experiencer": "patient"},
     "value_as_written": "12.4", "unit_as_written": "mg/dL"}
  ]
}""",
    ),
    "treatments": CategorySpec(
        name="treatments",
        instruction=(
            "Extract every treatment given, IN THE ORDER GIVEN.\n"
            "- line_of_therapy: 1 for the first treatment tried, 2 for the "
            "next, and so on. This ordering is the point of the extraction. "
            "Use null only if the report genuinely does not indicate order.\n"
            "- Record the RAW COMPONENTS of response, not a verdict: steroid "
            "dose before and after, CRP before and after, haemoglobin and "
            "platelets after, whether symptoms resolved, and follow-up "
            "duration.\n"
            "- author_response_label: if the authors themselves call it a "
            "'complete response' or 'partial response', record that phrase. "
            "Do NOT invent a label they did not use, and do not infer one "
            "from the components."
        ),
        json_shape="""{
  "treatments": [
    {"treatment": {"value": "ruxolitinib", "quote": "...",
                   "certainty": "positive", "temporality": "present",
                   "experiencer": "patient"},
     "line_of_therapy": 2,
     "dose_as_written": "10 mg twice daily",
     "steroid_dose_before": {"value": "40", "unit_as_written": "mg/day",
                             "quote": "...", "certainty": "positive",
                             "temporality": "present", "experiencer": "patient"},
     "steroid_dose_after": null,
     "crp_before": null, "crp_after": null,
     "hemoglobin_after": null, "platelets_after": null,
     "symptom_resolution": {"value": "yes", "quote": "...",
                            "certainty": "positive", "temporality": "present",
                            "experiencer": "patient"},
     "follow_up_duration": {"value": "12", "unit_as_written": "months",
                            "quote": "...", "certainty": "positive",
                            "temporality": "present", "experiencer": "patient"},
     "author_response_label": null}
  ]
}""",
    ),
    "outcome": CategorySpec(
        name="outcome",
        instruction=(
            "Extract the patient's final reported outcome and the reason the "
            "case was published. For reason_for_publication, quote the "
            "authors' own statement of novelty if present (e.g. 'the first "
            "reported case of', 'we describe an unusual'). This records "
            "publication bias per case; omit the field if the authors make "
            "no such claim rather than inferring one."
        ),
        json_shape="""{
  "outcome": {"value": "alive at last follow-up", "quote": "...",
              "certainty": "positive", "temporality": "present",
              "experiencer": "patient"},
  "reason_for_publication": {"value": "first reported case with cardiac involvement",
                             "quote": "...", "certainty": "positive",
                             "temporality": "present", "experiencer": "patient"}
}""",
    ),
}


def build_prompt(category: str, text: str) -> list[dict]:
    """Build a cache-friendly content block list.

    ORDER IS LOAD-BEARING. Prompt caching is a prefix match, so the stable
    content must come first and the varying content last:

        SHARED_RULES  (identical for every call, ever)
        SOURCE TEXT   (identical for all ~35 calls about one report)  <-- cached
        INSTRUCTION   (varies per category)

    Each report's text is re-sent ~35 times (5 category calls + ~30
    verification calls). Cached reads cost 0.1x, so getting this ordering
    right takes a full corpus run from roughly $292 to roughly $62.

    Putting SOURCE TEXT last would make it uncacheable and silently cost
    ~$230 per run. Do not reorder these blocks.
    """
    spec = CATEGORIES[category]
    return [
        {"type": "text", "text": SHARED_RULES},
        {
            "type": "text",
            "text": f"SOURCE TEXT:\n---\n{text}\n---",
            # Breakpoint: everything above this is reused across all calls
            # about this report. Reports average ~7,700 tokens, comfortably
            # over Opus 4.8's 4,096-token minimum cacheable prefix.
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"TASK: {spec.instruction}\n\n"
                f"Return JSON in exactly this shape:\n{spec.json_shape}\n\n"
                f"Omit any field you cannot support with a verbatim quote."
            ),
        },
    ]
