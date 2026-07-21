from vexas.aggregate.sequence import (
    first_manifestation,
    ordered_pairs,
    rank_precedence,
)
from vexas.schema import Certainty, GroundedField, PatientRecord, TimelineEvent


def _event(symptom, index, certainty=Certainty.POSITIVE):
    return TimelineEvent(
        symptom=GroundedField(value=symptom, quote=f"quote for {symptom}",
                              certainty=certainty),
        sequence_index=index,
    )


def _record(pmid, events):
    return PatientRecord(pmid=pmid, patient_id=f"{pmid}-p1", timeline=events)


def test_ordered_pairs_respects_sequence_index():
    record = _record("1", [_event("fever", 1), _event("chondritis", 2)])
    assert ordered_pairs(record) == {("fever", "chondritis")}


def test_three_events_yield_all_ordered_pairs():
    record = _record("1", [_event("fever", 1), _event("chondritis", 2),
                           _event("anemia", 3)])
    assert ordered_pairs(record) == {
        ("fever", "chondritis"), ("fever", "anemia"),
        ("chondritis", "anemia"),
    }


def test_unordered_events_produce_no_pairs():
    record = _record("1", [_event("fever", None), _event("chondritis", None)])
    assert ordered_pairs(record) == set()


def test_negated_symptoms_are_excluded():
    """A symptom the paper says was absent has no place in a sequence."""
    record = _record("1", [_event("fever", 1),
                           _event("chondritis", 2, Certainty.NEGATED)])
    assert ordered_pairs(record) == set()


def test_hypothetical_symptoms_are_excluded():
    record = _record("1", [_event("fever", 1),
                           _event("lymphoma", 2, Certainty.HYPOTHETICAL)])
    assert ordered_pairs(record) == set()


def test_first_manifestation_is_lowest_index():
    record = _record("1", [_event("chondritis", 2), _event("fever", 1)])
    assert first_manifestation(record) == "fever"


def test_first_manifestation_none_when_unordered():
    record = _record("1", [_event("fever", None)])
    assert first_manifestation(record) is None


def test_rank_precedence_requires_minimum_support():
    records = [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
               for i in range(2)]
    assert rank_precedence(records, min_support=3,
                           canonicalize=lambda s: s) == []


def test_rank_precedence_surfaces_supported_pattern():
    records = [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
               for i in range(5)]
    patterns = rank_precedence(records, min_support=3,
                               canonicalize=lambda s: s)
    assert len(patterns) == 1
    assert patterns[0].count.n == 5
    assert patterns[0].count.denominator == 5
    assert len(patterns[0].count.supporting_pmids) == 5


def test_denominator_counts_only_records_with_ordering():
    """Records that state no order cannot support or refute a precedence
    claim, so they must not inflate the denominator."""
    ordered = [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
               for i in range(4)]
    unordered = [_record("99", [_event("fever", None)])]
    patterns = rank_precedence(ordered + unordered, min_support=3,
                               canonicalize=lambda s: s)
    assert patterns[0].count.denominator == 4


def test_patterns_sorted_by_support_descending():
    records = (
        [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
         for i in range(5)]
        + [_record(f"b{i}", [_event("rash", 1), _event("anemia", 2)])
           for i in range(3)]
    )
    patterns = rank_precedence(records, min_support=3, canonicalize=lambda s: s)
    assert patterns[0].count.n >= patterns[1].count.n


def test_denominator_is_per_pair_not_global():
    records = [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
               for i in range(4)]
    records += [_record(f"r{i}", [_event("rash", 1), _event("anemia", 2)])
                for i in range(3)]
    patterns = rank_precedence(records, min_support=3, canonicalize=lambda s: s)
    fc = next(p for p in patterns if p.count.label == "fever->chondritis")
    assert fc.count.n == 4
    assert fc.count.denominator == 4   # NOT 7 -- rash/anemia records do not count


def test_conflicting_orders_share_the_pair_denominator():
    records = [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
               for i in range(3)]
    records += [_record("x", [_event("chondritis", 1), _event("fever", 2)])]
    patterns = rank_precedence(records, min_support=3, canonicalize=lambda s: s)
    fc = next(p for p in patterns if p.count.label == "fever->chondritis")
    assert fc.count.n == 3
    assert fc.count.denominator == 4   # all 4 placed both symptoms


def test_symptom_surface_variants_merge_via_canonicalization(monkeypatch):
    from vexas.aggregate import sequence as seq
    from vexas.normalize.ontology import OntologyTerm

    def fake_search_hpo(phrase, limit=1):
        # "fever" and "fevers" both map to the same HPO label.
        if "fever" in phrase:
            return [OntologyTerm(term_id="HP:0001945", label="Fever",
                                 ontology="hp")]
        if "chondritis" in phrase:
            return [OntologyTerm(term_id="HP:0002145", label="Chondritis",
                                 ontology="hp")]
        return []

    monkeypatch.setattr(seq, "search_hpo", fake_search_hpo)
    records = [_record("1", [_event("fever", 1), _event("chondritis", 2)]),
               _record("2", [_event("fevers", 1), _event("chondritis", 2)]),
               _record("3", [_event("recurrent fever", 1),
                             _event("chondritis", 2)])]
    patterns = rank_precedence(records, min_support=3)   # default canonicalize
    assert len(patterns) == 1
    assert patterns[0].count.n == 3          # all three merged despite surface diffs
    assert patterns[0].title == "fever precedes chondritis"
