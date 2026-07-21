"""Launch the UI from cached pipeline output."""

from vexas.aggregate.odyssey import rank_misdiagnoses
from vexas.aggregate.sequence import rank_precedence
from vexas.aggregate.treatment import rank_by_line
from vexas.pipeline import run
from vexas.web.app import create_app

if __name__ == "__main__":
    result = run()
    print(f"{len(result.records)} records, "
          f"{result.full_text_count} of {result.corpus_size} with full text")
    patterns = {
        "sequence": rank_precedence(result.records, min_support=3),
        "treatment": rank_by_line(result.records, min_support=3),
        "odyssey": rank_misdiagnoses(result.records, min_support=3),
    }
    create_app(patterns=patterns, records=result.records).run(
        debug=True, port=5001
    )
