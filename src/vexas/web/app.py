"""Flask app.

Patterns and records are injected rather than computed at request time, so
the UI never triggers an API call and the app starts instantly from cached
pipeline output.
"""

from flask import Flask, abort, render_template

from vexas.aggregate.denominators import STANDING_CAVEAT
from vexas.aggregate.treatment import RESPONSE_DEFINITION


def create_app(patterns: dict, records: list) -> Flask:
    app = Flask(__name__)
    records_by_pmid = {record.pmid: record for record in records}

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            patterns=patterns,
            caveat=STANDING_CAVEAT,
            response_definition=RESPONSE_DEFINITION,
        )

    @app.route("/pattern/<kind>/<int:index>")
    def pattern_detail(kind: str, index: int):
        if kind not in patterns:
            abort(404)
        try:
            pattern = patterns[kind][index]
        except IndexError:
            abort(404)
        supporting = [
            records_by_pmid.get(pmid) for pmid in pattern.count.supporting_pmids
        ]
        return render_template(
            "pattern.html",
            pattern=pattern,
            kind=kind,
            supporting=[(pmid, records_by_pmid.get(pmid))
                        for pmid in pattern.count.supporting_pmids],
            caveat=STANDING_CAVEAT,
        )

    return app
