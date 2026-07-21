import pytest

from vexas.aggregate.denominators import Count, Pattern
from vexas.web.app import create_app


@pytest.fixture
def app():
    patterns = {
        "sequence": [Pattern(
            title="fever precedes chondritis",
            count=Count(n=14, denominator=76, label="x",
                        supporting_pmids=["39012345", "39012346"]),
        )],
        "treatment": [],
        "odyssey": [],
    }
    application = create_app(patterns=patterns, records=[])
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200


def test_index_shows_pattern_with_denominator(client):
    body = client.get("/").get_data(as_text=True)
    assert "fever precedes chondritis" in body
    assert "14 of 76" in body


def test_index_never_shows_a_bare_percentage(client):
    """Claims discipline must hold in the rendered page, not just in
    the data layer."""
    import re
    body = client.get("/").get_data(as_text=True)
    # Strip the <style> block first: CSS percentages (color-mix, width:100%)
    # are not claims. This guard is about claim text in the rendered page.
    content = re.sub(r"<style.*?</style>", "", body, flags=re.DOTALL)
    assert not re.search(r"\d+\s*%", content)


def test_index_shows_the_standing_caveat(client):
    body = client.get("/").get_data(as_text=True)
    assert "absence in reports" in body.lower()


def test_pattern_detail_lists_supporting_pmids(client):
    body = client.get("/pattern/sequence/0").get_data(as_text=True)
    assert "39012345" in body
    assert "39012346" in body


def test_pattern_detail_links_to_pubmed(client):
    body = client.get("/pattern/sequence/0").get_data(as_text=True)
    assert "pubmed.ncbi.nlm.nih.gov/39012345" in body


def test_unknown_pattern_kind_returns_404(client):
    assert client.get("/pattern/nonexistent/0").status_code == 404


def test_out_of_range_pattern_index_returns_404(client):
    assert client.get("/pattern/sequence/99").status_code == 404
