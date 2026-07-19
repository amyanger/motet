from urllib.parse import parse_qs, urlparse

from vexas.fetch.pubmed import build_eutils_url, parse_id_list


def test_url_encodes_query_with_spaces_and_brackets():
    url = build_eutils_url("esearch.fcgi", db="pmc",
                           term='"VEXAS"[All Fields]')
    parsed = urlparse(url)
    assert parsed.netloc == "eutils.ncbi.nlm.nih.gov"
    assert " " not in url
    assert parse_qs(parsed.query)["term"] == ['"VEXAS"[All Fields]']


def test_url_includes_db_param():
    url = build_eutils_url("esearch.fcgi", db="pmc", term="x")
    assert parse_qs(urlparse(url).query)["db"] == ["pmc"]


def test_parse_id_list_extracts_ids_from_json():
    payload = {"esearchresult": {"count": "2", "idlist": ["111", "222"]}}
    assert parse_id_list(payload) == ["111", "222"]


def test_parse_id_list_handles_empty_result():
    payload = {"esearchresult": {"count": "0", "idlist": []}}
    assert parse_id_list(payload) == []
