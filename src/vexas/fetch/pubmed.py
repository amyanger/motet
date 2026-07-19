"""PubMed / PMC identifier search via NCBI E-utilities.

Rate limits: 3 req/s anonymous, 10 req/s with an API key. Exceeding them
gets the IP blocked, so every network call sleeps.
"""

import time
from urllib.parse import urlencode

import httpx

from vexas.config import settings

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def build_eutils_url(endpoint: str, **params) -> str:
    """Build a fully URL-encoded E-utilities URL.

    Encoding matters: PubMed query syntax contains spaces and square
    brackets, which produce an invalid URL if passed through raw.
    """
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    return f"{BASE}{endpoint}?{urlencode(params)}"


def parse_id_list(payload: dict) -> list[str]:
    return list(payload.get("esearchresult", {}).get("idlist", []))


def search_pmc_ids(query: str | None = None, retmax: int = 500) -> list[str]:
    """Return PMC IDs matching the corpus query."""
    term = query if query is not None else settings.pmc_query
    url = build_eutils_url(
        "esearch.fcgi", db="pmc", term=term, retmax=retmax, retmode="json"
    )
    response = httpx.get(url, timeout=60)
    response.raise_for_status()
    time.sleep(settings.ncbi_rate_delay)
    return parse_id_list(response.json())
