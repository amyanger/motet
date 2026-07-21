"""PMC full-text retrieval and section parsing.

JATS XML wraps inline markup (<italic>, <sup>, <xref>) inside paragraph
text. Naive `.text` extraction silently truncates at the first child tag,
which would produce quotes the gate then rejects. `itertext()` is used
throughout to flatten inline markup correctly.

Parsing uses defusedxml rather than the stdlib parser. JATS documents carry
DTD references, and stdlib ElementTree is vulnerable to entity-expansion
attacks (billion laughs, quadratic blowup). This XML comes from NCBI over
HTTPS so the risk is low, but the swap is one import and removes the
question entirely.
"""

import time
from dataclasses import dataclass, field
# defusedxml exposes the safe parser functions but not the Element type,
# which it returns unchanged from the stdlib. Import parsing from
# defusedxml, the type from stdlib. Never call stdlib fromstring/parse.
from xml.etree.ElementTree import Element

import httpx
from defusedxml import ElementTree as ET

from vexas.cache import DiskCache
from vexas.config import settings
from vexas.fetch.pubmed import build_eutils_url

_cache = DiskCache("pmc_xml")


@dataclass
class Article:
    pmc_id: str
    pmid: str | None
    title: str
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        parts = [f"{name}\n{body}" for name, body in self.sections.items()]
        return "\n\n".join(parts)


def _flatten(element: Element) -> str:
    """Collect all descendant text, flattening inline markup."""
    return " ".join(t.strip() for t in element.itertext() if t.strip())


def parse_sections(xml: str) -> dict[str, str]:
    """Map section title to flattened section text.

    Untitled sections are keyed 'Body'. Repeated titles are suffixed so
    later sections never silently overwrite earlier ones.
    """
    root = ET.fromstring(xml)
    sections: dict[str, str] = {}
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        title = _flatten(title_el) if title_el is not None else "Body"
        paragraphs = [_flatten(p) for p in sec.findall("p")]
        text = "\n".join(p for p in paragraphs if p)
        if not text:
            continue
        key = title
        suffix = 2
        while key in sections:
            key = f"{title} ({suffix})"
            suffix += 1
        sections[key] = text
    return sections


def _article_id(root: Element, id_type: str) -> str | None:
    for el in root.iter("article-id"):
        if el.get("pub-id-type") == id_type:
            return (el.text or "").strip()
    return None


def parse_article(xml: str) -> Article:
    root = ET.fromstring(xml)
    pmc_id = _article_id(root, "pmc") or ""
    if pmc_id and not pmc_id.startswith("PMC"):
        pmc_id = f"PMC{pmc_id}"
    title_el = root.find(".//article-title")
    return Article(
        pmc_id=pmc_id,
        pmid=_article_id(root, "pmid"),
        title=_flatten(title_el) if title_el is not None else "",
        sections=parse_sections(xml),
    )


def fetch_full_text(pmc_id: str) -> str:
    """Fetch raw JATS XML for one PMC article. Cached by PMC id."""
    cached = _cache.get(pmc_id)
    if cached is not None:
        return cached
    numeric = pmc_id.removeprefix("PMC")
    url = build_eutils_url("efetch.fcgi", db="pmc", id=numeric, retmode="xml")
    response = httpx.get(url, timeout=120)
    response.raise_for_status()
    time.sleep(settings.ncbi_rate_delay)
    _cache.put(pmc_id, response.text)
    return response.text
