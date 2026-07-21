"""Central configuration. Read once, imported everywhere."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    ncbi_api_key: str | None = field(
        default_factory=lambda: os.getenv("NCBI_API_KEY")
    )
    model: str = "claude-opus-4-8"
    #: Relevance triage runs on a cheaper model -- it is a 4-way label with a
    #: supporting quote, not structured extraction.
    relevance_model: str = "claude-haiku-4-5-20251001"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    data_dir: Path = PROJECT_ROOT / "data"

    # Corpus definition — see plan Global Constraints.
    pmc_query: str = (
        '"VEXAS"[All Fields] AND "open access"[filter] AND "case reports"[pt]'
    )

    @property
    def ncbi_rate_delay(self) -> float:
        """Seconds to sleep between NCBI calls (3/s without key, 10/s with)."""
        return 0.11 if self.ncbi_api_key else 0.34


settings = Settings()
