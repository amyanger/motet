"""Disk cache keyed by PMID.

Every pipeline stage caches its output so a mid-run failure never re-burns
API spend and development iterations never re-fetch. Namespaces keep stage
outputs from colliding.
"""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from vexas.config import settings

_SAFE_KEY = re.compile(r"^[A-Za-z0-9._-]+$")


class DiskCache:
    def __init__(self, namespace: str, root: Path | None = None) -> None:
        base = root if root is not None else settings.cache_dir
        self.dir = Path(base) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are usually PMIDs, but never trust that — a key containing
        # path separators would otherwise escape the cache directory.
        if not _SAFE_KEY.match(key):
            key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.dir / f"{key}.cache"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def put(self, key: str, value: str) -> None:
        # Write-then-rename so an interrupted write never leaves a
        # half-written file that looks like a valid cache hit. The temp
        # file gets a per-call unique name (in the same directory, so the
        # rename stays atomic) so concurrent writers for the same key
        # never collide on a shared temp path.
        path = self._path(key)
        fd, tmp_name = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(value)
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def get_json(self, key: str) -> object | None:
        raw = self.get(key)
        return None if raw is None else json.loads(raw)

    def put_json(self, key: str, value: object) -> None:
        self.put(key, json.dumps(value, ensure_ascii=False, indent=2))
