import threading

import pytest

from vexas.cache import DiskCache


@pytest.fixture
def cache(tmp_path):
    return DiskCache("test_ns", root=tmp_path)


def test_get_returns_none_when_missing(cache):
    assert cache.get("39012345") is None


def test_put_then_get_roundtrips(cache):
    cache.put("39012345", "<article>body</article>")
    assert cache.get("39012345") == "<article>body</article>"


def test_has_reflects_presence(cache):
    assert cache.has("39012345") is False
    cache.put("39012345", "x")
    assert cache.has("39012345") is True


def test_json_roundtrips(cache):
    payload = {"pmid": "39012345", "patients": 2}
    cache.put_json("39012345", payload)
    assert cache.get_json("39012345") == payload


def test_namespaces_are_isolated(tmp_path):
    a = DiskCache("raw", root=tmp_path)
    b = DiskCache("extracted", root=tmp_path)
    a.put("39012345", "raw text")
    assert b.get("39012345") is None


def test_unsafe_keys_are_sanitized(cache):
    cache.put("../../etc/passwd", "value")
    assert cache.get("../../etc/passwd") == "value"


def test_unicode_survives_roundtrip(cache):
    cache.put("39012345", "chondritis – M41T “classic”")
    assert cache.get("39012345") == "chondritis – M41T “classic”"


def test_concurrent_put_same_key_never_raises_or_mixes(tmp_path):
    # Regression test: put() used to derive its temp path from the key
    # alone, so concurrent writers for the same key shared one temp file.
    # The first writer's replace() would move that shared temp away, and
    # the second would hit FileNotFoundError on its own replace() (or, in
    # principle, interleave writes into a corrupt file). Each writer must
    # now get its own temp file, so every put() should complete cleanly
    # and the final content should be exactly one full payload.
    trials = 20
    writers = 4
    payload_size = 200_000

    for trial in range(trials):
        cache = DiskCache("test_ns", root=tmp_path / f"trial-{trial}")
        payloads = [chr(ord("a") + i) * payload_size for i in range(writers)]
        errors: list[BaseException] = []

        def write(value: str) -> None:
            try:
                cache.put("39012345", value)
            except BaseException as exc:  # noqa: BLE001 - captured for assertion
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(p,)) for p in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"trial {trial}: put() raised {errors!r}"

        result = cache.get("39012345")
        assert result is not None, f"trial {trial}: cache value missing after concurrent puts"
        assert result in payloads, (
            f"trial {trial}: cache value was not exactly one full payload "
            f"(len={len(result)}, mixed content suspected)"
        )
