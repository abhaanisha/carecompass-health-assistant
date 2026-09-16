"""Guards on the static bundle.

The Space serves `space/index.html`, not the repository, so the failure mode is
specific and quiet: add a module to `src/`, forget to list it in
``build_space.collect()``, and the published Space dies with a
``ModuleNotFoundError`` that never appears in local testing.
"""

from __future__ import annotations

import base64
import gzip
import json
import re

import pytest

import build_space
from build_space import ROOT, collect, encode


def test_bundle_includes_every_source_module():
    bundled = set(collect())
    for path in (ROOT / "src").glob("*.py"):
        name = path.relative_to(ROOT).as_posix()
        assert name in bundled, f"{name} is missing from build_space.collect()"


def test_bundle_includes_every_knowledge_document():
    bundled = set(collect())
    for path in (ROOT / "data" / "knowledge").glob("*.md"):
        assert path.relative_to(ROOT).as_posix() in bundled


def test_bundle_round_trips():
    bundle = collect()
    restored = json.loads(gzip.decompress(base64.b64decode(encode(bundle))).decode("utf-8"))
    assert restored == bundle


def test_loader_survives_html_parsing():
    """The entrypoint sits in HTML, so a bare "<" would start a phantom tag."""
    loader = build_space.LOADER.format(payload="QUJD")
    assert not set(loader) & set("<>&")


@pytest.mark.skipif(
    not (ROOT / "space" / "index.html").exists(),
    reason="run `python build_space.py` first",
)
def test_published_bundle_is_current():
    html = (ROOT / "space" / "index.html").read_text(encoding="utf-8")
    match = re.search(r'_PAYLOAD = "([A-Za-z0-9+/=]+)"', html)
    assert match, "no payload found in space/index.html"

    published = json.loads(gzip.decompress(base64.b64decode(match.group(1))).decode("utf-8"))
    current = collect()
    stale = [name for name, text in current.items() if published.get(name) != text]
    assert not stale, (
        f"space/index.html is out of date for {stale}. Run `python build_space.py`."
    )
