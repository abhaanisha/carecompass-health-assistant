"""Bundle the app into a single self-contained `space/index.html` for a free
Hugging Face **Static** Space, running under Gradio-Lite (Pyodide).

Why a bundler at all, rather than one `<gradio-file>` tag per source file:

* The `<gradio-file>` element's content is parsed as HTML, so any `<` in Python
  source ("if predicted < expected", "-> tuple[...]") starts a phantom tag and
  silently corrupts the file. Escaping every file is possible; getting it wrong
  once is a broken Space with a confusing error.
* Nested paths in `name=` have been unreliable across Gradio-Lite versions.

So everything is gzipped, base64-encoded, and unpacked into Pyodide's virtual
filesystem by a ten-line loader whose own source contains no HTML-special
character at all. The bundle is verified after writing by decoding it back and
diffing against the originals.

    python build_space.py
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "space"
OUT_FILE = OUT_DIR / "index.html"

# Everything the browser build needs. The server-only entry point (app.py) and
# the tests are deliberately left out.
def collect() -> dict[str, str]:
    paths: list[Path] = [
        ROOT / "app_lite.py",
        *sorted((ROOT / "src").glob("*.py")),
        ROOT / "data" / "red_flags.yaml",
        ROOT / "data" / "lexicon.yaml",
        ROOT / "data" / "translations.yaml",
        *sorted((ROOT / "data" / "knowledge").glob("*.md")),
        ROOT / "eval" / "__init__.py",
        ROOT / "eval" / "run_eval.py",
        ROOT / "eval" / "goldset.yaml",
        ROOT / "eval" / "results.json",
    ]
    bundle: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Cannot bundle missing file: {path}")
        bundle[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8")
    return bundle


def encode(bundle: dict[str, str]) -> str:
    raw = json.dumps(bundle, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, 9)).decode("ascii")


# Deliberately free of "<", ">" and "&" so it survives HTML parsing verbatim.
LOADER = '''import base64, gzip, json, pathlib
_PAYLOAD = "{payload}"
for _name, _text in json.loads(gzip.decompress(base64.b64decode(_PAYLOAD)).decode("utf-8")).items():
    _path = pathlib.Path(_name)
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(_text, encoding="utf-8")
exec(pathlib.Path("app_lite.py").read_text(encoding="utf-8"), globals())
'''

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CareCompass — health triage that runs in your browser</title>
<meta name="description" content="A health information assistant whose safety-critical
decisions are made by auditable rules, not a language model. Runs entirely client-side.">
<script type="module" crossorigin src="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.css" />
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; }}
  #cc-boot {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
             max-width: 620px; margin: 14vh auto; padding: 0 24px; line-height: 1.55; }}
  #cc-boot h1 {{ font-size: 1.5rem; margin: 0 0 6px; letter-spacing: -0.02em; }}
  #cc-boot p {{ opacity: 0.75; margin: 6px 0; }}
  #cc-boot small {{ opacity: 0.55; }}
  gradio-lite:defined ~ #cc-boot {{ display: none; }}
</style>
</head>
<body>
<gradio-lite>
<gradio-requirements>
numpy
pyyaml
</gradio-requirements>
<gradio-file name="app.py" entrypoint>
{loader}</gradio-file>
</gradio-lite>

<div id="cc-boot">
  <h1>CareCompass is starting</h1>
  <p>This page has no backend. Your browser is downloading a Python runtime and
     then runs the whole assistant locally — the knowledge base, the safety
     rules and the retrieval engine.</p>
  <p>First visit takes roughly half a minute. After that it is cached.</p>
  <p><small>Nothing you type leaves this tab. General health information only —
     in an emergency, call 112.</small></p>
</div>
</body>
</html>
"""


def build() -> Path:
    bundle = collect()
    payload = encode(bundle)

    # Verify the round trip before anything is written.
    restored = json.loads(gzip.decompress(base64.b64decode(payload)).decode("utf-8"))
    assert restored == bundle, "bundle does not round-trip"

    loader = LOADER.format(payload=payload)
    for char in "<>&":
        assert char not in loader, f"loader contains HTML-significant {char!r}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(TEMPLATE.format(loader=loader), encoding="utf-8")

    raw_bytes = sum(len(v.encode("utf-8")) for v in bundle.values())
    print(f"bundled {len(bundle)} files ({raw_bytes / 1024:.0f} KB of source)")
    print(f"wrote   {OUT_FILE.relative_to(ROOT)} ({OUT_FILE.stat().st_size / 1024:.0f} KB)")
    for name in bundle:
        print(f"        {name}")
    return OUT_FILE


if __name__ == "__main__":
    build()
