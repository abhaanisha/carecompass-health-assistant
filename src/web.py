"""The web retrieval tier: trusted public-health sources, fetched at answer time.

The curated corpus in ``data/knowledge`` is deliberately small. It covers the
eleven areas it claims to cover and nothing else, which is honest but leaves a
real gap: a question about shingles, thyroid or a drug interaction falls
straight through to "I could not find anything in my knowledge base".

This module closes that gap without giving up the property the rest of the
system is built on — **every factual sentence traces to a passage the reader
can open**. Web text enters the pipeline the same way corpus text does: as
numbered, citable passages with a URL and a retrieval timestamp. It is tagged
``[W1]`` rather than ``[S1]`` so a reader can always tell curated content from
content fetched a second ago.

Three constraints make this safe enough to ship in a health assistant:

* **An allow-list, not a search of the open web.** Only the domains in
  :data:`ALLOWED_DOMAINS` are ever read. A blog, a forum, a supplement seller
  or an SEO farm cannot become a source, whatever a search engine returns.
* **It never touches the safety-critical path.** Emergency and self-harm
  answers are fixed text and are produced before this module is consulted, so
  no network call can sit between a user and an ambulance number. See
  ``pipeline.CareCompass.answer``.
* **It cannot lower urgency.** The triage floor is computed from the user's own
  message by the rule engine, before retrieval of any kind. Nothing fetched
  here can talk it down.

The default tier needs no API key at all: MedlinePlus (US National Library of
Medicine) publishes a keyless search service that returns full topic summaries
as structured XML, so the common case involves no HTML scraping and no search
vendor. A search-vendor key, if one is present, widens the reach to the rest of
the allow-list; its absence costs coverage, never correctness.

Every failure mode here — no network, a slow host, a malformed payload, a
domain that redirected off the allow-list — returns an empty result with a
reason attached. The pipeline then answers from the corpus exactly as it did
before this file existed.
"""

from __future__ import annotations

import html
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .config import Settings, get_settings

# --------------------------------------------------------------------------
# the allow-list
# --------------------------------------------------------------------------

#: Domain -> the label shown to the reader in the sources panel.
#:
#: Government health services, WHO, and the two national libraries. India-first
#: where an Indian source exists, because dengue seasonality, TB protocol and
#: ambulance numbers are not the same everywhere and a US-centric answer to an
#: Indian question is a subtler failure than no answer at all.
ALLOWED_DOMAINS: dict[str, str] = {
    "www.who.int": "WHO",
    "who.int": "WHO",
    "www.mohfw.gov.in": "MoHFW, Government of India",
    "mohfw.gov.in": "MoHFW, Government of India",
    "www.nhp.gov.in": "National Health Portal of India",
    "nhp.gov.in": "National Health Portal of India",
    "www.icmr.gov.in": "ICMR",
    "icmr.gov.in": "ICMR",
    "ncvbdc.mohfw.gov.in": "NCVBDC, MoHFW",
    "www.nhs.uk": "NHS",
    "nhs.uk": "NHS",
    "www.cdc.gov": "CDC",
    "cdc.gov": "CDC",
    "medlineplus.gov": "MedlinePlus, US NLM",
    "www.medlineplus.gov": "MedlinePlus, US NLM",
    "www.ncbi.nlm.nih.gov": "NCBI Bookshelf",
    "www.nih.gov": "NIH",
    "nih.gov": "NIH",
    "www.unicef.org": "UNICEF",
    "www.healthdirect.gov.au": "Healthdirect Australia",
    "www.betterhealth.vic.gov.au": "Better Health Channel",
}

def _http():
    """Import ``requests`` at call time, not at module import.

    The same trick ``llm.py`` uses, and for the same reason: the browser build
    bundles every module under ``src/`` and runs it in Pyodide, where
    ``requests`` does not exist. A top-level import here would take the whole
    Space down on load rather than costing it one optional feature. The
    ImportError instead surfaces as an ordinary tier failure, and the corpus
    answers alone.
    """
    import requests

    return requests


MEDLINEPLUS_ENDPOINT = "https://wsearch.nlm.nih.gov/ws/query"

USER_AGENT = "CareCompass/1.1 (health information assistant; research demo)"

#: A fetched page is cut to this many characters before chunking. Whole NHS
#: condition pages run to tens of thousands of characters, almost all of it
#: navigation and "related conditions" tails.
MAX_PAGE_CHARS = 20_000

#: One passage. Long enough to carry a complete clinical idea, short enough
#: that five of them still leave the model room to answer.
PASSAGE_CHARS = 1_100

#: Below this the web tier reports itself ungrounded and the pipeline says so
#: rather than stretching a loosely-related page to fit.
MIN_WEB_SCORE = 0.18

_CACHE_TTL_S = 900


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WebPassage:
    """One citable passage fetched from an allow-listed source."""

    title: str
    heading: str
    text: str
    site: str
    url: str
    retrieved_at: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.title} — {self.site}"


@dataclass
class WebResult:
    query: str
    passages: list[WebPassage] = field(default_factory=list)
    #: Which tier produced these: ``medlineplus``, ``tavily``, ``brave``,
    #: ``serper``, or ``none``.
    provider: str = "none"
    grounded: bool = False
    latency_ms: int = 0
    #: Why the tier returned nothing, when it returned nothing. Surfaced in the
    #: decision trace so a silent web miss is never mistaken for a decision not
    #: to look.
    error: str | None = None
    #: False when the tier was switched off or not reached at all.
    attempted: bool = False

    @property
    def urls(self) -> list[str]:
        return list(dict.fromkeys(p.url for p in self.passages))


# --------------------------------------------------------------------------
# HTML and text handling
# --------------------------------------------------------------------------

_SCRIPT_RE = re.compile(
    r"<(script|style|noscript|svg|nav|footer|header|form|aside)\b.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_RE = re.compile(
    r"</(p|div|section|article|li|ul|ol|h1|h2|h3|h4|h5|h6|tr|br)\s*>|<br\s*/?>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def html_to_text(raw: str) -> str:
    """A small, dependency-free HTML-to-text pass.

    BeautifulSoup would do this better, but it is a dependency added for forty
    lines of regex in a project whose whole deployment story is fitting inside
    a free 512 MB instance. The same trade was made for BM25 and for ``.env``
    loading, and it is made consciously: this extractor is good enough for
    prose pages on the allow-list and would not be good enough for the open
    web, which is not somewhere this module is allowed to go.
    """
    text = _COMMENT_RE.sub(" ", raw or "")
    text = _SCRIPT_RE.sub(" ", text)
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANKS_RE.sub("\n\n", text).strip()


def _strip_markup(raw: str) -> str:
    """MedlinePlus wraps query-term hits in ``<span class="qt0">`` and escapes
    the summary as HTML inside XML, so it needs the same treatment as a page."""
    return html_to_text(html.unescape(raw or ""))


def site_for(url: str) -> str | None:
    """The reader-facing label for a URL, or ``None`` if it is off the list.

    Checked after every redirect as well as before the request, because a
    government short-link that bounces to a third-party CDN is exactly the case
    an allow-list exists to catch.
    """
    try:
        host = urllib.parse.urlparse(url).netloc.lower().split(":")[0]
    except ValueError:
        return None
    return ALLOWED_DOMAINS.get(host)


# --------------------------------------------------------------------------
# passage construction
# --------------------------------------------------------------------------

_HEADING_HINT_RE = re.compile(r"^(what|when|how|why|who|symptoms?|causes?|treatment|"
                              r"prevention|diagnosis|risk|warning|see a|call)\b",
                              re.IGNORECASE)


def _split_passages(text: str, limit: int = PASSAGE_CHARS) -> list[tuple[str, str]]:
    """Split page text into ``(heading, passage)`` pairs on paragraph bounds.

    The heading is a genuine signal, not decoration: MedlinePlus and NHS pages
    are written as question headings ("What are the symptoms of dengue?"), and
    carrying that into the citation is what lets a reader verify a claim
    without reading the whole page.
    """
    def as_heading(para: str) -> str:
        # MedlinePlus escapes its summaries as HTML inside XML, so a question
        # heading and the paragraph under it sometimes arrive already joined.
        # Keep the question, drop the body that ran into it.
        first, question, _ = para.partition("?")
        if question:
            return f"{first.strip()}?"
        return para.rstrip(":").strip()

    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 2]
    passages: list[tuple[str, str]] = []
    buffer: list[str] = []
    heading = "Overview"
    length = 0

    for para in paragraphs:
        is_heading = len(para) < 110 and (
            para.endswith("?") or _HEADING_HINT_RE.match(para)
        )
        if is_heading and buffer and length > 200:
            passages.append((heading, " ".join(buffer)))
            buffer, length = [], 0
            heading = as_heading(para)
            continue
        if is_heading and not buffer:
            heading = as_heading(para)
            continue

        buffer.append(para)
        length += len(para)
        if length >= limit:
            passages.append((heading, " ".join(buffer)))
            buffer, length = [], 0

    if buffer and length > 80:
        passages.append((heading, " ".join(buffer)))
    return passages


def _overlap_score(query: str, text: str) -> float:
    """Share of the query's content words present in the passage.

    Intentionally cruder than the BM25 machinery used on the corpus. There is
    no corpus-wide IDF to compute against for a page fetched moments ago, and
    this score is used for one decision only: whether a fetched passage is
    related enough to show at all.
    """
    from .retrieval import stem, tokenize

    q = {t for t in tokenize(query) if len(t) > 2}
    if not q:
        return 0.0
    body = {stem(t) for t in tokenize(text)}
    return round(len(q & body) / len(q), 3)


# --------------------------------------------------------------------------
# the retriever
# --------------------------------------------------------------------------


class WebRetriever:
    """Fetches supporting passages from the allow-listed sources.

    Stateless apart from a short-lived in-process cache, so it is safe to share
    across Streamlit reruns and across sessions.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._cache: dict[str, tuple[float, WebResult]] = {}

    # -- introspection -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.settings.web_mode != "off"

    @property
    def search_vendor(self) -> tuple[str, str] | None:
        """``(vendor, api_key)`` for the optional wide-search tier, if keyed."""
        import os

        for vendor, env_var in (
            ("tavily", "TAVILY_API_KEY"),
            ("brave", "BRAVE_API_KEY"),
            ("serper", "SERPER_API_KEY"),
        ):
            key = os.getenv(env_var, "").strip()
            if key:
                return vendor, key
        return None

    def describe(self) -> dict:
        vendor = self.search_vendor
        return {
            "enabled": self.enabled,
            "mode": self.settings.web_mode,
            "tiers": ["medlineplus"] + ([vendor[0]] if vendor else []),
            "domains": len(set(ALLOWED_DOMAINS.values())),
        }

    # -- main entry point --------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> WebResult:
        """Return allow-listed passages for ``query``. Never raises."""
        top_k = top_k or self.settings.web_top_k
        query = (query or "").strip()
        if not self.enabled:
            return WebResult(query=query, error="web tier disabled")
        if len(query) < 3:
            return WebResult(query=query, attempted=True, error="query too short")

        cache_key = f"{query.lower()}::{top_k}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < _CACHE_TTL_S:
            return cached[1]

        started = time.perf_counter()
        passages: list[WebPassage] = []
        provider = "none"
        error: str | None = None

        try:
            passages = self._medlineplus(query, top_k)
            if passages:
                provider = "medlineplus"
        except Exception as exc:  # network, DNS, malformed XML, anything
            error = f"medlineplus: {type(exc).__name__}"

        # Widen only if the keyless tier came back thin, so the common case
        # costs one request and a vendor quota is not spent on questions that
        # were already answered.
        if len(passages) < top_k:
            vendor = self.search_vendor
            if vendor:
                try:
                    extra = self._vendor_search(vendor, query, top_k - len(passages))
                    if extra:
                        passages.extend(extra)
                        provider = vendor[0] if provider == "none" else f"{provider}+{vendor[0]}"
                except Exception as exc:
                    error = f"{error + '; ' if error else ''}{vendor[0]}: {type(exc).__name__}"

        passages.sort(key=lambda p: p.score, reverse=True)
        passages = passages[:top_k]

        result = WebResult(
            query=query,
            passages=passages,
            provider=provider,
            grounded=bool(passages) and passages[0].score >= MIN_WEB_SCORE,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=error or (None if passages else "no allow-listed source matched"),
            attempted=True,
        )
        self._cache[cache_key] = (time.time(), result)
        return result

    # -- tier A: MedlinePlus, keyless --------------------------------------

    def _medlineplus(self, query: str, top_k: int) -> list[WebPassage]:
        """Search the NLM health-topics service and read its summaries.

        The service returns the full topic summary inline, so this tier makes
        exactly one HTTP request and never scrapes a page. That is why it is
        the default: it is the cheapest, most reliable and most citable of the
        three, and it works on a fresh clone with no keys configured.
        """
        response = _http().get(
            MEDLINEPLUS_ENDPOINT,
            params={
                "db": "healthTopics",
                "term": query,
                "retmax": str(max(3, top_k)),
                "rettype": "brief",
            },
            headers={"User-Agent": USER_AGENT, "Accept": "application/xml"},
            timeout=self.settings.web_timeout_s,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)

        stamp = time.strftime("%Y-%m-%d")
        passages: list[WebPassage] = []

        for document in root.iter("document"):
            url = (document.get("url") or "").strip()
            site = site_for(url)
            if not site:
                continue

            fields: dict[str, list[str]] = {}
            for content in document.findall("content"):
                fields.setdefault(content.get("name") or "", []).append(
                    "".join(content.itertext())
                )

            title = _strip_markup((fields.get("title") or [""])[0]) or "MedlinePlus topic"
            summary = _strip_markup((fields.get("FullSummary") or [""])[0])
            if len(summary) < 160:
                continue

            for heading, text in _split_passages(summary)[:2]:
                passages.append(
                    WebPassage(
                        title=title,
                        heading=heading,
                        text=text[:PASSAGE_CHARS],
                        site=site,
                        url=url,
                        retrieved_at=stamp,
                        score=_overlap_score(query, f"{title} {heading} {text}"),
                    )
                )
            if len(passages) >= top_k * 2:
                break

        return passages

    # -- tier B: an optional search vendor, restricted to the allow-list ----

    def _vendor_search(
        self, vendor: tuple[str, str], query: str, want: int
    ) -> list[WebPassage]:
        name, key = vendor
        domains = sorted({d for d in ALLOWED_DOMAINS if not d.startswith("www.")})
        timeout = self.settings.web_timeout_s

        if name == "tavily":
            payload = {
                "api_key": key,
                "query": query,
                "max_results": max(4, want * 2),
                "include_domains": domains,
                "include_raw_content": True,
                "search_depth": "basic",
            }
            data = _http().post(
                "https://api.tavily.com/search", json=payload, timeout=timeout
            ).json()
            raw = [
                (r.get("title", ""), r.get("url", ""),
                 r.get("raw_content") or r.get("content", ""))
                for r in (data.get("results") or [])
            ]
        elif name == "brave":
            data = _http().get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max(6, want * 3)},
                headers={"X-Subscription-Token": key, "Accept": "application/json"},
                timeout=timeout,
            ).json()
            raw = [
                (r.get("title", ""), r.get("url", ""), r.get("description", ""))
                for r in ((data.get("web") or {}).get("results") or [])
            ]
        else:  # serper
            data = _http().post(
                "https://google.serper.dev/search",
                json={"q": query, "num": max(6, want * 3)},
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                timeout=timeout,
            ).json()
            raw = [
                (r.get("title", ""), r.get("link", ""), r.get("snippet", ""))
                for r in (data.get("organic") or [])
            ]

        stamp = time.strftime("%Y-%m-%d")
        passages: list[WebPassage] = []
        for title, url, body in raw:
            site = site_for(url)
            if not site:
                # Brave and Serper cannot be told to restrict by domain the way
                # Tavily can, so the allow-list is enforced here instead.
                continue
            # A snippet is too thin to cite honestly; fetch the page when the
            # vendor did not hand back the body.
            if len(body) < 400:
                body = self._fetch(url) or body
            if len(body) < 200:
                continue
            for heading, text in _split_passages(body)[:2]:
                passages.append(
                    WebPassage(
                        title=html.unescape(title) or site,
                        heading=heading,
                        text=text[:PASSAGE_CHARS],
                        site=site,
                        url=url,
                        retrieved_at=stamp,
                        score=_overlap_score(query, f"{title} {heading} {text}"),
                    )
                )
            if len(passages) >= want * 2:
                break
        return passages

    # -- page fetch --------------------------------------------------------

    def _fetch(self, url: str) -> str:
        """Fetch and flatten one allow-listed page. Returns "" on any problem."""
        if not site_for(url):
            return ""
        try:
            response = _http().get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                timeout=self.settings.web_timeout_s,
                allow_redirects=True,
            )
            response.raise_for_status()
        except Exception:
            return ""

        # Re-check after redirects: where the response actually came from is
        # what matters, not where it was requested from.
        if not site_for(response.url):
            return ""
        if "html" not in response.headers.get("Content-Type", "").lower():
            return ""
        return html_to_text(response.text[:MAX_PAGE_CHARS * 4])[:MAX_PAGE_CHARS]


__all__ = [
    "ALLOWED_DOMAINS",
    "WebPassage",
    "WebResult",
    "WebRetriever",
    "html_to_text",
    "site_for",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import sys

    retriever = WebRetriever()
    print(retriever.describe())
    for question in sys.argv[1:] or ["dengue warning signs", "shingles treatment"]:
        result = retriever.search(question)
        print(f"\nQ: {question}  [{result.provider}, grounded={result.grounded}, "
              f"{result.latency_ms} ms, error={result.error}]")
        for passage in result.passages:
            print(f"  · {passage.title} — {passage.heading} "
                  f"({passage.site}, score {passage.score})")
            print(f"    {passage.text[:140]}…")
