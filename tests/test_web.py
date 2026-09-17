"""The web retrieval tier and the guards around it.

No test here touches the network. The interesting behaviour is not "can it
reach MedlinePlus" — it can, and a test asserting so fails on a train — but the
policy around the call: which questions are allowed to reach the network, which
domains are allowed to come back, and what happens to the rest of the answer
when the whole tier falls over.
"""

from __future__ import annotations

import pytest

from src.config import PROVIDER_NONE, get_settings
from src.knowledge import load_chunks
from src.llm import LLMResponse
from src.pipeline import MODE_EMERGENCY, MODE_WEB, CareCompass
from src.retrieval import Retriever
from src.web import (
    ALLOWED_DOMAINS,
    WebPassage,
    WebResult,
    WebRetriever,
    html_to_text,
    site_for,
)


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


class StubWeb:
    """A ``WebRetriever`` with the network removed.

    Records every query that got as far as ``search``, which is what the guard
    tests assert on: the question is not what the tier returns, it is whether
    the tier was reached at all.
    """

    def __init__(self, passages: list[WebPassage] | None = None, enabled: bool = True):
        self.passages = passages or []
        self.enabled = enabled
        self.queries: list[str] = []

    def describe(self):
        return {"enabled": self.enabled, "mode": "fallback", "tiers": [], "domains": 0}

    def search(self, query, top_k=None):
        self.queries.append(query)
        return WebResult(
            query=query,
            passages=self.passages,
            provider="stub" if self.passages else "none",
            grounded=bool(self.passages),
            attempted=True,
            error=None if self.passages else "no allow-listed source matched",
        )


class ExplodingWeb(StubWeb):
    """The tier is reachable but broken. The answer must survive it."""

    def search(self, query, top_k=None):
        raise RuntimeError("DNS is on fire")


class StubLLM:
    def __init__(self, text: str = "Answer. [W1]", ok: bool = True):
        self.text = text
        self.ok = ok
        self.available = True
        self.calls: list[tuple[str, list]] = []

    def describe(self):
        return {"provider": "stub", "label": "Stub", "model": "stub-1", "available": True}

    def chat(self, system, messages, **kwargs):
        self.calls.append((system, messages))
        return LLMResponse(
            text=self.text, ok=self.ok, provider="stub", model="stub-1", latency_ms=1
        )


def passage(title="Shingles", heading="Overview", score=0.8) -> WebPassage:
    return WebPassage(
        title=title,
        heading=heading,
        text="Shingles is a painful rash caused by the varicella-zoster virus. " * 4,
        site="MedlinePlus, US NLM",
        url="https://medlineplus.gov/shingles.html",
        retrieved_at="2026-09-17",
        score=score,
    )


@pytest.fixture(scope="module")
def retriever():
    settings = get_settings().with_(use_dense=False, log_events=False)
    return Retriever(chunks=load_chunks(), settings=settings)


def build(retriever, web, llm=None) -> CareCompass:
    settings = get_settings().with_(
        use_dense=False, log_events=False, provider=PROVIDER_NONE
    )
    return CareCompass(settings=settings, retriever=retriever, llm=llm, web=web)


# --------------------------------------------------------------------------
# the allow-list
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://medlineplus.gov/shingles.html", "MedlinePlus, US NLM"),
        ("https://www.nhs.uk/conditions/shingles/", "NHS"),
        ("https://www.mohfw.gov.in/whatever", "MoHFW, Government of India"),
        ("https://medlineplus.gov.evil.com/shingles", None),
        ("https://example.com/health", None),
        ("https://en.wikipedia.org/wiki/Shingles", None),
        ("not a url at all", None),
    ],
)
def test_allow_list_is_matched_on_the_host_not_the_string(url, expected):
    """A lookalike domain must not pass.

    ``medlineplus.gov.evil.com`` contains an allow-listed domain as a
    substring, which is precisely how a naive check gets bypassed.
    """
    assert site_for(url) == expected


def test_allow_list_covers_an_indian_source():
    """An India-first product whose every source is American has a problem."""
    labels = set(ALLOWED_DOMAINS.values())
    assert any("India" in label or "ICMR" in label or "MoHFW" in label for label in labels)


# --------------------------------------------------------------------------
# html extraction
# --------------------------------------------------------------------------


def test_html_to_text_drops_script_and_style_content():
    raw = """
    <html><head><style>body { color: red; }</style>
    <script>var tracking = 'do not surface me';</script></head>
    <body><h2>Warning signs</h2><p>Severe belly pain.</p>
    <ul><li>Vomiting blood</li></ul></body></html>
    """
    text = html_to_text(raw)
    assert "tracking" not in text
    assert "color: red" not in text
    assert "Warning signs" in text
    assert "Vomiting blood" in text


def test_html_to_text_unescapes_entities_and_keeps_block_breaks():
    text = html_to_text("<p>Fever &gt; 38&deg;C</p><p>Call 112</p>")
    assert "Fever > 38°C" in text
    assert "\n" in text


# --------------------------------------------------------------------------
# the guards: which questions reach the network
# --------------------------------------------------------------------------


def test_emergency_never_reaches_the_web(retriever):
    """The one path where a network round trip is unacceptable."""
    web = StubWeb([passage()])
    answer = build(retriever, web, StubLLM()).answer(
        "crushing chest pain spreading to my left arm"
    )
    assert answer.mode == MODE_EMERGENCY
    assert web.queries == [], "no network call may sit between a user and 112"
    assert not answer.web.attempted


def test_a_vague_message_is_asked_about_not_searched_for(retriever):
    """"I am in pain" matches a general page about pain, and that is the trap.

    Answering it with one is worse than asking where it hurts, so the message
    never reaches the tier.
    """
    web = StubWeb([passage()])
    build(retriever, web, None).answer("i am in pain what to do?")
    assert web.queries == []


def test_a_non_health_question_yields_no_web_sources(retriever):
    """The allow-list is the discriminator, not a local keyword guard.

    No cheap local signal separates "write me a python function" from "what is
    gout": the corpus grounds neither, and neither carries a symptom word. A
    guard strict enough to block the first blocks the second, which is the
    exact class of question this tier exists for. So the query is allowed out,
    and a health-topic index behind a domain allow-list returns nothing for it.
    """
    web = StubWeb()  # returns no passages, as the real tier would here
    answer = build(retriever, web, None).answer(
        "write me a python function to sort a list"
    )
    assert not [c for c in answer.citations if c.kind == "web"]
    assert not answer.web.grounded
    assert "knowledge base" in answer.body


def test_a_well_covered_question_does_not_reach_the_web(retriever):
    """Reviewed content beats fetched content, so there is nothing to gain."""
    web = StubWeb([passage()])
    answer = build(retriever, web, StubLLM(text="Give ORS. [S1]")).answer(
        "my 3 year old has loose motions and is drinking less"
    )
    assert web.queries == []
    assert "already grounded" in (answer.web.error or "")


def test_a_question_outside_the_corpus_does_reach_the_web(retriever):
    web = StubWeb([passage()])
    build(retriever, web, StubLLM()).answer("what is shingles and how is it treated")
    assert web.queries, "an out-of-corpus question is what this tier exists for"


def test_web_mode_off_disables_the_tier(retriever):
    web = StubWeb([passage()], enabled=False)
    answer = build(retriever, web, StubLLM()).answer("what is shingles")
    assert web.queries == []
    assert not answer.web.attempted


# --------------------------------------------------------------------------
# what the answer does with what comes back
# --------------------------------------------------------------------------


def test_web_passages_become_numbered_w_citations(retriever):
    web = StubWeb([passage(), passage(title="Chickenpox")])
    answer = build(retriever, web, StubLLM(text="Shingles is a rash. [W2]")).answer(
        "what is shingles and how is it treated"
    )

    web_citations = [c for c in answer.citations if c.kind == "web"]
    assert [c.tag for c in web_citations] == ["[W1]", "[W2]"]
    assert web_citations[1].used
    assert not web_citations[0].used
    assert answer.mode == MODE_WEB


def test_a_w_tag_pointing_at_nothing_is_deleted(retriever):
    """The two namespaces are validated independently.

    One web passage was retrieved, so [W1] is real and [W4] is not. A tag that
    survives display without being checked looks checked, which is worse than
    no tag at all.
    """
    web = StubWeb([passage()])
    answer = build(retriever, web, StubLLM(text="Real. [W1] Invented. [W4]")).answer(
        "what is shingles and how is it treated"
    )
    assert "[W1]" in answer.body
    assert "[W4]" not in answer.body


def test_full_width_brackets_still_count_as_citations(retriever):
    """Some providers emit 【W1】 instead of [W1], reliably.

    Left alone, a correctly cited answer is scored as uncited and every source
    in the panel reads "retrieved" when it was actually quoted.
    """
    web = StubWeb([passage()])
    answer = build(retriever, web, StubLLM(text="Shingles is a rash.【W1】")).answer(
        "what is shingles and how is it treated"
    )
    assert any(c.used and c.kind == "web" for c in answer.citations)
    assert "[W1]" in answer.body


def test_sources_panel_keeps_curated_and_fetched_apart(retriever):
    web = StubWeb([passage()])
    answer = build(retriever, web, StubLLM(text="Both. [S1][W1]")).answer(
        "what is shingles and how is it treated"
    )
    markdown = answer.sources_markdown()
    assert "Curated knowledge base" in markdown
    assert "Fetched from public-health sources" in markdown
    assert "fetched 2026-09-17" in markdown


def test_the_prompt_tells_the_model_the_two_kinds_are_not_equal(retriever):
    llm = StubLLM()
    build(retriever, StubWeb([passage()]), llm).answer(
        "what is shingles and how is it treated"
    )
    system = llm.calls[0][0]
    assert "[W1]" in system
    assert "retrieved 2026-09-17" in system
    assert "Prefer them, always" in system


def test_a_broken_web_tier_does_not_break_the_answer(retriever):
    """Every way this tier can fail has to end with the corpus answering alone."""
    assistant = build(retriever, ExplodingWeb(), StubLLM(text="Give ORS. [S1]"))
    with pytest.raises(RuntimeError):
        # Guard the guard: the stub really does raise, so the test below is
        # exercising the pipeline's tolerance and not a silent no-op.
        assistant.web.search("anything")

    answer = assistant.answer("what is shingles and how is it treated")
    assert answer.body
    assert "web tier failed" in " ".join(answer.warnings)


def test_web_describe_is_reported_in_health():
    health = CareCompass(
        settings=get_settings().with_(use_dense=False, log_events=False),
        web=WebRetriever(),
    ).health()
    assert "web" in health
    assert "tiers" in health["web"]
