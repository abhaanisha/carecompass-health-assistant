"""End-to-end behaviour, including the failure modes.

The LLM is stubbed rather than called: these tests assert what the pipeline
does *with* a model response, which is the part that has to hold regardless of
which provider is configured.
"""

from __future__ import annotations

import os

import pytest

from src.config import PROVIDER_NONE, get_settings
from src.knowledge import load_chunks
from src.llm import LLMResponse
from src.pipeline import MODE_CRISIS, MODE_EMERGENCY, MODE_EXTRACTIVE, MODE_MODEL, CareCompass
from src.retrieval import Retriever
from src.triage import Urgency


class StubLLM:
    """Stands in for a provider. ``available`` and the reply are controllable."""

    def __init__(self, text: str = "Rest and drink fluids. [S1]", ok: bool = True):
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


@pytest.fixture(scope="module")
def retriever():
    settings = get_settings().with_(use_dense=False, log_events=False)
    return Retriever(chunks=load_chunks(), settings=settings)


def build(retriever, llm=None) -> CareCompass:
    settings = get_settings().with_(use_dense=False, log_events=False, provider=PROVIDER_NONE)
    return CareCompass(settings=settings, retriever=retriever, llm=llm or StubLLM())


def test_emergency_never_reaches_the_model(retriever):
    llm = StubLLM()
    answer = build(retriever, llm).answer("crushing chest pain spreading to my left arm")
    assert answer.mode == MODE_EMERGENCY
    assert answer.triage.level == Urgency.EMERGENCY
    assert "112" in answer.text
    assert llm.calls == [], "the model must not be consulted on the emergency path"


def test_self_harm_gets_the_crisis_card(retriever):
    llm = StubLLM()
    answer = build(retriever, llm).answer("I don't want to live anymore")
    assert answer.mode == MODE_CRISIS
    assert "14416" in answer.text
    assert llm.calls == []


def test_model_answer_is_used_when_available(retriever):
    answer = build(retriever, StubLLM()).answer("I have a mild sore throat")
    assert answer.mode == MODE_MODEL
    assert "Rest and drink fluids" in answer.text


def test_failed_model_call_degrades_to_extractive(retriever):
    answer = build(retriever, StubLLM(ok=False, text="")).answer("I have a mild sore throat")
    assert answer.mode == MODE_EXTRACTIVE
    assert answer.text.strip()
    assert any("model call failed" in w for w in answer.warnings)


def test_hallucinated_citations_are_removed(retriever):
    llm = StubLLM(text="Drink fluids [S1] and see a doctor [S97].")
    answer = build(retriever, llm).answer("I have a mild sore throat")
    assert "[S97]" not in answer.text
    assert "[S1]" in answer.text


def test_uncited_model_answer_is_flagged(retriever):
    llm = StubLLM(text="Just rest, it will pass.")
    answer = build(retriever, llm).answer("I have a mild sore throat")
    assert any("without citing" in w for w in answer.warnings)


def test_out_of_scope_question_is_refused(retriever):
    settings = get_settings().with_(use_dense=False, log_events=False, provider=PROVIDER_NONE)
    assistant = CareCompass(settings=settings, retriever=retriever, llm=StubLLM())
    assistant.llm.available = False
    answer = assistant.answer("write me a python function to sort a list")
    assert not answer.retrieval.grounded
    assert "could not find" in answer.text.lower()


def test_every_answer_carries_a_disclaimer(retriever):
    for message in ["I have a mild sore throat", "chest pain and sweating", "hello"]:
        assert "112" in build(retriever).answer(message).text


def test_language_is_detected_and_echoed(retriever):
    assistant = build(retriever)
    assert assistant.answer("seene me dard nahi hai lekin bukhar hai").language == "hi"
    assert assistant.answer("আমার জ্বর হয়েছে").language == "bn"
    assert assistant.answer("I have a fever").language == "en"


def test_empty_message_is_handled(retriever):
    answer = build(retriever).answer("   ")
    assert answer.text
    assert answer.triage.level == Urgency.INFO


def test_extractive_path_voices_the_refusal(retriever):
    """Without a model key the decline still has to be said out loud."""
    settings = get_settings().with_(use_dense=False, log_events=False, provider=PROVIDER_NONE)
    assistant = CareCompass(settings=settings, retriever=retriever, llm=StubLLM())
    assistant.llm.available = False
    answer = assistant.answer("which antibiotic should I take for a sore throat")
    assert answer.mode == MODE_EXTRACTIVE
    assert "can't recommend a medicine" in answer.text


def test_dotenv_fills_only_missing_variables(tmp_path, monkeypatch):
    """A real environment variable must win over a stale local .env."""
    from src.config import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text(
        '# a comment\n'
        'export CARECOMPASS_TEST_NEW="from-file"\n'
        "CARECOMPASS_TEST_EXISTING='from-file'\n"
        "MALFORMED\n"
        "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CARECOMPASS_TEST_EXISTING", "from-environment")
    monkeypatch.delenv("CARECOMPASS_TEST_NEW", raising=False)

    applied = load_dotenv(env_file)

    assert applied == ["CARECOMPASS_TEST_NEW"]
    assert os.environ["CARECOMPASS_TEST_NEW"] == "from-file"
    assert os.environ["CARECOMPASS_TEST_EXISTING"] == "from-environment"


def test_dotenv_absent_is_not_an_error(tmp_path):
    from src.config import load_dotenv

    assert load_dotenv(tmp_path / "nope.env") == []


def test_vague_symptom_asks_instead_of_refusing(retriever):
    """"i am in pain" is in scope and underspecified, not out of scope."""
    from src.pipeline import MODE_CLARIFY

    settings = get_settings().with_(use_dense=False, log_events=False, provider=PROVIDER_NONE)
    assistant = CareCompass(settings=settings, retriever=retriever, llm=StubLLM())
    assistant.llm.available = False

    answer = assistant.answer("i am in pain what to do?")

    assert answer.mode == MODE_CLARIFY
    assert "could not find" not in answer.text.lower()
    assert "?" in answer.text, "a clarifying reply has to actually ask something"
    assert "112" in answer.text, "the severe-symptom exception must survive"


def test_vague_symptom_clarifies_in_the_users_language(retriever):
    from src.pipeline import MODE_CLARIFY

    settings = get_settings().with_(use_dense=False, log_events=False, provider=PROVIDER_NONE)
    assistant = CareCompass(settings=settings, retriever=retriever, llm=StubLLM())
    assistant.llm.available = False

    answer = assistant.answer("मुझे दर्द हो रहा है")
    assert answer.language == "hi"
    assert answer.mode == MODE_CLARIFY
    assert "112" in answer.text


def test_non_health_question_is_still_refused(retriever):
    """The clarifying path must not swallow genuinely out-of-scope questions."""
    from src.pipeline import MODE_EXTRACTIVE

    settings = get_settings().with_(use_dense=False, log_events=False, provider=PROVIDER_NONE)
    assistant = CareCompass(settings=settings, retriever=retriever, llm=StubLLM())
    assistant.llm.available = False

    answer = assistant.answer("write me a python function to sort a list")
    assert answer.mode == MODE_EXTRACTIVE
    assert "could not find" in answer.text.lower()


def test_answer_exposes_body_and_disclaimer_separately(retriever):
    """A chat UI shows the disclaimer once, so it has to be separable.

    Pinned as a contract: the Streamlit build reads `body`, and when a deploy
    left a cached assistant behind that predated this field, the app raised
    AttributeError at the user instead of answering.
    """
    answer = build(retriever).answer("I have a mild sore throat")

    assert answer.body, "body must carry the reply"
    assert answer.disclaimer, "disclaimer must be available on its own"
    assert answer.disclaimer not in answer.body, "body must not repeat the disclaimer"
    assert answer.body in answer.text
    assert answer.disclaimer in answer.text


# --------------------------------------------------------------------------
# follow-up suggestions
# --------------------------------------------------------------------------


def test_model_followups_are_parsed_and_stripped(retriever):
    """The tag is a transport detail and must never reach the reader."""
    llm = StubLLM(
        text=(
            "Give ORS. [S1]\n\n"
            "[[FOLLOWUPS: How do I make ORS at home? | "
            "When should I go to hospital? | Is it safe to keep feeding?]]"
        )
    )
    answer = build(retriever, llm).answer("my child has loose motions")

    assert answer.followups == [
        "How do I make ORS at home?",
        "When should I go to hospital?",
        "Is it safe to keep feeding?",
    ]
    assert "FOLLOWUPS" not in answer.body
    assert "FOLLOWUPS" not in answer.text


def test_followups_survive_a_model_that_forgets_the_tag(retriever):
    """A missing tag costs a convenience, never an answer."""
    answer = build(retriever, StubLLM(text="Give ORS. [S1]")).answer(
        "my child has loose motions"
    )
    assert answer.followups, "the rule-based fallback should have filled these in"
    assert all(len(f) < 90 for f in answer.followups)


def test_followups_are_never_offered_on_the_emergency_path(retriever):
    """Nothing on that card should invite a next question instead of a call."""
    answer = build(retriever, StubLLM()).answer(
        "crushing chest pain spreading to my left arm"
    )
    assert answer.mode == MODE_EMERGENCY
    assert answer.followups == []


def test_followups_follow_the_reply_language(retriever):
    answer = build(retriever, None).answer("मेरे बच्चे को दस्त हो रहे हैं", language="hi")
    assert answer.followups
    assert any("\u0900" <= char <= "\u097f" for char in " ".join(answer.followups))


def test_a_paragraph_is_not_a_followup_chip(retriever):
    """Models sometimes answer the follow-up instead of posing it."""
    llm = StubLLM(
        text=(
            "Give ORS. [S1]\n\n[[FOLLOWUPS: ok | "
            + "You should give oral rehydration solution after every loose stool "
            "and continue feeding normally throughout the illness, which " * 2
            + "]]"
        )
    )
    answer = build(retriever, llm).answer("my child has loose motions")
    # Both malformed items are dropped, and the rule-based fallback then fills
    # the gap — so the assertion is about what never reaches a chip, not about
    # ending up with none.
    assert not any(len(f) > 90 or len(f) < 8 for f in answer.followups)
    assert not any("oral rehydration solution after every" in f for f in answer.followups)
