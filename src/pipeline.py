"""The orchestrator: one call in, one fully-formed answer out.

Turn flow:

    message
      -> safety scan          (regex, deterministic)
      -> triage floor         (rules, deterministic)
      -> hybrid retrieval     (BM25 + embeddings, over the curated corpus)
      -> web tier             (allow-listed public-health sources, only when
                               the corpus came up empty -- never on the
                               emergency path, never on a vague message)
      -> EITHER a fixed emergency/crisis card   [no model]
         OR a grounded model answer             [model constrained by the floor]
         OR an extractive answer                [no model configured or call failed]
      -> citation validation, follow-up suggestions, disclaimer, event log

Three properties are worth noticing, because they are what separate this from
a prompt wrapped around a vector store:

* **The safety-critical path never reaches a model.** Emergencies and self-harm
  risk are answered from fixed text, so the ambulance number cannot be
  hallucinated and the response cannot be jailbroken by a later instruction.
* **The model cannot de-escalate.** Its urgency vote is merged with ``max()``
  against the rule floor.
* **There is always an answer.** No key, a dead provider, a rate limit or a
  network timeout all degrade to extractive output rather than an error. The
  same is true of the web tier: it is additive, and every way it can fail ends
  with the corpus answering alone.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

from .analytics import log_event
from .config import APP_NAME, PROVIDER_NONE, Settings, get_settings
from .knowledge import corpus_stats
from .language import resolve_language
from .llm import LLMClient
from .prompts import (
    build_messages,
    build_system_prompt,
    clarify_card,
    crisis_card,
    disclaimer_for,
    emergency_card,
    parse_followups,
    restricted_note,
)
from .retrieval import RetrievalResult, Retriever
from .safety import SafetyReport, redact, scan
from .web import WebResult, WebRetriever
from .triage import (
    LEVELS,
    TriageResult,
    Urgency,
    assess,
    mentions_symptom,
    merge_model_level,
    parse_level_tag,
)

#: Two source namespaces: [S] for the curated corpus, [W] for anything the
#: web tier fetched. One regex so a reply is scanned once and neither kind
#: can smuggle through a reference to a source that was never retrieved.
CITATION_RE = re.compile(r"\[([SW])(\d+)\]")

#: Models reach for these instead of ASCII brackets more often than one would
#: expect — the CJK lenticular pair especially, and reliably so on some
#: providers. Left alone, a perfectly cited answer scores as uncited: the tag
#: fails to match, so it is neither validated nor marked, and the sources panel
#: reports "retrieved" beside every passage the answer actually quoted. Cheap
#: to normalise, and invisible when there is nothing to normalise.
_BRACKET_VARIANTS = str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]"})

MODE_EMERGENCY = "deterministic-emergency"
MODE_CRISIS = "deterministic-crisis"
MODE_MODEL = "model-grounded"
MODE_EXTRACTIVE = "retrieval-extractive"
MODE_CLARIFY = "clarifying-question"
MODE_WEB = "web-grounded"


def _stopwords() -> set[str]:
    """The retrieval lexicon's stopword list, loaded once and shared."""
    global _STOPWORDS
    if _STOPWORDS is None:
        from .retrieval import _load_lexicon

        _STOPWORDS = _load_lexicon()[1]
    return _STOPWORDS


_STOPWORDS: set[str] | None = None

EMPTY_PROMPTS = {
    "en": "Tell me what is going on and I will help you work out what to do next.",
    "hi": "आप क्या महसूस कर रहे हैं, बताइए — मैं अगला कदम तय करने में मदद करूँगा।",
    "bn": "কী হচ্ছে বলুন, পরের ধাপ ঠিক করতে আমি সাহায্য করব।",
}


@dataclass
class Citation:
    index: int
    title: str
    heading: str
    source: str
    source_url: str
    last_reviewed: str
    matched_by: str
    used: bool = False
    #: ``corpus`` for a reviewed knowledge-base passage, ``web`` for one
    #: fetched at answer time from an allow-listed public-health body. The two
    #: are numbered independently and are never presented as equivalent: a
    #: reader deciding how much to trust a sentence needs to know which shelf
    #: it came off, and so does anyone auditing a bad answer afterwards.
    kind: str = "corpus"

    @property
    def tag(self) -> str:
        return f"[{'W' if self.kind == 'web' else 'S'}{self.index}]"


@dataclass
class Answer:
    #: The full reply, disclaimer included. What an API caller should use.
    text: str
    #: The reply without the trailing disclaimer, and the disclaimer on its own.
    #: A chat UI shows the disclaimer once, pinned near the input, rather than
    #: repeating it under every message where it stops being read.
    body: str
    disclaimer: str
    triage: TriageResult
    safety: SafetyReport
    retrieval: RetrievalResult
    citations: list[Citation]
    language: str
    mode: str
    provider: str
    model: str
    latency_ms: int
    warnings: list[str] = field(default_factory=list)
    #: What the web tier returned, or an empty result carrying the reason it
    #: returned nothing.
    web: WebResult = field(default_factory=lambda: WebResult(query=""))
    #: Two or three things the user might reasonably ask next. Written by the
    #: model when one is configured, derived from the triage level and the
    #: retrieved topic when one is not. Always safe to ignore.
    followups: list[str] = field(default_factory=list)

    @property
    def urgency_badge(self) -> str:
        meta = LEVELS[self.triage.level]
        return f"{meta.badge} · {meta.timeframe}"

    def sources_markdown(self) -> str:
        if not self.citations:
            return "_No knowledge-base passage was retrieved for this message._"

        def render(citation: Citation) -> str:
            marker = "**cited**" if citation.used else "retrieved"
            link = (
                f"[{citation.source}]({citation.source_url})"
                if citation.source_url
                else citation.source
            )
            stamp = (
                f"fetched {citation.last_reviewed}"
                if citation.kind == "web"
                else f"reviewed {citation.last_reviewed}"
            )
            return (
                f"**{citation.tag} {citation.title} — {citation.heading}**  \n"
                f"{marker} · matched by {citation.matched_by} · {link} · {stamp}"
            )

        corpus = [c for c in self.citations if c.kind == "corpus"]
        web = [c for c in self.citations if c.kind == "web"]

        # Grouped, not interleaved. The distinction between reviewed and
        # just-fetched is the whole reason both are labelled, and a mixed list
        # is exactly where a reader stops noticing it.
        blocks: list[str] = []
        if corpus:
            blocks.append("**Curated knowledge base**\n\n" + "\n\n".join(map(render, corpus)))
        if web:
            blocks.append(
                "**Fetched from public-health sources**\n\n"
                + "\n\n".join(map(render, web))
            )
        return "\n\n".join(blocks)

    def _web_trace(self) -> str:
        """One line on what the web tier did, including when it did nothing.

        A tier that quietly declines to run looks identical to one that ran and
        found nothing, and those are different bugs.
        """
        web = self.web
        if not web.attempted:
            return f"not consulted ({web.error or 'corpus answered the question'})"
        if not web.passages:
            return f"consulted, nothing usable ({web.error}) in {web.latency_ms} ms"
        sites = ", ".join(dict.fromkeys(p.site for p in web.passages))
        return (
            f"`{web.provider}` returned {len(web.passages)} passage"
            f"{'' if len(web.passages) == 1 else 's'} from {sites} "
            f"in {web.latency_ms} ms · grounded `{web.grounded}`"
        )

    def trace_markdown(self) -> str:
        """The audit trail, shown in the UI so the decision is inspectable."""
        flags = (
            "\n".join(
                f"- `{f.id}` ({f.urgency}) matched on `{f.evidence}`"
                for f in self.safety.red_flags
            )
            or "- none"
        )
        modifiers = ", ".join(m.label for m in self.safety.modifiers) or "none"
        restrictions = ", ".join(r.label for r in self.safety.restrictions) or "none"
        reasons = "\n".join(f"- {r}" for r in self.triage.reasons) or "- none"
        model_vote = (
            self.triage.model_level.name if self.triage.model_level else "not consulted"
        )
        return (
            f"**Red flags**\n{flags}\n\n"
            f"**Context modifiers:** {modifiers}\n\n"
            f"**Restricted intents:** {restrictions}\n\n"
            f"**Triage reasoning**\n{reasons}\n\n"
            f"**Rule floor:** `{self.triage.floor.name}` · "
            f"**model vote:** `{model_vote}` · "
            f"**final:** `{self.triage.level.name}`\n\n"
            f"**Retrieval:** mode `{self.retrieval.mode}`, "
            f"grounded `{self.retrieval.grounded}`, "
            f"best BM25 `{self.retrieval.max_lexical}`, "
            f"best cosine `{self.retrieval.max_dense}`, "
            f"term coverage `{self.retrieval.term_coverage}`\n\n"
            f"**Web tier:** {self._web_trace()}\n\n"
            f"**Answer path:** `{self.mode}` via `{self.provider}` "
            f"in {self.latency_ms} ms"
            + ("\n\n**Warnings:** " + "; ".join(self.warnings) if self.warnings else "")
        )


class CareCompass:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
        web: WebRetriever | None = None,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(settings=self.settings)
        self.llm = llm or LLMClient(self.settings)
        self.web = web or WebRetriever(self.settings)

    # -- introspection -----------------------------------------------------

    def health(self) -> dict:
        return {
            "app": APP_NAME,
            "retrieval": self.retriever.describe(),
            "corpus": corpus_stats(self.retriever.chunks),
            "llm": self.llm.describe(),
            "web": self.web.describe(),
        }

    # -- main entry point --------------------------------------------------

    def answer(
        self,
        message: str,
        history: list[dict] | None = None,
        language: str = "auto",
        session_id: str | None = None,
    ) -> Answer:
        started = time.perf_counter()
        history = history or []
        session_id = session_id or str(uuid.uuid4())[:8]
        message = (message or "").strip()

        lang = resolve_language(language, message)

        if not message:
            empty_safety = SafetyReport(query="", scanned_text="")
            prompt_text = EMPTY_PROMPTS.get(lang, EMPTY_PROMPTS["en"])
            return Answer(
                text=prompt_text,
                body=prompt_text,
                disclaimer=disclaimer_for(lang),
                triage=TriageResult(level=Urgency.INFO, floor=Urgency.INFO),
                safety=empty_safety,
                retrieval=RetrievalResult(query="", expanded_query=""),
                citations=[],
                language=lang,
                mode=MODE_EXTRACTIVE,
                provider=PROVIDER_NONE,
                model="",
                latency_ms=0,
            )

        history_text = " ".join(
            m.get("content", "") for m in history if m.get("role") == "user"
        )[-1200:]

        safety = scan(message, history_text=history_text)
        retrieval = self.retriever.search(message)
        triage = assess(message, safety)

        citations = [
            Citation(
                index=i,
                title=hit.chunk.doc_title,
                heading=hit.chunk.heading,
                source=hit.chunk.source,
                source_url=hit.chunk.source_url,
                last_reviewed=hit.chunk.last_reviewed,
                matched_by=hit.matched_by,
            )
            for i, hit in enumerate(retrieval.hits, start=1)
        ]

        warnings: list[str] = []
        web = WebResult(query=message)
        followups: list[str] = []

        # --- safety-critical paths: fixed text, no model ------------------
        #
        # Note what is *not* here: no retrieval decision, no network call, no
        # model. Someone describing a stroke gets the fixed card in the time it
        # takes to match a regex, and nothing downstream can delay that.
        if safety.is_crisis:
            triage.level = Urgency.EMERGENCY
            triage.floor = Urgency.EMERGENCY
            text = crisis_card(lang)
            mode, provider, model = MODE_CRISIS, PROVIDER_NONE, ""
            web.error = "safety-critical path; no network call is made here"
        elif safety.is_emergency:
            triage.level = Urgency.EMERGENCY
            triage.floor = Urgency.EMERGENCY
            text = emergency_card(safety, lang)
            mode, provider, model = MODE_EMERGENCY, PROVIDER_NONE, ""
            web.error = "safety-critical path; no network call is made here"
        else:
            web = self._consult_web(message, retrieval)
            if web.error and web.error.startswith("web tier failed"):
                warnings.append(web.error)
            citations.extend(
                Citation(
                    index=i,
                    title=passage.title,
                    heading=passage.heading,
                    source=passage.site,
                    source_url=passage.url,
                    last_reviewed=passage.retrieved_at,
                    matched_by=f"web · {web.provider}",
                    kind="web",
                )
                for i, passage in enumerate(web.passages, start=1)
            )

            text, mode, provider, model, followups, extra = self._compose(
                message, history, lang, triage, safety, retrieval, web
            )
            warnings.extend(extra)

        text, used = self._apply_citations(text, citations)
        if mode in (MODE_MODEL, MODE_WEB) and not used:
            warnings.append("model answered without citing a source")

        if not followups and mode not in (MODE_CRISIS, MODE_EMERGENCY):
            followups = self._fallback_followups(triage, retrieval, web, lang)

        body = text
        disclaimer = disclaimer_for(lang)
        text = f"{body}\n\n---\n*{disclaimer}*"
        latency_ms = int((time.perf_counter() - started) * 1000)

        answer = Answer(
            text=text,
            body=body,
            disclaimer=disclaimer,
            triage=triage,
            safety=safety,
            retrieval=retrieval,
            citations=citations,
            language=lang,
            mode=mode,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            warnings=warnings,
            web=web,
            followups=followups,
        )
        self._log(answer, session_id, message)
        return answer

    # -- internals ---------------------------------------------------------

    def _consult_web(self, message: str, retrieval: RetrievalResult) -> WebResult:
        """Decide whether to reach past the corpus, and do it if so.

        Two guards before any network call:

        * In the default ``fallback`` mode, a question the curated corpus
          already answers confidently does not go out to the network at all.
          Reviewed content beats fetched content, so there is nothing to gain.
        * An underspecified message does not go out either, and this is the
          guard that matters. "I am in pain" will happily match a general page
          about pain, and answering it with one is a worse response than asking
          where it hurts. The web tier exists for questions that are specific
          but outside the corpus, not for questions that are not yet questions.

        There is deliberately **no** off-topic guard, and the omission took a
        failing test to arrive at. The obvious one — refuse anything the corpus
        cannot ground and that contains no symptom word — does block "write me
        a python function". It also blocks "what is gout", because the corpus
        has nothing on gout either, and that is the exact class of question
        this tier was built for. No cheap local signal separates the two, so
        the discriminator is the search itself: MedlinePlus is a health-topic
        index behind a domain allow-list, and a question about sorting a list
        comes back empty. That costs about a second on a question nobody should
        be asking here, which is the cheaper of the two mistakes.
        """
        if not self.web.enabled:
            return WebResult(query=message, error="web tier disabled")
        if self.settings.web_mode == "fallback" and self._corpus_is_confident(retrieval):
            return WebResult(query=message, error="corpus already grounded")
        if self._underspecified(message):
            return WebResult(query=message, error="too vague to search; asking instead")

        # ``WebRetriever.search`` already swallows its own network failures, so
        # this catches the other kind: a retriever that is broken rather than
        # merely offline. The tier is additive by design and this is what makes
        # that true in practice — an exception here must cost the reader a
        # supporting source, never their answer.
        try:
            return self.web.search(message)
        except Exception as exc:
            return WebResult(
                query=message,
                attempted=True,
                error=f"web tier failed ({type(exc).__name__})",
            )

    #: A corpus hit above *either* of these is treated as the real thing, and
    #: the web is not consulted. Both sit well above the thresholds retrieval
    #: uses to call a result grounded at all, and the gap between them is the
    #: point of this method — see :meth:`_corpus_is_confident`.
    CONFIDENT_DENSE = 0.42
    CONFIDENT_LEXICAL = 8.0

    @classmethod
    def _corpus_is_confident(cls, retrieval: RetrievalResult) -> bool:
        """Whether the corpus answered this well enough to skip the web.

        Not the same question as ``retrieval.grounded``, and the difference is
        the reason this exists. ``grounded`` is tuned to decide *is this in
        scope at all* — it has to be generous, because refusing a question the
        corpus does cover is the worse error there. That generosity shows:
        "what is shingles" clears it on a lexical match against a passage about
        adult vaccination, and "thyroid test came back high" clears it on a
        passable cosine score against a passage about typhoid.

        Both of those are real questions the corpus does not answer, and before
        the web tier existed the model was handed the near-miss passage and
        answered from recall instead, uncited. So the bar for *not bothering to
        look further* is set separately, and higher: a strong lexical match or
        a strong semantic one, not merely enough of either to stay in scope.
        """
        return (
            retrieval.grounded
            and (
                retrieval.max_dense >= cls.CONFIDENT_DENSE
                or retrieval.max_lexical >= cls.CONFIDENT_LEXICAL
            )
        )

    #: Symptom words so general that a page about them tells a reader nothing
    #: they did not already know. Stripped before judging whether a message
    #: carries enough detail to search on.
    _GENERIC_TERMS = frozenset(
        {
            "pain", "ache", "aching", "hurt", "hurting", "sore", "unwell", "sick",
            "ill", "illness", "problem", "issue", "discomfort", "uncomfortable",
            "weak", "weakness", "tired", "fatigue", "feel", "feeling", "help",
            "advice", "doctor", "symptom", "symptoms", "body", "health", "well",
        }
    )

    @classmethod
    def _underspecified(cls, message: str) -> bool:
        """True when the message names no concrete thing to search for.

        A count of content words, after stopwords and the generic vocabulary
        above are removed. Crude on purpose — it gates a network call, not a
        clinical decision, and the cost of being wrong in either direction is
        one unnecessary question or one unnecessary second.

        The bar is deliberately at *nothing left*, not *little left*. One
        concrete noun is enough to search on: "what is hypothyroidism" reduces
        to a single token and is a perfectly good query, while "I am in pain"
        reduces to none.
        """
        from .retrieval import stem, tokenize

        generic = {stem(t) for t in cls._GENERIC_TERMS}
        tokens = {
            t for t in tokenize(message, _stopwords())
            if len(t) > 2 and t not in generic
        }
        return not tokens

    def _compose(self, message, history, lang, triage, safety, retrieval, web):
        """Non-emergency answer: model if available, extractive otherwise."""
        warnings: list[str] = []

        # Nothing matched on either shelf, but the person is plainly describing
        # a symptom. Ask, rather than refuse.
        vague = not retrieval.grounded and not web.grounded and mentions_symptom(message)

        if self.llm.available:
            system = build_system_prompt(
                app_name=APP_NAME,
                language=lang,
                triage=triage,
                safety=safety,
                retrieval=retrieval,
                web=web,
                vague=vague,
            )
            response = self.llm.chat(system, build_messages(history, message))
            if response.ok:
                level, cleaned = parse_level_tag(response.text)
                merge_model_level(triage, level)
                followups, cleaned = parse_followups(cleaned)
                # Named for whichever shelf actually carried the answer, not
                # for whichever one was consulted first. A turn where the
                # corpus only grazed the question and the web supplied the
                # substance is a web-grounded turn, and the analytics should
                # say so.
                if vague:
                    mode = MODE_CLARIFY
                elif web.grounded and not self._corpus_is_confident(retrieval):
                    mode = MODE_WEB
                else:
                    mode = MODE_MODEL
                return cleaned, mode, response.provider, response.model, followups, warnings
            warnings.append(f"model call failed ({response.error}); used retrieval only")

        if vague:
            return clarify_card(lang), MODE_CLARIFY, PROVIDER_NONE, "", [], warnings

        return (
            self._extractive(retrieval, triage, lang, safety, web),
            MODE_WEB
            if (web.grounded and not self._corpus_is_confident(retrieval))
            else MODE_EXTRACTIVE,
            PROVIDER_NONE,
            "",
            [],
            warnings,
        )

    # -- follow-up suggestions --------------------------------------------

    #: Used when no model is configured, or when a model forgot its tag.
    #: Deliberately keyed off the triage level rather than the topic: the level
    #: is the thing the reader most often wants to act on, and a suggestion
    #: that helps someone decide whether to go in today is worth more than one
    #: that offers to explain the condition further.
    _LEVEL_FOLLOWUPS: dict[str, dict[Urgency, list[str]]] = {
        "en": {
            Urgency.INFO: ["How can I prevent this?", "What should I watch for?"],
            Urgency.SELF_CARE: [
                "What can I safely do at home?",
                "When should I stop waiting and see someone?",
            ],
            Urgency.ROUTINE: [
                "What should I tell the doctor?",
                "Which tests are usually done for this?",
            ],
            Urgency.URGENT: [
                "Should I go to a hospital or a clinic?",
                "What should I take with me?",
            ],
            Urgency.EMERGENCY: ["What do I do while waiting for help?"],
        },
        "hi": {
            Urgency.INFO: ["इससे बचाव कैसे करें?", "किन बातों पर ध्यान दूँ?"],
            Urgency.SELF_CARE: ["घर पर क्या कर सकते हैं?", "डॉक्टर के पास कब जाना चाहिए?"],
            Urgency.ROUTINE: ["डॉक्टर को क्या बताना चाहिए?", "कौन सी जाँच होती है?"],
            Urgency.URGENT: ["अस्पताल जाऊँ या क्लिनिक?", "साथ में क्या ले जाऊँ?"],
            Urgency.EMERGENCY: ["मदद आने तक क्या करूँ?"],
        },
        "bn": {
            Urgency.INFO: ["কীভাবে প্রতিরোধ করব?", "কোন দিকে খেয়াল রাখব?"],
            Urgency.SELF_CARE: ["বাড়িতে কী করতে পারি?", "কখন ডাক্তার দেখাব?"],
            Urgency.ROUTINE: ["ডাক্তারকে কী বলব?", "সাধারণত কী পরীক্ষা হয়?"],
            Urgency.URGENT: ["হাসপাতাল না ক্লিনিক?", "সঙ্গে কী নিয়ে যাব?"],
            Urgency.EMERGENCY: ["সাহায্য আসা পর্যন্ত কী করব?"],
        },
    }

    def _fallback_followups(
        self,
        triage: TriageResult,
        retrieval: RetrievalResult,
        web: WebResult,
        lang: str,
    ) -> list[str]:
        table = self._LEVEL_FOLLOWUPS.get(lang, self._LEVEL_FOLLOWUPS["en"])
        items = list(table.get(triage.level, []))

        # One topic-specific suggestion, drawn from the heading actually
        # retrieved, so the list is not identical on every turn at a level.
        heading = ""
        if retrieval.hits:
            heading = retrieval.hits[0].chunk.heading
        elif web.passages:
            heading = web.passages[0].heading
        if heading and lang == "en" and not heading.endswith("?"):
            items.append(f"Tell me more about {heading[0].lower()}{heading[1:]}"[:80])

        return items[:3]

    @staticmethod
    def _key_points(text: str, limit: int = 4) -> list[str]:
        bullets = [
            line.strip().lstrip("-*").strip()
            for line in text.splitlines()
            if line.strip().startswith(("-", "*"))
        ]
        if bullets:
            return bullets[:limit]
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()][:limit]

    def _extractive(
        self,
        retrieval: RetrievalResult,
        triage: TriageResult,
        lang: str,
        safety: SafetyReport | None = None,
        web: WebResult | None = None,
    ) -> str:
        """Answer assembled from retrieved passages when no model is available.

        Honest by construction: it quotes the corpus and says what it is. A
        silent quality drop would be worse than a visible one.
        """
        web = web or WebResult(query="")

        # The corpus missed but the web tier landed: quote that instead, and
        # name the body it came from in the same breath. Without a model there
        # is no one to paraphrase, so attribution has to be structural.
        if (not retrieval.hits or not retrieval.grounded) and web.grounded:
            meta = LEVELS[triage.level]
            parts = [
                f"**{meta.label}** — {meta.timeframe}.",
                "",
                "This is outside my curated knowledge base, so here is what public "
                "health bodies publish on it:",
                "",
            ]
            for i, passage in enumerate(web.passages[:3], start=1):
                points = self._key_points(passage.text, limit=3)
                if not points:
                    continue
                parts.append(f"**{passage.heading}** — {passage.site} [W{i}]")
                parts.extend(f"- {p}" for p in points)
                parts.append("")
            parts.append(
                "_No language model is configured, so this reply quotes those pages "
                "directly rather than answering your question in your words. Follow "
                "the source links for the full guidance._"
            )
            return "\n".join(parts).strip()

        if not retrieval.hits or not retrieval.grounded:
            return (
                "I could not find anything in my knowledge base that covers this. "
                "I hold general public-health guidance on fever and infections, breathing "
                "problems, stomach and hydration, heart and stroke warning signs, diabetes, "
                "blood pressure, pregnancy and child health, mental health, medicine safety, "
                "first aid and preventive care.\n\n"
                "For anything outside that, a doctor or pharmacist is the right person to ask."
            )

        meta = LEVELS[triage.level]
        parts = [f"**{meta.label}** — {meta.timeframe}.", ""]

        for restriction in (safety.restrictions if safety else []):
            note = restricted_note(restriction.id, lang)
            if note:
                parts += [note, ""]

        for i, hit in enumerate(retrieval.hits[:3], start=1):
            points = self._key_points(hit.chunk.text, limit=4)
            if not points:
                continue
            parts.append(f"**{hit.chunk.heading}** [S{i}]")
            parts.extend(f"- {p}" for p in points)
            parts.append("")

        if lang != "en":
            parts.append(
                "_The knowledge base is written in English, so these passages are quoted "
                "in English._"
            )
        parts.append(
            "_No language model is configured, so this reply is quoted directly from the "
            "knowledge base rather than written for your question._"
        )
        return "\n".join(parts).strip()

    @staticmethod
    def _apply_citations(text: str, citations: list[Citation]) -> tuple[str, bool]:
        """Mark cited sources and delete references to sources that do not exist.

        The two namespaces are validated independently. A model that saw three
        corpus passages and two web passages can write ``[S3]`` and ``[W2]``
        but not ``[W4]``, and a tag pointing at a source that was never
        retrieved is deleted rather than shown — an unchecked citation is worse
        than no citation, because it looks checked.
        """
        text = (text or "").translate(_BRACKET_VARIANTS)

        by_kind: dict[str, dict[int, Citation]] = {"corpus": {}, "web": {}}
        for citation in citations:
            by_kind[citation.kind][citation.index] = citation
        used = False

        def replace(match: re.Match) -> str:
            nonlocal used
            kind = "web" if match.group(1).upper() == "W" else "corpus"
            citation = by_kind[kind].get(int(match.group(2)))
            if citation is None:
                return ""  # hallucinated reference
            used = True
            citation.used = True
            return match.group(0)

        return CITATION_RE.sub(replace, text).strip(), used

    def _log(self, answer: Answer, session_id: str, message: str) -> None:
        if not self.settings.log_events:
            return
        log_event(
            {
                "session_id": session_id,
                "language": answer.language,
                "query_chars": len(redact(message)),
                "triage_level": answer.triage.level.name,
                "triage_floor": answer.triage.floor.name,
                "model_level": (
                    answer.triage.model_level.name if answer.triage.model_level else None
                ),
                "red_flags": [f.id for f in answer.safety.red_flags],
                "modifiers": [m.id for m in answer.safety.modifiers],
                "restrictions": [r.id for r in answer.safety.restrictions],
                "retrieved_docs": list(dict.fromkeys(answer.retrieval.doc_ids)),
                "retrieval_mode": answer.retrieval.mode,
                "grounded": answer.retrieval.grounded,
                "web_provider": answer.web.provider if answer.web.attempted else None,
                "web_passages": len(answer.web.passages),
                "web_sites": sorted({p.site for p in answer.web.passages}),
                "citations": sum(1 for c in answer.citations if c.used),
                "web_citations": sum(
                    1 for c in answer.citations if c.used and c.kind == "web"
                ),
                "followups": len(answer.followups),
                "mode": answer.mode,
                "provider": answer.provider,
                "latency_ms": answer.latency_ms,
                "warnings": answer.warnings,
            }
        )


_SINGLETON: CareCompass | None = None


def get_assistant() -> CareCompass:
    """Process-wide instance so the index and embedding model load once."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = CareCompass()
    return _SINGLETON
