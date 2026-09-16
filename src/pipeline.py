"""The orchestrator: one call in, one fully-formed answer out.

Turn flow:

    message
      -> safety scan          (regex, deterministic)
      -> triage floor         (rules, deterministic)
      -> hybrid retrieval     (BM25 + embeddings)
      -> EITHER a fixed emergency/crisis card   [no model]
         OR a grounded model answer             [model constrained by the floor]
         OR an extractive answer                [no model configured or call failed]
      -> citation validation, disclaimer, event log

Three properties are worth noticing, because they are what separate this from
a prompt wrapped around a vector store:

* **The safety-critical path never reaches a model.** Emergencies and self-harm
  risk are answered from fixed text, so the ambulance number cannot be
  hallucinated and the response cannot be jailbroken by a later instruction.
* **The model cannot de-escalate.** Its urgency vote is merged with ``max()``
  against the rule floor.
* **There is always an answer.** No key, a dead provider, a rate limit or a
  network timeout all degrade to extractive output rather than an error.
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
    crisis_card,
    disclaimer_for,
    emergency_card,
    restricted_note,
)
from .retrieval import RetrievalResult, Retriever
from .safety import SafetyReport, redact, scan
from .triage import LEVELS, TriageResult, Urgency, assess, merge_model_level, parse_level_tag

CITATION_RE = re.compile(r"\[S(\d+)\]")

MODE_EMERGENCY = "deterministic-emergency"
MODE_CRISIS = "deterministic-crisis"
MODE_MODEL = "model-grounded"
MODE_EXTRACTIVE = "retrieval-extractive"

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


@dataclass
class Answer:
    text: str
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

    @property
    def urgency_badge(self) -> str:
        meta = LEVELS[self.triage.level]
        return f"{meta.badge} · {meta.timeframe}"

    def sources_markdown(self) -> str:
        if not self.citations:
            return "_No knowledge-base passage was retrieved for this message._"
        lines = []
        for citation in self.citations:
            marker = "**cited**" if citation.used else "retrieved"
            link = (
                f"[{citation.source}]({citation.source_url})"
                if citation.source_url
                else citation.source
            )
            lines.append(
                f"**[S{citation.index}] {citation.title} — {citation.heading}**  \n"
                f"{marker} · matched by {citation.matched_by} · {link} · "
                f"reviewed {citation.last_reviewed}"
            )
        return "\n\n".join(lines)

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
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(settings=self.settings)
        self.llm = llm or LLMClient(self.settings)

    # -- introspection -----------------------------------------------------

    def health(self) -> dict:
        return {
            "app": APP_NAME,
            "retrieval": self.retriever.describe(),
            "corpus": corpus_stats(self.retriever.chunks),
            "llm": self.llm.describe(),
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
            return Answer(
                text=EMPTY_PROMPTS.get(lang, EMPTY_PROMPTS["en"]),
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

        # --- safety-critical paths: fixed text, no model ------------------
        if safety.is_crisis:
            triage.level = Urgency.EMERGENCY
            triage.floor = Urgency.EMERGENCY
            text = crisis_card(lang)
            mode, provider, model = MODE_CRISIS, PROVIDER_NONE, ""
        elif safety.is_emergency:
            triage.level = Urgency.EMERGENCY
            triage.floor = Urgency.EMERGENCY
            text = emergency_card(safety, lang)
            mode, provider, model = MODE_EMERGENCY, PROVIDER_NONE, ""
        else:
            text, mode, provider, model, extra = self._compose(
                message, history, lang, triage, safety, retrieval
            )
            warnings.extend(extra)

        text, used = self._apply_citations(text, citations)
        if mode == MODE_MODEL and retrieval.grounded and not used:
            warnings.append("model answered without citing a source")

        text = f"{text}\n\n---\n*{disclaimer_for(lang)}*"
        latency_ms = int((time.perf_counter() - started) * 1000)

        answer = Answer(
            text=text,
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
        )
        self._log(answer, session_id, message)
        return answer

    # -- internals ---------------------------------------------------------

    def _compose(self, message, history, lang, triage, safety, retrieval):
        """Non-emergency answer: model if available, extractive otherwise."""
        warnings: list[str] = []

        if self.llm.available:
            system = build_system_prompt(
                app_name=APP_NAME,
                language=lang,
                triage=triage,
                safety=safety,
                retrieval=retrieval,
            )
            response = self.llm.chat(system, build_messages(history, message))
            if response.ok:
                level, cleaned = parse_level_tag(response.text)
                merge_model_level(triage, level)
                return cleaned, MODE_MODEL, response.provider, response.model, warnings
            warnings.append(f"model call failed ({response.error}); used retrieval only")

        return (
            self._extractive(retrieval, triage, lang, safety),
            MODE_EXTRACTIVE,
            PROVIDER_NONE,
            "",
            warnings,
        )

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
    ) -> str:
        """Answer assembled from retrieved passages when no model is available.

        Honest by construction: it quotes the corpus and says what it is. A
        silent quality drop would be worse than a visible one.
        """
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
        """Mark cited sources and delete references to sources that do not exist."""
        valid = {c.index for c in citations}
        used = False

        def replace(match: re.Match) -> str:
            nonlocal used
            index = int(match.group(1))
            if index in valid:
                used = True
                citations[index - 1].used = True
                return match.group(0)
            return ""  # hallucinated reference

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
                "citations": sum(1 for c in answer.citations if c.used),
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
