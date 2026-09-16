"""Deterministic safety layer.

Everything in this module runs *before* the language model is consulted, and
its output constrains what the model is allowed to do. The design rule is:

    A language model may raise the urgency of a conversation.
    It may never lower it below the floor set here.

That inversion is the whole point. A probabilistic system that is right 97% of
the time is fine for phrasing an explanation and unacceptable for deciding
whether someone with crushing chest pain is told to call an ambulance. So the
emergency decision is made by auditable regular expressions in
``data/red_flags.yaml`` that a clinician can review without reading Python.

Negation handling is deliberately conservative. The scope of a negation cue
ends at the first conjunction or punctuation mark, so "no fever but chest
pain" keeps the chest pain flag. Over-flagging costs a needless reassurance;
under-flagging costs a missed emergency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from .config import RED_FLAGS_PATH, TRANSLATIONS_PATH

# Cues where the negated span follows the cue (English word order).
PRE_NEGATION = re.compile(
    r"\b(no|not|without|never|none|denies|denying|free\s+of|"
    r"don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|wasn'?t|haven'?t|hasn'?t)\b",
    re.IGNORECASE,
)
# Cues where the negated span precedes the cue (Hindi/Bengali word order).
POST_NEGATION = re.compile(
    r"\b(nahi|nahin|nai|nei|noi)\b|नहीं|नही|নেই|নয়|না\b",
    re.IGNORECASE,
)
# A negation never reaches past one of these.
NEGATION_BOUNDARY = re.compile(
    r"\b(but|however|although|though|still|yet|except|and)\b|[.;,!?\n]",
    re.IGNORECASE,
)
NEGATION_WINDOW = 30

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PHONE_RE = re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{9}\b")

URGENCY_ORDER = {"INFO": 0, "SELF_CARE": 1, "ROUTINE": 2, "URGENT": 3, "EMERGENCY": 4}


# --------------------------------------------------------------------------
# rule loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledRule:
    id: str
    label: str
    urgency: str
    action: str
    guidance: str
    source_doc: str
    #: {"hi": {"label": ..., "guidance": ...}, "bn": {...}} — loaded from
    #: data/translations.yaml. Kept out of the rule bodies so the matching logic
    #: stays readable for whoever reviews it clinically, and so a translator can
    #: be handed one file with no regular expressions in it.
    translations: dict
    patterns: tuple[re.Pattern, ...]
    #: Some red flags are *phrased* as negations: "not breathing", "no urine
    #: since morning", "I don't want to live". Stripping negation before
    #: matching would erase exactly the phrase that makes them urgent. Those
    #: patterns go in ``patterns_raw`` and are matched against the original
    #: message. Kept per-pattern rather than per-rule so that a rule can hold
    #: both kinds without losing negation handling on the ordinary ones.
    raw_patterns: tuple[re.Pattern, ...] = ()


@dataclass(frozen=True)
class CompiledModifier:
    id: str
    label: str
    patterns: tuple[re.Pattern, ...]


@dataclass(frozen=True)
class CompiledRestriction:
    id: str
    label: str
    policy: str
    patterns: tuple[re.Pattern, ...]


@dataclass(frozen=True)
class RuleSet:
    rules: tuple[CompiledRule, ...]
    modifiers: tuple[CompiledModifier, ...]
    restrictions: tuple[CompiledRestriction, ...]
    version: int


def _compile(patterns) -> tuple[re.Pattern, ...]:
    compiled = []
    for raw in patterns or []:
        try:
            compiled.append(re.compile(str(raw), re.IGNORECASE))
        except re.error as exc:  # a broken rule must not break the app
            raise ValueError(f"Invalid red-flag pattern {raw!r}: {exc}") from exc
    return tuple(compiled)


@lru_cache(maxsize=1)
def load_rules() -> RuleSet:
    raw = yaml.safe_load(RED_FLAGS_PATH.read_text(encoding="utf-8")) or {}

    translations: dict = {}
    if TRANSLATIONS_PATH.exists():
        localised = yaml.safe_load(TRANSLATIONS_PATH.read_text(encoding="utf-8")) or {}
        translations = localised.get("rules") or {}

    rules = tuple(
        CompiledRule(
            id=str(r["id"]),
            label=str(r.get("label", r["id"])),
            urgency=str(r.get("urgency", "URGENT")).upper(),
            action=str(r.get("action", "see_doctor")),
            guidance=" ".join(str(r.get("guidance", "")).split()),
            source_doc=str(r.get("source_doc", "")),
            translations=translations.get(str(r["id"]), {}),
            patterns=_compile(r.get("patterns")),
            raw_patterns=_compile(r.get("patterns_raw")),
        )
        for r in raw.get("rules", [])
    )
    modifiers = tuple(
        CompiledModifier(
            id=str(m["id"]),
            label=str(m.get("label", m["id"])),
            patterns=_compile(m.get("patterns")),
        )
        for m in raw.get("context_modifiers", [])
    )
    restrictions = tuple(
        CompiledRestriction(
            id=str(x["id"]),
            label=str(x.get("label", x["id"])),
            policy=" ".join(str(x.get("response_policy", "")).split()),
            patterns=_compile(x.get("patterns")),
        )
        for x in raw.get("restricted_intents", [])
    )
    return RuleSet(rules, modifiers, restrictions, int(raw.get("version", 1)))


# --------------------------------------------------------------------------
# negation
# --------------------------------------------------------------------------


def _scope_end(text: str, start: int) -> int:
    """End of a forward negation scope: first boundary, or the window limit."""
    limit = min(len(text), start + NEGATION_WINDOW)
    boundary = NEGATION_BOUNDARY.search(text, start, limit)
    return boundary.start() if boundary else limit


def _scope_start(text: str, end: int) -> int:
    """Start of a backward negation scope, for Hindi/Bengali trailing cues."""
    limit = max(0, end - NEGATION_WINDOW)
    last = limit
    for match in NEGATION_BOUNDARY.finditer(text, limit, end):
        last = match.end()
    return last


def strip_negated(text: str) -> str:
    """Blank out negated spans so red-flag patterns never match them."""
    if not text:
        return ""
    keep = [True] * len(text)

    for match in PRE_NEGATION.finditer(text):
        for i in range(match.start(), _scope_end(text, match.end())):
            keep[i] = False
    for match in POST_NEGATION.finditer(text):
        for i in range(_scope_start(text, match.start()), match.end()):
            keep[i] = False

    return "".join(ch if ok else " " for ch, ok in zip(text, keep))


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RedFlagHit:
    id: str
    label: str
    urgency: str
    action: str
    guidance: str
    source_doc: str
    evidence: str
    translations: dict = field(default_factory=dict)

    def label_in(self, language: str) -> str:
        return (self.translations.get(language) or {}).get("label") or self.label

    def guidance_in(self, language: str) -> str:
        return (self.translations.get(language) or {}).get("guidance") or self.guidance


@dataclass(frozen=True)
class ModifierHit:
    id: str
    label: str
    evidence: str


@dataclass(frozen=True)
class RestrictionHit:
    id: str
    label: str
    policy: str
    evidence: str


@dataclass
class SafetyReport:
    query: str
    scanned_text: str
    red_flags: list[RedFlagHit] = field(default_factory=list)
    modifiers: list[ModifierHit] = field(default_factory=list)
    restrictions: list[RestrictionHit] = field(default_factory=list)

    @property
    def is_emergency(self) -> bool:
        return any(f.urgency == "EMERGENCY" for f in self.red_flags)

    @property
    def is_crisis(self) -> bool:
        """Self-harm risk, which needs a different response shape to an ambulance."""
        return any(f.action == "crisis_support" for f in self.red_flags)

    @property
    def floor_urgency(self) -> str:
        if not self.red_flags:
            return "INFO"
        return max((f.urgency for f in self.red_flags), key=lambda u: URGENCY_ORDER.get(u, 0))

    def summary(self) -> dict:
        return {
            "red_flags": [f.id for f in self.red_flags],
            "modifiers": [m.id for m in self.modifiers],
            "restrictions": [r.id for r in self.restrictions],
            "floor": self.floor_urgency,
        }


def scan(text: str, history_text: str = "") -> SafetyReport:
    """Scan a user message, optionally with earlier turns for context.

    ``history_text`` matters because people volunteer context across turns:
    "I'm 34 weeks pregnant" in turn one and "I have a bad headache" in turn
    three together mean something neither means alone. Only context modifiers
    look at history; red flags are matched on the current message so an
    emergency from three turns ago does not re-fire forever.
    """
    scanned = strip_negated(text or "")
    rule_set = load_rules()

    raw = text or ""
    red_flags: list[RedFlagHit] = []
    for rule in rule_set.rules:
        candidates = [(p, scanned) for p in rule.patterns]
        candidates += [(p, raw) for p in rule.raw_patterns]
        for pattern, target in candidates:
            match = pattern.search(target)
            if match:
                red_flags.append(
                    RedFlagHit(
                        id=rule.id,
                        label=rule.label,
                        urgency=rule.urgency,
                        action=rule.action,
                        guidance=rule.guidance,
                        source_doc=rule.source_doc,
                        evidence=match.group(0).strip()[:80],
                        translations=rule.translations,
                    )
                )
                break

    modifier_text = strip_negated(f"{history_text}\n{text}") if history_text else scanned
    modifiers: list[ModifierHit] = []
    for modifier in rule_set.modifiers:
        for pattern in modifier.patterns:
            match = pattern.search(modifier_text)
            if match:
                modifiers.append(
                    ModifierHit(modifier.id, modifier.label, match.group(0).strip()[:60])
                )
                break

    restrictions: list[RestrictionHit] = []
    for restriction in rule_set.restrictions:
        for pattern in restriction.patterns:
            match = pattern.search(text or "")
            if match:
                restrictions.append(
                    RestrictionHit(
                        restriction.id,
                        restriction.label,
                        restriction.policy,
                        match.group(0).strip()[:60],
                    )
                )
                break

    red_flags.sort(key=lambda f: URGENCY_ORDER.get(f.urgency, 0), reverse=True)
    return SafetyReport(
        query=text,
        scanned_text=scanned,
        red_flags=red_flags,
        modifiers=modifiers,
        restrictions=restrictions,
    )


def redact(text: str) -> str:
    """Remove direct identifiers before anything is written to the event log.

    Health questions carry incidental personal data. Logging is for product
    analytics, so identifiers are stripped at the point of writing rather than
    cleaned up later.
    """
    if not text:
        return ""
    out = EMAIL_RE.sub("[email]", text)
    out = AADHAAR_RE.sub("[id-number]", out)
    out = PHONE_RE.sub("[phone]", out)
    return out
