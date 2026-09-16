"""Urgency assessment.

Five levels, each mapped to a concrete timeframe rather than a vague adjective,
because "see a doctor soon" means different things to different readers.

The rule engine computes a *floor*. The model's opinion is merged with
:func:`merge_model_level`, which takes the maximum. A model that wants to
de-escalate an emergency simply cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum

from .safety import SafetyReport, URGENCY_ORDER


class Urgency(IntEnum):
    INFO = 0
    SELF_CARE = 1
    ROUTINE = 2
    URGENT = 3
    EMERGENCY = 4


@dataclass(frozen=True)
class LevelMeta:
    label: str
    badge: str
    colour: str
    timeframe: str


LEVELS: dict[Urgency, LevelMeta] = {
    Urgency.INFO: LevelMeta("General information", "INFO", "#4b6bfb", "No action needed"),
    Urgency.SELF_CARE: LevelMeta("Self-care is reasonable", "SELF-CARE", "#0e9f6e", "Watch for 24-48 hours"),
    Urgency.ROUTINE: LevelMeta("See a doctor", "ROUTINE", "#c27803", "Within a few days"),
    Urgency.URGENT: LevelMeta("Seek care today", "URGENT", "#e3651d", "Same day"),
    Urgency.EMERGENCY: LevelMeta("Emergency", "EMERGENCY", "#d1242f", "Call 112 now"),
}

_NAME_TO_LEVEL = {level.name: level for level in Urgency}

# Duration phrases that shift a symptom out of watchful waiting.
# The trailing "(?!\s*old)" matters more than it looks: without it, "my 3 year
# old has loose motion" parses as a three-year-long symptom and escalates every
# paediatric question.
_DURATION_RE = re.compile(
    r"\b(?:for|since|from|last|past)?\s*(\d+|a|one|two|three|four|five|six|seven|ten)\s*"
    r"(day|days|week|weeks|month|months|year|years)\b(?!\s*old)",
    re.IGNORECASE,
)
_WORD_NUMBERS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "ten": 10}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

# Presence of any of these means the user is describing a body complaint rather
# than asking a general question, which sets the baseline at SELF_CARE.
_SYMPTOM_RE = re.compile(
    r"\b(pain|ache|aching|fever|temperature|cough|cold|vomit|nausea|diarrhoea|diarrhea|"
    r"loose motions?|stool|rash|itch|swelling|swollen|bleed|dizzy|dizziness|giddy|faint|"
    r"breathless|breathing|wheez|tired|fatigue|weak|weakness|burning|cramp|sore|"
    r"injur|wound|burn|bite|sting|numb|tingl|blurred|discharge|constipat|insomnia|"
    r"anxious|anxiety|depress|panic|stress|infection|lump|ulcer|"
    r"headache|migraine|sad|hopeless|lonely|crying|low mood|sleepless)\b"
    # Romanised Hindi and Bengali, which is how a large share of Indian users
    # actually type. Without these, "bukhar hai" scores as a general question.
    r"|\b(dard|bukhar|khansi|ulti|dast|chakkar|kamzori|saans|behosh|sujan|"
    r"jor|kashi|bomi|byatha|matha ghurche|durbolota)\b"
    r"|दर्द|बुखार|खांसी|उल्टी|कमजोरी|सूजन|दस्त|चक्कर"
    r"|ব্যথা|জ্বর|কাশি|বমি|দুর্বলতা|ফোলা",
    re.IGNORECASE,
)

# "How much salt is safe with high blood pressure?" mentions a condition but
# describes no complaint. Treating it as a symptom pushed every informational
# question up a level, so an interrogative opening without a first-person
# complaint keeps the baseline at INFO. Red flags are unaffected: they are
# matched before this ever runs.
_GENERAL_QUESTION_RE = re.compile(
    r"^\s*(what|how|when|why|which|who|where|should i|is it|are there|is there|"
    r"can i|could i|does|do i|tell me|explain)\b",
    re.IGNORECASE,
)
_COMPLAINT_RE = re.compile(
    r"\b(i have|i've|i am having|i feel|i felt|i get|my (child|son|daughter|baby|"
    r"father|mother|wife|husband|brother|sister)\b|since (yesterday|morning|last|"
    r"two|three|\d)|for \d+\s*(day|week|month)|hai\b|ache)\b",
    re.IGNORECASE,
)


@dataclass
class TriageResult:
    level: Urgency
    floor: Urgency
    reasons: list[str] = field(default_factory=list)
    red_flag_ids: list[str] = field(default_factory=list)
    modifier_labels: list[str] = field(default_factory=list)
    escalated_by_modifier: bool = False
    model_level: Urgency | None = None

    @property
    def meta(self) -> LevelMeta:
        return LEVELS[self.level]

    @property
    def is_emergency(self) -> bool:
        return self.level == Urgency.EMERGENCY

    def to_dict(self) -> dict:
        return {
            "level": self.level.name,
            "floor": self.floor.name,
            "model_level": self.model_level.name if self.model_level else None,
            "reasons": self.reasons,
            "red_flags": self.red_flag_ids,
            "modifiers": self.modifier_labels,
            "escalated_by_modifier": self.escalated_by_modifier,
        }


def _max_duration_days(text: str) -> int:
    longest = 0
    for match in _DURATION_RE.finditer(text or ""):
        quantity_raw, unit = match.group(1).lower(), match.group(2).lower().rstrip("s")
        quantity = _WORD_NUMBERS.get(quantity_raw)
        if quantity is None:
            try:
                quantity = int(quantity_raw)
            except ValueError:
                continue
        longest = max(longest, quantity * _UNIT_DAYS.get(unit, 1))
    return longest


def assess(text: str, report: SafetyReport) -> TriageResult:
    """Compute the rule-based urgency floor for a message."""
    reasons: list[str] = []

    floor = Urgency.INFO
    if report.red_flags:
        top = report.floor_urgency
        floor = _NAME_TO_LEVEL.get(top, Urgency.URGENT)
        for flag in report.red_flags:
            if flag.urgency == top:
                reasons.append(f"Red flag: {flag.label}")
    elif _SYMPTOM_RE.search(text or ""):
        asking_generally = _GENERAL_QUESTION_RE.search(text or "") and not _COMPLAINT_RE.search(
            text or ""
        )
        if asking_generally:
            reasons.append("Informational question that mentions a health topic")
        else:
            floor = Urgency.SELF_CARE
            reasons.append("Symptom described, no red flag matched")
    else:
        reasons.append("General health question")

    # A complaint that has persisted deserves assessment even when benign.
    days = _max_duration_days(text)
    if days >= 14 and floor < Urgency.ROUTINE:
        floor = Urgency.ROUTINE
        reasons.append(f"Symptom reported for about {days} days")

    level = floor
    escalated = False
    if report.modifiers and Urgency.SELF_CARE <= floor < Urgency.EMERGENCY:
        level = Urgency(min(int(floor) + 1, int(Urgency.EMERGENCY)))
        escalated = True
        labels = ", ".join(m.label for m in report.modifiers)
        reasons.append(f"Escalated one level for higher-risk context ({labels})")

    return TriageResult(
        level=level,
        floor=level,
        reasons=reasons,
        red_flag_ids=[f.id for f in report.red_flags],
        modifier_labels=[m.label for m in report.modifiers],
        escalated_by_modifier=escalated,
    )


def merge_model_level(result: TriageResult, model_level: str | None) -> TriageResult:
    """Combine the model's judgement with the rule floor, taking the maximum."""
    if not model_level:
        return result

    parsed = _NAME_TO_LEVEL.get(str(model_level).strip().upper())
    if parsed is None:
        return result

    result.model_level = parsed
    if parsed > result.floor:
        result.level = parsed
        result.reasons.append(f"Model raised urgency to {parsed.name}")
    elif parsed < result.floor:
        result.reasons.append(
            f"Model suggested {parsed.name}; held at rule floor {result.floor.name}"
        )
    return result


def parse_level_tag(answer: str) -> tuple[str | None, str]:
    """Extract and remove a trailing ``[[URGENCY: X]]`` tag from a reply."""
    match = re.search(r"\[\[\s*URGENCY\s*:\s*([A-Z_]+)\s*\]\]", answer or "", re.IGNORECASE)
    if not match:
        return None, answer
    cleaned = (answer[: match.start()] + answer[match.end():]).strip()
    return match.group(1).strip().upper(), cleaned


def urgency_names() -> list[str]:
    return [level.name for level in Urgency]


__all__ = [
    "Urgency",
    "LEVELS",
    "LevelMeta",
    "TriageResult",
    "assess",
    "merge_model_level",
    "parse_level_tag",
    "urgency_names",
    "URGENCY_ORDER",
]
