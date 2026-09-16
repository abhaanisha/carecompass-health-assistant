"""Lightweight script-based language detection.

A full language-ID model is overkill here and adds a dependency. The three
languages this assistant supports use three distinguishable writing systems,
and the only genuinely ambiguous case is romanised Hindi or Bengali typed in
Latin script, which is handled with a small marker-word list.
"""

from __future__ import annotations

import re

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
BENGALI = re.compile(r"[ঀ-৿]")

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali"}

# Romanised markers. Kept short and high-precision: a false positive changes the
# reply language, which is worse than defaulting to English.
_HI_MARKERS = {
    "hai", "hain", "nahi", "nahin", "mujhe", "mera", "meri", "kya", "kaise",
    "dard", "bukhar", "khansi", "ulti", "dawa", "pet", "sir", "chakkar",
    "kamzori", "saans", "seene", "behosh", "raha", "rahi", "bahut",
}
_BN_MARKERS = {
    "ami", "amar", "hocche", "hoche", "kore", "khub", "jor", "kashi",
    "bomi", "byatha", "petae", "matha", "ghurche", "oshudh", "korchi",
}


def detect_language(text: str) -> str:
    """Return ``en``, ``hi`` or ``bn``."""
    if not text or not text.strip():
        return "en"

    if BENGALI.search(text):
        return "bn"
    if DEVANAGARI.search(text):
        return "hi"

    words = set(re.findall(r"[a-z]+", text.lower()))
    hi_score = len(words & _HI_MARKERS)
    bn_score = len(words & _BN_MARKERS)
    if hi_score >= 2 and hi_score >= bn_score:
        return "hi"
    if bn_score >= 2:
        return "bn"
    return "en"


def resolve_language(selection: str, text: str) -> str:
    """Honour an explicit UI selection, otherwise detect."""
    if selection and selection != "auto" and selection in LANGUAGE_NAMES:
        return selection
    return detect_language(text)
