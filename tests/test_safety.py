"""Safety layer tests.

These are the tests worth having. Retrieval quality is measured by the gold
set; what needs a hard assertion is the behaviour that must never regress: an
emergency phrase always fires, a negated phrase never does, and identifiers
never reach the log.
"""

from __future__ import annotations

import pytest

from src.safety import load_rules, redact, scan, strip_negated


@pytest.mark.parametrize(
    "message,expected_flag",
    [
        ("I have crushing chest pain going into my left arm", "cardiac_acs"),
        ("her face is drooping and speech is slurred", "stroke"),
        ("he is not breathing", "airway_breathing"),
        ("my child has not passed urine since morning", "severe_dehydration"),
        ("the wound will not stop bleeding", "severe_bleeding_trauma"),
        ("I don't want to live anymore", "self_harm"),
        ("a snake bit my brother", "poisoning_snakebite"),
        ("my 6 week old baby has a fever", "infant_fever"),
        ("I am 32 weeks pregnant with blurred vision and a headache", "obstetric_emergency"),
        ("fever with a stiff neck and a rash that does not fade", "meningitis"),
    ],
)
def test_emergency_phrases_fire(message, expected_flag):
    report = scan(message)
    assert expected_flag in {f.id for f in report.red_flags}
    assert report.is_emergency


@pytest.mark.parametrize(
    "message,forbidden_flag",
    [
        ("I have no chest pain, just a mild fever", "cardiac_acs"),
        ("seene me dard nahi hai, sirf bukhar hai", "cardiac_acs"),
        ("the doctor confirmed it is not a stroke", "stroke"),
        ("I would never hurt myself", "self_harm"),
        ("no bleeding and no breathlessness", "airway_breathing"),
    ],
)
def test_negated_phrases_do_not_fire(message, forbidden_flag):
    report = scan(message)
    assert forbidden_flag not in {f.id for f in report.red_flags}


def test_negation_scope_stops_at_conjunction():
    """"no fever but chest pain" must keep the chest pain."""
    report = scan("I have no fever but I do have chest pain")
    assert "cardiac_acs" in {f.id for f in report.red_flags}


def test_self_harm_is_crisis_not_ambulance():
    report = scan("I have been thinking of ending my life")
    assert report.is_crisis
    assert report.floor_urgency == "EMERGENCY"


def test_context_modifiers_read_conversation_history():
    report = scan("I have a bad headache", history_text="I am 30 weeks pregnant")
    assert "pregnancy" in {m.id for m in report.modifiers}


def test_restricted_intents_detected():
    assert "prescription_request" in {
        r.id for r in scan("which antibiotic should I take").restrictions
    }
    assert "test_interpretation" in {
        r.id for r in scan("my sugar is 145, what does that mean").restrictions
    }


def test_redaction_strips_identifiers():
    redacted = redact("call me on 9876543210 or ab.cd@example.com, aadhaar 1234 5678 9012")
    assert "9876543210" not in redacted
    assert "example.com" not in redacted
    assert "1234 5678 9012" not in redacted


def test_strip_negated_blanks_only_the_scope():
    stripped = strip_negated("no fever, chest pain present")
    assert "fever" not in stripped
    assert "chest pain" in stripped


def test_every_rule_compiles_and_is_documented():
    rule_set = load_rules()
    assert rule_set.rules
    for rule in rule_set.rules:
        assert rule.patterns or rule.raw_patterns, f"{rule.id} has no patterns"
        assert rule.guidance, f"{rule.id} has no user-facing guidance"
        assert rule.urgency in {"EMERGENCY", "URGENT", "ROUTINE"}


def test_every_rule_is_localised():
    """The emergency card is fixed text, so its translations must not drift."""
    for rule in load_rules().rules:
        for language in ("hi", "bn"):
            localised = rule.translations.get(language) or {}
            assert localised.get("label"), f"{rule.id} has no {language} label"
            assert localised.get("guidance"), f"{rule.id} has no {language} guidance"
