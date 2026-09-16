from __future__ import annotations

from src.safety import scan
from src.triage import Urgency, assess, merge_model_level, parse_level_tag


def level_for(message: str) -> Urgency:
    return assess(message, scan(message)).level


def test_red_flag_sets_emergency_floor():
    assert level_for("crushing chest pain radiating to my jaw") == Urgency.EMERGENCY


def test_plain_symptom_is_self_care():
    assert level_for("mild sore throat since yesterday") == Urgency.SELF_CARE


def test_informational_question_stays_info():
    assert level_for("how much salt per day is safe with high blood pressure") == Urgency.INFO


def test_long_duration_escalates_to_routine():
    assert level_for("I have had a headache every day for three weeks") >= Urgency.ROUTINE


def test_age_is_not_parsed_as_duration():
    """"my 3 year old" must not read as a three-year-long symptom."""
    result = assess("my 3 year old has loose motions since morning", scan("my 3 year old has loose motions since morning"))
    assert "about 1095 days" not in " ".join(result.reasons)


def test_context_modifier_escalates_one_level():
    message = "my 2 year old has a sore throat"
    result = assess(message, scan(message))
    assert result.escalated_by_modifier
    assert result.level == Urgency.ROUTINE


def test_model_may_escalate():
    message = "mild sore throat since yesterday"
    result = merge_model_level(assess(message, scan(message)), "URGENT")
    assert result.level == Urgency.URGENT


def test_model_may_not_de_escalate():
    message = "crushing chest pain radiating to my jaw"
    result = merge_model_level(assess(message, scan(message)), "SELF_CARE")
    assert result.level == Urgency.EMERGENCY
    assert any("held at rule floor" in r for r in result.reasons)


def test_model_garbage_level_is_ignored():
    message = "mild sore throat since yesterday"
    result = merge_model_level(assess(message, scan(message)), "VERY BAD INDEED")
    assert result.level == Urgency.SELF_CARE


def test_parse_level_tag_strips_the_tag():
    level, cleaned = parse_level_tag("Drink fluids and rest.\n\n[[URGENCY: SELF_CARE]]")
    assert level == "SELF_CARE"
    assert "URGENCY" not in cleaned
