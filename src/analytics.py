"""Event logging and the metrics behind the Insights tab.

A health assistant that cannot be measured cannot be improved, so every turn
writes one JSONL row: what was asked (redacted), what the rules decided, which
documents were retrieved, whether the answer was grounded, and how long it
took. The Insights tab reads those rows back.

Two constraints shaped this:

* **No free text is stored.** The query is redacted of direct identifiers and
  only its length and language are kept. Retrieval is recorded as document ids.
* **Failure is silent.** A read-only or full filesystem must never break a
  reply, so every write is wrapped. On Hugging Face Spaces the container
  filesystem is ephemeral, which is fine for a demo; a production deployment
  would point ``CARECOMPASS_LOG_DIR`` at a mounted volume or stream to a
  warehouse table instead.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .config import EVENT_LOG

_WRITE_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(event: dict, path: Path | None = None) -> bool:
    target = Path(path or EVENT_LOG)
    row = {"ts": utc_now(), **event}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except OSError:
        return False


def load_events(path: Path | None = None, limit: int | None = None) -> list[dict]:
    target = Path(path or EVENT_LOG)
    if not target.exists():
        return []
    events: list[dict] = []
    try:
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return events[-limit:] if limit else events


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[index])


def summarise(events: list[dict] | None = None) -> dict:
    events = events if events is not None else load_events()
    if not events:
        return {
            "turns": 0,
            "sessions": 0,
            "emergency_turns": 0,
            "emergency_rate": 0.0,
            "grounded_rate": 0.0,
            "citation_rate": 0.0,
            "restricted_turns": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "modes": {},
            "languages": {},
            "triage": {},
            "topics": {},
            "red_flags": {},
        }

    latencies = [float(e.get("latency_ms", 0)) for e in events]
    grounded = [bool(e.get("grounded")) for e in events]
    cited = [int(e.get("citations", 0)) > 0 for e in events]
    emergencies = [e for e in events if e.get("triage_level") == "EMERGENCY"]

    topics: Counter = Counter()
    red_flags: Counter = Counter()
    for event in events:
        topics.update(event.get("retrieved_docs") or [])
        red_flags.update(event.get("red_flags") or [])

    return {
        "turns": len(events),
        "sessions": len({e.get("session_id") for e in events if e.get("session_id")}),
        "emergency_turns": len(emergencies),
        "emergency_rate": round(100 * len(emergencies) / len(events), 1),
        "grounded_rate": round(100 * sum(grounded) / len(events), 1),
        "citation_rate": round(100 * sum(cited) / len(events), 1),
        "restricted_turns": sum(1 for e in events if e.get("restrictions")),
        "avg_latency_ms": int(mean(latencies)) if latencies else 0,
        "p95_latency_ms": int(_percentile(latencies, 95)),
        "modes": dict(Counter(e.get("mode", "unknown") for e in events)),
        "languages": dict(Counter(e.get("language", "en") for e in events)),
        "triage": dict(Counter(e.get("triage_level", "INFO") for e in events)),
        "topics": dict(topics.most_common(10)),
        "red_flags": dict(red_flags.most_common(10)),
    }


def counts_frame(mapping: dict, name_col: str = "name", value_col: str = "count"):
    """Counter-style dict to a DataFrame for the charts, sorted descending."""
    import pandas as pd

    if not mapping:
        return pd.DataFrame({name_col: [], value_col: []})
    ordered = sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(ordered, columns=[name_col, value_col])


def recent_frame(events: list[dict] | None = None, limit: int = 25):
    """Recent turns for the Insights table."""
    import pandas as pd

    events = (events if events is not None else load_events())[-limit:]
    if not events:
        return pd.DataFrame(
            columns=["time", "lang", "urgency", "mode", "red flags", "sources", "ms"]
        )
    rows = [
        {
            "time": str(e.get("ts", ""))[11:19],
            "lang": e.get("language", ""),
            "urgency": e.get("triage_level", ""),
            "mode": e.get("mode", ""),
            "red flags": ", ".join(e.get("red_flags") or []) or "-",
            "sources": ", ".join((e.get("retrieved_docs") or [])[:2]) or "-",
            "ms": e.get("latency_ms", 0),
        }
        for e in reversed(events)
    ]
    return pd.DataFrame(rows)


def clear_events(path: Path | None = None) -> None:
    target = Path(path or EVENT_LOG)
    try:
        if target.exists():
            target.unlink()
    except OSError:
        pass
