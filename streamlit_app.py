"""CareCompass — Streamlit front end.

The third face of the same engine, and the one built for free hosting with a
real model behind it. Streamlit Community Cloud gives 1 GB of RAM, keeps an app
awake for 12 quiet hours rather than 15 minutes, and stores API keys as secrets
the browser never sees — so this is the build where the assistant actually
writes prose instead of quoting passages.

Nothing clinical lives here. Every rule, threshold and piece of medical content
is in `src/` and `data/`, identical to the Gradio and browser builds. This file
is presentation only, which is the point: swapping the UI framework should not
be able to change what the assistant considers an emergency.
"""

from __future__ import annotations

import os

import streamlit as st

# Secrets must reach the environment before `src.config` is imported, because
# provider detection happens at import time.
_KEYS = (
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "CEREBRAS_API_KEY",
    "HF_TOKEN",
    "CARECOMPASS_PROVIDER",
    "CARECOMPASS_MODEL",
    "CARECOMPASS_DISABLE_DENSE",
)
for _key in _KEYS:
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
    except Exception:
        # No secrets.toml at all — normal when running locally.
        break

import json
import time
import uuid

from src.analytics import load_events, summarise
from src.config import APP_NAME, EVAL_DIR, SUPPORTED_LANGUAGES, VERSION
from src.pipeline import MODE_CRISIS, MODE_EMERGENCY, CareCompass
from src.triage import LEVELS, Urgency

st.set_page_config(
    page_title=f"{APP_NAME} — health triage assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLES = [
    "My 3 year old has had loose motions for two days and is drinking less",
    "I get chest tightness when I climb stairs, it goes away when I rest",
    "Cough for three weeks with weight loss and night sweats",
    "My HbA1c came back 7.4 — what does that mean?",
    "I am 32 weeks pregnant with a bad headache and blurred vision",
    "Which antibiotic should I take for a sore throat?",
]

CSS = """
<style>
  .cc-badge { display:inline-block; padding:9px 15px; border-radius:10px; color:#fff;
              font-weight:600; font-size:0.95rem; margin:2px 0 10px; }
  .cc-badge small { display:block; font-weight:400; opacity:0.92; font-size:0.76rem; }
  .cc-pill { display:inline-block; font-size:0.72rem; padding:3px 9px; border-radius:999px;
             border:1px solid rgba(128,128,128,0.35); opacity:0.85; margin:2px 4px 2px 0; }
  .cc-pill.on { border-color:#0e9f6e; color:#0e9f6e; opacity:1; }
  .cc-pill.off { border-color:#c27803; color:#c27803; opacity:1; }
  [data-testid="stChatMessage"] { padding-top:0.4rem; padding-bottom:0.4rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading knowledge base and safety rules…")
def get_assistant() -> CareCompass:
    """Built once per container, not once per rerun.

    Streamlit re-executes this file top to bottom on every interaction, so
    without the cache the corpus would be re-indexed on every keystroke.
    """
    return CareCompass()


ASSISTANT = get_assistant()
HEALTH = ASSISTANT.health()


def badge(level: Urgency, note: str) -> str:
    meta = LEVELS[level]
    return (
        f'<div class="cc-badge" style="background:{meta.colour}">'
        f"{meta.label}<small>{note}</small></div>"
    )


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("session_id", str(uuid.uuid4())[:8])
    st.session_state.setdefault("pending", None)


def reset_conversation() -> None:
    st.session_state.history = []
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.pending = None


init_state()


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### 🧭 {APP_NAME}")
    st.caption("Grounded health guidance with a safety floor you can audit.")

    llm = HEALTH["llm"]
    retrieval = HEALTH["retrieval"]
    corpus = HEALTH["corpus"]

    if llm["available"]:
        engine = f'<span class="cc-pill on">{llm["label"]} · {llm["model"]}</span>'
    else:
        engine = '<span class="cc-pill off">No model key — extractive answers</span>'
    st.markdown(
        engine
        + f'<span class="cc-pill">{retrieval["mode"]} retrieval</span>'
        + f'<span class="cc-pill">{corpus["chunks"]} passages</span>'
        + f'<span class="cc-pill">v{VERSION}</span>',
        unsafe_allow_html=True,
    )

    language_label = st.selectbox(
        "Reply language", list(SUPPORTED_LANGUAGES.values()), index=0
    )
    language = next(
        (code for code, label in SUPPORTED_LANGUAGES.items() if label == language_label),
        "auto",
    )

    st.button("New conversation", width="stretch", on_click=reset_conversation)

    st.divider()
    st.caption("**Try one**")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"ex{i}", width="stretch"):
            st.session_state.pending = example
            st.rerun()

    st.divider()
    st.caption(
        "General health information only. Not a diagnosis, not a prescription. "
        "In an emergency call **112**."
    )


# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------

chat_tab, insights_tab, eval_tab, about_tab = st.tabs(
    ["Chat", "Insights", "Evaluation", "How it works"]
)


def render_turn(entry: dict) -> None:
    """Draw one assistant turn plus its audit panels."""
    st.markdown(entry["content"])
    meta = entry.get("meta")
    if not meta:
        return

    st.markdown(badge(Urgency[meta["level"]], meta["note"]), unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        with st.expander("Sources used"):
            st.markdown(meta["sources"])
    with right:
        with st.expander("Decision trace"):
            st.markdown(meta["trace"])


with chat_tab:
    if not st.session_state.history:
        st.markdown(
            "#### What is going on?\n"
            "Describe a symptom in English, Hindi or Bengali. Every answer shows which "
            "rules fired and which passages it used."
        )

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            if entry["role"] == "assistant":
                render_turn(entry)
            else:
                st.markdown(entry["content"])

    typed = st.chat_input("Describe a health concern…")
    message = typed or st.session_state.pending
    st.session_state.pending = None

    if message:
        with st.chat_message("user"):
            st.markdown(message)

        with st.chat_message("assistant"):
            with st.spinner("Checking the rules and the knowledge base…"):
                result = ASSISTANT.answer(
                    message,
                    history=st.session_state.history,
                    language=language,
                    session_id=st.session_state.session_id,
                )

            # An emergency appears at once. Everything else streams, because a
            # wall of text arriving instantly reads as canned, and because the
            # one case where a reader must not wait is the one case where the
            # answer was already computed in single-digit milliseconds.
            if result.mode in (MODE_EMERGENCY, MODE_CRISIS):
                st.markdown(result.text)
            else:
                def stream():
                    for token in result.text.split(" "):
                        yield token + " "
                        time.sleep(0.012)

                st.write_stream(stream)

            note = f"{result.triage.meta.timeframe} · {result.latency_ms} ms · {result.mode}"
            st.markdown(badge(result.triage.level, note), unsafe_allow_html=True)
            left, right = st.columns(2)
            with left:
                with st.expander("Sources used"):
                    st.markdown(result.sources_markdown())
            with right:
                with st.expander("Decision trace"):
                    st.markdown(result.trace_markdown())

        st.session_state.history.append({"role": "user", "content": message})
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": result.text,
                "meta": {
                    "level": result.triage.level.name,
                    "note": note,
                    "sources": result.sources_markdown(),
                    "trace": result.trace_markdown(),
                },
            }
        )


with insights_tab:
    st.markdown(
        "One redacted row per turn: the rule outcome, the documents retrieved and the "
        "latency. No free text is stored. This is the instrumentation a product team "
        "would watch — escalation rate, grounding rate, and which topics people arrive "
        "with."
    )
    if st.button("Refresh", type="primary"):
        st.rerun()

    events = load_events()
    stats = summarise(events)

    columns = st.columns(6)
    for column, (label, value) in zip(
        columns,
        [
            ("Turns", stats["turns"]),
            ("Sessions", stats["sessions"]),
            ("Emergency", f"{stats['emergency_rate']}%"),
            ("Grounded", f"{stats['grounded_rate']}%"),
            ("Cited", f"{stats['citation_rate']}%"),
            ("p95 latency", f"{stats['p95_latency_ms']} ms"),
        ],
    ):
        column.metric(label, value)

    if not events:
        st.info("No turns yet. Ask something in the Chat tab.")
    else:
        left, right = st.columns(2)
        with left:
            st.caption("Urgency distribution")
            st.bar_chart(stats["triage"], horizontal=True, color="#4b6bfb")
        with right:
            st.caption("Most retrieved documents")
            st.bar_chart(stats["topics"], horizontal=True, color="#0e9f6e")
        if stats["red_flags"]:
            st.caption("Red flags triggered")
            st.bar_chart(stats["red_flags"], horizontal=True, color="#d1242f")

        st.caption("Recent turns")
        st.dataframe(
            [
                {
                    "time": str(e.get("ts", ""))[11:19],
                    "lang": e.get("language", ""),
                    "urgency": e.get("triage_level", ""),
                    "mode": e.get("mode", ""),
                    "red flags": ", ".join(e.get("red_flags") or []) or "-",
                    "sources": ", ".join((e.get("retrieved_docs") or [])[:2]) or "-",
                    "ms": e.get("latency_ms", 0),
                }
                for e in reversed(events[-25:])
            ],
            width="stretch",
            hide_index=True,
        )


with eval_tab:
    st.markdown(
        "### Does it actually work?\n"
        "52 hand-labelled cases in `eval/goldset.yaml` across 11 clinical categories and "
        "three languages. **Emergency recall** is the number to read first: a system that "
        "sends everyone to hospital scores badly on precision and is still safer than one "
        "that misses a stroke, so the two error directions are reported separately and "
        "never averaged into a single accuracy figure."
    )

    if st.button("Run the 52-case gold set now", type="primary"):
        from eval.run_eval import run

        with st.spinner("Running the gold set…"):
            st.session_state.eval_data = run(use_model=False, save=False)

    data = st.session_state.get("eval_data")
    if data is None:
        path = EVAL_DIR / "results.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    if data:
        metrics = data["metrics"]
        st.caption(
            f"{data['cases']} cases · retrieval `{data['retrieval_mode']}` · "
            f"run `{data['ran_at']}`"
        )
        headline = st.columns(4)
        headline[0].metric("Emergency recall", f"{metrics['emergency_recall_pct']}%")
        headline[1].metric("Under-triage", f"{metrics['under_triage_pct']}%")
        headline[2].metric("Over-triage", f"{metrics['over_triage_pct']}%")
        headline[3].metric("Retrieval recall@5", f"{metrics['retrieval_recall_at_k_pct']}%")

        st.dataframe(
            [
                {"metric": k.replace("_pct", " (%)").replace("_", " "), "value": v}
                for k, v in metrics.items()
            ],
            width="stretch",
            hide_index=True,
        )

        failures = data.get("failures") or []
        if failures:
            st.warning(
                f"**{len(failures)} failing cases** — reported rather than relabelled, "
                "because a gold set tuned to match the implementation measures nothing."
            )
            for failure in failures:
                st.markdown(f"- `{failure['id']}` — {failure['reason']}")
        else:
            st.success("All cases passed.")
    else:
        st.info("Press the button to run the gold set.")


with about_tab:
    st.markdown(
        f"""
## How an answer is produced

```
user message
    │
    ├─▶ 1. safety scan        19 rules in data/red_flags.yaml, 3 languages
    │                         negation scoped per-pattern, not per-rule
    ├─▶ 2. triage floor       5 levels, each tied to a timeframe
    │                         pregnancy / infant / elderly context adds one level
    ├─▶ 3. retrieval          BM25 (+ MiniLM embeddings where installed), fused
    │                         with Reciprocal Rank Fusion; a lay-term lexicon
    │                         maps "loose motion" to diarrhoea, ORS, dehydration
    │
    ├─▶ 4a. EMERGENCY or self-harm → fixed text, written in your language
    └─▶ 4b. otherwise → the model answers from the retrieved passages only,
                        must cite [S1]…[Sn], may raise urgency but never lower it
                        ↓ on any failure
                        extractive answer from the same passages
    │
    └─▶ 5. citation validation → disclaimer → redacted event log
```

### The three decisions worth defending

**The safety-critical path contains no language model.** When a red flag fires,
the words you see are fixed text — including the Hindi and Bengali versions. An
ambulance number cannot be paraphrased, and nothing in a later message can talk
the assistant out of it.

**The model can escalate, never de-escalate.** The rules compute a floor; the
model's urgency vote is merged with `max()`. Missing an emergency and sending
someone to hospital unnecessarily are not comparable errors, so the system is
not permitted to trade the first for fewer of the second.

**Every answer degrades instead of failing.** No key, a rate limit, a timeout or
a dead provider all fall through to an extractive answer built from the
retrieved passages and labelled as such.

### Current configuration

- Retrieval: `{HEALTH['retrieval']['mode']}` over {HEALTH['corpus']['chunks']} passages
  from {HEALTH['corpus']['documents']} documents
- Model: `{HEALTH['llm']['label']}` {'· `' + str(HEALTH['llm']['model']) + '`' if HEALTH['llm']['available'] else '(none configured)'}

### Limitations

The corpus is a compact demonstration set written from public health guidance,
not a clinical guideline library. Rules are regular expressions, so a red flag
phrased in a way no pattern anticipates will not fire — which is precisely why
the model is allowed to escalate. Event logs reset when the app restarts.

**It is not a doctor.** No diagnosis, no prescriptions, no reading your test
reports. In an emergency, call **112**.
"""
    )
