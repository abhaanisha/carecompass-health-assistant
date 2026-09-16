"""CareCompass — Streamlit front end.

Laid out the way people now expect a chat product to look: one narrow centred
column, a collapsed sidebar, suggestion cards on the empty state, and the input
pinned at the bottom. Everything that is not the conversation — analytics, the
evaluation harness, the architecture notes — moved behind sidebar navigation so
the chat surface has nothing competing with it.

Two deliberate departures from a plain chatbot, because they are the point of
the project rather than decoration:

* a small urgency chip under each answer, colour-coded to the triage level
* one collapsed "Sources and decision trace" control per answer, holding the
  passages used and the rules that fired

The medical disclaimer is rendered once under the input instead of repeated
under every message. Repetition is how a disclaimer stops being read; the
pipeline still carries it in `Answer.text` for any non-chat consumer.

Nothing clinical lives in this file. Every rule, threshold and piece of medical
content is in `src/` and `data/`, identical to the Gradio and browser builds.
"""

from __future__ import annotations

import os

import streamlit as st

# Secrets must reach the environment before `src.config` is imported, because
# provider detection happens at import time.
_KEYS = (
    "GROQ_API_KEY",
    "CEREBRAS_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
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
    page_title=f"{APP_NAME} — health guidance",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SUGGESTIONS = [
    ("Loose motions in a toddler", "My 3 year old has had loose motions for two days and is drinking less"),
    ("Chest tightness on the stairs", "I get chest tightness when I climb stairs, it goes away when I rest"),
    ("A cough that will not clear", "Cough for three weeks with weight loss and night sweats"),
    ("Making sense of a lab result", "My HbA1c came back 7.4 — what does that mean?"),
]

CSS = """
<style>
  /* Streamlit's chrome competes with the conversation. */
  [data-testid="stHeader"] { background: transparent; height: 2.2rem; }
  [data-testid="stToolbar"] { right: 0.5rem; }
  #MainMenu, footer { visibility: hidden; }

  .block-container { padding-top: 2.2rem; padding-bottom: 6rem; max-width: 46rem; }

  [data-testid="stChatMessage"] {
      background: transparent; padding: 0.35rem 0; gap: 0.7rem;
  }
  [data-testid="stChatMessage"] p { line-height: 1.62; }

  /* Hero, shown only on an empty conversation. */
  .cc-hero { text-align: center; margin: 3.4rem 0 1.9rem; }
  .cc-hero h1 { font-size: 1.72rem; font-weight: 650; margin: 0 0 0.5rem;
                letter-spacing: -0.02em; }
  .cc-hero p { opacity: 0.62; margin: 0; font-size: 0.95rem; }

  /* Suggestion cards. */
  .stButton > button {
      width: 100%; text-align: left; white-space: normal; height: auto;
      padding: 0.68rem 0.85rem; border-radius: 12px; font-weight: 450;
      font-size: 0.86rem; line-height: 1.35;
  }
  .stButton > button:hover { border-color: #0e9f6e; color: inherit; }

  /* Urgency chip + audit control. */
  .cc-chip { display: inline-block; padding: 3px 11px; border-radius: 999px;
             color: #fff; font-size: 0.72rem; font-weight: 600;
             letter-spacing: 0.02em; margin: 0.15rem 0 0.1rem; }
  .cc-chip span { font-weight: 400; opacity: 0.88; }
  [data-testid="stExpander"] { border: none; }
  [data-testid="stExpander"] summary { font-size: 0.78rem; opacity: 0.62; padding-left: 0; }
  [data-testid="stExpander"] summary:hover { opacity: 1; }

  .cc-foot { text-align: center; font-size: 0.72rem; opacity: 0.5;
             margin: 0.45rem 0 0; }
  .cc-note { font-size: 0.78rem; opacity: 0.68; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading knowledge base and safety rules…")
def get_assistant(version: str = VERSION) -> CareCompass:
    """Built once per container, not once per rerun.

    Streamlit re-executes this file top to bottom on every interaction, so
    without the cache the corpus would be re-indexed on every keystroke.

    ``version`` is part of the cache key and is never read. Without it, a
    deploy that reloads the script while keeping the cache alive leaves a stale
    assistant object behind — one whose class predates the new code. That is
    not hypothetical: it shipped, and the symptom was an AttributeError on a
    field the deployed UI had every right to expect.
    """
    return CareCompass()


ASSISTANT = get_assistant(VERSION)
HEALTH = ASSISTANT.health()


def chip(level: Urgency, note: str) -> str:
    meta = LEVELS[level]
    return (
        f'<div class="cc-chip" style="background:{meta.colour}">'
        f"{meta.badge} <span>· {note}</span></div>"
    )


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("session_id", str(uuid.uuid4())[:8])
    st.session_state.setdefault("pending", None)


def new_chat() -> None:
    st.session_state.history = []
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.pending = None


init_state()


# --------------------------------------------------------------------------
# sidebar: navigation and settings only
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"#### 🧭 {APP_NAME}")
    st.caption("Health guidance with a safety floor you can audit")

    page = st.radio(
        "View",
        ["Chat", "Insights", "Evaluation", "How it works"],
        label_visibility="collapsed",
    )

    st.button("New chat", width="stretch", on_click=new_chat)

    language_label = st.selectbox("Reply language", list(SUPPORTED_LANGUAGES.values()))
    language = next(
        (code for code, label in SUPPORTED_LANGUAGES.items() if label == language_label),
        "auto",
    )

    st.divider()
    llm = HEALTH["llm"]
    if llm["available"]:
        st.caption(f"**Model** · {llm['label']} · `{llm['model']}`")
    else:
        st.caption("**No model key** · answers are quoted from the corpus")
    st.caption(
        f"**Retrieval** · {HEALTH['retrieval']['mode']}, "
        f"{HEALTH['corpus']['chunks']} passages · v{VERSION}"
    )


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------


def render_audit(meta: dict) -> None:
    st.markdown(chip(Urgency[meta["level"]], meta["note"]), unsafe_allow_html=True)
    with st.expander(meta["audit_label"]):
        st.markdown("**Sources**")
        st.markdown(meta["sources"])
        st.markdown("**Decision trace**")
        st.markdown(meta["trace"])


def render_chat() -> None:
    history = st.session_state.history

    if not history:
        st.markdown(
            '<div class="cc-hero"><h1>What is going on?</h1>'
            "<p>Describe a symptom in English, Hindi or Bengali. "
            "Every answer shows the rules that fired and the passages it used.</p></div>",
            unsafe_allow_html=True,
        )
        left, right = st.columns(2, gap="small")
        for i, (label, prompt) in enumerate(SUGGESTIONS):
            with (left, right)[i % 2]:
                if st.button(label, key=f"sg{i}"):
                    st.session_state.pending = prompt
                    st.rerun()

    for entry in history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("meta"):
                render_audit(entry["meta"])

    typed = st.chat_input("Describe a health concern…")
    message = typed or st.session_state.pending
    st.session_state.pending = None

    st.markdown(
        '<p class="cc-foot">General health information, not a diagnosis or a '
        "prescription. In an emergency call 112.</p>",
        unsafe_allow_html=True,
    )

    if not message:
        return

    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        with st.spinner("Checking the rules and the knowledge base…"):
            result = ASSISTANT.answer(
                message,
                history=history,
                language=language,
                session_id=st.session_state.session_id,
            )

        # Tolerate an Answer from an older build that has no `body` field, so a
        # half-applied deploy degrades to a slightly longer message rather than
        # a traceback in the user's face.
        body = getattr(result, "body", None) or result.text

        # An emergency appears at once. Everything else streams, because a wall
        # of text arriving instantly reads as canned — and because the one case
        # where a reader must not wait is the one already computed in
        # single-digit milliseconds.
        if result.mode in (MODE_EMERGENCY, MODE_CRISIS):
            st.markdown(body)
        else:
            def stream():
                for token in body.split(" "):
                    yield token + " "
                    time.sleep(0.012)

            st.write_stream(stream)

        used = sum(1 for c in result.citations if c.used)
        meta = {
            "level": result.triage.level.name,
            "note": f"{result.triage.meta.timeframe} · {result.latency_ms} ms",
            "audit_label": (
                f"{used} source{'' if used == 1 else 's'} cited · decision trace"
                if used
                else "Decision trace"
            ),
            "sources": result.sources_markdown(),
            "trace": result.trace_markdown(),
        }
        render_audit(meta)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": body, "meta": meta})


# --------------------------------------------------------------------------
# insights
# --------------------------------------------------------------------------


def render_insights() -> None:
    st.subheader("Insights")
    st.markdown(
        '<p class="cc-note">One redacted row per turn: the rule outcome, the documents '
        "retrieved and the latency. No free text is stored. This is the instrumentation a "
        "product team would watch — escalation rate, grounding rate, and which topics "
        "people arrive with.</p>",
        unsafe_allow_html=True,
    )

    events = load_events()
    stats = summarise(events)

    row = st.columns(4)
    for column, (label, value) in zip(
        row,
        [
            ("Turns", stats["turns"]),
            ("Emergency", f"{stats['emergency_rate']}%"),
            ("Grounded", f"{stats['grounded_rate']}%"),
            ("p95 latency", f"{stats['p95_latency_ms']} ms"),
        ],
    ):
        column.metric(label, value)

    if not events:
        st.info("No turns yet. Ask something in the Chat view.")
        return

    st.caption("Urgency distribution")
    st.bar_chart(stats["triage"], horizontal=True, color="#4b6bfb", height=190)
    st.caption("Most retrieved documents")
    st.bar_chart(stats["topics"], horizontal=True, color="#0e9f6e", height=230)
    if stats["red_flags"]:
        st.caption("Red flags triggered")
        st.bar_chart(stats["red_flags"], horizontal=True, color="#d1242f", height=190)

    with st.expander("Recent turns"):
        st.dataframe(
            [
                {
                    "time": str(e.get("ts", ""))[11:19],
                    "lang": e.get("language", ""),
                    "urgency": e.get("triage_level", ""),
                    "mode": e.get("mode", ""),
                    "red flags": ", ".join(e.get("red_flags") or []) or "-",
                    "ms": e.get("latency_ms", 0),
                }
                for e in reversed(events[-25:])
            ],
            width="stretch",
            hide_index=True,
        )


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def render_evaluation() -> None:
    st.subheader("Does it actually work?")
    st.markdown(
        '<p class="cc-note">52 hand-labelled cases across 11 clinical categories and three '
        "languages. <b>Emergency recall</b> is the number to read first: a system that sends "
        "everyone to hospital scores badly on precision and is still safer than one that "
        "misses a stroke, so the two error directions are reported separately and never "
        "averaged into a single accuracy figure.</p>",
        unsafe_allow_html=True,
    )

    if st.button("Run the 52-case gold set now", type="primary"):
        from eval.run_eval import run

        with st.spinner("Running the gold set…"):
            st.session_state.eval_data = run(use_model=False, save=False)

    data = st.session_state.get("eval_data")
    if data is None:
        path = EVAL_DIR / "results.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if not data:
        st.info("Press the button to run the gold set.")
        return

    metrics = data["metrics"]
    row = st.columns(4)
    row[0].metric("Emergency recall", f"{metrics['emergency_recall_pct']}%")
    row[1].metric("Under-triage", f"{metrics['under_triage_pct']}%")
    row[2].metric("Over-triage", f"{metrics['over_triage_pct']}%")
    row[3].metric("Retrieval recall@5", f"{metrics['retrieval_recall_at_k_pct']}%")
    st.caption(f"{data['cases']} cases · retrieval `{data['retrieval_mode']}` · {data['ran_at']}")

    with st.expander("All metrics"):
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
            f"**{len(failures)} failing cases** — reported rather than relabelled, because "
            "a gold set tuned to match the implementation measures nothing."
        )
        for failure in failures:
            st.markdown(f"- `{failure['id']}` — {failure['reason']}")
    else:
        st.success("All cases passed.")


# --------------------------------------------------------------------------
# about
# --------------------------------------------------------------------------

ABOUT = f"""
### How an answer is produced

```
user message
    │
    ├─▶ 1. safety scan     19 rules, 3 languages, negation scoped per pattern
    ├─▶ 2. triage floor    5 levels, each tied to a timeframe
    ├─▶ 3. retrieval       BM25 (+ embeddings where installed), RRF fusion
    │
    ├─▶ 4a. emergency or self-harm  → fixed text, in your language
    ├─▶ 4b. symptom too vague       → ask what a clinician would ask
    └─▶ 4c. otherwise               → model answers from the passages only,
                                      must cite, may raise urgency, never lower it
```

**The safety-critical path contains no language model.** When a red flag fires,
the words you see are fixed text, including the Hindi and Bengali versions. An
ambulance number cannot be paraphrased, and nothing in a later message can talk
the assistant out of it.

**The model can escalate, never de-escalate.** The rules compute a floor; the
model's urgency vote is merged with `max()`. Missing an emergency and sending
someone to hospital unnecessarily are not comparable errors, so the system is
not permitted to trade the first for fewer of the second.

**Every answer degrades instead of failing.** No key, a rate limit, a timeout or
a dead provider all fall through to an answer quoted from the retrieved
passages, labelled as such.

### Current configuration

- Retrieval: `{HEALTH['retrieval']['mode']}` over {HEALTH['corpus']['chunks']} passages
  from {HEALTH['corpus']['documents']} documents
- Model: `{HEALTH['llm']['label']}`{' · `' + str(HEALTH['llm']['model']) + '`' if HEALTH['llm']['available'] else ' (none configured)'}

### Limitations

The corpus is a compact demonstration set written from public health guidance,
not a clinical guideline library. Rules are regular expressions, so a red flag
phrased in a way no pattern anticipates will not fire — which is exactly why the
model is allowed to escalate. Event logs reset when the app restarts.

**It is not a doctor.** No diagnosis, no prescriptions, no reading your test
reports. In an emergency, call **112**.
"""


if page == "Chat":
    render_chat()
elif page == "Insights":
    render_insights()
elif page == "Evaluation":
    render_evaluation()
else:
    st.subheader("How it works")
    st.markdown(ABOUT)
