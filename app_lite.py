"""CareCompass — browser-only build.

This is the entry point for the Gradio-Lite Space: the same `src/` package,
running under Pyodide inside the visitor's browser. There is no server, no API
key and no request leaving the tab.

Two differences from `app.py`, both forced by the runtime and both stated in
the UI rather than hidden:

* **Lexical retrieval only.** `sentence-transformers` needs torch, which has no
  WebAssembly build. The gold set scores identically without the dense half on
  this corpus, so the demonstration loses nothing measurable.
* **No generated prose.** A browser cannot hold an API key safely, so answers
  come from the extractive path — quoted from the retrieved passages. The
  deterministic engine, which is the part of this project worth showing, is
  fully intact and runs at exactly the same speed.

Charts are hand-rolled HTML rather than `gr.BarPlot`, to keep pandas and altair
off the critical path while Pyodide boots.

Runs locally too: `python app_lite.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("CARECOMPASS_DISABLE_DENSE", "1")

import json
import uuid
from pathlib import Path

import gradio as gr

from src.analytics import load_events, summarise
from src.config import APP_NAME, EVAL_DIR, SUPPORTED_LANGUAGES, VERSION
from src.pipeline import get_assistant
from src.triage import LEVELS, Urgency

ASSISTANT = get_assistant()
HEALTH = ASSISTANT.health()

EXAMPLES = [
    "I have crushing chest pain going into my left arm",
    "I have no chest pain, just a mild fever and body ache",
    "my 6 week old baby has a fever",
    "seene me dard ho raha hai aur paseena aa raha hai",
    "বুকে ব্যথা হচ্ছে আর ঘাম হচ্ছে",
    "my 3 year old has loose motions since morning",
    "which antibiotic should I take for a sore throat?",
    "how much salt per day is safe with high blood pressure?",
]

CSS = """
.cc-header { padding: 2px 0 10px; }
.cc-header h1 { margin: 0; font-size: 1.55rem; letter-spacing: -0.02em; }
.cc-header p { margin: 4px 0 0; opacity: 0.75; font-size: 0.9rem; }
.cc-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.cc-pill { font-size: 0.72rem; padding: 3px 9px; border-radius: 999px;
           border: 1px solid rgba(128,128,128,0.35); opacity: 0.85; }
.cc-pill.on { border-color: #0e9f6e; color: #0e9f6e; opacity: 1; }
.cc-badge { display: inline-block; padding: 8px 14px; border-radius: 10px;
            color: #fff; font-weight: 600; font-size: 0.92rem; }
.cc-badge small { display: block; font-weight: 400; opacity: 0.92; font-size: 0.76rem; }
.cc-kpi { display: flex; flex-wrap: wrap; gap: 10px; }
.cc-kpi div { flex: 1 1 120px; border: 1px solid rgba(128,128,128,0.25);
              border-radius: 10px; padding: 10px 12px; }
.cc-kpi b { display: block; font-size: 1.3rem; line-height: 1.2; }
.cc-kpi span { font-size: 0.72rem; opacity: 0.7; }
.cc-bars { display: flex; flex-direction: column; gap: 5px; margin-bottom: 18px; }
.cc-bar { display: grid; grid-template-columns: 150px 1fr 40px; align-items: center;
          gap: 10px; font-size: 0.8rem; }
.cc-bar span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; opacity: 0.85; }
.cc-bar div { background: rgba(128,128,128,0.16); border-radius: 4px; height: 15px; }
.cc-bar i { display: block; height: 15px; border-radius: 4px; }
.cc-bar b { text-align: right; font-variant-numeric: tabular-nums; }
.cc-muted { opacity: 0.6; font-size: 0.85rem; }
footer { display: none !important; }
"""


def header_html() -> str:
    corpus = HEALTH["corpus"]
    return f"""
<div class="cc-header">
  <h1>{APP_NAME}</h1>
  <p>Grounded health guidance with a safety floor you can audit —
     running entirely inside your browser.</p>
  <div class="cc-pills">
    <span class="cc-pill">v{VERSION}</span>
    <span class="cc-pill on">no server · no API key · nothing leaves this tab</span>
    <span class="cc-pill">{corpus['chunks']} passages · {corpus['documents']} documents</span>
    <span class="cc-pill">BM25 retrieval</span>
  </div>
</div>
"""


def badge_html(level: Urgency = Urgency.INFO, note: str = "Ask a question to begin") -> str:
    meta = LEVELS[level]
    return (
        f'<div class="cc-badge" style="background:{meta.colour}">'
        f"{meta.label}<small>{note}</small></div>"
    )


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------


def respond(message: str, history: list, language_label: str, session_id: str):
    message = (message or "").strip()
    if not message:
        return history, "", gr.update(), gr.update(), gr.update()

    language = next(
        (code for code, label in SUPPORTED_LANGUAGES.items() if label == language_label),
        "auto",
    )
    result = ASSISTANT.answer(message, history=history, language=language, session_id=session_id)

    new_history = [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.text},
    ]
    note = f"{result.triage.meta.timeframe} · {result.latency_ms} ms · {result.mode}"
    return (
        new_history,
        "",
        badge_html(result.triage.level, note),
        result.sources_markdown(),
        result.trace_markdown(),
    )


def reset_chat():
    return (
        [],
        badge_html(),
        "_Sources for the current answer appear here._",
        "_The rule trace for the current answer appears here._",
        str(uuid.uuid4())[:8],
    )


# --------------------------------------------------------------------------
# insights — hand-rolled charts, no pandas
# --------------------------------------------------------------------------


def bars_html(mapping: dict, title: str, colour: str = "#0e9f6e") -> str:
    if not mapping:
        return f"<h4>{title}</h4><p class='cc-muted'>No data yet — ask something in the Chat tab.</p>"
    largest = max(mapping.values()) or 1
    rows = "".join(
        f"<div class='cc-bar'><span title='{name}'>{name}</span>"
        f"<div><i style='width:{100 * count / largest:.0f}%;background:{colour}'></i></div>"
        f"<b>{count}</b></div>"
        for name, count in sorted(mapping.items(), key=lambda kv: -kv[1])
    )
    return f"<h4>{title}</h4><div class='cc-bars'>{rows}</div>"


def kpi_html(stats: dict) -> str:
    cells = [
        ("Turns", stats["turns"]),
        ("Sessions", stats["sessions"]),
        ("Emergency", f"{stats['emergency_rate']}%"),
        ("Grounded", f"{stats['grounded_rate']}%"),
        ("Avg latency", f"{stats['avg_latency_ms']} ms"),
        ("p95 latency", f"{stats['p95_latency_ms']} ms"),
    ]
    inner = "".join(f"<div><b>{value}</b><span>{label}</span></div>" for label, value in cells)
    return f'<div class="cc-kpi">{inner}</div>'


RECENT_HEADERS = ["time", "lang", "urgency", "red flags", "sources", "ms"]


def recent_rows(events: list, limit: int = 20) -> list:
    rows = []
    for event in reversed(events[-limit:]):
        rows.append(
            [
                str(event.get("ts", ""))[11:19],
                event.get("language", ""),
                event.get("triage_level", ""),
                ", ".join(event.get("red_flags") or []) or "-",
                ", ".join((event.get("retrieved_docs") or [])[:2]) or "-",
                event.get("latency_ms", 0),
            ]
        )
    return rows


def refresh_insights():
    events = load_events()
    stats = summarise(events)
    charts = (
        bars_html(stats["triage"], "Urgency distribution", "#4b6bfb")
        + bars_html(stats["topics"], "Most retrieved documents", "#0e9f6e")
        + bars_html(stats["red_flags"], "Red flags triggered", "#d1242f")
    )
    return kpi_html(stats), charts, recent_rows(events)


# --------------------------------------------------------------------------
# evaluation — the full gold set, run in the browser
# --------------------------------------------------------------------------

METRIC_HEADERS = ["metric", "value"]


def format_eval(data: dict):
    rows = [
        [key.replace("_pct", " (%)").replace("_", " "), value]
        for key, value in (data.get("metrics") or {}).items()
    ]
    header = (
        f"**{data.get('cases', 0)} cases** · retrieval `{data.get('retrieval_mode', '?')}` "
        f"· run `{data.get('ran_at', 'unknown')}`"
    )
    failures = data.get("failures") or []
    if failures:
        listed = "\n".join(f"- `{f['id']}` — {f['reason']}" for f in failures[:12])
        header += f"\n\n**Failing cases ({len(failures)}):**\n{listed}"
    else:
        header += "\n\nAll cases passed."
    return header, rows


def load_eval_results():
    path = EVAL_DIR / "results.json"
    if not path.exists():
        return "_Press the button to run the gold set._", []
    return format_eval(json.loads(path.read_text(encoding="utf-8")))


def run_offline_eval():
    from eval.run_eval import run

    return format_eval(run(use_model=False, save=False, lexical_only=True))


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

ABOUT = """
## This page is the whole application

There is no backend. When you loaded this tab your browser downloaded a Python
runtime (Pyodide), the knowledge base, the safety rules and about 1,500 lines of
application code, and it has been running them locally ever since. Nothing you
type is transmitted anywhere. You can disconnect from the network and the
assistant keeps working.

That is possible because the parts of this system that matter are deterministic
and dependency-light — regular expressions, BM25 and a rule engine — not a model
call.

## How a turn is processed

```
user message
    │
    ├─▶ 1. safety scan        19 rules in data/red_flags.yaml, 3 languages
    │                         negation scoped per-pattern, not per-rule
    ├─▶ 2. triage floor       5 levels, each tied to a timeframe
    │                         pregnancy / infant / elderly context adds one level
    ├─▶ 3. retrieval          BM25 with a lay-term lexicon, stemming, and
    │                         IDF-weighted grounding to detect out-of-scope
    │
    ├─▶ 4a. EMERGENCY or self-harm → fixed card, written in your language
    └─▶ 4b. otherwise → answer assembled from the retrieved passages, with citations
```

**The safety-critical path contains no language model.** When a red flag fires,
the words you see are fixed text, including the Hindi and Bengali versions. An
ambulance number cannot be paraphrased, and nothing in a later message can talk
the assistant out of it.

**The rule engine sets a floor a model may raise but never lower.** In the
server build, a language model writes the prose and returns its own urgency
vote, merged with `max()` against the floor computed here. That asymmetry is
deliberate: a missed emergency and an unnecessary hospital trip are not
comparable errors.

## What this build does not do

- **No generated prose.** A browser cannot hold an API key safely, so answers
  are quoted from the knowledge base rather than written for your question. The
  server build adds that layer on top of the identical engine.
- **Lexical retrieval only.** The dense half needs torch, which has no
  WebAssembly build. On the 52-case gold set the scores are identical either
  way.
- **Nothing is remembered.** Reload the page and the event log is empty again.

## It is not a doctor

CareCompass gives general health information from a curated public-health
corpus. It does not diagnose, prescribe, or read your test reports. In an
emergency, call **112**.
"""


with gr.Blocks(
    title=f"{APP_NAME} — health triage in your browser",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"),
    css=CSS,
) as demo:
    session_state = gr.State(str(uuid.uuid4())[:8])
    gr.HTML(header_html())

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        type="messages",
                        height=430,
                        show_label=False,
                        placeholder=(
                            "<div style='text-align:center;opacity:0.6'>"
                            "<h3>Describe a health concern</h3>"
                            "<p>English, Hindi or Bengali. General information only — "
                            "in an emergency, call 112.</p></div>"
                        ),
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="What is going on?",
                            show_label=False,
                            scale=7,
                            autofocus=True,
                        )
                        send = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        language = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.values()),
                            value="Auto-detect",
                            label="Reply language",
                            scale=2,
                        )
                        clear = gr.Button("New conversation", scale=1)
                    gr.Examples(examples=EXAMPLES, inputs=msg, label="Try one of these")

                with gr.Column(scale=2):
                    badge = gr.HTML(badge_html())
                    with gr.Accordion("Sources used", open=True):
                        sources = gr.Markdown("_Sources for the current answer appear here._")
                    with gr.Accordion("Decision trace", open=False):
                        trace = gr.Markdown("_The rule trace for the current answer appears here._")

        with gr.Tab("Insights"):
            gr.Markdown(
                "One redacted row per turn: the rule outcome, the documents retrieved and "
                "the latency. No free text is stored. In the server build this is the "
                "instrumentation a product team watches — escalation rate, grounding rate, "
                "and which topics people arrive with. Here it lives in browser memory and "
                "resets on reload."
            )
            refresh = gr.Button("Refresh", variant="primary")
            kpis = gr.HTML(kpi_html(summarise([])))
            charts = gr.HTML()
            recent = gr.DataFrame(
                headers=RECENT_HEADERS, value=[], label="Recent turns", wrap=True
            )

        with gr.Tab("Evaluation"):
            gr.Markdown(
                "### Does it actually work?\n"
                "52 hand-labelled cases in `eval/goldset.yaml` covering 11 clinical "
                "categories and three languages. The button below runs **the entire gold "
                "set in your browser** — the same harness used as a CI gate, which exits "
                "non-zero if emergency recall drops below 100%.\n\n"
                "Emergency recall is the number to read first. A system that sends "
                "everyone to hospital scores badly on precision and is still safer than "
                "one that misses a stroke, so the two error directions are reported "
                "separately and never averaged."
            )
            eval_run = gr.Button("Run the 52-case gold set now", variant="primary")
            eval_summary = gr.Markdown()
            eval_table = gr.DataFrame(headers=METRIC_HEADERS, value=[], label="Metrics", wrap=True)

        with gr.Tab("How it works"):
            gr.Markdown(ABOUT)

    submit_args = dict(
        fn=respond,
        inputs=[msg, chatbot, language, session_state],
        outputs=[chatbot, msg, badge, sources, trace],
    )
    msg.submit(**submit_args)
    send.click(**submit_args)
    clear.click(reset_chat, outputs=[chatbot, badge, sources, trace, session_state])
    refresh.click(refresh_insights, outputs=[kpis, charts, recent])
    eval_run.click(run_offline_eval, outputs=[eval_summary, eval_table])
    demo.load(load_eval_results, outputs=[eval_summary, eval_table])
    demo.load(refresh_insights, outputs=[kpis, charts, recent])

demo.launch()
