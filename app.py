"""CareCompass — Gradio application entry point for Hugging Face Spaces.

The interface is built around one idea: a health assistant should be
*inspectable*. Every answer shows the rules that fired, the passages that were
retrieved, and whether a language model was involved at all. A user who does
not care can ignore the panels; a reviewer who does can audit every decision
without reading the logs.
"""

from __future__ import annotations

import json
import uuid

import gradio as gr

from src.analytics import counts_frame, load_events, recent_frame, summarise
from src.config import APP_NAME, APP_TAGLINE, EVAL_DIR, SUPPORTED_LANGUAGES, VERSION
from src.llm import available_providers
from src.pipeline import get_assistant
from src.triage import LEVELS, Urgency

ASSISTANT = get_assistant()
HEALTH = ASSISTANT.health()

EXAMPLES = [
    "My 3 year old has had loose motions for two days and is drinking less",
    "I get chest tightness when I climb stairs, it goes away when I rest",
    "Cough for three weeks with weight loss and night sweats",
    "My HbA1c came back 7.4 — what does that mean?",
    "seene me halka dard hai aur ghabrahat ho rahi hai",
    "I am 32 weeks pregnant with a bad headache and blurred vision",
    "Which antibiotic should I take for a sore throat?",
    "How much salt is safe per day if I have high blood pressure?",
]

CSS = """
.cc-header { padding: 4px 0 12px; }
.cc-header h1 { margin: 0; font-size: 1.6rem; letter-spacing: -0.02em; }
.cc-header p { margin: 4px 0 0; opacity: 0.75; font-size: 0.92rem; }
.cc-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.cc-pill { font-size: 0.74rem; padding: 3px 9px; border-radius: 999px;
           border: 1px solid rgba(128,128,128,0.35); opacity: 0.85; }
.cc-badge { display: inline-block; padding: 8px 14px; border-radius: 10px;
            color: #fff; font-weight: 600; font-size: 0.92rem; }
.cc-badge small { display: block; font-weight: 400; opacity: 0.9; font-size: 0.76rem; }
.cc-kpi { display: flex; flex-wrap: wrap; gap: 10px; }
.cc-kpi div { flex: 1 1 130px; border: 1px solid rgba(128,128,128,0.25);
              border-radius: 10px; padding: 10px 12px; }
.cc-kpi b { display: block; font-size: 1.35rem; line-height: 1.2; }
.cc-kpi span { font-size: 0.74rem; opacity: 0.7; }
footer { display: none !important; }
"""


def header_html() -> str:
    retrieval = HEALTH["retrieval"]
    corpus = HEALTH["corpus"]
    llm = HEALTH["llm"]
    engine = (
        f"{llm['label']} · {llm['model']}" if llm["available"] else "No model key — retrieval only"
    )
    return f"""
<div class="cc-header">
  <h1>{APP_NAME}</h1>
  <p>{APP_TAGLINE}</p>
  <div class="cc-pills">
    <span class="cc-pill">v{VERSION}</span>
    <span class="cc-pill">{retrieval['mode']} retrieval</span>
    <span class="cc-pill">{corpus['chunks']} passages · {corpus['documents']} documents</span>
    <span class="cc-pill">{engine}</span>
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


def respond(message: str, history: list[dict], language_label: str, session_id: str):
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
    note = f"{result.triage.meta.timeframe} · answered via {result.mode}"
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
# insights
# --------------------------------------------------------------------------


def kpi_html(stats: dict) -> str:
    cells = [
        ("Turns", stats["turns"]),
        ("Sessions", stats["sessions"]),
        ("Emergency", f"{stats['emergency_rate']}%"),
        ("Grounded", f"{stats['grounded_rate']}%"),
        ("Cited", f"{stats['citation_rate']}%"),
        ("Avg latency", f"{stats['avg_latency_ms']} ms"),
        ("p95 latency", f"{stats['p95_latency_ms']} ms"),
    ]
    inner = "".join(f"<div><b>{value}</b><span>{label}</span></div>" for label, value in cells)
    return f'<div class="cc-kpi">{inner}</div>'


def refresh_insights():
    events = load_events()
    stats = summarise(events)
    return (
        kpi_html(stats),
        counts_frame(stats["triage"], "urgency", "turns"),
        counts_frame(stats["topics"], "document", "retrievals"),
        counts_frame(stats["red_flags"], "red flag", "hits"),
        recent_frame(events),
    )


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def load_eval_results():
    path = EVAL_DIR / "results.json"
    if not path.exists():
        return (
            "_No saved evaluation. Run `python -m eval.run_eval` from the project root, "
            "or press the button below to run the offline checks now._",
            None,
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return format_eval(data)


def format_eval(data: dict):
    import pandas as pd

    metrics = data.get("metrics", {})
    rows = [
        {"metric": k.replace("_", " "), "value": v}
        for k, v in metrics.items()
        if not isinstance(v, (dict, list))
    ]
    header = (
        f"**{data.get('cases', 0)} cases** · retrieval `{data.get('retrieval_mode', '?')}` "
        f"· run {data.get('ran_at', 'unknown')}"
    )
    failures = data.get("failures") or []
    if failures:
        listed = "\n".join(
            f"- `{f['id']}` — {f['reason']}" for f in failures[:12]
        )
        header += f"\n\n**Failing cases ({len(failures)}):**\n{listed}"
    else:
        header += "\n\nAll cases passed."
    return header, pd.DataFrame(rows)


def run_offline_eval():
    from eval.run_eval import run

    data = run(use_model=False, save=True)
    return format_eval(data)


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

ABOUT = f"""
## How an answer is produced

```
user message
    │
    ├─▶ 1. safety scan        regex rules in data/red_flags.yaml
    │                         negation-aware, multilingual, per-pattern scoping
    ├─▶ 2. triage floor       5 levels, each tied to a timeframe
    │                         context modifiers (pregnancy, infant, elderly…) add one level
    ├─▶ 3. hybrid retrieval   BM25 + MiniLM embeddings fused with Reciprocal Rank Fusion
    │                         lay-term lexicon expands "loose motion" → diarrhoea, ORS
    │
    ├─▶ 4a. EMERGENCY / self-harm → fixed text. No model is called at all.
    └─▶ 4b. otherwise → model answers, constrained to the retrieved passages,
                        allowed to raise urgency and never to lower it
                        no key, or a failed call → extractive answer from the passages
```

## The three design decisions worth defending

**The safety-critical path contains no model.** When a red flag fires, the words
shown are fixed text from `src/prompts.py`. An ambulance number cannot be
paraphrased, and no prompt injection in a later turn can talk the assistant out
of it. Generation is reserved for the cases where phrasing is the hard part.

**The model can escalate, never de-escalate.** Rules compute a floor; the
model's urgency vote is merged with `max()`. That is an asymmetry chosen on
purpose — the failure mode that matters is a missed emergency, not a
conservative one.

**Every answer degrades instead of failing.** No API key, a rate limit, a
timeout or a dead provider all fall through to an extractive answer built from
the retrieved passages, clearly labelled as such.

## What it does not do

It does not diagnose, prescribe, or read your test reports. It does not
remember you between sessions. The knowledge base is a compact demonstration
corpus written from public health guidance (WHO, NHS, MoHFW) — it is not a
clinical guideline set, and a real deployment would need clinician review,
versioned content with an approval workflow, and periodic re-review dates.

Answers in Hindi and Bengali are generated from an English corpus, so the
retrieved passage may be in English even when the reply is not.

## Configuring a model provider

The Space runs without any key. Add one of these as a Space secret to enable
generated answers:

{chr(10).join(f"- `{p['env_var']}` — {p['label']}, default model `{p['default_model']}`" for p in available_providers())}
"""


with gr.Blocks(title=f"{APP_NAME} — health triage assistant", theme=gr.themes.Soft(
    primary_hue="teal", secondary_hue="slate"
), css=CSS) as demo:
    session_state = gr.State(str(uuid.uuid4())[:8])
    gr.HTML(header_html())

    with gr.Tabs():
        # ---------------------------------------------------------------- chat
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        type="messages",
                        height=480,
                        show_label=False,
                        avatar_images=(None, None),
                        placeholder=(
                            "<div style='text-align:center;opacity:0.6'>"
                            "<h3>Describe a health concern</h3>"
                            "<p>General information only. In an emergency, call 112.</p>"
                            "</div>"
                        ),
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="What is going on? You can write in English, Hindi or Bengali.",
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

        # ------------------------------------------------------------ insights
        with gr.Tab("Insights"):
            gr.Markdown(
                "Operational view over the event log. Every turn writes one row: "
                "no free text is stored, only the rule outcome, the documents retrieved "
                "and the latency. This is the instrumentation a product team would "
                "actually watch — escalation rate, grounding rate, and which topics "
                "people arrive with."
            )
            refresh = gr.Button("Refresh", variant="primary")
            kpis = gr.HTML(kpi_html(summarise([])))
            with gr.Row():
                triage_plot = gr.BarPlot(
                    counts_frame({}, "urgency", "turns"),
                    x="urgency", y="turns", title="Urgency distribution", height=260,
                )
                topic_plot = gr.BarPlot(
                    counts_frame({}, "document", "retrievals"),
                    x="document", y="retrievals", title="Most retrieved documents", height=260,
                )
            flag_plot = gr.BarPlot(
                counts_frame({}, "red flag", "hits"),
                x="red flag", y="hits", title="Red flags triggered", height=240,
            )
            recent = gr.DataFrame(recent_frame([]), label="Recent turns", wrap=True)

        # ---------------------------------------------------------- evaluation
        with gr.Tab("Evaluation"):
            gr.Markdown(
                "### Does it actually work?\n"
                "A hand-labelled gold set in `eval/goldset.yaml` checks the things that "
                "matter: does every emergency get caught (recall on red flags is the "
                "metric to optimise, not accuracy), does retrieval surface the right "
                "document, and does the assistant refuse to prescribe. The offline "
                "checks below need no API key."
            )
            eval_run = gr.Button("Run offline evaluation", variant="primary")
            eval_summary = gr.Markdown()
            eval_table = gr.DataFrame(label="Metrics", wrap=True)

        # --------------------------------------------------------------- about
        with gr.Tab("How it works"):
            gr.Markdown(ABOUT)

    # -- wiring ------------------------------------------------------------
    submit_args = dict(
        fn=respond,
        inputs=[msg, chatbot, language, session_state],
        outputs=[chatbot, msg, badge, sources, trace],
    )
    msg.submit(**submit_args)
    send.click(**submit_args)
    clear.click(
        reset_chat, outputs=[chatbot, badge, sources, trace, session_state]
    )
    refresh.click(
        refresh_insights,
        outputs=[kpis, triage_plot, topic_plot, flag_plot, recent],
    )
    eval_run.click(run_offline_eval, outputs=[eval_summary, eval_table])
    demo.load(load_eval_results, outputs=[eval_summary, eval_table])
    demo.load(refresh_insights, outputs=[kpis, triage_plot, topic_plot, flag_plot, recent])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()
