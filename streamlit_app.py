"""CareCompass — Streamlit front end.

Laid out the way people now expect a chat product to look: one narrow centred
column, a collapsed sidebar, suggestion cards on the empty state, and the input
pinned at the bottom. Everything that is not the conversation — analytics, the
evaluation harness — moved behind sidebar navigation so the chat surface has
nothing competing with it.

Three deliberate departures from a plain chatbot, because they are the point of
the project rather than decoration:

* a small urgency chip under each answer, colour-coded to the triage level
* one collapsed "Sources and decision trace" control per answer, holding the
  passages used and the rules that fired, with curated and just-fetched sources
  kept visibly apart
* follow-up chips under the latest answer, so the next question costs a click

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
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    "SERPER_API_KEY",
    "CARECOMPASS_PROVIDER",
    "CARECOMPASS_MODEL",
    "CARECOMPASS_DISABLE_DENSE",
    "CARECOMPASS_WEB_MODE",
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

AUTHOR = "Abha Singh Sardar"
AUTHOR_AFFILIATION = "IISc"
AUTHOR_URL = "https://abhaanisha.github.io/"

st.set_page_config(
    page_title=f"{APP_NAME} — health guidance",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

#: Label, the message it sends, and a Material icon. The four are chosen to
#: show the range in one screen rather than to flatter the system: a
#: paediatric case, a cardiac red flag, a chronic-cough case that should
#: escalate, and a lab-result question the assistant is required to decline.
SUGGESTIONS = [
    (
        "Loose motions in a toddler",
        "My 3 year old has had loose motions for two days and is drinking less",
        ":material/child_care:",
    ),
    (
        "Chest tightness on the stairs",
        "I get chest tightness when I climb stairs, it goes away when I rest",
        ":material/monitor_heart:",
    ),
    (
        "A cough that will not clear",
        "Cough for three weeks with weight loss and night sweats",
        ":material/pulmonology:",
    ),
    (
        "Making sense of a lab result",
        "My HbA1c came back 7.4 — what does that mean?",
        ":material/science:",
    ),
]

# --------------------------------------------------------------------------
# styling
#
# Streamlit's defaults are competent and completely anonymous, which is the
# wrong register for something a person opens while worried about a child. The
# sheet below is doing three specific jobs, not decorating:
#
# 1. Establishing a type hierarchy. A serif display face for the one line that
#    asks the question, a humanist sans everywhere else, and a measure capped
#    near 68 characters so an answer reads like prose instead of a form.
# 2. Making the urgency chip legible without shouting. Solid fills on five
#    levels turn every answer into an alarm and the red one stops meaning
#    anything; a tinted ground with a saturated rule and label keeps the
#    EMERGENCY case distinguishable at a glance.
# 3. Giving the click targets — suggestion cards and follow-up chips — enough
#    surface, depth and hover response to read as affordances.
# --------------------------------------------------------------------------

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;650&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&display=swap');

  :root {
      --cc-ink:      #101a24;
      --cc-body:     #2c3a48;
      --cc-muted:    #64798c;
      --cc-faint:    #8fa3b4;
      --cc-line:     #e2e9ef;
      --cc-line-soft:#eef3f7;
      --cc-surface:  #ffffff;
      --cc-canvas:   #f7fafb;
      --cc-accent:   #0b7a6b;
      --cc-accent-d: #095f54;
      --cc-tint:     #f0f8f6;
      --cc-shadow:   0 1px 2px rgba(16,26,36,.04), 0 8px 24px -12px rgba(16,26,36,.14);
      --cc-shadow-h: 0 1px 2px rgba(16,26,36,.05), 0 14px 32px -14px rgba(11,122,107,.30);
  }

  /* Set the face on the roots and the form elements that do not inherit, and
     nowhere else. An earlier version of this rule used [class*="st-"], which
     also matched Streamlit's icon spans — and those icons are a *ligature*
     font, so overriding the family made every one of them render its own name
     as literal text ("child_care", "add") straight through the button label.
     The guard below is belt and braces against the same mistake returning. */
  html, body, .stApp, button, input, textarea, select, optgroup {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-feature-settings: 'cv05' 1, 'ss01' 1;
  }
  [data-testid="stIconMaterial"],
  .material-symbols-rounded,
  span[class*="material-symbols"] {
      font-family: 'Material Symbols Rounded' !important;
      font-feature-settings: 'liga' 1 !important;
  }
  .stApp { background: var(--cc-canvas); }

  /* Streamlit's chrome competes with the conversation. */
  [data-testid="stToolbar"] { right: 0.5rem; }
  #MainMenu, footer { visibility: hidden; }

  .block-container {
      padding-top: 3.6rem; padding-bottom: 2rem; max-width: 46rem;
  }

  /* The page is a column with the footer pushed to its end.
     Before this, the disclaimer and author line sat wherever the content
     happened to stop — which on the empty state was directly under the
     suggestion cards, stranded in the middle of the screen with a dead gap
     beneath it and the input floating far below. `margin-top: auto` on the
     last block is the whole fix: short conversations push the footer to the
     bottom of the viewport, long ones let it come to rest after the final
     message, which is where a footer belongs. */
  /* Sized so the empty state does not overflow its scroll container. It is
     not free height: the header sits above it and Streamlit's input dock is
     an in-flow sibling below it, so anything taller makes the container
     scroll — and because the chat pane scrolls itself to the bottom, the
     overflow is taken off the *top*, sliding the hero under the header. */
  .stMain .block-container > [data-testid="stVerticalBlock"] {
      min-height: calc(100vh - 14rem);
  }
  /* The keyed container is not the flex child — Streamlit wraps it in an
     stLayoutWrapper — so `margin-top: auto` has to go on the wrapper or it
     silently resolves to 0 and the footer stays where the content stopped. */
  [data-testid="stLayoutWrapper"]:has(> .st-key-cc-footer) { margin-top: auto; }
  .st-key-cc-footer { padding-top: 1.5rem; }
  /* Auto margins on both sides of the welcome block: the free space in the
     column splits between them, so the hero sits in the optical centre
     instead of hugging the header with a dead gap underneath. */
  [data-testid="stLayoutWrapper"]:has(> .st-key-cc-welcome) {
      margin-top: auto; margin-bottom: auto;
  }

  /* Short and narrow viewports: stop forcing the column taller than the
     screen. The reserved height is what pushes the footer down on a roomy
     display, but where the content already fills the screen it only creates
     overflow — and because the chat pane scrolls itself to the bottom, that
     overflow is taken off the top and slides the hero under the header. */
  @media (max-height: 780px) {
      .stMain .block-container > [data-testid="stVerticalBlock"] { min-height: 0; }
  }
  @media (max-width: 640px) {
      .stMain .block-container > [data-testid="stVerticalBlock"] { min-height: 0; }
      .block-container { padding-top: 3.2rem; padding-left: 1rem; padding-right: 1rem; }
      .cc-hero { margin: 1.5rem 0 1.4rem; }
      .cc-hero h1 { font-size: 1.78rem; }
      .cc-hero p { font-size: 0.88rem; }
      .cc-mark { margin-bottom: 0.9rem; font-size: 0.66rem; }
      [data-testid="stHeader"]::before { left: 3.1rem; font-size: 0.88rem; }
      /* Two suggestions, not four. Stacked single-file on a phone the other
         two push the hero off the top of the screen, and a starting point the
         reader has to scroll up to find is not a starting point. */
      .st-key-sg2, .st-key-sg3 { display: none; }
  }

  /* ---- brand ----------------------------------------------------------- */

  /* In Streamlit's own fixed header, which is the one strip that survives the
     chat container scrolling itself to the bottom. A brand rendered into the
     page column instead just scrolls away — the first attempt at this sat at
     y = -68px on load, present in the DOM and invisible to everyone. */
  [data-testid="stHeader"] {
      background: rgba(247,250,251,0.94);
      backdrop-filter: saturate(160%) blur(10px);
      border-bottom: 1px solid var(--cc-line-soft);
      height: 3rem;
  }
  [data-testid="stHeader"]::before {
      content: "🧭  CareCompass";
      position: absolute; left: 3.5rem; top: 50%; transform: translateY(-50%);
      font-size: 0.95rem; font-weight: 650; letter-spacing: -0.02em;
      color: var(--cc-ink); white-space: nowrap; pointer-events: none;
  }
  /* With the sidebar open its own header carries the name, so the chevron is
     gone and the offset that cleared it is dead space. */
  [data-testid="stSidebar"][aria-expanded="true"] ~ .stMain [data-testid="stHeader"]::before {
      left: 1.2rem;
  }

  /* The lockup on the empty state: the one screen where the product should
     say what it is at full size. */
  .cc-lockup {
      display: flex; align-items: center; justify-content: center;
      gap: 0.6rem; margin: 0 0 1.25rem;
  }
  .cc-logo {
      display: grid; place-items: center;
      width: 2.15rem; height: 2.15rem; border-radius: 10px;
      background: linear-gradient(145deg, #0e9a86, var(--cc-accent-d));
      color: #fff; font-size: 1.15rem; line-height: 1;
      box-shadow: 0 3px 10px -3px rgba(11,122,107,.6);
  }
  .cc-word {
      font-size: 1.32rem; font-weight: 650; letter-spacing: -0.024em;
      color: var(--cc-ink);
  }
  .cc-word em { font-style: normal; color: var(--cc-accent); }

  /* ---- conversation ------------------------------------------------- */

  [data-testid="stChatMessage"] {
      background: transparent; padding: 0.3rem 0 0.55rem; gap: 0.8rem;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] li {
      line-height: 1.68; color: var(--cc-body); font-size: 0.945rem;
  }
  [data-testid="stChatMessage"] strong { color: var(--cc-ink); font-weight: 600; }
  [data-testid="stChatMessage"] h3 {
      font-size: 1.06rem; font-weight: 600; letter-spacing: -0.01em;
      color: var(--cc-ink); margin: 0.1rem 0 0.45rem;
  }
  [data-testid="stChatMessage"] ul { margin: 0.1rem 0 0.55rem; padding-left: 1.15rem; }
  [data-testid="stChatMessage"] li::marker { color: var(--cc-faint); }
  [data-testid="stChatMessage"] a { color: var(--cc-accent); text-underline-offset: 2px; }

  /* The user's own turn, set apart from the assistant's without a bubble. */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
      background: var(--cc-surface);
      border: 1px solid var(--cc-line-soft);
      border-radius: 14px;
      padding: 0.55rem 0.95rem;
      margin-bottom: 0.3rem;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
      color: var(--cc-ink); font-weight: 450;
  }

  /* ---- hero, shown only on an empty conversation --------------------- */

  .cc-hero { text-align: center; margin: 2.6rem 0 1.9rem; }
  .cc-mark {
      display: inline-flex; align-items: center; gap: 0.44rem;
      font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
      text-transform: uppercase; color: var(--cc-accent);
      background: var(--cc-tint); border: 1px solid #d8ece7;
      padding: 0.3rem 0.7rem; border-radius: 999px; margin-bottom: 1.15rem;
  }
  .cc-hero h1 {
      font-family: 'Source Serif 4', Georgia, serif;
      font-size: 2.35rem; font-weight: 500; line-height: 1.12;
      letter-spacing: -0.022em; color: var(--cc-ink); margin: 0 0 0.62rem;
  }
  .cc-hero p {
      color: var(--cc-muted); margin: 0 auto; font-size: 0.935rem;
      line-height: 1.6; max-width: 31rem;
  }

  /* ---- suggestion cards ---------------------------------------------- */

  .st-key-cc-suggestions .stButton > button {
      width: 100%; text-align: left; white-space: normal; height: 100%;
      min-height: 3.45rem;
      padding: 0.8rem 0.95rem;
      border-radius: 14px;
      border: 1px solid var(--cc-line);
      background: var(--cc-surface);
      box-shadow: var(--cc-shadow);
      color: var(--cc-ink);
      font-weight: 500; font-size: 0.875rem; line-height: 1.4;
      transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
  }
  .st-key-cc-suggestions .stButton > button:hover {
      border-color: #bfe0d8; box-shadow: var(--cc-shadow-h);
      transform: translateY(-2px); color: var(--cc-ink);
  }
  .st-key-cc-suggestions .stButton > button:active { transform: translateY(0); }
  .st-key-cc-suggestions [data-testid="stIconMaterial"] {
      color: var(--cc-accent); font-size: 1.15rem; margin-right: 0.15rem;
  }

  /* ---- follow-up chips ------------------------------------------------ */

  .cc-nudge {
      font-size: 0.71rem; font-weight: 600; letter-spacing: 0.07em;
      text-transform: uppercase; color: var(--cc-faint);
      margin: 0.85rem 0 0.4rem;
  }
  [class*="st-key-cc-followups"] .stButton > button {
      width: 100%; text-align: left; white-space: normal; height: 100%;
      min-height: 2.5rem;
      padding: 0.5rem 0.85rem;
      /* Not a full pill: these wrap to two lines as often as not, and a 999px
         radius on a two-line box reads as a blob rather than a chip. */
      border-radius: 13px;
      border: 1px solid var(--cc-line);
      background: var(--cc-surface);
      color: var(--cc-body);
      font-size: 0.815rem; font-weight: 450; line-height: 1.35;
      box-shadow: none;
      transition: background .14s ease, border-color .14s ease, color .14s ease;
  }
  [class*="st-key-cc-followups"] .stButton > button:hover {
      background: var(--cc-tint); border-color: #bfe0d8; color: var(--cc-accent-d);
  }

  /* ---- urgency chip --------------------------------------------------- */

  .cc-chip {
      display: inline-flex; align-items: center; gap: 0.5rem;
      padding: 0.3rem 0.72rem 0.3rem 0.62rem;
      border-radius: 9px; font-size: 0.735rem;
      margin: 0.3rem 0 0.1rem;
      border: 1px solid; border-left-width: 3px;
  }
  .cc-chip b { font-weight: 650; letter-spacing: 0.035em; }
  .cc-chip span { font-weight: 450; opacity: 0.82; }

  /* ---- audit panel ---------------------------------------------------- */

  [data-testid="stExpander"] { border: none; background: transparent; }
  [data-testid="stExpander"] details { border: none; background: transparent; }
  [data-testid="stExpander"] summary {
      font-size: 0.775rem; font-weight: 500; color: var(--cc-faint);
      padding-left: 0; transition: color .14s ease;
  }
  [data-testid="stExpander"] summary:hover { color: var(--cc-accent); }
  [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
      border-left: 2px solid var(--cc-line);
      padding-left: 0.95rem; margin-left: 0.15rem;
  }
  [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
  [data-testid="stExpander"] [data-testid="stExpanderDetails"] li {
      font-size: 0.795rem; line-height: 1.55; color: var(--cc-muted);
  }
  [data-testid="stExpander"] code {
      font-size: 0.74rem; background: var(--cc-canvas);
      color: var(--cc-accent-d); padding: 0.05rem 0.28rem; border-radius: 4px;
  }

  /* ---- chat input ------------------------------------------------------ */

  [data-testid="stChatInput"] {
      border-radius: 15px; border: 1px solid var(--cc-line);
      background: var(--cc-surface); box-shadow: var(--cc-shadow);
  }
  [data-testid="stChatInput"]:focus-within {
      border-color: #bfe0d8;
      box-shadow: 0 0 0 3px rgba(11,122,107,.10), var(--cc-shadow);
  }
  [data-testid="stBottomBlockContainer"] { background: transparent; }

  /* ---- footer ---------------------------------------------------------- */

  .cc-foot {
      text-align: center; font-size: 0.735rem; color: var(--cc-faint);
      margin: 0; padding-top: 0.9rem; line-height: 1.6;
      border-top: 1px solid var(--cc-line-soft);
  }
  .cc-sig {
      text-align: center; font-size: 0.715rem; color: var(--cc-faint);
      margin: 0.32rem 0 0;
  }
  .cc-sig a { color: var(--cc-muted); text-decoration: none; font-weight: 500; }
  .cc-sig a:hover { color: var(--cc-accent); text-decoration: underline;
                    text-underline-offset: 2px; }
  .cc-sig .cc-dot { color: var(--cc-line); margin: 0 0.42rem; }

  /* ---- page furniture for the non-chat views --------------------------- */

  .cc-page-title {
      font-family: 'Source Serif 4', Georgia, serif;
      font-size: 1.62rem; font-weight: 500; letter-spacing: -0.015em;
      color: var(--cc-ink); margin: 0.6rem 0 0.35rem;
  }
  .cc-note {
      font-size: 0.855rem; line-height: 1.62; color: var(--cc-muted);
      margin: 0 0 1.1rem;
  }
  [data-testid="stMetric"] {
      background: var(--cc-surface); border: 1px solid var(--cc-line-soft);
      border-radius: 13px; padding: 0.7rem 0.85rem; box-shadow: var(--cc-shadow);
  }
  [data-testid="stMetricLabel"] p {
      font-size: 0.72rem !important; font-weight: 500; color: var(--cc-faint);
      letter-spacing: 0.02em;
  }
  [data-testid="stMetricValue"] {
      font-size: 1.42rem; font-weight: 600; color: var(--cc-ink);
      letter-spacing: -0.015em;
  }

  /* ---- sidebar ---------------------------------------------------------- */

  [data-testid="stSidebar"] {
      background: var(--cc-surface); border-right: 1px solid var(--cc-line-soft);
  }
  .cc-brand {
      display: flex; align-items: center; gap: 0.5rem;
      font-size: 1.02rem; font-weight: 600; letter-spacing: -0.015em;
      color: var(--cc-ink); margin: 0.2rem 0 0.15rem;
  }
  .cc-brand-sub {
      font-size: 0.755rem; color: var(--cc-faint); line-height: 1.5;
      margin: 0 0 0.1rem;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: 0.87rem; }
  .cc-status {
      font-size: 0.735rem; line-height: 1.62; color: var(--cc-muted);
  }
  .cc-status b { color: var(--cc-body); font-weight: 600; }
  .cc-dot-live {
      display: inline-block; width: 6px; height: 6px; border-radius: 50%;
      background: var(--cc-accent); margin-right: 0.38rem; vertical-align: middle;
  }
  .cc-dot-off { background: var(--cc-faint); }
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
    """The urgency chip: tinted ground, saturated rule and label.

    Colour is carried by a 3px left rule and the label text rather than a solid
    fill. Five solid-filled levels make every answer look like an alert, and
    once ROUTINE is shouting, EMERGENCY has nothing left to shout with.
    """
    meta = LEVELS[level]
    return (
        f'<div class="cc-chip" style="border-color:{meta.colour}33;'
        f"border-left-color:{meta.colour};background:{meta.colour}0d;"
        f'color:{meta.colour}">'
        f"<b>{meta.badge}</b><span>{note}</span></div>"
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
    st.markdown(
        f'<div class="cc-brand">🧭 {APP_NAME}</div>'
        '<p class="cc-brand-sub">Health guidance with a safety floor you can audit</p>',
        unsafe_allow_html=True,
    )
    st.write("")

    page = st.radio(
        "View",
        ["Chat", "Insights", "Evaluation"],
        label_visibility="collapsed",
    )

    st.button("New chat", width="stretch", on_click=new_chat, icon=":material/add:")

    language_label = st.selectbox("Reply language", list(SUPPORTED_LANGUAGES.values()))
    language = next(
        (code for code, label in SUPPORTED_LANGUAGES.items() if label == language_label),
        "auto",
    )

    st.divider()

    llm = HEALTH["llm"]
    web = HEALTH["web"]
    model_line = (
        f'<b>{llm["label"]}</b> · <code>{llm["model"]}</code>'
        if llm["available"]
        else "<b>No model key</b> · answers quoted from the corpus"
    )
    web_line = (
        f'<b>Web tier</b> · {", ".join(web["tiers"])}, {web["domains"]} trusted sources'
        if web["enabled"]
        else "<b>Web tier</b> · off"
    )
    st.markdown(
        '<div class="cc-status">'
        f'<span class="cc-dot-live{"" if llm["available"] else " cc-dot-off"}"></span>'
        f"{model_line}<br>"
        f'<span class="cc-dot-live{"" if web["enabled"] else " cc-dot-off"}"></span>'
        f"{web_line}<br>"
        f'<span class="cc-dot-live"></span><b>Retrieval</b> · '
        f'{HEALTH["retrieval"]["mode"]}, {HEALTH["corpus"]["chunks"]} passages'
        f"<br><span style='opacity:.6'>v{VERSION}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Disclaimer and author line, pinned to the end of the page.

    Wrapped in a keyed container purely for the CSS hook: `.st-key-cc-footer`
    takes `margin-top: auto`, which is what drops it to the bottom of the
    column instead of leaving it wherever the content happened to stop.
    """
    with st.container(key="cc-footer"):
        st.markdown(
            '<p class="cc-foot">General health information, not a diagnosis or a '
            "prescription. In an emergency call <b>112</b>.</p>",
            unsafe_allow_html=True,
        )
        signature()


def signature() -> None:
    """Author line. Rendered once at the foot of whichever view is open."""
    st.markdown(
        '<p class="cc-sig">'
        f'<a href="{AUTHOR_URL}" target="_blank" rel="noopener">{AUTHOR}</a>'
        f'<span class="cc-dot">·</span>{AUTHOR_AFFILIATION}'
        f'<span class="cc-dot">·</span>'
        f'<a href="{AUTHOR_URL}" target="_blank" rel="noopener">abhaanisha.github.io</a>'
        "</p>",
        unsafe_allow_html=True,
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


def render_followups(followups: list[str], turn: int) -> None:
    """Clickable next questions under the most recent answer.

    Only under the most recent one. A chip beside an answer three turns back is
    not a suggestion any more, it is clutter that silently rewrites the thread
    when someone clicks it.
    """
    if not followups:
        return
    st.markdown('<p class="cc-nudge">You might ask</p>', unsafe_allow_html=True)
    # The key has to be unique per turn — Streamlit refuses duplicates — but
    # the CSS hook is the shared `st-key-cc-followups` prefix it generates.
    with st.container(key=f"cc-followups-{turn}"):
        for i, (column, text) in enumerate(
            zip(st.columns(len(followups), gap="small"), followups)
        ):
            if column.button(text, key=f"fu{turn}-{i}"):
                st.session_state.pending = text
                st.rerun()


def render_chat() -> None:
    history = st.session_state.history

    # Read the input first even though it paints last: Streamlit pins the chat
    # input to the bottom of the viewport wherever it is called, so knowing
    # early whether a turn is in flight lets the rest of the view lay itself
    # out correctly — chips only under the answer that is actually last, and
    # the disclaimer genuinely at the foot of the page rather than stranded
    # above the reply that just arrived.
    typed = st.chat_input("Describe a health concern…")
    message = typed or st.session_state.pending
    st.session_state.pending = None

    if not history and not message:
        # One keyed block for the whole welcome state, so the stylesheet can
        # centre it as a unit between the header and the footer.
        with st.container(key="cc-welcome"):
            st.markdown(
                '<div class="cc-hero">'
                '<div class="cc-mark">Grounded · Auditable · Triaged</div>'
                "<h1>What is going on?</h1>"
                "<p>Describe a symptom in English, Hindi or Bengali. Every answer "
                "shows the rules that fired and the passages it used — from a "
                "curated knowledge base, and from public-health sources when the "
                "question reaches past it.</p></div>",
                unsafe_allow_html=True,
            )
            with st.container(key="cc-suggestions"):
                left, right = st.columns(2, gap="small")
                for i, (label, prompt, icon) in enumerate(SUGGESTIONS):
                    with (left, right)[i % 2]:
                        if st.button(label, key=f"sg{i}", icon=icon):
                            st.session_state.pending = prompt
                            st.rerun()

    last = len(history) - 1
    for index, entry in enumerate(history):
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("meta"):
                render_audit(entry["meta"])
                # Not when a new turn is about to be appended below: chips
                # belong to the last answer on screen, and for the next second
                # that is not this one.
                if index == last and not message:
                    render_followups(entry["meta"].get("followups", []), index)

    if not message:
        render_footer()
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
        web_used = sum(1 for c in result.citations if c.used and c.kind == "web")
        if used and web_used:
            label = (
                f"{used} source{'' if used == 1 else 's'} cited "
                f"({web_used} from the web) · decision trace"
            )
        elif used:
            label = f"{used} source{'' if used == 1 else 's'} cited · decision trace"
        else:
            label = "Decision trace"

        meta = {
            "level": result.triage.level.name,
            "note": f"{result.triage.meta.timeframe} · {result.latency_ms} ms",
            "audit_label": label,
            "sources": result.sources_markdown(),
            "trace": result.trace_markdown(),
            "followups": getattr(result, "followups", []),
        }
        render_audit(meta)
        render_followups(meta["followups"], len(history) + 1)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": body, "meta": meta})
    render_footer()


# --------------------------------------------------------------------------
# insights
# --------------------------------------------------------------------------


def render_insights() -> None:
    st.markdown('<h2 class="cc-page-title">Insights</h2>', unsafe_allow_html=True)
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
        render_footer()
        return

    st.caption("Urgency distribution")
    st.bar_chart(stats["triage"], horizontal=True, color="#4b6bfb", height=190)
    st.caption("Most retrieved documents")
    st.bar_chart(stats["topics"], horizontal=True, color="#0b7a6b", height=230)
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
                    "web": e.get("web_provider") or "-",
                    "red flags": ", ".join(e.get("red_flags") or []) or "-",
                    "ms": e.get("latency_ms", 0),
                }
                for e in reversed(events[-25:])
            ],
            width="stretch",
            hide_index=True,
        )

    render_footer()


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def render_evaluation() -> None:
    st.markdown(
        '<h2 class="cc-page-title">Does it actually work?</h2>', unsafe_allow_html=True
    )
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
        render_footer()
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

    render_footer()


if page == "Chat":
    render_chat()
elif page == "Insights":
    render_insights()
else:
    render_evaluation()
