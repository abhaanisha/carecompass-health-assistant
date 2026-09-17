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

#: The assistant's name, separate from the product's. "Disha" (दिशा / দিশা)
#: means *direction* in Hindi and Bengali — which is what a compass gives you
#: and, more to the point, what this thing actually does: it does not diagnose,
#: it tells you which way to go and how soon. One word, no honorific, and it
#: reads the same in all three languages the app answers in.
BOT_NAME = "Disha"

AUTHOR = "Abha Singh Sardar"
AUTHOR_AFFILIATION = "IISc"
AUTHOR_URL = "https://abhaanisha.github.io/"

#: Disha gets the brand mark; the reader keeps Streamlit's default avatar.
#: That asymmetry is load-bearing, not laziness: passing a custom avatar makes
#: Streamlit drop the `stChatMessageAvatarUser` test id, and that id is the
#: only stable way to tell the two roles apart in CSS — the rest of the markup
#: differs by an emotion-hash class that changes between releases.
AVATARS = {"assistant": "🧭", "user": None}

st.set_page_config(
    page_title=f"{APP_NAME} — {BOT_NAME}, your health guide",
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
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;650&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&display=swap');

  :root {
      --cc-ink:      #14231f;
      --cc-body:     #33443f;
      --cc-muted:    #64786f;
      --cc-faint:    #94a79d;
      --cc-line:     #e2e8e2;
      --cc-line-soft:#edf1ec;
      --cc-surface:  #ffffff;
      --cc-canvas:   #f4f6f1;
      --cc-canvas-2: #eef2ea;
      --cc-accent:   #2f6b4f;
      --cc-accent-d: #1f4c38;
      --cc-accent-l: #4a8b69;
      --cc-tint:     #eef5f0;
      --cc-shadow:   0 1px 2px rgba(20,35,31,.04), 0 10px 28px -14px rgba(20,35,31,.16);
      --cc-shadow-m: 0 2px 6px rgba(20,35,31,.05), 0 18px 44px -20px rgba(20,35,31,.24);
      --cc-bar:      4.6rem;
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

  /* A warm ground rather than a flat grey. Two very soft washes, so the card
     surfaces have something to sit on and the page does not read as a form. */
  .stApp {
      background:
        radial-gradient(70rem 40rem at 18% -10%, #ffffff 0%, rgba(255,255,255,0) 60%),
        radial-gradient(60rem 36rem at 92% 8%, var(--cc-tint) 0%, rgba(238,245,240,0) 55%),
        var(--cc-canvas);
  }

  #MainMenu, footer { visibility: hidden; }

  /* ---- the identity bar ------------------------------------------------
     Streamlit's own header, restyled. It is the one strip that survives the
     chat pane scrolling itself to the bottom, so it is where the assistant's
     name belongs; anything rendered into the page column scrolls away. */

  [data-testid="stHeader"] {
      height: var(--cc-bar);
      background: linear-gradient(135deg, #35755a 0%, var(--cc-accent) 45%, var(--cc-accent-d) 100%);
      box-shadow: 0 1px 0 rgba(20,35,31,.06), 0 10px 30px -18px rgba(20,35,31,.5);
      border: none;
  }
  [data-testid="stHeader"]::before {
      content: "🧭";
      position: absolute; left: 1.15rem; top: 50%; transform: translateY(-50%);
      width: 2.5rem; height: 2.5rem; border-radius: 12px;
      display: grid; place-items: center; font-size: 1.2rem;
      background: rgba(255,255,255,.16);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.22);
  }
  /* Two lines from one pseudo-element: `\\A` plus white-space: pre. */
  [data-testid="stHeader"]::after {
      content: "Disha — your health guide\\A Triage · cited sources · English, हिंदी, বাংলা";
      white-space: pre;
      position: absolute; left: 4.3rem; top: 50%; transform: translateY(-50%);
      font-size: 0.97rem; font-weight: 600; letter-spacing: -0.012em;
      line-height: 1.42; color: #ffffff; pointer-events: none;
  }
  /* Streamlit puts the expand chevron in this same corner, but only while the
     sidebar is closed. Shift the lockup clear of it exactly when it is there,
     rather than reserving the space permanently. */
  [data-testid="stHeader"]:has([data-testid="stExpandSidebarButton"])::before { left: 3.6rem; }
  [data-testid="stHeader"]:has([data-testid="stExpandSidebarButton"])::after  { left: 6.75rem; }
  /* Streamlit's controls now sit on a dark ground. */
  [data-testid="stHeader"] [data-testid="stIconMaterial"],
  [data-testid="stHeader"] button { color: rgba(255,255,255,.92) !important; }
  [data-testid="stHeader"] button:hover { background: rgba(255,255,255,.14) !important; }
  [data-testid="stToolbar"] { right: 0.6rem; }
  [data-testid="stToolbar"] * { color: rgba(255,255,255,.88) !important; }
  [data-testid="stExpandSidebarButton"] { margin-left: 0.15rem; }

  /* ---- the column ------------------------------------------------------ */

  .block-container {
      padding-top: calc(var(--cc-bar) + 1.2rem); padding-bottom: 3rem;
      max-width: 47rem;
  }
  .stMain .block-container > [data-testid="stVerticalBlock"] {
      min-height: calc(100vh - 17.2rem);
  }
  /* Auto margins top and bottom: the free space splits between them, so the
     welcome panel sits in the optical centre instead of hugging the bar with
     a dead gap underneath. */
  [data-testid="stLayoutWrapper"]:has(> .st-key-cc-welcome) {
      margin-top: auto; margin-bottom: auto;
  }

  /* ---- welcome --------------------------------------------------------- */

  .cc-welcome {
      background: var(--cc-surface);
      border: 1px solid var(--cc-line);
      border-radius: 20px;
      padding: 1.5rem 1.6rem 1.35rem;
      box-shadow: var(--cc-shadow);
      position: relative; overflow: hidden;
  }
  .cc-welcome::before {
      content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
      background: linear-gradient(90deg, var(--cc-accent-l), var(--cc-accent), #7fb08f);
  }
  .cc-greet {
      font-family: 'Fraunces', Georgia, serif;
      font-size: 1.62rem; font-weight: 500; letter-spacing: -0.015em;
      color: var(--cc-ink); margin: 0.15rem 0 0.5rem;
  }
  .cc-welcome p {
      color: var(--cc-muted); font-size: 0.925rem; line-height: 1.62; margin: 0;
      max-width: 36rem;
  }
  .cc-pills { display: flex; flex-wrap: wrap; gap: 0.42rem; margin-top: 1.05rem; }
  .cc-pill {
      font-size: 0.775rem; font-weight: 500; color: var(--cc-body);
      background: var(--cc-tint); border: 1px solid #dcebe1;
      padding: 0.3rem 0.68rem; border-radius: 999px; white-space: nowrap;
  }
  .cc-sub {
      font-size: 0.735rem; font-weight: 600; letter-spacing: 0.085em;
      text-transform: uppercase; color: var(--cc-faint);
      margin: 1.5rem 0 0.55rem; text-align: center;
  }

  /* ---- suggestion cards ------------------------------------------------- */

  .st-key-cc-suggestions .stButton > button {
      width: 100%; text-align: left; white-space: normal; height: 100%;
      min-height: 3.3rem; padding: 0.78rem 0.95rem;
      border-radius: 15px; border: 1px solid var(--cc-line);
      background: var(--cc-surface); box-shadow: var(--cc-shadow);
      color: var(--cc-ink); font-weight: 500; font-size: 0.875rem; line-height: 1.4;
      transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
  }
  .st-key-cc-suggestions .stButton > button:hover {
      border-color: #bcd9c6; box-shadow: var(--cc-shadow-m);
      transform: translateY(-2px); color: var(--cc-ink);
  }
  .st-key-cc-suggestions .stButton > button:active { transform: translateY(0); }
  .st-key-cc-suggestions [data-testid="stIconMaterial"] {
      color: var(--cc-accent); font-size: 1.15rem;
  }

  /* ---- conversation ----------------------------------------------------- */

  [data-testid="stChatMessage"] {
      background: transparent; padding: 0.3rem 0 0.5rem; gap: 0.72rem;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] li {
      line-height: 1.68; color: var(--cc-body); font-size: 0.945rem;
  }
  [data-testid="stChatMessage"] strong { color: var(--cc-ink); font-weight: 600; }
  [data-testid="stChatMessage"] h3 {
      font-size: 1.04rem; font-weight: 600; letter-spacing: -0.01em;
      color: var(--cc-ink); margin: 0.1rem 0 0.45rem;
  }
  [data-testid="stChatMessage"] ul { margin: 0.1rem 0 0.55rem; padding-left: 1.15rem; }
  [data-testid="stChatMessage"] li::marker { color: var(--cc-accent-l); }
  [data-testid="stChatMessage"] a { color: var(--cc-accent); text-underline-offset: 2px; }

  /* The assistant speaks from a card; the reader's own words sit in a tinted
     bubble. Two surfaces, so a long thread stays readable as a conversation
     rather than one undifferentiated column of prose. */
  [data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"]))
      [data-testid="stChatMessageContent"] {
      background: var(--cc-surface);
      border: 1px solid var(--cc-line-soft);
      border-radius: 16px 16px 16px 5px;
      padding: 0.85rem 1.05rem 0.7rem;
      box-shadow: var(--cc-shadow);
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
      [data-testid="stChatMessageContent"] {
      background: var(--cc-tint);
      border: 1px solid #dcebe1;
      border-radius: 16px 16px 5px 16px;
      padding: 0.62rem 0.95rem;
      /* Hugs the words. A four-word question stretched to the full column
         reads as a heading rather than as something the reader said. */
      width: fit-content; max-width: 100%;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
      color: var(--cc-ink); font-weight: 450; margin-bottom: 0;
  }
  /* Disha's avatar is a custom emoji, so it has no test id of its own — it is
     simply the first child of a message that is not the reader's. */
  [data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) > div:first-child {
      background: linear-gradient(145deg, var(--cc-accent-l), var(--cc-accent-d)) !important;
      box-shadow: 0 2px 8px -3px rgba(31,76,56,.7);
      border-radius: 10px !important;
  }
  [data-testid="stChatMessageAvatarUser"] {
      background: #dfe8e0 !important; color: var(--cc-accent-d) !important;
      border-radius: 10px !important;
  }

  /* ---- follow-up chips --------------------------------------------------- */

  .cc-nudge {
      font-size: 0.7rem; font-weight: 600; letter-spacing: 0.085em;
      text-transform: uppercase; color: var(--cc-faint);
      margin: 0.9rem 0 0.42rem;
  }
  [class*="st-key-cc-followups"] .stButton > button {
      width: 100%; text-align: left; white-space: normal; height: 100%;
      min-height: 2.5rem; padding: 0.5rem 0.85rem;
      /* Not a full pill: these wrap to two lines as often as not, and a 999px
         radius on a two-line box reads as a blob rather than a chip. */
      border-radius: 13px;
      border: 1px solid var(--cc-line); background: var(--cc-surface);
      color: var(--cc-body); font-size: 0.815rem; font-weight: 450; line-height: 1.35;
      box-shadow: 0 1px 2px rgba(20,35,31,.03);
      transition: background .14s ease, border-color .14s ease, color .14s ease;
  }
  [class*="st-key-cc-followups"] .stButton > button:hover {
      background: var(--cc-tint); border-color: #bcd9c6; color: var(--cc-accent-d);
  }

  /* ---- urgency chip ------------------------------------------------------ */

  .cc-chip {
      display: inline-flex; align-items: center; gap: 0.5rem;
      padding: 0.3rem 0.72rem 0.3rem 0.62rem;
      border-radius: 9px; font-size: 0.735rem;
      margin: 0.55rem 0 0.1rem;
      border: 1px solid; border-left-width: 3px;
  }
  .cc-chip b { font-weight: 650; letter-spacing: 0.035em; }
  .cc-chip span { font-weight: 450; opacity: 0.82; }

  /* ---- audit panel -------------------------------------------------------- */

  [data-testid="stExpander"] { border: none; background: transparent; }
  [data-testid="stExpander"] details { border: none; background: transparent; }
  [data-testid="stExpander"] summary {
      font-size: 0.775rem; font-weight: 500; color: var(--cc-faint);
      padding-left: 0; transition: color .14s ease;
  }
  [data-testid="stExpander"] summary:hover { color: var(--cc-accent); }
  [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
      border-left: 2px solid var(--cc-line); padding-left: 0.95rem; margin-left: 0.15rem;
  }
  [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
  [data-testid="stExpander"] [data-testid="stExpanderDetails"] li {
      font-size: 0.795rem; line-height: 1.55; color: var(--cc-muted);
  }
  [data-testid="stExpander"] code {
      font-size: 0.74rem; background: var(--cc-canvas-2);
      color: var(--cc-accent-d); padding: 0.05rem 0.28rem; border-radius: 4px;
  }

  /* ---- the input dock, and the footer beneath it --------------------------
     The disclaimer belongs under the box a person is about to type into, not
     above it, so the dock reserves room at its foot and the footer is fixed
     into that space. */

  [data-testid="stBottomBlockContainer"] {
      background: transparent;
      padding-bottom: 3.9rem;
      padding-top: 0.6rem;
  }
  [data-testid="stBottom"] > div {
      background: linear-gradient(to top, var(--cc-canvas) 62%, rgba(244,246,241,0));
  }
  [data-testid="stChatInput"] {
      border-radius: 17px; border: 1px solid var(--cc-line);
      background: var(--cc-surface); box-shadow: var(--cc-shadow-m);
  }
  [data-testid="stChatInput"]:focus-within {
      border-color: var(--cc-accent-l);
      box-shadow: 0 0 0 3px rgba(74,139,105,.14), var(--cc-shadow-m);
  }
  [data-testid="stChatInputSubmitButton"] {
      background: var(--cc-accent) !important; border-radius: 11px !important;
  }
  [data-testid="stChatInputSubmitButton"] [data-testid="stIconMaterial"],
  [data-testid="stChatInputSubmitButton"] svg { color: #fff !important; fill: #fff !important; }
  [data-testid="stChatInputSubmitButton"]:hover { background: var(--cc-accent-d) !important; }

  .cc-dock {
      /* This div is the fixed element, not the Streamlit container holding it.
         Fixing the container meant Streamlit decided its height — it settled
         on 49px for 63px of text and clipped the author line off the bottom of
         the screen. A plain div sizes to its own content.
         z-index clears the input dock, which Streamlit puts at 99. */
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 120;
      background: var(--cc-canvas);
      border-top: 1px solid var(--cc-line-soft);
      padding: 0.5rem 1rem 0.6rem;
  }
  body:has([data-testid="stSidebar"][aria-expanded="true"]) .cc-dock { left: 21rem; }
  .cc-foot { line-height: 1.45; }
  .cc-sig  { line-height: 1.45; }
  body:has([data-testid="stSidebar"][aria-expanded="true"]) .st-key-cc-footer {
      left: 21rem;
  }
  .cc-foot {
      text-align: center; font-size: 0.715rem; color: var(--cc-faint); margin: 0;
  }
  .cc-sig { text-align: center; font-size: 0.7rem; color: var(--cc-faint); margin: 0.12rem 0 0; }
  .cc-sig a { color: var(--cc-muted); text-decoration: none; font-weight: 500; }
  .cc-sig a:hover { color: var(--cc-accent); text-decoration: underline; text-underline-offset: 2px; }
  .cc-sig .cc-dot { color: var(--cc-line); margin: 0 0.4rem; }

  /* ---- the non-chat views -------------------------------------------------- */

  .cc-page-title {
      font-family: 'Fraunces', Georgia, serif;
      font-size: 1.62rem; font-weight: 500; letter-spacing: -0.015em;
      color: var(--cc-ink); margin: 0.4rem 0 0.35rem;
  }
  .cc-note { font-size: 0.855rem; line-height: 1.62; color: var(--cc-muted); margin: 0 0 1.1rem; }
  [data-testid="stMetric"] {
      background: var(--cc-surface); border: 1px solid var(--cc-line-soft);
      border-radius: 14px; padding: 0.7rem 0.85rem; box-shadow: var(--cc-shadow);
  }
  [data-testid="stMetricLabel"] p {
      font-size: 0.72rem !important; font-weight: 500; color: var(--cc-faint);
      letter-spacing: 0.02em;
  }
  [data-testid="stMetricValue"] {
      font-size: 1.42rem; font-weight: 600; color: var(--cc-ink); letter-spacing: -0.015em;
  }

  /* ---- sidebar -------------------------------------------------------------- */

  [data-testid="stSidebar"] {
      background: var(--cc-surface); border-right: 1px solid var(--cc-line-soft);
  }
  .cc-brand {
      display: flex; align-items: center; gap: 0.5rem;
      font-family: 'Fraunces', Georgia, serif;
      font-size: 1.16rem; font-weight: 600; letter-spacing: -0.012em;
      color: var(--cc-ink); margin: 0.2rem 0 0.2rem;
  }
  .cc-brand-sub { font-size: 0.755rem; color: var(--cc-faint); line-height: 1.5; margin: 0; }
  [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: 0.87rem; }
  .cc-status { font-size: 0.735rem; line-height: 1.62; color: var(--cc-muted); }
  .cc-status b { color: var(--cc-body); font-weight: 600; }
  .cc-dot-live {
      display: inline-block; width: 6px; height: 6px; border-radius: 50%;
      background: var(--cc-accent-l); margin-right: 0.38rem; vertical-align: middle;
  }
  .cc-dot-off { background: var(--cc-faint); }

  /* ---- small screens ---------------------------------------------------------
     Stop reserving height the screen does not have: the reservation is what
     pushes the welcome panel to the optical centre on a roomy display, but
     where content already fills the screen it only creates overflow — and
     because the chat pane scrolls itself to the bottom, that overflow comes
     off the top and slides the greeting under the identity bar. */

  @media (max-height: 820px) {
      .stMain .block-container > [data-testid="stVerticalBlock"] { min-height: 0; }
  }
  @media (max-width: 640px) {
      .stMain .block-container > [data-testid="stVerticalBlock"] { min-height: 0; }
      /* Still has to clear the bar — an earlier flat value here put the
         page titles underneath it on a phone. */
      .block-container {
          padding-top: calc(var(--cc-bar) + 0.9rem);
          padding-left: 1rem; padding-right: 1rem;
      }
      :root { --cc-bar: 4.1rem; }
      [data-testid="stHeader"]::after { font-size: 0.85rem; left: 3.9rem; }
      [data-testid="stHeader"]::before { width: 2.2rem; height: 2.2rem; font-size: 1.05rem; }
      .cc-greet { font-size: 1.38rem; }
      .cc-welcome { padding: 1.25rem 1.15rem 1.15rem; }
      /* Two suggestions, not four. Stacked single-file on a phone the other
         two push the greeting off the top of the screen, and a starting point
         the reader has to scroll up to find is not a starting point. */
      .st-key-sg2, .st-key-sg3 { display: none; }
      body:has([data-testid="stSidebar"][aria-expanded="true"]) .cc-dock { left: 0; }
  }
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
        f'<p class="cc-brand-sub">{BOT_NAME} — health guidance with a safety '
        "floor you can audit</p>",
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
    """Disclaimer and author line, docked beneath the input.

    Emitted as one markdown block rather than two. Streamlit lays a container's
    children out as a flex column with a 1rem gap, and with two children that
    gap pushed the author line out of the fixed box and off the bottom of the
    screen.

    The `.cc-dock` div is fixed into the space the input dock reserves at its
    foot. A disclaimer belongs
    under the box someone is about to type into — above it, it is read once and
    then scrolled past forever.
    """
    st.markdown(
            '<div class="cc-dock">'
            f'<p class="cc-foot">{BOT_NAME} gives general health information — not a '
            "diagnosis, not a prescription. In an emergency call <b>112</b>.</p>"
            '<p class="cc-sig">'
            f'<a href="{AUTHOR_URL}" target="_blank" rel="noopener">{AUTHOR}</a>'
            f'<span class="cc-dot">·</span>{AUTHOR_AFFILIATION}'
            f'<span class="cc-dot">·</span>'
            f'<a href="{AUTHOR_URL}" target="_blank" rel="noopener">abhaanisha.github.io</a>'
            "</p></div>",
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
                '<div class="cc-welcome">'
                f'<div class="cc-greet">Namaste — I am {BOT_NAME}. 🙏</div>'
                "<p>Tell me what is going on, in English, Hindi or Bengali. I will "
                "say how soon it needs attention and show you exactly what I based "
                "that on — the rules that fired and the passages I read.</p>"
                '<div class="cc-pills">'
                '<span class="cc-pill">Symptom triage</span>'
                '<span class="cc-pill">Cited sources</span>'
                '<span class="cc-pill">Red-flag rules</span>'
                '<span class="cc-pill">हिंदी · বাংলা</span>'
                "</div></div>"
                '<p class="cc-sub">Or start with one of these</p>',
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
        with st.chat_message(entry["role"], avatar=AVATARS[entry["role"]]):
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

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(message)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner(f"{BOT_NAME} is checking the rules and the sources…"):
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
