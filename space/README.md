---
title: CareCompass Health Assistant
emoji: 🧭
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: Health triage with auditable safety rules, running in your browser
---

# CareCompass — static build

A health information assistant whose safety-critical decisions are made by
auditable rules rather than by a language model. This Space runs **entirely in
your browser** under Gradio-Lite (Pyodide): no backend, no API key, and nothing
you type leaves the tab.

Everything — the knowledge base, the safety rules, the retrieval engine and the
52-case evaluation harness — is bundled into `index.html`.

## Try

- `I have crushing chest pain going into my left arm` — fixed emergency text, no model involved
- `I have no chest pain, just a mild fever` — same keywords, correctly not escalated
- `my 6 week old baby has a fever` — any fever under 3 months is an emergency
- `বুকে ব্যথা হচ্ছে আর ঘাম হচ্ছে` — hand-written Bengali emergency card, not a machine translation
- `which antibiotic should I take for a sore throat?` — declines to prescribe, explains why

The **Evaluation** tab runs the full gold set in your browser.

## Source

This file is generated. The full project — including the server build with
hybrid dense retrieval and generated answers — lives in the main repository,
where `python build_space.py` regenerates `index.html`.
