# Talking about this project in an interview

Notes for you, not for the repo's readers. Everything here is checkable against
the code, so you can be specific without overclaiming.

---

## The 45-second version

> CareCompass is a health triage assistant. The interesting part isn't the
> chatbot — it's that the safety-critical decisions are deliberately taken away
> from the language model. Red flags are matched by auditable rules in a YAML
> file, those rules set an urgency floor, and the model is only allowed to raise
> that floor, never lower it. When someone describes chest pain, no model is
> called at all — the response is fixed text, so the ambulance number can't be
> hallucinated and the answer can't be jailbroken. On a 52-case labelled set it
> gets 100% emergency recall with zero over-triage, and it runs with no API key
> because it falls back to extractive answers from the retrieved passages.

Then stop. Let them pick the thread they care about.

---

## The three claims, and where they live in the code

**1. The safety path has no model in it.**
[`src/pipeline.py`](src/pipeline.py) → `answer()`, the `if safety.is_crisis / elif
safety.is_emergency` branch. The text comes from `emergency_card()` and
`crisis_card()` in [`src/prompts.py`](src/prompts.py). There's a test that asserts
the stub LLM was never called on that path
(`test_emergency_never_reaches_the_model`).

**2. The model can escalate but not de-escalate.**
[`src/triage.py`](src/triage.py) → `merge_model_level()`. Two tests pin both
directions.

**3. It degrades instead of failing.**
[`src/llm.py`](src/llm.py) never raises — a failed call returns `ok=False` and
`pipeline._compose()` falls through to the extractive path, with the failure
recorded in the answer's warnings and visible in the decision trace.

---

## Questions you should expect

**"Why not just prompt the model to be careful?"**
Because a prompt is a request, not a guarantee. A model that follows a safety
instruction 97% of the time is fine for phrasing and unacceptable for deciding
whether someone with crushing chest pain is told to call an ambulance. Rules give
you something you can test, version, show a clinician, and regression-gate in CI.
The model still adds value — it handles paraphrase the regex misses, which is
exactly why it's allowed to escalate.

**"Regex for medical triage? That seems brittle."**
It is brittle, and that's a stated limitation in the README. The mitigation is
layered: rules catch the phrasings we anticipated, the model is the backstop for
the ones we didn't, and the gold set measures the gap. The alternative — a
classifier — needs labelled data I don't have and would be harder for a clinician
to review. If I had a few thousand labelled triage conversations I'd train one
and keep the rules as a floor underneath it.

**"Walk me through the retrieval."**
Hybrid BM25 + MiniLM embeddings, fused with Reciprocal Rank Fusion. RRF rather
than a weighted score blend because BM25 and cosine live on different,
corpus-dependent scales, so any fixed alpha needs retuning when the corpus
changes. Three things came out of testing rather than design: a lay-term lexicon
for Indian English ("loose motion", "shugar", "bukhar"), a small suffix stemmer
because users type "dehydrated" and the corpus says "dehydration", and
IDF-weighted grounding to decide whether a question is in scope at all.

**"How do you know it works?"**
52 hand-labelled cases across 11 clinical categories and three languages.
Emergency recall 100%, negation specificity 100%, retrieval recall@5 100%,
under-triage 3.8%, over-triage 0%. `run_eval` exits non-zero if emergency recall
drops below 100% or a dose-like string shows up in generated text, so it's a CI
gate rather than a report.
And I report accuracy separately by error direction — averaging over-triage and
under-triage into one number hides the only failure that matters.

**"What's the weakest part?"**
The corpus. Twelve documents I wrote from public health guidance is a
demonstration, not a clinical library. Anything real needs clinician-reviewed
content with an approval workflow, versioning and enforced re-review dates. The
second weakest is out-of-scope detection — it's a heuristic with a threshold, and
I can show you a query that slips past it.

**"Why not LangChain / a vector DB?"**
For 60 chunks, an in-memory numpy matrix is the whole vector database, and the
retrieval logic is forty lines I can reason about and unit test. The LLM layer is
two HTTP request shapes covering five providers, which keeps `requirements.txt`
to five lines and means no vendor SDK release can break the deployment. I'd reach
for a framework at a scale where its abstractions earn their cost.

**"Why are there three front ends?"**
Because hosting constraints are a real engineering input, not an excuse. All the
clinical logic is in `src/`; `streamlit_app.py`, `app.py` and `app_lite.py` are
presentation only. That split meant when Hugging Face put Gradio Spaces behind a
paywall I could ship a browser build and a Streamlit build in an afternoon,
without touching a single rule. If the triage logic had been tangled into the
UI, or into prompts, neither would have been possible. Good sentence to land:
*"a UI framework change should not be able to alter what the system considers an
emergency, and here it structurally cannot."*

**"Why does it run in the browser?"**
Two reasons, and the honest one comes first: Hugging Face now charges for Gradio
Spaces and I wanted a free public demo. But it only *worked* because of how the
system is built — the safety engine is regular expressions, BM25 and a rule
engine, so it compiles to WebAssembly and runs client-side with no server and no
key. A design that had put the triage logic inside model prompts could not have
shipped this way at all. The same `src/` package drives both builds; the static
one just loses the generation layer, which is the layer I already treat as
optional.

**"What would you build next?"**
A clinician review workflow for the corpus, then a second-opinion urgency vote
from a different model with disagreements logged for review — that's the cheapest
way to find the phrasings the rules miss. Then streaming, per-state emergency
numbers, and a real escalation handoff to a telemedicine queue.

---

## Connecting it to your internship

This is a step on from the Mohanpur Organics bot, and it's worth framing it that
way rather than as a separate project:

- That bot answered from a single clinic knowledge file; this one retrieves over
  a chunked corpus with citations, hybrid search and a grounding check.
- That bot put its safety rules in the system prompt; this one moved them into
  auditable data that a non-programmer can review and that CI can gate on.
- That bot's quality was assessed by trying it; this one has a labelled gold set
  and reported metrics.
- The analytics instrumentation is directly the Analytics & Outbound Automation
  angle: an event per turn, redacted at write time, with escalation rate,
  grounding rate and topic distribution surfaced in a dashboard tab — the numbers
  a product team would actually watch.

Good sentence to have ready: *"The clinic bot taught me what breaks in
production, which is why this one is built so the unsafe answers are impossible
rather than unlikely."*

---

## Demo script (3 minutes)

1. `I have crushing chest pain going into my left arm` — red badge, 112, then
   open the **Decision trace** and point at `answer path: deterministic-emergency`
   and the millisecond latency. "No model was called."
2. `I have no chest pain, just a mild fever` — same keywords, no escalation.
   "Negation scoping, and it's scoped per-pattern because some red flags are
   themselves phrased as negations — 'not breathing', 'no urine since morning'."
3. `my 6 week old baby has a fever` — emergency. "Age-aware. Any fever under
   three months."
4. `which antibiotic should I take for a sore throat?` — declines, explains,
   redirects. "Restricted intent, detected before the model sees it."
5. **Evaluation** tab. "52 cases, this is the CI gate."
6. **Insights** tab. "One redacted row per turn."

Don't demo the out-of-scope handling unless asked — it's the weakest heuristic
and you don't want to spend your best three minutes on it.
