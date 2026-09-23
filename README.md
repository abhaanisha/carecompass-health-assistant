---
title: CareCompass Health Assistant
emoji: "🧭"
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
short_description: "Safety-first health triage assistant with auditable rules"
---

## CareCompass Health Assistant

Live at: [https://abhaanisha-carecompass.streamlit.app/](https://abhaanisha-carecompass.streamlit.app/)

**A health information assistant where the safety-critical decisions are made by
auditable rules, not by a language model.**

Ask it about a symptom and it does three things before any text is generated: it
scans for red flags with negation-aware rules, assigns an urgency level tied to a
concrete timeframe, and retrieves passages from a curated public-health corpus.
Only then, and only if the situation is not an emergency, does a language model
write the reply — constrained to the retrieved passages, required to cite them,
and permitted to raise the urgency but never to lower it.

Every answer shows its working: which rules fired, which passages were used, and
whether a model was involved at all.

One engine, three front ends:

| | Streamlit | Gradio | Browser |
| --- | --- | --- | --- |
| Entry point | `streamlit_app.py` | `app.py` | `space/index.html` (generated) |
| Runs | On a server | On a server | Entirely in the visitor's browser, under Pyodide |
| Answers | Generated, streamed, cited | Generated and cited | Extractive, quoted from the corpus |
| Retrieval | BM25, hybrid where torch is installed | same | BM25 |
| Needs | A free API key | A free API key | Nothing at all |
| Free hosting | Streamlit Community Cloud | Render | Hugging Face Static Space |

The safety engine — red flags, triage floor, restricted intents, citation
validation — is byte-identical across all three and scores the same on the gold
set. `src/` contains every clinical decision; the three entry points are
presentation only. Swapping the UI framework cannot change what the assistant
treats as an emergency, which is the property the split is there to protect.

The browser build exists because a free Hugging Face account can host a Static
Space but not a Gradio one — and because "the safety-critical parts run
client-side with no network at all" turned out to be a fair summary of the whole
design argument.

---

## Try these

| Prompt | What it demonstrates |
| --- | --- |
| `I have crushing chest pain going into my left arm` | Emergency path — fixed text, no model call, under 10 ms |
| `I have no chest pain, just a mild fever` | Negation scoping — the same keywords, correctly *not* escalated |
| `my 6 week old baby has a fever` | Age-aware rules — any fever under 3 months is an emergency |
| `seene me dard ho raha hai aur paseena aa raha hai` | Romanised Hindi — same red flag, Hindi emergency card |
| `বুকে ব্যথা হচ্ছে আর ঘাম হচ্ছে` | Bengali — fixed emergency text, not a machine translation |
| `which antibiotic should I take for a sore throat?` | Restricted intent — declines to prescribe, explains why |
| `how much salt per day is safe with high blood pressure?` | Informational question, correctly not treated as a complaint |
| `i am in pain what to do?` | Too vague to ground — asks what a clinician would ask, instead of refusing |
| `write me a python function to sort a list` | Out-of-scope detection — refuses instead of improvising |

---

## Why it is built this way

### 1. The safety-critical path contains no language model

When a red flag fires, the words the user sees are fixed text from
[`src/prompts.py`](src/prompts.py). The ambulance number cannot be paraphrased,
the advice cannot drift between runs, and no instruction in a later turn can talk
the assistant out of it. Generation is reserved for the cases where phrasing is
the hard part, not for the cases where being wrong has a cost.

This also makes the emergency path fast — a median of **8 ms**, because nothing
leaves the process.

### 2. The model can escalate, never de-escalate

The rule engine computes an urgency **floor**. The model returns its own opinion
as a tag, and the two are merged with `max()`:

```python
if parsed > result.floor:
    result.level = parsed              # model raised it — accepted
elif parsed < result.floor:
    result.reasons.append(...)         # model lowered it — recorded and ignored
```

The asymmetry is deliberate. Missing an emergency and telling someone to rest is
a different category of error from sending someone to hospital unnecessarily, so
the system is not allowed to make the first mistake in exchange for fewer of the
second.

### 3. Every answer degrades rather than fails

No API key, an expired key, a rate limit, a timeout or a dead provider all fall
through to an extractive answer assembled from the retrieved passages and clearly
labelled as such. **The Space runs with no secrets configured at all** — which is
also why a reviewer can open it and get real answers without being asked for a
key.

---

## How a turn is processed

```
user message
    │
    ├─▶ 1. safety scan          data/red_flags.yaml — 19 rules, 3 languages
    │                           negation scoped per-pattern, not per-rule
    │
    ├─▶ 2. triage floor         5 levels, each tied to a timeframe
    │                           context modifiers (pregnancy, infant, elderly,
    │                           immunocompromised) add one level
    │
    ├─▶ 3. hybrid retrieval     BM25 + MiniLM embeddings, fused with Reciprocal
    │                           Rank Fusion; lay-term lexicon expands
    │                           "loose motion" → diarrhoea, ORS, dehydration
    │
    ├─▶ 3b. web tier            only if the corpus did not answer confidently,
    │                           and only for a specific question. An allow-list
    │                           of public-health bodies, not a web search.
    │                           Passages come back tagged [W1]…[Wn]
    │
    ├─▶ 4a. EMERGENCY or self-harm → fixed card. No model is called,
    │        and no network call is made.
    │
    ├─▶ 4b. a symptom too vague to match anything → ask what a clinician would
    │        ask, in the user's language, rather than refuse
    │
    └─▶ 4c. otherwise → model answers from the retrieved passages only,
                        must cite [S1]…[Sn] and [W1]…[Wn], may raise urgency
                        ↓ on any failure
                        extractive answer from the same passages
    │
    └─▶ 5. citation validation → follow-up suggestions → disclaimer
           → redacted event log
```

### The web tier

The curated corpus covers eleven areas well and nothing else at all. That is
honest, but it means a question about shingles, gout or thyroid used to fall
through to "I could not find anything in my knowledge base" — or worse, get
answered from a near-miss passage with no citation, because the grounding check
is tuned to be generous about what counts as in scope.

So there is a second shelf. When the corpus does not answer confidently, the
question goes to an allow-list of public-health bodies — WHO, MoHFW, ICMR, NHS,
CDC, MedlinePlus — and what comes back enters the pipeline exactly as corpus
text does: numbered, citable passages with a URL and a retrieval timestamp,
tagged `[W1]` rather than `[S1]` so the reader can always tell reviewed content
from content fetched a second ago.

Four things keep this from undoing the rest of the design:

- **It is an allow-list, not a search of the open web.** A blog, a forum or a
  supplement seller cannot become a source, whatever a search engine returns.
  The host is re-checked after redirects.
- **It never touches the safety-critical path.** Emergency and self-harm cards
  are produced before the tier is consulted, so no network call can sit between
  a user and an ambulance number.
- **It cannot lower urgency.** The triage floor comes from the user's own
  message, computed before any retrieval.
- **It is additive.** No network, a slow host, a malformed payload, a missing
  `requests` — every failure ends with the corpus answering alone.

It needs no API key: MedlinePlus (US National Library of Medicine) publishes a
free, unauthenticated search service that returns full topic summaries, so the
default path is one HTTP request and no HTML scraping. A Tavily, Brave or
Serper key, if present, widens the reach to the rest of the allow-list.

`CARECOMPASS_WEB_MODE=off` disables it entirely; `always` consults it on every
non-emergency turn.

### Follow-up suggestions

Each answer ends with two or three things the user might reasonably ask next,
rendered as chips under the latest reply. The model writes them in the user's
own voice and in the reply language, emitted as a `[[FOLLOWUPS: … | … ]]` tag
that is stripped before display. When no model is configured, or when one
forgets the tag, they are derived from the triage level and the retrieved
heading instead — keyed off the level rather than the topic, because "should I
go to a hospital or a clinic?" is worth more to someone who has just been told
to seek care today than an offer to explain the condition further.

They are never shown on the emergency or crisis path. Nothing on that card
should invite a next question instead of a phone call.

### Retrieval

Hybrid, because neither half is sufficient on its own:

- **Dense alone misses exact tokens.** A question containing `HbA1c` or `112`
  must land on the passage containing that literal string, and a 384-dimension
  MiniLM vector does not reliably preserve a rare token.
- **Lexical alone misses paraphrase.** "I feel like I can't catch my breath"
  shares no content word with "severe breathlessness".

The two rankings are fused with **Reciprocal Rank Fusion** rather than a weighted
score blend, because BM25 scores and cosine similarities live on different,
corpus-dependent scales — any fixed `alpha` needs retuning every time the corpus
changes, and rank fusion does not.

Three details that came out of testing rather than design:

- **Query expansion for Indian English.** `loose motion`, `sugar`, `bp`,
  `bukhar`, `শ্বাসকষ্ট` and ~70 other lay terms are mapped to the clinical
  vocabulary the corpus actually uses. Embedding models trained on English web
  text have never seen "shugar patient ko pair ki dekhbhal".
- **A small suffix stemmer**, because a user types "dehydrated" and the corpus
  says "dehydration".
- **IDF-weighted grounding.** Whether a question is in-scope is decided by how
  much of its *information content* the best passage accounts for, not by raw
  word overlap. "my 3 year old has loose motions since morning" is mostly
  incidental detail; "python" and "france" are absent from the corpus entirely
  and are charged maximum IDF.

---

## Does it work?

52 hand-labelled cases in [`eval/goldset.yaml`](eval/goldset.yaml), spanning
cardiac, stroke, respiratory, gastro, fever, maternal and child health, mental
health, medication safety, chronic disease, first aid, negation traps and
out-of-scope questions, in English, Hindi and Bengali.

| Metric | Result |
| --- | --- |
| **Emergency recall** | **100.0%** |
| Emergency precision | 100.0% |
| Red-flag recall | 100.0% |
| Negation specificity (no false alarms) | 100.0% |
| Retrieval recall@5 | 100.0% |
| Restricted-intent recall | 100.0% |
| Out-of-scope detected | 100.0% |
| Under-triage rate | 3.8% |
| Over-triage rate | 0.0% |
| Exact urgency match | 96.2% |
| Median latency (deterministic path) | 8 ms |

The two residual failures are both `INFO` vs `SELF_CARE` boundary cases
("can I take ibuprofen for dengue fever?"), where the two labels carry the same
practical advice. They are reported rather than relabelled, because a gold set
tuned to match the implementation measures nothing.

Accuracy is deliberately not the headline. **Under-triage** is: a system that
sends everyone to hospital scores badly on precision and is still safer than one
that misses a stroke, so the two error directions are reported separately and
never averaged together.

```bash
python -m eval.run_eval                 # deterministic paths, no key needed
python -m eval.run_eval --model         # also checks generated text for dose leakage
python -m eval.run_eval --lexical-only  # ablation: how much does the dense half earn?
python -m eval.compare_providers        # A/B the configured model vendors
```

### Which model provider?

Answered by measurement rather than by vendor benchmarks, which score things
this application does not care about. `eval/compare_providers.py` runs the gold
cases that actually reach a model and scores what matters here: citation
discipline, prescription leakage, agreement with the hand-labelled urgency, and
latency.

| Metric | Groq `openai/gpt-oss-120b` | Gemini `gemini-flash-latest` |
| --- | --- | --- |
| Answers completed, out of 10 | **10** | 1 |
| Citation rate | **88.9%** | n/a |
| Restricted intents handled cleanly | **100%** | n/a |
| Urgency agreement with gold label | **85.7%** | n/a |
| Urgency under-called | **0%** | n/a |
| Median latency | **1.9 s** | 3.2 s |
| Rate-limited into the fallback path | **0** | 9 |

Groq is the default, and the decisive number is the last row rather than any
quality metric: on the free tier Gemini was throttled out of nine of ten
answers, which is not a demo. Where Groq did answer, it under-called urgency
zero times. Gemini is kept configured as a second key, because the failure the
fallback exists for is exactly the one it demonstrated.

Two honest caveats. Gemini produced too few completed answers to judge its
quality, so this is a measure of usable throughput on a free tier, not of model
capability. And gpt-oss emits the `[[URGENCY]]` tag only ~70% of the time; when
it is missing the rule floor governs, which is the safe default, so the cost is
a lost second opinion rather than a wrong answer.

`run_eval` exits non-zero if emergency recall drops below 100% or a dose-like
instruction appears in generated text, so it works as a CI gate.

---

## Running it

```bash
git clone <this repo> && cd carecompass-health-assistant
pip install -r requirements.txt

streamlit run streamlit_app.py          # Streamlit build, http://localhost:8501
python app.py                           # Gradio build,    http://127.0.0.1:7860
python app_lite.py                      # browser build, run natively for debugging
python build_space.py                   # regenerate space/index.html

pip install -r requirements-dense.txt   # optional: adds the dense retriever (torch)
```

`sentence-transformers` is deliberately **not** in `requirements.txt`. It pulls
in torch, which does not fit the 512 MB–1 GB ceilings of free hosting, and the
gold set scores identically without it on this corpus. The retriever detects its
absence at startup and runs BM25-only, reporting `lexical` instead of `hybrid`.

It runs with no configuration. To enable generated answers, set any one of:

| Variable | Provider | Default model | Free tier |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Groq — **default, see the comparison above** | `openai/gpt-oss-120b` | Yes, no card |
| `CEREBRAS_API_KEY` | Cerebras | `llama-3.3-70b` | Yes, no card |
| `GEMINI_API_KEY` | Google Gemini | `gemini-flash-latest` | Yes |
| `OPENAI_API_KEY` | OpenAI | `gpt-4o-mini` | No |
| `ANTHROPIC_API_KEY` | Anthropic | `claude-sonnet-5` | No |
| `OPENROUTER_API_KEY` | OpenRouter | `meta-llama/llama-3.3-70b-instruct` | Some models |
| `HF_TOKEN` | Hugging Face Inference | `meta-llama/Llama-3.1-8B-Instruct` | Limited |

The first key found wins, in that order. Override with `CARECOMPASS_PROVIDER`
and `CARECOMPASS_MODEL`. There are no vendor SDKs in `requirements.txt` — two
HTTP request shapes cover all five providers, so no vendor's breaking release
can take the Space down.

Other useful switches: `CARECOMPASS_DISABLE_DENSE=1` (lexical-only, skips the
torch download), `CARECOMPASS_DISABLE_LOGGING=1`, `CARECOMPASS_TOP_K`,
`CARECOMPASS_DEBUG=1`.

Deployment to Hugging Face Spaces is in [DEPLOY.md](DEPLOY.md).

---

## Layout

```
streamlit_app.py            Streamlit front end — Disha, the assistant persona
app.py                      Gradio front end
app_lite.py                 browser front end — no pandas, no model layer
build_space.py              bundles everything into space/index.html for a Static Space
space/index.html            generated: the entire app in one 104 KB file
Dockerfile, render.yaml     deployment for container hosts and Render
src/
  config.py                 settings and the provider registry
  knowledge.py              markdown → citable chunks, split on section headings
  retrieval.py              BM25 + dense + RRF, stemming, IDF-weighted grounding
  safety.py                 red flags, negation scoping, restricted intents, PII redaction
  triage.py                 5 urgency levels, rule floor, model-merge logic
  prompts.py                system prompt + the fixed emergency and crisis cards
  llm.py                    provider-agnostic HTTP client, retry, graceful failure
  web.py                    the web tier: domain allow-list, MedlinePlus, HTML→text
  pipeline.py               orchestration: one call in, one audited answer out
  analytics.py              JSONL event log and the Insights metrics
  language.py               script-based language detection
data/
  knowledge/*.md            12 documents with source attribution and review dates
  red_flags.yaml            the safety rules, readable without knowing Python
  translations.yaml         Hindi and Bengali text for every rule
  lexicon.yaml              lay-term → clinical-term expansion
eval/
  goldset.yaml              52 labelled cases
  run_eval.py               metrics, failure list, CI gate
tests/                      110 tests covering safety, triage, retrieval, pipeline,
                            the web tier and its guards, follow-ups, bundle, detector
```

The rules live in YAML on purpose. A clinician reviewing whether "fever in an
infant under 3 months" is handled correctly should not have to read Python to
check, and adding a red flag should not require a deploy of new code.

---

## The knowledge base

Twelve documents written from public health guidance (WHO, NHS, UNICEF, and
Indian national programme material), each carrying `source`, `source_url` and
`last_reviewed` in frontmatter, which flows through to the citation shown in the
UI. Chunking is at the `##` section boundary rather than a fixed token count,
because each section is already a self-contained clinical idea, and because a
citation that names a heading is something a reader can actually go and check.

Topics: emergency services, fever and dengue, respiratory and TB, diarrhoea and
hydration, cardiac and stroke, diabetes, hypertension, maternal and child health,
mental health and crisis support, medication safety and AMR, first aid, and
preventive care.

Adding a document is dropping a markdown file into `data/knowledge/` with
frontmatter. The embedding cache invalidates itself on a content hash.

---

## Limitations

This is a demonstration project, and the honest list matters more than the
feature list:

- **The corpus is a compact demo set, not a clinical guideline library.** Real
  deployment needs clinician-reviewed content with an approval workflow,
  versioning, and enforced re-review dates.
- **Only the emergency path is natively multilingual.** Every red flag has
  hand-written Hindi and Bengali text in `data/translations.yaml`, so the
  critical response is fixed in all three languages. Everything else is
  translated on the fly by the model from an English corpus, and the retrieved
  passage shown in the sources panel stays in English.
- **Out-of-scope detection is a heuristic.** It catches the obvious cases; an
  oddly-phrased off-topic question can still retrieve a passage, at which point
  the system prompt is the remaining defence.
- **Rules are regex, and regex misses paraphrase.** A red flag phrased in a way
  no pattern anticipates will not fire. This is why the model is allowed to
  escalate — it is the backstop for exactly this gap.
- **Event logs are ephemeral on Spaces.** The container filesystem resets on
  restart. Point `CARECOMPASS_LOG_DIR` at a volume, or stream to a warehouse.
- **No memory between sessions, and no clinician in the loop.** Both would be
  prerequisites for anything beyond a demonstration.

### If this were going further

Clinician review workflow for the corpus · retrieval evaluation against a larger
labelled set with recall@k per topic · a second-opinion model vote on urgency
with disagreement logged for review · streaming responses · per-state emergency
numbers · a proper translation layer for the corpus · rate limiting and abuse
handling · structured escalation to a real telemedicine queue.

---

## Disclaimer

CareCompass provides general health information from a curated public-health
knowledge base. It does not diagnose, prescribe, or replace a qualified
clinician. In an emergency, call **112**.

MIT licensed. Built as a portfolio project.

**Abha Singh Sardar**, IISc — [abhaanisha.github.io](https://abhaanisha.github.io/)
