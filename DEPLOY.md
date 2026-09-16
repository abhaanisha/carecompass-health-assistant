# Deploying

One engine, three front ends. Pick the host first, and the entry point follows.

| Host | Cost | Entry point | Generated prose | Wakes from idle | Notes |
| --- | --- | --- | --- | --- | --- |
| **Streamlit Community Cloud** | Free | `streamlit_app.py` | **Yes** | Sleeps after ~12 quiet hours | 1 GB RAM, secrets supported. Best full demo. |
| **HF Static Space** | Free | `space/index.html` | No | Always on | Runs in the browser. No server, no key. |
| **Render** | Free | `app.py` (Gradio) | **Yes** | Spins down after 15 min, 30–60 s cold start | 512 MB RAM, 750 hrs/month. |
| Cloud Run / Koyeb / Fly / VPS | Free tier or cheap | `Dockerfile` | **Yes** | Scales to zero, fast cold start | Needs a card and Docker. |
| HF Gradio Space | $9/month (PRO) | `app.py` | **Yes** | Always on | Gradio and Docker Spaces are no longer free. |

**Recommended pairing for a portfolio:** Streamlit Community Cloud as the real
chatbot, and the HF Static Space as an always-on backup link that never cold
starts. They share the engine, so there is nothing to keep in sync but a
`git push` and a `python build_space.py`.

---

## First: get a free model key

The app runs without one, but "generated prose" needs one. Two have no credit
card requirement at all:

| Provider | Secret name | Free tier | Sign up |
| --- | --- | --- | --- |
| **Groq** (recommended) | `GROQ_API_KEY` | 30 req/min, ~1,000 req/day on `llama-3.3-70b-versatile`. Very fast. | <https://console.groq.com/keys> |
| **Cerebras** | `CEREBRAS_API_KEY` | Generous free tier, also no card | <https://cloud.cerebras.ai/> |
| **Google Gemini** | `GEMINI_API_KEY` | Free tier on `gemini-2.5-flash` | <https://aistudio.google.com/apikey> |
| OpenAI / Anthropic / OpenRouter / HF | `OPENAI_API_KEY` etc. | Paid | — |

The first key found wins, in the order above. Override the model with
`CARECOMPASS_MODEL` if a provider retires a default.

Groq's daily cap is the right size for a demo: enough for an interviewer to play
with, not enough to be worth scraping.

---

# A. Streamlit Community Cloud (recommended)

Free, no credit card, 1 GB RAM, and apps only sleep after roughly 12 quiet hours
— they wake in seconds on the next visit. That idle behaviour is what makes it
better than Render for a link on a CV that gets opened unpredictably.

## 1. Push the repository to GitHub

```bash
cd carecompass-health-assistant
git init
git add .
git commit -m "CareCompass: safety-first health triage assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/carecompass-health-assistant.git
git push -u origin main
```

`.gitignore` already excludes `.streamlit/secrets.toml`, the event log and the
embedding cache. Check `git status` before the first push and confirm no key is
staged.

## 2. Deploy

1. Sign in at <https://share.streamlit.io> with the same GitHub account
2. **Create app → Deploy a public app from GitHub**
3. Repository: `<your-username>/carecompass-health-assistant`
4. Branch: `main`
5. **Main file path: `streamlit_app.py`**
6. Pick a URL slug — this becomes the link you share
7. Deploy

The build takes two to three minutes. `requirements.txt` deliberately excludes
torch, so it stays well inside the memory limit.

## 3. Add the key

**Manage app → Settings → Secrets**, then paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Save. The app restarts on its own. The sidebar pill switches from
"No model key — extractive answers" to "Groq · llama-3.3-70b-versatile", which is
the quickest way to confirm it took.

`.streamlit/secrets.toml.example` in the repository has the full list. Never
commit the real file.

## 4. Check it

- `I have crushing chest pain going into my left arm` → red EMERGENCY badge,
  `112`, appears instantly, and the decision trace says
  `deterministic-emergency` — no model was called
- `My 3 year old has had loose motions for two days and is drinking less` →
  streamed prose with `[S1]`-style citations, and the sources panel naming the
  passages
- `I have no chest pain, just a mild fever` → same keywords, no escalation
- **Evaluation** tab → "Run the 52-case gold set now"

## Troubleshooting

**Still says "No model key".** The secret name must match exactly, including
case. Reboot the app from the Manage screen if a save did not trigger a restart.

**`ModuleNotFoundError`.** The main file path must be `streamlit_app.py` at the
repository root, so that `src/` is importable.

**App is slow on first message.** The corpus is indexed once per container, held
by `@st.cache_resource`. Only the first request pays for it.

**Resource limit exceeded.** Something pulled in torch. Confirm
`requirements.txt` has no `sentence-transformers` line; the dense extra lives in
`requirements-dense.txt` for local use only.

---

# B. Hugging Face Static Space (free, always on, no key)

Hugging Face now requires PRO for Gradio and Docker Spaces. Static Spaces remain
free, and the browser build targets exactly that: the whole app compiled into
one page that runs client-side under Pyodide.

No server means no API key can be held safely, so this build answers from the
retrieved passages instead of generating prose. Everything else — the rules, the
triage floor, the citations, the multilingual emergency cards, the 52-case
evaluation — runs identically, in the visitor's browser.

## 1. Build

```bash
python build_space.py
```

Gzips `src/`, the knowledge base, the safety rules and the evaluation harness
into a self-contained `space/index.html` (~104 KB). Re-run after **any** change
to `src/`, `data/`, `eval/` or `app_lite.py` — the Space serves the bundle, not
the repository. `tests/test_build_space.py` fails if you forget.

## 2. Create the Space

<https://huggingface.co/new-space> — SDK **Static**, template **Blank**, public.

## 3. Push the two files

```bash
cd ..
git clone https://huggingface.co/spaces/<your-username>/carecompass-health-assistant space-deploy
cp carecompass-health-assistant/space/index.html space-deploy/index.html
cp carecompass-health-assistant/space/README.md  space-deploy/README.md
cd space-deploy && git add -A && git commit -m "CareCompass browser build" && git push
```

Use an access token with write scope as the password
(<https://huggingface.co/settings/tokens>). Drag-and-drop through the Space's
**Files** tab works just as well — there is no build step.

## 4. Check it

First load takes ~30 seconds while the browser downloads the Python runtime;
after that it is cached. The boot screen explains the wait so it does not read
as a hang.

## Troubleshooting

**Stuck on the boot screen.** Open the browser console. A `micropip` failure on
`pyyaml` or `numpy` usually means `cdn.jsdelivr.net` is blocked — some corporate
networks do that. Try another network before assuming the build is broken.

**Changes not showing.** You did not re-run `python build_space.py`, or did not
copy the regenerated `index.html`. Hard refresh after uploading.

---

# C. Render (keeps the Gradio app exactly as built)

Free web service: 512 MB RAM, 0.1 CPU, 750 instance-hours a month. It spins down
after 15 minutes of inactivity, and the next request waits 30–60 seconds. Fine
if you open the link yourself before an interview; irritating if a recruiter
opens it cold.

`render.yaml` in the repository is a complete blueprint.

1. Push to GitHub (as in section A)
2. <https://dashboard.render.com> → **New → Blueprint** → select the repository
3. Render reads `render.yaml`; the free plan, start command and port binding are
   already set
4. Add `GROQ_API_KEY` in the dashboard when prompted — the blueprint marks it
   `sync: false` so it never lives in the repository

750 hours covers a month of continuous uptime (~730 hours), so a free uptime
pinger hitting the URL every 10 minutes keeps it warm within budget.

---

# D. Docker hosts

`Dockerfile` builds an unprivileged image that serves the Streamlit build on
`$PORT`, which is what Google Cloud Run expects.

```bash
docker build -t carecompass .
docker run -p 8080:8080 -e GROQ_API_KEY=gsk_... carecompass
```

Cloud Run's free tier is generous and scales to zero with a much faster cold
start than Render, at the cost of needing a card on file and the `gcloud` CLI:

```bash
gcloud run deploy carecompass --source . --allow-unauthenticated \
  --region asia-south1 --set-env-vars GROQ_API_KEY=gsk_...
```

For the Gradio build in the same image, override the command:

```bash
docker run -p 8080:8080 -e GRADIO_SERVER_NAME=0.0.0.0 -e GRADIO_SERVER_PORT=8080 \
  carecompass python app.py
```

---

# E. Hugging Face Gradio Space (PRO, $9/month)

Push the whole repository to a Gradio Space. The root `README.md` already has
the right frontmatter (`sdk: gradio`, `app_file: app.py`).

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/carecompass-health-assistant
git push hf main
```

Add `GROQ_API_KEY` under **Settings → Variables and secrets** as a **secret**,
not a variable — variables are visible to anyone who can see the Space.

To get hybrid dense retrieval here, add `sentence-transformers>=3.0` to
`requirements.txt`. The build then takes several minutes and needs more than the
free CPU tier's memory.

---

## Before any deploy

```bash
python -m pytest tests -q      # 64 tests
python -m eval.run_eval        # non-zero exit if emergency recall drops below 100%
python build_space.py          # regenerate the static bundle
```

And one rule that matters more than the rest: **never put an API key in the
static build.** It would ship to every visitor's browser. That is the whole
reason the browser build has no model layer.
