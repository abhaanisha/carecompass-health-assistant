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

The app runs without one — it falls back to extractive answers from the corpus.
A key is what turns it into a chatbot that writes prose.

| Provider | Variable name | Free tier | Sign up |
| --- | --- | --- | --- |
| **Groq** (recommended) | `GROQ_API_KEY` | 30 req/min on `openai/gpt-oss-120b`, no credit card. Completed 10/10 answers in the provider A/B; see the README. | <https://console.groq.com/keys> |
| **Cerebras** | `CEREBRAS_API_KEY` | Generous, no credit card | <https://cloud.cerebras.ai/> |
| **Google Gemini** | `GEMINI_API_KEY` | Free tier on `gemini-flash-latest`, but throttled out of 9 of 10 answers when measured. Good as a second key, not a primary. | <https://aistudio.google.com/apikey> |
| OpenAI / Anthropic / OpenRouter / HF | `OPENAI_API_KEY` etc. | Paid | — |

The first key found wins, in the order above. Override the model with
`CARECOMPASS_MODEL` if a provider retires a default.

### Step 1 — create the key (about two minutes)

1. Open <https://console.groq.com/keys>
2. Sign in with Google or GitHub. No card, no billing setup.
3. **Create API Key**, give it a name such as `carecompass`
4. Copy it immediately — it starts with `gsk_` and is shown **once**. If you
   lose it, delete that key and make another; there is no way to view it again.

### Step 2 — put it where the code will find it

There are three places, depending on where the app is running. Use whichever
matches; you do not need more than one.

**Locally, all three entry points** — create a file called `.env` in the project
root:

```
GROQ_API_KEY=gsk_paste_your_real_key_here
```

No quotes, no spaces around the `=`, no `export`. `src/config.py` reads it at
import, so `streamlit run streamlit_app.py`, `python app.py` and the tests all
pick it up. `.env` is gitignored. Real environment variables take precedence
over the file, so a shell export always wins.

On Windows you can create it from PowerShell without opening an editor:

```powershell
Set-Content -Path .env -Value 'GROQ_API_KEY=gsk_paste_your_real_key_here'
```

**Streamlit Community Cloud** — the `.env` file is not deployed, and must not
be. In the app's dashboard: **Manage app → Settings → Secrets**, then paste
TOML and save:

```toml
GROQ_API_KEY = "gsk_paste_your_real_key_here"
```

Note the TOML syntax here: quotes and spaces around `=`, unlike the `.env` form.
The app restarts by itself within a few seconds.

**Render, Cloud Run, or a Hugging Face Gradio Space** — add it as an environment
variable or **secret** in that platform's settings UI. On Hugging Face choose
*secret*, not *variable*: variables are readable by anyone who can see the Space.

### Step 3 — verify it works

```bash
python -m src.llm
```

This prints every provider, marks which variables were found, and then sends one
real test message to the selected provider. `SUCCESS` means generated answers
are live. A `401` means the key is wrong or truncated. A `404` on the model name
means the provider retired the default — set `CARECOMPASS_MODEL` to a current
one from their docs.

In the running app, check the sidebar (Streamlit) or the header pill (Gradio).
It flips from "No model key — extractive answers" to the provider and model
name. Per answer, the decision trace shows `model-grounded` instead of
`retrieval-extractive`.

### Never do this

- Do not commit `.env` or `.streamlit/secrets.toml`. Both are gitignored; run
  `git status` before your first push and confirm neither is staged.
- Do not put a key in the static browser build. It would ship to every visitor.
  That build has no model layer precisely so the mistake is impossible.
- If a key is ever exposed, delete it in the provider console and issue a new
  one. Rotating takes ten seconds; a leaked key on a public repo does not.

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
"No model key — extractive answers" to "Groq · openai/gpt-oss-120b", which is
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
python -m pytest tests -q      # 81 tests
python -m eval.run_eval        # non-zero exit if emergency recall drops below 100%
python build_space.py          # regenerate the static bundle
```

And one rule that matters more than the rest: **never put an API key in the
static build.** It would ship to every visitor's browser. That is the whole
reason the browser build has no model layer.
