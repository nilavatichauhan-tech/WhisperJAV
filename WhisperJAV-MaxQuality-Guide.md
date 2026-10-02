# WhisperJAV — Max-Quality Qwen-Only Uncensored Run

**Verified on this machine, 2026-08-30.** RTX 4060 Laptop (8188 MiB total, ~7.4 GiB
free), torch 2.10.0+cu128 with CUDA, Ollama running.

Qwen-only — no Whisper, no ensemble, no merge step. Every flag below was checked
against the real `--help` and proved to resolve with `--dump-params` (exit 0).

---

## 1. Run it in TWO stages — this is a VRAM requirement, not a preference

```bash
cd /home/Tobi/.gemini/antigravity/scratch/WhisperJAV-1.9.1
OUT="$HOME/Desktop/wj_out"

# ── STAGE 1 — ASR (Qwen only) ─────────────────────────────────────────────
./wj "/path/to/video.mp4" \
  --mode qwen \
  --qwen-model-id jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame \
  --qwen-generator qwen3 \
  --qwen-scene semantic \
  --qwen-segmenter silero-v6.2 \
  --qwen-enhancer none \
  --qwen-sensitivity aggressive \
  --qwen-timestamp-mode aligner_interpolation \
  --qwen-timestamps word \
  --output-dir "$OUT" \
  --log-level INFO

# ── STAGE 2 — uncensored translation ──────────────────────────────────────
./wj translate -i "$OUT"/*.srt \
  --provider ollama \
  --model huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M \
  --tone standard \
  --source japanese \
  --target english
```

`input` accepts **a file, several files, a directory, or a wildcard**, so stage 1
batches: `./wj "/media/jav/*.mp4" ...same flags...`

**Always `./wj`, never bare `whisperjav`.** The venv has whisperjav installed
*editable pointing at the old 1.8.14 tree*; `wj` prefixes `PYTHONPATH` with this
clone so v1.9.1 actually runs. Check with `./wj --version` → `WhisperJAV 1.9.1`.

### ⚠️ Why two stages and not one `--translate` flag

`--translate` **would work but will thrash your GPU.** Verified in `main.py`:
translation fires at line 1542, but `pipeline.cleanup()` is at line 1611 — so the
ASR model is **still resident** when the 4.7 GB translation model loads. The code
says so itself at line 1579:

> *"Free the LLM before the next file's ASR pass. Translation runs per-file here
> **while the ASR model stays loaded**, so leaving 4.6 GB of LLM resident
> overflows an 8 GB card."*

The `finally` block evicts the LLM *after* translation — protecting the **next**
file, not the current one. Do the arithmetic on your card:

| Resident at translation time | One command | Two stages |
|---|---|---|
| Qwen3-ASR finetune | 4.1 GB | 4.1 GB → **process exits, freed** |
| ForcedAligner 0.6B | 1.2 GB | 1.2 GB → freed |
| Ollama translation LLM | 4.7 GB | 4.7 GB (alone) |
| **Peak** | **≈10 GB — over your 8.19 GB** ❌ | **5.3 GB, then 4.7 GB** ✅ |

Overflow spills into shared system memory and transcription slows to a crawl
rather than crashing, so it's easy to miss. Two stages avoids it entirely.

> This split is specific to Qwen-only. In `--ensemble` mode each pass runs in a
> `spawn`ed subprocess that fully exits before the next starts, so its VRAM is
> reclaimed by the OS and one-command translation is safe. You lose that free
> cleanup by going single-pass.

---

## 2. Why each flag

| Flag | Reason |
|---|---|
| `--mode qwen` | Single-pass Qwen3-ASR. |
| `--qwen-model-id jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` | Full finetune of Qwen3-ASR-1.7B on `litagin/Galgame_Speech_ASR_16kHz` — Japanese visual-novel/anime speech. In-domain for JAV audio (breathy, close-mic, emotional Japanese) in a way generic ASR is not. 2.04 B params. **−42 % deletions vs base.** Already the code's built-in default, so this flag is belt-and-braces. |
| `--qwen-sensitivity aggressive` | Most permissive VAD — catches quiet moans, whispers, breathy speech that `balanced` drops. With no second pass to fill gaps, **recall is your binding constraint.** |
| `--qwen-scene semantic` | Semantic scene splitting rather than raw energy. |
| `--qwen-segmenter silero-v6.2` | Newest Silero VAD. Plain `silero` silently resolves to **v4.0** — name the version. |
| `--qwen-enhancer none` | Enhancement is denoising built for speech intelligibility; on JAV it scrubs the breathy/wet detail the finetune was trained on. Only enable per-file if audio is genuinely noisy. |
| `--qwen-timestamp-mode aligner_interpolation` | **The big quality win — see below.** |
| `--qwen-timestamps word` | Word granularity. Inert unless an aligner mode is on, which is why it pairs with the flag above. |

### The timestamp mode is the one real upgrade over defaults

The qwen-mode default is `vad_only`, which the code documents as:

> *"**Discard aligner timestamps entirely.** Distribute all segments
> proportionally by character count across the full duration."*

Cue boundaries are **guessed from character counts**, not acoustics. Several lines
inside one VAD region get evenly spaced timings regardless of when they were
actually spoken. `aligner_interpolation` instead:

> *"Keep valid aligner timestamps; mathematically interpolate null-timestamp
> segments based on character length between anchors."*

Real word-level forced-alignment anchors, with interpolation only for gaps. It is
the *pipeline's own* native default (`decoupled_pipeline.py:187`); qwen mode
overrides it to `vad_only` **only to save ~1 GB VRAM.** You have that GB free
precisely because you dropped the Whisper pass — so spend it.

Enabling an aligner mode also un-inerts two features that do nothing under
`vad_only`:

- `--qwen-stepdown` (default **on**) — retries collapsed groups at a fallback size.
- `--qwen-timestamps word` — word granularity instead of segment.

Other modes: `aligner_vad_fallback` (legacy proportional spread for nulls),
`aligner_only` (leaves nulls as zeros — diagnostic only, don't ship it).

### Already optimal — leave alone

- `--qwen-coverage-rescue` — **on by default.** Recovers acoustically active regions VAD missed. Directly serves recall; never disable this for JAV.
- `--qwen-temperature-fallback` — **on by default.** Retries degenerate/looping output.
- `--qwen-postprocess-preset high_moan` — **already the JAV default.** Nothing to do.
- `--qwen-repetition-penalty 1.0` — greedy base pass, the official Qwen3-ASR recipe. Escape-valve penalties engage only on fallback retries. **Raising this without `--qwen-do-sample` is wrong.**
- `--qwen-framer vad-grouped`, `--qwen-regroup standard`, `--qwen-assembly-cleaner` — current tuned defaults.

### One judgment call you may want to flip

`--qwen-drop-nonverbal-lines` is **on by default** and drops lone single-token
nonverbal lines (`あ。 は。 え。 ん。 つ。 ふ。 ふっ。 切。`). Those are usually ASR
noise, but on JAV some are real vocalisations. For absolute maximum recall:

```bash
--no-qwen-drop-nonverbal-lines
```

Expect more clutter — many will translate to bare "Ah." lines. Default (drop) is
tuned for JAV; try disabling only if you can hear content being lost.

---

## 3. First run downloads ~4 GB

| Model | Size | Status |
|---|---|---|
| `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` | **~4.1 GB** | Not cached yet |
| `Qwen/Qwen3-ForcedAligner-0.6B` | ~1.2 GB | **Already cached** ✅ |
| `huihui_ai/qwen2.5-abliterate:7b` (Ollama) | 4.7 GB | **Already installed** ✅ |

The HF repo also holds an 8.15 GB `optimizer.pt` **training artifact —
`from_pretrained` does not fetch it.** Ignore the ~12 GB repo figure on the
website. 80 GB free on `/var/home`.

**Going Qwen-only saves the ~3 GB Whisper `large-v2` download entirely** — nothing
here needs it.

Pre-pull if you'd rather not wait mid-run:

```bash
/home/Tobi/.gemini/antigravity/scratch/WhisperJAV/.venv/bin/huggingface-cli \
  download jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame \
  --include "*.json" "*.txt" "model.safetensors" "tokenizer.json"
```

---

## 4. VRAM commands

Two-stage already keeps peak at ~5.3 GB. To watch or force:

```bash
# Live
watch -n1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# What Ollama holds right now
curl -s http://localhost:11434/api/ps | python3 -m json.tool

# Force-evict translation models before stage 1
ollama stop huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M 2>/dev/null
for m in $(ollama ps 2>/dev/null | grep -o 'whisperjav-[a-f0-9]*'); do ollama stop "$m"; done
```

Ollama holds a model ~5 min after use, so **run that eviction before stage 1** if
you just translated something. `release_translation_vram()` handles it
automatically after stage 2 — and correctly: it asks the server what is actually
resident rather than guessing, because translation runs through a tuned
`whisperjav-<hash>` variant that unloading the base name would miss.

---

## 5. Why `standard` tone, not `pornify`

`standard` is already the default — confirmation, not a change. Measured on a
14-cue probe with the glossary active. `pornify` runs at **temperature 1.2** vs
standard's **0.5**, and that cost accuracy:

| Japanese | `standard` ✅ | `pornify` ❌ |
|---|---|---|
| 私のお**まんこ**、濡れてる | "My **pussy** is already so wet" | "I'm so wet already" — *anatomy word dropped* |
| 早く**挿れて** | "put it in faster" | "**Come in me** soon" — *wrong act; collides with cue 13* |
| もっと激しく**突いて** | "**thrust** me harder" | "Harder right there" — *verb dropped* |
| お願い**先生** | "please, **teacher**" | "Mr. **[Teacher]**" — *template leak* |
| あぁ…んっ…はぁ | "Ahh…mm…hah" | "Ahh…mm…**haha**" — *reads as laughter* |

pornify's only gains were cosmetic (keeps more interjections). Both held 14/14
cues. Higher temperature buys blunter phrasing at the cost of fidelity — and
fidelity is what the glossary exists to enforce.

### The glossary does the uncensoring — and it was broken until today

`whisperjav/translate/defaults/jav_glossary.txt` supplies mandatory renderings
(おちんちん→cock, おまんこ→pussy, おっぱい→tits, 中に出して→cum inside me,
イク→I'm cumming, 気持ちいい→feels so good *not* "wonderful").

There were **two** instruction resolvers. The standalone `translate` CLI used its
own, which never injected the glossary — so it shipped the tone file with no
terminology and produced clinical output. **Fixed.** Since stage 2 above *is* that
CLI, this fix is exactly what makes your run uncensored:

| source | before | after |
|---|---|---|
| おちんちん / おまんこ / おっぱい | penis / vagina / breasts | **cock / pussy / tits** |
| 咥えてあげる、口に出していいよ | "I'll take care of your mouth; you can lick it" | **"I'll take it in my mouth. You can cum in my mouth."** |
| 中に出して、全部中に | "Fill me up, from beginning to end" | **"Cum inside me, all of it!"** |

### Model: qwen2.5-abliterate, not hy-mt1.5

Hunyuan-MT-1.5 is the better *benchmark* translator (1st in 30 of 31 WMT25
categories) and was the previous default. Replaced because it returned **12 cues
for 14 input**, shifting every line after cue 8 into the wrong timestamp — silent,
fluent, wrong timing, non-deterministic. It also ignored glossary terms supplied
verbatim. qwen2.5 held 14/14 every run.

**An MT specialist resists vulgar register: abliteration removes refusal, not the
training bias toward clean prose.** hy-mt1.5 is still selectable via `--model`.

---

## 6. Sanity checks

```bash
cd /home/Tobi/.gemini/antigravity/scratch/WhisperJAV-1.9.1

./wj --version                                   # -> WhisperJAV 1.9.1
./wj --check                                     # dependency / env check
curl -s http://localhost:11434/api/tags >/dev/null && echo "ollama up"
./wj translate --list-ollama-models              # what's installed

# Resolve stage 1 WITHOUT transcribing
./wj "/path/to/video.mp4" <stage-1 flags> --dump-params /tmp/params.json
```

For stage 1 the dump should read `mode qwen | sens aggressive | ensemble False |
lang ja`. Note the dump is **shallow for non-ensemble mode** — it won't echo the
`--qwen-*` values back, so don't read their absence as them being ignored
(`main.py:1361` consumes `qwen_model_id`; verified).

### Refusal / leak detector

Translated output is scanned for refusals, `[Content omitted]` placeholders, and
bracketed template leaks like `Mr. [Teacher]`. It is **advisory — it warns, it
does not retry.** Read the `[TRANSLATE]` block at the end of stage 2. On the
measured probe: `standard` 0 hits, `pornify` 1.

---

## 7. Optional knobs (unmeasured here — change one at a time)

- `--no-qwen-drop-nonverbal-lines` — max recall of vocalisations; more clutter. See §2.
- `--qwen-num-beams 2` — beam search instead of greedy (default `1`). Plausibly more accurate, costs VRAM and time. You have headroom in stage 1. **Not measured.**
- `--qwen-enhancer clearvoice` — only for genuinely noisy audio.
- `--qwen-batch-size 2` — default `1`; may speed up long files if VRAM allows.
- `--qwen-context "…"` / `--qwen-context-file` — feed title/scene hints to bias recognition.
- `--output-format both` (SRT + VTT), `--skip-existing` (resume a batch), `--keep-temp` (keep intermediates for diagnosis).

Also available on stage 2: `--movie-title`, `--actress`, `--movie-plot` — context
that can improve pronoun and name handling.

---

## 8. Housekeeping

Nine `whisperjav-*` Ollama variants exist from A/B runs. `ollama list` shows
~4.7 GB each, but **real disk is 8.7 GB total** — Ollama dedupes shared layers.
To prune stale ones (keeps the newest):

```bash
ollama list | grep whisperjav- | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama rm
```

**Delete the `.subtrans` cache before re-translating the same file**, or already
translated batches are reused and you compare against stale output:

```bash
rm -f "$OUT"/*.subtrans
```

---

## 9. State of this tree

- Nothing committed; the 1.8.14 tree at `../WhisperJAV` is untouched.
- Test suite: **149 failed / 2094 passed / 73 skipped / 8 xfailed / 7 errors.** The
  149 are the **pre-existing upstream v1.9.1 baseline** — verified 0 regressions
  from this work. Only a *rise* above ~149 is meaningful.
- Running the suite needs 13 `--ignore=tests/...` flags; some test modules call
  `sys.exit()` or reassign `sys.stdout` at module level, killing collection.

### Security

The Hugging Face token pasted into chat should be **rotated** at
<https://huggingface.co/settings/tokens> — it sits in `~/.claude/settings.json`
and `~/.cache/huggingface/token` (both mode 600) and has been through a
transcript. Nothing above needs it: all models are public.
