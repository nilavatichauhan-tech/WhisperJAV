"""
Ollama Manager — detection, lifecycle, and model management for Ollama integration.

Central class for all Ollama interaction. Zero new pip dependencies (uses urllib.request).

Usage:
    from whisperjav.translate.ollama_manager import OllamaManager

    mgr = OllamaManager()
    readiness = mgr.ensure_ready(model="huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M", auto_start=True)
    # readiness = {model, num_ctx, batch_size, temperature, server_started,
    #              base_url, supports_system_messages, thinking_model, uncensored}

    # Or let it pick for the detected VRAM — defaults to the uncensored tier,
    # because JAV input is explicit and aligned models refuse rather than fail:
    readiness = mgr.ensure_ready(auto_start=True)
"""

import atexit
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# Exceptions
# ============================================================================

class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    pass


class OllamaNotInstalledError(OllamaError):
    """Ollama binary not found on the system."""
    pass


class OllamaNotRunningError(OllamaError):
    """Ollama is installed but the server is not running."""
    pass


class ModelNotAvailableError(OllamaError):
    """Requested model is not pulled locally."""
    pass


# ============================================================================
# Model configuration
# ============================================================================

@dataclass
class ModelRecommendation:
    """Hardware-aware model recommendation."""
    name: str           # e.g. "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M"
    download_size: str  # e.g. "4.3 GB"
    quality: str        # "basic" | "good" | "very_good" | "excellent"
    note: str           # Human-readable explanation
    num_ctx: int        # Recommended context window
    batch_size: int     # Recommended batch size for this context
    temperature: float  # Recommended temperature
    uncensored: bool = False  # True when the model won't refuse explicit input


# Optimized sampling defaults for JAV subtitle translation.
# Based on per-model analysis of 10 local LLMs (range across models):
#   top_k: 40–70, median 50    — constrains hallucination without starving
#   top_p: 0.90–0.95, median 0.92 — tight enough to limit noise
#   min_p: 0.05 (unanimous)    — floor for token probability
#   repeat_penalty: 1.05–1.15, median 1.1 — prevents ASR fragment duplication
#
# These are set server-side via Ollama Modelfile PARAMETER directives because
# PySubtrans CustomClient._generate_request_body() only passes temperature
# to the API. By baking them into a lightweight model variant, the parameters
# take effect without any PySubtrans modification.
#
# NOTE: these are *fallback* values for models with no curated entry. A model
# whose authors published their own recommended sampling gets those instead,
# via the per-entry "sampling" key below — see the Hunyuan-MT note there.
# Mixing the two sets is worse than either: top_p 0.92 with a model tuned for
# top_p 0.6 widens the sampling tail well past what its authors validated.
OLLAMA_SAMPLING_DEFAULTS = {
    'top_k': 50,
    'top_p': 0.92,
    'min_p': 0.05,
    'repeat_penalty': 1.1,
}


# Sampling published by Tencent for the Hunyuan-MT family, copied verbatim from
# generation_config.json in tencent/Hunyuan-MT-7B:
#   {"top_k": 20, "top_p": 0.6, "repetition_penalty": 1.05, "temperature": 0.7}
# The higher temperature is safe *because* top_p is 0.6 — the two are a matched
# pair, and the effective entropy is lower than WhisperJAV's 0.3/0.92 combo.
# min_p is deliberately absent: Tencent never validated it, and adding a
# probability floor on top of an already-tight nucleus can clip valid tokens.
#
# repeat_penalty 1.05 (vs. the 1.1 default) also matters for this domain: JAV
# dialogue is legitimately repetitive — repeated moans, "dame, dame", stuttered
# names — and an aggressive repetition penalty suppresses correct output. The
# Ollama build of hy-mt1.5 ships repeat_penalty 1 for exactly this reason.
HUNYUAN_MT_SAMPLING = {
    'top_k': 20,
    'top_p': 0.6,
    'repeat_penalty': 1.05,
}


# ============================================================================
# Model catalog
# ============================================================================
#
# Two tiers, and the distinction is the whole point of this table:
#
#   uncensored=True  — will translate explicit JAV dialogue faithfully.
#   uncensored=False — safety-aligned; refuses or self-censors on explicit
#                      input, writing boilerplate into the subtitle track.
#                      See whisperjav/translate/refusal.py for what that looks
#                      like in practice and how it is detected.
#
# JAV source material is explicit by definition, so an aligned model is the
# wrong tool: it does not fail loudly, it silently degrades the output. The
# aligned entries are kept only so users who already pulled them still get
# correct num_ctx/batch_size, and so recommend_model(prefer_uncensored=False)
# has something to return.
#
# Every uncensored entry below was verified against the live Ollama registry
# (registry.ollama.ai/v2/<name>/manifests/<tag> → HTTP 200) and its size is the
# real sum of manifest layer sizes, not a guess.
OLLAMA_MODEL_CONFIGS = {
    # ── Uncensored tier (default) ────────────────────────────────────────
    # Hunyuan-MT-1.5-7B, abliterated. A dedicated *translation* model, not a
    # general chat model that happens to translate: Tencent's Hunyuan-MT took
    # first place in 30 of the 31 WMT25 language categories it entered, and
    # supports Japanese natively (33 languages total).
    #
    # Verified: 4.31 GiB of layers, model_family "hunyuan-dense",
    # model_type 7.5B, file_type Q4_K_M, 32768 max_position_embeddings.
    # Its Ollama template handles {{ .System }} and multi-turn {{ .Messages }},
    # so PySubtrans's system instruction + numbered batch protocol is delivered
    # intact — this was checked, because a specialist MT model with a
    # translate-only template would have broken batching entirely.
    #
    # Caveat worth knowing: Hunyuan-MT was trained on single-segment prompts
    # ("Translate the following segment into <lang>"), so a 20-line numbered
    # batch is out of its training distribution. batch_size is set lower than
    # the general-purpose models for that reason. If lines come back merged or
    # unnumbered, drop --max-batch-size further before switching models.
    #
    # No longer the 8 GB default: see the ladder note above for the measured
    # cue-desync and register failures. Kept here so an explicit
    # --translate-model still gets correct num_ctx/batch_size/sampling.
    "huihui_ai/hy-mt1.5-abliterated:7b": {
        "num_ctx": 8192,
        "batch_size": 8,
        "temperature": 0.7,          # paired with top_p 0.6 — see HUNYUAN_MT_SAMPLING
        # 4.31 GiB of manifest layers = 4.6 GB decimal, which is what
        # `ollama pull` prints. All download_size values here are decimal GB
        # so they match the progress bar the user actually sees.
        "download_size": "4.6 GB",
        "quality": "excellent",
        "min_vram_gb": 8,
        "uncensored": True,
        "sampling": HUNYUAN_MT_SAMPLING,
        # Ollama must be able to run the hunyuan-dense GGUF architecture.
        # llama.cpp added LLM_ARCH_HUNYUAN_DENSE on 2025-07-25 (PR #14878);
        # older Ollama builds fail with "unable to load model". Probed at
        # runtime rather than version-gated — see _check_architecture_support.
        "requires_arch": "hunyuan-dense",
    },
    # Hunyuan-MT v1 (pre-1.5), abliterated. Same architecture and size; kept as
    # a fallback in case a given Ollama build chokes on the 1.5 GGUF.
    "huihui_ai/hunyuan-mt-abliterated:latest": {
        "num_ctx": 8192,
        "batch_size": 8,
        "temperature": 0.7,
        "download_size": "4.6 GB",
        "quality": "very_good",
        "min_vram_gb": 8,
        "uncensored": True,
        "sampling": HUNYUAN_MT_SAMPLING,
        "requires_arch": "hunyuan-dense",
    },
    # General-purpose uncensored fallback. Not an MT specialist, but a plain
    # instruct model — so it follows the numbered-batch protocol more reliably
    # than Hunyuan does, and tolerates a larger batch. Use this if Hunyuan's
    # output comes back badly formatted.
    # The 8 GB default. Not the strongest translator on MT benchmarks — the
    # Hunyuan entry above wins those — but the better *subtitle* model, which
    # is a different property: it held 14/14 cues on every JAV probe run and
    # followed jav_glossary.txt where hy-mt1.5 kept reverting to clinical
    # register. "very_good" over hy-mt1.5's benchmark-derived "excellent"
    # reflects measured behaviour on this domain, and matters because a user
    # reading --list-models would otherwise override the default in favour of
    # the model that desyncs cues.
    "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M": {
        "num_ctx": 8192,
        "batch_size": 11,
        "temperature": 0.3,
        "download_size": "4.7 GB",
        "quality": "very_good",
        "min_vram_gb": 8,
        "uncensored": True,
    },
    "huihui_ai/qwen3-abliterated:4b": {
        "num_ctx": 8192,
        "batch_size": 8,
        "temperature": 0.3,
        "download_size": "2.5 GB",
        "quality": "good",
        "min_vram_gb": 4,
        "uncensored": True,
    },
    "huihui_ai/qwen2.5-abliterate:3b-instruct-q4_K_M": {
        "num_ctx": 4096,
        "batch_size": 8,
        "temperature": 0.3,
        "download_size": "1.9 GB",
        "quality": "basic",
        "min_vram_gb": 0,
        "uncensored": True,
    },
    "huihui_ai/qwen2.5-abliterate:14b-instruct-q4_K_M": {
        "num_ctx": 16384,
        "batch_size": 20,
        "temperature": 0.3,
        "download_size": "9.0 GB",
        "quality": "excellent",
        # 16, not 12. The weights are 8.37 GiB and 48 layers × 8 KV heads ×
        # 128 head_dim × 2 (K,V) × 2 B = 0.1875 MiB/token, so a 16K context adds
        # 3.0 GiB → ~12.3 GiB total with compute buffers. That overflows a 12 GB
        # card into shared memory. 12 GB users get hy-mt1.5 instead, which is
        # the better translation model anyway.
        "min_vram_gb": 16,
        "uncensored": True,
    },

    # Older uncensored option, kept because it appears in the GUI's curated
    # dropdown and users may already have it. Its headline feature is a 256K
    # context, which is irrelevant here — 8192 is all a subtitle batch needs,
    # and a 256K KV cache would not fit any consumer card (0.125 MiB/token
    # × 262144 = 32 GiB). num_ctx is pinned low on purpose.
    "dolphin-llama3:8b-256k-v2.9-q4_K_M": {
        "num_ctx": 8192,
        "batch_size": 11,
        "temperature": 0.3,
        "download_size": "4.9 GB",     # 4.58 GiB of layers — verified
        "quality": "good",
        "min_vram_gb": 8,
        "uncensored": True,
    },

    # ── Safety-aligned tier (opt-in only) ────────────────────────────────
    # These refuse or self-censor on explicit input. Retained for users who
    # already have them pulled; not recommended for JAV.
    "qwen2.5:3b": {
        "num_ctx": 4096,
        "batch_size": 8,
        "temperature": 0.3,
        "download_size": "2.0 GB",
        "quality": "basic",
        "min_vram_gb": 0,
        "uncensored": False,
    },
    "gemma3:4b": {
        "num_ctx": 8192,
        "batch_size": 11,
        "temperature": 0.3,
        "download_size": "2.5 GB",
        "quality": "good",
        "min_vram_gb": 4,
        "uncensored": False,
    },
    "qwen2.5:7b": {
        "num_ctx": 8192,
        "batch_size": 11,
        "temperature": 0.5,
        "download_size": "4.7 GB",
        "quality": "good",
        "min_vram_gb": 8,
        "uncensored": False,
    },
    # Same weights as qwen2.5:7b — Ollama's bare :7b tag *is* the instruct
    # build. Listed separately because the GUI's curated dropdown uses the
    # explicit tag, and an unlisted name would silently fall back to untuned
    # defaults.
    "qwen2.5:7b-instruct": {
        "num_ctx": 8192,
        "batch_size": 11,
        "temperature": 0.5,
        "download_size": "4.7 GB",     # 4.36 GiB of layers — verified
        "quality": "good",
        "min_vram_gb": 8,
        "uncensored": False,
    },
    "gemma3:12b": {
        "num_ctx": 8192,
        "batch_size": 11,
        "temperature": 0.5,
        "download_size": "8.1 GB",     # 7.59 GiB of layers — verified, does not fit 8 GB
        "quality": "very_good",
        "min_vram_gb": 12,
        "uncensored": False,
    },
    "qwen2.5:14b": {
        "num_ctx": 16384,
        "batch_size": 20,
        "temperature": 0.5,
        "download_size": "9.0 GB",
        "quality": "excellent",
        "min_vram_gb": 16,
        "uncensored": False,
    },
}


# Fraction of a ladder floor that a card is allowed to fall short by and still
# be counted as meeting it.
#
# The floors below are nameplate sizes ("an 8 GB card"), but no card reports
# nameplate. Measured on the RTX 4060 Laptop this targets:
#
#     nvidia-smi memory.total        8188 MiB   (board total)
#     torch.cuda.mem_get_info total  7797 MiB = 7.61 GiB  (CUDA-addressable)
#
# The driver keeps the 391 MiB difference for itself, so the largest number a
# CUDA allocation can ever see is ~4.8% under nameplate. A bare `>= 8` put this
# card in the 4 GB tier and handed an 8 GB card the 4B — a silent downgrade of
# exactly the thing this module exists to get right.
#
# Relative rather than a flat GiB margin because the reserve scales with the
# card: ~0.4 GiB on 8 GB, proportionally more on 24 GB. 8% covers the measured
# 4.8% with room for cards that also drive a display, and is still far inside
# the 4 GiB minimum gap between floors, so it cannot promote a card into a tier
# it genuinely cannot hold (a 6 GB card reports at most 6.0 GiB, well under the
# 7.36 GiB the 8 GB floor becomes).
#
# It also makes the tier agree across both detection paths: 7.61 GiB (torch)
# and 8.00 GiB (nvidia-smi) now land in the same tier, so the recommendation no
# longer depends on whether torch happened to be importable.
_VRAM_FLOOR_SLACK = 0.08

# VRAM ladders, largest-first. recommend_model() walks the matching list and
# takes the first entry the card can hold.
#
# Sizing note for the 8 GB tier (the common case — RTX 4060/4060 Laptop):
# hy-mt1.5 is 4.31 GiB of Q4_K_M weights, and its KV cache costs
# 2 (K,V) x 32 layers x 8 KV heads x 128 head_dim x 2 B (f16) = 0.125 MiB per
# token, so num_ctx 8192 adds 1.0 GiB. With ~0.5 GiB of compute buffers and
# ~0.35 GiB of CUDA context that totals ~6.2 GiB — fits 8 GB, but only on a
# card that isn't also holding the ASR model, which is why WhisperJAV runs
# translation as a separate phase after transcription.
#
# Note the 12 GB gap: hy-mt1.5 covers 8-15 GB rather than handing 12 GB cards
# the 14B. That is deliberate — a dedicated MT model at Q4 beats a general
# instruct model two sizes up on translation, and the 14B needs 16 GB at its
# configured 16K context (see its min_vram_gb comment).
_UNCENSORED_LADDER = [
    (16, "huihui_ai/qwen2.5-abliterate:14b-instruct-q4_K_M"),
    # 8 GB tier: qwen2.5-abliterate, not hy-mt1.5. Hunyuan-MT-1.5 is the
    # stronger benchmark translator on paper (first in 30 of 31 WMT25
    # categories) and was the default here, but measured on a 14-cue JAV probe
    # it failed two ways qwen2.5 did not:
    #
    #   1. Cue desync — one run returned 12 cues for 14 input cues and shifted
    #      every line after cue 8 into the wrong slot. For subtitles that is
    #      catastrophic and silent: the text is fluent, the timing is wrong.
    #      Non-deterministic, so it cannot be relied out. qwen2.5 held 14/14
    #      across every run.
    #   2. Register, even with the glossary supplied — it still returned
    #      "about to reach orgasm" for イッちゃう and "collapse" for 壊れちゃう,
    #      the exact renderings jav_glossary.txt names as forbidden, and it
    #      dropped the 先生 honorific that rule 3 says to keep.
    #
    # An MT specialist resists vulgar register: abliteration removes refusal,
    # not the training bias toward clean prose. hy-mt1.5 stays available as an
    # explicit --translate-model choice and at rank 3 in the catalogue.
    (8,  "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M"),
    (4,  "huihui_ai/qwen3-abliterated:4b"),
    (0,  "huihui_ai/qwen2.5-abliterate:3b-instruct-q4_K_M"),
]

_ALIGNED_LADDER = [
    (16, "qwen2.5:14b"),
    (12, "gemma3:12b"),
    (8,  "qwen2.5:7b"),
    (4,  "gemma3:4b"),
    (0,  "qwen2.5:3b"),
]


def get_sampling_params(model: str) -> dict:
    """Return the sampling parameters to apply to *model*.

    A curated entry's own ``sampling`` block wins over
    ``OLLAMA_SAMPLING_DEFAULTS`` — model authors who published validated
    sampling know their model better than our cross-model median does.
    """
    cfg = OLLAMA_MODEL_CONFIGS.get(model)
    if cfg and cfg.get('sampling'):
        return dict(cfg['sampling'])
    return dict(OLLAMA_SAMPLING_DEFAULTS)


def is_uncensored(model: str) -> bool:
    """True when *model* is known not to refuse explicit input.

    Unknown models fall back to a name heuristic: the community tags for
    de-aligned weights are stable enough to be worth trusting for a warning,
    though not for a silent default.
    """
    cfg = OLLAMA_MODEL_CONFIGS.get(model)
    if cfg is not None:
        return bool(cfg.get('uncensored', False))
    lower = model.lower()
    return any(tag in lower for tag in
               ('abliterat', 'uncensored', 'dolphin', 'unalign', 'lexi', 'tiger-gemma'))



# ============================================================================
# OllamaManager
# ============================================================================

class OllamaManager:
    """Central manager for Ollama server detection, lifecycle, and model management."""

    def __init__(self, base_url: str = None):
        """Initialize OllamaManager.

        Resolution order for base URL:
            1. Explicit base_url parameter
            2. OLLAMA_HOST environment variable
            3. http://localhost:11434 (default)
        """
        if base_url:
            self.base_url = base_url.rstrip('/')
        elif os.environ.get('OLLAMA_HOST'):
            self.base_url = os.environ['OLLAMA_HOST'].rstrip('/')
        else:
            self.base_url = 'http://localhost:11434'

        self._server_process = None

    # ── Server Detection & Lifecycle ──────────────────────────────────

    def detect_server(self) -> bool:
        """Check if Ollama server is running.

        Returns True if GET / responds with "Ollama is running".
        """
        try:
            self._http_get('/', timeout=2)
            # Ollama returns plain text "Ollama is running" at root
            return True
        except Exception:
            return False

    def detect_installation(self) -> Optional[str]:
        """Check if Ollama binary is installed.

        Returns path to the ollama binary, or None if not found.
        """
        return shutil.which("ollama")

    def start_server(self, timeout: int = 15) -> bool:
        """Start Ollama server via 'ollama serve' subprocess.

        Only starts if server is not already running. Registers atexit
        handler for cleanup. On Windows, uses CREATE_NEW_PROCESS_GROUP
        for proper subprocess management.

        Returns True if server started (or was already running).
        """
        if self.detect_server():
            return True

        ollama_path = self.detect_installation()
        if not ollama_path:
            return False

        # Start ollama serve as a subprocess
        kwargs = {
            'stdout': subprocess.DEVNULL,
            'stderr': subprocess.DEVNULL,
        }
        if platform.system() == 'Windows':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self._server_process = subprocess.Popen(
                [ollama_path, 'serve'],
                **kwargs
            )
        except OSError as e:
            print(f"[OLLAMA] Failed to start server: {e}", file=sys.stderr)
            return False

        # Register cleanup
        atexit.register(self.stop_server)

        # Poll until server is ready
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if self.detect_server():
                return True
            # Check if process died
            if self._server_process.poll() is not None:
                print("[OLLAMA] Server process exited unexpectedly", file=sys.stderr)
                self._server_process = None
                return False

        print(f"[OLLAMA] Server did not become ready within {timeout}s", file=sys.stderr)
        return False

    def stop_server(self):
        """Stop the server subprocess we started (if any).

        Does NOT stop a system-level Ollama service or tray app.
        """
        if self._server_process is not None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except Exception:
                try:
                    self._server_process.kill()
                except Exception:
                    pass
            self._server_process = None

    # ── Model Management ──────────────────────────────────────────────

    def list_models(self) -> list:
        """List locally available models.

        Returns list of model dicts from GET /api/tags.
        """
        resp = self._http_get('/api/tags', timeout=5)
        return resp.get('models', [])

    def check_model(self, name: str) -> bool:
        """Check if a specific model is available locally.

        Returns True if model exists, False otherwise.
        """
        try:
            self._http_post('/api/show', {'name': name}, timeout=5)
            return True
        except Exception:
            return False

    def list_running_models(self) -> list:
        """List models currently resident in VRAM/RAM.

        Returns the model dicts from GET /api/ps (each carries ``name``,
        ``size``, ``size_vram`` and ``expires_at``), or an empty list if the
        server cannot be reached.

        This exists so callers can free VRAM without having to know which model
        was loaded. Guessing the name wrongly means the unload silently no-ops
        and the weights sit in VRAM until Ollama's idle timeout — which starves
        the next file's ASR pass on an 8 GB card.
        """
        try:
            resp = self._http_get('/api/ps', timeout=5)
            return resp.get('models', []) or []
        except Exception:
            return []

    def get_model_info(self, name: str) -> dict:
        """Get full model metadata via POST /api/show.

        Returns model info dict, or empty dict on failure.
        """
        try:
            return self._http_post('/api/show', {'name': name}, timeout=5)
        except Exception:
            return {}

    def get_context_length(self, name: str) -> int:
        """Extract context length from model metadata.

        Checks model_info dict and parameters string for context_length
        or num_ctx. Falls back to 8192 if not found (old Ollama versions).
        """
        info = self.get_model_info(name)
        if not info:
            return 8192

        # Check model_info dict (newer Ollama versions)
        model_info = info.get('model_info', {})
        if isinstance(model_info, dict):
            # Various key names Ollama uses across versions
            for key in ('context_length', 'num_ctx',
                        'llama.context_length',
                        'general.context_length'):
                val = model_info.get(key)
                if val and isinstance(val, int | float):
                    return int(val)

        # Check parameters string (some Ollama versions embed it here)
        params_str = info.get('parameters', '')
        if isinstance(params_str, str):
            for line in params_str.splitlines():
                line = line.strip()
                if line.startswith('num_ctx'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[-1])
                        except ValueError:
                            pass

        return 8192  # Safe fallback

    def is_thinking_model(self, name: str) -> bool:
        """Check if a model uses thinking/reasoning mode by default.

        Qwen3-family models (and derivatives like shisa-v2.1-qwen3-*) put
        their output in a ``reasoning`` JSON field with empty ``content``.
        PySubtrans reads ``content`` only, so all translations silently fail.

        Appending ``/no_think`` to the user message disables thinking mode,
        forcing output into the ``content`` field where PySubtrans expects it.

        Returns True if the model name indicates a Qwen3-family model.
        """
        # Normalize: "hf.co/bartowski/shisa-v2.1-qwen3-8b-GGUF:Q4_K_M" → lowercase
        lower = name.lower()
        # Match qwen3 but NOT qwen3.5 — only Qwen3 has thinking mode,
        # Qwen3.5 is a standard generation model without reasoning tokens.
        # 'qwen3' in 'qwen3.5-9b' would false-positive, so we check that
        # the character after 'qwen3' is not '.' (which starts '3.5').
        import re
        return bool(re.search(r'qwen3(?!\.)', lower))

    def supports_system_messages(self, name: str) -> bool:
        """Check if a model's chat template handles system messages.

        Models pulled from hf.co/ as raw GGUFs often have a bare
        ``{{ .Prompt }}`` template that drops the system role entirely.
        When that happens, instructions sent as a system message are
        silently discarded and the LLM never sees them.

        Returns True if the template contains a reference to .System
        (meaning system messages are rendered), False otherwise.
        """
        info = self.get_model_info(name)
        if not info:
            return True  # Optimistic fallback if we can't check

        template = info.get('template', '')
        if not template:
            return False  # No template at all — cannot handle chat roles

        # Ollama Go templates use {{ .System }} to render system messages.
        # If the template doesn't reference it, system messages are dropped.
        return '.System' in template

    def pull_model(self, name: str, progress_callback: Callable = None) -> bool:
        """Pull (download) a model. Resumable via Ollama's built-in support.

        Args:
            name: Model name (e.g., "gemma3:12b")
            progress_callback: Optional callback(status, completed, total)

        Returns True on success.
        """
        import urllib.error
        import urllib.request

        url = f"{self.base_url}/api/pull"
        data = json.dumps({'name': name, 'stream': True}).encode('utf-8')
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')

        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except (urllib.error.URLError, OSError) as e:
            print(f"[OLLAMA] Failed to pull model: {e}", file=sys.stderr)
            return False

        # Read streaming NDJSON response
        try:
            for line in resp:
                line = line.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                status = msg.get('status', '')
                completed = msg.get('completed', 0)
                total = msg.get('total', 0)

                if progress_callback:
                    progress_callback(status, completed, total)
                elif total > 0:
                    pct = (completed / total) * 100 if total else 0
                    print(f"\r[OLLAMA] {status}: {pct:.0f}%", end='', file=sys.stderr)

                if msg.get('error'):
                    print(f"\n[OLLAMA] Pull error: {msg['error']}", file=sys.stderr)
                    return False
        finally:
            resp.close()

        if not progress_callback:
            print(file=sys.stderr)  # Newline after progress
        return True

    def unload_model(self, name: str) -> bool:
        """Unload a model from VRAM/RAM immediately.

        Sends a generate request with keep_alive=0 which tells Ollama to
        evict the model from memory right away instead of waiting for the
        default idle timeout (typically 5 minutes).

        Args:
            name: Model name (e.g., "gemma3:4b")

        Returns True if the request succeeded, False otherwise.
        """
        try:
            self._http_post('/api/generate', {
                'model': name,
                'keep_alive': 0,
            }, timeout=10)
            return True
        except Exception as e:
            print(f"[OLLAMA] Failed to unload model: {e}", file=sys.stderr)
            return False

    # ── Optimized Model Variants ─────────────────────────────────────

    def _ensure_optimized_variant(self, model: str) -> str:
        """Create or reuse an optimized model variant with tuned sampling params.

        PySubtrans CustomClient only passes ``temperature`` to the Ollama API.
        Other sampling parameters (top_k, top_p, min_p, repeat_penalty) are
        silently ignored. This method works around that limitation by creating
        a lightweight Ollama model variant with our recommended defaults baked
        in, so Ollama applies them server-side as the model's own defaults.

        The variant is a thin metadata layer — no model weight duplication.
        Ollama reuses the same GGUF blobs.

        Args:
            model: Original model name (e.g., "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M")

        Returns:
            Variant model name on success, original model name on failure.
        """
        import hashlib

        params = get_sampling_params(model)
        if not params:
            return model  # No parameters to apply

        # Deterministic variant name: hash of (model + params) ensures:
        #   - Same model + same params → same variant (reuse)
        #   - Params change in code update → new variant created
        # Uses the *effective* params, so two models with different published
        # sampling never collide on one variant.
        _key = f"{model}:{json.dumps(params, sort_keys=True)}"
        _hash = hashlib.sha256(_key.encode()).hexdigest()[:10]
        variant = f"whisperjav-{_hash}"

        # Reuse an existing variant only if it actually carries the parameters.
        #
        # This check is not paranoia. WhisperJAV <= v1.8.14 built the variant by
        # POSTing a ``modelfile`` key to /api/create, but ``modelfile`` is not a
        # field of Ollama's api.CreateRequest — Go's json decoder drops unknown
        # fields without complaining, so the request succeeded with an empty
        # ``from`` and no parameters, producing a broken variant that then looked
        # "present" to check_model() forever. Anyone who ran that build has a
        # dud whisperjav-* model on disk; validating instead of trusting
        # presence lets us rebuild it.
        if self.check_model(variant):
            if self._variant_has_params(variant, params):
                print(f"[OLLAMA] Using optimized variant: {variant}", file=sys.stderr)
                return variant
            print(f"[OLLAMA] Existing variant {variant} is missing its sampling "
                  f"parameters — rebuilding", file=sys.stderr)

        # Modern /api/create contract (Ollama >= 0.5): 'from' names the base
        # model and 'parameters' is a dict. The old Modelfile-string form was
        # removed; see the note above.
        payload = {
            'model': variant,
            'from': model,
            'parameters': params,
            'stream': False,
        }
        try:
            self._http_post('/api/create', payload, timeout=60)
        except Exception as e:
            # Fall back to the pre-0.5 Modelfile form for genuinely old servers.
            lines = [f'FROM {model}']
            lines += [f'PARAMETER {k} {v}' for k, v in params.items()]
            try:
                self._http_post('/api/create', {
                    'model': variant,
                    'modelfile': '\n'.join(lines),
                    'stream': False,
                }, timeout=60)
            except Exception as e2:
                print(f"[OLLAMA] Could not create optimized variant: {e} / {e2}",
                      file=sys.stderr)
                print(f"[OLLAMA]   Continuing with original model "
                      f"(only temperature will be tuned)", file=sys.stderr)
                return model

        # Verify rather than assume — a 200 from /api/create does not guarantee
        # the parameters landed (that is exactly how the old bug hid).
        if not self._variant_has_params(variant, params):
            print(f"[OLLAMA] Variant {variant} was created but does not report "
                  f"the expected sampling parameters — using original model",
                  file=sys.stderr)
            return model

        print(f"[OLLAMA] Created optimized variant: {variant}", file=sys.stderr)
        _params_str = ', '.join(f'{k}={v}' for k, v in params.items())
        print(f"[OLLAMA]   Sampling: {_params_str}", file=sys.stderr)
        return variant

    def _variant_has_params(self, variant: str, params: dict) -> bool:
        """Check that *variant* reports every key in *params* via /api/show.

        Ollama renders a model's baked-in parameters as a newline-separated
        ``"<name> <value>"`` block in the ``parameters`` field. Only key
        presence is checked, not exact values: Ollama normalises numeric
        formatting (``0.6`` may come back as ``0.6`` or ``0.600000``), so
        comparing strings would produce spurious rebuilds.
        """
        info = self.get_model_info(variant)
        if not info:
            return False
        params_str = info.get('parameters', '')
        if not isinstance(params_str, str) or not params_str.strip():
            return False
        present = {
            line.split()[0]
            for line in params_str.splitlines()
            if line.strip() and line.split()
        }
        return all(key in present for key in params)

    # ── Hardware-Aware Recommendation ─────────────────────────────────

    def recommend_model(self, vram_gb: float = None,
                        prefer_uncensored: bool = True) -> ModelRecommendation:
        """Recommend a model based on available VRAM.

        Args:
            vram_gb: available VRAM. Auto-detected via torch.cuda.mem_get_info()
                or nvidia-smi when omitted.
            prefer_uncensored: pick from the uncensored ladder (the default).
                JAV source material is explicit by definition, so a
                safety-aligned model does not merely perform worse — it writes
                refusal boilerplate into the subtitle track instead of
                translating. Pass False only to deliberately opt into the
                aligned tier.
        """
        if vram_gb is None:
            vram_gb = self._detect_vram_gb() or 0

        ladder = _UNCENSORED_LADDER if prefer_uncensored else _ALIGNED_LADDER
        for floor, name in ladder:
            if vram_gb >= floor * (1 - _VRAM_FLOOR_SLACK):
                pick = name
                break
        else:  # pragma: no cover — ladders end at floor 0, so this is unreachable
            pick = ladder[-1][1]

        cfg = OLLAMA_MODEL_CONFIGS[pick]
        # Same slack as the ladder, so the label cannot contradict the tier: a
        # card just under 4 GiB is handed the 4B, and calling it "CPU/low VRAM"
        # in the same breath would just look like a bug. One decimal rather
        # than none because ``:.0f`` overstated the card — it printed
        # "8 GB VRAM" for anything from 7.5 GiB up.
        if vram_gb < 4 * (1 - _VRAM_FLOOR_SLACK):
            where = 'CPU/low VRAM'
        else:
            where = f'{vram_gb:.1f} GB VRAM'
        note = f"Recommended for {where}"
        if cfg.get('uncensored'):
            note += " — uncensored (translates explicit dialogue faithfully)"
        else:
            note += " — SAFETY-ALIGNED: may refuse explicit dialogue"

        return ModelRecommendation(
            name=pick,
            download_size=cfg['download_size'],
            quality=cfg['quality'],
            note=note,
            num_ctx=cfg['num_ctx'],
            batch_size=cfg['batch_size'],
            temperature=cfg['temperature'],
            uncensored=bool(cfg.get('uncensored', False)),
        )

    # ── Orchestration ─────────────────────────────────────────────────

    def ensure_ready(
        self,
        model: str = None,
        auto_start: bool = True,
        auto_pull: bool = False,
        interactive: bool = True,
    ) -> dict:
        """Full orchestration: detect server, start if needed, check model, pull if needed.

        Returns:
            dict with keys: model, num_ctx, batch_size, temperature, server_started, base_url

        Raises:
            OllamaNotInstalledError: Ollama not found on system
            OllamaNotRunningError: Server not running and couldn't start
            ModelNotAvailableError: Model not available and couldn't pull
        """
        server_started = False

        # Step 1: Server detection
        if not self.detect_server():
            if not self.detect_installation():
                install_msg = self._platform_install_guide()
                raise OllamaNotInstalledError(
                    f"Ollama is not installed.\n\n{install_msg}"
                )

            if auto_start:
                print("[OLLAMA] Server not running, starting...", file=sys.stderr)
                if self.start_server():
                    server_started = True
                    print("[OLLAMA] Server started successfully", file=sys.stderr)
                else:
                    raise OllamaNotRunningError(
                        "Failed to start Ollama server. Try running 'ollama serve' manually."
                    )
            elif interactive:
                answer = input("Ollama server is not running. Start it? [Y/n] ").strip().lower()
                if answer in ('', 'y', 'yes'):
                    if self.start_server():
                        server_started = True
                        print("[OLLAMA] Server started successfully", file=sys.stderr)
                    else:
                        raise OllamaNotRunningError(
                            "Failed to start Ollama server. Try running 'ollama serve' manually."
                        )
                else:
                    raise OllamaNotRunningError("Ollama server is not running.")
            else:
                raise OllamaNotRunningError(
                    "Ollama server is not running. Start it with: ollama serve"
                )

        # Step 2: Model selection
        if model is None:
            rec = self.recommend_model()
            model = rec.name
            print(f"[OLLAMA] Auto-selected model: {model} ({rec.quality}, {rec.download_size})",
                  file=sys.stderr)
            print(f"[OLLAMA]   {rec.note}", file=sys.stderr)

        # Step 3: Normalize model name (add :latest if no tag)
        if ':' not in model:
            # Check if it matches a known config key with tag
            for cfg_key in OLLAMA_MODEL_CONFIGS:
                if cfg_key.startswith(model + ':'):
                    model = cfg_key
                    break
            else:
                model = model + ':latest'

        # Step 4: Check model availability
        if not self.check_model(model):
            cfg = OLLAMA_MODEL_CONFIGS.get(model, {})
            size_str = cfg.get('download_size', 'unknown size')

            if auto_pull:
                print(f"[OLLAMA] Model '{model}' not found locally, downloading ({size_str})...",
                      file=sys.stderr)
                if not self.pull_model(model):
                    raise ModelNotAvailableError(f"Failed to download model: {model}")
            elif interactive:
                answer = input(
                    f"Model '{model}' not found locally ({size_str}). Download? [Y/n] "
                ).strip().lower()
                if answer in ('', 'y', 'yes'):
                    if not self.pull_model(model):
                        raise ModelNotAvailableError(f"Failed to download model: {model}")
                else:
                    raise ModelNotAvailableError(
                        f"Model '{model}' not available. Pull it with: ollama pull {model}"
                    )
            else:
                raise ModelNotAvailableError(
                    f"Model '{model}' not available locally.\n"
                    f"Pull it with: ollama pull {model}"
                )

        # Step 5: Get context length and build config
        # Use the ORIGINAL model for metadata queries — the variant inherits
        # the same GGUF blobs, context length, and template.
        actual_ctx = self.get_context_length(model)

        # Use curated config if available, otherwise use dynamic values
        cfg = OLLAMA_MODEL_CONFIGS.get(model, {})
        num_ctx = cfg.get('num_ctx', min(actual_ctx, 8192))
        batch_size = cfg.get('batch_size', 11)
        # Default 0.3: best-fit compromise across 10 tested local models for
        # JAV subtitle translation (range 0.2–0.4, median 0.3). Keeps strong
        # models faithful while giving weaker models enough flexibility for
        # fragmented ASR input. Previous default (0.5) was too high.
        temperature = cfg.get('temperature', 0.3)

        # Step 6: Check if model template handles system messages
        has_system = self.supports_system_messages(model)
        if not has_system:
            print(f"[OLLAMA] Model template does not handle system messages — "
                  f"instructions will be embedded in user message", file=sys.stderr)

        # Step 6b: Check if model uses thinking/reasoning mode
        is_thinking = self.is_thinking_model(model)
        if is_thinking:
            print(f"[OLLAMA] Qwen3-family thinking model detected — will patch "
                  f"response parsing to extract translations from reasoning field",
                  file=sys.stderr)

        # Step 6c: Verify the runner can actually load this architecture.
        # Uncensored MT picks like hy-mt1.5 are hunyuan-dense GGUFs; llama.cpp
        # only gained LLM_ARCH_HUNYUAN_DENSE on 2025-07-25 (PR #14878), so an
        # older Ollama fails at load time with an opaque 500. Probing beats
        # version-gating — Ollama's vendored llama.cpp does not track its own
        # release numbers predictably.
        self._warn_if_architecture_unsupported(model, cfg)

        # Step 6d: Warn when a safety-aligned model was chosen. JAV input is
        # explicit by definition, so an aligned model does not fail loudly — it
        # writes refusal boilerplate into the .srt. refusal.py catches the
        # result afterwards; this is the up-front warning.
        uncensored = is_uncensored(model)
        if not uncensored:
            rec = self.recommend_model()
            print(f"[OLLAMA] WARNING: '{model}' is a safety-aligned model.",
                  file=sys.stderr)
            print(f"[OLLAMA]   It may refuse explicit dialogue and write refusal "
                  f"text into your subtitles instead of translating.",
                  file=sys.stderr)
            print(f"[OLLAMA]   Uncensored alternative for your hardware: {rec.name} "
                  f"({rec.download_size})", file=sys.stderr)

        # Step 6e: Create optimized variant with tuned sampling parameters.
        # PySubtrans only passes temperature to the API — top_k, top_p, min_p,
        # repeat_penalty are silently dropped. By baking them into a variant,
        # Ollama applies them server-side as the model's own defaults.
        effective_model = self._ensure_optimized_variant(model)

        # Step 7: Return readiness info
        return {
            'model': effective_model,
            'num_ctx': num_ctx,
            'batch_size': batch_size,
            'temperature': temperature,
            'server_started': server_started,
            'base_url': self.base_url,
            'supports_system_messages': has_system,
            'thinking_model': is_thinking,
            'uncensored': uncensored,
        }

    def _warn_if_architecture_unsupported(self, model: str, cfg: dict) -> None:
        """Print actionable guidance when the server may not run *model*'s arch.

        Only advisory — it never blocks the run, because the probe is a
        heuristic and a false negative would be worse than a wasted warning.
        ``/api/show`` reports the architecture the server parsed out of the
        GGUF; if it comes back empty for a model whose config declares a
        ``requires_arch``, the runner likely cannot load it.
        """
        required = cfg.get('requires_arch')
        if not required:
            return

        info = self.get_model_info(model)
        details = info.get('details') or {}
        model_info = info.get('model_info') or {}
        reported = (
            details.get('family')
            or (details.get('families') or [None])[0]
            or model_info.get('general.architecture')
            or ''
        )
        if reported and required.lower() in str(reported).lower():
            return  # Server recognises the architecture.

        print(f"[OLLAMA] NOTE: '{model}' needs the '{required}' GGUF architecture, "
              f"but this server reports '{reported or 'unknown'}'.", file=sys.stderr)
        print(f"[OLLAMA]   If loading fails with 'unable to load model' or "
              f"'unsupported architecture', update Ollama:", file=sys.stderr)
        print(f"[OLLAMA]     ollama --version   (then update if older than the "
              f"llama.cpp build that added {required})", file=sys.stderr)
        print(f"[OLLAMA]   Fallback that needs no special architecture: "
              f"huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M", file=sys.stderr)

    # ── Internal Helpers ──────────────────────────────────────────────

    def _detect_vram_gb(self) -> Optional[float]:
        """Detect available GPU VRAM in GB.

        Tries torch.cuda.mem_get_info() first, then nvidia-smi subprocess.
        Returns None if no GPU detected.
        """
        # Try torch first (often already imported)
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                return total / (1024 ** 3)
        except Exception:
            pass

        # Try nvidia-smi subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # First GPU's total memory in MiB
                mem_mib = float(result.stdout.strip().splitlines()[0])
                return mem_mib / 1024
        except Exception:
            pass

        return None

    def _http_get(self, path: str, timeout: int = 5) -> dict:
        """HTTP GET request to Ollama API. Returns parsed JSON or empty dict."""
        import urllib.error
        import urllib.request

        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method='GET')
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = resp.read().decode('utf-8', errors='replace')
        resp.close()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def _http_post(self, path: str, data: dict, timeout: int = 10) -> dict:
        """HTTP POST request to Ollama API. Returns parsed JSON."""
        import urllib.error
        import urllib.request

        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        resp = urllib.request.urlopen(req, timeout=timeout)
        resp_body = resp.read().decode('utf-8', errors='replace')
        resp.close()
        try:
            return json.loads(resp_body)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _platform_install_guide() -> str:
        """Return platform-specific Ollama installation instructions."""
        system = platform.system()
        if system == 'Windows':
            return (
                "Install Ollama for Windows:\n"
                "  1. Download from https://ollama.com/download/windows\n"
                "  2. Run the installer\n"
                "  3. Ollama will start automatically"
            )
        elif system == 'Darwin':
            return (
                "Install Ollama for macOS:\n"
                "  1. Download from https://ollama.com/download/mac\n"
                "  2. Or via Homebrew: brew install ollama\n"
                "  3. Start with: ollama serve"
            )
        else:
            return (
                "Install Ollama for Linux:\n"
                "  curl -fsSL https://ollama.com/install.sh | sh\n"
                "  Then start with: ollama serve"
            )
