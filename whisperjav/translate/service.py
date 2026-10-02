"""
Translation Service Layer.

Provides a high-level API for subtitle translation that can be called
directly from the main pipeline without subprocess invocation.

This module encapsulates:
- Configuration resolution (settings + parameters)
- Provider setup and validation
- Instruction file fetching/caching
- Translation execution via core.py

Usage from main.py:
    from whisperjav.translate.service import translate_with_config

    result_path = translate_with_config(
        input_path=output_path,
        provider=args.translate_provider,
        target_lang=args.translate_target,
        tone=args.translate_tone,
        api_key=args.translate_api_key
    )
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .providers import PROVIDER_CONFIGS, SUPPORTED_TARGETS
from .settings import load_settings, DEFAULT_SETTINGS
from .instructions import get_instruction_content
from .core import translate_subtitle, _normalize_api_base, _api_base_to_custom_server, cap_batch_size_for_context, compute_max_output_tokens

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when translation fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def _resolve_api_key(provider_config: dict, api_key: Optional[str] = None) -> str:
    """
    Resolve API key from parameter or environment variable.

    Args:
        provider_config: Provider configuration dict with 'env_var' key
        api_key: Explicit API key (highest priority)

    Returns:
        Resolved API key

    Raises:
        ConfigurationError: If no API key found
    """
    if api_key:
        return api_key

    env_var = provider_config.get('env_var')
    if env_var:
        key = os.getenv(env_var)
        if key:
            return key

    raise ConfigurationError(
        f"API key not found. Set {env_var} environment variable or provide api_key parameter."
    )


def _resolve_model(provider_config: dict, model: Optional[str] = None, settings: Optional[dict] = None) -> str:
    """
    Resolve model name with precedence: parameter > settings > provider default.

    Args:
        provider_config: Provider configuration dict with 'model' key
        model: Explicit model override
        settings: User settings dict

    Returns:
        Resolved model name
    """
    if model:
        return model

    if settings and settings.get('model'):
        return settings['model']

    return provider_config['model']


_GLOSSARY_MARKER = "JAV TERMINOLOGY — MANDATORY RENDERINGS"


def _load_jav_glossary() -> Optional[str]:
    """Load the bundled JAV terminology glossary, or None if missing."""
    try:
        path = Path(__file__).parent / 'defaults' / 'jav_glossary.txt'
        return path.read_text(encoding='utf-8').strip() or None
    except Exception as e:
        logger.debug(f"JAV glossary unavailable: {e}")
        return None


def _inject_glossary(content: str) -> str:
    """Append the JAV terminology glossary to an instruction set's
    ``### instructions`` section.

    Why this is applied here rather than edited into ``defaults/*.txt``:
    ``get_instruction_content()`` prefers a remote gist and only falls back to
    the bundled files, so a glossary living in ``pornify.txt`` silently does
    nothing whenever the gist is reachable. This function sits downstream of
    every source (gist, cache, bundled), so the glossary applies to all three.

    Why it is needed at all: both bundled tones already say "avoid euphemisms",
    and measured against hy-mt1.5 that instruction alone was not enough —
    「中に出して」 came back as "let it happen" (meaning dropped entirely) and
    「イッちゃう」 as "I am about to reach orgasm". Naming the specific
    renderings fixed both. A translation-specialist model follows a term list
    far more reliably than it follows a register adjective.

    The glossary must land *inside* ``### instructions``: PySubtrans splits this
    file on ``###`` headings, so appending at the end of the file would file the
    glossary under ``### retry_instructions`` and it would only be seen on a
    retry.
    """
    glossary = _load_jav_glossary()
    if not glossary:
        return content
    if _GLOSSARY_MARKER in content:
        return content  # already present (e.g. a gist that bundles it)

    lines = content.splitlines()
    # Locate the ### instructions heading, then the next ### heading after it.
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().lower().startswith('### instructions')), None)
    if start is None:
        # No section to extend. Injecting blind risks landing in the wrong
        # section, and a wrong-section glossary is worse than none.
        logger.debug("No '### instructions' section; glossary not injected")
        return content
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].lstrip().startswith('###')), len(lines))

    body = lines[:end] + ['', glossary, ''] + lines[end:]
    return '\n'.join(body)


def _resolve_instruction_file(tone: str = "standard", refresh: bool = False) -> Optional[str]:
    """
    Resolve instruction file path by fetching content and caching to temp file.

    Args:
        tone: Translation tone ('standard' or 'pornify')
        refresh: Force refresh of cached content

    Returns:
        Path to instruction file, or None if unavailable
    """
    instruction_content = get_instruction_content(tone=tone, refresh=refresh)

    if not instruction_content:
        logger.debug(f"No instruction content available for tone: {tone}")
        return None

    instruction_content = _inject_glossary(instruction_content)

    # Save to temp file
    temp_dir = Path(tempfile.gettempdir()) / 'whisperjav_translate'
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f'instructions_{tone}.txt'

    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(instruction_content)

    return str(temp_file)


def _build_provider_options(
    tone: str = "standard",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    settings_model_params: Optional[dict] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Build provider options with tone-aware defaults.

    Precedence: explicit params > settings > curated model sampling > tone defaults

    The curated tier sits above the tone defaults because the values in
    ``OLLAMA_MODEL_CONFIGS[...]['sampling']`` come from the model authors'
    own published recommendations, which describe that model better than a
    single cross-model default can. hy-mt1.5, for instance, publishes
    ``top_p 0.6``; the tone default is 0.9.

    That mattered concretely: ``OllamaManager._ensure_optimized_variant()``
    bakes the curated sampling into the ``whisperjav-<hash>`` variant as
    Modelfile PARAMETERs, but Ollama lets per-request options override a
    Modelfile, so passing ``top_p 0.9`` here silently defeated the variant's
    0.6 on every single call. Meanwhile ``top_k`` and ``repeat_penalty`` were
    never sent at all, so they only survived by accident — via the variant —
    and were lost on the fallback path where variant creation fails and the
    base model is used directly. Sending all three explicitly makes the
    sampling identical on both paths.

    Args:
        tone: Translation tone for default selection
        temperature: Explicit temperature override
        top_p: Explicit top_p override
        settings_model_params: Model params from settings file
        model: Resolved model name, used to look up curated sampling

    Returns:
        Dict of provider options
    """
    # Start from tone-aware defaults
    if tone == 'pornify':
        default_temperature = 1.2
        default_top_p = 0.9
    else:
        default_temperature = 0.5
        default_top_p = 0.9

    result_temp = default_temperature
    result_top_p = default_top_p
    extra_sampling = {}

    # Curated per-model sampling beats the tone default.
    if model:
        try:
            from whisperjav.translate.ollama_manager import get_sampling_params
            curated = get_sampling_params(model) or {}
        except Exception as e:
            logger.debug(f"No curated sampling for {model}: {e}")
            curated = {}
        if curated.get('top_p') is not None:
            result_top_p = float(curated['top_p'])
        if curated.get('temperature') is not None:
            result_temp = float(curated['temperature'])
        # Forwarded so they do not depend on the tuned variant existing.
        for key in ('top_k', 'repeat_penalty', 'min_p'):
            if curated.get(key) is not None:
                extra_sampling[key] = curated[key]

    # Apply settings overrides
    if settings_model_params:
        if settings_model_params.get('temperature') is not None:
            try:
                result_temp = float(settings_model_params['temperature'])
            except (ValueError, TypeError):
                pass
        if settings_model_params.get('top_p') is not None:
            try:
                result_top_p = float(settings_model_params['top_p'])
            except (ValueError, TypeError):
                pass

    # Apply explicit parameter overrides (highest priority)
    if temperature is not None:
        result_temp = temperature
    if top_p is not None:
        result_top_p = top_p

    # Clamp to valid ranges
    result_temp = max(0.0, min(2.0, result_temp))
    result_top_p = max(0.0, min(1.0, result_top_p))

    return {
        'temperature': result_temp,
        'top_p': result_top_p,
        **extra_sampling,
    }


def translate_with_config(
    input_path: str,
    provider: str = "deepseek",
    target_lang: str = "english",
    tone: str = "standard",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_lang: str = "japanese",
    output_path: Optional[str] = None,
    instruction_file: Optional[str] = None,
    scene_threshold: Optional[float] = None,
    max_batch_size: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: bool = False,
    debug: bool = False,
    extra_context: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    n_gpu_layers: int = -1,
    endpoint: Optional[str] = None,
    ollama_url: Optional[str] = None,
    auto_confirm: bool = False,
    ollama_max_tokens: Optional[int] = None,
    ollama_num_ctx: Optional[int] = None,
) -> Optional[Path]:
    """
    Translate subtitle file with full configuration resolution.

    This is the primary entry point for programmatic translation.
    Can be called directly from main.py without subprocess overhead.

    Configuration is resolved with the following precedence:
    1. Explicit parameters (highest priority)
    2. User settings file (~/.config/WhisperJAV/translate/settings.json)
    3. Built-in defaults (lowest priority)

    Args:
        input_path: Path to input SRT file
        provider: AI provider name ('deepseek', 'openrouter', 'gemini', 'claude', 'gpt')
        target_lang: Target language ('english', 'chinese', 'indonesian', 'spanish')
        tone: Translation tone ('standard' or 'pornify')
        api_key: API key (or set via environment variable)
        model: Model override (uses provider default if not specified)
        source_lang: Source language ('japanese', 'korean', 'chinese')
        output_path: Output file path (auto-generated if not specified)
        instruction_file: Custom instruction file path
        scene_threshold: Scene threshold in seconds
        max_batch_size: Maximum batch size for translation
        temperature: Model temperature (0.0-2.0)
        top_p: Model top_p (0.0-1.0)
        stream: Stream translation progress
        debug: Enable debug output
        extra_context: Additional context for translation (movie title, etc.)
        progress_callback: Optional callback for progress updates
        endpoint: Custom API endpoint URL (for OpenAI-compatible APIs)

    Returns:
        Path to translated file, or None on failure

    Raises:
        ConfigurationError: Invalid provider or missing API key
        TranslationError: Translation execution failed
        FileNotFoundError: Input file not found
    """
    # Validate input file exists
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Validate provider
    provider = provider.lower()
    provider_config = PROVIDER_CONFIGS.get(provider)
    if not provider_config:
        valid_providers = list(PROVIDER_CONFIGS.keys())
        raise ConfigurationError(f"Unknown provider: {provider}. Valid providers: {valid_providers}")

    # Validate target language
    if target_lang not in SUPPORTED_TARGETS:
        raise ConfigurationError(f"Unsupported target language: {target_lang}. Valid: {SUPPORTED_TARGETS}")

    # Custom provider requires an endpoint
    if provider == 'custom' and not endpoint:
        raise ConfigurationError(
            "Custom provider requires --translate-endpoint. Example:\n"
            "  --translate-provider custom --translate-endpoint http://localhost:11434/v1"
        )

    # Override endpoint if custom endpoint provided (same logic as cli.py)
    if endpoint:
        provider_config = dict(provider_config)  # Copy to avoid mutating original
        psn = provider_config.get('pysubtrans_name')
        if psn in ('OpenAI', 'DeepSeek'):
            # These providers handle api_base natively via their SDKs
            provider_config['api_base'] = _normalize_api_base(endpoint)
        else:
            # Route through Custom Server for reliable /chat/completions
            # access without reasoning model misclassification (#178)
            server_addr, endpoint_path = _api_base_to_custom_server(endpoint)
            provider_config['pysubtrans_name'] = 'Custom Server'
            provider_config['server_address'] = server_addr
            provider_config['endpoint'] = endpoint_path
            provider_config.pop('api_base', None)

    # Load user settings
    settings = load_settings()

    # Resolve API key (not needed for local/custom/ollama providers)
    if provider in ('local', 'custom', 'ollama'):
        resolved_api_key = api_key or ''
    else:
        resolved_api_key = _resolve_api_key(provider_config, api_key)

    # Resolve model
    resolved_model = _resolve_model(provider_config, model, settings)

    # Resolve instruction file
    if instruction_file:
        if not Path(instruction_file).exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        resolved_instruction_file = instruction_file
    else:
        resolved_instruction_file = _resolve_instruction_file(tone)

    # Resolve processing options
    resolved_scene_threshold = scene_threshold if scene_threshold is not None else settings.get('scene_threshold', 60.0)
    resolved_max_batch_size = max_batch_size if max_batch_size is not None else settings.get('max_batch_size', 30)

    # Build provider options
    provider_options = _build_provider_options(
        tone=tone,
        temperature=temperature,
        top_p=top_p,
        settings_model_params=settings.get('model_params'),
        model=resolved_model,
    )

    # Generate output path if not specified
    if output_path:
        resolved_output_path = Path(output_path)
    else:
        stem = input_file.stem
        # Remove existing language suffix if present
        parts = stem.split('.')
        if len(parts) > 1 and parts[-1] in ['japanese', 'english', 'ja', 'en', 'jp', 'chinese', 'indonesian', 'spanish']:
            stem = '.'.join(parts[:-1])
        resolved_output_path = input_file.parent / f"{stem}.{target_lang}.srt"

    # Log configuration
    logger.info(f"Translation: {input_file.name} -> {target_lang}")
    logger.debug(f"Provider: {provider} ({resolved_model})")
    logger.debug(f"Tone: {tone}")

    # =========================================================================
    # DIAGNOSTIC: Configuration Summary
    # =========================================================================
    print(f"\n[SERVICE] Translation Configuration:", file=sys.stderr)
    print(f"[SERVICE]   Input file: {input_file}", file=sys.stderr)
    print(f"[SERVICE]   Output file: {resolved_output_path}", file=sys.stderr)
    print(f"[SERVICE]   Provider: {provider}", file=sys.stderr)
    print(f"[SERVICE]   Model: {resolved_model}", file=sys.stderr)
    print(f"[SERVICE]   Source: {source_lang} -> Target: {target_lang}", file=sys.stderr)
    print(f"[SERVICE]   Tone: {tone}", file=sys.stderr)
    print(f"[SERVICE]   Max batch size: {resolved_max_batch_size}", file=sys.stderr)
    print(f"[SERVICE]   Scene threshold: {resolved_scene_threshold}s", file=sys.stderr)
    print(f"[SERVICE]   Stream: {stream}", file=sys.stderr)
    print(f"[SERVICE]   Provider options: {provider_options}", file=sys.stderr)
    if resolved_instruction_file:
        print(f"[SERVICE]   Instructions: {resolved_instruction_file}", file=sys.stderr)
    else:
        print(f"[SERVICE]   Instructions: (none)", file=sys.stderr)

    # Report progress if callback provided
    if progress_callback:
        progress_callback(f"Translating {input_file.name} from {source_lang} to {target_lang}...")
        progress_callback(f"Provider: {provider} ({resolved_model})")

    # Execute translation
    try:
        # For Ollama provider: use OllamaManager for server detection + model readiness
        if provider == 'ollama':
            from .ollama_manager import OllamaManager

            print(f"\n[SERVICE] Ollama Provider Mode", file=sys.stderr)
            mgr = OllamaManager(base_url=ollama_url or endpoint)

            readiness = mgr.ensure_ready(
                model=resolved_model,
                auto_start=True,
                auto_pull=auto_confirm,
                interactive=False,  # Non-interactive from service layer
            )

            resolved_model = readiness['model']
            n_ctx = readiness['num_ctx']
            # User --ollama-num-ctx override applied before batch_size cap
            # and max_tokens computation so both honor the override.
            if ollama_num_ctx is not None:
                n_ctx = ollama_num_ctx
            resolved_max_batch_size = cap_batch_size_for_context(resolved_max_batch_size, n_ctx)
            max_tokens = compute_max_output_tokens(resolved_max_batch_size, n_ctx)
            if ollama_max_tokens is not None:
                max_tokens = ollama_max_tokens

            if temperature is None and readiness.get('temperature'):
                provider_options['temperature'] = readiness['temperature']
            provider_options['num_ctx'] = n_ctx

            provider_config = dict(provider_config)
            provider_config['max_tokens'] = max_tokens
            resolved_api_key = ''
            stream = True  # Always stream for local models

            # Override supports_system_messages based on actual model template.
            # Models with bare templates (e.g. {{ .Prompt }}) silently drop
            # system messages — instructions must go in the user message instead.
            if not readiness.get('supports_system_messages', True):
                provider_config['supports_system_messages'] = False

            if readiness.get('base_url'):
                server_addr, endpoint_path = _api_base_to_custom_server(readiness['base_url'])
                provider_config['server_address'] = server_addr
                provider_config['endpoint'] = endpoint_path

            print(f"[SERVICE]   Model: {resolved_model}", file=sys.stderr)
            print(f"[SERVICE]   num_ctx={n_ctx}, batch_size={resolved_max_batch_size}, max_tokens={max_tokens}", file=sys.stderr)

            result_path = translate_subtitle(
                input_path=str(input_file),
                output_path=resolved_output_path,
                provider_config=provider_config,
                model=resolved_model,
                api_key='',
                source_lang=source_lang,
                target_lang=target_lang,
                instruction_file=resolved_instruction_file,
                scene_threshold=resolved_scene_threshold,
                max_batch_size=resolved_max_batch_size,
                stream=stream,
                debug=debug,
                provider_options=provider_options,
                extra_context=extra_context,
                emit_raw_output=True,
            )

        # For local provider WITHOUT custom endpoint: start llama.cpp server
        # If local + endpoint: skip server launch, use the endpoint directly (cloud path)
        elif provider == 'local' and not endpoint:
            import warnings
            warnings.warn(
                "provider='local' is deprecated as of v1.8.10 and will be removed in v1.9.0. "
                "Use provider='ollama' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            from .local_backend import start_local_server, stop_local_server

            # =========================================================================
            # DIAGNOSTIC: Local LLM Provider
            # =========================================================================
            print(f"\n[SERVICE] Local LLM Provider Mode", file=sys.stderr)
            print(f"[SERVICE]   n_gpu_layers: {n_gpu_layers} (-1 = all layers on GPU)", file=sys.stderr)

            # Start local LLM server
            print(f"[SERVICE] Starting local LLM server...", file=sys.stderr)
            try:
                api_base, server_port, server_diagnostics = start_local_server(
                    model=resolved_model,
                    n_gpu_layers=n_gpu_layers
                )
                print(f"[SERVICE]   Server started at: {api_base}", file=sys.stderr)
                print(f"[SERVICE]   Port: {server_port}", file=sys.stderr)
                if server_diagnostics.inference_speed_tps > 0:
                    print(f"[SERVICE]   Inference speed: {server_diagnostics.inference_speed_tps:.1f} tokens/sec", file=sys.stderr)
            except Exception as e:
                print(f"[SERVICE]   ERROR: Failed to start local server: {e}", file=sys.stderr)
                raise TranslationError(f"Failed to start local server: {e}")

            # Use Custom Server provider - designed for local OpenAI-compatible servers
            # This uses /v1/chat/completions endpoint which llama-cpp-python supports
            server_address = api_base.replace('/v1', '')
            local_provider_config = {
                'pysubtrans_name': 'Custom Server',
                'server_address': server_address,
                'endpoint': '/v1/chat/completions',
                'supports_conversation': True,
                'supports_system_messages': True,
            }

            print(f"[SERVICE] Local provider config for PySubtrans:", file=sys.stderr)
            print(f"[SERVICE]   pysubtrans_name: Custom Server", file=sys.stderr)
            print(f"[SERVICE]   server_address: {server_address}", file=sys.stderr)
            print(f"[SERVICE]   endpoint: /v1/chat/completions", file=sys.stderr)

            # Auto-cap batch size to fit within local LLM context window.
            # Default n_ctx=8192 can't handle 30 subtitle lines — the prompt
            # exceeds the context and PySubtrans gets truncated/garbled responses
            # ("Hit API token limit" + "No matches"). See #183.
            local_n_ctx = 8192  # matches start_local_server() default
            capped_batch_size = cap_batch_size_for_context(resolved_max_batch_size, local_n_ctx)
            if capped_batch_size < resolved_max_batch_size:
                print(f"[SERVICE]   NOTE: Batch size auto-reduced from {resolved_max_batch_size} to "
                      f"{capped_batch_size} to fit {local_n_ctx}-token context window", file=sys.stderr)
                resolved_max_batch_size = capped_batch_size

            try:
                result_path = translate_subtitle(
                    input_path=str(input_file),
                    output_path=resolved_output_path,
                    provider_config=local_provider_config,
                    model='local',
                    api_key='',  # Local server doesn't require API key
                    source_lang=source_lang,
                    target_lang=target_lang,
                    instruction_file=resolved_instruction_file,
                    scene_threshold=resolved_scene_threshold,
                    max_batch_size=resolved_max_batch_size,
                    stream=stream,
                    debug=debug,
                    provider_options=provider_options,
                    extra_context=extra_context,
                    emit_raw_output=True
                )
            finally:
                # Always stop server when done
                print(f"[SERVICE] Stopping local LLM server...", file=sys.stderr)
                stop_local_server()
                print(f"[SERVICE]   Server stopped, GPU memory released", file=sys.stderr)
        else:
            # =========================================================================
            # DIAGNOSTIC: Cloud Provider
            # =========================================================================
            print(f"\n[SERVICE] Cloud Provider Mode: {provider}", file=sys.stderr)
            print(f"[SERVICE]   Using provider config: {provider_config.get('pysubtrans_name', provider)}", file=sys.stderr)
            if 'api_base' in provider_config:
                print(f"[SERVICE]   API base: {provider_config['api_base']}", file=sys.stderr)

            result_path = translate_subtitle(
                input_path=str(input_file),
                output_path=resolved_output_path,
                provider_config=provider_config,
                model=resolved_model,
                api_key=resolved_api_key,
                source_lang=source_lang,
                target_lang=target_lang,
                instruction_file=resolved_instruction_file,
                scene_threshold=resolved_scene_threshold,
                max_batch_size=resolved_max_batch_size,
                stream=stream,
                debug=debug,
                provider_options=provider_options,
                extra_context=extra_context,
                emit_raw_output=True  # Always emit progress to stderr (CLI parity)
            )

        if result_path:
            logger.info(f"Translation complete: {result_path}")
            if progress_callback:
                progress_callback(f"Translation complete: {Path(result_path).name}")
            return Path(result_path) if isinstance(result_path, str) else result_path
        else:
            raise TranslationError("Translation returned no result")

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise TranslationError(f"Translation failed: {e}") from e
