"""
Quality Gate for Qwen3-ASR transcription output.

Post-generation quality assessment that detects degenerate ASR output
(hallucinations, repetition loops, silence-generated text) at the scene
level before it enters the forced-alignment phase.

Metrics:
    1. Compression ratio  — zlib-compressed size / raw size. Repetitive
       text compresses extremely well (ratio > 2.4 is suspicious).
    2. Character density   — characters per second of audio. Japanese
       speech tops out at ~12 chars/sec; > 15 is superhuman.
    3. Repetition score    — fraction of text composed of the most-repeated
       n-gram (n=3). Score > 0.5 indicates looping behavior.
    4. Average log-prob    — mean per-token log-probability of the generated
       sequence (Whisper's `logprob_threshold` signal). Below -1.0 means
       the model was uncertain; combined with repetition this is a strong
       hallucination indicator. Only evaluated when explicitly provided
       (requires calling generate() with output_scores=True).
    5. No-speech probability — Whisper's `no_speech_threshold` signal. When
       the model reports a high probability that the segment contains no
       speech, the gate treats the output as silence (NOT degenerate) and
       suppresses fallback retries — matching openai-whisper's silence
       override in decode_with_fallback(). Only evaluated when provided.

The first three metrics run on the text alone and need no generation
internals. The last two are optional and only fill in when the caller has
access to generation scores (e.g. the speculative / direct-generate path).
All thresholds are tuned for Japanese speech patterns and can be overridden
via QualityGateConfig.
"""

import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from whisperjav.utils.logger import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QualityGateConfig:
    """Tunable thresholds for transcription quality assessment.

    Defaults are calibrated for Japanese JAV audio with the following
    rationale:

    - max_compression_ratio=2.4: normal Japanese text has ratio ~1.2-1.8.
      Repetitive hallucinations ("ああああああ") compress to 3.0+.
    - max_char_density=15.0: Japanese speech rate is typically 7-12 chars/sec.
      Values above 15 indicate text was generated faster than human speech.
    - max_repetition_score=0.5: if 50%+ of the text is a repeated trigram,
      the output is almost certainly a loop.
    - min_avg_logprob=-1.0: Whisper's logprob_threshold default. A mean
      per-token log-prob below this means the model was uncertain about
      the transcription. Only checked when avg_logprob is provided.
    - no_speech_prob_threshold=0.6: Whisper's no_speech_threshold default.
      When the no-speech probability is at or above this, the gate declares
      the segment silence (NOT degenerate) and suppresses fallback —
      matching openai-whisper's silence override. Only checked when
      no_speech_prob is provided.
    - min_text_length=2: texts shorter than 2 chars are too small to assess
      meaningfully; skip quality gating for these.
    """
    max_compression_ratio: float = 2.4
    max_char_density: float = 15.0
    max_repetition_score: float = 0.5
    min_avg_logprob: float = -1.0
    no_speech_prob_threshold: float = 0.6
    min_text_length: int = 2
    enabled: bool = True

    @classmethod
    def get_preset(cls, preset_name: str) -> "QualityGateConfig":
        """
        Get QualityGateConfig preset.
        
        Presets:
            - 'default': standard thresholds.
            - 'high_moan': tighter limits for JAV/ASMR where moans can trigger loops.
            - 'narrative': looser limits for dense narration/monologue.
        """
        preset_name = preset_name.lower().strip()
        if preset_name == "high_moan":
            return cls(
                max_compression_ratio=2.0,
                max_char_density=12.0,
                max_repetition_score=0.4,
                min_avg_logprob=-0.9,
                no_speech_prob_threshold=0.55
            )
        elif preset_name == "narrative":
            return cls(
                max_compression_ratio=2.8,
                max_char_density=18.0,
                max_repetition_score=0.6,
                min_avg_logprob=-1.2,
                no_speech_prob_threshold=0.7
            )
        else: # default
            return cls()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompressionAnalysis:
    """Quality assessment result for a single transcription output.

    Fields:
        compression_ratio: Ratio of raw bytes to zlib-compressed bytes.
            Normal text: 1.2-1.8. Repetitive hallucinations: 2.5+.
        char_density: Characters per second of audio duration.
            Normal Japanese speech: 7-12. Hallucinations: 15+.
        repetition_score: Fraction of text covered by the most-repeated
            trigram. Normal: <0.3. Loops: >0.5.
        avg_logprob: Mean per-token log-probability of the generated
            sequence, or None when not available. Normal: >-1.0.
            Hallucinations/uncertain: <-1.0.
        no_speech_prob: Model's reported probability that the segment
            contains no speech, or None when not available. >=0.6 means
            the segment is treated as silence (suppresses fallback).
        is_silence: True when no_speech_prob indicates a non-speech
            segment. When True, is_degenerate is forced False — the gate
            is declaring the segment silent, not hallucinated.
        is_degenerate: True if ANY metric exceeds its threshold (and the
            segment is not classified as silence).
        failure_reasons: Human-readable reasons why the gate failed.
    """
    compression_ratio: float = 0.0
    char_density: float = 0.0
    repetition_score: float = 0.0
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    is_silence: bool = False
    is_degenerate: bool = False
    failure_reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for diagnostics JSON."""
        return {
            "compression_ratio": round(self.compression_ratio, 3),
            "char_density": round(self.char_density, 2),
            "repetition_score": round(self.repetition_score, 3),
            "avg_logprob": round(self.avg_logprob, 3) if self.avg_logprob is not None else None,
            "no_speech_prob": round(self.no_speech_prob, 3) if self.no_speech_prob is not None else None,
            "is_silence": self.is_silence,
            "is_degenerate": self.is_degenerate,
            "failure_reasons": self.failure_reasons,
        }


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_transcription_quality(
    text: str,
    audio_duration_sec: float = 0.0,
    config: QualityGateConfig | None = None,
    avg_logprob: Optional[float] = None,
    no_speech_prob: Optional[float] = None,
) -> CompressionAnalysis:
    """
    Analyze raw transcription text for hallucination indicators.

    This function runs lightweight heuristic checks on the text to
    determine if the ASR output is likely degenerate. It does NOT modify
    the text — it only reports metrics and a pass/fail verdict.

    The text-only metrics (compression, density, repetition) always run.
    The two generation-score metrics (avg_logprob, no_speech_prob) only
    contribute when the caller supplies them — typically from a direct
    ``model.generate(..., output_scores=True)`` call (e.g. the
    speculative-decoding path). When unavailable, they are silently
    skipped and the gate falls back to the text-only heuristics.

    The no-speech override mirrors openai-whisper's ``decode_with_fallback``
    silence logic: if ``no_speech_prob >= threshold`` the segment is
    classified as silence (``is_silence=True``) and ``is_degenerate`` is
    forced False regardless of the text metrics — there is no point
    retrying a genuinely silent segment at a higher temperature.

    Args:
        text: Raw transcription text from ASR.
        audio_duration_sec: Duration of the source audio in seconds.
            Used for character density calculation. If 0 or unknown,
            the density check is skipped.
        config: Quality gate thresholds. Uses defaults if None.
        avg_logprob: Optional mean per-token log-probability of the
            generated sequence (Whisper's logprob_threshold signal).
            When provided and below ``min_avg_logprob``, the gate fails
            on low confidence — a strong hallucination signal when
            combined with high repetition.
        no_speech_prob: Optional model-reported probability that the
            segment contains no speech (Whisper's no_speech_threshold
            signal). When at or above ``no_speech_prob_threshold``, the
            gate declares the segment silent and suppresses the
            degenerate flag.

    Returns:
        CompressionAnalysis with metrics and degenerate flag.
    """
    if config is None:
        config = QualityGateConfig()

    result = CompressionAnalysis()
    result.avg_logprob = avg_logprob
    result.no_speech_prob = no_speech_prob

    # Skip analysis for very short text or when disabled
    if not config.enabled or not text or len(text) < config.min_text_length:
        return result

    # --- Silence override (Whisper no_speech_threshold semantics) ---
    # Checked first because it suppresses every other metric: a silent
    # segment that hallucinated a few characters should not trigger a
    # fallback retry — the model was right that there was no speech.
    if no_speech_prob is not None and no_speech_prob >= config.no_speech_prob_threshold:
        result.is_silence = True
        result.is_degenerate = False
        logger.debug(
            "Quality gate: silence override (no_speech_prob=%.2f >= %.2f) — "
            "suppressing fallback",
            no_speech_prob, config.no_speech_prob_threshold,
        )
        return result

    # --- Metric 1: Compression ratio ---
    result.compression_ratio = _compute_compression_ratio(text)
    if result.compression_ratio > config.max_compression_ratio:
        result.is_degenerate = True
        result.failure_reasons.append(
            f"compression_ratio={result.compression_ratio:.2f} > {config.max_compression_ratio}"
        )

    # --- Metric 2: Character density ---
    if audio_duration_sec > 0:
        result.char_density = len(text) / audio_duration_sec
        if result.char_density > config.max_char_density:
            result.is_degenerate = True
            result.failure_reasons.append(
                f"char_density={result.char_density:.1f} > {config.max_char_density} chars/sec"
            )

    # --- Metric 3: Repetition score (trigram dominance) ---
    result.repetition_score = _compute_repetition_score(text, n=3)
    if result.repetition_score > config.max_repetition_score:
        result.is_degenerate = True
        result.failure_reasons.append(
            f"repetition_score_n3={result.repetition_score:.2f} > {config.max_repetition_score}"
        )

    # Japanese-specific bigram/unigram repetition check (v1.9.0)
    # Detects short-unit loops common in Japanese ASR hallucinations
    is_japanese = any(ord(c) in range(0x3040, 0x30FF) or ord(c) in range(0x4E00, 0x9FFF) for c in text)
    if is_japanese:
        # Require enough observations for dominance to be meaningful. A
        # valid short response such as "はい" has one bigram and must not be
        # treated as a repetition loop merely because that sample is 100%.
        if len(text) >= 6:
            bigram_score = _compute_repetition_score(text, n=2)
            if bigram_score > max(config.max_repetition_score, 0.6):
                result.is_degenerate = True
                result.failure_reasons.append(
                    f"repetition_score_n2={bigram_score:.2f} > {max(config.max_repetition_score, 0.6)}"
                )
        if len(text) >= 8:
            unigram_score = _compute_repetition_score(text, n=1)
            if unigram_score > max(config.max_repetition_score, 0.7):
                result.is_degenerate = True
                result.failure_reasons.append(
                    f"repetition_score_n1={unigram_score:.2f} > {max(config.max_repetition_score, 0.7)}"
                )

    # --- Metric 4: Average log-probability (Whisper logprob_threshold) ---
    # Only when the caller has generation scores. Low avg_logprob means
    # the model was uncertain; on its own it is a weak signal (proper
    # nouns, noisy audio), but it strengthens the repetition/compression
    # signals. We add it as a standalone failure only to surface the
    # metric — the fallback ladder will still retry.
    if avg_logprob is not None and avg_logprob < config.min_avg_logprob:
        result.is_degenerate = True
        result.failure_reasons.append(
            f"avg_logprob={avg_logprob:.2f} < {config.min_avg_logprob}"
        )

    if result.is_degenerate:
        logger.warning(
            "Quality gate FAILED: %s | text[:%d]='%s...'",
            ", ".join(result.failure_reasons),
            min(40, len(text)),
            text[:40],
        )
    else:
        logger.debug(
            "Quality gate passed: compression=%.2f, density=%.1f, repetition=%.2f"
            + (", avg_logprob=%.2f" if avg_logprob is not None else ""),
            result.compression_ratio,
            result.char_density,
            result.repetition_score,
            *([avg_logprob] if avg_logprob is not None else []),
        )

    return result


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _compute_compression_ratio(text: str) -> float:
    """
    Compute zlib compression ratio for text.

    Returns raw_size / compressed_size. Higher values indicate more
    redundancy (repetitive text compresses very well).

    Normal Japanese text: ~1.2-1.8
    Repetitive hallucinations: 2.5-10+
    Single character: ~1.0

    Args:
        text: Input text string.

    Returns:
        Compression ratio (float). Returns 1.0 for empty/tiny text.
    """
    if not text:
        return 1.0

    raw_bytes = text.encode("utf-8")
    raw_size = len(raw_bytes)

    if raw_size < 4:
        # zlib needs a minimum input to be meaningful
        return 1.0

    compressed = zlib.compress(raw_bytes, level=6)
    compressed_size = len(compressed)

    if compressed_size == 0:
        return 1.0

    return raw_size / compressed_size


def _compute_repetition_score(text: str, n: int = 3) -> float:
    """
    Compute the dominance score of the most-repeated character n-gram.

    Extracts all character-level n-grams, finds the most common one,
    and returns the fraction of the text that could be "explained" by
    repeating that n-gram.

    Score interpretation:
        0.0-0.2: Normal, varied text
        0.2-0.4: Some repetition (common in natural speech with filler words)
        0.4-0.6: Suspicious repetition
        0.6-1.0: Almost certainly a hallucination loop

    Args:
        text: Input text string.
        n: N-gram size (default: 3 for trigrams).

    Returns:
        Repetition score between 0.0 and 1.0.
    """
    if not text or len(text) < n:
        return 0.0

    # Extract all character n-grams
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    total = len(ngrams)

    if total == 0:
        return 0.0

    # Find most common n-gram and its count
    counter = Counter(ngrams)
    _most_common_ngram, most_common_count = counter.most_common(1)[0]

    # Score: fraction of text positions dominated by the top n-gram
    # Multiply by n to approximate character coverage
    char_coverage = min(most_common_count * n, len(text))
    return char_coverage / len(text)


# ---------------------------------------------------------------------------
# Generation-score extraction (for callers with direct generate() access)
# ---------------------------------------------------------------------------


def extract_avg_logprob(generation_output) -> Optional[float]:
    """
    Extract the mean per-token log-probability from a HuggingFace
    ``generate(..., return_dict_in_generate=True, output_scores=True)``
    output.

    This is the Qwen3-ASR analogue of Whisper's ``avg_logprob``. The
    generation output's ``scores`` is a tuple of per-step logit tensors
    (one per generated step, in generation order, excluding the prompt).
    For each step the model's chosen token is the argmax, so we read the
    log-prob of that token and average across all steps.

    Returns None when scores are unavailable (e.g. the caller did not
    request output_scores, or the object shape is unexpected) so the
    quality gate can silently fall back to text-only heuristics.

    Args:
        generation_output: A HuggingFace ``GenerateDecoderOnlyOutput`` /
            ``GenerateSequenceEncoderDecoderOutput``-like object with a
            ``scores`` attribute, OR None.

    Returns:
        Mean per-token log-probability (float, typically in [-20, 0]),
        or None when it cannot be computed.
    """
    try:
        scores = getattr(generation_output, "scores", None)
        if not scores:
            return None

        import torch

        total = 0.0
        count = 0
        for step_logits in scores:
            # step_logits shape: (batch, vocab_size). We only care about
            # batch 0 (QwenASR transcribes one chunk at a time in the
            # direct-generate path).
            if step_logits is None or step_logits.numel() == 0:
                continue
            row = step_logits[0]
            # log-softmax gives per-token log-probs; the chosen token is
            # the argmax of the logits (= argmax of the log-probs).
            log_probs = torch.log_softmax(row.float(), dim=-1)
            chosen = int(torch.argmax(row, dim=-1).item())
            total += float(log_probs[chosen].item())
            count += 1

        if count == 0:
            return None
        return total / count
    except Exception as e:  # noqa: BLE001 — best-effort extraction
        logger.debug("extract_avg_logprob failed: %s", e)
        return None
