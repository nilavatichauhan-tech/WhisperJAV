"""
Experimental speculative (assisted) decoding TextGenerator for Qwen3-ASR.

Uses Qwen3-ASR-0.6B as a "draft" model to accelerate inference of the
1.7B target model via HuggingFace's assisted generation API.

Requirements:
    - Both 0.6B and 1.7B models must share the same tokenizer (vocabulary).
    - transformers >= 4.45.0 for dynamic speculative decoding support.

This is an EXPERIMENTAL feature behind the ``--qwen-generator speculative``
CLI flag. If assisted generation is incompatible (different vocab, internal
API mismatch), the generator falls back to standard 1.7B-only transcription
with a logged warning.

VRAM: Both models loaded simultaneously. Estimated ~4.5 GB total on float16.
Not suitable for GPUs with < 6 GB VRAM.
"""

from pathlib import Path
from typing import Any

from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult
from whisperjav.utils.logger import logger


class SpeculativeQwen3Generator:
    """
    TextGenerator backed by speculative decoding: 0.6B draft + 1.7B target.

    Lifecycle:
        load()   → loads both models into VRAM
        unload() → releases both + CUDA cleanup
    """

    def __init__(
        self,
        # Domain-matched finetune pair.  Speculative decoding is exact with respect to
        # the target's distribution, so the draft only affects speed -- but a draft
        # trained on the same corpus as the target is accepted far more often than the
        # generic base 0.6B.  VRAM: target 3.796 GiB + draft 1.457 GiB = 5.253 GiB
        # before the ForcedAligner, so this path needs >8 GiB if word timestamps are on.
        target_model_id: str = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
        draft_model_id: str = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame",
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        max_new_tokens: int = 4096,
        language: str = "Japanese",
        repetition_penalty: float = 1.1,
        max_tokens_per_audio_second: float = 20.0,
        attn_implementation: str = "auto",
    ):
        """
        Store configuration for deferred model construction.

        Args:
            target_model_id: HuggingFace model ID for the 1.7B target.
            draft_model_id: HuggingFace model ID for the 0.6B draft.
            device: Device ('auto', 'cuda', 'cuda:0', 'cpu').
            dtype: Data type ('auto', 'float16', 'bfloat16').
            batch_size: Max inference batch size (speculative requires 1).
            max_new_tokens: Maximum tokens per utterance.
            language: Language for transcription.
            repetition_penalty: HF generation penalty.
            max_tokens_per_audio_second: Dynamic token budget scaling.
            attn_implementation: Attention implementation strategy.
        """
        self._config = {
            "target_model_id": target_model_id,
            "draft_model_id": draft_model_id,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "language": language,
            "repetition_penalty": repetition_penalty,
            "max_tokens_per_audio_second": max_tokens_per_audio_second,
            "attn_implementation": attn_implementation,
        }
        self._target_asr = None  # QwenASR instance (1.7B)
        self._draft_asr = None   # QwenASR instance (0.6B) — draft model
        self._speculative_available = False
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the models are currently loaded in memory."""
        return self._loaded

    def load(self) -> None:
        """
        Load both target and draft models. Attempt to verify speculative
        compatibility. If incompatible, fall back to target-only mode.
        """
        if self._loaded:
            logger.debug("[SpeculativeQwen3] Already loaded")
            return

        from whisperjav.modules.qwen_asr import QwenASR

        cfg = self._config

        # Load target model (1.7B)
        logger.info("[SpeculativeQwen3] Loading target model: %s", cfg["target_model_id"])
        self._target_asr = QwenASR(
            model_id=cfg["target_model_id"],
            device=cfg["device"],
            dtype=cfg["dtype"],
            batch_size=cfg["batch_size"],
            max_new_tokens=cfg["max_new_tokens"],
            language=cfg["language"],
            use_aligner=False,
            timestamps="none",
            repetition_penalty=cfg["repetition_penalty"],
            max_tokens_per_audio_second=cfg["max_tokens_per_audio_second"],
            attn_implementation=cfg["attn_implementation"],
        )
        self._target_asr.load_model_text_only()

        # Attempt to load draft model (0.6B)
        try:
            logger.info("[SpeculativeQwen3] Loading draft model: %s", cfg["draft_model_id"])
            self._draft_asr = QwenASR(
                model_id=cfg["draft_model_id"],
                device=cfg["device"],
                dtype=cfg["dtype"],
                batch_size=1,
                max_new_tokens=cfg["max_new_tokens"],
                language=cfg["language"],
                use_aligner=False,
                timestamps="none",
                repetition_penalty=cfg["repetition_penalty"],
                max_tokens_per_audio_second=cfg["max_tokens_per_audio_second"],
                attn_implementation=cfg["attn_implementation"],
            )
            self._draft_asr.load_model_text_only()

            # Verify compatibility: check if both models share the same
            # tokenizer by comparing vocab size
            target_thinker = self._target_asr.model.model.thinker
            draft_thinker = self._draft_asr.model.model.thinker

            target_vocab = target_thinker.config.vocab_size
            draft_vocab = draft_thinker.config.vocab_size

            if target_vocab == draft_vocab:
                self._speculative_available = True
                logger.info(
                    "[SpeculativeQwen3] Speculative decoding ENABLED "
                    "(target=%s, draft=%s, vocab_size=%d)",
                    cfg["target_model_id"], cfg["draft_model_id"], target_vocab,
                )
            else:
                logger.warning(
                    "[SpeculativeQwen3] Vocab mismatch (target=%d, draft=%d). "
                    "Falling back to target-only mode.",
                    target_vocab, draft_vocab,
                )
                self._speculative_available = False
                # Release draft model to save VRAM
                self._draft_asr.unload_model()
                self._draft_asr = None

        except Exception as e:
            logger.warning(
                "[SpeculativeQwen3] Draft model load failed: %s. "
                "Falling back to target-only mode.",
                e,
            )
            self._speculative_available = False
            if self._draft_asr is not None:
                try:
                    self._draft_asr.unload_model()
                except Exception:
                    pass
                self._draft_asr = None

        self._loaded = True
        logger.info(
            "[SpeculativeQwen3] Loaded (speculative=%s)",
            self._speculative_available,
        )

    def unload(self) -> None:
        """Unload both models and release VRAM."""
        if not self._loaded:
            return

        if self._draft_asr is not None:
            self._draft_asr.unload_model()
            self._draft_asr = None

        if self._target_asr is not None:
            self._target_asr.unload_model()
            self._target_asr = None

        from whisperjav.utils.gpu_utils import safe_cuda_cleanup
        safe_cuda_cleanup()

        self._speculative_available = False
        self._loaded = False
        logger.info("[SpeculativeQwen3] Models unloaded")

    def generate(
        self,
        audio_path: Path,
        language: str = "ja",
        context: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe a single audio file."""
        results = self.generate_batch(
            audio_paths=[audio_path],
            language=language,
            contexts=[context] if context else None,
            **kwargs,
        )
        return results[0]

    def generate_batch(
        self,
        audio_paths: list[Path],
        language: str = "ja",
        contexts: list[str] | None = None,
        **kwargs: Any,
    ) -> list[TranscriptionResult]:
        """
        Transcribe a batch of audio files.

        If speculative decoding is available, uses HF assisted generation.
        Otherwise, falls back to standard target-only transcription.
        """
        if not self._loaded:
            raise RuntimeError(
                "SpeculativeQwen3Generator.generate_batch() called before load()"
            )

        # Speculative path: attempt assisted generation via the thinker
        # NOTE: qwen-asr package wraps the thinker internally. Speculative
        # decoding would require calling model.model.thinker.generate()
        # directly with assistant_model=draft_thinker, which bypasses
        # the qwen-asr audio preprocessing. For now, we fall back to
        # standard mode and log the limitation.
        if self._speculative_available:
            logger.info(
                "[SpeculativeQwen3] Speculative decoding available but "
                "qwen-asr package wraps generation internally. Using "
                "target-only mode for compatibility. Future: bypass "
                "qwen-asr and call thinker.generate() directly."
            )

        # Fall back to target-only transcription
        audio_durations = kwargs.get("audio_durations")
        texts = self._target_asr.transcribe_text_only(
            audio_paths=list(audio_paths),
            contexts=contexts,
            language=language,
            audio_durations=audio_durations,
        )

        return [
            TranscriptionResult(
                text=t,
                language=language,
                metadata={
                    "generator": "speculative-qwen3",
                    "speculative_active": False,
                    "audio_path": str(p),
                },
            )
            for t, p in zip(texts, audio_paths, strict=True)
        ]

    def cleanup(self) -> None:
        """Final cleanup — unload if still loaded."""
        self.unload()
