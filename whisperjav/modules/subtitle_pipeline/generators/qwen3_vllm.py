"""
vLLM Client TextGenerator adapter for Qwen3-ASR.

Delegates batch audio transcription to a local or remote vLLM API server
running Qwen3-ASR via the OpenAI-compatible /v1/audio/transcriptions endpoint.
This offloads the 1.7B generation model from local GPU memory to the server,
saving ~3.4GB VRAM and enabling high-concurrency concurrent API requests.
"""

from pathlib import Path
from typing import Any, Optional
import requests
from concurrent.futures import ThreadPoolExecutor

from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult
from whisperjav.utils.logger import logger

class Qwen3vLLMTextGenerator:
    """
    TextGenerator backed by vLLM running Qwen3-ASR via REST API.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        api_url: Optional[str] = None,
        language: str = "Japanese",
        max_new_tokens: int = 4096,
        max_workers: int = 8,
        **kwargs: Any,
    ):
        """
        Initialize the vLLM text generator adapter.

        Args:
            model_id: Model name served by vLLM (e.g. 'Qwen/Qwen3-ASR-1.7B').
            api_url: Base URL of the vLLM OpenAI-compatible server (e.g. 'http://localhost:8000/v1').
            language: Language for transcription.
            max_new_tokens: Maximum generated tokens per segment.
            max_workers: Max concurrent requests to send to vLLM.
        """
        self.model_id = model_id
        # Resolve API URL, default to local if not provided
        self.api_url = (api_url or "http://localhost:8000/v1").rstrip("/")
        # vLLM/OpenAI expects standard ISO code, e.g. "ja". If "Japanese" is given, map to "ja"
        lang_lower = language.lower() if language else "ja"
        if "japanese" in lang_lower or lang_lower == "ja":
            self.language = "ja"
        elif "chinese" in lang_lower or lang_lower == "zh":
            self.language = "zh"
        elif "english" in lang_lower or lang_lower == "en":
            self.language = "en"
        else:
            self.language = language

        self.max_new_tokens = max_new_tokens
        self.max_workers = max_workers
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """
        Pre-flight check: verify connectivity to the vLLM server.
        """
        if self._loaded:
            return

        logger.info(f"[Qwen3vLLMTextGenerator] Verifying vLLM server connectivity at {self.api_url}...")
        try:
            # Quick check to see if the server is up (e.g. query /v1/models)
            resp = requests.get(f"{self.api_url}/models", timeout=5)
            if resp.status_code == 200:
                logger.info(f"[Qwen3vLLMTextGenerator] Successfully connected to vLLM server: {resp.json()}")
            else:
                logger.warning(f"[Qwen3vLLMTextGenerator] Server responded with code {resp.status_code}. Proceeding anyway...")
        except Exception as e:
            logger.warning(
                f"[Qwen3vLLMTextGenerator] Could not connect to vLLM server at {self.api_url} during load: {e}. "
                "Will attempt transcription requests anyway."
            )

        self._loaded = True

    def unload(self) -> None:
        """No-op as there is no local model loaded."""
        self._loaded = False

    def cleanup(self) -> None:
        self.unload()

    def generate(
        self,
        audio_path: Path,
        language: str = "ja",
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file via vLLM ASR API.
        """
        resolved_lang = language or self.language
        # If "Japanese" is passed, map to "ja"
        if resolved_lang.lower() == "japanese":
            resolved_lang = "ja"

        # Prepare request parameters for /v1/audio/transcriptions
        url = f"{self.api_url}/audio/transcriptions"
        
        # Files dict for multipart/form-data
        files = {
            "file": (audio_path.name, open(audio_path, "rb"), "audio/wav")
        }
        data = {
            "model": self.model_id,
            "language": resolved_lang,
        }
        if context:
            data["prompt"] = context

        logger.debug(f"[Qwen3vLLMTextGenerator] Sending transcription request for {audio_path.name} to {url}")
        try:
            resp = requests.post(url, files=files, data=data, timeout=300)
            # Make sure we close files
            for f in files.values():
                f[1].close()

            if resp.status_code != 200:
                raise RuntimeError(
                    f"vLLM server error (HTTP {resp.status_code}): {resp.text}"
                )

            res_json = resp.json()
            text = res_json.get("text", "").strip()
            
            return TranscriptionResult(
                text=text,
                language=resolved_lang,
                metadata={
                    "generator": "qwen3-vllm",
                    "audio_path": str(audio_path),
                }
            )
        except Exception as e:
            # Ensure file descriptors are closed in case of exception
            for f in files.values():
                try:
                    f[1].close()
                except Exception:
                    pass
            logger.error(f"[Qwen3vLLMTextGenerator] Transcription failed for {audio_path.name}: {e}")
            raise

    def generate_batch(
        self,
        audio_paths: list[Path],
        language: str = "ja",
        contexts: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[TranscriptionResult]:
        """
        Transcribe a list of audio files concurrently using a thread pool.
        """
        if not self._loaded:
            self.load()

        logger.info(f"[Qwen3vLLMTextGenerator] Batch transcribing {len(audio_paths)} files via vLLM with max_workers={self.max_workers}...")

        contexts_list = contexts if contexts else [None] * len(audio_paths)
        if len(contexts_list) < len(audio_paths):
            contexts_list = list(contexts_list) + [None] * (len(audio_paths) - len(contexts_list))

        def _worker(item):
            idx, path, ctx = item
            try:
                res = self.generate(path, language=language, context=ctx, **kwargs)
                return idx, res
            except Exception as e:
                # Return empty result with error logged to avoid breaking the entire batch
                logger.error(f"[Qwen3vLLMTextGenerator] Batch item {idx} ({path.name}) failed: {e}")
                return idx, TranscriptionResult(
                    text="",
                    language=language,
                    metadata={"generator": "qwen3-vllm", "audio_path": str(path), "error": str(e)}
                )

        items = list(zip(range(len(audio_paths)), audio_paths, contexts_list))
        
        results_map = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for idx, res in executor.map(_worker, items):
                results_map[idx] = res

        # Sort back to original order
        return [results_map[i] for i in range(len(audio_paths))]
