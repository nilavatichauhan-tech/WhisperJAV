"""
Unit tests for the Qwen3 vLLM TextGenerator.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from whisperjav.modules.subtitle_pipeline.generators.qwen3_vllm import Qwen3vLLMTextGenerator
from whisperjav.modules.subtitle_pipeline.protocols import TextGenerator
from whisperjav.modules.subtitle_pipeline.generators.factory import TextGeneratorFactory
from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult

class TestQwen3vLLMProtocolCompliance:
    """The Qwen3vLLMTextGenerator class must satisfy the TextGenerator protocol."""

    def test_protocol_runtime_compliance(self):
        """isinstance check against the TextGenerator protocol."""
        gen = Qwen3vLLMTextGenerator()
        assert isinstance(gen, TextGenerator)

    def test_required_methods_exist(self):
        """All TextGenerator methods are present and callable."""
        gen = Qwen3vLLMTextGenerator()
        for method in ("load", "unload", "generate", "generate_batch", "cleanup"):
            assert hasattr(gen, method), f"missing method: {method}"
            assert callable(getattr(gen, method)), f"not callable: {method}"

    def test_is_loaded_starts_false(self):
        assert Qwen3vLLMTextGenerator().is_loaded is False


class TestQwen3vLLMFactoryRegistration:
    """The factory registers and instantiates the qwen3-vllm backend."""

    def test_qwen3_vllm_in_registry(self):
        assert "qwen3-vllm" in TextGeneratorFactory.available()

    def test_factory_create_returns_qwen3_vllm_class(self):
        gen = TextGeneratorFactory.create("qwen3-vllm")
        assert isinstance(gen, Qwen3vLLMTextGenerator)

    def test_factory_passes_kwargs_through(self):
        gen = TextGeneratorFactory.create(
            "qwen3-vllm",
            language="Japanese",
            api_url="http://192.168.1.100:8000/v1",
            max_new_tokens=1024,
        )
        assert gen.language == "ja"
        assert gen.api_url == "http://192.168.1.100:8000/v1"
        assert gen.max_new_tokens == 1024


class TestQwen3vLLMGenerateGuards:
    """generate() and generate_batch() check server connectivity on load."""

    @patch("requests.get")
    def test_load_checks_connectivity_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        gen = Qwen3vLLMTextGenerator(api_url="http://localhost:8000/v1")
        gen.load()
        assert gen.is_loaded is True
        mock_get.assert_called_once_with("http://localhost:8000/v1/models", timeout=5)

    @patch("requests.get")
    def test_load_checks_connectivity_failure_continues(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        gen = Qwen3vLLMTextGenerator(api_url="http://localhost:8000/v1")
        # Should not raise exception, just log warning and set loaded=True
        gen.load()
        assert gen.is_loaded is True


class TestQwen3vLLMGenerateMock:
    """Tests the generate and generate_batch functions with a mocked requests API."""

    @patch("requests.post")
    def test_generate_success(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "こんにちは、世界。"}
        mock_post.return_value = mock_resp

        gen = Qwen3vLLMTextGenerator(api_url="http://localhost:8000/v1")
        gen.load()

        dummy_file = tmp_path / "test.wav"
        dummy_file.write_bytes(b"mock_audio")

        res = gen.generate(dummy_file, language="ja", context="test prompt")

        assert isinstance(res, TranscriptionResult)
        assert res.text == "こんにちは、世界。"
        assert res.language == "ja"
        
        # Verify mock post args
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:8000/v1/audio/transcriptions"
        assert kwargs["data"]["model"] == "Qwen/Qwen3-ASR-1.7B"
        assert kwargs["data"]["language"] == "ja"
        assert kwargs["data"]["prompt"] == "test prompt"

    @patch("requests.post")
    def test_generate_batch(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "バッチテスト"}
        mock_post.return_value = mock_resp

        gen = Qwen3vLLMTextGenerator(api_url="http://localhost:8000/v1")
        gen.load()

        file1 = tmp_path / "1.wav"
        file1.write_bytes(b"audio1")
        file2 = tmp_path / "2.wav"
        file2.write_bytes(b"audio2")

        results = gen.generate_batch([file1, file2], language="ja", contexts=["ctx1", "ctx2"])
        
        assert len(results) == 2
        assert results[0].text == "バッチテスト"
        assert results[1].text == "バッチテスト"
        assert mock_post.call_count == 2
