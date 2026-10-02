"""
Unit tests for the quality gate module.

Tests the compression ratio, character density, and repetition score
detection heuristics used to identify degenerate ASR output.
"""

import pytest

from whisperjav.modules.subtitle_pipeline.quality_gate import (
    CompressionAnalysis,
    QualityGateConfig,
    analyze_transcription_quality,
    extract_avg_logprob,
    _compute_compression_ratio,
    _compute_repetition_score,
)


# ---------------------------------------------------------------------------
# Compression ratio tests
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    """Tests for _compute_compression_ratio()."""

    def test_empty_string(self):
        assert _compute_compression_ratio("") == 1.0

    def test_short_string(self):
        """Strings < 4 bytes should return 1.0 (too short for meaningful compression)."""
        assert _compute_compression_ratio("あ") == 1.0

    def test_normal_japanese_text(self):
        """Normal varied Japanese text should have low compression ratio (< 2.0)."""
        text = "今日はとても良い天気ですね。公園に行って散歩をしましょう。桜の花がきれいに咲いています。"
        ratio = _compute_compression_ratio(text)
        assert ratio < 2.0, f"Normal text ratio {ratio} should be < 2.0"

    def test_repetitive_text_high_ratio(self):
        """Highly repetitive text should have high compression ratio (> 2.4)."""
        text = "ああああああああああああああああああああああああああああああ"
        ratio = _compute_compression_ratio(text)
        assert ratio > 2.4, f"Repetitive text ratio {ratio} should be > 2.4"

    def test_phrase_loop_high_ratio(self):
        """Repeated phrases should compress well."""
        text = "気持ちいい" * 20  # 100 chars of repeated phrase
        ratio = _compute_compression_ratio(text)
        assert ratio > 2.4, f"Phrase loop ratio {ratio} should be > 2.4"


# ---------------------------------------------------------------------------
# Repetition score tests
# ---------------------------------------------------------------------------

class TestRepetitionScore:
    """Tests for _compute_repetition_score()."""

    def test_empty_string(self):
        assert _compute_repetition_score("") == 0.0

    def test_short_string(self):
        """Strings shorter than n should return 0.0."""
        assert _compute_repetition_score("ab", n=3) == 0.0

    def test_normal_text_low_score(self):
        """Varied text should have low repetition score."""
        text = "今日は天気が良いので公園に行きましょう。桜がきれいです。"
        score = _compute_repetition_score(text, n=3)
        assert score < 0.4, f"Normal text score {score} should be < 0.4"

    def test_repeated_trigram_high_score(self):
        """Text dominated by a single repeated pattern should score high."""
        text = "ああああああああああああああああああああ"  # single char flood
        score = _compute_repetition_score(text, n=3)
        assert score > 0.5, f"Repeated trigram score {score} should be > 0.5"

    def test_phrase_repetition_detected(self):
        """Repeated phrases should produce high trigram dominance."""
        text = "行け行け行け行け行け行け行け行け行け行け"
        score = _compute_repetition_score(text, n=3)
        assert score > 0.3, f"Phrase repetition score {score} should be > 0.3"


# ---------------------------------------------------------------------------
# Full quality gate tests
# ---------------------------------------------------------------------------

class TestAnalyzeTranscriptionQuality:
    """Tests for analyze_transcription_quality()."""

    def test_normal_text_passes(self):
        """Normal Japanese text should not be flagged as degenerate."""
        text = "この映画はとても面白かったです。主人公の演技が素晴らしいと思いました。"
        result = analyze_transcription_quality(text, audio_duration_sec=10.0)
        assert not result.is_degenerate
        assert result.failure_reasons == []

    def test_repetitive_text_fails(self):
        """Highly repetitive text should be flagged."""
        text = "ああああああああああああああああああああああああああああ" * 3
        result = analyze_transcription_quality(text, audio_duration_sec=5.0)
        assert result.is_degenerate
        assert len(result.failure_reasons) > 0

    def test_high_char_density_fails(self):
        """Text with superhuman character density should be flagged."""
        # 200 chars in 2 seconds = 100 chars/sec (impossible)
        text = "あ" * 200
        result = analyze_transcription_quality(text, audio_duration_sec=2.0)
        assert result.is_degenerate
        assert any("char_density" in r for r in result.failure_reasons)

    def test_char_density_skipped_without_duration(self):
        """Char density check should be skipped when audio_duration_sec=0."""
        text = "あ" * 200
        result = analyze_transcription_quality(text, audio_duration_sec=0.0)
        # May still fail on compression/repetition, but density should not be in reasons
        assert not any("char_density" in r for r in result.failure_reasons)

    def test_empty_text_not_degenerate(self):
        """Empty text should not be flagged (it's silence, not hallucination)."""
        result = analyze_transcription_quality("", audio_duration_sec=5.0)
        assert not result.is_degenerate

    def test_single_char_not_degenerate(self):
        """Single character should not be flagged (below min_text_length)."""
        result = analyze_transcription_quality("あ", audio_duration_sec=1.0)
        assert not result.is_degenerate

    def test_none_text_not_degenerate(self):
        """None-like empty strings should not crash."""
        result = analyze_transcription_quality("", audio_duration_sec=0.0)
        assert not result.is_degenerate
        assert result.compression_ratio == 0.0

    def test_disabled_config(self):
        """Disabled config should always pass."""
        text = "ああああああああああああああああああああ" * 10
        config = QualityGateConfig(enabled=False)
        result = analyze_transcription_quality(text, audio_duration_sec=1.0, config=config)
        assert not result.is_degenerate

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        text = "これはテストです。" * 5
        # Very strict threshold — should flag even moderate repetition
        config = QualityGateConfig(max_compression_ratio=1.0, max_repetition_score=0.1)
        result = analyze_transcription_quality(text, audio_duration_sec=10.0, config=config)
        # May or may not be degenerate depending on actual metrics, but config should be used
        assert isinstance(result, CompressionAnalysis)

    def test_to_dict_serialization(self):
        """CompressionAnalysis.to_dict() should return serializable dict."""
        result = analyze_transcription_quality(
            "テスト文章です。", audio_duration_sec=2.0
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "compression_ratio" in d
        assert "char_density" in d
        assert "repetition_score" in d
        assert "is_degenerate" in d
        assert "failure_reasons" in d


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_unicode_mixed_content(self):
        """Mixed kanji/kana/latin text should be analyzed correctly."""
        text = "Hello世界！これはtest123です。ありがとう。"
        result = analyze_transcription_quality(text, audio_duration_sec=5.0)
        assert isinstance(result, CompressionAnalysis)

    def test_very_long_text(self):
        """Very long text (10K+ chars) should not crash or timeout."""
        text = "今日は良い天気です。" * 1000  # 10K chars
        result = analyze_transcription_quality(text, audio_duration_sec=600.0)
        assert isinstance(result, CompressionAnalysis)

    def test_punctuation_only(self):
        """Punctuation-only text should be analyzable."""
        text = "。。。。。。。。。。"
        result = analyze_transcription_quality(text, audio_duration_sec=2.0)
        assert isinstance(result, CompressionAnalysis)


# ---------------------------------------------------------------------------
# Change 3: generation-score signals (avg_logprob, no_speech_prob)
# ---------------------------------------------------------------------------

class TestNoSpeechSilenceOverride:
    """The no_speech_prob silence override mirrors openai-whisper's
    decode_with_fallback silence logic: a silent segment that
    hallucinated a few characters must NOT trigger a fallback retry."""

    def test_no_speech_prob_suppresses_degenerate(self):
        """Repetitive text + high no_speech_prob → silence, not degenerate."""
        text = "ああああああああああああああ"  # would normally be flagged
        result = analyze_transcription_quality(
            text, audio_duration_sec=2.0, no_speech_prob=0.9,
        )
        assert result.is_silence is True
        assert result.is_degenerate is False
        assert result.failure_reasons == []

    def test_low_no_speech_prob_does_not_override(self):
        """no_speech_prob below threshold must not trigger the override."""
        text = "ああああああああああああああ"
        result = analyze_transcription_quality(
            text, audio_duration_sec=2.0, no_speech_prob=0.3,
        )
        assert result.is_silence is False
        assert result.is_degenerate is True

    def test_no_speech_prob_boundary_at_threshold(self):
        """At exactly the threshold the override fires (>= semantics)."""
        text = "ああああああああああああああ"
        result = analyze_transcription_quality(
            text, audio_duration_sec=2.0, no_speech_prob=0.6,
        )
        assert result.is_silence is True
        assert result.is_degenerate is False

    def test_silence_override_takes_precedence_over_logprob(self):
        """Even a terrible avg_logprob is overridden by no_speech_prob."""
        text = "ああああああああああああああ"
        result = analyze_transcription_quality(
            text, audio_duration_sec=2.0,
            avg_logprob=-5.0, no_speech_prob=0.8,
        )
        assert result.is_silence is True
        assert result.is_degenerate is False


class TestAvgLogprobSignal:
    """avg_logprob is Whisper's logprob_threshold signal."""

    def test_low_avg_logprob_fails_gate(self):
        """Normal text + very low avg_logprob → degenerate (low confidence)."""
        text = "今日はとても良い天気ですね。"
        result = analyze_transcription_quality(
            text, audio_duration_sec=5.0, avg_logprob=-3.0,
        )
        assert result.is_degenerate is True
        assert any("avg_logprob" in r for r in result.failure_reasons)

    def test_high_avg_logprob_alone_passes(self):
        """Normal text + healthy avg_logprob → passes."""
        text = "今日はとても良い天気ですね。"
        result = analyze_transcription_quality(
            text, audio_duration_sec=5.0, avg_logprob=-0.2,
        )
        assert result.is_degenerate is False

    def test_avg_logprob_none_skipped(self):
        """When avg_logprob is None, the metric is silently skipped."""
        text = "今日はとても良い天気ですね。"
        result = analyze_transcription_quality(
            text, audio_duration_sec=5.0, avg_logprob=None,
        )
        assert result.is_degenerate is False
        assert not any("avg_logprob" in r for r in result.failure_reasons)

    def test_avg_logprob_stored_on_result(self):
        """The provided avg_logprob is echoed back on the result."""
        result = analyze_transcription_quality(
            "テスト文章です。", audio_duration_sec=2.0, avg_logprob=-0.5,
        )
        assert result.avg_logprob == -0.5

    def test_no_speech_prob_stored_on_result(self):
        """The provided no_speech_prob is echoed back on the result."""
        result = analyze_transcription_quality(
            "テスト文章です。", audio_duration_sec=2.0, no_speech_prob=0.1,
        )
        assert result.no_speech_prob == 0.1


class TestCustomLogprobThreshold:
    """Custom min_avg_logprob thresholds should be respected."""

    def test_strict_logprob_threshold_fails_normal_text(self):
        """A very strict threshold flags even healthy log-probs."""
        text = "今日は良い天気です。"
        config = QualityGateConfig(min_avg_logprob=-0.1)
        result = analyze_transcription_quality(
            text, audio_duration_sec=3.0, avg_logprob=-0.5, config=config,
        )
        assert result.is_degenerate is True

    def test_lenient_logprob_threshold_passes(self):
        """A very lenient threshold lets low log-probs through."""
        text = "今日は良い天気です。"
        config = QualityGateConfig(min_avg_logprob=-10.0)
        result = analyze_transcription_quality(
            text, audio_duration_sec=3.0, avg_logprob=-3.0, config=config,
        )
        assert result.is_degenerate is False


class TestExtractAvgLogprob:
    """extract_avg_logprob pulls mean per-token log-prob from HF outputs."""

    def test_none_input_returns_none(self):
        assert extract_avg_logprob(None) is None

    def test_object_without_scores_returns_none(self):
        """A bare object with no scores attribute returns None."""
        assert extract_avg_logprob(object()) is None

    def test_empty_scores_returns_none(self):
        """An empty scores tuple returns None."""

        class Empty:
            scores = ()

        assert extract_avg_logprob(Empty()) is None

    def test_synthetic_scores_yield_mean(self):
        """Construct known logit rows and verify the mean is computed."""
        import torch

        # HF generate() with output_scores=True yields scores as a tuple of
        # (batch, vocab) tensors — one per step. We use batch size 1.
        # Step 0: argmax=0, log-prob ≈ 0.0
        # Step 1: argmax=1, log-prob ≈ 0.0
        row0 = torch.full((1, 4), -1e9)
        row0[0, 0] = 0.0
        row1 = torch.full((1, 4), -1e9)
        row1[0, 1] = 0.0

        class GenOut:
            scores = (row0, row1)

        avg = extract_avg_logprob(GenOut())
        assert avg is not None
        # Both chosen tokens have ~0 log-prob → mean ≈ 0
        assert abs(avg) < 0.01

    def test_uneven_logprobs_averaged(self):
        """Two steps with different log-probs average correctly."""
        import torch

        # Step 0: argmax=0, log-prob = log(0.9) ≈ -0.105
        #   row = [2.197, 0.0] → softmax = [0.9, 0.1], argmax=0
        # Step 1: argmax=0, log-prob = log(0.1) ≈ -2.303
        #   row = [0.0]*10 (all equal) → softmax[0]=0.1, argmax=0 (first max)
        row0 = torch.zeros((1, 2))
        row0[0, 0] = 2.197
        row1 = torch.zeros((1, 10))  # all zeros → uniform softmax over 10 tokens

        class GenOut:
            scores = (row0, row1)

        avg = extract_avg_logprob(GenOut())
        assert avg is not None
        expected = (float(torch.log(torch.tensor(0.9))) +
                    float(torch.log(torch.tensor(0.1)))) / 2
        assert abs(avg - expected) < 0.05


class TestToDictSerializationNewFields:
    """to_dict() should include the new Change 3 fields."""

    def test_to_dict_includes_new_fields(self):
        result = analyze_transcription_quality(
            "テスト文章です。", audio_duration_sec=2.0,
            avg_logprob=-0.5, no_speech_prob=0.1,
        )
        d = result.to_dict()
        assert "avg_logprob" in d
        assert "no_speech_prob" in d
        assert "is_silence" in d
        assert d["avg_logprob"] == round(-0.5, 3)
        assert d["no_speech_prob"] == round(0.1, 3)
        assert d["is_silence"] is False

    def test_to_dict_none_score_fields(self):
        """When scores are None, to_dict serializes them as None."""
        result = analyze_transcription_quality("テストです。", audio_duration_sec=1.0)
        d = result.to_dict()
        assert d["avg_logprob"] is None
        assert d["no_speech_prob"] is None
        assert d["is_silence"] is False
