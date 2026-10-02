"""Tests for VadGroupedFramer energy-gated coverage rescue.

The rescue pass exists to recover active speech that the primary VAD backend
dropped (the Qwen3-ASR recall regression) *without* flooding silent gaps with
frames.  These tests pin both halves of that contract.

``VadGroupedFramer.frame()`` runs the segmenter itself, so the tests inject a
stub segmenter into the private ``_segmenter`` slot -- ``_ensure_segmenter()``
short-circuits when it is already populated, so no factory patching is needed.
"""

import numpy as np
import pytest

from whisperjav.modules.speech_segmentation.base import (
    SegmentationResult,
    SpeechSegment,
)
from whisperjav.modules.subtitle_pipeline.framers.vad_grouped import VadGroupedFramer

SR = 16000
DURATION_S = 4.0


class _StubSegmenter:
    """Returns a canned SegmentationResult, mimicking a VAD backend."""

    def __init__(self, result: SegmentationResult):
        self._result = result
        self.cleanup_calls = 0

    def segment(self, audio, sample_rate=16000, **kwargs):
        return self._result

    def cleanup(self):
        self.cleanup_calls += 1


def _seg(start_sec: float, end_sec: float) -> SpeechSegment:
    return SpeechSegment(
        start_sec=start_sec,
        end_sec=end_sec,
        start_sample=int(start_sec * SR),
        end_sample=int(end_sec * SR),
    )


def _seg_result(groups: list[list[SpeechSegment]]) -> SegmentationResult:
    flat = [seg for group in groups for seg in group]
    return SegmentationResult(
        segments=flat,
        groups=groups,
        method="stub-vad",
        audio_duration_sec=DURATION_S,
        parameters={},
    )


def _audio_with_active_region(start_sec: float, end_sec: float) -> np.ndarray:
    """Silence except for a loud block, so energy gating has a clear target."""
    audio = np.zeros(int(SR * DURATION_S), dtype=np.float32)
    audio[int(SR * start_sec):int(SR * end_sec)] = 0.5
    return audio


def _make_framer(groups, **overrides) -> VadGroupedFramer:
    framer = VadGroupedFramer(**overrides)
    framer._segmenter = _StubSegmenter(_seg_result(groups))
    return framer


def test_coverage_rescue_finds_uncovered_speech():
    """Active audio outside every VAD group becomes a rescue frame."""
    framer = _make_framer([[_seg(0.5, 1.0)]], coverage_rescue=True)
    audio = _audio_with_active_region(2.0, 3.0)

    result = framer.frame(audio, SR)

    assert len(result.frames) == 2
    assert result.metadata["coverage_rescue_count"] == 1
    assert result.frames[0].source == "vad-grouped"

    rescued = result.frames[1]
    assert rescued.source == "vad-coverage-rescue"
    # 2.0s-3.0s active region padded by rescue_padding_s (0.15s default).
    assert rescued.start == pytest.approx(1.85, abs=1e-6)
    assert rescued.end == pytest.approx(3.15, abs=1e-6)
    assert result.metadata["coverage_rescue_regions"] == [
        pytest.approx((1.85, 3.15), abs=1e-6)
    ]

    # Rescued frames carry their own speech region for the sentinel.
    assert result.metadata["speech_regions"][1] == [
        pytest.approx((1.85, 3.15), abs=1e-6)
    ]


def test_coverage_rescue_frames_are_time_ordered():
    """A rescue frame earlier than a VAD group is sorted into place."""
    framer = _make_framer([[_seg(3.5, 3.9)]], coverage_rescue=True)
    audio = _audio_with_active_region(0.5, 1.5)

    result = framer.frame(audio, SR)

    assert result.metadata["coverage_rescue_count"] == 1
    starts = [frame.start for frame in result.frames]
    assert starts == sorted(starts)
    assert result.frames[0].source == "vad-coverage-rescue"
    assert result.frames[1].source == "vad-grouped"
    # speech_regions must stay index-aligned with frames after the sort.
    assert result.metadata["speech_regions"][1] == [
        pytest.approx((3.5, 3.9), abs=1e-6)
    ]


def test_coverage_rescue_disabled_leaves_frames_untouched():
    framer = _make_framer([[_seg(0.5, 1.0)]], coverage_rescue=False)
    audio = _audio_with_active_region(2.0, 3.0)

    result = framer.frame(audio, SR)

    assert len(result.frames) == 1
    assert result.metadata["coverage_rescue_enabled"] is False
    assert result.metadata["coverage_rescue_count"] == 0
    assert result.metadata["coverage_rescue_regions"] == []


def test_coverage_rescue_skips_regions_vad_already_covers():
    """No duplicate frames when VAD already spans the active audio."""
    framer = _make_framer([[_seg(1.8, 3.2)]], coverage_rescue=True)
    audio = _audio_with_active_region(2.0, 3.0)

    result = framer.frame(audio, SR)

    assert len(result.frames) == 1
    assert result.metadata["coverage_rescue_count"] == 0


def test_coverage_rescue_does_not_flood_silence():
    """Pure silence must not produce rescue frames (the anti-flood guard)."""
    framer = _make_framer([[_seg(0.5, 1.0)]], coverage_rescue=True)
    audio = np.zeros(int(SR * DURATION_S), dtype=np.float32)

    result = framer.frame(audio, SR)

    assert len(result.frames) == 1
    assert result.metadata["coverage_rescue_count"] == 0


def test_coverage_rescue_ignores_low_level_noise():
    """Audio below rescue_min_dbfs is treated as noise, not speech."""
    audio = np.zeros(int(SR * DURATION_S), dtype=np.float32)
    # -60 dBFS steady tone: above the -200 dB floor, below the -52 dBFS gate.
    audio[int(SR * 2.0):int(SR * 3.0)] = 10 ** (-60.0 / 20.0)

    framer = _make_framer([[_seg(0.5, 1.0)]], coverage_rescue=True)
    result = framer.frame(audio, SR)

    assert result.metadata["coverage_rescue_count"] == 0


def test_coverage_rescue_partial_overlap_keeps_only_the_gap():
    """Overlapping coverage is subtracted; only the uncovered tail is rescued."""
    framer = _make_framer([[_seg(0.5, 2.5)]], coverage_rescue=True)
    audio = _audio_with_active_region(2.0, 3.0)

    result = framer.frame(audio, SR)

    assert result.metadata["coverage_rescue_count"] == 1
    rescued = result.frames[1]
    assert rescued.source == "vad-coverage-rescue"
    assert rescued.start == pytest.approx(2.5, abs=1e-6)
    assert rescued.end == pytest.approx(3.15, abs=1e-6)


def test_coverage_rescue_splits_long_regions_at_max_group():
    """Rescued spans longer than max_group_duration_s are chunked, not truncated."""
    long_duration_s = 10.0
    audio = np.zeros(int(SR * long_duration_s), dtype=np.float32)
    audio[int(SR * 1.0):int(SR * 9.0)] = 0.5

    framer = VadGroupedFramer(coverage_rescue=True, max_group_duration_s=3.0)
    framer._segmenter = _StubSegmenter(
        SegmentationResult(
            segments=[],
            groups=[],
            method="stub-vad",
            audio_duration_sec=long_duration_s,
            parameters={},
        )
    )

    result = framer.frame(audio, SR)

    regions = result.metadata["coverage_rescue_regions"]
    assert len(regions) == 3  # 8.3s padded span / 3.0s chunks
    assert all(end - start <= 3.0 + 1e-6 for start, end in regions)
    # Chunks must tile the span contiguously with no gaps or overlaps.
    for (_, prev_end), (next_start, _) in zip(regions, regions[1:]):
        assert next_start == pytest.approx(prev_end, abs=1e-6)
    assert regions[0][0] == pytest.approx(0.85, abs=1e-6)
    assert regions[-1][1] == pytest.approx(9.15, abs=1e-6)


def test_coverage_rescue_handles_empty_audio():
    framer = _make_framer([], coverage_rescue=True)

    result = framer.frame(np.zeros(0, dtype=np.float32), SR)

    assert result.frames == []
    assert result.metadata["coverage_rescue_count"] == 0
