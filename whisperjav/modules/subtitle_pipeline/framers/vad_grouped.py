"""
VAD-grouped TemporalFramer — uses speech segmentation to produce frames.

Wraps the existing SpeechSegmenterFactory to detect speech regions, then
converts the segmenter's pre-grouped results into TemporalFrames.  This is
what current VAD_SLICING mode does: run VAD, group speech segments into
chunks within a duration limit, and process each group.

The raw speech regions are stored in FramingResult.metadata["speech_regions"]
so the orchestrator can pass them to the alignment sentinel for Strategy C
(VAD-guided) recovery.
"""

from typing import Any, Optional

import numpy as np

from whisperjav.modules.subtitle_pipeline.types import (
    FramingResult,
    TemporalFrame,
)
from whisperjav.utils.logger import logger


class VadGroupedFramer:
    """
    Produces temporal frames by running VAD and grouping speech regions.

    Each group of speech segments becomes a TemporalFrame spanning from
    the first segment's start to the last segment's end.

    Groups are produced by the SpeechSegmenter backend, which respects
    max_group_duration_s and chunk_threshold_s.
    """

    def __init__(
        self,
        segmenter_backend: str = "whisperseg",  # v1.8.13: was "ten"
        max_group_duration_s: float = 29.0,
        chunk_threshold_s: float = 1.0,
        min_frame_duration_s: float = 0.1,
        segmenter_config: Optional[dict[str, Any]] = None,
        coverage_rescue: bool = True,
        rescue_window_s: float = 0.25,
        rescue_min_duration_s: float = 0.20,
        rescue_merge_gap_s: float = 0.25,
        rescue_padding_s: float = 0.15,
        rescue_min_dbfs: float = -52.0,
        rescue_noise_margin_db: float = 8.0,
    ):
        """
        Configure the VAD-grouped framer.

        Args:
            segmenter_backend: SpeechSegmenter backend name (e.g., "ten", "silero").
            max_group_duration_s: Maximum duration per group (Whisper 30s limit).
            chunk_threshold_s: Silence gap threshold for splitting groups.
            min_frame_duration_s: Minimum frame duration; shorter groups are filtered.
            segmenter_config: Additional kwargs passed to the segmenter factory.
            coverage_rescue: Add energy-gated frames where VAD missed active audio.
            rescue_window_s: Analysis window size for coverage rescue.
            rescue_min_duration_s: Minimum active region duration to rescue.
            rescue_merge_gap_s: Merge active windows separated by shorter gaps.
            rescue_padding_s: Padding added around rescued active regions.
            rescue_min_dbfs: Absolute minimum RMS level considered active.
            rescue_noise_margin_db: Required level above the scene noise floor.
        """
        self._backend = segmenter_backend
        self._max_group = max_group_duration_s
        self._chunk_threshold = chunk_threshold_s
        self._min_duration = min_frame_duration_s
        self._segmenter_config = segmenter_config or {}
        self._coverage_rescue = bool(coverage_rescue)
        self._rescue_window_s = max(0.05, float(rescue_window_s))
        self._rescue_min_duration_s = max(0.05, float(rescue_min_duration_s))
        self._rescue_merge_gap_s = max(0.0, float(rescue_merge_gap_s))
        self._rescue_padding_s = max(0.0, float(rescue_padding_s))
        self._rescue_min_dbfs = float(rescue_min_dbfs)
        self._rescue_noise_margin_db = max(0.0, float(rescue_noise_margin_db))
        self._segmenter = None

    def _ensure_segmenter(self):
        """Lazily create the speech segmenter."""
        if self._segmenter is not None:
            return

        from whisperjav.modules.speech_segmentation.factory import (
            SpeechSegmenterFactory,
        )

        kwargs = {
            **self._segmenter_config,
            "max_group_duration_s": self._max_group,
            "chunk_threshold_s": self._chunk_threshold,
        }
        self._segmenter = SpeechSegmenterFactory.create(self._backend, **kwargs)

    def frame(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs: Any,
    ) -> FramingResult:
        """
        Run VAD on scene audio and convert groups to temporal frames.

        Each group from the segmenter becomes one TemporalFrame.
        Raw speech regions (per frame) are stored in metadata for
        the alignment sentinel's Strategy C (VAD-guided) recovery.

        Args:
            audio: Scene audio array (mono, typically 16kHz).
            sample_rate: Sample rate of the audio.

        Returns:
            FramingResult with one frame per VAD group.
        """
        self._ensure_segmenter()
        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

        # Run speech segmentation
        seg_result = self._segmenter.segment(audio, sample_rate=sample_rate)

        frames = []
        speech_regions_per_frame: list[list[tuple[float, float]]] = []
        speech_starts_per_frame: list[Optional[float]] = []
        skipped = 0

        for group in seg_result.groups:
            if not group:
                continue

            # Group boundaries: first segment start → last segment end
            group_start = group[0].start_sec
            group_end = group[-1].end_sec

            # Filter short groups
            if (group_end - group_start) < self._min_duration:
                skipped += 1
                continue

            frames.append(
                TemporalFrame(
                    start=group_start,
                    end=group_end,
                    source="vad-grouped",
                )
            )

            # Store raw speech regions within this frame (group-relative)
            # These are used by the sentinel for VAD-guided redistribution
            regions = [(seg.start_sec, seg.end_sec) for seg in group]
            speech_regions_per_frame.append(regions)

            # "3a" refined display start: first segment in the group that
            # recorded a speech_start_sec (first frame >= speech_start_threshold
            # in the backend). None -> consumers fall back to the frame start.
            speech_starts_per_frame.append(
                next(
                    (
                        s.metadata.get("speech_start_sec")
                        for s in group
                        if s.metadata.get("speech_start_sec") is not None
                    ),
                    None,
                )
            )

        rescue_regions: list[tuple[float, float]] = []
        if self._coverage_rescue and duration > 0:
            covered_regions = [(frame.start, frame.end) for frame in frames]
            rescue_regions = self._find_uncovered_active_regions(
                audio,
                sample_rate,
                covered_regions,
            )
            for start, end in rescue_regions:
                frames.append(
                    TemporalFrame(
                        start=start,
                        end=end,
                        source="vad-coverage-rescue",
                    )
                )
                speech_regions_per_frame.append([(start, end)])
                # A rescued frame comes from energy gating, not a VAD onset, so
                # it has no "3a" refined start. None keeps this list aligned with
                # ``frames`` (the orchestrator indexes speech_starts per frame)
                # and makes consumers fall back to the frame start.
                speech_starts_per_frame.append(None)

            if rescue_regions:
                ordered = sorted(
                    zip(frames, speech_regions_per_frame, speech_starts_per_frame),
                    key=lambda item: (item[0].start, item[0].end),
                )
                frames = [item[0] for item in ordered]
                speech_regions_per_frame = [item[1] for item in ordered]
                speech_starts_per_frame = [item[2] for item in ordered]
                logger.warning(
                    "[VadGroupedFramer] Coverage rescue added %d frame(s) for active audio missed by VAD",
                    len(rescue_regions),
                )

        if skipped > 0:
            logger.debug(
                "[VadGroupedFramer] Filtered %d groups below %.2fs minimum",
                skipped,
                self._min_duration,
            )

        logger.info(
            "[VadGroupedFramer] %s: %d segments → %d groups → %d frames (%.1fs audio, %.1f%% speech)",
            seg_result.method,
            seg_result.num_segments,
            seg_result.num_groups,
            len(frames),
            duration,
            seg_result.speech_coverage_ratio * 100,
        )

        return FramingResult(
            frames=frames,
            metadata={
                "strategy": "vad-grouped",
                "segmenter_backend": seg_result.method,
                "frame_count": len(frames),
                "total_segments": seg_result.num_segments,
                "total_groups": seg_result.num_groups,
                "groups_skipped": skipped,
                "audio_duration_sec": duration,
                "speech_coverage_ratio": seg_result.speech_coverage_ratio,
                "speech_regions": speech_regions_per_frame,
                "speech_starts": speech_starts_per_frame,
                "segmenter_params": seg_result.parameters,
                "coverage_rescue_enabled": self._coverage_rescue,
                "coverage_rescue_count": len(rescue_regions),
                "coverage_rescue_regions": rescue_regions,
            },
        )

    def _find_uncovered_active_regions(
        self,
        audio: np.ndarray,
        sample_rate: int,
        covered_regions: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Find active audio omitted by the primary speech segmenter."""
        if sample_rate <= 0 or audio.size == 0:
            return []

        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = np.mean(mono, axis=-1)
        mono = mono.reshape(-1)
        window_samples = max(1, int(round(self._rescue_window_s * sample_rate)))
        levels: list[float] = []
        windows: list[tuple[float, float]] = []

        for start_sample in range(0, len(mono), window_samples):
            end_sample = min(len(mono), start_sample + window_samples)
            chunk = mono[start_sample:end_sample]
            if chunk.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
            levels.append(20.0 * np.log10(max(rms, 1e-10)))
            windows.append((start_sample / sample_rate, end_sample / sample_rate))

        if not levels:
            return []

        noise_floor_db = float(np.percentile(levels, 20.0))
        active_threshold_db = max(
            self._rescue_min_dbfs,
            noise_floor_db + self._rescue_noise_margin_db,
        )
        active_windows = [
            window for window, level_db in zip(windows, levels)
            if level_db >= active_threshold_db
        ]
        if not active_windows:
            return []

        merged: list[list[float]] = []
        for start, end in active_windows:
            if merged and start - merged[-1][1] <= self._rescue_merge_gap_s:
                merged[-1][1] = end
            else:
                merged.append([start, end])

        duration = len(mono) / sample_rate
        rescued: list[tuple[float, float]] = []
        for start, end in merged:
            if end - start < self._rescue_min_duration_s:
                continue
            padded = (
                max(0.0, start - self._rescue_padding_s),
                min(duration, end + self._rescue_padding_s),
            )
            for piece in self._subtract_regions(padded, covered_regions):
                if piece[1] - piece[0] < self._rescue_min_duration_s:
                    continue
                piece_start = piece[0]
                while piece_start < piece[1]:
                    piece_end = min(piece[1], piece_start + self._max_group)
                    rescued.append((piece_start, piece_end))
                    piece_start = piece_end
        return rescued

    @staticmethod
    def _subtract_regions(
        region: tuple[float, float],
        covered_regions: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        pieces = [region]
        for covered_start, covered_end in sorted(covered_regions):
            next_pieces: list[tuple[float, float]] = []
            for start, end in pieces:
                if covered_end <= start or covered_start >= end:
                    next_pieces.append((start, end))
                else:
                    if covered_start > start:
                        next_pieces.append((start, min(end, covered_start)))
                    if covered_end < end:
                        next_pieces.append((max(start, covered_end), end))
            pieces = next_pieces
        return pieces

    def reframe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        max_group_duration_s: float,
        **kwargs: Any,
    ) -> FramingResult:
        """Re-frame with tighter grouping parameters (for step-down retry).

        Temporarily overrides ``max_group_duration_s`` and runs the same
        VAD + grouping pipeline as :meth:`frame`.  The original setting is
        restored on return.

        Args:
            audio: Scene audio array (mono, typically 16kHz).
            sample_rate: Sample rate of the audio.
            max_group_duration_s: Override for max group duration.
        """
        original = self._max_group
        self._max_group = max_group_duration_s

        # Force re-creation of the segmenter with the new max_group
        old_segmenter = self._segmenter
        self._segmenter = None

        try:
            result = self.frame(audio, sample_rate, **kwargs)
            return result
        finally:
            # Restore original settings and segmenter
            self._max_group = original
            if self._segmenter is not None:
                self._segmenter.cleanup()
            self._segmenter = old_segmenter

    def cleanup(self) -> None:
        """Release the speech segmenter."""
        if self._segmenter is not None:
            self._segmenter.cleanup()
            self._segmenter = None
