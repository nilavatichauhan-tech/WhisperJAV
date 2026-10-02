"""Utility helpers for deciding when to bypass VAD and transcribe the full clip."""
from __future__ import annotations

from typing import Iterable, List, Dict


def _flatten_segments(vad_segments: Iterable[Iterable[Dict[str, float]]]) -> List[Dict[str, float]]:
    """Return a flat list of VAD timestamp dictionaries."""
    flat: List[Dict[str, float]] = []
    for group in vad_segments or []:
        flat.extend(group or [])
    return flat


def _speech_coverage_seconds(flat_segments: List[Dict[str, float]]) -> float:
    """Compute total speech duration from flattened VAD segments."""
    total = 0.0
    for seg in flat_segments:
        start = float(seg.get("start_sec", 0.0))
        end = float(seg.get("end_sec", 0.0))
        if end > start:
            total += end - start
    return total


def should_force_full_transcribe(
    vad_segments: Iterable[Iterable[Dict[str, float]]],
    audio_duration: float,
    min_duration_for_fallback: float = 120.0,
    min_coverage_ratio: float = 0.01,
) -> bool:
    """Decide whether VAD output looks obviously wrong and warrants a full-clip pass.

    The heuristic keeps the original behaviour for short clips while catching the
    failure mode reported by users where long scenes (>10 minutes) end up with only
    a handful of subtitles or none at all because VAD returned almost no speech.
    """
    from whisperjav.utils.logger import logger

    if audio_duration <= 0:
        return False

    if audio_duration < min_duration_for_fallback:
        return False

    flat_segments = _flatten_segments(vad_segments)
    if not flat_segments:
        logger.warning(
            f"[VAD Failover] VAD returned 0 segments for {audio_duration:.1f}s audio! "
            f"Bypassing VAD to transcribe full clip."
        )
        return True

    speech_seconds = _speech_coverage_seconds(flat_segments)
    coverage_ratio = speech_seconds / audio_duration if audio_duration else 0.0
    if coverage_ratio < min_coverage_ratio:
        logger.warning(
            f"[VAD Failover] Speech coverage ratio too low: {coverage_ratio:.2%} "
            f"(speech: {speech_seconds:.1f}s, total: {audio_duration:.1f}s). "
            f"Bypassing VAD to transcribe full clip."
        )
        return True

    # If VAD only detected a single tiny blob on a very long clip, treat it as failure.
    if len(flat_segments) <= 2 and audio_duration >= (min_duration_for_fallback * 4):
        logger.warning(
            f"[VAD Failover] Pathological VAD result: only {len(flat_segments)} segments "
            f"detected on very long {audio_duration:.1f}s audio! "
            f"Bypassing VAD to transcribe full clip."
        )
        return True

    # Warn user if coverage is suspicious (1% to 10%) but don't bypass
    if coverage_ratio < 0.10:
        logger.warning(
            f"[VAD Warning] Suspicious low speech coverage ratio: {coverage_ratio:.2%} "
            f"(speech: {speech_seconds:.1f}s, total: {audio_duration:.1f}s). "
            f"VAD might be overly restrictive."
        )

    return False
