"""Audio Dataset Analyzer and Reference Candidate Selector.

Engineered for Micora Voice Dataset Ingestion (Stages 8 & 9 of Master Brief):
- Scans local directories of audio files without freezing or UI lockup.
- Non-crashing, robust reading of audio files (WAV, AIFF, FLAC).
- Measures objective technical acoustic properties:
  * Duration, sample rate, channels
  * Peak amplitude (dBFS) and RMS level (dBFS)
  * Clipping ratio (% samples at/near 0 dBFS)
  * Silence ratio (% of 50ms frames below -42 dBFS)
  * Speech ratio (1.0 - silence_ratio)
- Automatically deprioritizes/rejects corrupt, clipped, silent, or extreme files.
- Extracts optimal 8.0s continuous speech candidate segments from longer recordings.
- Ranks candidate reference clips so the user can easily audition and select the best reference.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf
import soxr

logger = logging.getLogger("micora.dataset.analyzer")

SUPPORTED_EXTENSIONS = {".wav", ".aiff", ".aif", ".flac", ".m4a"}
CANONICAL_SR = 48000
MIN_USABLE_DURATION_SEC = 2.0
MAX_USABLE_DURATION_SEC = 3600.0  # 1 hour
MAX_CLIPPING_RATIO = 0.01  # 1%
MAX_SILENCE_RATIO = 0.70  # 70%
TARGET_SEGMENT_DURATION_SEC = 8.0


@dataclass
class AudioMetrics:
    file_path: str
    filename: str
    duration_sec: float
    sample_rate: int
    channels: int
    peak_db: float
    rms_db: float
    clipping_ratio: float
    silence_ratio: float
    speech_ratio: float
    is_usable: bool
    rejection_reason: str | None = None
    extracted_segment_path: str | None = None
    score: float = 0.0


def compute_audio_metrics(
    audio_data: np.ndarray,
    sr: int,
    file_path: Path,
) -> AudioMetrics:
    """Computes technical audio quality and speech metrics on mono/stereo audio."""
    total_samples = audio_data.shape[0]
    duration = total_samples / float(sr)

    # Convert to mono for analysis if multi-channel
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        mono = np.mean(audio_data, axis=1)
    else:
        mono = audio_data.flatten()

    # Peak & RMS
    peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) > 0 else 0.0

    peak_db = 20.0 * math.log10(max(peak, 1e-9))
    rms_db = 20.0 * math.log10(max(rms, 1e-9))

    # Clipping detection: sample >= 0.999 (-0.01 dBFS)
    clipped_count = np.sum(np.abs(mono) >= 0.999)
    clipping_ratio = float(clipped_count) / max(total_samples, 1)

    # Frame-based silence analysis (50ms frames)
    frame_len = max(int(0.050 * sr), 1)
    num_frames = total_samples // frame_len
    if num_frames > 0:
        frames = mono[: num_frames * frame_len].reshape(num_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1))
        # Threshold: -42 dBFS (0.0079 amplitude)
        silent_frames = np.sum(frame_rms < 0.0079)
        silence_ratio = float(silent_frames) / num_frames
    else:
        silence_ratio = 1.0

    speech_ratio = max(0.0, 1.0 - silence_ratio)

    # Usability filter
    is_usable = True
    rejection_reason = None

    if duration < MIN_USABLE_DURATION_SEC:
        is_usable = False
        rejection_reason = f"Too short ({duration:.1f}s < {MIN_USABLE_DURATION_SEC}s)"
    elif duration > MAX_USABLE_DURATION_SEC:
        is_usable = False
        rejection_reason = f"Too long ({duration:.1f}s > {MAX_USABLE_DURATION_SEC}s)"
    elif clipping_ratio > MAX_CLIPPING_RATIO:
        is_usable = False
        rejection_reason = f"Excessive clipping ({clipping_ratio * 100:.1f}%)"
    elif silence_ratio > MAX_SILENCE_RATIO:
        is_usable = False
        rejection_reason = f"Excessive silence ({silence_ratio * 100:.1f}%)"
    elif rms_db < -45.0:
        is_usable = False
        rejection_reason = f"Audio level too low ({rms_db:.1f} dBFS)"

    # Cleanliness score: higher is better
    # Prefers RMS in conversational speech range [-26, -16] dBFS, high speech ratio, zero clipping
    score = 0.0
    if is_usable:
        level_penalty = abs(rms_db - (-20.0))  # ideal around -20 dBFS
        score = (speech_ratio * 50.0) - (clipping_ratio * 200.0) - (level_penalty * 1.5)
        # Optimal 6s-10s duration bonus for reference conditioning
        if 6.0 <= duration <= 12.0:
            score += 20.0

    return AudioMetrics(
        file_path=str(file_path.resolve()),
        filename=file_path.name,
        duration_sec=round(duration, 2),
        sample_rate=sr,
        channels=audio_data.shape[1] if audio_data.ndim > 1 else 1,
        peak_db=round(peak_db, 1),
        rms_db=round(rms_db, 1),
        clipping_ratio=round(clipping_ratio, 4),
        silence_ratio=round(silence_ratio, 3),
        speech_ratio=round(speech_ratio, 3),
        is_usable=is_usable,
        rejection_reason=rejection_reason,
        score=round(score, 2),
    )


def extract_best_segment(
    audio_data: np.ndarray,
    sr: int,
    source_file: Path,
    output_dir: Path,
    target_duration: float = TARGET_SEGMENT_DURATION_SEC,
) -> Path | None:
    """Finds and extracts an optimal ~8.0s speech window from longer audio."""
    total_samples = audio_data.shape[0]
    target_samples = int(target_duration * sr)

    if total_samples <= target_samples:
        return None

    # Slide window in 1.0s steps (scan up to 180 seconds of audio for efficiency)
    step_samples = int(1.0 * sr)
    best_rms = -1.0
    best_start = 0
    scan_limit = min(total_samples, int(180.0 * sr))

    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        mono = np.mean(audio_data, axis=1)
    else:
        mono = audio_data.flatten()

    for start in range(0, scan_limit - target_samples, step_samples):
        chunk = mono[start : start + target_samples]
        # Check clipping in chunk
        if np.sum(np.abs(chunk) >= 0.999) > 0:
            continue
        chunk_rms = float(np.sqrt(np.mean(chunk**2)))
        if chunk_rms > best_rms:
            best_rms = chunk_rms
            best_start = start

    if best_rms < 0.0079:  # No suitable speech window
        return None

    # Extract window and resample to canonical 48 kHz stereo if needed
    segment = audio_data[best_start : best_start + target_samples]
    if sr != CANONICAL_SR:
        segment = soxr.resample(segment, in_rate=sr, out_rate=CANONICAL_SR)

    if segment.ndim == 1 or segment.shape[1] == 1:
        segment = np.column_stack([segment.flatten(), segment.flatten()])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{source_file.stem}_seg_{int(best_start / sr)}s.wav"
    out_path = output_dir / out_name
    sf.write(str(out_path), segment.astype(np.float32), CANONICAL_SR)
    return out_path


class DatasetScanner:
    """Scans and analyzes an audio folder, extracting top reference candidates."""

    def __init__(self, folder_path: str | Path) -> None:
        self.folder_path = Path(folder_path).expanduser().resolve()
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {self.folder_path}")

    def scan(
        self,
        max_candidates: int = 5,
        extract_segments: bool = True,
    ) -> dict[str, Any]:
        """Scans folder, analyzes audio metrics, and extracts top reference candidates."""
        files = [
            p for p in self.folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
        ]

        logger.info("Discovered %d audio files in %s", len(files), self.folder_path)

        metrics_list: list[AudioMetrics] = []
        candidates_dir = self.folder_path / ".micora_candidates"

        for file_path in files:
            try:
                data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
                metrics = compute_audio_metrics(data, sr, file_path)

                # Extract 8s speech segment if long file
                if metrics.is_usable and extract_segments and metrics.duration_sec > 12.0:
                    seg_path = extract_best_segment(data, sr, file_path, candidates_dir)
                    if seg_path:
                        metrics.extracted_segment_path = str(seg_path)

                metrics_list.append(metrics)
            except Exception as ex:
                logger.warning("Failed to decode %s: %s", file_path.name, ex)
                metrics_list.append(
                    AudioMetrics(
                        file_path=str(file_path),
                        filename=file_path.name,
                        duration_sec=0.0,
                        sample_rate=0,
                        channels=0,
                        peak_db=-99.0,
                        rms_db=-99.0,
                        clipping_ratio=0.0,
                        silence_ratio=1.0,
                        speech_ratio=0.0,
                        is_usable=False,
                        rejection_reason=f"Decoder error: {str(ex)}",
                    )
                )

        usable_files = [m for m in metrics_list if m.is_usable]
        # Sort by score descending
        usable_files.sort(key=lambda m: m.score, reverse=True)
        top_candidates = usable_files[:max_candidates]

        return {
            "type": "dataset_scanned",
            "folder_path": str(self.folder_path),
            "total_files": len(files),
            "usable_count": len(usable_files),
            "rejected_count": len(files) - len(usable_files),
            "candidates": [asdict(c) for c in top_candidates],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Micora Audio Dataset Analyzer")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing audio files")
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates to return")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    scanner = DatasetScanner(args.folder)
    result = scanner.scan(max_candidates=args.top)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nScanned: {result['folder_path']}")
        print(f"Total: {result['total_files']} files | Usable: {result['usable_count']} | Rejected: {result['rejected_count']}")
        print("\n--- Top Reference Candidates ---")
        for idx, c in enumerate(result["candidates"], 1):
            seg_note = f" (Segment: {Path(c['extracted_segment_path']).name})" if c.get('extracted_segment_path') else ""
            print(
                f"{idx}. {c['filename']} — {c['duration_sec']}s, "
                f"Peak: {c['peak_db']} dBFS, RMS: {c['rms_db']} dBFS, "
                f"Speech: {int(c['speech_ratio'] * 100)}%{seg_note} [Score: {c['score']}]"
            )


if __name__ == "__main__":
    main()
