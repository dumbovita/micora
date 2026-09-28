"""Micora Standalone Voice Synthesis CLI.

Command-line tool for Phase 1 voice synthesis validation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from worker.engine import MicoraTtsEngine, SAMPLE_RATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micora Local Voice Cloning TTS CLI")
    parser.add_argument(
        "--text",
        "-t",
        required=True,
        help="Text to synthesize (casual Turkish, English, numbers, phonemes, etc.)",
    )
    parser.add_argument(
        "--reference",
        "-r",
        default=None,
        help="Path to reference audio file for zero-shot voice cloning.",
    )
    parser.add_argument(
        "--voice",
        "-v",
        default="Ava",
        help="Built-in voice preset name if no reference audio provided (e.g. Ava, Junhao).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.wav",
        help="Path to output WAV file (default: output.wav).",
    )
    parser.add_argument(
        "--model-dir",
        default="models/MOSS-TTS-Nano-100M-ONNX",
        help="Path to ONNX model directory.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="ONNX Runtime intra-op threads (default: 2 for M2).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream chunks incrementally and print latency per chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_dir).resolve()
    if not model_path.is_dir():
        print(f"Error: Model directory not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing Micora TTS Engine (threads={args.threads})...")
    t0 = time.perf_counter()
    engine = MicoraTtsEngine(model_dir=model_path, thread_count=args.threads)
    init_time = time.perf_counter() - t0
    print(f"Engine initialized in {init_time:.2f}s")

    # Resolve voice conditioning
    if args.reference:
        ref_path = Path(args.reference).resolve()
        print(f"Encoding reference audio: {ref_path.name}...")
        t_ref = time.perf_counter()
        prompt_codes = engine.encode_reference(ref_path)
        ref_time = time.perf_counter() - t_ref
        print(f"Reference encoded: {len(prompt_codes)} frames in {ref_time:.3f}s")
    else:
        print(f"Using built-in voice: {args.voice}")
        prompt_codes = engine.get_builtin_voice(args.voice)

    print(f"Synthesizing: \"{args.text}\"")
    t_start = time.perf_counter()
    chunks: list[np.ndarray] = []
    first_chunk_time: float | None = None

    for i, chunk in enumerate(engine.synthesize_stream(args.text, prompt_codes)):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - t_start
        chunks.append(chunk)
        if args.stream:
            chunk_ms = (chunk.shape[0] / SAMPLE_RATE) * 1000
            print(f"  [Chunk {i+1:2d}] +{time.perf_counter() - t_start:.3f}s | {chunk.shape[0]} samples ({chunk_ms:.0f}ms)")

    gen_time = time.perf_counter() - t_start

    if not chunks:
        print("Error: No audio was generated.", file=sys.stderr)
        sys.exit(1)

    full_audio = np.concatenate(chunks, axis=0)
    audio_dur = full_audio.shape[0] / SAMPLE_RATE
    rtf = gen_time / audio_dur if audio_dur > 0 else 0

    out_path = Path(args.output).resolve()
    sf.write(str(out_path), full_audio, SAMPLE_RATE)

    print("\n=== Synthesis Complete ===")
    print(f"Output File:     {out_path}")
    print(f"Audio Duration:  {audio_dur:.2f}s ({SAMPLE_RATE} Hz, 2 channels)")
    print(f"Generation Time: {gen_time:.3f}s")
    print(f"Time to First:   {first_chunk_time or 0.0:.3f}s")
    print(f"RTF:             {rtf:.3f} ({(1.0/rtf if rtf > 0 else 0):.1f}x faster than real-time)")


if __name__ == "__main__":
    main()
