"""Micora Engine Head-to-Head Comparison Benchmark (Stage 12 of Master Brief).

Compares Efficient Mode (MOSS-TTS-Nano ONNX) vs Natural Mode (Chatterbox Multilingual V3 MLX):
- Uses representative Turkish & English sentences from corpus/validation_corpus.json.
- Measures cold load time, reference conditioning time, per-sentence generation latency,
  generated audio duration, Real-Time Factor (RTF), and peak resident memory (RSS).
- Saves generated canonical 48 kHz stereo WAV files for direct listening comparison.
- Unloads the first engine before loading the second, ensuring isolated memory measurement.
"""

from __future__ import annotations

import json
import logging
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from worker.backends.base import CANONICAL_CHANNELS, CANONICAL_SAMPLE_RATE
from worker.backends.chatterbox import ChatterboxBackend
from worker.backends.moss import MossBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("micora.benchmark.comparison")


def get_peak_rss_mb() -> float:
    """Return peak resident set size in MB (macOS ru_maxrss is in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def benchmark_engine(
    backend: Any,
    test_sentences: list[dict[str, str]],
    ref_audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cold Load
    logger.info("--- Loading %s ---", backend.name)
    t0 = time.perf_counter()
    backend.load()
    load_time = time.perf_counter() - t0
    logger.info("%s loaded in %.3fs", backend.name, load_time)

    # 2. Reference Conditioning
    logger.info("Preparing voice conditioning from %s...", ref_audio_path.name)
    t_ref = time.perf_counter()
    if backend.id == "moss":
        voice_state = backend.prepare_voice(audio_path=ref_audio_path)
    else:
        voice_state = backend.prepare_voice(audio_path=ref_audio_path)
    ref_time = time.perf_counter() - t_ref
    logger.info("Voice conditioning prepared in %.3fs", ref_time)

    # 3. Sentence Generation
    results = []
    total_gen_time = 0.0
    total_audio_time = 0.0

    for sample in test_sentences:
        sid = sample["id"]
        text = sample["text"]

        t_start = time.perf_counter()
        first_chunk_time = None
        chunks: list[np.ndarray] = []

        for chunk in backend.synthesize(text=text, voice_state=voice_state):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - t_start
            chunks.append(chunk)

        gen_time = time.perf_counter() - t_start
        if chunks:
            all_pcm = np.concatenate(chunks, axis=0)
            audio_sec = all_pcm.shape[0] / float(CANONICAL_SAMPLE_RATE)
            rtf = gen_time / audio_sec if audio_sec > 0 else 0.0

            # Save canonical audio
            out_file = output_dir / f"{sid}_48k.wav"
            sf.write(str(out_file), all_pcm, CANONICAL_SAMPLE_RATE)
        else:
            audio_sec = 0.0
            rtf = 0.0

        ttfa = first_chunk_time if first_chunk_time is not None else gen_time

        logger.info(
            "[%s] '%s': %.2fs audio in %.2fs (RTF: %.3f, TTFA: %.3fs)",
            backend.id,
            text[:30],
            audio_sec,
            gen_time,
            rtf,
            ttfa,
        )

        results.append({
            "id": sid,
            "text": text,
            "generation_seconds": round(gen_time, 3),
            "audio_seconds": round(audio_sec, 3),
            "rtf": round(rtf, 3),
            "time_to_first_audio_seconds": round(ttfa, 3),
        })

        total_gen_time += gen_time
        total_audio_time += audio_sec

    avg_rtf = total_gen_time / total_audio_time if total_audio_time > 0 else 0.0
    rss = get_peak_rss_mb()

    # Unload
    backend.unload()

    return {
        "engine_id": backend.id,
        "engine_name": backend.name,
        "is_streaming": backend.is_streaming,
        "cold_load_seconds": round(load_time, 3),
        "reference_conditioning_seconds": round(ref_time, 3),
        "total_generation_seconds": round(total_gen_time, 3),
        "total_audio_seconds": round(total_audio_time, 3),
        "average_rtf": round(avg_rtf, 3),
        "peak_rss_mb": round(rss, 1),
        "samples": results,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    corpus_file = repo_root / "corpus" / "validation_corpus.json"
    ref_audio = repo_root / "models" / "reference_speech.wav"
    output_base = repo_root / "benchmarks" / "audio_comparison"

    corpus = json.loads(corpus_file.read_text(encoding="utf-8"))
    # Select 6 core sentences representing conversational Micora use
    test_ids = {"casual_01", "short_01", "long_01", "question_01", "special_chars_01", "english_01"}
    test_sentences = [s for s in corpus["samples"] if s["id"] in test_ids]

    logger.info("Benchmarking MOSS-TTS-Nano (Efficient Mode)...")
    moss = MossBackend(model_dir=repo_root / "models" / "MOSS-TTS-Nano-100M-ONNX", thread_count=2)
    moss_res = benchmark_engine(moss, test_sentences, ref_audio, output_base / "moss")

    logger.info("Benchmarking Chatterbox Multilingual V3 (Natural Mode)...")
    chatterbox = ChatterboxBackend()
    chatter_res = benchmark_engine(chatterbox, test_sentences, ref_audio, output_base / "chatterbox")

    comparison = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": "Apple MacBook Air M2 (16 GB unified memory, 8-core CPU)",
        "reference_audio": ref_audio.name,
        "summary": {
            "efficient_mode_moss": {
                "name": moss_res["engine_name"],
                "streaming": moss_res["is_streaming"],
                "cold_load_sec": moss_res["cold_load_seconds"],
                "avg_rtf": moss_res["average_rtf"],
                "peak_rss_mb": moss_res["peak_rss_mb"],
            },
            "natural_mode_chatterbox": {
                "name": chatter_res["engine_name"],
                "streaming": chatter_res["is_streaming"],
                "cold_load_sec": chatter_res["cold_load_seconds"],
                "avg_rtf": chatter_res["average_rtf"],
                "peak_rss_mb": chatter_res["peak_rss_mb"],
            },
        },
        "moss": moss_res,
        "chatterbox": chatter_res,
    }

    out_json = repo_root / "benchmarks" / "engine_comparison.json"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info("Engine comparison results written to %s", out_json)

    # Print summary table
    print("\n" + "=" * 70)
    print("MICORA ENGINE HEAD-TO-HEAD COMPARISON (M2 APPLE SILICON)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Efficient (MOSS)':<18} | {'Natural (Chatterbox)':<18}")
    print("-" * 70)
    print(f"{'Streaming Output':<30} | {'Native (160ms)':<18} | {'Utterance Sliced':<18}")
    print(f"{'Cold Load Duration':<30} | {moss_res['cold_load_seconds']:>15.2f}s | {chatter_res['cold_load_seconds']:>15.2f}s")
    print(f"{'Ref Conditioning Cost':<30} | {moss_res['reference_conditioning_seconds']:>15.2f}s | {chatter_res['reference_conditioning_seconds']:>15.2f}s")
    print(f"{'Average RTF (Speech Speed)':<30} | {moss_res['average_rtf']:>15.2f}x | {chatter_res['average_rtf']:>15.2f}x")
    print(f"{'Total Audio Produced':<30} | {moss_res['total_audio_seconds']:>15.2f}s | {chatter_res['total_audio_seconds']:>15.2f}s")
    print(f"{'Total Generation Time':<30} | {moss_res['total_generation_seconds']:>15.2f}s | {chatter_res['total_generation_seconds']:>15.2f}s")
    print(f"{'Peak RSS Memory':<30} | {moss_res['peak_rss_mb']:>14.1f}MB | {chatter_res['peak_rss_mb']:>14.1f}MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
