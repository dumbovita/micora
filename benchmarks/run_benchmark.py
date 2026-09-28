#!/usr/bin/env python3
"""Micora M2 TTS Benchmark Tool.

Measures:
- Cold model initialization time
- Reference conditioning cost
- Warm synthesis
- Time to first playable audio (TTFA)
- Total generation time
- Generated audio duration
- Real-time factor (RTF)
- Peak resident memory (RSS)
- CPU time (process time)
Across 1, 2, and 4 thread configurations.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import sentencepiece as spm

# Add reference module to path for OrtCpuRuntime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".reference" / "MOSS-TTS-Nano"))
from ort_cpu_runtime import OrtCpuRuntime, SAMPLE_MODE_FIXED


def get_peak_rss_mb() -> float:
    """Return peak resident memory in MB (macOS ru_maxrss is in bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024.0 * 1024.0)


def encode_reference_audio(
    runtime: OrtCpuRuntime,
    audio_path: Path,
    target_sr: int = 48000,
    target_channels: int = 2,
) -> tuple[list[list[int]], float]:
    """Load reference audio with soundfile/soxr and encode using ONNX codec."""
    t0 = time.perf_counter()
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if sr != target_sr:
        data = soxr.resample(data, in_rate=sr, out_rate=target_sr)
    if data.shape[1] == 1 and target_channels == 2:
        waveform = np.repeat(data.T, 2, axis=0)
    elif data.shape[1] == target_channels:
        waveform = data.T
    else:
        raise ValueError(f"Channel mismatch: {data.shape[1]} vs {target_channels}")

    waveform = np.expand_dims(waveform, axis=0).astype(np.float32)
    waveform_len = waveform.shape[-1]

    outputs = runtime.sessions["codec_encode"].run(
        None,
        {"waveform": waveform, "input_lengths": np.asarray([waveform_len], dtype=np.int32)},
    )
    named = dict(zip([o.name for o in runtime.sessions["codec_encode"].get_outputs()], outputs))
    codes = np.asarray(named["audio_codes"], dtype=np.int32)
    code_len = int(named["audio_code_lengths"].reshape(-1)[0])
    num_quantizers = int(runtime.codec_meta["codec_config"]["num_quantizers"])
    prompt_codes = [[int(codes[0, f, q]) for q in range(num_quantizers)] for f in range(code_len)]
    elapsed = time.perf_counter() - t0
    return prompt_codes, elapsed


def run_single_benchmark(
    thread_count: int,
    test_sentences: list[dict[str, str]],
    prompt_codes: list[list[int]],
    model_dir: Path,
    tokenizer: spm.SentencePieceProcessor,
) -> dict[str, Any]:
    print(f"\n--- Benchmarking thread_count={thread_count} ---")
    rss_start = get_peak_rss_mb()

    # Measure cold init
    t_init_start = time.perf_counter()
    runtime = OrtCpuRuntime(model_dir=model_dir, thread_count=thread_count, sample_mode=SAMPLE_MODE_FIXED)
    cold_init_time = time.perf_counter() - t_init_start
    print(f"Cold model initialization: {cold_init_time:.3f}s")

    # Warmup step
    t_warm_start = time.perf_counter()
    runtime.warmup()
    warmup_time = time.perf_counter() - t_warm_start
    print(f"Warmup time: {warmup_time:.3f}s")

    sentence_results: list[dict[str, Any]] = []
    frames_per_chunk = 2  # 160ms per streaming chunk

    for item in test_sentences:
        text = item["text"]
        text_id = item["id"]
        text_tokens = tokenizer.encode(text, out_type=int)
        req_rows = runtime.build_voice_clone_request_rows(prompt_codes, text_tokens)

        runtime.codec_streaming_session.reset()
        pending_frames: list[list[int]] = []
        ttfa: float | None = None
        pcm_chunks: list[np.ndarray] = []

        t_gen_start = time.perf_counter()
        cpu_start = time.process_time()

        def on_frame(_all_frames: list[list[int]], _step_idx: int, frame: list[int]) -> None:
            nonlocal ttfa
            pending_frames.append(list(frame))
            if len(pending_frames) >= frames_per_chunk:
                decoded = runtime.codec_streaming_session.run_frames(pending_frames)
                pending_frames.clear()
                if decoded is not None:
                    audio, length = decoded
                    if length > 0:
                        if ttfa is None:
                            ttfa = time.perf_counter() - t_gen_start
                        pcm_chunks.append(audio[0, :, :length].T)

        gen_frames = runtime.generate_audio_frames(req_rows, on_frame=on_frame)
        if pending_frames:
            decoded = runtime.codec_streaming_session.run_frames(pending_frames)
            if decoded is not None:
                audio, length = decoded
                if length > 0:
                    if ttfa is None:
                        ttfa = time.perf_counter() - t_gen_start
                    pcm_chunks.append(audio[0, :, :length].T)

        runtime.codec_streaming_session.reset()
        gen_time = time.perf_counter() - t_gen_start
        cpu_time = time.process_time() - cpu_start

        total_samples = sum(c.shape[0] for c in pcm_chunks)
        audio_dur = total_samples / 48000.0
        rtf = gen_time / audio_dur if audio_dur > 0 else 0.0

        print(
            f"[{text_id:15s}] dur={audio_dur:.2f}s gen={gen_time:.3f}s "
            f"ttfa={ttfa or 0.0:.3f}s rtf={rtf:.3f} cpu={cpu_time:.3f}s"
        )

        sentence_results.append({
            "id": text_id,
            "text": text,
            "audio_duration_seconds": round(audio_dur, 3),
            "generation_time_seconds": round(gen_time, 3),
            "time_to_first_audio_seconds": round(ttfa or 0.0, 3),
            "real_time_factor": round(rtf, 3),
            "cpu_time_seconds": round(cpu_time, 3),
            "generated_frames": len(gen_frames),
        })

    avg_rtf = float(np.mean([s["real_time_factor"] for s in sentence_results]))
    avg_ttfa = float(np.mean([s["time_to_first_audio_seconds"] for s in sentence_results]))
    total_audio = float(sum(s["audio_duration_seconds"] for s in sentence_results))
    total_gen = float(sum(s["generation_time_seconds"] for s in sentence_results))
    peak_rss = get_peak_rss_mb()

    return {
        "thread_count": thread_count,
        "cold_init_seconds": round(cold_init_time, 3),
        "warmup_seconds": round(warmup_time, 3),
        "total_audio_seconds": round(total_audio, 3),
        "total_generation_seconds": round(total_gen, 3),
        "average_rtf": round(avg_rtf, 3),
        "average_ttfa_seconds": round(avg_ttfa, 3),
        "peak_rss_mb": round(peak_rss, 1),
        "sentences": sentence_results,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    model_dir = repo_root / "models" / "MOSS-TTS-Nano-100M-ONNX"
    corpus_path = repo_root / "corpus" / "validation_corpus.json"
    benchmark_dir = repo_root / "benchmarks"
    benchmark_dir.mkdir(exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(model_dir / "tokenizer.model"))

    print("Loading corpus...")
    corpus_data = json.loads(corpus_path.read_text(encoding="utf-8"))
    # Select representative subset of corpus for standard benchmark: short, casual, questions, long, special chars, English
    benchmark_ids = {
        "short_01", "casual_01", "question_01", "long_01", "special_chars_01", "english_01"
    }
    test_sentences = [s for s in corpus_data["samples"] if s["id"] in benchmark_ids]

    # Reference conditioning test
    ref_audio_path = repo_root / ".reference" / "MOSS-TTS-Nano" / "assets" / "audio" / "en_2.wav"
    init_runtime = OrtCpuRuntime(model_dir=model_dir, thread_count=2, sample_mode=SAMPLE_MODE_FIXED)
    prompt_codes, ref_enc_time = encode_reference_audio(init_runtime, ref_audio_path)
    print(f"Reference conditioning cost for {ref_audio_path.name}: {ref_enc_time:.3f}s ({len(prompt_codes)} frames)")
    del init_runtime

    thread_configs = [1, 2, 4]
    all_runs = []
    for tc in thread_configs:
        res = run_single_benchmark(tc, test_sentences, prompt_codes, model_dir, tokenizer)
        all_runs.append(res)

    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": "Apple M2 (Mac14,2), 8 Cores (4P+4E), 16 GB unified memory",
        "reference_conditioning_cost_seconds": round(ref_enc_time, 3),
        "reference_audio": str(ref_audio_path.name),
        "thread_benchmarks": all_runs,
    }

    out_file = benchmark_dir / "results_m2.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBenchmark completed successfully! Results written to {out_file}")

    print("\n=== SUMMARY COMPARISON ===")
    print(f"{'Threads':<10} | {'Avg RTF':<10} | {'Avg TTFA':<12} | {'Total Gen':<12} | {'Peak RSS':<10}")
    print("-" * 62)
    for r in all_runs:
        print(
            f"{r['thread_count']:<10} | {r['average_rtf']:<10.3f} | "
            f"{r['average_ttfa_seconds']:<10.3f}s | {r['total_generation_seconds']:<10.3f}s | {r['peak_rss_mb']:<8.1f} MB"
        )


if __name__ == "__main__":
    main()
