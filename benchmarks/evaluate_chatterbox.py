#!/usr/bin/env python3
"""Micora Chatterbox Multilingual V3 Evaluation on Apple Silicon (M2).

Phase 2 Standalone Evaluation:
- Measures cold model initialization time
- Verifies model variant (Multilingual V3)
- Measures reference conditioning cost and verifies Conditionals reuse
- Evaluates Turkish and English synthesis across validation corpus
- Measures RTF, generation latency, and output audio properties (sample rate, channels)
- Measures resident memory (RSS) and MLX Metal memory
- Tests repeated generation stability and cache clearing
- Saves generated audio files to benchmarks/audio_chatterbox/
- Exports structured benchmark results to benchmarks/chatterbox_m2_evaluation.json
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import soundfile as sf
import soxr

from mlx_audio.tts.utils import load_model


def get_peak_rss_mb() -> float:
    """Return peak resident memory in MB (macOS ru_maxrss is in bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024.0 * 1024.0)


def get_hardware_info() -> str:
    """Get Mac hardware info string via sysctl."""
    try:
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
        ncpu = subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
        mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        mem_gb = mem_bytes // (1024**3)
        return f"{model}, {ncpu} cores, {mem_gb} GB unified memory"
    except Exception:
        return "Apple Silicon Mac"


def get_mlx_active_mem_mb() -> float:
    try:
        return mx.get_active_memory() / (1024.0 * 1024.0)
    except AttributeError:
        return mx.metal.get_active_memory() / (1024.0 * 1024.0)


def get_mlx_peak_mem_mb() -> float:
    try:
        return mx.get_peak_memory() / (1024.0 * 1024.0)
    except AttributeError:
        return mx.metal.get_peak_memory() / (1024.0 * 1024.0)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    corpus_path = repo_root / "corpus" / "validation_corpus.json"
    output_audio_dir = repo_root / "benchmarks" / "audio_chatterbox"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = repo_root / "benchmarks" / "chatterbox_m2_evaluation.json"

    ref_audio_candidates = [
        repo_root / "tests" / "fixtures" / "reference_speech.wav",
        repo_root / ".reference" / "MOSS-TTS-Nano" / "assets" / "audio" / "en_2.wav",
    ]
    ref_audio_path = next((p for p in ref_audio_candidates if p.exists()), None)
    if not ref_audio_path:
        print("ERROR: No reference audio file found in tests/fixtures or .reference/MOSS-TTS-Nano")
        sys.exit(1)

    print(f"=== Micora Chatterbox Multilingual V3 Evaluation ===")
    hw_info = get_hardware_info()
    print(f"Hardware: {hw_info}")
    print(f"Reference audio: {ref_audio_path}")

    # 1. Cold Load
    print("\n--- Measuring Cold Model Initialization ---")
    rss_before_load = get_peak_rss_mb()
    t0_load = time.perf_counter()
    model_id = "mlx-community/chatterbox-multilingual-v3"
    model = load_model(model_id)
    load_time = time.perf_counter() - t0_load
    rss_after_load = get_peak_rss_mb()
    metal_active_after_load = get_mlx_active_mem_mb()

    print(f"Cold model load time: {load_time:.3f}s")
    print(f"Process peak RSS: {rss_after_load:.1f} MB (delta: +{rss_after_load - rss_before_load:.1f} MB)")
    print(f"Metal active memory: {metal_active_after_load:.1f} MB")

    # 2. Checkpoint Verification
    is_multilingual = getattr(model.config, "multilingual", False)
    text_preprocessing = getattr(model.config, "text_preprocessing", "")
    native_sample_rate = getattr(model, "sample_rate", 24000)
    print(f"Model ID: {model_id}")
    print(f"Config multilingual: {is_multilingual}")
    print(f"Config text_preprocessing: {text_preprocessing}")
    print(f"Native model sample_rate: {native_sample_rate} Hz")

    is_v3 = is_multilingual and text_preprocessing.lower() == "nfkd,fullcase"
    if is_v3:
        print("✓ Verified: Exact Chatterbox Multilingual V3 architecture confirmed (multilingual=True, text_preprocessing=NFKD,fullcase)")
    else:
        print(f"WARNING: Model configuration does not match expected Multilingual V3! multilingual={is_multilingual}, text_preprocessing={text_preprocessing}")

    # 3. Reference Conditioning & Conditionals Caching
    print("\n--- Measuring Reference Conditioning ---")
    t0_cond = time.perf_counter()
    conds = model.prepare_conditionals(str(ref_audio_path), ref_sr=24000, exaggeration=0.1)
    cond_time = time.perf_counter() - t0_cond
    print(f"prepare_conditionals elapsed: {cond_time:.3f}s")
    print(f"Conditionals prepared: t3_cond speaker_emb shape: {conds.t3.speaker_emb.shape}")

    # 4. Validation Corpus Synthesis
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    # Select core representative sentences across categories
    sentence_ids_to_test = [
        "short_01",              # "Tamamdır."
        "casual_01",             # "Selam, naber? Akşama oyunda mısın?"
        "special_chars_01",      # "Ilık süt içerken, ışıltılı gökyüzüne bakıp çiğdem çıtlatıyoruz."
        "question_01",           # "Yarın saat kaçta buluşuyoruz, plan belli oldu mu?"
        "long_01",               # "Bugün hava biraz yağmurlu olduğu için dışarı çıkmak yerine evde kalıp kod yazmaya karar verdim."
        "numeric_currency_percent", # "Fiyatı 250,50 TL olmuş, yani yaklaşık %20 zam gelmiş."
        "english_01",            # "Hello! Voice cloning is working directly on Apple Silicon."
        "gaming_codeswitching",  # "Mid lane push yapalım, ulti hazır olunca fight başlatırız."
    ]

    sentences_map = {item["id"]: item for item in corpus_data["samples"]}
    test_items = [sentences_map[sid] for sid in sentence_ids_to_test if sid in sentences_map]

    sentence_results: list[dict[str, Any]] = []

    print("\n--- Synthesizing Test Corpus ---")
    for item in test_items:
        sid = item["id"]
        text = item["text"]
        lang = "en" if item.get("category") == "english_secondary" else "tr"

        t0_gen = time.perf_counter()
        results = list(model.generate(
            text=text,
            conds=conds,
            lang_code=lang,
            exaggeration=0.1,
            cfg_weight=0.5,
            temperature=0.8,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
            max_new_tokens=1000,
            verbose=False,
        ))
        gen_time = time.perf_counter() - t0_gen

        if not results:
            print(f"ERROR: No results returned for {sid}")
            continue

        result = results[0]
        audio_mlx = result.audio
        mx.eval(audio_mlx)
        audio_np = np.array(audio_mlx, copy=False).astype(np.float32)

        audio_duration = len(audio_np) / float(result.sample_rate)
        rtf = gen_time / audio_duration if audio_duration > 0 else 0.0

        # Save native 24 kHz audio
        native_wav_path = output_audio_dir / f"{sid}_24k.wav"
        sf.write(str(native_wav_path), audio_np, result.sample_rate)

        # Convert to Micora Canonical format: 48 kHz stereo float32
        resampled = soxr.resample(audio_np, in_rate=result.sample_rate, out_rate=48000)
        stereo_48k = np.stack([resampled, resampled], axis=1)
        canonical_wav_path = output_audio_dir / f"{sid}_canonical_48k.wav"
        sf.write(str(canonical_wav_path), stereo_48k, 48000)

        res_dict = {
            "id": sid,
            "category": item.get("category"),
            "language": lang,
            "text": text,
            "generation_time_seconds": round(gen_time, 3),
            "audio_duration_seconds": round(audio_duration, 3),
            "real_time_factor": round(rtf, 3),
            "native_sample_rate": result.sample_rate,
            "native_samples": len(audio_np),
            "canonical_format": "48000 Hz, stereo float32",
            "audio_file": str(canonical_wav_path.name),
        }
        sentence_results.append(res_dict)

        print(f"[{sid}] ({lang}) \"{text[:40]}...\" -> {audio_duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.3f})")

    # 5. Stability & Repeated Generation Check
    print("\n--- Repeated Generation & Memory Leak Check ---")
    memory_progression = []
    short_text = "Tamamdır, anladım."
    for i in range(5):
        t0_rep = time.perf_counter()
        _ = list(model.generate(
            text=short_text,
            conds=conds,
            lang_code="tr",
            exaggeration=0.1,
            verbose=False,
        ))
        gen_rep = time.perf_counter() - t0_rep
        act_mem = get_mlx_active_mem_mb()
        peak_mem = get_mlx_peak_mem_mb()
        memory_progression.append({
            "iteration": i + 1,
            "generation_time_seconds": round(gen_rep, 3),
            "metal_active_mb": round(act_mem, 1),
            "metal_peak_mb": round(peak_mem, 1),
        })

    # Clear cache test
    mx.clear_cache()
    post_clear_mem = get_mlx_active_mem_mb()
    print(f"Metal active memory after mx.clear_cache(): {post_clear_mem:.1f} MB")

    total_gen_time = sum(s["generation_time_seconds"] for s in sentence_results)
    total_audio_time = sum(s["audio_duration_seconds"] for s in sentence_results)
    avg_rtf = round(total_gen_time / total_audio_time, 3) if total_audio_time > 0 else 0.0

    output_report = {
        "model_id": model_id,
        "hardware": hw_info,
        "is_multilingual": is_multilingual,
        "text_preprocessing": text_preprocessing,
        "native_sample_rate": native_sample_rate,
        "cold_load_seconds": round(load_time, 3),
        "peak_rss_mb": round(get_peak_rss_mb(), 1),
        "metal_active_load_mb": round(metal_active_after_load, 1),
        "reference_conditioning_seconds": round(cond_time, 3),
        "reference_audio": str(ref_audio_path.name),
        "total_audio_seconds": round(total_audio_time, 3),
        "total_generation_seconds": round(total_gen_time, 3),
        "average_rtf": avg_rtf,
        "memory_progression": memory_progression,
        "post_clear_metal_active_mb": round(post_clear_mem, 1),
        "sentence_results": sentence_results,
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete! Results written to {results_json_path}")
    print(f"Average RTF across all test sentences: {avg_rtf:.3f}")
    print(f"Total audio synthesized: {total_audio_time:.2f}s in {total_gen_time:.2f}s")


if __name__ == "__main__":
    main()
