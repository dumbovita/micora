"""Chatterbox Multilingual V3 Synthesis Backend using MLX.

Engineered specifically for Apple Silicon (MacBook Air M2, 16 GB).
- Runs natively on Apple Silicon GPU/ANE via MLX.
- Zero-shot multilingual voice cloning with Turkish (primary) and English support.
- Derives reusable in-memory Conditionals (speaker embedding + vocoder reference).
- S3Gen native 24,000 Hz output is resampled via soxr to Micora's canonical
  48,000 Hz stereo float32 format at the backend boundary.
- Non-streaming full-utterance generation delivered in canonical 160ms chunks.
"""

from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from typing import Any, Callable, Generator

import mlx.core as mx
import numpy as np
import soxr

from worker.backends.base import CANONICAL_SAMPLE_RATE, SynthesisBackend
from worker.normalizer import is_english_text, normalize_text_for_tts

logger = logging.getLogger("micora.backend.chatterbox")

CHATTERBOX_MODEL_ID = "mlx-community/chatterbox-multilingual-v3"
NATIVE_SAMPLE_RATE = 24000
CHUNK_FRAMES_48K = 7680  # 160ms per streaming chunk at 48 kHz


class ChatterboxBackend(SynthesisBackend):
    """Chatterbox Multilingual V3 backend using MLX."""

    id = "chatterbox"
    name = "Chatterbox Multilingual V3 (MLX)"
    is_streaming = False
    native_sample_rate = NATIVE_SAMPLE_RATE

    def __init__(
        self,
        model_id: str = CHATTERBOX_MODEL_ID,
    ) -> None:
        self.model_id = model_id
        self.model: Any | None = None
        self._fallback_ref_path: Path | None = None

    def _find_fallback_ref(self) -> Path:
        if self._fallback_ref_path and self._fallback_ref_path.exists():
            return self._fallback_ref_path
        candidates = [
            Path("models/reference_speech.wav").resolve(),
            Path(".reference/MOSS-TTS-Nano/assets/audio/en_2.wav").resolve(),
            Path("tests/fixtures/reference_speech.wav").resolve(),
        ]
        for c in candidates:
            if c.exists():
                self._fallback_ref_path = c
                return c
        raise FileNotFoundError("No reference audio file available for Chatterbox fallback.")

    def load(self) -> None:
        if self.model is None:
            logger.info("Loading Chatterbox Multilingual V3 (%s) on MLX...", self.model_id)
            from mlx_audio.tts.utils import load_model

            self.model = load_model(self.model_id)
            # Evaluate all lazy RoPE frequencies so any background thread can safely run inference
            if hasattr(self.model, "t3") and hasattr(self.model.t3, "tfmr"):
                for layer in self.model.t3.tfmr.model.layers:
                    rope = getattr(getattr(layer, "self_attn", None), "rope", None)
                    if hasattr(rope, "_freqs"):
                        mx.eval(rope._freqs)

            logger.info("Chatterbox Multilingual V3 loaded successfully.")

    def unload(self) -> None:
        if self.model is not None:
            logger.info("Unloading Chatterbox model and releasing unified memory...")
            self.model = None
            try:
                mx.clear_cache()
            except Exception:
                pass
            gc.collect()

    def prepare_voice(
        self,
        audio_path: str | Path | None = None,
        preset: str | None = None,
    ) -> Any:
        if self.model is None:
            self.load()
        assert self.model is not None

        ref_path = Path(audio_path).expanduser().resolve() if audio_path else self._find_fallback_ref()
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        logger.info("Preparing Chatterbox conditionals from %s...", ref_path.name)
        conds = self.model.prepare_conditionals(str(ref_path), ref_sr=NATIVE_SAMPLE_RATE, exaggeration=0.1)
        logger.info("Chatterbox voice conditioning ready (speaker embedding: %s)", conds.t3.speaker_emb.shape)
        return conds

    def synthesize(
        self,
        text: str,
        voice_state: Any,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Generator[np.ndarray, None, None]:
        if re.search(r"\d", text):
            text = normalize_text_for_tts(text)
        if not text or not text.strip():
            return

        if is_cancelled is not None and is_cancelled():
            logger.info("Chatterbox synthesis cancelled before start")
            return

        if self.model is None:
            self.load()
        assert self.model is not None

        # Resolve conditioning
        conds = voice_state
        if conds is None:
            logger.warning("No voice state provided for Chatterbox; preparing default fallback conditionals")
            conds = self.prepare_voice()

        # Language selection: "en" if text matches English heuristic, otherwise default "tr"
        lang_code = "en" if is_english_text(text) else "tr"

        # Generate complete waveform
        results = list(
            self.model.generate(
                text=text.strip(),
                conds=conds,
                lang_code=lang_code,
                exaggeration=0.1,
                cfg_weight=0.5,
                temperature=0.8,
                repetition_penalty=1.2,
                min_p=0.05,
                top_p=1.0,
                max_new_tokens=1000,
                verbose=False,
            )
        )

        if is_cancelled is not None and is_cancelled():
            logger.info("Chatterbox synthesis cancelled immediately after model generation")
            return

        if not results:
            logger.error("Chatterbox generation returned no audio chunks")
            return

        result = results[0]
        audio_mlx = result.audio
        mx.eval(audio_mlx)
        audio_np = np.array(audio_mlx, copy=False).astype(np.float32)

        if len(audio_np) == 0:
            return

        # Resample native 24 kHz mono to canonical 48 kHz stereo
        resampled_mono = soxr.resample(audio_np, in_rate=result.sample_rate, out_rate=CANONICAL_SAMPLE_RATE)
        stereo_48k = np.stack([resampled_mono, resampled_mono], axis=1).astype(np.float32)

        # Slice into canonical 160ms chunks (7680 frames) and yield with cancellation checks
        total_frames = stereo_48k.shape[0]
        offset = 0

        while offset < total_frames:
            if is_cancelled is not None and is_cancelled():
                logger.info("Chatterbox PCM delivery cancelled at frame %d/%d", offset, total_frames)
                return

            end = min(offset + CHUNK_FRAMES_48K, total_frames)
            chunk = stereo_48k[offset:end]
            yield chunk
            offset = end
