"""MOSS-TTS-Nano ONNX Synthesis Backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np

from worker.backends.base import CANONICAL_SAMPLE_RATE, SynthesisBackend
from worker.engine import MicoraTtsEngine

logger = logging.getLogger("micora.backend.moss")


class MossBackend(SynthesisBackend):
    """MOSS-TTS-Nano ONNX backend.

    Efficient streaming baseline running on CPU (2 threads on Apple Silicon M2).
    Yields 48 kHz stereo PCM chunks natively.
    """

    id = "moss"
    name = "MOSS-TTS-Nano (ONNX)"
    is_streaming = True
    native_sample_rate = CANONICAL_SAMPLE_RATE

    def __init__(
        self,
        model_dir: str | Path = "models/MOSS-TTS-Nano-100M-ONNX",
        thread_count: int = 2,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.thread_count = thread_count
        self.engine: MicoraTtsEngine | None = None

    def load(self) -> None:
        if self.engine is None:
            logger.info("Initializing MOSS-TTS-Nano engine from %s...", self.model_dir)
            self.engine = MicoraTtsEngine(model_dir=self.model_dir, thread_count=self.thread_count)

    def unload(self) -> None:
        if self.engine is not None:
            logger.info("Unloading MOSS-TTS-Nano engine...")
            self.engine = None

    def prepare_voice(
        self,
        audio_path: str | Path | None = None,
        preset: str | None = None,
    ) -> list[list[int]]:
        if self.engine is None:
            self.load()
        assert self.engine is not None

        if audio_path:
            return self.engine.encode_reference(audio_path)
        elif preset:
            return self.engine.get_builtin_voice(preset)
        else:
            raise ValueError("MOSS prepare_voice requires either audio_path or preset")

    def synthesize(
        self,
        text: str,
        voice_state: Any,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Generator[np.ndarray, None, None]:
        if self.engine is None:
            self.load()
        assert self.engine is not None

        prompt_codes = voice_state if voice_state is not None else self.engine.get_builtin_voice("Ava")
        yield from self.engine.synthesize_stream(
            text=text,
            prompt_codes=prompt_codes,
            is_cancelled=is_cancelled,
        )
