"""Micora Synthesis Backend Protocol and Base Classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np

# Canonical audio format expected by Micora CoreAudio pipeline
CANONICAL_SAMPLE_RATE = 48000
CANONICAL_CHANNELS = 2


class SynthesisBackend(ABC):
    """Synthesis engine interface.

    Guarantees:
    - Yields canonical interleaved 48 kHz stereo Float32 PCM chunks of shape (samples, 2).
    - Supports voice state conditioning caching per voice profile.
    - Honors instant cancellation via `is_cancelled`.
    - Fully releases resident memory upon `unload()`.
    """

    id: str
    name: str
    is_streaming: bool
    native_sample_rate: int

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize sessions."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free GPU/Metal/CPU memory."""
        ...

    @abstractmethod
    def prepare_voice(
        self,
        audio_path: str | Path | None = None,
        preset: str | None = None,
    ) -> Any:
        """Derive reusable voice conditioning state."""
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_state: Any,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """Yield canonical (samples, 2) float32 PCM chunks at 48,000 Hz."""
        ...
