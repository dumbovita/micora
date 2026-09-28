# Micora

Micora is a local-first macOS application that allows a user to type text and have that text spoken through a virtual microphone using a locally cloned version of the user's own voice.

Designed specifically for live communication in applications such as Discord, gaming voice chat, OBS, and meetings, Micora operates as a keyboard-controlled microphone running entirely on-device with zero cloud dependence.

---

## Runtime Path

```
Text Input ──► Selected Synthesis Backend ──► Canonical PCM (48kHz stereo) ──► Lock-free Ring Buffer ──► CoreAudio HAL
               ├─ Efficient: MOSS-TTS-Nano (ONNX)                                                            │
               └─ Natural: Chatterbox V3 (MLX)                                    ┌────────────────────────┴────────────────────────┐
                                                                                  ▼                                                 ▼
                                                                       Virtual Mic (BlackHole 2ch)                      Local Monitor (Speakers)
                                                                                  │                                         (15–20% vol)
                                                                                  ▼
                                                                   Discord / Games / Browser / OBS
```

---

## Dual Synthesis Engines: Efficient & Natural Modes

Micora provides two distinct, dynamically switchable local synthesis engines to fit different communication scenarios:

| Metric / Feature | **Efficient Mode** (Default) | **Natural Mode** (Optional) |
| :--- | :--- | :--- |
| **Model** | MOSS-TTS-Nano (100M ONNX) | Chatterbox Multilingual V3 (0.5B MLX) |
| **Primary Advantage** | Sub-second latency, near-zero CPU | Exceptional vocal naturalness & inflection |
| **Time to First Audio (TTFA)** | **~260–350 ms** (streamed) | ~3.3s (short) / ~6.5s (average sentence) |
| **Real-Time Factor (RTF)** | **0.42–0.48x** (~2.2x faster than playback) | 1.62–1.83x |
| **Peak Unified Memory** | **~2,000 MB** | **~2,978 MB** |
| **Streaming Output** | Native 160ms chunk pipeline | Sliced into canonical 160ms chunks |
| **Memory Management** | Active session kept resident | Dynamically loaded; unloads on switch |

---

## Target Hardware & Benchmarks

Micora is designed and benchmarked for fanless Apple Silicon (specifically the **MacBook Air M2 with 16 GB unified memory**):

- **Intra-op Tuning**: 2 threads for MOSS ONNX avoids CPU contention and thermal throttling on fanless designs.
- **MLX Hardware Acceleration**: Chatterbox Multilingual V3 leverages Apple Silicon GPU & Neural Engine via MLX with in-memory speaker conditioning caching.
- **Isolated In-Process Switching**: Only one large model remains resident in memory at a time; switching backends explicitly clears unified memory caches (`mx.clear_cache()`).
- **Canonical PCM Boundary**: Both engines output to canonical 48,000 Hz stereo 32-bit floating point PCM, matching macOS CoreAudio hardware.

---

## Subsystem Architecture

### 1. Dual Backend Synthesis Worker (`worker/`)
- Abstracted behind a lightweight `SynthesisBackend` protocol (`worker/backends/base.py`).
- **MOSS-TTS-Nano (`worker/backends/moss.py`)**: ONNX Runtime session for lightweight real-time streaming with zero PyTorch dependencies.
- **Chatterbox V3 (`worker/backends/chatterbox.py`)**: Apple Silicon MLX inference with automated 24 kHz → 48 kHz stereo resampling via `soxr`.
- **Dataset Analyzer (`worker/analyzer.py`)**: Offline technical acoustic analyzer for local audio folders; extracts optimal 8.0s continuous speech segments for reference conditioning.
- Framed binary IPC protocol over Unix domain sockets with mid-stream cancellation.

### 2. Lock-Free Audio Ring Buffer (`AudioRingBuffer.swift`)
- Single-Producer / Single-Consumer (SPSC) ring buffer implemented with Swift 6 `Synchronization.Atomic`.
- Zero heap allocation during real-time render cycles.
- Automatic zero-padded underflow protection prevents clicks, noise, or stutter.
- Atomic discard on cancellation for instantaneous silence.

### 3. CoreAudio Real-Time Output (`AudioPlayer.swift`, `AudioOutputRouter.swift`)
- Direct low-latency `HALOutput` AudioUnit integration.
- Hardware discovery via CoreAudio HAL (`AudioDeviceManager.swift`).
- Dual routing: primary stream to **BlackHole 2ch** virtual microphone driver; secondary quiet local monitor (15–20% volume) to physical speakers/headphones.

### 4. Sequential Message Queue (`MessageQueue.swift`)
- Swift 6 actor-based queue managing message lifecycle: `queued`, `generating`, `speaking`, `completed`, `cancelled`, `failed`.
- Configurable submission behavior:
  - **Add to Queue** (default): sequential execution without interrupting ongoing speech.
  - **Interrupt Current Speech**: cancels active speech, clears audio buffers, and begins new message immediately.
- Dedicated Stop control halts speech and immediately advances or cleanly idles.

### 5. Multi-Source Voice Profile Manager (`VoiceProfile.swift`)
- Stores voice profiles locally in `~/Library/Application Support/Micora/`.
- Supports single-file audio import (`.wav`) or folder-based dataset discovery.
- Automatically selects backend-appropriate references (`selectedReferences` per backend).
- Built-in default voice preset (`Ava`).

### 6. Native macOS Desktop App & CLI (`MicoraApp`, `MicoraCli`)
- **MicoraApp**: Modern, responsive native macOS interface built with SwiftUI, featuring:
  - Text input with Speak / Stop controls.
  - Mode switcher: `[ Efficient | Natural ]`.
  - Voice selector with single-file and folder dataset import.
  - Local monitor volume control.
  - Live message queue with real-time status badges.
- **MicoraCli**: Interactive terminal console for keyboard-driven speech synthesis.

---

## Building and Running

### Prerequisites
- macOS 15.0+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.12+ (managed with `uv`)
- Swift 6.0 toolchain

```bash
# Set up Python inference environment with uv
uv sync

# Build all Swift targets
swift build
```

### Running the Test Suite
Run the complete unit, CoreAudio, and integration test suite:
```bash
swift run micora-tests
```

### Running the Interactive CLI
```bash
# Interactive REPL mode
swift run micora-cli

# Or speak a single phrase directly:
swift run micora-cli -t "Micora gerçek zamanlı ses sentezi başarıyla çalışıyor."
```

### Running the Desktop App
```bash
swift run micora-app
```

### Virtual Microphone Setup (Discord / OBS / Games)
To route Micora's output as a virtual microphone:
```bash
# Install BlackHole 2ch virtual audio driver
brew install blackhole-2ch
```
1. In Micora, verify that **BlackHole 2ch** is detected as the primary output device.
2. In Discord (or other app), set your **Input Device (Microphone)** to **BlackHole 2ch**.
3. Keep your **Output Device (Headphones)** set to your physical headphones.
4. Adjust Micora's local monitor volume slider so you can hear your cloned voice at comfortable volume without loopback feedback.
