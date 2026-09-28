"""Micora ONNX TTS Inference Engine.

Engineered specifically for Apple Silicon (MacBook Air M2, 16 GB).
- Fully decoupled from PyTorch and torchaudio (uses soundfile + soxr + numpy + onnxruntime).
- Zero-shot voice cloning with reference audio encoding and caching.
- True incremental PCM streaming (yields 48 kHz 2-channel float32 / int16 chunks).
- Cancellation flag support for instantaneous interruption.
- 2-thread intra-op configuration for optimal sustained M2 CPU/thermal efficiency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Sequence

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import soundfile as sf
import soxr

logger = logging.getLogger("micora.engine")

SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLES_PER_FRAME = 3840  # 48000 / 12.5 fps = 3840 samples (80 ms) per frame
FRAMES_PER_STREAM_CHUNK = 2  # 160 ms per streaming PCM chunk


def load_and_resample_audio(
    audio_path: Path | str,
    target_sample_rate: int = SAMPLE_RATE,
    target_channels: int = CHANNELS,
) -> np.ndarray:
    """Load audio file without PyTorch and convert to (1, channels, samples) float32."""
    resolved_path = Path(audio_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {resolved_path}")

    data, sample_rate = sf.read(str(resolved_path), dtype="float32", always_2d=True)
    if sample_rate != target_sample_rate:
        data = soxr.resample(data, in_rate=sample_rate, out_rate=target_sample_rate)

    current_channels = data.shape[1]
    if current_channels == target_channels:
        waveform = data.T
    elif current_channels == 1 and target_channels == 2:
        waveform = np.repeat(data.T, 2, axis=0)
    elif current_channels > 1 and target_channels == 1:
        waveform = np.mean(data.T, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unsupported channel conversion: {current_channels} -> {target_channels}")

    return np.expand_dims(waveform, axis=0).astype(np.float32, copy=False)


def clean_turkish_text(text: str) -> str:
    """Conservative text preparation for Turkish.

    Preserves Turkish phonemes (ı, I, i, İ, ğ, ş, ç, ö, ü) and punctuation prosody.
    Normalizes numbers and percentages to natural spoken words if not already normalized.
    Cleans up multiple whitespaces and control characters.
    """
    import re
    from worker.normalizer import normalize_text_for_tts

    if re.search(r"\d", text):
        return normalize_text_for_tts(text)
    return text.strip()


@dataclass
class CodecStreamingSession:
    codec_meta: dict
    session: ort.InferenceSession

    def __post_init__(self) -> None:
        self.transformer_specs = list(self.codec_meta.get("streaming_decode", {}).get("transformer_offsets", []))
        self.attention_specs = list(self.codec_meta.get("streaming_decode", {}).get("attention_caches", []))
        self.state_feeds: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self.state_feeds = {}
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = np.zeros(tuple(spec["shape"]), dtype=np.int32)
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = np.zeros(tuple(spec["offset_shape"]), dtype=np.int32)
            self.state_feeds[str(spec["cached_keys_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            self.state_feeds[str(spec["cached_values_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            self.state_feeds[str(spec["cached_positions_input_name"])] = np.full(tuple(spec["positions_shape"]), -1, dtype=np.int32)

    def decode_frames(self, frame_rows: list[list[int]]) -> np.ndarray | None:
        """Decode frames into (channels, samples) float32 audio."""
        if not frame_rows:
            return None
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        frame_count = len(frame_rows)
        audio_codes = np.zeros((1, frame_count, num_quantizers), dtype=np.int32)
        for f_idx, row in enumerate(frame_rows):
            for q_idx in range(num_quantizers):
                audio_codes[0, f_idx, q_idx] = int(row[q_idx] if q_idx < len(row) else 0)

        feeds = {
            "audio_codes": audio_codes,
            "audio_code_lengths": np.asarray([frame_count], dtype=np.int32),
            **self.state_feeds,
        }
        outputs = self.session.run(None, feeds)
        out_names = [o.name for o in self.session.get_outputs()]
        named = dict(zip(out_names, outputs, strict=True))

        # Update cache states
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = named[str(spec["output_name"])]
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = named[str(spec["offset_output_name"])]
            self.state_feeds[str(spec["cached_keys_input_name"])] = named[str(spec["cached_keys_output_name"])]
            self.state_feeds[str(spec["cached_values_input_name"])] = named[str(spec["cached_values_output_name"])]
            self.state_feeds[str(spec["cached_positions_input_name"])] = named[str(spec["cached_positions_output_name"])]

        audio = named["audio"]  # (1, channels, samples)
        length = int(named["audio_lengths"].reshape(-1)[0])
        if length <= 0:
            return None
        return audio[0, :, :length]


class MicoraTtsEngine:
    """Self-contained MOSS-TTS-Nano ONNX synthesis engine."""

    def __init__(
        self,
        model_dir: Path | str,
        thread_count: int = 2,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.thread_count = max(1, int(thread_count))

        manifest_path = self.model_dir / "browser_poc_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.tts_meta_path = self.model_dir / self.manifest["model_files"]["tts_meta"]
        self.tts_meta = json.loads(self.tts_meta_path.read_text(encoding="utf-8"))

        codec_meta_rel = self.manifest["model_files"]["codec_meta"]
        self.codec_meta_path = (self.model_dir / codec_meta_rel).resolve()
        self.codec_meta = json.loads(self.codec_meta_path.read_text(encoding="utf-8"))

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(str(self.model_dir / "tokenizer.model"))

        self.prefill_out_to_past = {
            out_name: out_name.replace("present_", "past_")
            for out_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }
        self.decode_out_to_past = {
            out_name: out_name.replace("present_", "past_")
            for out_name in self.tts_meta["onnx"]["decode_output_names"][1:]
        }
        self.rng = np.random.default_rng(1234)
        self.sessions = self._load_sessions()
        self.codec_streaming = CodecStreamingSession(
            codec_meta=self.codec_meta,
            session=self.sessions["codec_decode_step"],
        )
        self.voice_cache: dict[str, list[list[int]]] = {}
        self._warmup()

    def _create_session(self, path: Path) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.thread_count
        options.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def _load_sessions(self) -> dict[str, ort.InferenceSession]:
        tts_dir = self.tts_meta_path.parent
        codec_dir = self.codec_meta_path.parent
        sessions = {
            "prefill": self._create_session(tts_dir / self.tts_meta["files"]["prefill"]),
            "decode": self._create_session(tts_dir / self.tts_meta["files"]["decode_step"]),
            "codec_encode": self._create_session(codec_dir / self.codec_meta["files"]["encode"]),
            "codec_decode_step": self._create_session(codec_dir / self.codec_meta["files"]["decode_step"]),
        }
        if self.tts_meta["files"].get("local_fixed_sampled_frame"):
            sessions["local_fixed_sampled_frame"] = self._create_session(
                tts_dir / self.tts_meta["files"]["local_fixed_sampled_frame"]
            )
        return sessions

    def _warmup(self) -> None:
        """Run a minimal warmup iteration to prime all ONNX graph kernels."""
        t0 = time.perf_counter()
        voice = self.manifest["builtin_voices"][0]
        dummy_prompt = voice["prompt_audio_codes"][:10]
        text_tokens = self.tokenizer.encode("Isınma.", out_type=int)
        req = self._build_request_rows(dummy_prompt, text_tokens)

        prefill_ids = np.asarray([req["inputIds"]], dtype=np.int32)
        prefill_mask = np.asarray(req["attentionMask"], dtype=np.int32)
        outs = self.sessions["prefill"].run(
            None,
            {"input_ids": prefill_ids, "attention_mask": prefill_mask},
        )
        named = dict(zip([o.name for o in self.sessions["prefill"].get_outputs()], outs, strict=True))
        global_hidden = named["global_hidden"][:, -1, :].astype(np.float32, copy=False)
        asst_u = np.asarray([0.5], dtype=np.float32)
        audio_u = np.full((1, 16), 0.5, dtype=np.float32)
        self.sessions["local_fixed_sampled_frame"].run(
            None,
            {
                "global_hidden": global_hidden,
                "repetition_seen_mask": np.zeros((1, 16, 1024), dtype=np.int32),
                "assistant_random_u": asst_u,
                "audio_random_u": audio_u,
            },
        )
        empty_frame = [[0] * 16]
        self.codec_streaming.reset()
        self.codec_streaming.decode_frames(empty_frame)
        self.codec_streaming.reset()
        logger.info("Engine warmed up in %.3fs", time.perf_counter() - t0)

    def encode_reference(self, audio_path: Path | str) -> list[list[int]]:
        """Encode reference audio into prompt codes with SHA-256 caching."""
        path = Path(audio_path).expanduser().resolve()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if file_hash in self.voice_cache:
            return self.voice_cache[file_hash]

        waveform = load_and_resample_audio(path, target_sample_rate=SAMPLE_RATE, target_channels=CHANNELS)
        waveform_len = waveform.shape[-1]
        outputs = self.sessions["codec_encode"].run(
            None,
            {"waveform": waveform, "input_lengths": np.asarray([waveform_len], dtype=np.int32)},
        )
        named = dict(zip([o.name for o in self.sessions["codec_encode"].get_outputs()], outputs, strict=True))
        codes = np.asarray(named["audio_codes"], dtype=np.int32)
        code_len = int(named["audio_code_lengths"].reshape(-1)[0])
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        prompt_codes = [[int(codes[0, f, q]) for q in range(num_quantizers)] for f in range(code_len)]
        self.voice_cache[file_hash] = prompt_codes
        return prompt_codes

    def get_builtin_voice(self, name: str = "Ava") -> list[list[int]]:
        voice = next((v for v in self.manifest["builtin_voices"] if v["voice"].lower() == name.lower()), None)
        if voice is None:
            raise ValueError(f"Unknown builtin voice: {name}")
        return list(voice["prompt_audio_codes"])

    def _build_request_rows(self, prompt_audio_codes: list[list[int]], text_token_ids: list[int]) -> dict[str, list[list[int]]]:
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        row_width = n_vq + 1
        pad_token = int(self.manifest["tts_config"]["audio_pad_token_id"])
        user_slot = int(self.manifest["tts_config"]["audio_user_slot_token_id"])

        def text_rows(token_ids: Sequence[int]) -> list[list[int]]:
            return [[int(t)] + [pad_token] * n_vq for t in token_ids]

        prefix_ids = [
            *self.manifest["prompt_templates"]["user_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        suffix_ids = [
            int(self.manifest["tts_config"]["audio_end_token_id"]),
            *self.manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
            *text_token_ids,
            *self.manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]

        audio_rows = []
        for code_row in prompt_audio_codes:
            r = [user_slot] + [pad_token] * n_vq
            for idx in range(min(len(code_row), n_vq)):
                r[idx + 1] = int(code_row[idx])
            audio_rows.append(r)

        all_rows = text_rows(prefix_ids) + audio_rows + text_rows(suffix_ids)
        return {
            "inputIds": all_rows,
            "attentionMask": [[1 for _ in all_rows]],
        }

    def synthesize_stream(
        self,
        text: str,
        prompt_codes: list[list[int]],
        max_new_frames: int = 375,
        is_cancelled: Callable[[], bool] | None = None,
        frames_per_chunk: int = FRAMES_PER_STREAM_CHUNK,
    ) -> Generator[np.ndarray, None, None]:
        """Incremental streaming generator yielding (samples, 2) float32 PCM chunks."""
        cleaned_text = clean_turkish_text(text)
        if not cleaned_text:
            return

        text_tokens = self.tokenizer.encode(cleaned_text, out_type=int)
        req = self._build_request_rows(prompt_codes, text_tokens)

        # Prefill stage
        prefill_ids = np.asarray([req["inputIds"]], dtype=np.int32)
        prefill_mask = np.asarray(req["attentionMask"], dtype=np.int32)
        outs = self.sessions["prefill"].run(
            None,
            {"input_ids": prefill_ids, "attention_mask": prefill_mask},
        )
        named = dict(zip([o.name for o in self.sessions["prefill"].get_outputs()], outs, strict=True))
        global_hidden = named["global_hidden"][:, -1, :].astype(np.float32, copy=False)
        past_valid_length = len(req["inputIds"])
        past_by_name = {
            self.prefill_out_to_past[out_name]: named[out_name]
            for out_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }

        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        asst_slot = int(self.manifest["tts_config"]["audio_assistant_slot_token_id"])
        pad_token = int(self.manifest["tts_config"]["audio_pad_token_id"])

        previous_token_sets = [set() for _ in range(n_vq)]
        self.codec_streaming.reset()
        pending_frames: list[list[int]] = []

        # Pre-allocate reusable buffers across autoregressive steps
        repetition_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        asst_u = np.zeros((1,), dtype=np.float32)
        audio_u = np.zeros((1, n_vq), dtype=np.float32)
        next_row = np.full((1, 1, row_width), pad_token, dtype=np.int32)
        next_row[0, 0, 0] = asst_slot

        try:
            for step_index in range(max_new_frames):
                if is_cancelled is not None and is_cancelled():
                    logger.info("Synthesis cancelled at step %d", step_index)
                    return

                # Local fixed sampled frame
                repetition_mask.fill(0)
                for ch_idx, token_set in enumerate(previous_token_sets):
                    for tok in token_set:
                        if 0 <= tok < audio_codebook_size:
                            repetition_mask[0, ch_idx, tok] = 1

                asst_u[0] = min(0.99999994, max(0.0, float(self.rng.random())))
                for q_idx in range(n_vq):
                    audio_u[0, q_idx] = min(0.99999994, max(0.0, float(self.rng.random())))

                local_outs = self.sessions["local_fixed_sampled_frame"].run(
                    None,
                    {
                        "global_hidden": global_hidden,
                        "repetition_seen_mask": repetition_mask,
                        "assistant_random_u": asst_u,
                        "audio_random_u": audio_u,
                    },
                )
                local_named = dict(zip([o.name for o in self.sessions["local_fixed_sampled_frame"].get_outputs()], local_outs, strict=True))
                should_continue = bool(int(np.asarray(local_named["should_continue"]).reshape(-1)[0]))
                if not should_continue:
                    break

                frame = np.asarray(local_named["frame_token_ids"]).reshape(-1).astype(np.int32).tolist()
                for ch_idx, tok in enumerate(frame):
                    previous_token_sets[ch_idx].add(tok)

                pending_frames.append(frame)
                if len(pending_frames) >= frames_per_chunk:
                    audio_chunk = self.codec_streaming.decode_frames(pending_frames)
                    pending_frames.clear()
                    if audio_chunk is not None and audio_chunk.shape[-1] > 0:
                        yield audio_chunk.T  # (samples, channels)

                # Decode step for next autoregressive token
                next_row[0, 0, 1:] = pad_token
                for idx, tok in enumerate(frame):
                    next_row[0, 0, idx + 1] = int(tok)

                decode_feeds = {
                    "input_ids": next_row,
                    "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
                }
                for in_name in self.tts_meta["onnx"]["decode_input_names"][2:]:
                    decode_feeds[in_name] = past_by_name[in_name]

                decode_outs = self.sessions["decode"].run(None, decode_feeds)
                decode_named = dict(zip([o.name for o in self.sessions["decode"].get_outputs()], decode_outs, strict=True))
                global_hidden = decode_named["global_hidden"][:, -1, :].astype(np.float32, copy=False)
                past_valid_length += 1
                past_by_name = {
                    self.decode_out_to_past[out_name]: decode_named[out_name]
                    for out_name in self.tts_meta["onnx"]["decode_output_names"][1:]
                }

            # Emit remaining pending frames
            if pending_frames and (is_cancelled is None or not is_cancelled()):
                audio_chunk = self.codec_streaming.decode_frames(pending_frames)
                pending_frames.clear()
                if audio_chunk is not None and audio_chunk.shape[-1] > 0:
                    yield audio_chunk.T
        finally:
            self.codec_streaming.reset()
