"""Integration test for Micora Worker IPC over Unix Domain Socket."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.protocol import (
    PROTOCOL_VERSION,
    PacketType,
    decode_pcm_payload,
    encode_control_packet,
    read_packet,
)

SOCKET_PATH = Path("/tmp/micora_test.sock")
MODEL_DIR = Path("models/MOSS-TTS-Nano-100M-ONNX").resolve()
REF_AUDIO = (
    Path("models/reference_speech.wav").resolve()
    if Path("models/reference_speech.wav").is_file()
    else Path(".reference/MOSS-TTS-Nano/assets/audio/en_2.wav").resolve()
)


def run_ipc_test():
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    print("Starting worker process in background...")
    proc = subprocess.Popen([
        sys.executable,
        "-m",
        "worker.server",
        "--socket",
        str(SOCKET_PATH),
        "--model-dir",
        str(MODEL_DIR),
        "--threads",
        "2",
    ])

    try:
        # Wait for socket to appear
        for _ in range(50):
            if SOCKET_PATH.exists():
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("Worker socket failed to appear within 10s")

        print("Connecting to worker Unix domain socket...")
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(SOCKET_PATH))

        # 1. Test Handshake
        print("\n--- Test 1: Handshake ---")
        client.sendall(encode_control_packet({"type": "init", "protocol_version": PROTOCOL_VERSION}))
        ptype, payload = read_packet(client)
        assert ptype == PacketType.CONTROL_JSON, f"Expected CONTROL_JSON, got {ptype}"
        msg = json.loads(payload.decode("utf-8"))
        print("Received handshake:", msg)
        assert msg["type"] == "ready"
        assert msg["protocol_version"] == PROTOCOL_VERSION
        assert msg["sample_rate"] == 48000
        assert msg["channels"] == 2

        # 2. Test Voice Preparation & Caching
        print("\n--- Test 2: Voice Preparation & Caching ---")
        client.sendall(encode_control_packet({
            "type": "prepare_voice",
            "voice_id": "test_cloned",
            "audio_path": str(REF_AUDIO),
        }))
        ptype, payload = read_packet(client)
        msg = json.loads(payload.decode("utf-8"))
        print("Voice prep response:", msg)
        assert msg["type"] == "voice_ready"
        assert msg["voice_id"] == "test_cloned"
        assert msg["frame_count"] > 0

        # 3. Test Synthesis & Streaming PCM Delivery
        print("\n--- Test 3: Synthesis & Streaming PCM Delivery ---")
        req_id = "req_ipc_01"
        client.sendall(encode_control_packet({
            "type": "synthesize",
            "request_id": req_id,
            "voice_id": "test_cloned",
            "text": "Unix soket üzerinden gerçek zamanlı ses akışı başarıyla çalışıyor.",
        }))

        pcm_chunks = []
        eos_received = False
        t_start = time.perf_counter()
        ttfa = None

        while True:
            packet = read_packet(client)
            assert packet is not None, "Connection dropped"
            ptype, payload = packet

            if ptype == PacketType.AUDIO_PCM:
                chunk_req_id, pcm_bytes = decode_pcm_payload(payload)
                assert chunk_req_id == req_id
                if ttfa is None:
                    ttfa = time.perf_counter() - t_start
                # Reconstruct numpy float32 (samples, 2)
                chunk_floats = np.frombuffer(pcm_bytes, dtype=np.float32).reshape(-1, 2)
                pcm_chunks.append(chunk_floats)
            elif ptype == PacketType.AUDIO_EOS:
                chunk_req_id, _ = decode_pcm_payload(payload)
                assert chunk_req_id == req_id
                eos_received = True
                break
            else:
                raise RuntimeError(f"Unexpected packet type during synthesis: {ptype}")

        total_time = time.perf_counter() - t_start
        all_audio = np.concatenate(pcm_chunks, axis=0)
        dur = all_audio.shape[0] / 48000
        rtf = total_time / dur
        print(f"Synthesis complete! TTFA: {ttfa:.3f}s, Chunks: {len(pcm_chunks)}, Dur: {dur:.2f}s, Time: {total_time:.3f}s, RTF: {rtf:.3f}")
        assert eos_received
        assert dur > 1.0
        assert rtf < 0.6
        sf.write("test_ipc_stream.wav", all_audio, 48000)

        # 4. Test Cancellation Mid-Stream
        print("\n--- Test 4: Mid-Stream Cancellation ---")
        cancel_req_id = "req_cancel_02"
        client.sendall(encode_control_packet({
            "type": "synthesize",
            "request_id": cancel_req_id,
            "voice_id": "test_cloned",
            "text": "Bu çok uzun bir cümle olacak ve sentez sırasında iptal edilecek, böylece arka plandaki model anında duracak.",
        }))

        # Read first chunk, then cancel
        first_packet = read_packet(client)
        assert first_packet is not None and first_packet[0] == PacketType.AUDIO_PCM
        print("Received first chunk of request to cancel. Sending cancel...")
        t_cancel_sent = time.perf_counter()
        client.sendall(encode_control_packet({"type": "cancel", "request_id": cancel_req_id}))

        # Read until cancellation ack or drain
        cancelled_ack = False
        chunks_after_cancel = 0
        while True:
            packet = read_packet(client)
            if packet is None:
                break
            ptype, payload = packet
            if ptype == PacketType.CONTROL_JSON:
                msg = json.loads(payload.decode("utf-8"))
                if msg.get("type") == "cancelled" and msg.get("request_id") == cancel_req_id:
                    cancelled_ack = True
                    break
            elif ptype == PacketType.AUDIO_PCM:
                chunks_after_cancel += 1
                if chunks_after_cancel > 3:
                    raise RuntimeError("Too many chunks arrived after cancellation!")

        cancel_duration = time.perf_counter() - t_cancel_sent
        print(f"Cancellation acknowledged in {cancel_duration:.3f}s (chunks after cancel: {chunks_after_cancel})")
        assert cancelled_ack
        assert cancel_duration < 0.35  # Must cancel within 350 ms

        # 5. Test Subsequent Synthesis After Cancellation (Clean State)
        print("\n--- Test 5: Synthesis After Cancellation ---")
        req3_id = "req_after_cancel_03"
        client.sendall(encode_control_packet({
            "type": "synthesize",
            "request_id": req3_id,
            "voice_id": "test_cloned",
            "text": "İptal işleminden sonra sistem temiz çalışıyor.",
        }))

        pcm_chunks3 = []
        eos3_received = False
        while True:
            packet = read_packet(client)
            assert packet is not None
            ptype, payload = packet
            if ptype == PacketType.AUDIO_PCM:
                chunk_req_id, pcm_bytes = decode_pcm_payload(payload)
                assert chunk_req_id == req3_id
                pcm_chunks3.append(np.frombuffer(pcm_bytes, dtype=np.float32).reshape(-1, 2))
            elif ptype == PacketType.AUDIO_EOS:
                chunk_req_id, _ = decode_pcm_payload(payload)
                assert chunk_req_id == req3_id
                eos3_received = True
                break

        all_audio3 = np.concatenate(pcm_chunks3, axis=0)
        dur3 = all_audio3.shape[0] / 48000
        print(f"Synthesis after cancellation succeeded: {dur3:.2f}s audio in {len(pcm_chunks3)} chunks")
        assert eos3_received
        assert dur3 > 1.0

        # 6. Test Shutdown
        print("\n--- Test 6: Clean Shutdown ---")
        client.sendall(encode_control_packet({"type": "shutdown"}))
        ptype, payload = read_packet(client)
        msg = json.loads(payload.decode("utf-8"))
        print("Shutdown response:", msg)
        assert msg["type"] == "shutting_down"
        client.close()

        proc.wait(timeout=5)
        print("\nALL WORKER IPC TESTS PASSED!")

    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=2)
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()


if __name__ == "__main__":
    run_ipc_test()
