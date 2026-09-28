"""Integration test for Chatterbox backend and dynamic backend switching in Micora worker."""

import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

from worker.protocol import (
    PROTOCOL_VERSION,
    PacketType,
    encode_control_packet,
    read_packet,
    decode_pcm_payload,
)


def run_integration_test():
    repo_root = Path(__file__).resolve().parent.parent
    socket_path = Path(f"/tmp/micora_test_chatterbox_{os.getpid()}.sock")
    if socket_path.exists():
        socket_path.unlink()

    ref_audio = repo_root / "tests" / "fixtures" / "reference_speech.wav"
    if not ref_audio.exists():
        ref_audio = repo_root / ".reference" / "MOSS-TTS-Nano" / "assets" / "audio" / "en_2.wav"

    print(f"=== Starting Worker Server with --backend chatterbox ===")
    worker_proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "worker.server",
            "--socket", str(socket_path),
            "--backend", "chatterbox",
        ],
        cwd=str(repo_root),
    )

    try:
        # Wait for socket
        t_start = time.time()
        connected = False
        client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        while time.time() - t_start < 60:
            if socket_path.exists():
                try:
                    client_sock.connect(str(socket_path))
                    connected = True
                    break
                except OSError:
                    pass
            time.sleep(0.1)

        assert connected, "Failed to connect to worker socket within 60s"
        print("✓ Connected to worker socket")

        # 1. Handshake
        client_sock.sendall(encode_control_packet({"type": "init", "protocol_version": PROTOCOL_VERSION}))
        ptype, payload = read_packet(client_sock)
        assert ptype == PacketType.CONTROL_JSON
        import json
        ready_msg = json.loads(payload.decode("utf-8"))
        print("Handshake response:", ready_msg)
        assert ready_msg.get("type") == "ready"
        assert ready_msg.get("sample_rate") == 48000
        assert ready_msg.get("channels") == 2

        # 2. Query Capabilities
        client_sock.sendall(encode_control_packet({"type": "get_capabilities"}))
        ptype, payload = read_packet(client_sock)
        assert ptype == PacketType.CONTROL_JSON
        cap_msg = json.loads(payload.decode("utf-8"))
        print("Capabilities response:", cap_msg)
        assert cap_msg.get("backend") == "chatterbox"
        assert cap_msg.get("is_streaming") is False
        assert cap_msg.get("sample_rate") == 48000
        assert cap_msg.get("channels") == 2
        print("✓ Capabilities verified: Chatterbox is correctly reported as non-streaming, 48 kHz stereo canonical")

        # 3. Prepare Voice
        print(f"Preparing voice from {ref_audio}...")
        client_sock.sendall(encode_control_packet({
            "type": "prepare_voice",
            "voice_id": "test_voice",
            "audio_path": str(ref_audio),
        }))
        ptype, payload = read_packet(client_sock)
        assert ptype == PacketType.CONTROL_JSON
        v_ready = json.loads(payload.decode("utf-8"))
        print("Voice ready response:", v_ready)
        assert v_ready.get("type") == "voice_ready"
        print("✓ Chatterbox voice preparation verified")

        # 4. Synthesize Turkish speech
        test_text = "Selam, Chatterbox backend başarıyla çalışıyor."
        req_id = f"test_req_{int(time.time()*1000)}"
        print(f"Requesting synthesis: '{test_text}' [req={req_id}]")
        t0_synth = time.perf_counter()
        client_sock.sendall(encode_control_packet({
            "type": "synthesize",
            "request_id": req_id,
            "voice_id": "test_voice",
            "text": test_text,
        }))

        chunks_received = 0
        total_bytes = 0
        while True:
            ptype, payload = read_packet(client_sock)
            if ptype == PacketType.AUDIO_PCM:
                r_id, pcm_bytes = decode_pcm_payload(payload)
                assert r_id == req_id
                chunks_received += 1
                total_bytes += len(pcm_bytes)
            elif ptype == PacketType.AUDIO_EOS:
                r_id, _ = decode_pcm_payload(payload)
                assert r_id == req_id
                print(f"✓ EOS received for {r_id}")
                break
            elif ptype == PacketType.ERROR:
                err = json.loads(payload.decode("utf-8"))
                raise RuntimeError(f"Synthesis error: {err}")
            else:
                print(f"Unexpected packet type: {ptype}")

        elapsed = time.perf_counter() - t0_synth
        total_samples = total_bytes // (4 * 2) # float32 stereo
        audio_dur = total_samples / 48000.0
        print(f"✓ Chatterbox synthesis complete: {chunks_received} chunks, {audio_dur:.2f}s audio generated in {elapsed:.2f}s (RTF: {elapsed/audio_dur:.3f})")

        # 5. Dynamic Backend Switching: Switch to MOSS
        print("\n--- Testing Dynamic Backend Switching: Chatterbox -> MOSS ---")
        client_sock.sendall(encode_control_packet({"type": "set_backend", "backend": "moss"}))
        ptype, payload = read_packet(client_sock)
        assert ptype == PacketType.CONTROL_JSON
        switch_msg = json.loads(payload.decode("utf-8"))
        print("Switch backend response:", switch_msg)
        assert switch_msg.get("backend") == "moss"
        assert switch_msg.get("is_streaming") is True
        print("✓ Switched backend to MOSS (verified is_streaming=True)")

        # 6. Prepare Voice on MOSS and Synthesize
        client_sock.sendall(encode_control_packet({
            "type": "prepare_voice",
            "voice_id": "moss_voice",
            "audio_path": str(ref_audio),
        }))
        ptype, payload = read_packet(client_sock)
        v_ready = json.loads(payload.decode("utf-8"))
        assert v_ready.get("type") == "voice_ready"

        moss_req_id = f"moss_req_{int(time.time()*1000)}"
        client_sock.sendall(encode_control_packet({
            "type": "synthesize",
            "request_id": moss_req_id,
            "voice_id": "moss_voice",
            "text": "MOSS arka planına geçildi ve sentez yapıldı.",
        }))

        moss_chunks = 0
        while True:
            ptype, payload = read_packet(client_sock)
            if ptype == PacketType.AUDIO_PCM:
                moss_chunks += 1
            elif ptype == PacketType.AUDIO_EOS:
                break

        print(f"✓ MOSS synthesis complete: {moss_chunks} chunks received")

        # 7. Shutdown
        client_sock.sendall(encode_control_packet({"type": "shutdown"}))
        client_sock.close()
        print("✓ Clean shutdown complete")

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            worker_proc.wait(timeout=5)
        if socket_path.exists():
            socket_path.unlink()

    print("\nALL WORKER BACKEND & SWITCHING TESTS PASSED!")


if __name__ == "__main__":
    run_integration_test()
