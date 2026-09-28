"""Micora Persistent Local Inference Worker.

Runs as a background daemon process communicating over a Unix domain socket.
Maintains ONNX models in memory across requests.
Streams binary PCM directly to Swift application.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import ctypes
from pathlib import Path

import numpy as np

# On macOS, prevent Python from registering as a GUI application or showing in the Dock
if sys.platform == "darwin":
    try:
        _as_lib = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        class _PSN(ctypes.Structure):
            _fields_ = [("highLongOfPSN", ctypes.c_uint32), ("lowLongOfPSN", ctypes.c_uint32)]
        _psn = _PSN(0, 2)  # kCurrentProcess
        _as_lib.TransformProcessType(ctypes.byref(_psn), 2)  # kProcessTransformToBackgroundApplication
    except Exception:
        pass

from worker.backends import (
    CANONICAL_CHANNELS,
    CANONICAL_SAMPLE_RATE,
    ChatterboxBackend,
    MossBackend,
    SynthesisBackend,
)
from worker.normalizer import normalize_text_for_tts
from worker.engine import SAMPLE_RATE, CHANNELS
from worker.protocol import (
    PROTOCOL_VERSION,
    PacketType,
    encode_control_packet,
    encode_error_packet,
    encode_eos_packet,
    encode_pcm_packet,
    read_packet,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [micora-worker] %(message)s",
)
logger = logging.getLogger("micora.worker")


class MicoraWorkerServer:
    def __init__(
        self,
        socket_path: str | Path,
        model_dir: str | Path,
        thread_count: int = 2,
        backend_type: str = "moss",
    ) -> None:
        self.socket_path = Path(socket_path).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.thread_count = thread_count
        self.backend_type = backend_type.lower()
        self.state = "starting"

        self.server_sock: socket.socket | None = None
        self.active_conn: socket.socket | None = None
        self.running = True
        self.active_request_id: str | None = None
        self.cancel_event = threading.Event()
        self.state_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.flow_lock = threading.Lock()
        self.flow_cv = threading.Condition(self.flow_lock)
        self.flow_credit = 0
        self.synthesis_thread: threading.Thread | None = None
        self.voice_cache: dict[str, Any] = {}

        logger.info("Initializing %s backend...", self.backend_type)
        self.state = "loading"
        if self.backend_type == "chatterbox":
            self.backend: SynthesisBackend = ChatterboxBackend()
        else:
            self.backend = MossBackend(model_dir=self.model_dir, thread_count=self.thread_count)

        self.backend.load()
        self.state = "ready"
        logger.info("Worker ready. State: %s (backend: %s)", self.state, self.backend.name)

    def _send(self, conn: socket.socket, data: bytes) -> None:
        with self.send_lock:
            conn.sendall(data)

    def start(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()

        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(str(self.socket_path))
        self.server_sock.listen(1)
        logger.info("Worker listening on Unix domain socket: %s", self.socket_path)

        def signal_handler(_sig, _frame):
            self.running = False
            self.cancel_event.set()
            with self.flow_lock:
                self.flow_cv.notify_all()
            if self.active_conn:
                try:
                    self.active_conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    self.active_conn.close()
                except OSError:
                    pass
            if self.server_sock:
                try:
                    self.server_sock.close()
                except OSError:
                    pass
            os._exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Watchdog thread: automatically terminate if parent process exits
        def _parent_watchdog():
            parent_pid = os.getppid()
            while self.running:
                time.sleep(1.0)
                if os.getppid() != parent_pid:
                    self.running = False
                    if self.server_sock:
                        try:
                            self.server_sock.close()
                        except OSError:
                            pass
                    os._exit(0)

        watchdog = threading.Thread(target=_parent_watchdog, daemon=True)
        watchdog.start()

        while self.running:
            try:
                conn, _ = self.server_sock.accept()
                logger.info("Client connected.")
                self._handle_client(conn)
            except OSError:
                break

        self.cleanup()

    def _handle_client(self, conn: socket.socket) -> None:
        self.active_conn = conn
        try:
            while self.running:
                logger.debug("Server: waiting for packet...")
                packet = read_packet(conn)
                if packet is None:
                    logger.info("Server: client disconnected.")
                    self.running = False
                    break

                ptype, payload = packet
                logger.debug("Server: received packet type=%s, len=%d", ptype, len(payload))
                if ptype != PacketType.CONTROL_JSON:
                    logger.warning("Ignoring unexpected packet type: %s", ptype)
                    continue

                try:
                    cmd = json.loads(payload.decode("utf-8"))
                except Exception as exc:
                    logger.error("Invalid JSON payload: %s", exc)
                    self._send(conn, encode_control_packet({
                        "type": "error",
                        "error": f"Invalid JSON payload: {exc}",
                    }))
                    continue

                action = cmd.get("type")
                corr_id = cmd.get("correlation_id")

                if action == "init":
                    client_ver = cmd.get("protocol_version", 0)
                    if client_ver != PROTOCOL_VERSION:
                        err = f"Protocol version mismatch: worker={PROTOCOL_VERSION}, client={client_ver}"
                        logger.error(err)
                        self._send(conn, encode_control_packet({"type": "error", "error": err, "correlation_id": corr_id}))
                        break
                    self._send(
                        conn,
                        encode_control_packet({
                            "type": "ready",
                            "protocol_version": PROTOCOL_VERSION,
                            "sample_rate": SAMPLE_RATE,
                            "channels": CHANNELS,
                            "state": self.state,
                            "correlation_id": corr_id,
                        })
                    )

                elif action == "prepare_voice":
                    voice_id = cmd.get("voice_id", "default")
                    audio_path = cmd.get("audio_path")
                    preset = cmd.get("preset")

                    try:
                        voice_state = self.backend.prepare_voice(audio_path=audio_path, preset=preset)
                        self.voice_cache[voice_id] = voice_state
                        frame_count = len(voice_state) if isinstance(voice_state, (list, tuple)) else 1
                        self._send(conn, encode_control_packet({
                            "type": "voice_ready",
                            "voice_id": voice_id,
                            "frame_count": frame_count,
                            "correlation_id": corr_id,
                        }))
                    except Exception as exc:
                        logger.error("Failed to prepare voice %s: %s", voice_id, exc)
                        self._send(conn, encode_control_packet({
                            "type": "voice_error",
                            "voice_id": voice_id,
                            "error": str(exc),
                            "correlation_id": corr_id,
                        }))

                elif action == "ack_frames":
                    req_id = cmd.get("request_id")
                    frames = int(cmd.get("frames", 0))
                    with self.flow_lock:
                        if self.active_request_id == req_id:
                            self.flow_credit += frames
                            self.flow_cv.notify_all()

                elif action == "cancel":
                    req_id = cmd.get("request_id")
                    logger.info("Cancellation requested for request: %s", req_id)
                    with self.state_lock:
                        if self.active_request_id == req_id:
                            self.cancel_event.set()
                            with self.flow_lock:
                                self.flow_cv.notify_all()
                            self.state = "cancelling"
                    self._send(conn, encode_control_packet({
                        "type": "cancelled",
                        "request_id": req_id,
                        "correlation_id": corr_id,
                    }))

                elif action == "synthesize":
                    req_id = cmd.get("request_id", f"req_{int(time.time()*1000)}")
                    text = cmd.get("text", "")
                    voice_id = cmd.get("voice_id", "default")
                    flow_control = bool(cmd.get("flow_control", False))
                    initial_credit = int(cmd.get("initial_credit_frames", 96000))

                    # If previous synthesis is still running, cancel it first
                    with self.state_lock:
                        if self.synthesis_thread is not None and self.synthesis_thread.is_alive():
                            self.cancel_event.set()
                            with self.flow_lock:
                                self.flow_cv.notify_all()
                    if self.synthesis_thread is not None and self.synthesis_thread.is_alive():
                        self.synthesis_thread.join(timeout=1.0)

                    self.cancel_event.clear()
                    with self.flow_lock:
                        self.flow_credit = initial_credit
                    self.active_request_id = req_id
                    self.state = "synthesizing"

                    self.synthesis_thread = threading.Thread(
                        target=self._run_synthesis,
                        args=(conn, req_id, text, voice_id, flow_control),
                        daemon=True,
                    )
                    self.synthesis_thread.start()

                elif action == "get_capabilities":
                    self._send(
                        conn,
                        encode_control_packet({
                            "type": "capabilities",
                            "backend": self.backend.id,
                            "name": self.backend.name,
                            "is_streaming": self.backend.is_streaming,
                            "sample_rate": CANONICAL_SAMPLE_RATE,
                            "channels": CANONICAL_CHANNELS,
                            "correlation_id": corr_id,
                        }),
                    )

                elif action in ("set_backend", "switch_backend"):
                    target = str(cmd.get("backend", "")).lower().strip()
                    logger.info("Request to switch backend to '%s' (current: '%s')", target, self.backend.id)
                    if target and target != self.backend.id:
                        with self.state_lock:
                            if self.synthesis_thread is not None and self.synthesis_thread.is_alive():
                                self.cancel_event.set()
                        if self.synthesis_thread is not None and self.synthesis_thread.is_alive():
                            self.synthesis_thread.join(timeout=2.0)

                        self.state = "loading"
                        self.backend.unload()
                        self.voice_cache.clear()

                        if target == "chatterbox":
                            self.backend = ChatterboxBackend()
                        else:
                            self.backend = MossBackend(model_dir=self.model_dir, thread_count=self.thread_count)

                        self.backend.load()
                        self.backend_type = self.backend.id
                        self.state = "ready"
                        logger.info("Switched active backend to %s", self.backend.name)

                    self._send(
                        conn,
                        encode_control_packet({
                            "type": "backend_switched",
                            "backend": self.backend.id,
                            "name": self.backend.name,
                            "is_streaming": self.backend.is_streaming,
                            "sample_rate": CANONICAL_SAMPLE_RATE,
                            "channels": CANONICAL_CHANNELS,
                            "correlation_id": corr_id,
                        }),
                    )

                elif action == "scan_dataset":
                    folder_path = cmd.get("folder_path", "")
                    max_candidates = int(cmd.get("max_candidates", 5))
                    try:
                        from worker.analyzer import DatasetScanner
                        scanner = DatasetScanner(folder_path)
                        result = scanner.scan(max_candidates=max_candidates)
                        self._send(
                            conn,
                            encode_control_packet({
                                "type": "dataset_scanned",
                                "correlation_id": corr_id,
                                **result,
                            }),
                        )
                    except Exception as exc:
                        logger.error("Failed to scan dataset %s: %s", folder_path, exc)
                        self._send(
                            conn,
                            encode_control_packet({
                                "type": "dataset_scan_error",
                                "error": str(exc),
                                "correlation_id": corr_id,
                            }),
                        )

                elif action == "import_hf":
                    repo_id = cmd.get("repo_id", "")
                    max_samples = int(cmd.get("max_samples", 30))
                    max_candidates = int(cmd.get("max_candidates", 5))
                    try:
                        from worker.hf_importer import HuggingFaceVoiceImporter
                        importer = HuggingFaceVoiceImporter()
                        result = importer.import_voice(
                            repo_id,
                            max_samples=max_samples,
                            max_candidates=max_candidates,
                        )
                        self._send(
                            conn,
                            encode_control_packet({
                                "type": "hf_imported",
                                "correlation_id": corr_id,
                                **result,
                            }),
                        )
                    except Exception as exc:
                        logger.error("Failed to import Hugging Face voice %s: %s", repo_id, exc)
                        self._send(
                            conn,
                            encode_control_packet({
                                "type": "hf_import_error",
                                "error": str(exc),
                                "correlation_id": corr_id,
                            }),
                        )

                elif action == "shutdown":
                    logger.info("Shutdown command received.")
                    self.cancel_event.set()
                    if self.synthesis_thread and self.synthesis_thread.is_alive():
                        self.synthesis_thread.join(timeout=1.0)
                    self._send(conn, encode_control_packet({"type": "shutting_down", "correlation_id": corr_id}))
                    self.running = False
                    break

                else:
                    err_msg = f"Unknown command type: {action}"
                    logger.warning(err_msg)
                    self._send(conn, encode_control_packet({
                        "type": "error",
                        "error": err_msg,
                        "correlation_id": corr_id,
                    }))

        finally:
            self.cancel_event.set()
            with self.flow_lock:
                self.flow_cv.notify_all()
            if self.synthesis_thread and self.synthesis_thread.is_alive():
                self.synthesis_thread.join(timeout=1.0)
            self.active_conn = None
            conn.close()

    def _run_synthesis(
        self,
        conn: socket.socket,
        request_id: str,
        text: str,
        voice_id: str,
        flow_control: bool = False,
    ) -> None:
        text = normalize_text_for_tts(text)
        logger.info("Starting synthesis [req=%s]: \"%s\"", request_id, text)
        t_start = time.perf_counter()
        first_chunk_sent = False

        try:
            # Resolve voice state
            voice_state = self.voice_cache.get(voice_id)

            chunk_count = 0
            total_samples = 0

            for chunk in self.backend.synthesize(
                text=text,
                voice_state=voice_state,
                is_cancelled=self.cancel_event.is_set,
            ):
                if self.cancel_event.is_set():
                    logger.info("Aborting PCM delivery for cancelled request: %s", request_id)
                    break

                chunk_frames = chunk.shape[0]
                if flow_control:
                    with self.flow_cv:
                        while self.flow_credit < chunk_frames and not self.cancel_event.is_set() and self.running:
                            self.flow_cv.wait(timeout=0.05)
                        if self.cancel_event.is_set() or not self.running:
                            logger.info("Aborting PCM delivery during backpressure wait [req=%s]", request_id)
                            break
                        self.flow_credit -= chunk_frames

                # Ensure float32 interleaved bytes
                pcm_bytes = chunk.astype(np.float32, copy=False).tobytes()
                self._send(conn, encode_pcm_packet(request_id, pcm_bytes))
                chunk_count += 1
                total_samples += chunk.shape[0]

                if not first_chunk_sent:
                    first_chunk_sent = True
                    ttfa = time.perf_counter() - t_start
                    logger.info("First PCM chunk sent [req=%s] at +%.3fs", request_id, ttfa)

            if self.cancel_event.is_set():
                logger.info("Request %s was cancelled. Skipping EOS.", request_id)
                with self.state_lock:
                    if self.active_request_id == request_id:
                        self.active_request_id = None
                        self.state = "ready"
                return

            # Send EOS
            self._send(conn, encode_eos_packet(request_id))
            elapsed = time.perf_counter() - t_start
            dur = total_samples / SAMPLE_RATE
            rtf = elapsed / dur if dur > 0 else 0
            logger.info(
                "Synthesis complete [req=%s]: %d chunks (%.2fs audio) in %.3fs (RTF: %.3f)",
                request_id,
                chunk_count,
                dur,
                elapsed,
                rtf,
            )

        except Exception as exc:
            if not self.cancel_event.is_set():
                logger.exception("Synthesis error [req=%s]: %s", request_id, exc)
                try:
                    self._send(conn, encode_error_packet(request_id, str(exc)))
                except OSError:
                    pass
        finally:
            with self.flow_lock:
                self.flow_cv.notify_all()
            with self.state_lock:
                if self.active_request_id == request_id:
                    self.active_request_id = None
                    self.state = "ready"

    def cleanup(self) -> None:
        logger.info("Cleaning up worker...")
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Micora Persistent Inference Worker")
    parser.add_argument("--socket", "-s", required=True, help="Path to Unix domain socket")
    parser.add_argument("--model-dir", "-m", default="models/MOSS-TTS-Nano-100M-ONNX", help="Model path")
    parser.add_argument("--threads", "-t", type=int, default=2, help="Intra-op thread count")
    parser.add_argument("--backend", "-b", default="moss", choices=["moss", "chatterbox"], help="Synthesis engine backend")
    args = parser.parse_args()

    server = MicoraWorkerServer(
        socket_path=args.socket,
        model_dir=args.model_dir,
        thread_count=args.threads,
        backend_type=args.backend,
    )
    server.start()


if __name__ == "__main__":
    main()
