"""Micora IPC Protocol Definitions.

Transport: Unix Domain Socket
Framing:
    Byte 0:       Packet Type (1 byte)
                  0x01 = CONTROL_JSON
                  0x02 = AUDIO_PCM
                  0x03 = AUDIO_EOS
                  0x04 = ERROR
    Bytes 1..4:   Payload Length (4 bytes, Big-Endian uint32)
    Bytes 5..end: Payload

AUDIO_PCM Payload:
    Byte 0:            Request ID Length (1 byte uint8, N)
    Bytes 1..N:        Request ID (UTF-8)
    Bytes N+1..end:    Interleaved 48 kHz stereo Float32 raw PCM bytes
"""

from __future__ import annotations

import json
import struct
from enum import IntEnum
from typing import Any

PROTOCOL_VERSION = 1
MAX_PAYLOAD_SIZE = 8 * 1024 * 1024  # 8 MB upper bound (sufficient for ~35s uncompressed 48kHz stereo or large metadata)

class PacketType(IntEnum):
    CONTROL_JSON = 0x01
    AUDIO_PCM = 0x02
    AUDIO_EOS = 0x03
    ERROR = 0x04


def encode_control_packet(data: dict[str, Any]) -> bytes:
    """Encode a JSON control dictionary into a framed packet."""
    json_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    header = struct.pack("!BI", PacketType.CONTROL_JSON, len(json_bytes))
    return header + json_bytes


def encode_pcm_packet(request_id: str, pcm_bytes: bytes) -> bytes:
    """Encode raw Float32 PCM chunk with associated request ID."""
    req_id_bytes = request_id.encode("utf-8")
    if len(req_id_bytes) > 255:
        raise ValueError("request_id exceeds 255 bytes")
    payload = struct.pack("!B", len(req_id_bytes)) + req_id_bytes + pcm_bytes
    header = struct.pack("!BI", PacketType.AUDIO_PCM, len(payload))
    return header + payload


def encode_eos_packet(request_id: str) -> bytes:
    """Encode end-of-stream signal for a given request ID."""
    req_id_bytes = request_id.encode("utf-8")
    payload = struct.pack("!B", len(req_id_bytes)) + req_id_bytes
    header = struct.pack("!BI", PacketType.AUDIO_EOS, len(payload))
    return header + payload


def encode_error_packet(request_id: str, error_message: str) -> bytes:
    """Encode error message for a given request ID."""
    data = {"type": "error", "request_id": request_id, "error": error_message}
    json_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    header = struct.pack("!BI", PacketType.ERROR, len(json_bytes))
    return header + json_bytes


def read_packet(sock) -> tuple[PacketType, bytes] | None:
    """Read a single packet from socket. Returns (packet_type, payload) or None on disconnect."""
    header_bytes = b""
    while len(header_bytes) < 5:
        chunk = sock.recv(5 - len(header_bytes))
        if not chunk:
            return None
        header_bytes += chunk

    packet_type_val, length = struct.unpack("!BI", header_bytes)
    if length > MAX_PAYLOAD_SIZE:
        raise ValueError(f"Packet payload exceeds maximum size limit ({length} > {MAX_PAYLOAD_SIZE})")

    try:
        packet_type = PacketType(packet_type_val)
    except ValueError:
        raise ValueError(f"Unknown packet type: {packet_type_val}")

    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            return None
        payload += chunk

    return packet_type, payload


def decode_pcm_payload(payload: bytes) -> tuple[str, bytes]:
    """Decode AUDIO_PCM or AUDIO_EOS payload into (request_id, pcm_bytes)."""
    id_len = payload[0]
    request_id = payload[1 : 1 + id_len].decode("utf-8")
    pcm_bytes = payload[1 + id_len :]
    return request_id, pcm_bytes
