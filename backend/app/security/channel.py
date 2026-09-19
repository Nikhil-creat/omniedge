"""
AEAD-encrypted channel built on top of a PQC-derived session key.
ChaCha20-Poly1305 with a monotonically increasing nonce counter and a
frame-count / time-based rekey trigger.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

REKEY_AFTER_FRAMES = 1_048_576  # 2^20
REKEY_AFTER_SECONDS = 600  # 10 minutes


class ChannelExpired(Exception):
    """Raised when a channel needs rekeying before it can send/receive."""


@dataclass
class SecureChannel:
    session_key: bytes
    _aead: ChaCha20Poly1305 = field(init=False)
    _send_counter: int = field(default=0, init=False)
    _recv_high_watermark: int = field(default=-1, init=False)
    _opened_at: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self._aead = ChaCha20Poly1305(self.session_key)

    # -- helpers -----------------------------------------------------
    def _nonce(self, counter: int) -> bytes:
        # 12-byte nonce: 4 bytes random salt (fixed per channel) + 8 byte counter
        return struct.pack(">Q", counter).rjust(12, b"\x00")

    def needs_rekey(self) -> bool:
        return (
            self._send_counter >= REKEY_AFTER_FRAMES
            or (time.time() - self._opened_at) >= REKEY_AFTER_SECONDS
        )

    # -- API -----------------------------------------------------------
    def encrypt(self, plaintext: bytes, associated_data: bytes = b"") -> bytes:
        if self.needs_rekey():
            raise ChannelExpired("channel requires rekey before further sends")
        nonce = self._nonce(self._send_counter)
        ct = self._aead.encrypt(nonce, plaintext, associated_data)
        frame = struct.pack(">Q", self._send_counter) + ct
        self._send_counter += 1
        return frame

    def decrypt(self, frame: bytes, associated_data: bytes = b"") -> bytes:
        counter = struct.unpack(">Q", frame[:8])[0]
        if counter <= self._recv_high_watermark:
            raise InvalidTag("replayed or out-of-order frame rejected")
        nonce = self._nonce(counter)
        pt = self._aead.decrypt(nonce, frame[8:], associated_data)
        self._recv_high_watermark = counter
        return pt


def new_random_salt(n: int = 16) -> bytes:
    return os.urandom(n)
