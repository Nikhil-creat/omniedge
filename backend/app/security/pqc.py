"""
Post-Quantum Cryptography abstraction layer.

Tries to use real liboqs bindings (CRYSTALS-Kyber / ML-KEM and
CRYSTALS-Dilithium / ML-DSA) via `oqs-python` when available. Falls back
to a clearly-labeled classical simulation backend so the reference
implementation runs anywhere without native liboqs binaries.

DO NOT treat the fallback backend as quantum-safe. It exists purely so
this reference architecture is runnable out of the box.
"""

from __future__ import annotations

import abc
import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("omniedge.security.pqc")

KYBER_VARIANT = "Kyber768"
DILITHIUM_VARIANT = "Dilithium3"

try:
    import oqs  # type: ignore

    _HAS_OQS = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_OQS = False
    logger.warning(
        "liboqs-python (oqs) not found. Falling back to SimulatedKEM / "
        "SimulatedSignature. This is NOT post-quantum secure and must "
        "not be used in production. Install `liboqs-python` and set "
        "OMNIEDGE_PQC_BACKEND=oqs to use real Kyber/Dilithium."
    )


# --------------------------------------------------------------------------
# Key Encapsulation Mechanism interface
# --------------------------------------------------------------------------


class KEMBackend(abc.ABC):
    """Common interface for a Kyber-family KEM implementation."""

    variant: str

    @abc.abstractmethod
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Return (public_key, secret_key)."""

    @abc.abstractmethod
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Return (ciphertext, shared_secret)."""

    @abc.abstractmethod
    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        """Return shared_secret."""


class OQSKemBackend(KEMBackend):
    """Real ML-KEM (Kyber) backend via liboqs."""

    def __init__(self, variant: str = KYBER_VARIANT) -> None:
        self.variant = variant

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        with oqs.KeyEncapsulation(self.variant) as kem:  # type: ignore
            pk = kem.generate_keypair()
            sk = kem.export_secret_key()
        return pk, sk

    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        with oqs.KeyEncapsulation(self.variant) as kem:  # type: ignore
            ct, ss = kem.encap_secret(public_key)
        return ct, ss

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        with oqs.KeyEncapsulation(self.variant, secret_key) as kem:  # type: ignore
            ss = kem.decap_secret(ciphertext)
        return ss


class SimulatedKEM(KEMBackend):
    """
    NOT POST-QUANTUM SAFE. X25519-based stand-in with the same call
    signature as the Kyber backend, used only when liboqs is unavailable
    so the reference system remains runnable end-to-end.
    """

    def __init__(self, variant: str = KYBER_VARIANT) -> None:
        self.variant = f"SIMULATED-{variant}"
        from cryptography.hazmat.primitives.asymmetric import x25519

        self._x25519 = x25519

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        sk = self._x25519.X25519PrivateKey.generate()
        pk = sk.public_key()
        from cryptography.hazmat.primitives import serialization

        pk_bytes = pk.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        sk_bytes = sk.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return pk_bytes, sk_bytes

    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        from cryptography.hazmat.primitives import serialization

        eph_sk = self._x25519.X25519PrivateKey.generate()
        eph_pk_bytes = eph_sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        peer_pk = self._x25519.X25519PublicKey.from_public_bytes(public_key)
        shared = eph_sk.exchange(peer_pk)
        shared_secret = hashlib.sha384(shared).digest()
        # "ciphertext" carries the ephemeral public key, mimicking a KEM ct
        return eph_pk_bytes, shared_secret

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        sk = self._x25519.X25519PrivateKey.from_private_bytes(secret_key)
        peer_eph_pk = self._x25519.X25519PublicKey.from_public_bytes(ciphertext)
        shared = sk.exchange(peer_eph_pk)
        return hashlib.sha384(shared).digest()


# --------------------------------------------------------------------------
# Digital signature interface
# --------------------------------------------------------------------------


class SignatureBackend(abc.ABC):
    variant: str

    @abc.abstractmethod
    def generate_keypair(self) -> Tuple[bytes, bytes]: ...

    @abc.abstractmethod
    def sign(self, secret_key: bytes, message: bytes) -> bytes: ...

    @abc.abstractmethod
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool: ...


class OQSSignatureBackend(SignatureBackend):
    """Real ML-DSA (Dilithium) backend via liboqs."""

    def __init__(self, variant: str = DILITHIUM_VARIANT) -> None:
        self.variant = variant

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        with oqs.Signature(self.variant) as sig:  # type: ignore
            pk = sig.generate_keypair()
            sk = sig.export_secret_key()
        return pk, sk

    def sign(self, secret_key: bytes, message: bytes) -> bytes:
        with oqs.Signature(self.variant, secret_key) as sig:  # type: ignore
            return sig.sign(message)

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        with oqs.Signature(self.variant) as sig:  # type: ignore
            return sig.verify(message, signature, public_key)


class SimulatedSignature(SignatureBackend):
    """NOT POST-QUANTUM SAFE. Ed25519 stand-in for Dilithium."""

    def __init__(self, variant: str = DILITHIUM_VARIANT) -> None:
        self.variant = f"SIMULATED-{variant}"
        from cryptography.hazmat.primitives.asymmetric import ed25519

        self._ed25519 = ed25519

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        from cryptography.hazmat.primitives import serialization

        sk = self._ed25519.Ed25519PrivateKey.generate()
        pk_bytes = sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        sk_bytes = sk.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return pk_bytes, sk_bytes

    def sign(self, secret_key: bytes, message: bytes) -> bytes:
        sk = self._ed25519.Ed25519PrivateKey.from_private_bytes(secret_key)
        return sk.sign(message)

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        pk = self._ed25519.Ed25519PublicKey.from_public_bytes(public_key)
        try:
            pk.verify(signature, message)
            return True
        except Exception:
            return False


# --------------------------------------------------------------------------
# Backend selection + session key derivation
# --------------------------------------------------------------------------


def _select_backend_flag() -> bool:
    forced = os.environ.get("OMNIEDGE_PQC_BACKEND", "").lower()
    if forced == "simulated":
        return False
    if forced == "oqs":
        if not _HAS_OQS:
            raise RuntimeError("OMNIEDGE_PQC_BACKEND=oqs but liboqs is not installed")
        return True
    return _HAS_OQS


_USE_OQS = _select_backend_flag()


def get_kem() -> KEMBackend:
    return OQSKemBackend() if _USE_OQS else SimulatedKEM()


def get_signature() -> SignatureBackend:
    return OQSSignatureBackend() if _USE_OQS else SimulatedSignature()


def hkdf_sha384(shared_secret: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """RFC 5869 HKDF-SHA384 session-key derivation."""
    prk = hmac.new(salt, shared_secret, hashlib.sha384).digest()
    t = b""
    okm = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha384).digest()
        okm += t
        counter += 1
    return okm[:length]


@dataclass
class NodeIdentity:
    """A node's long-term PQC identity."""

    node_id: str
    kyber_pk: bytes
    kyber_sk: bytes
    dilithium_pk: bytes
    dilithium_sk: bytes

    @classmethod
    def generate(cls, node_id: str) -> "NodeIdentity":
        kem = get_kem()
        sig = get_signature()
        kyber_pk, kyber_sk = kem.generate_keypair()
        dil_pk, dil_sk = sig.generate_keypair()
        return cls(node_id, kyber_pk, kyber_sk, dil_pk, dil_sk)


def establish_session_hybrid(
    initiator: NodeIdentity, responder_kyber_pk: bytes, responder_node_id: str
) -> Tuple[bytes, bytes]:
    """
    Initiator side of the hybrid Kyber(+X25519 fallback) handshake.
    Returns (ciphertext_to_send, derived_session_key).
    """
    kem = get_kem()
    ciphertext, shared_secret = kem.encapsulate(responder_kyber_pk)
    salt = f"{initiator.node_id}|{responder_node_id}".encode()
    session_key = hkdf_sha384(shared_secret, salt=salt, info=b"omniedge-session-v1")
    return ciphertext, session_key


def complete_session_hybrid(
    responder: NodeIdentity, initiator_node_id: str, ciphertext: bytes
) -> bytes:
    """Responder side: derive the same session key from the ciphertext."""
    kem = get_kem()
    shared_secret = kem.decapsulate(responder.kyber_sk, ciphertext)
    salt = f"{initiator_node_id}|{responder.node_id}".encode()
    return hkdf_sha384(shared_secret, salt=salt, info=b"omniedge-session-v1")
