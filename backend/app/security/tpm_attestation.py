"""
TPM 2.0-style remote attestation handshake for edge-node admission.

Reference implementation using a software `SimulatedTPM` so it runs
without physical hardware. In production, replace `SimulatedTPM` with a
thin wrapper around `tpm2-pytss` talking to `/dev/tpm0` (or a vTPM), and
have the Admission Authority validate the EK certificate chain against
the OEM root CA offline.

Steps modeled (see docs/ARCHITECTURE.md §3):
  1. EK (Endorsement Key) creation + certificate
  2. AK (Attestation Key) creation, certified by EK
  3. MakeCredential / ActivateCredential proof-of-possession
  4. Quote over PCRs, verified against a golden measurement set
  5. Issuance of a short-lived signed Membership Certificate
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.security import pqc

MEMBERSHIP_CERT_TTL_SECONDS = 24 * 60 * 60  # 24h, per architecture doc

def _compute_golden_pcr_digest(pcr_count: int = 8) -> str:
    """
    In production this "golden" digest comes from a signed reference
    manifest shipped alongside each approved firmware/agent release
    (i.e. it is computed once at release-build time, not derived here).
    We compute it the same way `SimulatedTPM` derives a healthy node's
    PCR digest so the reference admission flow succeeds end-to-end
    without needing an external manifest file.
    """
    pcr_state = {p: hashlib.sha256(f"pcr{p}-healthy-boot".encode()).hexdigest() for p in range(pcr_count)}
    return hashlib.sha256("".join(pcr_state[p] for p in range(pcr_count)).encode()).hexdigest()


# "Golden" PCR measurement an approved boot chain must reproduce.
APPROVED_PCR_DIGEST = _compute_golden_pcr_digest()


@dataclass
class SimulatedTPM:
    """Stand-in for a physical/virtual TPM 2.0 chip."""

    node_id: str
    _ek_sk: bytes = field(init=False, repr=False)
    ek_pk: bytes = field(init=False)
    _ak_sk: bytes = field(init=False, repr=False)
    ak_pk: bytes = field(init=False)
    pcr_state: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sig = pqc.get_signature()
        self.ek_pk, self._ek_sk = sig.generate_keypair()
        self.ak_pk, self._ak_sk = sig.generate_keypair()
        # Simulate measured-boot PCRs 0-7 for a healthy node.
        for pcr in range(8):
            self.pcr_state[pcr] = hashlib.sha256(f"pcr{pcr}-healthy-boot".encode()).hexdigest()

    def ek_certificate(self) -> "tuple[bytes, bytes]":
        """Self-signed stand-in for an OEM-issued EK certificate.

        Returns (payload, signature) rather than a delimited blob, since
        the signature is raw binary and may itself contain any delimiter
        byte we could otherwise choose.
        """
        payload = json.dumps({"node_id": self.node_id, "ek_pk": self.ek_pk.hex()}).encode()
        sig = pqc.get_signature()
        signature = sig.sign(self._ek_sk, payload)
        return payload, signature

    def certify_ak(self) -> bytes:
        """TPM2_Certify: EK vouches the AK was generated inside this TPM."""
        payload = json.dumps({"node_id": self.node_id, "ak_pk": self.ak_pk.hex()}).encode()
        sig = pqc.get_signature()
        return sig.sign(self._ek_sk, payload)

    def activate_credential(self, encrypted_credential: bytes, wrapping_key: bytes) -> bytes:
        """
        TPM2_ActivateCredential: only a TPM holding the private AK (bound
        to this physical chip) can unwrap the credential the verifier
        encrypted under the AK's public identity.
        """
        # Simulated unwrap: HMAC-based "decryption" keyed on the AK secret,
        # standing in for the TPM-internal asymmetric unwrap.
        mac_key = hashlib.sha256(self._ak_sk).digest()
        expected = hmac.new(mac_key, wrapping_key, hashlib.sha256).digest()
        return expected  # proof-of-possession value returned to verifier

    def quote(self, nonce: bytes, pcr_selection: Optional[list] = None) -> Dict[str, str]:
        """TPM2_Quote: signed attestation over selected PCRs + a fresh nonce."""
        pcr_selection = pcr_selection or list(range(8))
        pcr_digest = hashlib.sha256(
            "".join(self.pcr_state[p] for p in pcr_selection).encode()
        ).hexdigest()
        message = json.dumps(
            {"pcr_digest": pcr_digest, "nonce": nonce.hex(), "node_id": self.node_id}
        ).encode()
        sig = pqc.get_signature()
        signature = sig.sign(self._ak_sk, message)
        return {
            "pcr_digest": pcr_digest,
            "nonce": nonce.hex(),
            "message": message.hex(),
            "signature": signature.hex(),
            "ak_pk": self.ak_pk.hex(),
        }


@dataclass
class MembershipCertificate:
    node_id: str
    ak_pk_hex: str
    kyber_pk_hex: str
    dilithium_pk_hex: str
    issued_at: float
    expires_at: float
    authority_signature_hex: str

    def is_valid(self) -> bool:
        return time.time() < self.expires_at

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "ak_pk_hex": self.ak_pk_hex,
            "kyber_pk_hex": self.kyber_pk_hex,
            "dilithium_pk_hex": self.dilithium_pk_hex,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "authority_signature_hex": self.authority_signature_hex,
        }


class AdmissionAuthority:
    """
    Verifies a candidate node's TPM attestation and, on success, issues a
    short-lived Membership Certificate binding its hardware identity to
    its PQC (Kyber/Dilithium) mesh identity.
    """

    def __init__(self) -> None:
        sig = pqc.get_signature()
        self._sig_backend = sig
        self.authority_pk, self._authority_sk = sig.generate_keypair()
        self._issued: Dict[str, MembershipCertificate] = {}

    def admit(
        self,
        tpm: SimulatedTPM,
        node_identity: pqc.NodeIdentity,
    ) -> MembershipCertificate:
        # 1. EK cert "chain" check (self-signed stand-in; in prod: verify
        #    against OEM root CA)
        ek_payload, ek_signature = tpm.ek_certificate()
        if not self._sig_backend.verify(tpm.ek_pk, ek_payload, ek_signature):
            raise PermissionError("EK certificate verification failed")

        # 2. AK certified by EK
        ak_cert_sig = tpm.certify_ak()
        ak_payload = json.dumps({"node_id": tpm.node_id, "ak_pk": tpm.ak_pk.hex()}).encode()
        if not self._sig_backend.verify(tpm.ek_pk, ak_payload, ak_cert_sig):
            raise PermissionError("AK certification by EK failed")

        # 3. MakeCredential / ActivateCredential proof of possession
        wrapping_key = os.urandom(32)
        mac_key = hashlib.sha256(tpm._ak_sk).digest()  # verifier side simulation only
        expected_proof = hmac.new(mac_key, wrapping_key, hashlib.sha256).digest()
        actual_proof = tpm.activate_credential(b"", wrapping_key)
        if not hmac.compare_digest(expected_proof, actual_proof):
            raise PermissionError("ActivateCredential proof-of-possession failed")

        # 4. Quote + PCR golden-measurement comparison
        nonce = os.urandom(16)
        quote = tpm.quote(nonce)
        if not self._sig_backend.verify(
            bytes.fromhex(quote["ak_pk"]),
            bytes.fromhex(quote["message"]),
            bytes.fromhex(quote["signature"]),
        ):
            raise PermissionError("TPM quote signature invalid")
        if quote["nonce"] != nonce.hex():
            raise PermissionError("quote nonce mismatch (possible replay)")
        if quote["pcr_digest"] != APPROVED_PCR_DIGEST:
            raise PermissionError(
                f"PCR measurement mismatch for {tpm.node_id}: node not admitted "
                "(unapproved firmware/boot state)"
            )

        # 5. Issue Membership Certificate
        now = time.time()
        cert_payload = json.dumps(
            {
                "node_id": node_identity.node_id,
                "ak_pk_hex": tpm.ak_pk.hex(),
                "kyber_pk_hex": node_identity.kyber_pk.hex(),
                "dilithium_pk_hex": node_identity.dilithium_pk.hex(),
                "issued_at": now,
                "expires_at": now + MEMBERSHIP_CERT_TTL_SECONDS,
            }
        ).encode()
        authority_sig = self._sig_backend.sign(self._authority_sk, cert_payload)
        cert = MembershipCertificate(
            node_id=node_identity.node_id,
            ak_pk_hex=tpm.ak_pk.hex(),
            kyber_pk_hex=node_identity.kyber_pk.hex(),
            dilithium_pk_hex=node_identity.dilithium_pk.hex(),
            issued_at=now,
            expires_at=now + MEMBERSHIP_CERT_TTL_SECONDS,
            authority_signature_hex=authority_sig.hex(),
        )
        self._issued[node_identity.node_id] = cert
        return cert

    def revoke(self, node_id: str) -> None:
        self._issued.pop(node_id, None)

    def is_member_in_good_standing(self, node_id: str) -> bool:
        cert = self._issued.get(node_id)
        return cert is not None and cert.is_valid()
