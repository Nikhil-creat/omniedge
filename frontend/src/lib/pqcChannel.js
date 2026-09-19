/**
 * Client-side counterpart to backend/app/security/{pqc,channel}.py.
 *
 * The backend's default (no-liboqs) mode uses an X25519-based
 * `SimulatedKEM` behind the same call shape as real Kyber, and derives
 * session keys with HKDF-SHA384. This module reproduces that exact
 * derivation in the browser using audited `@noble/*` primitives so the
 * dashboard can decrypt the telemetry stream without a server-side
 * plaintext fallback.
 *
 * IMPORTANT: like the backend's SimulatedKEM, this is NOT post-quantum
 * secure. It exists to demonstrate the full hybrid-handshake data flow
 * end-to-end. In a production deployment, swap the backend to the real
 * `oqs` (Kyber/Dilithium) backend and this module to a WASM Kyber
 * implementation (e.g. via liboqs-js) with a matching wire format.
 */

import { x25519 } from "@noble/curves/ed25519";
import { hkdf } from "@noble/hashes/hkdf";
import { sha384 } from "@noble/hashes/sha512";
import { chacha20poly1305 } from "@noble/ciphers/chacha";

const SESSION_INFO = new TextEncoder().encode("omniedge-session-v1");

export function bytesToHex(bytes) {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export function hexToBytes(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return out;
}

/** Generates the client's ephemeral "Kyber-shaped" X25519 identity. */
export function generateClientIdentity() {
  const secretKey = x25519.utils.randomPrivateKey();
  const publicKey = x25519.getPublicKey(secretKey);
  return { secretKey, publicKey };
}

/**
 * Completes the responder side of the hybrid handshake, mirroring
 * `pqc.complete_session_hybrid` on the backend:
 *   shared_secret = SHA384(X25519(clientSecretKey, serverEphemeralPubKey))
 *   session_key   = HKDF-SHA384(shared_secret, salt=`${serverId}|${clientId}`,
 *                                info="omniedge-session-v1", length=32)
 */
export function completeSessionHybrid({ clientSecretKey, serverEphemeralPubKey, serverNodeId, clientNodeId = "browser-client" }) {
  const rawShared = x25519.getSharedSecret(clientSecretKey, serverEphemeralPubKey);
  const sharedSecret = sha384(rawShared);
  const salt = new TextEncoder().encode(`${serverNodeId}|${clientNodeId}`);
  const sessionKey = hkdf(sha384, sharedSecret, salt, SESSION_INFO, 32);
  return sessionKey;
}

/**
 * AEAD-encrypted channel matching backend/app/security/channel.py's
 * `SecureChannel`: 12-byte nonce = 4 zero bytes || 8-byte big-endian
 * frame counter; wire frame = 8-byte counter || ChaCha20-Poly1305(ct).
 */
export class SecureChannel {
  constructor(sessionKey) {
    this.key = sessionKey;
    this.recvHighWatermark = -1n;
  }

  _nonce(counterBigInt) {
    const nonce = new Uint8Array(12);
    const view = new DataView(nonce.buffer);
    // bytes 0-3 stay zero; bytes 4-11 hold the big-endian 64-bit counter
    view.setBigUint64(4, counterBigInt, false);
    return nonce;
  }

  /** frame: Uint8Array received over the WebSocket (binary frame). */
  decrypt(frame, associatedData = new Uint8Array()) {
    const counterBytes = frame.slice(0, 8);
    const counterView = new DataView(counterBytes.buffer, counterBytes.byteOffset, 8);
    const counter = counterView.getBigUint64(0, false);

    if (counter <= this.recvHighWatermark) {
      throw new Error("replayed or out-of-order frame rejected");
    }
    const nonce = this._nonce(counter);
    const ciphertext = frame.slice(8);
    const cipher = chacha20poly1305(this.key, nonce, associatedData);
    const plaintext = cipher.decrypt(ciphertext);
    this.recvHighWatermark = counter;
    return plaintext;
  }
}

/**
 * Full handshake driver: given an already-open WebSocket, sends the
 * client's ephemeral public key, awaits the server's Kyber-shaped
 * ciphertext response, and returns a ready `SecureChannel`.
 */
export async function performHandshake(ws, { serverNodeId }) {
  const { secretKey, publicKey } = generateClientIdentity();

  ws.send(JSON.stringify({ kyber_pk: bytesToHex(publicKey) }));

  const serverMessage = await new Promise((resolve, reject) => {
    const onMessage = (event) => {
      ws.removeEventListener("message", onMessage);
      try {
        resolve(JSON.parse(event.data));
      } catch (err) {
        reject(err);
      }
    };
    ws.addEventListener("message", onMessage);
  });

  const ciphertextHex = serverMessage.ciphertext ?? serverMessage.rekey_ciphertext;
  if (!ciphertextHex) {
    throw new Error("handshake failed: server did not return a ciphertext");
  }
  const serverEphemeralPubKey = hexToBytes(ciphertextHex);
  const sessionKey = completeSessionHybrid({
    clientSecretKey: secretKey,
    serverEphemeralPubKey,
    serverNodeId,
  });

  return new SecureChannel(sessionKey);
}
