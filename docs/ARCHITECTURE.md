# OmniEdge — System Architecture & Cryptographic Specification

## 1. Overview

OmniEdge is a decentralized mesh of edge-AI agent nodes that:

- authenticate each other using **hardware-rooted trust** (TPM 2.0 attestation),
- secure all inter-node traffic with **post-quantum cryptography** (Kyber KEM + Dilithium signatures),
- run a **self-healing GNN-based topology manager** that reroutes traffic around failed/compromised nodes,
- stream live telemetry to a **WebXR-capable 3D dashboard** over authenticated WebSockets,
- expose an **LLM copilot** for natural-language operational commands.

```
┌─────────────────────────────────────────────────────────────────┐
│                         OmniEdge Mesh                             │
│                                                                     │
│   ┌──────────┐  PQC-secured   ┌──────────┐  PQC-secured  ┌──────┐ │
│   │ Node A    │◄──channel────►│ Node B    │◄──channel────►│Node C│ │
│   │ (agent)   │                │ (agent)   │               │      │ │
│   └────┬─────┘                └────┬─────┘               └──┬───┘ │
│        │  TPM quote + Dilithium sig on admission              │     │
│        ▼                                                       ▼     │
│                     Gossip / GNN topology graph                     │
└───────────────────────────────┬───────────────────────────────────┘
                                  │ wss:// (Kyber-derived session key)
                                  ▼
                     FastAPI Telemetry Gateway
                                  │
                                  ▼
                React + Three.js Spatial Digital Twin (GitHub Pages)
```

> **Implementation honesty note:** this repo ships a *reference/simulation*
> implementation. It uses the `pqcrypto` / `liboqs-python` bindings where
> available and falls back to a clearly-labeled `SimulatedKEM` /
> `SimulatedSignature` backend (HKDF + Ed25519, NOT quantum-safe) so the
> system runs on any machine without native liboqs binaries installed. Swap
> `security/pqc.py`'s backend selection to `oqs` in a real deployment and
> run a security review before using this in production — this is a
> portfolio/reference architecture, not an audited cryptographic product.

## 2. Post-Quantum Secured Channels

### 2.1 Key Encapsulation — CRYSTALS-Kyber (ML-KEM)

Every node generates a long-term Kyber768 keypair at provisioning time,
sealed by the local TPM (see §3). Session establishment between two nodes
`A` and `B`:

1. `A` fetches `B`'s Kyber public key (pinned via the mesh's signed
   membership list, §2.3).
2. `A` runs `Encapsulate(pk_B) -> (ciphertext, shared_secret_A)`.
3. `A` sends `ciphertext` to `B`, signed with `A`'s Dilithium key.
4. `B` verifies the signature, then runs
   `Decapsulate(sk_B, ciphertext) -> shared_secret_B`.
5. Both sides derive a symmetric session key via
   `HKDF-SHA384(shared_secret, salt=node_ids, info="omniedge-session-v1")`.
6. All subsequent traffic on that link is AEAD-encrypted
   (ChaCha20-Poly1305) under the derived key, with a monotonically
   increasing nonce counter and periodic re-keying (every 10 min or 2^20
   frames, whichever is first).

### 2.2 Authentication — CRYSTALS-Dilithium (ML-DSA)

- Every control-plane message (topology updates, admission requests,
  chaos/failover events) is signed with the sender's Dilithium3 key.
- Signatures are verified against the **mesh membership ledger** — a
  gossiped, append-only, signature-chained list of `{node_id, kyber_pk,
  dilithium_pk, tpm_ek_cert_hash, joined_at}` records, so a compromised
  node cannot silently swap its own keys without breaking the hash chain.

### 2.3 Hybrid mode

For defense-in-depth during the PQC transition period, the reference
handshake is **hybrid**: the final session key mixes the Kyber shared
secret with an X25519 ECDH shared secret
(`HKDF(kyber_ss || x25519_ss)`), so the link stays secure even if a
classical or a post-quantum primitive alone is later broken.

## 3. TPM 2.0 Hardware Attestation — Node Admission Handshake

Goal: a new edge node cannot join the mesh (and thus cannot inject
telemetry or vote in the GNN topology graph) without proving it holds a
genuine TPM-backed identity key whose PCRs match an approved firmware/boot
state.

```
New Node (Prover)                          Admission Authority (Verifier)
------------------                          ------------------------------
1. TPM2_CreatePrimary(EK)  -----EK cert-------------------------------->
2. TPM2_Create(AK, bound to EK)
3. TPM2_Certify(AK by EK)  -----AK public + cert----------------------->
                                             4. Verify EK cert chains to
                                                OEM root CA (offline OK)
                                             5. Generate nonce N, encrypt
                                                a "credential" under AK
                                                pub using TPM2_MakeCredential
                              <----(encrypted credential, N)------------
6. TPM2_ActivateCredential
   (proves possession of the
   private AK, bound to *this*
   physical TPM)
7. TPM2_Quote(PCRs 0-7, N)  -----signed PCR quote + N echo------------->
                                             8. Verify quote signature
                                                with AK; compare PCR
                                                digest to the approved
                                                "golden" measurement set
                                             9. If match: issue a signed
                                                Membership Certificate
                                                binding {node_id, AK pub,
                                                Kyber pk, Dilithium pk}
   <----Membership Certificate + mesh peer list--------------------------
10. Node joins gossip topology, begins
    PQC-secured sessions with peers
```

Key properties:

- **Remote attestation** (steps 4–9) proves the node's boot chain
  (bootloader → kernel → agent binary hashes in PCRs 0–7) matches an
  approved measurement, not just that "some TPM" answered.
- **`TPM2_ActivateCredential`** binds the proof to a *specific physical
  TPM* — a cloned key on different hardware cannot complete it.
- The **Membership Certificate** is short-lived (default 24h) and must be
  renewed via an abbreviated re-quote, so a node whose PCRs drift (e.g.
  unauthorized firmware change) is automatically evicted from the mesh at
  renewal time.
- Reference code models this handshake in
  `backend/app/security/tpm_attestation.py` using a software TPM
  simulator (`SimulatedTPM`) so it runs without physical hardware; swap in
  `tpm2-pytss` against `/dev/tpm0` for real deployments.

## 4. Self-Healing Topology

See `backend/app/agents/gnn_topology.py`. Each node maintains a live graph
(`networkx.Graph`) of mesh links, weighted by measured latency and PQC
handshake success rate. A lightweight GNN (`torch_geometric`-style message
passing, with a pure-NumPy fallback so the reference implementation has no
hard GPU/torch dependency) scores each node's "health embedding" from
neighbor telemetry; when a node's embedding crosses a failure threshold or
it misses N consecutive heartbeats, the manager:

1. marks the node `SUSPECT`, then `DOWN` after a confirmation window,
2. recomputes shortest augmenting paths for every route that traversed
   the dead node (`networkx.shortest_path` over the surviving subgraph),
3. broadcasts a signed `TopologyUpdate` so all peers reroute
   consistently, and
4. keeps the node's slot "reservable" — if it heartbeats again with a
   valid PQC session, it's re-admitted without a full TPM re-attestation
   as long as its Membership Certificate hasn't expired.

## 5. Data Flow to the Dashboard

`FastAPI` node(s) expose `wss://.../ws/telemetry`. The frontend opens a
WebSocket, performs the same Kyber-derived session-key handshake over a
small JSON control channel, then receives AEAD-encrypted telemetry frames
containing CPU/mem, PQC handshake latency, anomaly scores, CNN vision
predictions (§6), and topology deltas, which drive both the 2D charts
and the Three.js digital twin.

## 6. Edge Computer Vision (CNN)

Every node runs a small convolutional neural network locally against
its camera/sensor feed, classifying each frame into one of `normal`,
`obstruction`, `intrusion`, `fire_smoke`, or `low_visibility`. The
network is intentionally tiny — two conv+maxpool stages feeding a
fully-connected softmax head — so it runs on constrained edge hardware
(e.g. a Raspberry-Pi-class device) with no GPU required.

```
input (32x32 grayscale)
   -> Conv(1->4, 5x5) -> ReLU -> MaxPool(2)
   -> Conv(4->8, 5x5) -> ReLU -> MaxPool(2)
   -> Flatten -> FC -> softmax(5 classes)
```

`backend/app/ai/cnn_vision.py` implements this twice, behind the same
interface: a `TorchCNN` (used automatically if `torch` is installed)
and a dependency-free `NumpyCNN` fallback. Out of the box, both run
with **randomly initialized weights** against synthetic frames — the
module demonstrates a correctly-shaped, correctly-performing inference
pipeline, not a trained detector. To use real weights:

- **Torch backend**: train a matching `nn.Module` (same layer shapes)
  and call `EdgeVisionService.load_weights(path)`, which calls
  `model.load_state_dict(torch.load(path))`.
- **NumPy backend**: save `conv1_kernels`, `conv2_kernels`,
  `fc_weights`, `fc_bias` as a `.npz` archive and call the same
  `load_weights(path)`, which calls `NumpyCNN.load_weights`.

A vision prediction is emitted on every telemetry tick and folded into
the same WebSocket stream as the GNN topology and neuromorphic anomaly
score, so the dashboard can correlate "the camera sees an obstruction"
with "the node's anomaly score just spiked" as one operational picture.

## 7. Retrieval-Augmented Generation (RAG)

`backend/app/ai/rag.py`'s `KnowledgeBase` indexes the markdown
architecture docs and operational runbook (`app/ai/knowledge/*.md` plus
this file) by splitting them into per-heading chunks, embedding each
chunk, and answering queries by cosine similarity. Two embedding
backends behind one interface:

- `SentenceTransformerEmbedder` — real dense embeddings via
  `sentence-transformers`, used automatically if installed.
- `TfidfEmbedder` — a dependency-free TF-IDF + cosine-similarity
  fallback (NumPy only), used otherwise, with each chunk's heading
  re-weighted into its vector so a query naming a specific situation
  ("node shows DOWN") ranks the chunk titled that way first.

This is intentionally an in-memory reference implementation. For a
knowledge base too large to hold in memory, swap `KnowledgeBase` for a
real vector database (pgvector, Qdrant, Chroma, Pinecone) behind the
same `query()` / `answer_with_citations()` interface — nothing else in
the system needs to change.

## 8. Agentic AI Orchestrator

`backend/app/ai/agentic_orchestrator.py`'s `AgenticCopilot` is the
mesh's single natural-language entry point. It wires the mesh's
operational surface — topology control, CNN vision inference, and RAG
knowledge retrieval — up as callable **tools**:

| Tool | Backs |
|---|---|
| `get_mesh_snapshot` | §4 GNN topology manager |
| `fail_node` / `rejoin_node` | §4 self-healing + chaos engineering |
| `run_vision_inference` | §6 CNN edge vision |
| `query_knowledge` | §7 RAG knowledge base |

Two operating modes, same interface:

- **Real agentic loop** (`ANTHROPIC_API_KEY` set + `anthropic`
  installed): drives an actual Claude tool-calling loop. The model
  decides which tool(s) a request needs, in which order, and can chain
  several calls (e.g. "check the camera, and if it looks bad, fail
  that node") before composing a final answer.
- **Deterministic fallback planner** (default, no API key required):
  a rule-based router that matches the request's intent (fail/rejoin,
  status, vision, or — by default — a knowledge-base question) to the
  same tools, so the whole stack is runnable and testable with zero
  external API dependencies.

Every response includes a full **trace** — one Thought/Action/
Observation step per tool call — not just the final reply, so the
dashboard's Copilot drawer can show its reasoning, not only its
answer.

