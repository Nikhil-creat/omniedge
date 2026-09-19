"""
OmniEdge FastAPI gateway.

Exposes:
  - GET  /api/health                simple liveness probe
  - GET  /api/mesh/snapshot         current topology snapshot (REST)
  - POST /api/mesh/chaos/fail-node  operator/chaos-engineering endpoint
  - POST /api/vision/infer          run the CNN edge-vision classifier on demand
  - POST /api/knowledge/query       RAG query against the architecture docs + runbook
  - POST /api/copilot/command       agentic natural-language operator command -> action
  - WS   /ws/telemetry              PQC-session-secured telemetry stream (now includes
                                     CNN vision predictions alongside GNN/anomaly data)

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Set ANTHROPIC_API_KEY (and/or GROQ_API_KEY) to give
/api/copilot/command a real tool-calling loop instead of the
deterministic fallback planner (see app/ai/agentic_orchestrator.py).
Groq's free tier (https://console.groq.com, no card required) is a
good no-cost way to try the real agentic loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agents.node import AgentNode
from app.ai.agentic_orchestrator import AgenticCopilot
from app.ai.cnn_vision import synthetic_frame
from app.ai.rag import KnowledgeBase
from app.security import pqc
from app.security.channel import ChannelExpired, SecureChannel
from app.security.tpm_attestation import AdmissionAuthority

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("omniedge.main")

PEER_IDS = ["edge-node-berlin", "edge-node-tokyo", "edge-node-saopaulo", "edge-node-austin"]

admission_authority = AdmissionAuthority()
agent = AgentNode("edge-node-primary", admission_authority)
agent.seed_peers(PEER_IDS)

knowledge_base = KnowledgeBase()
copilot = AgenticCopilot(agent, knowledge_base=knowledge_base)

_broadcast_queue: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=256)
_background_task: Optional[asyncio.Task] = None


async def _telemetry_producer() -> None:
    async for frame in agent.run_telemetry_stream(interval_s=1.5):
        payload = frame.to_dict()
        if _broadcast_queue.full():
            _ = _broadcast_queue.get_nowait()
        await _broadcast_queue.put(payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _background_task
    _background_task = asyncio.create_task(_telemetry_producer())
    logger.info("OmniEdge telemetry producer started")
    yield
    if _background_task:
        agent.stop()
        _background_task.cancel()


app = FastAPI(title="OmniEdge Mesh Gateway", version="1.0.0", lifespan=lifespan)

# NOTE: in production, restrict allow_origins to the exact GitHub Pages
# origin (e.g. "https://<user>.github.io") rather than "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# REST endpoints
# --------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "node_id": agent.node_id, "pqc_backend": pqc.get_kem().variant}


@app.get("/api/mesh/snapshot")
async def mesh_snapshot() -> dict:
    return agent.topology.snapshot()


class FailNodeRequest(BaseModel):
    node_id: str


@app.post("/api/mesh/chaos/fail-node")
async def chaos_fail_node(req: FailNodeRequest) -> dict:
    if req.node_id not in agent.topology.telemetry:
        raise HTTPException(status_code=404, detail=f"unknown node_id {req.node_id!r}")
    agent.force_fail_node(req.node_id)
    return {"status": "fault injected", "node_id": req.node_id}


class CopilotCommand(BaseModel):
    text: str


@app.post("/api/copilot/command")
async def copilot_command(cmd: CopilotCommand) -> dict:
    """
    Agentic natural-language command endpoint. Routes through
    `AgenticCopilot`, which either drives a real Claude tool-calling
    loop (if ANTHROPIC_API_KEY is set) or a deterministic fallback
    planner — either way it returns a full reasoning trace, not just a
    final string, so the dashboard can show what the copilot actually
    did.
    """
    response = copilot.handle(cmd.text)
    return response.to_dict()


class VisionInferRequest(BaseModel):
    simulate_anomaly: bool = False


@app.post("/api/vision/infer")
async def vision_infer(req: VisionInferRequest) -> dict:
    """On-demand CNN inference against a fresh (synthetic) camera frame."""
    frame = synthetic_frame(anomalous=req.simulate_anomaly)
    result = agent.vision.infer(frame)
    return result.to_dict()


class KnowledgeQuery(BaseModel):
    question: str
    top_k: int = 3


@app.post("/api/knowledge/query")
async def knowledge_query(q: KnowledgeQuery) -> dict:
    """Direct RAG query against the architecture docs + operational runbook."""
    answer, retrieved = knowledge_base.answer_with_citations(q.question, top_k=q.top_k)
    return {
        "answer": answer,
        "citations": [
            {"chunk_id": r.chunk.chunk_id, "heading": r.chunk.heading, "score": round(r.score, 4)}
            for r in retrieved
        ],
        "backend": knowledge_base.backend_name,
    }


# --------------------------------------------------------------------------
# Secure WebSocket telemetry stream
# --------------------------------------------------------------------------


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket) -> None:
    """
    Handshake:
      1. Client sends {"kyber_pk": "<hex>"} (an ephemeral client keypair;
         in production this would be the browser's WebCrypto-derived
         PQC-hybrid identity).
      2. Server responds {"ciphertext": "<hex>"} — the Kyber
         encapsulation against the client's public key.
      3. Both sides now hold the same HKDF-derived session key and every
         subsequent frame is a base64 ChaCha20-Poly1305 AEAD frame.
    """
    await websocket.accept()

    try:
        hello_raw = await websocket.receive_text()
        hello = json.loads(hello_raw)
        client_kyber_pk = bytes.fromhex(hello["kyber_pk"])
    except Exception:
        await websocket.close(code=4400, reason="malformed handshake")
        return

    ciphertext, session_key = pqc.establish_session_hybrid(
        agent.identity, client_kyber_pk, responder_node_id="browser-client"
    )
    await websocket.send_text(json.dumps({"ciphertext": ciphertext.hex()}))

    channel = SecureChannel(session_key=session_key)
    logger.info("Telemetry WS session established (PQC handshake complete)")

    try:
        while True:
            payload = await _broadcast_queue.get()
            plaintext = json.dumps(payload).encode()
            try:
                frame = channel.encrypt(plaintext)
            except ChannelExpired:
                # Re-key: derive a fresh session via a new ephemeral encapsulation
                ciphertext, session_key = pqc.establish_session_hybrid(
                    agent.identity, client_kyber_pk, responder_node_id="browser-client"
                )
                await websocket.send_text(json.dumps({"rekey_ciphertext": ciphertext.hex()}))
                channel = SecureChannel(session_key=session_key)
                frame = channel.encrypt(plaintext)
            await websocket.send_bytes(frame)
    except WebSocketDisconnect:
        logger.info("Telemetry WS client disconnected")
