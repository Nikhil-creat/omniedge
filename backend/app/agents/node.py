"""
The asynchronous edge-agent node: wires together PQC identity, TPM
admission, the neuromorphic anomaly processor, and the GNN topology
manager into one long-running asyncio task that periodically emits
telemetry frames.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Dict, List, Optional

from app.agents.gnn_topology import GNNTopologyManager, NodeTelemetry
from app.agents.neuromorphic_anomaly import NeuromorphicAnomalyProcessor, SpikeEvent
from app.ai.cnn_vision import EdgeVisionService, synthetic_frame
from app.security import pqc
from app.security.tpm_attestation import AdmissionAuthority, SimulatedTPM

logger = logging.getLogger("omniedge.agent.node")


@dataclass
class TelemetryFrame:
    node_id: str
    timestamp: float
    cpu_pct: float
    mem_pct: float
    pqc_handshake_latency_ms: float
    anomaly_score: float
    topology_snapshot: dict
    vision_prediction: str
    vision_confidence: float
    vision_inference_ms: float

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "cpu_pct": round(self.cpu_pct, 2),
            "mem_pct": round(self.mem_pct, 2),
            "pqc_handshake_latency_ms": round(self.pqc_handshake_latency_ms, 3),
            "anomaly_score": round(self.anomaly_score, 4),
            "topology": self.topology_snapshot,
            "vision": {
                "predicted_class": self.vision_prediction,
                "confidence": round(self.vision_confidence, 4),
                "inference_ms": round(self.vision_inference_ms, 3),
            },
        }


class AgentNode:
    """
    One mesh participant. In this reference deployment a single FastAPI
    process typically hosts one real `AgentNode` (the local node) plus
    several *simulated peer* nodes so the dashboard has an interesting
    mesh to visualize without needing a physical multi-machine cluster.
    """

    def __init__(self, node_id: str, admission_authority: AdmissionAuthority) -> None:
        self.node_id = node_id
        self.identity = pqc.NodeIdentity.generate(node_id)
        self.tpm = SimulatedTPM(node_id)
        self.admission_authority = admission_authority
        self.membership_cert = admission_authority.admit(self.tpm, self.identity)
        self.anomaly_processor = NeuromorphicAnomalyProcessor()
        self.topology = GNNTopologyManager(self_node_id=node_id)
        self.vision = EdgeVisionService(node_id, use_torch=False)
        self._running = False
        self._rng = random.Random(hash(node_id) & 0xFFFF)
        self._manually_failed: set[str] = set()

        logger.info(
            "Node %s admitted to mesh (membership expires %s)",
            node_id,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.membership_cert.expires_at)),
        )

    # -- simulated peers, for a self-contained runnable demo ----------
    def seed_peers(self, peer_ids: List[str]) -> None:
        for pid in peer_ids:
            t = NodeTelemetry(
                node_id=pid,
                cpu_pct=self._rng.uniform(10, 40),
                mem_pct=self._rng.uniform(20, 50),
                handshake_latency_ms=self._rng.uniform(15, 60),
                anomaly_score=0.05,
            )
            self.topology.add_node(pid, t)
            self.topology.add_link(
                self.node_id, pid, latency_ms=t.handshake_latency_ms, handshake_ok=True
            )
        # Mesh a few peer-to-peer links too, for a realistic-looking graph
        for i in range(len(peer_ids)):
            for j in range(i + 1, len(peer_ids)):
                if self._rng.random() < 0.5:
                    self.topology.add_link(
                        peer_ids[i], peer_ids[j], latency_ms=self._rng.uniform(15, 80), handshake_ok=True
                    )

    def _simulate_peer_drift(self) -> None:
        """Randomly evolve peer telemetry + occasionally fail a node, so
        the self-healing loop has something to demonstrate live."""
        for pid, t in self.topology.telemetry.items():
            if pid in self._manually_failed:
                # Under an operator/chaos-injected fault: stop heartbeating
                # so the GNN liveness check can actually mark it DOWN.
                continue
            t.cpu_pct = min(100.0, max(0.0, t.cpu_pct + self._rng.uniform(-5, 5)))
            t.mem_pct = min(100.0, max(0.0, t.mem_pct + self._rng.uniform(-3, 3)))
            t.handshake_latency_ms = max(2.0, t.handshake_latency_ms + self._rng.uniform(-4, 4))
            t.anomaly_score = max(0.0, min(1.0, t.anomaly_score + self._rng.uniform(-0.03, 0.03)))
            if self._rng.random() < 0.004:  # rare simulated outage
                t.cpu_pct = 99.9
                t.anomaly_score = 0.95
                logger.warning("Simulated fault injected on peer %s", pid)
            else:
                self.topology.heartbeat(pid, t)

    # -- core telemetry loop -------------------------------------------
    async def run_telemetry_stream(self, interval_s: float = 1.0) -> AsyncIterator[TelemetryFrame]:
        self._running = True
        while self._running:
            handshake_start = time.perf_counter()
            # Demonstrate a real Kyber-derived session establishment cost
            peer_identity = pqc.NodeIdentity.generate("ephemeral-probe")
            _, _ = pqc.establish_session_hybrid(
                self.identity, peer_identity.kyber_pk, peer_identity.node_id
            )
            handshake_latency_ms = (time.perf_counter() - handshake_start) * 1000.0

            event = SpikeEvent(
                timestamp=time.time(),
                channel=self._rng.randint(0, 31),
                amplitude=self._rng.uniform(0.1, 0.6),
            )
            anomaly = self.anomaly_processor.process_event(event)

            self._simulate_peer_drift()
            for update in self.topology.check_liveness_and_heal():
                logger.info(
                    "TopologyUpdate: %s -> %s (rerouted=%s)",
                    update.changed_node,
                    update.new_state.value,
                    update.rerouted_edges,
                )

            # Run the local CNN vision classifier on this tick's camera
            # frame (synthetic in this reference deployment — see
            # app/ai/cnn_vision.py for wiring a real camera source).
            # Bias toward an anomalous-looking frame when the anomaly
            # processor is already flagging trouble, so the two signals
            # correlate the way they would on real hardware.
            frame_img = synthetic_frame(anomalous=anomaly.score > 1.0)
            vision_result = self.vision.infer(frame_img)

            frame = TelemetryFrame(
                node_id=self.node_id,
                timestamp=time.time(),
                cpu_pct=self._rng.uniform(15, 55),
                mem_pct=self._rng.uniform(25, 60),
                pqc_handshake_latency_ms=handshake_latency_ms,
                anomaly_score=anomaly.score,
                topology_snapshot=self.topology.snapshot(),
                vision_prediction=vision_result.predicted_class,
                vision_confidence=vision_result.confidence,
                vision_inference_ms=vision_result.inference_ms,
            )
            yield frame
            await asyncio.sleep(interval_s)

    def stop(self) -> None:
        self._running = False

    def force_fail_node(self, node_id: str) -> None:
        """Used by the chaos engineering module / copilot 'fail node X' command."""
        if node_id in self.topology.telemetry:
            t = self.topology.telemetry[node_id]
            t.cpu_pct = 100.0
            t.anomaly_score = 1.0
            t.last_heartbeat = 0.0
            t.missed_heartbeats = 4  # forces DOWN on the next liveness check
            self._manually_failed.add(node_id)

    def rejoin_node(self, node_id: str) -> None:
        """Clears an operator/chaos-injected fault and lets the node heartbeat again."""
        self._manually_failed.discard(node_id)
        self.topology.rejoin(node_id)
