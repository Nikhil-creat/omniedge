"""
Self-healing topology manager.

Maintains a live mesh graph and scores each node's health via a small
message-passing ("GNN-style") update over neighbor telemetry. When a
node's health embedding collapses or it misses too many heartbeats, the
manager marks it DOWN, recomputes routes around it, and emits a signed
TopologyUpdate so every peer reroutes consistently.

Uses `torch_geometric` when available for a real graph-neural-network
forward pass; otherwise falls back to an equivalent pure-NumPy message
-passing implementation so this module has no hard GPU/torch dependency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv

    _HAS_PYG = True
except ImportError:  # pragma: no cover
    _HAS_PYG = False


HEARTBEAT_TIMEOUT_S = 8.0
SUSPECT_AFTER_MISSED = 2
DOWN_AFTER_MISSED = 4
FAILURE_SCORE_THRESHOLD = 0.75


class NodeState(str, Enum):
    HEALTHY = "HEALTHY"
    SUSPECT = "SUSPECT"
    DOWN = "DOWN"


@dataclass
class NodeTelemetry:
    node_id: str
    cpu_pct: float
    mem_pct: float
    handshake_latency_ms: float
    anomaly_score: float
    last_heartbeat: float = field(default_factory=time.time)
    missed_heartbeats: int = 0
    state: NodeState = NodeState.HEALTHY


@dataclass
class TopologyUpdate:
    changed_node: str
    new_state: NodeState
    rerouted_edges: List[Tuple[str, str]]
    timestamp: float

    def to_signed_payload(self, sign_fn) -> dict:
        body = {
            "changed_node": self.changed_node,
            "new_state": self.new_state.value,
            "rerouted_edges": self.rerouted_edges,
            "timestamp": self.timestamp,
        }
        message = json.dumps(body, sort_keys=True).encode()
        signature = sign_fn(message)
        return {"body": body, "signature_hex": signature.hex()}


class _NumpyMessagePasser:
    """Pure-NumPy 1-layer mean-aggregation GNN forward pass fallback."""

    def __init__(self, feature_dim: int = 4, hidden_dim: int = 8, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W_self = rng.normal(0, 0.4, size=(feature_dim, hidden_dim))
        self.W_neigh = rng.normal(0, 0.4, size=(feature_dim, hidden_dim))
        self.W_out = rng.normal(0, 0.4, size=(hidden_dim, 1))

    def forward(self, features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        deg = adjacency.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        neigh_mean = (adjacency @ features) / deg
        h = np.tanh(features @ self.W_self + neigh_mean @ self.W_neigh)
        health = 1.0 / (1.0 + np.exp(-(h @ self.W_out)))  # sigmoid -> [0,1] "healthiness"
        return health.flatten()


if _HAS_PYG:

    class _TorchSAGE(torch.nn.Module):  # type: ignore
        def __init__(self, feature_dim: int = 4, hidden_dim: int = 8) -> None:
            super().__init__()
            self.conv1 = SAGEConv(feature_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, 1)

        def forward(self, x, edge_index):
            h = F.tanh(self.conv1(x, edge_index))
            out = torch.sigmoid(self.conv2(h, edge_index))
            return out.flatten()


class GNNTopologyManager:
    """
    Owns the live mesh graph + per-node telemetry, runs periodic health
    scoring, and produces reroute decisions when a node fails.
    """

    def __init__(self, self_node_id: str) -> None:
        self.self_node_id = self_node_id
        self.graph = nx.Graph()
        self.graph.add_node(self_node_id)
        self.telemetry: Dict[str, NodeTelemetry] = {}
        self._np_model = _NumpyMessagePasser()
        self._torch_model = _TorchSAGE() if _HAS_PYG else None
        self.pending_updates: List[TopologyUpdate] = []

    # -- graph membership ---------------------------------------------
    def add_node(self, node_id: str, telemetry: NodeTelemetry) -> None:
        self.graph.add_node(node_id)
        self.telemetry[node_id] = telemetry

    def add_link(self, a: str, b: str, latency_ms: float, handshake_ok: bool) -> None:
        weight = latency_ms * (1.0 if handshake_ok else 5.0)
        self.graph.add_edge(a, b, weight=weight, latency_ms=latency_ms, handshake_ok=handshake_ok)

    def heartbeat(self, node_id: str, telemetry: NodeTelemetry) -> None:
        telemetry.last_heartbeat = time.time()
        telemetry.missed_heartbeats = 0
        if node_id in self.telemetry and self.telemetry[node_id].state != NodeState.HEALTHY:
            self._transition(node_id, NodeState.HEALTHY)
        self.telemetry[node_id] = telemetry

    # -- health scoring --------------------------------------------------
    def _feature_matrix(self) -> Tuple[List[str], np.ndarray]:
        node_ids = list(self.graph.nodes)
        rows = []
        for nid in node_ids:
            t = self.telemetry.get(nid)
            if t is None:
                rows.append([0.0, 0.0, 0.0, 0.0])
            else:
                rows.append(
                    [t.cpu_pct / 100.0, t.mem_pct / 100.0, t.handshake_latency_ms / 500.0, t.anomaly_score]
                )
        return node_ids, np.array(rows, dtype=np.float64)

    def score_health(self) -> Dict[str, float]:
        node_ids, features = self._feature_matrix()
        if len(node_ids) == 0:
            return {}
        adjacency = nx.to_numpy_array(self.graph, nodelist=node_ids)

        if _HAS_PYG and self._torch_model is not None and len(node_ids) > 1:
            x = torch.tensor(features, dtype=torch.float32)
            edge_index = torch.tensor(
                np.array(np.nonzero(adjacency)), dtype=torch.long
            )
            with torch.no_grad():
                health = self._torch_model(x, edge_index).numpy()
        else:
            health = self._np_model.forward(features, adjacency)

        return {nid: float(h) for nid, h in zip(node_ids, health)}

    # -- failure detection + self-healing ---------------------------
    def check_liveness_and_heal(self) -> List[TopologyUpdate]:
        """Call periodically (e.g. every 2s) from the agent's event loop."""
        now = time.time()
        health_scores = self.score_health()
        updates: List[TopologyUpdate] = []

        for node_id, t in list(self.telemetry.items()):
            if node_id == self.self_node_id:
                continue
            missed_interval = now - t.last_heartbeat
            if missed_interval > HEARTBEAT_TIMEOUT_S:
                t.missed_heartbeats += 1

            healthiness = health_scores.get(node_id, 1.0)
            failing = (1.0 - healthiness) >= FAILURE_SCORE_THRESHOLD

            if t.missed_heartbeats >= DOWN_AFTER_MISSED or (
                failing and t.missed_heartbeats >= SUSPECT_AFTER_MISSED
            ):
                if t.state != NodeState.DOWN:
                    rerouted = self._reroute_around(node_id)
                    self._transition(node_id, NodeState.DOWN)
                    updates.append(
                        TopologyUpdate(node_id, NodeState.DOWN, rerouted, time.time())
                    )
            elif t.missed_heartbeats >= SUSPECT_AFTER_MISSED and t.state == NodeState.HEALTHY:
                self._transition(node_id, NodeState.SUSPECT)
                updates.append(TopologyUpdate(node_id, NodeState.SUSPECT, [], time.time()))

        self.pending_updates.extend(updates)
        return updates

    def _transition(self, node_id: str, new_state: NodeState) -> None:
        if node_id in self.telemetry:
            self.telemetry[node_id].state = new_state

    def _reroute_around(self, dead_node: str) -> List[Tuple[str, str]]:
        """
        Remove the dead node from the routable subgraph and recompute
        shortest paths for every pair that previously traversed it,
        returning the set of new direct edges chosen as replacement hops.
        """
        if dead_node not in self.graph:
            return []
        neighbors = list(self.graph.neighbors(dead_node))
        subgraph = self.graph.copy()
        subgraph.remove_node(dead_node)

        rerouted: List[Tuple[str, str]] = []
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a, b = neighbors[i], neighbors[j]
                if a not in subgraph or b not in subgraph:
                    continue
                try:
                    path = nx.shortest_path(subgraph, a, b, weight="weight")
                    if len(path) == 2:  # direct alternate link already exists
                        continue
                    # Establish a new direct link to shortcut the reroute
                    est_latency = sum(
                        subgraph[u][v]["latency_ms"]
                        for u, v in zip(path[:-1], path[1:])
                    )
                    subgraph.add_edge(a, b, weight=est_latency, latency_ms=est_latency, handshake_ok=True)
                    rerouted.append((a, b))
                except nx.NetworkXNoPath:
                    continue

        # Commit the healed subgraph (dead node fully evicted from routing)
        self.graph = subgraph
        return rerouted

    def rejoin(self, node_id: str) -> None:
        """Re-admit a node that heartbeats again with a valid session."""
        if node_id in self.telemetry:
            self.telemetry[node_id].missed_heartbeats = 0
            self._transition(node_id, NodeState.HEALTHY)
        if node_id not in self.graph:
            self.graph.add_node(node_id)

    def snapshot(self) -> dict:
        health = self.score_health()
        return {
            "nodes": [
                {
                    "node_id": nid,
                    "state": t.state.value,
                    "cpu_pct": t.cpu_pct,
                    "mem_pct": t.mem_pct,
                    "handshake_latency_ms": t.handshake_latency_ms,
                    "anomaly_score": t.anomaly_score,
                    "health": round(health.get(nid, 1.0), 4),
                }
                for nid, t in self.telemetry.items()
            ],
            "edges": [
                {"a": a, "b": b, **data} for a, b, data in self.graph.edges(data=True)
            ],
        }
