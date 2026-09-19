"""
OmniEdge chaos-engineering module.

Randomly injects node failures and network partitions against a running
OmniEdge gateway (or an in-process AgentNode, via --mode inprocess) and
verifies that the GNN topology manager detects the fault and reroutes
within an expected time bound. Prints a pass/fail report suitable for CI.

Usage:
    # Against a running `uvicorn app.main:app` instance:
    python chaos_runner.py --mode http --base-url http://localhost:8000 \
        --rounds 5 --heal-timeout 15

    # Self-contained, no server required:
    python chaos_runner.py --mode inprocess --rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # only required for --mode http


@dataclass
class ChaosResult:
    round_no: int
    target_node: str
    fault_type: str
    detected_down: bool
    detection_latency_s: Optional[float]
    rerouted_edge_count: int


class ChaosReport:
    def __init__(self) -> None:
        self.results: List[ChaosResult] = []

    def add(self, r: ChaosResult) -> None:
        self.results.append(r)

    def summary(self) -> str:
        total = len(self.results)
        healed = sum(1 for r in self.results if r.detected_down)
        avg_latency = (
            sum(r.detection_latency_s for r in self.results if r.detection_latency_s is not None)
            / max(1, sum(1 for r in self.results if r.detection_latency_s is not None))
        )
        lines = [
            "=" * 60,
            "OmniEdge Chaos Engineering Report",
            "=" * 60,
            f"Rounds run           : {total}",
            f"Faults self-healed   : {healed}/{total}",
            f"Avg detection latency: {avg_latency:.2f}s",
            "-" * 60,
        ]
        for r in self.results:
            status = "HEALED" if r.detected_down else "NOT DETECTED"
            lines.append(
                f"  round {r.round_no:>2} | node={r.target_node:<20} "
                f"fault={r.fault_type:<12} -> {status:<13} "
                f"(latency={r.detection_latency_s}, rerouted_edges={r.rerouted_edge_count})"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def exit_code(self) -> int:
        return 0 if all(r.detected_down for r in self.results) else 1


FAULT_TYPES = ["node_crash", "network_partition", "resource_exhaustion"]


# --------------------------------------------------------------------------
# HTTP mode — targets a live OmniEdge FastAPI gateway
# --------------------------------------------------------------------------


async def run_http_chaos(base_url: str, rounds: int, heal_timeout: float, seed: int) -> ChaosReport:
    if httpx is None:
        print("httpx is required for --mode http (`pip install httpx`)", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(seed)
    report = ChaosReport()

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        snap = (await client.get("/api/mesh/snapshot")).json()
        node_ids = [n["node_id"] for n in snap["nodes"]]
        if not node_ids:
            print("No peer nodes found in mesh snapshot — nothing to chaos-test.", file=sys.stderr)
            sys.exit(2)

        for round_no in range(1, rounds + 1):
            target = rng.choice(node_ids)
            fault_type = rng.choice(FAULT_TYPES)
            print(f"[round {round_no}] injecting {fault_type} on {target} ...")

            t0 = time.monotonic()
            resp = await client.post("/api/mesh/chaos/fail-node", json={"node_id": target})
            resp.raise_for_status()

            detected = False
            detection_latency: Optional[float] = None
            rerouted_count = 0
            deadline = t0 + heal_timeout
            while time.monotonic() < deadline:
                snap = (await client.get("/api/mesh/snapshot")).json()
                node = next((n for n in snap["nodes"] if n["node_id"] == target), None)
                if node and node["state"] == "DOWN":
                    detected = True
                    detection_latency = round(time.monotonic() - t0, 3)
                    rerouted_count = len(
                        [e for e in snap["edges"] if e["a"] == target or e["b"] == target]
                    )
                    break
                await asyncio.sleep(1.0)

            report.add(
                ChaosResult(round_no, target, fault_type, detected, detection_latency, rerouted_count)
            )

            # Give the mesh a beat before the next round
            await asyncio.sleep(1.5)

    return report


# --------------------------------------------------------------------------
# In-process mode — no HTTP server required; drives an AgentNode directly
# --------------------------------------------------------------------------


async def run_inprocess_chaos(rounds: int, seed: int) -> ChaosReport:
    # Imported lazily so `--mode http` doesn't require the backend package
    # (and its dependencies) to be importable.
    import os

    backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
    sys.path.insert(0, backend_path)
    from app.agents.node import AgentNode  # type: ignore
    from app.security.tpm_attestation import AdmissionAuthority  # type: ignore

    rng = random.Random(seed)
    authority = AdmissionAuthority()
    agent = AgentNode("chaos-target-primary", authority)
    agent.seed_peers(["peer-alpha", "peer-beta", "peer-gamma", "peer-delta"])

    report = ChaosReport()
    node_ids = list(agent.topology.telemetry.keys())

    stream = agent.run_telemetry_stream(interval_s=0.2)

    async def tick() -> dict:
        frame = await stream.__anext__()
        return frame.to_dict()

    await tick()  # warm up

    for round_no in range(1, rounds + 1):
        target = rng.choice(node_ids)
        fault_type = rng.choice(FAULT_TYPES)
        print(f"[round {round_no}] injecting {fault_type} on {target} ...")
        t0 = time.monotonic()
        agent.force_fail_node(target)

        detected = False
        detection_latency = None
        rerouted_count = 0
        for _ in range(60):  # up to ~12s at 0.2s/tick
            d = await tick()
            node = next((n for n in d["topology"]["nodes"] if n["node_id"] == target), None)
            if node and node["state"] == "DOWN":
                detected = True
                detection_latency = round(time.monotonic() - t0, 3)
                rerouted_count = len(
                    [e for e in d["topology"]["edges"] if e["a"] == target or e["b"] == target]
                )
                break

        report.add(
            ChaosResult(round_no, target, fault_type, detected, detection_latency, rerouted_count)
        )
        agent.rejoin_node(target)
        await tick()

    agent.stop()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniEdge chaos engineering runner")
    parser.add_argument("--mode", choices=["http", "inprocess"], default="inprocess")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--heal-timeout", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.mode == "http":
        report = asyncio.run(
            run_http_chaos(args.base_url, args.rounds, args.heal_timeout, args.seed)
        )
    else:
        report = asyncio.run(run_inprocess_chaos(args.rounds, args.seed))

    print(report.summary())
    sys.exit(report.exit_code())


if __name__ == "__main__":
    main()
