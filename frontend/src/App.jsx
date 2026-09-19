import React, { useCallback, useState } from "react";
import DigitalTwin3D from "./components/DigitalTwin3D.jsx";
import TelemetryCharts from "./components/TelemetryCharts.jsx";
import MeshNodeList from "./components/MeshNodeList.jsx";
import VisionPanel from "./components/VisionPanel.jsx";
import CopilotDrawer from "./components/CopilotDrawer.jsx";
import { useTelemetrySocket } from "./lib/useTelemetrySocket.js";

// Configure at build time via a `.env` file (see `.env.example`):
//   VITE_OMNIEDGE_API_URL=https://your-gateway.example.com
//   VITE_OMNIEDGE_WS_URL=wss://your-gateway.example.com/ws/telemetry
//   VITE_OMNIEDGE_NODE_ID=edge-node-primary
const API_BASE_URL = import.meta.env.VITE_OMNIEDGE_API_URL || "http://localhost:8000";
const WS_URL = import.meta.env.VITE_OMNIEDGE_WS_URL || "ws://localhost:8000/ws/telemetry";
const SERVER_NODE_ID = import.meta.env.VITE_OMNIEDGE_NODE_ID || "edge-node-primary";

export default function App() {
  const { connected, simulated, latest, history } = useTelemetrySocket({
    wsUrl: WS_URL,
    serverNodeId: SERVER_NODE_ID,
  });
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [manualVision, setManualVision] = useState(null);
  const [visionBusy, setVisionBusy] = useState(false);

  const handleFail = useCallback(
    async (nodeId) => {
      if (simulated) return;
      try {
        await fetch(`${API_BASE_URL}/api/mesh/chaos/fail-node`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ node_id: nodeId }),
        });
      } catch (err) {
        console.error("[omniedge] fail-node request failed", err);
      }
    },
    [simulated]
  );

  const handleRejoin = useCallback(
    async (nodeId) => {
      if (simulated) return;
      try {
        await fetch(`${API_BASE_URL}/api/copilot/command`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: `rejoin node ${nodeId}` }),
        });
      } catch (err) {
        console.error("[omniedge] rejoin request failed", err);
      }
    },
    [simulated]
  );

  const handleRunVisionInference = useCallback(async () => {
    if (simulated) return;
    setVisionBusy(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/vision/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ simulate_anomaly: Math.random() < 0.3 }),
      });
      const data = await res.json();
      setManualVision(data);
    } catch (err) {
      console.error("[omniedge] vision inference request failed", err);
    } finally {
      setVisionBusy(false);
    }
  }, [simulated]);

  const topology = latest?.topology ?? { nodes: [], edges: [] };
  const vision = manualVision ?? latest?.vision;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex items-center justify-between border-b border-panelline px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-md bg-gradient-to-br from-pulse to-signal" />
          <div>
            <h1 className="font-mono text-sm tracking-tight text-slate-100">OmniEdge Mesh Control</h1>
            <p className="text-xs text-mist">Post-quantum secured decentralized edge-AI mesh</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <StatusPill connected={connected} simulated={simulated} />
          <button
            onClick={() => setCopilotOpen(true)}
            className="rounded-lg border border-pulse/40 bg-pulse/10 px-3 py-1.5 text-xs font-medium text-slate-100 hover:bg-pulse/20 transition-colors"
          >
            Copilot
          </button>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[1.3fr_1fr] gap-4 p-4 lg:p-6">
        <section className="flex flex-col gap-4 min-h-[420px]">
          <div className="flex-1 rounded-xl border border-panelline bg-panel/40 overflow-hidden shadow-glow">
            <DigitalTwin3D topology={topology} />
          </div>
          <TelemetryCharts history={history} latest={latest} />
        </section>

        <section className="flex flex-col gap-4">
          <div className="rounded-xl border border-panelline bg-panel/40 p-4">
            <h2 className="mb-3 font-mono text-xs uppercase tracking-wide text-mist">Mesh nodes</h2>
            <MeshNodeList nodes={topology.nodes} onFail={handleFail} onRejoin={handleRejoin} />
          </div>
          <VisionPanel vision={vision} onRunInference={handleRunVisionInference} busy={visionBusy} />
          <div className="rounded-xl border border-panelline bg-panel/40 p-4 text-xs text-mist leading-relaxed">
            <p className="mb-1 font-mono text-slate-300">
              {simulated ? "Local simulation mode" : "Live gateway"}
            </p>
            <p>
              {simulated
                ? "No OmniEdge gateway was reachable, so this dashboard is rendering a local telemetry simulation. Run the FastAPI backend and set VITE_OMNIEDGE_WS_URL to see live, PQC-encrypted mesh data."
                : "Telemetry is streaming over a Kyber-hybrid-derived, ChaCha20-Poly1305-encrypted WebSocket channel."}
            </p>
          </div>
        </section>
      </main>

      <CopilotDrawer
        apiBaseUrl={API_BASE_URL}
        open={copilotOpen}
        onClose={() => setCopilotOpen(false)}
        meshApiUnreachable={simulated}
      />
    </div>
  );
}

function StatusPill({ connected, simulated }) {
  const label = !connected ? "Connecting…" : simulated ? "Simulated" : "Live · PQC secured";
  const color = !connected ? "bg-mist" : simulated ? "bg-warn" : "bg-signal";
  return (
    <span className="flex items-center gap-2 rounded-full border border-panelline px-3 py-1 text-xs text-slate-300">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}
