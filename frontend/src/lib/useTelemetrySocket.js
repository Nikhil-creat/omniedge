import { useEffect, useRef, useState } from "react";
import { performHandshake } from "./pqcChannel.js";

const MAX_HISTORY = 120;

/**
 * Opens the OmniEdge telemetry WebSocket, performs the PQC-hybrid
 * handshake, and decrypts each incoming frame into a telemetry sample.
 * Falls back to a local simulator (clearly flagged) if no gateway is
 * reachable, so the dashboard is still explorable as a static demo.
 */
export function useTelemetrySocket({ wsUrl, serverNodeId }) {
  const [connected, setConnected] = useState(false);
  const [simulated, setSimulated] = useState(false);
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const channelRef = useRef(null);
  const wsRef = useRef(null);
  const simTimerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    function pushFrame(frame) {
      if (cancelled) return;
      setLatest(frame);
      setHistory((prev) => {
        const next = [...prev, frame];
        return next.length > MAX_HISTORY ? next.slice(next.length - MAX_HISTORY) : next;
      });
    }

    function startSimulation() {
      setSimulated(true);
      setConnected(true);
      let t = 0;
      const nodeIds = ["edge-node-berlin", "edge-node-tokyo", "edge-node-saopaulo", "edge-node-austin"];
      const visionClasses = ["normal", "obstruction", "intrusion", "fire_smoke", "low_visibility"];
      simTimerRef.current = setInterval(() => {
        t += 1;
        const wobble = (base, amp) => base + Math.sin(t / 6) * amp + (Math.random() - 0.5) * amp;
        const visionIsAnomalous = t % 53 === 0;
        pushFrame({
          node_id: "edge-node-primary (simulated)",
          timestamp: Date.now() / 1000,
          cpu_pct: Math.max(5, Math.min(95, wobble(35, 12))),
          mem_pct: Math.max(5, Math.min(95, wobble(42, 8))),
          pqc_handshake_latency_ms: Math.max(0.1, wobble(0.4, 0.2)),
          anomaly_score: Math.max(0, Math.min(1, 0.08 + (t % 47 === 0 ? 0.7 : 0) + Math.random() * 0.05)),
          vision: {
            predicted_class: visionIsAnomalous
              ? visionClasses[1 + Math.floor(Math.random() * 3)]
              : "normal",
            confidence: 0.25 + Math.random() * 0.15,
            inference_ms: 18 + Math.random() * 6,
          },
          topology: {
            nodes: nodeIds.map((id, i) => ({
              node_id: id,
              state: t % (40 + i * 10) < 3 ? "DOWN" : "HEALTHY",
              cpu_pct: wobble(30 + i * 8, 10),
              mem_pct: wobble(35 + i * 5, 8),
              handshake_latency_ms: wobble(25 + i * 5, 10),
              anomaly_score: Math.random() * 0.15,
              health: 0.6 + Math.random() * 0.4,
            })),
            edges: [
              { a: "edge-node-primary (simulated)", b: nodeIds[0] },
              { a: "edge-node-primary (simulated)", b: nodeIds[1] },
              { a: "edge-node-primary (simulated)", b: nodeIds[2] },
              { a: "edge-node-primary (simulated)", b: nodeIds[3] },
              { a: nodeIds[0], b: nodeIds[1] },
              { a: nodeIds[2], b: nodeIds[3] },
            ],
          },
        });
      }, 1200);
    }

    async function connect() {
      try {
        const ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        await new Promise((resolve, reject) => {
          ws.addEventListener("open", resolve, { once: true });
          ws.addEventListener("error", reject, { once: true });
          setTimeout(() => reject(new Error("connection timeout")), 4000);
        });

        const channel = await performHandshake(ws, { serverNodeId });
        channelRef.current = channel;
        if (cancelled) return;
        setConnected(true);
        setSimulated(false);

        ws.addEventListener("message", (event) => {
          if (!(event.data instanceof ArrayBuffer)) return; // JSON control frames handled separately
          try {
            const plaintext = channelRef.current.decrypt(new Uint8Array(event.data));
            const frame = JSON.parse(new TextDecoder().decode(plaintext));
            pushFrame(frame);
          } catch (err) {
            console.error("[omniedge] failed to decrypt telemetry frame", err);
          }
        });

        ws.addEventListener("close", () => {
          if (!cancelled) {
            setConnected(false);
            startSimulation();
          }
        });
      } catch (err) {
        console.warn("[omniedge] gateway unreachable, using local simulation:", err.message);
        startSimulation();
      }
    }

    connect();

    return () => {
      cancelled = true;
      if (simTimerRef.current) clearInterval(simTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [wsUrl, serverNodeId]);

  return { connected, simulated, latest, history };
}
