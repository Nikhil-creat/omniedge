import React from "react";

const CLASS_STYLES = {
  normal: "text-signal border-signal/30 bg-signal/10",
  obstruction: "text-warn border-warn/30 bg-warn/10",
  intrusion: "text-danger border-danger/30 bg-danger/10",
  fire_smoke: "text-danger border-danger/30 bg-danger/10",
  low_visibility: "text-warn border-warn/30 bg-warn/10",
};

export default function VisionPanel({ vision, onRunInference, busy }) {
  const predicted = vision?.predicted_class ?? "—";
  const confidence = vision?.confidence != null ? `${(vision.confidence * 100).toFixed(1)}%` : "—";

  return (
    <div className="rounded-xl border border-panelline bg-panel/40 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-mono text-xs uppercase tracking-wide text-mist">CNN edge vision</h2>
        <button
          onClick={onRunInference}
          disabled={busy}
          className="rounded-md border border-pulse/30 px-2 py-1 text-xs text-pulse hover:bg-pulse/10 transition-colors disabled:opacity-50"
        >
          {busy ? "Running…" : "Run inference"}
        </button>
      </div>
      <div className="flex items-center gap-3">
        <span
          className={`rounded-full border px-3 py-1 text-sm font-mono ${
            CLASS_STYLES[predicted] ?? "text-mist border-mist/20 bg-mist/10"
          }`}
        >
          {predicted}
        </span>
        <span className="text-xs text-mist">confidence {confidence}</span>
        {vision?.inference_ms != null && (
          <span className="text-xs text-mist">· {vision.inference_ms.toFixed(1)}ms</span>
        )}
      </div>
      <p className="mt-3 text-xs text-mist leading-relaxed">
        Demo weights are randomly initialized (untrained) — this panel demonstrates the live
        inference pipeline's shape and latency, not real detection accuracy.
      </p>
    </div>
  );
}
