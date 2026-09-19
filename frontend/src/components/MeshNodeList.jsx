import React from "react";

const STATE_STYLES = {
  HEALTHY: "bg-signal/15 text-signal border-signal/30",
  SUSPECT: "bg-warn/15 text-warn border-warn/30",
  DOWN: "bg-danger/15 text-danger border-danger/30",
};

export default function MeshNodeList({ nodes = [], onFail, onRejoin }) {
  if (!nodes.length) {
    return <p className="text-sm text-mist">Waiting for mesh telemetry…</p>;
  }

  return (
    <ul className="flex flex-col gap-2">
      {nodes.map((node) => (
        <li
          key={node.node_id}
          className="flex items-center justify-between rounded-lg border border-panelline bg-panel/50 px-3 py-2"
        >
          <div className="flex items-center gap-3 min-w-0">
            <span
              className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-mono ${
                STATE_STYLES[node.state] ?? "bg-mist/10 text-mist border-mist/20"
              }`}
            >
              {node.state}
            </span>
            <span className="truncate text-sm text-slate-200">{node.node_id}</span>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <span className="font-mono text-xs text-mist">
              {node.cpu_pct != null ? `${node.cpu_pct.toFixed(0)}% cpu` : ""}
            </span>
            {node.state === "DOWN" ? (
              <button
                onClick={() => onRejoin?.(node.node_id)}
                className="rounded-md border border-signal/30 px-2 py-1 text-xs text-signal hover:bg-signal/10 transition-colors"
              >
                Rejoin
              </button>
            ) : (
              <button
                onClick={() => onFail?.(node.node_id)}
                className="rounded-md border border-danger/30 px-2 py-1 text-xs text-danger hover:bg-danger/10 transition-colors"
              >
                Fail
              </button>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
