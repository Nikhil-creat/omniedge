import React, { useMemo } from "react";

/**
 * Small self-contained SVG line chart. Deliberately dependency-free
 * (no chart library) so the GitHub Pages bundle stays tiny; swap for
 * recharts/visx if you want richer interactions.
 */
function Sparkline({ values, color, height = 64, fillOpacity = 0.12 }) {
  const path = useMemo(() => {
    if (!values.length) return { line: "", area: "" };
    const w = 300;
    const h = height;
    const max = Math.max(...values, 1);
    const min = Math.min(...values, 0);
    const range = max - min || 1;
    const stepX = w / Math.max(values.length - 1, 1);

    const points = values.map((v, i) => {
      const x = i * stepX;
      const y = h - ((v - min) / range) * (h - 8) - 4;
      return [x, y];
    });

    const line = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const area = `${line} L${w},${h} L0,${h} Z`;
    return { line, area };
  }, [values, height]);

  return (
    <svg viewBox={`0 0 300 ${height}`} preserveAspectRatio="none" className="w-full" style={{ height }}>
      <path d={path.area} fill={color} opacity={fillOpacity} />
      <path d={path.line} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function MetricCard({ label, value, unit, values, color, formatter }) {
  return (
    <div className="rounded-lg border border-panelline bg-panel/60 p-4">
      <div className="flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wide text-mist">{label}</span>
        <span className="font-mono text-lg text-slate-100">
          {formatter ? formatter(value) : value?.toFixed?.(1) ?? "—"}
          <span className="ml-1 text-xs text-mist">{unit}</span>
        </span>
      </div>
      <div className="mt-2">
        <Sparkline values={values} color={color} />
      </div>
    </div>
  );
}

export default function TelemetryCharts({ history, latest }) {
  const cpu = history.map((f) => f.cpu_pct ?? 0);
  const mem = history.map((f) => f.mem_pct ?? 0);
  const latency = history.map((f) => f.pqc_handshake_latency_ms ?? 0);
  const anomaly = history.map((f) => f.anomaly_score ?? 0);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <MetricCard label="CPU" value={latest?.cpu_pct} unit="%" values={cpu} color="#38F5C8" />
      <MetricCard label="Memory" value={latest?.mem_pct} unit="%" values={mem} color="#7C6CF6" />
      <MetricCard
        label="PQC handshake latency"
        value={latest?.pqc_handshake_latency_ms}
        unit="ms"
        values={latency}
        color="#F5A623"
        formatter={(v) => v?.toFixed?.(2) ?? "—"}
      />
      <MetricCard
        label="Anomaly score"
        value={latest?.anomaly_score}
        unit=""
        values={anomaly}
        color="#F5456C"
        formatter={(v) => v?.toFixed?.(3) ?? "—"}
      />
    </div>
  );
}
