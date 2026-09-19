import React, { useRef, useState, useEffect } from "react";

/**
 * Chat drawer that posts natural-language operator commands to
 * POST /api/copilot/command (see backend/app/main.py). The backend now
 * routes this through the agentic orchestrator (app/ai/agentic_orchestrator.py),
 * which returns not just a reply but a reasoning trace (one entry per
 * tool call — mesh control, CNN vision, RAG knowledge lookup) and, for
 * knowledge questions, source citations. Falls back to a canned local
 * responder when no gateway is reachable, so the drawer is still
 * explorable in the static-only GitHub Pages demo.
 */
export default function CopilotDrawer({ apiBaseUrl, open, onClose, meshApiUnreachable }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "I'm the OmniEdge agentic copilot. I can check mesh status, fail/rejoin a node, run the CNN vision classifier, or answer questions from the architecture docs and runbook — try \"fail node tokyo\" or \"what happens when a node shows DOWN\".",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, open]);

  async function sendCommand(text) {
    setMessages((m) => [...m, { role: "user", text }]);
    setBusy(true);
    try {
      if (meshApiUnreachable) {
        throw new Error("offline");
      }
      const res = await fetch(`${apiBaseUrl}/api/copilot/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`gateway returned ${res.status}`);
      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: data.reply,
          trace: data.trace,
          citations: data.citations,
          mode: data.mode,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: "The mesh gateway isn't reachable from this static demo, so I can't act on that here. Run the FastAPI backend and set VITE_OMNIEDGE_API_URL to try live commands.",
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    sendCommand(text);
  }

  return (
    <div
      className={`fixed inset-y-0 right-0 z-40 flex w-full max-w-sm flex-col border-l border-panelline bg-panel/95 backdrop-blur transition-transform duration-300 ${
        open ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <div className="flex items-center justify-between border-b border-panelline px-4 py-3">
        <div>
          <p className="font-mono text-sm text-signal">Copilot</p>
          <p className="text-xs text-mist">Agentic mesh operations</p>
        </div>
        <button onClick={onClose} className="text-mist hover:text-slate-100" aria-label="Close copilot">
          ✕
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[90%] rounded-xl px-3 py-2 text-sm ${
                m.role === "user"
                  ? "bg-pulse/20 text-slate-100"
                  : "bg-void border border-panelline text-slate-200"
              }`}
            >
              <p>{m.text}</p>

              {m.trace?.length > 0 && (
                <details className="mt-2 border-t border-panelline pt-2">
                  <summary className="cursor-pointer text-[11px] font-mono text-mist hover:text-signal">
                    {m.mode === "llm-agentic" ? "Claude tool-calling trace" : "Reasoning trace"} (
                    {m.trace.length} step{m.trace.length > 1 ? "s" : ""})
                  </summary>
                  <ol className="mt-2 space-y-1.5">
                    {m.trace.map((step, si) => (
                      <li key={si} className="text-[11px] font-mono text-mist">
                        <span className="text-pulse">{step.action}</span>
                        {Object.keys(step.action_input ?? {}).length > 0 && (
                          <span className="text-mist"> {JSON.stringify(step.action_input)}</span>
                        )}
                      </li>
                    ))}
                  </ol>
                </details>
              )}

              {m.citations?.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1 border-t border-panelline pt-2">
                  {m.citations.map((c, ci) => (
                    <span
                      key={ci}
                      className="rounded border border-signal/20 bg-signal/5 px-1.5 py-0.5 text-[10px] font-mono text-signal"
                      title={`relevance ${c.score}`}
                    >
                      {c.chunk_id}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {busy && <div className="text-xs text-mist">Copilot is thinking…</div>}
      </div>

      <form onSubmit={handleSubmit} className="border-t border-panelline p-3 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. fail node berlin"
          className="flex-1 rounded-lg border border-panelline bg-void px-3 py-2 text-sm text-slate-100 placeholder:text-mist focus:outline-none focus:ring-2 focus:ring-pulse/50"
        />
        <button
          type="submit"
          disabled={busy}
          className="rounded-lg bg-pulse px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}

