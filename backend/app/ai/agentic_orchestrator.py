"""
Agentic AI orchestrator for the OmniEdge copilot.

Wraps the mesh's operational surface (topology control, CNN vision
inference, RAG knowledge retrieval) as a set of callable **tools**, and
exposes them through a single natural-language entry point,
`AgenticCopilot.handle(text)`, that:

  1. plans which tool(s) the request needs,
  2. executes them against the live `AgentNode`,
  3. composes a final answer grounded in the tool outputs,

and returns a full **trace** (Thought / Action / Observation, per tool
call) so the dashboard can show its reasoning — not just the answer.

Three operating modes, same public interface, tried in this order and
falling through on failure so the copilot degrades gracefully instead
of erroring out:

  1. **Claude tool-calling loop** — if `ANTHROPIC_API_KEY` is set and
     `anthropic` is installed.
  2. **Groq tool-calling loop** — if `GROQ_API_KEY` is set and `groq`
     is installed. Groq's free tier (https://console.groq.com — no
     card required, generous rate limits, low latency) makes this a
     good no-cost/low-friction way to get a real agentic loop running
     without an Anthropic key. Neither Claude nor Anthropic can
     generate a working Groq API key for you — it's tied to your own
     account — but once you have one, dropping it into `GROQ_API_KEY`
     is all this module needs.
  3. **Deterministic fallback planner** — otherwise (or if both LLM
     backends error out), a rule-based router picks a tool from the
     request's keywords. Zero external dependencies, so the reference
     stack is always fully runnable and testable with no API keys at
     all.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.ai.cnn_vision import EdgeVisionService, synthetic_frame
from app.ai.rag import KnowledgeBase

logger = logging.getLogger("omniedge.ai.agentic")

try:
    import anthropic

    _HAS_ANTHROPIC = True
except ImportError:  # pragma: no cover
    _HAS_ANTHROPIC = False

try:
    import groq

    _HAS_GROQ = True
except ImportError:  # pragma: no cover
    _HAS_GROQ = False


ANTHROPIC_MODEL = "claude-sonnet-4-6"
# llama-3.3-70b-versatile is Groq's strongest free-tier model with solid
# tool-calling support as of this writing; override via GROQ_MODEL if
# Groq deprecates/renames it (check https://console.groq.com/docs/models).
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


# --------------------------------------------------------------------------
# Tool definitions
# --------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    fn: Callable[..., dict]


@dataclass
class TraceStep:
    thought: str
    action: str
    action_input: dict
    observation: dict

    def to_dict(self) -> dict:
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class AgentResponse:
    reply: str
    trace: List[TraceStep] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)
    mode: str = "deterministic"

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "trace": [s.to_dict() for s in self.trace],
            "citations": self.citations,
            "mode": self.mode,
        }


class AgenticCopilot:
    """
    The multi-agent orchestrator. In practice this hosts three logical
    sub-agents behind one tool-calling surface:
      - a **topology agent**  (mesh_snapshot, fail_node, rejoin_node)
      - a **vision agent**    (run_vision_inference)
      - a **knowledge agent** (query_knowledge, backed by RAG)
    and either an LLM (Claude, then Groq) or the deterministic planner
    acts as the orchestrating "manager" that decides which sub-agent's
    tool(s) a given request needs.
    """

    def __init__(self, agent_node, knowledge_base: Optional[KnowledgeBase] = None) -> None:
        self.agent_node = agent_node
        self.kb = knowledge_base or KnowledgeBase()
        self.vision_service = EdgeVisionService(agent_node.node_id, use_torch=False)

        self.anthropic_client: Optional["anthropic.Anthropic"] = None
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if _HAS_ANTHROPIC and anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            logger.info("AgenticCopilot: Claude tool-calling loop enabled (%s)", ANTHROPIC_MODEL)

        self.groq_client: Optional["groq.Groq"] = None
        groq_key = os.environ.get("GROQ_API_KEY")
        if _HAS_GROQ and groq_key:
            self.groq_client = groq.Groq(api_key=groq_key)
            logger.info("AgenticCopilot: Groq tool-calling loop enabled (%s, free tier)", GROQ_MODEL)

        if not self.anthropic_client and not self.groq_client:
            logger.info(
                "AgenticCopilot: no ANTHROPIC_API_KEY or GROQ_API_KEY found; using the "
                "deterministic fallback planner. Get a free Groq key at "
                "https://console.groq.com to enable a real agentic loop at no cost."
            )

        self.tools: Dict[str, Tool] = {}
        self._register_tools()

    # -- tool registry ---------------------------------------------------

    def _register_tools(self) -> None:
        self._add_tool(
            "get_mesh_snapshot",
            "Get the current state of every node in the mesh: health status "
            "(HEALTHY/SUSPECT/DOWN), CPU, memory, handshake latency, and anomaly score.",
            {"type": "object", "properties": {}},
            self._tool_get_mesh_snapshot,
        )
        self._add_tool(
            "fail_node",
            "Inject a simulated fault into a named mesh node, forcing it toward the DOWN state "
            "so self-healing rerouting can be observed. Use only when the operator explicitly "
            "asks to fail, kill, or take down a node.",
            {
                "type": "object",
                "properties": {"node_id": {"type": "string", "description": "Exact node_id to fail."}},
                "required": ["node_id"],
            },
            self._tool_fail_node,
        )
        self._add_tool(
            "rejoin_node",
            "Clear a previously injected fault on a node and let it rejoin the mesh.",
            {
                "type": "object",
                "properties": {"node_id": {"type": "string", "description": "Exact node_id to rejoin."}},
                "required": ["node_id"],
            },
            self._tool_rejoin_node,
        )
        self._add_tool(
            "run_vision_inference",
            "Run the local CNN edge-vision classifier on the node's current camera frame and "
            "return the predicted category (normal/obstruction/intrusion/fire_smoke/low_visibility) "
            "with confidence. Use when the operator asks what a camera/site currently looks like.",
            {
                "type": "object",
                "properties": {
                    "simulate_anomaly": {
                        "type": "boolean",
                        "description": "For demo purposes only: bias the synthetic frame toward an anomalous pattern.",
                    }
                },
            },
            self._tool_run_vision_inference,
        )
        self._add_tool(
            "query_knowledge",
            "Search the OmniEdge architecture documentation and operational runbook for an "
            "answer to a question about how the system works or how to respond to an alert.",
            {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
            self._tool_query_knowledge,
        )

    def _add_tool(self, name: str, description: str, input_schema: dict, fn: Callable[..., dict]) -> None:
        self.tools[name] = Tool(name=name, description=description, input_schema=input_schema, fn=fn)

    # -- tool implementations ---------------------------------------------

    def _tool_get_mesh_snapshot(self) -> dict:
        return self.agent_node.topology.snapshot()

    def _tool_fail_node(self, node_id: str) -> dict:
        if node_id not in self.agent_node.topology.telemetry:
            return {"error": f"unknown node_id {node_id!r}", "known_nodes": list(self.agent_node.topology.telemetry.keys())}
        self.agent_node.force_fail_node(node_id)
        # Force an immediate liveness re-check so a status query right after
        # this call reflects the fault instead of waiting for the next
        # telemetry tick (which normally drives check_liveness_and_heal()).
        self.agent_node.topology.check_liveness_and_heal()
        return {"status": "fault_injected", "node_id": node_id}

    def _tool_rejoin_node(self, node_id: str) -> dict:
        if node_id not in self.agent_node.topology.telemetry:
            return {"error": f"unknown node_id {node_id!r}", "known_nodes": list(self.agent_node.topology.telemetry.keys())}
        self.agent_node.rejoin_node(node_id)
        return {"status": "rejoined", "node_id": node_id}

    def _tool_run_vision_inference(self, simulate_anomaly: bool = False) -> dict:
        frame = synthetic_frame(anomalous=simulate_anomaly)
        result = self.vision_service.infer(frame)
        return result.to_dict()

    def _tool_query_knowledge(self, question: str) -> dict:
        answer, retrieved = self.kb.answer_with_citations(question)
        return {
            "answer": answer,
            "citations": [{"chunk_id": r.chunk.chunk_id, "score": round(r.score, 4)} for r in retrieved],
        }

    def _dispatch(self, name: str, tool_input: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return tool.fn(**tool_input)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}

    # -- public entry point ------------------------------------------------

    def handle(self, text: str) -> AgentResponse:
        if self.anthropic_client is not None:
            try:
                return self._handle_with_anthropic(text)
            except Exception:
                logger.exception("Claude agentic loop failed; falling back")

        if self.groq_client is not None:
            try:
                return self._handle_with_groq(text)
            except Exception:
                logger.exception("Groq agentic loop failed; falling back to deterministic planner")

        return self._handle_deterministic(text)

    # -- real agentic tool-calling loop (Claude) ----------------------------

    def _anthropic_tool_specs(self) -> List[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self.tools.values()
        ]

    def _handle_with_anthropic(self, text: str, max_turns: int = 5) -> AgentResponse:
        system = self._system_prompt()
        messages: List[dict] = [{"role": "user", "content": text}]
        trace: List[TraceStep] = []
        citations: List[dict] = []

        for _ in range(max_turns):
            response = self.anthropic_client.messages.create(  # type: ignore
                model=ANTHROPIC_MODEL,
                max_tokens=1024,
                system=system,
                tools=self._anthropic_tool_specs(),
                messages=messages,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b.text for b in response.content if b.type == "text"]

            if not tool_uses:
                final_text = " ".join(text_blocks).strip() or "I wasn't able to produce an answer."
                return AgentResponse(reply=final_text, trace=trace, citations=citations, mode="claude-agentic")

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tool_use in tool_uses:
                observation = self._dispatch(tool_use.name, tool_use.input or {})
                if tool_use.name == "query_knowledge":
                    citations.extend(observation.get("citations", []))
                trace.append(
                    TraceStep(
                        thought=" ".join(text_blocks) or f"Deciding to call {tool_use.name}",
                        action=tool_use.name,
                        action_input=tool_use.input or {},
                        observation=observation,
                    )
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(observation),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return AgentResponse(
            reply="I made several tool calls but didn't reach a final answer in time — try a more specific request.",
            trace=trace,
            citations=citations,
            mode="claude-agentic",
        )

    # -- real agentic tool-calling loop (Groq, OpenAI-compatible API) -------

    def _openai_style_tool_specs(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self.tools.values()
        ]

    def _handle_with_groq(self, text: str, max_turns: int = 5) -> AgentResponse:
        messages: List[dict] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": text},
        ]
        trace: List[TraceStep] = []
        citations: List[dict] = []

        for _ in range(max_turns):
            response = self.groq_client.chat.completions.create(  # type: ignore
                model=GROQ_MODEL,
                max_tokens=1024,
                tools=self._openai_style_tool_specs(),
                messages=messages,
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                final_text = (message.content or "").strip() or "I wasn't able to produce an answer."
                return AgentResponse(reply=final_text, trace=trace, citations=citations, mode="groq-agentic")

            # Groq's chat-completions API (OpenAI-compatible) needs the
            # assistant's tool-call turn echoed back verbatim before the
            # tool results, so the model can see what it asked for.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    tool_input = {}
                observation = self._dispatch(tc.function.name, tool_input)
                if tc.function.name == "query_knowledge":
                    citations.extend(observation.get("citations", []))
                trace.append(
                    TraceStep(
                        thought=(message.content or "").strip() or f"Deciding to call {tc.function.name}",
                        action=tc.function.name,
                        action_input=tool_input,
                        observation=observation,
                    )
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(observation)}
                )

        return AgentResponse(
            reply="I made several tool calls but didn't reach a final answer in time — try a more specific request.",
            trace=trace,
            citations=citations,
            mode="groq-agentic",
        )

    # -- shared system prompt for both LLM backends --------------------------

    def _system_prompt(self) -> str:
        return (
            "You are the OmniEdge mesh operations copilot. You control a live "
            "decentralized edge-AI mesh through the tools available to you: "
            "inspecting mesh topology, injecting/clearing simulated node faults, "
            "running the on-node CNN vision classifier, and searching the "
            "architecture docs and runbook. Use tools to ground every factual "
            "claim; don't guess at node names or system behavior. Keep your "
            "final answer to a few sentences."
        )

    # -- deterministic fallback planner ------------------------------------

    def _find_node_mention(self, text: str) -> Optional[str]:
        for node_id in self.agent_node.topology.telemetry:
            short = node_id.split("-")[-1]
            if node_id in text or (short and short in text):
                return node_id
        return None

    def _handle_deterministic(self, text: str) -> AgentResponse:
        lower = text.lower().strip()
        trace: List[TraceStep] = []

        def step(thought: str, action: str, action_input: dict, observation: dict) -> None:
            trace.append(TraceStep(thought, action, action_input, observation))

        # A "why/what/how did X fail" question contains the word "fail"
        # but is asking for an explanation, not asking us to inject a
        # fault — route questions to the knowledge base instead of the
        # action handlers below.
        is_question = bool(re.search(r"\?\s*$|^\s*(why|what|how|when|where|who)\b", lower))

        if not is_question and re.search(r"\b(fail|kill|take down|crash)\b", lower):
            node_id = self._find_node_mention(lower)
            if not node_id:
                return AgentResponse(
                    reply="Which node should I fail? Try: 'fail node tokyo'.", trace=trace
                )
            obs = self._tool_fail_node(node_id)
            step(f"Operator asked to fail a node; matched '{node_id}'.", "fail_node", {"node_id": node_id}, obs)
            reply = f"Injected a fault on {node_id}. Watch the mesh self-heal on the dashboard."
            return AgentResponse(reply=reply, trace=trace)

        if not is_question and re.search(r"\b(rejoin|restore|heal|recover)\b", lower):
            node_id = self._find_node_mention(lower)
            if not node_id:
                return AgentResponse(
                    reply="Which node should rejoin? Try: 'rejoin node tokyo'.", trace=trace
                )
            obs = self._tool_rejoin_node(node_id)
            step(f"Operator asked to rejoin a node; matched '{node_id}'.", "rejoin_node", {"node_id": node_id}, obs)
            reply = f"{node_id} rejoined the mesh."
            return AgentResponse(reply=reply, trace=trace)

        if re.search(r"\b(camera|vision|frame|see|look|cnn|intrusion|obstruction)\b", lower):
            obs = self._tool_run_vision_inference(simulate_anomaly="anomal" in lower or "intrusion" in lower)
            step("Operator asked about the camera feed.", "run_vision_inference", {}, obs)
            reply = (
                f"Vision classifier predicts '{obs['predicted_class']}' "
                f"with {obs['confidence']*100:.1f}% confidence "
                f"({obs['inference_ms']:.1f}ms inference). Note: demo weights are untrained."
            )
            return AgentResponse(reply=reply, trace=trace)

        if re.search(r"\b(status|snapshot|health|how.*mesh|overview)\b", lower):
            obs = self._tool_get_mesh_snapshot()
            step("Operator asked for mesh status.", "get_mesh_snapshot", {}, obs)
            healthy = sum(1 for n in obs["nodes"] if n["state"] == "HEALTHY")
            reply = f"{healthy}/{len(obs['nodes'])} peer nodes healthy."
            return AgentResponse(reply=reply, trace=trace)

        # Default: treat it as a knowledge question and retrieve from the RAG KB
        obs = self._tool_query_knowledge(text)
        step("No operational intent matched; treating this as a knowledge-base question.", "query_knowledge", {"question": text}, obs)
        return AgentResponse(reply=obs["answer"], trace=trace, citations=obs["citations"])
