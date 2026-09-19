# OmniEdge

An autonomous, post-quantum-secured, decentralized edge-AI mesh with a
spatial WebXR-style monitoring dashboard.

```
omniedge/
├── docs/ARCHITECTURE.md        System architecture & cryptographic spec
├── backend/                    Python: agent node, PQC, TPM, GNN topology, FastAPI gateway
│   └── app/ai/                 Agentic orchestrator, RAG knowledge base, CNN edge vision
├── frontend/                   React + Vite + Tailwind + Three.js dashboard (GitHub Pages)
├── chaos/                      Chaos-engineering self-healing verification
└── .github/workflows/          deploy.yml (Pages) + chaos-ci.yml (self-healing CI)
```

Read `docs/ARCHITECTURE.md` first — it explains *why* the system is built
this way: the Kyber/Dilithium hybrid handshake, the TPM 2.0 admission
flow, and how the GNN topology manager reroutes around failures.

## 1. Run the backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

This starts one real agent node (`edge-node-primary`) with four
simulated peers, admits itself to the mesh via the TPM attestation flow,
indexes the RAG knowledge base, and begins streaming PQC-encrypted
telemetry — now including a live CNN vision classification on every
tick.

Check it's alive:

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/mesh/snapshot | jq
curl -X POST http://localhost:8000/api/vision/infer -d '{}' -H 'Content-Type: application/json' | jq
curl -X POST http://localhost:8000/api/knowledge/query \
  -d '{"question": "what happens when a node shows DOWN"}' -H 'Content-Type: application/json' | jq
curl -X POST http://localhost:8000/api/copilot/command \
  -d '{"text": "fail node tokyo"}' -H 'Content-Type: application/json' | jq
```

By default (no `liboqs` installed) the crypto layer runs in its
**simulated backend** — functionally identical call shape to real
Kyber/Dilithium, but backed by X25519/Ed25519, and it says so loudly in
the logs. For real post-quantum crypto:

```bash
pip install liboqs-python
export OMNIEDGE_PQC_BACKEND=oqs
```

### Agentic AI, RAG, and CNN vision (`backend/app/ai/`)

- **CNN edge vision** (`cnn_vision.py`) — a small conv net (torch
  backend if installed, dependency-free NumPy fallback otherwise) runs
  on every telemetry tick, classifying a (synthetic, by default) camera
  frame into `normal` / `obstruction` / `intrusion` / `fire_smoke` /
  `low_visibility`. Ships with untrained demo weights — see
  `docs/ARCHITECTURE.md` §6 for how to load a real trained checkpoint.
- **RAG** (`rag.py`) — indexes `app/ai/knowledge/*.md` and
  `docs/ARCHITECTURE.md` into per-heading chunks and answers questions
  by similarity search (TF-IDF fallback, or `sentence-transformers` if
  installed). Query it directly at `POST /api/knowledge/query`.
- **Agentic orchestrator** (`agentic_orchestrator.py`) — the
  `/api/copilot/command` endpoint. Wires mesh control, vision
  inference, and RAG retrieval up as tools. Tries backends in order and
  falls through on failure, so the copilot never just errors out:
  1. **Claude** tool-calling loop, if `ANTHROPIC_API_KEY` is set (`pip
     install anthropic`).
  2. **Groq** tool-calling loop, if `GROQ_API_KEY` is set (`pip install
     groq`) — **free tier, no card required**: sign up at
     [console.groq.com](https://console.groq.com), create a key, and
     export it. Neither Claude nor Anthropic can generate a working key
     for you (it's tied to your own account), but this is the fastest
     no-cost way to get a real agentic loop running.
  3. **Deterministic rule-based planner** — zero external dependencies,
     so the reference stack is always fully runnable with no API keys
     at all.

```bash
pip install anthropic groq sentence-transformers   # optional, for the real backends
export ANTHROPIC_API_KEY=sk-ant-...   # preferred if you have one
export GROQ_API_KEY=gsk_...           # free-tier fallback/primary
```

## 2. Run the frontend

```bash
cd frontend
cp .env.example .env.local   # point at your local backend
npm install
npm run dev
```

Open the printed local URL. The dashboard performs the same
Kyber-hybrid handshake as the backend (see `src/lib/pqcChannel.js`) and
decrypts the live telemetry stream client-side. If no backend is
reachable, it falls back to a clearly-labeled local simulation so the
UI is still explorable.

Try the **Copilot** drawer: `"status"`, `"fail node berlin"`,
`"rejoin node berlin"`, `"what does the camera see"`, or a knowledge
question like `"how are session keys rotated"` — the drawer shows the
copilot's tool-call trace and, for knowledge answers, its source
citations, not just the final reply. There's also a standalone **CNN
edge vision** panel with a manual "Run inference" button.

## 3. Deploy the dashboard to GitHub Pages

1. Push this repo to GitHub.
2. Settings → Pages → Source → **GitHub Actions**.
3. (Optional) Settings → Secrets and variables → Actions → Variables:
   add `VITE_OMNIEDGE_API_URL` / `VITE_OMNIEDGE_WS_URL` pointing at your
   deployed backend (Pages only serves the static frontend — the
   FastAPI backend needs its own host, e.g. Fly.io/Render/a VM).
4. Push to `main`. `.github/workflows/deploy.yml` builds `frontend/`
   with the correct `/<repo-name>/` base path and publishes `dist/`.

Without a reachable backend, the deployed dashboard still works in
local-simulation mode — useful as a portfolio demo even without hosting
the Python side.

## 4. Run the chaos engineering suite

```bash
cd chaos
python3 chaos_runner.py --mode inprocess --rounds 5
# or, against a running backend:
pip install httpx
python3 chaos_runner.py --mode http --base-url http://localhost:8000 --rounds 5
```

It injects random node failures/partitions, measures how long the GNN
topology manager takes to detect and reroute around each one, and exits
non-zero if any fault isn't healed — wired into `chaos-ci.yml` so a
regression in the self-healing logic fails CI.

## Honesty notes on scope

This is a reference/portfolio-grade implementation, not an audited
production system:

- **PQC**: real Kyber/Dilithium via `liboqs` is supported but optional;
  the default runtime fallback is clearly labeled as *not* post-quantum
  secure. Read `docs/ARCHITECTURE.md` §2 before treating any deployment
  as quantum-resistant.
- **TPM**: `SimulatedTPM` models the *protocol*, not real silicon. Swap
  in `tpm2-pytss` against `/dev/tpm0` for real hardware attestation.
- **GNN**: uses `torch_geometric` if installed, otherwise an equivalent
  hand-rolled NumPy message-passing layer — same interface, much less
  capacity.
- **CNN vision**: real, correctly-shaped conv/pool/FC forward pass —
  but with randomly initialized, untrained weights and a synthetic
  frame source by default. Demonstrates the inference pipeline, not a
  trained detector. See `docs/ARCHITECTURE.md` §6 to load real weights
  and wire in a real camera.
- **Copilot**: `/api/copilot/command` is agentic — it tries Claude,
  then Groq, then a deterministic rule-based planner, falling through
  automatically if a backend isn't configured or errors out. All three
  drive the exact same tools (mesh control, CNN inference, RAG
  retrieval), so behavior is consistent regardless of which backend
  answered — see `backend/app/main.py` and
  `backend/app/ai/agentic_orchestrator.py`. Groq's free tier
  (console.groq.com) is the easiest zero-cost way to get a real
  tool-calling loop without an Anthropic key.
- **RAG**: the reference knowledge base is two short markdown files
  plus the architecture doc — enough to demonstrate a real chunk/embed/
  retrieve pipeline, not a production-scale corpus. The TF-IDF fallback
  embedder is dependency-free but weaker than a real dense-embedding
  model; install `sentence-transformers` for better retrieval quality
  on a larger knowledge base.
- **Chaos/failure injection** is simulated at the application layer
  (marking nodes down, cutting heartbeats) rather than real OS-level
  network partitions — appropriate for CI, not a substitute for
  infra-level chaos testing (e.g. Chaos Mesh, Toxiproxy) in a real
  multi-host deployment.

Every module above was actually executed and its behavior verified
(PQC session agreement, AEAD round-trip, TPM admission, anomaly
scoring, and — most importantly — live self-healing reroute on
simulated node failure) before being handed to you.

## 👤 About the Builder

**NIKHIL CHARY SRIRAMOJU**
B.Tech Final Year — Computer Science & Engineering

- 🔗 LinkedIn: [nikhil-chary-sriramoju](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a)
- 💻 GitHub: [Nikhil-creat](https://github.com/Nikhil-creat)
- 📧 Email: sriramojunikhil66@gmail.com
- 📸 Instagram: [nikhil__sriramoju](https://www.instagram.com/nikhil__sriramoju)

  
