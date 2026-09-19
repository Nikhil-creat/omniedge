# OmniEdge Knowledge Base — Operational Runbook

## A node shows SUSPECT

SUSPECT means the node has missed at least two heartbeats, or the GNN
health scorer has flagged early degradation, but it has not yet crossed
the DOWN threshold of four missed heartbeats. No rerouting happens yet.
Check the node's CPU and memory telemetry in the dashboard; if both are
climbing steadily, expect it to transition to DOWN within the next few
heartbeat intervals. No operator action is required unless the node
stays SUSPECT for longer than five minutes, which usually indicates a
flapping network link rather than a genuine failure.

## A node shows DOWN

DOWN means the topology manager has already recomputed routes around
the node and broadcast a signed TopologyUpdate to every peer. Traffic
that used to pass through it now takes the shortest surviving path. The
dead node's slot stays reservable: if it heartbeats again with a valid
PQC session before its Membership Certificate expires, it rejoins
automatically without a full TPM re-attestation. Use the "Rejoin"
control in the dashboard, or the copilot command "rejoin node
<name>", to manually clear a fault you've confirmed is resolved.

## Injecting a test failure

Use the chaos engineering runner (`chaos/chaos_runner.py`) or the
dashboard's "Fail" button on any healthy node to simulate an outage.
Detection should complete within roughly 6 to 12 seconds depending on
heartbeat interval; if it consistently takes longer, check that the
liveness-check loop is actually running (it ties to the telemetry
producer task in `backend/app/main.py`).

## High anomaly score with no topology impact

A high neuromorphic anomaly score without a corresponding DOWN/SUSPECT
state usually means a local sensor or event source is behaving
unusually (e.g. a burst of malformed packets) but the node's mesh
connectivity itself is still healthy. Investigate the local event
source before assuming a network-level problem.

## Vision classifier flags "intrusion" or "fire_smoke"

The CNN edge-vision classifier ships with untrained, randomly
initialized weights in this reference deployment, so its predictions
are not meaningful until you load a real trained checkpoint via
`EdgeVisionService.load_weights()`. Treat vision-classifier output as a
demonstration of the inference pipeline's shape and latency, not as a
real detection signal, until real weights are loaded.

## PQC handshake latency spikes

A brief handshake latency spike right after a rekey event (every 10
minutes, or after 2^20 frames) is expected — a new Kyber encapsulation
and HKDF derivation costs more than the ChaCha20-Poly1305 encrypt/decrypt
of a normal telemetry frame. Sustained high latency outside of rekey
windows usually points to CPU contention on the node.
