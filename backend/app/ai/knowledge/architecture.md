# OmniEdge Knowledge Base — Architecture Reference

## Post-quantum handshake

OmniEdge secures every inter-node link with a hybrid Kyber768 (ML-KEM)
key encapsulation mechanism combined with an X25519 elliptic-curve
Diffie-Hellman exchange. The two shared secrets are mixed with
HKDF-SHA384 to derive a session key, which then protects traffic with
ChaCha20-Poly1305 AEAD encryption. Session keys are rotated every 10
minutes or after 2^20 frames, whichever comes first. All control-plane
messages, including topology updates and admission requests, are signed
with Dilithium3 (ML-DSA) so a compromised node cannot forge mesh state.

## TPM admission

A new edge node cannot join the mesh without completing a TPM 2.0
remote-attestation handshake: it presents an Endorsement Key
certificate, has its Attestation Key certified by that EK, proves
possession of the AK via TPM2_ActivateCredential, and produces a signed
Quote over PCRs 0 through 7. The Admission Authority compares the
quoted PCR digest against an approved "golden" measurement set. Only on
a match does it issue a Membership Certificate, valid for 24 hours,
binding the node's hardware identity to its Kyber and Dilithium mesh
keys.

## Self-healing topology

The GNN topology manager tracks every node's telemetry (CPU, memory,
handshake latency, anomaly score) as node features in a live graph and
scores each node's health with a small graph neural network. A node
that misses four consecutive heartbeats, or whose health score collapses
below a threshold while showing early signs of trouble, is marked DOWN.
The manager then removes it from the routable subgraph and recomputes
shortest augmenting paths for every route that previously traversed it,
broadcasting a signed TopologyUpdate so every peer reroutes consistently.
A node that heartbeats again with a valid PQC session can rejoin without
a full TPM re-attestation, as long as its Membership Certificate has not
expired.

## Neuromorphic anomaly processing

Each node runs a small leaky-integrate-and-fire spiking neuron layer
over its local event stream (sensor deltas, packet inter-arrival times).
The anomaly score is derived from how far the current spike rate
deviates from an exponentially-weighted baseline; a burst of unusually
large or frequent events — for example from a failing sensor or a
denial-of-service pattern — drives the score up sharply.

## Edge computer vision

Nodes with an attached camera or video feed run a small convolutional
neural network locally to classify frames into normal, obstruction,
intrusion, fire/smoke, or low-visibility categories. The architecture is
two convolution-and-pool stages feeding a fully connected softmax
classifier, sized to run on constrained edge hardware such as a
Raspberry-Pi-class device with no GPU required.

## Chaos engineering

The chaos-engineering module injects random node failures, network
partitions, and simulated resource exhaustion against a running mesh (or
an in-process agent, for CI) and measures how long the GNN topology
manager takes to detect and reroute around each fault, reporting a
pass/fail summary suitable for continuous integration.
