"""
Neuromorphic event-stream anomaly processor (simulator).

Models a spiking-neural-network-style leaky-integrate-and-fire (LIF)
layer that consumes an asynchronous event stream (e.g. sensor deltas,
packet inter-arrival events) and raises an anomaly score when the
membrane potential pattern deviates from a learned baseline. This is a
lightweight, dependency-free (NumPy only) simulation of the neuromorphic
processing style used on real event-based hardware (e.g. Intel Loihi /
IBM TrueNorth-class chips), suitable for demonstrating the mesh's
edge-inference pipeline without requiring specialized hardware.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np


@dataclass
class SpikeEvent:
    timestamp: float
    channel: int
    amplitude: float


@dataclass
class LIFNeuronLayer:
    """A small leaky-integrate-and-fire layer used as the anomaly scorer."""

    n_neurons: int = 32
    tau_membrane_ms: float = 20.0
    threshold: float = 1.0
    reset: float = 0.0
    leak_per_ms: float = field(init=False)
    potentials: np.ndarray = field(init=False)
    last_update: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.leak_per_ms = 1.0 - math.exp(-1.0 / self.tau_membrane_ms)
        self.potentials = np.zeros(self.n_neurons, dtype=np.float64)
        # Fixed random projection from input channel space to neurons,
        # standing in for a trained synaptic weight matrix.
        rng = np.random.default_rng(seed=1337)
        self.weights = rng.normal(loc=0.0, scale=0.6, size=(self.n_neurons,))

    def step(self, event: SpikeEvent) -> np.ndarray:
        now_ms = event.timestamp * 1000.0
        dt_ms = max(now_ms - self.last_update * 1000.0, 0.0)
        self.last_update = event.timestamp

        # Leak
        self.potentials *= max(0.0, 1.0 - self.leak_per_ms * (dt_ms / 10.0 + 0.01))

        # Integrate the incoming event, modulated by each neuron's synapse
        channel_phase = (event.channel % self.n_neurons)
        drive = np.zeros(self.n_neurons)
        drive[channel_phase] = event.amplitude * self.weights[channel_phase]
        self.potentials += drive

        # Fire + reset neurons that crossed threshold
        fired = self.potentials >= self.threshold
        spikes = fired.astype(np.float64)
        self.potentials[fired] = self.reset
        return spikes


@dataclass
class AnomalyScore:
    timestamp: float
    score: float  # 0.0 (nominal) .. 1.0+ (highly anomalous)
    spike_rate: float
    detail: str


class NeuromorphicAnomalyProcessor:
    """
    Consumes an event stream, runs it through a LIF layer, and computes a
    rolling anomaly score from spike-rate deviation vs. an exponentially
    weighted baseline. Designed to run per-node on constrained edge
    hardware (O(n_neurons) work per event, no batching required).
    """

    def __init__(
        self,
        n_neurons: int = 32,
        baseline_alpha: float = 0.02,
        window: int = 256,
    ) -> None:
        self.layer = LIFNeuronLayer(n_neurons=n_neurons)
        self.baseline_alpha = baseline_alpha
        self._baseline_rate: Optional[float] = None
        self._recent_rates: Deque[float] = deque(maxlen=window)

    def process_event(self, event: SpikeEvent) -> AnomalyScore:
        spikes = self.layer.step(event)
        rate = float(spikes.mean())
        self._recent_rates.append(rate)

        if self._baseline_rate is None:
            self._baseline_rate = rate
        else:
            self._baseline_rate = (
                self.baseline_alpha * rate + (1 - self.baseline_alpha) * self._baseline_rate
            )

        std = float(np.std(self._recent_rates)) if len(self._recent_rates) > 4 else 1e-6
        std = max(std, 1e-6)
        z = abs(rate - self._baseline_rate) / std
        score = min(float(z) / 4.0, 3.0)  # squash into a friendlier range

        detail = "nominal"
        if score > 1.5:
            detail = "critical deviation from spike-rate baseline"
        elif score > 0.8:
            detail = "elevated spike-rate deviation"

        return AnomalyScore(
            timestamp=event.timestamp, score=round(score, 4), spike_rate=rate, detail=detail
        )

    # -- Convenience: simulate an incoming event stream for demos --------
    def simulate_stream(
        self, n_events: int = 500, anomaly_at: Optional[int] = 400, seed: int = 7
    ) -> List[AnomalyScore]:
        rng = random.Random(seed)
        results: List[AnomalyScore] = []
        t0 = time.time()
        for i in range(n_events):
            is_anomalous_window = anomaly_at is not None and i >= anomaly_at
            channel = rng.randint(0, self.layer.n_neurons - 1)
            amplitude = rng.uniform(0.1, 0.5)
            if is_anomalous_window:
                # Inject a burst pattern to simulate e.g. a DoS/failing sensor
                amplitude *= rng.uniform(3.0, 6.0)
            event = SpikeEvent(timestamp=t0 + i * 0.01, channel=channel, amplitude=amplitude)
            results.append(self.process_event(event))
        return results
