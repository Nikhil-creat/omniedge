"""
CNN edge-vision inference service.

Each OmniEdge node is assumed to have a local camera/sensor feed (a
factory floor cam, a perimeter cam, a drone gimbal, etc.). This module
runs a small convolutional neural network locally on each node to
classify incoming frames into an operational category (normal /
obstruction / intrusion / fire-smoke / low-visibility), which feeds
into the same mesh telemetry stream as the neuromorphic anomaly score
and the GNN topology health so the dashboard has one unified picture of
"is this node/site okay".

Two backends, same interface, mirroring the honesty pattern used for
PQC/GNN elsewhere in this codebase:

  - `TorchCNN`  — a real `torch.nn` ConvNet, used if `torch` is
    installed. Supports loading trained weights via `load_weights()`.
  - `NumpyCNN`  — a pure-NumPy forward pass (conv2d + ReLU + maxpool +
    FC) with *no* torch dependency, used as the runnable-anywhere
    fallback.

IMPORTANT: in both backends the weights are randomly initialized unless
you call `load_weights(path)` with a real trained checkpoint. Out of
the box this module demonstrates a working, correctly-shaped inference
pipeline on synthetic frames — it is NOT a trained defect/intrusion
detector. See `docs/ARCHITECTURE.md` §6 for how to plug in real weights
and a real camera/RTSP frame source.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("omniedge.ai.cnn_vision")

CLASSES = ["normal", "obstruction", "intrusion", "fire_smoke", "low_visibility"]
FRAME_SIZE = 32  # grayscale, FRAME_SIZE x FRAME_SIZE, for a lightweight edge model

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    logger.info("torch not found; EdgeVisionService will use the NumPy CNN backend.")


# --------------------------------------------------------------------------
# NumPy backend (no hard dependency on torch)
# --------------------------------------------------------------------------


def _conv2d_single_channel(frame: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """Valid-mode 2D convolution for one input channel, one kernel."""
    kh, kw = kernel.shape
    h, w = frame.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            patch = frame[i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _maxpool2d(x: np.ndarray, size: int = 2) -> np.ndarray:
    h, w = x.shape
    out_h, out_w = h // size, w // size
    out = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.max(x[i * size : i * size + size, j * size : j * size + size])
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass
class NumpyCNN:
    """
    Architecture: Conv(1->4, 5x5) -> ReLU -> MaxPool(2) -> Conv(4->8, 5x5)
    -> ReLU -> MaxPool(2) -> Flatten -> FC -> softmax(len(CLASSES))

    Deliberately tiny (fits comfortably on constrained edge hardware,
    e.g. a Raspberry Pi class device) and pure NumPy so it runs with no
    GPU and no torch dependency.
    """

    n_classes: int = len(CLASSES)
    seed: int = 2024
    conv1_kernels: List[np.ndarray] = field(init=False)
    conv2_kernels: Dict[Tuple[int, int], np.ndarray] = field(init=False)
    fc_weights: Optional[np.ndarray] = field(init=False, default=None)
    fc_bias: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # 4 first-layer 5x5 kernels over the single grayscale input channel
        self.conv1_kernels = [rng.normal(0, 0.15, size=(5, 5)) for _ in range(4)]
        # 8 second-layer kernels, each consuming all 4 first-layer feature maps
        self.conv2_kernels = {
            (out_c, in_c): rng.normal(0, 0.15, size=(5, 5)) for out_c in range(8) for in_c in range(4)
        }
        self._fc_initialized = False

    def _forward_features(self, frame: np.ndarray) -> np.ndarray:
        # Layer 1
        feat1 = [_maxpool2d(_relu(_conv2d_single_channel(frame, k))) for k in self.conv1_kernels]
        # Layer 2: each output channel sums contributions across all input channels
        feat2 = []
        for out_c in range(8):
            acc = None
            for in_c in range(4):
                contrib = _conv2d_single_channel(feat1[in_c], self.conv2_kernels[(out_c, in_c)])
                acc = contrib if acc is None else acc + contrib
            feat2.append(_maxpool2d(_relu(acc)))
        flat = np.concatenate([f.flatten() for f in feat2])
        return flat

    def _ensure_fc(self, flat_dim: int) -> None:
        if self._fc_initialized:
            return
        rng = np.random.default_rng(self.seed + 1)
        self.fc_weights = rng.normal(0, 0.1, size=(flat_dim, self.n_classes))
        self.fc_bias = np.zeros(self.n_classes)
        self._fc_initialized = True

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """frame: (FRAME_SIZE, FRAME_SIZE) float32 in [0, 1]. Returns class probabilities."""
        flat = self._forward_features(frame)
        self._ensure_fc(flat.shape[0])
        logits = flat @ self.fc_weights + self.fc_bias
        return _softmax(logits)

    def load_weights(self, npz_path: str) -> None:
        """Load a real trained checkpoint (see docs/ARCHITECTURE.md §6 for the expected format)."""
        data = np.load(npz_path, allow_pickle=True)
        self.conv1_kernels = list(data["conv1_kernels"])
        self.conv2_kernels = {tuple(k): v for k, v in data["conv2_kernels"].item().items()}
        self.fc_weights = data["fc_weights"]
        self.fc_bias = data["fc_bias"]
        self._fc_initialized = True
        logger.info("Loaded trained CNN weights from %s", npz_path)


# --------------------------------------------------------------------------
# Torch backend (used automatically if torch is installed)
# --------------------------------------------------------------------------

if _HAS_TORCH:

    class TorchCNN(nn.Module):  # type: ignore
        def __init__(self, n_classes: int = len(CLASSES)) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, kernel_size=5)
            self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
            self.pool = nn.MaxPool2d(2)
            # 32x32 -> conv5 -> 28 -> pool -> 14 -> conv5 -> 10 -> pool -> 5
            self.fc = nn.Linear(8 * 5 * 5, n_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.flatten(1)
            return F.softmax(self.fc(x), dim=-1)

        def load_weights(self, path: str) -> None:
            self.load_state_dict(torch.load(path, map_location="cpu"))
            self.eval()
            logger.info("Loaded trained CNN weights from %s", path)


@dataclass
class VisionInferenceResult:
    node_id: str
    timestamp: float
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    inference_ms: float

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "class_probabilities": {k: round(v, 4) for k, v in self.class_probabilities.items()},
            "inference_ms": round(self.inference_ms, 3),
        }


class EdgeVisionService:
    """
    Runs CNN inference on camera frames local to a node. In this
    reference deployment, frames are synthetic (see `synthetic_frame`)
    since no physical camera is attached; wire a real RTSP/USB camera
    reader into `infer_from_camera` for production use.
    """

    def __init__(self, node_id: str, use_torch: bool = True) -> None:
        self.node_id = node_id
        self.backend_name = "torch" if (use_torch and _HAS_TORCH) else "numpy"
        if self.backend_name == "torch":
            self.model = TorchCNN()  # type: ignore
            self.model.eval()
        else:
            self.model = NumpyCNN()

    def infer(self, frame: np.ndarray) -> VisionInferenceResult:
        t0 = time.perf_counter()
        if self.backend_name == "torch":
            with torch.no_grad():  # type: ignore
                x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # type: ignore
                probs = self.model(x).numpy().flatten()  # type: ignore
        else:
            probs = self.model.forward(frame)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        top_idx = int(np.argmax(probs))
        return VisionInferenceResult(
            node_id=self.node_id,
            timestamp=time.time(),
            predicted_class=CLASSES[top_idx],
            confidence=float(probs[top_idx]),
            class_probabilities=dict(zip(CLASSES, (float(p) for p in probs))),
            inference_ms=inference_ms,
        )

    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)


def synthetic_frame(seed: Optional[int] = None, anomalous: bool = False) -> np.ndarray:
    """
    Generates a plausible-looking synthetic grayscale frame for demoing
    the vision pipeline without a real camera. `anomalous=True` injects
    a bright, sharp-edged region to loosely simulate an obstruction/
    intrusion silhouette so the demo endpoint has something interesting
    to classify.
    """
    rng = np.random.default_rng(seed)
    frame = rng.normal(0.35, 0.08, size=(FRAME_SIZE, FRAME_SIZE))
    if anomalous:
        y0, x0 = rng.integers(4, FRAME_SIZE - 12, size=2)
        h, w = rng.integers(6, 12, size=2)
        frame[y0 : y0 + h, x0 : x0 + w] += rng.uniform(0.4, 0.6)
    return np.clip(frame, 0.0, 1.0)
