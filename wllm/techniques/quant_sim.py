"""Int8 symmetric per-row weight quantization, simulated exactly.

Simulates the numerics of int8 weight-only quantization for linear
layers (row-wise absmax scaling, round-to-nearest, dequantized matmul)
without requiring specialized kernels, so quality impact is measurable
anywhere. Authenticity signal: ``layers_quantized`` — a candidate that
quantized nothing must not be reported as a quantization win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Matrix = list[list[float]]


def _quantize_row(row: list[float]) -> tuple[list[int], float]:
    absmax = max((abs(v) for v in row), default=0.0)
    if absmax == 0.0:
        return [0] * len(row), 1.0
    scale = absmax / 127.0
    q = [max(-127, min(127, round(v / scale))) for v in row]
    return q, scale


@dataclass
class QuantizedLinears:
    """A set of named linear layers with int8-simulated weights."""

    weights: dict[str, Matrix]
    layers_quantized: int = 0
    _q: dict[str, tuple[list[list[int]], list[float]]] = field(
        default_factory=dict, repr=False)

    def __post_init__(self):
        for name, w in self.weights.items():
            if not w or any(len(r) != len(w[0]) for r in w):
                raise ValueError(f"layer {name!r}: ragged or empty weight")
            rows, scales = [], []
            for r in w:
                q, s = _quantize_row(r)
                rows.append(q)
                scales.append(s)
            self._q[name] = (rows, scales)
            self.layers_quantized += 1

    def forward(self, name: str, x: list[float]) -> list[float]:
        """Dequantized matmul: y_i = scale_i * (q_i . x)."""
        rows, scales = self._q[name]
        if any(len(r) != len(x) for r in rows):
            raise ValueError(f"layer {name!r}: input dim mismatch")
        return [s * sum(qv * xv for qv, xv in zip(qrow, x))
                for qrow, s in zip(rows, scales)]

    def forward_exact(self, name: str, x: list[float]) -> list[float]:
        w = self.weights[name]
        return [sum(wv * xv for wv, xv in zip(row, x)) for row in w]

    def authenticity(self) -> dict[str, float]:
        return {"layers_quantized": float(self.layers_quantized)}

    def max_weight_error(self) -> float:
        worst = 0.0
        for name, w in self.weights.items():
            rows, scales = self._q[name]
            for row, qrow, s in zip(w, rows, scales):
                for v, q in zip(row, qrow):
                    worst = max(worst, abs(v - q * s))
        return worst
