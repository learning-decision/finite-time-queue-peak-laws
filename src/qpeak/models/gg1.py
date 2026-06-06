"""Discrete-time single-server queue (G/G/1-style): state is a length-1 vector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qpeak.registries import MODEL_REGISTRY, register

# State dimension for single-server queue.
GG1_DIM = 1


@dataclass(frozen=True)
class GG1Model:
    """
    Single-server queue in discrete time.

    Heavy-traffic default: Bernoulli arrival mean ``1 - epsilon`` per slot (when arrivals
    are model-derived). Optional explicit rates live under ``arrivals`` in config.
    """

    epsilon: float

    @property
    def name(self) -> str:
        return "gg1"

    @property
    def d(self) -> int:
        """State dimension ``Q_t in R^d``; here ``d=1``."""
        return GG1_DIM

    def describe(self) -> str:
        return f"GG/1 (discrete-time) d={self.d}, epsilon={self.epsilon}"


@register(MODEL_REGISTRY, "gg1")
def build_model_gg1(params: dict[str, Any]) -> GG1Model:
    return GG1Model(epsilon=float(params["epsilon"]))
