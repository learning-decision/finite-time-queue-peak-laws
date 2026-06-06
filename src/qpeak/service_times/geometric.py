"""Geometric (Bernoulli-completion) service-time distribution."""

from __future__ import annotations

from typing import Any

import numpy as np

from qpeak.service_times.base import ServiceTime, register_service_time


class GeometricServiceTime:
    """
    Duration D ~ Geom(mu) on {1, 2, ...} with P(D=k) = (1-mu)^(k-1) mu, E[D] = 1/mu.

    Equivalent to: each slot the server is busy, it completes with independent
    probability mu (memoryless).
    """

    def __init__(self, mu: float):
        if not (0.0 < float(mu) <= 1.0):
            raise ValueError("service_time.mu must lie in (0, 1].")
        self._mu = float(mu)

    @property
    def name(self) -> str:
        return "geometric"

    @property
    def mu(self) -> float:
        return self._mu

    def describe(self) -> str:
        return f"geometric service times (mu={self._mu:g}, E[D]={1.0 / self._mu:g})"

    def sample_duration(self, rng: np.random.Generator) -> int:
        return int(rng.geometric(self._mu))


@register_service_time("geometric")
def build_service_time_geometric(params: dict[str, Any]) -> GeometricServiceTime:
    if "mu" not in params:
        raise KeyError("service_time.mu is required when service_time.type is 'geometric'.")
    return GeometricServiceTime(float(params["mu"]))
