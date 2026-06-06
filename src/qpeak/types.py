"""Shared typing protocols for models, arrivals, and policies."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class QueueingModel(Protocol):
    """Network instance: state shape, dynamics hooks, geometry for policies."""

    @property
    def name(self) -> str: ...

    def describe(self) -> str:
        """Human-readable one-liner for logging."""
        ...


class ArrivalProcess(Protocol):
    """Model-aware arrival law: dimension and sampling match the given network."""

    @property
    def name(self) -> str: ...

    def describe(self) -> str: ...


class SchedulingPolicy(Protocol):
    """Chooses a feasible service matrix given the current queue state."""

    @property
    def name(self) -> str: ...

    def describe(self) -> str: ...

    def schedule(self, q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return feasible service matrix S (same shape as VOQ queue matrix q)."""
        ...
