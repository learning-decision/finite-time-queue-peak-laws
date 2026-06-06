"""Service-time distributions for persistent-server models (parallel_server)."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np


class ServiceTime(Protocol):
    """Distribution of per-customer service duration, in discrete slots (>= 1)."""

    @property
    def name(self) -> str: ...

    def describe(self) -> str: ...

    def sample_duration(self, rng: np.random.Generator) -> int:
        """Return a strictly positive integer service duration in slots."""
        ...


SERVICE_TIME_REGISTRY: dict[str, Callable[[dict[str, Any]], ServiceTime]] = {}


def register_service_time(key: str) -> Callable[
    [Callable[[dict[str, Any]], ServiceTime]],
    Callable[[dict[str, Any]], ServiceTime],
]:
    def decorator(
        fn: Callable[[dict[str, Any]], ServiceTime],
    ) -> Callable[[dict[str, Any]], ServiceTime]:
        SERVICE_TIME_REGISTRY[key] = fn
        return fn

    return decorator


def build_service_time(params: dict[str, Any]) -> ServiceTime:
    if not isinstance(params, dict):
        raise TypeError("model.service_time must be an object/mapping.")
    if "type" not in params:
        raise KeyError("model.service_time.type is required.")
    key = params["type"]
    if key not in SERVICE_TIME_REGISTRY:
        raise KeyError(
            f"Unknown service_time type {key!r}. Registered: {sorted(SERVICE_TIME_REGISTRY)}"
        )
    return SERVICE_TIME_REGISTRY[key](params)
