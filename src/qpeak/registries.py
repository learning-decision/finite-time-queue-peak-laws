"""String-keyed factories for model / arrivals / policy blocks."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from qpeak.types import ArrivalProcess, QueueingModel, SchedulingPolicy

ModelFactory = Callable[[dict[str, Any]], QueueingModel]
ArrivalFactory = Callable[[QueueingModel, dict[str, Any]], ArrivalProcess]
PolicyFactory = Callable[[QueueingModel, dict[str, Any]], SchedulingPolicy]

MODEL_REGISTRY: dict[str, ModelFactory] = {}
ARRIVAL_REGISTRY: dict[str, ArrivalFactory] = {}
POLICY_REGISTRY: dict[str, PolicyFactory] = {}

T = TypeVar("T")


def register(registry: dict[str, T], key: str) -> Callable[[T], T]:
    def decorator(fn: T) -> T:
        registry[key] = fn  # type: ignore[assignment]
        return fn

    return decorator
