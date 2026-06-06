"""Parallel-server / skill-based routing model (persistent servers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qpeak.registries import MODEL_REGISTRY, register
from qpeak.service_times.base import ServiceTime, build_service_time


@dataclass(frozen=True)
class ParallelServerModel:
    """
    Skill-based routing with persistent servers.

    - Left nodes: L customer classes (queues).
    - Right nodes: K persistent servers.
    - State visible to engine: Q in Z_+^L (customer queues only).
    - Auxiliary server state (busy/remaining/assigned_class) is owned by the policy.
    """

    L: int
    K: int
    edges: tuple[tuple[int, int], ...]  # (l, k) 0-based pairs
    service_time: ServiceTime
    epsilon: float

    @property
    def name(self) -> str:
        return "parallel_server"

    @property
    def dim(self) -> int:
        return int(self.L)

    def describe(self) -> str:
        return (
            f"parallel_server L={self.L}, K={self.K}, |E|={len(self.edges)}, "
            f"epsilon={self.epsilon}; {self.service_time.describe()}"
        )


def _parse_edges(params: dict[str, Any], L: int, K: int) -> tuple[tuple[int, int], ...]:
    raw = params.get("edges")
    if not isinstance(raw, list) or len(raw) < 1:
        raise TypeError("model.edges must be a non-empty list of [l, k] pairs (0-based).")
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i, e in enumerate(raw):
        if not (isinstance(e, list) or isinstance(e, tuple)) or len(e) != 2:
            raise TypeError(f"model.edges[{i}] must be a length-2 list/tuple [l, k].")
        l, k = e[0], e[1]
        if not isinstance(l, int) or isinstance(l, bool):
            raise TypeError(f"model.edges[{i}][0] (l) must be an int.")
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError(f"model.edges[{i}][1] (k) must be an int.")
        if not (0 <= l < L):
            raise ValueError(f"model.edges[{i}][0] out of range: l={l}, expected 0..{L-1}.")
        if not (0 <= k < K):
            raise ValueError(f"model.edges[{i}][1] out of range: k={k}, expected 0..{K-1}.")
        key = (l, k)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if not out:
        raise ValueError("model.edges must contain at least one unique edge.")
    return tuple(out)


@register(MODEL_REGISTRY, "parallel_server")
def build_model_parallel_server(params: dict[str, Any]) -> ParallelServerModel:
    if "L" not in params or "K" not in params:
        raise KeyError("model.L and model.K are required for model.type 'parallel_server'.")
    L = int(params["L"])
    K = int(params["K"])
    if L < 1 or K < 1:
        raise ValueError("model.L and model.K must be positive integers.")
    if "service_time" not in params:
        raise KeyError("model.service_time is required for model.type 'parallel_server'.")
    svc = build_service_time(params["service_time"])
    eps = float(params["epsilon"])
    edges = _parse_edges(params, L=L, K=K)
    return ParallelServerModel(L=L, K=K, edges=edges, service_time=svc, epsilon=eps)
