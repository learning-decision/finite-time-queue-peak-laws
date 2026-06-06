"""Bipartite stochastic matching model (Ata–Xu): node-queue state in R^{L+K}_+."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qpeak.registries import MODEL_REGISTRY, register


@dataclass(frozen=True)
class BipartiteMatchingModel:
    """
    Bipartite matching network with node-queue state.

    - Left nodes: 0..L-1
    - Right nodes: 0..K-1
    - State Q_t = (Q^L, Q^R) in R^{L+K}_+
    """

    L: int
    K: int
    edges: tuple[tuple[int, int], ...]  # (l, r) pairs, 0-based
    epsilon: float

    @property
    def name(self) -> str:
        return "bipartite_matching"

    @property
    def dim(self) -> int:
        return int(self.L + self.K)

    def describe(self) -> str:
        return f"bipartite_matching L={self.L}, K={self.K}, |E|={len(self.edges)}, epsilon={self.epsilon}"


def _parse_edges(params: dict[str, Any], L: int, K: int) -> tuple[tuple[int, int], ...]:
    raw = params.get("edges")
    if not isinstance(raw, list) or len(raw) < 1:
        raise TypeError("model.edges must be a non-empty list of [l, r] pairs (0-based).")
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i, e in enumerate(raw):
        if not (isinstance(e, list) or isinstance(e, tuple)) or len(e) != 2:
            raise TypeError(f"model.edges[{i}] must be a length-2 list/tuple [l, r].")
        l, r = e[0], e[1]
        if not isinstance(l, int) or isinstance(l, bool):
            raise TypeError(f"model.edges[{i}][0] (l) must be an int.")
        if not isinstance(r, int) or isinstance(r, bool):
            raise TypeError(f"model.edges[{i}][1] (r) must be an int.")
        if not (0 <= l < L):
            raise ValueError(f"model.edges[{i}][0] out of range: l={l}, expected 0..{L-1}.")
        if not (0 <= r < K):
            raise ValueError(f"model.edges[{i}][1] out of range: r={r}, expected 0..{K-1}.")
        key = (l, r)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if not out:
        raise ValueError("model.edges must contain at least one unique edge.")
    return tuple(out)


@register(MODEL_REGISTRY, "bipartite_matching")
def build_model_bipartite_matching(params: dict[str, Any]) -> BipartiteMatchingModel:
    if "L" not in params or "K" not in params:
        raise KeyError("model.L and model.K are required for model.type 'bipartite_matching'.")
    L = int(params["L"])
    K = int(params["K"])
    if L < 1 or K < 1:
        raise ValueError("model.L and model.K must be positive integers.")
    eps = float(params["epsilon"])
    edges = _parse_edges(params, L=L, K=K)
    return BipartiteMatchingModel(L=L, K=K, edges=edges, epsilon=eps)

