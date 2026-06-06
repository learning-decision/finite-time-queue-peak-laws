"""Bernoulli arrivals for bipartite matching: node-level Q in R^{L+K}."""

from __future__ import annotations

from typing import Any

import numpy as np

from qpeak.models.bipartite_matching import BipartiteMatchingModel
from qpeak.registries import ARRIVAL_REGISTRY, register


class BernoulliBipartiteArrivals:
    """
    Node-level Bernoulli arrivals for bipartite matching.

    Default (U1 side-load calibration using model.epsilon):
      Let m = min(L, K). Set
        lambda_L = (1 - eps) * m / L   on each left node
        lambda_R = (1 - eps) * m / K   on each right node

    Override:
      - arrivals.lambda_L: list[float] length L, entries in (0,1)
      - arrivals.lambda_R: list[float] length K, entries in (0,1)
    """

    def __init__(self, model: BipartiteMatchingModel, params: dict[str, Any]):
        self._model = model
        self._params = params

        L, K = model.L, model.K
        has_lam_L = "lambda_L" in params
        has_lam_R = "lambda_R" in params
        if has_lam_L and has_lam_R:
            # Both provided: use as-is.
            lam_L = params["lambda_L"]
            lam_R = params["lambda_R"]
            if not isinstance(lam_L, list) or len(lam_L) != L:
                raise TypeError("arrivals.lambda_L must be a list of length L when provided.")
            if not isinstance(lam_R, list) or len(lam_R) != K:
                raise TypeError("arrivals.lambda_R must be a list of length K when provided.")
            self._lam_L = np.array([float(x) for x in lam_L], dtype=float)
            self._lam_R = np.array([float(x) for x in lam_R], dtype=float)
        elif has_lam_R and not has_lam_L:
            # Asymmetric mode: fixed service rates on right side,
            # customer arrival rates on left computed from epsilon.
            # lambda_L[i] = (1 - eps) * fair_share_capacity[i]
            # where fair_share_capacity[i] = sum_{j:(i,j) in E} lambda_R[j] / deg_R(j)
            lam_R = params["lambda_R"]
            if not isinstance(lam_R, list) or len(lam_R) != K:
                raise TypeError("arrivals.lambda_R must be a list of length K when provided.")
            self._lam_R = np.array([float(x) for x in lam_R], dtype=float)
            eps = float(model.epsilon)
            # Compute right-node degrees (number of compatible left nodes per right node)
            deg_R = np.zeros(K, dtype=float)
            for (l, r) in model.edges:
                deg_R[r] += 1.0
            # Fair-share capacity for each left node
            capacity = np.zeros(L, dtype=float)
            for (l, r) in model.edges:
                capacity[l] += self._lam_R[r] / deg_R[r]
            self._lam_L = (1.0 - eps) * capacity
        elif has_lam_L and not has_lam_R:
            raise ValueError(
                "arrivals.lambda_L without arrivals.lambda_R is not supported. "
                "Set both, or set only lambda_R for asymmetric (parallel-server) mode."
            )
        else:
            m = min(L, K)
            eps = float(model.epsilon)
            lam_L = (1.0 - eps) * (m / L)
            lam_R = (1.0 - eps) * (m / K)
            self._lam_L = np.full((L,), lam_L, dtype=float)
            self._lam_R = np.full((K,), lam_R, dtype=float)

        if not (np.all(self._lam_L > 0.0) and np.all(self._lam_L < 1.0)):
            raise ValueError("All entries of lambda_L must lie strictly in (0,1).")
        if not (np.all(self._lam_R > 0.0) and np.all(self._lam_R < 1.0)):
            raise ValueError("All entries of lambda_R must lie strictly in (0,1).")

    @property
    def name(self) -> str:
        return "bernoulli_bipartite"

    def describe(self) -> str:
        L, K = self._model.L, self._model.K
        return (
            f"bernoulli_bipartite node arrivals (L={L},K={K}); "
            f"mean(sum A_L)={float(np.sum(self._lam_L)):.6g}, mean(sum A_R)={float(np.sum(self._lam_R)):.6g}"
        )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        aL = rng.binomial(1, self._lam_L).astype(float)
        aR = rng.binomial(1, self._lam_R).astype(float)
        return np.concatenate([aL, aR], axis=0)


@register(ARRIVAL_REGISTRY, "bernoulli_bipartite")
def build_arrivals_bernoulli_bipartite(model: Any, params: dict[str, Any]) -> BernoulliBipartiteArrivals:
    if not isinstance(model, BipartiteMatchingModel):
        raise TypeError("bernoulli_bipartite requires model.type 'bipartite_matching'.")
    return BernoulliBipartiteArrivals(model, params)

