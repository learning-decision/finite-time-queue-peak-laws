"""IID Bernoulli arrivals for IQS (rates from model.epsilon)."""

from __future__ import annotations

from typing import Any

import numpy as np

from qpeak.models.iqs import IQSModel
from qpeak.registries import ARRIVAL_REGISTRY, register


class IIDBernoulliIQSArrivals:
    """Bernoulli arrivals with λ_ij = (1 - ε) / n (independent across i,j)."""

    def __init__(self, model: IQSModel, params: dict[str, Any]):
        self._model = model
        self._params = params
        self._lam = (1.0 - model.epsilon) / model.n

    @property
    def name(self) -> str:
        return "iid_bernoulli"

    def describe(self) -> str:
        return f"iid_bernoulli per-VOQ rate lam={(1.0 - self._model.epsilon) / self._model.n:.6g}"

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        n = self._model.n
        return rng.binomial(1, self._lam, size=(n, n)).astype(float)


@register(ARRIVAL_REGISTRY, "iid_bernoulli")
def build_arrivals_iid_bernoulli(model: Any, params: dict[str, Any]) -> IIDBernoulliIQSArrivals:
    if not isinstance(model, IQSModel):
        raise TypeError("iid_bernoulli in this package currently requires model.type iqs.")
    return IIDBernoulliIQSArrivals(model, params)
