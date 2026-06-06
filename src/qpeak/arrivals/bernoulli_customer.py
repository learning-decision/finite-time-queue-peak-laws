"""Per-class Bernoulli customer arrivals for parallel_server model."""

from __future__ import annotations

from typing import Any

import numpy as np

from qpeak.models.parallel_server import ParallelServerModel
from qpeak.registries import ARRIVAL_REGISTRY, register


class BernoulliCustomerArrivals:
    """
    Class-level Bernoulli arrivals A_{t+1} in {0,1}^L with independent entries
    Bernoulli(lambdas[l]) per slot.

    If `lambdas` is omitted, defaults to the U1 uniform calibration
        lambda_l = (1 - eps) * mu * K / L
    where eps = model.epsilon and mu = model.service_time.mu. This lets the
    CLI epsilons-sweep work without requiring a lambdas list per epsilon.
    """

    def __init__(self, model: ParallelServerModel, params: dict[str, Any]):
        self._model = model
        self._params = params

        L = model.L
        if "lambdas" in params:
            raw = params["lambdas"]
            if not isinstance(raw, list) or len(raw) != L:
                raise TypeError("arrivals.lambdas must be a list of length L.")
            lam = np.array([float(x) for x in raw], dtype=float)
        else:
            eps = float(model.epsilon)
            mu = float(getattr(model.service_time, "mu"))
            val = (1.0 - eps) * mu * float(model.K) / float(L)
            lam = np.full((L,), val, dtype=float)
        if not (np.all(lam > 0.0) and np.all(lam < 1.0)):
            raise ValueError("All entries of arrivals.lambdas must lie strictly in (0, 1).")
        self._lam = lam

    @property
    def name(self) -> str:
        return "bernoulli_customer"

    def describe(self) -> str:
        return (
            f"bernoulli_customer class arrivals (L={self._model.L}); "
            f"mean(sum A)={float(np.sum(self._lam)):.6g}"
        )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.binomial(1, self._lam).astype(float)


@register(ARRIVAL_REGISTRY, "bernoulli_customer")
def build_arrivals_bernoulli_customer(
    model: Any, params: dict[str, Any]
) -> BernoulliCustomerArrivals:
    if not isinstance(model, ParallelServerModel):
        raise TypeError("bernoulli_customer requires model.type 'parallel_server'.")
    return BernoulliCustomerArrivals(model, params)
