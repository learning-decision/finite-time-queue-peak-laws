"""Bernoulli arrivals for GG1: one potential arrival per slot."""

from __future__ import annotations

from typing import Any

import numpy as np

from qpeak.models.gg1 import GG1Model
from qpeak.registries import ARRIVAL_REGISTRY, register


class BernoulliGG1Arrivals:
    """
    Per-slot Bernoulli arrival with mean ``lambda``.

    Default: ``lambda = 1 - model.epsilon`` (heavy traffic towards unit service rate).
    Override: set ``arrivals.lambda`` in config (strictly in ``(0, 1)``).
    """

    def __init__(self, model: GG1Model, params: dict[str, Any]):
        self._model = model
        self._params = params
        if "lambda" in params:
            self._lam = float(params["lambda"])
        else:
            self._lam = 1.0 - model.epsilon

    @property
    def name(self) -> str:
        return "bernoulli_gg1"

    def describe(self) -> str:
        return f"bernoulli_gg1 lambda={self._lam:.6g}"

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        a = rng.binomial(1, self._lam, size=(1,)).astype(float)
        return a


@register(ARRIVAL_REGISTRY, "bernoulli_gg1")
def build_arrivals_bernoulli_gg1(model: Any, params: dict[str, Any]) -> BernoulliGG1Arrivals:
    if not isinstance(model, GG1Model):
        raise TypeError("bernoulli_gg1 requires model.type gg1.")
    return BernoulliGG1Arrivals(model, params)
