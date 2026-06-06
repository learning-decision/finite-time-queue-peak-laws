"""n×n input-queued switch (IQS) model parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qpeak.registries import MODEL_REGISTRY, register


@dataclass(frozen=True)
class IQSModel:
    n: int
    epsilon: float

    @property
    def name(self) -> str:
        return "iqs"

    def describe(self) -> str:
        return f"IQS n={self.n}, epsilon={self.epsilon}"


@register(MODEL_REGISTRY, "iqs")
def build_model_iqs(params: dict[str, Any]) -> IQSModel:
    return IQSModel(n=int(params["n"]), epsilon=float(params["epsilon"]))
