"""Build model, arrivals, and policy from a validated config dict."""

from __future__ import annotations

from typing import Any

from qpeak.registries import ARRIVAL_REGISTRY, MODEL_REGISTRY, POLICY_REGISTRY
from qpeak.types import ArrivalProcess, QueueingModel, SchedulingPolicy


def build_model(cfg: dict[str, Any]) -> QueueingModel:
    block = cfg["model"]
    key = block["type"]
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model type {key!r}. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](block)


def build_arrivals(model: QueueingModel, cfg: dict[str, Any]) -> ArrivalProcess:
    block = cfg["arrivals"]
    key = block["type"]
    if key not in ARRIVAL_REGISTRY:
        raise KeyError(f"Unknown arrivals type {key!r}. Registered: {sorted(ARRIVAL_REGISTRY)}")
    return ARRIVAL_REGISTRY[key](model, block)


def build_policy(model: QueueingModel, cfg: dict[str, Any]) -> SchedulingPolicy:
    block = cfg["policy"]
    key = block["type"]
    if key not in POLICY_REGISTRY:
        raise KeyError(f"Unknown policy type {key!r}. Registered: {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[key](model, block)


def compose_experiment(cfg: dict[str, Any]) -> tuple[QueueingModel, ArrivalProcess, SchedulingPolicy]:
    model = build_model(cfg)
    arrivals = build_arrivals(model, cfg)
    policy = build_policy(model, cfg)
    return model, arrivals, policy
