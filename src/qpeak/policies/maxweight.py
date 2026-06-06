"""MaxWeight scheduling for IQS and parallel_server models."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from qpeak.models.iqs import IQSModel
from qpeak.models.parallel_server import ParallelServerModel
from qpeak.registries import POLICY_REGISTRY, register


def maxweight_permutation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Return an n×n permutation (0/1) matrix S maximising ⟨Q, S⟩.

    Uses ``scipy.optimize.linear_sum_assignment(-Q)``, equivalent to a
    maximum-weight perfect matching on the complete bipartite graph with
    edge weights ``Q_ij``.

    Tie-breaking: SciPy returns *one* optimal matching; the choice is
    implementation-defined, not uniformly random over all max-weight
    permutations. Complexity is O(n³) per call.
    """
    q = np.asarray(q, dtype=float)
    n = q.shape[0]
    if q.shape != (n, n):
        raise ValueError("q must be square (n×n VOQ matrix).")
    row_ind, col_ind = linear_sum_assignment(-q)
    pi = np.empty(n, dtype=int)
    pi[row_ind] = col_ind
    s = np.zeros((n, n), dtype=float)
    for i in range(n):
        s[i, pi[i]] = 1.0
    return s


def maxweight_parallel_server_assign(
    q: np.ndarray,
    model: ParallelServerModel,
    busy: np.ndarray,
) -> np.ndarray:
    """
    MaxWeight assignment of idle servers to non-empty compatible classes.

    Returns a length-K array ``assign`` where:
      - ``assign[k] == -1`` iff server k is idle and unassigned this slot.
      - ``assign[k] == l >= 0`` iff server k is newly assigned to class l.
      - For busy servers, ``assign[k]`` is set to ``-1`` (not used).

    Weight is w(l, k) = Q[l]. Implementation: extended bipartite assignment
    via scipy.linear_sum_assignment, replicating each class l into
    ``min(Q[l], n_idle)`` columns.
    """
    L, K = int(model.L), int(model.K)
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape != (L,):
        raise ValueError("q shape must be (L,) for model.type parallel_server.")

    assign = np.full((K,), -1, dtype=int)
    idle_servers = [int(k) for k in range(K) if int(busy[k]) == 0]
    n_idle = len(idle_servers)
    if n_idle == 0:
        return assign

    adj_lk = [set() for _ in range(L)]
    for (l, k) in model.edges:
        adj_lk[l].add(int(k))

    class_cols: list[int] = []
    for l in range(L):
        ql = int(q[l])
        if ql <= 0:
            continue
        class_cols.extend([l] * min(ql, n_idle))

    n_real = len(class_cols)
    n_cols = n_real + n_idle

    BIG = 1e12
    cost = np.full((n_idle, n_cols), BIG, dtype=float)
    cost[:, n_real:] = 0.0

    for i, k in enumerate(idle_servers):
        for j in range(n_real):
            l = class_cols[j]
            if k in adj_lk[l]:
                cost[i, j] = -float(q[l])

    row_ind, col_ind = linear_sum_assignment(cost)
    for i, k in enumerate(idle_servers):
        j = int(col_ind[i])
        if j < n_real and cost[i, j] < BIG / 2:
            assign[k] = int(class_cols[j])
    return assign


class ParallelServerMaxWeightPolicy:
    """
    Stateful MaxWeight for parallel_server. Owns per-server aux state
    (busy, remaining, assigned_class). Engine calls ``reset(rng)`` at the
    start of each replication and ``schedule(q, rng)`` once per slot.

    Per-slot order (inside ``schedule``):
      1. Tick: decrement ``remaining`` for busy servers; release those that hit 0.
      2. Match: assign idle servers to non-empty compatible classes (MaxWeight).
      3. For newly assigned server k: set busy, sample service duration, build S.

    Returns S in Z_+^L with ``S[l] =`` number of idle servers newly assigned
    to class l; the engine applies ``Q_{t+1} = (Q_t + A_{t+1} - S)^+``.
    """

    def __init__(self, model: ParallelServerModel, params: dict[str, Any]):
        self._model = model
        self._params = params
        K = int(model.K)
        self._busy = np.zeros(K, dtype=np.int8)
        self._remaining = np.zeros(K, dtype=np.int64)
        self._assigned_class = np.full(K, -1, dtype=np.int64)

    @property
    def name(self) -> str:
        return "maxweight"

    def describe(self) -> str:
        return (
            "MaxWeight (parallel_server): extended bipartite assignment, "
            "weight w(l,k)=Q[l]; persistent servers with per-class capacity Q[l]"
        )

    def reset(self, rng: np.random.Generator) -> None:
        _ = rng
        self._busy.fill(0)
        self._remaining.fill(0)
        self._assigned_class.fill(-1)

    def schedule(self, q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        L, K = int(self._model.L), int(self._model.K)
        if q.shape != (L,):
            raise ValueError("q shape must be (L,) for model.type parallel_server.")

        for k in range(K):
            if self._busy[k]:
                self._remaining[k] -= 1
                if self._remaining[k] <= 0:
                    self._busy[k] = 0
                    self._assigned_class[k] = -1
                    self._remaining[k] = 0

        assign = maxweight_parallel_server_assign(q, self._model, self._busy)

        s = np.zeros((L,), dtype=float)
        for k in range(K):
            l = int(assign[k])
            if l >= 0:
                self._busy[k] = 1
                self._assigned_class[k] = l
                self._remaining[k] = int(self._model.service_time.sample_duration(rng))
                s[l] += 1.0
        return s


class IQSMaxWeightPolicy:
    """MaxWeight for IQS: scipy linear_sum_assignment on -Q (O(n³) per slot)."""

    def __init__(self, model: IQSModel, params: dict[str, Any]):
        self._model = model
        self._params = params

    @property
    def name(self) -> str:
        return "maxweight"

    def describe(self) -> str:
        return (
            "MaxWeight (IQS): scipy.optimize.linear_sum_assignment(-Q); "
            "one optimal matching, ties not uniform over all max-weight permutations"
        )

    def reset(self, rng: np.random.Generator) -> None:
        _ = rng

    def schedule(self, q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        _ = rng
        if q.shape != (self._model.n, self._model.n):
            raise ValueError("q shape must match model.n × model.n.")
        return maxweight_permutation_matrix(q)


@register(POLICY_REGISTRY, "maxweight")
def build_policy_maxweight(
    model: Any, params: dict[str, Any]
) -> IQSMaxWeightPolicy | ParallelServerMaxWeightPolicy:
    if isinstance(model, ParallelServerModel):
        return ParallelServerMaxWeightPolicy(model, params)
    if isinstance(model, IQSModel):
        return IQSMaxWeightPolicy(model, params)
    raise TypeError(
        f"policy maxweight requires model.type iqs or parallel_server; "
        f"got {type(model).__name__!r}."
    )
