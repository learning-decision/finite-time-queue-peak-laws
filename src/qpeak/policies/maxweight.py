"""MaxWeight scheduling for IQS via SciPy minimum-cost assignment on -Q."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from qpeak.models.bipartite_matching import BipartiteMatchingModel
from qpeak.models.gg1 import GG1Model
from qpeak.models.iqs import IQSModel
from qpeak.models.parallel_server import ParallelServerModel
from qpeak.registries import POLICY_REGISTRY, register


def maxweight_single_server(q: np.ndarray) -> np.ndarray:
    """Feasible service in ``{0,1}``; maximize ``q^T s`` (``d=1``)."""
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape != (1,):
        raise ValueError("GG1 expects q with shape (1,).")
    s = 1.0 if q[0] > 0 else 0.0
    return np.array([s], dtype=float)


def maxweight_permutation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Return an n×n permutation (0/1) matrix S maximizing ⟨Q, S⟩.

    Uses ``scipy.optimize.linear_sum_assignment(-Q)``, which is equivalent to a
    maximum-weight perfect matching on the complete bipartite graph with edge
    weights ``Q_ij``.

    **Tie-breaking:** SciPy returns *one* optimal matching. If several permutations
    share the same maximum weight, the choice is **implementation-defined**, not
    uniformly random over all max-weight permutations. See ``NON_NEGOTIABLES.md``.

    Complexity is **O(n³)** per call.
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


def maxweight_bipartite_node_matching(q: np.ndarray, model: BipartiteMatchingModel) -> np.ndarray:
    """
    MaxWeight for bipartite matching with node-queue state Q=(Q^L,Q^R) in R^{L+K}_+.

    Action is a 0/1 matching on edges with node degree constraints, represented here by the
    induced service vector S in R^{L+K}:
      S_L[l] = 1 iff left node l is matched; S_R[r] = 1 iff right node r is matched.

    Edge weights for matching are w(l,r) = Q^L[l] + Q^R[r].

    State-dependent feasibility (critical): only allow matching an edge (l,r) if
    both endpoint queues are strictly positive at the start of the slot. Otherwise
    matching would "waste" service on the empty endpoint and (because service here
    subtracts from both endpoints) can incorrectly cancel arrivals on the nonempty
    side under the shared recursion Q_{t+1}=(Q_t + A_{t+1} - S_t)^+.

    Implementation uses linear_sum_assignment on an augmented N×N assignment problem where
    N=max(L,K), with dummy nodes allowing unmatched left/right nodes at zero weight.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    L, K = int(model.L), int(model.K)
    if q.shape != (L + K,):
        raise ValueError("q shape must be (L+K,) for model.type bipartite_matching.")
    qL = q[:L]
    qR = q[L:]

    N = max(L, K)
    # Rows: left_aug (size N). Cols: right_aug (size N).
    # Use a large cost for forbidden edges; allowed edges use cost=-weight.
    BIG = 1e12
    cost = np.zeros((N, N), dtype=float)
    cost.fill(0.0)

    # Default: matching to dummy nodes has 0 cost (0 weight).
    # For real-left (rows < L) to real-right (cols < K), only allow edges in E.
    cost[:L, :K] = BIG
    for (l, r) in model.edges:
        if qL[l] <= 0.0 or qR[r] <= 0.0:
            # Not feasible to match if either endpoint is empty.
            continue
        w = float(qL[l] + qR[r])
        cost[l, r] = -w

    # Dummy left rows (L..N-1) to real right cols (0..K-1): 0 cost (unmatched rights)
    # Real left rows to dummy right cols (K..N-1): 0 cost (unmatched lefts)
    # Dummy-dummy: 0 cost

    row_ind, col_ind = linear_sum_assignment(cost)

    sL = np.zeros((L,), dtype=float)
    sR = np.zeros((K,), dtype=float)
    for rr, cc in zip(row_ind.tolist(), col_ind.tolist()):
        if rr < L and cc < K:
            # Matched on a real edge (must be allowed).
            if cost[rr, cc] >= BIG / 2:
                # Should not happen: unmatched left should choose dummy right at cost 0.
                continue
            sL[rr] = 1.0
            sR[cc] = 1.0

    return np.concatenate([sL, sR], axis=0)


def maxweight_parallel_server_assign(
    q: np.ndarray,
    model: ParallelServerModel,
    busy: np.ndarray,
) -> np.ndarray:
    """
    MaxWeight assignment of idle servers to non-empty compatible classes.

    Returns a length-K array ``assign`` where:
      - ``assign[k] == -1`` iff server k is idle and unassigned this slot.
      - ``assign[k] == l >= 0`` iff server k is newly assigned to class l this slot.
      - For busy servers (``busy[k] == 1``), ``assign[k]`` is set to ``-1`` (not used).

    Objective: max sum_k Q[assign(k)] over feasible assignments with capacity
    ``#{k : assign(k) = l} <= Q[l]`` and only edges in ``model.edges`` allowed.

    Weight is w(l, k) = Q[l] (no mu-weighting in v1).

    Implementation: extended bipartite assignment via scipy.linear_sum_assignment,
    replicating each class l into ``min(Q[l], n_idle)`` columns.
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

    # Adjacency: class l -> list of compatible servers
    adj_lk = [set() for _ in range(L)]
    for (l, k) in model.edges:
        adj_lk[l].add(int(k))

    # Build column list: each column is (class_label, copy_index); a class has
    # min(Q[l], n_idle) real copies. Append n_idle dummy columns (label = -1).
    class_cols: list[int] = []  # parallel list of class labels, one per real col
    for l in range(L):
        ql = int(q[l])
        if ql <= 0:
            continue
        n_copies = min(ql, n_idle)
        class_cols.extend([l] * n_copies)

    n_real = len(class_cols)
    n_cols = n_real + n_idle  # dummies guarantee n_cols >= n_rows = n_idle

    BIG = 1e12
    cost = np.full((n_idle, n_cols), BIG, dtype=float)
    # Dummy columns: cost 0 for every idle server
    cost[:, n_real:] = 0.0

    for i, k in enumerate(idle_servers):
        for j in range(n_real):
            l = class_cols[j]
            if k in adj_lk[l]:
                cost[i, j] = -float(q[l])

    row_ind, col_ind = linear_sum_assignment(cost)
    for i, k in enumerate(idle_servers):
        j = int(col_ind[i])
        if j < n_real:
            c = cost[i, j]
            if c < BIG / 2:
                assign[k] = int(class_cols[j])
    return assign


class ParallelServerMaxWeightPolicy:
    """
    Stateful MaxWeight for parallel_server. Owns per-server aux state
    (busy, remaining, assigned_class). Engine calls ``reset(rng)`` at the
    start of each replication and ``schedule(q, rng)`` once per slot.

    Per-slot order (inside ``schedule``):
      1. Tick: for busy servers, decrement ``remaining``; release those that hit 0.
      2. Match: assign idle servers to non-empty compatible classes (MaxWeight).
      3. For each newly assigned server k: set ``busy[k]=1``,
         ``remaining[k] = service_time.sample_duration(rng)``, ``assigned_class[k]=l``.

    Returns S in Z_+^L with ``S[l] =`` number of idle servers newly assigned
    to class l this slot; the engine applies ``Q_{t+1} = (Q_t + A_{t+1} - S)^+``.
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
        _ = rng  # reset is deterministic (all servers idle)
        self._busy.fill(0)
        self._remaining.fill(0)
        self._assigned_class.fill(-1)

    def schedule(self, q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        L, K = int(self._model.L), int(self._model.K)
        if q.shape != (L,):
            raise ValueError("q shape must be (L,) for model.type parallel_server.")

        # 1. Server tick: decrement remaining; release completions.
        for k in range(K):
            if self._busy[k]:
                self._remaining[k] -= 1
                if self._remaining[k] <= 0:
                    self._busy[k] = 0
                    self._assigned_class[k] = -1
                    self._remaining[k] = 0

        # 2. Match idle servers to non-empty compatible classes.
        assign = maxweight_parallel_server_assign(q, self._model, self._busy)

        # 3. Apply assignments: mark busy, sample service durations, build S.
        s = np.zeros((L,), dtype=float)
        for k in range(K):
            l = int(assign[k])
            if l >= 0:
                self._busy[k] = 1
                self._assigned_class[k] = l
                self._remaining[k] = int(self._model.service_time.sample_duration(rng))
                s[l] += 1.0
        return s


class MaxWeightPolicy:
    def __init__(self, model: IQSModel | GG1Model | BipartiteMatchingModel, params: dict[str, Any]):
        self._model = model
        self._params = params

    @property
    def name(self) -> str:
        return "maxweight"

    def describe(self) -> str:
        if isinstance(self._model, GG1Model):
            return "MaxWeight: single server, s=1 iff Q>0 else feasible s=0"
        if isinstance(self._model, BipartiteMatchingModel):
            return "MaxWeight: maximum-weight matching on bipartite graph with w(l,r)=Q_L(l)+Q_R(r)"
        return (
            "MaxWeight: scipy.optimize.linear_sum_assignment(-Q); "
            "one optimal matching, ties not uniform over all max-weight π"
        )

    def reset(self, rng: np.random.Generator) -> None:
        _ = rng  # stateless

    def schedule(self, q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        _ = rng  # SciPy matching is deterministic; rng kept for ``SchedulingPolicy`` API
        if isinstance(self._model, GG1Model):
            if q.shape != (1,):
                raise ValueError("q shape must be (1,) for model.type gg1.")
            return maxweight_single_server(q)
        if isinstance(self._model, BipartiteMatchingModel):
            return maxweight_bipartite_node_matching(q, self._model)
        if isinstance(self._model, IQSModel):
            if q.shape != (self._model.n, self._model.n):
                raise ValueError("q shape must match model.n × model.n.")
            return maxweight_permutation_matrix(q)
        raise TypeError(f"MaxWeightPolicy does not support model type {type(self._model).__name__!r}.")


@register(POLICY_REGISTRY, "maxweight")
def build_policy_maxweight(
    model: Any, params: dict[str, Any]
) -> MaxWeightPolicy | ParallelServerMaxWeightPolicy:
    if isinstance(model, ParallelServerModel):
        return ParallelServerMaxWeightPolicy(model, params)
    if not isinstance(model, (IQSModel, GG1Model, BipartiteMatchingModel)):
        raise TypeError(
            "policy maxweight requires model.type iqs, gg1, bipartite_matching, or parallel_server."
        )
    return MaxWeightPolicy(model, params)
