"""Discrete-time queue simulation and replicated aggregate recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from qpeak.compose import compose_experiment
from qpeak.models.iqs import IQSModel
from qpeak.models.parallel_server import ParallelServerModel
from qpeak.recording import write_replication_path
from qpeak.simulation.fast_parallel_server import run_fast_parallel_server_ring_if_supported


def _initial_queue_state(model: Any) -> tuple[np.ndarray, int]:
    """
    Return ``(q0, n_meta)`` for the first slot.

    ``n_meta`` is IQS port count for IQS, or state dimension ``d`` for vector models (e.g. GG1).
    """
    if isinstance(model, IQSModel):
        n = model.n
        return np.zeros((n, n), dtype=float), n
    if isinstance(model, ParallelServerModel):
        d = model.dim  # == L
        return np.zeros((d,), dtype=float), d
    raise NotImplementedError(
        f"Simulation initial state is not implemented for model {type(model).__name__!r}. "
        "Add a branch in qpeak.simulation.engine._initial_queue_state."
    )

def _evenly_spaced_t_indices(T: int, K: int) -> np.ndarray:
    if K < 2:
        raise ValueError("K must be >= 2.")
    if K > T:
        raise ValueError("K cannot exceed T.")
    # Include both endpoints: 0 and T-1.
    return np.unique(np.round(np.linspace(0, T - 1, num=K)).astype(int))


def _metrics_t_indices(cfg: dict[str, Any]) -> np.ndarray:
    T = int(cfg["simulation"]["T"])
    metrics = cfg.get("metrics", {})
    ds = metrics.get("downsample")
    if ds is None:
        return np.arange(T, dtype=int)
    K = int(ds["K"])
    t_idx = _evenly_spaced_t_indices(T, K)
    if t_idx.size < 2:
        # Extremely small T/K rounding edge case (shouldn't happen with validator K<=T and K>=2)
        raise RuntimeError("Downsample grid collapsed to <2 unique indices; adjust K.")
    return t_idx


def _fro_norm(q: np.ndarray) -> float:
    # Frobenius norm for matrix state (np.linalg.norm default is Frobenius for 2D).
    return float(np.linalg.norm(q))


_SERIES_REGISTRY: dict[str, tuple[str, Any]] = {
    # name -> (kind, fn) where kind is "level" (value at time t) or "peak" (running max).
    "totalQ": ("level", lambda q: float(np.sum(q))),
    "q_norm_l1": ("level", lambda q: float(np.sum(np.abs(q)))),
    "q_norm_l2": ("level", _fro_norm),
    "peak_totalQ_so_far": ("peak", lambda q: float(np.sum(q))),
    "peak_l1_so_far": ("peak", lambda q: float(np.sum(np.abs(q)))),
    "peak_l2_so_far": ("peak", _fro_norm),
}


def _build_series_registry(model: Any) -> dict[str, tuple[str, Any]]:
    """Extend the global series registry with model-specific metrics."""
    return dict(_SERIES_REGISTRY)


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory of VOQ queues: ``q_history[t]`` is :math:`Q_t` at the start of slot ``t``."""

    q_history: np.ndarray  # shape (T,) + per-slot queue shape; IQS is (T, n, n)
    T: int
    n: int
    seed: int
    model_name: str
    arrivals_name: str
    policy_name: str


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{ss:02d}s"
    if m > 0:
        return f"{m:d}m{ss:02d}s"
    return f"{ss:d}s"


def _progress_line(msg: str) -> None:
    """
    Single-line refresh progress output.

    Uses carriage return and clears to end-of-line so updates do not scroll.
    """
    # \x1b[K clears from cursor to end of line (ANSI). \r returns to column 0.
    print("\r" + msg + "\x1b[K", end="", flush=True)


def _progress_done_line(msg: str) -> None:
    """Finalize the single-line progress output with a newline."""
    _progress_line(msg)
    print("", flush=True)


def run_simulation(cfg: dict[str, Any]) -> SimulationResult:
    """
    Run the slot recursion for ``cfg['simulation']['T']`` steps.

    Uses whatever ``compose_experiment`` builds: ``policy.schedule(q, rng)`` and
    ``arrivals.sample(rng)`` must exist and return arrays broadcastable with ``q``.

    Dynamics per NON_NEGOTIABLES: observe :math:`Q_t`, choose feasible :math:`S_t`,
    sample arrivals :math:`A_{t+1}`, then :math:`Q_{t+1}=(Q_t+A_{t+1}-S_t)^+`.
    """
    sim = cfg["simulation"]
    T = int(sim["T"])
    seed = int(sim["seed"])

    model, arrivals, policy = compose_experiment(cfg)
    q, n = _initial_queue_state(model)

    ss = np.random.SeedSequence(seed)
    arr_seq, pol_seq = ss.spawn(2)
    rng_arr = np.random.default_rng(arr_seq)
    rng_pol = np.random.default_rng(pol_seq)
    policy.reset(rng_pol)

    q_history = np.zeros((T,) + q.shape, dtype=float)

    progress = bool(sim.get("progress", False))
    progress_every = int(sim.get("progress_every", 1_000_000))
    if progress_every < 1:
        progress_every = 1
    if progress:
        start = time.perf_counter()
        last_t = 0
        last_wall = start
        _progress_line(f"[progress] begin: T={T} every={progress_every}")

    for t in range(T):
        q_history[t] = q
        s = policy.schedule(q, rng_pol)
        a = arrivals.sample(rng_arr)
        q = np.maximum(0.0, q + a - s)
        if progress and (t + 1 == T or ((t + 1) % progress_every == 0)):
            now = time.perf_counter()
            dt_slots = (t + 1) - last_t
            dt_wall = max(1e-12, now - last_wall)
            rate = dt_slots / dt_wall
            frac = (t + 1) / max(1, T)
            eta = (T - (t + 1)) / max(1e-12, rate)
            _progress_line(
                f"[progress] t={t + 1}/{T}  {100 * frac:6.2f}%  "
                f"{rate:10.1f} slots/s  ETA {_format_eta(eta)}"
            )
            last_t = t + 1
            last_wall = now
    if progress:
        elapsed = time.perf_counter() - start
        _progress_done_line(f"[progress] done: elapsed={_format_eta(elapsed)}")

    return SimulationResult(
        q_history=q_history,
        T=T,
        n=n,
        seed=seed,
        model_name=model.name,
        arrivals_name=arrivals.name,
        policy_name=policy.name,
    )


@dataclass(frozen=True)
class AggregateResult:
    """Aggregate time series across replications (mean + standard error)."""

    t_indices: np.ndarray  # shape (K,)
    arrays: dict[str, np.ndarray]  # each shape (K,)


def run_replications_and_write_aggregate(cfg: dict[str, Any], output_dir: Path) -> Path:
    """
    Run ``R`` replications and write ``aggregate.npz`` under ``output_dir``.

    Always writes (on the chosen time grid):
      - E_peak_l1_so_far, SE_peak_l1_so_far
      - E_peak_l2_so_far, SE_peak_l2_so_far

    Additional series may be requested via ``metrics.series`` using names in
    ``_SERIES_REGISTRY``. Each requested series is stored as mean + SE with keys
    ``E_<name>`` and ``SE_<name>``.

    Accuracy contract: peak series at a stored index t_k equal the true running maxima
    over the full underlying path up to t_k (even if we only emit them at t_indices).
    """
    sim = cfg["simulation"]
    T = int(sim["T"])
    seed = int(sim["seed"])
    R = int(sim.get("num_replications", 10))

    model, arrivals, policy = compose_experiment(cfg)
    q0, n = _initial_queue_state(model)
    series_registry = _build_series_registry(model)

    t_indices = _metrics_t_indices(cfg)
    K = int(t_indices.size)
    t_set = set(int(x) for x in t_indices.tolist())

    recording = cfg.get("recording", {})
    if not isinstance(recording, dict):
        raise TypeError("recording must be an object/mapping when present.")
    save_full_state = bool(recording.get("full_state", True))
    P = int(recording.get("num_paths_to_save", min(R, 10)))
    if P < 0:
        raise ValueError("recording.num_paths_to_save must be >= 0.")
    P = min(P, R)

    # Per-path derived series: only those explicitly listed in metrics.series.
    per_path_series: list[str] = list(cfg.get("metrics", {}).get("series", []))

    # Always-on peak series
    series_names: list[str] = ["peak_l1_so_far", "peak_l2_so_far"]
    requested = cfg.get("metrics", {}).get("series", [])
    for name in requested:
        if name in ("peak_l1_so_far", "peak_l2_so_far"):
            continue
        if name not in series_registry:
            raise KeyError(
                f"Unknown metrics.series name {name!r}. "
                f"Allowed: {sorted(k for k in series_registry if not k.startswith('peak_l'))}"
            )
        series_names.append(name)

    peak_needed: list[str] = []
    for name in list(dict.fromkeys(series_names + per_path_series)):
        if name not in series_registry:
            continue
        kind, _fn = series_registry[name]
        if kind == "peak":
            peak_needed.append(name)

    # Native fast path for the long paper-facing parallel-server ring runs.
    # It preserves the aggregate/path file contract and falls back to the
    # general Python engine whenever the config is outside its narrow contract.
    fast_agg_path = run_fast_parallel_server_ring_if_supported(
        cfg,
        output_dir,
        model=model,
        series_names=series_names,
        per_path_series=per_path_series,
        t_indices=t_indices,
    )
    if fast_agg_path is not None:
        return fast_agg_path

    # Pre-allocate per-rep sampled series values: shape (R, K)
    per_rep: dict[str, np.ndarray] = {name: np.zeros((R, K), dtype=float) for name in series_names}

    # SeedSequence-based RNG splitting
    ss = np.random.SeedSequence(seed)
    child_seqs = ss.spawn(R)

    progress = bool(sim.get("progress", False))
    progress_every = int(sim.get("progress_every", 1_000_000))
    if progress_every < 1:
        progress_every = 1
    if progress:
        start = time.perf_counter()
        last_done = 0
        last_wall = start
        total_slots = int(R) * int(T)
        eps = cfg.get("model", {}).get("epsilon")
        eps_s = f"{float(eps):g}" if isinstance(eps, (int, float)) and not isinstance(eps, bool) else "?"
        _progress_line(
            f"[progress] begin: model={getattr(model, 'name', '?')} eps={eps_s} "
            f"R={R} T={T} total_slots={total_slots} every={progress_every}"
        )

    for r in range(R):
        # Independent sub-streams per replication: separate RNGs for arrivals
        # and policy so adding policy-internal randomness does not perturb the
        # arrival realization for a fixed seed.
        arr_seq, pol_seq = child_seqs[r].spawn(2)
        rng_arr = np.random.default_rng(arr_seq)
        rng_pol = np.random.default_rng(pol_seq)
        policy.reset(rng_pol)
        q = q0.copy()

        # Running maxima for "peak" series
        running_max: dict[str, float] = {}
        for name in peak_needed:
            running_max[name] = float("-inf")

        # Diagnostic path buffers (only for first P reps)
        if r < P:
            if save_full_state:
                t_idx_path = np.arange(T, dtype=int)
            else:
                t_idx_path = t_indices
            Kp = int(t_idx_path.size)
            t_set_path = set(int(x) for x in t_idx_path.tolist())
            q_path = np.zeros((Kp,) + q.shape, dtype=np.float32)
            derived_path: dict[str, np.ndarray] = {}
            for name in per_path_series:
                if name not in series_registry:
                    raise KeyError(
                        f"Unknown metrics.series name {name!r} for diagnostic paths. "
                        f"Allowed: {sorted(series_registry)}"
                    )
                derived_path[name] = np.zeros((Kp,), dtype=float)
            kp = 0

        k = 0
        # Slot loop: record state at start of slot t (Q_t), then update to Q_{t+1}.
        for t in range(T):
            # Update running maxima at time t based on Q_t (for any peak series we may emit).
            for name in peak_needed:
                _kind, fn = series_registry[name]
                val = float(fn(q))
                if val > running_max[name]:
                    running_max[name] = val

            if r < P and t in t_set_path:
                while kp < Kp and int(t_idx_path[kp]) < t:
                    kp += 1
                if kp < Kp and int(t_idx_path[kp]) == t:
                    q_path[kp] = q.astype(np.float32, copy=False)
                    for name in per_path_series:
                        kind, fn = series_registry[name]
                        if kind == "peak":
                            derived_path[name][kp] = running_max[name]
                        else:
                            val = float(fn(q))
                            derived_path[name][kp] = val
                    kp += 1

            if t in t_set:
                # Emit values at this grid time t.
                while k < K and int(t_indices[k]) < t:
                    # Should not happen: t_indices are nondecreasing and we only emit when t in t_set.
                    k += 1
                if k < K and int(t_indices[k]) == t:
                    for name in series_names:
                        kind, fn = series_registry[name]
                        if kind == "peak":
                            per_rep[name][r, k] = running_max[name]
                        else:
                            per_rep[name][r, k] = float(fn(q))
                    k += 1

            s = policy.schedule(q, rng_pol)
            a = arrivals.sample(rng_arr)
            q = np.maximum(0.0, q + a - s)
            if progress and (t + 1 == T or ((t + 1) % progress_every == 0)):
                done = r * T + (t + 1)
                now = time.perf_counter()
                dslots = done - last_done
                dwall = max(1e-12, now - last_wall)
                rate = dslots / dwall
                frac = done / max(1, total_slots)
                eta = (total_slots - done) / max(1e-12, rate)
                _progress_line(
                    f"[progress] rep={r + 1}/{R}  t={t + 1}/{T}  "
                    f"{100 * frac:6.2f}%  {rate:10.1f} slots/s  ETA {_format_eta(eta)}"
                )
                last_done = done
                last_wall = now

        if k != K:
            raise RuntimeError("Internal error: did not emit all requested t_indices for a replication.")
        if r < P and kp != Kp:
            raise RuntimeError("Internal error: did not emit all requested diagnostic t_indices for a replication.")
        if r < P:
            write_replication_path(
                Path(output_dir),
                rep_idx=r,
                t_indices=t_idx_path,
                Q=q_path,
                derived_series=derived_path,
            )

    if progress:
        elapsed = time.perf_counter() - start
        _progress_done_line(f"[progress] done: elapsed={_format_eta(elapsed)}")

    def _mean_and_se(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(x, axis=0)
        if x.shape[0] <= 1:
            se = np.zeros_like(mean)
        else:
            sd = np.std(x, axis=0, ddof=1)
            se = sd / np.sqrt(x.shape[0])
        return mean, se

    arrays: dict[str, np.ndarray] = {"t_indices": t_indices.astype(int)}
    # Always-on peaks under canonical keys
    m1, se1 = _mean_and_se(per_rep["peak_l1_so_far"])
    m2, se2 = _mean_and_se(per_rep["peak_l2_so_far"])
    arrays["E_peak_l1_so_far"] = m1
    arrays["SE_peak_l1_so_far"] = se1
    arrays["E_peak_l2_so_far"] = m2
    arrays["SE_peak_l2_so_far"] = se2
    # Additional requested series
    for name in series_names:
        if name in ("peak_l1_so_far", "peak_l2_so_far"):
            continue
        m, se = _mean_and_se(per_rep[name])
        arrays[f"E_{name}"] = m
        arrays[f"SE_{name}"] = se

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "aggregate.npz"
    np.savez_compressed(path, **arrays)
    return path
