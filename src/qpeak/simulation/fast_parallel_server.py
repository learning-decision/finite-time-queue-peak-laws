"""Native fast path for long parallel-server ring experiments.

The public engine keeps the original Python implementation as a general fallback.
For the paper's large ``parallel_server`` ring runs, this module compiles and calls
an O(1)-per-slot C++ simulator that avoids Python slot overhead and the per-slot
Hungarian solve.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from qpeak.models.parallel_server import ParallelServerModel
from qpeak.recording import write_replication_path


_SERIES_CODES: dict[str, int] = {
    "peak_l1_so_far": 0,
    "peak_l2_so_far": 1,
    "totalQ": 2,
    "q_norm_l1": 3,
    "q_norm_l2": 4,
    "peak_totalQ_so_far": 5,
}


def _mean_and_se(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    if x.shape[0] <= 1:
        se = np.zeros_like(mean)
    else:
        sd = np.std(x, axis=0, ddof=1)
        se = sd / np.sqrt(x.shape[0])
    return mean, se


def _ring_edges_match(model: ParallelServerModel) -> bool:
    if int(model.L) != int(model.K):
        return False
    L = int(model.L)
    expected = {(i, i) for i in range(L)} | {(i, (i + 1) % L) for i in range(L)}
    return set((int(l), int(k)) for (l, k) in model.edges) == expected


def _arrival_lambda_if_uniform(cfg: dict[str, Any], model: ParallelServerModel) -> float | None:
    arr = cfg.get("arrivals", {})
    if arr.get("type") != "bernoulli_customer":
        return None
    L = int(model.L)
    if "lambdas" in arr:
        raw = arr["lambdas"]
        if not isinstance(raw, list) or len(raw) != L:
            return None
        vals = np.asarray([float(x) for x in raw], dtype=float)
        if vals.size == 0 or not np.allclose(vals, vals[0], rtol=0.0, atol=1e-15):
            return None
        lam = float(vals[0])
    else:
        mu = float(getattr(model.service_time, "mu"))
        lam = (1.0 - float(model.epsilon)) * mu * float(model.K) / float(model.L)
    if not (0.0 < lam < 1.0):
        return None
    return lam


def _shared_library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":
        return ".dll"
    return ".so"


def _compile_native_library() -> Path:
    here = Path(__file__).resolve().parent
    cpp = here / "fast_parallel_server.cpp"
    build_dir = here / "_native_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    lib = build_dir / ("libqpeak_fast_parallel_server" + _shared_library_suffix())

    if lib.exists() and lib.stat().st_mtime >= cpp.stat().st_mtime:
        return lib

    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise RuntimeError("No C++ compiler found. Install g++/clang++ or set CXX for the qpeak native fast path.")

    base = [
        cxx,
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        str(cpp),
        "-o",
        str(lib),
    ]
    if os.environ.get("QPEAK_NATIVE_ARCH", "1") not in {"0", "false", "False"}:
        base.insert(1, "-march=native")

    attempts: list[list[str]] = []
    # Prefer OpenMP.  If that toolchain does not support it, fall back to a serial shared library.
    if os.name != "nt":
        attempts.append(base[:1] + ["-fopenmp"] + base[1:] + ["-fopenmp"])
    attempts.append(base)

    errors: list[str] = []
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return lib
        except subprocess.CalledProcessError as e:
            errors.append("$ " + " ".join(cmd) + "\n" + (e.stderr or e.stdout or ""))
    raise RuntimeError("Failed to build qpeak native fast path.\n" + "\n---\n".join(errors))


def _load_native_library() -> ctypes.CDLL:
    lib_path = _compile_native_library()
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.qpeak_simulate_parallel_server_ring
    fn.argtypes = [
        ctypes.c_longlong,  # T
        ctypes.c_int,  # R
        ctypes.c_int,  # L
        ctypes.c_double,  # lambda
        ctypes.c_double,  # mu
        ctypes.c_ulonglong,  # seed
        ctypes.c_void_p,  # t_indices int64*
        ctypes.c_int,  # G
        ctypes.c_void_p,  # series_codes int*
        ctypes.c_int,  # S
        ctypes.c_void_p,  # per_rep_out double*
        ctypes.c_int,  # P
        ctypes.c_void_p,  # q_paths float*
        ctypes.c_void_p,  # path_series_codes int*
        ctypes.c_int,  # PS
        ctypes.c_void_p,  # path_series_out double*
        ctypes.c_int,  # progress
        ctypes.c_longlong,  # progress_every
        ctypes.c_int,  # num_threads
    ]
    fn.restype = ctypes.c_int
    return lib


def run_fast_parallel_server_ring_if_supported(
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    model: Any,
    series_names: list[str],
    per_path_series: list[str],
    t_indices: np.ndarray,
) -> Path | None:
    """Run native ring simulator when ``cfg`` matches its contract; otherwise return ``None``."""
    sim = cfg["simulation"]
    fast_mode = str(sim.get("fast_mode", "auto")).lower()
    if fast_mode in {"off", "false", "0", "python"}:
        return None

    if not isinstance(model, ParallelServerModel):
        return None
    if cfg.get("policy", {}).get("type") != "maxweight":
        return None
    if not _ring_edges_match(model):
        if fast_mode in {"force", "native"}:
            raise RuntimeError("simulation.fast_mode requested native, but the parallel_server graph is not the L=K ring zigzag.")
        return None
    if int(model.L) > 63:
        if fast_mode in {"force", "native"}:
            raise RuntimeError("native ring fast path supports L<=63 because it uses uint64 bit masks.")
        return None
    if getattr(model.service_time, "name", None) != "geometric" or not hasattr(model.service_time, "mu"):
        return None

    lam = _arrival_lambda_if_uniform(cfg, model)
    if lam is None:
        if fast_mode in {"force", "native"}:
            raise RuntimeError("native ring fast path requires uniform bernoulli_customer arrivals.")
        return None
    mu = float(getattr(model.service_time, "mu"))

    rec = cfg.get("recording", {})
    if bool(rec.get("full_state", True)):
        # The large runs deliberately use downsampled diagnostic paths.  Saving a full
        # T-by-L path would dominate memory/time and is left to the general Python path.
        if fast_mode in {"force", "native"}:
            raise RuntimeError("native fast path requires recording.full_state=false.")
        return None

    unsupported = [s for s in list(dict.fromkeys(series_names + per_path_series)) if s not in _SERIES_CODES]
    if unsupported:
        if fast_mode in {"force", "native"}:
            raise RuntimeError(f"native fast path does not support metric series: {unsupported}")
        return None

    T = int(sim["T"])
    R = int(sim.get("num_replications", 10))
    L = int(model.L)
    seed = int(sim["seed"])
    t_indices = np.ascontiguousarray(t_indices.astype(np.int64, copy=False))
    G = int(t_indices.size)
    if G <= 0:
        return None

    series_codes = np.ascontiguousarray(np.array([_SERIES_CODES[s] for s in series_names], dtype=np.int32))
    S = int(series_codes.size)
    per_rep_native = np.zeros((S, R, G), dtype=np.float64)

    P = int(rec.get("num_paths_to_save", min(R, 10)))
    if P < 0:
        P = 0
    P = min(P, R)
    q_paths = np.zeros((P, G, L), dtype=np.float32) if P > 0 else None
    path_codes = np.ascontiguousarray(np.array([_SERIES_CODES[s] for s in per_path_series], dtype=np.int32))
    PS = int(path_codes.size)
    path_series_native = np.zeros((P, PS, G), dtype=np.float64) if (P > 0 and PS > 0) else None

    progress = 1 if bool(sim.get("progress", False)) else 0
    progress_every = int(sim.get("progress_every", 1_000_000))
    if progress_every < 1:
        progress_every = 1
    default_threads = min(max(1, os.cpu_count() or 1), R)
    num_threads = int(sim.get("num_threads", os.environ.get("QPEAK_NUM_THREADS", default_threads)))
    if num_threads < 1:
        num_threads = default_threads

    lib = _load_native_library()
    fn = lib.qpeak_simulate_parallel_server_ring

    def ptr(arr: np.ndarray | None) -> ctypes.c_void_p:
        if arr is None:
            return ctypes.c_void_p(0)
        return ctypes.c_void_p(arr.ctypes.data)

    print(
        f"[fast] native parallel_server ring path: L=K={L}, R={R}, T={T}, "
        f"lambda={lam:.8g}, mu={mu:.8g}, threads={num_threads}",
        flush=True,
    )
    rc = fn(
        ctypes.c_longlong(T),
        ctypes.c_int(R),
        ctypes.c_int(L),
        ctypes.c_double(lam),
        ctypes.c_double(mu),
        ctypes.c_ulonglong(seed & ((1 << 64) - 1)),
        ptr(t_indices),
        ctypes.c_int(G),
        ptr(series_codes),
        ctypes.c_int(S),
        ptr(per_rep_native),
        ctypes.c_int(P),
        ptr(q_paths),
        ptr(path_codes),
        ctypes.c_int(PS),
        ptr(path_series_native),
        ctypes.c_int(progress),
        ctypes.c_longlong(progress_every),
        ctypes.c_int(num_threads),
    )
    if int(rc) != 0:
        raise RuntimeError(f"qpeak native fast path failed with return code {rc}.")

    arrays: dict[str, np.ndarray] = {"t_indices": t_indices.astype(int)}
    # Preserve the aggregate key contract of the Python engine.
    by_name = {name: per_rep_native[i] for i, name in enumerate(series_names)}
    m1, se1 = _mean_and_se(by_name["peak_l1_so_far"])
    m2, se2 = _mean_and_se(by_name["peak_l2_so_far"])
    arrays["E_peak_l1_so_far"] = m1
    arrays["SE_peak_l1_so_far"] = se1
    arrays["E_peak_l2_so_far"] = m2
    arrays["SE_peak_l2_so_far"] = se2
    for name in series_names:
        if name in ("peak_l1_so_far", "peak_l2_so_far"):
            continue
        m, se = _mean_and_se(by_name[name])
        arrays[f"E_{name}"] = m
        arrays[f"SE_{name}"] = se

    output_dir.mkdir(parents=True, exist_ok=True)
    agg_path = output_dir / "aggregate.npz"
    np.savez_compressed(agg_path, **arrays)

    if P > 0 and q_paths is not None:
        for r in range(P):
            derived: dict[str, np.ndarray] = {}
            if path_series_native is not None:
                for j, name in enumerate(per_path_series):
                    derived[name] = path_series_native[r, j]
            write_replication_path(
                Path(output_dir),
                rep_idx=r,
                t_indices=t_indices,
                Q=q_paths[r],
                derived_series=derived,
            )

    return agg_path
