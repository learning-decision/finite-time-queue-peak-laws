"""Write diagnostic sample paths under an experiment output directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def rep_path_file(output_dir: Path, rep_idx: int) -> Path:
    return output_dir / "paths" / f"rep_{rep_idx:04d}.npz"


def write_replication_path(
    output_dir: Path,
    *,
    rep_idx: int,
    t_indices: np.ndarray,
    Q: np.ndarray,
    derived_series: dict[str, np.ndarray] | None = None,
) -> Path:
    """
    Write one replication's diagnostic path.

    Arrays are written with ``np.savez`` (uncompressed) for speed.

    Required arrays:
      - ``t_indices``: shape (K,), int
      - ``Q``: shape (K, ...) float32 (IQS: (K, n, n))

    Optional derived arrays must be shape (K,).
    """
    output_dir = Path(output_dir)
    out = rep_path_file(output_dir, rep_idx)
    out.parent.mkdir(parents=True, exist_ok=True)

    t_indices = np.asarray(t_indices, dtype=int)
    Q = np.asarray(Q, dtype=np.float32)

    payload: dict[str, Any] = {"t_indices": t_indices, "Q": Q}
    if derived_series:
        for k, v in derived_series.items():
            payload[k] = np.asarray(v, dtype=float)

    np.savez(out, **payload)
    return out

