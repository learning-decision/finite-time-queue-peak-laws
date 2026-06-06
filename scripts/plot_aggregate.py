from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_aggregate(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "aggregate.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _get_epsilon(manifest: dict[str, Any]) -> float | None:
    """Read ``model.epsilon`` from manifest (any model that stores a resolved scalar slack)."""
    try:
        cfg = manifest["config"]
        model = cfg["model"]
        if "epsilon" in model:
            eps = model["epsilon"]
            if isinstance(eps, (int, float)) and not isinstance(eps, bool):
                return float(eps)
    except Exception:
        return None
    return None


def _expand_run_dir(run_dir: Path) -> list[Path]:
    """
    Resolve a run directory to one or more leaf dirs that contain ``aggregate.npz``.

    If ``run_dir`` is a sweep parent (only ``run_manifest.json`` at top level, real outputs under
    ``eps_<slug>/``), expand to those children so overlays match ``python main.py`` sweep layout.
    """
    run_dir = run_dir.resolve()
    agg = run_dir / "aggregate.npz"
    eps_children = sorted(
        p for p in run_dir.glob("eps_*") if p.is_dir() and (p / "aggregate.npz").is_file()
    )
    if eps_children:
        # Prefer per-epsilon bundles whenever present (typical for model.epsilons sweeps).
        return eps_children
    if agg.is_file():
        return [run_dir]
    raise FileNotFoundError(
        f"No aggregate.npz under {run_dir} and no eps_*/aggregate.npz children. "
        f"Pass each run directory explicitly, e.g. {run_dir}/eps_0p1"
    )


def _legend_label(manifest: dict[str, Any]) -> str:
    """Short legend entry: only resolved slack (for overlays by ε), mathtext ``\\varepsilon``."""
    eps = _get_epsilon(manifest)
    if eps is not None:
        return rf"$\varepsilon = {eps:g}$"
    return r"$\varepsilon = ?$"


def _figure_title(manifests: list[dict[str, Any]], *, x_axis_is_raw_t: bool) -> str:
    """
    Title line for model / R / (optionally) T.

    When the horizontal axis is raw slot index ``t`` covering the simulated horizon, ``T`` is
    redundant with the axis range and is omitted. For ``tau = epsilon^2 * t`` plots, include ``T``.
    """
    if not manifests:
        return ""
    cfg0 = manifests[0].get("config", {})
    model = cfg0.get("model", {})
    sim = cfg0.get("simulation", {})
    mtype = model.get("type", "?")
    R = sim.get("num_replications", "?")
    T = sim.get("T", "?")
    parts = [f"model={mtype}", f"R={R}"]
    if not x_axis_is_raw_t:
        parts.append(f"T={T}")
    return ", ".join(parts)


def _effective_beta(t: np.ndarray, y: np.ndarray,
                    log_half_width: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute local effective growth exponent: beta(T) = d log(peak) / d log(T).

    Uses a log-proportional window: for each T_i, computes the slope between
    data points near T_i * e^{-h} and T_i * e^{+h}.
    """
    mask = (t > 0) & (y > 0)
    t, y = t[mask], y[mask]
    if len(t) < 3:
        return np.array([]), np.array([])
    log_t = np.log(t)
    log_y = np.log(y)
    h = log_half_width
    t_out, beta_out = [], []
    for i in range(len(t)):
        lo_target = log_t[i] - h
        hi_target = log_t[i] + h
        j_lo = np.searchsorted(log_t, lo_target)
        j_hi = np.searchsorted(log_t, hi_target) - 1
        if j_lo < 0 or j_hi >= len(t) or j_lo >= j_hi:
            continue
        dlog_t = log_t[j_hi] - log_t[j_lo]
        if dlog_t < 1e-12:
            continue
        t_out.append(t[i])
        beta_out.append((log_y[j_hi] - log_y[j_lo]) / dlog_t)
    return np.array(t_out), np.array(beta_out)


_LINESTYLES = ("-", "--", "-.", ":")
_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*", "h", "p")


def _series_style(i: int, n_x: int) -> dict[str, Any]:
    """Distinct linestyle + markers so overlapping y-values remain visible (e.g. GG1 peaks)."""
    markevery = max(1, n_x // 8) if n_x > 0 else 1
    return {
        "linestyle": _LINESTYLES[i % len(_LINESTYLES)],
        "marker": _MARKERS[i % len(_MARKERS)],
        "markersize": 4.5,
        "markevery": markevery,
        "markeredgewidth": 0.9,
        "markeredgecolor": "white",
    }


def _plot_with_band(
    ax,
    x,
    y,
    se,
    label: str,
    band_mult: float,
    *,
    color: Any = None,
    zorder: int | None = None,
    series_index: int = 0,
    fill_alpha: float = 0.2,
) -> None:
    """Draw SE band first (low z), then the mean line + markers on top (high z)."""
    n_x = int(np.asarray(x).size)
    st = _series_style(series_index, n_x)
    line_z = (zorder + 1000) if zorder is not None else 1000
    band_z = (zorder - 1) if zorder is not None else 0

    if se is not None:
        lo = y - band_mult * se
        hi = y + band_mult * se
        fb_kw: dict[str, Any] = {"alpha": fill_alpha, "zorder": band_z}
        if color is not None:
            fb_kw["color"] = color
        ax.fill_between(x, lo, hi, **fb_kw)

    lw_kw: dict[str, Any] = {
        "label": label,
        "zorder": line_z,
        "linewidth": 1.6,
        **st,
    }
    if color is not None:
        lw_kw["color"] = color
    ax.plot(x, y, **lw_kw)


def _latex_running_max_norm_q(p: int) -> str:
    return rf"$\mathbb{{E}}\left[\max_{{0\leq s\leq t}}\|Q_s\|_{p}\right]$"


def _latex_running_max_norm_q_over(p: int, denom: str) -> str:
    return rf"$\mathbb{{E}}\left[\max_{{0\leq s\leq t}}\|Q_s\|_{p}\right]\,/\,{denom}$"


def _drop_first_point_if_t_starts_at_zero(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For ratio plots vs ``t``, the first grid point is often ``t=0`` where ``Q_0=0`` and the running
    peak is 0, which creates a misleading edge next to ``t>0``. Drop that first sample when present.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size >= 2 and float(t.flat[0]) == 0.0:
        return t[1:], y[1:]
    return t, y


def main() -> None:
    p = argparse.ArgumentParser(description="Plot qpeak aggregate.npz runs.")
    p.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more run output directories containing aggregate.npz and run_manifest.json.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to write figures. Default: first run_dir.",
    )
    p.add_argument(
        "--band_mult",
        type=float,
        default=1.0,
        help="Multiplier for SE bands (default: 1.0). Use 1.96 for approx 95%% bands.",
    )
    p.add_argument(
        "--format",
        dest="fmt",
        choices=("pdf", "png", "svg"),
        default="pdf",
        help="Output figure format (default: pdf, vectorized).",
    )
    args = p.parse_args()
    fmt = args.fmt

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it (pip/conda).") from e

    cmap = plt.get_cmap("tab10")

    run_dirs_in = [d.resolve() for d in args.run_dirs]
    run_dirs: list[Path] = []
    for d in run_dirs_in:
        run_dirs.extend(_expand_run_dir(d))
    # Deduplicate (e.g. user passes both sweep parent and eps_* child).
    _seen: set[Path] = set()
    _dedup: list[Path] = []
    for p in run_dirs:
        r = p.resolve()
        if r not in _seen:
            _seen.add(r)
            _dedup.append(r)
    run_dirs = _dedup
    out_dir = (args.out_dir or run_dirs_in[0]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for d in run_dirs:
        manifest = _load_manifest(d)
        agg = _load_aggregate(d)
        t = agg["t_indices"].astype(float)
        eps = _get_epsilon(manifest)
        tau = (eps**2) * t if eps is not None else None
        runs.append(
            {
                "dir": d,
                "manifest": manifest,
                "t": t,
                "tau": tau,
                "agg": agg,
            }
        )

    def _save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote {path}")

    n_runs = len(runs)
    band_alpha = 0.11 if n_runs > 1 else 0.2
    manifests_all = [r["manifest"] for r in runs]
    title_vs_t = _figure_title(manifests_all, x_axis_is_raw_t=True)

    # 1) Running max norms vs t (with beta subplot)
    fig, (ax, ax_b) = plt.subplots(2, 1, figsize=(7.2, 6.0),
                                    height_ratios=[3, 1], sharex=True)
    for i, r in enumerate(runs):
        color = cmap(i % 10)
        z = 2 + i * 0.01
        agg = r["agg"]
        _plot_with_band(
            ax,
            r["t"],
            agg["E_peak_l1_so_far"],
            agg.get("SE_peak_l1_so_far"),
            label=_legend_label(r["manifest"]),
            band_mult=args.band_mult,
            color=color,
            zorder=z,
            series_index=i,
            fill_alpha=band_alpha,
        )
        t_b, beta = _effective_beta(r["t"], agg["E_peak_l1_so_far"])
        if len(t_b) > 0:
            ax_b.plot(t_b, beta, label=_legend_label(r["manifest"]),
                      color=color, linewidth=1.2, alpha=0.85)
    ax.set_title(title_vs_t)
    ax.set_ylabel(_latex_running_max_norm_q(1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax_b.set_xlabel("t")
    ax_b.set_ylabel(r"$\beta(T)$")
    ax_b.axhline(0.5, color="gray", ls="--", lw=0.8, label=r"$\sqrt{T}$")
    ax_b.axhline(0.0, color="gray", ls=":", lw=0.8, label=r"$\log T$")
    ax_b.legend(fontsize=7, ncol=2, loc="upper right")
    ax_b.grid(True, alpha=0.3)
    _save(fig, f"peak_l1_vs_t.{fmt}")

    fig, (ax, ax_b) = plt.subplots(2, 1, figsize=(7.2, 6.0),
                                    height_ratios=[3, 1], sharex=True)
    for i, r in enumerate(runs):
        color = cmap(i % 10)
        z = 2 + i * 0.01
        agg = r["agg"]
        _plot_with_band(
            ax,
            r["t"],
            agg["E_peak_l2_so_far"],
            agg.get("SE_peak_l2_so_far"),
            label=_legend_label(r["manifest"]),
            band_mult=args.band_mult,
            color=color,
            zorder=z,
            series_index=i,
            fill_alpha=band_alpha,
        )
        t_b, beta = _effective_beta(r["t"], agg["E_peak_l2_so_far"])
        if len(t_b) > 0:
            ax_b.plot(t_b, beta, label=_legend_label(r["manifest"]),
                      color=color, linewidth=1.2, alpha=0.85)
    ax.set_title(title_vs_t)
    ax.set_ylabel(_latex_running_max_norm_q(2))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax_b.set_xlabel("t")
    ax_b.set_ylabel(r"$\beta(T)$")
    ax_b.axhline(0.5, color="gray", ls="--", lw=0.8, label=r"$\sqrt{T}$")
    ax_b.axhline(0.0, color="gray", ls=":", lw=0.8, label=r"$\log T$")
    ax_b.legend(fontsize=7, ncol=2, loc="upper right")
    ax_b.grid(True, alpha=0.3)
    _save(fig, f"peak_l2_vs_t.{fmt}")

    # 2) Normalized vs t for both L1 and L2
    for p_norm, y_key in [(1, "E_peak_l1_so_far"), (2, "E_peak_l2_so_far")]:
        # peak / log(t+1)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for i, r in enumerate(runs):
            t = r["t"]
            agg = r["agg"]
            y = agg[y_key] / np.maximum(1.0, np.log1p(t))
            t_p, y_p = _drop_first_point_if_t_starts_at_zero(t, y)
            st = _series_style(i, int(np.asarray(t_p).size))
            ax.plot(t_p, y_p, label=_legend_label(r["manifest"]),
                    color=cmap(i % 10), zorder=2 + i * 0.01, linewidth=1.6, **st)
        ax.set_title(title_vs_t)
        ax.set_xlabel("t")
        ax.set_ylabel(_latex_running_max_norm_q_over(p_norm, r"\log(t+1)"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, f"peak_l{p_norm}_over_log_vs_t.{fmt}")

        # peak / sqrt(t)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for i, r in enumerate(runs):
            t = r["t"]
            agg = r["agg"]
            y = agg[y_key] / np.maximum(1.0, np.sqrt(t))
            t_p, y_p = _drop_first_point_if_t_starts_at_zero(t, y)
            st = _series_style(i, int(np.asarray(t_p).size))
            ax.plot(t_p, y_p, label=_legend_label(r["manifest"]),
                    color=cmap(i % 10), zorder=2 + i * 0.01, linewidth=1.6, **st)
        ax.set_title(title_vs_t)
        ax.set_xlabel("t")
        ax.set_ylabel(_latex_running_max_norm_q_over(p_norm, r"\sqrt{t}"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, f"peak_l{p_norm}_over_sqrt_vs_t.{fmt}")

    # 3) Peaks vs tau (only runs that have epsilon)
    runs_tau = [r for r in runs if r["tau"] is not None]
    if runs_tau:
        title_vs_tau = _figure_title([r["manifest"] for r in runs_tau], x_axis_is_raw_t=False)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for i, r in enumerate(runs_tau):
            color = cmap(i % 10)
            z = 2 + i * 0.01
            agg = r["agg"]
            _plot_with_band(
                ax,
                r["tau"],
                agg["E_peak_l1_so_far"],
                agg.get("SE_peak_l1_so_far"),
                label=_legend_label(r["manifest"]),
                band_mult=args.band_mult,
                color=color,
                zorder=z,
                series_index=i,
                fill_alpha=band_alpha,
            )
        ax.set_title(title_vs_tau)
        ax.set_xlabel(r"tau = epsilon^2 * t")
        ax.set_ylabel(_latex_running_max_norm_q(1))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, f"peak_l1_vs_tau.{fmt}")


if __name__ == "__main__":
    main()

