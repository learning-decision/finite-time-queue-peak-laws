"""Plot peak queue norms with T on log-scale x-axis. Works for any model.

Usage:
    PYTHONPATH=src python scripts/plot_logscale.py _runs/smoke_iqs
    PYTHONPATH=src python scripts/plot_logscale.py _runs/smoke_iqs --out_dir _runs/smoke_iqs_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _effective_beta(t: np.ndarray, y: np.ndarray,
                    log_half_width: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute local effective growth exponent: beta(T) = d log(peak) / d log(T).

    Uses a **log-proportional window**: for each T_i, finds the nearest data
    points at T_i * e^{-h} and T_i * e^{+h} (where h = log_half_width) and
    computes the finite-difference slope in log-log space.  This keeps the
    window resolution constant on a log scale, so small-T behaviour is not
    smeared out by a window that is too wide relative to T.

    Returns (t_beta, beta) arrays.  Points where y <= 0 are excluded.
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


def _load_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "aggregate.npz") as z:
        agg = {k: z[k] for k in z.files}
    eps = float(manifest["config"]["model"]["epsilon"])
    return {"dir": run_dir, "manifest": manifest, "agg": agg, "eps": eps}


def _expand(run_dir: Path) -> list[Path]:
    eps_children = sorted(
        p for p in run_dir.glob("eps_*") if p.is_dir() and (p / "aggregate.npz").is_file()
    )
    if eps_children:
        return eps_children
    if (run_dir / "aggregate.npz").is_file():
        return [run_dir]
    raise FileNotFoundError(f"No aggregate.npz under {run_dir}")


def _suptitle(runs: list[dict[str, Any]]) -> str:
    cfg0 = runs[0]["manifest"]["config"]
    model = cfg0["model"]
    sim = cfg0["simulation"]
    mtype = model.get("type", "?")
    R = sim.get("num_replications", "?")
    parts = [f"model={mtype}", f"R={R}"]
    if mtype == "iqs":
        parts.append(f"n={model.get('n', '?')}")
    elif mtype == "bipartite_matching":
        parts.append(f"L={model.get('L', '?')}, K={model.get('K', '?')}")
    return ", ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--band_mult", type=float, default=1.0)
    p.add_argument("--format", dest="fmt", choices=("pdf", "png", "svg"), default="pdf",
                   help="Output figure format (default: pdf, vectorized).")
    args = p.parse_args()
    fmt = args.fmt

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dirs = _expand(args.run_dir.resolve())
    runs = sorted([_load_run(d) for d in dirs], key=lambda r: r["eps"])
    out_dir = (args.out_dir or args.run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("tab10")
    n = len(runs)
    band_alpha = 0.15 if n > 1 else 0.25
    title = _suptitle(runs)

    def _save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote {path}")

    # Detect if customer-only metrics are available
    has_customer = "E_peak_customer_l1_so_far" in runs[0]["agg"]

    # Which peak series to plot
    peak_series = [
        ("E_peak_l1_so_far", "SE_peak_l1_so_far", 1, "full"),
        ("E_peak_l2_so_far", "SE_peak_l2_so_far", 2, "full"),
    ]
    if has_customer:
        peak_series += [
            ("E_peak_customer_l1_so_far", "SE_peak_customer_l1_so_far", 1, "customer"),
            ("E_peak_customer_l2_so_far", "SE_peak_customer_l2_so_far", 2, "customer"),
        ]

    for y_key, se_key, p_norm, scope in peak_series:
        suffix = f"{'customer_' if scope == 'customer' else ''}l{p_norm}"
        if scope == "customer":
            ylabel = rf"$\mathbb{{E}}\left[\max_{{0\leq s\leq t}}\|Q^{{\mathrm{{cust}}}}_s\|_{p_norm}\right]$"
        else:
            ylabel = rf"$\mathbb{{E}}\left[\max_{{0\leq s\leq t}}\|Q_s\|_{p_norm}\right]$"

        # --- Log-scale x (with beta subplot) ---
        fig, (ax, ax_b) = plt.subplots(2, 1, figsize=(7.5, 6.5),
                                        height_ratios=[3, 1], sharex=True)
        for i, r in enumerate(runs):
            t = r["agg"]["t_indices"].astype(float)
            y = r["agg"][y_key]
            se = r["agg"].get(se_key)
            color = cmap(i % 10)
            label = rf"$\varepsilon = {r['eps']:g}$"
            ax.plot(t, y, label=label, color=color, linewidth=1.6, zorder=10 + i)
            if se is not None:
                ax.fill_between(t, y - args.band_mult * se, y + args.band_mult * se,
                                color=color, alpha=band_alpha, zorder=i)
            t_b, beta = _effective_beta(t, y)
            if len(t_b) > 0:
                ax_b.plot(t_b, beta, label=label, color=color, linewidth=1.2, alpha=0.85)
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax_b.set_xscale("log")
        ax_b.set_xlabel(r"$T$ (log scale)")
        ax_b.set_ylabel(r"$\beta(T)$")
        ax_b.axhline(0.5, color="gray", ls="--", lw=0.8, label=r"$\sqrt{T}$")
        ax_b.axhline(0.0, color="gray", ls=":", lw=0.8, label=r"$\log T$")
        ax_b.legend(fontsize=7, ncol=2, loc="upper right")
        ax_b.grid(True, alpha=0.3, which="both")
        _save(fig, f"peak_{suffix}_vs_t_logscale.{fmt}")

        # --- Linear x (with beta subplot) ---
        fig, (ax, ax_b) = plt.subplots(2, 1, figsize=(7.5, 6.5),
                                        height_ratios=[3, 1], sharex=True)
        for i, r in enumerate(runs):
            t = r["agg"]["t_indices"].astype(float)
            y = r["agg"][y_key]
            se = r["agg"].get(se_key)
            color = cmap(i % 10)
            label = rf"$\varepsilon = {r['eps']:g}$"
            ax.plot(t, y, label=label, color=color, linewidth=1.6, zorder=10 + i)
            if se is not None:
                ax.fill_between(t, y - args.band_mult * se, y + args.band_mult * se,
                                color=color, alpha=band_alpha, zorder=i)
            t_b, beta = _effective_beta(t, y)
            if len(t_b) > 0:
                ax_b.plot(t_b, beta, label=label, color=color, linewidth=1.2, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax_b.set_xlabel(r"$T$")
        ax_b.set_ylabel(r"$\beta(T)$")
        ax_b.axhline(0.5, color="gray", ls="--", lw=0.8, label=r"$\sqrt{T}$")
        ax_b.axhline(0.0, color="gray", ls=":", lw=0.8, label=r"$\log T$")
        ax_b.legend(fontsize=7, ncol=2, loc="upper right")
        ax_b.grid(True, alpha=0.3)
        _save(fig, f"peak_{suffix}_vs_t_linear.{fmt}")

    # --- Normalized plots: peak/log(T+1) and peak/sqrt(T) for l1 and l2 ---
    norm_series = [
        ("E_peak_l1_so_far", 1, "full"),
        ("E_peak_l2_so_far", 2, "full"),
    ]
    if has_customer:
        norm_series += [
            ("E_peak_customer_l1_so_far", 1, "customer"),
            ("E_peak_customer_l2_so_far", 2, "customer"),
        ]

    for y_key, p_norm, scope in norm_series:
        suffix = f"{'customer_' if scope == 'customer' else ''}l{p_norm}"

        # peak / log(T+1) vs T (log scale)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for i, r in enumerate(runs):
            t = r["agg"]["t_indices"].astype(float)
            y = r["agg"][y_key]
            mask = t > 1
            ratio = y[mask] / np.log1p(t[mask])
            color = cmap(i % 10)
            label = rf"$\varepsilon = {r['eps']:g}$"
            ax.plot(t[mask], ratio, label=label, color=color, linewidth=1.6, zorder=10 + i)
        ax.set_xscale("log")
        ax.set_xlabel(r"$T$ (log scale)")
        ax.set_ylabel(r"$\mathrm{peak} \,/\, \log(T+1)$")
        ax.set_title(title + " — coefficient extraction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        _save(fig, f"peak_{suffix}_over_logT_vs_t.{fmt}")

        # peak / sqrt(T) vs T (log scale)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for i, r in enumerate(runs):
            t = r["agg"]["t_indices"].astype(float)
            y = r["agg"][y_key]
            mask = t > 1
            ratio = y[mask] / np.sqrt(t[mask])
            color = cmap(i % 10)
            label = rf"$\varepsilon = {r['eps']:g}$"
            ax.plot(t[mask], ratio, label=label, color=color, linewidth=1.6, zorder=10 + i)
        ax.set_xscale("log")
        ax.set_xlabel(r"$T$ (log scale)")
        ax.set_ylabel(r"$\mathrm{peak} \,/\, \sqrt{T}$")
        ax.set_title(title + r" — $\sqrt{T}$ normalization")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        _save(fig, f"peak_{suffix}_over_sqrtT_vs_t.{fmt}")

    # --- eps * peak vs log(T+1) (slope collapse) ---
    for y_key, p_norm, scope in norm_series:
        suffix = f"{'customer_' if scope == 'customer' else ''}l{p_norm}"
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for i, r in enumerate(runs):
            t = r["agg"]["t_indices"].astype(float)
            y = r["agg"][y_key]
            mask = t > 0
            color = cmap(i % 10)
            label = rf"$\varepsilon = {r['eps']:g}$"
            ax.plot(np.log1p(t[mask]), r["eps"] * y[mask],
                    label=label, color=color, linewidth=1.6, zorder=10 + i)
        ax.set_xlabel(r"$\log(T+1)$")
        ax.set_ylabel(r"$\varepsilon \times \mathrm{peak}$")
        ax.set_title(title + r" — slope collapse ($\varepsilon \times$ peak)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _save(fig, f"eps_x_peak_{suffix}_vs_logT.{fmt}")


if __name__ == "__main__":
    main()
