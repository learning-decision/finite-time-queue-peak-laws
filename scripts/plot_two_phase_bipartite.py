"""Plot two-phase envelope for asymmetric bipartite matching (customer-only metrics).

Usage:
    PYTHONPATH=src python scripts/plot_two_phase_bipartite.py _runs/two_phase_bipartite_asym

Produces figures with all epsilon values overlaid, x-axis on log scale.
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

    cfg0 = runs[0]["manifest"]["config"]
    model_desc = cfg0["model"]["type"]
    L = cfg0["model"].get("L", "?")
    K = cfg0["model"].get("K", "?")
    R = cfg0["simulation"].get("num_replications", "?")
    suptitle = f"{model_desc} (L={L}, K={K}, R={R}) — customer queues only"

    # Helper to plot peak + beta as 2x1 subplot
    def _plot_peak_with_beta(y_key, se_key, ylabel, xscale, fname):
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
        if xscale == "log":
            ax.set_xscale("log")
            ax_b.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(suptitle)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both" if xscale == "log" else "major")
        ax_b.set_xlabel(r"$T$ (log scale)" if xscale == "log" else r"$T$")
        ax_b.set_ylabel(r"$\beta(T)$")
        ax_b.axhline(0.5, color="gray", ls="--", lw=0.8, label=r"$\sqrt{T}$")
        ax_b.axhline(0.0, color="gray", ls=":", lw=0.8, label=r"$\log T$")
        ax_b.legend(fontsize=7, ncol=2, loc="upper right")
        ax_b.grid(True, alpha=0.3, which="both" if xscale == "log" else "major")
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote {path}")

    # ---------- Figure 1a: peak customer L1 vs T (log scale) ----------
    _plot_peak_with_beta(
        "E_peak_customer_l1_so_far", "SE_peak_customer_l1_so_far",
        r"$\mathbb{E}\left[\max_{0\leq s\leq t}\|Q^{\mathrm{cust}}_s\|_1\right]$",
        "log", f"peak_customer_l1_vs_t_logscale.{fmt}")

    # ---------- Figure 1b: peak customer L1 vs T (linear scale) ----------
    _plot_peak_with_beta(
        "E_peak_customer_l1_so_far", "SE_peak_customer_l1_so_far",
        r"$\mathbb{E}\left[\max_{0\leq s\leq t}\|Q^{\mathrm{cust}}_s\|_1\right]$",
        "linear", f"peak_customer_l1_vs_t_linear.{fmt}")

    # ---------- Figure 2a: peak customer L2 vs T (log scale) ----------
    _plot_peak_with_beta(
        "E_peak_customer_l2_so_far", "SE_peak_customer_l2_so_far",
        r"$\mathbb{E}\left[\max_{0\leq s\leq t}\|Q^{\mathrm{cust}}_s\|_2\right]$",
        "log", f"peak_customer_l2_vs_t_logscale.{fmt}")

    # ---------- Figure 2b: peak customer L2 vs T (linear scale) ----------
    _plot_peak_with_beta(
        "E_peak_customer_l2_so_far", "SE_peak_customer_l2_so_far",
        r"$\mathbb{E}\left[\max_{0\leq s\leq t}\|Q^{\mathrm{cust}}_s\|_2\right]$",
        "linear", f"peak_customer_l2_vs_t_linear.{fmt}")

    # ---------- Normalized and collapse plots for both L1 and L2 ----------
    norm_series = [
        ("E_peak_customer_l1_so_far", 1),
        ("E_peak_customer_l2_so_far", 2),
    ]

    for y_key, p_norm in norm_series:
        suffix = f"customer_l{p_norm}"
        peak_label = rf"\|Q^{{\mathrm{{cust}}}}_s\|_{p_norm}"

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
        ax.set_title(suptitle + " — coefficient extraction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        path = out_dir / f"peak_{suffix}_over_logT_vs_t.{fmt}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote {path}")

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
        ax.set_title(suptitle + r" — $\sqrt{T}$ normalization")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        path = out_dir / f"peak_{suffix}_over_sqrtT_vs_t.{fmt}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote {path}")

        # eps * peak vs log(T+1) (slope collapse)
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
        ax.set_ylabel(rf"$\varepsilon \times \mathrm{{peak}}$")
        ax.set_title(suptitle + r" — slope collapse ($\varepsilon \times$ peak)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"eps_x_peak_{suffix}_vs_logT.{fmt}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
