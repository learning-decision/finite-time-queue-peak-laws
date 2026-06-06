"""Command-line entry: load config, validate, print summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import qpeak  # noqa: F401 — register model/arrival/policy factories

from qpeak.config_io import (
    epsilon_output_slug,
    load_config_file,
    remove_comment_keys,
    resolved_config_with_epsilon,
    summarize_config,
    uses_model_epsilons_sweep,
    validate_config_shape,
    write_run_manifest,
)
from qpeak.simulation import run_replications_and_write_aggregate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q-peak experiments (config-driven).")
    p.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=None,
        help="Path to experiment config (.json, .yaml, .yml).",
    )
    return p.parse_args()

def _default_output_dir(config_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("_runs") / config_path.stem / stamp


def main() -> None:
    args = parse_args()
    if args.config is None:
        import qpeak as pkg

        print((pkg.__doc__ or "qpeak package").strip())
        print("\nUsage: python main.py <config.json|yaml>")
        return

    raw = load_config_file(args.config)
    cfg = remove_comment_keys(raw)
    validate_config_shape(cfg)
    print(summarize_config(cfg))

    model = cfg["model"]
    if uses_model_epsilons_sweep(model):
        eps_values = [float(x) for x in model["epsilons"]]
        sim0 = cfg["simulation"]
        base_out = Path(sim0["output_dir"]) if sim0.get("output_dir") else _default_output_dir(args.config)
        for eps in eps_values:
            sub_cfg = resolved_config_with_epsilon(cfg, eps)
            sim = sub_cfg["simulation"]
            out_dir = base_out / f"eps_{epsilon_output_slug(eps)}"
            sim["output_dir"] = str(out_dir)

            manifest_path = write_run_manifest(out_dir, sub_cfg, config_source=args.config.resolve())
            print(f"\n--- epsilon={eps:g} -> {out_dir} ---")
            print(f"Wrote {manifest_path} (config snapshot for this run).")

            agg_path = run_replications_and_write_aggregate(sub_cfg, out_dir)
            with np.load(agg_path) as z:
                t_indices = z["t_indices"]
                last_t = int(t_indices[-1])
                e1 = float(z["E_peak_l1_so_far"][-1])
                se1 = float(z["SE_peak_l1_so_far"][-1])
                e2 = float(z["E_peak_l2_so_far"][-1])
                se2 = float(z["SE_peak_l2_so_far"][-1])
            print(
                f"Wrote {agg_path} (R={sim.get('num_replications', 10)}; last grid t={last_t}; "
                f"E[peak_l1_so_far]={e1:.6g}±{se1:.3g}, E[peak_l2_so_far]={e2:.6g}±{se2:.3g})"
            )
        return

    sim = cfg["simulation"]
    out_dir_cfg = sim.get("output_dir")
    out_dir = Path(out_dir_cfg) if out_dir_cfg else _default_output_dir(args.config)
    # Make the resolved directory part of the config snapshot for reproducibility.
    sim["output_dir"] = str(out_dir)

    manifest_path = write_run_manifest(out_dir, cfg, config_source=args.config.resolve())
    print(f"Wrote {manifest_path} (config snapshot for this run).")

    agg_path = run_replications_and_write_aggregate(cfg, out_dir)
    with np.load(agg_path) as z:
        t_indices = z["t_indices"]
        last_t = int(t_indices[-1])
        e1 = float(z["E_peak_l1_so_far"][-1])
        se1 = float(z["SE_peak_l1_so_far"][-1])
        e2 = float(z["E_peak_l2_so_far"][-1])
        se2 = float(z["SE_peak_l2_so_far"][-1])
    print(
        f"Wrote {agg_path} (R={sim.get('num_replications', 10)}; last grid t={last_t}; "
        f"E[peak_l1_so_far]={e1:.6g}±{se1:.3g}, E[peak_l2_so_far]={e2:.6g}±{se2:.3g})"
    )
