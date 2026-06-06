# Reproducing the Experiments

This repository contains the simulation code and plotting scripts used for the
queueing experiments. Generated outputs are intentionally not part of the source
artifact; the commands below recreate them from the checked-in configs.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, install the package in editable mode:

```bash
python -m pip install -e .
```

The long parallel-server runs use a native C++ fast path when available. A C++17
compiler such as `g++` or `clang++` is required for that path; otherwise use
`"fast_mode": "off"` in the config to fall back to the Python implementation.

## Quick Smoke Test

Verify the end-to-end pipeline before running paper experiments:

```bash
python main.py configs/smoke_parallel_server.json
PYTHONPATH=src python scripts/plot_logscale.py _runs/smoke_parallel_server --out_dir _runs/smoke_parallel_server_figs --band_mult 1.96
```

---

## Experiment 1 — Two-Phase Envelope (IQS + Parallel Server)

### Appendix: large-slack experiments

```bash
python main.py configs/two_phase_iqs.json
PYTHONPATH=src python scripts/plot_logscale.py _runs/two_phase_iqs_large_eps --out_dir _runs/two_phase_iqs_large_eps_figs --band_mult 1.96

python main.py configs/two_phase_parallel_server_1.json
PYTHONPATH=src python scripts/plot_logscale.py _runs/two_phase_parallel_server_large_eps_1 --out_dir _runs/two_phase_parallel_server_large_eps_1_figs --band_mult 1.96
```

Key output PDFs:

```
_runs/two_phase_iqs_large_eps_figs/peak_l2_vs_t_logscale.pdf
_runs/two_phase_iqs_large_eps_figs/peak_l2_over_logT_vs_t.pdf
_runs/two_phase_parallel_server_large_eps_1_figs/peak_l2_vs_t_logscale.pdf
_runs/two_phase_parallel_server_large_eps_1_figs/peak_l2_over_logT_vs_t.pdf
```

### Main figures: long parallel-server runs

```bash
python main.py configs/two_phase_parallel_server_0.005.json
python main.py configs/two_phase_parallel_server_0.001.json
python main.py configs/two_phase_parallel_server_0.0005.json
python main.py configs/two_phase_parallel_server_0.0001.json
```

The native fast path is used automatically for these ring parallel-server configs.
Plot the sweep parent with:

```bash
PYTHONPATH=src python scripts/plot_logscale.py _runs/two_phase_parallel_server --out_dir _runs/two_phase_parallel_server_figs --band_mult 1.96
```

---

## Experiment 2 — Two-Queue MaxWeight: Simulation vs Theoretical Bounds

Self-contained notebook; requires only `numpy` and `matplotlib`.

```bash
cd notebooks
jupyter notebook 2q.ipynb
```

Run all cells top-to-bottom. The notebook generates figures comparing the empirical
peak of a two-queue MaxWeight system against the global corollary bound and the local
high-probability bound from the paper.

---

## Experiment 3 — CRP vs Non-CRP, Synchronous vs Independent Arrivals

Self-contained notebook; requires `numpy`, `scipy`, and `matplotlib`.

```bash
cd notebooks
jupyter notebook CRP_in.ipynb
```

Run all cells top-to-bottom. Two cells:
- **Cell 1** — synchronous vs independent Bernoulli arrivals in IQS
- **Cell 2** — CRP vs non-CRP arrival profiles in IQS

---

## Building a Clean Submission Archive

To create a source-only zip without generated outputs, local caches, or the
historical nested `Code/` copy, run:

```bash
python scripts/make_submission_archive.py
```

The archive is written to `dist/qpeak_neurips_code.zip`.
