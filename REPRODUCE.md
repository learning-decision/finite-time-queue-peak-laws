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

## Quick Smoke Tests

These commands check the end-to-end pipeline on small instances.

```bash
python main.py configs/smoke_iqs.json
python main.py configs/smoke_parallel_server.json
PYTHONPATH=src python scripts/plot_logscale.py _runs/smoke_iqs --out_dir _runs/smoke_iqs_figs --band_mult 1.96
```

## Appendix Large-Slack Experiments

These are the small-horizon experiments used to show that logarithmic behavior
can emerge quickly when the slack is not extremely small.

```bash
python main.py configs/two_phase_iqs.json
PYTHONPATH=src python scripts/plot_logscale.py _runs/two_phase_iqs_large_eps --out_dir _runs/two_phase_iqs_large_eps_figs --band_mult 1.96

python main.py configs/two_phase_parallel_server_1.json
PYTHONPATH=src python scripts/plot_logscale.py _runs/two_phase_parallel_server_large_eps_1 --out_dir _runs/two_phase_parallel_server_large_eps_1_figs --band_mult 1.96
```

Suggested PDFs for the appendix:

```text
_runs/two_phase_iqs_large_eps_figs/peak_l2_vs_t_logscale.pdf
_runs/two_phase_iqs_large_eps_figs/peak_l2_over_logT_vs_t.pdf
_runs/two_phase_parallel_server_large_eps_1_figs/peak_l2_vs_t_logscale.pdf
_runs/two_phase_parallel_server_large_eps_1_figs/peak_l2_over_logT_vs_t.pdf
```

## Long Parallel-Server Experiments

The main long-horizon parallel-server configs are:

```bash
python main.py configs/two_phase_parallel_server_0.005.json
python main.py configs/two_phase_parallel_server_0.001.json
python main.py configs/two_phase_parallel_server_0.0005.json
python main.py configs/two_phase_parallel_server_0.0001.json
```

The native fast path is used automatically for these ring parallel-server
configs. Plot the sweep parent with:

```bash
PYTHONPATH=src python scripts/plot_logscale.py _runs/two_phase_parallel_server --out_dir _runs/two_phase_parallel_server_figs --band_mult 1.96
```

## Building a Clean Submission Archive

To create a source-only zip without generated outputs, local caches, or the
historical nested `Code/` copy, run:

```bash
python scripts/make_submission_archive.py
```

The archive is written to `dist/qpeak_neurips_code.zip`.
