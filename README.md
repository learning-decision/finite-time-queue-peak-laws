# Finite-Time Queue Peak Laws in Stochastic Networks

Simulation code for the paper:

> **Finite-Time Queue Peak Laws in Stochastic Networks: Logarithmic Scaling After Geometric Thresholds**  
> Hao Liang, Cheng Tang, Yunzong Xu  
> [arXiv:2606.18218](https://arxiv.org/abs/2606.18218)

---

## Overview

This repository contains the config-driven simulation framework used to produce
all numerical experiments in the paper. The code simulates discrete-time stochastic
queueing networks under MaxWeight scheduling and measures the finite-time peak of
various queue norms. The main finding explored numerically is a **two-phase envelope**:
peak queue growth scales as √T for small horizons, then crosses over to logarithmic
scaling after a geometry-dependent threshold.

### Experiments and models

| Experiment | System | Implementation |
|---|---|---|
| **Exp 1 — Two-phase envelope** | IQS (n×n VOQ) and parallel-server network | `qpeak` package + JSON configs |
| **Exp 2 — Two-queue theory bounds** | Two-queue MaxWeight (`S = {(1,0), (0,a)}`) | `notebooks/2q.ipynb` |
| **Exp 3 — CRP and arrival structure** | IQS with CRP/non-CRP profiles; synchronous vs independent arrivals | `notebooks/CRP_in.ipynb` |

The `qpeak` package (experiments 1) supports two model types:

| `model.type` | System |
|---|---|
| `iqs` | Input-queued switch (n×n VOQ, MaxWeight scheduling) |
| `parallel_server` | Skill-based routing with persistent servers and geometric service times |

Experiments 2 and 3 are self-contained notebooks that do not use the `qpeak` package.

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/learning-decision/finite-time-queue-peak-laws.git
cd finite-time-queue-peak-laws
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e .
```

### Native fast path (optional, for long parallel-server runs)

The large parallel-server experiments use a C++17/OpenMP native simulator that is
compiled automatically on first use. This requires a C++17 compiler (`g++` or `clang++`)
on your PATH. No extra Python package is needed.

```bash
# Verify your compiler is available:
g++ --version
```

If no compiler is available, add `"fast_mode": "off"` to the `simulation` block of
the config to fall back to the pure-Python engine. The Python engine is slower
(~100× for the ring MaxWeight assignment) but produces equivalent results.

To control threading explicitly:

```json
"simulation": { "fast_mode": "auto", "num_threads": 10 }
```

`fast_mode` accepts `"auto"` (default), `"off"`, or `"native"` / `"force"`.

---

## Reproducing paper figures

### Quick smoke test (verify the pipeline)

```bash
python main.py configs/smoke_iqs.json
python main.py configs/smoke_parallel_server.json
PYTHONPATH=src python scripts/plot_logscale.py _runs/smoke_iqs --out_dir _runs/smoke_iqs_figs --band_mult 1.96
```

---

### Experiment 1 — Two-phase envelope (IQS + parallel server)

**Appendix figures — large-slack experiments (~minutes on a laptop)**

```bash
bash scripts/reproduce_fig_appendix.sh
```

Key output PDFs:

```
_runs/two_phase_iqs_large_eps_figs/peak_l2_vs_t_logscale.pdf
_runs/two_phase_iqs_large_eps_figs/peak_l2_over_logT_vs_t.pdf
_runs/two_phase_parallel_server_large_eps_1_figs/peak_l2_vs_t_logscale.pdf
_runs/two_phase_parallel_server_large_eps_1_figs/peak_l2_over_logT_vs_t.pdf
```

**Main figures — long parallel-server runs (~30–45 min on 10 cores)**

```bash
bash scripts/reproduce_fig_main.sh
```

Runs four epsilon values (0.005, 0.001, 0.0005, 0.0001) and produces:

```
_runs/two_phase_parallel_server_figs/
```

See `REPRODUCE.md` for detailed commands and parameter notes.

---

### Experiment 2 — Two-queue MaxWeight: simulation vs theoretical bounds

Self-contained notebook; requires only `numpy` and `matplotlib`.

```bash
cd notebooks
jupyter notebook 2q.ipynb
```

Produces figures comparing the empirical peak of a two-queue MaxWeight system
against the global corollary bound and the local high-probability bound from the paper.

---

### Experiment 3 — CRP vs non-CRP, synchronous vs independent arrivals

Self-contained notebook; requires `numpy`, `scipy`, and `matplotlib`.

```bash
cd notebooks
jupyter notebook CRP_in.ipynb
```

Two cells:
- **Cell 1** — synchronous vs independent Bernoulli arrivals in IQS
- **Cell 2** — CRP vs non-CRP arrival profiles in IQS

---

### Reproduce all experiments

```bash
bash scripts/reproduce_all.sh   # Experiment 1 only (shell scripts)
```

For experiments 2 and 3, open the notebooks above.

---

## Running a single experiment

All experiments are driven by JSON configs:

```bash
python main.py configs/<name>.json
```

Outputs land in `_runs/<name>/` (or the path set in `simulation.output_dir`).
Each run writes:

- `aggregate.npz` — mean and SE of peak queue norms across replications, on a downsampled grid
- `run_manifest.json` — frozen config snapshot with timestamps
- `paths/rep_XXXX.npz` — raw sample paths (first `num_paths_to_save` replications)

For an epsilon sweep (`model.epsilons: [...]`), one sub-directory is created per
epsilon value: `_runs/<name>/eps_0p005/`, `eps_0p001/`, etc.

---

## Plotting

All three plotting scripts read `aggregate.npz` and auto-expand sweep parents.

```bash
# Log-scale x-axis, beta(T) panel, per-epsilon overlay (main paper figures)
PYTHONPATH=src python scripts/plot_logscale.py _runs/<name> --out_dir _runs/<name>_figs --band_mult 1.96

# Linear x-axis, overlay multiple run dirs
PYTHONPATH=src python scripts/plot_aggregate.py _runs/<run1> _runs/<run2> --out_dir _runs/combined_figs
```

Common flags: `--band_mult 1.96` (95% bands), `--format pdf` (default) or `png`.

---

## Repository structure

```
finite-time-queue-peak-laws/
├── main.py                    entry point: python main.py <config>
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── README.md
├── REPRODUCE.md               detailed reproduction guide with all commands
├── DESIGN_DECISIONS.md        model spec, config schema, simulation contracts
├── CONTRIBUTING.md
├── CODE_RELEASE_CHECKLIST.md
├── configs/                   JSON experiment configs (one file = one experiment)
│   ├── smoke_iqs.json                          quick IQS smoke test
│   ├── smoke_parallel_server.json              quick parallel-server smoke test
│   ├── two_phase_iqs.json                      IQS appendix experiment
│   ├── two_phase_parallel_server_1.json        parallel-server appendix
│   └── two_phase_parallel_server_0.00*.json    main long-horizon runs
├── scripts/
│   ├── reproduce_fig_appendix.sh
│   ├── reproduce_fig_main.sh
│   ├── reproduce_all.sh
│   ├── plot_logscale.py       main figure script (log x-axis, beta panel)
│   ├── plot_aggregate.py      overlay multiple runs (linear x-axis)
│   └── make_submission_archive.py
├── src/qpeak/                 simulation package
│   ├── models/                iqs, parallel_server
│   ├── arrivals/              bernoulli arrival processes
│   ├── policies/              maxweight
│   ├── service_times/         geometric service-time distribution
│   ├── simulation/            Python engine + C++/OpenMP native fast path
│   ├── recording/             sample-path saving
│   └── cli.py, config_io.py, compose.py, registries.py, types.py
├── notebooks/
│   ├── 2q.ipynb               experiment 2: two-queue MaxWeight vs theory bounds
│   └── CRP_in.ipynb           experiment 3: CRP vs non-CRP, synchronous vs independent arrivals
└── figures/                   selected output figures (PDF)
```

---

## Contact

For questions about the paper or code, please open a GitHub issue or contact the authors.

Hao Liang, Cheng Tang, Yunzong Xu ({hliang17,chengt6,xyz}@illinois.edu)
