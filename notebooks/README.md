# Notebooks

These notebooks contain the simulation code for experiments 2 and 3 of the paper.
Each notebook is self-contained and requires only `numpy`, `scipy`, and `matplotlib`.

## `2q.ipynb` — Experiment 2: Two-Queue Example

Simulates a two-queue MaxWeight system with capacity set `S = {(1, 0), (0, a)}`
and compares the empirical peak against two theoretical upper bounds:

- **Global theory estimate** (Corollary 2.1): `(A_max / ε) log(t+1) + (A_max + S_max)² / (ε · α₂)`
- **Local high-probability estimate** (Proposition): `1 + (1/ε)(1 + log(t+1))`

Run the notebook and the figures are saved as PDF files in the same directory.

**Key parameters:**
- `a` — asymmetry parameter of the capacity set
- `eps` — slack `ε`
- `T` — horizon
- `num_runs` — number of Monte Carlo replications

## `CRP_in.ipynb` — Experiment 3: CRP vs non-CRP, Synchronous vs Independent Arrivals

Two cells demonstrating geometric effects on peak behavior in IQS (input-queued switch):

**Cell 0 — Synchronous vs independent arrivals:**
Compares `E[max ||Q||₁]` under two arrival laws with identical marginals:
- Synchronous: all entries arrive together with probability `(1-ε)/n`
- Independent: each entry is iid Bernoulli(`(1-ε)/n`)

**Cell 1 — CRP vs non-CRP arrival profiles:**
Compares `E[max ||Q||₁]` under two structured arrival profiles:
- CRP profile: first row uniform `1/n`, diagonal `1/n`, rest `1/n²`
- non-CRP profile: first row uniform `1/n`, first column `1/n`, rest `1/n²`

Run each cell to produce PDF figures in the same directory.

## Running the notebooks

```bash
cd notebooks
jupyter notebook
```

Or run them non-interactively:

```bash
jupyter nbconvert --to notebook --execute notebooks/2q.ipynb
jupyter nbconvert --to notebook --execute notebooks/CRP_in.ipynb
```
