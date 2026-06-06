# Design Decisions

This document records the key design decisions for the `qpeak` simulator.
It serves as the authoritative specification for model semantics, config schema,
and simulation contracts.

---

## Repository layout

- **Package code** lives under `src/qpeak/`.
- **Entry point:** repo-root `main.py` prepends `src` to `sys.path` so runs work with:
  ```bash
  python main.py <config.json|yaml>
  ```
- **Dependencies:** see `requirements.txt` (NumPy, SciPy, Matplotlib, PyYAML).

---

## Configuration

- **Formats:** YAML or JSON on disk.
- **JSON documentation keys:** any `"_comment"` key is stripped recursively before validation (see `qpeak.config_io.remove_comment_keys`).
- **Required top-level blocks:** `model`, `arrivals`, `policy`, `simulation`.
- **Registry-driven blocks** (`type` field required): `model`, `arrivals`, `policy` only — **`simulation` has no `type`.**
- **Optional top-level:** `recording`, `metrics` (if present, `metrics` must be a mapping).

### Single slack parameter

- **One canonical slack value per simulation run** where the model uses heavy-traffic slack (IQS and GG1): after loading config, the engine always sees **`model.epsilon`** as a single scalar in **(0, 1)**.
- **On disk:** each `run_manifest.json` stores that resolved **`model.epsilon`** (and does **not** retain `model.epsilons`), so plotting and analysis read one slack per bundle without ambiguity.
- **Default arrival laws** should be **derived from the model** (no duplicate "slack" in `arrivals` unless you intentionally override rates).

### `model.epsilon` vs `model.epsilons` (all slack-based models)

- Set **exactly one** of:
  - **`model.epsilon`:** a single number in **(0, 1)** — one CLI invocation produces **one** output directory.
  - **`model.epsilons`:** a **non-empty list** of numbers, each in **(0, 1)** — one CLI invocation runs **one experiment per list entry**.
- **Do not** set both `epsilon` and `epsilons` in the same model block.
- **Legacy note:** `model.epsilon` may appear as a list in some configs; this is treated as an alias for `model.epsilons` but new configs should use `model.epsilons` explicitly.
- **Output layout for `epsilons`:** let `B` be the directory from `simulation.output_dir` if set, otherwise `_runs/<config_stem>/<utc_timestamp>`. The `i`-th value is written under **`B / eps_<slug>`**, where `<slug>` is a filesystem-safe encoding of that ε (e.g. `0.1` → `0p1`). Each subdirectory is a full artifact bundle (`run_manifest.json`, `aggregate.npz`, optional `paths/`).

### IQS model block (`model.type: iqs`)

- **`model.n`:** required, positive integer.
- **Slack:** exactly one of **`model.epsilon`** or **`model.epsilons`**; each entry strictly in **(0, 1)**.
- **State:** \(Q_t\) is an **`n×n`** VOQ matrix.

### Parallel-server model block (`model.type: parallel_server`)

Skill-based routing with **persistent servers**.

- **Graph:** bipartite compatibility \(G=(\mathcal L,\mathcal K,\mathcal E)\). Left = customer classes, right = servers.
- **`model.L`, `model.K`:** required, positive integers.
- **`model.edges`:** required, non-empty list of **0-based** `[[l, k], ...]` pairs.
- **Service-time distribution (`model.service_time`):** `{"type": "geometric", "mu": <x>}` with `mu` in \((0, 1]\); \(E[D] = 1/\mu\).
- **Slack:** exactly one of **`model.epsilon`** or **`model.epsilons`** in \((0, 1)\).
- **Primary state:** customer queues \(Q_t \in \mathbb{Z}_+^L\).
- **Auxiliary state (policy-internal):** per-server `busy[k]`, `remaining[k]`, `assigned_class[k]`. Reset via `policy.reset(rng)` at the start of each replication.
- **Initial condition:** \(Q_0 = 0\); all servers idle.
- **Per-slot event order:**
  1. Observe \(Q_t\).
  2. **Server tick:** decrement `remaining[k]`; if 0, mark idle.
  3. **Matching:** assign idle servers to non-empty compatible classes (MaxWeight).
  4. **Arrivals:** sample \(A_{t+1}\).
  5. **Update:** \(Q_{t+1} = (Q_t + A_{t+1} - S_t)^+\).

### Arrivals (registry)

- **`iid_bernoulli`** (IQS): per-VOQ Bernoulli rate \(\lambda_{ij}=(1-\varepsilon)/n\).
- **`bernoulli_customer`** (parallel server): default \(\lambda_\ell=(1-\varepsilon)\mu K/L\); optional `arrivals.lambdas` override.

### Simulation block

- **`T`:** number of slots; trajectories use \(t = 0, \ldots, T-1\).
- **`seed`:** RNG seed.
- **`num_replications` (optional):** default **10**.
- **`output_dir`:** output directory. If omitted, defaults to `_runs/<config_stem>/<utc_timestamp>`.
- **`progress`** (bool, default `false`): enable periodic progress output.
- **`progress_every`** (int, default `1000000`): print update every this many slots.

---

## Metrics and time range

- **No default burn-in discard:** metrics use the **full** range **`[0, T-1]`**.

---

## Recording (trajectories)

- **`recording` omitted** ⇒ default behavior.
- **`num_paths_to_save` (optional):** default `min(num_replications, 10)`.
- **`full_state: true`:** store \(Q_t\) at every slot for saved paths.
- **`full_state: false`:** store downsampled \(Q\) on the `metrics.downsample` grid.

### Metrics (paper-facing aggregates)

- **Always-on:** `peak_l1_so_far` and `peak_l2_so_far` (mean and SE across replications).
- **Additional:** specify via `metrics.series: [...]`.
- **Downsampling:** `metrics.downsample` with `K` (target output length) and `grid` (`raw_time` | `scaled_tau`).

### On-disk layout

- **`run_manifest.json`:** metadata and full config snapshot.
- **`aggregate.npz`:** aggregate time series on the chosen grid.
- **`paths/rep_XXXX.npz` (optional):** up to `num_paths_to_save` diagnostic sample paths.

---

## Dynamics and composition order

- **Discrete-time slot order:**
  1. Observe \(Q_t\).
  2. Policy chooses feasible \(S_t\).
  3. Sample arrivals \(A_{t+1}\).
  4. Update \(Q_{t+1} = (Q_t + A_{t+1} - S_t)^+\).

- **Config composition order:** build `model` → `arrivals` → `policy`.

- **Independent RNG streams per replication:** derived from the master seed via `numpy.random.SeedSequence(seed).spawn(...)`.

---

## Policies

### MaxWeight

- Supported for **IQS** and **parallel_server**.
- **IQS solver:** `scipy.optimize.linear_sum_assignment` on `-Q` (O(n³) per slot).
- **Parallel server:** extended bipartite assignment using SciPy's linear sum assignment.

---

## Registries

- **`MODEL_REGISTRY`**, **`ARRIVAL_REGISTRY`**, **`POLICY_REGISTRY`:** `type` strings dispatch to factories.
- Importing `qpeak` registers all factories as a side effect.
