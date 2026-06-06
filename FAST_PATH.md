# Native fast path for parallel-server ring experiments

This codebase includes a native simulator for the large parallel-server ring experiments.

## What it accelerates

The fast path is used automatically when all of the following hold:

- `model.type == "parallel_server"`
- `L == K <= 63`
- edges are exactly the ring/zigzag graph: `(l,l)` and `(l,(l+1) mod L)`
- `arrivals.type == "bernoulli_customer"` with uniform rates, or omitted rates so the code uses `(1-epsilon)*mu*K/L`
- `service_time.type == "geometric"`
- `policy.type == "maxweight"`
- `recording.full_state == false`

For other configs the original Python engine remains the fallback.

## Main changes

- Replaced per-slot Python/SciPy assignment with a C++17/OpenMP simulator.
- Used the memoryless equivalence of geometric service times: a busy server completes at the next scheduling epoch with probability `mu`.
- Sampled uniform Bernoulli vectors exactly as `Binomial(n,p)` plus a uniform subset, instead of drawing one RNG value per class/server.
- Added an exact dynamic program for ring MaxWeight assignment. In heavy traffic the code usually takes an even faster greedy branch; when a `q_l=1` capacity conflict is possible, it falls back to the exact DP.
- Maintained `totalQ` and `||Q||_2^2` incrementally, so peak metrics do not rescan the 40-dimensional queue vector each slot.
- Parallelized replications across CPU threads.

## Usage

Run the same command:

```bash
python main.py configs/two_phase_parallel_server_0.005.json
```

Optional controls in the `simulation` block:

```json
{
  "fast_mode": "auto",
  "num_threads": 10
}
```

`fast_mode: "off"` disables the native path. `fast_mode: "native"` or `"force"` raises an error if the config is not eligible.

The first run compiles `src/qpeak/simulation/fast_parallel_server.cpp` into a local shared object under `src/qpeak/simulation/_native_build/`. Set `CXX=/path/to/g++` if the compiler is not on `PATH`.

## Benchmark used for the estimate

On the execution environment used to prepare this patch, the native simulator sustained approximately:

- 4.2 million slots/s with 1 thread
- 8.4 million slots/s with 2 threads
- 16.0 million slots/s with 4 threads
- 28.6 million slots/s with 10 threads

The target config has `T=1,000,000,000` and `R=50`, i.e. `50,000,000,000` simulated slots. At 28.6 million slots/s, that is about 29 minutes of simulator time, plus a small compile/startup overhead and output writing. A conservative estimate on a 10-CPU machine is 30--45 minutes.

## Reproducibility note

The native path samples the same arrival law and the same geometric-service law, and it solves the ring MaxWeight optimization exactly. It does not reproduce the old path bit-for-bit because it uses a different RNG and deterministic tie-breaking instead of SciPy's assignment tie-breaking.
