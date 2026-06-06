# Contributing

Thank you for your interest in this repository.

This is primarily a research-code release accompanying the paper
"Finite-Time Queue Peak Laws in Stochastic Networks: Logarithmic Scaling After Geometric Thresholds."
The main purpose of the repository is reproducibility, not ongoing feature development.
That said, bug reports and targeted improvements are welcome.

## Ground rules

- Keep `main` runnable. The smoke tests (`configs/smoke_*.json`) should always pass.
- Put reusable simulation and model code in `src/qpeak/`.
- Put experiment-running scripts (Python or shell) in `scripts/`.
- Avoid hard-coded absolute paths. Use `Path(__file__).resolve().parent` or relative paths.
- Document key parameters and random seeds in configs or in `REPRODUCE.md`.

## Repository layout

```
src/qpeak/     simulation package (models, arrivals, policies, engine)
configs/       JSON experiment configs (one config = one experiment)
scripts/       plotting scripts and shell reproduction wrappers
main.py        entry point: python main.py <config.json>
```

## Running experiments

```bash
python main.py configs/smoke_iqs.json           # quick smoke test
bash scripts/reproduce_fig_appendix.sh          # appendix figures (~minutes)
bash scripts/reproduce_fig_main.sh              # main figures (~30-45 min, 10 cores)
```

## Adding a new model or policy

1. Add a module under `src/qpeak/models/` or `src/qpeak/policies/`.
2. Register it with the appropriate registry using `@register(REGISTRY, "type_name")`.
3. Import the new module in the corresponding `__init__.py` so the registry entry is created on package import.
4. Add a smoke config under `configs/` to verify the end-to-end pipeline.

## Reporting issues

Please open a GitHub issue with:
- the exact command that failed,
- the config file used (or a minimal reproduction),
- the Python version and OS.
