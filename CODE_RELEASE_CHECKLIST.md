# Code Release Checklist

Work through this list before switching the repository to public.

## Documentation
- [ ] README is written for an external reader (not internal team notes)
- [ ] arXiv link added to README once the paper appears
- [ ] Paper title and authors are correct in README
- [ ] `REPRODUCE.md` tested end-to-end on a clean environment
- [ ] Reproduction scripts (`scripts/reproduce_*.sh`) run without error on a fresh clone

## Dependencies
- [ ] `requirements.txt` matches all packages actually imported
- [ ] `pyproject.toml` version constraints are consistent with `requirements.txt`
- [ ] Python version requirement stated in README and `pyproject.toml`
- [ ] C++ compiler requirement documented (for native fast path)

## Reproducibility
- [ ] Random seeds are fixed in all paper configs and documented in `REPRODUCE.md`
- [ ] All figures in the paper can be regenerated from configs + scripts
- [ ] Smoke tests pass: `python main.py configs/smoke_iqs.json` and `python main.py configs/smoke_parallel_server.json`
- [ ] Native fast path compiles and runs correctly on a clean machine

## Code hygiene
- [ ] No hard-coded absolute paths (`/Users/...`) anywhere in committed code
- [ ] No credentials, tokens, API keys, or private data in any file
- [ ] `.gitignore` covers `_runs/`, `_native_build/`, `__pycache__/`, `.DS_Store`, `.venv/`
- [ ] `_runs/` directory is not tracked by git (check with `git status`)
- [ ] Compiled binary `_native_build/libqpeak_fast_parallel_server.dylib` is not tracked
- [ ] No large unnecessary files committed (check with `git ls-files | xargs wc -c | sort -n | tail`)

## Licensing
- [ ] `LICENSE` file present at repo root
- [ ] License type and year are correct
- [ ] All authors listed in LICENSE and README

## Repository visibility
- [ ] Repository name set to `finite-time-queue-peak-laws` (or agreed name)
- [ ] Repository is under the `learning-decision` organization
- [ ] All team members have appropriate access before switching to public
- [ ] Switch repository visibility to public after arXiv submission
