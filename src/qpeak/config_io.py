"""Load experiment configs (YAML / JSON), strip JSON-only comments, validate shape."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

REQUIRED_TOP_LEVEL = ("model", "arrivals", "policy", "simulation")
BLOCKS_REQUIRING_TYPE = ("model", "arrivals", "policy")

STREAMING_PATH_REGISTRY: Final[frozenset[str]] = frozenset(
    {
        "peak_l1_so_far",
        "peak_l2_so_far",
        "argmax_time_l1",
        "argmax_time_l2",
    }
)

_DOWNSAMPLE_GRIDS: Final[tuple[str, str]] = ("raw_time", "scaled_tau")


def uses_model_epsilons_sweep(model: dict[str, Any]) -> bool:
    """
    True if the model block lists multiple slack values.

    Supported encodings:
      - model.epsilons: list[float]
      - model.epsilon:  list[float]  (legacy alias; normalized to epsilons during validation)
    """
    if "epsilons" in model:
        return True
    return isinstance(model.get("epsilon"), list)


def epsilon_output_slug(eps: float) -> str:
    """Filesystem-friendly token for a sweep subdirectory, e.g. ``0.01`` → ``0p01``."""
    x = float(eps)
    s = f"{x:.12g}"
    if "e" in s or "E" in s:
        s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def resolved_config_with_epsilon(cfg: dict[str, Any], eps: float) -> dict[str, Any]:
    """
    Deep copy of ``cfg`` with a single canonical ``model.epsilon`` and no ``model.epsilons``.

    Each on-disk run manifest uses this resolved shape so consumers always read one slack value.
    """
    out = copy.deepcopy(cfg)
    mm = dict(out["model"])
    mm["epsilon"] = float(eps)
    mm.pop("epsilons", None)
    # If a legacy epsilon-list was present, it is overwritten by the scalar above.
    out["model"] = mm
    return out


def load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load .yaml/.yml configs. "
                "Install pyyaml or use a .json config."
            ) from e
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension {suffix!r}; use .json, .yaml, or .yml")
    if not isinstance(data, dict):
        raise TypeError("Config root must be a mapping (JSON object / YAML mapping).")
    return data


def remove_comment_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: remove_comment_keys(v) for k, v in obj.items() if k != "_comment"}
    if isinstance(obj, list):
        return [remove_comment_keys(x) for x in obj]
    return obj


def validate_config_shape(cfg: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise KeyError(f"Config missing required top-level keys: {missing}")
    for block in REQUIRED_TOP_LEVEL:
        inner = cfg[block]
        if not isinstance(inner, dict):
            raise TypeError(f"Config[{block!r}] must be an object/mapping.")
        if block in BLOCKS_REQUIRING_TYPE and "type" not in inner:
            raise KeyError(f"Config[{block!r}] must include a 'type' field.")
    if "metrics" in cfg and not isinstance(cfg["metrics"], dict):
        raise TypeError("Optional key 'metrics' must be an object/mapping when present.")
    validate_model_parameters(cfg["model"])
    validate_arrivals_for_model(cfg)
    validate_recording(cfg)
    validate_metrics(cfg)
    validate_simulation_options(cfg["simulation"])


def _validate_slack_scalar_or_list(model: dict[str, Any], *, kind: str) -> None:
    """
    Require exactly one of:
      - model.epsilon: scalar in (0,1)
      - model.epsilons: non-empty list of scalars in (0,1)

    Also supports a legacy alias: model.epsilon may be a list, which is treated as model.epsilons.
    """
    has_epsilons = "epsilons" in model
    eps_val = model.get("epsilon", None)
    epsilon_is_list = isinstance(eps_val, list)
    has_epsilon_key = "epsilon" in model

    if has_epsilons and has_epsilon_key and not epsilon_is_list:
        raise ValueError(
            f"When model.type is {kind!r}, set exactly one of 'model.epsilon' (scalar) or "
            f"'model.epsilons' (non-empty list), not both."
        )

    if has_epsilons:
        raw = model["epsilons"]
    elif epsilon_is_list:
        # Normalize legacy encoding: treat epsilon:[...] as epsilons:[...]
        raw = eps_val
        model["epsilons"] = raw
        model.pop("epsilon", None)
    else:
        if "epsilon" not in model:
            raise ValueError(
                f"When model.type is {kind!r}, set one of 'model.epsilon' (scalar) or "
                f"'model.epsilons' (non-empty list)."
            )
        eps = model["epsilon"]
        if not isinstance(eps, (int, float)) or isinstance(eps, bool):
            raise TypeError("model.epsilon must be a numeric value in (0, 1).")
        if not (0 < float(eps) < 1):
            raise ValueError("model.epsilon must lie strictly in (0, 1) (heavy-traffic slack).")
        return

    if not isinstance(raw, list) or len(raw) < 1:
        raise TypeError("model.epsilons must be a non-empty list when present.")
    for i, eps in enumerate(raw):
        if not isinstance(eps, (int, float)) or isinstance(eps, bool):
            raise TypeError(f"model.epsilons[{i}] must be a numeric value in (0, 1).")
        if not (0 < float(eps) < 1):
            raise ValueError(f"model.epsilons[{i}] must lie strictly in (0, 1).")


def validate_model_parameters(model: dict[str, Any]) -> None:
    mtype = model["type"]
    if mtype == "iqs":
        _validate_slack_scalar_or_list(model, kind="iqs")
        if "n" not in model:
            raise KeyError("model.n is required when model.type is 'iqs'.")
        n = model["n"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise TypeError("model.n must be a positive integer when model.type is 'iqs'.")
    elif mtype == "gg1":
        _validate_slack_scalar_or_list(model, kind="gg1")
    elif mtype == "bipartite_matching":
        _validate_slack_scalar_or_list(model, kind="bipartite_matching")
        if "L" not in model or "K" not in model:
            raise KeyError("model.L and model.K are required when model.type is 'bipartite_matching'.")
        L = model["L"]
        K = model["K"]
        if not isinstance(L, int) or isinstance(L, bool) or L < 1:
            raise TypeError("model.L must be a positive integer for model.type 'bipartite_matching'.")
        if not isinstance(K, int) or isinstance(K, bool) or K < 1:
            raise TypeError("model.K must be a positive integer for model.type 'bipartite_matching'.")
        if "edges" not in model:
            raise KeyError("model.edges is required for model.type 'bipartite_matching'.")
        edges = model["edges"]
        if not isinstance(edges, list) or len(edges) < 1:
            raise TypeError("model.edges must be a non-empty list of [l, r] pairs (0-based).")
        for i, e in enumerate(edges):
            if not isinstance(e, list) or len(e) != 2:
                raise TypeError(f"model.edges[{i}] must be a length-2 list [l, r].")
            l, r = e[0], e[1]
            if not isinstance(l, int) or isinstance(l, bool):
                raise TypeError(f"model.edges[{i}][0] must be an int.")
            if not isinstance(r, int) or isinstance(r, bool):
                raise TypeError(f"model.edges[{i}][1] must be an int.")
            if not (0 <= l < int(L)):
                raise ValueError(f"model.edges[{i}][0] out of range for L={L}.")
            if not (0 <= r < int(K)):
                raise ValueError(f"model.edges[{i}][1] out of range for K={K}.")
    elif mtype == "parallel_server":
        _validate_slack_scalar_or_list(model, kind="parallel_server")
        if "L" not in model or "K" not in model:
            raise KeyError("model.L and model.K are required when model.type is 'parallel_server'.")
        L = model["L"]
        K = model["K"]
        if not isinstance(L, int) or isinstance(L, bool) or L < 1:
            raise TypeError("model.L must be a positive integer for model.type 'parallel_server'.")
        if not isinstance(K, int) or isinstance(K, bool) or K < 1:
            raise TypeError("model.K must be a positive integer for model.type 'parallel_server'.")
        if "edges" not in model:
            raise KeyError("model.edges is required for model.type 'parallel_server'.")
        edges = model["edges"]
        if not isinstance(edges, list) or len(edges) < 1:
            raise TypeError("model.edges must be a non-empty list of [l, k] pairs (0-based).")
        for i, e in enumerate(edges):
            if not isinstance(e, list) or len(e) != 2:
                raise TypeError(f"model.edges[{i}] must be a length-2 list [l, k].")
            l, k = e[0], e[1]
            if not isinstance(l, int) or isinstance(l, bool):
                raise TypeError(f"model.edges[{i}][0] must be an int.")
            if not isinstance(k, int) or isinstance(k, bool):
                raise TypeError(f"model.edges[{i}][1] must be an int.")
            if not (0 <= l < int(L)):
                raise ValueError(f"model.edges[{i}][0] out of range for L={L}.")
            if not (0 <= k < int(K)):
                raise ValueError(f"model.edges[{i}][1] out of range for K={K}.")
        if "service_time" not in model:
            raise KeyError("model.service_time is required for model.type 'parallel_server'.")
        st = model["service_time"]
        if not isinstance(st, dict):
            raise TypeError("model.service_time must be an object/mapping.")
        if "type" not in st:
            raise KeyError("model.service_time.type is required.")
        if st["type"] == "geometric":
            if "mu" not in st:
                raise KeyError("model.service_time.mu is required when service_time.type is 'geometric'.")
            mu = st["mu"]
            if not isinstance(mu, (int, float)) or isinstance(mu, bool):
                raise TypeError("model.service_time.mu must be numeric in (0, 1].")
            if not (0.0 < float(mu) <= 1.0):
                raise ValueError("model.service_time.mu must lie in (0, 1].")


def validate_arrivals_for_model(cfg: dict[str, Any]) -> None:
    mtype = cfg["model"]["type"]
    arr = cfg["arrivals"]
    atype = arr["type"]
    if mtype == "gg1":
        if atype != "bernoulli_gg1":
            raise ValueError("When model.type is 'gg1', arrivals.type must be 'bernoulli_gg1'.")
        if "lambda" in arr:
            lam = arr["lambda"]
            if not isinstance(lam, (int, float)) or isinstance(lam, bool):
                raise TypeError("arrivals.lambda must be numeric in (0, 1) when present.")
            if not (0 < float(lam) < 1):
                raise ValueError("arrivals.lambda must lie strictly in (0, 1) when present.")
    if mtype == "parallel_server":
        if atype != "bernoulli_customer":
            raise ValueError(
                "When model.type is 'parallel_server', arrivals.type must be 'bernoulli_customer'."
            )
        L = int(cfg["model"]["L"])
        if "lambdas" in arr:
            lam = arr["lambdas"]
            if not isinstance(lam, list) or len(lam) != L:
                raise TypeError("arrivals.lambdas must be a list of length L.")
            for i, x in enumerate(lam):
                if not isinstance(x, (int, float)) or isinstance(x, bool):
                    raise TypeError(f"arrivals.lambdas[{i}] must be numeric in (0, 1).")
                if not (0 < float(x) < 1):
                    raise ValueError(f"arrivals.lambdas[{i}] must lie strictly in (0, 1).")
    if mtype == "bipartite_matching":
        if atype != "bernoulli_bipartite":
            raise ValueError(
                "When model.type is 'bipartite_matching', arrivals.type must be 'bernoulli_bipartite'."
            )
        L = int(cfg["model"]["L"])
        K = int(cfg["model"]["K"])
        if "lambda_L" in arr and "lambda_R" not in arr:
            raise ValueError(
                "arrivals.lambda_L without arrivals.lambda_R is not supported. "
                "Set both, or set only lambda_R for asymmetric (parallel-server) mode."
            )
        if "lambda_L" in arr:
            lam_L = arr["lambda_L"]
            if not isinstance(lam_L, list) or len(lam_L) != L:
                raise TypeError("arrivals.lambda_L must be a list of length L when present.")
            for i, x in enumerate(lam_L):
                if not isinstance(x, (int, float)) or isinstance(x, bool):
                    raise TypeError(f"arrivals.lambda_L[{i}] must be numeric in (0,1).")
                if not (0 < float(x) < 1):
                    raise ValueError(f"arrivals.lambda_L[{i}] must lie strictly in (0,1).")
        if "lambda_R" in arr:
            lam_R = arr["lambda_R"]
            if not isinstance(lam_R, list) or len(lam_R) != K:
                raise TypeError("arrivals.lambda_R must be a list of length K when present.")
            for i, x in enumerate(lam_R):
                if not isinstance(x, (int, float)) or isinstance(x, bool):
                    raise TypeError(f"arrivals.lambda_R[{i}] must be numeric in (0,1).")
                if not (0 < float(x) < 1):
                    raise ValueError(f"arrivals.lambda_R[{i}] must lie strictly in (0,1).")


def validate_recording(cfg: dict[str, Any]) -> None:
    if "recording" not in cfg:
        return
    rec = cfg["recording"]
    if not isinstance(rec, dict):
        raise TypeError("Optional key 'recording' must be an object/mapping when present.")

    if "num_paths_to_save" in rec:
        P = rec["num_paths_to_save"]
        if not isinstance(P, int) or isinstance(P, bool) or P < 0:
            raise TypeError("recording.num_paths_to_save must be an integer >= 0 when present.")

    full_state = rec.get("full_state", True)
    if not isinstance(full_state, bool):
        raise TypeError("recording.full_state must be a boolean when present.")

    if full_state:
        sp = rec.get("streaming_paths", [])
        if sp:
            if not isinstance(sp, list):
                raise TypeError("recording.streaming_paths must be a list when present.")
            if len(sp) > 0:
                raise ValueError(
                    "When recording.full_state is true, recording.streaming_paths must be "
                    "absent or empty (reconstruct running series from full Q_t offline)."
                )
        # recording.downsample is deprecated; ignore if present.
        return

    # When full_state is false, diagnostic paths use the *metrics* downsample grid.
    metrics = cfg.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("Optional key 'metrics' must be an object/mapping when present.")
    if "downsample" not in metrics:
        raise KeyError(
            "When recording.full_state is false, metrics.downsample is required "
            "(single canonical downsample spec for aggregates and diagnostic paths)."
        )

    sp = rec.get("streaming_paths", [])
    if not isinstance(sp, list):
        raise TypeError("recording.streaming_paths must be a list when present.")
    for name in sp:
        if not isinstance(name, str):
            raise TypeError("Each entry in recording.streaming_paths must be a string.")
        if name not in STREAMING_PATH_REGISTRY:
            raise ValueError(
                f"Unknown recording.streaming_paths name {name!r}. "
                f"Allowed: {sorted(STREAMING_PATH_REGISTRY)}"
            )

def validate_metrics(cfg: dict[str, Any]) -> None:
    if "metrics" not in cfg:
        return
    m = cfg["metrics"]
    if not isinstance(m, dict):
        raise TypeError("Optional key 'metrics' must be an object/mapping when present.")
    if "series" in m:
        s = m["series"]
        if not isinstance(s, list):
            raise TypeError("metrics.series must be a list of strings when present.")
        for name in s:
            if not isinstance(name, str):
                raise TypeError("Each entry in metrics.series must be a string.")
    if "downsample" in m:
        ds = m["downsample"]
        if not isinstance(ds, dict):
            raise TypeError("metrics.downsample must be an object/mapping when present.")
        if "K" not in ds:
            raise KeyError("metrics.downsample.K is required when metrics.downsample is present.")
        K = ds["K"]
        if not isinstance(K, int) or isinstance(K, bool) or K < 2:
            raise TypeError("metrics.downsample.K must be an integer >= 2.")
        T = cfg.get("simulation", {}).get("T")
        if isinstance(T, int) and not isinstance(T, bool) and K > T:
            raise ValueError("metrics.downsample.K cannot exceed simulation.T.")
        if "grid" in ds:
            g = ds["grid"]
            if g not in _DOWNSAMPLE_GRIDS:
                raise ValueError("metrics.downsample.grid must be 'raw_time' or 'scaled_tau' when present.")


def validate_simulation_options(sim: dict[str, Any]) -> None:
    if "T" not in sim:
        raise KeyError("simulation.T is required (number of slots; trajectory indices 0..T-1).")
    T = sim["T"]
    if not isinstance(T, int) or isinstance(T, bool) or T < 1:
        raise TypeError("simulation.T must be a positive integer.")
    if "seed" not in sim:
        raise KeyError("simulation.seed is required (RNG seed for the run).")
    seed = sim["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("simulation.seed must be an integer.")
    if "num_replications" in sim:
        R = sim["num_replications"]
        if not isinstance(R, int) or isinstance(R, bool) or R < 1:
            raise TypeError("simulation.num_replications must be an integer >= 1 when present.")
    if "output_dir" in sim:
        od = sim["output_dir"]
        if not isinstance(od, str) or not od.strip():
            raise TypeError("simulation.output_dir must be a non-empty string when present.")

    if "progress" in sim:
        prog = sim["progress"]
        if not isinstance(prog, bool):
            raise TypeError("simulation.progress must be a boolean when present.")
    if "progress_every" in sim:
        pe = sim["progress_every"]
        if not isinstance(pe, int) or isinstance(pe, bool) or pe < 1:
            raise TypeError("simulation.progress_every must be an integer >= 1 when present.")


def write_run_manifest(
    output_dir: Path,
    cfg: dict[str, Any],
    *,
    config_source: Path | None = None,
) -> Path:
    """
    Write ``run_manifest.json``: UTC timestamp, full config dict, optional path to
    the config file that was loaded. Call this whenever outputs are saved so runs
    stay reproducible.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
    }
    if config_source is not None:
        payload["config_source_path"] = str(config_source.resolve())
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def summarize_config(cfg: dict[str, Any]) -> str:
    sim = cfg["simulation"]
    m = cfg["model"]
    lines = [
        "Experiment composition (scaffold):",
        f"  model.type     = {m['type']!r}",
    ]
    if m.get("type") == "iqs":
        lines.append(f"  model.n        = {m['n']!r}")
        if "epsilons" in m:
            lines.append(f"  model.epsilons = {m['epsilons']!r}  (CLI runs one bundle per value)")
        else:
            lines.append(f"  model.epsilon  = {m['epsilon']!r}")
    if m.get("type") == "gg1":
        if "epsilons" in m:
            lines.append(f"  model.epsilons = {m['epsilons']!r}  (CLI runs one bundle per value)")
        else:
            lines.append(f"  model.epsilon  = {m['epsilon']!r}")
    if m.get("type") == "bipartite_matching":
        lines.append(f"  model.L        = {m['L']!r}")
        lines.append(f"  model.K        = {m['K']!r}")
        lines.append(f"  model.|E|      = {len(m.get('edges', []))!r}")
        if "epsilons" in m:
            lines.append(f"  model.epsilons = {m['epsilons']!r}  (CLI runs one bundle per value)")
        else:
            lines.append(f"  model.epsilon  = {m['epsilon']!r}")
    if m.get("type") == "parallel_server":
        lines.append(f"  model.L        = {m['L']!r}")
        lines.append(f"  model.K        = {m['K']!r}")
        lines.append(f"  model.|E|      = {len(m.get('edges', []))!r}")
        lines.append(f"  service_time   = {m.get('service_time')!r}")
        if "epsilons" in m:
            lines.append(f"  model.epsilons = {m['epsilons']!r}  (CLI runs one bundle per value)")
        else:
            lines.append(f"  model.epsilon  = {m['epsilon']!r}")
    lines += [
        f"  arrivals.type  = {cfg['arrivals']['type']!r}",
        f"  policy.type    = {cfg['policy']['type']!r}",
        f"  simulation     = {sim!r}",
    ]
    if "metrics" in cfg:
        lines.append(f"  metrics        = {cfg['metrics']!r}")
    lines.append(f"  recording      = {_recording_summary(cfg)!r}")
    return "\n".join(lines)


def _recording_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    if "recording" not in cfg:
        return {
            "full_state": True,
            "note": "default recording omitted; use metrics for aggregate series, recording for diagnostic paths",
        }
    rec = cfg["recording"]
    full_state = rec.get("full_state", True)
    out: dict[str, Any] = {"full_state": full_state}
    if "num_paths_to_save" in rec:
        out["num_paths_to_save"] = rec.get("num_paths_to_save")
    if not full_state:
        out["downsample"] = rec.get("downsample")
        out["streaming_paths"] = list(rec.get("streaming_paths", []))
    return out
