"""Build evaluation/usage_report.md from the LLM call log written by code/evidence.py.

Usage: python code/evaluation/main.py
Token counts come from code/cache/usage.json (one entry per model call, as reported by the provider).
Prices come from LiteLLM's bundled model price table; override with env vars
LLM_INPUT_PRICE_PER_M / LLM_OUTPUT_PRICE_PER_M (USD per million tokens) if needed.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
USAGE = CODE / "cache" / "usage.json"
REPORT = HERE / "usage_report.md"


def prices(model):
    env_in, env_out = os.getenv("LLM_INPUT_PRICE_PER_M"), os.getenv("LLM_OUTPUT_PRICE_PER_M")
    if env_in and env_out:
        return float(env_in) / 1e6, float(env_out) / 1e6, "environment override"
    try:
        import litellm

        info = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1]) or {}
        pin, pout = info.get("input_cost_per_token"), info.get("output_cost_per_token")
        if pin is not None and pout is not None:
            return float(pin), float(pout), f"LiteLLM price table ({litellm.__name__})"
    except Exception:
        pass
    return None, None, "price unavailable"


def main():
    calls = json.loads(USAGE.read_text(encoding="utf-8")) if USAGE.exists() else []
    n_requests = len(pd.read_csv(ROOT / "dataset" / "requests.csv"))
    per_model = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0})
    per_stage = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0})
    for c in calls:
        m = per_model[c["model"]]
        stage = "images" if c["label"].startswith("image") else "messages"
        for bucket in (m, per_stage[stage]):
            bucket["calls"] += 1
            bucket["input"] += int(c["input_tokens"] or 0)
            bucket["output"] += int(c["output_tokens"] or 0)

    lines = [
        "# Token Usage and Cost Report",
        "",
        "Final full-dataset run that produced `output.csv`.",
        "",
        "## How the model is used",
        "",
        "- The decision engine (forecast, plan generation, ranking, validation, explanations) is deterministic Python and makes **no model calls** per request.",
        "- A one-time evidence-extraction pass (`code/evidence.py`) uses the LLM to structure the 215 messages (batched, 30 per call) and to read the linked images. Results are cached in `code/cache/evidence.json`; re-running `python code/main.py` reuses the cache and costs 0 tokens.",
        f"- Requests evaluated: {n_requests}.",
        "",
        "## Per model",
        "",
        "| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Est. cost (USD) | Price source |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    tot = {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
    cost_known = True
    for model, m in sorted(per_model.items()):
        pin, pout, src = prices(model)
        cost = (m["input"] * pin + m["output"] * pout) if pin is not None else None
        cost_known &= cost is not None
        provider = model.split("/", 1)[0] if "/" in model else "unknown"
        lines.append(f"| {provider} | {model} | {m['calls']} | {m['input']:,} | {m['output']:,} | "
                     f"{m['input'] + m['output']:,} | {'' if cost is None else f'{cost:.4f}'} | {src} |")
        tot["calls"] += m["calls"]
        tot["input"] += m["input"]
        tot["output"] += m["output"]
        tot["cost"] += cost or 0.0

    total_tokens = tot["input"] + tot["output"]
    lines += [
        "",
        "## Per stage",
        "",
        "| Stage | Calls | Input tokens | Output tokens |",
        "|---|---:|---:|---:|",
    ]
    for stage, s in sorted(per_stage.items()):
        lines.append(f"| {stage} | {s['calls']} | {s['input']:,} | {s['output']:,} |")
    lines += [
        "",
        "## Overall",
        "",
        f"- Model calls: {tot['calls']}",
        f"- Input tokens: {tot['input']:,}",
        f"- Output tokens (includes reasoning tokens billed as output): {tot['output']:,}",
        f"- Total tokens: {total_tokens:,}",
        f"- Average tokens per request: {total_tokens / n_requests:,.1f}",
        f"- Estimated total cost: {'USD %.4f' % tot['cost'] if cost_known else 'unavailable (set LLM_INPUT_PRICE_PER_M / LLM_OUTPUT_PRICE_PER_M)'}",
        f"- Estimated cost per request: {'USD %.6f' % (tot['cost'] / n_requests) if cost_known else 'unavailable'}",
        "",
        "Notes: token counts are the provider-reported `usage` values captured for every call, including calls for",
        "sample-request evidence. Failed attempts (e.g. HTTP 503/429) returned no usage and are not billed.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
