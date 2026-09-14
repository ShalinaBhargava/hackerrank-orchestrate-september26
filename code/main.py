"""Buy or Wait? — entry point.

Usage:
    python code/main.py                 # evaluate dataset/requests.csv -> ./output.csv
    python code/main.py --samples       # evaluate dataset/sample_requests.csv -> code/cache/sample_output.csv
    python code/main.py --extract       # (re)run LLM evidence extraction first (needs GEMINI_API_KEY)
"""
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import ROOT, Data  # noqa: E402
from engine import D  # noqa: E402
from explain import explain, fmt_num  # noqa: E402
from forecast import build_forecast  # noqa: E402
from guardrails import downgrade, verify_row  # noqa: E402
from planner import _drop_fn, decide  # noqa: E402

COLUMNS = ["request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
           "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"]
GUARD_REPORT = HERE / "cache" / "guardrail_report.csv"


def validate_input(data, req):
    """Reject malformed requests before any forecasting (fail closed)."""
    problems = []
    if req["user_id"] not in data.profiles.index or req["user_id"] not in data.events_by_user:
        problems.append("unknown user or no financial history")
    try:
        amount = float(req["requested_amount"])
        if not math.isfinite(amount) or amount <= 0:
            problems.append("requested_amount must be a positive number")
    except (TypeError, ValueError):
        problems.append("requested_amount is not numeric")
    try:
        if D(req["desired_completion_date"]) < D(req["request_date"]):
            problems.append("desired_completion_date is before request_date")
    except (TypeError, ValueError):
        problems.append("invalid request or completion date")
    if not problems:
        p = data.profiles.loc[req["user_id"]]
        for col in ("current_available_balance", "minimum_balance_to_keep"):
            try:
                v = float(p[col])
            except (TypeError, ValueError):
                v = float("nan")
            if not math.isfinite(v) or v < 0:
                problems.append(f"profile {col} invalid")
    return problems


def solve(data, req):
    fc = build_forecast(data, req)
    dec = decide(data, req, fc)
    plan = dec["plan"]
    requested = dec["requested"]
    earliest = dec["earliest"]
    start = fc.start

    if plan is None:
        status, method, plan_txt, changes = "not_affordable", "not_recommended", "none", "none"
        if earliest is not None and earliest > D(req["desired_completion_date"]):
            status = "affordable_later"  # capacity exists later, but not within the request's timeline
    else:
        method = plan.method
        plan_txt = "|".join(f"{d}:{fmt_num(a)}" for d, a in plan.payments)
        if plan.changes:
            status = "affordable_with_plan"
            changes = "|".join(
                f"stop:{f.event_id}" if k == "stop" else f"reduce_to:{f.event_id}:{fmt_num(v)}"
                for k, f, v in plan.changes)
        else:
            changes = "none"
            status = {"full_payment": "affordable_now", "wait": "affordable_later",
                      "partial_payment": "affordable_with_plan", "installments": "affordable_with_plan"}[method]
    if status == "affordable_now":
        earliest = start

    row = {
        "request_id": req["request_id"],
        "amount_safe_to_pay": fmt_num(min(max(dec["safe"], 0), requested)),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": plan_txt,
        "earliest_date_for_full_payment": str(earliest) if earliest else "",
        "spending_changes_needed": changes,
        "decision_explanation": explain(req, data.profiles.loc[req["user_id"]], dec, fc),
    }
    guard = {"request_id": req["request_id"], "violations": "", "blocked": "|".join(dec.get("blocked", [])),
             "notes": " ; ".join(n for n in fc.notes if n.startswith("guard")), "downgraded": False}
    violations = verify_row(data, req, fc, row, _drop_fn)
    if violations:
        row = downgrade(row, req, data.profiles.loc[req["user_id"]], violations, dec["safe"])
        guard.update(violations="|".join(violations), downgraded=True)
        # the downgraded row must itself pass verification
        assert not verify_row(data, req, fc, row, _drop_fn), row
    return row, guard


def fallback_row(req, reason):
    return {"request_id": req["request_id"], "amount_safe_to_pay": "0",
            "affordability_status": "not_affordable", "recommended_payment_method": "not_recommended",
            "payment_plan": "none", "earliest_date_for_full_payment": "",
            "spending_changes_needed": "none",
            "decision_explanation": f"Do not proceed. The request could not be verified as safe ({reason})."}


def run(samples=False, out_path=None):
    data = Data()
    df = data.samples if samples else data.requests
    out_path = out_path or (HERE / "cache" / "sample_output.csv" if samples else ROOT / "output.csv")
    rows, guards = [], []
    for req in df.to_dict("records"):
        problems = validate_input(data, req)
        if problems:
            rows.append(fallback_row(req, "; ".join(problems)))
            guards.append({"request_id": req["request_id"], "violations": "|".join(problems), "blocked": "",
                           "notes": "input validation failed", "downgraded": True})
            continue
        try:
            row, guard = solve(data, req)
        except Exception as exc:  # never drop a request and never guess: fall back to not_recommended
            print(f"{req['request_id']}: fallback ({type(exc).__name__}: {exc})", file=sys.stderr)
            row = fallback_row(req, "internal check failed")
            guard = {"request_id": req["request_id"], "violations": f"exception: {type(exc).__name__}: {exc}",
                     "blocked": "", "notes": "", "downgraded": True}
        rows.append(row)
        guards.append(guard)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if not samples:
        with open(GUARD_REPORT, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["request_id", "downgraded", "violations", "blocked", "notes"])
            w.writeheader()
            w.writerows(guards)
    n_down = sum(bool(g["downgraded"]) for g in guards)
    n_notes = sum(bool(g["notes"] or g["blocked"]) for g in guards)
    print(f"wrote {len(rows)} rows -> {out_path} | guardrails: {n_down} downgraded, {n_notes} with evidence guards")
    return rows


if __name__ == "__main__":
    if "--extract" in sys.argv:
        from evidence import extract
        extract()
    run(samples="--samples" in sys.argv)
