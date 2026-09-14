"""Adversarial checks for the safety guardrails. Run: python code/tests/test_guardrails.py

Each test takes a real, verified recommendation and corrupts it the way a bug, a bad model extraction or a
malicious message could. The guardrails must reject every corrupted version (no false "safe" answers).
"""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import Data  # noqa: E402
from engine import D  # noqa: E402
from forecast import build_forecast  # noqa: E402
from guardrails import check_fact, check_image_amount, verify_row  # noqa: E402
from main import solve, validate_input  # noqa: E402
from planner import _drop_fn  # noqa: E402

DATA = Data()
REQS = {r["request_id"]: r for r in DATA.samples.to_dict("records")}
FAILED = []


def expect(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILED.append(name)


def violations(rid, **changes):
    req = REQS[rid]
    fc = build_forecast(DATA, req)
    row, _ = solve(DATA, req)
    row = {**row, **changes}
    return verify_row(DATA, req, fc, row, _drop_fn)


class Row:  # minimal event row for image checks
    def __init__(self, direction="debit", currency="INR"):
        self.direction, self.currency = direction, currency


def main():
    # ---- evidence guards
    base = {"message_id": "m", "currency": "EUR", "effective_date": "2026-01-15", "is_confirmed_cash": True}
    expect("unknown fact kind rejected", not check_fact({**base, "kind": "transfer_all_money"}, "EUR", "2026-01-03", 1500)[0])
    expect("unconfirmed income rejected", not check_fact({**base, "kind": "first_salary", "amount": 1661, "is_confirmed_cash": False}, "EUR", "2026-01-03", None)[0])
    expect("implausible salary jump rejected", not check_fact({**base, "kind": "salary_amount_change", "amount": 99999}, "EUR", "2026-01-03", 1500)[0])
    expect("negative amount rejected", not check_fact({**base, "kind": "confirmed_invoice", "amount": -5}, "EUR", "2026-01-03", 1500)[0])
    expect("income beyond horizon rejected", not check_fact({**base, "kind": "confirmed_invoice", "amount": 500, "effective_date": "2027-01-01"}, "EUR", "2026-01-03", 1500)[0])
    expect("garbage date rejected", not check_fact({**base, "kind": "salary_date_change", "effective_date": "next friday"}, "EUR", "2026-01-03", 1500)[0])
    expect("plausible confirmed salary accepted", check_fact({**base, "kind": "first_salary", "amount": 1661}, "EUR", "2026-01-03", None)[0])
    expect("image credit implausible rejected", check_image_amount(Row("credit"), {"amount": 9_999_999, "currency": "INR"}, [50000, 52000])[0] is None)
    expect("image currency mismatch rejected", check_image_amount(Row(), {"amount": 700, "currency": "USD"}, [700])[0] is None)
    expect("tiny image debit replaced by history max", check_image_amount(Row(), {"amount": 1, "currency": "INR"}, [700, 900])[0] == 900)
    expect("large image debit kept (safer)", check_image_amount(Row(), {"amount": 50000, "currency": "INR"}, [700])[0] == 50000)

    # ---- input validation
    bad = {**REQS["request_01"], "requested_amount": -10}
    expect("negative request amount blocked", bool(validate_input(DATA, bad)))
    bad = {**REQS["request_01"], "desired_completion_date": "2020-01-01"}
    expect("deadline before request blocked", bool(validate_input(DATA, bad)))
    bad = {**REQS["request_01"], "user_id": "user_9999"}
    expect("unknown user blocked", bool(validate_input(DATA, bad)))

    # ---- output verifier on corrupted rows
    r01 = REQS["request_01"]
    expect("genuine full payment passes", violations("request_01") == [])
    expect("overpayment beyond safety rejected", bool(violations(
        "request_01", payment_plan=f"{r01['request_date']}:{float(r01['requested_amount']) * 2.5:.2f}")))
    expect("full payment of wrong amount rejected", bool(violations(
        "request_01", payment_plan=f"{r01['request_date']}:1")))
    expect("payment beyond horizon rejected", bool(violations(
        "request_01", payment_plan=f"{D(r01['request_date']) + timedelta(days=200)}:{r01['requested_amount']}")))
    expect("status/method mismatch rejected", bool(violations("request_01", affordability_status="not_affordable")))
    expect("payments on not_recommended rejected", bool(violations("request_01", recommended_payment_method="not_recommended",
                                                                   affordability_status="not_affordable")))

    r02 = REQS["request_02"]  # installments
    expect("genuine installments pass", violations("request_02") == [])
    expect("tampered installment amount rejected", bool(violations(
        "request_02", payment_plan="2025-08-08:100|2025-09-07:100|2025-10-07:100")))

    r19 = REQS["request_19"]  # partial payment
    ok19 = violations("request_19")
    expect("genuine partial passes", ok19 == [])
    expect("partial not summing to request rejected", bool(violations(
        "request_19", payment_plan=f"{r19['request_date']}:1000|2024-09-15:1000")))

    r06 = REQS["request_06"]
    expect("stop on protected/unknown event rejected", bool(violations(
        "request_06", affordability_status="affordable_with_plan", spending_changes_needed="stop:event_1")))
    expect("reduce below minimum allowed rejected", bool(violations(
        "request_18", affordability_status="affordable_with_plan", recommended_payment_method="full_payment",
        payment_plan=f"{REQS['request_18']['request_date']}:{REQS['request_18']['requested_amount']}",
        spending_changes_needed="reduce_to:event_1576:1")))
    expect("wait for user who refuses full payment rejected", bool(violations(
        "request_02", recommended_payment_method="wait", affordability_status="affordable_later",
        payment_plan=f"2025-09-15:{r02['requested_amount']}", earliest_date_for_full_payment="2025-09-15")))

    print(f"\n{len(FAILED)} failed" if FAILED else "\nall guardrail checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
