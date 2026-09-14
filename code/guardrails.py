"""Safety guardrails for a financial recommendation system.

Principle: a false "go ahead" (recommending a payment that is not actually safe) is far more harmful than a
missed opportunity. Every guard therefore fails closed:

1. Evidence guards   - LLM-extracted message facts and image amounts are schema-checked and sanity-checked
                       against the user's own history before they may change the forecast. Facts that would
                       *increase* available money need stronger proof than facts that decrease it.
2. Forecast guards   - unresolved blank amounts are reserved at a conservative estimate; anything that still
                       cannot be quantified marks the forecast as uncertain, which blocks immediate payments.
3. Plan guards       - plans must finish by the deadline and inside the verifiable 90-day horizon,
                       installments must match a supplied option exactly, and spending changes must target
                       the user's own flexible, permitted, non-protected expenses.
4. Output verifier   - each finished row is parsed back from its CSV text and re-simulated day by day
                       independently of the planner. Any violation downgrades the row to not_recommended.
"""
import math
from datetime import date, timedelta

from engine import EPS, HORIZON_DAYS, D, is_blank, plan_is_safe

KNOWN_CURRENCIES = {"INR", "ZAR", "IDR", "USD", "EUR"}
INCOME_KINDS = {"salary_amount_change", "first_salary", "salary_resumes", "confirmed_invoice", "fx_settlement_note",
                "pending_unconfirmed_income"}
MAX_INCOME_JUMP = 3.0        # an income fact above this multiple of observed salary is not trusted
MAX_IMAGE_DEVIATION = 20.0   # an image amount this many times away from same-category history is not trusted
VALID_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
STATUS_FOR_METHOD = {
    "full_payment": {"affordable_now", "affordable_with_plan"},
    "partial_payment": {"affordable_with_plan"},
    "installments": {"affordable_with_plan"},
    "wait": {"affordable_later"},
    "not_recommended": {"not_affordable", "affordable_later"},
}


# ---------------------------------------------------------------- evidence guards
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _date(x):
    try:
        return D(x) if x else None
    except ValueError:
        return None


def check_fact(fact, home, request_date, observed_salary):
    """Return (ok, reason). Rejected facts are ignored by the forecast (and logged)."""
    from evidence import MESSAGE_KINDS

    kind = fact.get("kind")
    if kind not in MESSAGE_KINDS:
        return False, f"unknown kind {kind!r}"
    cur = fact.get("currency")
    if cur and cur not in KNOWN_CURRENCIES:
        return False, f"unknown currency {cur!r}"
    amount = fact.get("amount")
    if amount is not None and (_num(amount) is None or _num(amount) <= 0):
        return False, f"invalid amount {amount!r}"
    eff = fact.get("effective_date")
    if eff and _date(eff) is None:
        return False, f"invalid date {eff!r}"
    if kind in INCOME_KINDS:
        # money-increasing facts need explicit confirmation and a plausible size
        if kind != "pending_unconfirmed_income" and fact.get("is_confirmed_cash") is False:
            return False, "income not confirmed"
        if kind in ("first_salary", "confirmed_invoice", "salary_resumes") and (_date(eff) is None or amount is None):
            return False, "income fact without amount and date"
        if _date(eff) and _date(eff) > D(request_date) + timedelta(days=HORIZON_DAYS):
            return False, "income outside forecast horizon"
        if observed_salary and amount is not None and _num(amount) is not None and cur == home \
                and _num(amount) > MAX_INCOME_JUMP * observed_salary:
            return False, f"income {amount} implausible vs observed salary {observed_salary:.2f}"
    return True, ""


def check_image_amount(event_row, info, category_history):
    """Validate an image-extracted amount for a blank event. Returns (amount, currency, reason)."""
    amount = _num(info.get("amount"))
    cur = info.get("currency") or event_row.currency
    if amount is None or amount <= 0:
        return None, None, "image amount missing or non-positive"
    if cur not in KNOWN_CURRENCIES:
        return None, None, f"image currency {cur!r} unknown"
    if cur != event_row.currency:
        return None, None, f"image currency {cur} differs from event currency {event_row.currency}"
    hist = [a for a in category_history if a and a > 0]
    if hist:
        med = sorted(hist)[len(hist) // 2]
        if amount > MAX_IMAGE_DEVIATION * med or amount < med / MAX_IMAGE_DEVIATION:
            if event_row.direction == "credit":
                return None, None, f"image credit {amount} implausible vs history median {med:.2f}"
            # an implausibly large debit is kept (safer); an implausibly small one is replaced by history
            if amount < med / MAX_IMAGE_DEVIATION:
                return max(hist), cur, f"image debit {amount} too small; reserved history max"
    return amount, cur, ""


def conservative_blank_estimate(event_row, category_history):
    """Reserve for a debit whose amount could not be read: the largest same-category amount seen."""
    hist = [a for a in category_history if a and a > 0]
    if event_row.direction == "debit" and hist:
        return max(hist)
    return None


# ---------------------------------------------------------------- output verifier
def _parse_plan(text):
    if text == "none":
        return []
    out = []
    for part in text.split("|"):
        d, a = part.split(":")
        out.append((D(d), float(a)))
    return out


def verify_row(data, req, fc, row, drop_fn_factory):
    """Independently re-check a finished output row. Returns a list of violations (empty = safe)."""
    v = []
    requested = float(req["requested_amount"])
    start, deadline = D(req["request_date"]), D(req["desired_completion_date"])
    prof = data.profiles.loc[req["user_id"]]
    methods = set(str(prof.payment_methods_user_will_consider).split("|"))
    status, method = row["affordability_status"], row["recommended_payment_method"]

    if status not in VALID_STATUSES:
        v.append(f"invalid status {status}")
    if method not in VALID_METHODS:
        v.append(f"invalid method {method}")
    elif status not in STATUS_FOR_METHOD[method]:
        v.append(f"status {status} inconsistent with method {method}")

    safe = float(row["amount_safe_to_pay"])
    if not (0 <= safe <= requested + EPS):
        v.append("amount_safe_to_pay out of bounds")
    try:
        pays = _parse_plan(row["payment_plan"])
    except Exception:
        return v + ["unparseable payment_plan"]
    earliest = _date(row["earliest_date_for_full_payment"])

    # spending changes
    changes = []
    if row["spending_changes_needed"] != "none":
        items = row["spending_changes_needed"].split("|")
        if len(items) > 3:
            v.append("more than three spending changes")
        protected = set(str(prof.expense_categories_to_protect).split("|"))
        can_reduce = set(str(prof.expense_categories_user_is_willing_to_reduce).split("|"))
        can_stop = set(str(prof.expense_categories_user_is_willing_to_stop).split("|"))
        user_events = data.events_by_user[req["user_id"]].set_index("event_id")
        seen = set()
        for it in items:
            parts = it.split(":")
            eid = parts[1] if len(parts) > 1 else ""
            if eid in seen:
                v.append(f"event {eid} changed twice")
            seen.add(eid)
            if eid not in user_events.index:
                v.append(f"change targets unknown/foreign event {eid}")
                continue
            e = user_events.loc[eid]
            if e.category in protected:
                v.append(f"change targets protected category {e.category}")
            if parts[0] == "stop":
                if e.category not in can_stop or e.flexibility not in ("stoppable", "reducible_or_stoppable"):
                    v.append(f"stop not permitted for {eid}")
            elif parts[0] == "reduce_to" and len(parts) == 3:
                new_amt = float(parts[2])
                if e.category not in can_reduce or e.flexibility not in ("reducible", "reducible_or_stoppable"):
                    v.append(f"reduce not permitted for {eid}")
                if is_blank(e.minimum_allowed_amount) or new_amt + EPS < float(e.minimum_allowed_amount):
                    v.append(f"reduce_to below minimum_allowed_amount for {eid}")
            else:
                v.append(f"malformed change {it}")
        # rebuild the change list against forecast flows for re-simulation
        flows_by_id = {f.event_id: f for f in fc.flows if f.event_id}
        for it in items:
            parts = it.split(":")
            f = flows_by_id.get(parts[1]) if len(parts) > 1 else None
            if f is None:
                v.append(f"change {it} has no projected expense to change")
                continue
            changes.append(("stop" if parts[0] == "stop" else "reduce_to", f,
                            float(parts[2]) if parts[0] == "reduce_to" and len(parts) == 3 else 0.0))

    if method in ("full_payment", "partial_payment", "installments", "wait"):
        if not pays:
            v.append("payment method without payment plan")
        else:
            dates = [d for d, _ in pays]
            if dates != sorted(dates):
                v.append("payment plan not chronological")
            if dates[0] < start:
                v.append("payment before request date")
            if dates[-1] > fc.end:
                v.append("payment beyond verifiable forecast horizon")
            if any(a <= 0 for _, a in pays):
                v.append("non-positive payment")
            # every payment in the plan must keep the balance at or above the minimum, day by day
            ok, lowest = plan_is_safe(fc, pays, drop_fn_factory(changes) if changes else None)
            if not ok:
                v.append(f"plan breaches minimum balance (lowest {lowest:.2f} < {fc.minimum:.2f})")
            if method != "wait" and dates[-1] > deadline:
                v.append("plan completes after desired_completion_date")
    elif pays:
        v.append("not_recommended row carries payments")

    if method != "not_recommended" and method not in ("wait",) and method not in methods:
        v.append(f"method {method} not accepted by user")
    if method == "wait" and "full_payment" not in methods:
        v.append("wait requires the user to accept full_payment")

    if method == "full_payment":
        if len(pays) != 1 or abs(pays[0][1] - requested) > 0.011 or pays[0][0] != start:
            v.append("full payment must be the requested amount today")
        if status == "affordable_now" and (changes or safe + 0.011 < requested):
            v.append("affordable_now requires full safety without spending changes")
    if method == "wait":
        if len(pays) != 1 or abs(pays[0][1] - requested) > 0.011 or earliest is None or pays[0][0] != earliest:
            v.append("wait must be one full payment on earliest_date_for_full_payment")
    if method == "partial_payment":
        if not str(req["allows_partial_payment"]).strip().lower() == "true":
            v.append("request does not allow partial payment")
        if len(pays) != 2 or pays[0][0] != start or abs(pays[0][1] - safe) > 0.011 \
                or earliest is None or pays[1][0] != earliest or earliest > deadline \
                or abs(sum(a for _, a in pays) - requested) > 0.011 or not (0 < safe < requested):
            v.append("partial payment structure invalid")
    if method == "installments":
        opts = data.options_by_request.get(req["request_id"])
        match = False
        if opts is not None:
            for o in opts[opts.payment_method == "installments"].itertuples():
                n, freq = int(o.number_of_payments), int(o.payment_frequency_days)
                sched = [(D(o.first_payment_date) + timedelta(days=freq * k), round(float(o.payment_amount), 2)) for k in range(n)]
                if len(sched) == len(pays) and all(d1 == d2 and abs(a1 - a2) < 0.011 for (d1, a1), (d2, a2) in zip(sched, pays)):
                    match = True
                    if is_blank(prof.max_installment_months) or n > int(prof.max_installment_months):
                        v.append("installment option exceeds user's max_installment_months")
        if not match:
            v.append("installment plan does not match a supplied option")
    if status == "affordable_now" and row["earliest_date_for_full_payment"] != str(start):
        v.append("affordable_now requires earliest date = request date")
    if earliest is not None and (earliest < start or earliest > fc.end):
        v.append("earliest date outside forecast horizon")
    return v


def downgrade(row, req, profile, reasons, safe):
    """Fail closed: replace an unverifiable recommendation with a conservative one."""
    cur = profile.home_currency
    minimum = profile.minimum_balance_to_keep
    fmt = lambda x: f"{cur} {x:,.2f}".replace(".00", "")
    row = dict(row)
    row.update({
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "spending_changes_needed": "none",
        "decision_explanation": (f"Do not proceed with the {fmt(float(req['requested_amount']))} request yet. "
                                 f"A safe plan could not be fully verified against the {fmt(minimum)} minimum balance, "
                                 f"so no payment is recommended."),
    })
    return row
