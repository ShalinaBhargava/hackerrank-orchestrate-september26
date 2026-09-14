"""Generate candidate payment plans, verify them against the forecast, and pick the best one."""
import itertools
import math
from dataclasses import dataclass, field
from datetime import timedelta

from engine import EPS, D, is_blank, plan_is_safe, safe_amount_and_earliest


@dataclass
class Plan:
    method: str
    payments: list                      # [(date, amount)]
    total: float
    option_id: str = ""
    changes: list = field(default_factory=list)   # [("stop"|"reduce_to", Flow, new_amount)]

    @property
    def start(self):
        return self.payments[0][0]

    def by_deadline(self, deadline):
        return self.payments[-1][0] <= deadline


def _split(v):
    return set() if is_blank(v) else {x.strip() for x in str(v).split("|") if x.strip()}


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _option_num(opt_id):
    try:
        return int(str(opt_id).rsplit("_", 1)[-1])
    except ValueError:
        return 10 ** 9


def _drop_fn(changes):
    if not changes:
        return None
    by_id = {}
    for kind, flow, new_amt in changes:
        by_id[flow.event_id] = (kind, new_amt)

    def drop(f):
        c = by_id.get(f.event_id)
        if not c or f.amount >= 0:
            return f.amount
        kind, new_amt = c
        if f.kind == "recurring_variable":
            return f.amount + (f.save_stop if kind == "stop" else f.save_reduce)
        return 0.0 if kind == "stop" else -min(abs(f.amount), new_amt)
    return drop


def candidate_plans(data, req, fc, earliest, safe, changes=None):
    prof = data.profiles.loc[req["user_id"]]
    methods = _split(prof.payment_methods_user_will_consider)
    requested = float(req["requested_amount"])
    start = fc.start
    deadline = D(req["desired_completion_date"])
    drop = _drop_fn(changes)
    plans = []

    if "full_payment" in methods:
        p = Plan("full_payment", [(start, requested)], requested, changes=changes or [])
        if plan_is_safe(fc, p.payments, drop)[0]:
            plans.append(p)
        elif not changes and earliest is not None and earliest > start:
            plans.append(Plan("wait", [(earliest, requested)], requested))

    if not changes and _truthy(req["allows_partial_payment"]) and "partial_payment" in methods \
            and 0 < safe < requested and earliest is not None and earliest <= deadline:
        p = Plan("partial_payment", [(start, round(safe, 2)), (earliest, round(requested - safe, 2))], requested)
        if plan_is_safe(fc, p.payments, drop)[0]:
            plans.append(p)

    max_months = prof.max_installment_months
    if "installments" in methods and not is_blank(max_months):
        opts = data.options_by_request.get(req["request_id"])
        if opts is not None:
            for o in opts[opts.payment_method == "installments"].itertuples():
                n = int(o.number_of_payments)
                if n > int(max_months):
                    continue
                freq = int(o.payment_frequency_days) if not is_blank(o.payment_frequency_days) else 30
                first = D(o.first_payment_date)
                pays = [(first + timedelta(days=freq * k), float(o.payment_amount)) for k in range(n)]
                # payments after the deadline or beyond the 90-day horizon cannot be verified as safe
                if pays[-1][0] > deadline or pays[-1][0] > fc.end or pays[0][0] < start:
                    continue
                p = Plan("installments", pays, float(o.total_payable_amount), option_id=o.payment_option_id,
                         changes=changes or [])
                if plan_is_safe(fc, pays, drop)[0]:
                    plans.append(p)
    return plans


def rank_key(plan, deadline):
    return (
        0 if plan.by_deadline(deadline) else 1,
        0 if not plan.changes else 1,
        round(plan.total, 2),
        plan.start,
        len(plan.payments),
        _option_num(plan.option_id),
    )


def change_actions(data, req, fc):
    """Allowed spending-change actions on flexible, non-protected recurring expenses."""
    prof = data.profiles.loc[req["user_id"]]
    protected = _split(prof.expense_categories_to_protect)
    can_reduce = _split(prof.expense_categories_user_is_willing_to_reduce)
    can_stop = _split(prof.expense_categories_user_is_willing_to_stop)
    series = {}
    for f in fc.flows:
        if f.amount < 0 and f.kind in ("recurring_fixed", "recurring_variable") and f.event_id \
                and f.category not in protected and f.flexibility != "fixed":
            series.setdefault(f.event_id, f)
    actions = []
    for eid, f in sorted(series.items()):
        variable = f.kind == "recurring_variable"
        if f.category in can_stop and f.flexibility in ("stoppable", "reducible_or_stoppable") \
                and (not variable or f.save_stop > 0):
            actions.append(("stop", f, 0.0))
        if f.category in can_reduce and f.flexibility in ("reducible", "reducible_or_stoppable") \
                and not math.isnan(f.min_allowed) \
                and (f.save_reduce > 0 if variable else f.min_allowed < abs(f.amount) - EPS):
            actions.append(("reduce_to", f, f.min_allowed))
    return actions


def decide(data, req, fc):
    requested = float(req["requested_amount"])
    deadline = D(req["desired_completion_date"])
    safe, earliest, _ = safe_amount_and_earliest(fc, requested)
    if fc.flags:
        # an unquantifiable obligation makes every projection unreliable: fail closed
        return {"safe": 0.0, "earliest": None, "plan": None, "deadline": deadline, "requested": requested,
                "blocked": list(fc.flags)}
    plans = candidate_plans(data, req, fc, earliest, safe)
    in_time = [p for p in plans if p.by_deadline(deadline)]

    if not in_time:
        actions = change_actions(data, req, fc)
        found = []
        for size in (1, 2, 3):
            for combo in itertools.combinations(actions, size):
                if len({a[1].event_id for a in combo}) < size:
                    continue  # stop and reduce on the same event are exclusive
                cand = [p for p in candidate_plans(data, req, fc, earliest, safe, changes=list(combo))
                        if p.by_deadline(deadline)]
                for p in cand:
                    saved = sum(abs(a[1].amount) - a[2] for a in combo)
                    found.append((rank_key(p, deadline), saved, p))
            if found:
                break
        if found:
            found.sort(key=lambda x: (x[0], x[1]))
            plans.append(found[0][2])

    best = min(plans, key=lambda p: rank_key(p, deadline)) if plans else None
    return {"safe": safe, "earliest": earliest, "plan": best, "deadline": deadline, "requested": requested}
