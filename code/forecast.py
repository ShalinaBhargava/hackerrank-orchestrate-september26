"""Reconstruct a user's cash position and project it over the 90-day horizon."""
from datetime import timedelta

import numpy as np
import pandas as pd

from engine import CFG, DISCRETIONARY, HORIZON_DAYS, D, Flow, Forecast, add_month, is_blank
from guardrails import check_fact, check_image_amount, conservative_blank_estimate

SALARY_CATS = {"salary"}
IRREGULAR_INCOME = r"commission|bonus|arrears|incentive|overtime|one-time|one-off|reimburse|adjustment|allowance"


def _prepare_events(data, req, home, fc):
    ev = data.events_by_user[req["user_id"]].copy()
    amts = []
    for r in ev.itertuples():
        amount, cur = r.amount, r.currency
        if is_blank(amount):
            # never treat a blank amount as zero: use a validated image amount, else a conservative estimate
            hist = ev[(ev.category == r.category) & (ev.direction == r.direction) & (ev.currency == r.currency)
                      & ev.amount.notna()].amount.tolist()
            amount = None
            info = data.image_info.get(r.event_id)
            if info is not None:
                amount, img_cur, reason = check_image_amount(r, info, hist)
                cur = img_cur or r.currency
                if reason:
                    fc.notes.append(f"guard {r.event_id}: {reason}")
            if amount is None and r.direction == "debit" and r.status in ("pending", "scheduled"):
                amount = conservative_blank_estimate(r, hist)
                cur = r.currency
                if amount is None:
                    fc.flags.append(f"open debit {r.event_id} has no verifiable amount")
                else:
                    fc.notes.append(f"guard {r.event_id}: amount unreadable, reserved {amount:.2f}")
            elif amount is None and r.status in ("pending", "scheduled"):
                fc.notes.append(f"guard {r.event_id}: unreadable credit not counted")
        day = r.settlement_date if not is_blank(r.settlement_date) else r.event_date
        amts.append(data.fx.convert(float(amount), cur, home, day) if not is_blank(amount) else float("nan"))
    ev["amt"] = amts
    # repeated identical credits would overstate income, so only credits are de-duplicated; look-alike debits
    # (e.g. minimum payments on two separate cards) are all kept
    credits = ev[ev.direction == "credit"].drop_duplicates(
        subset=["description", "category", "direction", "amt", "event_date", "status"])
    ev = pd.concat([ev[ev.direction != "credit"], credits]).sort_index()
    return ev


def _min_allowed(row):
    return float(row.minimum_allowed_amount) if not is_blank(row.minimum_allowed_amount) else float("nan")


def _project_salary(fc, ev, hist, home, data):
    """Salary: a scheduled confirmed salary repeats monthly on its day; otherwise the recent payroll pattern."""
    start, end = fc.start, fc.end
    sched = ev[(ev.status == "scheduled") & (ev.direction == "credit") & (ev.event_type == "income")
               & ev.amt.notna()].sort_values("settlement_date")
    if len(sched):
        x = sched.iloc[0]
        d0 = D(x.settlement_date)
        k = 0
        while True:
            d = add_month(d0, k, d0.day)
            k += 1
            if d > end:
                break
            if d >= start:
                fc.flows.append(Flow(d, float(x.amt), "salary", key="salary", event_id=x.event_id,
                                     category="salary", description=x.description))
        return
    sal = hist[(hist.direction == "credit") & (hist.category.isin(SALARY_CATS))].sort_values("event_date")
    # commissions, bonuses and one-off adjustments are not regular confirmed salary
    sal = sal[~sal.description.str.contains(IRREGULAR_INCOME, case=False, regex=True)]
    recent = sal[sal.event_date >= str(start - timedelta(days=CFG["salary_lookback_days"]))]
    if not len(recent):
        return
    if recent.description.str.contains("final", case=False).any():
        return  # a final payroll ends the income series
    days = pd.Series([D(x).day for x in recent.event_date], index=recent.index)
    dom = int(days.mode().iloc[0])
    main = recent[days == dom]
    if len(main) < 2:
        return
    amt = float(main.amt.iloc[-2:].max())
    last = D(main.event_date.iloc[-1])
    k = 1
    while True:
        d = add_month(last, k, dom)
        k += 1
        if d > end:
            break
        if d >= start:
            fc.flows.append(Flow(d, amt, "salary", key="salary", event_id=main.iloc[-1].event_id,
                                 category="salary", description=main.iloc[-1].description))


def build_forecast(data, req):
    uid = req["user_id"]
    prof = data.profiles.loc[uid]
    home = prof.home_currency
    start = D(req["request_date"])
    end = start + timedelta(days=HORIZON_DAYS)
    fc = Forecast(start=start, end=end, balance=float(prof.current_available_balance),
                  minimum=float(prof.minimum_balance_to_keep))
    ev = _prepare_events(data, req, home, fc)
    hist = ev[(ev.status == "settled") & (ev.event_date < str(start)) & ev.direction.isin(["debit", "credit"])]

    # 1. income
    _project_salary(fc, ev, hist, home, data)

    # 2. monthly debits by description (rent, bills, subscriptions, loan payments)
    deb = hist[hist.direction == "debit"]
    used = set()
    for key, g in deb.groupby(["description", "category"]):
        g = g.sort_values("event_date")
        g = g[g.amt.notna()]
        if len(g) < CFG["min_occurrences"]:
            continue
        dates = [D(x) for x in g.event_date]
        days = pd.Series([d.day for d in dates])
        dom = int(days.mode().iloc[0])
        gaps = np.diff([d.toordinal() for d in dates])
        if (days == dom).mean() < CFG["monthly_day_share"] or np.median(gaps) < 25:
            continue
        used.add(key)
        if (start - dates[-1]).days > CFG["stale_days_monthly"]:
            continue  # stopped series
        last = g.iloc[-1]
        amt = float(last.amt)
        k = 1
        while True:
            d = add_month(dates[-1], k, dom)
            k += 1
            if d > end:
                break
            if d >= start:
                fc.flows.append(Flow(d, -amt, "recurring_fixed", key="|".join(key), event_id=last.event_id,
                                     category=key[1], flexibility=last.flexibility, min_allowed=_min_allowed(last),
                                     description=key[0]))

    # 3. variable spending: trailing-window daily average per category, spread across the horizon
    rest = deb[[k not in used for k in zip(deb.description, deb.category)]]
    rest = rest[rest.event_type == "expense"]
    window = CFG["variable_window_days"]
    protected = set(str(prof.expense_categories_to_protect).split("|"))
    for cat, g in rest.groupby("category"):
        weight = 1.0
        if CFG["var_cats"] == "essential" and cat in DISCRETIONARY:
            weight = CFG.get("discretionary_weight", 0.0)
            if weight <= 0:
                continue
        if CFG["var_cats"] == "protected" and cat not in protected:
            continue
        g = g.sort_values("event_date")
        g = g[g.amt.notna()]
        if CFG["outlier_mult"] and len(g) >= 3:
            g = g[g.amt <= CFG["outlier_mult"] * g.amt.median()]  # one-off purchases are not recurring spend
        if len(g) < CFG["min_occurrences"] or (start - D(g.event_date.iloc[-1])).days > CFG["stale_days_variable"]:
            continue
        if CFG["var_mode"] == "cadence":
            dates = [D(x) for x in g.event_date]
            cad = max(1, int(np.median(np.diff([d.toordinal() for d in dates]))))
            amt = float(np.mean(g.amt.values[-6:]))
            d = dates[-1] + timedelta(days=cad)
            flex_rows = g[g.flexibility != "fixed"]
            ref = flex_rows.iloc[-1] if len(flex_rows) else g.iloc[-1]
            while d <= end:
                if d >= start:
                    fc.flows.append(Flow(d, -amt, "recurring_variable", key=cat, event_id=ref.event_id, category=cat,
                                         flexibility=ref.flexibility, min_allowed=_min_allowed(ref),
                                         description=ref.description,
                                         save_stop=amt if ref.flexibility != "fixed" and ref.description in set(g.description.values[-6:]) else 0.0,
                                         save_reduce=max(0.0, amt - _min_allowed(ref)) if not np.isnan(_min_allowed(ref)) else 0.0))
                d += timedelta(days=cad)
            continue
        daily = weight * g[g.event_date >= str(start - timedelta(days=window))].amt.sum() / window
        if daily <= 0:
            continue
        flex_rows = g[g.flexibility != "fixed"]
        ref = flex_rows.iloc[-1] if len(flex_rows) else g.iloc[-1]
        # daily savings if the flexible expense (ref description) is reduced to its minimum or stopped
        in_win = g[(g.event_date >= str(start - timedelta(days=window))) & (g.description == ref.description)]
        save_stop = in_win.amt.sum() / window if ref.flexibility != "fixed" else 0.0
        mn_allowed = _min_allowed(ref)
        save_reduce = (np.clip(in_win.amt - mn_allowed, 0, None).sum() / window) if not np.isnan(mn_allowed) else 0.0
        t = start
        while t <= end:
            fc.flows.append(Flow(t, -daily, "recurring_variable", key=cat, event_id=ref.event_id, category=cat,
                                 flexibility=ref.flexibility, min_allowed=mn_allowed, description=ref.description,
                                 save_stop=min(save_stop, daily), save_reduce=min(save_reduce, daily)))
            t += timedelta(days=1)

    # 4. pending / scheduled debits are reserved; pending credits are ignored until they settle
    open_rows = ev[ev.status.isin(["pending", "scheduled"]) & (ev.direction == "debit")]
    for r in open_rows.itertuples():
        if is_blank(r.amt):
            fc.notes.append(f"missing amount for {r.event_id}")
            continue
        day = D(r.settlement_date) if not is_blank(r.settlement_date) else D(r.event_date)
        day = max(day, start)
        if day > end:
            continue
        fc.flows.append(Flow(day, -float(r.amt), "pending", key=r.description, event_id=r.event_id,
                             category=r.category, description=r.description))

    _apply_messages(data, req, fc, ev, home)
    fc.flows.sort(key=lambda f: f.day)
    return fc


def _salary_flows(fc):
    return sorted([f for f in fc.flows if f.amount > 0 and f.category in SALARY_CATS], key=lambda f: f.day)


def _monthly_salary(fc, first_day, amount, label):
    k = 0
    while True:
        d = add_month(first_day, k, first_day.day)
        k += 1
        if d > fc.end:
            break
        if d >= fc.start:
            fc.flows.append(Flow(d, amount, "salary", key=label, category="salary", description=label))


def _apply_messages(data, req, fc, ev, home):
    observed_salary = max((f.amount for f in fc.flows if f.amount > 0 and f.category in SALARY_CATS), default=None)
    for fact in data.message_facts(req["user_id"], req["request_date"]):
        ok, reason = check_fact(fact, home, req["request_date"], observed_salary)
        if not ok:
            fc.notes.append(f"guard ignored {fact.get('message_id')}: {reason}")
            continue
        kind = fact.get("kind")
        cur = fact.get("currency") or home
        eff = None
        if fact.get("effective_date"):
            try:
                eff = D(fact["effective_date"])
            except ValueError:
                eff = None
        amount = fact.get("amount")
        if amount is not None:
            try:
                amount = data.fx.convert(float(amount), cur, home, eff or fc.start)
            except (TypeError, ValueError):
                amount = None
        sal = _salary_flows(fc)
        summary = str(fact.get("summary", "")).lower()

        if kind == "salary_amount_change" and amount:
            for f in sal:
                if eff is None or f.day >= eff:
                    f.amount = amount
            fc.notes.append(f"salary changes to {amount:.2f}")
        elif kind == "pending_unconfirmed_income" and amount and "salary" in summary:
            # only the confirmed base salary counts; unapproved commission/bonus is excluded
            for f in sal:
                f.amount = amount
            fc.notes.append(f"confirmed base salary {amount:.2f}")
        elif kind in ("salary_temporary_amount", "salary_with_one_time") and amount:
            nxt = [f for f in sal if f.day >= fc.start]
            if nxt:
                nxt[0].amount = amount
            fc.notes.append(f"next salary {amount:.2f}")
        elif kind == "salary_date_change" and eff:
            nxt = [f for f in sal if f.day >= fc.start]
            if nxt and fc.start <= eff:
                nxt[0].day = eff
            fc.notes.append(f"salary moved to {eff}")
        elif kind == "first_salary" and amount and eff:
            # only the confirmed first payment is counted; later pay is not yet evidenced
            if fc.start <= eff <= fc.end and not any(abs((f.day - eff).days) <= 5 for f in sal):
                fc.flows.append(Flow(eff, amount, "salary", key="first salary", category="salary",
                                     description="first salary"))
            fc.notes.append(f"first salary {amount:.2f} on {eff}")
        elif kind == "salary_resumes" and amount and eff:
            for f in sal:
                fc.flows.remove(f)
            _monthly_salary(fc, eff, amount, "salary")
            fc.notes.append(f"salary resumes {amount:.2f} from {eff}")
        elif kind == "fx_settlement_note" and amount and eff and "salary" in summary:
            near = [f for f in sal if abs((f.day - eff).days) <= 10]
            if near:
                near[0].day, near[0].amount = eff, amount
            else:
                _monthly_salary(fc, eff, amount, "confirmed salary")
            fc.notes.append(f"confirmed salary {amount:.2f} on {eff}")
        elif kind in ("employment_ended", "income_source_ended") and amount:
            for f in sal:
                f.amount = amount  # one income stream ended; the stated remaining salary continues
            fc.notes.append(f"remaining salary {amount:.2f}")
        elif kind in ("employment_ended", "seasonal_contract_ended"):
            for f in [f for f in fc.flows if f.amount > 0]:
                fc.flows.remove(f)
            fc.notes.append("no further salary confirmed")
        elif kind == "confirmed_invoice" and amount and eff:
            if fc.start <= eff <= fc.end:
                fc.flows.append(Flow(eff, amount, "scheduled", key="confirmed invoice", category="income",
                                     description="confirmed invoice"))
            fc.notes.append(f"confirmed invoice {amount:.2f} on {eff}")
        elif kind == "rent_increase" and fact.get("percent"):
            pct = float(fact["percent"])
            for f in fc.flows:
                if f.category in ("rent", "housing") and f.amount < 0 and f.kind == "recurring_fixed":
                    f.amount *= 1 + pct / 100.0
            fc.notes.append(f"rent up {pct:g}%")
        elif kind == "failed_debit_retry" and fact.get("related_event_id"):
            if (ev.linked_event_id == fact["related_event_id"]).any():
                fc.notes.append("failed debit retry already scheduled")  # avoid reserving it twice
                continue
            row = ev[ev.event_id == fact["related_event_id"]]
            if len(row) and not is_blank(row.iloc[0].amt):
                r = row.iloc[0]
                fc.flows.append(Flow(fc.start, -float(r.amt), "pending", key=r.description, event_id=r.event_id,
                                     category=r.category, description=r.description))
                fc.notes.append(f"failed debit {r.event_id} still due")
