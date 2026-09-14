"""Output formatting and grounded, template-based decision explanations."""


def fmt_num(x):
    """Plain number for CSV fields: integers without decimals, otherwise two decimals."""
    x = round(float(x), 2)
    return str(int(round(x))) if abs(x - round(x)) < 0.005 else f"{x:.2f}"


def fmt_money(cur, x):
    x = round(float(x), 2)
    body = f"{int(round(x)):,}" if abs(x - round(x)) < 0.005 else f"{x:,.2f}"
    return f"{cur} {body}"


def fmt_date(d):
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _change_phrase(changes, cur):
    parts = []
    for kind, flow, new_amt in changes:
        name = (flow.description or flow.category).lower()
        if kind == "stop":
            parts.append(f"stop the {name}")
        else:
            parts.append(f"reduce the {name} to {fmt_money(cur, new_amt)}")
    text = " and ".join(parts)
    return text[0].upper() + text[1:]


def explain(req, profile, decision, fc):
    cur = profile.home_currency
    minimum = fmt_money(cur, profile.minimum_balance_to_keep)
    requested = decision["requested"]
    plan = decision["plan"]
    safe = decision["safe"]
    deadline = decision["deadline"]
    req_txt = fmt_money(cur, requested)

    if plan is None:
        if safe > 0:
            return (f"Do not proceed with the {req_txt} request. Although {fmt_money(cur, safe)} is available today, "
                    f"the full amount cannot be completed safely by {fmt_date(deadline)} while keeping the {minimum} minimum.")
        return (f"Do not make this payment by {fmt_date(deadline)}. None of the available options keeps the "
                f"{minimum} minimum protected.")

    if plan.method == "full_payment" and plan.changes:
        return (f"{_change_phrase(plan.changes, cur)}, then pay {req_txt} today. "
                f"This leaves at least {minimum} available.")
    if plan.method == "installments" and plan.changes:
        n = len(plan.payments)
        return (f"{_change_phrase(plan.changes, cur)}, then use {n} installments of "
                f"{fmt_money(cur, plan.payments[0][1])}, starting {fmt_date(plan.start)}. This leaves at least {minimum} available.")
    if plan.method == "full_payment":
        return f"Pay {req_txt} today. This leaves at least {minimum} available over the next 90 days."
    if plan.method == "wait":
        d = plan.payments[0][0]
        late = "" if d <= deadline else f" This is after the requested {fmt_date(deadline)} date."
        return (f"Wait until {fmt_date(d)}, then pay {req_txt} in full. Paying earlier would take the balance "
                f"below the {minimum} minimum.{late}")
    if plan.method == "partial_payment":
        (d1, a1), (d2, a2) = plan.payments
        return (f"Pay {fmt_money(cur, a1)} today and the remaining {fmt_money(cur, a2)} on {fmt_date(d2)}. "
                f"This completes the full request by {fmt_date(deadline)} while keeping the {minimum} minimum.")
    if plan.method == "installments":
        n = len(plan.payments)
        return (f"Use {n} installments of {fmt_money(cur, plan.payments[0][1])}, starting {fmt_date(plan.start)}. "
                f"This leaves at least {minimum} available.")
    return "No safe recommendation."
