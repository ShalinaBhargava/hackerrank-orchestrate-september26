"""Deterministic financial engine for Buy or Wait?

Pipeline per request:
  1. Load the user's events, convert to home currency, fill blank amounts from image evidence.
  2. Detect recurring income/expenses from settled history and project them over a 90-day horizon.
  3. Reserve pending/scheduled debits, count scheduled (confirmed) income, apply message facts.
  4. Compute the daily balance path, amount_safe_to_pay and earliest_date_for_full_payment.
  5. Build candidate plans (full, wait, partial, installments, spending changes), verify each
     against the minimum balance, and rank them with the challenge's ordering rules.
"""
import calendar
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

HORIZON_DAYS = 90
EPS = 1e-6

# Forecast configuration (calibrated on dataset/sample_requests.csv)
CFG = {
    "fixed_stat": "max",        # amount statistic for monthly items whose amount varies (e.g. utilities)
    "var_stat": "median",       # amount statistic for variable category spending
    "var_lookback": 3,          # number of most recent occurrences used for variable spending
    "min_occurrences": 3,       # occurrences required to call something recurring
    "monthly_day_share": 0.8,   # share of occurrences on the modal day-of-month
    "stale_days_monthly": 40,   # monthly item is dropped if not seen within this many days
    "stale_days_variable": 45,
    "salary_lookback_days": 70,  # payroll pattern is read from this recent window
    "variable_window_days": 120, # trailing window for the daily variable-spending average
    "var_mode": "spread",        # spread: daily average across the horizon | cadence: typical gap and amount
    "var_cats": "essential",     # all | essential (skip discretionary) | protected (profile-protected only)
    "outlier_mult": 3.0,         # drop one-off amounts above this multiple of the category median
    "discretionary_weight": 0.25, # share of discretionary category spend kept when var_cats == "essential"
}

DISCRETIONARY = {"dining", "entertainment", "shopping"}

STATS = {
    "mean": lambda a: float(np.mean(a)),
    "median": lambda a: float(np.median(a)),
    "max": lambda a: float(np.max(a)),
    "last": lambda a: float(a[-1]),
}


def D(s):
    return date.fromisoformat(str(s)[:10])


def add_month(d, k, day):
    y = d.year + (d.month - 1 + k) // 12
    m = (d.month - 1 + k) % 12 + 1
    return date(y, m, min(day, calendar.monthrange(y, m)[1]))


def is_blank(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x.strip() == "")


class FX:
    def __init__(self, df):
        self.rates = {(r.rate_date, r.from_currency, r.to_currency): float(r.rate) for r in df.itertuples()}

    def convert(self, amount, cur, home, day):
        if is_blank(amount) or cur == home or is_blank(cur):
            return amount
        day = str(day)[:10]
        if (day, cur, home) in self.rates:
            return amount * self.rates[(day, cur, home)]
        if (day, home, cur) in self.rates:
            return amount / self.rates[(day, home, cur)]
        best = None
        for (d, f, t), v in self.rates.items():
            if {f, t} == {cur, home}:
                dist = abs((D(d) - D(day)).days)
                rate = v if f == cur else 1.0 / v
                if best is None or dist < best[0]:
                    best = (dist, rate)
        return amount * best[1] if best else amount


@dataclass
class Flow:
    day: date
    amount: float          # signed: credit > 0, debit < 0
    kind: str              # recurring_fixed | recurring_variable | salary | pending | scheduled | message | payment
    key: str = ""          # grouping key (description/category)
    event_id: str = ""     # latest event id for the recurring series (used for spending changes)
    category: str = ""
    flexibility: str = "fixed"
    min_allowed: float = float("nan")
    description: str = ""
    save_stop: float = 0.0     # variable flows: daily saving if the flexible expense is stopped
    save_reduce: float = 0.0   # variable flows: daily saving if reduced to minimum_allowed_amount


@dataclass
class Forecast:
    start: date
    end: date
    balance: float
    minimum: float
    flows: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    flags: list = field(default_factory=list)   # blocking uncertainties: no payment may be recommended

    def path(self, extra=None, drop=None):
        """Daily end-of-day balances from start..end. extra: list of (day, signed amount).
        drop: function(flow) -> adjusted amount (for spending changes)."""
        daily = defaultdict(float)
        for f in self.flows:
            amt = drop(f) if drop else f.amount
            daily[f.day] += amt
        for d, a in extra or []:
            daily[d] += a
        out = []
        b = self.balance
        t = self.start
        while t <= self.end:
            b += daily.get(t, 0.0)
            out.append((t, b))
            t += timedelta(days=1)
        return out


def suffix_min(path):
    res = [0.0] * len(path)
    m = float("inf")
    for i in range(len(path) - 1, -1, -1):
        m = min(m, path[i][1])
        res[i] = m
    return res


def safe_amount_and_earliest(fc, requested, drop=None):
    p = fc.path(drop=drop)
    sm = suffix_min(p)
    # floor to cents so a plan paying exactly this amount never dips below the minimum by rounding
    safe = max(0.0, min(requested, math.floor((sm[0] - fc.minimum) * 100 + 1e-6) / 100))
    earliest = None
    for i, (t, _) in enumerate(p):
        if sm[i] - fc.minimum >= requested - EPS:
            earliest = t
            break
    return round(safe, 2), earliest, p


def plan_is_safe(fc, payments, drop=None):
    p = fc.path(extra=[(d, -a) for d, a in payments], drop=drop)
    lowest = min(b for _, b in p)
    return lowest >= fc.minimum - EPS, lowest
