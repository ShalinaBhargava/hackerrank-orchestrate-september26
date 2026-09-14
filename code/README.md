# Buy or Wait? — Financial Decision Agent

For every request in `dataset/requests.csv` the agent decides whether the user should pay in full, pay
partially, use a supplied installment option, wait, or not proceed, and writes `output.csv` in the repository root.

## Architecture

```
code/
  main.py            entry point: loads data, solves every request, validates, writes ./output.csv
  data.py            loads and indexes the CSVs and the cached evidence
  evidence.py        LLM step (LiteLLM + Gemini): structures messages and reads images -> cache/evidence.json
  engine.py          FX conversion, daily balance path, safety checks, calibrated configuration (CFG)
  forecast.py        reconstructs the 90-day cash forecast for a request
  planner.py         candidate plans, safety verification, spending changes, ranking
  explain.py         number/date formatting and grounded explanation templates
  guardrails.py      evidence validation, plan constraints, independent output verifier (fail closed)
  score_samples.py   compares predictions with dataset/sample_requests.csv
  evaluation/main.py builds evaluation/usage_report.md from cache/usage.json
  tests/             test_guardrails.py: adversarial safety checks
  ui/                FastAPI + Chart.js explorer (app.py, static/index.html)
  cache/             evidence.json (cached LLM facts), usage.json (token log), guardrail_report.csv
```

**Hybrid design.** The only model usage is a one-time evidence extraction: messages (English/Indonesian) are
classified into a fixed set of fact types (salary change, first salary, pay-date move, confirmed invoice, pending
income, rent increase, failed debit, scam, ...) with amounts and dates, and images linked to blank-amount events are
read for the missing amount. Message and image content is treated as untrusted data: the prompts request facts only,
and the engine applies each fact type with fixed rules, so embedded instructions cannot change a decision.
Everything else is deterministic Python.

**Forecast (90 days from `request_date`).**
- Starting point: `current_available_balance`; pending and scheduled debits are reserved on their settlement date
  (or today if already due). Pending credits, refunds, bonuses, prizes, unrealized gains, failed and cancelled rows are ignored.
- Foreign-currency amounts use `exchange_rates.csv` on the settlement date (inverse pair if only that direction exists).
- Salary: a scheduled confirmed salary is counted on its date and repeated monthly; otherwise the recent payroll
  pattern (modal pay day, higher of the last two regular amounts). Commissions, bonuses, arrears and
  reimbursements are not regular salary. A final payroll stops the series; a confirmed first salary counts once.
- Monthly debits (rent, bills, subscriptions, loan payments) are detected by description with a consistent day of month.
- Variable spending (groceries, transport and other irregular spend) is a trailing 120-day daily average per
  category, excluding one-off amounts above 3x the category median. Discretionary categories (dining, entertainment,
  shopping) are counted at 25%, a weight calibrated on the 25 solved samples.
- Message facts adjust the forecast (salary amount/date changes, first or resumed salary, ended income, confirmed invoices,
  rent increases, failed debits still due).

**Decision.**
- `amount_safe_to_pay` = lowest projected balance over the horizon minus `minimum_balance_to_keep`, clamped to [0, requested].
- `earliest_date_for_full_payment` = first day a single full payment keeps every later day at or above the minimum.
- Candidates: full payment today, wait (full payment on the earliest date), partial payment (safe amount today, remainder
  on the earliest date), each supplied installment option within `max_installment_months`. Every plan is re-simulated
  day by day and kept only if the balance never falls below the minimum.
- If no plan completes by `desired_completion_date`, up to three spending changes (`stop` / `reduce_to` the
  `minimum_allowed_amount`) on flexible, non-protected expenses in categories the user allows are searched, fewest first.
- Ranking: completes by deadline, no spending changes, lowest total paid, earliest start, fewest payments, lowest option id.
- Each row is validated (bounds, allowed values, chronological plan, partial sums, change count) before writing.

## Safety guardrails (`code/guardrails.py`)

A wrong "go ahead" costs the user real money, so every guard fails closed:

| Layer | Guard |
|---|---|
| Input | Unknown user, non-positive/non-numeric amount, invalid dates, deadline before request, invalid profile balances → `not_recommended`. |
| Model evidence | Message facts must use a known fact type, known currency, positive amount and valid date. Facts that *add* money must be explicitly confirmed, fall inside the horizon and be ≤ 3× observed salary; otherwise they are ignored. Scam / "pay a fee to receive funds" messages never change the forecast. |
| Image evidence | Amount must be positive, in the event's currency and plausible against same-category history. An implausible credit is dropped; an implausibly small debit is replaced by the category maximum. |
| Blank amounts | Never treated as zero. An unreadable open debit is reserved at the largest same-category amount; if nothing can be estimated the forecast is flagged and no payment is recommended. |
| Income | Pending credits, refunds, bonuses, commissions, prizes, unrealized gains are never counted; a first salary counts once (no invented repeats); duplicate credits are removed, look-alike debits are kept. |
| Double counting | A failed debit already rescheduled as a retry event is not reserved twice. |
| Plans | Installments must match a supplied option, respect `max_installment_months`, start on/after the request date and finish by the deadline and inside the 90-day horizon. Partial payments must be allowed by the request and the user and sum exactly to the request. Amounts are floored to cents so rounding cannot breach the minimum. |
| Output verifier | Each finished row is parsed back from its CSV text and re-simulated day by day, independently of the planner: bounds, status/method consistency, user-accepted method, chronological plan, balance ≥ minimum on every day, spending changes only on the user's own flexible, permitted, non-protected events with `reduce_to` ≥ `minimum_allowed_amount`. Any violation downgrades the row to `not_recommended`. |
| Crashes | Any exception for a request yields a conservative `not_recommended` row instead of a guess. |

Every run writes `code/cache/guardrail_report.csv` (downgrades, blocked forecasts, ignored evidence).
`python code/tests/test_guardrails.py` feeds corrupted rows and malicious/implausible evidence to the guards
(27 checks) and fails if any unsafe answer gets through.

## Setup

Requires Python 3.10+ (developed on 3.14).

```bash
pip install -r code/requirements.txt
```

The cached evidence in `code/cache/evidence.json` is included, so **no API key is needed to reproduce `output.csv`**.
To re-run the LLM extraction, create `.env` in the repository root (see `.env.example`):

```
GEMINI_API_KEY=...
LLM_MODEL=gemini/gemini-3.6-flash
```

## Run

```bash
python code/main.py                 # -> output.csv (250 rows)
python code/score_samples.py        # accuracy on the 25 solved samples
python code/evidence.py             # (optional) extract/refresh evidence; resumes from cache
python code/main.py --extract       # extraction then prediction
python code/evaluation/main.py      # regenerate evaluation/usage_report.md
```

## Explorer UI

```bash
python code/ui/app.py               # then open http://127.0.0.1:8000
```

A FastAPI + Chart.js page to correlate each request with its user: profile (balance, minimum, protected /
reducible / stoppable categories, accepted methods), the decision (with sample-answer comparison), the 90-day
balance forecast with and without the recommended plan (minimum line, deadline and earliest-date markers,
payment points), monthly cash flow, 90-day spending by category, projected flows, payment options, open events,
messages with their extracted facts, linked images, and recent events.

Runtime: about 20 seconds for the full dataset on a laptop, deterministic given the cached evidence.
