"""Compare predictions on dataset/sample_requests.csv with their published answers."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from main import run  # noqa: E402

FIELDS = ["amount_safe_to_pay", "affordability_status", "recommended_payment_method", "payment_plan",
          "earliest_date_for_full_payment", "spending_changes_needed"]


def main(verbose=True):
    pred = pd.DataFrame(run(samples=True)).set_index("request_id")
    truth = pd.read_csv(HERE.parent / "dataset" / "sample_requests.csv", dtype=str).fillna("").set_index("request_id")
    score = {f: 0 for f in FIELDS}
    close = 0
    for rid, t in truth.iterrows():
        p = pred.loc[rid]
        diffs = []
        for f in FIELDS:
            tv, pv = str(t[f]), str(p[f])
            if f == "amount_safe_to_pay":
                ok = abs(float(tv) - float(pv)) < 0.011
                if abs(float(tv) - float(pv)) <= 0.02 * max(float(t["requested_amount"]), 1):
                    close += 1
            elif f == "payment_plan":
                norm = lambda s: "|".join(f"{x.split(':')[0]}:{float(x.split(':')[1]):.2f}" for x in s.split("|")) if s != "none" else s
                ok = norm(tv) == norm(pv)
            else:
                ok = tv == pv
            score[f] += ok
            if not ok:
                diffs.append(f"{f}: pred={pv} true={tv}")
        if verbose and diffs:
            print(rid, "|", "; ".join(diffs))
    n = len(truth)
    print("\n".join(f"{f:32s} {score[f]}/{n}" for f in FIELDS))
    print(f"{'safe within 2% of request':32s} {close}/{n}")


if __name__ == "__main__":
    main()
