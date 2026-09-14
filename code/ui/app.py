"""Buy or Wait? Explorer — a small FastAPI UI to correlate requests, user profiles, evidence and decisions.

Run from the repository root:
    python code/ui/app.py            # http://127.0.0.1:8000
"""
import math
import sys
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from data import DATASET, ROOT, Data  # noqa: E402
from engine import D  # noqa: E402
from forecast import build_forecast  # noqa: E402
from main import solve  # noqa: E402
from planner import _drop_fn, decide  # noqa: E402

app = FastAPI(title="Buy or Wait? Explorer")
DATA = Data()
OUTPUT_FIELDS = ["amount_safe_to_pay", "affordability_status", "recommended_payment_method", "payment_plan",
                 "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"]

REQUESTS = {}
for r in DATA.requests.to_dict("records"):
    REQUESTS[r["request_id"]] = (r, "evaluation")
for r in DATA.samples.to_dict("records"):
    REQUESTS[r["request_id"]] = (r, "sample")


def clean(x):
    """Make pandas/numpy values JSON-safe."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) or math.isinf(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return x


def _read_output(path):
    return pd.read_csv(path, dtype=str).fillna("").set_index("request_id") if Path(path).exists() else pd.DataFrame()


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/media/{image_id}")
def media(image_id: str):
    if not image_id.replace("_", "").isalnum():
        raise HTTPException(400, "bad image id")
    path = DATASET / "media" / "images" / f"{image_id}.png"
    if not path.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(path)


@app.get("/api/requests")
def list_requests():
    out = pd.concat([_read_output(ROOT / "output.csv"), _read_output(CODE / "cache" / "sample_output.csv")])
    items = []
    for rid, (r, kind) in REQUESTS.items():
        prof = DATA.profiles.loc[r["user_id"]]
        o = out.loc[rid] if rid in out.index else {}
        items.append({
            "request_id": rid, "user_id": r["user_id"], "set": kind, "request_type": r["request_type"],
            "requested_amount": r["requested_amount"], "currency": prof.home_currency,
            "status": o.get("affordability_status", "") if len(o) else "",
            "method": o.get("recommended_payment_method", "") if len(o) else "",
            "truth_status": r.get("affordability_status", "") if kind == "sample" else "",
        })
    items.sort(key=lambda x: int(x["request_id"].split("_")[1]))
    return clean(items)


@app.get("/api/request/{rid}")
def request_detail(rid: str):
    if rid not in REQUESTS:
        raise HTTPException(404, "unknown request")
    req, kind = REQUESTS[rid]
    uid = req["user_id"]
    prof = DATA.profiles.loc[uid]
    home = prof.home_currency

    row, guard = solve(DATA, req)
    fc = build_forecast(DATA, req)
    dec = decide(DATA, req, fc)

    # balance paths: baseline, and with the recommended plan (and its spending changes) applied
    baseline = fc.path()
    plan_pays = []
    if row["payment_plan"] != "none":
        plan_pays = [(D(p.split(":")[0]), float(p.split(":")[1])) for p in row["payment_plan"].split("|")]
    changes = dec["plan"].changes if dec["plan"] is not None and row["spending_changes_needed"] != "none" else []
    with_plan = fc.path(extra=[(d, -a) for d, a in plan_pays], drop=_drop_fn(changes)) if plan_pays else None

    flows = {}
    for f in fc.flows:
        k = (f.kind, f.key)
        a = flows.setdefault(k, {"kind": f.kind, "key": f.key, "category": f.category, "count": 0, "total": 0.0,
                                 "first": f.day, "last": f.day, "flexibility": f.flexibility})
        a["count"] += 1
        a["total"] += f.amount
        a["last"] = f.day

    # history: last 6 months before the request, converted to home currency
    ev = DATA.events_by_user[uid].copy()
    start = D(req["request_date"])
    since = str(start - timedelta(days=183))
    ev["amt_home"] = [
        DATA.fx.convert(float(a), c, home, s if isinstance(s, str) else e) if not pd.isna(a) else None
        for a, c, s, e in zip(ev.amount, ev.currency, ev.settlement_date, ev.event_date)]
    hist = ev[(ev.event_date >= since) & (ev.event_date < str(start)) & (ev.status == "settled")].copy()
    hist["month"] = hist.event_date.str[:7]
    monthly = {}
    for (m, direction, cat), g in hist.groupby(["month", "direction", "category"]):
        monthly.setdefault(m, {"income": 0.0, "expenses": {}})
        total = float(g.amt_home.fillna(0).sum())
        if direction == "credit":
            monthly[m]["income"] += total
        elif direction == "debit":
            monthly[m]["expenses"][cat] = monthly[m]["expenses"].get(cat, 0.0) + total
    recent90 = hist[(hist.event_date >= str(start - timedelta(days=90))) & (hist.direction == "debit")]
    by_cat = recent90.groupby("category").amt_home.sum().sort_values(ascending=False)

    open_events = ev[ev.status.isin(["pending", "scheduled", "failed", "cancelled", "unrealized"])]
    msgs = DATA.messages_by_user.get(uid)
    # source links: images and messages that describe a specific event row
    img_by_event = {im.related_event_id: im.image_id for im in DATA.images[DATA.images.user_id == uid].itertuples()}
    msg_by_event = {}
    if msgs is not None:
        for m in msgs.itertuples():
            if m.related_event_id:
                msg_by_event.setdefault(m.related_event_id, []).append(m.message_id)
    linked = set(img_by_event) | set(msg_by_event)
    before = ev[ev.event_date < str(start)].sort_values("event_date", ascending=False)
    # latest 60 events, plus every event that has a source document even if it is older
    recent_events = pd.concat([before.head(60), before[before.event_id.isin(linked)]])
    recent_events = recent_events.drop_duplicates("event_id").sort_values("event_date", ascending=False)
    messages = []
    if msgs is not None:
        for m in msgs.sort_values("sent_at").itertuples():
            fact = DATA.evidence.get("messages", {}).get(m.message_id, {})
            messages.append({"message_id": m.message_id, "sent_at": m.sent_at, "source_type": m.source_type,
                             "request_id": m.request_id, "related_event_id": m.related_event_id,
                             "text": m.message_text, "kind": fact.get("kind"), "summary": fact.get("summary"),
                             "amount": fact.get("amount"), "effective_date": fact.get("effective_date"),
                             "after_request": m.sent_at[:10] > req["request_date"]})
    images = []
    for im in DATA.images[DATA.images.user_id == uid].itertuples():
        info = DATA.evidence.get("images", {}).get(im.image_id, {})
        images.append({"image_id": im.image_id, "request_id": im.request_id, "related_event_id": im.related_event_id,
                       "url": f"/media/{im.image_id}", "amount": info.get("amount"), "currency": info.get("currency"),
                       "document_type": info.get("document_type"), "notes": info.get("notes")})
    opts = DATA.options_by_request.get(rid)
    cols = ["event_id", "event_type", "description", "category", "direction", "amount", "currency", "event_date",
            "settlement_date", "status", "linked_event_id", "flexibility", "minimum_allowed_amount"]

    def event_records(df):
        recs = df[cols].to_dict("records")
        for r in recs:
            iid = img_by_event.get(r["event_id"])
            info = DATA.evidence.get("images", {}).get(iid, {}) if iid else {}
            r.update({"image_id": iid, "image_url": f"/media/{iid}" if iid else None,
                      "image_amount": info.get("amount"), "image_currency": info.get("currency"),
                      "image_doc": info.get("document_type"), "image_notes": info.get("notes"),
                      "message_ids": msg_by_event.get(r["event_id"], [])})
        return recs

    return clean({
        "request": {k: req.get(k) for k in ["request_id", "user_id", "request_date", "request_type", "requested_amount",
                                            "desired_completion_date", "allows_partial_payment", "request_text"]},
        "set": kind,
        "profile": {"user_id": uid, **prof.to_dict()},
        "decision": {k: row[k] for k in OUTPUT_FIELDS},
        "truth": {k: req.get(k) for k in OUTPUT_FIELDS} if kind == "sample" else None,
        "guard": guard,
        "forecast": {
            "minimum": fc.minimum, "start_balance": fc.balance, "start": fc.start, "end": fc.end,
            "deadline": req["desired_completion_date"], "requested": float(req["requested_amount"]),
            "dates": [d for d, _ in baseline], "baseline": [b for _, b in baseline],
            "with_plan": [b for _, b in with_plan] if with_plan else None,
            "payments": [{"date": d, "amount": a} for d, a in plan_pays],
            "notes": fc.notes, "flags": fc.flags,
            "flows": sorted(flows.values(), key=lambda x: x["total"]),
        },
        "history": {
            "months": sorted(monthly), "monthly": monthly,
            "category_90d": [{"category": c, "total": float(v)} for c, v in by_cat.items()],
        },
        "open_events": event_records(open_events),
        "recent_events": event_records(recent_events),
        "options": opts.to_dict("records") if opts is not None else [],
        "messages": messages,
        "images": images,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
