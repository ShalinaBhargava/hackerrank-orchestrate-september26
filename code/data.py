"""Load the challenge dataset once and index it for per-request lookups."""
import json
from pathlib import Path

import pandas as pd

from engine import FX

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
EVIDENCE_PATH = Path(__file__).resolve().parent / "cache" / "evidence.json"


class Data:
    def __init__(self, dataset_dir=DATASET, evidence_path=EVIDENCE_PATH):
        d = Path(dataset_dir)
        self.profiles = pd.read_csv(d / "financial_profiles.csv").set_index("user_id")
        events = pd.read_csv(d / "financial_events.csv", dtype={"linked_event_id": str})
        self.events_by_user = {u: g.copy() for u, g in events.groupby("user_id")}
        options = pd.read_csv(d / "request_payment_options.csv")
        self.options_by_request = {r: g.copy() for r, g in options.groupby("request_id")}
        messages = pd.read_csv(d / "messages.csv", dtype=str).fillna("")
        self.messages_by_user = {u: g.copy() for u, g in messages.groupby("user_id")}
        self.images = pd.read_csv(d / "images.csv", dtype=str).fillna("")
        self.fx = FX(pd.read_csv(d / "exchange_rates.csv"))
        self.requests = pd.read_csv(d / "requests.csv")
        self.samples = pd.read_csv(d / "sample_requests.csv")

        ev_path = Path(evidence_path)
        self.evidence = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else {"messages": {}, "images": {}}
        # event_id -> (amount, currency) extracted from the linked image
        self.image_amounts = {}
        # event_id -> full extracted image record (validated by guardrails before use)
        self.image_info = {info.get("related_event_id"): info for info in self.evidence.get("images", {}).values()}
        for img_id, info in self.evidence.get("images", {}).items():
            amt = info.get("amount")
            if amt is None:
                continue
            try:
                self.image_amounts[info.get("related_event_id")] = (float(amt), info.get("currency"))
            except (TypeError, ValueError):
                continue

    def message_facts(self, user_id, request_date):
        """Facts from messages for this user sent on or before the request date, oldest first."""
        msgs = self.messages_by_user.get(user_id)
        if msgs is None:
            return []
        out = []
        for r in msgs.sort_values("sent_at").itertuples():
            if r.sent_at[:10] > str(request_date)[:10]:
                continue
            fact = self.evidence.get("messages", {}).get(r.message_id)
            if fact:
                out.append({**fact, "message_id": r.message_id, "related_event_id": r.related_event_id,
                            "request_id": r.request_id, "sent_at": r.sent_at})
        return out
