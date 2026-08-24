# Shared audit trail utility.
# Logs every AI-generated output with module, inputs, and a human-readable
# summary, so the tool can honestly claim a basic governance trail —
# aligned with the AI-CFO Maturity Model (stage 4) discussed in the thesis,
# and loosely inspired by the kind of technical documentation the EU AI Act
# (Annex IV) expects for high-risk systems. This is a lightweight local log,
# not a compliance-grade implementation.
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "audit_log.jsonl")


def log_event(module: str, action: str, inputs: Dict[str, Any], output_summary: str, approved: bool = False) -> None:
    """Append one JSON line describing an AI-assisted action.

    module: e.g. "forecasting", "scenario", "genai_reports"
    action: short verb phrase, e.g. "generated_forecast", "generated_report"
    inputs: dict of the key parameters used (models, tickers, sliders...)
    output_summary: short human-readable description of what was produced
    approved: whether a human explicitly reviewed/approved the output before use
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": module,
        "action": action,
        "inputs": inputs,
        "output_summary": output_summary,
        "human_approved": approved,
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Never let logging break the app.
        pass


def read_log(limit: int = 50):
    """Return the most recent `limit` audit records, newest first."""
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        records = [json.loads(l) for l in lines[-limit:]]
        return list(reversed(records))
    except Exception:
        return []
