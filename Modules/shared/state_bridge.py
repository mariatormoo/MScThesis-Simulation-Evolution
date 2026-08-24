# Shared session_state bridge between modules.
# Centralizes the keys used to pass results from Forecasting / Scenario
# Planning into the GenAI Reports module, so a user's numbers flow through
# the tool instead of three isolated demos.
from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st

KEY_LAST_FORECAST = "bridge_last_forecast"
KEY_LAST_SCENARIO = "bridge_last_scenario"


def set_last_forecast(ticker: str, best_model: str, mape: float, rmse: float, horizon_months: int) -> None:
    st.session_state[KEY_LAST_FORECAST] = {
        "ticker": ticker,
        "best_model": best_model,
        "mape": mape,
        "rmse": rmse,
        "horizon_months": horizon_months,
    }


def get_last_forecast() -> Optional[Dict[str, Any]]:
    return st.session_state.get(KEY_LAST_FORECAST)


def set_last_scenario(profit: float, margin: float, npv: float, years: int,
                       inflation: float, sales_change: float, interest_rate: float,
                       loss_probability: Optional[float] = None,
                       covenant_breach_years: Optional[list] = None) -> None:
    st.session_state[KEY_LAST_SCENARIO] = {
        "profit": profit,
        "margin": margin,
        "npv": npv,
        "years": years,
        "inflation": inflation,
        "sales_change": sales_change,
        "interest_rate": interest_rate,
        "loss_probability": loss_probability,
        "covenant_breach_years": covenant_breach_years or [],
    }


def get_last_scenario() -> Optional[Dict[str, Any]]:
    return st.session_state.get(KEY_LAST_SCENARIO)


def has_any_results() -> bool:
    return bool(get_last_forecast() or get_last_scenario())


def as_metrics_text() -> str:
    """Render whatever is available as a compact text block for the GenAI prompt."""
    lines = []
    fc = get_last_forecast()
    if fc:
        lines.append(
            f"Forecast ({fc['ticker']}): best model {fc['best_model']}, "
            f"MAPE {fc['mape']:.2f}%, RMSE {fc['rmse']:.2f}, horizon {fc['horizon_months']} months."
        )
    sc = get_last_scenario()
    if sc:
        lp = f", loss probability {sc['loss_probability']:.1%}" if sc.get("loss_probability") is not None else ""
        breaches = sc.get("covenant_breach_years") or []
        cov = f" COVENANT BREACH projected in year(s) {breaches} — flag this explicitly." if breaches else " No covenant breaches projected."
        lines.append(
            f"Scenario ({sc['years']}y): inflation {sc['inflation']:.1f}%, sales Δ {sc['sales_change']:.1f}%, "
            f"rate {sc['interest_rate']:.1f}% -> profit ${sc['profit']:,.0f} (margin {sc['margin']:.1f}%), "
            f"NPV ${sc['npv']:,.0f}{lp}.{cov}"
        )
    return "\n".join(lines)
