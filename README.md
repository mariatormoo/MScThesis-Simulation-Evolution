# CFO Simulation Tool

A prototype exploring how AI can support CFO decision-making — built alongside
the MSc thesis *"The Impact of AI on Decision-Making in Business: How is AI
transforming the role of the CFO"* (Esade, 2025).

## What it does

| Module | Purpose |
|---|---|
| 📈 Financial Forecasting | Compares ARIMA, Holt-Winters, Prophet and XGBoost against a manual CFO baseline, with a 90% prediction interval on the best-performing model. |
| 🧮 Scenario Planning | Stress-tests inflation, sales and interest-rate assumptions with a waterfall chart, tornado sensitivity, Monte Carlo simulation, a real FCF/NPV bridge, and **covenant compliance checks** (interest coverage, leverage) against bank-style thresholds. |
| 📝 Generative AI Decision Reports | Drafts board-ready reports from your KPIs, with mandatory source citation on every quantitative claim. |

The three modules are connected: run a forecast or a scenario, and the report
generator offers to use those results automatically — no retyping numbers
between tabs.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: set `OPENAI_API_KEY` as an environment variable (or paste it into
the UI) to enable real report generation — without it, the app runs fully in
demo mode with sample data.

## Development

```bash
pip install pytest ruff pre-commit
pytest -v                    # 23 unit tests for the forecasting logic
ruff check .                 # static analysis (undefined names, dead imports)
pre-commit install           # runs ruff automatically before each commit
```

## Design notes

- **Grounding over eloquence.** Every AI-generated report must cite the exact
  source field behind each number — the top adoption blocker CFOs flagged in
  interviews was hallucination and lack of explainability, not tone.
- **A basic audit trail** (`Modules/shared/audit_log.py`) logs every
  AI-assisted action with its inputs and whether a human approved it —
  a first, honest step toward the governance a real deployment would need.
- **Probability, not just a point estimate.** Forecasts ship with an
  approximate 90% interval (native for Prophet, bootstrapped from holdout
  residuals otherwise) rather than a single number presented as certain.

See `CHANGELOG.md` for the full history of fixes and design decisions.

## Status

Functional prototype, not production software. Known gaps: `_call_openai`
uses a single fixed model (`gpt-4o-mini`); the Monte Carlo and bootstrap
methods are intentionally simple approximations, not calibrated risk models.
