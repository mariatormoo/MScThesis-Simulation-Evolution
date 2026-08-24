# Unit tests for Modules/module1_financial_forecasting/forecasting_models.py
#
# Run with:  pytest -v tests/test_forecasting_models.py
#
# Why this file exists: several of the bugs found during manual review of the
# prototype (the 'apend' typo, the 'susbet' typo, the 2D-history bug in the
# XGBoost forecaster) were one-line mistakes that a two-line unit test would
# have caught instantly, instead of surfacing only when clicking through the
# Streamlit UI. These tests target the pure, deterministic logic in
# forecasting_models.py — no Streamlit, no network calls.

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from Modules.module1_financial_forecasting import forecasting_models as fm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_series() -> pd.DataFrame:
    """A perfectly linear, noise-free series: easy to reason about by hand."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    values = np.arange(60, dtype=float) * 2.0 + 10.0  # y = 2x + 10
    return pd.DataFrame({"ds": dates, "y": values})


@pytest.fixture
def flat_series() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    values = np.full(40, 50.0)
    return pd.DataFrame({"ds": dates, "y": values})


# ---------------------------------------------------------------------------
# mape / rmse
# ---------------------------------------------------------------------------

class TestMape:
    def test_perfect_prediction_is_zero(self):
        y_true = np.array([10.0, 20.0, 30.0])
        assert fm.mape(y_true, y_true) == pytest.approx(0.0)

    def test_known_value_matches_hand_calculation(self):
        # |100-90|/100 = 10% ; |200-180|/200 = 10%  -> mean = 10%
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([90.0, 180.0])
        assert fm.mape(y_true, y_pred) == pytest.approx(10.0)

    def test_does_not_divide_by_zero_when_actual_is_zero(self):
        # Regression guard: mape() uses an eps floor specifically so a zero
        # in y_true can't blow up into inf/NaN.
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([5.0, 10.0])
        result = fm.mape(y_true, y_pred)
        assert math.isfinite(result)
        assert result > 0

    def test_returns_a_python_float(self):
        result = fm.mape(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert isinstance(result, float)


class TestRmse:
    def test_perfect_prediction_is_zero(self):
        y_true = np.array([5.0, 15.0, 25.0])
        assert fm.rmse(y_true, y_true) == pytest.approx(0.0)

    def test_known_value_matches_hand_calculation(self):
        # errors = [3, -4] -> squared = [9, 16] -> mean = 12.5 -> sqrt = 3.5355...
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([7.0, 24.0])
        assert fm.rmse(y_true, y_pred) == pytest.approx(math.sqrt(12.5))

    def test_symmetric_in_error_sign(self):
        # RMSE shouldn't care whether the model over- or under-shoots.
        y_true = np.array([10.0, 20.0])
        over = fm.rmse(y_true, np.array([12.0, 22.0]))
        under = fm.rmse(y_true, np.array([8.0, 18.0]))
        assert over == pytest.approx(under)


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def test_normal_split_sizes(self, linear_series):
        train, test = fm.train_test_split(linear_series, test_size=10)
        assert len(train) == 50
        assert len(test) == 10
        # chronological: train must end right before test begins
        assert train["ds"].max() < test["ds"].min()

    def test_zero_test_size_returns_full_train_empty_test(self, linear_series):
        train, test = fm.train_test_split(linear_series, test_size=0)
        assert len(train) == len(linear_series)
        assert len(test) == 0

    def test_test_size_larger_than_df_returns_full_train_empty_test(self, linear_series):
        train, test = fm.train_test_split(linear_series, test_size=999)
        assert len(train) == len(linear_series)
        assert len(test) == 0

    def test_empty_dataframe_does_not_raise(self):
        empty = pd.DataFrame({"ds": [], "y": []})
        train, test = fm.train_test_split(empty, test_size=10)
        assert len(train) == 0
        assert len(test) == 0


# ---------------------------------------------------------------------------
# manual_cfo_* baselines
# ---------------------------------------------------------------------------

class TestManualCfoBaselines:
    def test_naive_last_value_is_flat_and_correct_length(self, flat_series):
        steps = 15
        preds = fm.manual_cfo_naive_last_value(flat_series, steps)
        assert len(preds) == steps
        assert np.all(preds == flat_series["y"].iloc[-1])

    def test_growth_rate_zero_percent_is_flat(self, flat_series):
        preds = fm.manual_cfo_growth_rate(flat_series, steps=10, monthly_growth_pct=0.0)
        assert preds == pytest.approx(np.full(10, flat_series["y"].iloc[-1]))

    def test_growth_rate_compounds_upward_over_time(self, flat_series):
        preds = fm.manual_cfo_growth_rate(flat_series, steps=90, monthly_growth_pct=5.0)
        # With positive growth, day 90 should be noticeably higher than day 1.
        assert preds[-1] > preds[0]
        # And the whole path should be monotonically increasing (steady positive growth).
        assert np.all(np.diff(preds) > 0)

    def test_growth_rate_output_length_matches_steps(self, flat_series):
        preds = fm.manual_cfo_growth_rate(flat_series, steps=37, monthly_growth_pct=1.2)
        assert len(preds) == 37


# ---------------------------------------------------------------------------
# XGBoost — this is the regression test for the 'apend' + 2D-history bugs
# ---------------------------------------------------------------------------

xgb = pytest.importorskip("xgboost", reason="xgboost not installed; skipping XGBoost tests")


class TestXgboostForecast:
    def test_fit_and_forecast_roundtrip_returns_correct_length(self, linear_series):
        model_tuple = fm.fit_xgboost(linear_series, max_lag=7)
        assert model_tuple[0] is not None, "xgboost is installed but fit_xgboost returned None"

        history = linear_series["y"].values.reshape(-1, 1)  # exactly how forecasting_app.py calls it
        steps = 10
        preds = fm.forecast_xgboost(model_tuple, steps=steps, history=history)

        # Regression guard for the 'history_list.apend(...)' typo: if that
        # typo were reintroduced, this call would raise AttributeError
        # before ever reaching this assertion.
        assert preds is not None
        assert len(preds) == steps

        # Regression guard for the 2D-history-not-flattened bug: predictions
        # must be a flat, finite, numeric sequence — not nested arrays/NaNs
        # coming from a malformed lag window.
        assert preds.ndim == 1
        assert np.all(np.isfinite(preds))

    def test_forecast_is_deterministic_for_same_inputs(self, linear_series):
        model_tuple = fm.fit_xgboost(linear_series, max_lag=7)
        history = linear_series["y"].values.reshape(-1, 1)
        preds_1 = fm.forecast_xgboost(model_tuple, steps=5, history=history)
        preds_2 = fm.forecast_xgboost(model_tuple, steps=5, history=history)
        assert np.allclose(preds_1, preds_2)


# ---------------------------------------------------------------------------
# Holt-Winters / ARIMA — light smoke tests (heavier statistical fitting,
# so we keep these to "does it run and return the right shape", not exact values)
# ---------------------------------------------------------------------------

class TestHoltWintersSmoke:
    def test_fit_and_forecast_returns_correct_length(self, linear_series):
        model = fm.fit_holtwinters(linear_series, seasonal=None, seasonal_periods=None)
        preds = fm.forecast_holtwinters(model, steps=12)
        assert len(preds) == 12
        assert np.all(np.isfinite(preds))


class TestArimaSmoke:
    def test_fit_and_forecast_returns_correct_length(self, linear_series):
        model = fm.fit_arima(linear_series, order=(1, 1, 1))
        preds = fm.forecast_arima(model, steps=8)
        assert len(preds) == 8
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# CSV loading — header detection and basic cleaning
# ---------------------------------------------------------------------------

class TestLoadDataFromCsv(object):
    def test_recognizes_standard_date_value_headers(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("date,value\n2024-01-01,100\n2024-01-02,101\n2024-01-03,102\n")
        df = fm.load_data_from_csv(str(csv_path))
        assert list(df.columns) == ["ds", "y"]
        assert len(df) == 3
        assert df["y"].tolist() == [100.0, 101.0, 102.0]

    def test_falls_back_to_first_two_columns_when_headers_unrecognized(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("timestamp,revenue\n2024-01-01,500\n2024-01-02,510\n")
        df = fm.load_data_from_csv(str(csv_path))
        assert list(df.columns) == ["ds", "y"]
        assert len(df) == 2

    def test_drops_rows_with_unparseable_dates(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("date,value\n2024-01-01,100\nNOT-A-DATE,101\n2024-01-03,102\n")
        df = fm.load_data_from_csv(str(csv_path))
        assert len(df) == 2  # the malformed row is dropped, not left to crash downstream

    def test_returns_none_for_single_column_csv(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("value\n100\n101\n")
        df = fm.load_data_from_csv(str(csv_path))
        assert df is None
