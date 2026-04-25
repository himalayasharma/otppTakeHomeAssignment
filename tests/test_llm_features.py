from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.llm_features import (  # noqa: E402
    FINBERT_FEATURE_COLUMNS,
    _join_finbert,
    attach_finbert,
)


def _synthetic_price_frame(start: str = "2024-01-02", periods: int = 200) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=periods, name="date")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": np.linspace(100, 200, n),
            "high": np.linspace(101, 201, n),
            "low": np.linspace(99, 199, n),
            "close": np.linspace(100, 200, n),
            "volume": np.linspace(1e6, 5e6, n),
            "returns": np.linspace(-0.01, 0.01, n),
            "realized_vol_5d": np.linspace(0.1, 0.4, n),
        },
        index=idx,
    ).astype("float64")


def _synthetic_scores(call_dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "call_date": pd.to_datetime(call_dates).astype("datetime64[ns]"),
            "pos_mean": np.linspace(0.1, 0.5, len(call_dates)),
            "neg_mean": np.linspace(0.05, 0.2, len(call_dates)),
            "neu_mean": np.linspace(0.3, 0.85, len(call_dates)),
            "pos_frac": np.linspace(0.2, 0.6, len(call_dates)),
            "neg_frac": np.linspace(0.05, 0.3, len(call_dates)),
        }
    )


def test_join_strict_past_no_leakage() -> None:
    prices = _synthetic_price_frame(periods=60)
    call_date = str(prices.index[20].date())
    baseline = _join_finbert(prices, _synthetic_scores([call_date]))

    attached_rows = baseline["finbert_pos_mean"].notna()
    assert attached_rows.any()
    for index_date in baseline.index[attached_rows]:
        assert index_date > pd.Timestamp(call_date)

    future_scores = _synthetic_scores([call_date, str((prices.index[-1] + pd.offsets.BDay(1)).date())])
    mutated_future_scores = future_scores.copy()
    mutated_future_scores.loc[1, ["pos_mean", "neg_mean", "neu_mean", "pos_frac", "neg_frac"]] = 999.0

    with_future = _join_finbert(prices, future_scores)
    mutated_future = _join_finbert(prices, mutated_future_scores)
    pdt.assert_frame_equal(with_future.loc[: prices.index[-1]], mutated_future.loc[: prices.index[-1]])


def test_pre_first_call_rows_are_nan() -> None:
    prices = _synthetic_price_frame(periods=60)
    first_call = str(prices.index[20].date())
    out = _join_finbert(prices, _synthetic_scores([first_call]))

    vals = out.loc[out.index <= pd.Timestamp(first_call), FINBERT_FEATURE_COLUMNS].to_numpy()
    assert np.isnan(vals).all()
    assert not (vals == 0).any()


def test_join_preserves_row_count_and_index_and_payload() -> None:
    prices = _synthetic_price_frame(periods=60)
    out = _join_finbert(prices, _synthetic_scores([str(prices.index[20].date())]))

    assert len(out) == len(prices)
    assert out.index.equals(prices.index)
    pdt.assert_frame_equal(out[prices.columns], prices)


def test_join_is_deterministic() -> None:
    prices = _synthetic_price_frame(periods=60)
    scores = _synthetic_scores([str(prices.index[20].date()), str(prices.index[40].date())])

    a = _join_finbert(prices, scores)
    b = _join_finbert(prices, scores)
    pdt.assert_frame_equal(a, b)


def test_attach_finbert_reads_default_parquet(tmp_path: Path) -> None:
    prices = _synthetic_price_frame(periods=60)
    scores = _synthetic_scores([str(prices.index[20].date())])
    parquet_path = tmp_path / "f.parquet"
    scores.to_parquet(parquet_path)

    expected = _join_finbert(prices, scores)
    actual = attach_finbert(prices, scores_path=parquet_path)
    pdt.assert_frame_equal(actual, expected)
