import numpy as np
import pytest

from screamer import (
    BacktestOHLCTarget,
    PortfolioResult,
    portfolio_columns,
    portfolio_ohlc_target,
)


def _bars(close):
    close = np.asarray(close, dtype=float)
    return close.copy(), close.copy(), close.copy(), close.copy()


def test_portfolio_result_has_fixed_shapes_and_columns():
    close = np.array([[100.0, 50.0], [101.0, 51.0], [102.0, 50.0]])
    open_, high, low, close = _bars(close)
    targets = np.array([[1.0, -1.0], [1.0, -1.0], [0.0, 0.0]])
    result = portfolio_ohlc_target(
        targets, open_, high, low, close,
        multipliers=[10.0, 100.0],
        initial_cash=1_000.0,
        margin_per_contract=[200.0, 50.0],
    )
    assert isinstance(result, PortfolioResult)
    assert result.asset_state.shape == (3, 2, 4)
    assert result.portfolio_state.shape == (3, 10)
    assert result.portfolio_state.dtype == float
    assert portfolio_columns == (
        "equity", "pnl", "cash", "gross_exposure", "net_exposure",
        "margin_used", "cost", "turnover", "trades", "drawdown",
    )
    assert result.summary["num_trades"] == 2.0


def test_single_asset_matches_native_cpp_engine_after_multiplier_scaling():
    close = np.array([100.0, 101.0, 99.0, 102.0])
    open_, high, low, close = _bars(close)
    target = np.array([1.0, 1.0, -1.0, 0.0])
    multiplier = 25.0
    result = portfolio_ohlc_target(
        target, open_, high, low, close,
        multipliers=multiplier,
        taker_fee=0.0002,
        tick_size=0.01,
    )
    expected = BacktestOHLCTarget(
        taker_fee=0.0002,
        tick_size=0.01 * multiplier,
    )(
        target,
        open_ * multiplier,
        high * multiplier,
        low * multiplier,
        close * multiplier,
    )
    np.testing.assert_allclose(result.asset_state[:, 0, :], expected)
    np.testing.assert_allclose(
        result.portfolio_state[:, 1],
        expected[:, 1],
    )


def test_vectorized_assets_match_independent_native_columns():
    rng = np.random.default_rng(4)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.2, (80, 3)), axis=0)
    open_, high, low, close = _bars(close)
    targets = np.sign(rng.normal(size=close.shape))
    multipliers = np.array([10.0, 50.0, 100.0])
    result = portfolio_ohlc_target(
        targets, open_, high, low, close, multipliers=multipliers,
        taker_fee=0.0001,
    )
    for asset, multiplier in enumerate(multipliers):
        expected = BacktestOHLCTarget(taker_fee=0.0001)(
            targets[:, asset],
            open_[:, asset] * multiplier,
            high[:, asset] * multiplier,
            low[:, asset] * multiplier,
            close[:, asset] * multiplier,
        )
        np.testing.assert_allclose(
            np.nan_to_num(result.asset_state[:, asset, :]),
            np.nan_to_num(expected),
        )


def test_ohlc_target_is_causal_in_portfolio_helper():
    close = np.array([
        [100.0, 50.0],
        [105.0, 55.0],
        [106.0, 56.0],
    ])
    open_, high, low, close = _bars(close)
    targets = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
    full = portfolio_ohlc_target(targets, open_, high, low, close)
    trunc = portfolio_ohlc_target(targets[:2], open_[:2], high[:2], low[:2], close[:2])
    np.testing.assert_allclose(full.asset_state[:2], trunc.asset_state)
    np.testing.assert_allclose(full.portfolio_state[:2], trunc.portfolio_state)


def test_portfolio_aggregates_contract_pnl_exposure_margin_and_drawdown():
    close = np.array([[100.0, 50.0], [101.0, 52.0], [99.0, 51.0]])
    open_, high, low, close = _bars(close)
    targets = np.ones_like(close)
    result = portfolio_ohlc_target(
        targets, open_, high, low, close,
        multipliers=[10.0, 100.0],
        initial_cash=1_000.0,
        margin_per_contract=[200.0, 50.0],
    )
    # Targets are deferred: both contracts are long from row 1.
    np.testing.assert_allclose(result.asset_state[:, :, 2], [[0, 0], [1, 1], [1, 1]])
    np.testing.assert_allclose(result.portfolio_state[:, 0], [1000.0, 1000.0, 880.0])
    np.testing.assert_allclose(result.portfolio_state[:, 3], [0.0, 6210.0, 6090.0])
    np.testing.assert_allclose(result.portfolio_state[:, 4], [0.0, 6210.0, 6090.0])
    np.testing.assert_allclose(result.portfolio_state[:, 5], [0.0, 250.0, 250.0])
    np.testing.assert_allclose(result.portfolio_state[:, 9], [0.0, 0.0, -120.0])
    assert result.summary["total_pnl"] == -120.0
    assert result.summary["max_drawdown"] == -120.0


def test_skipped_asset_bar_carries_state_and_aggregate_pnl():
    close = np.array([
        [100.0, 50.0],
        [101.0, np.nan],
        [103.0, 52.0],
    ])
    open_, high, low, close = _bars(close)
    targets = np.ones_like(close)
    result = portfolio_ohlc_target(targets, open_, high, low, close)
    assert np.all(np.isnan(result.asset_state[1, 1]))
    assert result.asset_state[2, 1, 2] == 1.0
    # The deferred position marks the second asset from 50 to 52 across the gap.
    np.testing.assert_allclose(result.portfolio_state[:, 1], [0.0, 0.0, 2.0])


def test_index_and_validation_contract():
    close = np.ones((3, 2)) * 100.0
    open_, high, low, close = _bars(close)
    targets = np.zeros_like(close)
    index = np.array(["a", "b", "c"])
    result = portfolio_ohlc_target(targets, open_, high, low, close, index=index)
    np.testing.assert_array_equal(result.index, index)
    with pytest.raises(ValueError, match="same shape"):
        portfolio_ohlc_target(targets, open_[:, :1], high[:, :1], low[:, :1], close[:, :1])
    with pytest.raises(ValueError, match="strictly positive"):
        portfolio_ohlc_target(targets, open_, high, low, close, multipliers=[1.0, 0.0])
    with pytest.raises(ValueError, match="non-zero tick_size"):
        portfolio_ohlc_target(
            targets, open_, high, low, close,
            multipliers=[10.0, 100.0], tick_size=0.01,
        )
    with pytest.raises(ValueError, match="length T"):
        portfolio_ohlc_target(targets, open_, high, low, close, index=[1, 2])


def test_portfolio_call_is_repeatable():
    close = np.array([[100.0, 50.0], [101.0, 51.0], [102.0, 52.0]])
    open_, high, low, close = _bars(close)
    targets = np.array([[1.0, -1.0], [1.0, -1.0], [0.0, 0.0]])
    first = portfolio_ohlc_target(targets, open_, high, low, close, multipliers=[10.0, 20.0])
    second = portfolio_ohlc_target(targets, open_, high, low, close, multipliers=[10.0, 20.0])
    np.testing.assert_allclose(first.asset_state, second.asset_state, equal_nan=True)
    np.testing.assert_allclose(first.portfolio_state, second.portfolio_state, equal_nan=True)
    assert first.summary == second.summary
