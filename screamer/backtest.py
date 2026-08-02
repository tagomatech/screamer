"""Backtest reporting helper.

A thin wrapper over the C++ ``BacktestReport`` node. That node does all the
aggregation (drawdown, cumulative cost, turnover, trade count, running Sharpe);
this helper only labels its columns and reads the last row for the summary, so
pure-C++ users get the same functionality by calling ``BacktestReport`` directly.
No pandas: ``running`` is a dict of numpy arrays and ``summary`` a dict of floats.
Wrap ``running`` in ``pandas.DataFrame`` yourself if you want a frame.
"""
import math
from typing import NamedTuple

import numpy as np

from .screamer_bindings import BacktestReport
from .screamer_bindings import BacktestOHLCTarget

#: Market-order price sentinel for the backtest engines. A quote or limit price of
#: ``MARKET`` (or any non-finite price) is a market order in the aggressive
#: direction; ``NaN`` works as a side-agnostic shorthand.
MARKET = math.inf

__all__ = ["backtest_report", "portfolio_ohlc_target", "PortfolioResult", "portfolio_columns", "MARKET"]

# Column order emitted by the C++ BacktestReport node.
_REPORT_COLUMNS = ("drawdown", "cum_cost", "turnover", "trades", "max_drawdown", "sharpe")

_PORTFOLIO_COLUMNS = (
    "equity", "pnl", "cash", "gross_exposure", "net_exposure",
    "margin_used", "cost", "turnover", "trades", "drawdown",
)


class PortfolioResult(NamedTuple):
    """Fixed-shape result returned by portfolio_ohlc_target."""

    asset_state: np.ndarray
    portfolio_state: np.ndarray
    summary: dict
    index: object = None


portfolio_columns = _PORTFOLIO_COLUMNS


def backtest_report(values, index=None):
    """Running report columns and a summary for a backtest engine's output.

    ``values`` is the ``(T, 4)`` array a backtest engine emits, with columns
    ``[equity, pnl, position, cost]``. Returns ``(running, summary)``:

    - ``running``: a dict of numpy arrays, the four engine columns plus the
      ``BacktestReport`` node's ``drawdown`` (dollar), ``cum_cost``, ``turnover``
      (units traded), ``trades`` (count), ``max_drawdown`` (running worst), and
      ``sharpe`` (running). Each is a causal series whose last finite value is the
      summary.
    - ``summary``: a dict of the final statistics: ``total_pnl``, ``max_drawdown``,
      ``total_cost``, ``turnover``, ``num_trades``, ``sharpe``.

    The aggregation lives in the C++ ``BacktestReport`` node; this wrapper only
    labels its columns and reads the last finite row. No pandas dependency.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(
            "values must be a (T, 4) array of [equity, pnl, position, cost]")
    equity, pnl, position, cost = (values[:, i] for i in range(4))

    report = np.asarray(BacktestReport()(equity, pnl, position, cost), dtype=float)
    running = {"equity": equity, "pnl": pnl, "position": position, "cost": cost}
    for i, name in enumerate(_REPORT_COLUMNS):
        running[name] = report[:, i]
    if index is not None:
        running["index"] = np.asarray(index)

    def _last_finite(a):
        finite = a[np.isfinite(a)]
        return float(finite[-1]) if finite.size else float("nan")

    summary = {
        "total_pnl": _last_finite(equity),
        "max_drawdown": _last_finite(running["max_drawdown"]),
        "total_cost": _last_finite(running["cum_cost"]),
        "turnover": _last_finite(running["turnover"]),
        "num_trades": _last_finite(running["trades"]),
        "sharpe": _last_finite(running["sharpe"]),
    }
    return running, summary


def _as_time_asset(values, name):
    """Return a floating (T, A) matrix, accepting a single asset vector."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    return array


def _as_asset_vector(values, assets, name, *, nonnegative=False):
    """Broadcast a scalar or validate a per-asset vector."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(assets, float(array), dtype=float)
    elif array.ndim == 1 and array.shape[0] == assets:
        array = array.astype(float, copy=False)
    else:
        raise ValueError(f"{name} must be a scalar or a vector of length {assets}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _carry_forward(values, initial=0.0):
    """Carry finite column values over skipped rows without a Python data loop."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    rows = np.broadcast_to(np.arange(values.shape[0])[:, None], values.shape)
    last = np.maximum.accumulate(np.where(finite, rows, -1), axis=0)
    take = np.maximum(last, 0)
    carried = np.take_along_axis(values, take, axis=0)
    return np.where(last >= 0, carried, initial)


def _portfolio_summary(portfolio_state, initial_cash):
    """Build scalar statistics using the backtest_report conventions."""
    equity = portfolio_state[:, 0]
    pnl = portfolio_state[:, 1]
    cost = portfolio_state[:, 6]
    turnover = portfolio_state[:, 7]
    trades = portfolio_state[:, 8]
    drawdown = portfolio_state[:, 9]
    sd = float(np.std(pnl, ddof=1)) if pnl.size > 1 else float("nan")
    sharpe = float(np.mean(pnl) / sd) if np.isfinite(sd) and sd > 0.0 else float("nan")
    return {
        "total_pnl": float(equity[-1] - initial_cash) if equity.size else float("nan"),
        "max_drawdown": float(np.min(drawdown)) if drawdown.size else float("nan"),
        "total_cost": float(np.sum(cost)) if cost.size else float("nan"),
        "turnover": float(np.sum(turnover)) if turnover.size else float("nan"),
        "num_trades": float(np.sum(trades)) if trades.size else float("nan"),
        "sharpe": sharpe,
    }


def portfolio_ohlc_target(
    targets,
    open_,
    high,
    low,
    close,
    *,
    multipliers=1.0,
    initial_cash=0.0,
    margin_per_contract=0.0,
    taker_fee=0.0,
    tick_size=0.0,
    min_position=-math.inf,
    max_position=math.inf,
    index=None,
):
    """Run causal OHLC target backtests for several instruments at once.

    The execution path is the existing C++ BacktestOHLCTarget node, which
    processes trailing NumPy columns independently and returns (T, A, 4).
    This helper adds contract multipliers and a vectorized portfolio report;
    it does not loop over assets in Python and does not implement shared cash
    allocation yet.

    Multipliers may be a scalar or a length-A vector. Market prices are scaled
    before entering the C++ mark-to-market engine, making PnL and fees
    dollar-denominated while preserving positions in contracts. A non-zero
    tick_size is accepted only when all multipliers are equal, because the
    current C++ engine has one tick-size constructor parameter.
    margin_per_contract is a reporting input and does not reserve cash.

    Returns a PortfolioResult with asset_state shape (T, A, 4), using
    [equity, pnl, position, cost], and portfolio_state shape (T, 10), using
    portfolio_columns.
    """
    arrays = {
        "targets": _as_time_asset(targets, "targets"),
        "open": _as_time_asset(open_, "open"),
        "high": _as_time_asset(high, "high"),
        "low": _as_time_asset(low, "low"),
        "close": _as_time_asset(close, "close"),
    }
    shape = arrays["targets"].shape
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError("targets and OHLC arrays must have the same shape")
    periods, assets = shape
    if periods == 0 or assets == 0:
        raise ValueError("targets and OHLC arrays must not be empty")

    multiplier = _as_asset_vector(multipliers, assets, "multipliers")
    if np.any(multiplier <= 0.0):
        raise ValueError("multipliers must be strictly positive")
    margin = _as_asset_vector(
        margin_per_contract, assets, "margin_per_contract", nonnegative=True
    )

    scalar_parameters = {
        "initial_cash": initial_cash,
        "taker_fee": taker_fee,
        "tick_size": tick_size,
        "min_position": min_position,
        "max_position": max_position,
    }
    for name, value in scalar_parameters.items():
        if np.asarray(value).ndim != 0:
            raise ValueError(f"{name} must be a scalar in this first portfolio slice")
    initial_cash = float(initial_cash)
    taker_fee = float(taker_fee)
    tick_size = float(tick_size)
    min_position = float(min_position)
    max_position = float(max_position)
    if not np.isfinite(initial_cash):
        raise ValueError("initial_cash must be finite")
    if not np.isfinite(taker_fee):
        raise ValueError("taker_fee must be finite")
    if tick_size < 0.0 or not np.isfinite(tick_size):
        raise ValueError("tick_size must be finite and non-negative")
    if min_position > max_position:
        raise ValueError("min_position must not exceed max_position")
    if tick_size != 0.0 and not np.allclose(multiplier, multiplier[0]):
        raise ValueError("non-zero tick_size requires equal contract multipliers")

    # Scaling OHLC values by a contract multiplier is algebraically equivalent
    # to multiplier-aware mark-to-market accounting. The existing C++ column
    # engine can therefore process all instruments in one call.
    scaled_open = arrays["open"] * multiplier[None, :]
    scaled_high = arrays["high"] * multiplier[None, :]
    scaled_low = arrays["low"] * multiplier[None, :]
    scaled_close = arrays["close"] * multiplier[None, :]
    scaled_tick = tick_size * multiplier[0]
    asset_state = np.asarray(
        BacktestOHLCTarget(
            taker_fee=taker_fee,
            tick_size=scaled_tick,
            min_position=min_position,
            max_position=max_position,
        )(
            arrays["targets"],
            scaled_open,
            scaled_high,
            scaled_low,
            scaled_close,
        ),
        dtype=float,
    )
    if asset_state.shape != (periods, assets, 4):
        raise RuntimeError("BacktestOHLCTarget returned an unexpected output shape")

    pnl = np.nan_to_num(asset_state[:, :, 1], nan=0.0)
    cost = np.nan_to_num(asset_state[:, :, 3], nan=0.0)
    position = _carry_forward(asset_state[:, :, 2], initial=0.0)
    mark = _carry_forward(arrays["close"], initial=np.nan)
    notional_position = position * mark * multiplier[None, :]
    dposition = np.diff(
        position,
        axis=0,
        prepend=np.zeros((1, assets), dtype=float),
    )
    period_pnl = np.sum(pnl, axis=1)
    period_cost = np.sum(cost, axis=1)
    equity = initial_cash + np.cumsum(period_pnl)
    gross_exposure = np.nansum(np.abs(notional_position), axis=1)
    net_exposure = np.nansum(notional_position, axis=1)
    margin_used = np.sum(np.abs(position) * margin[None, :], axis=1)
    turnover = np.nansum(
        np.abs(dposition) * np.abs(mark) * multiplier[None, :],
        axis=1,
    )
    trades = np.sum(np.abs(dposition) > 0.0, axis=1, dtype=float)
    drawdown = equity - np.maximum.accumulate(equity)
    portfolio_state = np.column_stack(
        [
            equity,
            period_pnl,
            equity,
            gross_exposure,
            net_exposure,
            margin_used,
            period_cost,
            turnover,
            trades,
            drawdown,
        ]
    )
    if index is not None:
        index = np.asarray(index)
        if index.ndim != 1 or index.shape[0] != periods:
            raise ValueError("index must be one-dimensional with length T")

    return PortfolioResult(
        asset_state=asset_state,
        portfolio_state=portfolio_state,
        summary=_portfolio_summary(portfolio_state, initial_cash),
        index=index,
    )
