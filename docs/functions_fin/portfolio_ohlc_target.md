---
name: portfolio_ohlc_target
title: Multi-asset OHLC target backtest
kind: function
short: Causal multi-asset target backtest with fixed-shape contract and portfolio arrays.
topics:
- backtesting
---

# portfolio_ohlc_target

portfolio_ohlc_target applies the existing causal OHLC target engine to several
instrument columns in one vectorized call. A target decided on bar t executes
at bar t+1's open, matching BacktestOHLCTarget.

The result is a tuple-like PortfolioResult:

- asset_state: a (T, A, 4) NumPy array with [equity, pnl, position, cost] on
  the last axis.
- portfolio_state: a (T, 10) NumPy array with columns portfolio_columns.
- summary: scalar PnL, cost, turnover, trade-count, drawdown, and Sharpe values.
- index: optional user-supplied row labels.

multipliers can be a scalar or one value per instrument. It converts price
changes and transaction costs to dollar-denominated contract PnL while keeping
position measured in contracts. margin_per_contract contributes to the reported
margin_used column but does not reserve cash in this first slice.

This API deliberately models independent instrument accounting. Shared cash,
group budgets, partial fills, and deterministic cross-asset execution ordering
are separate follow-up features; the helper does not silently pretend to model
them.

## Signature

portfolio_ohlc_target(targets, open_, high, low, close, *, multipliers=1.0, initial_cash=0.0, margin_per_contract=0.0, taker_fee=0.0, tick_size=0.0, min_position=-inf, max_position=inf, index=None)

All market arrays have shape (T, A); a one-dimensional array is treated as one
asset. The scalar taker_fee, tick_size, and position limits are shared by all
assets in this first slice.

## Example

~~~python
import numpy as np
from screamer import portfolio_ohlc_target

close = np.array([[75.0, 420.0], [75.4, 421.0], [74.8, 418.5]])
targets = np.array([[1.0, -1.0], [1.0, -1.0], [0.0, 0.0]])
open_ = high = low = close

result = portfolio_ohlc_target(
    targets,
    open_,
    high,
    low,
    close,
    multipliers=[1_000.0, 50.0],
    margin_per_contract=[8_000.0, 4_000.0],
    initial_cash=100_000.0,
)

result.asset_state.shape       # (3, 2, 4)
result.portfolio_state.shape  # (3, 10)
result.summary
~~~
