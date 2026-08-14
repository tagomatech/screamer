"""Helpers for the corn-futures indicator study notebook.

This module deliberately stays outside Screamer's public API. Bloomberg data is
licensed and must remain local; the notebook therefore accepts Bloomberg,
CSV/Parquet, or an explicitly-labelled deterministic demo fixture.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BLOOMBERG_FIELDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "VOLUME", "OPEN_INT"]
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "open_interest"]


def _flatten_columns(columns) -> list[str]:
    """Return field-like names from Bloomberg's optional MultiIndex columns."""
    if isinstance(columns, pd.MultiIndex):
        return [str(item[-1]) for item in columns]
    return [str(item) for item in columns]


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Bloomberg or local OHLCV data to a strict datetime-indexed frame."""
    data = frame.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        # xbbg may return a plain Index of datetime.date objects, while CSV
        # exports often carry an explicit date column. Accept both forms.
        parsed_index = pd.to_datetime(data.index, errors="coerce")
        if not parsed_index.isna().any():
            data.index = parsed_index
        else:
            timestamp = next((column for column in data.columns if str(column).lower() in {"date", "datetime", "timestamp", "time"}), None)
            if timestamp is None:
                raise ValueError("OHLCV data needs a DatetimeIndex or date/timestamp column")
            data = data.set_index(timestamp)

    data.columns = _flatten_columns(data.columns)
    aliases = {
        "PX_OPEN": "open", "OPEN": "open", "O": "open",
        "PX_HIGH": "high", "HIGH": "high", "H": "high",
        "PX_LOW": "low", "LOW": "low", "L": "low",
        "PX_LAST": "close", "LAST_PRICE": "close", "CLOSE": "close", "C": "close",
        "VOLUME": "volume", "VOL": "volume",
        "OPEN_INT": "open_interest", "OPEN_INTEREST": "open_interest", "OI": "open_interest",
    }
    data = data.rename(columns={column: aliases.get(str(column).upper(), str(column).lower()) for column in data.columns})
    data = data.loc[:, ~data.columns.duplicated(keep="last")]
    if "close" not in data:
        raise ValueError("OHLCV data must contain PX_LAST/close")
    data["open"] = data.get("open", data["close"])
    data["high"] = data.get("high", data[["open", "close"]].max(axis=1))
    data["low"] = data.get("low", data[["open", "close"]].min(axis=1))
    data["volume"] = data.get("volume", np.nan)
    data["open_interest"] = data.get("open_interest", np.nan)
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_convert("UTC").tz_localize(None)
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    for column in OHLCV_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["close"])
    data["volume"] = data["volume"].fillna(0.0).clip(lower=0.0)
    return data[OHLCV_COLUMNS]


def load_local(path: str | Path) -> pd.DataFrame:
    """Load a user-exported CSV or Parquet file and normalize its columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return normalize_ohlcv(pd.read_parquet(path))
    return normalize_ohlcv(pd.read_csv(path))


def load_bloomberg_daily(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Read daily Bloomberg history through the locally-running Terminal API."""
    try:
        from xbbg import blp
    except ImportError as exc:  # pragma: no cover - depends on local terminal setup
        raise ImportError("Install xbbg/blpapi in the notebook environment") from exc
    raw = blp.bdh(tickers=ticker, flds=BLOOMBERG_FIELDS, start_date=start_date, end_date=end_date, Per="DAILY")
    if raw is None or raw.empty:
        raise ValueError(f"Bloomberg returned no daily rows for {ticker!r}")
    return normalize_ohlcv(raw)


def make_demo_data(start: str = "2018-01-01", periods: int = 1_900, seed: int = 7) -> pd.DataFrame:
    """Create a deterministic corn-like OHLCV fixture for notebook smoke tests only."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=periods)
    t = np.arange(periods)
    seasonal = 0.035 * np.sin(2 * np.pi * t / 252) + 0.018 * np.sin(2 * np.pi * t / 126)
    regime = np.where((t // 190) % 2 == 0, 0.006, -0.003)
    shocks = rng.normal(0, 0.018, periods) * (1 + 0.7 * ((t // 80) % 3 == 2))
    log_close = np.log(430.0) + np.cumsum(seasonal / 20 + regime / 20 + shocks)
    close = np.exp(log_close)
    open_ = close * np.exp(rng.normal(0, 0.006, periods))
    intraday = np.maximum(np.abs(rng.normal(0.012, 0.006, periods)), 0.002)
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday * (0.85 + rng.random(periods) * 0.3))
    harvest = 1 + 0.55 * np.maximum(0, np.cos(2 * np.pi * (t - 185) / 252))
    volume = np.maximum(1.0, rng.lognormal(np.log(115_000), 0.22, periods) * harvest)
    open_interest = np.maximum(1.0, 1_350_000 + np.cumsum(rng.normal(0, 4_500, periods)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume, "open_interest": open_interest}, index=index)


def load_dataset(source: str, ticker: str, start_date: str, end_date: str, local_path: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    """Load data and return ``(frame, provenance_label)``.

    ``source='auto'`` tries Bloomberg, then a local file, then the clearly
    labelled demo fixture. For an actual research run, use ``source='bloomberg'``
    or ``source='local'`` so a missing data source fails loudly.
    """
    if source == "demo":
        return make_demo_data(start_date), "DEMO FIXTURE — not market data"
    if source == "local":
        if local_path is None:
            raise ValueError("local_path is required when source='local'")
        return load_local(local_path), f"LOCAL FILE — {Path(local_path).name}"
    if source == "bloomberg":
        return load_bloomberg_daily(ticker, start_date, end_date), f"BLOOMBERG — {ticker}"
    if source != "auto":
        raise ValueError("source must be one of: auto, bloomberg, local, demo")
    try:
        return load_bloomberg_daily(ticker, start_date, end_date), f"BLOOMBERG — {ticker}"
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"Bloomberg unavailable ({exc}); trying local/demo data", RuntimeWarning)
    if local_path is not None and Path(local_path).exists():
        return load_local(local_path), f"LOCAL FILE — {Path(local_path).name}"
    warnings.warn("No local data supplied; using the deterministic demo fixture", RuntimeWarning)
    return make_demo_data(start_date), "DEMO FIXTURE — not market data"


def resample_ohlcv(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Resample local/intraday data without look-ahead or volume interpolation."""
    data = normalize_ohlcv(data)
    if frequency in {"1D", "D", "daily"}:
        return data
    if frequency not in {"1h", "60min"}:
        raise ValueError("frequency must be daily/'1D' or '1h'/'60min'")
    return data.resample("1h", label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "open_interest": "last"}).dropna(subset=["close"])


def _array(operator, *values) -> np.ndarray:
    return np.asarray(operator(*[np.asarray(value, dtype=float) for value in values]), dtype=float)


def compute_features(data: pd.DataFrame, window: int = 20, long_window: int = 60) -> pd.DataFrame:
    """Compute OHLCV indicators and causal forward-return labels with Screamer."""
    from screamer import (ADX, ADOSC, ATR, KAMA, MFI, Momentum, NATR, OBV, ROC, RollingGarmanKlassVol, RollingPoly1, RollingRSI, RollingRogersSatchellVol, RollingTSF, RollingVWAP, RollingYangZhangVol, RollingZscore, TRIX)
    frame = normalize_ohlcv(data).copy()
    open_, high, low, close, volume = (frame[name].to_numpy(float) for name in ["open", "high", "low", "close", "volume"])
    log_close = np.log(close)
    frame["log_return"] = pd.Series(np.diff(log_close, prepend=np.nan), index=frame.index)
    frame["low_lag"] = _array(RollingPoly1(long_window, 0), close)
    frame["trend_slope_pct"] = 100.0 * _array(RollingPoly1(long_window, 1), close) / close
    frame["trend_forecast"] = _array(RollingTSF(long_window), close)
    frame["kama"] = _array(KAMA(long_window), close)
    frame["vwap"] = _array(RollingVWAP(window), high, low, close, volume)
    frame["vwap_distance_bps"] = 10_000.0 * (close / frame["vwap"].to_numpy(float) - 1.0)
    frame["atr"] = _array(ATR(window), high, low, close)
    frame["natr_pct"] = _array(NATR(window), high, low, close)
    frame["gk_vol"] = _array(RollingGarmanKlassVol(window), open_, high, low, close)
    frame["rs_vol"] = _array(RollingRogersSatchellVol(window), open_, high, low, close)
    frame["yz_vol"] = _array(RollingYangZhangVol(window), open_, high, low, close)
    adx = _array(ADX(14), high, low, close)
    frame["plus_di"], frame["minus_di"], frame["adx"] = adx[:, 0], adx[:, 1], adx[:, 2]
    frame["rsi"] = _array(RollingRSI(14), close)
    frame["roc"] = _array(ROC(window), close)
    frame["momentum"] = _array(Momentum(window), close)
    frame["trix"] = _array(TRIX(window), close)
    frame["zscore"] = _array(RollingZscore(long_window), close)
    frame["obv"] = _array(OBV(), close, volume)
    frame["mfi"] = _array(MFI(14), high, low, close, volume)
    frame["adosc"] = _array(ADOSC(3, 10), high, low, close, volume)
    for horizon in (1, 5, 20):
        frame[f"forward_{horizon}d"] = close[np.minimum(np.arange(len(close)) + horizon, len(close) - 1)] / close - 1.0
        frame.loc[frame.index[-horizon:], f"forward_{horizon}d"] = np.nan
    return frame


SIGNAL_COLUMNS = {"Trend slope": "trend_slope_pct", "Z-score": "zscore", "RSI − 50": "rsi", "VWAP distance": "vwap_distance_bps", "ADX": "adx", "NATR": "natr_pct"}


INDICATOR_GUIDE = [
    {"key": "low_lag", "group": "Trend", "label": "Low-lag level", "operator": "RollingPoly1(60, 0)", "formula": "Right-edge value of a trailing linear fit.", "corn_test": "Does it turn earlier than a plain moving average without producing false reversals around limit moves?"},
    {"key": "trend_forecast", "group": "Trend", "label": "Trend forecast", "operator": "RollingTSF(60)", "formula": "Trailing linear-fit forecast at the current bar.", "corn_test": "Does the forecast-minus-price gap predict continuation over 1–5 sessions after weather or report shocks?"},
    {"key": "trend_slope_pct", "group": "Trend", "label": "Normalized trend slope", "operator": "RollingPoly1(60, 1) / close", "formula": "Linear-fit slope divided by price, expressed in percent per bar.", "corn_test": "Is the sign persistent across planting, pollination, and harvest regimes, or only a delayed price trend?"},
    {"key": "kama", "group": "Trend", "label": "Adaptive trend", "operator": "KAMA(60)", "formula": "Kaufman adaptive average; smoothing increases with directional efficiency.", "corn_test": "Does it stay smooth in choppy summer trade yet follow genuine crop-supply repricing faster?"},
    {"key": "vwap", "group": "Trend / value", "label": "Rolling VWAP", "operator": "RollingVWAP(20)", "formula": "Volume-weighted typical price over the trailing window.", "corn_test": "Do large deviations mean-revert, and does that relationship survive roll and delivery-month changes?"},
    {"key": "atr", "group": "Volatility", "label": "Average true range", "operator": "ATR(20)", "formula": "Wilder-smoothed max of range and gaps.", "corn_test": "Does ATR forecast the next session’s range well enough to improve stops and position sizing?"},
    {"key": "natr_pct", "group": "Volatility", "label": "Normalized ATR", "operator": "NATR(20)", "formula": "100 × ATR / close.", "corn_test": "Does it identify high-risk report/weather regimes consistently across price levels?"},
    {"key": "gk_vol", "group": "Volatility", "label": "Garman–Klass volatility", "operator": "RollingGarmanKlassVol(20)", "formula": "OHLC range estimator using log(H/L) and log(C/O).", "corn_test": "Is it more efficient than close-to-close volatility for corn’s intraday range, or distorted by overnight gaps?"},
    {"key": "rs_vol", "group": "Volatility", "label": "Rogers–Satchell volatility", "operator": "RollingRogersSatchellVol(20)", "formula": "Drift-robust OHLC range estimator.", "corn_test": "Does it remain informative during directional rallies and selloffs where Parkinson/GK can misread drift?"},
    {"key": "yz_vol", "group": "Volatility", "label": "Yang–Zhang volatility", "operator": "RollingYangZhangVol(20)", "formula": "Combines overnight, open-to-close, and Rogers–Satchell components.", "corn_test": "Does its treatment of overnight gaps help around USDA releases and the reopen after Globex breaks?"},
    {"key": "rsi", "group": "Momentum", "label": "RSI", "operator": "RollingRSI(14)", "formula": "Wilder-smoothed ratio of recent gains to losses.", "corn_test": "Are extreme readings mean-reverting in quiet ranges but continuation signals in supply shocks?"},
    {"key": "roc", "group": "Momentum", "label": "Rate of change", "operator": "ROC(20)", "formula": "Percentage price change over 20 bars.", "corn_test": "Does the magnitude separate sustained repricing from one-day report gaps?"},
    {"key": "momentum", "group": "Momentum", "label": "Momentum", "operator": "Momentum(20)", "formula": "Current close minus the close 20 bars ago.", "corn_test": "Does the absolute-dollar signal remain comparable across contract price levels, or should it be normalized?"},
    {"key": "trix", "group": "Momentum", "label": "TRIX", "operator": "TRIX(20)", "formula": "Rate of change of a triple-smoothed price.", "corn_test": "Does the extra smoothing improve multi-session signal quality enough to justify its delay?"},
    {"key": "zscore", "group": "Momentum / value", "label": "Rolling z-score", "operator": "RollingZscore(60)", "formula": "Distance from rolling mean in rolling standard deviations.", "corn_test": "Do extreme deviations mean-revert only inside a stable seasonal regime, or across all corn regimes?"},
    {"key": "plus_di", "group": "Trend strength", "label": "+DI", "operator": "ADX(14)[0]", "formula": "Smoothed positive directional movement scaled by true range.", "corn_test": "Does +DI lead sustained upward moves, or merely confirm bars that already moved?"},
    {"key": "minus_di", "group": "Trend strength", "label": "−DI", "operator": "ADX(14)[1]", "formula": "Smoothed negative directional movement scaled by true range.", "corn_test": "Does −DI identify persistent downside supply shocks rather than ordinary two-sided noise?"},
    {"key": "adx", "group": "Trend strength", "label": "ADX", "operator": "ADX(14)[2]", "formula": "Smoothed directional-index separation; strength, not direction.", "corn_test": "Does a high ADX tell us when trend-following beats mean-reversion, conditional on direction?"},
    {"key": "obv", "group": "Volume", "label": "On-balance volume", "operator": "OBV()", "formula": "Cumulative volume signed by close-to-close direction.", "corn_test": "Does volume confirmation distinguish genuine participation from price moves on thin holiday trade?"},
    {"key": "mfi", "group": "Volume", "label": "Money flow index", "operator": "MFI(14)", "formula": "RSI-like oscillator using typical price and volume.", "corn_test": "Do extreme money-flow readings anticipate exhaustion around crop-report repricing?"},
    {"key": "adosc", "group": "Volume", "label": "Accumulation/distribution oscillator", "operator": "ADOSC(3, 10)", "formula": "Fast-minus-slow average of close-location-value money flow.", "corn_test": "Does intrabar close location plus volume add information beyond the close and volume separately?"},
    {"key": "open_interest", "group": "Futures context", "label": "Open interest", "operator": "Bloomberg field OPEN_INT", "formula": "Contracts outstanding; not a Screamer indicator.", "corn_test": "Does price-plus-open-interest classification improve regime labels around rolls and new-crop positioning?"},
]


def indicator_guide_frame() -> pd.DataFrame:
    """Return the study's formula and corn-specific diagnostic guide."""
    return pd.DataFrame(INDICATOR_GUIDE).set_index("key")


def indicator_figure(frame: pd.DataFrame, key: str) -> go.Figure:
    """Plot one indicator in its own price-linked panel."""
    spec = next(item for item in INDICATOR_GUIDE if item["key"] == key)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.48, 0.52], subplot_titles=("Corn price (rebased to 100)", f"{spec['label']} · {spec['operator']}"))
    price = 100.0 * frame["close"] / frame["close"].dropna().iloc[0]
    fig.add_trace(go.Scatter(x=frame.index, y=price, name="price", line={"color": "#264653", "width": 1.5}), row=1, col=1)
    if key == "adx":
        for column, label, color in [("plus_di", "+DI", "#2a9d8f"), ("minus_di", "−DI", "#d1495b"), ("adx", "ADX", "#264653")]:
            fig.add_trace(go.Scatter(x=frame.index, y=frame[column], name=label, line={"color": color, "width": 1.5}), row=2, col=1)
    elif key == "open_interest":
        fig.add_trace(go.Scatter(x=frame.index, y=frame[key], name=spec["label"], line={"color": "#f4a261", "width": 1.5}), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[key], name=spec["label"], line={"color": "#7b2cbf", "width": 1.6}), row=2, col=1)
    fig.add_hline(y=0, line_color="#999", line_width=1, row=2, col=1)
    fig.update_yaxes(title_text="rebased price", row=1, col=1)
    fig.update_yaxes(title_text="value", row=2, col=1)
    # Plotly's native x-axis spike is limited to the subplot under the
    # pointer. A transparent, full-height trace provides one hover target for
    # both panels and its x-axis spike consequently spans both y-domains.
    hover_series = [(price, "price", ".2f")]
    if key == "adx":
        hover_series.extend(
            [
                (frame["plus_di"], "+DI", ".3f"),
                (frame["minus_di"], "−DI", ".3f"),
                (frame["adx"], "ADX", ".3f"),
            ]
        )
    else:
        hover_series.append((frame[key], spec["label"], ".3f" if key != "open_interest" else ".0f"))
    customdata = np.column_stack([series.to_numpy(dtype=float) for series, _, _ in hover_series])
    date_format = "%{x|%Y-%m-%d}" if isinstance(frame.index, pd.DatetimeIndex) else "%{x}"
    hover_lines = [f"{label}: %{{customdata[{column}]:{format_spec}}}" for column, (_, label, format_spec) in enumerate(hover_series)]
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=np.full(len(frame), 0.5),
            xaxis="x3",
            yaxis="y3",
            mode="markers",
            marker={"size": 20, "color": "rgba(0,0,0,0)"},
            customdata=customdata,
            hovertemplate="<b>" + date_format + "</b><br>" + "<br>".join(hover_lines) + "<extra></extra>",
            name="_crosshair",
            showlegend=False,
        )
    )
    fig.update_xaxes(showspikes=False)
    fig.update_layout(
        title=f"{spec['group']} · {spec['label']}",
        template="plotly_white",
        height=560,
        hovermode="x unified",
        spikedistance=-1,
        xaxis3={
            "domain": [0, 1],
            "anchor": "y3",
            "matches": "x2",
            "layer": "below traces",
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikedash": "dot",
            "spikethickness": 1,
            "spikecolor": "#777",
        },
        yaxis3={
            "domain": [0, 1],
            "layer": "below traces",
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "showline": False,
        },
        legend={"orientation": "h", "y": 1.03, "x": 0},
        margin={"l": 55, "r": 35, "t": 95, "b": 40},
        xaxis_rangeslider_visible=False,
    )
    return fig


def decile_table(frame: pd.DataFrame, horizon: int = 5, buckets: int = 5) -> pd.DataFrame:
    """Return mean forward returns by indicator bucket for visual diagnostics."""
    rows = []
    target = f"forward_{horizon}d"
    for label, column in SIGNAL_COLUMNS.items():
        valid = frame[[column, target]].dropna()
        if len(valid) < buckets * 4:
            continue
        valid = valid.assign(bucket=pd.qcut(valid[column], buckets, labels=False, duplicates="drop") + 1)
        grouped = valid.groupby("bucket", observed=True)[target].agg(["mean", "count"]).reset_index()
        grouped["indicator"] = label
        grouped["mean_return_pct"] = grouped["mean"] * 100
        rows.append(grouped[["indicator", "bucket", "mean_return_pct", "count"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def signal_summary(frame: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Summarize monotonicity and predictive spread without fitting a model."""
    target = f"forward_{horizon}d"
    rows = []
    for label, column in SIGNAL_COLUMNS.items():
        valid = frame[[column, target]].dropna()
        if len(valid) < 40:
            continue
        ranked = valid.assign(bucket=pd.qcut(valid[column], 5, labels=False, duplicates="drop"))
        means = ranked.groupby("bucket", observed=True)[target].mean()
        rows.append({"indicator": label, "observations": len(valid), "rank_corr": valid[column].corr(valid[target], method="spearman"), "low_bucket_%": means.iloc[0] * 100, "high_bucket_%": means.iloc[-1] * 100, "high_minus_low_%": (means.iloc[-1] - means.iloc[0]) * 100})
    return pd.DataFrame(rows).set_index("indicator").sort_values("high_minus_low_%", ascending=False)


def causal_backtests(frame: pd.DataFrame, cost_bps: float = 1.5) -> pd.DataFrame:
    """Compare simple next-bar strategies; signals are shifted before returns."""
    raw = frame["log_return"].fillna(0.0)
    signals = {"trend-follow": np.sign(frame["trend_slope_pct"]), "zscore-revert": -np.where(np.abs(frame["zscore"]) >= 1.0, np.sign(frame["zscore"]), 0.0), "rsi-revert": np.select([frame["rsi"] < 30, frame["rsi"] > 70], [1.0, -1.0], default=0.0), "vwap-revert": -np.where(np.abs(frame["vwap_distance_bps"]) >= 20, np.sign(frame["vwap_distance_bps"]), 0.0)}
    curves = []
    for name, signal in signals.items():
        position = pd.Series(signal, index=frame.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).shift(1).fillna(0.0)
        turnover = position.diff().abs().fillna(position.abs())
        pnl = position * raw - turnover * cost_bps / 10_000.0
        equity = np.exp(pnl.cumsum())
        drawdown = equity / equity.cummax() - 1.0
        curves.append(pd.DataFrame({"strategy": name, "position": position, "pnl": pnl, "equity": equity, "drawdown": drawdown}))
    return pd.concat(curves, axis=0)


def dashboard_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    """Interactive price, trend, volatility, momentum, and volume dashboard."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.42, 0.20, 0.20, 0.18], specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]], subplot_titles=("Price, trend estimate, and rolling VWAP", "Volatility regime", "Momentum and trend strength", "Volume and open interest"))
    fig.add_trace(go.Candlestick(x=frame.index, open=frame.open, high=frame.high, low=frame.low, close=frame.close, name="OHLC", increasing_line_color="#168aad", decreasing_line_color="#d1495b"), row=1, col=1)
    for column, name, color in [("trend_forecast", "TSF", "#f4a261"), ("kama", "KAMA", "#2a9d8f"), ("vwap", "rolling VWAP", "#7b2cbf")]:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[column], name=name, line={"color": color, "width": 1.5}), row=1, col=1)
    for column, name, color in [("natr_pct", "NATR %", "#e76f51"), ("gk_vol", "GK vol", "#457b9d"), ("rs_vol", "RS vol", "#2a9d8f"), ("yz_vol", "YZ vol", "#6a4c93")]:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[column], name=name, line={"color": color, "width": 1.3}), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame.rsi, name="RSI", line={"color": "#e9c46a"}), row=3, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame.adx, name="ADX", line={"color": "#264653"}), row=3, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame.zscore, name="z-score", line={"color": "#f4a261"}), row=3, col=1)
    fig.add_trace(go.Bar(x=frame.index, y=frame.volume, name="volume", marker_color="#8ecae6", opacity=0.65), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=frame.index, y=frame.open_interest, name="open interest", line={"color": "#ffb703", "width": 1.4}), row=4, col=1, secondary_y=True)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="volatility", row=2, col=1)
    fig.update_yaxes(title_text="indicator", row=3, col=1)
    fig.update_yaxes(title_text="volume", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="open interest", row=4, col=1, secondary_y=True)
    fig.update_layout(title=title, template="plotly_white", height=1_050, hovermode="x unified", legend={"orientation": "h", "y": 1.02, "x": 0}, margin={"l": 55, "r": 35, "t": 90, "b": 40}, xaxis_rangeslider_visible=False)
    return fig


def decile_figure(table: pd.DataFrame, horizon: int) -> go.Figure:
    """Show whether indicator buckets separate subsequent returns."""
    fig = go.Figure()
    for indicator in table["indicator"].unique():
        subset = table[table.indicator == indicator]
        fig.add_trace(go.Scatter(x=subset.bucket, y=subset.mean_return_pct, mode="lines+markers", name=indicator, customdata=subset[["count"]], hovertemplate="bucket %{x}<br>mean forward return %{y:.2f}%<br>n=%{customdata[0]}<extra>%{fullData.name}</extra>"))
    fig.add_hline(y=0, line_color="#777", line_width=1)
    fig.update_layout(title=f"Mean forward return by indicator bucket ({horizon}-bar horizon)", xaxis_title="low → high indicator bucket", yaxis_title="mean forward return (%)", template="plotly_white", height=520, hovermode="x unified")
    return fig


def equity_figure(curves: pd.DataFrame) -> go.Figure:
    """Plot causal strategy equity and drawdown, with no-look-ahead labels."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.07, subplot_titles=("Causal equity curves (net of illustrative costs)", "Drawdown"))
    for name, subset in curves.groupby("strategy"):
        fig.add_trace(go.Scatter(x=subset.index, y=subset.equity, name=name, mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=subset.index, y=subset.drawdown * 100, name=name + " drawdown", mode="lines", showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="growth of $1", row=1, col=1)
    fig.update_yaxes(title_text="drawdown (%)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=680, hovermode="x unified", legend={"orientation": "h", "y": 1.03, "x": 0})
    return fig
