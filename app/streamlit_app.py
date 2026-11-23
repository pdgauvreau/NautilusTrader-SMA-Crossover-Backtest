"""Interactive dashboard for the BTCUSDT SMA crossover backtest.

Run from the repo root:
    streamlit run app/streamlit_app.py

Every run is a fresh NautilusTrader backtest on the locally downloaded Binance
archive data (data/raw/). Nothing here connects to an exchange.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.data_loader import TAKER_FEE  # noqa: E402
from src.data_loader import load_klines_csv  # noqa: E402
from src.data_loader import validate_daily  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.strategies.sma_crossover import Signal  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
STRATEGY_COLOR = "#1f6feb"
BNH_COLOR = "#8b949e"
FAST_COLOR = "#d29922"
SLOW_COLOR = "#8250df"
BUY_COLOR = "#1a7f37"
SELL_COLOR = "#cf222e"

st.set_page_config(page_title="BTC SMA Crossover Backtest", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = load_klines_csv(RAW_DIR)
    validate_daily(df)
    return df[["open", "high", "low", "close", "volume", "close_time"]]


@st.cache_data(show_spinner=False)
def backtest(
    start: date,
    end: date,
    fast: int,
    slow: int,
    trade_size: float,
    capital: float,
) -> dict:
    """Run one backtest and return plain, cacheable objects."""
    df = load_data()
    window = df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
    result = run_pipeline(window, fast, slow, trade_size, capital)
    return {
        "closes": result.closes,
        "equity": result.equity,
        "bnh": result.buy_and_hold,
        "strategy": result.strategy_stats.as_dict(),
        "bnh_stats": result.buy_and_hold_stats.as_dict(),
        "trades": result.trades,
        "signals": [(pd.Timestamp(ts, tz="UTC"), sig.value) for ts, sig in result.output.signals],
        "bars": len(window),
    }


# ----------------------------------------------------------------------- formatting
def pct(x: float, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:+.2f}%" if signed else f"{x * 100:.1f}%"


def num(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.2f}"


# ---------------------------------------------------------------------------- chart
def build_chart(res: dict, fast: int, slow: int) -> go.Figure:
    closes, equity, bnh = res["closes"], res["equity"], res["bnh"]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.45, 0.35, 0.20],
        subplot_titles=("Equity (USDT)", "BTCUSDT close & SMAs", "Drawdown (%)"),
    )
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name=f"SMA {fast}/{slow} strategy",
                             line=dict(color=STRATEGY_COLOR, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bnh.index, y=bnh, name="Buy & hold",
                             line=dict(color=BNH_COLOR, width=1.5, dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=closes.index, y=closes, name="Close",
                             line=dict(color="#24292f", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=closes.index, y=closes.rolling(fast).mean(), name=f"SMA {fast}",
                             line=dict(color=FAST_COLOR, width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=closes.index, y=closes.rolling(slow).mean(), name=f"SMA {slow}",
                             line=dict(color=SLOW_COLOR, width=1.2)), row=2, col=1)
    for kind, symbol, color, label in (
        (Signal.ENTER_LONG.value, "triangle-up", BUY_COLOR, "Buy"),
        (Signal.EXIT_LONG.value, "triangle-down", SELL_COLOR, "Sell"),
    ):
        ts = [t for t, s in res["signals"] if s == kind]
        if ts:
            fig.add_trace(go.Scatter(x=ts, y=[closes.asof(t) for t in ts], mode="markers", name=label,
                                     marker=dict(symbol=symbol, size=12, color=color)), row=2, col=1)

    dd = (equity / equity.cummax() - 1) * 100
    dd_bnh = (bnh / bnh.cummax() - 1) * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Strategy DD", fill="tozeroy",
                             line=dict(color=STRATEGY_COLOR, width=1), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=dd_bnh.index, y=dd_bnh, name="Buy & hold DD",
                             line=dict(color=BNH_COLOR, width=1, dash="dash"), showlegend=False), row=3, col=1)

    fig.update_layout(height=780, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0))
    return fig


# ------------------------------------------------------------------------------- UI
st.title("📈 BTC SMA crossover backtest")
st.caption(
    "Long-only fast/slow SMA crossover on Binance BTCUSDT daily bars, simulated with "
    "NautilusTrader 2.0 (spot cash account, 0.1% taker fee per side). Historical "
    "simulation only. No exchange connection and no orders are placed.",
)

try:
    data = load_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Couldn't load data from `{RAW_DIR}`: {exc}")
    st.code("python scripts/download_data.py", language="bash")
    st.stop()

first_day, last_day = data.index[0].date(), data.index[-1].date()

with st.sidebar:
    st.header("Parameters")
    with st.form("params"):
        period = st.date_input("Date range", value=(first_day, last_day),
                               min_value=first_day, max_value=last_day)
        c1, c2 = st.columns(2)
        fast = c1.number_input("Fast SMA", min_value=2, max_value=200, value=10, step=1)
        slow = c2.number_input("Slow SMA", min_value=3, max_value=300, value=50, step=1)
        trade_size = st.number_input("Trade size (BTC)", min_value=0.01, max_value=5.0,
                                     value=0.5, step=0.05, format="%.2f")
        capital = st.number_input("Starting capital (USDT)", min_value=1_000, max_value=10_000_000,
                                  value=100_000, step=10_000)
        st.form_submit_button("Run backtest", type="primary", use_container_width=True)
    st.caption(f"Data: {first_day} → {last_day} ({len(data)} daily bars) from data.binance.vision")

if not isinstance(period, (tuple, list)) or len(period) != 2:
    st.info("Pick both a start and an end date.")
    st.stop()
start, end = period
n_bars = len(data.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")])
if fast >= slow:
    st.error("Fast SMA must be shorter than slow SMA.")
    st.stop()
if n_bars <= slow:
    st.error(f"The selected range has {n_bars} bars; it needs more than {slow} to warm up the slow SMA.")
    st.stop()

with st.spinner("Running NautilusTrader backtest…"):
    res = backtest(start, end, int(fast), int(slow), float(trade_size), float(capital))

s, b = res["strategy"], res["bnh_stats"]
max_notional = float(trade_size) * float(res["closes"].max())
if max_notional > capital:
    st.warning(
        f"{trade_size} BTC can cost up to {max_notional:,.0f} USDT in this period, more than the "
        f"starting capital. Entries the account can't afford are denied by Nautilus's risk engine "
        f"(the signal is skipped).",
    )

cols = st.columns(5)
cols[0].metric("Total return", pct(s["total_return"]),
               delta=f"{(s['total_return'] - b['total_return']) * 100:+.2f} pp vs B&H")
cols[1].metric("Max drawdown", pct(s["max_drawdown"]),
               delta=f"{(s['max_drawdown'] - b['max_drawdown']) * 100:+.2f} pp vs B&H")
cols[2].metric("Sharpe (ann.)", num(s["sharpe_ratio"]),
               delta=None if math.isnan(s["sharpe_ratio"]) or math.isnan(b["sharpe_ratio"])
               else f"{s['sharpe_ratio'] - b['sharpe_ratio']:+.2f} vs B&H")
cols[3].metric("Trades", f"{s['num_trades']}")
cols[4].metric("Win rate", pct(s["win_rate"], signed=False))

st.plotly_chart(build_chart(res, int(fast), int(slow)), use_container_width=True)

left, right = st.columns([3, 2])
with left:
    st.subheader("Trades")
    if res["trades"].empty:
        st.info("No completed trades in this window.")
    else:
        st.dataframe(
            res["trades"], hide_index=True, use_container_width=True,
            column_config={
                "opened": "Opened (UTC)",
                "closed": "Closed (UTC)",
                "entry_px": st.column_config.NumberColumn("Entry", format="%.2f"),
                "exit_px": st.column_config.NumberColumn("Exit", format="%.2f"),
                "quantity": st.column_config.NumberColumn("Qty (BTC)", format="%.4f"),
                "net_pnl_usdt": st.column_config.NumberColumn("Net PnL (USDT)", format="%.2f"),
                "return": st.column_config.NumberColumn("Return", format="percent"),
            },
        )
        st.download_button("Download trades.csv", res["trades"].to_csv(index=False),
                           file_name="trades.csv", mime="text/csv")
with right:
    st.subheader("Strategy vs buy & hold")
    st.dataframe(
        pd.DataFrame(
            {
                "Strategy": [pct(s["total_return"]), f"{s['final_equity']:,.2f}", str(s["num_trades"]),
                             pct(s["win_rate"], False), pct(s["max_drawdown"]), num(s["sharpe_ratio"])],
                "Buy & hold": [pct(b["total_return"]), f"{b['final_equity']:,.2f}", str(b["num_trades"]),
                               pct(b["win_rate"], False), pct(b["max_drawdown"]), num(b["sharpe_ratio"])],
            },
            index=["Total return", "Final equity (USDT)", "Trades", "Win rate", "Max drawdown", "Sharpe"],
        ),
        use_container_width=True,
    )

with st.expander("How this is simulated, and why it isn't proof"):
    st.markdown(
        f"""
- **Fills:** market orders fill at the close of the signal bar as a taker, with a
  {float(TAKER_FEE) * 100:.1f}% fee per side. No slippage is modelled.
- **No look-ahead:** each daily bar is stamped at its close, so a signal only uses closed days.
- **Exposure:** the strategy trades a fixed BTC quantity. Buy & hold invests all capital,
  so the drawdown comparison partly reflects position size, not timing.
- **Warmup:** no trades until the slow SMA has {int(slow)} bars.
- **Small sample:** {s['num_trades']} trades over {res['bars']} days says little about the
  future. Tuning parameters on this screen until the numbers look good is overfitting.
""",
    )
