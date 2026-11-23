"""Interactive dashboard for the ETF SMA crossover backtests and research.

Run from the repo root:
    streamlit run app/streamlit_app.py

The "Backtest" tab runs a fresh NautilusTrader backtest on the locally downloaded
Alpaca daily ETF data (data/raw/). The "Research" tab shows the output of
scripts/run_research.py. Nothing here connects to an exchange.
"""

from __future__ import annotations

import json
import os
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

from src.data_loader import DEFAULT_COST_RATE  # noqa: E402
from src.data_loader import DEFAULT_SYMBOLS  # noqa: E402
from src.data_loader import csv_path  # noqa: E402
from src.data_loader import load_bars  # noqa: E402
from src.data_loader import validate_daily  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.strategies.sma_crossover import Signal  # noqa: E402

RAW_DIR = Path(os.environ.get("ETF_RAW_DIR", ROOT / "data" / "raw"))  # override for demos/tests
RESEARCH_DIR = ROOT / "reports" / "research"
ML_DIR = ROOT / "reports" / "ml"
SIZING_DIR = ROOT / "reports" / "ml_sizing"
STRATEGY_COLOR = "#1f6feb"
BNH_COLOR = "#8b949e"
FAST_COLOR = "#d29922"
SLOW_COLOR = "#8250df"
BUY_COLOR = "#1a7f37"
SELL_COLOR = "#cf222e"

st.set_page_config(page_title="SMA Crossover Backtest", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- data
def available_symbols() -> list[str]:
    return [s for s in DEFAULT_SYMBOLS if csv_path(RAW_DIR, s).exists()]


@st.cache_data(show_spinner=False)
def load_data(symbol: str) -> pd.DataFrame:
    df = load_bars(RAW_DIR, symbol)
    validate_daily(df)
    return df[["open", "high", "low", "close", "volume"]]


@st.cache_data(show_spinner=False)
def backtest(
    symbol: str,
    start: date,
    end: date,
    fast: int,
    slow: int,
    trade_size: float,
    position_fraction: float | None,
    capital: float,
    cost_rate: float,
    delay: int,
) -> dict:
    """Run one backtest and return plain, cacheable objects."""
    df = load_data(symbol)
    window = df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
    result = run_pipeline(
        window, fast, slow, trade_size, capital,
        symbol=symbol, position_fraction=position_fraction,
        cost_rate=cost_rate, execution_delay_bars=delay,
    )
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


def signed(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2f}"


def num(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.2f}"


# ---------------------------------------------------------------------------- chart
def build_chart(res: dict, symbol: str, fast: int, slow: int, log_scale: bool) -> go.Figure:
    closes, equity, bnh = res["closes"], res["equity"], res["bnh"]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.45, 0.35, 0.20],
        subplot_titles=("Equity (USD)", f"{symbol} close & SMAs (adjusted)", "Drawdown (%)"),
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
    for kind, symbol_marker, color, label in (
        (Signal.ENTER_LONG.value, "triangle-up", BUY_COLOR, "Buy"),
        (Signal.EXIT_LONG.value, "triangle-down", SELL_COLOR, "Sell"),
    ):
        ts = [t for t, s in res["signals"] if s == kind]
        if ts:
            fig.add_trace(go.Scatter(x=ts, y=[closes.asof(t) for t in ts], mode="markers", name=label,
                                     marker=dict(symbol=symbol_marker, size=11, color=color)), row=2, col=1)

    dd = (equity / equity.cummax() - 1) * 100
    dd_bnh = (bnh / bnh.cummax() - 1) * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Strategy DD", fill="tozeroy",
                             line=dict(color=STRATEGY_COLOR, width=1), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=dd_bnh.index, y=dd_bnh, name="Buy & hold DD",
                             line=dict(color=BNH_COLOR, width=1, dash="dash"), showlegend=False), row=3, col=1)

    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=2, col=1)
    fig.update_layout(height=780, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0))
    return fig


# ------------------------------------------------------------------------- header
st.title("📈 SMA crossover backtest: US ETFs and stocks")
st.caption(
    "Long-only fast/slow SMA crossover on US ETF and stock daily bars (Alpaca, split/dividend-adjusted), simulated "
    "with NautilusTrader 2.0: decide at the close, fill at the next open. Historical simulation only; "
    "no broker connection and no orders are placed.",
)

symbols = available_symbols()
if not symbols:
    st.error(f"No kline data in `{RAW_DIR}`.")
    st.code("python scripts/download_data.py", language="bash")
    st.caption("Needs free Alpaca API keys in a .env file; see .env.example.")
    st.stop()

# ----------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Parameters")
    symbol = st.selectbox("Symbol", symbols, index=0)
    data = load_data(symbol)
    first_day, last_day = data.index[0].date(), data.index[-1].date()
    default_start = max(first_day, date(last_day.year, 1, 1))
    with st.form("params"):
        period = st.date_input("Date range", value=(default_start, last_day),
                               min_value=first_day, max_value=last_day)
        c1, c2 = st.columns(2)
        fast = c1.number_input("Fast SMA", min_value=2, max_value=200, value=10, step=1)
        slow = c2.number_input("Slow SMA", min_value=3, max_value=300, value=50, step=1)
        fraction_pct = st.slider("Cash per entry (%)", 10, 100, 99)
        trade_size, position_fraction = 1.0, fraction_pct / 100
        capital = st.number_input("Starting capital (USD)", min_value=1_000, max_value=10_000_000,
                                  value=100_000, step=10_000)
        cost_bps = st.slider("Cost per side (bps)", 0, 30, int(round(DEFAULT_COST_RATE * 1e4)),
                             help="Commission + slippage, on top of a 1 bp half-spread at the open")
        delay = st.slider("Execution delay (bars)", 0, 5, 0,
                          help="Act on each signal this many sessions late")
        log_scale = st.checkbox("Log-scale price and equity", value=False)
        st.form_submit_button("Run backtest", type="primary", width="stretch")
    st.caption(f"Data: {first_day} → {last_day} ({len(data)} sessions) from Alpaca")

tab_bt, tab_research, tab_ml = st.tabs(["Backtest", "Stage 1 research", "Stage 2 ML"])

# --------------------------------------------------------------------- backtest tab
def render_backtest() -> None:
    """Backtest tab. Returns early (instead of st.stop) so the research tab still renders."""
    if not isinstance(period, (tuple, list)) or len(period) != 2:
        st.info("Pick both a start and an end date.")
        return
    start, end = period
    n_bars = len(data.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")])
    if fast >= slow:
        st.error("Fast SMA must be shorter than slow SMA.")
        return
    if n_bars <= slow:
        st.error(f"The selected range has {n_bars} bars; it needs more than {slow} to warm up the slow SMA.")
        return

    cost_rate = cost_bps / 10_000
    with st.spinner("Running NautilusTrader backtest…"):
        res = backtest(symbol, start, end, int(fast), int(slow), float(trade_size), position_fraction,
                       float(capital), cost_rate, int(delay))

    s, b = res["strategy"], res["bnh_stats"]

    cols = st.columns(5)
    cols[0].metric("Total return", pct(s["total_return"]),
                   delta=f"{(s['total_return'] - b['total_return']) * 100:+.1f} pp vs B&H")
    cols[1].metric("Max drawdown", pct(s["max_drawdown"]),
                   delta=f"{(s['max_drawdown'] - b['max_drawdown']) * 100:+.1f} pp vs B&H")
    cols[2].metric("Sharpe (ann.)", num(s["sharpe_ratio"]),
                   delta=None if math.isnan(s["sharpe_ratio"]) or math.isnan(b["sharpe_ratio"])
                   else f"{s['sharpe_ratio'] - b['sharpe_ratio']:+.2f} vs B&H")
    cols[3].metric("Trades", f"{s['num_trades']}")
    cols[4].metric("Win rate", pct(s["win_rate"], signed=False))

    st.plotly_chart(build_chart(res, symbol, int(fast), int(slow), log_scale), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Trades")
        if res["trades"].empty:
            st.info("No completed trades in this window.")
        else:
            st.dataframe(
                res["trades"], hide_index=True, width="stretch",
                column_config={
                    "opened": "Opened (UTC)",
                    "closed": "Closed (UTC)",
                    "entry_px": st.column_config.NumberColumn("Entry", format="%.4f"),
                    "exit_px": st.column_config.NumberColumn("Exit", format="%.4f"),
                    "quantity": st.column_config.NumberColumn("Qty", format="%.4f"),
                    "net_pnl_usd": st.column_config.NumberColumn("Net PnL (USD)", format="%.2f"),
                    "return": st.column_config.NumberColumn("Return", format="percent"),
                },
            )
            st.download_button("Download trades.csv", res["trades"].to_csv(index=False),
                               file_name=f"{symbol}_trades.csv", mime="text/csv")
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
                index=["Total return", "Final equity (USD)", "Trades", "Win rate", "Max drawdown", "Sharpe"],
            ),
            width="stretch",
        )

    with st.expander("How this is simulated, and why it isn't proof"):
        st.markdown(
            f"""
- **Execution:** each signal is decided on the daily close and filled at the next session's
  open (plus any delay), paying a 1 bp half-spread and {cost_rate * 100:.3f}% per side.
- **No look-ahead:** each daily bar is stamped at its close, so a signal only uses closed days.
- **Sizing:** each entry invests the chosen share of free cash in whole shares.
  Buy & hold invests everything at the first close.
- **Warmup:** no trades until the slow SMA has {int(slow)} bars.
- **Overfitting:** tuning parameters on this screen until the numbers look good is exactly
  what the walk-forward test in the Research tab guards against.
""",
        )


with tab_bt:
    render_backtest()

# --------------------------------------------------------------------- research tab
with tab_research:
    results_path = RESEARCH_DIR / "results.json"
    if not results_path.exists():
        st.info("No research results yet. Generate them with:")
        st.code("python scripts/run_research.py", language="bash")
    else:
        results = {r["symbol"]: r for r in json.loads(results_path.read_text(encoding="utf-8"))}
        st.markdown(
            "Each question below tests whether the SMA rule's edge is real or just a lucky backtest. "
            "Every run decides at the close, fills at the next open (1 bp half-spread) and pays 0.03% per side, "
            "investing 99% of cash per entry.",
        )
        rsym = st.radio("Symbol", list(results), horizontal=True, key="research_symbol")
        r = results[rsym]
        wf = r["walk_forward"]

        st.subheader("1. Does optimising parameters help out of sample?")
        st.caption(
            f"Walk-forward: choose the best SMA pair on 2 years, trade it for the next 6 months, roll forward "
            f"({wf['folds']} folds, {wf['oos_start']} → {wf['oos_end']}).",
        )
        c = st.columns(3)
        for col, (label, key) in zip(c, (("Walk-forward optimised", "walk_forward"),
                                          ("Fixed SMA 10/50", "baseline"),
                                          ("Buy & hold", "buy_and_hold"))):
            col.metric(label, pct(wf[key]["total_return"]),
                       delta=f"Sharpe {num(wf[key]['sharpe_ratio'])}, max DD {pct(wf[key]['max_drawdown'])}",
                       delta_color="off")
        st.image(str(RESEARCH_DIR / f"{rsym}_walk_forward.png"), width="stretch")
        st.caption("Parameters chosen per fold: " + ", ".join(wf["chosen_pairs"]))

        left, right = st.columns(2)
        with left:
            st.subheader("2. Is it one lucky parameter pair?")
            sens = r["sensitivity"]
            st.caption(
                f"{sens['pairs']} pairs: Sharpe {num(sens['sharpe_min'])} to {num(sens['sharpe_max'])} "
                f"(median {num(sens['sharpe_median'])}); {sens['share_positive_return']:.0%} profitable.",
            )
            st.image(str(RESEARCH_DIR / f"{rsym}_sensitivity.png"), width="stretch")
        with right:
            st.subheader("3. Is the timing better than luck?")
            rb = r["random_baseline"]
            st.caption(
                f"10/50 beats {rb['return_percentile']:.0%} of {rb['samples']} random schedules with the same "
                f"trade count and holding lengths.",
            )
            st.image(str(RESEARCH_DIR / f"{rsym}_random_baseline.png"), width="stretch")

        st.subheader("4. Does it survive higher costs and late fills?")
        rob = pd.DataFrame(r["robustness"])
        piv = rob.pivot(index="cost_per_side", columns="delay_bars", values="sharpe_ratio")
        piv.index = [f"{c * 100:.2f}% / side" for c in piv.index]
        piv.columns = [f"delay {d} bar{'s' if d != 1 else ''}" for d in piv.columns]
        st.dataframe(piv.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1.5),
                     width="stretch")
        st.caption("Sharpe ratio of SMA 10/50 over the full period.")


# --------------------------------------------------------------------------- ML tab
with tab_ml:
    ml_path = ML_DIR / "results.json"
    if not ml_path.exists():
        st.info("No ML results yet. Generate them with:")
        st.code("python scripts/run_ml.py", language="bash")
    else:
        ml = json.loads(ml_path.read_text(encoding="utf-8"))
        cfg = ml["settings"]
        st.markdown(
            f"Can a model predict whether the next **{cfg['horizon_days']} days** will be up? "
            f"{len(cfg['features'])} backward-looking features, one model trained across all symbols, "
            f"purged walk-forward (train {cfg['train_days']} days, test {cfg['test_days']}). Every prediction "
            "below is out-of-sample. Trades go through Nautilus at "
            f"{cfg['cost_per_side'] * 100:.2f}% per side, entering at P ≥ {cfg['enter']} and exiting at "
            f"P ≤ {cfg['exit']}.",
        )

        st.subheader("1. Does the model predict direction?")
        names = {"lgbm": "LightGBM", "logreg": "Logistic regression", "shuffled": "Shuffled labels (control)"}
        cols = st.columns(3)
        for col, (key, label) in zip(cols, names.items()):
            c = ml["classification"][key]
            col.metric(f"{label}: AUC", f"{c['pooled_auc']:.3f}",
                       delta=f"{c['pooled_auc'] - 0.5:+.3f} vs coin flip", delta_color="off")
        st.image(str(ML_DIR / "fold_auc.png"), width="stretch")
        left, right = st.columns(2)
        left.image(str(ML_DIR / "calibration.png"), width="stretch")
        right.image(str(ML_DIR / "feature_importance.png"), width="stretch")

        st.subheader("2. Does it make money after costs?")
        msym = st.radio("Symbol", list(ml["symbols"]), horizontal=True, key="ml_symbol")
        r = ml["symbols"][msym]
        labels = {"lgbm": "LightGBM", "logreg": "Logistic regression", "shuffled": "Shuffled control",
                  "sma": "SMA 10/50", "bnh": "Buy & hold"}
        table = pd.DataFrame(
            {
                labels[k]: [pct(v["total_return"]), pct(v["max_drawdown"]), num(v["sharpe_ratio"]),
                            str(v["num_trades"])]
                for k, v in r["stats"].items()
            },
            index=["Return", "Max drawdown", "Sharpe", "Trades"],
        )
        st.caption(f"Out-of-sample: {r['oos_start']} → {r['oos_end']}")
        st.dataframe(table, width="stretch")
        st.image(str(ML_DIR / f"{msym}_equity.png"), width="stretch")
        with st.expander("Threshold sensitivity (LightGBM)"):
            th = pd.DataFrame(r["thresholds"])
            th["total_return"] = th["total_return"].map(pct)
            th["sharpe_ratio"] = th["sharpe_ratio"].map(num)
            th["max_drawdown"] = th["max_drawdown"].map(pct)
            th["time_in_market"] = th["time_in_market"].map(lambda x: f"{x:.0%}")
            st.dataframe(th, hide_index=True, width="stretch")

        st.subheader("3. Can ML at least size the SMA trades?")
        sz_path = SIZING_DIR / "results.json"
        if not sz_path.exists():
            st.info("Run `python scripts/run_ml_sizing.py` to add this section.")
        else:
            sz = json.loads(sz_path.read_text(encoding="utf-8"))
            st.markdown(
                "Same SMA 10/50 entries and exits every time; only the amount bet on each entry changes. "
                f"The model is trained on SMA-long days to predict whether the next {sz['settings']['horizon_days']} "
                f"days are up (out-of-sample AUC {sz['auc']['lgbm']:.3f}).",
            )
            scheme_names = {
                "fixed": "Fixed 99%", "vol_target": "Vol target (no ML)", "ml_size": "ML size",
                "ml_gate": "ML gate", "ml_size_logreg": "ML size (logreg)",
                "ml_size_shuffled": "ML size (shuffled control)", "bnh": "Buy & hold",
            }
            sharpe = pd.DataFrame(
                {sym: {scheme_names[r["scheme"]]: r["sharpe_ratio"] for r in v["stats"]}
                 for sym, v in sz["symbols"].items()},
            )
            sharpe["Average"] = sharpe.mean(axis=1)
            st.dataframe(sharpe.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=0.5, vmax=1.5),
                         width="stretch")
            st.caption("Sharpe ratio, out-of-sample.")
            corr = sz["bet_correlation"]
            st.markdown(
                f"Did bigger bets go to better trades? Rank correlation between bet size and trade return "
                f"over {sz['n_entries']} entries: **ML {signed(corr['ml_size'])}**, shuffled control "
                f"{signed(corr['ml_size_shuffled'])}, vol target {signed(corr['vol_target'])}.",
            )
            st.image(str(SIZING_DIR / "bets_vs_outcomes.png"), width="stretch")
            ssym = st.radio("Symbol", list(sz["symbols"]), horizontal=True, key="sizing_symbol")
            st.image(str(SIZING_DIR / f"{ssym}_equity.png"), width="stretch")

