# BTC SMA Crossover Backtest (NautilusTrader v2)

A small, fully reproducible backtest of a classic trend-following rule on Bitcoin,
built on the [NautilusTrader](https://nautilustrader.io) 2.0 release candidate's
event-driven `BacktestEngine`.

**Rule:** on BTCUSDT daily bars, buy a fixed 0.5 BTC when the 10-day simple moving
average crosses above the 50-day SMA, and sell back to flat when it crosses below.
Long only, no leverage, Binance spot taker fees (0.1% per side) on every fill.

**Data:** calendar year 2025 BTCUSDT 1d klines from Binance's public archive,
[data.binance.vision](https://data.binance.vision). No API keys, no accounts, and
nothing in this repo places live or paper orders.

## Results (2025-01-01 → 2025-12-31)

![Equity curve](reports/equity_curve.png)

| Metric | SMA 10/50 crossover | Buy & hold BTC |
|---|---:|---:|
| Total return | +1.04% | −7.53% |
| Final equity (USDT) | 101,037.54 | 92,474.30 |
| Number of trades (round trips) | 4 | 1 |
| Win rate | 50% | 0% |
| Max drawdown | −13.32% | −32.02% |
| Sharpe ratio (daily returns, annualised ×√365, rf = 0) | 0.15 | 0.02 |

Starting capital is 100,000 USDT for both. Buy & hold puts all of it into BTC at the
first daily close and sells at the last, paying the 0.1% fee on both sides.

| Opened (UTC) | Closed (UTC) | Entry | Exit | Qty (BTC) | Net PnL (USDT) |
|---|---|---:|---:|---:|---:|
| 2025-04-21 | 2025-06-23 | 85,179.24 | 100,963.87 | 0.5 | +7,799.24 |
| 2025-07-02 | 2025-08-24 | 105,681.14 | 115,438.05 | 0.5 | +4,767.90 |
| 2025-09-18 | 2025-09-27 | 116,447.59 | 109,643.46 | 0.5 | −3,515.11 |
| 2025-10-04 | 2025-10-18 | 122,232.00 | 106,431.68 | 0.5 | −8,014.49 |

Full outputs are in [`reports/`](reports): `summary.md`, `metrics.json`, `trades.csv`
and `equity_curve.png`.

### How to read these numbers

- **The comparison isn't like for like on exposure.** The strategy holds a fixed 0.5 BTC,
  roughly 40–60% of the account depending on price. Buy & hold is 100% invested.
  A good part of the smaller drawdown comes from simply holding less BTC, not from
  the timing signal.
- **Four trades is not a sample.** One year of daily bars gives a handful of round
  trips. A win rate of 50% on four trades says almost nothing, and the result swings
  on one or two of them. In 2025 the rule caught the April–August rally, then gave
  back most of it in two whipsaws around the October top.
- **Warmup.** The 50-day SMA needs 50 bars, so the strategy can't act before
  mid-February. Buy & hold is measured over the full year.

## How it works

```
scripts/download_data.py   →  data/raw/BTCUSDT-1d-2025-MM.csv   (SHA256-verified)
src/data_loader.py         →  pandas OHLCV → validated → Nautilus Bar objects
src/strategies/sma_crossover.py  SmaCrossover(Strategy) + SmaCrossoverConfig
src/backtest.py            →  BacktestEngine, simulated Binance spot venue, equity curve
src/metrics.py             →  return, drawdown, Sharpe, win rate, buy-and-hold baseline
scripts/run_backtest.py    →  runs it all, writes reports/
```

Modelling choices:

- **No look-ahead.** Each daily bar is stamped with its *close* time (open + 24h),
  so the strategy only sees a day's close once that day is over.
- **Fills.** Market orders fill at the close of the bar that triggered the signal,
  as a taker, with a `MakerTakerFeeModel(0.001, 0.001)` on the venue. BTC trades
  24/7, so the daily close is effectively the next day's open, but there is
  **no slippage model**.
- **Venue.** Binance spot is simulated as a multi-currency `CASH` account
  (USDT + BTC) with `NETTING` positions. Equity is marked to market every bar as
  `USDT + BTC × close`.
- **End of test.** Any open position is closed on the final bar, so every trade is
  realised in the stats. In 2025 the strategy was already flat by then.
- **Signal rule.** `crossover_signal()` is a pure function: the spread `fast − slow`
  must move from `≤ 0` to `> 0` to enter, or from `≥ 0` to `< 0` to exit. It is unit
  tested separately from the engine.

## How to run

Requires Python 3.12. Commands are shown for Windows PowerShell. On macOS or Linux,
use `source .venv/bin/activate` instead.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt     # pins nautilus-trader==2.0.0rc6
```

```powershell
python scripts/download_data.py     # ~12 small zips from data.binance.vision → data/raw/
```

```powershell
python scripts/run_backtest.py      # prints the table, writes reports/
```

```powershell
python -m pytest -q                 # 39 tests, ~2 s
```

Useful flags for the runner: `--fast 20 --slow 100`, `--trade-size 0.25`, `--capital 50000`,
`--start 2025-03-01 --end 2025-12-31`. The downloader accepts `--year`, `--symbol`,
`--interval` and `--force`.

The data loader can be run directly to re-verify the dataset (365 rows, no gaps,
OHLC sanity):

```powershell
python -m src.data_loader
```

## Tests

- `tests/test_signals.py`: crossover edge cases (touching zero, no prior value),
  known crossovers on hand-built price series, and config validation.
- `tests/test_warmup.py`: no signals or fills before 50 bars, even when the trend
  is obvious from bar 10, and a cross right after warmup *does* trade at that
  bar's close.
- `tests/test_backtest_smoke.py`: a synthetic run reconciles final equity with
  realised PnL and checks fees are charged. An end-to-end run of the script
  produces every report file (skipped if `data/raw/` is empty).

## Project layout

```
src/            strategy, data loader, engine wrapper, metrics
scripts/        download_data.py, run_backtest.py
tests/          pytest suite
data/raw/       downloaded CSVs (git-ignored)
reports/        generated summary, metrics, trades and chart
```

## Notes on NautilusTrader 2.0

The 2.0 RC is a Rust/pyo3 rewrite, and several 1.x patterns no longer apply:

- Fees live on the venue's `fee_model` (required), not on the instrument.
- Reports come from `engine.generate_*_report()`. There is no `engine.trader`.
- A spot cash venue needs `base_currency=None`.
- `StrategyConfig` subclasses must set custom fields in `__init__`.
- Exceptions inside strategy handlers are logged, not raised. The runner and tests
  check `bars_seen` and fill counts so a silent failure can't pass.

## Disclaimer: a backtest is not evidence the rule works

This is one rule, one asset and one year, tested on the same data used to write it.
Even with fees included and look-ahead avoided, the result is a description of what
*would have happened* in 2025, not a prediction.

- The sample is tiny: 4 trades.
- The outcome depends heavily on the regime. Moving-average crossovers do well in long
  trends and lose money in choppy markets.
- Slippage, spread, funding of idle cash, taxes and execution delays are not modelled.
- Any parameter tweak made after looking at this chart would overfit it.

To learn anything durable you would need multiple years and assets, out-of-sample
and walk-forward testing, sensitivity analysis across parameters, and a realistic
execution model. Nothing here is investment advice.
