# Backtest summary

- Period: 2025-01-01 to 2025-12-31 (365 daily bars, BTCUSDT Binance spot)
- Strategy: long 0.5 BTC when SMA10 crosses above SMA50; flat on cross below
- Starting capital: 100,000 USDT; fees: 0.1% taker per side
- Buy & hold: all capital into BTC at the first close, sold at the last close (fees on both sides)
- Fills: market orders at the close of the signal bar; open position closed at the final bar

| Metric | SMA 10/50 crossover | Buy & hold BTC |
|---|---:|---:|
| Total return | +1.04% | -7.53% |
| Final equity (USDT) | 101,037.54 | 92,474.30 |
| Number of trades | 4 | 1 |
| Win rate | 50.00% | 0.00% |
| Max drawdown | -13.32% | -32.02% |
| Sharpe ratio (daily, ann. 365) | 0.15 | 0.02 |

## Trades

| opened | closed | entry_px | exit_px | quantity | net_pnl_usdt | return |
|---|---|---|---|---|---|---|
| 2025-04-21 | 2025-06-23 | 85,179.2400 | 100,963.8700 | 0.5000 | 7,799.2400 | 0.1853 |
| 2025-07-02 | 2025-08-24 | 105,681.1400 | 115,438.0500 | 0.5000 | 4,767.9000 | 0.0923 |
| 2025-09-18 | 2025-09-27 | 116,447.5900 | 109,643.4600 | 0.5000 | -3,515.1100 | -0.0584 |
| 2025-10-04 | 2025-10-18 | 122,232.0000 | 106,431.6800 | 0.5000 | -8,014.4900 | -0.1293 |
