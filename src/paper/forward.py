"""Daily forward test (paper trading without a broker).

Each run replays the *same* Nautilus SMA strategy over the full price history,
but only allows trading from a fixed start session onward (indicators still
warm up on earlier data). The result is what the strategy would have done
since the forward test began:

* current position and equity per ETF,
* every trade since the start (filled at the next open, with costs),
* the action due at the **next open**, if the latest close produced a signal.

Because each run recomputes from the start date, the forward test survives
restarts, missed days and machine reboots: nothing is held in memory. No
orders are sent anywhere.

Note: prices are dividend-adjusted, so after each dividend Alpaca rescales
older prices slightly. Historical equity values in the forward test can shift
by small amounts as a result; decisions are unaffected because SMA crossovers
are scale-invariant.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pandas as pd

from src.data_loader import DEFAULT_COST_RATE
from src.data_loader import bar_close_times
from src.pipeline import run_pipeline


@dataclass(frozen=True)
class ForwardState:
    start_session: str  # YYYY-MM-DD: trading may begin at this session's close
    symbols: tuple[str, ...]
    fast: int = 10
    slow: int = 50
    capital_per_symbol: float = 10_000.0
    cost_rate: float = DEFAULT_COST_RATE
    created: str = ""
    # Per-symbol start sessions for symbols added after the forward test began.
    starts: dict[str, str] = field(default_factory=dict)

    def start_for(self, symbol: str) -> str:
        return self.starts.get(symbol, self.start_session)

    def with_symbols(self, symbols, start_session: dict[str, str]) -> ForwardState:
        """A copy that also tracks ``symbols`` (new ones start at their given session)."""
        new = [s for s in symbols if s not in self.symbols]
        starts = dict(self.starts) | {s: start_session[s] for s in new}
        return replace(self, symbols=tuple(self.symbols) + tuple(new), starts=starts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self) | {"symbols": list(self.symbols)}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ForwardState:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["symbols"] = tuple(data["symbols"])
        data.setdefault("starts", {})
        return cls(**data)

    @classmethod
    def new(cls, start_session: str, symbols, **kwargs) -> ForwardState:
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return cls(start_session=start_session, symbols=tuple(symbols), created=created, **kwargs)


@dataclass
class SymbolStatus:
    symbol: str
    last_session: str
    close: float
    sma_fast: float
    sma_slow: float
    trend: str  # "UP" if fast > slow
    shares: float
    equity: float
    return_since_start: float
    trades_since_start: int
    next_open_action: str  # "BUY", "SELL" or "HOLD"
    fills: list[dict]


def forward_status(df: pd.DataFrame, symbol: str, state: ForwardState) -> SymbolStatus:
    """Replay the strategy on ``df`` with trading allowed from the symbol's start session."""
    sessions = df.index.strftime("%Y-%m-%d")
    wanted = state.start_for(symbol)
    if wanted not in set(sessions):
        later = [s for s in sessions if s >= wanted]
        if not later:
            raise ValueError(f"{symbol}: no data on or after the start session {wanted}")
        start_session = later[0]
    else:
        start_session = wanted
    start_close = bar_close_times(df)[list(sessions).index(start_session)]

    res = run_pipeline(
        df, state.fast, state.slow, capital=state.capital_per_symbol, symbol=symbol,
        position_fraction=0.99, cost_rate=state.cost_rate, trade_start=start_close,
        enter_if_above_on_start=True, close_on_stop=False,
    )
    fills = res.output.fills
    shares = 0.0
    fill_rows = []
    if not fills.empty:
        for _, f in fills.sort_values("ts_last").iterrows():
            qty = float(str(f["filled_qty"]).split()[0])
            side = str(f["side"]).upper()
            shares += qty if side == "BUY" else -qty
            fill_rows.append({"time": str(f["ts_last"]), "side": side, "qty": qty,
                              "avg_px": float(str(f["avg_px"]).split()[0])})

    pending = res.output.pending
    action = "HOLD"
    if pending is not None:
        action = "BUY" if pending[0].value == "enter_long" else "SELL"

    closes = df["close"]
    fast = float(closes.rolling(state.fast).mean().iloc[-1])
    slow = float(closes.rolling(state.slow).mean().iloc[-1])
    return SymbolStatus(
        symbol=symbol,
        last_session=sessions[-1],
        close=float(closes.iloc[-1]),
        sma_fast=fast,
        sma_slow=slow,
        trend="UP" if fast > slow else "DOWN",
        shares=shares,
        equity=float(res.equity.iloc[-1]),
        return_since_start=float(res.equity.iloc[-1] / state.capital_per_symbol - 1),
        trades_since_start=len(fill_rows),
        next_open_action=action,
        fills=fill_rows,
    )
