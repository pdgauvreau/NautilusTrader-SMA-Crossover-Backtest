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
from src.backtest import run_backtest
from src.data_loader import bar_close_times
from src.data_loader import bar_type_for
from src.data_loader import equity_instrument
from src.data_loader import open_quotes
from src.data_loader import to_nautilus_bars
from src.pipeline import run_pipeline
from src.strategies.signals import STRATEGIES
from src.strategies.target_position import TargetPositionConfig
from src.strategies.target_position import TargetPositionStrategy


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

    def without_symbols(self, removed) -> ForwardState:
        """A copy that no longer tracks ``removed`` (e.g. symbols deleted from the symbol list)."""
        removed = set(removed)
        keep = tuple(s for s in self.symbols if s not in removed)
        return replace(self, symbols=keep, starts={k: v for k, v in self.starts.items() if k not in removed})

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


def _start_close(df: pd.DataFrame, symbol: str, state: ForwardState) -> pd.Timestamp:
    """Close time of the symbol's start session (or the first session after it)."""
    sessions = df.index.strftime("%Y-%m-%d")
    wanted = state.start_for(symbol)
    later = [i for i, s in enumerate(sessions) if s >= wanted]
    if not later:
        raise ValueError(f"{symbol}: no data on or after the start session {wanted}")
    return bar_close_times(df)[later[0]]


def _fills(fills: pd.DataFrame) -> tuple[float, list[dict]]:
    """Net shares held and a list of fills from an order-fills report."""
    shares, rows = 0.0, []
    if fills.empty:
        return shares, rows
    for _, f in fills.sort_values("ts_last").iterrows():
        qty = float(str(f["filled_qty"]).split()[0])
        side = str(f["side"]).upper()
        shares += qty if side == "BUY" else -qty
        rows.append({"time": str(f["ts_last"]), "side": side, "qty": qty,
                     "avg_px": float(str(f["avg_px"]).split()[0])})
    return shares, rows


def _action(pending) -> str:
    if pending is None:
        return "HOLD"
    return "BUY" if pending[0].value == "enter_long" else "SELL"


def forward_status(df: pd.DataFrame, symbol: str, state: ForwardState) -> SymbolStatus:
    """Replay the strategy on ``df`` with trading allowed from the symbol's start session."""
    sessions = df.index.strftime("%Y-%m-%d")
    start_close = _start_close(df, symbol, state)

    res = run_pipeline(
        df, state.fast, state.slow, capital=state.capital_per_symbol, symbol=symbol,
        position_fraction=0.99, cost_rate=state.cost_rate, trade_start=start_close,
        enter_if_above_on_start=True, close_on_stop=False,
    )
    shares, fill_rows = _fills(res.output.fills)
    action = _action(res.output.pending)

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


# --------------------------------------------------------------------------- all rules
@dataclass
class RuleStatus:
    symbol: str
    rule: str
    long_now: bool  # holding shares after the latest close
    target_long: bool  # what the rule wants after the latest close
    shares: float
    equity: float
    return_since_start: float
    trades_since_start: int
    next_open_action: str  # "BUY", "SELL" or "HOLD"


def forward_rules(
    df: pd.DataFrame,
    symbol: str,
    state: ForwardState,
    keys: list[str] | None = None,
) -> list[RuleStatus]:
    """Forward-test every rule in :data:`STRATEGIES` on one symbol, same start and costs.

    Each rule starts with ``state.capital_per_symbol`` of simulated cash at the
    symbol's start session. A rule whose target is already "long" at the start
    buys at the next open, just as the SMA forward test joins an existing trend.
    """
    keys = keys or list(STRATEGIES)
    instrument = equity_instrument(symbol)
    bar_type = bar_type_for(symbol)
    bars = to_nautilus_bars(df, instrument, bar_type)
    quotes = open_quotes(df, instrument)
    close_ns = [int(t.value) for t in bar_close_times(df)]
    start_ns = int(_start_close(df, symbol, state).value)

    out_rows = []
    for key in keys:
        target = STRATEGIES[key][1](df)
        cfg = TargetPositionConfig(
            bar_type=bar_type,
            targets=dict(zip(close_ns, target.to_numpy().tolist())),
            position_fraction=0.99,
            trade_start_ns=start_ns,
            close_on_stop=False,
        )
        out = run_backtest(bars, instrument, cfg, starting_capital=state.capital_per_symbol,
                           cost_rate=state.cost_rate, strategy_cls=TargetPositionStrategy, quotes=quotes)
        shares, fills = _fills(out.fills)
        equity = float(out.equity.iloc[-1])
        out_rows.append(RuleStatus(
            symbol=symbol,
            rule=key,
            long_now=shares > 0,
            target_long=bool(target.iloc[-1]),
            shares=shares,
            equity=equity,
            return_since_start=equity / state.capital_per_symbol - 1,
            trades_since_start=len(fills),
            next_open_action=_action(out.pending),
        ))
    return out_rows


def rule_leaderboard(statuses: list[RuleStatus], capital_per_symbol: float) -> pd.DataFrame:
    """Totals per rule across symbols, best first."""
    df = pd.DataFrame([asdict(s) for s in statuses])
    g = df.groupby("rule")
    board = pd.DataFrame({
        "equity": g["equity"].sum(),
        "symbols": g.size(),
        "holding": g["long_now"].sum(),
        "trades": g["trades_since_start"].sum(),
        "buys_next_open": g["next_open_action"].apply(lambda a: int((a == "BUY").sum())),
        "sells_next_open": g["next_open_action"].apply(lambda a: int((a == "SELL").sum())),
    })
    board["return_since_start"] = board["equity"] / (board["symbols"] * capital_per_symbol) - 1
    return board.sort_values("equity", ascending=False)

