"""Build a paper-trading node: live public market data, simulated execution.

Safety by construction:

* The only execution client is Nautilus's **sandbox** (in-process simulated
  fills). No exchange execution client is imported or created, so no order can
  reach any exchange.
* No API keys are read or required. Market data uses the public JSON streams.

Region: ``api.binance.com`` refuses connections from restricted locations
(HTTP 451). ``region="us"`` uses Binance.US public market data instead.
Prices are close to, but not identical with, Binance.com. Choose the region
your exchange account would actually trade on.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.binance import BinanceDataClientFactory
from nautilus_trader.adapters.binance import BinanceEnvironment
from nautilus_trader.adapters.binance import BinanceInstrumentProviderConfig
from nautilus_trader.adapters.binance import BinanceProductType
from nautilus_trader.adapters.binance import BinanceSpotMarketDataMode
from nautilus_trader.adapters.sandbox import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox import SandboxExecutionClientFactory
from nautilus_trader.common import Environment
from nautilus_trader.common import LoggerConfig
from nautilus_trader.common import LogLevel
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.live import LiveNode
from nautilus_trader.model import AccountType
from nautilus_trader.model import Currency
from nautilus_trader.model import Money
from nautilus_trader.model import OmsType
from nautilus_trader.model import TraderId
from nautilus_trader.model import Venue

REGIONS = ("us", "com")


@dataclass(frozen=True)
class PaperSettings:
    symbol: str = "BTCUSDT"
    region: str = "us"
    starting_usdt: float = 10_000.0
    fee_rate: float = 0.001  # per side; set to your actual exchange tier
    quote_fills: bool = True  # fill at live best bid/ask (needs quote subscription) instead of bar close
    log_level: LogLevel = LogLevel.WARNING

    def __post_init__(self) -> None:
        if self.region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}")
        if not self.symbol.endswith("USDT") or not self.symbol.isalnum():
            raise ValueError("symbol must be an alphanumeric *USDT pair, e.g. BTCUSDT")
        if self.starting_usdt <= 0 or not 0 <= self.fee_rate < 0.05:
            raise ValueError("starting_usdt must be positive and fee_rate in [0, 0.05)")


def data_client_config(settings: PaperSettings) -> BinanceDataClientConfig:
    return BinanceDataClientConfig(
        product_type=BinanceProductType.SPOT,
        environment=BinanceEnvironment.LIVE,  # real public prices; nothing is traded
        spot_market_data_mode=BinanceSpotMarketDataMode.Json,  # public streams, no keys needed
        us=settings.region == "us",
        instrument_provider=BinanceInstrumentProviderConfig(
            load_all=False, load_ids=[f"{settings.symbol}.BINANCE"],
        ),
    )


def sandbox_exec_config(settings: PaperSettings) -> SandboxExecutionClientConfig:
    base = Currency.from_str(settings.symbol[:-4])
    rate = Decimal(str(settings.fee_rate))
    return SandboxExecutionClientConfig(
        venue=Venue("BINANCE"),
        starting_balances=[Money(settings.starting_usdt, Currency.from_str("USDT")), Money(0, base)],
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        fee_model=MakerTakerFeeModel(maker_rate=rate, taker_rate=rate),
        # With quote fills, the simulated book is driven by live bid/ask only, so a market
        # BUY pays the ask and a SELL hits the bid. Otherwise bars set the price (backtest-like).
        bar_execution=not settings.quote_fills,
    )


def build_paper_node(settings: PaperSettings, name: str = "paper") -> LiveNode:
    """A SANDBOX-environment node: Binance public data in, simulated fills out."""
    return (
        LiveNode.builder(name, TraderId("PAPER-001"), Environment.SANDBOX)
        .with_logging(LoggerConfig(stdout_level=settings.log_level))
        .add_data_client(None, BinanceDataClientFactory(), data_client_config(settings))
        .add_simulated_exec_client(None, SandboxExecutionClientFactory(), sandbox_exec_config(settings))
        .build()
    )
