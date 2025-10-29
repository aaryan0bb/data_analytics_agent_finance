"""Vectorized backtesting utilities built on top of vectorbt.

The tool accepts either (1) pre-loaded price/signal data frames provided as
records, or (2) instructions to fetch data using the existing extraction tools
(`extract_daily_stock_data`, `bulk_extract_daily_closing_prices_from_polygon`,
`extract_macro_data_from_fred`, etc.).  Signals can be supplied explicitly or
generated via simple recipes (currently an SMA cross demo).

The public entry point is :func:`backtesting_tool`, which validates input using
Pydantic models and executes a vectorbt ``Portfolio.from_signals`` backtest.
The companion ``BacktestingPlugin`` integrates with ``tools_registry`` so the
tool can be invoked through the agent tooling layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt
from pydantic import BaseModel, Field, ValidationError, model_validator

# Do not import data-extraction implementations directly. We go through the
# ToolRegistry to ensure the backtesting tool depends only on the orchestrator
# tooling layer.
from quant_research_agent.orchestrator.capabilities.tools_registry import (
    ToolRegistry,
    StockDataPlugin,
    MacroDataPlugin,
    FundamentalsPlugin,
    AnalystEstimatesPlugin,
    BulkPricesPlugin,
)

# Import helper function for date alignment
from quant_research_agent.tools.data.extraction import date_alignment_for_series


# -----------------------------------------------------------------------------
# Pydantic models describing the request/response payloads
# -----------------------------------------------------------------------------


class BacktestEngine(str, Enum):
    """Supported execution engines."""

    VECTORBT = "vectorbt"


class SignalMode(str, Enum):
    """How trading signals are supplied/generated."""

    PRECOMPUTED = "precomputed"
    SMA_CROSS = "sma_cross"


class BacktestParams(BaseModel):
    """Execution parameters passed to vectorbt."""

    engine: BacktestEngine = Field(default=BacktestEngine.VECTORBT)
    init_cash: float = 100_000.0
    fees: float = Field(default=0.0005, ge=0.0)
    slippage: float = Field(default=0.0005, ge=0.0)
    stop_loss: Optional[float] = Field(default=None, ge=0.0)
    trailing_stop: bool = False
    take_profit: Optional[float] = Field(default=None, ge=0.0)
    cash_sharing: bool = True
    freq: Optional[str] = None


class SignalRecipe(BaseModel):
    """Configuration for generating or ingesting signals."""

    mode: SignalMode = SignalMode.SMA_CROSS
    fast: int = Field(default=10, ge=1, description="Fast window for SMA cross")
    slow: int = Field(default=50, ge=2, description="Slow window for SMA cross")
    entries: Optional[List[Dict[str, Any]]] = None
    exits: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def _check_windows(cls, values: "SignalRecipe") -> "SignalRecipe":
        if values.mode == SignalMode.SMA_CROSS and values.fast >= values.slow:
            raise ValueError("For SMA cross strategy, fast window must be < slow window")
        return values


class PriceSourceKind(str, Enum):
    SINGLE_DAILY = "single_daily"
    SINGLE_INTRADAY = "single_intraday"
    BULK_DAILY = "bulk_daily"


class PriceSource(BaseModel):
    """Instruction for fetching prices using existing extraction tools."""

    kind: PriceSourceKind
    ticker: Optional[str] = None
    tickers: Optional[List[str]] = None
    start_date: str
    end_date: str
    interval: str = "1d"

    @model_validator(mode="after")
    def _validate(self) -> "PriceSource":
        if self.kind in {PriceSourceKind.SINGLE_DAILY, PriceSourceKind.SINGLE_INTRADAY} and not self.ticker:
            raise ValueError("ticker required for single_* price source")
        if self.kind == PriceSourceKind.BULK_DAILY:
            if not self.tickers or len(self.tickers) < 1:
                raise ValueError("tickers list required for bulk_daily source")
        return self


class DataFramePayload(BaseModel):
    """Generic representation of a pandas DataFrame serialized as records."""

    records: List[Dict[str, Any]]
    index_field: str = "date"


class BacktestRequest(BaseModel):
    """Full request payload for running a backtest."""

    price_data: Optional[DataFramePayload] = None
    price_source: Optional[PriceSource] = None
    price_is_returns: bool = False

    signal_recipe: SignalRecipe = Field(default_factory=SignalRecipe)
    entries_data: Optional[DataFramePayload] = None
    exits_data: Optional[DataFramePayload] = None

    use_macro: Optional[Dict[str, str]] = None
    use_estimates: bool = False
    use_fundamentals: bool = False

    params: BacktestParams = Field(default_factory=BacktestParams)

    @model_validator(mode="after")
    def _validate_price_inputs(self) -> "BacktestRequest":
        if not self.price_data and not self.price_source:
            raise ValueError("Either price_data or price_source must be supplied")
        if self.signal_recipe.mode == SignalMode.PRECOMPUTED:
            if not self.entries_data or not self.exits_data:
                raise ValueError("PRECOMPUTED signals require entries_data and exits_data")
        return self


class BacktestResult(BaseModel):
    """Structured backtest result returned to the caller."""

    success: bool
    engine: str
    stats: Dict[str, Any] = Field(default_factory=dict)
    equity: Optional[List[Dict[str, Any]]] = None
    returns: Optional[List[Dict[str, Any]]] = None
    trades_count: Optional[int] = None
    columns: Optional[List[str]] = None

    # Raw record payloads for post-processing
    trade_records: Optional[List[Dict[str, Any]]] = None
    order_records: Optional[List[Dict[str, Any]]] = None

    error_message: Optional[str] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _records_to_frame(payload: DataFramePayload, *, parse_dates: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(payload.records)
    if payload.index_field in df.columns:
        if parse_dates:
            df[payload.index_field] = pd.to_datetime(df[payload.index_field])
        df = df.set_index(payload.index_field)
    df = df.sort_index()
    return df


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.copy()
    for col in df.columns:
        try:
            numeric_df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return numeric_df


def _build_local_registry() -> ToolRegistry:
    """Create a minimal registry with the data-extraction plugins registered.

    Avoids importing heavy dependencies in this module's global scope. Tests can
    swap this with a custom registry if desired.
    """
    reg = ToolRegistry()
    reg.register(StockDataPlugin())
    reg.register(MacroDataPlugin())
    reg.register(FundamentalsPlugin())
    reg.register(AnalystEstimatesPlugin())
    reg.register(BulkPricesPlugin())
    return reg


def _validated_execute(reg: ToolRegistry, name: str, params: Dict[str, Any]):
    ok, fixed, issues = reg.validate(name, params)
    if not ok:
        raise ValueError(f"Validation failed for {name}: {issues}")
    return reg.execute(name, fixed)


def _load_prices(req: BacktestRequest, reg: Optional[ToolRegistry] = None) -> pd.DataFrame:
    if req.price_data is not None:
        return _ensure_numeric(_records_to_frame(req.price_data))

    # Fetch via tool registry
    assert req.price_source is not None
    source = req.price_source
    reg = reg or _build_local_registry()
    if source.kind == PriceSourceKind.SINGLE_DAILY:
        resp = _validated_execute(
            reg,
            "extract_daily_stock_data",
            {
                "ticker": source.ticker,
                "start_date": source.start_date,
                "end_date": source.end_date,
                "interval": "1d",
            },
        )
        if getattr(resp, "success", False) and getattr(resp, "data", None):
            df = pd.DataFrame(resp.data)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])  # type: ignore[assignment]
                df = df.set_index("datetime").sort_index()
            return df[["open", "high", "low", "close"]]
        return pd.DataFrame()

    if source.kind == PriceSourceKind.SINGLE_INTRADAY:
        resp = _validated_execute(
            reg,
            "extract_daily_stock_data",
            {
                "ticker": source.ticker,
                "start_date": source.start_date,
                "end_date": source.end_date,
                "interval": source.interval,
            },
        )
        if getattr(resp, "success", False) and getattr(resp, "data", None):
            df = pd.DataFrame(resp.data)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])  # type: ignore[assignment]
                df = df.set_index("datetime").sort_index()
            return df[["open", "high", "low", "close"]]
        return pd.DataFrame()

    if source.kind == PriceSourceKind.BULK_DAILY:
        df = _validated_execute(
            reg,
            "bulk_extract_daily_closing_prices_from_polygon",
            {"tickers": source.tickers, "start_date": source.start_date, "end_date": source.end_date},
        )
        return df
    raise ValueError("Unsupported price source")


def _add_optional_features(base_prices: pd.DataFrame, req: BacktestRequest, reg: Optional[ToolRegistry] = None) -> pd.DataFrame:
    series_dict: Dict[str, pd.DataFrame] = {"prices": base_prices.reset_index().rename(columns={base_prices.index.name or "index": "date"})}
    reg = reg or _build_local_registry()

    if req.use_macro and req.price_source:
        for series_id, alias in req.use_macro.items():
            macro_resp = _validated_execute(
                reg,
                "extract_economic_data_from_fred",
                {
                    "series_id": series_id,
                    "start_date": req.price_source.start_date,
                    "end_date": req.price_source.end_date,
                },
            )
            if getattr(macro_resp, "success", False) and getattr(macro_resp, "data", None):
                series = pd.DataFrame(macro_resp.data)
                series = series.rename(columns={series_id: alias}) if series_id in series.columns else series
                series_dict[alias] = series

    ticker = req.price_source.ticker if req.price_source else None
    if req.use_estimates and ticker:
        est_resp = _validated_execute(
            reg,
            "extract_analyst_estimates_from_fmp",
            {
                "ticker": ticker,
                "start_date": req.price_source.start_date,
                "end_date": req.price_source.end_date,
                "period": "quarter",
            },
        )
        if getattr(est_resp, "success", False) and getattr(est_resp, "data", None):
            series_dict["analyst_estimates"] = pd.DataFrame(est_resp.data)

    if req.use_fundamentals and ticker:
        fund_resp = _validated_execute(
            reg,
            "extract_fundamentals_from_fmp",
            {
                "ticker": ticker,
                "start_date": req.price_source.start_date,
                "end_date": req.price_source.end_date,
            },
        )
        if getattr(fund_resp, "success", False) and getattr(fund_resp, "data", None):
            series_dict["fundamentals"] = pd.DataFrame(fund_resp.data)

    try:
        return date_alignment_for_series(series_dict)
    except Exception:
        return base_prices


def _build_signals(price_frame: pd.DataFrame, req: BacktestRequest) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    if req.signal_recipe.mode == SignalMode.PRECOMPUTED:
        entries = _records_to_frame(req.entries_data, parse_dates=True)  # type: ignore[arg-type]
        exits = _records_to_frame(req.exits_data, parse_dates=True)  # type: ignore[arg-type]
        return entries.astype(bool), exits.astype(bool)

    # SMA cross
    px = price_frame
    if {"open", "high", "low", "close"}.issubset(px.columns):
        px = px["close"]
    fast = px.rolling(req.signal_recipe.fast).mean()
    slow = px.rolling(req.signal_recipe.slow).mean()
    entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    # Safety: avoid look-ahead by shifting forward one bar
    entries = entries.shift(1).fillna(False).astype(bool)
    exits = exits.shift(1).fillna(False).astype(bool)
    return entries, exits


def _returns_to_prices(df: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + df).cumprod()


def run_backtest_vectorbt(
    prices: pd.DataFrame,
    entries: pd.DataFrame | pd.Series,
    exits: pd.DataFrame | pd.Series,
    params: BacktestParams,
) -> BacktestResult:
    try:
        if {"open", "high", "low", "close"}.issubset(prices.columns):
            price = prices["close"]
            open_ = prices["open"]
            high_ = prices["high"]
            low_ = prices["low"]
        else:
            price = prices
            open_ = high_ = low_ = None

        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            fees=params.fees,
            slippage=params.slippage,
            sl_stop=params.stop_loss if params.stop_loss is not None else np.nan,
            sl_trail=params.trailing_stop,
            tp_stop=params.take_profit if params.take_profit is not None else np.nan,
            open=open_,
            high=high_,
            low=low_,
            init_cash=params.init_cash,
            cash_sharing=params.cash_sharing,
            freq=params.freq,
        )

        stats = pf.stats().to_dict()
        equity_series = pf.value().rename("equity")
        equity = [
            {"date": idx.isoformat() if hasattr(idx, "isoformat") else idx, "equity": float(val)}
            for idx, val in equity_series.items()
        ]

        # Calculate returns
        returns_series = pf.value().pct_change().dropna().rename("ret")
        returns = [
            {"date": idx.isoformat() if hasattr(idx, "isoformat") else idx, "ret": float(val)}
            for idx, val in returns_series.items()
        ]

        # Extract trade and order records for post-trade analytics
        try:
            trades_df = pf.trades.records_readable.reset_index(drop=True)
            trade_records = trades_df.to_dict("records") if not trades_df.empty else []
        except Exception:
            trade_records = []

        try:
            orders_df = pf.orders.records_readable.reset_index(drop=True)
            order_records = orders_df.to_dict("records") if not orders_df.empty else []
        except Exception:
            order_records = []

        columns = list(price.columns) if isinstance(price, pd.DataFrame) else [getattr(price, "name", "asset")]
        trades_count = int(pf.trades.count())

        return BacktestResult(
            success=True,
            engine=BacktestEngine.VECTORBT.value,
            stats=stats,
            equity=equity,
            returns=returns,
            trades_count=trades_count,
            columns=columns,
            trade_records=trade_records,
            order_records=order_records,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return BacktestResult(success=False, engine=BacktestEngine.VECTORBT.value, error_message=str(exc))


def backtesting_tool(req: BacktestRequest, *, registry: Optional[ToolRegistry] = None) -> BacktestResult:
    """Main entry point for running a backtest."""

    prices = _load_prices(req, reg=registry)
    if prices.empty:
        return BacktestResult(
            success=False,
            engine=req.params.engine.value,
            error_message="No price data available for requested period",
        )

    if req.price_is_returns:
        prices = _returns_to_prices(prices)

    feature_frame = prices
    if req.price_source and (req.use_macro or req.use_estimates or req.use_fundamentals):
        feature_frame = _add_optional_features(prices, req, reg=registry)

    entries, exits = _build_signals(feature_frame, req)
    entries = entries.reindex(prices.index).fillna(False).infer_objects(copy=False).astype(bool)
    exits = exits.reindex(prices.index).fillna(False).infer_objects(copy=False).astype(bool)

    if req.params.engine == BacktestEngine.VECTORBT:
        return run_backtest_vectorbt(prices, entries, exits, req.params)

    return BacktestResult(
        success=False,
        engine=req.params.engine.value,
        error_message=f"Unsupported backtest engine: {req.params.engine.value}",
    )


# -----------------------------------------------------------------------------
# ToolRegistry plugin wrapper
# -----------------------------------------------------------------------------


@dataclass
class BacktestingPlugin:
    """ToolRegistry plugin wrapper for the backtesting tool."""

    name: str = "run_backtest"
    description: str = "Run vectorized backtest (vectorbt) on supplied or fetched prices"
    semantic_key: str = "backtesting_results"
    rate_limit_cps: float = 2.0
    timeout_sec: float = 180.0
    max_retries: int = 1

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "price_data": {"type": "object", "description": "Records representing a DataFrame of prices"},
                    "price_source": {"type": "object", "description": "Instructions for fetching prices via existing tools"},
                    "signal_recipe": {"type": "object"},
                    "entries_data": {"type": "object"},
                    "exits_data": {"type": "object"},
                    "params": {"type": "object"},
                },
            },
        }

    def validate(self, params: Dict[str, Any]):
        try:
            req = BacktestRequest(**params)
            return True, req.model_dump(), []
        except ValidationError as exc:  # pragma: no cover - validation is straightforward
            return False, params, [str(exc)]

    def execute(self, params: Dict[str, Any]):
        req = BacktestRequest(**params)
        result = backtesting_tool(req)
        return result.model_dump()


__all__ = [
    "BacktestRequest",
    "BacktestResult",
    "BacktestParams",
    "SignalRecipe",
    "SignalMode",
    "DataFramePayload",
    "PriceSource",
    "PriceSourceKind",
    "backtesting_tool",
    "BacktestingPlugin",
]
