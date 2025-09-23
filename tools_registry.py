from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from tools_clean import (
    StockDataRequest,
    MacroDataRequest,
    FundamentalsRequest,
    AnalystEstimatesRequest,
    get_stock_data,
    get_macro_data,
    get_fundamentals_data,
    get_analyst_estimates,
    bulk_extract_daily_closing_prices_from_polygon,
)
from validator import validate_tool_params


class ToolPlugin(ABC):
    # Optional defaults for executor behavior; can be overridden in subclasses
    rate_limit_cps: float = 5.0
    timeout_sec: float = 30.0
    max_retries: int = 3

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def semantic_key(self) -> str: ...

    
    @abstractmethod
    def get_schema(self) -> dict: ...

    @abstractmethod
    def validate(self, params: Dict[str, Any]) -> tuple[bool, Dict[str, Any], list[str]]: ...

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Any: ...


class StockDataPlugin(ToolPlugin):
    def __init__(self):
        self._name = "extract_daily_stock_data"
        self._desc = "Extract daily stock price data for a given ticker symbol, only works for a single ticker like 'AAPL' or 'MSFT' it wont work with the list of tickers like ['AAPL', 'MSFT', 'GOOGL']"
        self._semantic_key = "closing_prices"

    @property
    def name(self) -> str: return self._name

    @property
    def description(self) -> str: return self._desc

    @property
    def semantic_key(self) -> str: return self._semantic_key

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "interval": {"type": "string", "enum": ["1d", "1wk", "1mo"]},
                },
                "required": ["ticker", "start_date", "end_date"],
            },
        }

    def validate(self, params: Dict[str, Any]) -> tuple[bool, Dict[str, Any], list[str]]:
        return validate_tool_params(self.name, params)

    def execute(self, params: Dict[str, Any]) -> Any:
        req = StockDataRequest(**params)
        return get_stock_data(req)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolPlugin] = {}

    def register(self, plugin: ToolPlugin):
        self._tools[plugin.name] = plugin

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_semantic_key(self, name: str) -> str:
        return self._tools[name].semantic_key if name in self._tools else name

    def get_available_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_schemas(self) -> Dict[str, dict]:
        return {name: plugin.get_schema() for name, plugin in self._tools.items()}

    def get_plugin(self, name: str) -> Optional[ToolPlugin]:
        """Return the plugin instance by name if registered, else None."""
        return self._tools.get(name)

    def execute(self, name: str, params: Dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ValueError(f"Tool {name} not registered")
        return self._tools[name].execute(params)

    def validate(self, name: str, params: Dict[str, Any]) -> tuple[bool, Dict[str, Any], list[str]]:
        if name not in self._tools:
            return True, params, []
        return self._tools[name].validate(params)


def auto_register(registry: "ToolRegistry") -> None:
    """Discover and register all ToolPlugin subclasses in this module.

    Assumes plugins have no-arg constructors.
    """
    import inspect
    current_module = globals()
    for _name, obj in list(current_module.items()):
        if inspect.isclass(obj) and issubclass(obj, ToolPlugin) and obj is not ToolPlugin:
            try:
                registry.register(obj())
            except Exception:
                continue


class MacroDataPlugin(ToolPlugin):
    def __init__(self):
        self._name = "extract_economic_data_from_fred"
        self._desc = "Extract economic data from FRED database"
        self._semantic_key = "macro_series"
    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._desc
    @property
    def semantic_key(self) -> str: return self._semantic_key
    def get_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": {"type": "object", "properties": {"series_id": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["series_id", "start_date", "end_date"]}}
    def validate(self, params):
        return validate_tool_params(self.name, params)
    def execute(self, params):
        req = MacroDataRequest(**params)
        return get_macro_data(req)

class FundamentalsPlugin(ToolPlugin):
    def __init__(self):
        self._name = "extract_fundamentals_from_fmp"
        self._desc = "Extract fundamental data from Financial Modeling Prep"
        self._semantic_key = "fundamentals"
    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._desc
    @property
    def semantic_key(self) -> str: return self._semantic_key
    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "ticker": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["tickers"],
            },
        }
    def validate(self, params):
        return validate_tool_params(self.name, params)
    def execute(self, params):
        req = FundamentalsRequest(**params)
        return get_fundamentals_data(req)

class AnalystEstimatesPlugin(ToolPlugin):
    def __init__(self):
        self._name = "extract_analyst_estimates_from_fmp"
        self._desc = "Extract analyst estimates from Financial Modeling Prep"
        self._semantic_key = "analyst_estimates"
    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._desc
    @property
    def semantic_key(self) -> str: return self._semantic_key
    def get_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, "period": {"type": "string", "enum": ["quarter", "annual"]}}, "required": ["ticker"]}}
    def validate(self, params):
        return validate_tool_params(self.name, params)
    def execute(self, params):
        req = AnalystEstimatesRequest(**params)
        return get_analyst_estimates(req)

class BulkPricesPlugin(ToolPlugin):
    def __init__(self):
        self._name = "bulk_extract_daily_closing_prices_from_polygon"
        self._desc = "Extract daily closing prices for multiple tickers; for example if tickers is ['AAPL', 'MSFT', 'GOOGL'] then extract the daily closing prices for these tickers"
        self._semantic_key = "closing_prices_bulk"
    # Stricter defaults: heavy endpoints and larger payloads
    rate_limit_cps = 1.0
    timeout_sec = 120.0
    max_retries = 2
    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._desc
    @property
    def semantic_key(self) -> str: return self._semantic_key
    def get_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": {"type": "object", "properties": {"tickers": {"type": "array", "items": {"type": "string"}}, "start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["tickers", "start_date", "end_date"]}}
    def validate(self, params):
        return validate_tool_params(self.name, params)
    def execute(self, params):
        return bulk_extract_daily_closing_prices_from_polygon(params.get("tickers", ["AAPL", "MSFT"]), params.get("start_date", "2023-01-01"), params.get("end_date", "2023-12-31"))
