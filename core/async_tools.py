from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from aiolimiter import AsyncLimiter

from .caching import DataCache

ToolExecFn = Callable[[str, Dict[str, Any]], Any]


class ToolConfig:
    """Per-tool knobs; defaults are conservative."""

    def __init__(self, calls_per_second: float = 5.0, timeout_sec: float = 30.0, max_retries: int = 3):
        self.calls_per_second = calls_per_second
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries


class AsyncToolExecutor:
    """
    Executes tool calls with:
      - per-tool rate limits
      - exponential backoff + jitter retries
      - asyncio parallelism
      - disk cache (raw + semantic)
    It assumes the registry.execute(name, params) returns a DataFrame or dict-like.
    """

    def __init__(self, registry, data_dir: str, tool_overrides: Optional[Dict[str, ToolConfig]] = None):
        self.registry = registry
        self.cache = DataCache(data_dir)
        self.tool_cfg: Dict[str, ToolConfig] = tool_overrides or {}
        self.limiters: Dict[str, AsyncLimiter] = {}

    def _cfg(self, tool: str) -> ToolConfig:
        # Explicit override provided at construction
        if tool in self.tool_cfg:
            return self.tool_cfg[tool]
        # Try to read defaults from registry plugin attributes
        try:
            plugin = getattr(self.registry, "get_plugin", lambda name: None)(tool)
        except Exception:
            plugin = None
        if plugin is not None:
            cps = getattr(plugin, "rate_limit_cps", 5.0)
            timeout = getattr(plugin, "timeout_sec", 30.0)
            retries = getattr(plugin, "max_retries", 3)
            return ToolConfig(calls_per_second=float(cps), timeout_sec=float(timeout), max_retries=int(retries))
        # Fallback to conservative defaults
        return ToolConfig()

    def _limiter(self, tool: str) -> AsyncLimiter:
        if tool not in self.limiters:
            cps = max(self._cfg(tool).calls_per_second, 0.1)
            self.limiters[tool] = AsyncLimiter(max_rate=cps, time_period=1.0)
        return self.limiters[tool]

    async def _execute_one(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        sem_key = self.registry.get_semantic_key(tool)
        sig = self.cache.signature(tool, params)

        # Semantic cache (preferred)
        cached_df = self.cache.sem_get(sem_key, sig)
        if cached_df is not None:
            # Backward-compatibility: migrate older artifacts to include a 'date' column when possible
            try:
                needs_rewrite = False
                if "date" not in cached_df.columns:
                    lower = {c.lower(): c for c in cached_df.columns}
                    if "datetime" in lower:
                        cached_df = cached_df.rename(columns={lower["datetime"]: "date"})
                        needs_rewrite = True
                    elif isinstance(cached_df.index, pd.DatetimeIndex):
                        idx_name = cached_df.index.name or "date"
                        cached_df = cached_df.reset_index().rename(columns={idx_name: "date"})
                        needs_rewrite = True
                if needs_rewrite:
                    # Normalize dtype and rewrite artifact in-place
                    s = pd.to_datetime(cached_df["date"], errors="coerce")
                    cached_df["date"] = s
                    self.cache.sem_set(sem_key, sig, cached_df)
            except Exception:
                pass
            return {
                "tool": tool,
                "params": params,
                "sig": sig,
                "sem_key": sem_key,
                "artifact_path": self.cache.sem_path(sem_key, sig),
                "from_cache": True,
                "rows": len(cached_df),
                "cols": len(cached_df.columns),
            }

        cfg = self._cfg(tool)
        limiter = self._limiter(tool)

        @retry(
            reraise=True,
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=8.0),
            retry=retry_if_exception_type((TimeoutError, Exception)),
        )
        async def _run_with_retry():
            async with limiter:
                loop = asyncio.get_running_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, self.registry.execute, tool, params),
                        timeout=cfg.timeout_sec,
                    )
                except asyncio.TimeoutError as te:
                    raise TimeoutError(f"Tool `{tool}` timed out @ {cfg.timeout_sec}s") from te
                return result

        result = await _run_with_retry()

        # -------- Normalize result to DataFrame for semantic cache --------
        def _to_df(res: Any) -> pd.DataFrame:
            # Already a DataFrame
            if isinstance(res, pd.DataFrame):
                return res
            # Pydantic v2 model -> model_dump; v1 -> dict
            if hasattr(res, "model_dump") and callable(getattr(res, "model_dump")):
                try:
                    obj = res.model_dump()
                except Exception:
                    obj = None
            else:
                obj = None
            if obj is None and hasattr(res, "dict") and callable(getattr(res, "dict")):
                try:
                    obj = res.dict()  # type: ignore[attr-defined]
                except Exception:
                    obj = None
            # If we got a mapping, prefer DataExtractionResponse-like shape
            if isinstance(obj, dict):
                data_part = obj.get("data")
                if isinstance(data_part, list) and (len(data_part) == 0 or isinstance(data_part[0], dict)):
                    try:
                        return pd.DataFrame.from_records(data_part)
                    except Exception:
                        pass
                # Fallback: dataframe from mapping (columns from keys)
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    pass
            # If res itself is dict-like
            if isinstance(res, dict):
                data_part = res.get("data")
                if isinstance(data_part, list) and (len(data_part) == 0 or isinstance(data_part[0], dict)):
                    try:
                        return pd.DataFrame.from_records(data_part)
                    except Exception:
                        pass
                try:
                    return pd.DataFrame(res)
                except Exception:
                    pass
            # If res is a list of dicts
            if isinstance(res, list) and (len(res) == 0 or isinstance(res[0], dict)):
                try:
                    return pd.DataFrame.from_records(res)
                except Exception:
                    pass
            # Last resort: wrap as a string value
            return pd.DataFrame([{"value": str(res)}])

        df = _to_df(result)

        # Ensure a stable 'date' column for time series outputs when index is datetime-like
        try:
            if isinstance(df.index, pd.DatetimeIndex) and ("date" not in df.columns):
                idx_name = df.index.name or "date"
                df = df.reset_index().rename(columns={idx_name: "date"})
            else:
                # Try to coalesce common date-like columns to 'date'
                date_like = None
                lower = {c.lower(): c for c in df.columns}
                for cand in ("date", "datetime", "timestamp", "publisheddate", "time"):
                    if cand in lower:
                        date_like = lower[cand]
                        break
                if date_like and date_like != "date":
                    df = df.rename(columns={date_like: "date"})
            # Normalize 'date' dtype and position
            if "date" in df.columns:
                s = pd.to_datetime(df["date"], errors="coerce")
                try:
                    if getattr(s.dt, 'tz', None) is not None:
                        s = s.dt.tz_localize(None)
                except Exception:
                    try:
                        s = s.dt.tz_convert(None)
                    except Exception:
                        pass
                df["date"] = s
                # Move 'date' to first column if not already
                cols = list(df.columns)
                cols.remove("date")
                df = df[["date"] + cols]
        except Exception:
            pass

        artifact_path = self.cache.sem_set(sem_key, sig, df)

        try:
            self.cache.raw_set(tool, sig, {"tool": tool, "params": params, "rowcount": len(df)})
        except Exception:
            pass

        return {
            "tool": tool,
            "params": params,
            "sig": sig,
            "sem_key": sem_key,
            "artifact_path": artifact_path,
            "from_cache": False,
            "rows": len(df),
            "cols": len(df.columns),
        }

    async def execute_many(self, calls: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        tasks = [self._execute_one(tool, params) for tool, params in calls]
        return await asyncio.gather(*tasks, return_exceptions=False)
