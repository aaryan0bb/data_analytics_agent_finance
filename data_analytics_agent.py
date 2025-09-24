"""
Data Analytics Agent with LangGraph orchestration, OpenAI Function Calling, and FAISS-based few-shot retrieval.

This agent provides a complete workflow for:
1. Tool selection using OpenAI function calling
2. FAISS-based few-shot example retrieval with re-ranking
3. Code generation using OpenAI structured outputs
4. Code execution with reflection loop for error handling
5. LangGraph state management throughout the process
"""

import os
import sys
import shutil
import json
import uuid
import subprocess
import logging
import tempfile
import pickle
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal
from typing_extensions import TypedDict

import pandas as pd
import numpy as np
try:
    import faiss  # type: ignore[assignment]
except Exception:
    faiss = None  # type: ignore[assignment]
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore[assignment]
except Exception:
    SentenceTransformer = None  # type: ignore[assignment]
    CrossEncoder = None  # type: ignore[assignment]

try:
    from tavily import TavilyClient  # type: ignore[assignment]
except Exception:
    TavilyClient = None  # type: ignore[assignment]

from openai import OpenAI
from pydantic import BaseModel, Field, validator
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
from validator import validate_tool_params
from tools_registry import (
    ToolRegistry,
    auto_register,
)
from core.manifests import ResultManifest, TableArtifact, FigureArtifact
from core.manifest_utils import sanitize_or_build_manifest
from core.prompting_blocks import (
    PLOTLY_CONVENTIONS,
    BACKTEST_HYGIENE,
    STATS_RIGOR,
    OUTPUT_CONTRACT,
)
from core.utils import params_signature
from datetime import datetime
import re
from langsmith import __version__ as _ls_ver  # optional presence
try:
    from langsmith.run_tree import RunTree  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
except Exception:
    RunTree = None  # graceful fallback if LangSmith is unavailable

# Tool wrappers for execution
# Legacy tool wrappers are no longer imported here; plugins handle execution.
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()

# Tavily integration (optional; degrades gracefully when unavailable)
tavily_client = None  # type: ignore[assignment]
if TavilyClient is None:
    tavily_client = None
    logger.warning("TavilyClient import failed; Tavily research will be skipped.")
else:
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_api_key:
        tavily_client = TavilyClient(api_key=tavily_api_key)
    else:
        tavily_client = None
        logger.warning("TAVILY_API_KEY not set; Tavily research will be skipped.")
# Environment setup
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def tavily_search(**kwargs) -> Dict[str, Any]:
    """Execute a Tavily search via the Responses API compatible wrapper."""
    if tavily_client is None:
        raise RuntimeError("Tavily client is unavailable (missing dependency or API key).")
    return tavily_client.search(**kwargs)


TAVILY_TOOL_DEF = [{
    "type": "function",
    "name": "tavily_search",
    "description": "Search the web using Tavily's Responses API and return JSON results.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "High-signal Tavily query string with the key terms we need evidence for."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of Tavily documents to retrieve (1-20).",
                "default": 5
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": "Depth of Tavily crawl; use 'advanced' for complex quantitative research.",
                "default": "advanced"
            },
            "include_answer": {
                "type": "boolean",
                "description": "Whether to include Tavily's synthesized answer payload in the response.",
                "default": False
            },
        },
        "required": ["query", "max_results"],
        "additionalProperties": False
    },
    "strict": True
}]

# Constants
# Allow overriding the OpenAI model via environment, default to a widely available model
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
PLANNER_MAX_TURNS = int(os.environ.get("PLANNER_MAX_TURNS", "3"))
MAX_REFLECTION_ATTEMPTS = 3
CODE_EXECUTION_TIMEOUT = 1800
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# Phase 1: global tool registry (thin)
REGISTRY = ToolRegistry()
try:
    auto_register(REGISTRY)
except Exception:
    pass

# Global executor wiring (set during agent initialization)
GLOBAL_TOOL_EXECUTOR = None  # type: ignore[var-annotated]
GLOBAL_BASE_DATA_DIR = None  # type: ignore[var-annotated]

# Lazy import for async executor to avoid import cycles at module import time
try:
    from core.async_tools import AsyncToolExecutor, ToolConfig  # noqa: F401
except Exception:
    AsyncToolExecutor = None  # type: ignore[assignment]
    ToolConfig = None  # type: ignore[assignment]

# Function names
TOOL_SELECTION_FUNCTION_NAME = "select_tools"

######################################################################
# Utility helpers (Responses API parsing + Pydantic schema)
######################################################################

def _model_schema(model_cls):
    try:
        return model_cls.model_json_schema()
    except Exception:
        # Fallback for older Pydantic versions
        return model_cls.schema()

def _extract_text(resp):
    """Extract text content from a Responses API response object."""
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    chunks = []
    for item in getattr(resp, "output", []) or []:
        for seg in getattr(item, "content", []) or []:
            txt = getattr(seg, "text", None)
            if txt:
                chunks.append(getattr(txt, "value", None) or str(txt))
    return "".join(chunks)

# JSON default serializer for numpy/pandas types
def _json_default(o):
    if isinstance(o, (pd.Timestamp, datetime)):
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

def _persist_df_as_csv(df: pd.DataFrame, data_dir: str, sem_key: str) -> dict:
    """Save DataFrame to CSV with a standardized 'date' column and return metadata + small sample.
    - Ensures a single 'date' column (case-insensitive from ['date','datetime','timestamp','publisheddate','time']).
    - Removes timezone info; keeps naive datetimes.
    - Writes CSV with index=False; 'date' as first column.
    - Returns description dict with file_path, rows, columns, dtypes, nulls, date_start, date_end, sample_head.
    """
    if df is None:
        return {"ok": False, "error": "empty dataframe"}

    df_out = df.copy()

    # Materialize index as 'date' if it's datetime-like and no date-like column exists
    date_like_names = ("date", "datetime", "timestamp", "publisheddate", "time")
    has_date_like_col = any(c.lower() in date_like_names for c in df_out.columns)
    if pd.api.types.is_datetime64_any_dtype(df_out.index) and not has_date_like_col:
        idx_name = df_out.index.name or df_out.columns[0]
        df_out = df_out.reset_index().rename(columns={idx_name: "date"})

    # Coalesce any existing date-like column to a single 'date' column and drop others
    col_lower_map = {c.lower(): c for c in df_out.columns}
    chosen = None
    for cand in ("date", "datetime", "timestamp", "publisheddate", "time"):
        if cand in col_lower_map:
            chosen = col_lower_map[cand]
            break
    if chosen is not None:
        if chosen != "date":
            df_out = df_out.rename(columns={chosen: "date"})
        # Drop other date-like columns
        for cand in ("datetime", "timestamp", "publisheddate", "time"):
            cname = col_lower_map.get(cand)
            if cname and cname in df_out.columns and cname != chosen and cname != "date":
                df_out = df_out.drop(columns=[cname])

        # Normalize 'date' to naive datetime (no timezone)
        s_dt = pd.to_datetime(df_out["date"], errors="coerce")
        try:
            if getattr(s_dt.dt, 'tz', None) is not None:
                s_dt = s_dt.dt.tz_localize(None)
        except Exception:
            try:
                s_dt = s_dt.dt.tz_convert(None)
            except Exception:
                pass
        df_out["date"] = s_dt

    # Reorder to place 'date' first if present
    if "date" in df_out.columns:
        other_cols = [c for c in df_out.columns if c != "date"]
        df_out = df_out[["date"] + other_cols]
        try:
            df_out = df_out.sort_values("date")
        except Exception:
            pass

    # Build metadata
    rows, cols = int(df_out.shape[0]), int(df_out.shape[1])
    dtypes = {str(k): str(v) for k, v in df_out.dtypes.to_dict().items()}
    nulls = {str(k): int(v) for k, v in df_out.isna().sum().to_dict().items()}

    # Date range formatting (no timezone, plain date/datetime)
    date_start = date_end = None
    if "date" in df_out.columns:
        s_all = pd.to_datetime(df_out["date"], errors="coerce").dropna()
        if not s_all.empty:
            try:
                is_midnight = ((s_all.dt.hour.fillna(0) == 0) & (s_all.dt.minute.fillna(0) == 0) & (s_all.dt.second.fillna(0) == 0)).all()
            except Exception:
                is_midnight = False
            fmt = "%Y-%m-%d" if is_midnight else "%Y-%m-%d %H:%M:%S"
            date_start = s_all.min().strftime(fmt)
            date_end = s_all.max().strftime(fmt)

    # Write CSV (no index)
    file_path = os.path.join(data_dir, f"{sem_key}.csv")
    try:
        df_out.to_csv(file_path, index=False)
    except Exception:
        df_out.astype(str).to_csv(file_path, index=False)

    # Sample head (format datetime columns without timezone, no 'T')
    head_df = df_out.head(5).copy()
    for c in head_df.columns:
        if pd.api.types.is_datetime64_any_dtype(head_df[c]):
            head_df[c] = pd.to_datetime(head_df[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    sample_head = head_df.to_dict(orient="records")

    return {
        "ok": True,
        "file_path": file_path,
        "rows": rows,
        "columns": cols,
        "dtypes": dtypes,
        "nulls": nulls,
        "date_start": date_start,
        "date_end": date_end,
        "sample_head": sample_head,
    }


def _order_preserving_dedupe(items: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _code_context_snippet(state: AgentState, max_chars: int = 2000) -> str:
    """Return a truncated view of generated code for planner prompts."""
    code = state.get("generated_code") or ""
    if not code:
        return ""
    snippet = code[:max_chars]
    if len(code) > max_chars:
        snippet += "\n# ... truncated for tool parameterization context ..."
    return snippet


def _build_dataset_descriptor(path: str) -> Dict[str, Any]:
    """Generate a standardized dataset descriptor for tool outputs."""

    try:
        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        rows = int(len(df))
        cols = int(len(df.columns))
        dtypes = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        nulls = {str(k): int(v) for k, v in df.isna().sum().to_dict().items()}
        date_start = date_end = None
        if "date" in df.columns:
            s_all = pd.to_datetime(df["date"], errors="coerce").dropna()
            if not s_all.empty:
                try:
                    is_midnight = ((s_all.dt.hour.fillna(0) == 0) & (s_all.dt.minute.fillna(0) == 0) & (s_all.dt.second.fillna(0) == 0)).all()
                except Exception:
                    is_midnight = False
                fmt = "%Y-%m-%d" if is_midnight else "%Y-%m-%d %H:%M:%S"
                date_start = s_all.min().strftime(fmt)
                date_end = s_all.max().strftime(fmt)
        head_df = df.head(5).copy()
        for c in head_df.columns:
            if pd.api.types.is_datetime64_any_dtype(head_df[c]):
                head_df[c] = pd.to_datetime(head_df[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        sample_head = head_df.to_dict(orient="records")

        return {
            "ok": True,
            "file_path": path,
            "rows": rows,
            "columns": cols,
            "dtypes": dtypes,
            "nulls": nulls,
            "date_start": date_start,
            "date_end": date_end,
            "sample_head": sample_head,
        }
    except Exception as exc:
        return {"ok": False, "file_path": path, "error": str(exc)}


def _save_graph_state(initial_state: Dict[str, Any], final_state: Dict[str, Any]) -> Optional[str]:
    """Persist the entire graph state locally and emit a LangSmith run.

    Returns the JSON file path if written.
    """
    try:
        # Prefer the state's data_dir when present
        out_dir = final_state.get("data_dir") or initial_state.get("data_dir") or tempfile.mkdtemp(prefix="agent_state_")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "graph_state.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "initial_state": initial_state,
                "final_state": final_state,
            }, f, default=_json_default)

        # Send a single summary run to LangSmith if available and configured
        if RunTree is not None:
            try:
                root = RunTree(
                    name="DataAnalyticsAgent",
                    run_type="chain",
                    inputs={
                        "user_input": initial_state.get("user_input", ""),
                        "task_description": initial_state.get("task_description", ""),
                        "selected_tools": initial_state.get("selected_tools", []),
                    },
                    extra={"out_dir": out_dir},
                )
                root.end(outputs={
                    "final_response": final_state.get("final_response", ""),
                    "tool_files": final_state.get("tool_files", {}),
                    "tool_outputs": final_state.get("tool_outputs", {}),
                    "execution_result": final_state.get("execution_result", {}),
                })
                root.post()
            except Exception as e:
                logger.warning(f"LangSmith run post failed: {e}")

        return out_path
    except Exception as e:
        logger.warning(f"Failed to save graph state: {e}")
        return None


######################################################################
# Pydantic Models for Structured Outputs (OpenAI Response Format)
######################################################################

class ToolSelection(BaseModel):
    """Selected tools for the given task."""
    selected_tools: List[str] = Field(description="List of relevant tool names for the task")
    reasoning: str = Field(description="Brief explanation of why these tools were selected")

class CodeGeneration(BaseModel):
    """Generated Python code for the task."""
    code: str = Field(description="Complete Python code to solve the task")
    explanation: str = Field(description="Brief explanation of what the code does")
    imports_needed: List[str] = Field(description="List of Python imports required")

class CodeReflection(BaseModel):
    """Reflected and fixed Python code."""
    fixed_code: str = Field(description="Corrected Python code")
    changes_made: str = Field(description="Description of changes made to fix the issues")
    explanation: str = Field(description="Explanation of the fix")

# Pydantic code parser helpers
class CodeContent(BaseModel):
    code: str = Field(description='complete python code without any markdown or natural language text')

class CodeResponse(BaseModel):
    code: str

    @classmethod
    def parse_llm_response(cls, content: str) -> "CodeResponse":
        # Try JSON with a 'code' field first
        if content:
            try:
                obj = json.loads(content)
                if isinstance(obj, dict) and isinstance(obj.get('code'), str):
                    return cls(code=obj['code'])
            except Exception:
                pass
        # Fallback: extract fenced code blocks ``` ... ```
        lines = content.split('\n') if content else []
        code_lines = []
        in_code_block = False
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                code_lines.append(line)
        if code_lines:
            return cls(code='\n'.join(code_lines))
        # Last resort: return raw content
        return cls(code=content or '')

# Tool parameter models for LLM planning
class StockParams(BaseModel):
    ticker: Optional[str] = Field(default="SPY", description="Stock ticker symbol")
    start_date: Optional[str] = Field(default="2023-01-01", description="YYYY-MM-DD")
    end_date: Optional[str] = Field(default="2023-12-31", description="YYYY-MM-DD")
    interval: Optional[str] = Field(default="1d", description="Data interval like '1d'")

class MacroParams(BaseModel):
    series_id: Optional[str] = Field(default="UNRATE", description="FRED series id")
    start_date: Optional[str] = Field(default="2023-01-01", description="YYYY-MM-DD")
    end_date: Optional[str] = Field(default="2023-12-31", description="YYYY-MM-DD")

class AnalystParams(BaseModel):
    ticker: Optional[str] = Field(default="AAPL")
    start_date: Optional[str] = Field(default="2023-01-01")
    end_date: Optional[str] = Field(default="2023-12-31")
    period: Optional[str] = Field(default="quarter", description="'quarter' or 'annual'")

class FundamentalsParams(BaseModel):
    ticker: Optional[str] = Field(default="AAPL")
    start_date: Optional[str] = Field(default="2023-01-01")
    end_date: Optional[str] = Field(default="2023-12-31")

class BulkPricesParams(BaseModel):
    tickers: Optional[List[str]] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"]) 
    start_date: Optional[str] = Field(default="2023-01-01")
    end_date: Optional[str] = Field(default="2023-12-31")

class ToolParamsPlan(BaseModel):
    extract_daily_stock_data: Optional[StockParams] = None
    extract_economic_data_from_fred: Optional[MacroParams] = None
    extract_fundamentals_from_fmp: Optional[FundamentalsParams] = None
    extract_analyst_estimates_from_fmp: Optional[AnalystParams] = None
    bulk_extract_daily_closing_prices_from_polygon: Optional[BulkPricesParams] = None


######################################################################
# Post-manifest Improvement Contracts
######################################################################

Category = Literal[
    "financial_factors",
    "visualizations",
    "data_sources",
    "methodology",
    "risk_controls",
    "benchmarks",
    "diagnostics",
]
Intent = Literal["discover", "validate", "challenge", "visualize", "ops"]


class ResearchQuery(BaseModel):
    """Minimal Tavily research query specification."""

    id: str = Field(..., description="Stable identifier for correlating research evidence to the query.")
    question: str = Field(..., min_length=10, description="Full natural-language question the research should answer.")
    category: Category
    intent: Intent
    priority: int = Field(5, ge=1, le=10, description="Lower numbers run first (1 is highest priority).")
    queries: List[str] = Field(..., min_items=1, max_items=5, description="Candidate Tavily query strings.")
    include_domains: List[str] = Field(default_factory=list, description="Domains Tavily should prioritize.")
    exclude_domains: List[str] = Field(default_factory=list, description="Domains Tavily should avoid.")
    recency_days: Optional[int] = Field(None, description="Optional recency filter in days.")
    k: int = Field(5, ge=1, le=20, description="Maximum number of Tavily search results to retrieve.")
    citations_required: bool = Field(True, description="Whether synthesized answers must include citations.")

    @validator("queries")
    def _uniq_queries(cls, v: List[str]) -> List[str]:
        dedup = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("Each query must be a string.")
            stripped = item.strip()
            if len(stripped) < 8:
                raise ValueError("Each query must be at least 8 characters long.")
            if stripped not in dedup:
                dedup.append(stripped)
        if len(dedup) != len(v):
            raise ValueError("queries must be unique")
        return dedup


class Evidence(BaseModel):
    """Evidence captured for a single research query."""

    query_id: str
    used_query: str
    docs: List[Dict[str, Any]] = Field(default_factory=list)
    synthesis: str
    caveats: List[str] = Field(default_factory=list)


class ResearchBundle(BaseModel):
    """Aggregate research payload shared across the improvement loop."""

    goal: Optional[str] = ""
    queries: List[ResearchQuery] = Field(default_factory=list)
    evidences: List[Evidence] = Field(default_factory=list)
    summary: str = ""
    conflicts: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


ImprovementKind = Literal["code", "result", "data", "validation", "visualization", "infra"]
RiskLevel = Literal["low", "medium", "high"]


class Improvement(BaseModel):
    """Single actionable improvement proposed by the evaluator."""

    type: ImprovementKind
    description: str
    rationale: str
    acceptance_criteria: List[str] = Field(default_factory=list)
    risk: RiskLevel = "low"
    artifacts_to_add: List[str] = Field(default_factory=list)


class ToolSuggestion(BaseModel):
    """Suggestion for running an additional registry tool during improvements."""

    tool_name: str
    reasoning: str


class EvalReport(BaseModel):
    """Evaluator output capturing improvements, research plan, and tool ideas."""

    improvements: List[Improvement] = Field(default_factory=list)
    research_goal: Optional[str] = ""
    research_queries: List[ResearchQuery] = Field(default_factory=list)
    new_tool_suggestions: List[ToolSuggestion] = Field(default_factory=list)
    notes: str = ""


class ToolNamePick(BaseModel):
    """Tool name recommendation with reasoning."""
    tool_name: str = Field(..., description="Exact registry tool name")
    reasoning: str = Field(..., description="Concise justification for recommendation")


class ToolRecommendationResponse(BaseModel):
    """LLM response for tool name recommendations."""
    recommendations: List[ToolNamePick] = Field(
        default_factory=list,
        description="List of recommended tools with reasoning"
    )


class DeltaCodePatch(BaseModel):
    """Structured response for delta code generation during improvement rounds."""

    new_code: str = Field(..., description="Complete Python script replacement incorporating improvements.")
    changes_summary: str = Field(
        ..., description="Natural language summary of the code changes applied in this iteration."
    )
    reasoning: str = Field(
        ..., description="Detailed reasoning explaining how the changes satisfy requirements."
    )


######################################################################
# Deterministic Prompt Templates for Improvement Loop
######################################################################


EVAL_REPORT_SYSTEM = """
You are a rigorous evaluation agent for a deterministic quant workflow.

Context you MUST respect:
- No network access in generated code; only local artifacts.
- Output contract: write result.json with tables, figures, metrics, explanation.
- Deterministic behavior: avoid time-based randomness; make assumptions explicit.
- Backtest hygiene: no lookahead, use 1-bar delay on signals, realistic frictions.

Your job now:
1) Read the user request, current manifest summary, execution stdout/stderr, any warnings, dataset summaries and assumptions.
2) Produce a high-signal EvalReport that includes:
   - improvements: actionable deltas (code/result/data/validation/visualization/infra)
   - research_goal: short statement of the research objective (optional)
   - research_queries: prioritized ResearchQuery objects Tavily will run
   - new_tool_suggestions: registry tools to add value (optional)
   - notes: concise evaluator notes
3) Your research questions should reflect these themes:
   * What new financial factors could we generate?
   * What more visualizations would materially improve insight or risk diagnostics?
   * What unique/better data sources could we use (and licensing constraints)?
   * How to validate/challenge our methodology (robustness, alternative specs, benchmarks)?
   * What risk controls or portfolio construction knobs could reduce drawdown?
4) Each research query should be concrete, include 1-5 high-signal entries in `queries`,
   surface optional include/exclude domains or recency_days when relevant, and set `k` (max results).

Return ONLY JSON conforming to the EvalReport schema. Do NOT include any extra text.
"""


def _eval_input_from_state(state: "AgentState") -> Dict[str, Any]:
    """Transform agent state into the structured payload required by eval_report_node."""

    tool_outputs = state.get("tool_outputs", {}) or {}
    datasets: List[Dict[str, Any]] = []
    for key, desc in tool_outputs.items():
        if not desc:
            continue
        datasets.append(
            {
                "key": key,
                "file_path": desc.get("file_path") or "",
                "rows": int(desc.get("rows") or 0),
                "columns": int(desc.get("columns") or 0),
                "dtypes": desc.get("dtypes") or {},
                "nulls": desc.get("nulls") or {},
                "date_start": desc.get("date_start"),
                "date_end": desc.get("date_end"),
                "sample_head": desc.get("sample_head") or [],
            }
        )

    return {
        "user_request": state.get("user_input", ""),
        "manifest_summary": state.get("manifest_summary", {}),
        "execution_result": state.get("execution_result", {}),
        "warnings": state.get("warnings", []),
        "datasets": datasets,
        "assumptions": state.get("assumptions", []),
    }


TAVILY_QA_SYSTEM = """
You are a research assistant using Tavily web search via function-calling.
For the given research query, first CALL the 'tavily_search' tool with the most appropriate 'query' from the provided list of queries.
- Use 'advanced' search_depth for non-trivial topics.
- Prefer <= 7 results from credible domains. Use include/exclude domains if provided.
- Use recency_days only when time-sensitive.
- Set `max_results` to match the provided `k` value.
After tool results are added back to the conversation, you will be asked to synthesize findings with citations.
"""


TAVILY_SYNTHESIS_INSTRUCTIONS = """
Synthesize a concise, actionable summary for this research query. Requirements:
- Start with a 2–3 sentence direct answer.
- Then add 3–6 bullet points with specific, citable facts (include domain + year/month if available).
- Call out any conflicts, limitations, or licensing constraints.
- End with 1–3 recommended actions tailored to our workflow.
- Include citations inline as [domain] after each bullet/claim.

Return a short, well-structured paragraph + bullets. Avoid fluff.
"""


DELTA_CODE_SYSTEM = """
You are a senior Python quant engineer. You will produce a FULL replacement script.

Hard constraints:
- Read ONLY local artifacts listed by the agent (no network calls).
- Preserve output contract: write result.json with {tables, figures, metrics, explanation}.
- Preserve determinism: fix seeds where sampling is used; avoid time.now; document assumptions.
- Preserve backtest hygiene: 1-bar signal delay, realistic friction settings, no look-ahead.
- Keep DATA_DIR semantics: write outputs under os.environ['DATA_DIR'] (already set).

Your tasks:
- Apply the 'improvements' from the evaluator and the 'recommended_actions' from research synthesis.
- If new artifacts were requested (artifacts_to_add), produce them in DATA_DIR.
- Keep failure-safe: if a non-critical plot fails, write a 1x1 placeholder PNG.

Return JSON strictly matching DeltaCodePatch with `new_code` containing ONLY executable Python code (no markdown fences).
"""

######################################################################
# LangGraph State Definition
######################################################################


class AgentState(TypedDict):
    """State for the data analytics agent workflow."""
    user_input: str
    task_description: str
    selected_tools: List[str]
    retrieved_few_shots: List[Dict[str, Any]]
    composite_prompt: str
    generated_code: str
    code_explanation: str
    execution_result: Dict[str, Any]
    error_message: str
    reflection_count: int
    final_response: str
    intermediate_results: Dict[str, Any]
    # Tooling integration
    tool_params: Dict[str, Dict[str, Any]]
    tool_results: Dict[str, Any]
    tool_files: Dict[str, str]
    data_dir: str
    tool_calls_log: List[Dict[str, Any]]
    tool_outputs: Dict[str, Any]
    intermediate_context: str
    # Phase 1 additions
    run_id: str
    warnings: List[str]
    # Optional: executor artifacts
    tool_artifacts: List[Dict[str, Any]]
    base_data_dir: str
    # Phase 3 additions (ingestion)
    run_dir: str
    result_manifest: Any
    iteration_workdirs: List[str]
    final_manifest_path: str
    exec_workdir: str
    manifest_ok: bool
    iteration_manifests: List[Any]
    iteration_index: int
    # Multi-turn planner fields
    planning_turn: int
    plan_history: List[List[Dict[str, Any]]]
    assumptions: List[Dict[str, Any]]
    stop_reason: str
    # Post-manifest improvement loop fields
    eval_report: Dict[str, Any]
    tavily_results: Dict[str, Any]
    delta_tool_outputs: Dict[str, Any]
    improvement_round: int
    should_finalize_after_eval: bool
    # Unused tools tracking fields
    unused_tool_picks: List[Dict[str, str]]
    unused_tool_files: Dict[str, str]
    unused_tool_outputs: Dict[str, Dict[str, Any]]
    unused_tool_names: List[str]

######################################################################
# OpenAI Function Definitions (Provided via Registry)
######################################################################

######################################################################
# Few-Shot Retrieval System with FAISS
######################################################################

class FewShotRetriever:
    """FAISS-based retrieval system for few-shot examples with re-ranking."""
    
    def __init__(self, few_shots_dir: str):
        self.few_shots_dir = Path(few_shots_dir)
        # Enable only if deps are available
        self.enabled = bool(SentenceTransformer) and bool(faiss)
        if self.enabled:
            try:
                self.encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                self.encoder = None
                self.enabled = False
            try:
                self.reranker = CrossEncoder(RERANKER_MODEL) if CrossEncoder else None
            except Exception:
                self.reranker = None
        else:
            self.encoder = None
            self.reranker = None
        self.few_shots_data = []
        self.index = None
        
        try:
            self._load_few_shots()
            self._build_faiss_index()
        except Exception as e:
            logger.warning(f"FewShotRetriever initialization degraded: {e}")
    
    def _load_few_shots(self):
        """Load few-shot examples from files in the few_shots directory."""
        logger.info(f"Loading few-shot examples from: {self.few_shots_dir}")
        
        # Support both .py (legacy) and .json files in the directory
        for file_path in list(self.few_shots_dir.glob("*.py")) + list(self.few_shots_dir.glob("*.json")):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse examples from files
                if file_path.suffix.lower() == '.json':
                    try:
                        examples = json.loads(content)
                    except json.JSONDecodeError as e:
                        # Try to sanitize common escape issues (e.g., backslash-newline, stray backslashes)
                        import re
                        sanitized = content.replace('\\\r\n', '\\n').replace('\\\n', '\\n')
                        sanitized = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", sanitized)
                        try:
                            examples = json.loads(sanitized)
                        except Exception as e2:
                            logger.warning(f"Failed to parse JSON in {file_path} after sanitization: {e2}")
                            continue
                else:
                    # For .py files: prefer AST-based extraction of a top-level list/tuple literal
                    examples = None
                    import ast
                    try:
                        tree = ast.parse(content)
                        for node in tree.body:
                            # case 1: a bare list literal as a top-level expression
                            if isinstance(node, ast.Expr) and isinstance(node.value, (ast.List, ast.Tuple)):
                                try:
                                    val = ast.literal_eval(node.value)
                                    if isinstance(val, list):
                                        examples = val
                                        break
                                except Exception:
                                    pass
                            # case 2: a top-level assignment to a list
                            if isinstance(node, ast.Assign):
                                try:
                                    val = ast.literal_eval(node.value)
                                    if isinstance(val, list):
                                        examples = val
                                        break
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.warning(f"AST parse failed for {file_path}: {e}")

                    # Fallbacks for files that start with a JSON-like array string
                    if examples is None:
                        stripped = content.strip()
                        if stripped.startswith('[') and stripped.rfind(']') != -1:
                            arr_text = stripped[: stripped.rfind(']') + 1]
                            try:
                                examples = json.loads(arr_text)
                            except Exception:
                                try:
                                    examples = ast.literal_eval(arr_text)
                                except Exception as e2:
                                    logger.warning(f"Failed to parse top-level list in {file_path}: {e2}")

                    if examples is None:
                        # Skip files that don't contain a parseable list
                        logger.warning(f"No parseable examples found in {file_path}")
                        continue

                for example in examples:
                    example['source_file'] = file_path.name
                    # Normalize code field and unescape
                    if 'executable_code' in example:
                        example['code'] = example['executable_code'].replace('\\n', '\n').replace('\\', '')
                    elif 'code' in example and isinstance(example['code'], str):
                        example['code'] = example['code'].replace('\\n', '\n').replace('\\', '')
                    # Normalize description field
                    if 'description' not in example or not isinstance(example.get('description'), str):
                        if isinstance(example.get('code_description'), str):
                            example['description'] = example['code_description']
                        elif isinstance(example.get('question'), str):
                            example['description'] = example['question']
                        else:
                            # Fallback to first line of code as a description
                            code_str = example.get('code', '') or ''
                            first_line = code_str.strip().splitlines()[0] if code_str.strip().splitlines() else ''
                            example['description'] = first_line[:120] if first_line else 'Example from ' + file_path.name
                    self.few_shots_data.append(example)
                        
            except (json.JSONDecodeError, ValueError, FileNotFoundError) as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.few_shots_data)} few-shot examples")
    
    def _build_faiss_index(self):
        """Build FAISS index for semantic search."""
        if not self.enabled:
            logger.warning("FewShotRetriever disabled (missing faiss or sentence_transformers)")
            return
        if not self.few_shots_data:
            logger.warning("No few-shot examples loaded")
            return
        
        logger.info("Building FAISS index...")
        
        # Extract descriptions for embedding
        descriptions = []
        for ex in self.few_shots_data:
            desc = ex.get('description') or ex.get('code_description') or ex.get('question')
            if not desc:
                code_preview = (ex.get('code') or '')
                desc = (code_preview.strip().splitlines()[0] if code_preview else '') or 'Example'
            descriptions.append(desc)
        
        # Generate embeddings
        if self.encoder is None:
            logger.warning("Encoder not available; skipping FAISS index build")
            return
        embeddings = self.encoder.encode(descriptions)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        if not faiss:
            logger.warning("faiss not available; cannot build index")
            return
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        if hasattr(faiss, 'normalize_L2'):
            faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        logger.info(f"FAISS index built with {self.index.ntotal} examples")
    
    def retrieve(self, query: str, top_k: int = 10, rerank_top_k: int = 3) -> List[Dict]:
        """Retrieve and re-rank few-shot examples."""
        if (not self.enabled) or (not self.index) or (self.encoder is None):
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        if hasattr(faiss, 'normalize_L2'):
            faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Get candidates
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.few_shots_data):
                candidate = self.few_shots_data[idx].copy()
                candidate['faiss_score'] = float(scores[0][i])
                candidates.append(candidate)
        
        # Re-rank with cross-encoder
        if len(candidates) > rerank_top_k and self.reranker is not None:
            pairs = [(query, candidate['description']) for candidate in candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            # Sort by re-rank scores
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(rerank_scores[i])
            
            candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        return candidates[:rerank_top_k]

######################################################################
# LangGraph Workflow Nodes
######################################################################

def select_tools_node(state: AgentState) -> AgentState:
    """Select relevant tools based on user input using OpenAI function calling."""
    logger.info("Selecting tools...")
    
    # Create function calling prompt
    registry_tools = []
    try:
        registry_tools = REGISTRY.get_available_tools()
    except Exception:
        registry_tools = []
    available_tool_names = sorted(registry_tools)
    system_content = f"""You are a tool selection expert. Based on the user's request, determine which tools from the available tools are needed.

Available tools: {', '.join(available_tool_names)}

Call the select_tools function with your selection."""
    
    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user", 
            "content": f"User request: {state['user_input']}\nTask description: {state.get('task_description', '')}"
        }
    ]
    
    # Define function for tool selection with enum constraint
    select_tools_function = {
        "name": TOOL_SELECTION_FUNCTION_NAME,
        "description": "Select relevant tools for the given task",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_tools": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": available_tool_names
                    },
                    "description": f"List of relevant tool names from: {available_tool_names}"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of tool selection"
                }
            },
            "required": ["selected_tools", "reasoning"]
        }
    }
    
    tools = [{
        "type": "function",
        "name": select_tools_function["name"],
        "description": select_tools_function["description"],
        "parameters": select_tools_function["parameters"],
    }]
    
    # Make API call with function calling (Responses API)
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "medium"},
        input=messages,
        tools=tools,
        tool_choice={"type": "function", "name": TOOL_SELECTION_FUNCTION_NAME}
    )
    
    # Parse function call result
    # Extract first tool call arguments directly from response.output
    calls = [item for item in (response.output or []) if getattr(item, "type", None) == "function_call"]
    if not calls:
        raise RuntimeError("Tool selection did not return a tool call")
    tc = calls[0]
    fn_args = getattr(tc, "arguments", None) or (getattr(getattr(tc, "function", None), "arguments", None))
    if not fn_args:
        raise RuntimeError("Tool call missing arguments")
    result = json.loads(fn_args)
    
    state["selected_tools"] = _order_preserving_dedupe(result["selected_tools"])
    print(result["reasoning"])
    logger.info(f"Selected tools: {result['selected_tools']}")
    
    return state

## Legacy plan_tool_params_node removed (registry handles validation)


def _ensure_data_dir(state: AgentState) -> str:
    """Ensure a stable run-scoped data directory exists and return it.

    Preference order for base directory:
      1) state['base_data_dir'] if provided
      2) $AGENT_DATA_DIR environment variable
      3) '.agent_data' folder under current working directory
    """
    if state.get("data_dir") and os.path.isdir(state["data_dir"]):
        return state["data_dir"]

    base_dir = (
        state.get("base_data_dir")
        or os.environ.get("AGENT_DATA_DIR")
        or os.path.join(os.getcwd(), ".agent_data")
    )
    run_id = state.get("run_id") or uuid.uuid4().hex
    run_dir = os.path.join(base_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    state["data_dir"] = run_dir
    state.setdefault("run_id", run_id)
    return run_dir


## Legacy execute_tools_node removed (orchestration handles execution)

def retrieve_few_shots_node(state: AgentState, retriever: FewShotRetriever) -> AgentState:
    """Retrieve relevant few-shot examples."""
    logger.info("Retrieving few-shot examples...")
    
    query = f"{state['user_input']} {' '.join(state['selected_tools'])}"
    retrieved = retriever.retrieve(query, top_k=10, rerank_top_k=3)
    
    state["retrieved_few_shots"] = retrieved
    logger.info(f"Retrieved {len(retrieved)} few-shot examples")
    
    return state

# Mapping from function names to semantic keys and executors
## Legacy semantic key map removed; plugins supply semantic_key

## Legacy executor mapping removed; registry handles execution

def llm_tool_orchestration_node(state: AgentState) -> AgentState:
    """Multi-turn planning: plan -> validate -> execute -> summarize -> refine.

    Executes registry tools in parallel using AsyncToolExecutor with retries, rate limits, and caching.
    No human interaction: the LLM adopts explicit assumptions when parameters are ambiguous.
    """
    logger.info("Orchestrating tools with multi-turn planner...")
    _ = _ensure_data_dir(state)

    # Build tool definitions from registry for function calling
    selected = state.get("selected_tools", [])
    registry_schemas: Dict[str, dict] = {}
    try:
        registry_schemas = REGISTRY.get_tool_schemas()
    except Exception:
        registry_schemas = {}
    tool_defs: List[dict] = []
    for name in selected:
        if name in registry_schemas:
            sch = registry_schemas[name]
            tool_defs.append({
                "type": "function",
                "name": sch.get("name", name),
                "description": sch.get("description", ""),
                "parameters": sch.get("parameters", {}),
            })

    # Planner instructions: autonomous assumptions; registry-only tools
    plan_sys = (
        "You are a data extraction planner.\n"
        "- Use ONLY tools from the provided registry.\n"
        "- Do NOT ask the user anything. If parameters are ambiguous, adopt explicit assumptions (JSON list) and continue.\n"
        "- Prefer minimal sufficient calls.\n"
        "- After you receive validation feedback or tool summaries, refine your plan if needed.\n"
        "Return planned tool function calls directly via function-calling."
    )
    system_msg = {"role": "system", "content": plan_sys}
    user_msg = {
        "role": "user",
        "content": (
            f"User request: {state.get('user_input','')}\n"
            f"Task description: {state.get('task_description','')}\n"
            f"Selected tools available: {', '.join(selected)}"
        ),
    }
    messages: List[Dict[str, Any]] = [system_msg, user_msg]

    # Accumulators
    state.setdefault("tool_artifacts", [])
    state.setdefault("tool_files", {})
    state.setdefault("tool_outputs", {})
    state.setdefault("warnings", [])
    state.setdefault("assumptions", [])
    plan_history: List[List[Dict[str, Any]]] = []
    executed_sigs: set[str] = set()
    calls_log: List[Dict[str, Any]] = []
    stop_reason = "budget"

    for turn in range(1, PLANNER_MAX_TURNS + 1):
        logger.info(f"Planner turn {turn} ...")
        resp = openai_client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": "medium"},
            input=messages,
            tools=tool_defs,
            tool_choice="auto",
        )
        items = resp.output or []
        tool_calls = [it for it in items if getattr(it, "type", None) == "function_call"]
        if not tool_calls:
            stop_reason = "no_calls"
            break

        # Build plan from calls with validation
        planned: List[Tuple[str, Dict[str, Any]]] = []
        planned_rec: List[Dict[str, Any]] = []
        issues_report: List[Dict[str, Any]] = []
        for tc in tool_calls:
            tool_name = getattr(tc, "name", None) or (getattr(tc, "function", None).name if getattr(tc, "function", None) else None)
            if not tool_name or tool_name not in registry_schemas:
                continue
            try:
                if hasattr(tc, "arguments"):
                    raw_args = tc.arguments
                elif hasattr(tc, "function") and getattr(tc.function, "arguments", None):
                    raw_args = tc.function.arguments
                else:
                    raw_args = "{}"
                args = json.loads(raw_args) if raw_args else {}
            except Exception:
                args = {}

            ok, fixed, issues = REGISTRY.validate(tool_name, args)
            use_args = fixed or args
            sem_key = REGISTRY.get_semantic_key(tool_name)
            sig = params_signature(tool_name, use_args)
            calls_log.append({"tool": tool_name, "semantic_key": sem_key, "args": use_args, "param_issues": issues or []})
            if issues and not fixed and not ok:
                issues_report.append({"tool": tool_name, "provided": args, "issues": issues})
            else:
                planned.append((tool_name, use_args))
                planned_rec.append({"tool": tool_name, "params": use_args, "sig": sig})

        plan_history.append(planned_rec)

        # If all invalid, feed validation report and continue
        if planned and GLOBAL_TOOL_EXECUTOR is None:
            logger.warning("GLOBAL_TOOL_EXECUTOR not initialized; skipping execution phase.")
            stop_reason = "no_executor"
            break
        if not planned:
            if issues_report:
                messages.append({
                    "role": "user",
                    "content": "VALIDATION_REPORT: " + json.dumps(issues_report),
                })
                continue
            else:
                stop_reason = "no_valid_calls"
                break

        # Check stability: if all sigs already executed, end
        planned_sigs = {params_signature(n, p) for n, p in planned}
        if planned_sigs.issubset(executed_sigs):
            stop_reason = "stable_plan"
            break

        # Execute batch (cached / parallel)
        records = asyncio.run(GLOBAL_TOOL_EXECUTOR.execute_many(planned))
        files_delta: Dict[str, str] = {}
        outputs_delta: Dict[str, Dict[str, Any]] = {}
        executed_tool_names = [name for name, _ in planned]
        # Update executed sigs and attach artifacts
        for rec in records:
            executed_sigs.add(rec.get("sig"))
            sem_key = rec.get("sem_key")
            path = rec.get("artifact_path")
            state["tool_artifacts"].append(rec)
            if sem_key and path:
                files_delta[sem_key] = path
                outputs_delta[sem_key] = _build_dataset_descriptor(path)

        if files_delta or outputs_delta:
            warnings = state.setdefault("warnings", [])
            if files_delta:
                store_files = state.setdefault("tool_files", {})
                for key, new_path in files_delta.items():
                    old_path = store_files.get(key)
                    if old_path and old_path != new_path:
                        warnings.append(f"Semantic key {key} file changed from {old_path} to {new_path}")
                    store_files[key] = new_path
            if outputs_delta:
                store_outputs = state.setdefault("tool_outputs", {})
                expected_fields = {"ok", "file_path", "rows", "columns", "dtypes", "nulls", "date_start", "date_end", "sample_head"}
                for key, payload in outputs_delta.items():
                    old_payload = store_outputs.get(key)
                    if old_payload:
                        old_path = old_payload.get("file_path")
                        new_path = payload.get("file_path")
                        if old_path and new_path and old_path != new_path:
                            warnings.append(f"Semantic key {key} file changed from {old_path} to {new_path}")
                    missing = sorted(list(expected_fields - set(payload.keys())))
                    if missing:
                        warnings.append(f"Semantic key {key} missing fields: {', '.join(missing)}")
                    store_outputs[key] = payload
        # Provide summaries back to the planner
        messages.append({
            "role": "user",
            "content": "TOOL_OUTPUTS_JSON: " + json.dumps(outputs_delta, default=_json_default),
        })

        # Loop continues for potential refinement

    # Finalize orchestration state
    state["tool_calls_log"] = calls_log
    state["tool_results"] = {k: v for k, v in (state.get("tool_outputs", {}) or {}).items()}
    state["planning_turn"] = len(plan_history)
    state["plan_history"] = plan_history
    state["stop_reason"] = stop_reason
    return state


def collect_tool_outputs_node(state: AgentState) -> AgentState:
    """Normalize tool outputs into descriptions and ensure file mapping in state."""
    logger.info("Collecting tool outputs...")
    # Prefer outputs set during orchestration
    if state.get("tool_outputs"):
        return state
    outputs: Dict[str, Any] = {}
    for entry in state.get("tool_calls_log", []):
        sem_key = entry.get("semantic_key")
        desc = entry.get("result", {})
        if not sem_key:
            continue
        outputs[sem_key] = {k: desc.get(k) for k in (
            "ok", "file_path", "rows", "columns", "dtypes", "nulls", "date_start", "date_end", "sample_head"
        )}
    state["tool_outputs"] = outputs
    return state


def build_manifest_node(state: AgentState) -> AgentState:
    """Sanitize or construct a manifest for the current iteration workdir, validate artifacts, and record provenance."""
    logger.info("Building/sanitizing manifest for current iteration...")
    # Default: manifest not ok; will be set True upon success
    state["manifest_ok"] = False

    # Determine iteration working directory
    run_dir = _ensure_data_dir(state)
    iteration_index = state.get("iteration_index")
    if iteration_index is None:
        iteration_index = int(state.get("reflection_count", 0) or 0) + int(state.get("improvement_round", 0) or 0)
    else:
        iteration_index = int(iteration_index)
    state["iteration_index"] = iteration_index
    exec_workdir = os.path.join(run_dir, f"reflection_{iteration_index}")
    os.makedirs(exec_workdir, exist_ok=True)
    state["exec_workdir"] = exec_workdir

    try:
        manifest, warns = sanitize_or_build_manifest(exec_workdir)
        # Validate the listed paths exist
        missing: List[str] = []
        for t in manifest.tables:
            if not os.path.exists(t.path):
                missing.append(t.path)
        for g in manifest.figures:
            if not os.path.exists(g.path):
                missing.append(g.path)
        if missing:
            state["execution_result"] = {"success": False, **(state.get("execution_result", {}) or {})}
            state["error_message"] = f"Manifest lists missing artifacts: {missing}"
            return state

        # Attach and mark ok
        state["result_manifest"] = manifest
        state.setdefault("iteration_manifests", [])
        if len(state["iteration_manifests"]) <= iteration_index:
            state["iteration_manifests"].append(manifest)
        else:
            state["iteration_manifests"][iteration_index] = manifest
        state["final_manifest_path"] = os.path.join(exec_workdir, "result.json")
        state.setdefault("warnings", []).extend(warns or [])

        # Build a serializable manifest summary for graph state (post-run audit)
        # - counts of coercions
        # - included tables/figures with key metadata
        counts: Dict[str, int] = {}
        for w in (warns or []):
            counts[w] = counts.get(w, 0) + 1
        tables_summary = [
            {
                "path": t.path,
                "rows": int(getattr(t, "rows", 0) or 0),
                "columns": (list(getattr(t, "columns", []) or [])[:20]),
                "n_columns": len(getattr(t, "columns", []) or []),
            }
            for t in getattr(manifest, "tables", []) or []
        ]
        figures_summary = [
            {
                "path": f.path,
                "format": getattr(f, "format", None),
                "caption": getattr(f, "caption", None),
            }
            for f in getattr(manifest, "figures", []) or []
        ]
        state["manifest_summary"] = {
            "iteration": iteration_index,
            "coercion_counts": counts,
            "table_count": len(tables_summary),
            "figure_count": len(figures_summary),
            "tables": tables_summary,
            "figures": figures_summary,
            "manifest_path": state["final_manifest_path"],
            "workdir": exec_workdir,
        }
        state["manifest_ok"] = True
        return state
    except Exception as e:
        state["execution_result"] = {"success": False, **(state.get("execution_result", {}) or {})}
        state["error_message"] = f"Manifest build failed: {e}"
        return state


def eval_report_node(state: AgentState) -> AgentState:
    """Generate an evaluation report that guides post-manifest improvements."""

    logger.info("EvalReport: generating eval plan and research queries...")
    eval_in = _eval_input_from_state(state)

    # Add unused tools analysis
    selected_tools = set(state.get("selected_tools", []))
    try:
        all_available = set(REGISTRY.get_available_tools())
        registry_schemas = REGISTRY.get_tool_schemas()
    except Exception:
        all_available = set()
        registry_schemas = {}

    unused_tools = all_available - selected_tools
    unused_tools_context = ""
    if unused_tools:
        tool_descriptions = []
        for tool_name in sorted(unused_tools):
            if tool_name in registry_schemas:
                schema = registry_schemas[tool_name]
                desc = schema.get("description", "")
                tool_descriptions.append(f"• {tool_name}: {desc}")
        unused_tools_context = f"\nAVAILABLE UNUSED TOOLS:\n{chr(10).join(tool_descriptions)}"

    # Enhanced evaluation prompt including unused tools
    enhanced_eval_input = eval_in.copy()
    enhanced_eval_input["unused_tools_context"] = unused_tools_context

    messages = [
        {"role": "system", "content": EVAL_REPORT_SYSTEM + "\n\nAlso consider unused tools that could enhance the analysis.\nReturn ONLY a JSON object that strictly conforms to this JSON Schema: " + json.dumps(_model_schema(EvalReport))},
        {"role": "user", "content": json.dumps(enhanced_eval_input, default=_json_default)},
    ]

    try:
        resp = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=messages,
        )
        raw = _extract_text(resp)
        parsed = EvalReport.model_validate(json.loads(raw))
    except Exception as exc:
        logger.warning(f"EvalReport parse failed: {exc}")
        parsed = EvalReport(
            improvements=[],
            research_goal="(fallback)",
            research_queries=[],
            new_tool_suggestions=[],
            notes="(fallback)",
        )

    state["eval_report"] = parsed.model_dump()
    has_actions = bool(parsed.improvements) or bool(parsed.research_queries) or bool(parsed.new_tool_suggestions)
    state["should_finalize_after_eval"] = not has_actions
    return state


def _run_tavily_for_one_query(rq: ResearchQuery, context: Dict[str, Any]) -> Evidence:
    """Execute Tavily research for a single query specification."""

    payload = {
        "query_id": rq.id,
        "question": rq.question,
        "queries": rq.queries,
        "recency_days": rq.recency_days,
        "k": rq.k,
        "max_results": rq.k,
    }

    input_list: List[Dict[str, Any]] = [
        {"role": "system", "content": TAVILY_QA_SYSTEM},
        {"role": "user", "content": json.dumps(payload)},
    ]
    if context:
        input_list.append({"role": "user", "content": json.dumps({"context": context}, default=_json_default)})

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        tools=TAVILY_TOOL_DEF,
        input=input_list,
    )
    input_list.extend(response.output or [])

    used_query = rq.queries[0]
    for item in response.output or []:
        if getattr(item, "type", "") == "function_call" and getattr(item, "name", "") == "tavily_search":
            try:
                parsed_args = json.loads(item.arguments or "{}")
            except Exception:
                parsed_args = {}
            if "max_results" not in parsed_args and "k" in parsed_args:
                parsed_args["max_results"] = parsed_args.pop("k")
            parsed_args.pop("k", None)
            used_query = parsed_args.get("query", used_query)
            try:
                results = tavily_search(**parsed_args)
                input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({"results": results}, default=_json_default),
                    }
                )
            except Exception as exc:
                input_list.append(
                    {
                        "role": "system",
                        "content": f"(tavily execution error) {exc}",
                    }
                )

    response2 = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=TAVILY_SYNTHESIS_INSTRUCTIONS,
        input=input_list,
    )

    docs: List[Dict[str, Any]] = []
    for msg in input_list:
        if isinstance(msg, dict) and msg.get("type") == "function_call_output":
            try:
                payload = json.loads(msg["output"])
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            for d in payload.get("results", {}).get("results", []):
                docs.append({
                    "title": d.get("title", ""),
                    "url": d.get("url", ""),
                    "snippet": d.get("content") or d.get("snippet"),
                    "published_at": d.get("published_time"),
                    "score": d.get("score"),
                })

    synthesis = _extract_text(response2)
    return Evidence(
        query_id=rq.id,
        used_query=used_query,
        docs=docs,
        synthesis=synthesis,
        caveats=[],
    )


def tavily_research_node(state: AgentState) -> AgentState:
    """Execute Tavily research queries derived from the evaluation report."""

    report_dict = state.get("eval_report") or {}
    try:
        report = EvalReport.model_validate(report_dict)
    except Exception as exc:
        logger.warning(f"EvalReport parse failed in tavily_research_node: {exc}")
        report = EvalReport()

    if state.get("should_finalize_after_eval") or not report.research_queries:
        bundle = ResearchBundle(goal=report.research_goal, queries=[], evidences=[], summary="", conflicts=[], recommended_actions=[])
        state["tavily_results"] = bundle.model_dump()
        return state

    if tavily_client is None:
        logger.warning("Tavily research unavailable; skipping evidence gathering.")
        bundle = ResearchBundle(goal=report.research_goal, queries=report.research_queries, evidences=[], summary="", conflicts=[], recommended_actions=[])
        state["tavily_results"] = bundle.model_dump()
        return state

    evidences: List[Evidence] = []
    context_base = {
        "user_request": state.get("user_input", ""),
        "task_description": state.get("task_description", ""),
        "research_goal": report.research_goal,
        "evaluator_notes": report.notes,
        "warnings": state.get("warnings", []),
        "manifest_summary": state.get("manifest_summary", {}),
        "execution_result": state.get("execution_result", {}),
    }
    for rq in sorted(report.research_queries, key=lambda item: item.priority):
        try:
            query_context = dict(context_base)
            query_context.update({
                "category": rq.category,
                "intent": rq.intent,
                "priority": rq.priority,
                "include_domains": rq.include_domains,
                "exclude_domains": rq.exclude_domains,
                "recency_days": rq.recency_days,
                "citations_required": rq.citations_required,
            })
            evidences.append(_run_tavily_for_one_query(rq, query_context))
        except Exception as exc:
            logger.warning(f"Tavily query failed for {rq.id}: {exc}")

    bundle = ResearchBundle(
        goal=report.research_goal,
        queries=report.research_queries,
        evidences=evidences,
        summary="",
        conflicts=[],
        recommended_actions=[],
    )
    state["tavily_results"] = bundle.model_dump()
    return state


def extra_tool_exec_node(state: AgentState) -> AgentState:
    """Execute additional registry tools suggested by the evaluator."""

    logger.info("Extra tool execution (from suggestions)...")
    if state.get("should_finalize_after_eval"):
        state["delta_tool_outputs"] = {}
        return state

    try:
        # Get evaluation suggestions
        suggestions = (state.get("eval_report", {}) or {}).get("new_tool_suggestions", [])
        if not suggestions:
            state["delta_tool_outputs"] = {}
            return state

        try:
            registry_schemas = REGISTRY.get_tool_schemas()
        except Exception:
            registry_schemas = {}

        # Build tool definitions for function calling
        tool_defs = []
        tool_names = []
        for suggestion in suggestions:
            try:
                ts = ToolSuggestion.model_validate(suggestion)
                tool_name = ts.tool_name
                if tool_name in registry_schemas:
                    sch = registry_schemas[tool_name]
                    tool_defs.append({
                        "type": "function",
                        "name": sch.get("name", tool_name),
                        "description": sch.get("description", ""),
                        "parameters": sch.get("parameters", {}),
                    })
                    tool_names.append(tool_name)
            except Exception:
                continue

        if not tool_defs:
            state["delta_tool_outputs"] = {}
            return state

        # LLM parameterization prompt
        plan_sys = (
            "You are a data extraction planner for newly suggested tools.\n"
            "- Use ONLY the provided tools that were suggested by the evaluator.\n"
            "- Fill parameters based on the user request, generated code, and dataset context.\n"
            "- If parameters are ambiguous, adopt explicit assumptions and continue.\n"
            "- These tools will supplement the existing analysis.\n"
            "Return planned tool function calls directly via function-calling."
        )

        user_content_parts = [
            f"User request: {state.get('user_input','')}",
            f"Task description: {state.get('task_description','')}",
            f"Suggested tools to parameterize: {', '.join(tool_names)}",
        ]

        code_context = _code_context_snippet(state)
        if code_context:
            user_content_parts.append("Current generated code context:\n```python\n" + code_context + "\n```")

        messages = [
            {"role": "system", "content": plan_sys},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ]

        # Multi-turn execution loop
        executed_sigs = set()
        calls_log = []
        new_outputs_delta = {}
        new_files_delta = {}
        executed_tool_names = []
        MAX_TURNS = 2

        for turn in range(1, MAX_TURNS + 1):
            logger.info(f"Tool execution turn {turn} ...")

            resp = openai_client.responses.create(
                model=OPENAI_MODEL,
                reasoning={"effort": "medium"},
                input=messages,
                tools=tool_defs,
                tool_choice="auto",
            )

            items = resp.output or []
            tool_calls = [it for it in items if getattr(it, "type", None) == "function_call"]
            if not tool_calls:
                break

            # Build and validate plan
            planned = []
            issues_report = []

            for tc in tool_calls:
                tool_name = getattr(tc, "name", None) or (getattr(tc, "function", None).name if getattr(tc, "function", None) else None)
                if not tool_name or tool_name not in registry_schemas:
                    continue

                try:
                    if hasattr(tc, "arguments"):
                        raw_args = tc.arguments
                    elif hasattr(tc, "function") and getattr(tc.function, "arguments", None):
                        raw_args = tc.function.arguments
                    else:
                        raw_args = "{}"
                    args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    args = {}

                ok, fixed, issues = REGISTRY.validate(tool_name, args)
                use_args = fixed or args
                sem_key = REGISTRY.get_semantic_key(tool_name)
                sig = params_signature(tool_name, use_args)

                calls_log.append({
                    "tool": tool_name,
                    "semantic_key": sem_key,
                    "args": use_args,
                    "param_issues": issues or []
                })

                if issues and not fixed and not ok:
                    issues_report.append({"tool": tool_name, "provided": args, "issues": issues})
                else:
                    planned.append((tool_name, use_args))

            if not planned:
                if issues_report:
                    messages.append({
                        "role": "user",
                        "content": "VALIDATION_REPORT: " + json.dumps(issues_report),
                    })
                    continue
                else:
                    break

            # Check stability
            planned_sigs = {params_signature(n, p) for n, p in planned}
            if planned_sigs.issubset(executed_sigs):
                break

            # Execute tools
            if GLOBAL_TOOL_EXECUTOR is None:
                logger.warning("GLOBAL_TOOL_EXECUTOR not initialized; skipping execution phase.")
                break

            records = asyncio.run(GLOBAL_TOOL_EXECUTOR.execute_many(planned))

            # Process results and build summaries
            for rec in records:
                executed_sigs.add(rec.get("sig"))
                sem_key = rec.get("sem_key")
                path = rec.get("artifact_path")
                tool_name = rec.get("tool")

                if tool_name:
                    executed_tool_names.append(tool_name)

                if sem_key and path:
                    new_files_delta[sem_key] = path
                    new_outputs_delta[sem_key] = _build_dataset_descriptor(path)

            # Provide summaries back to planner
            if new_outputs_delta:
                messages.append({
                    "role": "user",
                    "content": "TOOL_OUTPUTS_JSON: " + json.dumps(new_outputs_delta, default=_json_default),
                })

        # DUAL STATE TRACKING (Your exact pattern)
        state["unused_tool_picks"] = calls_log
        state["unused_tool_files"] = new_files_delta
        state["unused_tool_outputs"] = new_outputs_delta
        state["unused_tool_names"] = list(set(executed_tool_names))

        # Merge into main state (SAME as orchestrator)
        state.setdefault("tool_files", {}).update(new_files_delta)
        state.setdefault("tool_outputs", {}).update(new_outputs_delta)
        state["selected_tools"] = state.get("selected_tools", []) + state["unused_tool_names"]

        # Also set delta_tool_outputs for compatibility with existing workflow
        state["delta_tool_outputs"] = new_outputs_delta

        logger.info(f"New tools executed: {state['unused_tool_names']}")
        return state

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        state["unused_tool_picks"] = []
        state["unused_tool_files"] = {}
        state["unused_tool_outputs"] = {}
        state["unused_tool_names"] = []
        state["delta_tool_outputs"] = {}
        return state


def delta_code_gen_node(state: AgentState) -> AgentState:
    """Generate a full code replacement that incorporates evaluated improvements."""

    logger.info("Delta code generation with research + tool deltas...")
    if state.get("should_finalize_after_eval"):
        logger.info("Eval report requested no further improvements; skipping delta code generation.")
        state["delta_tool_outputs"] = {}
        return state
    payload = {
        "eval_report": state.get("eval_report", {}),
        "tavily_results": state.get("tavily_results", {}),
        "delta_tool_outputs": list((state.get("delta_tool_outputs", {}) or {}).keys()),
        "unused_tool_outputs": list((state.get("unused_tool_outputs", {}) or {}).keys()),
        "current_code_excerpt": state.get("generated_code", "")[:2000],
        "manifest_summary": state.get("manifest_summary", {}),
        "warnings": state.get("warnings", []),
    }
    messages = [
        {"role": "system", "content": DELTA_CODE_SYSTEM + "\nReturn ONLY a JSON object that strictly conforms to this JSON Schema: " + json.dumps(_model_schema(DeltaCodePatch))},
        {"role": "user", "content": json.dumps(payload, default=_json_default)},
    ]

    try:
        resp = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=messages,
        )
        raw = _extract_text(resp)
        patch = DeltaCodePatch.model_validate(json.loads(raw))
    except Exception as exc:
        logger.warning(f"Delta code generation parse failed: {exc}")
        patch = DeltaCodePatch(
            new_code=state.get("generated_code", ""),
            changes_summary="(fallback)",
            reasoning="",
        )

    parsed_code = CodeResponse.parse_llm_response(patch.new_code).code
    state["generated_code"] = parsed_code
    prev_explanation = state.get("code_explanation", "")
    delta_notes = (prev_explanation + "\nDelta changes:\n" + patch.changes_summary).strip()
    state["code_explanation"] = delta_notes
    state["improvement_round"] = int(state.get("improvement_round", 0)) + 1
    state["eval_report"] = {}
    state["tavily_results"] = {}
    state["delta_tool_outputs"] = {}
    state["should_finalize_after_eval"] = False
    state["manifest_ok"] = False
    state.pop("result_manifest", None)
    state.pop("final_manifest_path", None)
    state.pop("exec_workdir", None)
    return state


class IntermediateContext(BaseModel):
    context: str = Field(description="Guidance text mapping data keys to problem parts with data previews")


def generate_intermediate_context_node(state: AgentState) -> AgentState:
    """Ask LLM to generate usage context for codegen using metadata only.

    Relies on state['tool_outputs'] descriptions: file_path, rows, columns, dtypes,
    nulls, date_start, date_end, sample_head. Does not re-read CSV files.
    """
    logger.info("Generating intermediate context...")
    summary_lines = []
    for key, desc in (state.get("tool_outputs", {}) or {}).items():
        block = {
            "key": key,
            "file_path": desc.get("file_path"),
            "rows": desc.get("rows"),
            "columns": desc.get("columns"),
            "dtypes": desc.get("dtypes"),
            "nulls": desc.get("nulls"),
            "date_start": desc.get("date_start"),
            "date_end": desc.get("date_end"),
            "sample_head": desc.get("sample_head", []),
        }
        summary_lines.append(json.dumps(block, default=_json_default))
    summary = "\n".join(summary_lines) if summary_lines else "(no data)"

    sys = """You are a planning assistant. Based on the available datasets and the user goal, produce a concise plan telling which dataset key to use for which part of the task. Use the provided descriptions (columns, dtypes, dates, sample_head). Output MUST follow this exact format for each dataset you reference:

<csv_header_line>
<csv_sample_row>
col_desc:
<col_name>: <short description>
<col_name>: <short description>
...

Notes: Choose columns relevant to solve the user's request. If the example headers (symbol,publisheddate,pricetarget,adjpricetarget,pricewhenposted) fit the dataset, use them; otherwise, adapt the headers to the dataset's actual columns. Keep descriptions concise."""
    usr = f"""User request: {state.get('user_input','')}
Task description: {state.get('task_description','')}
Available dataset summaries (one JSON object per line):
{summary}
Generate the formatted context blocks for the datasets that should be used."""

    schema =  _model_schema(IntermediateContext)
    ctx_sys = sys + "\nReturn ONLY a JSON object that strictly conforms to this JSON Schema: " + json.dumps(schema)
    resp = openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": ctx_sys},
            {"role": "user", "content": usr},
        ],
    )
    raw = _extract_text(resp)
    obj = json.loads(raw)
    try:
        parsed = IntermediateContext.model_validate(obj)
    except Exception:
        parsed = IntermediateContext.parse_obj(obj)
    state["intermediate_context"] = parsed.context
    return state


def create_prompt_node(state: AgentState) -> AgentState:
    """Create composite prompt with few-shot examples."""
    logger.info("Creating composite prompt...")
    
    # Build few-shot context
    few_shot_context = "\n\n".join([
        f"Example {i+1}:\nDescription: {ex['description']}\nCode:\n```python\n{ex['code']}\n```"
        for i, ex in enumerate(state["retrieved_few_shots"])
    ])
    
    # Build artifacts block
    artifacts = []
    for t, p in (state.get("tool_files", {}) or {}).items():
        artifacts.append(f"- {t}: {p}")
    artifacts_block = "\n".join(artifacts) if artifacts else "(none)"

    # Create composite prompt
    prompt = f"""
    You are a quantitative research code generator. You will receive local artifact paths (CSV/Parquet) and must analyze them. Follow the conventions and the output contract strictly.

    {PLOTLY_CONVENTIONS}

    {BACKTEST_HYGIENE}

    {STATS_RIGOR}

    {OUTPUT_CONTRACT}

    Implementation notes:
    - Use pandas (and optionally polars) and Plotly. Set the Plotly default template to 'plotly_dark'.
    - Load inputs from the artifact list. Prefer pd.read_parquet when a Parquet path is provided; otherwise use pd.read_csv.
    - Persist all tabular outputs under ${'{'}DATA_DIR{'}'}.
    - At the end, write 'result.json' with tables, figures, metrics, explanation as specified.
    - Do not access any network resources; only read local artifacts.

    EXAMPLES:
    {few_shot_context}

    USER REQUEST: {state['user_input']}
    SELECTED TOOLS: {', '.join(state['selected_tools'])}

    INTERMEDIATE USAGE CONTEXT:
    {state.get('intermediate_context','(no context)')}

    AVAILABLE DATA ARTIFACTS (load these files in your code):
    {artifacts_block}
    """
    
    state["composite_prompt"] = prompt
    return state

def generate_code_node(state: AgentState) -> AgentState:
    """Generate code using OpenAI with structured output."""
    logger.info("Generating code...")
    
    # Create completion with structured output (Responses API)
    def _model_schema(m):
        try:
            return m.model_json_schema()
        except Exception:
            return m.schema()
    code_schema = _model_schema(CodeGeneration)
    cg_sys = ("You are an expert Python programmer specializing in data analytics.\n" "Return ONLY a JSON object that strictly conforms to this JSON Schema: " f"%s")
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": cg_sys % json.dumps(code_schema)},
            {"role": "user", "content": state["composite_prompt"]}
        ],
    )
    def _extract_text(resp):
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for seg in getattr(item, "content", []) or []:
                txt = getattr(seg, "text", None)
                if txt:
                    chunks.append(getattr(txt, "value", None) or str(txt))
        return "".join(chunks)
    raw = _extract_text(response)
    obj = json.loads(raw)
    try:
        result = CodeGeneration.model_validate(obj)
    except Exception:
        result = CodeGeneration.parse_obj(obj)
    
    sanitized = CodeResponse.parse_llm_response(result.code).code
    state["generated_code"] = sanitized
    state["code_explanation"] = result.explanation
    
    logger.info("Code generated successfully")
    return state

def execute_code_node(state: AgentState) -> AgentState:
    """Execute generated code with error handling."""
    logger.info("Executing code...")
    
    try:
        # Pre-flight heuristics for output contract adherence
        code = state.get("generated_code", "")
        for s in ["result.json", "plotly_dark", "DATA_DIR"]:
            if s not in code:
                state.setdefault("warnings", []).append(f"Generated code missing hint: {s}")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(state["generated_code"])
            temp_file = f.name
        
        # Prepare per-iteration execution working directory
        run_dir = _ensure_data_dir(state)
        iteration_index = int(state.get("reflection_count", 0) or 0) + int(state.get("improvement_round", 0) or 0)
        state["iteration_index"] = iteration_index
        exec_workdir = os.path.join(run_dir, f"reflection_{iteration_index}")
        os.makedirs(exec_workdir, exist_ok=True)
        state.setdefault("iteration_workdirs", [])
        if len(state["iteration_workdirs"]) <= iteration_index:
            state["iteration_workdirs"].append(exec_workdir)
        else:
            state["iteration_workdirs"][iteration_index] = exec_workdir

        # Execute code with iteration working directory and environment
        current_dir = exec_workdir
        env = os.environ.copy()
        env['PYTHONPATH'] = current_dir + ':' + env.get('PYTHONPATH', '')
        # Provide DATA_DIR for the generated code (per-iteration)
        env['DATA_DIR'] = current_dir
        
        # Select Python interpreter: prefer current interpreter, then PATH fallbacks, then last-resort
        python_executable = sys.executable or ''
        if not (python_executable and os.path.exists(python_executable)):
            # Try project-local .venv first if present
            venv_candidate = os.path.join(os.getcwd(), '.venv', 'bin', 'python')
            if os.path.exists(venv_candidate):
                python_executable = venv_candidate
            else:
                python_executable = shutil.which('python3') or shutil.which('python') or '/usr/bin/python3'
        
        result = subprocess.run(
            [python_executable, temp_file],
            capture_output=True,
            text=True,
            timeout=CODE_EXECUTION_TIMEOUT,
            cwd=current_dir,
            env=env
        )
        
        # Clean up
        os.unlink(temp_file)
        
        if result.returncode == 0:
            # Defer manifest handling to build_manifest_node
            state["execution_result"] = {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            state["error_message"] = ""
            state["exec_workdir"] = current_dir
        else:
            state["execution_result"] = {
                "success": False,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            state["error_message"] = result.stderr
            
    except subprocess.TimeoutExpired:
        state["execution_result"] = {"success": False}
        state["error_message"] = "Code execution timed out"
    except Exception as e:
        state["execution_result"] = {"success": False}
        state["error_message"] = str(e)
    
    logger.info(f"Code execution: {'success' if state['execution_result']['success'] else 'failed'}")
    return state

def reflect_code_node(state: AgentState) -> AgentState:
    """Reflect on and fix code errors using structured output."""
    logger.info("Reflecting on code errors...")
    
    reflection_prompt = f"""
    The following code failed to execute:

    ```python
    {state['generated_code']}
    ```

    Error message:
    {state['error_message']}

    Please fix the code and explain what went wrong.
    """
    
    # Create completion with structured output (Responses API)
    schema = _model_schema(CodeReflection)
    dbg_sys = ("You are an expert Python debugger. Fix the given code.\n" "Return ONLY a JSON object that strictly conforms to this JSON Schema: " f"%s")
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": dbg_sys % json.dumps(schema)},
            {"role": "user", "content": reflection_prompt}
        ],
    )
    raw = _extract_text(response)
    obj = json.loads(raw)
    try:
        result = CodeReflection.model_validate(obj)
    except Exception:
        result = CodeReflection.parse_obj(obj)
    
    sanitized = CodeResponse.parse_llm_response(result.fixed_code).code
    state["generated_code"] = sanitized
    state["code_explanation"] = result.explanation
    state["reflection_count"] = state.get("reflection_count", 0) + 1
    # Reset manifest flags for next iteration
    state["manifest_ok"] = False
    state.pop("result_manifest", None)
    state.pop("final_manifest_path", None)
    state.pop("exec_workdir", None)
    
    logger.info(f"Code reflected (attempt {state['reflection_count']})")
    return state

def finalize_node(state: AgentState) -> AgentState:
    """Finalize the response."""
    logger.info("Finalizing response...")
    
    if state["execution_result"].get("success"):
        parts = [
            "Code executed successfully!",
            state.get("code_explanation", ""),
            f"Output:\n{state['execution_result'].get('stdout','')}",
        ]
        if state.get("exec_workdir"):
            parts.append(f"Artifacts available in: {state['exec_workdir']}")
        if state.get("warnings"):
            parts.append("Warnings: " + ", ".join(state.get("warnings", [])))
        state["final_response"] = "\n\n".join([p for p in parts if p])
    else:
        state["final_response"] = (
            f"Failed to execute code after {state.get('reflection_count', 0)} attempts.\n\n"
            f"Last error: {state.get('error_message','')}"
        )
    
    return state

######################################################################
# LangGraph Workflow Definition
######################################################################

def should_continue_improving(state: AgentState) -> str:
    """Decide whether to trigger the improvement loop after a successful manifest."""

    max_rounds = 3
    if int(state.get("improvement_round", 0)) >= max_rounds:
        return "finalize"
    if state.get("should_finalize_after_eval"):
        return "finalize"
    if not state.get("eval_report"):
        return "eval_report"
    return "eval_report"


def post_execute_router(state: AgentState) -> str:
    """Route after code execution to reflection or manifest building."""

    success = bool(state.get("execution_result", {}).get("success"))
    if success:
        return "build_manifest"
    if state.get("reflection_count", 0) < MAX_REFLECTION_ATTEMPTS:
        return "reflect_code"
    return "finalize"


def post_manifest_router(state: AgentState) -> str:
    """Route the workflow after a manifest build completes."""

    if not state.get("manifest_ok"):
        return "finalize"
    return should_continue_improving(state)

def create_workflow(retriever: FewShotRetriever) -> StateGraph:
    """Create the LangGraph workflow with reflection and improvement loop."""

    wf = StateGraph(AgentState)

    # Core generation path
    wf.add_node("select_tools", select_tools_node)
    wf.add_node("llm_tool_orchestration", llm_tool_orchestration_node)
    wf.add_node("collect_tool_outputs", collect_tool_outputs_node)
    wf.add_node("retrieve_few_shots", lambda state: retrieve_few_shots_node(state, retriever))
    wf.add_node("generate_intermediate_context", generate_intermediate_context_node)
    wf.add_node("create_prompt", create_prompt_node)
    wf.add_node("generate_code", generate_code_node)
    wf.add_node("execute_code", execute_code_node)
    wf.add_node("build_manifest", build_manifest_node)
    wf.add_node("reflect_code", reflect_code_node)
    wf.add_node("finalize", finalize_node)

    # Improvement loop nodes
    wf.add_node("eval_report", eval_report_node)
    wf.add_node("tavily_research", tavily_research_node)
    wf.add_node("extra_tool_exec", extra_tool_exec_node)
    wf.add_node("delta_code_gen", delta_code_gen_node)

    # Linear edges for core flow
    wf.add_edge(START, "select_tools")
    wf.add_edge("select_tools", "llm_tool_orchestration")
    wf.add_edge("llm_tool_orchestration", "collect_tool_outputs")
    wf.add_edge("collect_tool_outputs", "retrieve_few_shots")
    wf.add_edge("retrieve_few_shots", "generate_intermediate_context")
    wf.add_edge("generate_intermediate_context", "create_prompt")
    wf.add_edge("create_prompt", "generate_code")
    wf.add_edge("generate_code", "execute_code")
    wf.add_edge("reflect_code", "execute_code")  # retry after fixes
    wf.add_edge("delta_code_gen", "execute_code")
    wf.add_edge("eval_report", "tavily_research")
    wf.add_edge("tavily_research", "extra_tool_exec")
    wf.add_edge("extra_tool_exec", "delta_code_gen")
    wf.add_edge("finalize", END)

    # Route execution failures to reflection before manifest work
    wf.add_conditional_edges(
        "execute_code",
        post_execute_router,
        {
            "build_manifest": "build_manifest",
            "reflect_code": "reflect_code",
            "finalize": "finalize",
        },
    )

    # Decide on iterative improvement after each successful manifest
    wf.add_conditional_edges(
        "build_manifest",
        post_manifest_router,
        {
            "finalize": "finalize",
            "eval_report": "eval_report",
        },
    )

    return wf

######################################################################
# Main Agent Class
######################################################################

class DataAnalyticsAgent:
    """Main data analytics agent with LangGraph orchestration."""
    
    def __init__(self, few_shots_dir: str = None):
        # Resolve few-shots directory (absolute path requested by user)
        default_fs_dir = os.environ.get(
            "FEW_SHOTS_DIR",
            "/Users/aaryangoyal/Desktop/coffee_code/data_analytics_agent/few_shots/",
        )
        fs_dir = few_shots_dir or default_fs_dir
        self.few_shot_retriever = FewShotRetriever(fs_dir)
        # Registry is initialized globally as REGISTRY
        self.workflow = create_workflow(self.few_shot_retriever)
        self.app = self.workflow.compile()
        # Base data directory and async executor
        base_dir = os.environ.get("AGENT_DATA_DIR", os.path.join(os.getcwd(), ".agent_data"))
        os.makedirs(base_dir, exist_ok=True)
        self.base_data_dir = base_dir
        global GLOBAL_BASE_DATA_DIR
        GLOBAL_BASE_DATA_DIR = base_dir
        # Initialize a global AsyncToolExecutor instance if available
        global GLOBAL_TOOL_EXECUTOR
        try:
            if AsyncToolExecutor is not None:
                GLOBAL_TOOL_EXECUTOR = AsyncToolExecutor(registry=REGISTRY, data_dir=base_dir, tool_overrides={})
        except Exception as e:
            logger.warning(f"Failed to initialize AsyncToolExecutor: {e}")
        
        logger.info("DataAnalyticsAgent initialized successfully")
    
    def process_request(self, user_input: str, task_description: str = None) -> str:
        """Process a user request through the complete workflow."""
        logger.info(f"Processing request: {user_input}")
        
        # Initialize state
        initial_state = AgentState(
            user_input=user_input,
            task_description=task_description or "",
            selected_tools=[],
            retrieved_few_shots=[],
            composite_prompt="",
            generated_code="",
            code_explanation="",
            execution_result={},
            error_message="",
            reflection_count=0,
            final_response="",
            intermediate_results={},
            tool_params={},
            tool_results={},
            tool_files={},
            data_dir="",
            tool_calls_log=[],
            tool_outputs={},
            intermediate_context="",
            # Phase 1 additions
            run_id=str(uuid.uuid4()),
            warnings=[],
            iteration_manifests=[],
            iteration_index=0,
            eval_report={},
            tavily_results={},
            delta_tool_outputs={},
            improvement_round=0,
            should_finalize_after_eval=False,
            # Unused tools tracking fields
            unused_tool_picks=[],
            unused_tool_files={},
            unused_tool_outputs={},
            unused_tool_names=[],
            # Critical missing field initializations
            tool_artifacts=[],
            result_manifest=None,
            iteration_workdirs=[],
            final_manifest_path="",
            exec_workdir="",
            manifest_ok=False,
            planning_turn=0,
            plan_history=[],
            assumptions=[],
            stop_reason="",
            run_dir="",
        )
        # Provide base_data_dir in state for directory resolution
        initial_state["base_data_dir"] = getattr(self, "base_data_dir", os.environ.get("AGENT_DATA_DIR", os.path.join(os.getcwd(), ".agent_data")))
        
        # Execute workflow
        try:
            final_state = self.app.invoke(initial_state)
            _ = _save_graph_state(initial_state, final_state)
            return final_state["final_response"]
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return f"Error processing request: {str(e)}"

######################################################################
# Example Usage
######################################################################

if __name__ == "__main__":
    # Initialize agent
    agent = DataAnalyticsAgent()
    
    # # Example request
    # response = agent.process_request(
    #     "For the period of 2024-01-01 to 2025-08-01 can you please create moving average crossover of 10 day vs 50 day for AAPL and TSLA, once done please create a trading strategy for the same that goes long on these stocks if the 10 day moving average crosses above the 50 day moving average, and goes short if the 10 day moving average crosses below the 50 day moving average, and make a chart for the same"
    # )

        # Example request
    response = agent.process_request(
        # "For the last 3 years can you extract P/E ration and other financials for AAPL, TSLA, NVDA, GOOGL, AMZN, MSFT, NFLX, and create a strategy where you rank P/E ratio of the stocks and go long on the stocks with the lowest P/E ratio and go short on the stocks with the highest P/E ratio"
        "For the last 4 years can you extract US CPI data on monthly basis and then calculate the rolling 2 year beta of AAPL, TSLA, NVDA, GOOGL, AMZN, MSFT, NFLX returns with CPI change then, and backtest a strategy by  rank ordering them each month and go long on stocks with low rank values and short stock with high rank values"
    )
    
    print(response)
