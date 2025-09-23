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
from typing import List, Dict, Any, Optional, Tuple
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

from openai import OpenAI
from pydantic import BaseModel, Field
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
# Environment setup
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
    # Iteration-scoped execution + manifest
    planning_turn: int
    plan_history: List[List[Dict[str, Any]]]
    assumptions: List[Dict[str, Any]]
    stop_reason: str
    iteration_workdirs: List[str]
    final_manifest_path: str
    exec_workdir: str
    manifest_ok: bool
    # Multi-turn planner fields
    planning_turn: int
    plan_history: List[List[Dict[str, Any]]]
    assumptions: List[Dict[str, Any]]
    stop_reason: str

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
    
    state["selected_tools"] = result["selected_tools"]
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
        # Update executed sigs and attach artifacts
        outputs_delta: Dict[str, Any] = {}
        for rec in records:
            executed_sigs.add(rec.get("sig"))
            sem_key = rec.get("sem_key")
            path = rec.get("artifact_path")
            state["tool_artifacts"].append(rec)
            if sem_key and path:
                state["tool_files"][sem_key] = path
                try:
                    df = pd.read_parquet(path)
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
                    outputs_delta[sem_key] = {
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
                except Exception as e:
                    outputs_delta[sem_key] = {"ok": False, "file_path": path, "error": str(e)}

        # Merge new outputs
        state["tool_outputs"].update(outputs_delta)
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
    iteration = int(state.get("reflection_count", 0) or 0)
    exec_workdir = os.path.join(run_dir, f"reflection_{iteration}")
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
        if len(state["iteration_manifests"]) <= iteration:
            state["iteration_manifests"].append(manifest)
        else:
            state["iteration_manifests"][iteration] = manifest
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
            "iteration": iteration,
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
        iteration = int(state.get("reflection_count", 0) or 0)
        exec_workdir = os.path.join(run_dir, f"reflection_{iteration}")
        os.makedirs(exec_workdir, exist_ok=True)
        state.setdefault("iteration_workdirs", [])
        if len(state["iteration_workdirs"]) <= iteration:
            state["iteration_workdirs"].append(exec_workdir)
        else:
            state["iteration_workdirs"][iteration] = exec_workdir

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

def should_reflect(state: AgentState) -> str:
    """Determine if code reflection is needed."""
    success = bool(state.get("execution_result", {}).get("success"))
    manifest_ok = bool(state.get("manifest_ok", success))  # default ok if success and not set
    if success and manifest_ok:
        return "finalize"
    elif state.get("reflection_count", 0) < MAX_REFLECTION_ATTEMPTS:
        return "reflect_code"
    else:
        return "finalize"

def create_workflow(retriever: FewShotRetriever) -> StateGraph:
    """Create the LangGraph workflow."""
    
    # Create workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("select_tools", select_tools_node)
    workflow.add_node("llm_tool_orchestration", llm_tool_orchestration_node)
    workflow.add_node("collect_tool_outputs", collect_tool_outputs_node)
    workflow.add_node("generate_intermediate_context", generate_intermediate_context_node)
    workflow.add_node("retrieve_few_shots", lambda state: retrieve_few_shots_node(state, retriever))
    workflow.add_node("create_prompt", create_prompt_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("build_manifest", build_manifest_node)
    workflow.add_node("reflect_code", reflect_code_node)
    workflow.add_node("finalize", finalize_node)
    
    # Add edges
    workflow.add_edge(START, "select_tools")
    workflow.add_edge("select_tools", "llm_tool_orchestration")
    workflow.add_edge("llm_tool_orchestration", "collect_tool_outputs")
    workflow.add_edge("collect_tool_outputs", "retrieve_few_shots")
    workflow.add_edge("retrieve_few_shots", "generate_intermediate_context")
    workflow.add_edge("generate_intermediate_context", "create_prompt")
    workflow.add_edge("create_prompt", "generate_code")
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", "build_manifest")
    
    # Conditional edge for reflection (after manifest build)
    workflow.add_conditional_edges(
        "build_manifest",
        should_reflect,
        {
            "finalize": "finalize",
            "reflect_code": "reflect_code",
        },
    )
    workflow.add_edge("reflect_code", "execute_code")  # Re-execute after reflection
    workflow.add_edge("finalize", END)
    
    return workflow

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
