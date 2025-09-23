from __future__ import annotations
import os
import json
from typing import List, Tuple, Dict, Any

import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None  # type: ignore

from .manifests import ResultManifest, TableArtifact, FigureArtifact


def _infer_format(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in {"parquet", "pq"}: return "parquet"
    if ext in {"csv"}: return "csv"
    if ext in {"json"}: return "json"
    if ext in {"html", "png", "svg"}: return ext
    return None


def _rows_columns_for_table(path: str) -> Tuple[int, List[str]]:
    """Return (rows, columns) for a tabular artifact. Prefer Parquet metadata."""
    try:
        low = path.lower()
        if low.endswith((".parquet", ".pq")) and pq is not None:
            try:
                pf = pq.ParquetFile(path)
                rows = int(pf.metadata.num_rows) if pf.metadata is not None else 0
                # Column names via schema if available
                cols = [str(n) for n in pf.schema.names] if getattr(pf, "schema", None) else []
                return rows, cols
            except Exception:
                pass
        # Fallback: load first to get shape
        if low.endswith((".parquet", ".pq")):
            df = pd.read_parquet(path)
        elif low.endswith(".csv"):
            df = pd.read_csv(path)
        elif low.endswith(".json"):
            df = pd.read_json(path)
        else:
            return 0, []
        return int(len(df)), [str(c) for c in df.columns]
    except Exception:
        return 0, []


def sanitize_or_build_manifest(exec_workdir: str) -> Tuple[ResultManifest, List[str]]:
    """Load result.json if present; sanitize entries; else build a minimal manifest by scanning exec_workdir.

    Returns (ResultManifest, warnings).
    """
    warnings: List[str] = []
    manifest_path = os.path.join(exec_workdir, "result.json")

    # Helper to coerce table entries
    def _coerce_tables(raw_tables: List[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in raw_tables or []:
            if isinstance(t, str):
                rows, cols = _rows_columns_for_table(t)
                out.append({"path": t, "rows": rows, "columns": cols})
                warnings.append("coerced_table_string")
            elif isinstance(t, dict):
                path = t.get("path")
                if not path:
                    warnings.append("table_missing_path")
                    continue
                rows = int(t.get("rows") or 0)
                cols = t.get("columns")
                if not rows or not cols:
                    r2, c2 = _rows_columns_for_table(path)
                    rows = rows or r2
                    cols = cols or c2
                    warnings.append("filled_table_shape")
                out.append({"path": path, "rows": int(rows), "columns": [str(c) for c in (cols or [])]})
            # else ignore unknown types
        return out

    # Helper to coerce figure entries
    def _coerce_figures(raw_figs: List[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for g in raw_figs or []:
            if isinstance(g, str):
                out.append({"path": g, "caption": None, "format": _infer_format(g)})
                warnings.append("coerced_figure_string")
            elif isinstance(g, dict):
                path = g.get("path")
                if not path:
                    warnings.append("figure_missing_path")
                    continue
                fmt = g.get("format") or _infer_format(path)
                out.append({"path": path, "caption": g.get("caption"), "format": fmt})
            # else ignore unknown types
        return out

    if os.path.exists(manifest_path):
        # Sanitize existing manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        tables = _coerce_tables(raw.get("tables", []) or [])
        figures = _coerce_figures(raw.get("figures", []) or [])
        metrics = raw.get("metrics", {}) or {}
        explanation = str(raw.get("explanation", ""))

        manifest = ResultManifest(
            tables=[TableArtifact(**t) for t in tables],
            figures=[FigureArtifact(**g) for g in figures],
            metrics=metrics,
            explanation=explanation,
        )
        # Rewrite sanitized manifest
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(manifest.to_json())
        except Exception:
            pass
        return manifest, warnings

    # Build from directory scan
    table_paths: List[str] = []
    fig_paths: List[str] = []
    for root, _, files in os.walk(exec_workdir):
        for fn in files:
            p = os.path.join(root, fn)
            low = fn.lower()
            if low.endswith((".parquet", ".pq", ".csv", ".json")):
                table_paths.append(p)
            elif low.endswith((".html", ".png", ".svg")):
                fig_paths.append(p)

    tables = [{"path": p, "rows": _rows_columns_for_table(p)[0], "columns": _rows_columns_for_table(p)[1]} for p in table_paths]
    figures = [{"path": p, "caption": None, "format": _infer_format(p)} for p in fig_paths]

    manifest = ResultManifest(
        tables=[TableArtifact(**t) for t in tables],
        figures=[FigureArtifact(**g) for g in figures],
        metrics={},
        explanation="",  # code explanation (if any) can be added by caller
    )
    # Write new manifest
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.to_json())
    except Exception:
        pass
    return manifest, warnings

