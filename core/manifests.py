from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json


@dataclass
class TableArtifact:
    path: str
    rows: int
    columns: List[str]
    schema: Optional[Dict[str, str]] = None
    description: Optional[str] = None


@dataclass
class FigureArtifact:
    path: str
    caption: Optional[str] = None
    format: Optional[str] = None  # html/png/svg


@dataclass
class ResultManifest:
    tables: List[TableArtifact]
    figures: List[FigureArtifact]
    metrics: Dict[str, Any]
    explanation: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_file(path: str) -> "ResultManifest":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for key in ["tables", "figures", "metrics", "explanation"]:
            if key not in obj:
                raise ValueError(f"Manifest missing `{key}`")
        tables = [TableArtifact(**t) for t in obj.get("tables", [])]
        figures = [FigureArtifact(**g) for g in obj.get("figures", [])]
        return ResultManifest(
            tables=tables,
            figures=figures,
            metrics=obj.get("metrics", {}),
            explanation=obj.get("explanation", ""),
        )
