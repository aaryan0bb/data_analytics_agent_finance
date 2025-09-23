from __future__ import annotations
import os
import json
import gzip
from typing import Optional, Any, Dict

import pandas as pd
from filelock import FileLock

from .utils import params_signature


class DataCache:
    """
    Two-layer cache rooted at a base directory:
      1) raw_cache/{tool}/{sig}.json.gz      -- small JSON envelopes
      2) semantic_cache/{sem_key}/{sig}.parquet -- normalized tabular outputs
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.raw_root = os.path.join(base_dir, "raw_cache")
        self.sem_root = os.path.join(base_dir, "semantic_cache")
        os.makedirs(self.raw_root, exist_ok=True)
        os.makedirs(self.sem_root, exist_ok=True)

    # ---------- RAW ----------
    def raw_path(self, tool: str, sig: str) -> str:
        d = os.path.join(self.raw_root, tool)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{sig}.json.gz")

    def raw_get(self, tool: str, sig: str) -> Optional[Dict[str, Any]]:
        p = self.raw_path(tool, sig)
        if not os.path.exists(p):
            return None
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)

    def raw_set(self, tool: str, sig: str, payload: Dict[str, Any]) -> None:
        p = self.raw_path(tool, sig)
        lock = FileLock(p + ".lock")
        with lock:
            tmp = p + ".tmp"
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, p)

    # ---------- SEMANTIC (DataFrame) ----------
    def sem_path(self, sem_key: str, sig: str) -> str:
        d = os.path.join(self.sem_root, sem_key)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{sig}.parquet")

    def sem_get(self, sem_key: str, sig: str) -> Optional[pd.DataFrame]:
        p = self.sem_path(sem_key, sig)
        if not os.path.exists(p):
            return None
        return pd.read_parquet(p)

    def sem_set(self, sem_key: str, sig: str, df: pd.DataFrame) -> str:
        p = self.sem_path(sem_key, sig)
        lock = FileLock(p + ".lock")
        with lock:
            tmp = p + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, p)
        return p

    # ---------- Convenience ----------
    def signature(self, tool_name: str, params: dict) -> str:
        return params_signature(tool_name, params)

