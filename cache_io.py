"""
Caching helpers for neighborhoods agglomeration.

Stores/loads adjacency and initial edge scores, with a lightweight
metadata key to validate cache correctness.

We use Parquet for data frames and JSON for metadata. No dependency
on project-specific classes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import pandas as pd

ALGO_VERSION = "2025-11-14a"  # bump when adjacency/score logic changes


def _file_fingerprint(path: str) -> dict:
    try:
        p = Path(path)
        st = p.stat()
        return {
            "path": str(p.resolve()),
            "size": int(st.st_size),
            "mtime": float(st.st_mtime),
        }
    except Exception:
        return {"path": path, "size": None, "mtime": None}


def _tiles_signature(tiles_gdf) -> dict:
    try:
        keys = tiles_gdf["tile_key"].astype(str)
    except Exception:
        # Fallback if structure unexpected
        try:
            keys = tiles_gdf["key"].astype(str)
        except Exception:
            return {"n_tiles": None, "keys_hash": None}
    keys_list = list(keys.values if hasattr(keys, "values") else keys)
    keys_sorted = sorted(keys_list)
    h = hashlib.sha256("\n".join(keys_sorted).encode("utf-8")).hexdigest()
    return {
        "n_tiles": int(len(keys_sorted)),
        "keys_hash": h,
    }


def make_cache_key(
    parcels_path: str,
    tiles_path: str,
    buffer_feet: float,
    crs_epsg_feet: Optional[int],
    k_neighbors: int,
    tiles_after_filter,
) -> dict:
    return {
        "parcels": _file_fingerprint(parcels_path),
        "tiles": _file_fingerprint(tiles_path),
        "params": {
            "buffer_feet": float(buffer_feet),
            "crs_epsg_feet": int(crs_epsg_feet) if crs_epsg_feet is not None else None,
            "k_neighbors": int(k_neighbors),
        },
        "tiles_sig": _tiles_signature(tiles_after_filter),
        "algo_version": ALGO_VERSION,
    }


def cache_paths(cache_dir: str, prefix: str) -> dict:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "meta": str(d / f"{prefix}_meta.json"),
        "adj": str(d / f"{prefix}_adj.parquet"),
        "scores": str(d / f"{prefix}_scores.parquet"),
    }


def save_cache(
    adjacency: Dict[str, Set[str]],
    edge_scores: Dict[frozenset, object],
    meta: dict,
    cache_dir: str = ".cache",
    prefix: str = "init",
) -> None:
    paths = cache_paths(cache_dir, prefix)

    # Save metadata
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Save adjacency as undirected edge list a<b
    rows_adj = []
    for a, nbs in adjacency.items():
        for b in nbs:
            a_s = str(a); b_s = str(b)
            if a_s < b_s:
                rows_adj.append({"tile_a": a_s, "tile_b": b_s})
    df_adj = pd.DataFrame(rows_adj)
    df_adj.to_parquet(paths["adj"])

    # Save edge scores
    rows_sc = []
    for k, es in edge_scores.items():
        try:
            a, b = sorted(list(k))  # type: ignore
        except Exception:
            continue
        # Accept either dataclass-like or dict-like objects
        r2 = getattr(es, "r2", None)
        n_obs = getattr(es, "n_obs", None)
        n_sales = getattr(es, "n_sales", None)
        if r2 is None and isinstance(es, dict):
            r2 = es.get("r2", 0.0)
            n_obs = es.get("n_obs", 0)
            n_sales = es.get("n_sales", 0)
        rows_sc.append({
            "tile_a": str(a),
            "tile_b": str(b),
            "r2": float(r2 if r2 is not None else 0.0),
            "n_obs": int(n_obs if n_obs is not None else 0),
            "n_sales": int(n_sales if n_sales is not None else 0),
        })
    df_sc = pd.DataFrame(rows_sc)
    df_sc.to_parquet(paths["scores"])


def load_cache(cache_dir: str = ".cache", prefix: str = "init") -> Tuple[Optional[dict], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    paths = cache_paths(cache_dir, prefix)
    try:
        meta = None
        meta_path = Path(paths["meta"]) 
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        df_adj = pd.read_parquet(paths["adj"]) if Path(paths["adj"]).exists() else None
        df_scores = pd.read_parquet(paths["scores"]) if Path(paths["scores"]).exists() else None
        return meta, df_adj, df_scores
    except Exception:
        return None, None, None


def cache_valid(cached_meta: dict, current_meta: dict) -> bool:
    try:
        return (
            cached_meta.get("algo_version") == current_meta.get("algo_version")
            and cached_meta.get("params") == current_meta.get("params")
            and cached_meta.get("tiles_sig") == current_meta.get("tiles_sig")
            and cached_meta.get("parcels", {}).get("mtime") == current_meta.get("parcels", {}).get("mtime")
            and cached_meta.get("parcels", {}).get("size") == current_meta.get("parcels", {}).get("size")
            and cached_meta.get("tiles", {}).get("mtime") == current_meta.get("tiles", {}).get("mtime")
            and cached_meta.get("tiles", {}).get("size") == current_meta.get("tiles", {}).get("size")
        )
    except Exception:
        return False


def df_to_adjacency(df_adj: pd.DataFrame) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {}
    if df_adj is None or len(df_adj) == 0:
        return adj
    for _, row in df_adj.iterrows():
        a = str(row["tile_a"]) ; b = str(row["tile_b"])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def df_to_edge_scores(df_scores: pd.DataFrame):
    out: Dict[frozenset, dict] = {}
    if df_scores is None or len(df_scores) == 0:
        return out
    for _, row in df_scores.iterrows():
        a = str(row.get("tile_a"))
        b = str(row.get("tile_b"))
        if not a or not b:
            continue
        out[frozenset([a, b])] = {
            "r2": float(row.get("r2", 0.0)),
            "n_obs": int(row.get("n_obs", 0)),
            "n_sales": int(row.get("n_sales", 0)),
        }
    return out
