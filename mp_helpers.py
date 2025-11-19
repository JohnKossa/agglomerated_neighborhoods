"""
Multiprocessing helpers for neighborhoods agglomeration.

These functions are defined in a regular Python module so that Windows
multiprocessing (spawn) can import them in child processes.
"""
from __future__ import annotations

from typing import Dict, Set, Tuple, Optional, List

import numpy as np
import pandas as pd

# ---------------------------
# Globals shared in workers
# ---------------------------
# For adjacency: list of buffered geometries (Shapely geometries)
_BUFFERED_GEOMS: Optional[List[object]] = None

# For edge scoring: slim parcels dataframe and tile->parcel index mapping
_PARCELS_MIN_DF: Optional[pd.DataFrame] = None
_PARCEL_IDX_BY_TILE: Optional[Dict[str, Set[int]]] = None


# ---------------------------
# Initializers
# ---------------------------

def _init_pool_buffered(geoms_list: List[object]):
    """Pool initializer to set the buffered geometries list.
    The list is treated as read-only by worker processes.
    """
    global _BUFFERED_GEOMS
    _BUFFERED_GEOMS = geoms_list


def _init_pool_edges(parcels_min_df: pd.DataFrame, parcel_idx_by_tile: Dict[str, Set[int]]):
    """Initializer for multiprocessing pool for edge scoring.
    Stores minimal read-only data in module globals for workers.
    """
    global _PARCELS_MIN_DF, _PARCEL_IDX_BY_TILE
    _PARCELS_MIN_DF = parcels_min_df
    _PARCEL_IDX_BY_TILE = parcel_idx_by_tile


# ---------------------------
# Worker functions
# ---------------------------

def _pair_overlaps_area(pair: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Given indices (i, j), return the pair if buffered geometries overlap with
    positive area; otherwise return None. Safe against exceptions.
    """
    try:
        i, j = pair
        inter = _BUFFERED_GEOMS[i].intersection(_BUFFERED_GEOMS[j])  # type: ignore[index]
        if getattr(inter, "area", 0.0) > 0:
            return (i, j)
        return None
    except Exception:
        return None


def _score_edge_pair_worker(pair: Tuple[str, str]) -> Tuple[str, str, float, int, int]:
    """Compute EdgeScore components for a pair of tiles using globals set by _init_pool_edges.
    Returns (a, b, r2, n_obs, n_sales). Robust to errors and missing data.
    """
    try:
        a, b = pair
        mapping = _PARCEL_IDX_BY_TILE or {}
        idxs = list((mapping.get(a, set()) or set()) | (mapping.get(b, set()) or set()))
        if len(idxs) == 0:
            return a, b, 0.0, 0, 0
        sub = _PARCELS_MIN_DF.iloc[idxs]
        n_sales = int(sub["adj_sale_price"].notna().sum()) if "adj_sale_price" in sub.columns else 0
        if n_sales < 3:
            return a, b, 0.0, int(len(sub)), n_sales
        cols = ["market_value_proxy", "built_area_sqft", "land_area_sqft"]
        if not all(c in sub.columns for c in cols):
            return a, b, 0.0, int(len(sub)), n_sales
        df = sub[cols].astype(float).dropna()
        if len(df) < 3:
            return a, b, 0.0, int(len(sub)), n_sales
        y = df["market_value_proxy"].values
        X = df[["built_area_sqft", "land_area_sqft"]].values
        # Compute OLS R^2 with intercept
        X_ = np.column_stack([np.ones(len(X)), X])
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X_, y, rcond=None)
            yhat = X_.dot(beta)
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 0.0 if ss_tot <= 0 else 1.0 - (ss_res / ss_tot)
        except Exception:
            r2 = 0.0
        if not np.isfinite(r2):
            r2 = 0.0
        # Clamp r2
        r2 = float(max(min(r2, 1.0), -1.0))
        return a, b, r2, int(len(df)), n_sales
    except Exception:
        # On any error, return safe defaults so the main process can continue
        try:
            a, b = pair
        except Exception:
            a = b = ""
        return a, b, 0.0, 0, 0
