import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Optional

def ensure_feet_crs(gdf: gpd.GeoDataFrame, target_epsg: Optional[int]) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        warnings.warn("Input GeoDataFrame has no CRS; proceeding without reprojection. Distances may be incorrect.")
        return gdf
    if target_epsg is None:
        # Try to infer unit; many commonly used projected CRSs are in meters. If meters, we convert feet->meters when buffering later.
        return gdf
    # if gdf.crs.to_epsg() == target_epsg:
    #     return gdf
    try:
        return gdf.to_crs(epsg=target_epsg)
    except Exception as e:
        warnings.warn(f"Failed to reproject to EPSG:{target_epsg}: {e}. Proceeding in original CRS.")
        return gdf

def compute_market_value_proxy(
        parcels: gpd.GeoDataFrame,
        sale_field="adj_sale_price",
        assessed_field="assessed_value"
) -> pd.Series:
    sale = parcels.get(sale_field)
    assessed = parcels.get(assessed_field)
    if sale is None or sale.isna().all():
        proxy = assessed.astype(float)
    else:
        sale = sale.astype(float)
        assessed = assessed.astype(float)
        both = (~sale.isna()) & (~assessed.isna())
        proxy = assessed.copy().astype(float)
        proxy[both] = (sale[both] + assessed[both]) / 2.0
        # where sale is nan, keep assessed; where assessed is nan but sale exists, use sale
        only_sale = (~sale.isna()) & (assessed.isna())
        proxy[only_sale] = sale[only_sale]
    return proxy

def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    # Add intercept
    X_ = np.column_stack([np.ones(len(X)), X])
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X_, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    yhat = X_.dot(beta)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    # Guard against numerical issues
    if not np.isfinite(r2):
        return 0.0
    return float(max(min(r2, 1.0), -1.0))