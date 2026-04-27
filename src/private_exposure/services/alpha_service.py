from __future__ import annotations
from typing import NamedTuple
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_IC          = 0.05
_FALLBACK_VOL = 0.30
_CACHE_TTL   = 86_400
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0

class _CacheEntry(NamedTuple):
    log_rets: np.ndarray
    mom: float
    vol: float
    ts: float

_cache: dict[str, _CacheEntry | None] = {}
# None = known missing; absent = never fetched


def _fetch_prices(tickers: list[str]) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=400)
    for attempt in range(_MAX_RETRIES):
        try:
            df = yf.download(tickers, start=start, end=end,
                             auto_adjust=True, progress=False, threads=True)
            if df is None or df.empty:
                return pd.DataFrame()
            close = df["Close"] if "Close" in df.columns else pd.DataFrame()
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            return close if isinstance(close, pd.DataFrame) else pd.DataFrame()
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                logger.warning("yfinance attempt %d failed: %s — retrying", attempt + 1, e)
                time.sleep(_RETRY_DELAY)
            else:
                logger.error("yfinance failed after %d attempts: %s", _MAX_RETRIES, e)
    return pd.DataFrame()


def _refresh_cache(tickers: list[str]) -> None:
    prices = _fetch_prices(tickers)
    now = time.time()
    for ticker in tickers:
        if prices.empty or ticker not in prices.columns:
            logger.debug("No price data for %s — marking missing", ticker)
            _cache[ticker] = None
            continue
        series: pd.Series = prices[ticker].dropna()
        if len(series) < 60:
            _cache[ticker] = None
            continue
        log_ret_series: pd.Series = np.log(series / series.shift(1)).dropna()
        log_rets: np.ndarray = log_ret_series.to_numpy()
        lookback = min(252, len(series) - 1)
        mom = float(series.iloc[-21] / series.iloc[-lookback] - 1)
        vol = float(log_rets[-60:].std() * np.sqrt(252)) if len(log_rets) >= 60 else _FALLBACK_VOL
        _cache[ticker] = _CacheEntry(log_rets, mom, vol, now)


def _ensure_cached(tickers: list[str]) -> None:
    now = time.time()
    stale = []
    for t in tickers:
        entry = _cache.get(t)
        if entry is None and t not in _cache:
            stale.append(t)  # never fetched
        elif isinstance(entry, _CacheEntry) and now - entry.ts > _CACHE_TTL:
            stale.append(t)  # fetched but expired
    if stale:
        _refresh_cache(stale)


# --- public API ---

def get_return_series(tickers: list[str]) -> dict[str, np.ndarray]:
    _ensure_cached(tickers)
    return {t: e.log_rets for t in tickers if (e := _cache.get(t)) is not None}


def get_signals(tickers: list[str]) -> dict[str, tuple[float, float]]:
    _ensure_cached(tickers)
    return {t: (e.mom, e.vol) for t in tickers if (e := _cache.get(t)) is not None}


def compute_alphas(tickers: list[str], cov_matrix: np.ndarray | None = None) -> dict[str, float]:
    """
    12-1 month momentum → rank → z-score → Σ-risk-normalised alpha.
    Mirrors production alpha procedure without beta neutralisation.
    If cov_matrix is None falls back to diagonal vol-scaling.
    """
    raw = get_signals(tickers)
    if not raw:
        return {}

    keys = list(raw)
    n    = len(keys)

    # --- momentum z-score (rank-based, same as production) ---
    moms   = np.array([raw[t][0] for t in keys])
    ranked = moms.argsort().argsort().astype(float)          # rank in [0, n-1]
    w      = (ranked - ranked.mean()) / (ranked.std() + 1e-8)  # z-score

    if cov_matrix is not None and cov_matrix.shape == (n, n):
        try:
            # risk-normalise: v = w / sqrt(w^T Σ w), then alpha = Σv
            # this gives Σ-adjusted scores in expected-return units
            port_var = float(w @ cov_matrix @ w)
            v        = w / np.sqrt(max(port_var, 1e-8))
            alphas   = _IC * (cov_matrix @ v)
        except np.linalg.LinAlgError:
            logger.warning("Cov normalisation failed — falling back to vol-scaling")
            vols   = np.array([raw[t][1] for t in keys])
            alphas = _IC * vols * w
    else:
        vols   = np.array([raw[t][1] for t in keys])
        alphas = _IC * vols * w

    return dict(zip(keys, alphas.tolist()))


def compute_covariance(tickers: list[str]) -> np.ndarray:
    """Annualised, PSD-regularised covariance matrix for tickers (n x n)."""
    n    = len(tickers)
    rets = get_return_series(tickers)
    vols = get_signals(tickers)

    if rets:
        min_len = min(len(rets[t]) for t in tickers if t in rets)
        if min_len >= 2:
            mat     = np.stack(
                [rets[t][-min_len:] if t in rets else np.zeros(min_len) for t in tickers],
                axis=1,
            )
            cov_raw = np.cov(mat, rowvar=False) * 12
            return cov_raw + 1e-4 * np.eye(n)

    # fallback: diagonal
    v = np.array([vols[t][1] if t in vols else _FALLBACK_VOL for t in tickers])
    return np.diag(v ** 2) + 1e-4 * np.eye(n)