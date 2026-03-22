"""
dom_median.py — ATR(100)*mult диапазон на 5m ТФ + медиана стакана.

Логика:
  1. Берём 5m свечи, считаем ATR(100) с mult=5.
  2. Диапазон = [last_close - atr_band, last_close + atr_band]
  3. Из стакана (bids+asks) оставляем только уровни внутри этого диапазона.
  4. Считаем медиану по объёму (qty * price) среди отфильтрованных уровней.
  5. Проверяем глубину стакана вокруг медианы (±depth уровней).
"""

import numpy as np
import pandas as pd
import requests as _requests
from typing import Optional


# ─── ATR ─────────────────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> np.ndarray:
    h = df["high"].values
    l = df["low"].values
    c_prev = np.roll(df["close"].values, 1)
    c_prev[0] = df["open"].values[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    return tr


def calc_atr(df: pd.DataFrame, period: int = 100) -> float:
    """Wilder's ATR on last `period` bars. Returns last value."""
    if df is None or len(df) < period + 1:
        if df is not None and len(df) >= 2:
            # fallback: simple mean TR
            tr = _true_range(df)
            return float(np.mean(tr[-min(period, len(tr)):]))
        return 0.0

    tr = _true_range(df)

    # Wilder smoothing: first value = SMA, then EMA with alpha=1/period
    atr = float(np.mean(tr[1:period + 1]))
    alpha = 1.0 / period
    for i in range(period + 1, len(tr)):
        atr = atr * (1 - alpha) + tr[i] * alpha

    return atr


def calc_atr_band(df_5m: pd.DataFrame, period: int = 100, mult: float = 5.0) -> dict:
    """
    Returns:
        {
          "atr":    float,      # ATR value
          "band":   float,      # atr * mult
          "mid":    float,      # last close
          "lo":     float,      # mid - band
          "hi":     float,      # mid + band
        }
    or None if not enough data.
    """
    if df_5m is None or df_5m.empty:
        return None

    atr  = calc_atr(df_5m, period)
    band = atr * mult
    mid  = float(df_5m["close"].iloc[-1])

    return {
        "atr":  round(atr, 10),
        "band": round(band, 10),
        "mid":  round(mid, 10),
        "lo":   round(mid - band, 10),
        "hi":   round(mid + band, 10),
    }


# ─── DOM fetch ────────────────────────────────────────────────────────────────

def fetch_dom(binance_sym: str, depth: int = 500) -> Optional[dict]:
    """Fetch raw order book from Binance Futures. Returns {"bids":[], "asks":[]}."""
    # Binance only accepts specific limits
    for snap in (5, 10, 20, 50, 100, 500, 1000):
        if snap >= depth:
            snap_limit = snap
            break
    else:
        snap_limit = 1000

    try:
        r = _requests.get(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": binance_sym, "limit": snap_limit},
            timeout=5,
        )
        r.raise_for_status()
        raw  = r.json()
        bids = [(float(p), float(q)) for p, q in raw.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in raw.get("asks", [])]
        return {"bids": bids, "asks": asks}
    except Exception:
        return None


# ─── Median calculation ───────────────────────────────────────────────────────

def calc_dom_median(
    dom: dict,
    lo: float,
    hi: float,
    depth: int = 10,
) -> dict:
    """
    Filter DOM levels within [lo, hi] ATR range,
    compute volume-weighted median price, then analyse
    ±depth levels around that median.

    Returns:
        {
          "median_price": float,            # volume-weighted median
          "median_usd":   float,            # total USD at median zone
          "bid_wall":     float | None,     # strongest bid near median
          "ask_wall":     float | None,     # strongest ask near median
          "bid_depth_usd": float,           # total bid USD in depth zone
          "ask_depth_usd": float,           # total ask USD in depth zone
          "imbalance":    float,            # bid_usd / (bid_usd + ask_usd)
          "levels_in_range": int,           # how many levels inside ATR range
          "bids_filtered": list,            # [{price, qty, usd}]
          "asks_filtered": list,
        }
    or None if not enough data.
    """
    if not dom:
        return None

    bids_raw = dom.get("bids", [])
    asks_raw = dom.get("asks", [])

    # Filter within ATR band
    bids = [(p, q) for p, q in bids_raw if lo <= p <= hi]
    asks = [(p, q) for p, q in asks_raw if lo <= p <= hi]

    all_levels = [(p, q, p * q, "bid") for p, q in bids] + \
                 [(p, q, p * q, "ask") for p, q in asks]

    if not all_levels:
        return None

    all_levels.sort(key=lambda x: x[0])  # sort by price

    prices  = np.array([x[0] for x in all_levels])
    volumes = np.array([x[2] for x in all_levels])  # USD value

    if volumes.sum() == 0:
        return None

    # Volume-weighted median: find price where cumulative volume >= 50%
    total_vol  = volumes.sum()
    cumvol     = np.cumsum(volumes)
    median_idx = int(np.searchsorted(cumvol, total_vol * 0.5))
    median_idx = min(median_idx, len(prices) - 1)
    median_price = float(prices[median_idx])

    # Depth zone: ±depth levels around median index
    lo_idx = max(0, median_idx - depth)
    hi_idx = min(len(all_levels) - 1, median_idx + depth)
    zone   = all_levels[lo_idx: hi_idx + 1]

    bid_levels = [(p, q, usd) for p, q, usd, side in zone if side == "bid"]
    ask_levels = [(p, q, usd) for p, q, usd, side in zone if side == "ask"]

    bid_depth_usd = sum(usd for _, _, usd in bid_levels)
    ask_depth_usd = sum(usd for _, _, usd in ask_levels)
    total_zone    = bid_depth_usd + ask_depth_usd

    # Strongest wall near median
    bid_wall = max(bid_levels, key=lambda x: x[2])[0] if bid_levels else None
    ask_wall = max(ask_levels, key=lambda x: x[2])[0] if ask_levels else None

    imbalance = bid_depth_usd / total_zone if total_zone > 0 else 0.5

    # Median USD (volume at the exact median level)
    median_usd = float(volumes[median_idx])

    return {
        "median_price":    round(median_price, 10),
        "median_usd":      round(median_usd, 2),
        "bid_wall":        round(bid_wall, 10) if bid_wall is not None else None,
        "ask_wall":        round(ask_wall, 10) if ask_wall is not None else None,
        "bid_depth_usd":   round(bid_depth_usd, 2),
        "ask_depth_usd":   round(ask_depth_usd, 2),
        "imbalance":       round(imbalance, 4),
        "levels_in_range": len(all_levels),
        "bids_filtered":   [{"price": p, "qty": round(q, 6), "usd": round(usd, 2)}
                            for p, q, usd in bid_levels],
        "asks_filtered":   [{"price": p, "qty": round(q, 6), "usd": round(usd, 2)}
                            for p, q, usd in ask_levels],
    }


# ─── Main entry point ─────────────────────────────────────────────────────────

def get_dom_median_full(
    symbol: str,
    df_5m: pd.DataFrame,
    atr_period: int = 100,
    atr_mult: float = 5.0,
    dom_depth_fetch: int = 500,
    analysis_depth: int = 10,
) -> dict:
    """
    Full pipeline: ATR range on 5m → fetch DOM → compute median + depth analysis.

    symbol: ccxt-style e.g. "BTC/USDT:USDT"
    df_5m:  5-minute OHLCV DataFrame with DatetimeIndex
    """
    base  = symbol.split("/")[0]
    quote = symbol.split("/")[1].split(":")[0]
    bsym  = base + quote

    # 1. ATR band
    band = calc_atr_band(df_5m, atr_period, atr_mult)
    if band is None:
        return {"error": "not enough 5m bars for ATR"}

    # 2. Fetch DOM
    dom = fetch_dom(bsym, dom_depth_fetch)
    if dom is None:
        return {"error": "failed to fetch DOM", "band": band}

    # 3. Median analysis
    result = calc_dom_median(dom, band["lo"], band["hi"], analysis_depth)
    if result is None:
        return {"error": "no DOM levels inside ATR range", "band": band}

    return {
        "band":   band,
        "median": result,
    }
