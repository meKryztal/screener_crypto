"""
dom_median.py — ATR(100)*mult диапазон на 5m ТФ + медиана стакана (логика из 11111.py).

Логика:
  1. Берём 5m свечи, считаем ATR(100) с mult=5.
  2. Диапазон = [last_close - atr_band, last_close + atr_band]
  3. Из стакана (bids+asks) оставляем только уровни внутри этого диапазона.
  4. Считаем робастную медиану по qty (нижние 90%) среди отфильтрованных уровней —
     это referens-уровень для сравнения стен, не смещаемый крупными аномалиями.
  5. Для каждого уровня считаем score = qty / median_qty — это показатель аномальности.
  6. Возвращаем все уровни со своим score (подход как в детекторе аномалий).
"""

import statistics
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
    compute robust qty-based median (bottom 90% of levels by qty).
    
    Подход из 11111.py (детектор аномалий):
    - Медиана считается по qty (не по USD) — устойчива к крупным стенам.
    - Обрезаются топ 10% уровней по qty перед расчётом медианы — аномалии
      не смещают референс (это именно то что нужно для детектора стен).
    - Score каждого уровня = qty / median_qty (простое деление).
    - Возвращаются все уровни со своим score и информацией.

    Returns:
        {
          "median_qty":      float,            # робастная медиана qty (referens)
          "levels_in_range": int,             # всего уровней в ATR диапазоне
          "bids_with_score": list,            # [{price, qty, usd, score}]
          "asks_with_score": list,            # [{price, qty, usd, score}]
          "bid_total_usd":   float,           # сумма USD по всем бидам
          "ask_total_usd":   float,           # сумма USD по всем аскам
          "imbalance":       float,           # bid_usd / (bid_usd + ask_usd)
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

    all_levels = [(p, q, "bid") for p, q in bids] + \
                 [(p, q, "ask") for p, q in asks]

    if not all_levels:
        return None

    # ── Robust qty-median (bottom 90%) — подход из 11111.py ──────────────────
    # Сортируем все qty, обрезаем топ 10% чтобы аномальные стены не тянули медиану
    zone_qtys = sorted(q for _, q, _ in all_levels)
    cutoff    = max(1, int(len(zone_qtys) * 0.90))
    median_qty = statistics.median(zone_qtys[:cutoff])

    if median_qty <= 0:
        return None

    # ── Считаем score для каждого уровня ─────────────────────────────────────
    # score = qty / median_qty — это показатель аномальности уровня
    bids_with_score = []
    asks_with_score = []
    bid_total_usd = 0.0
    ask_total_usd = 0.0

    for price, qty, side in all_levels:
        usd_val = qty * price
        score   = qty / median_qty if median_qty > 0 else 0.0
        
        level_dict = {
            "price": round(price, 10),
            "qty":   round(qty, 6),
            "usd":   round(usd_val, 2),
            "score": round(score, 4),
        }
        
        if side == "bid":
            bids_with_score.append(level_dict)
            bid_total_usd += usd_val
        else:
            asks_with_score.append(level_dict)
            ask_total_usd += usd_val

    total_usd = bid_total_usd + ask_total_usd
    imbalance = bid_total_usd / total_usd if total_usd > 0 else 0.5

    return {
        "median_qty":      round(median_qty, 6),
        "levels_in_range": len(all_levels),
        "bids_with_score": bids_with_score,
        "asks_with_score": asks_with_score,
        "bid_total_usd":   round(bid_total_usd, 2),
        "ask_total_usd":   round(ask_total_usd, 2),
        "imbalance":       round(imbalance, 4),
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
