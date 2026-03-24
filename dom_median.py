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
import asyncio
import aiohttp
import logging
from typing import Optional, Dict, List
import time

logger = logging.getLogger(__name__)


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
            timeout=10,  # FIX #5: Increase timeout from 5 to 10 seconds
        )
        r.raise_for_status()
        raw  = r.json()
        bids = [(float(p), float(q)) for p, q in raw.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in raw.get("asks", [])]
        return {"bids": bids, "asks": asks}
    except _requests.Timeout:
        logger.warning(f"fetch_dom {binance_sym}: timeout after 10s")
        return None
    except _requests.ConnectionError as e:
        logger.warning(f"fetch_dom {binance_sym}: connection error: {e}")
        return None
    except _requests.HTTPError as e:
        logger.warning(f"fetch_dom {binance_sym}: HTTP error: {e}")
        return None
    except Exception as e:
        logger.error(f"fetch_dom {binance_sym}: unexpected error: {type(e).__name__}: {e}")
        return None


# ─── Sequential DOM Fetcher (Rate-limited, 6 req/sec) ─────────────────────────

class SequentialDOMFetcher:
    """
    Fetches DOM for multiple symbols sequentially at 6 req/sec rate.
    Safe for Binance Futures API (limit: 20 req/sec).
    
    FIX #5: Parallelized DOM fetching with proper rate limiting
    """
    
    def __init__(self, rate: float = 6.0):
        self.rate = rate
        self.interval = 1.0 / rate  # seconds between requests
        self.metrics = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "rate_limit_hits": 0,
        }
    
    async def fetch_all(self, symbols: List[str]) -> Dict[str, Optional[dict]]:
        """
        Fetch DOM for all symbols at rate-limited speed.
        
        For 550 symbols at 6 req/sec: 550 / 6 ≈ 92 seconds
        Safe against Binance rate limits (20 req/sec limit)
        """
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            try:
                for i, symbol in enumerate(symbols):
                    dom = await self._fetch_one_async(session, symbol)
                    results[symbol] = dom

                    # Log progress every 50 symbols
                    if (i + 1) % 50 == 0:
                        logger.info(
                            f"DOM fetch progress: {i+1}/{len(symbols)} "
                            f"({(i+1)*100//len(symbols)}%) - "
                            f"Success: {self.metrics['success']}, "
                            f"Failed: {self.metrics['failed']}"
                        )
                    # Rate limit: sleep between requests
                    if i < len(symbols) - 1:
                        await asyncio.sleep(self.interval)
            finally:
                # Гарантируем закрытие сессии
                pass
        logger.info(
            f"DOM fetch complete: {self.metrics['success']}/{len(symbols)} successful, "
            f"{self.metrics['failed']} failed"
        )

        return results

    async def _fetch_one_async(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        depth: int = 500
    ) -> Optional[dict]:
        """Fetch DOM for single symbol with retries"""

        # Find valid limit
        for snap in (5, 10, 20, 50, 100, 500, 1000):
            if snap >= depth:
                snap_limit = snap
                break
        else:
            snap_limit = 1000

        binance_sym = symbol.replace("/", "").replace(":", "")

        for attempt in range(3):
            try:
                async with session.get(
                    "https://fapi.binance.com/fapi/v1/depth",
                    params={"symbol": binance_sym, "limit": snap_limit},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    if r.status == 429:  # Too Many Requests
                        self.metrics["rate_limit_hits"] += 1
                        logger.warning(f"{symbol}: Rate limited (429), waiting...")
                        await asyncio.sleep(2.0)
                        continue

                    r.raise_for_status()
                    data = await r.json()

                    self.metrics["total"] += 1
                    self.metrics["success"] += 1

                    return {
                        "bids": [(float(p), float(q)) for p, q in data.get("bids", [])],
                        "asks": [(float(p), float(q)) for p, q in data.get("asks", [])],
                    }

            except asyncio.TimeoutError:
                logger.debug(f"{symbol}: timeout (attempt {attempt+1}/3)")
                if attempt < 2:
                    await asyncio.sleep(1.0)

            except aiohttp.ClientError as e:
                logger.debug(f"{symbol}: {type(e).__name__} (attempt {attempt+1}/3)")
                if attempt < 2:
                    await asyncio.sleep(1.0)

        self.metrics["total"] += 1
        self.metrics["failed"] += 1
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