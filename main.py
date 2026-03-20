"""
main.py v8 — все фьючи Binance, без WebSocket
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import json
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from data_manager import get_manager, LTF
ALL_TFS = ["1m", "5m", "15m", "1h"]
from indicator import calc_absorption, absorption_to_json, ohlcv_to_json
from poc_indicator import calc_all_pocs, pocs_to_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Per-TF display limits — HTF bars are resampled from 1m so limits scale accordingly
MAX_DISPLAY = {
    "1m":  6000,   # ~2 days
    "5m":  6000,   # ~3000 5m bars = ~10 days
    "15m": 6000,   # ~6000 15m bars = ~60 days
}
MAX_DISPLAY_DEFAULT = 6000

# ─── Кэш скринера ───
# Пересчитывается в фоне раз в 10 сек, а не при каждом запросе
import time as _time
_symbols_cache: list = []
_symbols_cache_ts: float = 0.0
_SYMBOLS_TTL = 10.0  # секунд


def _calc_symbols_list(m) -> list:
    """Считает скринер из 1m данных — они всегда актуальны через WS."""
    result = []
    for sym in m.store.symbols:
        df = m.store.get(sym, "1m")
        if df is None or df.empty:
            result.append({"symbol": sym, "loaded": False,
                           "price": None, "change": None,
                           "vol24h": None, "vol2h": None, "vol7d": None})
            continue
        last   = df.iloc[-1]
        first  = df.iloc[0]
        change = round((last["close"] - first["close"]) / first["close"] * 100, 2)

        dollar_vol = df["volume"] * df["close"]
        vol24h = float(dollar_vol.iloc[-1440:].sum())  # 1440 × 1m = 24h
        vol2h  = float(dollar_vol.iloc[-120:].sum())   # 120  × 1m = 2h
        vol7d  = float(dollar_vol.iloc[-10080:].sum())  # 10080 × 1m = 7 days

        avg_2h_in_24h = vol24h / 12.0 if vol24h > 0 else 1
        spike_2h_24h  = round(vol2h / avg_2h_in_24h, 2) if avg_2h_in_24h > 0 else None

        avg_24h_in_7d = vol7d / 7.0 if vol7d > 0 else 1
        spike_24h_7d  = round(vol24h / avg_24h_in_7d, 2) if avg_24h_in_7d > 0 else None

        close_prices = df["close"]
        cur = float(last["close"])

        low3h = float(close_prices.iloc[-180:].min()) if len(close_prices) >= 180 else None
        pump_3h = round((cur / low3h - 1) * 100, 2) if low3h and low3h > 0 else None

        result.append({
            "symbol": sym,
            "loaded": True,
            "price":  round(cur, 6),
            "change": change,
            "vol24h":       vol24h,
            "vol2h":        vol2h,
            "vol7d":        vol7d,
            "spike_2h_24h": spike_2h_24h,
            "spike_24h_7d": spike_24h_7d,
            "pump_3h":  pump_3h,
        })
    return result

TEST_SYMBOLS = [
    "BTC/USDT:USDT",
    "1INCH/USDT:USDT",
    "1000XEC/USDT:USDT",
    "1000SHIB/USDT:USDT",
    "2Z/USDT:USDT",
]
@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = get_manager()
    symbols = await manager.fetch_all_futures_symbols()
    # symbols = TEST_SYMBOLS
    asyncio.create_task(manager.start(symbols))
    asyncio.create_task(_symbols_refresh_loop())
    asyncio.create_task(_dom_poll_loop())
    yield
    await manager.stop()


async def _symbols_refresh_loop():
    """Фоновый пересчёт скринера каждые 10 сек — не блокирует запросы."""
    global _symbols_cache, _symbols_cache_ts
    while True:
        try:
            await asyncio.sleep(_SYMBOLS_TTL)
            m = get_manager()
            if m.store.ready or m.store._loaded:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, _calc_symbols_list, m)
                _symbols_cache = data
                _symbols_cache_ts = _time.monotonic()
        except Exception as e:
            logger.warning(f"symbols_refresh: {e}")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/status")
async def api_status():
    m = get_manager()
    total  = len(m.store.symbols)
    loaded = len(m.store._loaded)
    return {
        "ready":   m.store.ready,
        "loaded":  loaded,
        "total":   total,
        "percent": round(loaded / total * 100, 1) if total else 0,
    }


@app.get("/api/symbols")
async def api_symbols():
    global _symbols_cache, _symbols_cache_ts
    # Отдаём кэш — он обновляется фоново каждые 10 сек
    if _symbols_cache:
        return _symbols_cache
    # Первый запрос до готовности кэша — считаем синхронно
    m = get_manager()
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _calc_symbols_list, m)
    _symbols_cache = data
    _symbols_cache_ts = _time.monotonic()
    return data


TF_RESAMPLE = {"5m": "5min", "15m": "15min", "1h": "1h"}

def resample_from_1m(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    freq = TF_RESAMPLE[tf]
    df = df_1m.resample(freq, label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })
    # Убираем только полностью пустые периоды (нет ни одной 1m свечи)
    df = df[df["open"].notna()]
    return df


@app.get("/api/bars/{symbol:path}")
async def api_bars(symbol: str, tf: str = "5m"):
    if tf not in ALL_TFS:
        raise HTTPException(400, f"Неизвестный TF: {tf}")
    m = get_manager()
    # Всегда работаем от 1m
    await m.ensure_tf(symbol, "1m")
    df_1m = m.store.get(symbol, "1m")
    if df_1m is None or df_1m.empty:
        raise HTTPException(404, f"Нет данных {symbol}")

    if tf == "1m":
        df = df_1m
    else:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, resample_from_1m, df_1m, tf)

    limit = MAX_DISPLAY.get(tf, MAX_DISPLAY_DEFAULT)
    display_df = df.iloc[-limit:] if len(df) > limit else df
    return ohlcv_to_json(display_df)


@app.get("/api/absorption/{symbol:path}")
async def api_absorption(
    symbol:     str,
    tf:         str   = "5m",
    mode:       str   = "Auto",
    percentile: float = 99.0,
    manual_vol: float = 100.0,
    lookback:   int   = 4900,
):
    m = get_manager()
    await m.ensure_tf(symbol, "1m")
    df_1m = m.store.get(symbol, "1m")
    if df_1m is None or df_1m.empty:
        raise HTTPException(404, f"Нет данных {symbol}")

    loop = asyncio.get_event_loop()
    if tf == "1m":
        base_df = df_1m
    else:
        base_df = await loop.run_in_executor(None, resample_from_1m, df_1m, tf)

    ltf_df = df_1m  # всегда 1m

    result = calc_absorption(
        base_df, ltf_df,
        base_tf=tf,
        mode=mode,
        manual_vol=manual_vol,
        percentile=percentile,
        lookback=lookback,
    )

    # Calculate current auto threshold for display
    import numpy as np
    import pandas as pd
    auto_thresh = None
    if mode == "Auto":
        vol_series = ltf_df["volume"] if tf != "1m" else base_df["volume"]
        thresh_series = (
            vol_series.shift(1)
            .rolling(window=lookback, min_periods=10)
            .quantile(percentile / 100)
        )
        if not thresh_series.empty:
            last_val = thresh_series.dropna()
            if not last_val.empty:
                auto_thresh = float(last_val.iloc[-1])



    return {
        "signals": absorption_to_json(result),
        "auto_thresh": auto_thresh,
    }


@app.get("/api/poc/{symbol:path}")
async def api_poc(
    symbol: str,
    tf:     str  = "5m",
    poc4h:  bool = True,
    poc1d:  bool = True,
    poc1w:  bool = True,
    poc1m:  bool = False,
):
    m = get_manager()
    await m.ensure_tf(symbol, "1m")
    df_1m = m.store.get(symbol, "1m")
    if df_1m is None or df_1m.empty:
        raise HTTPException(404, f"Нет данных {symbol}")

    loop = asyncio.get_event_loop()
    if tf == "1m":
        df = df_1m
    else:
        df = await loop.run_in_executor(None, resample_from_1m, df_1m, tf)

    show = {"4H": poc4h, "1D": poc1d, "1W": poc1w, "1M": poc1m}
    pocs = calc_all_pocs(df, tf, show)
    return pocs_to_json(pocs, max_display=MAX_DISPLAY.get(tf, MAX_DISPLAY_DEFAULT))


@app.get("/api/oi/{symbol:path}")
async def api_oi(symbol: str, tf: str = "5m"):
    """
    Возвращает историю OI с гранулярностью под TF графика.
    1m/5m → 5m OI, 15m → 15m OI, 1h → 1h OI и т.д.
    """
    m = get_manager()
    await m.ensure_oi(symbol, tf)
    df = m.store.get_oi(symbol, tf)
    if df is None or df.empty:
        raise HTTPException(404, f"OI не доступен для {symbol}")

    result = [
        {"time": int(ts.timestamp()), "value": float(v)}
        for ts, v in df["oi"].items()
    ]
    return result



@app.get("/api/dom/{symbol:path}")
async def api_dom(symbol: str, depth: int = 50):
    """
    Возвращает стакан (DOM) через Binance Futures REST.
    depth: число уровней с каждой стороны (без ограничений сверху — Binance даёт до 1000).
    """
    import requests
    try:
        base  = symbol.split("/")[0]
        quote = symbol.split("/")[1].split(":")[0]
        binance_sym = base + quote
        depth = max(5, depth)
        # Binance fapi/v1/depth допускает limit: 5,10,20,50,100,500,1000
        # Выбираем минимально достаточный лимит
        for snap in (5, 10, 20, 50, 100, 500, 1000):
            if snap >= depth:
                snap_limit = snap
                break
        else:
            snap_limit = 1000
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": binance_sym, "limit": snap_limit},
            timeout=5,
        )
        r.raise_for_status()
        raw = r.json()
        bids = [[float(p), float(q)] for p, q in raw.get("bids", [])]
        asks = [[float(p), float(q)] for p, q in raw.get("asks", [])]
        # Обрезаем до запрошенной глубины
        bids = bids[:depth]
        asks = asks[:depth]
        # Максимальный объём по всем уровням для нормализации баров
        max_qty = max(
            max((q for _, q in bids), default=0),
            max((q for _, q in asks), default=0),
            1e-12,
        )
        return {
            "bids": [{"price": p, "qty": q, "pct": q / max_qty} for p, q in bids],
            "asks": [{"price": p, "qty": q, "pct": q / max_qty} for p, q in asks],
            "max_qty": max_qty,
        }
    except Exception as e:
        raise HTTPException(500, f"DOM error: {e}")



import collections
import statistics
import requests as _requests

# ─── DOM Anomaly Detector ───────────────────────────────────────────────────
# Хранит снапшоты стакана: symbol → deque of {ts, bids:[(price,qty)], asks:[(price,qty)]}
_DOM_SNAPSHOTS: dict = collections.defaultdict(lambda: collections.deque(maxlen=6))  # 6×30s = 3 min
_DOM_ANOMALIES: dict = {}
_DOM_FIRST_SEEN: dict = collections.defaultdict(dict)   # symbol → list of {price, qty, side, distance_pct, score}
_DOM_POLL_INTERVAL = 30     # секунд между циклами
_DOM_DEPTH         = 100    # уровней стакана
_DOM_MIN_PERSIST   = 60     # секунд — лимитка должна висеть минимум столько
_DOM_ANOMALY_MULT  = 8.0    # во сколько раз qty > медианы чтобы считаться аномалией


def _fetch_dom_sync(binance_sym: str) -> dict:
    try:
        r = _requests.get(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": binance_sym, "limit": _DOM_DEPTH},
            timeout=5,
        )
        r.raise_for_status()
        raw = r.json()
        bids = [(float(p), float(q)) for p, q in raw.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in raw.get("asks", [])]
        return {"bids": bids, "asks": asks}
    except Exception:
        return {}


def _detect_anomalies(symbol: str, current_price: float) -> list:
    snaps = list(_DOM_SNAPSHOTS[symbol])
    if len(snaps) < 2:
        return []
    now    = _time.time()
    latest = snaps[-1]
    all_levels = latest.get("bids", []) + latest.get("asks", [])
    if not all_levels or len(all_levels) < 5:
        return []
    all_qtys   = [q for _, q in all_levels]
    median_qty = statistics.median(all_qtys)
    if median_qty <= 0:
        return []
    old_snaps = [s for s in snaps if now - s["ts"] >= _DOM_MIN_PERSIST]
    if not old_snaps:
        return []
    def snap_prices(snap, side):
        return {round(p, 8) for p, q in snap.get(side, [])}
    persistent_bids = set.intersection(*[snap_prices(s, "bids") for s in old_snaps])
    persistent_asks = set.intersection(*[snap_prices(s, "asks") for s in old_snaps])
    fs        = _DOM_FIRST_SEEN[symbol]
    seen_now  = set()
    anomalies = []
    for side, levels, persistent in [
        ("bid", latest.get("bids", []), persistent_bids),
        ("ask", latest.get("asks", []), persistent_asks),
    ]:
        for price, qty in levels:
            rounded = round(price, 8)
            if rounded not in persistent:
                continue
            score = qty / median_qty
            if score < _DOM_ANOMALY_MULT:
                continue
            if current_price <= 0:
                continue
            key = f"{side}_{rounded}"
            seen_now.add(key)
            if key not in fs:
                fs[key] = now
            first_seen = fs[key]
            distance_pct = abs(price - current_price) / current_price * 100
            usd_val      = qty * price
            anomalies.append({
                "price":        rounded,
                "qty":          round(qty, 4),
                "usd_val":      round(usd_val, 0),
                "side":         side,
                "distance_pct": round(distance_pct, 4),
                "score":        round(score, 2),
                "first_seen":   int(first_seen),
                "age_sec":      int(now - first_seen),
            })
    stale = [k for k in fs if k not in seen_now and now - fs[k] > 28800]
    for k in stale:
        del fs[k]
    anomalies.sort(key=lambda x: x["distance_pct"])
    return anomalies[:5]  # максимум 5 аномалий на символ

async def _dom_poll_loop():
    """Фоновый поллер стакана — обходит все символы каждые 30 сек."""
    await asyncio.sleep(15)  # небольшая задержка перед стартом
    while True:
        try:
            m = get_manager()
            symbols = [s for s in m.store.symbols if s in m.store._loaded]
            if not symbols:
                await asyncio.sleep(_DOM_POLL_INTERVAL)
                continue

            loop = asyncio.get_event_loop()
            now  = _time.time()

            for sym in symbols:
                try:
                    base  = sym.split("/")[0]
                    quote = sym.split("/")[1].split(":")[0]
                    bsym  = base + quote

                    dom = await loop.run_in_executor(None, _fetch_dom_sync, bsym)
                    if not dom:
                        continue

                    dom["ts"] = now
                    _DOM_SNAPSHOTS[sym].append(dom)

                    # Текущая цена
                    df = m.store.get_mini(sym)
                    cur_price = float(df.iloc[-1]["close"]) if df is not None and not df.empty else 0

                    _DOM_ANOMALIES[sym] = _detect_anomalies(sym, cur_price)

                    # Пауза между запросами чтобы не словить rate limit
                    await asyncio.sleep(0.06)

                except Exception as e:
                    logger.debug(f"dom_poll {sym}: {e}")

            logger.debug(f"dom_poll: обработано {len(symbols)} символов")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"dom_poll_loop: {e}")

        await asyncio.sleep(_DOM_POLL_INTERVAL)


@app.post("/api/set_limit_mult")
async def api_set_limit_mult(mult: float = 8.0):
    global _DOM_ANOMALY_MULT
    _DOM_ANOMALY_MULT = max(2.0, min(100.0, mult))
    # Сбрасываем кэш аномалий чтобы пересчитались
    _DOM_ANOMALIES.clear()
    return {"mult": _DOM_ANOMALY_MULT}


@app.get("/api/limit_levels/{symbol:path}")
async def api_limit_levels(symbol: str):
    return _DOM_ANOMALIES.get(symbol, [])




@app.get("/api/dom_debug/{symbol:path}")
async def api_dom_debug(symbol: str):
    snaps = list(_DOM_SNAPSHOTS.get(symbol, []))
    now   = _time.time()
    return {
        "snapshots_count": len(snaps),
        "snap_ages_sec":   [round(now - s["ts"]) for s in snaps],
        "oldest_snap_sec": round(now - snaps[0]["ts"]) if snaps else None,
        "anomalies":       _DOM_ANOMALIES.get(symbol, []),
        "first_seen_keys": list(_DOM_FIRST_SEEN.get(symbol, {}).keys()),
    }

@app.get("/api/anomalies")
async def api_anomalies():
    """Возвращает аномальные лимитки по всем символам."""
    result = []
    m = get_manager()
    for sym in m.store.symbols:
        anomalies = _DOM_ANOMALIES.get(sym, [])
        if not anomalies:
            continue
        # Ближайшая аномалия
        closest = anomalies[0]
        result.append({
            "symbol":       sym,
            "distance_pct": closest["distance_pct"],
            "side":         closest["side"],
            "score":        closest["score"],
            "price":        closest["price"],
            "count":        len(anomalies),
            "all":          anomalies[:5],  # топ-5 аномалий
        })

    result.sort(key=lambda x: x["distance_pct"])
    return result


SETTINGS_FILE = "settings.json"
DRAWINGS_FILE = "drawings.json"
DRAWINGS_TTL  = 7 * 24 * 3600  # 1 неделя


def _read_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _write_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"write_json {path}: {e}")


@app.get("/api/settings")
async def get_settings():
    return _read_json(SETTINGS_FILE, {})


@app.post("/api/settings")
async def post_settings(request: Request):
    data = await request.json()
    _write_json(SETTINGS_FILE, data)
    return {"ok": True}


@app.get("/api/drawings")
async def get_drawings():
    drawings = _read_json(DRAWINGS_FILE, [])
    # Обрезаем рисования старше недели
    cutoff = _time.time() * 1000 - DRAWINGS_TTL * 1000
    drawings = [d for d in drawings if d.get("createdAt", 0) > cutoff]
    return drawings


@app.post("/api/drawings")
async def post_drawings(request: Request):
    data = await request.json()
    # Обрезаем старые перед сохранением
    cutoff = _time.time() * 1000 - DRAWINGS_TTL * 1000
    data = [d for d in data if d.get("createdAt", 0) > cutoff]
    _write_json(DRAWINGS_FILE, data)
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")
