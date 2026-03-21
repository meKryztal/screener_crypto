"""
main.py v8 — все фьючи Binance, без WebSocket
+ авторизация по коду (без логина/пароля)
"""

import asyncio
import builtins
import collections
import hashlib
import hmac
import json
import logging
import os
import secrets
import statistics
import time as _time
from collections import defaultdict
from contextlib import asynccontextmanager

import pandas as pd
import requests as _requests
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from data_manager import get_manager, LTF
from indicator import calc_absorption, absorption_to_json, ohlcv_to_json
from poc_indicator import calc_all_pocs, pocs_to_json

load_dotenv()

def _get_codes() -> set:
    """Читаем .env при каждом вызове — изменения подхватываются без перезапуска."""
    load_dotenv(override=True)
    raw = os.environ.get("ACCESS_CODES", "")
    return {c.strip() for c in raw.split(",") if c.strip()}

if not _get_codes():
    raise RuntimeError("ACCESS_CODES не задан в .env! Пример: ACCESS_CODES=code1,code2")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALL_TFS = ["1m", "5m", "15m", "1h"]

# Per-TF display limits
MAX_DISPLAY = {
    "1m":  6000,
    "5m":  6000,
    "15m": 6000,
}
MAX_DISPLAY_DEFAULT = 6000

# ─── Кэш скринера ───
_symbols_cache:    list  = []
_symbols_cache_ts: float = 0.0
_SYMBOLS_TTL = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

# Пути которые доступны без авторизации
_SKIP_AUTH = ("/login", "/static")

# Пути WebSocket — нужна отдельная проверка (нет кук в стандартном WS)
_WS_PREFIX = "/ws"

# Защита от брутфорса
_failed_attempts: dict = defaultdict(list)
_MAX_ATTEMPTS    = 5
_BLOCK_SECONDS   = 300   # 5 минут


def _is_blocked(ip: str) -> bool:
    now = _time.time()
    _failed_attempts[ip] = [t for t in _failed_attempts[ip] if now - t < _BLOCK_SECONDS]
    return len(_failed_attempts[ip]) >= _MAX_ATTEMPTS


def _record_fail(ip: str):
    _failed_attempts[ip].append(_time.time())


def _make_cookie(code: str) -> str:
    """Создаём подписанную куку: code.hmac_sig"""
    sig = hmac.new(code.encode(), code.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{code}.{sig}"


def _verify_cookie(cookie_value: str) -> bool:
    """
    Кука валидна если:
    1. Подпись верна
    2. Код всё ещё есть в .env (проверяется динамически — без перезапуска)
    """
    try:
        code, sig = cookie_value.rsplit(".", 1)
        expected  = hmac.new(code.encode(), code.encode(), hashlib.sha256).hexdigest()[:16]
        return (
            secrets.compare_digest(sig, expected) and
            code in _get_codes()
        )
    except Exception:
        return False


def _verify_ws_cookie(websocket) -> bool:
    """Парсим куку из заголовка WS handshake."""
    cookie_header = websocket.headers.get("cookie", "")
    cookies = dict(
        c.strip().split("=", 1)
        for c in cookie_header.split(";")
        if "=" in c
    )
    return _verify_cookie(cookies.get("access_token", ""))


# ─── Login page ───────────────────────────────────────────────────────────────

_LOGIN_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Вход</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh; background: #0d0d0d; font-family: sans-serif;
    }}
    .card {{
      background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px;
      padding: 40px 36px; width: 320px; display: flex;
      flex-direction: column; gap: 16px;
    }}
    h2 {{ color: #e0e0e0; font-size: 18px; text-align: center; }}
    input[type=password] {{
      padding: 11px 14px; font-size: 15px; border-radius: 7px;
      border: 1px solid #333; background: #111; color: #fff; outline: none;
      transition: border-color .2s;
    }}
    input[type=password]:focus {{ border-color: #2962ff; }}
    button {{
      padding: 11px; font-size: 15px; border: none; border-radius: 7px;
      background: #2962ff; color: #fff; cursor: pointer; transition: background .2s;
    }}
    button:hover {{ background: #1a4fd6; }}
    .error {{ color: #ff5252; font-size: 13px; text-align: center; }}
  </style>
</head>
<body>
  <form class="card" method="post" action="/login?next_url={next}">
    <h2>Введи код доступа</h2>
    {error_block}
    <input type="password" name="code" placeholder="Код" autofocus autocomplete="current-password">
    <button type="submit">Войти</button>
  </form>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _calc_symbols_list(m) -> list:
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
        vol24h = float(dollar_vol.iloc[-1440:].sum())
        vol2h  = float(dollar_vol.iloc[-120:].sum())
        vol7d  = float(dollar_vol.iloc[-10080:].sum())

        avg_2h_in_24h = vol24h / 12.0 if vol24h > 0 else 1
        spike_2h_24h  = round(vol2h / avg_2h_in_24h, 2) if avg_2h_in_24h > 0 else None

        avg_24h_in_7d = vol7d / 7.0 if vol7d > 0 else 1
        spike_24h_7d  = round(vol24h / avg_24h_in_7d, 2) if avg_24h_in_7d > 0 else None

        close_prices = df["close"]
        cur = float(last["close"])

        low3h   = float(close_prices.iloc[-180:].min()) if len(close_prices) >= 180 else None
        pump_3h = round((cur / low3h - 1) * 100, 2) if low3h and low3h > 0 else None

        result.append({
            "symbol":       sym,
            "loaded":       True,
            "price":        round(cur, 6),
            "change":       change,
            "vol24h":       vol24h,
            "vol2h":        vol2h,
            "vol7d":        vol7d,
            "spike_2h_24h": spike_2h_24h,
            "spike_24h_7d": spike_24h_7d,
            "pump_3h":      pump_3h,
        })
    return result


TEST_SYMBOLS = [
    "BTC/USDT:USDT",
    "1INCH/USDT:USDT",
    "1000XEC/USDT:USDT",
    "1000SHIB/USDT:USDT",
    "2Z/USDT:USDT",
]


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

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
    global _symbols_cache, _symbols_cache_ts
    while True:
        try:
            await asyncio.sleep(_SYMBOLS_TTL)
            m = get_manager()
            if m.store.ready or m.store._loaded:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, _calc_symbols_list, m)
                _symbols_cache    = data
                _symbols_cache_ts = _time.monotonic()
        except Exception as e:
            logger.warning(f"symbols_refresh: {e}")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE — защищает все HTTP маршруты
# ══════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path

    # Открытые пути
    if any(path.startswith(p) for p in _SKIP_AUTH):
        return await call_next(request)

    # Проверяем куку
    cookie = request.cookies.get("access_token", "")
    if _verify_cookie(cookie):
        return await call_next(request)

    # API — отвечаем 401, не редиректим
    if path.startswith("/api/"):
        return Response("Unauthorized", status_code=401)

    # Остальное — на страницу логина
    return RedirectResponse(f"/login?next_url={path}", status_code=302)


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN / LOGOUT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
async def login_page(next_url: str = "/"):
    return _LOGIN_HTML.format(next=next_url, error_block="")


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, next_url: str = "/", code: str = Form(default="")):
    ip = request.client.host

    if _is_blocked(ip):
        mins = _BLOCK_SECONDS // 60
        return HTMLResponse(
            _LOGIN_HTML.format(
                next=next_url,
                error_block=f'<p class="error">Слишком много попыток. Подожди {mins} мин.</p>',
            ),
            status_code=429,
        )

    if not code:
        return HTMLResponse(
            _LOGIN_HTML.format(next=next_url, error_block='<p class="error">Введи код.</p>'),
            status_code=400,
        )

    # Ищем совпадение среди всех кодов (читаем свежий список из .env)
    codes   = _get_codes()
    matched = builtins.next((c for c in codes if secrets.compare_digest(code, c)), None)

    if matched:
        response = RedirectResponse(next_url, status_code=302)
        response.set_cookie(
            "access_token",
            _make_cookie(matched),
            httponly=True,
            samesite="strict",
            max_age=60 * 60 * 24 * 30,   # 30 дней
            secure=False,                  # поставь True если HTTPS
        )
        return response

    _record_fail(ip)
    remaining = _MAX_ATTEMPTS - len(_failed_attempts[ip])
    return HTMLResponse(
        _LOGIN_HTML.format(
            next=next_url,
            error_block=f'<p class="error">Неверный код. Осталось попыток: {remaining}</p>',
        ),
        status_code=403,
    )


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("access_token")
    return response


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def api_status():
    m      = get_manager()
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
    if _symbols_cache:
        return _symbols_cache
    m    = get_manager()
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _calc_symbols_list, m)
    _symbols_cache    = data
    _symbols_cache_ts = _time.monotonic()
    return data


TF_RESAMPLE = {"5m": "5min", "15m": "15min", "1h": "1h"}


def resample_from_1m(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    freq = TF_RESAMPLE[tf]
    df   = df_1m.resample(freq, label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })
    return df[df["open"].notna()]


@app.get("/api/bars/{symbol:path}")
async def api_bars(symbol: str, tf: str = "5m"):
    if tf not in ALL_TFS:
        raise HTTPException(400, f"Неизвестный TF: {tf}")
    m = get_manager()
    await m.ensure_tf(symbol, "1m")
    df_1m = m.store.get(symbol, "1m")
    if df_1m is None or df_1m.empty:
        raise HTTPException(404, f"Нет данных {symbol}")

    if tf == "1m":
        df = df_1m
    else:
        loop = asyncio.get_event_loop()
        df   = await loop.run_in_executor(None, resample_from_1m, df_1m, tf)

    limit      = MAX_DISPLAY.get(tf, MAX_DISPLAY_DEFAULT)
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
    import numpy as np

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

    ltf_df = df_1m

    result = calc_absorption(
        base_df, ltf_df,
        base_tf=tf,
        mode=mode,
        manual_vol=manual_vol,
        percentile=percentile,
        lookback=lookback,
    )

    auto_thresh = None
    if mode == "Auto":
        vol_series    = ltf_df["volume"] if tf != "1m" else base_df["volume"]
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
        "signals":    absorption_to_json(result),
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
    m = get_manager()
    await m.ensure_oi(symbol, tf)
    df = m.store.get_oi(symbol, tf)
    if df is None or df.empty:
        raise HTTPException(404, f"OI не доступен для {symbol}")

    return [
        {"time": int(ts.timestamp()), "value": float(v)}
        for ts, v in df["oi"].items()
    ]


@app.get("/api/dom/{symbol:path}")
async def api_dom(symbol: str, depth: int = 50):
    import requests
    try:
        base        = symbol.split("/")[0]
        quote       = symbol.split("/")[1].split(":")[0]
        binance_sym = base + quote
        depth       = max(5, depth)
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
        raw  = r.json()
        bids = [[float(p), float(q)] for p, q in raw.get("bids", [])][:depth]
        asks = [[float(p), float(q)] for p, q in raw.get("asks", [])][:depth]
        max_qty = max(
            max((q for _, q in bids), default=0),
            max((q for _, q in asks), default=0),
            1e-12,
        )
        return {
            "bids":    [{"price": p, "qty": q, "pct": q / max_qty} for p, q in bids],
            "asks":    [{"price": p, "qty": q, "pct": q / max_qty} for p, q in asks],
            "max_qty": max_qty,
        }
    except Exception as e:
        raise HTTPException(500, f"DOM error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# DOM Anomaly Detector
# ──────────────────────────────────────────────────────────────────────────────

_DOM_SNAPSHOTS:  dict = collections.defaultdict(lambda: collections.deque(maxlen=6))
_DOM_ANOMALIES:  dict = {}
_DOM_FIRST_SEEN: dict = collections.defaultdict(dict)
_DOM_POLL_INTERVAL = 30
_DOM_DEPTH         = 100
_DOM_MIN_PERSIST   = 60
_DOM_ANOMALY_MULT  = 8.0


def _fetch_dom_sync(binance_sym: str) -> dict:
    try:
        r = _requests.get(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": binance_sym, "limit": _DOM_DEPTH},
            timeout=5,
        )
        r.raise_for_status()
        raw  = r.json()
        bids = [(float(p), float(q)) for p, q in raw.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in raw.get("asks", [])]
        return {"bids": bids, "asks": asks}
    except Exception:
        return {}


def _detect_anomalies(symbol: str, current_price: float) -> list:
    snaps = list(_DOM_SNAPSHOTS[symbol])
    if len(snaps) < 2:
        return []
    now        = _time.time()
    latest     = snaps[-1]
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
    fs       = _DOM_FIRST_SEEN[symbol]
    seen_now = set()
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
            if score < _DOM_ANOMALY_MULT or current_price <= 0:
                continue
            key = f"{side}_{rounded}"
            seen_now.add(key)
            if key not in fs:
                fs[key] = now
            distance_pct = abs(price - current_price) / current_price * 100
            usd_val      = qty * price
            anomalies.append({
                "price":        rounded,
                "qty":          round(qty, 4),
                "usd_val":      round(usd_val, 0),
                "side":         side,
                "distance_pct": round(distance_pct, 4),
                "score":        round(score, 2),
                "first_seen":   int(fs[key]),
                "age_sec":      int(now - fs[key]),
            })

    stale = [k for k in fs if k not in seen_now and now - fs[k] > 28800]
    for k in stale:
        del fs[k]

    anomalies.sort(key=lambda x: x["distance_pct"])
    return anomalies[:5]


async def _dom_poll_loop():
    await asyncio.sleep(15)
    while True:
        try:
            m       = get_manager()
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

                    df        = m.store.get_mini(sym)
                    cur_price = float(df.iloc[-1]["close"]) if df is not None and not df.empty else 0
                    _DOM_ANOMALIES[sym] = _detect_anomalies(sym, cur_price)

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
    result = []
    m      = get_manager()
    for sym in m.store.symbols:
        anomalies = _DOM_ANOMALIES.get(sym, [])
        if not anomalies:
            continue
        closest = anomalies[0]
        result.append({
            "symbol":       sym,
            "distance_pct": closest["distance_pct"],
            "side":         closest["side"],
            "score":        closest["score"],
            "price":        closest["price"],
            "count":        len(anomalies),
            "all":          anomalies[:5],
        })
    result.sort(key=lambda x: x["distance_pct"])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Settings & Drawings
# ──────────────────────────────────────────────────────────────────────────────

SETTINGS_FILE = "settings.json"
DRAWINGS_FILE = "drawings.json"
DRAWINGS_TTL  = 7 * 24 * 3600


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
    cutoff   = _time.time() * 1000 - DRAWINGS_TTL * 1000
    return [d for d in drawings if d.get("createdAt", 0) > cutoff]


@app.post("/api/drawings")
async def post_drawings(request: Request):
    data   = await request.json()
    cutoff = _time.time() * 1000 - DRAWINGS_TTL * 1000
    data   = [d for d in data if d.get("createdAt", 0) > cutoff]
    _write_json(DRAWINGS_FILE, data)
    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────────────
# Root
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")
