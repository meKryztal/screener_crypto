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

import numpy as np
import pandas as pd
import requests as _requests
from dom_orderbook import get_ob_manager, OrderBookManager
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

# ─── Кэш результатов absorption и poc ───────────────────────────────────────
# Ключ: (symbol, tf, mode, percentile, manual_vol, lookback) → (result, timestamp)
# TTL 60 сек — данные меняются только при появлении новых баров
_result_cache: dict = {}
_RESULT_TTL = 60.0

# ─── Кэш ресемплированных датафреймов ────────────────────────────────────────
# Ключ: (symbol, tf) → (df, monotonic_ts)
# TTL 10 сек — исключает тройной resample за одно открытие символа
_resample_cache: dict = {}
_RESAMPLE_TTL = 10.0


def resample_cached(df_1m: pd.DataFrame, symbol: str, tf: str) -> pd.DataFrame:
    """Возвращает ресемплированный df из кэша или пересчитывает."""
    key   = (symbol, tf)
    entry = _resample_cache.get(key)
    if entry is not None and _time.monotonic() - entry[1] < _RESAMPLE_TTL:
        return entry[0]
    df = resample_from_1m(df_1m, tf)
    _resample_cache[key] = (df, _time.monotonic())
    # Чистим устаревшие записи
    if len(_resample_cache) > 200:
        now = _time.monotonic()
        stale = [k for k, (_, ts) in _resample_cache.items() if now - ts > _RESAMPLE_TTL]
        for k in stale:
            del _resample_cache[k]
    return df

def _cache_get(key: tuple):
    entry = _result_cache.get(key)
    if entry is None:
        return None
    result, ts = entry
    if _time.monotonic() - ts > _RESULT_TTL:
        del _result_cache[key]
        return None
    return result

def _cache_set(key: tuple, value):
    _result_cache[key] = (value, _time.monotonic())
    # Чистим старые записи если накопилось много
    if len(_result_cache) > 500:
        now = _time.monotonic()
        stale = [k for k, (_, ts) in _result_cache.items() if now - ts > _RESULT_TTL]
        for k in stale:
            del _result_cache[k]


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


# Последняя активная сессия на каждый код: code → cookie_value
# Новый вход по тому же коду инвалидирует предыдущую сессию
_active_sessions: dict = {}


def _make_cookie(code: str) -> str:
    """Создаём куку с nonce — каждый вход уникален."""
    nonce = secrets.token_hex(8)
    sig   = hmac.new(code.encode(), (code + nonce).encode(), hashlib.sha256).hexdigest()[:16]
    value = f"{code}.{nonce}.{sig}"
    _active_sessions[code] = value  # старая сессия этого кода слетает
    return value


def _verify_cookie(cookie_value: str) -> bool:
    """
    Кука валидна если:
    1. Подпись верна
    2. Код есть в .env
    3. Это последняя выданная сессия для данного кода
       (вошёл с другого устройства — старая кука слетает)
    """
    try:
        code, nonce, sig = cookie_value.rsplit(".", 2)
        expected = hmac.new(code.encode(), (code + nonce).encode(), hashlib.sha256).hexdigest()[:16]
        if not secrets.compare_digest(sig, expected):
            return False
        if code not in _get_codes():
            return False
        return _active_sessions.get(code) == cookie_value
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
    #symbols = TEST_SYMBOLS
    asyncio.create_task(manager.start(symbols))
    asyncio.create_task(_symbols_refresh_loop())

    # Запускаем order book ПОСЛЕ того как data_manager загрузит историю —
    # чтобы не конкурировать за лимит Binance REST при старте
    async def _delayed_ob_start():
        # Ждём пока загрузится хотя бы 80% символов
        while True:
            m = get_manager()
            total  = len(m.store.symbols)
            loaded = len(m.store._loaded)
            if total > 0 and loaded >= total * 0.8:
                break
            await asyncio.sleep(5)
        logger.info("OrderBook: история загружена, запускаем стаканы...")
        # Дополнительная пауза чтобы weight счётчик Binance успел сбросится
        await asyncio.sleep(15)
        ob_manager = get_ob_manager()
        await ob_manager.start(symbols)

    asyncio.create_task(_delayed_ob_start())
    asyncio.create_task(_anomaly_detect_loop())
    yield
    await manager.stop()
    ob_manager = get_ob_manager()
    await ob_manager.stop()


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

    # Всегда ждём полный chart — mini только для скринера
    await m.ensure_tf(symbol, "1m")
    df_1m = m.store.get(symbol, "1m")

    if df_1m is None or df_1m.empty:
        raise HTTPException(404, f"Нет данных {symbol}")

    if tf == "1m":
        df = df_1m
    else:
        loop = asyncio.get_event_loop()
        df   = await loop.run_in_executor(None, resample_cached, df_1m, symbol, tf)

    limit      = MAX_DISPLAY.get(tf, MAX_DISPLAY_DEFAULT)
    display_df = df.iloc[-limit:] if len(df) > limit else df
    return ohlcv_to_json(display_df)


@app.get("/api/chart_ready/{symbol:path}")
async def api_chart_ready(symbol: str):
    m = get_manager()
    return {"ready": m.store.has_chart(symbol)}


async def _ensure_chart_bg(symbol: str):
    """Загружает полную историю в фоне не блокируя текущий запрос."""
    try:
        m    = get_manager()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, m._load_chart, symbol)
    except Exception as e:
        logger.warning(f"_ensure_chart_bg {symbol}: {e}")


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

    # #7: ключ кэша включает last_ts — инвалидируется при появлении нового бара
    last_ts   = int(df_1m.index[-1].timestamp())
    cache_key = ("absorption", symbol, tf, mode, percentile, manual_vol, lookback, last_ts)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    loop = asyncio.get_event_loop()
    if tf == "1m":
        base_df = df_1m
    else:
        base_df = await loop.run_in_executor(None, resample_cached, df_1m, symbol, tf)

    ltf_df = df_1m

    result, auto_thresh = await loop.run_in_executor(
        None,
        lambda: calc_absorption(
            base_df, ltf_df,
            base_tf=tf,
            mode=mode,
            manual_vol=manual_vol,
            percentile=percentile,
            lookback=lookback,
        ),
    )
    # #2: auto_thresh уже вычислен внутри calc_absorption —
    # убран повторный rolling().quantile() который считался здесь второй раз

    response = {
        "signals":     absorption_to_json(result),
        "auto_thresh": auto_thresh,
    }
    _cache_set(cache_key, response)
    return response


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

    # #7: ключ кэша включает last_ts — инвалидируется при появлении нового бара
    last_ts   = int(df_1m.index[-1].timestamp())
    cache_key = ("poc", symbol, tf, poc4h, poc1d, poc1w, poc1m, last_ts)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    loop = asyncio.get_event_loop()
    if tf == "1m":
        df = df_1m
    else:
        df = await loop.run_in_executor(None, resample_cached, df_1m, symbol, tf)

    show   = {"4H": poc4h, "1D": poc1d, "1W": poc1w, "1M": poc1m}
    pocs   = await loop.run_in_executor(None, calc_all_pocs, df, tf, show)
    result = pocs_to_json(pocs, max_display=MAX_DISPLAY.get(tf, MAX_DISPLAY_DEFAULT))
    _cache_set(cache_key, result)
    return result


@app.post("/api/reload/{symbol:path}")
async def api_reload(symbol: str):
    """
    Принудительная перезагрузка данных монеты:
    - удаляет chart из RAM и кэш на диске
    - заново скачивает полную историю с биржи
    - сбрасывает result_cache для этого символа
    """
    import os as _os
    m = get_manager()

    # Удаляем chart из RAM
    if symbol in m.store._chart:
        del m.store._chart[symbol]
    if symbol in m.store._chart_access:
        del m.store._chart_access[symbol]

    # Удаляем кэш с диска
    from data_manager import _cache_path
    cache_file = _cache_path(symbol, "1m")
    if _os.path.exists(cache_file):
        try:
            _os.remove(cache_file)
        except Exception as e:
            logger.warning(f"reload: не удалось удалить кэш {symbol}: {e}")

    # Чистим result_cache для этого символа
    stale_keys = [k for k in _result_cache if len(k) > 1 and k[1] == symbol]
    for k in stale_keys:
        del _result_cache[k]

    # Перезагружаем полную историю
    loop = asyncio.get_event_loop()
    try:
        ok = await loop.run_in_executor(None, m._load_chart, symbol)
    except Exception as e:
        raise HTTPException(500, f"Ошибка загрузки {symbol}: {e}")

    if not ok:
        raise HTTPException(500, f"Не удалось загрузить данные для {symbol}")

    df = m.store.get(symbol, "1m")
    bars = len(df) if df is not None else 0
    logger.info(f"reload: {symbol} перезагружен, {bars} баров")
    return {"ok": True, "bars": bars, "symbol": symbol}



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
    """Возвращает стакан из локального order book (без REST запросов к Binance)."""
    ob = get_ob_manager().get(symbol)
    if ob is None:
        raise HTTPException(503, detail={"reason": "no_book", "msg": f"Order book not initialized for {symbol}"})
    if not ob.synced:
        raise HTTPException(503, detail={
            "reason":          "not_synced",
            "msg":             "Order book syncing, please retry in a few seconds",
            "last_update_id":  ob.last_update_id,
            "buf_size":        len(ob._buf),
        })
    depth = max(5, min(depth, 1000))
    bids  = ob.get_bids(depth)
    asks  = ob.get_asks(depth)
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


# ──────────────────────────────────────────────────────────────────────────────
# DOM Anomaly Detector  —  локальный order book через diff stream
# ──────────────────────────────────────────────────────────────────────────────

_DOM_ANOMALIES:   dict  = {}
_DOM_FIRST_SEEN:  dict  = collections.defaultdict(dict)
_DOM_LAST_DETECT: dict  = {}          # symbol → время последнего детекта

_DOM_DEPTH        = 1000              # уровней для детектора (из локального book)
_DOM_MIN_PERSIST  = 120                # сек — стена должна простоять
_DOM_ANOMALY_MULT = 50.0               # score порог
_DOM_MIN_USD      = 50000.0               # мин. размер стены в USD
_DETECT_INTERVAL  = 10                # сек между запусками детектора на символ

_ATR_PERIOD       = 100               # баров 5m для ATR
_ATR_MULT         = 5.0               # диапазон = цена ± ATR * mult

# История уровней для проверки персистентности: symbol → {side_price → deque[ts]}
_LEVEL_HISTORY: dict = collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.deque(maxlen=20)
))


def _calc_atr(df_1m, period: int = _ATR_PERIOD) -> float:
    """ATR(period) по 5m барам из 1m mini_store."""
    try:
        df5 = df_1m.resample("5min", label="left", closed="left").agg({
            "high":  "max",
            "low":   "min",
            "close": "last",
        }).dropna()
        if len(df5) < period + 1:
            return 0.0
        df5 = df5.iloc[-(period + 1):]
        h, l, c = df5["high"].values, df5["low"].values, df5["close"].values
        hl  = h[1:] - l[1:]
        hcp = abs(h[1:] - c[:-1])
        lcp = abs(l[1:] - c[:-1])
        tr  = np.maximum(hl, np.maximum(hcp, lcp))
        return float(tr[-period:].mean())
    except Exception:
        return 0.0


def _detect_anomalies(symbol: str, current_price: float, df_1m=None) -> list:
    """
    Детектор аномальных стен в стакане.
    Медиана считается по ATR-зоне (цена ± ATR*5 по 5m барам).
    Проверяется весь стакан на глубину _DOM_DEPTH.
    Персистентность: уровень должен присутствовать >= _DOM_MIN_PERSIST сек.
    """
    if current_price <= 0:
        return []

    ob = get_ob_manager().get(symbol)
    if ob is None or not ob.synced:
        return []

    bids = ob.get_bids(_DOM_DEPTH)   # [(price, qty), ...] сортировка ↓
    asks = ob.get_asks(_DOM_DEPTH)   # [(price, qty), ...] сортировка ↑

    if not bids and not asks:
        return []

    now = _time.time()

    # ── Обновляем историю уровней для проверки персистентности ───────────────
    hist = _LEVEL_HISTORY[symbol]
    seen_keys = set()
    for price, qty in bids:
        k = f"bid_{round(price, 8)}"
        hist[k].append(now)
        seen_keys.add(k)
    for price, qty in asks:
        k = f"ask_{round(price, 8)}"
        hist[k].append(now)
        seen_keys.add(k)
    # Чистим исчезнувшие уровни
    gone = [k for k in hist if k not in seen_keys]
    for k in gone:
        del hist[k]

    # ── ATR-диапазон для медианы ──────────────────────────────────────────────
    atr = _calc_atr(df_1m) if df_1m is not None else 0.0
    all_levels = bids + asks

    if atr > 0:
        lo = current_price - atr * _ATR_MULT
        hi = current_price + atr * _ATR_MULT
        near = [(p, q) for p, q in all_levels if lo <= p <= hi]
    else:
        near = sorted(all_levels, key=lambda x: abs(x[0] - current_price))
        near = near[:max(5, len(near) // 2)]

    if len(near) < 3:
        near = all_levels

    # Робастная медиана: нижние 90% по qty
    zone_qtys = sorted(q for _, q in near)
    cutoff    = max(1, int(len(zone_qtys) * 0.90))
    median_qty = statistics.median(zone_qtys[:cutoff])
    if median_qty <= 0:
        return []

    # ── Поиск аномалий ────────────────────────────────────────────────────────
    fs       = _DOM_FIRST_SEEN[symbol]
    seen_now = set()
    anomalies = []

    for side, levels in [("bid", bids), ("ask", asks)]:
        for price, qty in levels:
            rounded = round(price, 8)
            key     = f"{side}_{rounded}"
            hist_k  = hist.get(key)

            # Персистентность: уровень должен быть в истории >= _DOM_MIN_PERSIST сек
            if not hist_k or len(hist_k) < 2:
                continue
            age_in_book = now - hist_k[0]
            if age_in_book < _DOM_MIN_PERSIST:
                continue

            score = qty / median_qty
            if score < _DOM_ANOMALY_MULT:
                continue

            usd_val = qty * price
            if _DOM_MIN_USD > 0 and usd_val < _DOM_MIN_USD:
                continue

            seen_now.add(key)
            if key not in fs:
                fs[key] = now

            distance_pct = abs(price - current_price) / current_price * 100
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

    # Чистим first_seen для исчезнувших уровней
    stale = [k for k in fs if k not in seen_now and now - fs[k] > 28800]
    for k in stale:
        del fs[k]

    anomalies.sort(key=lambda x: x["distance_pct"])
    return anomalies[:5]


import re as _re

# Символы-стейблы и прочий мусор которые не нужны в Big Bid/Ask
_EXCLUDE_BASES = {
    # Стейблы
    "USDC", "USDE", "USDP", "TUSD", "BUSD", "FDUSD", "PYUSD", "USDD",
    "EURC", "EURI", "EURS", "USDT", "DAI", "FRAX", "LUSD", "GUSD",
    "SUSD", "CUSD", "CEUR", "HUSD", "USDX", "USDK", "USDJ",
    # Прочие исключения
    "BTCDOM", "DEFI",
}
# Паттерны для исключения (квартальные фьючи: содержат 6-значную дату типа 260626)
_EXCLUDE_PATTERN = _re.compile(r"-\d{6}$|_\d{6}$|\d{6}$")

# Китайский мем-токен
_EXCLUDE_NAMES = {"我踏马来了", "WTMLL"}


def _should_skip_anomaly(symbol: str) -> bool:
    """Возвращает True если символ не нужно проверять на аномалии стакана."""
    base = symbol.split("/")[0].upper()
    # Стейблы и спецсимволы
    if base in _EXCLUDE_BASES:
        return True
    # Квартальные фьючи с датой в тикере (BTC/USDT:USDT-260626, ETH-260327 и т.п.)
    # Дата может быть как в base так и в суффиксе после :
    if _EXCLUDE_PATTERN.search(symbol):
        return True
    # Китайские/Unicode тикеры — содержат не-ASCII символы
    if not base.isascii():
        return True
    # Явные имена
    if base in _EXCLUDE_NAMES:
        return True
    return False


async def _anomaly_detect_loop():
    """
    Периодически запускает детектор аномалий для всех символов.
    Не чаще _DETECT_INTERVAL сек на символ.
    """
    await asyncio.sleep(30)  # ждём пока стаканы синхронизируются
    while True:
        try:
            m   = get_manager()
            obm = get_ob_manager()
            now = _time.time()

            for sym in m.store.symbols:
                if _should_skip_anomaly(sym):
                    continue
                if now - _DOM_LAST_DETECT.get(sym, 0) < _DETECT_INTERVAL:
                    continue
                ob = obm.get(sym)
                if ob is None or not ob.synced:
                    continue
                _DOM_LAST_DETECT[sym] = now
                df        = m.store.get_mini(sym)
                cur_price = float(df.iloc[-1]["close"]) if df is not None and not df.empty else 0
                _DOM_ANOMALIES[sym] = _detect_anomalies(sym, cur_price, df_1m=df)

            await asyncio.sleep(2)   # проверяем каждые 2 сек

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"anomaly_detect_loop: {e}")
            await asyncio.sleep(5)


@app.post("/api/set_limit_mult")
async def api_set_limit_mult(mult: float = 8.0):
    global _DOM_ANOMALY_MULT
    _DOM_ANOMALY_MULT = max(2.0, min(100.0, mult))
    _DOM_ANOMALIES.clear()
    return {"mult": _DOM_ANOMALY_MULT}


@app.post("/api/set_limit_params")
async def api_set_limit_params(mult: float = 8.0):
    global _DOM_ANOMALY_MULT
    _DOM_ANOMALY_MULT = max(2.0, min(100.0, mult))
    _DOM_ANOMALIES.clear()
    return {"mult": _DOM_ANOMALY_MULT, "depth": _DOM_DEPTH, "min_usd": _DOM_MIN_USD}


@app.get("/api/limit_levels/{symbol:path}")
async def api_limit_levels(symbol: str):
    return _DOM_ANOMALIES.get(symbol, [])


@app.get("/api/dom_debug/{symbol:path}")
async def api_dom_debug(symbol: str):
    ob  = get_ob_manager().get(symbol)
    now = _time.time()
    return {
        "synced":          ob.synced if ob else False,
        "last_update_id":  ob.last_update_id if ob else None,
        "bids_count":      len(ob.bids) if ob else 0,
        "asks_count":      len(ob.asks) if ob else 0,
        "best_bid":        ob.best_bid() if ob else None,
        "best_ask":        ob.best_ask() if ob else None,
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
