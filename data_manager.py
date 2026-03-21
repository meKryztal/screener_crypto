"""
data_manager.py v8 — два уровня хранения:

  mini_store  — последние MINI_BARS баров 1m для ВСЕХ символов (скринер)
  chart_store — полная история для АКТИВНЫХ символов (график)
                выгружается из RAM через CHART_EVICT_SEC неактивности
                кэш на диске живёт вечно, старые бары (>1 нед) обрезаются при загрузке
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import ccxt.pro as ccxtpro
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

MARKET_TYPE = "future"
LTF         = "1m"
ALL_TFS     = ["1m", "5m", "15m", "1h"]
CACHE_DIR   = "cache"

# Сколько баров держим в mini_store (для скринера vol/spike)
MINI_BARS = 2000          # 1500 × 1m = 25 часов — достаточно для vol24h

# Полная история для графика
CHART_LIMIT = 50400        # баров 1m в chart_store

# Через сколько секунд неактивности выгружать символ из chart_store
CHART_EVICT_SEC = 1200     # 10 минут

# Неделя в секундах — старше этого обрезаем из кэша
CACHE_WEEK_SEC  = 35 * 24 * 3600

MAX_WORKERS = 3

TF_MS = {
    "1m":  60_000,   "3m":  180_000,  "5m":  300_000,
    "15m": 900_000,  "30m": 1_800_000,"1h":  3_600_000,
    "2h":  7_200_000,"4h":  14_400_000,"1d": 86_400_000,
}

OI_INTERVAL   = 60
OI_HIST_LIMIT = 1000


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _cache_path(symbol: str, tf: str) -> str:
    safe = symbol.replace("/", "_").replace(":", "_")
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{safe}_{tf}.parquet")


def cache_save(symbol: str, tf: str, df: pd.DataFrame):
    try:
        df.to_parquet(_cache_path(symbol, tf), compression="snappy")
    except Exception as e:
        logger.warning(f"cache_save {symbol} {tf}: {e}")


def cache_load(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    """Load cache. Always returns data regardless of file age — we trim inside."""
    path = _cache_path(symbol, tf)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        # Trim rows older than 1 week — keeps recent data, removes old
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=CACHE_WEEK_SEC)
        df = df[df.index >= cutoff]
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"cache_load {symbol} {tf}: {e}")
        return None


def fetch_ohlcv_sync(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    max_per  = 1000
    ms_bar   = TF_MS.get(timeframe, 60_000)
    since    = exchange.milliseconds() - limit * ms_bar
    all_data = []

    while len(all_data) < limit:
        batch = min(max_per, limit - len(all_data))
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch)
        except Exception as e:
            logger.warning(f"fetch {symbol} {timeframe}: {e}")
            break
        if not raw:
            break
        all_data.extend(raw)
        since = raw[-1][0] + 1
        if len(raw) < batch:
            break

    if not all_data:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    df = pd.DataFrame(all_data, columns=["ts","open","high","low","close","volume"])
    df.drop_duplicates("ts", inplace=True)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df


# ─── DataStore ────────────────────────────────────────────────────────────────

class DataStore:
    def __init__(self):
        # mini: symbol → last MINI_BARS 1m bars (always in RAM for all symbols)
        self._mini:  Dict[str, pd.DataFrame]            = {}
        # chart: symbol → full history 1m (only active symbols)
        self._chart: Dict[str, pd.DataFrame]            = {}
        # last access time for chart eviction
        self._chart_access: Dict[str, float]            = {}

        self._locks: Dict[str, asyncio.Lock]            = defaultdict(asyncio.Lock)
        self._subs:  Dict[str, Set[asyncio.Queue]]      = defaultdict(set)
        self.symbols:      List[str]                    = []
        self.ready                                      = False
        self._loaded:      Set[str]                     = set()
        self._loaded_tfs:  Dict[str, Set[str]]          = defaultdict(set)
        self._last_prices: Dict[str, float]             = {}
        self._oi:  Dict[str, pd.DataFrame]              = {}

    # ── mini store ──

    def init_mini(self, symbol: str, df: pd.DataFrame):
        """Store last MINI_BARS rows in mini store."""
        self._mini[symbol] = df.iloc[-MINI_BARS:].copy() if len(df) > MINI_BARS else df.copy()
        self._loaded.add(symbol)
        self._loaded_tfs[symbol].add("1m")

    def get_mini(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._mini.get(symbol)

    # ── chart store ──

    def init_chart(self, symbol: str, df: pd.DataFrame):
        """Store full history in chart store, update access time."""
        self._chart[symbol] = df.iloc[-CHART_LIMIT:].copy() if len(df) > CHART_LIMIT else df.copy()
        self._chart_access[symbol] = time.monotonic()

    def get_chart(self, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self._chart:
            self._chart_access[symbol] = time.monotonic()
            return self._chart[symbol]
        return None

    def has_chart(self, symbol: str) -> bool:
        return symbol in self._chart

    def evict_stale_charts(self):
        """Remove chart data for symbols not accessed recently."""
        now = time.monotonic()
        to_evict = [
            sym for sym, t in self._chart_access.items()
            if now - t > CHART_EVICT_SEC
        ]
        for sym in to_evict:
            del self._chart[sym]
            del self._chart_access[sym]
            logger.debug(f"chart evicted: {sym}")

    # ── legacy get() — returns chart if available, else mini ──

    def get(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        """Compatibility: returns chart data (full) or mini data."""
        if tf == "1m":
            chart = self.get_chart(symbol)
            if chart is not None:
                return chart
            return self.get_mini(symbol)
        return None

    def needs_load(self, symbol: str, tf: str) -> bool:
        return symbol not in self._loaded

    def is_tf_loaded(self, symbol: str, tf: str) -> bool:
        return tf in self._loaded_tfs.get(symbol, set())

    def has_bar_subscribers(self, symbol: str) -> bool:
        for tf in ALL_TFS:
            if self._subs.get(f"{symbol}:{tf}"):
                return True
        return False

    @staticmethod
    def _upsert_bar(df: pd.DataFrame, ts, row: dict, limit: int) -> pd.DataFrame:
        """
        Вставляет или обновляет бар в DataFrame.
        Оптимизация: сравниваем только с последним индексом (O(1))
        вместо `ts in df.index` (O(n) линейный поиск).
        """
        if len(df) > 0 and df.index[-1] == ts:
            # Обновляем последний бар (самый частый случай при live-данных)
            for k, v in row.items():
                df.iat[-1, df.columns.get_loc(k)] = v
        else:
            df.loc[ts] = row
            if len(df) > limit + 200:
                return df.iloc[-limit:]
        return df

    async def update_bar(self, symbol: str, tf: str, bar: dict, closed: bool):
        async with self._locks[symbol]:
            ts  = bar["ts"]
            row = {k: bar[k] for k in ["open","high","low","close","volume"]}

            # Update mini store
            mini = self._mini.get(symbol)
            if mini is not None:
                self._mini[symbol] = self._upsert_bar(mini, ts, row, MINI_BARS)

            # Update chart store if loaded
            chart = self._chart.get(symbol)
            if chart is not None:
                self._chart[symbol] = self._upsert_bar(chart, ts, row, CHART_LIMIT)

        price = bar.get("close")
        if price:
            self._last_prices[symbol] = float(price)
        await self._notify(symbol, tf, bar, closed)

    async def _notify(self, symbol: str, tf: str, bar: dict, closed: bool):
        key  = f"{symbol}:{tf}"
        dead = set()
        for q in list(self._subs.get(key, set())):
            try:
                q.put_nowait({"bar": bar, "closed": closed, "symbol": symbol, "tf": tf})
            except asyncio.QueueFull:
                dead.add(q)
        self._subs[key] -= dead

    def subscribe(self, symbol: str, tf: str) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=500)
        self._subs[f"{symbol}:{tf}"].add(q)
        return q

    def unsubscribe(self, symbol: str, tf: str, q: asyncio.Queue):
        self._subs[f"{symbol}:{tf}"].discard(q)

    # ── OI ──
    def set_oi(self, symbol: str, tf: str, df: pd.DataFrame):
        self._oi[f"{symbol}:{tf}"] = df

    def get_oi(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        return self._oi.get(f"{symbol}:{tf}")


# ─── DataManager ──────────────────────────────────────────────────────────────

class DataManager:
    def __init__(self):
        self.store    = DataStore()
        self._running = False
        self._ws_task:  Optional[asyncio.Task] = None
        self._oi_task:  Optional[asyncio.Task] = None
        self._evict_task: Optional[asyncio.Task] = None

    def _make_sync_ex(self):
        return (ccxt.binanceusdm if MARKET_TYPE == "future" else ccxt.binance)(
            {"enableRateLimit": True}
        )

    def _make_ws_ex(self):
        return (ccxtpro.binanceusdm if MARKET_TYPE == "future" else ccxtpro.binance)(
            {"enableRateLimit": True}
        )

    async def fetch_all_futures_symbols(self) -> List[str]:
        loop = asyncio.get_event_loop()
        def _fetch():
            ex = self._make_sync_ex()
            markets = ex.load_markets()
            syms = []
            for sym, m in markets.items():
                if (m.get("quote") == "USDT" and
                    m.get("type") in ("swap", "future") and
                    m.get("active", True) and
                    ":USDT" in sym):
                    syms.append(sym)
            return sorted(syms)
        try:
            symbols = await loop.run_in_executor(None, _fetch)
            logger.info(f"Загружено {len(symbols)} фьючей")
            return symbols
        except Exception as e:
            logger.error(f"fetch_all_futures_symbols: {e}")
            return ["BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT"]

    def _load_mini(self, symbol: str) -> bool:
        """Load only MINI_BARS into mini store for screener. Fast."""
        cached = cache_load(symbol, "1m")
        if cached is not None:
            # Incremental: fetch only missing bars since last cached
            last_ts_ms = int(cached.index[-1].timestamp() * 1000)
            ms_bar     = TF_MS["1m"]
            now_ms     = int(time.time() * 1000)
            missing    = (now_ms - last_ts_ms) // ms_bar
            if missing > 1:
                try:
                    ex  = self._make_sync_ex()
                    raw = ex.fetch_ohlcv(symbol, "1m", since=last_ts_ms,
                                         limit=min(missing + 5, 1000))
                    if raw:
                        df_new = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
                        df_new["ts"] = pd.to_datetime(df_new["ts"], unit="ms", utc=True)
                        df_new.set_index("ts", inplace=True)
                        combined = pd.concat([cached, df_new])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined.sort_index(inplace=True)
                        cached = combined
                        # Save full updated cache to disk (trimming handled in cache_load)
                        cache_save(symbol, "1m", cached)
                except Exception as e:
                    logger.warning(f"mini incremental {symbol}: {e}")
            self.store.init_mini(symbol, cached)
            return True

        # No cache — fetch minimal bars for screener
        try:
            ex = self._make_sync_ex()
            df = fetch_ohlcv_sync(ex, symbol, "1m", MINI_BARS)
            if df.empty:
                return False
            cache_save(symbol, "1m", df)
            self.store.init_mini(symbol, df)
            return True
        except Exception as e:
            logger.error(f"load_mini {symbol}: {e}")
            return False

    def _load_chart(self, symbol: str) -> bool:
        """Load full chart history (CHART_LIMIT bars). Called on demand when user opens a symbol."""
        ex     = self._make_sync_ex()
        cached = cache_load(symbol, "1m")

        if cached is not None:
            # How many bars are missing from cache end to now
            last_ts_ms = int(cached.index[-1].timestamp() * 1000)
            now_ms     = int(time.time() * 1000)
            missing    = (now_ms - last_ts_ms) // TF_MS["1m"]

            # Also check if cache has fewer bars than CHART_LIMIT — backfill if so
            need_backfill = len(cached) < CHART_LIMIT

            if need_backfill:
                # Fetch full history — cache only had mini bars
                try:
                    df_full = fetch_ohlcv_sync(ex, symbol, "1m", CHART_LIMIT)
                    if not df_full.empty:
                        combined = pd.concat([df_full, cached])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined.sort_index(inplace=True)
                        cached = combined.iloc[-CHART_LIMIT:]
                        cache_save(symbol, "1m", cached)
                        logger.info(f"chart backfilled: {symbol} ({len(cached)} bars)")
                except Exception as e:
                    logger.warning(f"chart backfill {symbol}: {e}")
            elif missing > 1:
                # Incremental: only fetch what's new
                try:
                    raw = ex.fetch_ohlcv(symbol, "1m", since=last_ts_ms,
                                         limit=min(missing + 5, 1000))
                    if raw:
                        df_new = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
                        df_new["ts"] = pd.to_datetime(df_new["ts"], unit="ms", utc=True)
                        df_new.set_index("ts", inplace=True)
                        combined = pd.concat([cached, df_new])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined.sort_index(inplace=True)
                        cached = combined.iloc[-CHART_LIMIT:]
                        cache_save(symbol, "1m", cached)
                        logger.info(f"chart incremental: {symbol} +{len(df_new)} bars")
                except Exception as e:
                    logger.warning(f"chart incremental {symbol}: {e}")
            else:
                logger.info(f"chart cache hit: {symbol} ({len(cached)} bars)")

            self.store.init_chart(symbol, cached)
            return True

        # No cache at all — fetch full history
        try:
            df = fetch_ohlcv_sync(ex, symbol, "1m", CHART_LIMIT)
            if df.empty:
                return False
            cache_save(symbol, "1m", df)
            self.store.init_chart(symbol, df)
            logger.info(f"chart loaded fresh: {symbol} ({len(df)} bars)")
            return True
        except Exception as e:
            logger.error(f"load_chart {symbol}: {e}")
            return False

    # ── OI ──────────────────────────────────────────────────────────────────

    _OI_TF_MAP = {
        "1m": "5m", "5m": "5m", "15m": "15m",
        "1h": "1h", "4h": "4h", "1d":  "1d",
    }
    _OI_PAGES = {
        "5m": 6, "15m": 4, "1h": 2, "4h": 2, "1d": 1,
    }

    def _fetch_oi_history(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        import requests
        period  = self._OI_TF_MAP.get(tf, "1h")
        n_pages = self._OI_PAGES.get(period, 2)
        try:
            base        = symbol.split("/")[0]
            quote       = symbol.split("/")[1].split(":")[0]
            binance_sym = base + quote
            url         = "https://fapi.binance.com/futures/data/openInterestHist"
            all_rows = []
            end_time = None
            for _ in range(n_pages):
                params = {"symbol": binance_sym, "period": period, "limit": 500}
                if end_time:
                    params["endTime"] = end_time
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                if not data:
                    break
                all_rows = data + all_rows
                end_time = int(data[0]["timestamp"]) - 1
                if len(data) < 500:
                    break
            if not all_rows:
                return None
            rows = [
                {"ts": pd.Timestamp(int(d["timestamp"]), unit="ms", tz="UTC"),
                 "oi": float(d["sumOpenInterest"])}
                for d in all_rows
            ]
            df = pd.DataFrame(rows).set_index("ts")
            df = df[~df.index.duplicated(keep="last")].sort_index()
            logger.info(f"OI loaded: {symbol} {tf} ({period}) — {len(df)} bars")
            return df
        except Exception as e:
            logger.warning(f"fetch_oi_history {symbol} {tf}: {e}")
            return None

    def _fetch_oi_snapshot(self, symbol: str) -> Optional[float]:
        import requests
        try:
            base        = symbol.split("/")[0]
            quote       = symbol.split("/")[1].split(":")[0]
            binance_sym = base + quote
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/openInterest",
                params={"symbol": binance_sym}, timeout=5
            )
            r.raise_for_status()
            return float(r.json()["openInterest"])
        except Exception as e:
            logger.debug(f"fetch_oi_snapshot {symbol}: {e}")
            return None

    async def ensure_oi(self, symbol: str, tf: str = "5m"):
        if self.store.get_oi(symbol, tf) is not None:
            return
        loop = asyncio.get_event_loop()
        df   = await loop.run_in_executor(None, self._fetch_oi_history, symbol, tf)
        if df is not None:
            self.store.set_oi(symbol, tf, df)

    async def _oi_poll_loop(self, symbols: List[str]):
        # Poll OI for each cached symbol:tf at the interval matching that TF
        await asyncio.sleep(30)

        _POLL_SEC = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        _FLOOR    = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
        _last_update: Dict[str, float] = {}

        while self._running:
            try:
                loop        = asyncio.get_event_loop()
                cached_keys = list(self.store._oi.keys())
                now         = time.time()

                for key in cached_keys:
                    if not self._running:
                        break
                    sym, tf  = key.rsplit(":", 1)
                    period   = self._OI_TF_MAP.get(tf, "1h")
                    poll_sec = _POLL_SEC.get(period, 300)

                    # Skip if not yet time for this key
                    if now - _last_update.get(key, 0) < poll_sec:
                        continue

                    val = await loop.run_in_executor(None, self._fetch_oi_snapshot, sym)
                    if val is not None:
                        df = self.store.get_oi(sym, tf)
                        if df is not None:
                            ts = pd.Timestamp.now(tz="UTC").floor(_FLOOR.get(period, "5min"))
                            df.loc[ts] = val
                            if len(df) > OI_HIST_LIMIT + 100:
                                self.store.set_oi(sym, tf, df.iloc[-OI_HIST_LIMIT:])
                        _last_update[key] = now
                    await asyncio.sleep(0.2)

                await asyncio.sleep(10)  # check every 10s if any key needs update
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"oi_poll: {e}")
                await asyncio.sleep(10)

    # ── Public API ────────────────────────────────────────────────────────────

    async def load_all_history(self, symbols: List[str]):
        """Startup: load MINI_BARS for all symbols (fast, low RAM)."""
        logger.info(f"Загрузка мини-истории {len(symbols)} монет...")
        sem   = asyncio.Semaphore(MAX_WORKERS)
        done  = 0
        total = len(symbols)

        async def load_one(sym):
            nonlocal done
            async with sem:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_mini, sym)
                done += 1
                if done % 20 == 0 or done == total:
                    logger.info(f"  {done}/{total} загружено")

        await asyncio.gather(*[load_one(s) for s in symbols])
        self.store.ready = True
        logger.info("Мини-история загружена ✓")

    async def ensure_tf(self, symbol: str, tf: str):
        """Called before serving /api/bars — ensures full chart history is loaded."""
        if not self.store.has_chart(symbol):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_chart, symbol)

    async def _evict_loop(self):
        """Periodically evict chart data for inactive symbols."""
        while self._running:
            await asyncio.sleep(60)
            self.store.evict_stale_charts()

    async def _ws_loop(self, symbols: List[str]):
        chunk_size = 50
        chunks = [symbols[i:i+chunk_size] for i in range(0, len(symbols), chunk_size)]
        logger.info(f"WS: {len(symbols)} монет, {len(chunks)} соединений")
        await asyncio.gather(*[self._ws_chunk(chunk) for chunk in chunks])

    async def _ws_chunk(self, symbols: List[str]):
        ex = self._make_ws_ex()
        last_cache_save = time.time()
        CACHE_INTERVAL  = 300

        try:
            while self._running:
                try:
                    watch  = [[s, "1m"] for s in symbols]
                    ohlcvs = await ex.watch_ohlcv_for_symbols(watch)
                    now_ms = int(time.time() * 1000)
                    for symbol, tf_data in ohlcvs.items():
                        for tf, candles in tf_data.items():
                            if not candles:
                                continue
                            c      = candles[-1]
                            ts     = pd.Timestamp(c[0], unit="ms", tz="UTC")
                            closed = now_ms >= c[0] + TF_MS.get(tf, 60_000)
                            bar    = {
                                "ts":     ts,
                                "open":   c[1], "high": c[2],
                                "low":    c[3], "close": c[4],
                                "volume": c[5],
                            }
                            await self.store.update_bar(symbol, "1m", bar, closed)

                    if time.time() - last_cache_save >= CACHE_INTERVAL:
                        last_cache_save = time.time()
                        loop = asyncio.get_event_loop()
                        for sym in symbols:
                            # Save chart data if loaded, else mini
                            df = self.store.get_chart(sym) or self.store.get_mini(sym)
                            if df is not None:
                                await loop.run_in_executor(None, cache_save, sym, "1m", df)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if self._running:
                        logger.warning(f"WS chunk error: {e}, reconnect in 3s...")
                        await asyncio.sleep(3)
        finally:
            try:
                await ex.close()
            except Exception:
                pass

    async def start(self, symbols: List[str]):
        self._running      = True
        self.store.symbols = symbols
        await self.load_all_history(symbols)
        self._ws_task     = asyncio.create_task(self._ws_loop(symbols))
        self._oi_task     = asyncio.create_task(self._oi_poll_loop(symbols))
        self._evict_task  = asyncio.create_task(self._evict_loop())

    async def stop(self):
        self._running = False
        for task in [self._ws_task, self._oi_task, self._evict_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


_manager: Optional[DataManager] = None

def get_manager() -> DataManager:
    global _manager
    if _manager is None:
        _manager = DataManager()
    return _manager
