"""
dom_orderbook.py — локальный order book через Binance Futures diff stream

Алгоритм (по документации Binance):
  1. Подписываемся на WS стрим <symbol>@depth@500ms (буферизируем события)
  2. Через REST /fapi/v1/depth?limit=1000 получаем снапшот (lastUpdateId)
  3. Отбрасываем буферизированные события где u < lastUpdateId
  4. Первое валидное: U <= lastUpdateId+1 <= u
  5. Далее каждое событие: pu == предыдущий u (иначе рестарт)
  6. Применяем дельту: qty==0 → удалить уровень, иначе → обновить

Итог: полный стакан глубиной 1000 уровней, актуальный каждые 500мс,
      без REST запросов в реальном времени.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import requests
import websockets

logger = logging.getLogger(__name__)

# ── Константы ─────────────────────────────────────────────────────────────────
_WS_BASE          = "wss://fstream.binance.com/stream"
_REST_URL         = "https://fapi.binance.com/fapi/v1/depth"
_SNAPSHOT_DEPTH   = 1000      # уровней в REST снапшоте
_WS_CHUNK_SIZE    = 200       # символов на одно WS соединение
_SNAP_RATE        = 2         # REST снапшотов в секунду при инициализации (2/s = 120/min)
_REST_SESSION     = requests.Session()
_REST_SESSION.headers.update({"Connection": "keep-alive"})


def _ccxt_to_binance(sym: str) -> str:
    """'BTC/USDT:USDT' → 'BTCUSDT'"""
    base  = sym.split("/")[0]
    quote = sym.split("/")[1].split(":")[0]
    return (base + quote).upper()


# ── Order Book ────────────────────────────────────────────────────────────────

class OrderBook:
    """
    Локальная копия стакана одного символа.
    bids/asks хранятся как dict {price_str → qty_float} для O(1) обновлений.
    """
    __slots__ = ("bids", "asks", "last_update_id", "synced", "_buf")

    def __init__(self):
        self.bids:           Dict[str, float] = {}
        self.asks:           Dict[str, float] = {}
        self.last_update_id: int              = 0
        self.synced:         bool             = False
        self._buf:           list             = []   # буфер событий до синхронизации

    def apply_snapshot(self, data: dict):
        """Применяет REST снапшот."""
        self.bids = {p: float(q) for p, q in data["bids"]}
        self.asks = {p: float(q) for p, q in data["asks"]}
        self.last_update_id = data["lastUpdateId"]
        self.synced = False   # будет True после первого валидного WS события

    def apply_event(self, event: dict) -> bool:
        """
        Применяет WS diff событие.
        Возвращает False если нужна ресинхронизация.
        """
        U  = event["U"]   # first update id
        u  = event["u"]   # final update id
        pu = event.get("pu", None)  # previous final update id

        if not self.synced:
            # Ждём первое валидное событие: U <= lastUpdateId+1 <= u
            if u < self.last_update_id:
                return True   # устаревшее — пропускаем
            if U > self.last_update_id + 1:
                return False  # пропустили событие — рестарт
            self.synced = True
        else:
            # Проверяем непрерывность: pu должен == нашему last_update_id
            if pu is not None and pu != self.last_update_id:
                return False  # разрыв — рестарт

        self._apply_delta(self.bids, event.get("b", []))
        self._apply_delta(self.asks, event.get("a", []))
        self.last_update_id = u
        return True

    @staticmethod
    def _apply_delta(side: dict, updates: list):
        for price, qty in updates:
            qty_f = float(qty)
            if qty_f == 0.0:
                side.pop(price, None)
            else:
                side[price] = qty_f

    def get_bids(self, depth: int = 1000) -> List[Tuple[float, float]]:
        """Возвращает список (price, qty) отсортированный по убыванию цены."""
        items = sorted(
            ((float(p), q) for p, q in self.bids.items()),
            key=lambda x: -x[0]
        )
        return items[:depth]

    def get_asks(self, depth: int = 1000) -> List[Tuple[float, float]]:
        """Возвращает список (price, qty) отсортированный по возрастанию цены."""
        items = sorted(
            ((float(p), q) for p, q in self.asks.items()),
            key=lambda x: x[0]
        )
        return items[:depth]

    def best_bid(self) -> Optional[float]:
        if not self.bids:
            return None
        return max(float(p) for p in self.bids)

    def best_ask(self) -> Optional[float]:
        if not self.asks:
            return None
        return min(float(p) for p in self.asks)


# ── OrderBook Manager ─────────────────────────────────────────────────────────

class OrderBookManager:
    """
    Управляет локальными стаканами для всех символов.
    Публичный интерфейс:
        await manager.start(symbols)  — запустить
        manager.get(symbol)           — получить OrderBook
        await manager.stop()          — остановить
    """

    def __init__(self):
        self._books:    Dict[str, OrderBook]       = {}
        self._running:  bool                       = False
        self._tasks:    List[asyncio.Task]         = []
        # Очередь для ресинхронизации: символы которым нужен новый снапшот
        self._resync_q: asyncio.Queue              = asyncio.Queue()

    def get(self, symbol: str) -> Optional[OrderBook]:
        return self._books.get(symbol)

    def get_bids_asks(self, symbol: str, depth: int = 1000):
        """Удобный метод: возвращает (bids, asks) или ([], [])."""
        book = self._books.get(symbol)
        if book is None or not book.synced:
            return [], []
        return book.get_bids(depth), book.get_asks(depth)

    async def start(self, symbols: List[str]):
        self._running = True

        # Инициализируем пустые стаканы
        for sym in symbols:
            self._books[sym] = OrderBook()

        # Разбиваем на чанки для WS
        chunks = [
            symbols[i:i + _WS_CHUNK_SIZE]
            for i in range(0, len(symbols), _WS_CHUNK_SIZE)
        ]
        logger.info(f"OrderBook: {len(symbols)} символов → {len(chunks)} WS соединений")

        # Запускаем WS чанки — они начнут буферизировать события
        for chunk in chunks:
            t = asyncio.create_task(self._ws_chunk(chunk))
            self._tasks.append(t)

        # Загружаем REST снапшоты с ограничением скорости
        t = asyncio.create_task(self._snapshot_loader(symbols))
        self._tasks.append(t)

        # Обработчик ресинхронизаций
        t = asyncio.create_task(self._resync_worker())
        self._tasks.append(t)

    async def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # ── REST снапшот ──────────────────────────────────────────────────────────

    def _fetch_snapshot(self, binance_sym: str) -> Optional[dict]:
        """Синхронный REST запрос снапшота. При 429 — ждёт и повторяет."""
        for attempt in range(5):
            try:
                r = _REST_SESSION.get(
                    _REST_URL,
                    params={"symbol": binance_sym, "limit": _SNAPSHOT_DEPTH},
                    timeout=10,
                )
                # Читаем weight — при приближении к лимиту притормаживаем
                used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", 0))
                if used > 1800:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Binance weight {used}/2400 — sleeping {wait}s")
                    time.sleep(wait)

                if r.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"snapshot {binance_sym}: 429, retry in {wait}s")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                return r.json()
            except Exception as e:
                if "429" in str(e) or "Too Many" in str(e):
                    wait = 30 * (attempt + 1)
                    logger.warning(f"snapshot {binance_sym}: rate limit, retry in {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"snapshot {binance_sym}: {e}")
                    return None
        logger.warning(f"snapshot {binance_sym}: all retries failed")
        return None

    async def _snapshot_loader(self, symbols: List[str]):
        """
        Загружает REST снапшоты для всех символов с ограничением скорости.
        _SNAP_RATE штук в секунду.
        """
        loop     = asyncio.get_event_loop()
        interval = 1.0 / _SNAP_RATE   # сек между запросами

        for sym in symbols:
            if not self._running:
                return
            bsym = _ccxt_to_binance(sym)
            data = await loop.run_in_executor(None, self._fetch_snapshot, bsym)
            if data:
                book = self._books.get(sym)
                if book:
                    book.apply_snapshot(data)
                    # Применяем накопленный буфер
                    self._drain_buffer(sym, book)
                    logger.debug(f"snapshot OK: {sym} lastUpdateId={book.last_update_id}")
            await asyncio.sleep(interval)

        logger.info("OrderBook: все снапшоты загружены")

    def _drain_buffer(self, symbol: str, book: OrderBook):
        """Применяет буферизированные WS события после получения снапшота."""
        buf = book._buf
        book._buf = []
        for event in buf:
            ok = book.apply_event(event)
            if not ok:
                # Нужна ресинхронизация
                self._resync_q.put_nowait(symbol)
                return

    # ── WS чанк ───────────────────────────────────────────────────────────────

    async def _ws_chunk(self, symbols: List[str]):
        """
        Один WS коннект для чанка символов.
        Получает diff события и либо буферизирует (до снапшота),
        либо применяет к стакану.
        """
        streams  = [f"{_ccxt_to_binance(s).lower()}@depth@500ms" for s in symbols]
        url      = f"{_WS_BASE}?streams={'/'.join(streams)}"
        sym_map  = {_ccxt_to_binance(s).lower(): s for s in symbols}
        backoff  = 1

        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=16 * 1024 * 1024,
                ) as ws:
                    logger.info(f"OrderBook WS подключён: {len(symbols)} символов")
                    backoff = 1

                    async for raw in ws:
                        if not self._running:
                            return
                        try:
                            msg      = json.loads(raw)
                            stream   = msg.get("stream", "")
                            bsym_low = stream.split("@")[0]
                            ccxt_sym = sym_map.get(bsym_low)
                            if ccxt_sym is None:
                                continue

                            event = msg.get("data", msg)
                            book  = self._books.get(ccxt_sym)
                            if book is None:
                                continue

                            if not book.synced and book.last_update_id == 0:
                                # Снапшот ещё не получен — буферизируем
                                book._buf.append(event)
                                continue

                            if not book.synced:
                                # Снапшот есть — применяем из буфера
                                book._buf.append(event)
                                self._drain_buffer(ccxt_sym, book)
                                continue

                            # Нормальная работа — применяем напрямую
                            ok = book.apply_event(event)
                            if not ok:
                                logger.debug(f"OrderBook desync: {ccxt_sym}, resync...")
                                self._resync_q.put_nowait(ccxt_sym)
                                book.synced = False
                                book.last_update_id = 0

                        except Exception as e:
                            logger.debug(f"OrderBook WS parse: {e}")

            except asyncio.CancelledError:
                return
            except Exception as e:
                if self._running:
                    logger.warning(f"OrderBook WS error: {e}, reconnect in {backoff}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)

    # ── Ресинхронизация ───────────────────────────────────────────────────────

    async def _resync_worker(self):
        """
        Обрабатывает очередь ресинхронизаций.
        Берёт символ, ждёт 1 сек (чтобы не спамить), загружает новый снапшот.
        """
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                sym = await asyncio.wait_for(self._resync_q.get(), timeout=5.0)
                await asyncio.sleep(1.0)  # небольшая пауза перед запросом

                bsym = _ccxt_to_binance(sym)
                data = await loop.run_in_executor(None, self._fetch_snapshot, bsym)
                if data:
                    book = self._books.get(sym)
                    if book:
                        book.apply_snapshot(data)
                        self._drain_buffer(sym, book)
                        logger.debug(f"resync OK: {sym}")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"resync_worker: {e}")


# ── Глобальный экземпляр ──────────────────────────────────────────────────────

_ob_manager: Optional[OrderBookManager] = None


def get_ob_manager() -> OrderBookManager:
    global _ob_manager
    if _ob_manager is None:
        _ob_manager = OrderBookManager()
    return _ob_manager