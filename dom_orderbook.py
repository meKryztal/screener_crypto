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

try:
    from sortedcontainers import SortedList
    _HAVE_SORTED = True
except ImportError:
    _HAVE_SORTED = False
    logging.getLogger(__name__).warning(
        "sortedcontainers not installed — falling back to sorted(). "
        "Run: pip install sortedcontainers"
    )

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

    Хранение уровней:
      _bids_dict / _asks_dict  — dict {price_float → qty_float} для O(1) обновлений.
      _bids_sl   / _asks_sl    — SortedList[price_float] для O(log n) вставки/удаления
                                  и O(1) best_bid/best_ask (первый/последний элемент).

    При наличии пакета sortedcontainers get_bids/get_asks работают за O(depth),
    а не O(N log N) как раньше.  best_bid/best_ask — O(1) вместо O(N).
    Без пакета — автоматический fallback на прежнюю sorted() логику.
    """
    __slots__ = (
        "bids", "asks",                   # публичные dict (совместимость)
        "_bids_sl", "_asks_sl",           # SortedList (None если нет пакета)
        "last_update_id", "synced", "_buf",
    )

    def __init__(self):
        # dict {price_float → qty_float}
        self.bids:           Dict[float, float] = {}
        self.asks:           Dict[float, float] = {}
        # SortedList для быстрого доступа к отсортированным ценам
        if _HAVE_SORTED:
            self._bids_sl = SortedList()   # возрастающий порядок, best = [-1]
            self._asks_sl = SortedList()   # возрастающий порядок, best = [0]
        else:
            self._bids_sl = None
            self._asks_sl = None
        self.last_update_id: int  = 0
        self.synced:         bool = False
        self._buf:           list = []

    def apply_snapshot(self, data: dict):
        """Применяет REST снапшот."""
        self.bids = {float(p): float(q) for p, q in data["bids"]}
        self.asks = {float(p): float(q) for p, q in data["asks"]}
        if _HAVE_SORTED:
            self._bids_sl = SortedList(self.bids.keys())
            self._asks_sl = SortedList(self.asks.keys())
        self.last_update_id = data["lastUpdateId"]
        self.synced = False

    def apply_event(self, event: dict) -> bool:
        """
        Применяет WS diff событие.
        Возвращает False если нужна ресинхронизация.
        """
        U  = event["U"]
        u  = event["u"]
        pu = event.get("pu", None)

        if not self.synced:
            if u < self.last_update_id:
                return True
            if U > self.last_update_id + 1:
                return False
            self.synced = True
        else:
            if pu is not None and pu != self.last_update_id:
                return False

        self._apply_delta(self.bids, self._bids_sl, event.get("b", []))
        self._apply_delta(self.asks, self._asks_sl, event.get("a", []))
        self.last_update_id = u
        return True

    @staticmethod
    def _apply_delta(side_dict: dict, side_sl, updates: list):
        """
        Обновляет dict и SortedList одновременно.
        qty == 0 → удалить уровень, иначе → вставить/обновить.
        """
        for price_s, qty_s in updates:
            price = float(price_s)
            qty   = float(qty_s)
            if qty == 0.0:
                if price in side_dict:
                    del side_dict[price]
                    if side_sl is not None:
                        side_sl.discard(price)
            else:
                if side_sl is not None and price not in side_dict:
                    side_sl.add(price)
                side_dict[price] = qty

    def get_bids(self, depth: int = 1000) -> List[Tuple[float, float]]:
        """
        Возвращает список (price, qty) отсортированный по убыванию цены.
        O(depth) при наличии SortedList, O(N log N) при fallback.
        """
        if _HAVE_SORTED and self._bids_sl is not None:
            sl   = self._bids_sl
            n    = len(sl)
            take = min(depth, n)
            # SortedList хранит в возрастающем порядке — берём с конца
            return [(sl[n - 1 - i], self.bids[sl[n - 1 - i]]) for i in range(take)]
        # fallback
        items = sorted(self.bids.items(), key=lambda x: -x[0])
        return items[:depth]

    def get_asks(self, depth: int = 1000) -> List[Tuple[float, float]]:
        """
        Возвращает список (price, qty) отсортированный по возрастанию цены.
        O(depth) при наличии SortedList, O(N log N) при fallback.
        """
        if _HAVE_SORTED and self._asks_sl is not None:
            sl   = self._asks_sl
            take = min(depth, len(sl))
            return [(sl[i], self.asks[sl[i]]) for i in range(take)]
        # fallback
        items = sorted(self.asks.items(), key=lambda x: x[0])
        return items[:depth]

    def best_bid(self) -> Optional[float]:
        """O(1) при наличии SortedList, O(N) при fallback."""
        if not self.bids:
            return None
        if _HAVE_SORTED and self._bids_sl:
            return self._bids_sl[-1]   # максимум — последний элемент
        return max(self.bids.keys())

    def best_ask(self) -> Optional[float]:
        """O(1) при наличии SortedList, O(N) при fallback."""
        if not self.asks:
            return None
        if _HAVE_SORTED and self._asks_sl:
            return self._asks_sl[0]    # минимум — первый элемент
        return min(self.asks.keys())


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
        # Создаём в start() чтобы попасть в правильный event loop (Python 3.8)
        self._resync_q: asyncio.Queue              = None

    def get(self, symbol: str) -> Optional[OrderBook]:
        return self._books.get(symbol)

    def get_bids_asks(self, symbol: str, depth: int = 1000):
        """Удобный метод: возвращает (bids, asks) или ([], [])."""
        book = self._books.get(symbol)
        if book is None or not book.synced:
            return [], []
        return book.get_bids(depth), book.get_asks(depth)

    async def start(self, symbols: List[str]):
        self._running  = True
        self._resync_q = asyncio.Queue()  # создаём здесь — гарантированно в нужном loop

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

    def _drain_buffer(self, symbol: str, book: "OrderBook"):
        """
        Применяет буферизированные WS события после получения снапшота.
        Обрабатывает весь накопленный буфер за один вызов.
        """
        buf       = book._buf
        book._buf = []
        for event in buf:
            ok = book.apply_event(event)
            if not ok:
                # Разрыв — буфер уже очищен, ставим в очередь на resync
                if self._resync_q is not None:
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

                    # При (пере)подключении сбрасываем состояние всех книг чанка.
                    # Без этого старые события от предыдущего соединения могут
                    # смешаться с новыми → бесконечный resync loop.
                    for sym in symbols:
                        book = self._books.get(sym)
                        if book is not None:
                            book._buf           = []
                            book.synced         = False
                            book.last_update_id = 0

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

                            if not book.synced:
                                # Всегда буферизируем пока не получим снапшот
                                # (last_update_id == 0) или пока не sync-нулись.
                                # _drain_buffer вызывается из _snapshot_loader
                                # после apply_snapshot — не надо дренировать здесь.
                                book._buf.append(event)
                                # Ограничиваем размер буфера: если накопилось
                                # слишком много событий до снапшота — обрезаем
                                # старые (они всё равно устарели).
                                if len(book._buf) > 2000:
                                    book._buf = book._buf[-1000:]
                                continue

                            # Нормальная работа — применяем напрямую
                            ok = book.apply_event(event)
                            if not ok:
                                logger.debug(f"OrderBook desync: {ccxt_sym}, resync...")
                                book.synced         = False
                                book.last_update_id = 0
                                book._buf           = []
                                if self._resync_q is not None:
                                    self._resync_q.put_nowait(ccxt_sym)

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
        Пауза 1 сек перед запросом нужна чтобы WS успел накопить свежие
        события в буфер — они будут применены через _drain_buffer после
        apply_snapshot.
        """
        loop = asyncio.get_event_loop()
        # Дедупликация: не берём один символ дважды подряд
        _in_resync: set = set()

        while self._running:
            try:
                sym = await asyncio.wait_for(self._resync_q.get(), timeout=5.0)

                if sym in _in_resync:
                    continue
                _in_resync.add(sym)

                book = self._books.get(sym)
                if book:
                    # Сброс перед паузой: WS начнёт складывать свежие события
                    # в book._buf пока мы ждём
                    book.synced         = False
                    book.last_update_id = 0
                    book._buf           = []

                await asyncio.sleep(1.0)

                bsym = _ccxt_to_binance(sym)
                data = await loop.run_in_executor(None, self._fetch_snapshot, bsym)
                if data:
                    book = self._books.get(sym)
                    if book:
                        book.apply_snapshot(data)
                        self._drain_buffer(sym, book)
                        logger.debug(f"resync OK: {sym} id={book.last_update_id}")
                _in_resync.discard(sym)  # снимаем блокировку только после завершения
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
