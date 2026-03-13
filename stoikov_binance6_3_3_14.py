import asyncio
import time
import math
import json
import uuid
import hmac
import hashlib
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlencode
import aiohttp

import numpy as np
import ccxt.pro as ccxtpro

# ---------------------------
# CONFIGURATION
# ---------------------------
EXCHANGE_ID = "aster"
SYMBOL = "ETHUSDT"
API_KEY = ""
API_SECRET = ""

FALLBACK_TICK_SIZE = 0.1
FALLBACK_LOT_SIZE = 0.001
MIN_NOTIONAL_FALLBACK = 3
MIN_ORDER_NOTIONAL_OVERRIDE = 3
ADAPT_TO_HIGHER_RUNTIME = True

QUOTE_INTERVAL_SEC = 0.1
POSITION_POLL_INTERVAL_SEC = 2.0
OPEN_ORDERS_POLL_INTERVAL_SEC = 2.0

MAX_POSITION_BASE = 0.003 * 50
ORDER_QTY_BASE = 0.003
GRID_NUM = 10
HALF_SPREAD_COEFF = 1
SKEW_COEFF = 0.5
MAX_NORM_POS = 50.0

# GLFT parameters
GLFT_GAMMA = 0.005

# ---------------------------
# TIMING & WINDOWING CONFIG (0.1 SECOND SAMPLING)
# ---------------------------

# Arrival depth calculation (0.1 second sampling)
ARRIVAL_DEPTH_WINDOW_SIZE_SEC = 0.1  # 100ms window for max depth aggregation
ARRIVAL_DEPTH_BUFFER_SIZE = 3000  # 60,000 samples × 0.1s = 6000 seconds (10 minutes)
WARMUP_MIN_ARRIVAL_DEPTHS = 100  # 600 samples × 0.1s = 60 seconds minimum

# Volatility calculation (0.1 second sampling → 1 second volatility)
MID_PRICE_WINDOW_SIZE_SEC = 0.1  # 100ms window for mid-price aggregation
GLFT_VOL_WINDOW = 3000  # 3000 samples × 0.1s = 300 seconds (5 minutes)
VOLATILITY_SCALING_FACTOR = math.sqrt(10)  # Scale from 0.1-sec to 1-sec: √(1.0 / 0.1) = √10

# K parameter estimation (uses 0.1 second arrival depth samples)
K_ESTIMATION_MAX_WINDOW_SEC = 3000.0  # Cap observation window at 10 minutes
K_ESTIMATION_MIN_SAMPLES = 100  # 600 samples × 0.1s = 60 seconds minimum
K_ESTIMATION_UPDATE_INTERVAL = 50  # Update k every N computation cycles

# ---------------------------
# GLFT PARAMETERS
# ---------------------------

# Dynamic GLFT_DELTA parameters
GLFT_DELTA_MIN = 1
GLFT_DELTA_SLOPE = 0.0
GLFT_DELTA_MAX = 1

# WARMUP PARAMETERS
WARMUP_TIMEOUT_SEC = 30.0
MIN_VALID_K = 1e-6

# BACKGROUND GLFT COMPUTATION
ENABLE_BACKGROUND_GLFT = True
GLFT_COMPUTE_INTERVAL_SEC = 0.1
GLFT_METRICS_TIMEOUT_SEC = 2.0

# WEBSOCKET ORDER BOOK CACHING
ENABLE_WEBSOCKET_OB = True
ORDERBOOK_CACHE_MAX_AGE_MS = 2000  # Increased tolerance to 2 seconds
WEBSOCKET_RECONNECT_INITIAL_WAIT = 1.0
WEBSOCKET_RECONNECT_MAX_WAIT = 30.0
WEBSOCKET_RECONNECT_BACKOFF_MULTIPLIER = 2.0
WEBSOCKET_HEALTH_CHECK_INTERVAL_SEC = 5.0  # Check connection health every 5s

# BATCH ORDER SETTINGS
ENABLE_NATIVE_BATCH = True
BATCH_MAX_CONCURRENT = 5
BATCH_PLACE_TIMEOUT_SEC = 5.0
BATCH_CANCEL_TIMEOUT_SEC = 5.0
CANCEL_TO_PLACE_DELAY_SEC = 10.0  # Delay between cancel and place operations

# Arrival depth measurement (REQUIRED for k estimation)
ENABLE_ARRIVAL_DEPTH = True

# OBI parameters
ENABLE_OBI = False
OBI_LEVELS = 40
OBI_SMOOTH_ALPHA = 0.4

# TFI parameters
ENABLE_TRADE_FLOW = False
TFI_WINDOW_SEC = 15.0
TFI_SMOOTH_ALPHA = 0.4

# Adaptive scaling
OBI_FRACTION_OF_HALFSPREAD = 0.25
TFI_FRACTION_OF_HALFSPREAD = 0.25
OBI_BETA_MIN_TICKS = 0.5
OBI_BETA_MAX_TICKS = 50
TFI_BETA_MIN_TICKS = 0.5
TFI_BETA_MAX_TICKS = 50

# Smoothing
SKEW_SMOOTH_ALPHA = 0.3
NORM_POS_SMOOTH_ALPHA = 0.3
SKEW_FRACTION_OF_HALFSPREAD_CAP = 10

DRY_RUN = True
POST_ONLY = True
MAX_CONSEC_ERRORS = 15
VERBOSE_FAILURES = True
CANCEL_STRATEGY = 'stale'
USE_SANDBOX = False


async def native_batch_place_orders_aster(
        exchange,
        symbol: str,
        orders: List[Dict[str, Any]],
        post_only: bool = True,
        max_batch_size: int = 5,  # Aster limit appears to be 5 orders per batch
) -> Tuple[List[Dict], List[Dict]]:
    if not orders:
        return [], []

    if exchange.id != 'aster':
        raise ValueError("This function requires Aster DEX exchange")

    all_successful = []
    all_failed = []

    try:
        # Split orders into chunks
        for i in range(0, len(orders), max_batch_size):
            chunk = orders[i:i + max_batch_size]

            if VERBOSE_FAILURES:
                print(f"[BATCH] Processing batch {i // max_batch_size + 1}: {len(chunk)} orders")

            batch_orders = []
            now_ms = int(time.time() * 1000)

            for o in chunk:
                side_upper = o['side'].upper()
                price_precise = exchange.price_to_precision(symbol, o['price'])
                amount_precise = exchange.amount_to_precision(symbol, o['amount'])
                client_id = f"mm_{now_ms}_{uuid.uuid4().hex[:8]}"

                order_spec = {
                    "symbol": symbol,
                    "side": side_upper,
                    "type": "LIMIT",
                    "timeInForce": "GTX" if post_only else "GTC",
                    "quantity": str(amount_precise),
                    "price": str(price_precise),
                    "newClientOrderId": client_id,
                }
                batch_orders.append(order_spec)

            timestamp = int(time.time() * 1000)

            batch_orders_json = json.dumps(batch_orders, separators=(',', ':'))

            params = {
                'batchOrders': batch_orders_json,
                'timestamp': str(timestamp),
            }

            query_string = urlencode(params)
            signature = sign_request(exchange.secret, query_string)
            full_query_string = f"{query_string}&signature={signature}"

            base_url = exchange.urls.get('api', {}).get('fapi', 'https://fapi.asterdex.com')
            if isinstance(base_url, dict):
                base_url = 'https://fapi.asterdex.com'
            base_url = str(base_url).rstrip('/')

            full_url = f"{base_url}/fapi/v1/batchOrders"

            headers = {
                'X-MBX-APIKEY': exchange.apiKey,
                'Content-Type': 'application/x-www-form-urlencoded',
            }

            if VERBOSE_FAILURES:
                print(f"[BATCH] Posting {len(chunk)} orders to {full_url}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        full_url,
                        data=full_query_string,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=BATCH_PLACE_TIMEOUT_SEC)
                ) as resp:
                    response_text = await resp.text()

                    if VERBOSE_FAILURES:
                        print(f"[BATCH] Status: {resp.status}")
                        if resp.status != 200:
                            print(f"[BATCH] Response: {response_text[:200]}")

                    if resp.status == 200:
                        response = json.loads(response_text)
                        successful = response if isinstance(response, list) else [response]
                        all_successful.extend(successful)
                        if VERBOSE_FAILURES:
                            print(f"[BATCH] ✓ Placed {len(successful)} orders from chunk")
                    else:
                        all_failed.extend(chunk)

        if VERBOSE_FAILURES:
            print(f"[BATCH] ✓ Total: Placed {len(all_successful)}/{len(orders)} orders")

        return all_successful, all_failed

    except Exception as e:
        if VERBOSE_FAILURES:
            print(f"[BATCH] Native batch placement failed: {e}")
            import traceback
            traceback.print_exc()
        return all_successful, all_failed + orders


async def native_batch_cancel_orders_aster(
        exchange,
        symbol: str,
        order_ids: List[str],
        max_batch_size: int = 5,
) -> Tuple[List[Dict], List[str]]:
    if not order_ids:
        return [], []

    if exchange.id != 'aster':
        raise ValueError("This function requires Aster DEX exchange")

    all_successful = []
    all_failed = []

    try:
        # Split order IDs into chunks of 5
        for i in range(0, len(order_ids), max_batch_size):
            chunk = order_ids[i:i + max_batch_size]

            if VERBOSE_FAILURES:
                print(f"[BATCH] Canceling batch {i // max_batch_size + 1}: {len(chunk)} orders")

            timestamp = int(time.time() * 1000)

            # Build parameters - orderIdList as compact JSON array
            order_id_list_json = json.dumps(chunk, separators=(',', ':'))

            # Build params dict
            params = {
                'orderIdList': order_id_list_json,
                'symbol': symbol,
                'timestamp': str(timestamp),
            }

            # Create query string (URL-encoded) - this is what we sign
            query_string = urlencode(sorted(params.items()))

            if VERBOSE_FAILURES:
                print(f"[BATCH] Query string to sign: {query_string[:150]}...")

            # Sign the query string
            signature = sign_request(exchange.secret, query_string)

            if VERBOSE_FAILURES:
                print(f"[BATCH] Signature: {signature}")

            # Append signature
            full_query_string = f"{query_string}&signature={signature}"

            # Get base URL
            base_url = exchange.urls.get('api', {}).get('fapi', 'https://fapi.asterdex.com')
            if isinstance(base_url, dict):
                base_url = 'https://fapi.asterdex.com'
            base_url = str(base_url).rstrip('/')

            full_url = f"{base_url}/fapi/v1/batchOrders"

            headers = {
                'X-MBX-APIKEY': exchange.apiKey,
                'Content-Type': 'application/x-www-form-urlencoded',
            }

            if VERBOSE_FAILURES:
                print(f"[BATCH] Deleting {len(chunk)} orders")
                print(f"[BATCH] Sending as DELETE with body (not query params)")

            async with aiohttp.ClientSession() as session:
                # Send DELETE with body, just like POST
                async with session.delete(
                        full_url,
                        data=full_query_string,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=BATCH_CANCEL_TIMEOUT_SEC)
                ) as resp:
                    response_text = await resp.text()

                    if VERBOSE_FAILURES:
                        print(f"[BATCH] Status: {resp.status}")
                        if resp.status != 200:
                            print(f"[BATCH] Response: {response_text[:500]}")

                    if resp.status == 200:
                        response = json.loads(response_text)
                        successful = response if isinstance(response, list) else [response]
                        all_successful.extend(successful)
                        if VERBOSE_FAILURES:
                            print(f"[BATCH] ✓ Canceled {len(successful)} orders")
                    else:
                        all_failed.extend(chunk)

        if VERBOSE_FAILURES:
            print(f"[BATCH] ✓ Total: Canceled {len(all_successful)}/{len(order_ids)} orders")

        return all_successful, all_failed

    except Exception as e:
        if VERBOSE_FAILURES:
            print(f"[BATCH] Native batch cancel failed: {e}")
            import traceback
            traceback.print_exc()
        return all_successful, all_failed + order_ids


async def ccxt_batch_place_orders_sequential(
        exchange,
        symbol: str,
        orders: List[Dict[str, Any]],
        post_only: bool = True,
        max_concurrent: int = 10,
) -> Tuple[List[Dict], List[Dict]]:
    if not orders:
        return [], []

    sem = asyncio.Semaphore(max_concurrent)

    async def place_single_order(order_spec):
        async with sem:
            try:
                side = order_spec['side'].lower()
                amount = exchange.amount_to_precision(symbol, order_spec['amount'])
                price = exchange.price_to_precision(symbol, order_spec['price'])

                params = {}
                if post_only:
                    params['timeInForce'] = 'GTX'

                now_ms = int(time.time() * 1000)
                params['newClientOrderId'] = f"mm_{now_ms}_{uuid.uuid4().hex[:8]}"

                result = await exchange.create_order(
                    symbol,
                    'limit',
                    side,
                    amount,
                    price,
                    params,
                )
                return ('success', result)
            except Exception as e:
                return ('error', {'spec': order_spec, 'error': str(e)})

    results = await asyncio.gather(
        *[place_single_order(o) for o in orders],
        return_exceptions=False,
    )

    successful = [r[1] for r in results if r[0] == 'success']
    failed = [r[1] for r in results if r[0] == 'error']

    if VERBOSE_FAILURES:
        print(f"[BATCH] Sequential placement: placed {len(successful)}/{len(orders)} orders")

    return successful, failed


# ---------------------------
# UTILS
# ---------------------------
def quantize_bid(price: float, tick: float) -> float:
    return math.floor(price / tick) * tick


def quantize_ask(price: float, tick: float) -> float:
    return math.ceil(price / tick) * tick


def quantize_amount(amount, step):
    steps = math.floor(float(amount) / step)
    return steps * step if steps > 0 else 0.0


def passive_buy_price(price, best_ask, tick_size):
    if price >= best_ask:
        return quantize_bid(best_ask - tick_size, tick_size)
    return quantize_bid(price, tick_size)


def passive_sell_price(price, best_bid, tick_size):
    if price <= best_bid:
        return quantize_ask(best_bid + tick_size, tick_size)
    return quantize_ask(price, tick_size)


def price_to_tick_index(price: float, tick: float) -> int:
    return int(round(float(price) / tick))


def ewma(prev, new, alpha):
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new


def linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b


def compute_coeff(gamma, delta, A, k):
    """
    Backtest-aligned formula with xi = gamma

    More aggressive than GLFT:
    - Tighter spreads (2.5× narrower)
    - Weaker position adjustment (2.2× smaller)
    """
    xi = gamma  # Key: xi = gamma in your backtest

    inv_k = 1 / k
    c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
    c2 = np.sqrt((gamma / (2 * A * delta * k)) *
                 ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1)))

    return c1, c2


# def compute_coeff(gamma, delta, A, k):
#     if np.isnan(A) or np.isnan(k) or k == 0:
#         return 0.5, 0.5
#     c1 = 1 / k * np.log(1 + k / gamma)
#     c2 = 1 / (gamma + k)
#     return c1, c2


def estimate_intensity(arrival_depths, window_sec=600.0):
    good = arrival_depths[~np.isnan(arrival_depths)]
    if len(good) == 0:
        return np.full(70, 1e-6, dtype=np.float64)

    counts, _ = np.histogram(good, bins=70, range=(0, 40))
    return (counts + 1e-6) / float(window_sec)


def extract_min_notional_from_filters(filters):
    candidates = []
    for f in filters:
        ftype = f.get('filterType', '')
        if ftype in ('MIN_NOTIONAL', 'NOTIONAL'):
            for key in ('minNotional', 'notional', 'notionalFloor'):
                val = f.get(key)
                if val is not None:
                    try:
                        fv = float(val)
                        if fv > 0:
                            candidates.append(fv)
                    except:
                        pass
    return max(candidates) if candidates else None


def sign_request(secret: str, query_string: str) -> str:
    if VERBOSE_FAILURES:
        print(f"[BATCH] Signing query string: {query_string[:150]}...")

    signature = hmac.new(
        secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    if VERBOSE_FAILURES:
        print(f"[BATCH] Generated signature: {signature}")

    return signature


async def ccxt_batch_cancel_orders_sequential(
        exchange,
        symbol: str,
        order_ids: List[str],
        max_concurrent: int = 10,
) -> Tuple[List[Dict], List[str]]:
    """
    Fallback: Cancel multiple orders using CCXT's standard cancelOrder with controlled concurrency.
    """
    if not order_ids:
        return [], []

    sem = asyncio.Semaphore(max_concurrent)

    async def cancel_single_order(oid):
        async with sem:
            try:
                result = await exchange.cancel_order(oid, symbol)
                return ('success', result)
            except Exception as e:
                if "Unknown order" not in str(e) and "order does not exist" not in str(e).lower():
                    if VERBOSE_FAILURES:
                        print(f"[BATCH] Cancel {oid} failed: {e}")
                return ('error', oid)

    results = await asyncio.gather(
        *[cancel_single_order(oid) for oid in order_ids],
        return_exceptions=False,
    )

    successful = [r[1] for r in results if r[0] == 'success']
    failed = [r[1] for r in results if r[0] == 'error']

    if VERBOSE_FAILURES:
        print(f"[BATCH] Sequential cancellation: canceled {len(successful)}/{len(order_ids)} orders")

    return successful, failed


async def smart_batch_place_orders(
        exchange,
        symbol: str,
        orders: List[Dict[str, Any]],
        post_only: bool = True,
        use_native: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    if not orders:
        return [], []

    successful, failed = [], []

    # Try native first (faster)
    if use_native and exchange.id == 'aster':
        try:
            successful, failed = await asyncio.wait_for(
                native_batch_place_orders_aster(exchange, symbol, orders, post_only),
                timeout=BATCH_PLACE_TIMEOUT_SEC,
            )
            if successful:
                return successful, failed
        except asyncio.TimeoutError:
            if VERBOSE_FAILURES:
                print(f"[BATCH] Native endpoint timed out, falling back to concurrent CCXT")
        except Exception as e:
            if VERBOSE_FAILURES:
                print(f"[BATCH] Native endpoint unavailable, falling back to concurrent CCXT")

    # Fallback to concurrent sequential
    if VERBOSE_FAILURES:
        print(f"[BATCH] Using fallback concurrent CCXT placement for {len(orders)} orders")

    try:
        successful, failed = await asyncio.wait_for(
            ccxt_batch_place_orders_sequential(
                exchange, symbol, orders, post_only, max_concurrent=BATCH_MAX_CONCURRENT
            ),
            timeout=BATCH_PLACE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print(f"[ERROR] Batch placement timed out after {BATCH_PLACE_TIMEOUT_SEC}s")

    return successful, failed


async def smart_batch_cancel_orders(
        exchange,
        symbol: str,
        order_ids: List[str],
        use_native: bool = True,
) -> Tuple[List[Dict], List[str]]:
    if not order_ids:
        return [], []

    successful, failed = [], []

    # Try native first (faster)
    if use_native and exchange.id == 'aster':
        try:
            successful, failed = await asyncio.wait_for(
                native_batch_cancel_orders_aster(exchange, symbol, order_ids),
                timeout=BATCH_CANCEL_TIMEOUT_SEC,
            )
            if successful:
                return successful, failed
        except asyncio.TimeoutError:
            if VERBOSE_FAILURES:
                print(f"[BATCH] Native cancel timed out, falling back to concurrent CCXT")
        except Exception as e:
            if VERBOSE_FAILURES:
                print(f"[BATCH] Native cancel unavailable, falling back to concurrent CCXT")

    # Fallback to concurrent sequential
    if VERBOSE_FAILURES:
        print(f"[BATCH] Using fallback concurrent CCXT cancellation for {len(order_ids)} orders")

    try:
        successful, failed = await asyncio.wait_for(
            ccxt_batch_cancel_orders_sequential(
                exchange, symbol, order_ids, max_concurrent=BATCH_MAX_CONCURRENT
            ),
            timeout=BATCH_CANCEL_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print(f"[ERROR] Batch cancellation timed out after {BATCH_CANCEL_TIMEOUT_SEC}s")
        failed = order_ids

    return successful, failed


# ---------------------------
# EXCHANGE HELPERS
# ---------------------------
async def fetch_market_filters(exchange, symbol):
    """Fetch market info and filters from Aster DEX."""
    market = exchange.market(symbol)
    tick_size = market.get('precision', {}).get('price', FALLBACK_TICK_SIZE)
    lot_step = market.get('precision', {}).get('amount', FALLBACK_LOT_SIZE)
    min_qty = lot_step
    max_qty = None
    min_notional = MIN_NOTIONAL_FALLBACK

    info = market.get("info") or {}
    filters = info.get("filters", [])

    for f in filters:
        ftype = f.get('filterType')
        if ftype == 'PRICE_FILTER':
            step = f.get('tickSize')
            if step and float(step) > 0:
                tick_size = float(step)
        elif ftype == 'LOT_SIZE':
            step = f.get('stepSize')
            if step and float(step) > 0:
                lot_step = float(step)
            if f.get('minQty'):
                min_qty = float(f['minQty'])
            if f.get('maxQty'):
                max_qty = float(f['maxQty'])

    extracted = extract_min_notional_from_filters(filters)
    if ADAPT_TO_HIGHER_RUNTIME and extracted and extracted > min_notional:
        min_notional = extracted

    if MIN_ORDER_NOTIONAL_OVERRIDE and MIN_ORDER_NOTIONAL_OVERRIDE > min_notional:
        min_notional = MIN_ORDER_NOTIONAL_OVERRIDE

    return dict(
        tick_size=tick_size,
        lot_step=lot_step,
        min_qty=min_qty,
        max_qty=max_qty,
        min_notional=min_notional
    )


async def fetch_position(exchange, symbol):
    """Fetch position from Aster DEX."""
    pos_base = pos_notional = 0.0
    try:
        positions = await exchange.fetch_positions([symbol])
        for p in positions:
            if symbol in (p.get('symbol'), p.get('info', {}).get('symbol')):
                c = float(p.get('contracts') or 0)
                side = (p.get('side') or '').lower()
                pos_base = -c if side == 'short' else c
                pos_notional = float(p.get('notional') or 0)
                break
    except Exception as e:
        if VERBOSE_FAILURES:
            print(f"[WARN] fetch_position error: {e}")
    return pos_base, pos_notional


async def fetch_open_orders(exchange, symbol):
    """Fetch open orders from Aster DEX."""
    try:
        return await exchange.fetch_open_orders(symbol)
    except Exception as e:
        print(f"[WARN] fetch_open_orders error: {e}")
        return []


# ---------------------------
# GLFTGridMarketMaker
# ---------------------------
class GLFTGridMarketMaker:
    def __init__(self, exchange, exchange_trades, symbol, tick_size, lot_step, min_qty, max_qty, min_notional):
        self.k = 0.05
        self.A = 1e-6
        self._k_ready = False
        self.exchange = exchange
        self.exchange_trades = exchange_trades
        self.symbol = symbol
        self.tick_size = tick_size
        self.lot_step = lot_step
        self.min_qty = min_qty
        self.max_qty = max_qty
        self.min_notional = min_notional

        # ========== GLFT CALCULATION STATE (0.1-SECOND SAMPLING) ==========
        self.mid_history = deque(maxlen=GLFT_VOL_WINDOW)
        self.arrival_depth = deque(maxlen=ARRIVAL_DEPTH_BUFFER_SIZE)
        self.last_mid = None

        # 0.1-second windowed arrival depths
        self.arrival_depth_windowed = deque(maxlen=ARRIVAL_DEPTH_BUFFER_SIZE)
        self.current_window_start = time.time()
        self.current_window_samples = []

        # 0.1-second windowed mid prices
        self.mid_history_windowed = deque(maxlen=GLFT_VOL_WINDOW)
        self.current_mid_window_start = time.time()
        self.current_mid_window_prices = []

        self.volatility = np.nan
        self.volatility_ticks = np.nan

        self._norm_pos_smooth = 0.0
        self._skew_ticks_smooth = 0.0

        # ========== WARMUP STATE ==========
        self._warmup_started = time.time()
        self._k_ready = False
        self._warmup_logged_count = 0

        # ========== CACHE LOCKS & STATE ==========
        self._obi_smooth = 0.0
        self._obi_lock = asyncio.Lock()

        self._pos_base = 0.0
        self._pos_notional = 0.0
        self._position_lock = asyncio.Lock()

        self._open_orders = []
        self._orders_lock = asyncio.Lock()

        self._trades = deque(maxlen=5000)
        self._tfi_smooth = 0.0
        self._trades_lock = asyncio.Lock()

        # Order Book Cache
        self._ob_cache = {
            'best_bid': None,
            'best_ask': None,
            'bids': [],
            'asks': [],
            'timestamp_ms': 0,
            'updated': False,
        }
        self._ob_cache_lock = asyncio.Lock()
        self._ws_connected = False
        self._last_ws_update = 0

        # GLFT Metrics Cache
        self._glft_cache = {
            'A': np.nan,
            'k': np.nan,
            'volatility': np.nan,
            'volatility_ticks': np.nan,
            'half_spread_tick': np.nan,
            'skew_ticks': np.nan,
            'timestamp_ms': 0,
            'k_ready': False,
        }
        self._glft_cache_lock = asyncio.Lock()

        # ========== BACKGROUND TASKS ==========
        self._position_task = None
        self._orders_task = None
        self._trades_task = None
        self._orderbook_task = None
        self._orderbook_obi_task = None
        self._glft_task = None
        self._running = False

    # ========== WARM-UP CHECKING ==========
    def is_warmup_complete(self) -> bool:
        return self._k_ready

    async def check_k_readiness(self) -> bool:
        async with self._glft_cache_lock:
            k_ready = self._glft_cache.get('k_ready', False)
            if k_ready:
                self._k_ready = True
                return True
        return False

    async def log_warmup_status(self, t: int):
        elapsed = time.time() - self._warmup_started

        if self._k_ready:
            if self._warmup_logged_count == 0:
                print(f"[WARMUP] ✓ k parameter ready after {elapsed:.1f}s. Starting order placement.")
                self._warmup_logged_count += 1
            return

        if t % 5 == 0:
            windowed_count = len(self.arrival_depth_windowed)
            timeout_remaining = WARMUP_TIMEOUT_SEC - elapsed

            if timeout_remaining > 0:
                print(f"[WARMUP] Waiting for k parameter... "
                      f"windowed_samples={windowed_count}/{WARMUP_MIN_ARRIVAL_DEPTHS} "
                      f"(need {WARMUP_MIN_ARRIVAL_DEPTHS * ARRIVAL_DEPTH_WINDOW_SIZE_SEC:.0f}s of trade data) "
                      f"timeout_in={timeout_remaining:.1f}s")
            else:
                print(f"[WARMUP] Timeout reached. Using fallback spread (volatility-based).")
                self._k_ready = True

    # ========== BACKGROUND TASK MANAGEMENT ==========
    async def start_background_tasks(self):

        # Initial position and orders fetch
        try:
            pos_base, pos_notional = await fetch_position(self.exchange, self.symbol)
            async with self._position_lock:
                self._pos_base = pos_base
                self._pos_notional = pos_notional
        except Exception as e:
            print(f"[WARN] Initial position fetch failed: {e}")

        try:
            orders = await fetch_open_orders(self.exchange, self.symbol)
            async with self._orders_lock:
                self._open_orders = orders
        except Exception as e:
            print(f"[WARN] Initial open orders fetch failed: {e}")

        # Start REST API polling tasks (no conflicts)
        self._position_task = asyncio.create_task(
            self.update_position_task(POSITION_POLL_INTERVAL_SEC)
        )
        self._orders_task = asyncio.create_task(
            self.update_orders_task(OPEN_ORDERS_POLL_INTERVAL_SEC)
        )

        # WebSocket 1: Order book (uses self.exchange)
        if ENABLE_WEBSOCKET_OB:
            print("[INIT] Starting order book WebSocket task...")
            self._orderbook_task = asyncio.create_task(self.update_orderbook_cache_task())
            await asyncio.sleep(3.0)
            print("[INIT] Order book WebSocket initialization complete")

        # WebSocket 2: Trades (uses self.exchange_trades - separate instance)
        if (ENABLE_ARRIVAL_DEPTH or ENABLE_TRADE_FLOW):
            print("[INIT] Starting trades WebSocket task...")
            self._trades_task = asyncio.create_task(self.update_trades_task())
            await asyncio.sleep(3.0)
            print("[INIT] Trades WebSocket initialization complete")

        # OBI (uses orderbook data)
        if ENABLE_OBI:
            self._orderbook_obi_task = asyncio.create_task(self.update_orderbook_obi_task())

        # GLFT computation
        if ENABLE_BACKGROUND_GLFT:
            print("[INIT] Starting GLFT computation task...")
            self._glft_task = asyncio.create_task(self.compute_glft_metrics_task())

        print("[INIT] All background tasks started")

    async def stop_background_tasks(self):
        tasks = [
            self._position_task,
            self._orders_task,
            self._trades_task,
            self._orderbook_task,
            self._orderbook_obi_task,
            self._glft_task,
        ]
        for task in tasks:
            if task:
                task.cancel()
        for task in tasks:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._position_task = None
        self._orders_task = None
        self._trades_task = None
        self._orderbook_task = None
        self._orderbook_obi_task = None
        self._glft_task = None

    async def update_position_task(self, poll_interval=2.0):
        while True:
            try:
                pos_base, pos_notional = await fetch_position(self.exchange, self.symbol)
                async with self._position_lock:
                    self._pos_base = pos_base
                    self._pos_notional = pos_notional
            except Exception as e:
                print(f"[WARN] Position poll error: {e}")
            await asyncio.sleep(poll_interval)

    async def update_orders_task(self, poll_interval=2.0):
        while True:
            try:
                orders = await fetch_open_orders(self.exchange, self.symbol)
                async with self._orders_lock:
                    self._open_orders = orders
            except Exception as e:
                print(f"[WARN] Orders poll error: {e}")
            await asyncio.sleep(poll_interval)

    async def update_trades_task(self):
        """
        WebSocket task for trades with 0.1-second windowing.

        SAMPLING STRATEGY:
        - Aggregates trades into 0.1-second windows (100ms)
        - Takes maximum arrival depth per window
        - Maintains 60,000 samples (6000 seconds = 100 minutes of history)
        """

        if not self.exchange_trades.has.get('watchTrades'):
            print(f"[TRADES] ✗ watchTrades not supported")
            return

        print(f"[TRADES] Starting trades WebSocket for {self.symbol} (0.1s windowing)...")

        # ✅ Initialize to current 0.1-second boundary (decisecond)
        self.current_window_start = float(int(time.time() * 10) / 10.0)

        total_trades_seen = 0
        message_count = 0

        try:
            while True:
                trades = await self.exchange_trades.watch_trades(self.symbol)

                message_count += 1
                if message_count == 1:
                    print(f"[TRADES] ✓✓✓ Trades WebSocket CONNECTED (0.1s sampling)")

                now_ms = int(time.time() * 1000)
                current_time = time.time()
                current_decisecond = float(int(current_time * 10) / 10.0)  # ✅ Floor to 0.1 second

                # ✅ Clock-aligned windowing: Finalize at 0.1-second boundaries
                if current_decisecond > self.current_window_start:
                    async with self._trades_lock:
                        if self.current_window_samples:
                            window_depth = float(np.max(self.current_window_samples))
                            self.arrival_depth_windowed.append(window_depth)

                            # Enhanced logging with window timestamps
                            if len(self.arrival_depth_windowed) % 50 == 0 or len(self.arrival_depth_windowed) <= 10:
                                print(
                                    f"[TRADES] Window [{self.current_window_start:.1f}s → {current_decisecond:.1f}s]: "
                                    f"{len(self.current_window_samples)} trades → "
                                    f"max_depth={window_depth:.3f} ticks | "
                                    f"buffer={len(self.arrival_depth_windowed)}/{ARRIVAL_DEPTH_BUFFER_SIZE}"
                                )
                        else:
                            # No trades in this window - still finalize with empty window
                            if VERBOSE_FAILURES and len(self.arrival_depth_windowed) <= 10:
                                print(
                                    f"[TRADES] Window [{self.current_window_start:.1f}s → {current_decisecond:.1f}s]: "
                                    f"0 trades (no depth)"
                                )

                        self.current_window_start = current_decisecond  # ✅ Move to next 0.1-second boundary
                        self.current_window_samples = []

                # Get mid-price for depth calculation
                best_bid, best_ask, _, _, _ = await self.get_cached_orderbook()

                if best_bid is None or best_ask is None:
                    try:
                        ob = await self.exchange.fetch_order_book(self.symbol)
                        best_bid = ob['bids'][0][0]
                        best_ask = ob['asks'][0][0]
                    except Exception:
                        continue

                mid = (best_bid + best_ask) / 2.0
                mid_tick = mid / self.tick_size

                # Process trades
                async with self._trades_lock:
                    for trade in trades:
                        total_trades_seen += 1

                        side = (trade.get('side') or '').lower()
                        price = float(trade.get('price') or 0.0)
                        amount = float(trade.get('amount') or 0.0)
                        timestamp = trade.get('timestamp') or now_ms

                        # Arrival depth calculation
                        if self.tick_size > 0 and price > 0:
                            trade_price_tick = price / self.tick_size

                            if side == 'buy':
                                depth_ticks = trade_price_tick - mid_tick
                            else:
                                depth_ticks = mid_tick - trade_price_tick

                            if depth_ticks > 0:
                                self.current_window_samples.append(depth_ticks)
                                self.arrival_depth.append(depth_ticks)

                        # TFI calculation (if enabled)
                        if ENABLE_TRADE_FLOW:
                            qvol = amount * price
                            if qvol > 0:
                                signed = qvol if side == 'buy' else -qvol
                                self._trades.append((timestamp, signed))

                        # Debug: Log first few trades
                        if total_trades_seen <= 5:
                            print(
                                f"[TRADES] {self.exchange_trades.id} {self.symbol} | "
                                f"{trade.get('datetime', 'N/A')} | "
                                f"{side.upper()} {amount:.4f} @ {price:.2f}"
                            )

                    # TFI cleanup
                    if ENABLE_TRADE_FLOW:
                        cutoff = now_ms - int(TFI_WINDOW_SEC * 1000)
                        while self._trades and self._trades[0][0] < cutoff:
                            self._trades.popleft()

                        buys = sum(v for _, v in self._trades if v > 0)
                        sells = -sum(v for _, v in self._trades if v < 0)
                        denom = buys + sells
                        self._tfi_smooth = (buys - sells) / denom if denom > 1e-9 else 0.0

                # Periodic status
                if message_count % 100 == 0:
                    print(
                        f"[TRADES] ✓ {message_count} batches | {total_trades_seen} total trades | "
                        f"buffer={len(self.arrival_depth_windowed)}/{ARRIVAL_DEPTH_BUFFER_SIZE}"
                    )

        except asyncio.CancelledError:
            print(f"[TRADES] Task cancelled")
        except Exception as e:
            print(f"[TRADES] Error: {type(e).__name__}: {e}")
            if VERBOSE_FAILURES:
                import traceback
                traceback.print_exc()

    async def update_orderbook_obi_task(self):
        """WebSocket task for OBI calculation with automatic reconnection."""
        backoff_wait = WEBSOCKET_RECONNECT_INITIAL_WAIT
        consecutive_failures = 0
        connection_count = 0

        while True:
            try:
                consecutive_failures = 0
                connection_count += 1

                if connection_count > 1:
                    print(f"[OBI] Reconnecting to orderbook WebSocket (attempt #{connection_count})...")
                else:
                    print(f"[OBI] Connecting to orderbook WebSocket for {self.symbol}...")

                message_count = 0
                while True:
                    try:
                        ob = await asyncio.wait_for(
                            self.exchange.watch_order_book(self.symbol),
                            timeout=30.0
                        )
                        message_count += 1

                        obi_raw = self.compute_obi_from_ob(ob, OBI_LEVELS)
                        async with self._obi_lock:
                            self._obi_smooth = ewma(self._obi_smooth, obi_raw, OBI_SMOOTH_ALPHA)

                        if message_count == 1:
                            print(f"[OBI] ✓ Orderbook WebSocket connected")

                        backoff_wait = WEBSOCKET_RECONNECT_INITIAL_WAIT

                        # Periodic status
                        if message_count % 100 == 0:
                            print(f"[OBI] ✓ Received {message_count} orderbook updates, OBI={self._obi_smooth:.4f}")

                    except asyncio.TimeoutError:
                        print(f"[OBI] Timeout waiting for orderbook update (30s)")
                        raise

            except asyncio.CancelledError:
                print("[OBI] Orderbook WebSocket task cancelled.")
                break

            except Exception as e:
                consecutive_failures += 1
                error_str = str(e).lower()

                print(f"[OBI] ✗ Orderbook WebSocket error (attempt {consecutive_failures}): {type(e).__name__}: {e}")

                is_critical = any(x in error_str for x in [
                    'not found',
                    'invalid',
                    'unauthorized',
                    'not supported',
                    'method',
                    'authentication'
                ])

                if is_critical:
                    print(f"[OBI] CRITICAL ERROR - disabling OBI calculation")
                    async with self._obi_lock:
                        self._obi_smooth = 0.0
                    break

                if consecutive_failures >= 5:
                    backoff_wait = WEBSOCKET_RECONNECT_MAX_WAIT

                print(f"[OBI] Reconnecting in {backoff_wait:.1f}s...")
                await asyncio.sleep(backoff_wait)

                backoff_wait = min(
                    backoff_wait * WEBSOCKET_RECONNECT_BACKOFF_MULTIPLIER,
                    WEBSOCKET_RECONNECT_MAX_WAIT
                )

    async def update_orderbook_cache_task(self):
        """WebSocket task for caching best bid/ask with automatic reconnection."""

        print(f"[OB_CACHE] Task started for {self.symbol}")

        backoff_wait = WEBSOCKET_RECONNECT_INITIAL_WAIT
        consecutive_failures = 0
        last_health_check = time.time()
        connection_count = 0

        while True:
            try:
                consecutive_failures = 0
                self._ws_connected = False
                connection_count += 1

                if connection_count > 1:
                    print(f"[OB_CACHE] Reconnecting (attempt #{connection_count})...")
                else:
                    print(f"[OB_CACHE] Connecting to WebSocket for {self.symbol}...")

                message_count = 0
                first_message_received = False

                while True:
                    try:
                        if not first_message_received:
                            print(f"[OB_CACHE] Waiting for first orderbook message (30s timeout)...")

                        ob = await asyncio.wait_for(
                            self.exchange.watch_order_book(self.symbol),
                            timeout=30.0
                        )

                        if not first_message_received:
                            print(f"[OB_CACHE] ✓ First message received!")
                            first_message_received = True

                        message_count += 1

                        if not ob or not ob.get('bids') or not ob.get('asks'):
                            print(f"[OB_CACHE] WARNING: Empty orderbook received")
                            await asyncio.sleep(0.1)
                            continue

                        best_bid = float(ob['bids'][0][0])
                        best_ask = float(ob['asks'][0][0])
                        now_ms = int(time.time() * 1000)

                        async with self._ob_cache_lock:
                            self._ob_cache.update({
                                'best_bid': best_bid,
                                'best_ask': best_ask,
                                'bids': ob['bids'][:20],
                                'asks': ob['asks'][:20],
                                'timestamp_ms': now_ms,
                                'updated': True,
                            })

                        if not self._ws_connected:
                            self._ws_connected = True
                            print(f"[OB_CACHE] ✓✓✓ WebSocket CONNECTED: bid={best_bid:.2f} ask={best_ask:.2f}")

                        self._last_ws_update = now_ms
                        backoff_wait = WEBSOCKET_RECONNECT_INITIAL_WAIT

                        # Periodic health check
                        current_time = time.time()
                        if current_time - last_health_check > WEBSOCKET_HEALTH_CHECK_INTERVAL_SEC:
                            age_sec = (now_ms - self._last_ws_update) / 1000.0
                            print(
                                f"[OB_CACHE] ✓ Health: bid={best_bid:.2f} ask={best_ask:.2f} "
                                f"age={age_sec:.2f}s msgs={message_count}"
                            )
                            last_health_check = current_time

                    except asyncio.TimeoutError:
                        print(f"[OB_CACHE] ✗ TIMEOUT - no orderbook update in 30 seconds")
                        print(f"[OB_CACHE] This might be a connection reuse issue")
                        raise

            except asyncio.CancelledError:
                print("[OB_CACHE] Task cancelled")
                break

            except Exception as e:
                consecutive_failures += 1
                self._ws_connected = False

                print(f"[OB_CACHE] ✗ Error (attempt {consecutive_failures}): {type(e).__name__}: {e}")

                if VERBOSE_FAILURES:
                    import traceback
                    traceback.print_exc()

                if consecutive_failures >= 5:
                    print(f"[OB_CACHE] Too many failures - using max backoff")
                    backoff_wait = WEBSOCKET_RECONNECT_MAX_WAIT

                print(f"[OB_CACHE] Reconnecting in {backoff_wait:.1f}s...")
                await asyncio.sleep(backoff_wait)

                backoff_wait = min(
                    backoff_wait * WEBSOCKET_RECONNECT_BACKOFF_MULTIPLIER,
                    WEBSOCKET_RECONNECT_MAX_WAIT
                )

    async def compute_glft_metrics_task(self):
        """
        Compute GLFT metrics with 0.1-second sampling.

        SAMPLING STRATEGY:
        - Mid-prices: 0.1-second windows (mean per window)
        - Volatility: std dev of 0.1-second changes, scaled to 1-second
        - K parameter: estimated from 0.1-second arrival depth samples
        - Maintains 300 seconds (5 minutes) of mid-price history
        """

        # ✅ Initialize to current 0.1-second boundary (clock-aligned)
        self.current_mid_window_start = float(int(time.time() * 10) / 10.0)

        computation_count = 0
        last_k_estimation = 0

        print(f"[GLFT_CALC] Task started - 0.1-second clock-aligned windowing enabled")

        while True:
            try:
                await asyncio.sleep(GLFT_COMPUTE_INTERVAL_SEC)

                computation_count += 1

                # ========== GET CURRENT POSITION ==========
                async with self._position_lock:
                    pos_base = self._pos_base

                # ========== GET CURRENT ORDERBOOK ==========
                best_bid, best_ask = None, None

                # Try cached orderbook first
                async with self._ob_cache_lock:
                    if self._ob_cache.get('updated'):
                        now_ms = int(time.time() * 1000)
                        age_ms = now_ms - self._ob_cache['timestamp_ms']

                        if age_ms < ORDERBOOK_CACHE_MAX_AGE_MS:
                            best_bid = self._ob_cache['best_bid']
                            best_ask = self._ob_cache['best_ask']

                # Fallback to REST if cache stale
                if best_bid is None or best_ask is None:
                    try:
                        ob = await self.exchange.fetch_order_book(self.symbol)
                        best_bid = float(ob['bids'][0][0])
                        best_ask = float(ob['asks'][0][0])
                    except Exception as e:
                        if VERBOSE_FAILURES:
                            print(f"[GLFT_CALC] Could not fetch orderbook: {e}")
                        continue

                mid = (best_bid + best_ask) / 2.0
                current_time = time.time()
                current_decisecond = float(int(current_time * 10) / 10.0)  # ✅ Floor to 0.1 second

                # ========== MID-PRICE WINDOWING (0.1-SECOND CLOCK-ALIGNED) ==========
                # Finalize window at exact 0.1-second boundaries
                if current_decisecond > self.current_mid_window_start:
                    if self.current_mid_window_prices:
                        # Compute mean mid-price for the completed 0.1-second window
                        window_mid = float(np.mean(self.current_mid_window_prices))

                        # CRITICAL: Convert to ticks before storing (backtest-consistent)
                        if self.tick_size > 0:
                            window_mid_tick = window_mid / self.tick_size
                        else:
                            window_mid_tick = np.nan

                        # Store windowed mid-price in ticks
                        self.mid_history_windowed.append(window_mid_tick)

                        # Debug logging
                        if VERBOSE_FAILURES and (
                                len(self.mid_history_windowed) % 50 == 0 or len(self.mid_history_windowed) <= 10):
                            print(
                                f"[GLFT_CALC] Mid window [{self.current_mid_window_start:.1f}s → {current_decisecond:.1f}s]: "
                                f"{len(self.current_mid_window_prices)} samples → mean_tick={window_mid_tick:.4f}"
                            )
                    else:
                        # Empty window (no samples collected)
                        if VERBOSE_FAILURES and len(self.mid_history_windowed) <= 10:
                            print(
                                f"[GLFT_CALC] Mid window [{self.current_mid_window_start:.1f}s → {current_decisecond:.1f}s]: "
                                f"0 samples (no data)"
                            )

                    # Reset window to current 0.1-second boundary
                    self.current_mid_window_start = current_decisecond
                    self.current_mid_window_prices = []

                # Collect current mid-price sample
                self.current_mid_window_prices.append(mid)

                # ========== K PARAMETER ESTIMATION (0.1-SECOND SAMPLING → 1-SECOND INTENSITY) ==========
                k_is_valid = False

                if (computation_count - last_k_estimation >= K_ESTIMATION_UPDATE_INTERVAL) and \
                        len(self.arrival_depth_windowed) >= K_ESTIMATION_MIN_SAMPLES:
                    try:
                        # Get windowed arrival depths (each sample = 0.1 second window)
                        depths = np.array(self.arrival_depth_windowed, dtype=np.float64)

                        # CRITICAL: Each sample represents 0.1 seconds
                        # Total observation window in seconds
                        window_sec = len(depths) * ARRIVAL_DEPTH_WINDOW_SIZE_SEC

                        # Cap at K_ESTIMATION_MAX_WINDOW_SEC for backtest consistency
                        if window_sec > K_ESTIMATION_MAX_WINDOW_SEC:
                            window_sec = K_ESTIMATION_MAX_WINDOW_SEC
                            # Use only the most recent samples that fit in the window
                            max_samples = int(K_ESTIMATION_MAX_WINDOW_SEC / ARRIVAL_DEPTH_WINDOW_SIZE_SEC)
                            depths = depths[-max_samples:]
                            window_sec = len(depths) * ARRIVAL_DEPTH_WINDOW_SIZE_SEC

                        if VERBOSE_FAILURES:
                            print(
                                f"[GLFT_CALC] Estimating k from {len(depths)} windowed samples (0.1s each) "
                                f"(observation window = {window_sec:.1f}s)"
                            )

                        # Filter valid depths
                        good = depths[~np.isnan(depths)]

                        if len(good) > 0:
                            # ✅ MATCH BACKTEST: 500 bins, range [0, 500) ticks
                            counts, _ = np.histogram(good, bins=250, range=(0, 250))
                            lambda_ = (counts + 1e-6) / float(window_sec)
                        else:
                            lambda_ = np.full(250, 1e-6, dtype=np.float64)

                        # Ticks from mid-price
                        ticks = np.arange(len(lambda_)) + 0.5  # [0.5, 1.5, ..., 499.5]

                        # ✅ MATCH BACKTEST: Refit to first 70 ticks
                        x_shallow = ticks[:15]  # [0.5, 1.5, ..., 69.5]
                        lambda_shallow = lambda_[:15]
                        y_shallow = np.log(lambda_shallow)

                        # Linear regression
                        k_, logA = linear_regression(x_shallow, y_shallow)
                        k_ = max(-k_, 1e-6)  # k is negative slope

                        # Validate k
                        if np.isfinite(k_) and not np.isnan(k_) and k_ > MIN_VALID_K:
                            self.A = np.exp(logA)
                            self.k = k_
                            k_is_valid = True

                            print(
                                f"[GLFT_CALC] ✓ k updated (BACKTEST-ALIGNED): k={self.k:.6f} A={self.A:.3g} | "
                                f"fitted 0-70 ticks from {len(depths)} samples (0.1s) over {window_sec:.1f}s"
                            )
                        else:
                            if VERBOSE_FAILURES:
                                print(
                                    f"[GLFT_CALC] k estimation failed: k={k_:.6f} (below MIN_VALID_K={MIN_VALID_K})"
                                )

                    except Exception as e:
                        if VERBOSE_FAILURES:
                            print(f"[GLFT_CALC] Error computing k: {e}")
                            import traceback
                            traceback.print_exc()

                # ========== VOLATILITY CALCULATION (0.1-SECOND → 1-SECOND) ==========
                if len(self.mid_history_windowed) > 3:
                    # mid_history_windowed contains mid-prices in TICKS (0.1-second sampling)
                    mid_arr = np.array(self.mid_history_windowed, dtype=np.float64)
                    diffs = np.diff(mid_arr)  # 0.1-second changes in ticks

                    # Standard deviation of 0.1-second tick changes
                    vol_0_1_sec = float(np.nanstd(diffs))

                    # Scale to 1-second volatility: σ(1s) = σ(0.1s) × √(1.0 / 0.1) = σ(0.1s) × √10
                    # Theoretical justification: Random walk scaling law σ(Δt₂) = σ(Δt₁) × √(Δt₂/Δt₁)
                    self.volatility_ticks = vol_0_1_sec * VOLATILITY_SCALING_FACTOR

                    if VERBOSE_FAILURES and computation_count % 10 == 0:
                        print(
                            f"[GLFT_CALC] Volatility: "
                            f"0.1s_std={vol_0_1_sec:.4f} ticks → "
                            f"1s_vol={self.volatility_ticks:.4f} ticks (×√10)"
                        )
                else:
                    # Not enough history - use safe default
                    self.volatility_ticks = 1.0

                # Validate volatility
                if not np.isfinite(self.volatility_ticks) or self.volatility_ticks <= 0:
                    self.volatility_ticks = 1.0

                # ========== SPREAD CALCULATION ==========
                half_spread_tick = 1.0  # Default fallback

                if k_is_valid or self._k_ready:
                    try:
                        # Get order quantity for normalized position calculation
                        dyn_qty = self.compute_order_quantity(mid_price=mid)

                        # Compute normalized position: q_norm = q / q_order
                        normalized_position = float(np.clip(
                            (pos_base / dyn_qty) if dyn_qty else 0.0,
                            -MAX_NORM_POS,
                            MAX_NORM_POS
                        ))

                        # Dynamic delta (inventory risk aversion parameter)
                        delta = GLFT_DELTA_MIN + GLFT_DELTA_SLOPE * abs(normalized_position)
                        delta = float(np.clip(delta, GLFT_DELTA_MIN, GLFT_DELTA_MAX))

                        # Compute GLFT coefficients
                        c1, c2 = compute_coeff(GLFT_GAMMA, GLFT_DELTA_MIN, self.A, self.k)

                        # BACKTEST-CONSISTENT SPREAD FORMULA (GLFT paper Eq. 13):
                        # δ^± = c1 + (δ/2) * c2 * σ
                        half_spread_tick = (c1 + (delta / 2.0) * c2 * self.volatility_ticks) * HALF_SPREAD_COEFF

                        # Validate spread
                        if not np.isfinite(half_spread_tick) or half_spread_tick <= 0:
                            half_spread_tick = 1.0

                        # Debug logging
                        if VERBOSE_FAILURES and computation_count % 5 == 0:
                            print(
                                f"[GLFT_CALC] Spread calc: c1={c1:.4f} c2={c2:.4f} "
                                f"delta={delta:.3f} vol_ticks={self.volatility_ticks:.3f} "
                                f"→ half_spread={half_spread_tick:.2f} ticks"
                            )

                    except Exception as e:
                        if VERBOSE_FAILURES:
                            print(f"[GLFT_CALC] Error computing spread: {e}")
                        half_spread_tick = max(1.0, self.volatility_ticks * 1.5)
                else:
                    # Fallback during warmup: use volatility-based spread
                    half_spread_tick = max(1.0, self.volatility_ticks * 1.5)

                # ========== UPDATE CACHE ==========
                async with self._glft_cache_lock:
                    self._glft_cache.update({
                        'A': self.A,
                        'k': self.k,
                        'volatility': self.volatility_ticks * self.tick_size,  # Convert to dollars for display
                        'volatility_ticks': self.volatility_ticks,
                        'half_spread_tick': half_spread_tick,
                        'mid': mid,
                        'timestamp_ms': int(time.time() * 1000),
                        'k_ready': k_is_valid or self._k_ready,
                    })

                # ========== K READINESS NOTIFICATION ==========
                if k_is_valid and not self._k_ready:
                    self._k_ready = True
                    elapsed = time.time() - self._warmup_started
                    print(
                        f"[GLFT_CALC] ✓✓✓ k parameter ready after {elapsed:.1f}s (0.1s TRADE-BASED arrival depths)"
                    )
                    print(
                        f"[GLFT_CALC] Parameters: k={self.k:.6f} A={self.A:.3g} "
                        f"vol_ticks={self.volatility_ticks:.3f} half_spread={half_spread_tick:.2f}"
                    )

                # ========== PERIODIC STATUS LOGGING ==========
                if computation_count % 5 == 0:
                    windowed_count = len(self.arrival_depth_windowed)
                    windowed_mid_count = len(self.mid_history_windowed)
                    ws_status = "✓ WS connected" if self._ws_connected else "✗ WS disconnected"

                    if not self._k_ready:
                        # Warmup status
                        print(
                            f"[GLFT_CALC] warming up (waiting for trades) | {ws_status} | "
                            f"trade_depths={windowed_count}/{WARMUP_MIN_ARRIVAL_DEPTHS} "
                            f"({windowed_count * ARRIVAL_DEPTH_WINDOW_SIZE_SEC:.0f}s of data) "
                            f"mid_hist_ticks={windowed_mid_count} "
                            f"A={self.A:.3g} k={self.k:.6f} vol_ticks={self.volatility_ticks:.3f} "
                            f"half_spread={half_spread_tick:.2f}"
                        )
                    else:
                        # Operational status
                        dyn_qty = self.compute_order_quantity(mid_price=mid)
                        normalized_position = float(np.clip(
                            (pos_base / dyn_qty) if dyn_qty else 0.0,
                            -MAX_NORM_POS,
                            MAX_NORM_POS
                        ))

                        print(
                            f"[GLFT_CALC] pos={pos_base:.6f} norm={normalized_position:.4f} | "
                            f"A={self.A:.3g} k={self.k:.6f} vol_ticks={self.volatility_ticks:.3f} | "
                            f"depths={windowed_count}/{ARRIVAL_DEPTH_BUFFER_SIZE} mid_hist={windowed_mid_count}/{GLFT_VOL_WINDOW}"
                        )

            except asyncio.CancelledError:
                print("[GLFT_CALC] Task cancelled")
                break

            except Exception as e:
                if VERBOSE_FAILURES:
                    print(f"[GLFT_CALC] Error in computation loop: {e}")
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.5)

        print("[GLFT_CALC] Task stopped")

    # ========== CACHED ACCESSORS ==========
    async def get_cached_position(self):
        async with self._position_lock:
            return self._pos_base, self._pos_notional

    async def get_cached_open_orders(self):
        async with self._orders_lock:
            return list(self._open_orders)

    async def get_cached_tfi(self):
        async with self._trades_lock:
            return float(self._tfi_smooth)

    async def get_cached_obi(self):
        async with self._obi_lock:
            return float(self._obi_smooth)

    async def get_cached_orderbook(self):
        async with self._ob_cache_lock:
            now_ms = int(time.time() * 1000)
            age_ms = now_ms - self._ob_cache['timestamp_ms']

            if not self._ob_cache.get('updated'):
                return None, None, [], [], age_ms

            if age_ms > ORDERBOOK_CACHE_MAX_AGE_MS:
                if VERBOSE_FAILURES:
                    print(
                        f"[OB_CACHE] WARNING: Cached data is {age_ms}ms old (max {ORDERBOOK_CACHE_MAX_AGE_MS}ms) - using REST fallback")
                return None, None, [], [], age_ms

            return (
                self._ob_cache['best_bid'],
                self._ob_cache['best_ask'],
                self._ob_cache['bids'],
                self._ob_cache['asks'],
                age_ms
            )

    async def get_cached_glft_metrics(self):
        async with self._glft_cache_lock:
            now_ms = int(time.time() * 1000)
            age_ms = now_ms - self._glft_cache['timestamp_ms']

            if age_ms > GLFT_METRICS_TIMEOUT_SEC * 1000:
                if VERBOSE_FAILURES:
                    print(f"[GLFT_CACHE] WARNING: Metrics are {age_ms}ms old (max {GLFT_METRICS_TIMEOUT_SEC * 1000}ms)")

            return dict(self._glft_cache), age_ms

    async def mutate_cached_orders_remove(self, order_ids):
        if not order_ids:
            return
        async with self._orders_lock:
            before = len(self._open_orders)
            self._open_orders = [o for o in self._open_orders if o.get('id') not in set(order_ids)]
            after = len(self._open_orders)
            if VERBOSE_FAILURES and before != after:
                print(f"[DEBUG] Removed {before - after} orders from cache (cancels).")

    async def mutate_cached_orders_add(self, orders):
        if not orders:
            return
        async with self._orders_lock:
            existing_ids = {o.get('id') for o in self._open_orders}
            for o in orders:
                oid = o.get('id')
                if oid and oid not in existing_ids:
                    self._open_orders.append(o)

    # ========== CORE LOGIC ==========
    def compute_order_quantity(self, mid_price):
        baseline = ORDER_QTY_BASE
        min_qty_for_notional = self.min_notional / mid_price
        raw_qty = max(baseline, min_qty_for_notional, self.min_qty)
        steps = math.ceil(raw_qty / self.lot_step)
        q = steps * self.lot_step
        if self.max_qty and q > self.max_qty:
            q = self.max_qty
        return q

    @staticmethod
    def compute_obi_from_ob(ob, levels=5):
        bids = ob.get('bids', [])[:max(0, int(levels))]
        asks = ob.get('asks', [])[:max(0, int(levels))]
        sum_bid = float(sum(q for _, q in bids)) if bids else 0.0
        sum_ask = float(sum(q for _, q in asks)) if asks else 0.0
        denom = sum_bid + sum_ask
        return (sum_bid - sum_ask) / denom if denom > 1e-12 else 0.0

    async def run_once(self, t=0):
        """Main quote cycle: Read from caches, compute orders, place/cancel."""
        await self.check_k_readiness()
        await self.log_warmup_status(t)

        if not self.is_warmup_complete():
            return

        try:
            # ========== GET MARKET DATA ==========
            best_bid, best_ask, bids, asks, ob_age = await self.get_cached_orderbook()

            if best_bid is None or best_ask is None:
                if VERBOSE_FAILURES:
                    print("[WARN] Order book cache empty or stale, fetching via REST...")
                try:
                    ob = await self.exchange.fetch_order_book(self.symbol)
                    best_bid = ob['bids'][0][0]
                    best_ask = ob['asks'][0][0]
                    ob_age = 0
                except Exception as e:
                    print(f"[ERROR] Could not fetch order book: {e}")
                    return

            # ========== GET CACHED DATA ==========
            glft_metrics, glft_age = await self.get_cached_glft_metrics()
            pos_base, pos_notional = await self.get_cached_position()
            open_orders = await self.get_cached_open_orders()

            mid = glft_metrics.get('mid', (best_bid + best_ask) / 2.0)
            dyn_qty = self.compute_order_quantity(mid_price=mid)

            if dyn_qty * mid < self.min_notional - 1e-9:
                print(
                    f"[WARN] Computed qty notional {dyn_qty * mid:.4f} < min_notional {self.min_notional:.4f}, skipping.")
                return

            # ========== POSITION & DELTA ==========
            normalized_position = float(np.clip(
                (pos_base / dyn_qty) if dyn_qty else 0.0,
                -MAX_NORM_POS,
                MAX_NORM_POS
            ))
            dyn_delta = GLFT_DELTA_MIN + GLFT_DELTA_SLOPE * abs(normalized_position)
            dyn_delta = float(np.clip(dyn_delta, GLFT_DELTA_MIN, GLFT_DELTA_MAX))

            # ========== CALCULATE SPREAD ==========
            if self._k_ready and np.isfinite(self.k):
                c1, c2 = compute_coeff(GLFT_GAMMA, GLFT_DELTA_MIN, self.A, self.k)
                volatility_ticks = glft_metrics.get('volatility_ticks', 1.0)
                half_spread_tick = (c1 + (dyn_delta / 2) * c2 * volatility_ticks) * HALF_SPREAD_COEFF
                if not np.isfinite(half_spread_tick) or half_spread_tick <= 0:
                    half_spread_tick = 1.0
            else:
                half_spread_tick = glft_metrics.get('half_spread_tick', 1.0)

            # ========== CALCULATE SKEW WITH DETAILED BREAKDOWN ==========
            tfi = await self.get_cached_tfi() if ENABLE_TRADE_FLOW else 0.0
            obi = await self.get_cached_obi() if ENABLE_OBI else 0.0

            # Initialize skew components
            inv_skew_ticks = 0.0
            obi_skew_ticks = 0.0
            tfi_skew_ticks = 0.0
            skew_ticks = 0.0

            if self._k_ready and np.isfinite(self.k):
                c1, c2 = compute_coeff(GLFT_GAMMA, GLFT_DELTA_MIN, self.A, self.k)
                volatility_ticks = glft_metrics.get('volatility_ticks', 1.0)

                # 1. INVENTORY SKEW (position-based)
                base_skew_per_unit = c2 * volatility_ticks
                inv_skew_ticks = base_skew_per_unit * SKEW_COEFF * normalized_position

                # 2. OBI SKEW (order book imbalance)
                beta_obi_eff = 0.0
                if ENABLE_OBI:
                    beta_obi_raw = OBI_FRACTION_OF_HALFSPREAD * half_spread_tick
                    beta_obi_eff = float(np.clip(beta_obi_raw, OBI_BETA_MIN_TICKS, OBI_BETA_MAX_TICKS))
                    obi_skew_ticks = beta_obi_eff * obi

                # 3. TFI SKEW (trade flow imbalance)
                beta_tfi_eff = 0.0
                if ENABLE_TRADE_FLOW:
                    beta_tfi_raw = TFI_FRACTION_OF_HALFSPREAD * half_spread_tick
                    beta_tfi_eff = float(np.clip(beta_tfi_raw, TFI_BETA_MIN_TICKS, TFI_BETA_MAX_TICKS))
                    tfi_skew_ticks = beta_tfi_eff * tfi

                # 4. TOTAL ALPHA SKEW (market signals)
                alpha_skew_ticks = obi_skew_ticks + tfi_skew_ticks

                # 5. COMBINED SKEW (inventory adjustment minus market signals)
                skew_ticks_target = inv_skew_ticks - alpha_skew_ticks
                self._skew_ticks_smooth = skew_ticks_target

                # 6. APPLY CAP
                skew_cap_ticks = SKEW_FRACTION_OF_HALFSPREAD_CAP * half_spread_tick
                skew_ticks = float(np.clip(self._skew_ticks_smooth, -skew_cap_ticks, skew_cap_ticks))

                # ========== DETAILED SKEW LOGGING ==========
                print(
                    f"[SKEW] pos={pos_base:.6f} norm_pos={normalized_position:.4f} | "
                    f"INV_SKEW={inv_skew_ticks:+.3f} ticks "
                    f"(c2={c2:.3f} × vol={volatility_ticks:.3f} × coeff={SKEW_COEFF:.3f} × norm={normalized_position:.2f})"
                )

                if ENABLE_OBI or ENABLE_TRADE_FLOW:
                    print(
                        f"[SKEW] obi={obi:+.4f} tfi={tfi:+.4f} | "
                        f"OBI_SKEW={obi_skew_ticks:+.3f} ticks (beta={beta_obi_eff:.2f} × obi) | "
                        f"TFI_SKEW={tfi_skew_ticks:+.3f} ticks (beta={beta_tfi_eff:.2f} × tfi)"
                    )

                print(
                    f"[SKEW] TOTAL: inventory={inv_skew_ticks:+.3f} - alpha={alpha_skew_ticks:+.3f} "
                    f"= raw={skew_ticks_target:+.3f} → capped={skew_ticks:+.3f} ticks "
                    f"(cap=±{skew_cap_ticks:.2f})"
                )

        except Exception as e:
            print(f"[ERROR] Cache read error: {e}")
            import traceback
            traceback.print_exc()
            return

        # ========== GRID PLACEMENT ==========
        reservation_price_tick = (mid / self.tick_size) - skew_ticks

        bid_tick = min(np.round(reservation_price_tick - half_spread_tick), best_bid / self.tick_size)
        ask_tick = max(np.round(reservation_price_tick + half_spread_tick), best_ask / self.tick_size)

        bid_price = passive_buy_price(bid_tick * self.tick_size, best_ask, self.tick_size)
        ask_price = passive_sell_price(ask_tick * self.tick_size, best_bid, self.tick_size)

        grid_interval = max(np.round(half_spread_tick) * self.tick_size, self.tick_size)

        # Build bid grid
        new_bids = {}
        if pos_base < MAX_POSITION_BASE and np.isfinite(bid_price):
            price = bid_price
            for _ in range(GRID_NUM):
                safe = passive_buy_price(price, best_ask, self.tick_size)
                if safe <= 0:
                    break
                new_bids[price_to_tick_index(safe, self.tick_size)] = safe
                price -= grid_interval
                if price <= 0:
                    break

        # Build ask grid
        new_asks = {}
        if pos_base > -MAX_POSITION_BASE and np.isfinite(ask_price):
            price = ask_price
            for _ in range(GRID_NUM):
                safe = passive_sell_price(price, best_bid, self.tick_size)
                if safe <= 0:
                    break
                new_asks[price_to_tick_index(safe, self.tick_size)] = safe
                price += grid_interval

        # ========== DETERMINE ORDERS TO CANCEL ==========
        grid_bid_ticks = set(new_bids.keys())
        grid_ask_ticks = set(new_asks.keys())

        order_ids_to_cancel = []
        if CANCEL_STRATEGY == 'stale':
            for o in open_orders:
                side = o['side'].lower()
                tick = price_to_tick_index(o['price'], self.tick_size)
                if (side == 'buy' and tick not in grid_bid_ticks) or \
                        (side == 'sell' and tick not in grid_ask_ticks):
                    order_ids_to_cancel.append(o['id'])
        elif CANCEL_STRATEGY == 'all':
            order_ids_to_cancel = [o['id'] for o in open_orders]

        # ========== STEP 1: CANCEL ORDERS ==========
        canceled_ids = []
        if order_ids_to_cancel:
            print(f"[ORDER_FLOW] Step 1: Canceling {len(order_ids_to_cancel)} orders...")

            canceled_success, failed_cancel = await smart_batch_cancel_orders(
                self.exchange,
                self.symbol,
                order_ids_to_cancel,
                use_native=ENABLE_NATIVE_BATCH,
            )

            # Extract canceled IDs
            for item in canceled_success:
                if isinstance(item, str):
                    canceled_ids.append(item)
                elif isinstance(item, dict):
                    oid = item.get('orderId') or item.get('id') or item.get('clientOrderId')
                    if oid:
                        canceled_ids.append(str(oid))

            # If no IDs extracted but no errors, assume all canceled
            if not canceled_ids and not failed_cancel and order_ids_to_cancel:
                canceled_ids = order_ids_to_cancel

            # Update cache
            await self.mutate_cached_orders_remove(canceled_ids)

            print(f"[ORDER_FLOW] ✓ Canceled {len(canceled_ids)}/{len(order_ids_to_cancel)} orders")

            # Delay before placing
            if canceled_ids and CANCEL_TO_PLACE_DELAY_SEC > 0:
                await asyncio.sleep(CANCEL_TO_PLACE_DELAY_SEC)
        else:
            if VERBOSE_FAILURES and t % 10 == 0:
                print(f"[ORDER_FLOW] No orders to cancel")

        # ========== STEP 2: BUILD NEW ORDER SPECS ==========
        active_order_price_ticks = set(
            price_to_tick_index(o['price'], self.tick_size)
            for o in open_orders
            if o['id'] not in canceled_ids
        )

        new_orders_spec = []

        # Bid candidates (not already on book)
        bid_candidates = [
            new_bids[t]
            for t in sorted(grid_bid_ticks, reverse=True)
            if t not in active_order_price_ticks
        ]

        # Ask candidates (not already on book)
        ask_candidates = [
            new_asks[t]
            for t in sorted(grid_ask_ticks)
            if t not in active_order_price_ticks
        ]

        # Interleave bids and asks for better fill probability
        i = j = 0
        while i < len(bid_candidates) or j < len(ask_candidates):
            if i < len(bid_candidates):
                if pos_base + dyn_qty <= MAX_POSITION_BASE:
                    new_orders_spec.append({
                        "side": "buy",
                        "amount": dyn_qty,
                        "price": bid_candidates[i]
                    })
                i += 1

            if j < len(ask_candidates):
                if pos_base - dyn_qty >= -MAX_POSITION_BASE:
                    new_orders_spec.append({
                        "side": "sell",
                        "amount": dyn_qty,
                        "price": ask_candidates[j]
                    })
                j += 1

        # ========== STEP 3: PLACE NEW ORDERS ==========
        placed_results = []
        if new_orders_spec:
            print(f"[ORDER_FLOW] Step 2: Placing {len(new_orders_spec)} new orders...")

            placed_results, failed_place = await smart_batch_place_orders(
                self.exchange,
                self.symbol,
                new_orders_spec,
                post_only=POST_ONLY,
                use_native=ENABLE_NATIVE_BATCH,
            )

            # Update cache
            await self.mutate_cached_orders_add(placed_results)

            print(f"[ORDER_FLOW] ✓ Placed {len(placed_results)}/{len(new_orders_spec)} orders")
        else:
            if VERBOSE_FAILURES and t % 10 == 0:
                print(f"[ORDER_FLOW] No new orders to place")

        # ========== SUMMARY LOGGING ==========
        ws_status = "WS" if self._ws_connected else "REST"
        print(
            f"[GLFT] t={t} mid={mid:.2f} dyn_qty={dyn_qty:.6f} notional≈{dyn_qty * mid:.2f} | "
            f"A={glft_metrics.get('A', self.A):.3g} k={glft_metrics.get('k', self.k):.6f} vol_ticks={glft_metrics.get('volatility_ticks', 1.0):.3f} | "
            f"half_spread={half_spread_tick:.2f} ticks res_price_tick={reservation_price_tick:.2f} | "
            f"bid={bid_price:.2f} ask={ask_price:.2f} | "
            f"open={len(open_orders)} canceled={len(canceled_ids)} placed={len(placed_results)} | "
            f"{ws_status} ob_age={ob_age}ms glft_age={glft_age}ms"
        )

    async def run_live(self):
        """Main event loop: Quote continuously at QUOTE_INTERVAL_SEC."""
        if self._running:
            print("[WARN] run_live already in progress.")
            return
        self._running = True
        await self.start_background_tasks()
        t = 0
        consec_errors = 0
        try:
            while True:
                try:
                    await self.run_once(t)
                    t += 1
                    consec_errors = 0
                except Exception as e:
                    consec_errors += 1
                    print(f"[ERROR] Main loop: {e} (consec={consec_errors})")
                    if VERBOSE_FAILURES:
                        import traceback
                        traceback.print_exc()
                    if consec_errors > MAX_CONSEC_ERRORS:
                        print("[FATAL] Exceeded max consecutive errors. Stopping.")
                        break
                await asyncio.sleep(QUOTE_INTERVAL_SEC)
        finally:
            await self.stop_background_tasks()
            self._running = False


# ---------------------------
# MAIN
# ---------------------------
async def main():
    if not API_KEY or not API_SECRET:
        print("Set API_KEY / API_SECRET")
        return

    # ========== Create TWO Exchange Instances ==========
    print(f"[INFO] Creating exchange instances...")

    # Main exchange: orderbook, orders, positions
    exchange = ccxtpro.aster({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    await exchange.load_markets()

    # Separate exchange for trades WebSocket
    exchange_trades = ccxtpro.aster({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    await exchange_trades.load_markets()

    print(f"[INFO] Exchange: {exchange.id}")
    print(f"[INFO] Symbol: {SYMBOL}")
    print(f"[INFO] Using 0.1-second sampling for arrival depth and volatility\n")

    # ========== Open Order Book WebSocket ==========
    print(f"[INFO] Opening order book WebSocket connection...")
    try:
        ob = await asyncio.wait_for(
            exchange.watch_order_book(SYMBOL, limit=5),
            timeout=10.0
        )
        print(f"[INFO] ✓ Order book WebSocket connection OPEN: bid={ob['bids'][0][0]:.2f} ask={ob['asks'][0][0]:.2f}")
    except Exception as e:
        print(f"[WARN] Failed to open order book WebSocket: {e}")

    # Delay before opening trades
    print(f"[INFO] Waiting 2 seconds before opening trades connection...")
    await asyncio.sleep(2.0)

    # ========== Open Trades WebSocket ==========
    if exchange_trades.has.get('watchTrades'):
        print(f"[INFO] Opening trades WebSocket connection...")
        try:
            trades = await asyncio.wait_for(
                exchange_trades.watch_trades(SYMBOL),
                timeout=10.0
            )
            print(f"[INFO] ✓ Trades WebSocket connection OPEN ({len(trades)} trades received)")
        except Exception as e:
            print(f"[WARN] Failed to open trades WebSocket: {e}")

    print(f"[INFO] ✓ Both WebSocket connections established\n")
    await asyncio.sleep(1.0)

    # ========== Continue with Bot Setup ==========
    filters = await fetch_market_filters(exchange, SYMBOL)
    print(f"[INFO] Market filters (effective): {filters}")

    # Pass BOTH exchange instances to the bot
    mm = GLFTGridMarketMaker(
        exchange,
        exchange_trades,
        SYMBOL,
        filters['tick_size'],
        filters['lot_step'],
        filters['min_qty'],
        filters['max_qty'],
        filters['min_notional'],
    )

    try:
        await mm.run_live()
    finally:
        await exchange.close()
        await exchange_trades.close()


if __name__ == "__main__":
    asyncio.run(main())