import ccxt.pro
import asyncio
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict, deque
import time

EXCHANGE_ID = 'hyperliquid'
SYMBOL = 'HYPE/USDC:USDC'
INTERVAL = 60  # seconds
NUM_INTERVALS = 15  # Number of intervals to collect data
ROLLING_WINDOW = NUM_INTERVALS  # Rolling window size for rolling stats

def pool_and_fit(fill_counts, bin_labels, side_label):
    avg_crosses = []
    spreads = []
    for i, (left, right) in enumerate(bin_labels):
        crosses = fill_counts[i]
        avg = np.mean(crosses) if crosses else 0.0
        avg_crosses.append(avg)
        spreads.append((left + right) / 2)
    spreads = np.array(spreads)
    avg_crosses = np.array(avg_crosses)
    mask = avg_crosses > 0
    spreads = spreads[mask]
    avg_crosses = avg_crosses[mask]
    if len(spreads) == 0:
        print(f"No crosses detected for {side_label}!")
        return None, None, None, None
    log_crosses = np.log(avg_crosses)
    reg = LinearRegression().fit(spreads.reshape(-1, 1), log_crosses)
    k = -reg.coef_[0]
    A = np.exp(reg.intercept_)
    print(f"Decay parameter k for {side_label} = {k:.4f}")
    print(f"Base intensity A for {side_label} = {A:.4f}")
    return spreads, avg_crosses, k, A

def rolling_pool_and_fit(fill_counts_window, bin_labels, side_label):
    # fill_counts_window: list of fill_counts dicts for each interval in window
    pooled_counts = defaultdict(list)
    for fill_counts in fill_counts_window:
        for i in range(len(bin_labels)):
            pooled_counts[i].extend(fill_counts[i])
    return pool_and_fit(pooled_counts, bin_labels, f"{side_label} (rolling)")

def calculate_lambda(total_fills_per_interval, interval_length_sec=INTERVAL):
    if not total_fills_per_interval:
        return 0.0
    avg_fills_per_interval = np.mean(total_fills_per_interval)
    lambda_per_minute = avg_fills_per_interval / (interval_length_sec / 60)
    return lambda_per_minute

async def collect_data():
    exchange = getattr(ccxt.pro, EXCHANGE_ID)()
    await exchange.load_markets()

    spread_bins = np.arange(0.0, 0.51, 0.01)
    bin_labels = [(round(spread_bins[i], 2), round(spread_bins[i + 1], 2)) for i in range(len(spread_bins) - 1)]

    bid_total_fills_per_interval = []
    ask_total_fills_per_interval = []
    lambda_m_per_interval = []
    lambda_p_per_interval = []
    bid_fill_counts_window = deque(maxlen=ROLLING_WINDOW)
    ask_fill_counts_window = deque(maxlen=ROLLING_WINDOW)

    try:
        interval = 0
        while True:
            interval += 1
            print(f"\nInterval {interval}")

            bid_fill_counts = defaultdict(list)
            ask_fill_counts = defaultdict(list)

            # Priming order book and trades
            order_book = await exchange.watch_order_book(SYMBOL)
            if not order_book['bids'] or not order_book['asks']:
                print("Order book empty, skipping this interval.")
                continue
            mid = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
            bid_levels = []
            ask_levels = []
            for price, _ in order_book['bids'][:200]:
                if price < mid:
                    bid_levels.append((price, abs(price - mid)))
            bid_levels.sort(reverse=True)
            for price, _ in order_book['asks'][:200]:
                if price > mid:
                    ask_levels.append((price, abs(price - mid)))
            ask_levels.sort()

            bid_level_map = {price: spread for price, spread in bid_levels}
            ask_level_map = {price: spread for price, spread in ask_levels}
            bid_cross_counts = {price: 0 for price, _ in bid_levels}
            ask_cross_counts = {price: 0 for price, _ in ask_levels}

            prev_mid = mid

            trades_on_bid = 0
            trades_on_ask = 0
            last_trade_timestamp = 0

            start_time = time.time()
            while time.time() - start_time < INTERVAL:
                ob_task = asyncio.create_task(exchange.watch_order_book(SYMBOL))
                trades_task = asyncio.create_task(exchange.watch_trades(SYMBOL))
                done, pending = await asyncio.wait(
                    [ob_task, trades_task],
                    return_when=asyncio.ALL_COMPLETED
                )
                ob = ob_task.result()
                if not ob['bids'] or not ob['asks']:
                    continue
                mid = (ob['bids'][0][0] + ob['asks'][0][0]) / 2

                for price, _ in ask_levels:
                    if prev_mid < price <= mid:
                        for p, _ in ask_levels:
                            if p <= price:
                                ask_cross_counts[p] += 1
                for price, _ in bid_levels:
                    if prev_mid > price >= mid:
                        for p, _ in bid_levels:
                            if p >= price:
                                bid_cross_counts[p] += 1

                prev_mid = mid

                trades = trades_task.result()
                for trade in trades:
                    if 'timestamp' in trade and trade['timestamp'] <= last_trade_timestamp:
                        continue
                    last_trade_timestamp = max(last_trade_timestamp, trade.get('timestamp', last_trade_timestamp))
                    if trade['side'] == 'buy':
                        trades_on_ask += 1
                    elif trade['side'] == 'sell':
                        trades_on_bid += 1

                await asyncio.sleep(0.2)

            total_ask_fills = sum(ask_cross_counts[p] for p, _ in ask_levels)
            total_bid_fills = sum(bid_cross_counts[p] for p, _ in bid_levels)
            ask_total_fills_per_interval.append(total_ask_fills)
            bid_total_fills_per_interval.append(total_bid_fills)

            for price, spread in ask_level_map.items():
                for i, (left, right) in enumerate(bin_labels):
                    if left <= spread < right:
                        ask_fill_counts[i].append(ask_cross_counts[price])
                        break
            for price, spread in bid_level_map.items():
                for i, (left, right) in enumerate(bin_labels):
                    if left <= spread < right:
                        bid_fill_counts[i].append(bid_cross_counts[price])
                        break

            bid_fill_counts_window.append(bid_fill_counts)
            ask_fill_counts_window.append(ask_fill_counts)

            lambda_m = trades_on_bid / (INTERVAL / 60)
            lambda_p = trades_on_ask / (INTERVAL / 60)
            lambda_m_per_interval.append(lambda_m)
            lambda_p_per_interval.append(lambda_p)

            # Rolling lambda
            rolling_bid_fills = bid_total_fills_per_interval[-ROLLING_WINDOW:]
            rolling_ask_fills = ask_total_fills_per_interval[-ROLLING_WINDOW:]
            rolling_lambda_m = lambda_m_per_interval[-ROLLING_WINDOW:]
            rolling_lambda_p = lambda_p_per_interval[-ROLLING_WINDOW:]

            lambda_m_cross = calculate_lambda(rolling_bid_fills, INTERVAL)
            lambda_p_cross = calculate_lambda(rolling_ask_fills, INTERVAL)
            print(f"Rolling λₘ (mid crosses hitting bid per min, last {ROLLING_WINDOW}): {lambda_m_cross:.4f}")
            print(f"Rolling λₚ (mid crosses hitting ask per min, last {ROLLING_WINDOW}): {lambda_p_cross:.4f}")

            avg_lambda_m = np.mean(rolling_lambda_m) if rolling_lambda_m else 0.0
            avg_lambda_p = np.mean(rolling_lambda_p) if rolling_lambda_p else 0.0
            print(f"Rolling λₘ (market orders hitting bid per min, last {ROLLING_WINDOW}): {avg_lambda_m:.4f}")
            print(f"Rolling λₚ (market orders hitting ask per min, last {ROLLING_WINDOW}): {avg_lambda_p:.4f}")

            # Rolling window decay and base intensity
            rolling_pool_and_fit(
                list(bid_fill_counts_window), bin_labels, "bids")
            rolling_pool_and_fit(
                list(ask_fill_counts_window), bin_labels, "asks")

            # if interval >= NUM_INTERVALS:
            #     break

    finally:
        await exchange.close()

async def main():
    await collect_data()

if __name__ == "__main__":
    asyncio.run(main())