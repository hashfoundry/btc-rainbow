"""Price history loading and top-up.

The CSV in ``data/`` is the source of truth. On each run any missing days are
pulled from a public exchange and appended, so a container restart is enough to
refresh every chart.
"""

import datetime
import time

import ccxt
import numpy as np
import pandas as pd
from dateutil.parser import parse

# Tried in order; the first one that answers wins. Binance returns HTTP 451 in
# several jurisdictions, so the spot venues below act as fallbacks.
DEFAULT_EXCHANGES = ("binance", "coinbase", "kraken", "bitstamp")
SYMBOL_CANDIDATES = ("BTC/USDT", "BTC/USD", "BTC/USDC")
MAX_FETCH_ROUNDS = 40


def get_data(file_path: str, exchanges=DEFAULT_EXCHANGES, update: bool = True) -> pd.DataFrame:
    """Load the price history, topping it up from an exchange when stale.

    Args:
        file_path: Path to the CSV holding ``Date,Value`` rows.
        exchanges: Exchange ids to try, in order.
        update: Set False to work offline from the CSV as-is.

    Returns:
        A ``Date``/``Value`` frame, sorted, deduplicated and stripped of the
        zero-price rows that predate Bitcoin's first quote.
    """
    raw_data = pd.read_csv(file_path)
    raw_data["Date"] = pd.to_datetime(raw_data["Date"])
    raw_data = _clean(raw_data)

    last_date = raw_data["Date"].max()
    diff_days = (pd.Timestamp.today().normalize() - last_date.normalize()).days

    if update and diff_days > 1:
        print(f"Price data is {diff_days} days old. Fetching updates...")
        new_data = fetch_data(
            since=last_date + pd.Timedelta(days=1),
            limit=diff_days,
            exchanges=exchanges,
        )
        if new_data is not None and not new_data.empty:
            raw_data = _clean(pd.concat([raw_data, new_data], ignore_index=True))
            raw_data.to_csv(file_path, index=False)
            print(f"Added {len(new_data)} rows, history now ends {raw_data['Date'].max().date()}")
        else:
            print("No new rows retrieved; continuing with the data on disk.")
    elif update:
        print(f"Price data is current through {last_date.date()}.")
    else:
        print(f"Update disabled; using data through {last_date.date()}.")

    return raw_data


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date, drop duplicate days and rows without a real price."""
    df = df[["Date", "Value"]].copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df[np.isfinite(df["Value"]) & (df["Value"] > 0)]
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    return df.reset_index(drop=True)


def supported_exchanges() -> list:
    """Exchange ids that expose OHLCV history.

    Built lazily: instantiating every ccxt exchange takes seconds, and the list
    is only needed when an unknown exchange id has to be validated.
    """
    supported = []
    for exchange_id in ccxt.exchanges:
        try:
            if getattr(ccxt, exchange_id)().has["fetchOHLCV"]:
                supported.append(exchange_id)
        except Exception:
            continue
    return supported


def fetch_data(
    exchanges=DEFAULT_EXCHANGES,
    since=None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Fetch daily closes, trying each exchange until one succeeds.

    Args:
        exchanges: Exchange ids to try in order, or a single id as a string.
        since: Start date - datetime, parseable string, or UTC ms timestamp.
        limit: Number of daily rows wanted; paginated across calls as needed.

    Returns:
        A ``Date``/``Value`` frame, or an empty frame if every exchange failed.
    """
    if isinstance(exchanges, str):
        exchanges = [part.strip() for part in exchanges.split(",") if part.strip()]

    since_ms = _to_milliseconds(since)
    errors = []

    for exchange_id in exchanges:
        try:
            frame = _fetch_from(exchange_id, since_ms, limit)
        except Exception as exc:  # network, geo-block, delisted symbol, ...
            errors.append(f"{exchange_id}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if not frame.empty:
            print(f"Fetched {len(frame)} daily candles from {exchange_id}.")
            return frame
        errors.append(f"{exchange_id}: returned no candles")

    for message in errors:
        print(f"  ! {message}")
    return pd.DataFrame(columns=["Date", "Value"])


def _fetch_from(exchange_id: str, since_ms, limit) -> pd.DataFrame:
    """Pull daily candles from one exchange, paginating until ``limit`` rows."""
    exchange = getattr(ccxt, exchange_id.lower())()
    if not exchange.has["fetchOHLCV"]:
        raise ValueError(f"{exchange_id} does not expose OHLCV history")

    symbol = _pick_symbol(exchange)
    rate_limit = exchange.rateLimit / 1000  # ms -> s, for time.sleep

    data = exchange.fetch_ohlcv(symbol, "1d", since_ms, limit)
    rounds = 0
    while limit and len(data) < limit and rounds < MAX_FETCH_ROUNDS:
        rounds += 1
        cursor = data[-1][0] + 86_400_000
        time.sleep(rate_limit)
        new_data = exchange.fetch_ohlcv(symbol, "1d", cursor, limit - len(data))
        # Stop when the exchange stops advancing, otherwise this spins forever.
        new_data = [row for row in new_data if row[0] > data[-1][0]]
        if not new_data:
            break
        data += new_data

    if not data:
        return pd.DataFrame(columns=["Date", "Value"])

    df = pd.DataFrame(data, columns=["Timestamp", "open", "high", "low", "close", "volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"] / 1000, unit="s").dt.normalize()
    df["Value"] = pd.to_numeric(df["close"])
    return df[["Date", "Value"]]


def _pick_symbol(exchange) -> str:
    """First quote pair the exchange actually lists."""
    try:
        markets = exchange.load_markets()
    except Exception:
        return SYMBOL_CANDIDATES[0]

    for symbol in SYMBOL_CANDIDATES:
        if symbol in markets:
            return symbol
    raise ValueError(f"{exchange.id} lists none of {list(SYMBOL_CANDIDATES)}")


def _to_milliseconds(since):
    if since is None:
        return None
    if isinstance(since, str):
        since = parse(since)
    if isinstance(since, (pd.Timestamp, datetime.datetime)):
        return int(pd.Timestamp(since).timestamp() * 1000)
    return int(since)
