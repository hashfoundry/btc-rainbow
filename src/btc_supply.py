"""Bitcoin issuance schedule helpers.

Used by the Stock-to-Flow model, which needs circulating supply (stock) and
annual issuance (flow) for every date in the price series. Both are derived
from the consensus rules rather than an external API, so the container stays
self-contained.
"""

import numpy as np
import pandas as pd

GENESIS = pd.Timestamp("2009-01-03")

HALVING_INTERVAL = 210_000  # blocks
INITIAL_REWARD = 50.0  # BTC
BLOCKS_PER_DAY = 144.0  # 10 minute target
BLOCKS_PER_YEAR = BLOCKS_PER_DAY * 365.25

# Observed (date, height) anchors. Block production has run slightly ahead of
# the 10 minute target, so interpolating between the real halvings is more
# accurate than assuming 144 blocks/day since genesis.
HEIGHT_ANCHORS = [
    (GENESIS, 0),
    (pd.Timestamp("2012-11-28"), 210_000),
    (pd.Timestamp("2016-07-09"), 420_000),
    (pd.Timestamp("2020-05-11"), 630_000),
    (pd.Timestamp("2024-04-20"), 840_000),
]


def halving_dates(until: pd.Timestamp) -> list:
    """Halving dates up to ``until``, projecting future ones at 10 min/block."""
    dates = [date for date, _ in HEIGHT_ANCHORS[1:]]
    step = pd.Timedelta(days=HALVING_INTERVAL / BLOCKS_PER_DAY)
    while dates[-1] < until:
        dates.append(dates[-1] + step)
    return dates


def block_height(dates) -> np.ndarray:
    """Estimate block height for each date.

    Piecewise-linear between the observed halving anchors, then 144 blocks/day
    past the most recent one.
    """
    days = _days_since_genesis(dates)
    anchor_days = np.array([(d - GENESIS).days for d, _ in HEIGHT_ANCHORS], dtype=float)
    anchor_heights = np.array([h for _, h in HEIGHT_ANCHORS], dtype=float)

    height = np.interp(days, anchor_days, anchor_heights)

    # np.interp clamps past the last anchor; extrapolate at the target rate.
    beyond = days > anchor_days[-1]
    height[beyond] = anchor_heights[-1] + (days[beyond] - anchor_days[-1]) * BLOCKS_PER_DAY
    return np.maximum(height, 0.0)


def block_reward(height) -> np.ndarray:
    """Block subsidy in BTC at a given height (zero after the 33rd epoch)."""
    height = np.asarray(height, dtype=float)
    epoch = np.floor(height / HALVING_INTERVAL)
    reward = INITIAL_REWARD / np.power(2.0, epoch)
    return np.where(epoch >= 33, 0.0, reward)


def circulating_supply(height) -> np.ndarray:
    """Cumulative mined BTC at a given height."""
    height = np.asarray(height, dtype=float)
    epoch = np.floor(height / HALVING_INTERVAL).astype(int)
    epoch = np.clip(epoch, 0, 33)

    # Supply mined by all fully completed epochs: sum of a geometric series.
    completed = HALVING_INTERVAL * INITIAL_REWARD * 2.0 * (1.0 - np.power(0.5, epoch))
    # Plus the partial current epoch.
    into_epoch = height - epoch * HALVING_INTERVAL
    current = into_epoch * block_reward(height)
    return completed + current


def stock_to_flow(dates) -> np.ndarray:
    """Stock-to-flow ratio: circulating supply divided by annual issuance."""
    height = block_height(dates)
    stock = circulating_supply(height)
    flow = BLOCKS_PER_YEAR * block_reward(height)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(flow > 0, stock / flow, np.nan)
    return ratio


def _days_since_genesis(dates) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(dates))
    return (dates - GENESIS).dt.total_seconds().to_numpy() / 86400.0
