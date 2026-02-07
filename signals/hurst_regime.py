"""
Hurst Exponent via Rescaled Range (R/S) Analysis
==================================================
Detects regime type:
  H > 0.6  = Trending (persistent) - GOOD for momentum
  H ~ 0.5  = Random walk - NEUTRAL
  H < 0.4  = Mean-reverting - BAD for momentum (good for grid)

Lightweight implementation suitable for PythonAnywhere.
No external dependencies beyond numpy.
"""

import numpy as np
from typing import Optional


def compute_hurst_exponent(
    prices: list[float],
    min_window: int = 10,
    max_window: Optional[int] = None,
) -> float:
    """
    Compute Hurst Exponent using R/S analysis.

    Args:
        prices: List of closing prices (at least 50 values recommended).
        min_window: Minimum sub-series window size.
        max_window: Maximum window size (defaults to len/2).

    Returns:
        Hurst exponent H in [0, 1].
        H > 0.5 = trending, H < 0.5 = mean-reverting, H ~ 0.5 = random.
    """
    if len(prices) < 20:
        return 0.5  # Not enough data, assume random walk

    ts = np.array(prices, dtype=np.float64)
    # Work with log returns for stationarity
    returns = np.diff(np.log(ts))
    n = len(returns)

    if max_window is None:
        max_window = n // 2

    # Generate window sizes (powers of 2 for clean division)
    window_sizes = []
    size = min_window
    while size <= max_window:
        window_sizes.append(size)
        size = int(size * 1.5)  # Geometric spacing

    if len(window_sizes) < 3:
        return 0.5  # Not enough scale variation

    rs_values = []
    for w in window_sizes:
        rs_list = []
        num_segments = n // w
        if num_segments < 1:
            continue

        for i in range(num_segments):
            segment = returns[i * w: (i + 1) * w]
            mean_seg = np.mean(segment)
            devs = np.cumsum(segment - mean_seg)
            r = np.max(devs) - np.min(devs)
            s = np.std(segment, ddof=1)
            if s > 1e-10:  # Avoid division by zero
                rs_list.append(r / s)

        if rs_list:
            rs_values.append((np.log(w), np.log(np.mean(rs_list))))

    if len(rs_values) < 3:
        return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])

    # Least squares fit
    n_pts = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    denom = n_pts * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-10:
        return 0.5

    hurst = (n_pts * sum_xy - sum_x * sum_y) / denom

    # Clamp to valid range
    return float(np.clip(hurst, 0.0, 1.0))


def classify_regime(hurst: float) -> str:
    """
    Classify market regime from Hurst exponent.

    Returns: 'trending', 'random', or 'mean_reverting'
    """
    if hurst > 0.6:
        return 'trending'
    elif hurst < 0.4:
        return 'mean_reverting'
    else:
        return 'random'


def is_trending(hurst: float, threshold: float = 0.55) -> bool:
    """Check if market is in trending regime (good for momentum)."""
    return hurst > threshold
