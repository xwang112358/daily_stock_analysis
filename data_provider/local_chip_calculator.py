# -*- coding: utf-8 -*-
"""
本地筹码分布计算器（外部筹码数据源全部失败时的零网络兜底）。

算法忠实移植东方财富前端 CYQCalculator（akshare ``stock_cyq_em`` 拉取远端
K 线后本地运行的同一套算法）：150 档价格网格上，逐日按三角分布（峰值在
均价 (open+close+high+low)/4）叠加当日筹码，并以换手率对存量筹码衰减，
窗口为最近 120 根日线。

与东财的差异仅在换手率来源：东财 K 线自带每日换手率，本地兜底用
「最新实时换手率 + 同日成交量」反推流通股本，再重建历史换手率序列
（窗口内股本不变的近似；有增发/解禁时会有偏差，但量级正确）。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

from .realtime_types import ChipDistribution

logger = logging.getLogger(__name__)

PRICE_BUCKETS = 150          # 东财 factor
WINDOW_BARS = 120            # 东财 this.range
MIN_BARS_REQUIRED = 30       # 样本太少时结果不可信，放弃兜底
_MIN_ACCURACY = 0.01

LOCAL_CHIP_SOURCE = "local_calc"


def estimate_turnover_rates(
    volumes: Sequence[float],
    *,
    float_shares: Optional[float] = None,
    latest_turnover_rate_pct: Optional[float] = None,
) -> Optional[List[float]]:
    """重建历史换手率序列（百分比值）。

    优先使用 ``float_shares``（流通股本，可由 流通市值/现价 求得）；缺失时由
    「最新换手率 + 同日成交量」反推。``volumes`` 按时间升序，日线 volume 单位
    须与 float_shares 一致（A 股日线均为股）。
    """
    if not volumes:
        return None

    shares = None
    try:
        if float_shares is not None and float(float_shares) > 0 and math.isfinite(float(float_shares)):
            shares = float(float_shares)
    except (TypeError, ValueError):
        shares = None

    if shares is None and latest_turnover_rate_pct is not None:
        try:
            latest_rate = float(latest_turnover_rate_pct)
            latest_volume = float(volumes[-1])
        except (TypeError, ValueError):
            return None
        if latest_rate <= 0 or latest_volume <= 0 or not math.isfinite(latest_rate):
            return None
        shares = latest_volume / (latest_rate / 100.0)

    if shares is None or shares <= 0:
        return None

    rates: List[float] = []
    for volume in volumes:
        try:
            value = float(volume)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0 or not math.isfinite(value):
            rates.append(0.0)
            continue
        rates.append(min(100.0, value / shares * 100.0))
    return rates


def compute_chip_distribution(
    bars: Sequence[Dict[str, Any]],
    turnover_rates_pct: Sequence[float],
    *,
    code: str,
    current_price: Optional[float] = None,
) -> Optional[ChipDistribution]:
    """按东财 CYQCalculator 计算最新一日的筹码分布指标。

    Args:
        bars: 时间升序日线，元素需含 open/high/low/close（数值）。
        turnover_rates_pct: 与 bars 等长的换手率序列（百分比值，如 5.2）。
        code: 股票代码（仅用于结果标注）。
        current_price: 计算获利比例的现价；缺省用最后一根收盘价。
    """
    if len(bars) != len(turnover_rates_pct):
        return None
    if len(bars) < MIN_BARS_REQUIRED:
        return None

    window = list(bars[-WINDOW_BARS:])
    rates = list(turnover_rates_pct[-WINDOW_BARS:])

    try:
        highs = [float(b["high"]) for b in window]
        lows = [float(b["low"]) for b in window]
    except (KeyError, TypeError, ValueError):
        return None
    max_price = max(highs)
    min_price = min(lows)
    if not (math.isfinite(max_price) and math.isfinite(min_price)) or max_price <= 0:
        return None

    accuracy = max(_MIN_ACCURACY, (max_price - min_price) / (PRICE_BUCKETS - 1))
    chips = [0.0] * PRICE_BUCKETS

    for bar, rate_pct in zip(window, rates):
        try:
            open_ = float(bar["open"])
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
        except (KeyError, TypeError, ValueError):
            continue
        avg = (open_ + close + high + low) / 4
        turnover = min(1.0, max(0.0, float(rate_pct) / 100.0))

        bucket_high = int(math.floor((high - min_price) / accuracy))
        bucket_low = int(math.ceil((low - min_price) / accuracy))
        g_height = (PRICE_BUCKETS - 1) if high == low else 2 / (high - low)
        g_bucket = int(math.floor((avg - min_price) / accuracy))

        # 存量筹码按换手率衰减
        for n in range(PRICE_BUCKETS):
            chips[n] *= (1 - turnover)

        if high == low:
            # 一字板：全部堆在均价档（矩形面积为三角形 2 倍，故除 2）
            chips[g_bucket] += g_height * turnover / 2
            continue
        for j in range(bucket_low, min(bucket_high, PRICE_BUCKETS - 1) + 1):
            cur_price = min_price + accuracy * j
            if cur_price <= avg:
                if abs(avg - low) < 1e-8:
                    chips[j] += g_height * turnover
                else:
                    chips[j] += (cur_price - low) / (avg - low) * g_height * turnover
            else:
                if abs(high - avg) < 1e-8:
                    chips[j] += g_height * turnover
                else:
                    chips[j] += (high - cur_price) / (high - avg) * g_height * turnover

    total_chips = sum(chips)
    if total_chips <= 0:
        return None

    def cost_by_chip(target: float) -> float:
        cumulative = 0.0
        for i in range(PRICE_BUCKETS):
            if cumulative + chips[i] > target:
                return min_price + i * accuracy
            cumulative += chips[i]
        return min_price + (PRICE_BUCKETS - 1) * accuracy

    def percent_range(percent: float) -> tuple:
        low_cost = cost_by_chip(total_chips * (1 - percent) / 2)
        high_cost = cost_by_chip(total_chips * (1 + percent) / 2)
        concentration = 0.0 if (low_cost + high_cost) == 0 else (high_cost - low_cost) / (low_cost + high_cost)
        return low_cost, high_cost, concentration

    try:
        price_now = float(current_price) if current_price is not None else float(window[-1]["close"])
    except (KeyError, TypeError, ValueError):
        return None
    benefit = sum(
        chips[i] for i in range(PRICE_BUCKETS) if price_now >= min_price + i * accuracy
    ) / total_chips

    cost_90_low, cost_90_high, concentration_90 = percent_range(0.9)
    cost_70_low, cost_70_high, concentration_70 = percent_range(0.7)

    bar_date = window[-1].get("date")
    return ChipDistribution(
        code=code,
        date=str(bar_date) if bar_date is not None else "",
        source=LOCAL_CHIP_SOURCE,
        profit_ratio=round(benefit, 4),
        avg_cost=round(cost_by_chip(total_chips * 0.5), 2),
        cost_90_low=round(cost_90_low, 2),
        cost_90_high=round(cost_90_high, 2),
        concentration_90=round(concentration_90, 4),
        cost_70_low=round(cost_70_low, 2),
        cost_70_high=round(cost_70_high, 2),
        concentration_70=round(concentration_70, 4),
    )
