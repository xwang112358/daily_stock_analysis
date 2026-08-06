# -*- coding: utf-8 -*-
"""Tests for the local chip-distribution fallback calculator."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from data_provider.local_chip_calculator import (
    MIN_BARS_REQUIRED,
    compute_chip_distribution,
    estimate_turnover_rates,
)


def _make_bars(n=60, start=10.0, step=0.1):
    """Synthetic ascending bars: slow uptrend with fixed intraday range."""
    bars = []
    price = start
    for i in range(n):
        bars.append({
            "date": f"2026-01-{i + 1:02d}",
            "open": round(price, 2),
            "close": round(price + step, 2),
            "high": round(price + step * 2, 2),
            "low": round(price - step, 2),
            "volume": 1_000_000,
        })
        price += step
    return bars


class TestEstimateTurnoverRates(unittest.TestCase):
    def test_by_float_shares(self):
        rates = estimate_turnover_rates([1_000_000, 2_000_000], float_shares=100_000_000)
        self.assertEqual(rates, [1.0, 2.0])

    def test_by_latest_turnover_rate(self):
        rates = estimate_turnover_rates(
            [1_000_000, 2_000_000],
            latest_turnover_rate_pct=2.0,
        )
        # float_shares = 2M / 2% = 100M → 第一天 1%
        self.assertAlmostEqual(rates[0], 1.0)
        self.assertAlmostEqual(rates[1], 2.0)

    def test_float_shares_wins_over_rate(self):
        rates = estimate_turnover_rates(
            [1_000_000],
            float_shares=200_000_000,
            latest_turnover_rate_pct=50.0,
        )
        self.assertAlmostEqual(rates[0], 0.5)

    def test_missing_inputs_return_none(self):
        self.assertIsNone(estimate_turnover_rates([]))
        self.assertIsNone(estimate_turnover_rates([1_000_000]))
        self.assertIsNone(estimate_turnover_rates([1_000_000], latest_turnover_rate_pct=0))
        self.assertIsNone(estimate_turnover_rates([0], latest_turnover_rate_pct=5.0))

    def test_rates_capped_at_100(self):
        rates = estimate_turnover_rates([500_000_000], float_shares=100_000_000)
        self.assertEqual(rates, [100.0])


class TestComputeChipDistribution(unittest.TestCase):
    def test_basic_metrics_are_sane(self):
        bars = _make_bars(60)
        rates = [5.0] * 60
        chip = compute_chip_distribution(bars, rates, code="600021")
        self.assertIsNotNone(chip)
        self.assertEqual(chip.source, "local_calc")
        self.assertEqual(chip.date, bars[-1]["date"])
        self.assertGreaterEqual(chip.profit_ratio, 0.0)
        self.assertLessEqual(chip.profit_ratio, 1.0)
        # 上涨趋势 + 现价为最新收盘 → 大部分筹码获利
        self.assertGreater(chip.profit_ratio, 0.5)
        # 平均成本落在历史价格区间内
        lows = min(b["low"] for b in bars)
        highs = max(b["high"] for b in bars)
        self.assertGreaterEqual(chip.avg_cost, lows)
        self.assertLessEqual(chip.avg_cost, highs)
        # 90% 区间包含 70% 区间，集中度非负
        self.assertLessEqual(chip.cost_90_low, chip.cost_70_low)
        self.assertGreaterEqual(chip.cost_90_high, chip.cost_70_high)
        self.assertGreaterEqual(chip.concentration_90, chip.concentration_70)

    def test_profit_ratio_extremes(self):
        bars = _make_bars(60)
        rates = [5.0] * 60
        top = compute_chip_distribution(bars, rates, code="X", current_price=1e6)
        bottom = compute_chip_distribution(bars, rates, code="X", current_price=0.01)
        self.assertAlmostEqual(top.profit_ratio, 1.0, places=3)
        self.assertAlmostEqual(bottom.profit_ratio, 0.0, places=3)

    def test_limit_board_bars_do_not_crash(self):
        bars = _make_bars(40)
        # 后 5 天一字板（open=close=high=low）
        for bar in bars[-5:]:
            bar["open"] = bar["close"] = bar["high"] = bar["low"] = 20.0
        chip = compute_chip_distribution(bars, [5.0] * 40, code="X")
        self.assertIsNotNone(chip)

    def test_too_few_bars_returns_none(self):
        n = MIN_BARS_REQUIRED - 1
        self.assertIsNone(compute_chip_distribution(_make_bars(n), [5.0] * n, code="X"))

    def test_zero_turnover_returns_none(self):
        bars = _make_bars(60)
        self.assertIsNone(compute_chip_distribution(bars, [0.0] * 60, code="X"))

    def test_length_mismatch_returns_none(self):
        bars = _make_bars(60)
        self.assertIsNone(compute_chip_distribution(bars, [5.0] * 59, code="X"))


class TestLocalChipFallbackIntegration(unittest.TestCase):
    """base.DataFetcherManager._compute_local_chip_fallback wiring."""

    @staticmethod
    def _make_rows(n=60):
        rows = []
        price = 10.0
        for i in range(n):
            rows.append(SimpleNamespace(
                date=f"2026-01-{i + 1:02d}",
                open=price, close=price + 0.1, high=price + 0.2, low=price - 0.1,
                volume=1_000_000,
            ))
            price += 0.1
        rows.reverse()  # get_latest_data 返回降序
        return rows

    def _manager(self):
        from data_provider.base import DataFetcherManager
        mgr = DataFetcherManager.__new__(DataFetcherManager)
        return mgr

    def test_fallback_success_with_circ_mv(self):
        mgr = self._manager()
        quote = SimpleNamespace(price=16.0, circ_mv=1_600_000_000, turnover_rate=0.0)
        mgr.get_realtime_quote = MagicMock(return_value=quote)
        db = MagicMock()
        db.get_latest_data.return_value = self._make_rows()

        with patch("src.storage.get_db", return_value=db):
            chip = mgr._compute_local_chip_fallback("600021")

        self.assertIsNotNone(chip)
        self.assertEqual(chip.source, "local_calc")
        self.assertGreater(chip.avg_cost, 0)

    def test_fallback_skips_non_a_share(self):
        mgr = self._manager()
        self.assertIsNone(mgr._compute_local_chip_fallback("AAPL"))
        self.assertIsNone(mgr._compute_local_chip_fallback("hk00700"))

    def test_fallback_gives_up_without_shares_info(self):
        mgr = self._manager()
        quote = SimpleNamespace(price=16.0, circ_mv=None, turnover_rate=0.0)
        mgr.get_realtime_quote = MagicMock(return_value=quote)
        db = MagicMock()
        db.get_latest_data.return_value = self._make_rows()

        with patch("src.storage.get_db", return_value=db):
            self.assertIsNone(mgr._compute_local_chip_fallback("600021"))

    def test_fallback_survives_db_errors(self):
        mgr = self._manager()
        with patch("src.storage.get_db", side_effect=RuntimeError("db down")):
            self.assertIsNone(mgr._compute_local_chip_fallback("600021"))

    def test_chip_local_first_skips_external_fetchers(self):
        """CHIP_LOCAL_FIRST=true 时本地计算成功则不触碰外部数据源。"""
        mgr = self._manager()
        sentinel = SimpleNamespace(profit_ratio=0.5)
        mgr._compute_local_chip_fallback = MagicMock(return_value=sentinel)
        mgr._get_fetchers_snapshot = MagicMock(side_effect=AssertionError("external fetchers must not be touched"))

        config = SimpleNamespace(enable_chip_distribution=True, chip_local_first=True)
        with patch("src.config.get_config", return_value=config):
            chip = mgr.get_chip_distribution("600021")

        self.assertIs(chip, sentinel)
        mgr._compute_local_chip_fallback.assert_called_once_with("600021")

    def test_chip_local_first_falls_back_to_external_when_local_empty(self):
        mgr = self._manager()
        mgr._compute_local_chip_fallback = MagicMock(return_value=None)
        mgr._get_fetchers_snapshot = MagicMock(return_value=[])

        config = SimpleNamespace(enable_chip_distribution=True, chip_local_first=True)
        with patch("src.config.get_config", return_value=config):
            chip = mgr.get_chip_distribution("600021")

        self.assertIsNone(chip)
        # 本地失败后进入外部数据源遍历（本例为空列表）,且不再二次尝试本地
        mgr._get_fetchers_snapshot.assert_called_once()
        mgr._compute_local_chip_fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
