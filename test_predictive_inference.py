from __future__ import annotations

import unittest

import numpy as np

from predictive_inference import _moving_block_indices, paired_moving_block_bootstrap


class MovingBlockBootstrapTests(unittest.TestCase):
    def test_sampler_returns_valid_contiguous_blocks(self):
        rng = np.random.default_rng(11)
        idx = _moving_block_indices(30, 5, rng)
        self.assertEqual(len(idx), 30)
        self.assertTrue(((idx >= 0) & (idx < 30)).all())
        for chunk_start in range(0, 25, 5):
            block = idx[chunk_start:chunk_start + 5]
            if len(block) == 5:
                self.assertTrue(np.all(np.diff(block) == 1))

    def test_date_clustered_comparison_is_reproducible(self):
        dates = np.repeat(np.arange("2025-01-01", "2025-03-02", dtype="datetime64[D]"), 4)
        y_true = np.tile([0, 1, 2, 0], len(dates) // 4)
        ref = y_true.copy()
        candidate = y_true.copy()

        first = paired_moving_block_bootstrap(y_true, ref, candidate, dates,
                                              block_lengths=(3, 5), n_resamples=500,
                                              random_state=5)
        second = paired_moving_block_bootstrap(y_true, ref, candidate, dates,
                                               block_lengths=(3, 5), n_resamples=500,
                                               random_state=5)
        self.assertEqual(len(first), 2)
        self.assertAlmostEqual(first.iloc[0]["Accuracy Difference (pp)"], 0.0, places=8)
        self.assertTrue(first.equals(second))
        self.assertTrue(first["Two-Sided Bootstrap p-value"].between(0, 1).all())

    def test_candidate_improvement_is_positive(self):
        n_dates, n_stocks = 40, 5
        dates = np.repeat(np.arange("2025-01-01", "2025-02-10", dtype="datetime64[D]"), n_stocks)
        y = np.zeros(n_dates * n_stocks, dtype=int)
        reference = np.ones_like(y)
        candidate = np.zeros_like(y)
        result = paired_moving_block_bootstrap(y, reference, candidate, dates,
                                               block_lengths=(5,), n_resamples=300)
        self.assertAlmostEqual(result.iloc[0]["Accuracy Difference (pp)"], 100.0)
        self.assertGreater(result.iloc[0]["95% CI Lower (pp)"], 0.0)

    def test_rejects_inconsistent_array_lengths(self):
        with self.assertRaises(ValueError):
            paired_moving_block_bootstrap([0, 1], [0], [0, 1], ["2025-01-01", "2025-01-02"])

    def test_rejects_too_short_series(self):
        dates = np.repeat(np.arange("2025-01-01", "2025-01-11", dtype="datetime64[D]"), 2)
        y = np.zeros(len(dates), dtype=int)
        with self.assertRaises(ValueError):
            paired_moving_block_bootstrap(y, y, y, dates, block_lengths=(3,), n_resamples=100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
