import itertools
import math
import unittest

import numpy as np

from selection import (
    evaluate_catalog, exact_expectation, paired_artist_jackknife,
    rank_candidates, rank_weights, select_song, selection_scores,
)


class SelectionTests(unittest.TestCase):
    def test_exact_expectation_matches_enumeration(self):
        rng = np.random.default_rng(3)
        for n in (4, 6, 9):
            ids = rng.permutation(np.arange(10, 10 + n))
            scores = np.round(rng.random(n), 1)
            values = rng.random(n)
            for budget in range(1, n + 1):
                with self.subTest(n=n, budget=budget):
                    outcomes = []
                    for pool in itertools.combinations(range(n), budget):
                        winner = min(pool, key=lambda j: (scores[j], ids[j]))
                        outcomes.append(values[winner])
                    self.assertAlmostEqual(
                        exact_expectation(values, scores, ids, budget),
                        np.mean(outcomes), places=12,
                    )
                    expected = [
                        math.comb(n - r - 1, budget - 1) / math.comb(n, budget)
                        if n - r >= budget else 0.0 for r in range(n)
                    ]
                    np.testing.assert_allclose(rank_weights(n, budget), expected)
                    self.assertAlmostEqual(rank_weights(n, budget).sum(), 1.0)

    def test_aggregation_order_and_zero_penalty(self):
        members = np.array([[0.2, 0.8, 0.5], [0.4, 0.4, 0.4]])
        single, ids = np.array([0.5, 0.4]), np.array([10, 20])
        self.assertEqual(select_song(members, single, ids, 0.5), 10)
        self.assertEqual(
            select_song(members, single, ids, 0.5, "mean_distance"), 20,
        )
        np.testing.assert_array_equal(
            selection_scores(members, single, 0.5, "penalty", 0),
            selection_scores(members, single, 0.5, "mean_distance"),
        )

    def test_penalty_uses_sample_standard_deviation(self):
        score = selection_scores([[0.1, 0.4, 0.9]], [0.5], 0.5, "penalty", 2)
        self.assertAlmostEqual(score[0], 0.3 + 2 * math.sqrt(0.03))

    def test_ties_use_numeric_song_ids(self):
        ids = np.array([90, 2, 17, 4])
        np.testing.assert_array_equal(
            rank_candidates(np.zeros(4), ids), [1, 3, 2, 0],
        )
        self.assertEqual(select_song(np.full((4, 3), 0.5), [0.5] * 4, ids, 0.5), 2)

    def test_oracle_bounds_and_single_candidate_pools(self):
        rng = np.random.default_rng(7)
        y, single = rng.random(8), rng.random(8)
        members, ids = rng.random((8, 3)), np.arange(8)
        policies = ("single", "mean_prediction", "mean_distance", "penalty", "random", "oracle")
        for policy in policies:
            for budget in (1, 4, 8):
                with self.subTest(policy=policy, budget=budget):
                    result = evaluate_catalog(y, members, single, ids, 0.25, budget, policy, 1)
                    self.assertEqual(set(result), {"error", "regret", "oracle_error"})
                    self.assertGreaterEqual(result["regret"], 0.0)
                    self.assertAlmostEqual(result["regret"], result["error"] - result["oracle_error"])
                    if budget == 1 or policy == "random":
                        self.assertAlmostEqual(result["error"], np.abs(y - 0.25).mean())
                    if policy == "oracle":
                        self.assertEqual(result["regret"], 0.0)

    def test_jackknife_removes_whole_groups(self):
        groups = np.array(["a", "a", "b", "c", "c", "c"])
        values, masks = np.array([1., 1., 3., 5., 5., 5.]), []

        def statistic(mask):
            masks.append(mask.copy())
            return values[mask].mean()

        result = paired_artist_jackknife(statistic, groups)
        deleted = np.array([9 / 2, 17 / 5, 5 / 3])
        expected_se = np.sqrt(2 / 3 * np.sum((deleted - deleted.mean()) ** 2))
        self.assertEqual(result["n_groups"], 3)
        self.assertAlmostEqual(result["estimate"], 10 / 3)
        self.assertAlmostEqual(result["se"], expected_se)
        self.assertAlmostEqual(result["ci_low"], 10 / 3 - 1.96 * expected_se)
        self.assertAlmostEqual(result["ci_high"], 10 / 3 + 1.96 * expected_se)
        self.assertEqual(len(masks), 4)
        for mask in masks[1:]:
            self.assertEqual(len(set(groups[~mask])), 1)
            self.assertFalse(set(groups[mask]) & set(groups[~mask]))
        with self.assertRaises(ValueError):
            paired_artist_jackknife(statistic, groups, minimum_catalog=4)

    def test_invalid_inputs(self):
        for budget in (0, 4, 1.5, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                rank_weights(3, budget)
        for scores, ids in (([0.1, 0.2], [1, 1]), ([np.nan], [1]), ([], [])):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                rank_candidates(scores, ids)
        for members in ([[1.1, 0.5]], [[np.nan, 0.5]]):
            with self.subTest(members=members), self.assertRaises(ValueError):
                selection_scores(members, [0.5], 0.5, "mean_prediction")
        with self.assertRaises(ValueError):
            selection_scores([[0.5]], [0.5], 0.5, "penalty", 1)
        with self.assertRaises(ValueError):
            selection_scores([[0.5, 0.5]], [0.5], 0.5, "unknown")
        with self.assertRaises(ValueError):
            exact_expectation([0.1], [0.1, 0.2], [1, 2], 1)
        with self.assertRaises(ValueError):
            paired_artist_jackknife(lambda mask: 0.0, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
