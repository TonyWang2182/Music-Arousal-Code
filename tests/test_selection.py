import itertools
import math
import numpy as np
import pytest

from arousal.selection import (rank_weights, rank_candidates, select_song, selection_scores,
                               exact_expectation, evaluate_catalog, paired_artist_jackknife)


def test_exact_expectation_matches_every_distinct_candidate_set():
    rng = np.random.default_rng(3)
    for n in (4, 6, 9):
        ids = rng.permutation(np.arange(10, 10+n))
        scores = np.round(rng.random(n), 1)  # deliberate score ties
        outcomes = rng.random(n)
        for budget in range(1, n+1):
            exhaustive = []
            for subset in itertools.combinations(range(n), budget):
                winner = min(subset, key=lambda j: (scores[j], ids[j]))
                exhaustive.append(outcomes[winner])
            assert exact_expectation(outcomes, scores, ids, budget) == pytest.approx(np.mean(exhaustive), abs=1e-12)
            assert rank_weights(n, budget).sum() == pytest.approx(1.0)
            expected = [math.comb(n-r-1,budget-1)/math.comb(n,budget) if n-r>=budget else 0 for r in range(n)]
            assert np.allclose(rank_weights(n,budget), expected)


def test_aggregation_order_is_a_real_intervention():
    members = np.array([[0.2,0.8,0.5], [0.4,0.4,0.4]])
    single = np.array([0.5,0.4])
    ids = np.array([10,20])
    assert select_song(members,single,ids,0.5,"mean_prediction") == 10
    assert select_song(members,single,ids,0.5,"mean_distance") == 20
    assert np.array_equal(selection_scores(members,single,0.5,"penalty",0),
                          selection_scores(members,single,0.5,"mean_distance"))


def test_oracle_bounds_and_budget_one():
    rng = np.random.default_rng(7)
    y, single = rng.random(8), rng.random(8)
    members = rng.random((8,3))
    ids = np.arange(8)
    for policy in ("random","single","mean_prediction","mean_distance","penalty","oracle"):
        b1 = evaluate_catalog(y,members,single,ids,0.25,1,policy,1)
        assert b1["error"] == pytest.approx(np.abs(y-0.25).mean())
        for budget in (4,8):
            result = evaluate_catalog(y,members,single,ids,0.25,budget,policy,1)
            assert result["regret"] >= -1e-12
            if policy == "oracle":
                assert result["regret"] == 0


def test_test_labels_do_not_change_selection_or_tie_break():
    members = np.full((4,3), 0.5)
    single, ids = np.full(4,0.5), np.array([9,2,7,4])
    winner = select_song(members,single,ids,0.5)
    assert winner == 2
    for y in (np.array([0.,0.,0.,0.]),np.array([1.,0.5,1.,1.])):
        evaluate_catalog(y,members,single,ids,0.5,4,"mean_prediction")
        assert select_song(members,single,ids,0.5) == winner


def test_artist_jackknife_deletes_whole_groups_and_self_contrast_is_zero():
    groups = np.repeat(["a","b","c","d"], [1,2,3,4])
    masks = []
    def statistic(mask):
        masks.append(mask.copy())
        return 0.0
    result = paired_artist_jackknife(statistic,groups)
    assert result["estimate"] == result["se"] == result["ci_low"] == result["ci_high"] == 0
    assert len(masks) == 5
    for mask in masks[1:]:
        assert len(set(groups[~mask])) == 1
        assert not set(groups[mask]) & set(groups[~mask])
    with pytest.raises(ValueError, match="fewer"):
        paired_artist_jackknife(statistic,groups,minimum_catalog=8)


@pytest.mark.parametrize("budget", [0, 4, 1.5, True])
def test_invalid_budgets_fail(budget):
    with pytest.raises(ValueError):
        rank_weights(3,budget)


def test_duplicate_candidate_ids_fail():
    with pytest.raises(ValueError,match="unique"):
        rank_candidates([0.1,0.2],[1,1])
