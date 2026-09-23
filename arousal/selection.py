"""Label-free selectors and exact finite-catalog evaluation.

Ranks are one-based in the mathematical description and zero-based in arrays.
Only evaluation functions receive human labels. A selector receives predictions,
target and song IDs, never human labels or their standard deviations.
"""
from __future__ import annotations

import numpy as np


def rank_weights(n, budget):
    if isinstance(budget, bool) or int(budget) != budget or not 1 <= budget <= n:
        raise ValueError(f"Require integer 1 <= budget <= catalog size; got B={budget}, N={n}")
    budget = int(budget)
    weights = np.zeros(n, dtype=np.float64)
    weights[0] = budget / n
    for r in range(1, n - budget + 1):
        weights[r] = weights[r - 1] * (n - r - budget + 1) / (n - r)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ArithmeticError("Rank probabilities do not sum to one")
    return weights


def selection_scores(members, single, target, selector, beta=0.0):
    members = np.asarray(members, dtype=np.float64)
    single = np.asarray(single, dtype=np.float64)
    if members.ndim != 2 or single.shape != (len(members),) or members.shape[1] < 1:
        raise ValueError("Expected members[N,K] and single[N]")
    if not np.isfinite(members).all() or not np.isfinite(single).all():
        raise ValueError("Non-finite predictions")
    if np.any((members < 0) | (members > 1)) or np.any((single < 0) | (single > 1)):
        raise ValueError("Predictions must already be clipped to [0,1]")
    if not 0 <= target <= 1 or not np.isfinite(beta) or beta < 0:
        raise ValueError("Invalid target or penalty")
    if selector == "single":
        return np.abs(single - target)
    if selector == "mean_prediction":
        return np.abs(members.mean(axis=1) - target)
    distance = np.abs(members - target)
    if selector == "mean_distance":
        return distance.mean(axis=1)
    if selector == "penalty":
        if members.shape[1] < 2:
            raise ValueError("Sample disagreement requires at least two members")
        return distance.mean(axis=1) + beta * distance.std(axis=1, ddof=1)
    raise ValueError(f"Unknown label-free selector: {selector}")


def rank_candidates(scores, song_ids):
    scores, song_ids = np.asarray(scores), np.asarray(song_ids)
    if scores.shape != song_ids.shape or scores.ndim != 1 or not len(scores):
        raise ValueError("Nonempty score and ID vectors required")
    if len(np.unique(song_ids)) != len(song_ids) or not np.isfinite(scores).all():
        raise ValueError("Song IDs must be unique and scores finite")
    return np.lexsort((song_ids, scores))


def select_song(members, single, song_ids, target, selector="mean_prediction", beta=0.0):
    order = rank_candidates(selection_scores(members, single, target, selector, beta), song_ids)
    return int(np.asarray(song_ids)[order[0]])


def exact_expectation(values, scores, song_ids, budget):
    values = np.asarray(values, dtype=np.float64)
    order = rank_candidates(scores, song_ids)
    if values.shape != order.shape or not np.isfinite(values).all():
        raise ValueError("Finite per-song evaluation values required")
    return float(rank_weights(len(order), budget) @ values[order])


def evaluate_catalog(y, members, single, song_ids, target, budget, selector, beta=0.0):
    y, members = np.asarray(y, dtype=np.float64), np.asarray(members, dtype=np.float64)
    if y.shape != (len(members),) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Invalid human labels")
    # Even random/oracle cases validate budget and unique IDs.
    weights = rank_weights(len(y), budget)
    rank_candidates(np.zeros(len(y)), song_ids)
    error = np.abs(y - target)
    oracle = exact_expectation(error, error, song_ids, budget)
    if selector == "random":
        measured = float(error.mean())
        predicted = float(np.abs(members.mean(axis=1) - target).mean())
    elif selector == "oracle":
        measured, predicted = oracle, None
    else:
        score = selection_scores(members, single, target, selector, beta)
        order = rank_candidates(score, song_ids)
        measured = float(weights @ error[order])
        # The diagnostic excludes the beta penalty itself.
        proxy = (np.abs(np.asarray(single) - target) if selector == "single" else
                 np.abs(members.mean(axis=1) - target) if selector == "mean_prediction" else
                 np.abs(members - target).mean(axis=1))
        predicted = float(weights @ proxy[order])
    regret = measured - oracle
    if regret < -1e-10:
        raise ArithmeticError("Oracle cannot have worse expected target error")
    return {"error": measured, "regret": max(0.0, regret),
            "optimism_gap": None if predicted is None else measured - predicted,
            "oracle_error": oracle}


def paired_artist_jackknife(statistic, groups, minimum_catalog=1):
    """Pointwise normal-approximation CI conditional on fixed fitted models.

    statistic(mask) evaluates the same paired contrast on the retained catalog.
    Groups, not pool rows or training seeds, are the sampling units.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three independent artist/duplicate groups required")
    point = float(statistic(np.ones(len(groups), dtype=bool)))
    deleted = []
    for group in unique:
        mask = groups != group
        if mask.sum() < minimum_catalog:
            raise ValueError(f"Deleting group {group!r} leaves fewer than B={minimum_catalog} distinct songs")
        deleted.append(float(statistic(mask)))
    deleted = np.asarray(deleted)
    se = float(np.sqrt((len(unique) - 1) / len(unique) * np.sum((deleted - deleted.mean()) ** 2)))
    return {"estimate": point, "se": se, "ci_low": point - 1.96 * se,
            "ci_high": point + 1.96 * se, "n_groups": len(unique),
            "interval": "pointwise_approximate_artist_jackknife_fixed_models"}
