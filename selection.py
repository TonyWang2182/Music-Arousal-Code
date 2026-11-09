"""Target-conditioned selection and exact candidate-pool evaluation."""

from numbers import Integral

import numpy as np


def rank_weights(n, budget):
    """Selection probabilities for a uniform pool of distinct candidates."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (n, budget)):
        raise ValueError("Catalog size and budget must be integers")
    if not 1 <= budget <= n:
        raise ValueError("Require 1 <= budget <= catalog size")
    weights = np.zeros(n, dtype=np.float64)
    weights[0] = budget / n
    for r in range(1, n - budget + 1):
        weights[r] = weights[r - 1] * (n - r - budget + 1) / (n - r)
    return weights


def selection_scores(members, single, target, selector, beta=0.0):
    """Score candidates from member predictions (N, K) and a single fit (N,)."""
    members = np.asarray(members, dtype=np.float64)
    single = np.asarray(single, dtype=np.float64)
    if members.ndim != 2 or members.shape[1] < 1 or single.shape != (len(members),):
        raise ValueError("Expected members[N, K] and single[N]")
    if not np.isfinite(members).all() or not np.isfinite(single).all():
        raise ValueError("Predictions must be finite")
    if np.any((members < 0) | (members > 1)) or np.any((single < 0) | (single > 1)):
        raise ValueError("Predictions must lie in [0, 1]")
    if not 0 <= target <= 1 or not np.isfinite(beta) or beta < 0:
        raise ValueError("Invalid target or penalty weight")
    if selector == "single":
        return np.abs(single - target)
    if selector == "mean_prediction":
        return np.abs(members.mean(axis=1) - target)
    distance = np.abs(members - target)
    if selector == "mean_distance":
        return distance.mean(axis=1)
    if selector == "penalty":
        if members.shape[1] < 2:
            raise ValueError("Sample standard deviation requires at least two members")
        return distance.mean(axis=1) + beta * distance.std(axis=1, ddof=1)
    raise ValueError(f"Unknown selection rule: {selector}")


def rank_candidates(scores, song_ids):
    """Rank by ascending score, breaking ties by integer song ID."""
    scores, song_ids = np.asarray(scores), np.asarray(song_ids)
    if scores.ndim != 1 or scores.shape != song_ids.shape or not len(scores):
        raise ValueError("Expected nonempty score and ID vectors of equal length")
    if not np.issubdtype(song_ids.dtype, np.integer):
        raise ValueError("Song IDs must be integers")
    if len(np.unique(song_ids)) != len(song_ids) or not np.isfinite(scores).all():
        raise ValueError("Song IDs must be unique and scores finite")
    return np.lexsort((song_ids, scores))


def select_song(members, single, song_ids, target, selector="mean_prediction", beta=0.0):
    """Return the selected song ID from the supplied candidates."""
    scores = selection_scores(members, single, target, selector, beta)
    order = rank_candidates(scores, song_ids)
    return int(np.asarray(song_ids)[order[0]])


def exact_expectation(values, scores, song_ids, budget):
    """Expected selected value over uniform pools, without replacement."""
    values = np.asarray(values, dtype=np.float64)
    order = rank_candidates(scores, song_ids)
    if values.shape != order.shape or not np.isfinite(values).all():
        raise ValueError("Expected one finite value per candidate")
    return float(rank_weights(len(order), budget) @ values[order])


def evaluate_catalog(y, members, single, song_ids, target, budget, selector, beta=0.0):
    """Return expected target error, oracle error and regret for a catalog."""
    y = np.asarray(y, dtype=np.float64)
    members = np.asarray(members, dtype=np.float64)
    scores = selection_scores(
        members, single, target,
        "mean_prediction" if selector in {"random", "oracle"} else selector,
        beta,
    )
    if y.shape != scores.shape or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Expected one label in [0, 1] per candidate")
    error = np.abs(y - target)
    oracle = exact_expectation(error, error, song_ids, budget)
    if selector == "random":
        measured = float(error.mean())
    elif selector == "oracle":
        measured = oracle
    else:
        measured = exact_expectation(error, scores, song_ids, budget)
    return {"error": measured, "regret": max(0.0, measured - oracle), "oracle_error": oracle}


def paired_artist_jackknife(statistic, groups, minimum_catalog=1):
    """Approximate 95% intervals using leave-one-artist-group-out contrasts.

    statistic(mask) recomputes a paired difference with fitted predictors fixed.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("Expected a nonempty artist-group vector")
    if isinstance(minimum_catalog, bool) or not isinstance(minimum_catalog, Integral) or minimum_catalog < 1:
        raise ValueError("Minimum catalog size must be a positive integer")
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three artist groups are required")
    point = float(statistic(np.ones(len(groups), dtype=bool)))
    deleted = []
    for group in unique:
        mask = groups != group
        if mask.sum() < minimum_catalog:
            raise ValueError("Deleting an artist group leaves too few candidates")
        deleted.append(float(statistic(mask)))
    deleted = np.asarray(deleted)
    if not np.isfinite(point) or not np.isfinite(deleted).all():
        raise ValueError("Contrasts must be finite")
    se = float(np.sqrt((len(unique) - 1) / len(unique) * np.sum((deleted - deleted.mean()) ** 2)))
    return {"estimate": point, "se": se, "ci_low": point - 1.96 * se,
            "ci_high": point + 1.96 * se, "n_groups": len(unique)}
