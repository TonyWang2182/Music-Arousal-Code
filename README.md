# TCAS

Core methods for target-conditioned music arousal selection.

`selection.py` implements single-model, mean-prediction, mean-distance and
dispersion-penalized selection. Evaluation uses exact expectations over uniformly
sampled candidate pools, with random and oracle baselines, target error, regret
and artist-group jackknife comparisons.

## Install

Python 3.12 and NumPy.

```bash
python -m pip install -r requirements.txt
```

## Usage

Provide member predictions shaped `(N, K)`, separate single-model predictions
shaped `(N,)`, and unique integer song IDs. Predictions, labels and targets use
the range `[0, 1]`. Equal scores are ordered by song ID.

```python
from selection import select_song, evaluate_catalog

song_id = select_song(members, single, song_ids, target,
                      selector="mean_prediction")
metrics = evaluate_catalog(labels, members, single, song_ids, target,
                           budget=budget, selector="mean_prediction")
```

Selection rules are `single`, `mean_prediction`, `mean_distance` and `penalty`.
The penalty rule accepts `beta`. Evaluation also accepts `random` and `oracle`;
the budget is the number of distinct songs in each candidate pool.

## Tests

```bash
python -m unittest discover -s tests -v
```
