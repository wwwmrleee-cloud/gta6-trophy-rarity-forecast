"""Pinball scorer on the logit scale, following the competition rules.

Loss per item: 2 * max(tau * d, (tau - 1) * d) where
d = logit(true rate) - logit(quantile), rates clamped to 0.01..99.99
before the logit. Also reports typical miss (mean |true - median| in
percentage points) and 90% coverage (share of trophies whose true rate
lies inside [q05, q95]).
"""

import json
import math

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pinball(true_pct, quantile, tau):
    d = logit_of(true_pct) - logit_of(quantile)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(forecast, truth):
    """forecast: {"trophies": [{"id": n, "q": [5 numbers]}]},
    truth: {trophy id: true rate in percent}.
    Returns (loss, typical_miss_pp, coverage_90)."""
    rows = {row["id"]: row["q"] for row in forecast["trophies"]}
    losses = []
    misses = []
    covered = 0
    for tid, y in truth.items():
        q = rows[tid]
        for tau, qv in zip(LEVELS, q):
            losses.append(pinball(y, qv, tau))
        misses.append(abs(y - q[2]))
        if q[0] <= y <= q[4]:
            covered += 1
    n = len(truth)
    return (sum(losses) / len(losses),
            sum(misses) / n,
            covered / n)


def validate_output(forecast, trophy_ids):
    """Strict shape check: exactly one key, one row per trophy, each row
    exactly id + q of five finite non-decreasing numbers in [0, 100]."""
    assert set(forecast.keys()) == {"trophies"}, "top-level keys"
    rows = forecast["trophies"]
    assert len(rows) == len(trophy_ids), "row count"
    seen = set()
    for row in rows:
        assert set(row.keys()) == {"id", "q"}, "row keys"
        assert row["id"] in trophy_ids and row["id"] not in seen, "id once"
        seen.add(row["id"])
        q = row["q"]
        assert len(q) == 5, "five quantiles"
        for v in q:
            assert isinstance(v, (int, float)) and math.isfinite(v), "finite"
            assert 0.0 <= v <= 100.0, "in range"
        assert all(q[i] <= q[i + 1] for i in range(4)), "non-decreasing"
    return True


def load_out(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
