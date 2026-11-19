"""Trend-only baseline forecaster, for comparison in tests only.

Per trophy: a straight weighted least-squares line of the logit rate
against ln(1 + hours since release midnight) over the last 24 hours of
distinct snapshots, read at the target; spread is the random-walk term
(0.5 per square-root day) combined with the fit residual. Falls back to
persistence when there are fewer than three distinct snapshots. This is
a deliberately plain variant, not the entry.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import (INPUT_DIR, OUTPUT_DIR, LEVELS, Z, clamp_pct, logit_of,
                  round4, sigmoid_of, weighted_line, collapse_snapshots,
                  parse_time, quantiles)


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
    release = (case.get("game") or {}).get("release_date") or "2026-11-19"
    midnight = datetime.fromisoformat(release + "T00:00:00+00:00")

    def time_var(moment):
        hours = (moment - midnight).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    target_x = time_var(parse_time(case["target"])) if case.get("target") \
        else time_var(parse_time(case["at"]))
    rows = []
    if not history:
        for trophy in trophies:
            rows.append({"id": trophy["id"],
                         "q": quantiles(logit_of(1.0), 1.5,
                                        [Z[LEVELS.index(l)] for l in LEVELS])})
        return {"trophies": rows}

    snaps = collapse_snapshots(history)
    times = [parse_time(r["at"]) for r in snaps]
    last_ts = times[-1].timestamp()
    keep = [i for i, t in enumerate(times)
            if t.timestamp() >= last_ts - 24.0 * 3600.0]
    xs = [time_var(times[i]) for i in keep]
    zs = [Z[LEVELS.index(l)] for l in (case.get("levels") or LEVELS)]
    rw = 0.5 * math.sqrt(max(0.0, horizon_days))
    for j, trophy in enumerate(trophies):
        last_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if j < len(rates) and rates[j] is not None:
                last_rate = rates[j]
                break
        if last_rate is None:
            rows.append({"id": trophy["id"],
                         "q": quantiles(logit_of(1.0), 1.5, zs)})
            continue
        if last_rate >= 100.0:
            rows.append({"id": trophy["id"],
                         "q": quantiles(logit_of(99.99), 0.5, zs)})
            continue
        last_logit = logit_of(last_rate)
        dx, dy = [], []
        for i in keep:
            rates = snaps[i].get("rates") or []
            rate = rates[j] if j < len(rates) else None
            if rate is None:
                continue
            dx.append(xs[keep.index(i)] if False else time_var(times[i]))
            dy.append(logit_of(rate))
        if len(set(dx)) >= 3:
            a, b = weighted_line(dx, dy, [1.0] * len(dx))
            med = a + b * target_x
            resid = [y - (a + b * x) for x, y in zip(dx, dy)]
            sd = math.sqrt(sum(r * r for r in resid) / max(1, len(resid) - 2))
            sigma = math.sqrt(rw ** 2 + max(0.30, sd) ** 2)
        else:
            med = last_logit
            sigma = math.sqrt(rw ** 2 + 0.50 ** 2)
        sigma = min(2.0, max(0.60, sigma))
        rows.append({"id": trophy["id"], "q": quantiles(med, sigma, zs)})
    return {"trophies": rows}


def main():
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        (out_dir / f"{path.stem}.out").write_text(
            json.dumps(forecast_case(case), separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
