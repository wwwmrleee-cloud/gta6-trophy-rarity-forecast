"""Synthetic final-like case generator, for test 4 of the spec.

Generates a history from 2026-11-18 23:50 UTC to 2026-11-21 11:30 UTC
with reads every 10 minutes, anchored at the real points:
  * 638 players tracked at 23:50 on 18 Nov,
  * 3,868 tracked at 04:30 on 19 Nov,
  * about 5,000 tracked at 09:37 on 19 Nov,
with the tracked count growing along a decelerating curve afterwards.
Rates change only at refresh instants a few hours apart (the first
snapshot keeps the real launch-night rates; later snapshots follow a
log-time curve per trophy, anchored at the real 09:37 morning rates,
rising to a trophy-specific ceiling at the target). Percents are
rounded to two decimals. The target is 2026-11-26 00:00 UTC with a
known truth (the curve's value there, rounded to two decimals).

Trophy ceilings: early story trophies fastest, grind trophies slowest.
"""

import json
import math
from datetime import datetime, timedelta, timezone

# Real 09:37 UTC 19 Nov morning rates, in trophy order (from the spec).
MORNING = [0.10, 100.00, 0.49, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
           0.23, 0.23, 0.23, 0.10, 0.10, 0.13, 0.10, 0.10, 0.13, 1.16,
           0.39, 0.13, 0.13, 0.16, 0.10, 0.10, 0.13, 0.10, 0.13, 0.10,
           0.10, 0.10, 0.18, 0.16, 0.39, 0.13, 0.23, 0.31, 0.18, 0.13,
           0.16, 0.16, 0.13, 0.21, 0.13, 0.16, 0.16, 0.18, 0.16, 0.18,
           0.18, 0.13, 0.18, 0.16, 0.16]

MIDNIGHT = datetime(2026, 11, 19, tzinfo=timezone.utc)
T_MORNING = 9.0 + 37.0 / 60.0          # hours after midnight
T_TARGET = 7 * 24.0                    # 2026-11-26 00:00 UTC
L_MORNING = math.log1p(T_MORNING)
L_TARGET = math.log1p(T_TARGET)


def players_tracked(t_hours):
    """Decelerating growth through the three anchors."""
    if t_hours <= -1.0 / 6.0:
        return 638
    anchors = [(-1.0 / 6.0, 638), (4.5, 3868), (T_MORNING, 5000)]
    if t_hours <= T_MORNING:
        for (t0, p0), (t1, p1) in zip(anchors, anchors[1:]):
            if t0 <= t_hours <= t1:
                f = (math.log1p(max(0.0, t_hours)) - math.log1p(max(0.0, t0))) / \
                    (math.log1p(t1) - math.log1p(max(0.0, t0)))
                return round(p0 + f * (p1 - p0))
    return round(5000 + 3000 * (1.0 - math.exp(-(t_hours - T_MORNING) / 48.0)))


def ceiling_multiple(trophy_id):
    """How many times the morning rate the trophy reaches by the target."""
    if trophy_id == 1:      # platinum: slowest of all
        return 10.0
    if trophy_id in (7, 14):  # gold story trophies
        return 9.0
    if 3 <= trophy_id <= 13:  # early story
        return 8.0
    if trophy_id == 20:     # fishing challenge, already 1.16 in the morning
        return 5.0
    if trophy_id == 21:     # arcade
        return 8.0
    return 4.0              # everything else: grind / side content


def rate_at(trophy_id, t_hours):
    """True rate (percent) at t hours after release midnight."""
    morning = MORNING[trophy_id - 1]
    if morning >= 100.0:
        return 100.0
    mult = ceiling_multiple(trophy_id)
    beta = math.log(mult) / math.log(L_TARGET / L_MORNING)
    return morning * (math.log1p(max(0.0, t_hours)) / L_MORNING) ** beta


def build_case(practice_path, out_path):
    with open(practice_path, encoding="utf-8") as fh:
        practice = json.load(fh)
    trophies = practice["trophies"]
    launch_rates = practice["history"][-1]["rates"]

    start = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
    end = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
    refresh_every = timedelta(hours=4)
    first_refresh = datetime(2026, 11, 19, 4, 30, tzinfo=timezone.utc)

    history = []
    current_rates = [round(float(r), 2) if r is not None else None
                     for r in launch_rates]
    moment = start
    while moment < end:
        t_hours = (moment - MIDNIGHT).total_seconds() / 3600.0
        if moment >= first_refresh and (
                (moment - first_refresh) % refresh_every) == timedelta(0):
            current_rates = [
                round(rate_at(i + 1, t_hours), 2) if r is not None else None
                for i, r in enumerate(launch_rates)]
        players = players_tracked(t_hours)
        earners = sum(round(r / 100.0 * players)
                      for r in current_rates if r is not None)
        history.append({
            "slot": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": moment.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "players_tracked": players,
            "trophies_earned": earners,
            "rates": list(current_rates),
        })
        moment += timedelta(minutes=10)

    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "f00",
        "at": "2026-11-21T11:30:00Z",
        "horizon_minutes": 6510,
        "target": "2026-11-26T00:00:00Z",
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": practice["game"],
        "trophies": trophies,
        "history": history,
        "notes": "Synthetic final-like case built by tests/simulate.py.",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(case, fh)
    truth = {i + 1: round(rate_at(i + 1, T_TARGET), 2)
             for i in range(len(trophies))}
    return case, truth


if __name__ == "__main__":
    import sys
    case, truth = build_case(sys.argv[1], sys.argv[2])
    print(f"reads: {len(case['history'])}, "
          f"first players {case['history'][0]['players_tracked']}, "
          f"last players {case['history'][-1]['players_tracked']}")
    print("truth sample:", {k: truth[k] for k in (1, 2, 3, 7, 20, 55)})
