"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Reads every <id>.in case in the input directory (MAKEFUN_INPUT_DIR, else
/input) and writes one <id>.out forecast per case to the output directory
(MAKEFUN_OUTPUT_DIR, else /output). Each forecast gives, per trophy, five
quantiles (5, 25, 50, 75, 95 percent levels) of its earn rate in percent.

Method, per trophy, in words:
  * Collapse consecutive reads whose rates vector and players-tracked count
    are identical into one snapshot kept at the first read's time, because
    Exophase only changes at its own refreshes and the plateaus between them
    carry no new information.
  * Measure time as ln(1 + hours since release midnight, 2026-11-19 00:00
    UTC, taken from the case's release date). Reads before midnight count
    as zero hours.
  * Over the last 24 hours of snapshots, fit two weighted straight lines
    (weights halving every 12 hours of age): one of the logit's rate
    against the time variable, and one of log earners and one of log
    players tracked against the same variable.
  * The median is the average, in logit space, of the direct fit read at
    the target and the ratio forecast (earners fit divided by players fit,
    turned back into a percent). Each candidate's move away from the last
    observed logit is shrunk by n / (n + 4), where n is the number of
    distinct snapshots in the window, and the averaged move is capped at
    2.0 logit units.
  * The spread is the square root of three squared terms: a random-walk
    term (0.35 per square-root day of horizon), the direct fit's residual
    spread (floored at 0.30), and a jump term (half the largest absolute
    logit change between consecutive distinct snapshots, capped at 0.5:
    the launch-night dilution drop was a one-off, not the size of an
    ordinary refresh jump, so the cap keeps it from dominating).
    The total is floored at 0.60 and capped at 2.0. A trophy whose
    residual rests on fewer than five distinct snapshots has its residual
    term blended fifty-fifty with the median residual term across trophies.
  * Quantiles come from the median logit plus the standard normal
    quantiles (-1.645, -0.674, 0, 0.674, 1.645) times sigma, passed
    through the sigmoid and scaled to percent, rounded to four decimals
    half-up.

Edge cases: a trophy last at 100% is forecast at 99.99% with sigma 0.5;
a 0.00% rate is treated as 0.01% for the logit; a trophy with no usable
rate anywhere gets median 1.0% and sigma 1.5; with no history at all every
trophy gets median 1.0% and sigma 1.5 and a valid file is still written.

Standard library only, no network, no randomness, no wall-clock reads.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Named model constants (the only knobs the calibration may adjust).
FIT_WINDOW_HOURS = 24.0
WEIGHT_HALF_LIFE_HOURS = 12.0
SHRINK_PLUS = 4.0
MAX_LOGIT_MOVE = 2.0
RW_COEF = 0.35
RESID_FLOOR = 0.30
RESID_FLOOR_FEW = 0.50
JUMP_CAP = 0.5
SIGMA_FLOOR = 0.60
SIGMA_CAP = 2.0
MIN_SNAPSHOTS_FOR_FIT = 3
MIN_SNAPSHOTS_FOR_FULL_RESID = 5


def parse_time(text):
    """Parse an ISO 8601 timestamp (with Z or an offset) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid_of(logit_value):
    if logit_value >= 0:
        return 1.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def inv_normal(p):
    """Acklam's approximation of the standard normal quantile."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def z_for_level(level):
    if level in LEVELS:
        return Z[LEVELS.index(level)]
    return inv_normal(level)


def weighted_line(xs, ys, ws):
    """Weighted least squares for y = a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return (sum(ys) / len(ys) if ys else 0.0, 0.0)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def collapse_snapshots(history):
    """Collapse consecutive identical reads into snapshots at first time."""
    snaps = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, read))
    return [read for _, read in snaps]


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes") or 0
    horizon_days = horizon_minutes / 1440.0
    levels = case.get("levels") or LEVELS
    zs = [z_for_level(lv) for lv in levels]

    release = case.get("game") or {}
    release_date = release.get("release_date") or "2026-11-19"
    try:
        midnight = datetime.fromisoformat(release_date + "T00:00:00+00:00")
    except ValueError:
        midnight = datetime(2026, 11, 19, tzinfo=timezone.utc)

    def time_var(moment):
        hours = (moment - midnight).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    target_text = case.get("target")
    if target_text:
        target_x = time_var(parse_time(target_text))
    else:
        target_x = time_var(parse_time(case["at"])) + math.log1p(
            max(0.0, horizon_minutes / 60.0))

    if not history:
        rows = []
        for trophy in trophies:
            med = logit_of(1.0)
            rows.append({"id": trophy["id"], "q": quantiles(med, 1.5, zs)})
        return {"trophies": rows}

    snaps = collapse_snapshots(history)
    snap_times = [parse_time(r["at"]) for r in snaps]
    last_time = snap_times[-1]
    window_start = last_time.timestamp() - FIT_WINDOW_HOURS * 3600.0
    win_idx = [i for i, t in enumerate(snap_times)
               if t.timestamp() >= window_start]
    if not win_idx:
        win_idx = [len(snaps) - 1]
    win_snaps = [snaps[i] for i in win_idx]
    win_times = [snap_times[i] for i in win_idx]
    n_win = len(win_snaps)

    # Last non-missing rate per trophy (from the newest read backwards).
    last_rates = []
    for j in range(len(trophies)):
        rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if j < len(rates) and rates[j] is not None:
                rate = rates[j]
                break
        last_rates.append(rate)

    # Weights: halve every 12 hours of age from the newest window snapshot.
    newest_ts = win_times[-1].timestamp()
    weights = [0.5 ** max(0.0, (newest_ts - t.timestamp()) / 3600.0
                          / WEIGHT_HALF_LIFE_HOURS) for t in win_times]
    xs = [time_var(t) for t in win_times]

    # Log players-tracked fit over the window (shared across trophies).
    log_players = []
    for read in win_snaps:
        players = read.get("players_tracked") or 0
        log_players.append(math.log(max(1.0, float(players))))
    players_fit = weighted_line(xs, log_players, weights)

    # First pass per trophy: direct fit, earners fit, residual, jump.
    per_trophy = []
    resid_terms = []
    for j, trophy in enumerate(trophies):
        last_rate = last_rates[j]
        if last_rate is None:
            per_trophy.append(None)
            continue
        if last_rate >= 100.0:
            per_trophy.append(("top", None))
            continue
        last_logit = logit_of(last_rate)
        # Direct logit fit over snapshots where this trophy has a rate.
        dx, dy, dw = [], [], []
        ex, ey, ew = [], [], []
        for k, read in enumerate(win_snaps):
            rates = read.get("rates") or []
            rate = rates[j] if j < len(rates) else None
            if rate is None:
                continue
            players = read.get("players_tracked") or 0
            dx.append(xs[k])
            dy.append(logit_of(rate))
            dw.append(weights[k])
            earners = clamp_pct(rate) / 100.0 * max(1.0, float(players))
            ex.append(xs[k])
            ey.append(math.log(max(1e-9, earners)))
            ew.append(weights[k])
        if len(set(dx)) >= MIN_SNAPSHOTS_FOR_FIT and len(dx) >= MIN_SNAPSHOTS_FOR_FIT:
            a_dir, b_dir = weighted_line(dx, dy, dw)
            a_earn, b_earn = weighted_line(ex, ey, ew)
            resid = [y - (a_dir + b_dir * x) for x, y in zip(dx, dy)]
            dof = max(1, len(resid) - 2)
            resid_sd = math.sqrt(sum(r * r for r in resid) / dof)
        else:
            a_dir = b_dir = None
            a_earn = b_earn = None
            resid_sd = RESID_FLOOR_FEW
        # Jump: largest absolute logit change between consecutive snapshots.
        jump = 0.0
        prev = None
        for read in snaps:
            rates = read.get("rates") or []
            rate = rates[j] if j < len(rates) else None
            if rate is None:
                continue
            lg = logit_of(rate)
            if prev is not None:
                jump = max(jump, abs(lg - prev))
            prev = lg
        jump_term = min(JUMP_CAP, 0.5 * jump)
        resid_term = max(RESID_FLOOR, resid_sd)
        if a_dir is None:
            resid_term = RESID_FLOOR_FEW
        resid_terms.append(resid_term)
        per_trophy.append((last_logit, a_dir, b_dir, a_earn, b_earn,
                           resid_term, jump_term, len(set(dx))))

    median_resid = 0.0
    if resid_terms:
        ordered = sorted(resid_terms)
        mid = len(ordered) // 2
        median_resid = (ordered[mid] if len(ordered) % 2 == 1
                        else 0.5 * (ordered[mid - 1] + ordered[mid]))

    rw_term = RW_COEF * math.sqrt(max(0.0, horizon_days))

    rows = []
    for j, trophy in enumerate(trophies):
        entry = per_trophy[j]
        if entry is None:
            rows.append({"id": trophy["id"],
                         "q": quantiles(logit_of(1.0), 1.5, zs)})
            continue
        if entry[0] == "top":
            rows.append({"id": trophy["id"],
                         "q": quantiles(logit_of(99.99), 0.5, zs)})
            continue
        (last_logit, a_dir, b_dir, a_earn, b_earn,
         resid_term, jump_term, n_fit) = entry
        if a_dir is None:
            med_logit = last_logit
            resid_use = RESID_FLOOR_FEW
        else:
            shrink = n_win / (n_win + SHRINK_PLUS)
            cand1 = last_logit + (a_dir + b_dir * target_x - last_logit) * shrink
            earners_t = math.exp(a_earn + b_earn * target_x)
            players_t = math.exp(players_fit[0] + players_fit[1] * target_x)
            ratio_pct = 100.0 * earners_t / max(1.0, players_t)
            cand2 = last_logit + (logit_of(ratio_pct) - last_logit) * shrink
            med_logit = 0.5 * (cand1 + cand2)
            move = med_logit - last_logit
            if move > MAX_LOGIT_MOVE:
                med_logit = last_logit + MAX_LOGIT_MOVE
            elif move < -MAX_LOGIT_MOVE:
                med_logit = last_logit - MAX_LOGIT_MOVE
            resid_use = resid_term
            if n_fit < MIN_SNAPSHOTS_FOR_FULL_RESID:
                resid_use = 0.5 * resid_term + 0.5 * median_resid
        sigma = math.sqrt(rw_term ** 2 + resid_use ** 2 + jump_term ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        rows.append({"id": trophy["id"],
                     "q": quantiles(med_logit, sigma, zs)})
    return {"trophies": rows}


def quantiles(med_logit, sigma, zs):
    out = []
    prev = None
    for z in zs:
        value = round4(100.0 * sigmoid_of(med_logit + z * sigma))
        value = min(100.0, max(0.0, value))
        if prev is not None and value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        data = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(data.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name} forecast exceeds 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(data, encoding="utf-8")


if __name__ == "__main__":
    main()
