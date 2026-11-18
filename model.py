"""The forecasting model: a scale-matched pool of past launches' release-day curves.

Idea: a launch day's shape, measured as the log ratio ln(viewers_h / viewers_1) against
hour 1, depends strongly on how big hour 1 already was. A game whose first UTC hour is
small usually has not unlocked for its main audience yet and can grow a hundredfold by
the evening; a game whose first hour is already huge has the world piled in at once and
stays flat or dips. So the forecast for a target pools the release-day log-ratio curves
of past launches, weighting each launch by a Gaussian kernel on the distance between
its log10 hour-1 audience and the target's, multiplied when the launch's unlock
pattern (rolling midnight-local vs one simultaneous global moment, read from the
case's text) matches the target's, and reads the five scored quantiles off the
weighted empirical distribution of those curves, hour by hour.
"""

import math
from datetime import datetime, timezone

# Kernel bandwidth in log10 units of the hour-1 audience: launches whose first hour is
# within about half an order of magnitude of the target's dominate the pool. Chosen by
# leave-one-out backtests over the kit's comparable launches (see README).
BANDWIDTH = 0.45

# Weight multiplier for a pooled launch whose unlock pattern matches the target's
# (rolling midnight-local vs a single simultaneous unlock): the two patterns produce
# different day shapes at the same scale, and matching them sharpened every cut of
# the leave-one-out backtest (0.397 against 0.410 overall; 0.241 against 0.258 on
# launches with hour 1 >= 20,000).
TYPE_BONUS = 3.0

# A comparable must cover at least this many of the release day's 24 hours, and have a
# positive hour 1, to enter the pool: without an hour 1 there is no curve to normalise.
MIN_DAY_HOURS = 20


def parse_ts(text):
    """Parse the case files' ISO 8601 UTC timestamps ('...Z') to aware datetimes."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def release_day_hours(comparable):
    """Map hour-of-release-day (1..24) -> viewers for one comparable launch.

    The comparable's hourly rows run from 14 days before its release day to 3 days
    after; the release day is the 24 whole hours starting at its `day_start`.
    """
    start = parse_ts(comparable["day_start"])
    hours = {}
    for row in comparable["hourly"]:
        offset = (parse_ts(row["from"]) - start).total_seconds()
        hour = int(offset // 3600) + 1
        if 1 <= hour <= 24:
            hours[hour] = row["viewers"]
    return hours


def is_launch(comparable):
    """True for real past launches; the target game's own pre-release category history
    rides along in every case's comparables and is not a launch, so it stays out."""
    if "before release" in comparable["name"]:
        return False
    if comparable["unlock"].startswith("Not a launch"):
        return False
    return True


def fill_gaps(curve):
    """Fill isolated missing hours of a log-ratio curve by linear interpolation
    between the nearest printed neighbours on either side. Only interior gaps are
    filled: the curve is never extrapolated past its first or last printed hour, so
    a launch contributes at an hour only when its own day brackets that hour."""
    if len(curve) < 2:
        return curve
    filled = dict(curve)
    printed = sorted(curve)
    for lo, hi in zip(printed, printed[1:]):
        if hi - lo <= 1:
            continue
        for hour in range(lo + 1, hi):
            frac = (hour - lo) / (hi - lo)
            filled[hour] = curve[lo] + frac * (curve[hi] - curve[lo])
    return filled


def is_rolling(text):
    """True when a free-text unlock description says the launch rolled out at
    midnight local time, territory by territory, rather than at one global moment."""
    lowered = str(text).lower()
    return "midnight local" in lowered or "rolling" in lowered


def target_is_rolling(case):
    """The target's unlock pattern, read from the case's own notes."""
    return is_rolling(case.get("notes", ""))


def build_pool(case):
    """The pool of usable past launches in a case: [{name, first, rolling, curve}]
    where `first` is the launch's hour-1 viewers, `rolling` its unlock pattern, and
    `curve` maps hour -> ln(viewers/ first) for the hours it printed a positive
    count, with isolated gaps interpolated."""
    pool = []
    for comparable in case["comparables"]:
        if not is_launch(comparable):
            continue
        hours = release_day_hours(comparable)
        first = hours.get(1, 0)
        if first <= 0 or len(hours) < MIN_DAY_HOURS:
            continue
        curve = {h: math.log(v / first) for h, v in hours.items() if v > 0}
        pool.append(
            {
                "name": comparable["name"],
                "first": first,
                "rolling": is_rolling(comparable["unlock"]),
                "curve": fill_gaps(curve),
            }
        )
    return pool


def kernel_weights(pool, target_first, target_rolling):
    """One weight per pooled launch: a Gaussian kernel on log10 hour-1 distance,
    multiplied by TYPE_BONUS when the launch's unlock pattern matches the target's."""
    centre = math.log10(max(1, target_first))
    weights = []
    for launch in pool:
        distance = (math.log10(launch["first"]) - centre) / BANDWIDTH
        weight = math.exp(-0.5 * distance * distance)
        if launch["rolling"] == target_rolling:
            weight *= TYPE_BONUS
        weights.append(weight)
    return weights


def weighted_quantile(pairs, tau):
    """The tau-quantile of (value, weight) pairs, linearly interpolated across the
    sorted values by cumulative weight. Deterministic: the sort is total and the
    arithmetic is plain floats."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    goal = tau * total
    acc = 0.0
    prev_value, prev_acc = ordered[0][0], 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= goal:
            if acc == prev_acc:
                return value
            frac = (goal - prev_acc) / (acc - prev_acc)
            return prev_value + frac * (value - prev_value)
        prev_value, prev_acc = value, acc
    return ordered[-1][0]


def forecast_curves(pool, target_first, target_rolling, levels):
    """{hour: [log-ratio quantile per level]} for hours 2..24, from the weighted pool.

    Hours no pooled launch printed fall back to a flat 0.0 log ratio at every level;
    the caller replaces the whole forecast with persistence if the pool is empty.
    """
    weights = kernel_weights(pool, target_first, target_rolling)
    curves = {}
    for hour in range(2, 25):
        pairs = [
            (launch["curve"][hour], w)
            for launch, w in zip(pool, weights)
            if hour in launch["curve"]
        ]
        if not pairs:
            curves[hour] = [0.0] * len(levels)
            continue
        curves[hour] = [weighted_quantile(pairs, tau) for tau in levels]
    return curves
