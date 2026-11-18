"""The forecasting model: a scale-matched pool of past launches' release-day curves.

Idea: a launch day's shape, measured as the log ratio ln(viewers_h / viewers_1) against
hour 1, depends strongly on how big hour 1 already was. A game whose first UTC hour is
small usually has not unlocked for its main audience yet and can grow a hundredfold by
the evening; a game whose first hour is already huge has the world piled in at once and
stays flat or dips. So the forecast for a target pools the release-day log-ratio curves
of past launches, weighting each launch by a Gaussian kernel on the distance between
its log10 hour-1 audience and the target's, multiplied when the launch's unlock pattern
(rolling midnight-local vs one simultaneous global moment) and its early-access shape
match the target's, and fits, hour by hour, a local line in log10 hour-1 through the
pooled curves: the line carries the pool's centre to the target's own scale instead of
averaging launches of different scales flat, and the five scored quantiles are read off
the weighted distribution of the line's residuals. The kernel narrows as the day goes
on (wide while the day's shape is still up for grabs, tight once scale-matched
launches track the decay closely), and the fitted quantile curves are smoothed
across neighbouring hours before they are read.
"""

import math
import re
from datetime import datetime, timezone

# Kernel bandwidth in log10 units of the hour-1 audience, at the day's middle
# (hour 13): launches whose first hour is within about two-thirds of an order of
# magnitude of the target's dominate the pool there. The bandwidth ramps linearly
# with the hour (BW_SLOPE per hour from the middle): wide in the first hours,
# when a launch's shape still depends on unlock timing more than on scale, and
# narrow late, when the closest scale matches track the decay best. Both numbers
# were chosen by leave-one-out backtests over the kit's comparable launches (see
# README): the ramp alone took the loss from 0.397 to 0.389.
BW_CENTRE = 0.65
BW_SLOPE = -0.5

# Weight multiplier for a pooled launch whose unlock pattern matches the target's
# (rolling midnight-local vs a single simultaneous unlock): the two patterns produce
# different day shapes at the same scale, and matching them sharpened every cut of
# the leave-one-out backtest.
TYPE_BONUS = 3.0

# Weight multiplier for a pooled launch whose early-access shape matches the
# target's: a launch that sold (or beta-tested) its first days before the release
# day starts warm and stays flat, where a cold launch explodes, and the match is
# worth about as much as the unlock-pattern match in the backtests.
EARLY_BONUS = 3.0

# The local line's slope is shrunk toward flat by this factor and capped at
# SLOPE_CAP log-ratio units per log10 of hour-1 audience: with three dozen
# launches in the pool the raw slope is real but noisy, and the shrinkage is
# what the backtests settled on.
SLOPE_SHRINK = 0.8
SLOPE_CAP = 1.5

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


def has_early_access(text):
    """True when an unlock description reports an early-access period before the
    release day (paid head starts, premium editions, network tests): the launch
    reaches its release day already warm. A bare 'no early access' does not
    count, and neither do the words on their own in a denial."""
    lowered = str(text).lower()
    return any(
        match.group(0) != "no early access"
        for match in re.finditer(r"(no )?early access", lowered)
    )


def target_has_early_access(case):
    """The target's early-access shape, read from the case's own notes (which
    state it the way the comparables' unlock texts do)."""
    return has_early_access(case.get("notes", ""))


def build_pool(case):
    """The pool of usable past launches in a case: [{name, first, rolling, early,
    curve}] where `first` is the launch's hour-1 viewers, `rolling` its unlock
    pattern, `early` whether it had an early-access period, and `curve` maps
    hour -> ln(viewers / first) for the hours it printed a positive count, with
    isolated gaps interpolated."""
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
                "early": has_early_access(comparable["unlock"]),
                "curve": fill_gaps(curve),
            }
        )
    return pool


def bandwidth_at(hour):
    """The kernel bandwidth for one hour of the day: BW_CENTRE at hour 13,
    ramping by BW_SLOPE of itself across the day's span, so the first hours
    pool widely (0.975) and the last hours pool tightly (0.325)."""
    return BW_CENTRE * (1.0 + BW_SLOPE * (hour - 13) / 11.0)


def kernel_weights(pool, target_first, target_rolling, target_early, hour):
    """One weight per pooled launch, for one hour: a Gaussian kernel on log10
    hour-1 distance at that hour's bandwidth, multiplied by TYPE_BONUS when
    the launch's unlock pattern matches the target's and by EARLY_BONUS when
    its early-access shape does."""
    centre = math.log10(max(1, target_first))
    bandwidth = bandwidth_at(hour)
    weights = []
    for launch in pool:
        distance = (math.log10(launch["first"]) - centre) / bandwidth
        weight = math.exp(-0.5 * distance * distance)
        if launch["rolling"] == target_rolling:
            weight *= TYPE_BONUS
        if launch["early"] == target_early:
            weight *= EARLY_BONUS
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


def local_quantiles(rows, centre, levels):
    """The scored quantiles of one hour's log ratio at the target's scale.

    `rows` holds (log10 hour-1, log ratio, weight) for each pooled launch that
    printed this hour. A weighted line is fitted through the rows: its slope
    says how the day's shape tilts with scale, and it is shrunk by
    SLOPE_SHRINK and capped at SLOPE_CAP because three dozen launches make a
    noisy line. The quantiles are the weighted quantiles of the rows'
    residuals around the line, carried to the target's own log10 hour-1
    (`centre`), so the spread comes from how launches scattered around their
    shared tilt, not from the tilt itself.
    """
    total = sum(w for _, _, w in rows)
    x_mean = sum(w * x for x, _, w in rows) / total
    y_mean = sum(w * y for _, y, w in rows) / total
    spread = sum(w * (x - x_mean) ** 2 for x, _, w in rows)
    if spread > 1e-9:
        slope = (
            sum(w * (x - x_mean) * (y - y_mean) for x, y, w in rows) / spread
        )
    else:
        slope = 0.0
    slope = max(-SLOPE_CAP, min(SLOPE_CAP, SLOPE_SHRINK * slope))
    base = y_mean + slope * (centre - x_mean)
    residuals = [
        (y - (y_mean + slope * (x - x_mean)), w) for x, y, w in rows
    ]
    return [base + weighted_quantile(residuals, tau) for tau in levels]


def smooth_curves(curves):
    """Smooth each level's log-ratio curve across neighbouring hours with
    triangular weights (1, 2, 3, 2, 1 over two hours either side): the pool
    behind one hour is a slightly different set of launches than the pool
    behind the next, and the smoothing irons out the steps that difference
    prints into the curves. At the day's edges the weights that fall outside
    hours 2..24 are dropped and the rest renormalised."""
    weights = (1.0, 2.0, 3.0, 2.0, 1.0)
    smoothed = {}
    for hour in range(2, 25):
        acc = None
        total = 0.0
        for step, weight in zip(range(-2, 3), weights):
            near = hour + step
            if near not in curves:
                continue
            if acc is None:
                acc = [0.0] * len(curves[hour])
            for i, value in enumerate(curves[near]):
                acc[i] += weight * value
            total += weight
        smoothed[hour] = [value / total for value in acc]
    return smoothed


def forecast_curves(pool, target_first, target_rolling, target_early, levels):
    """{hour: [log-ratio quantile per level]} for hours 2..24, from the weighted pool.

    Hours no pooled launch printed fall back to a flat 0.0 log ratio at every level;
    the caller replaces the whole forecast with persistence if the pool is empty.
    """
    centre = math.log10(max(1, target_first))
    curves = {}
    for hour in range(2, 25):
        weights = kernel_weights(
            pool, target_first, target_rolling, target_early, hour
        )
        rows = [
            (math.log10(launch["first"]), launch["curve"][hour], w)
            for launch, w in zip(pool, weights)
            if hour in launch["curve"]
        ]
        if not rows:
            curves[hour] = [0.0] * len(levels)
            continue
        curves[hour] = local_quantiles(rows, centre, levels)
    return smooth_curves(curves)
