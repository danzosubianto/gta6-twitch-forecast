"""gta6-twitch-ccu-forecast-01 entry: forecast a launch day's Twitch viewers, hour by hour.

Reads each case file in the input directory (schema makefun/twitch-ccu-case/v1) and
writes, for hours 2 to 24 of the release day, the five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's concurrent-viewer count in the game's Twitch category.

The model (model.py): the target's hour-1 print anchors the level; the day's shape
comes from past big launches in the case's `comparables`, each normalised to its own
hour 1 and weighted by a Gaussian kernel on log10 hour-1 distance (narrowing as the
day goes on), with matching unlock patterns and early-access shapes weighted up. A
local line in log10 hour-1 is fitted through each hour's pooled curves and the
scored quantiles are read off its residuals at the target's scale, then smoothed
across neighbouring hours. With no usable comparable, the entry falls back to
persistence (every hour at hour 1's count, random-walk spread), the house baseline.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0), Python 3.13,
standard library only, no network. Outside the grader's container, MAKEFUN_INPUT_DIR
and MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds the model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level, for the persistence fallback.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The fallback's spread, in natural-log units per square-root hour.
SIGMA = 0.15
# The largest quantile the answer format allows.
MAX_VIEWERS = 1_000_000_000


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100
    rounds it. A whole number comes back as an int, so the JSON writes 1240, not
    1240.0."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def persistence_quantiles(p1, hour):
    """The fallback forecast of one hour: hour 1's count, random-walk spread."""
    spread = SIGMA * math.sqrt(hour - 1)
    return [round2(p1 * math.exp(z * spread)) for z in Z]


def clamp_quantiles(values):
    """Bring five raw quantiles into the answer format: each above 0 and at most
    MAX_VIEWERS, and non-decreasing (a later level never below the one before)."""
    out = []
    for value in values:
        value = min(max(value, 0.01), MAX_VIEWERS)
        if out and value < out[-1]:
            value = out[-1]
        out.append(round2(value))
    return out


def forecast_is_valid(forecast, case):
    """True when a forecast matches the answer format exactly: one row per forecast
    hour, five finite quantiles per row, each above 0 and at most MAX_VIEWERS, and
    non-decreasing within the row. An invalid case scores as flat persistence and
    takes no prize, so the caller checks before anything is written."""
    rows = forecast["hours"]
    if len(rows) != len(case["forecast"]):
        return False
    for row, want in zip(rows, case["forecast"]):
        if row["hour"] != want["hour"] or len(row["q"]) != len(LEVELS):
            return False
        quantiles = row["q"]
        if any(not math.isfinite(q) or q <= 0 or q > MAX_VIEWERS for q in quantiles):
            return False
        if any(b < a for a, b in zip(quantiles, quantiles[1:])):
            return False
    return True


def forecast_case(case, name):
    """One case's forecast: {"hours": [{hour, q}, ...]} for its forecast rows."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    pool = model.build_pool(case)
    curves = (
        model.forecast_curves(
            pool,
            p1,
            model.target_is_rolling(case),
            model.target_has_early_access(case),
            LEVELS,
        )
        if pool
        else None
    )

    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        if curves is None:
            quantiles = persistence_quantiles(p1, hour)
        else:
            raw = [p1 * math.exp(x) for x in curves[hour]]
            quantiles = clamp_quantiles(raw)
        rows.append({"hour": hour, "q": quantiles})
    forecast = {"hours": rows}
    if not forecast_is_valid(forecast, case):
        # Never write an invalid file: persistence rows are always in format.
        forecast = {
            "hours": [
                {"hour": row["hour"], "q": persistence_quantiles(p1, row["hour"])}
                for row in case["forecast"]
            ]
        }
    return forecast


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
