# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/danzosubianto/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/danzosubianto/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch audience forecaster. Given a case file holding everything known
at the end of a release day's first hour — that hour's viewer count and the hourly
histories of dozens of past big launches — it writes the 5%, 25%, 50%, 75% and 95%
quantiles of the viewer count for each of hours 2 to 24.

The model is a scale-matched pool of past launch curves. Each comparable launch is
normalised to its own hour 1, giving a log-ratio curve for the day; the launches are
then weighted by a Gaussian kernel on the distance between their log10 hour-1
audience and the target's — wide in the day's first hours (0.975), narrowing to
0.325 by hour 24, because early shape owes more to unlock timing than to scale
while late decay tracks scale closely — multiplied by 3 when the launch's unlock
pattern (rolling midnight-local vs one simultaneous global moment) matches the
target's, and by 3 again when its early-access shape matches. Each hour's forecast
is a local line fitted through the pooled curves in log10 hour-1 (slope shrunk and
capped), with the five quantiles read off the weighted residuals at the target's
own scale and smoothed across neighbouring hours, anchored on the target's hour-1
print. In leave-one-out backtests over the kit's 37 usable comparable launches the
method's mean pinball loss is 0.359 against the persistence baseline's 0.930
(0.229 against 0.321 on launches with hour 1 ≥ 20,000, the scale of the practice
targets). Python 3.13, standard library only, deterministic, offline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=qy81p54j&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
