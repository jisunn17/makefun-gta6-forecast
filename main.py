"""gta6-twitch-ccu-forecast-01: a robust scale-regression forecast of a launch day's Twitch curve.

Reads every ``*.in`` case in ``$MAKEFUN_INPUT_DIR`` (else ``/input``) and writes ``<id>.out`` to
``$MAKEFUN_OUTPUT_DIR`` (else ``/output``): for each hour h = 2..24 of the release day, five quantiles of that hour's
concurrent viewers in the game's Twitch category.

The model, in log space (r_h = v_h / v_1 is the hour's ratio to hour 1's print):

  1. Every comparable past launch supplies one point per hour: log r_h against log v_1, keeping comparables whose
     hour-1 value is at least 1,000 viewers and whose hour h has a value.
  2. Per hour a robust (Huber IRLS, weighted) affine fit log r_h = a_h + b_h * log v_1, plus a small quadratic term in
     log v_1 so the decline with scale bends instead of exploding when the case's launch is bigger than every
     comparable. b_h is clamped to [-0.80, 0.10]; with fewer than 8 surviving comparables the hour falls back to the
     median of log r_h.
  3. Comparables are weighted: launch-quality tier read from the case's own metadata (a live-service or free-to-play
     launch is a weaker analogue than a paid single-player one), platform overlap with the case, and a Gaussian kernel
     on log v_1 that leans on the comparables of a similar size.
  4. The forecast's median is m_h = a_h + b_h * log p1 (+ the quadratic term), evaluated at the case's hour-1 print p1.
  5. The spread is empirical, not Gaussian: residuals of the fit are standardised by each hour's MAD, pooled across the
     day, and their 5/25/50/75/95% quantiles become the per-hour offsets delta_tau,h = t_tau * s_h. The forecast is
     q_tau = p1 * exp(m_h + delta_tau,h), which keeps the fat left tail a launch day actually has (most launches fall
     below the persistence of hour 1, a few fall far below it) instead of assuming a symmetric normal.
  6. When the case carries minute reads, the last half hour before the checkpoint gives the within-hour-1 slope: a spike
     still falling at the checkpoint decays further, so the median is nudged down and the spread widened, both capped.

Everything is deterministic: no clock, no randomness, no network, standard library only.

Run it outside the grader's container with:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level; used by the fallback forecast only.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

HOURS = list(range(2, 25))
LEVELS_OUT = LEVELS

# Model constants (tuned by leave-one-out over the 38 practice comparables; see README).
CFG = {
    "bmin": -0.90,
    "bmax": 0.10,
    "qmin": -0.80,
    "qmax": 0.80,
    "hw": 1.8,
    "mult": 1.0,
    "kscale": -0.18,
    "smooth": 1,
    "span": 0,
    "hwq": 1.8,
    "wmad": False,
    "stretch_floor": 0.6,
    "minn": 8,
    "min_v1": 1000.0,
    "tier": True,
    "quad": True,
    "plat": True,
    "cap_delta": 3.6,
}


# ---------------------------------------------------------------- time helpers


def _epoch_hour(stamp):
    """The integer hour (Unix hours) an ISO 8601 UTC string names, or None."""
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        text = stamp.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        moment = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(math.floor(moment.timestamp() / 3600.0))


def _median(values):
    """The median of a non-empty list (mean of the middle two when even)."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _quantile(values, p, weights=None):
    """The p-quantile of a list, linearly interpolating between order statistics; weighted when asked."""
    if not values:
        return 0.0
    if weights is None:
        ordered = sorted(values)
        n = len(ordered)
        if n == 1:
            return ordered[0]
        pos = (n - 1) * p
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    if total <= 0:
        return _quantile(values, p)
    target = p * total
    acc = 0.0
    for i, (value, weight) in enumerate(pairs):
        acc += weight
        if acc >= target:
            return value
    return pairs[-1][0]


def _mad_sigma(values, weights=None):
    """1.4826 x the median absolute deviation: a robust standard deviation."""
    if not values:
        return 0.0
    if weights is None:
        centre = _median(values)
        return 1.4826 * _median([abs(v - centre) for v in values])
    pairs = sorted(zip([abs(v) for v in values], weights))
    total = sum(w for _, w in pairs)
    if total <= 0:
        return _mad_sigma(values)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= total / 2.0:
            return 1.4826 * value
    return 1.4826 * pairs[-1][0]


# ---------------------------------------------------------------- case reading


def _platform_profile(text):
    """Which platform families a launch's blurb names: (pc, playstation, xbox, nintendo)."""
    t = (text or "").lower()
    pc = ("pc" in t) or ("stadia" in t) or ("steam" in t)
    playstation = ("ps5" in t) or ("ps4" in t) or ("playstation" in t)
    xbox = ("xbox" in t) or ("xb1" in t) or ("series x" in t)
    nintendo = ("switch" in t) or ("nintendo" in t)
    return (pc, playstation, xbox, nintendo)


def _launch_weight(entries):
    """How good an analogue a comparable is, from the case's own metadata (never from its future hours).

    A pre-release or non-launch row counts for nothing, a live-service or free-to-play launch is a weak analogue, and
    an early-access-contaminated row (a premium edition days before the real unlock) is a middling one.
    """
    text = " ".join(str(entries.get(k) or "") for k in ("unlock", "source", "name", "platforms")).lower()
    if "not a launch" in text or "before release" in text:
        return 0.0
    if "tier: weak" in text:
        return 0.35
    if "early-access-contaminated" in text:
        return 0.5
    return 1.0


def _series(hourly, start_hour):
    """Hour-indexed viewers and channels for a comparable: index 0 is hour 1 of its release day."""
    value = [None] * 24
    channels = [None] * 24
    by_hour = {}
    for row in hourly or []:
        key = _epoch_hour(row.get("from"))
        if key is None:
            continue
        by_hour[key] = (row.get("viewers"), row.get("channels"))
    for i in range(24):
        hit = by_hour.get(start_hour + i)
        if hit is None:
            continue
        value[i], channels[i] = hit
    return value, channels


def _comparables(case):
    """Every comparable as a row: hour-indexed viewers/channels, its metadata weight and platform profile."""
    rows = []
    for comp in case.get("comparables") or []:
        start = _epoch_hour(comp.get("day_start"))
        if start is None:
            continue
        value, channels = _series(comp.get("hourly"), start)
        rows.append({
            "name": comp.get("name") or "",
            "v": value,
            "ch": channels,
            "w": _launch_weight(comp),
            "plat": _platform_profile(comp.get("platforms")),
        })
    return rows


# ---------------------------------------------------------------- robust fitting


def _solve(matrix, vector):
    """Gaussian elimination with partial pivoting; None when the system is singular."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            if factor:
                for k in range(col, n + 1):
                    a[row][k] -= factor * a[col][k]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = a[i][n] - sum(a[i][j] * out[j] for j in range(i + 1, n))
        out[i] = acc / a[i][i]
    return out


def _huber_irls(points, scale_hint=0.6, iterations=15):
    """Weighted Huber IRLS for y = sum(coef_j * f_j), points = [(features, y, weight)]."""
    nf = len(points[0][0])
    coef = [0.0] * nf
    coef[0] = _median([p[1] for p in points])
    for _ in range(iterations):
        gram = [[0.0] * nf for _ in range(nf)]
        rhs = [0.0] * nf
        for feats, y, weight in points:
            resid = y - sum(c * f for c, f in zip(coef, feats))
            u = abs(resid) / scale_hint
            huber = weight * (1.0 if u <= 1.345 else 1.345 / u)
            for i in range(nf):
                fi = feats[i]
                if fi == 0.0:
                    continue
                rhs[i] += huber * fi * y
                for j in range(nf):
                    gram[i][j] += huber * fi * feats[j]
        for i in range(nf):
            gram[i][i] += 1e-9
        step = _solve(gram, rhs)
        if step is None:
            break
        change = max(abs(step[i] - coef[i]) for i in range(nf))
        coef = step
        if change < 1e-8:
            break
    return coef


# ---------------------------------------------------------------- the model


class Model:
    """The scale-regression forecast: per-hour centre offsets and residual-quantile spreads."""

    def __init__(self, cfg=None):
        self.cfg = dict(CFG)
        if cfg:
            self.cfg.update(cfg)
        self.coef = {}
        self.delta = {}

    def fit(self, rows, target_x, target_plat):
        cfg = self.cfg
        usable = [r for r in rows if r["v"][0] and r["v"][0] >= cfg["min_v1"]]
        if not usable:
            raise ValueError("no usable comparables")
        xs = [math.log(r["v"][0]) for r in usable]
        x_med = _median(xs)
        weights = []
        for i, row in enumerate(usable):
            w = row["w"] if cfg["tier"] else 1.0
            if cfg["plat"] and target_plat is not None:
                same = sum(1 for k in range(4) if row["plat"][k] == target_plat[k])
                w *= 1.0 if same == 4 else (0.8 if same == 3 else 0.55)
            if cfg["hw"] > 0:
                w *= math.exp(-0.5 * ((xs[i] - target_x) / cfg["hw"]) ** 2)
            weights.append(max(w, 1e-6))
        top = max(weights)
        weights = [w / top for w in weights]

        residuals = {}
        points_by_hour = {}
        points_idx = {}
        for h in HOURS:
            points = []
            index = []
            for i, row in enumerate(usable):
                v = row["v"][h - 1]
                if not v or v <= 0:
                    continue
                feats = [1.0, xs[i]]
                if cfg["quad"]:
                    feats.append(((xs[i] - x_med) ** 2 - 1.0) / 10.0)
                points.append((feats, math.log(v / row["v"][0]), weights[i]))
                index.append(i)
            points_by_hour[h] = points
            points_idx[h] = index
            if len(points) < cfg["minn"]:
                ys = [p[1] for p in points]
                centre = _median(ys) if ys else 0.0
                self.coef[h] = ("median", centre)
                continue
            coef = _huber_irls(points)
            if len(coef) > 1:
                coef[1] = min(cfg["bmax"], max(cfg["bmin"], coef[1]))
            if cfg["quad"] and len(coef) > 2:
                coef[2] = min(cfg["qmax"], max(cfg["qmin"], coef[2]))
            self.coef[h] = ("reg", coef)

        # Neighbouring hours share a shape: averaging the coefficients over +/- smooth hours steadies the fit and keeps
        # the curve from wobbling when the case's launch sits above every comparable in the set.
        window = cfg.get("smooth", 0)
        if window:
            raw = {h: self.coef[h] for h in HOURS}
            for h in HOURS:
                if raw[h][0] != "reg":
                    continue
                acc = [0.0] * len(raw[h][1])
                count = 0
                for j in range(h - window, h + window + 1):
                    if j in raw and raw[j][0] == "reg" and len(raw[j][1]) == len(acc):
                        for k in range(len(acc)):
                            acc[k] += raw[j][1][k]
                        count += 1
                self.coef[h] = ("reg", [value / count for value in acc])

        for h in HOURS:
            points = points_by_hour[h]
            if self.coef[h][0] != "reg" or not points:
                residuals[h] = [p[1] - self.coef[h][1] for p in points]
                continue
            coef = self.coef[h][1]
            # Re-centre the intercept as a weighted median of the leftover, so one wild comparable cannot drag it.
            offsets = [sum(coef[j] * p[0][j] for j in range(1, len(coef))) for p in points]
            pairs = sorted((p[1] - o, p[2]) for p, o in zip(points, offsets))
            total = sum(w for _, w in pairs)
            acc = 0.0
            intercept = pairs[-1][0]
            for value, weight in pairs:
                acc += weight
                if acc >= total / 2.0:
                    intercept = value
                    break
            coef[0] = intercept
            residuals[h] = [p[1] - sum(c * f for c, f in zip(coef, p[0])) for p in points]

        # Spread: standardise each hour's residuals by its MAD, pool them, and take the pooled quantile shape.
        hwq = cfg.get("hwq", 0.0)
        shape_weights = None
        if hwq:
            # The residual shape itself depends on launch size: pool the standardised residuals with a kernel around
            # the case's own scale, so a launch the size of this one lends its tails instead of a 500-viewer stream.
            shape_weights = {}
            for h in HOURS:
                wl = []
                for i in points_idx.get(h, []):
                    w = usable[i]["w"] if cfg["tier"] else 1.0
                    if cfg["plat"] and target_plat is not None:
                        same = sum(1 for k in range(4) if usable[i]["plat"][k] == target_plat[k])
                        w *= 1.0 if same == 4 else (0.8 if same == 3 else 0.55)
                    wl.append(max(w * math.exp(-0.5 * ((xs[i] - target_x) / hwq) ** 2), 1e-6))
                top = max(wl) if wl else 1.0
                shape_weights[h] = [w / top for w in wl]
        span = cfg.get("span", 0)
        scale = {}
        for h in HOURS:
            scale[h] = max(1e-6, _mad_sigma(residuals.get(h, []),
                                             shape_weights[h] if (shape_weights and cfg.get("wmad", False)) else None))
        shape_by_hour = {}
        if span:
            # Neighbouring hours share a residual shape: pool the standardised residuals over a small window so each
            # hour's tails come from a hundred points rather than thirty, which keeps the 5% and 95% edges stable.
            for h in HOURS:
                pooled = []
                for j in range(h - span, h + span + 1):
                    pooled.extend([e / scale[j] for e in residuals.get(j, [])])
                shape_by_hour[h] = [_quantile(pooled, tau) for tau in LEVELS] if pooled else None
        pooled = []
        pooled_w = []
        for h in HOURS:
            for idx, e in enumerate(residuals.get(h, [])):
                pooled.append(e / scale[h])
                pooled_w.append(shape_weights[h][idx] if shape_weights else 1.0)
        if not pooled:
            raise ValueError("no residuals")
        global_shape = [_quantile(pooled, tau, pooled_w if cfg.get("hwq", 0.0) else None) for tau in LEVELS]
        stretch = math.exp(cfg["kscale"] * (target_x - x_med))
        stretch = max(cfg.get("stretch_floor", 0.0), min(cfg.get("stretch_cap", 4.0), stretch))
        for h in HOURS:
            shape = shape_by_hour.get(h) or global_shape
            deltas = [t * scale[h] * cfg["mult"] * stretch for t in shape]
            deltas = [max(-cfg["cap_delta"], min(cfg["cap_delta"], d)) for d in deltas]
            for k in range(1, len(deltas)):
                if deltas[k] < deltas[k - 1]:
                    deltas[k] = deltas[k - 1]
            self.delta[h] = deltas
        self.x_med = x_med

    def centre(self, h, p1):
        """The hour's log ratio to hour 1, at the case's hour-1 print."""
        x = math.log(max(1.0, p1))
        kind = self.coef[h][0]
        if kind == "median":
            return self.coef[h][1]
        coef = self.coef[h][1]
        out = coef[0] + coef[1] * x
        if self.cfg["quad"] and len(coef) > 2:
            out += coef[2] * ((x - self.x_med) ** 2 - 1.0) / 10.0
        return out


# ---------------------------------------------------------------- hour-1 shape


def _history_tail(case, start_hour):
    """The last half hour of reads before the checkpoint: (points, slope per hour) of log viewers.

    Empty when the case has no minute-grain history (a practice case holds one hourly point, the final holds a minute
    by minute recording), so the adjustment below simply does not apply there.
    """
    start = start_hour * 3600
    points = []
    for row in case.get("history") or []:
        at = row.get("at")
        if not isinstance(at, str):
            continue
        try:
            text = at[:-1] + "+00:00" if at.endswith("Z") else at
            moment = datetime.fromisoformat(text)
        except (ValueError, TypeError):
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        stamp = moment.timestamp()
        viewers = row.get("viewers")
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        if start - 1800 <= stamp <= start:
            points.append((stamp - start, math.log(float(viewers))))
    if len(points) < 8:
        return []
    points.sort()
    return points


def _tail_slope(points):
    """A Theil-Sen slope of log viewers per hour over the tail; positive while viewers still climb."""
    xs = [p[0] / 3600.0 for p in points]
    ys = [p[1] for p in points]
    slopes = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = xs[j] - xs[i]
            if abs(dx) > 1e-9:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return 0.0
    slope = _median(slopes)
    if not math.isfinite(slope):
        return 0.0
    return max(-6.0, min(6.0, slope))


def _tail_adjustment(case, start_hour):
    """(log shift, spread multiplier) for hour 1's own slope: a spike still falling decays further.

    A launch that peaks inside hour 1 and is already sliding at the checkpoint gives up more of its height afterwards.
    The nudge is capped: at most 0.10 in log on the median and a quarter of the spread, and it never tightens.
    """
    points = _history_tail(case, start_hour)
    if not points:
        return 0.0, 1.0
    slope = _tail_slope(points)
    if slope < 0:
        drop = min(2.0, -slope)
        return -min(0.10, 0.15 * drop), 1.0 + min(0.25, 0.5 * drop)
    climb = min(2.0, slope)
    return min(0.04, 0.06 * climb), 1.0 + min(0.10, 0.10 * climb)


# ---------------------------------------------------------------- output


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def _fallback(p1):
    """Persistence with a widening spread: the house baseline's shape, used only if the model cannot be fitted."""
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in HOURS}


def forecast_case(case):
    """The forecast rows for one case: 23 rows of five quantiles, hours 2..24."""
    known = case.get("known") or []
    if isinstance(known, dict):
        known = [known]
    first = None
    for row in known:
        if isinstance(row, dict) and row.get("hour") == 1:
            first = row
            break
    p1 = 1.0
    if first is not None:
        viewers = first.get("viewers")
        if isinstance(viewers, (int, float)) and math.isfinite(viewers):
            p1 = max(1.0, float(viewers))
    start_hour = _epoch_hour(case.get("day_start"))
    if start_hour is None:
        start_hour = _epoch_hour((first or {}).get("from"))

    try:
        if start_hour is None:
            raise ValueError("no day_start")
        rows = _comparables(case)
        target_x = math.log(p1)
        blurb = " ".join(str(case.get(k) or "") for k in ("notes", "game"))
        target_plat = _platform_profile(blurb) if blurb.strip() else None
        model = Model()
        model.fit(rows, target_x, target_plat)
        shift, widen = _tail_adjustment(case, start_hour)
        plan = {}
        for h in HOURS:
            centre = model.centre(h, p1) + shift
            plan[h] = [p1 * math.exp(centre + d * widen) for d in model.delta[h]]
    except Exception:
        plan = _fallback(p1)

    wanted = sorted({row["hour"] for row in (case.get("forecast") or [])
                     if isinstance(row, dict) and isinstance(row.get("hour"), int)
                     and row["hour"] in HOURS}) or list(HOURS)
    hours = []
    for h in wanted:
        quantiles = []
        previous = 0.0
        for value in plan[h]:
            q = value
            if not math.isfinite(q) or q <= 0:
                q = 1.0
            q = min(q, 1e9)
            q = max(q, previous, 0.01)
            previous = q
            quantiles.append(round2(q))
        hours.append({"hour": h, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            case = {}
        if not isinstance(case, dict):
            case = {}
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case without a file: a plain persistence fan is void-proof, if not sharp.
            p1 = 1.0
            try:
                known = case.get("known") or []
                if isinstance(known, dict):
                    known = [known]
                for row in known:
                    if isinstance(row, dict) and row.get("hour") == 1:
                        viewers = row.get("viewers")
                        if isinstance(viewers, (int, float)) and math.isfinite(viewers):
                            p1 = max(1.0, float(viewers))
            except Exception:
                p1 = 1.0
            plan = _fallback(p1)
            forecast = {"hours": [{"hour": h, "q": [round2(max(0.01, min(1e9, v))) for v in plan[h]]} for h in HOURS]}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        name = path.stem
        (OUTPUT_DIR / f"{name}.out").write_text(text, encoding="utf-8")
        case_id = case.get("id")
        if isinstance(case_id, str) and case_id and case_id != name and "/" not in case_id and "\\" not in case_id:
            (OUTPUT_DIR / f"{case_id}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()