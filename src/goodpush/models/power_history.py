"""The Rider page's power over time (2026-11-13; replaced the calibrated 20-min level, CurrentLevel; moved out of the
model ladder when that went): the physio
model's own all-out power at a few durations - predicted (fresh), observed (each ride day's best, with the fit's
probability it was all-out) and the plan's forecast - and the whole power curve over time (CurveHistory)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class PowerHistory:
    """Per day, at DURATIONS (s):

    * predicted: the physio model's all-out power, fresh (capacity: no fatigue, no calibration), from its daily
      states (processed/physio_states.parquet) through physio.PowerCurve;
    * observed: each ride day's best power (models.physio.PhysioInputs.efforts, the efforts the model sees), the ride
      that rode it, its kind (physio KINDS) and `all_out` = the fit's posterior probability, at the mode, that it was
      all-out: the two terms of the effort mixture (physio.model.PhysioModel.effort_terms);
    * forecast: the plan's forecast (cp, map per day ahead) as the same curve - W' from its MAP, Pmax and fade held at
      the last day's (they are not forecast) - fresh too.

    No range: the stored files hold the states' own 5-95 % ranges, not the predictive distribution of an effort at a
    duration, and the plan's forecast has none."""
    DURATIONS = (60, 300, 1200, 3600)

    def __init__(self, states: pd.DataFrame, acts: pd.DataFrame, fit: dict | None = None,
                 forecast: pd.DataFrame | None = None, max_hr: float | None = None):
        """fit: the stored fit's parameters ({"names", "z"}: models.physio.PhysioRun.stored) or None (no all_out);
        forecast: the plan's physio forecast (cp, map per future day) or None; max_hr: your maximum heart rate (the
        "maxed" kind: models.physio.PhysioInputs.efforts)."""
        self.states = states.dropna(subset=["cp"])
        self.acts, self.fit, self.forecast, self.max_hr = acts, fit, forecast, max_hr

    def predicted(self, s: pd.DataFrame) -> np.ndarray:
        """(durations, days) all-out power, fresh, of the curves in `s` (cp, w_prime, pmax, fade per day)."""
        from ..physio import PowerCurve
        t = np.asarray(self.DURATIONS, float)[:, None]
        return np.asarray(PowerCurve.power(t, *(s[c].to_numpy(float)[None] for c in ("cp", "w_prime", "pmax", "fade"))))

    def ahead(self) -> np.ndarray | None:
        """(durations, forecast days) the forecast's curve, or None without a forecast."""
        fc = self.forecast
        if fc is None or not len(fc):
            return None
        from ..physio import PowerCurve
        from ..physio.model import MAP_S
        last = self.states.iloc[-1]
        cp, mp = fc["cp"].to_numpy(float), fc["map"].to_numpy(float)
        wp = np.asarray(PowerCurve.w_prime_for(MAP_S, mp, cp, float(last["pmax"])))
        return self.predicted(pd.DataFrame({"cp": cp, "w_prime": wp, "pmax": last["pmax"], "fade": last["fade"]}))

    def efforts(self) -> pd.DataFrame:
        """the model's efforts on the states' days: day (index), s, w, kind (index into KINDS) and the ride's id."""
        from ..models.physio import PhysioInputs
        rides = self.acts[self.acts["sport"].eq("ride")]
        idx = self.states.index.get_indexer(pd.to_datetime(rides["date"]).dt.normalize())
        rides, idx = rides[idx >= 0], idx[idx >= 0]
        e = pd.DataFrame(dict(zip(("day", "s", "w", "kind"), PhysioInputs.efforts(rides, idx, self.max_hr))))
        if not len(e):
            return e.assign(id=pd.Series(dtype=str))
        cols = [f"mmp_{d}" for d in PhysioInputs.EFFORT_S]
        each = pd.DataFrame(rides[cols].to_numpy(float), columns=np.asarray(PhysioInputs.EFFORT_S, float))
        each = each.assign(day=idx, id=rides["id"].astype(str).str.removesuffix(".0").to_numpy())
        each = each.melt(id_vars=["day", "id"], var_name="s", value_name="w").astype({"s": float})
        return e.merge(each.drop_duplicates(["day", "s", "w"]), on=["day", "s", "w"], how="left")   # the ride that rode it

    def mode(self) -> dict:
        """the effort mixture's parameters at the fit's mode, by block name."""
        names = pd.Series(np.asarray(self.fit["names"]).astype(str))
        block = names.str.replace(r"\[\d+\]$", "", regex=True).to_numpy()
        z = np.asarray(self.fit["z"], float)
        return {b: z[block == b] for b in ("logit_pi", "log_shape_m2", "log_deficit", "log_scale_kind")}

    def odds(self, e: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        """each effort's posterior probability of being all-out, and of being a bad reading, at the mode (None
        without a fit): all-out = N(0, ALL_OUT_SD) in log power around the fatigued curve, else a Gamma log deficit
        below capacity, weighed by the kind's prior share; either way BAD_READING_SHARE of them bad readings, spread
        evenly - as PhysioModel.effort_terms."""
        if not self.fit or not len(e):
            return None
        from scipy.special import log_expit
        from scipy.stats import gamma, norm
        from ..physio import PhysioModel, PowerCurve
        from ..physio.model import ALL_OUT_SD, BAD_READING_SHARE, BAD_READING_WIDTH, MIN_SHAPE
        p, st = self.mode(), self.states.iloc[e["day"].to_numpy(int)]
        k, t = e["kind"].to_numpy(int), e["s"].to_numpy(float)
        cap = np.asarray(PowerCurve.power(t, *(st[c].to_numpy(float) for c in ("cp", "w_prime", "pmax", "fade"))))
        u = np.log(e["w"].to_numpy(float)) - np.log(np.maximum(cap, 1.0))
        top = log_expit(p["logit_pi"][k]) + norm.logpdf(u - np.log(st["form"].to_numpy(float)), 0.0, ALL_OUT_SD)
        durations = np.unique(t)
        x = PhysioModel.standard_log(durations)[np.searchsorted(durations, t)]
        shape = MIN_SHAPE + np.exp(p["log_shape_m2"][0])
        scale = np.exp(p["log_deficit"][0] + p["log_deficit"][1] * x + np.r_[0.0, p["log_scale_kind"]][k]) / shape
        with np.errstate(divide="ignore"):
            below = np.where(u < 0, log_expit(-p["logit_pi"][k]) + gamma.logpdf(np.maximum(-u, 1e-9), shape, scale=scale),
                             -np.inf)
        good = np.log1p(-BAD_READING_SHARE) + np.logaddexp(top, below)
        bad = np.log(BAD_READING_SHARE / BAD_READING_WIDTH)
        total = np.logaddexp(good, bad)
        return np.exp(np.log1p(-BAD_READING_SHARE) + top - total), np.exp(bad - total)

    def build(self) -> dict:
        s = self.states
        if s.empty:
            return {}
        from ..physio.model import ALL_OUT_SD, KINDS
        pred, fut = self.predicted(s), self.ahead()
        n, m = len(s), 0 if fut is None else fut.shape[1]
        dates = s.index.append(pd.DatetimeIndex(self.forecast.index)) if m else s.index
        e = self.efforts()
        p, bad = self.odds(e) or (None, None)
        num = lambda a: [None if not np.isfinite(v) else round(float(v), 1) for v in np.asarray(a, float)]  # noqa: E731
        observed = []
        for d in self.DURATIONS:                   # sparse: the days with an effort that long
            at = e["s"].to_numpy(float) == d
            observed.append({"i": e["day"].to_numpy(int)[at].tolist(), "w": num(e["w"][at]),
                             "id": [None if pd.isna(v) else str(v) for v in e["id"][at]],
                             "kind": [KINDS[k] for k in e["kind"].to_numpy(int)[at]],
                             "all_out": num(p[at] * 100) if p is not None else None,    # %
                             "bad": num(bad[at] * 100) if bad is not None else None})   # %: a bad reading
        return {"date": [str(x.date()) for x in dates], "today": str(s.index[-1].date()), "s": list(self.DURATIONS),
                "all_out_sd": ALL_OUT_SD,
                "predicted": [num(np.r_[row, np.full(m, np.nan)]) for row in pred],
                "forecast": [num(np.r_[np.full(n - 1, np.nan), pred[j, -1], fut[j]]) for j in range(len(pred))] if m else [],
                "observed": observed}


class CurveHistory:
    """The whole power-duration curve over time (Rider page heatmap, 2026-11-06): weekly, at DURATIONS, (a) the physio
    model's all-out curve that day (fatigue not included: capacity), (b) what you measured - your best of the
    WINDOW_DAYS centred on the column's week (2026-11-12; was the WINDOW_DAYS before it, which trailed the model by
    up to six weeks; the latest weeks have fewer days ahead) and (c) the measured as a percentage of the model (NaN
    where nothing that long was ridden): all-out efforts near 100, ordinary training below, above 100 = the model
    is low there."""
    DURATIONS = (5, 10, 15, 30, 60, 120, 180, 300, 600, 1200, 1800, 3600, 5400, 7200)
    WINDOW_DAYS = 42                 # measured: half of it each side of the week's middle
    STEP = "W-SUN"                   # one column a week (its last day)
    MID_DAYS = 3                     # a column's day less this = the middle of its week

    def __init__(self, states: pd.DataFrame, acts: pd.DataFrame):
        self.states, self.acts = states, acts

    def model(self, days: pd.DatetimeIndex) -> np.ndarray:
        """(durations, weeks) all-out power of the model's curve on those days."""
        from ..physio import PowerCurve
        s = self.states.reindex(days)
        t = np.asarray(self.DURATIONS, float)[:, None]
        return np.asarray(PowerCurve.power(t, s["cp"].to_numpy(float)[None], s["w_prime"].to_numpy(float)[None],
                                           s["pmax"].to_numpy(float)[None], s["fade"].to_numpy(float)[None]))

    def measured(self, days: pd.DatetimeIndex) -> np.ndarray:
        """(durations, weeks) best power within WINDOW_DAYS / 2 of each week's middle (centred rolling max of the
        daily bests)."""
        r = self.acts[self.acts["sport"].eq("ride")]
        cols = [f"mmp_{d}" for d in self.DURATIONS if f"mmp_{d}" in r]
        daily = r.groupby(r["date"].dt.normalize())[cols].max()
        mid = days - pd.Timedelta(days=self.MID_DAYS)
        full = daily.reindex(pd.date_range(min(daily.index.min(), mid.min()), max(daily.index.max(), days.max())))
        half = self.WINDOW_DAYS // 2
        best = full.rolling(2 * half + 1, center=True, min_periods=1).max().reindex(mid)
        return np.stack([best[f"mmp_{d}"].to_numpy(float) if f"mmp_{d}" in best else np.full(len(days), np.nan)
                         for d in self.DURATIONS])

    @staticmethod
    def percent(measured: np.ndarray, model: np.ndarray) -> np.ndarray:
        """measured as % of the model (NaN where either is missing)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(model > 0, 100 * measured / model, np.nan)

    def build(self) -> dict:
        s = self.states.dropna(subset=["cp"])
        if s.empty:
            return {}
        days = pd.Series(s.index, index=s.index).resample(self.STEP).last().dropna()
        days = pd.DatetimeIndex(days.to_numpy())
        num = lambda a: [[None if not np.isfinite(v) else int(round(float(v))) for v in row] for row in a]  # noqa: E731
        model, measured = self.model(days), self.measured(days)
        return {"date": [str(d.date()) for d in days], "s": list(self.DURATIONS),
                "model": num(model), "measured": num(measured), "percent": num(self.percent(measured, model))}
