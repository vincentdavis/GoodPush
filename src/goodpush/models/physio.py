"""goodpush tables -> physio.PhysioData: daily power histograms, ride best powers, threshold readings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics.load import hist_matrix
from ..metrics.power import POWER_BIN_W, POWER_BINS
from ..physio import KINDS, QUANTITIES, PhysioData, Source
from .ftp_model import HR_REP_CV, HR_REP_S

# FTP-type readings measure CP (threshold, ~LT2), not a 1-h all-out effort
# Your own rides, tests and entries only (2026-11-04: an analysis independent of Garmin's - its FTP and VO2max are
# its own estimates, and its VO2max, one smoothed series, counted as dozens of readings held MAP too firmly)
# Steady rides and reps are no readings: they go in as power at their heart rate (PhysioInputs.steady, the model's
# heart-rate curve), not scaled to threshold HR (2026-11-09)
SOURCES = (Source("my_ftp", "cp", 0.03, 0.03),            # [ftp.history]
           Source("alpha1_lt1", "lt1", 0.08, 0.08),       # alpha1 = 0.75 on ordinary rides
           # a ramp test reads its thresholds high (alpha1 and heart rate lag the rising power; steady-state
           # thresholds sit at or below them): an upper bound, soft (alpha1 noise, a strap dropping beats, a test that
           # comes out low) - only a threshold above it costs. The ramp's top is no MAP reading: legs can give out
           # before the aerobic ceiling, after all the ramp took (2026-11-04) - the ride's best powers are efforts.
           Source("ramp_lt1", "lt1", 0.01, 0.035, bound=True),
           Source("ramp_lt2", "cp", 0.01, 0.035, bound=True),
           Source("ramp_hr_lt1", "lt1", 0.01, 0.05, bound=True),     # ramp power at your LT1 / LT2 heart rate (no
           Source("ramp_hr_lt2", "cp", 0.01, 0.05, bound=True))      # alpha1): heart rate lags the ramp too

class PhysioInputs:
    EFFORT_S = (5, 15, 30, 60, 180, 300, 600, 1200, 2400, 3600, 5400, 7200)
    BINS = 150                                      # 0..1500 W; more goes in the top bin
    POWER_W = (np.arange(BINS) + 0.5) * POWER_BIN_W  # the bins' centres
    # rides without power: HR zone seconds put at these fractions of that day's FTP
    HR_ZONE_FTP = np.array([0.5, 0.6, 0.75, 0.88, 0.98, 1.08, 1.2])
    LONG_AFTER_H = 2.0
    KIND_OF_WORKOUT = {"test": 2, "race": 1, "threshold": 1, "vo2max": 1, "anaerobic": 1, "sweetspot": 1, "tempo": 1}
    TEST_SHARE = 0.9                                # a test duration: >= this x the previous TEST_PRIOR_DAYS' best
    TEST_PRIOR_DAYS = 90
    STEP_BEST_SHARE = 1.01                          # an all-out step's best within this of the ride's: the ride's best
    TEST_MAX_S = 1200                              # tests are 20 min or shorter: a 20-min test's 40-90 min bests aren't
                                                    # all-out there. Races are hard rides (kind 1), not tests: their best
                                                    # powers come inside repeated efforts, not from fresh steady ones
                                                    # (coverage 2026-11-11: races' 20+ min far below the curve)
    STEADY_RIDE_S = 1800                            # a steady ride for the heart-rate curve: >= 30 min,
    STEADY_VI = 1.15                                # normalised / average power at most this, power recorded >= 90 %
    STEADY_X = (0.55, 1.10)                         # heart rate / LTHR kept: not near resting, not past threshold
    # a ride-day's best powers (up to 12 durations) come from overlapping windows of the same riding - one hard
    # effort shows at several durations -, so they are not 12 independent observations: together they count as at
    # most this many (each weighs min(this / n, 1) with n on its day). Ranges were too narrow with all at 1; the
    # levels barely move with the weights (2026-11-10)
    EFFORTS_PER_DAY = 3
    # a severe illness (models.illness: >= 10 days off or resting HR +7) takes fitness the training doesn't explain:
    # its weeks, from onset to ILLNESS_AFTER_DAYS past the comeback note, may change ILLNESS_NOISE x the usual. Unaware
    # of a long illness the fit read part of it as fast detraining (2026-11-11: the best powers fitted better with
    # it, and the MAP time constant came out longer)
    ILLNESS_NOISE = 5.0
    ILLNESS_AFTER_DAYS = 7

    @classmethod
    def start(cls, acts: pd.DataFrame, daily: pd.DataFrame, since_days: int = 0) -> pd.Timestamp:
        """the model's first day: your first ride with power (else the first day). Months of rides without power
        before it left the starting state to its prior: an FTP setting from those months read as a career best.
        since_days ([physio] since_days; 0 = all your data): no earlier than `window_start`."""
        first = daily.index[0]
        if "has_power" in acts:
            r = acts[acts["sport"].eq("ride") & acts["has_power"].fillna(False).astype(bool)]
            ride = pd.to_datetime(r["date"]).min().normalize() if len(r) else None
            first = ride if ride is not None and ride >= first else first
        if since_days and len(daily):
            first = max(first, cls.window_start(pd.Timestamp(daily.index[-1]), since_days))
        return first

    @staticmethod
    def window_start(last: pd.Timestamp, since_days: int) -> pd.Timestamp:
        """the first day of the last `since_days` days up to `last` (your newest data day), moved back to the start
        of its quarter (1 Jan, 1 Apr, 1 Jul, 1 Oct): the model's weekly noise terms count from its first day, so a
        start that moved every day would make every day's fit a full one and drop the ranges' draws. It moves four
        times a year instead (each time: a full fit, and the ranges need a full run)."""
        day = (last - pd.Timedelta(days=int(since_days) - 1)).normalize()
        return pd.Timestamp(year=day.year, month=3 * ((day.month - 1) // 3) + 1, day=1)

    @classmethod
    def build(cls, acts: pd.DataFrame, daily: pd.DataFrame, ftp_history: dict | None, lthr: float,
              intervals: pd.DataFrame | None = None, max_hr: float | None = None,
              illnesses: list | None = None, severe=None, since_days: int = 0) -> PhysioData:
        """intervals: models.intervals.IntervalLog (steady reps for the heart-rate curve), if there is one; max_hr: your
        maximum heart rate (efforts that reached it are all-out: metrics.effort_hr); illnesses: severe illnesses as
        (onset, comeback) dates (`noise_scale`). Starts at the first ride with power (`start`).
        severe: metrics.severe_hr.SevereSplit - the rides' seconds with heart rate by their evidence of being severe
        (hr_hist_h); None: none (evidence 0, the power edges alone). since_days: only about the last this many days
        (`start`; 0 = all)."""
        daily = daily.loc[cls.start(acts, daily, since_days):]
        days = pd.DatetimeIndex(daily.index)
        rides = acts[acts["sport"].eq("ride")]
        dates = pd.to_datetime(rides["date"])
        idx = days.get_indexer(dates)
        rides, idx = rides[idx >= 0], idx[idx >= 0]
        week = (np.arange(len(days)) + days[0].day_of_week) // 7
        z = cls.recovery(daily)
        e_day, e_s, e_w, e_kind = cls.efforts(rides, idx, max_hr)
        r_day, r_w, r_src = cls.readings(rides, idx, daily, ftp_history)
        h_day, h_w, h_x, h_rep = cls.steady(rides, idx, days, lthr, intervals, max_hr)
        rhr = daily["rhr"].median() if "rhr" in daily else np.nan
        long_h = np.bincount(idx, cls.long_hours(rides["moving_s"].fillna(rides["duration_s"])), minlength=len(days))
        hist = cls.histograms(rides, idx, len(days))
        split = {}
        if severe is not None and lthr:
            share = severe.fractions(rides, cls.lthr_then(rides, idx, len(days), lthr, max_hr)[idx], cls.BINS)
            if share.any():
                split = dict(hr_hist_h=cls.histograms(rides, idx, len(days), share))
        return PhysioData(days=days.to_numpy().astype("datetime64[D]"),
                          power_w=cls.POWER_W,
                          hist_h=hist, long_h=long_h, **split,
                          recovery_z=np.clip(np.nan_to_num(z), -3, 3), recovery_seen=np.isfinite(z), week=week,
                          effort_day=e_day, effort_s=e_s, effort_w=e_w, effort_kind=e_kind,
                          effort_weight=cls.effort_weights(e_day),
                          noise_scale=cls.noise_scale(days, week, illnesses or []),
                          reading_day=r_day, reading_w=r_w, reading_source=r_src, sources=SOURCES,
                          hr_day=h_day, hr_w=h_w, hr_x=h_x, hr_rep=h_rep,
                          hr_rest=float(rhr / lthr) if lthr and np.isfinite(rhr) else 0.3,
                          hr_fatigue_ref=cls.usual_fatigue(hist, e_s, e_w, h_day),
                          level_w=cls.own_level(e_day, e_s, e_w))

    @classmethod
    def effort_weights(cls, effort_day: np.ndarray) -> np.ndarray:
        """each best power's weight in the likelihood: min(EFFORTS_PER_DAY / n, 1), n = best powers on its day."""
        day = np.asarray(effort_day, int)
        n = np.bincount(day)[day] if day.size else np.zeros(0)
        return np.minimum(cls.EFFORTS_PER_DAY / np.maximum(n, 1), 1.0)

    @classmethod
    def noise_scale(cls, days: pd.DatetimeIndex, week: np.ndarray, illnesses: list) -> np.ndarray:
        """per day: ILLNESS_NOISE in every week that overlaps an illness (onset .. comeback + ILLNESS_AFTER_DAYS),
        else 1."""
        if not len(days) or not len(illnesses):
            return np.ones(len(days))
        span = np.array([(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=cls.ILLNESS_AFTER_DAYS))
                         for a, b in illnesses], "datetime64[D]").reshape(-1, 2)
        d = days.to_numpy().astype("datetime64[D]")
        # inside one: more illnesses begun by the day than ended before it
        ill = np.searchsorted(np.sort(span[:, 0]), d, "right") > np.searchsorted(np.sort(span[:, 1]), d, "left")
        ill_week = np.bincount(week, ill, minlength=int(week.max()) + 1) > 0
        return np.where(ill_week[week], cls.ILLNESS_NOISE, 1.0)

    LEVEL_DAYS = 365                             # own_level: the rider's first year of efforts
    LEVEL_MIN_RIDES = 10                            # ... with at least this many rides, else no level of their own
    LEVEL_SPRINT_PCT = 98                           # the sprint level: this percentile (a 5-s best can be a spike)

    @classmethod
    def own_level(cls, effort_day: np.ndarray, effort_s: np.ndarray, effort_w: np.ndarray) -> tuple | None:
        """about where this rider's MAP and Pmax start (W), for the priors' centre: the best 5-min power and the
        LEVEL_SPRINT_PCT percentile of the 5-s bests in the first LEVEL_DAYS with efforts (a year's best 5 min is
        about all-out for most riders; the priors stay wide around it). None with under LEVEL_MIN_RIDES of either."""
        if not len(effort_day):
            return None
        early = effort_day < effort_day.min() + cls.LEVEL_DAYS
        five, sprint = effort_w[early & (effort_s == 300)], effort_w[early & (effort_s == 5)]
        if min(len(five), len(sprint)) < cls.LEVEL_MIN_RIDES:
            return None
        return float(five.max()), float(np.percentile(sprint, cls.LEVEL_SPRINT_PCT))

    FATIGUE_TAU_D = 7.0                             # usual_fatigue: the model's fatigue time constant, about

    @classmethod
    def usual_fatigue(cls, hist: np.ndarray, effort_s: np.ndarray, effort_w: np.ndarray, days: np.ndarray) -> float:
        """about the model's fatigue on the steady days (median): hours x (power / CP)^2 fading over FATIGUE_TAU_D,
        CP taken as 0.95 x the 90th percentile of the 20-min bests. Only where the heart-rate fatigue term is measured
        from, so a rough value does."""
        from scipy.signal import lfilter
        best = effort_w[effort_s == 1200]
        if not len(days) or not len(best):
            return 0.0
        cp = 0.95 * float(np.percentile(best, 90))
        power = (np.arange(cls.BINS) + 0.5) * POWER_BIN_W
        work = hist @ (power / cp) ** 2
        k = np.exp(-1.0 / cls.FATIGUE_TAU_D)
        f = lfilter([0.0, 1.0], [1.0, -k], work)              # the morning's fatigue: up to the day before
        return float(np.median(f[days]))

    @classmethod
    def steady(cls, rides: pd.DataFrame, idx: np.ndarray, days: pd.DatetimeIndex, lthr: float,
               intervals: pd.DataFrame | None = None, max_hr: float | None = None):
        """steady stretches for the model's heart-rate curve (day index, power, heart rate / LTHR then, rep?): whole
        steady rides (average power and heart rate) and steady reps (models.intervals.IntervalLog: >= HR_REP_S, power
        within HR_REP_CV, heart rate once settled); heart rate between STEADY_X of LTHR. With max_hr, threshold heart
        rate moves with maximum heart rate (`lthr_then`): the same watts at a lower heart rate years later is age, not
        fitness or fatigue (2026-11-09: a fixed LTHR let the heart-rate fatigue term pick up the years)."""
        if not lthr:
            return np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0, bool)
        ok = ((rides["duration_s"].to_numpy(float) >= cls.STEADY_RIDE_S) & rides["avg_hr"].notna().to_numpy()
              & (rides["avg_power"].to_numpy(float) > 0) & (rides["power_coverage"].to_numpy(float) > 0.9)
              & (rides["np_used"].to_numpy(float) <= cls.STEADY_VI * rides["avg_power"].to_numpy(float)))
        day, w, hr = [idx[ok]], [rides["avg_power"].to_numpy(float)[ok]], [rides["avg_hr"].to_numpy(float)[ok]]
        rep = [np.zeros(ok.sum(), bool)]
        if intervals is not None and len(intervals):
            st = intervals[(intervals["seconds"] >= HR_REP_S) & (intervals["cv"] <= HR_REP_CV) & intervals["hr_late"].notna()]
            d = days.get_indexer(pd.to_datetime(st["date"]).dt.normalize())
            day.append(d), w.append(st["watts"].to_numpy(float)), hr.append(st["hr_late"].to_numpy(float))
            rep.append(np.ones(len(st), bool))
        day = np.concatenate(day)
        thr = cls.lthr_then(rides, idx, len(days), lthr, max_hr)[np.clip(day, 0, max(len(days) - 1, 0))]
        w, x, rep = np.concatenate(w), np.concatenate(hr) / thr, np.concatenate(rep)
        keep = (day >= 0) & np.isfinite(w) & (w > 0) & (x >= cls.STEADY_X[0]) & (x <= cls.STEADY_X[1])
        return day[keep].astype(int), w[keep], x[keep], rep[keep]

    @classmethod
    def recovery(cls, daily: pd.DataFrame) -> np.ndarray:
        """The physio model's recovery markers per day (physio RECOVERY, + = better, NaN = not measured):
        sleep z over the last two nights (one short night isn't what costs you), HRV z, -resting HR z,
        -(hours later to bed than your usual: median of the BED_BASE_DAYS nights before)."""
        from ..planning.readiness import BED_BASE_DAYS, bedtime_hours
        col = daily.reindex(columns=["sleep_h_z", "hrv_z", "rhr_z", "bedtime_h"])
        sleep2 = col["sleep_h_z"].rolling(2, min_periods=1).mean()
        bed = bedtime_hours(col["bedtime_h"])
        late = bed - bed.rolling(BED_BASE_DAYS, min_periods=10).median().shift(1)
        return np.c_[sleep2, col["hrv_z"], -col["rhr_z"], -late].astype(float)

    @classmethod
    def power_hours(cls, H: np.ndarray) -> np.ndarray:
        """rides' seconds per power bin (rides, POWER_BINS; metrics.power.power_histogram) -> hours in the model's
        bins (rides, BINS): coasting left out, everything above the top bin in it."""
        H = np.nan_to_num(np.asarray(H, float))
        out = H[:, :cls.BINS].copy()
        out[:, 0] = 0.0                                                  # coasting
        out[:, -1] += H[:, cls.BINS:].sum(1)
        return out / 3600.0

    @classmethod
    def long_hours(cls, seconds: np.ndarray) -> np.ndarray:
        """each ride's hours past LONG_AFTER_H (from its moving seconds)."""
        return np.maximum(np.asarray(seconds, float) / 3600.0 - cls.LONG_AFTER_H, 0.0)

    @classmethod
    def histograms(cls, rides: pd.DataFrame, idx: np.ndarray, n_days: int, share: np.ndarray | None = None) -> np.ndarray:
        """hours per power bin per day; rides without power from their HR zones and that day's FTP. share: (rides,
        BINS, K) - each ride's power bins split K ways by these shares -> (days, BINS, K) (rides without power: none)."""
        H = hist_matrix(rides["power_hist"], POWER_BINS)
        has = np.isfinite(H).all(1) & (np.nan_to_num(H).sum(1) > 0)
        H = cls.power_hours(H)
        out = np.zeros((n_days, cls.BINS))
        if share is not None:
            part = np.zeros((n_days,) + share.shape[1:])
            np.add.at(part, idx[has], H[has][:, :, None] * share[has])
            return part
        np.add.at(out, idx[has], H[has])
        # no power: each HR zone's seconds at a fraction of FTP
        hz = rides[[f"hz{i}_s" for i in range(1, 8)]].to_numpy(float)[~has]
        ftp = rides["ftp"].to_numpy(float)[~has]
        ok = np.isfinite(ftp) & (np.nan_to_num(hz).sum(1) > 0)
        b = np.clip((ftp[ok, None] * cls.HR_ZONE_FTP / POWER_BIN_W).astype(int), 0, cls.BINS - 1)
        flat = idx[~has][ok, None] * cls.BINS + b
        out += np.bincount(flat.ravel(), np.nan_to_num(hz[ok]).ravel() / 3600.0,
                           minlength=n_days * cls.BINS).reshape(n_days, cls.BINS)
        return out

    @classmethod
    def efforts(cls, rides: pd.DataFrame, idx: np.ndarray, max_hr: float | None = None):
        """each day's best power per duration (only durations the ride lasted), and its kind (physio KINDS: 0 easy,
        1 hard workout or race, 2 test - the day's hardest -, 3 maxed: the ride that set it reached your maximum
        heart rate at its end, EffortHeartRate)."""
        from ..metrics.effort_hr import EffortHeartRate
        cols = [f"mmp_{s}" for s in cls.EFFORT_S]
        M = rides[cols].to_numpy(float)
        moving = rides["moving_s"].fillna(rides["duration_s"]).to_numpy(float)
        ok = (rides["power_coverage"].to_numpy(float) > 0.9)[:, None] & (np.array(cls.EFFORT_S) <= moving[:, None])
        M = np.where(ok & (M > 0), M, np.nan)
        kind = rides["workout"].map(cls.KIND_OF_WORKOUT).fillna(0).to_numpy(int)
        mx = EffortHeartRate.maxed(rides, max_hr) if max_hr else pd.DataFrame(index=rides.index)
        hit = np.column_stack([mx[s].to_numpy(bool) if s in mx else np.zeros(len(rides), bool) for s in cls.EFFORT_S])
        e = pd.DataFrame(M, columns=cls.EFFORT_S).assign(day=idx)
        best = e.groupby("day")[list(cls.EFFORT_S)].transform("max").to_numpy(float)
        maxed = pd.DataFrame((M == best) & hit, columns=cls.EFFORT_S).assign(day=idx).groupby("day").any()
        # a test is one only where it went near your best: >= TEST_SHARE of the previous TEST_PRIOR_DAYS' best
        # at that duration (2026-11-06: a 20-min FTP test's 5 s, a 10 s / 60 s test's 20 min, LSCTs, failed tests sat
        # at 0.3-0.7 of it and taught the model that tests can fall far below the curve); elsewhere a hard ride
        day_best = e.groupby("day")[list(cls.EFFORT_S)].max().reindex(np.arange(int(idx.max()) + 1 if len(idx) else 0))
        prior = day_best.rolling(cls.TEST_PRIOR_DAYS, min_periods=1).max().shift(1).to_numpy(float)[idx]
        with np.errstate(invalid="ignore"):
            tested = np.isnan(prior) | (M >= cls.TEST_SHARE * prior)
        tested &= (np.array(cls.EFFORT_S) <= cls.TEST_MAX_S)[None, :]
        K = np.where(kind[:, None] == 2, np.where(tested, 2, 1), kind[:, None]) * np.ones((1, M.shape[1]), int)
        if "all_out_s" in rides:                     # a planned all-out step (models.all_out): its length a test
            step_s, step_w = (rides[c].to_numpy(float) for c in ("all_out_s", "all_out_w"))
            with np.errstate(invalid="ignore"):
                stepped = (np.array(cls.EFFORT_S)[None, :] == step_s[:, None]) & tested & \
                          (M <= cls.STEP_BEST_SHARE * step_w[:, None])    # the ride's best is the step's
            K = np.where(stepped, 2, K)
        kd = pd.DataFrame(np.where(np.isfinite(M), K, -1), columns=cls.EFFORT_S).assign(day=idx).groupby("day").max()
        f = e.groupby("day").max()
        long = f[list(cls.EFFORT_S)].stack().dropna()
        day, s = long.index.get_level_values(0).to_numpy(int), long.index.get_level_values(1).to_numpy(float)
        k = np.maximum(kd.stack().reindex(long.index).to_numpy(float), 0).astype(int)
        at = maxed.stack().reindex(long.index).fillna(False).to_numpy(bool)
        return day, s, long.to_numpy(float), np.where(at, len(KINDS) - 1, k)

    @staticmethod
    def lthr_then(rides: pd.DataFrame, idx: np.ndarray, n_days: int, lthr: float, max_hr: float | None) -> np.ndarray:
        """threshold heart rate on each day: `lthr` (today's) scaled by maximum heart rate then / now
        (metrics.effort_hr.EffortHeartRate.max_hr_then: never rises later); without max_hr, `lthr` throughout."""
        from ..metrics.effort_hr import EffortHeartRate
        if not max_hr or not len(idx) or not n_days:
            return np.full(n_days, float(lthr))
        then = pd.Series(EffortHeartRate.max_hr_then(rides, max_hr).to_numpy(float), index=idx).groupby(level=0).max()
        then = then.reindex(np.arange(n_days)).ffill().bfill().to_numpy(float)
        return lthr * then / then[-1]

    @staticmethod
    def readings(rides: pd.DataFrame, idx: np.ndarray, daily: pd.DataFrame, ftp_history: dict | None):
        """threshold readings (day index, watts, source index) from every source that has some."""
        days = pd.DatetimeIndex(daily.index)
        parts = {"my_ftp": pd.Series({pd.Timestamp(k): float(v) for k, v in (ftp_history or {}).items()}, dtype=float)}
        for name, col in (("alpha1_lt1", "hrvt1_w"), ("ramp_lt1", "ramp_lt1_w"), ("ramp_lt2", "ramp_lt2_w"),
                          ("ramp_hr_lt1", "ramp_hr_lt1_w"), ("ramp_hr_lt2", "ramp_hr_lt2_w")):
            if col in rides:
                parts[name] = pd.Series(rides[col].to_numpy(float), index=days[idx])
        names = [s.name for s in SOURCES]
        s = pd.concat({k: v for k, v in parts.items() if len(v)}, names=["source", "day"])
        s = s[np.isfinite(s.to_numpy(float)) & (s.to_numpy(float) > 0)]
        d = days.get_indexer(pd.DatetimeIndex(s.index.get_level_values("day")))
        keep = d >= 0
        return (d[keep], s.to_numpy(float)[keep],
                pd.Index(names).get_indexer(s.index.get_level_values("source"))[keep])


class PhysioRun:
    """The physio model for this workspace, kept current cheaply:

    * full fit (`fit`, `goodpush physio`): posterior mode over everything (+ optional posterior draws);
    * daily update (`current`, used by the planner and the app): the new days taken in - the posterior mode
      re-fitted, warm-started from the stored one (about a second). The stored draws are kept as they are (by
      name; new weeks' noise from the prior), dated by the run that made them (`draws_day`);
    * ranges (`refresh`, `goodpush physio --refresh`; the app runs it in its own process on the home computer
      about weekly, [physio] ranges_every_days): a short NUTS run from the stored run's last draws on today's data.

    A full NUTS run (FULL_RUN) is only needed when there are no draws or the model changed.

    Kept in processed/physio_fit.npz (parameters, what was seen; synced), physio_draws.npz (draws; this machine
    only) and processed/physio_states.parquet (daily capacities; with draws also their 5 / 95 % quantiles)."""
    FIT = "physio_fit.npz"           # the coefficients (synced to your other computers)
    DRAWS = "physio_draws.npz"       # the posterior draws: large, kept on the machine that made them (not synced)
    CHECKPOINT = "physio_nuts.ckpt"  # a NUTS run in progress (`goodpush physio --nuts` goes on from it)
    WHITENING = "physio_whitening.pkl"   # the last full NUTS run's learned whitening, for refreshes (this machine only)
    WARMUP = 400                     # NUTS warm-up steps a chain (whitening learned in 3 blocks, then the step size)
    # `refresh` (the ranges): a short NUTS run from the last run's draws, with its own checkpoint (peer sync skips
    # physio_nuts.*); after REFRESH_DRAWS a chain, more until REFRESH_ESS and REFRESH_R_HAT (looser than a full run's
    # PhysioFit.MAX_R_HAT 1.01), or REFRESH_MINUTES of drawing
    REFRESH_CHECKPOINT = "physio_nuts.refresh.ckpt"
    REFRESH_DRAWS = 800
    REFRESH_ESS = 200
    REFRESH_R_HAT = 1.02
    REFRESH_MINUTES = 12
    STATES = "physio_states.parquet"
    COLUMNS = ("cp", "lt1", "map", "f_t", "w_prime", "pmax", "fade", "ftp", "fatigue", "form")
    MAX_TRAJECTORIES = 100
    MAXITER = 15000                  # L-BFGS iterations for a posterior mode (a warm start needs a few hundred)
    RECENT_DAYS = 28                 # these last days may change (late syncs) without a full refit
    JAX_CACHE = "jax_cache"          # processed/: compiled model code (this machine and jax version only)
    LOCK = "physio_nuts.lock"        # held by a NUTS run (full or refresh): one at a time (peer sync skips *.lock)
    PROGRESS = "physio_nuts_progress.log"   # while LOCK is held: its start and last progress line (the Model page
                                            # shows a run started elsewhere; peer sync skips *.log)
    SEVERE = "severe"                # the illnesses (models.illness severity) the model allows a larger change in
    FULL_RUN = "goodpush physio --nuts --samples 2500 --ess 400 --max-minutes 540"   # what to run for new draws

    def __init__(self, proj, since_days: int | None = None):
        """since_days: fit on about the last this many days only (`goodpush physio --since-days`); None: [physio]
        since_days (0 or unset: all your data)."""
        self.proj = proj
        self.processed = proj.processed
        self._since_days = since_days
        self._start = None                   # the model's first day, once `inputs` has read the data
        self.keep_compiled(self.processed / self.JAX_CACHE)

    @staticmethod
    def keep_compiled(path) -> None:
        """JAX keeps its compiled code in `path`, so a new app or CLI process loads it instead of compiling
        again. Specific to this machine and jax version (jax keys it by both), so never in git. A cache folder you set
        yourself (JAX_COMPILATION_CACHE_DIR; the tests share one across runs) wins.

        One subfolder per compression (`cache_codec`): jax names an entry the same whether it was compressed with
        Python 3.14's zstd or with zlib, so the installed app (Python 3.12) and a checkout (3.14) sharing one folder
        read each other's entries with the wrong one ("Unable to decompress Zstandard data"). Entries from before the
        subfolders, a mix of both, are removed."""
        import os
        from pathlib import Path
        import jax
        if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
            return
        path = Path(path)
        for old in [*path.glob("*-cache"), *path.glob("*-atime")]:
            if old.is_file():
                old.unlink(missing_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(path / PhysioRun.cache_codec()))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)   # the many small ones add up

    @staticmethod
    def cache_codec() -> str:
        """What jax compresses its cache entries with, in its own order: Python 3.14's compression.zstd, else the
        zstandard package, else zlib."""
        import importlib
        for codec, module in (("zstd", "compression.zstd"), ("zstandard", "zstandard")):
            try:
                importlib.import_module(module)
                return codec
            except ImportError:
                continue
        return "zlib"

    def inputs(self) -> PhysioData:
        import json
        acts = pd.read_parquet(self.processed / "activities.parquet")
        daily = pd.read_parquet(self.processed / "daily.parquet")
        ath = self.proj.cfg.get("athlete") or {}
        try:
            res = json.loads((self.proj.reports / "results.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            res = {}
        info = res.get("athlete") or {}
        ill = [(e["onset"], e["comeback_note"]) for e in (res.get("illness") or {}).get("episodes") or []
               if e.get("severity") == self.SEVERE]
        lthr = float(info.get("lthr") or ath.get("lthr") or 0) or None
        max_hr = float(info.get("max_hr") or ath.get("max_hr") or 0) or None
        from ..metrics.severe_hr import SevereSplit
        from .intervals import IntervalLog
        data = PhysioInputs.build(acts, daily, (self.proj.cfg.get("ftp") or {}).get("history") or {}, lthr,
                                  IntervalLog.load(self.processed), max_hr, ill, SevereSplit(self.processed),
                                  self.since_days)
        self._start = str(data.days[0]) if len(data.days) else None
        return data

    @property
    def since_days(self) -> int:
        """fit on about the last this many days only (PhysioInputs.window_start); 0 = all your data."""
        if self._since_days is not None:
            return max(int(self._since_days), 0)
        try:
            return max(int((self.proj.cfg.get("physio") or {}).get("since_days") or 0), 0)
        except (TypeError, ValueError):
            return 0

    def stored(self) -> dict | None:
        """the saved fit, if it was made with this noise period (else its noise terms mean other weeks) and these
        states (else the same parameter names mean other things: a structure change, e.g. 2026-10-31)."""
        from ..physio.model import NOISE_WEEKS
        try:
            with np.load(self.processed / self.FIT, allow_pickle=False) as f:
                st = {k: f[k] for k in f.files}
        except (OSError, KeyError, ValueError):
            return None
        if int(st.get("noise_weeks", -1)) != NOISE_WEEKS or str(st.get("states", "")) != self.structure():
            return None
        if self._start is not None and self.first_day(st) not in (None, self._start):
            return None                            # another first day ([physio] since_days): its noise weeks are others
        st["names"] = st["names"].tolist()
        st["draws_day"] = str(st.get("draws_day", st.get("full_date", "")))   # (full_date: before 2026-11-10)
        if "samples" not in st:                    # the draws, if this machine has the ones made with this fit
            st["samples"] = np.zeros((0, len(st["names"])))
            try:                                   # (an older file's weights, log_w, are ignored)
                with np.load(self.processed / self.DRAWS, allow_pickle=False) as f:
                    if str(f["fit_id"]) == str(st.get("fit_id", "")):
                        st["samples"] = f["samples"]
            except (OSError, KeyError, ValueError):
                pass
        st.pop("log_w", None)
        return st

    def same_model(self, model, st: dict | None) -> bool:
        """the stored fit has this model's parameters (noise weeks aside): its draws can be kept / gone on from."""
        return st is not None and ([n for n in st["names"] if not n.startswith("eta")]
                                   == [n for n in model.spec.names() if not n.startswith("eta")])

    def kept(self, model, st: dict | None, data) -> tuple[np.ndarray, str]:
        """the stored draws on this model's parameters (new weeks' noise from the prior) and the data day of the run
        that made them; none (and "") when there are none or they belong to another model."""
        if not self.same_model(model, st) or not len(st["samples"]):
            return np.zeros((0, model.spec.size)), ""
        return (model.spec.align(st["names"], st["samples"], rng=np.random.default_rng(len(data.days))),
                st["draws_day"])

    @classmethod
    def nuts_running(cls, processed) -> bool:
        """a NUTS run (full or refresh, any process) holds LOCK right now."""
        from filelock import FileLock, Timeout
        lock = FileLock(str(Path(processed) / cls.LOCK))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return True
        lock.release()
        return False

    @classmethod
    def nuts_progress(cls, processed) -> dict | None:
        """the NUTS run holding LOCK (any process): {"kind" (full | refresh), "started", "t" (epoch s), "line" (its
        last progress line)}; None when no run holds it, or it hasn't said anything yet."""
        import json
        if not cls.nuts_running(processed):
            return None
        try:
            return json.loads((Path(processed) / cls.PROGRESS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def tracked(self, progress, kind: str):
        """`progress` that also keeps PROGRESS up to date (call with LOCK held; untrack() when the run ends)."""
        import functools
        import time
        say = functools.partial(self.said, progress, kind, time.time())
        say("physio model: starting the sampling run ...")
        return say

    def said(self, progress, kind: str, started: float, line: str) -> None:
        """one progress line: passed on, and written to PROGRESS (tracked)."""
        import json
        import time
        progress(line)
        path = self.processed / self.PROGRESS
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"kind": kind, "started": started, "t": time.time(), "line": str(line)}),
                           encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass

    def untrack(self) -> None:
        (self.processed / self.PROGRESS).unlink(missing_ok=True)

    @staticmethod
    def priors() -> str:
        """how the threshold readings are weighed (SOURCES): a fit made with others is redone (warm-started)."""
        return ",".join(f"{s.name}:{s.quantity}:{s.bias_sd}:{s.sd}" + (":bound" if s.bound else "") for s in SOURCES)

    @staticmethod
    def structure() -> str:
        """what the model's states are (names and version): a stored fit with another one is not reused."""
        from ..physio.model import STATES, VERSION
        return ",".join(STATES) + f"|v{VERSION}"

    @staticmethod
    def first_day(st: dict) -> str | None:
        """the stored fit's first day (from its last day and its number of days), None if it doesn't say."""
        try:
            last = np.datetime64(str(st["day"]), "D")
            return str(last - np.timedelta64(int(st["days"]) - 1, "D"))
        except (KeyError, TypeError, ValueError):
            return None

    def warm_start(self, model) -> np.ndarray | None:
        """the last fit's parameters on this model's vector (new weeks / names at their prior mean)."""
        st = self.stored()
        return None if st is None else model.spec.align(st["names"], st["z"])[0]

    @staticmethod
    def key(data: PhysioData) -> str:
        """what the fit saw: nothing to do while it is the same."""
        return (f"{len(data.days)}:{data.days[-1]}:{len(data.effort_w)}:{data.effort_w.sum():.0f}:"
                f"{len(data.reading_w)}:{data.reading_w.sum():.0f}:{data.noise_scales.sum():.0f}")

    @staticmethod
    def history_key(data: PhysioData, days: int) -> str:
        """the training, efforts and readings of the first `days` days (recovery z-scores left out: their
        baselines move a little every day)."""
        e, r = data.effort_day < days, data.reading_day < days
        from ..physio.model import EVIDENCE
        return (f"{days}:{data.hist_h[:days].sum():.2f}:{(data.hr_hours[:days] @ EVIDENCE).sum():.2f}:"
                f"{data.long_h[:days].sum():.2f}:{int(e.sum())}:"
                f"{data.effort_w[e].sum():.0f}:{int(r.sum())}:{data.reading_w[r].sum():.0f}:"
                f"{data.noise_scales[:days].sum():.0f}")

    def save(self, model, data, z, draws, draws_day: str, info: dict | None = None, progress=print) -> pd.DataFrame:
        """the fit (coefficients; draws on this machine), its daily states, what training does and the Model
        page's diagnostics (`info`: a NUTS run's, else the last run's convergence is kept). `draws_day`: the data
        day of the run that made the draws ("" without). Each file is replaced whole (`write_whole`): the app may
        read them while a refresh in another process saves. Stored draws this fit doesn't take (another model
        version, a fresh fit) are deleted - and said so (2026-11-13: a v14 run's draws went without a word)."""
        from ..physio.model import NOISE_WEEKS
        import secrets
        n = len(data.days)
        fit_id = secrets.token_hex(8)              # ties the draws (this machine only) to these coefficients
        draws_day = draws_day if len(draws) else ""
        if info:                                   # after a NUTS run each stage is named first (2026-11-13: two
            progress(f"physio model: saving 1/5: the draws ({len(draws)}) ...")   # resets in the write-out)
        if len(draws):
            self.write_whole(self.processed / self.DRAWS, lambda f: np.savez(f, samples=draws, fit_id=fit_id))
        elif (self.processed / self.DRAWS).exists():
            try:
                with np.load(self.processed / self.FIT, allow_pickle=False) as f:
                    was = str(f["states"]).rsplit("|", 1)[-1]
            except (OSError, KeyError, ValueError):
                was = "an unknown model version"
            progress(f"physio model: the stored posterior draws (model {was}) don't belong to this fit (model "
                     f"{self.structure().rsplit('|', 1)[-1]}): deleted - ranges need a full run: {self.FULL_RUN}")
            (self.processed / self.DRAWS).unlink()
        self.write_whole(self.processed / self.FIT, lambda f: np.savez_compressed(
            f, z=z, names=np.array(model.spec.names()), fit_id=fit_id, key=self.key(data), days=n,
            day=str(data.days[-1]), draws_day=draws_day, noise_weeks=NOISE_WEEKS, states=self.structure(),
            priors=self.priors(), hist_key=self.history_key(data, max(n - self.RECENT_DAYS, 0))))
        if info:
            progress(f"physio model: saving 2/5: daily states of {min(len(draws), self.MAX_TRAJECTORIES)} draws ...")
        T = self.picked(model, draws)
        states = self.states(model, z, draws, T)
        self.write_whole(self.processed / self.STATES, states.to_parquet)
        if info:
            progress("physio model: saving 3/5: what training does ...")
        TrainingResponse(model, z, draws).save(self.processed)
        if info:
            progress("physio model: saving 4/5: the sampler's checks for the Model page ...")
        sampling = info and PhysioDiagnostics.sampling(model, draws, info)
        if info:
            progress("physio model: saving 5/5: the Model page ...")
        PhysioDiagnostics(model, z, T, draws).save(self.processed, sampling, draws_day)
        return states

    WRITE_WAITS = (0.2, 0.5, 1, 2, 4, None)    # Windows: a file another process has open can't be replaced yet

    @classmethod
    def write_whole(cls, path: Path, write) -> None:
        """`write(file)` into a temporary file next to `path`, then put it in `path`'s place: a reader sees the old
        file or the new one, never half of one."""
        import time
        import os
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
            if not f.closed:                       # (a writer may close it itself)
                f.flush()
                os.fsync(f.fileno())               # on the disk before the swap (2026-11-13: a reset after a swap
        for wait in cls.WRITE_WAITS:               # left a checkpoint of zeros)
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if wait is None:
                    raise
                time.sleep(wait)

    @staticmethod
    def version_on_disk() -> int:
        """the model VERSION in physio/model.py as it is on disk now (a process may have loaded an older one)."""
        import re
        from ..physio import model
        from ..physio.model import VERSION
        m = re.search(r"^VERSION = (\d+)", Path(model.__file__).read_text(encoding="utf-8"), re.M)
        return int(m.group(1)) if m else VERSION

    def same_code(self) -> None:
        """raise unless this process runs the model version on disk: an app started before an update would refit with
        its old model after a sync, write it over the newer fit and drop its draws (2026-11-13). Restart it."""
        from ..physio.model import VERSION
        disk = self.version_on_disk()
        if disk != VERSION:
            raise RuntimeError(f"this process runs physio model v{VERSION}, the code on disk is v{disk}: restart it "
                               "(no refit until then)")

    def current(self, progress=print):
        """(model, parameters) for today's data: stored if it saw the same data, else updated with the new
        days (about a second), or - when older data changed or there is no fit - a full posterior-mode fit.
        Never on another model version than the code on disk (`same_code`)."""
        from ..physio import PhysioModel
        self.same_code()
        data = self.inputs()
        model = PhysioModel(data)
        st = self.stored()
        weights = st is not None and str(st.get("priors", "")) == self.priors()   # the readings weighed as now
        if st is not None and str(st["key"]) == self.key(data) and st["names"] == model.spec.names() and weights:
            return model, st["z"]
        seen = int(st["days"]) if st is not None and "days" in st else 0
        cut = seen - self.RECENT_DAYS
        if (self.same_model(model, st) and weights and cut > 0 and len(data.days) >= seen
                and str(st.get("hist_key")) == self.history_key(data, cut)):
            return model, self.update(model, data, st, progress)
        progress(f"physio model {self.structure().rsplit('|', 1)[-1]}: no fit yet, the model or the weight of its "
                 "readings changed, or older data changed: full fit")
        _, res, _, _ = self.fit(progress=progress, data=data, model=model)
        return model, res.z

    def update(self, model, data, st, progress=print) -> np.ndarray:
        """the stored fit brought up to date with the days since: the posterior mode over everything, warm-started
        from the stored one by name (new weeks at their prior mean; about a second); the stored draws kept as they
        are, with their own date (`refresh` makes today's)."""
        import time
        from ..physio import PhysioFit
        t = time.time()
        seen = int(st["days"])
        z = PhysioFit(model).map(model.spec.align(st["names"], st["z"])[0], maxiter=self.MAXITER).z
        draws, draws_day = self.kept(model, st, data)
        self.save(model, data, z, draws, draws_day, progress=progress)
        progress(f"physio model {self.structure().rsplit('|', 1)[-1]}: updated with {len(data.days) - seen} new day(s) "
                 f"in {time.time() - t:.0f}s" + (f"; ranges from the run of {draws_day}" if len(draws) else f"; no ranges yet: {self.FULL_RUN}"))
        return z

    def fit(self, samples: int = 0, maxiter: int = MAXITER, progress=print, data=None, model=None, nuts: bool = False,
            fresh: bool = False, target_ess: float | None = None, max_minutes: float | None = None,
            warmup: int = WARMUP):
        """the posterior mode over everything (+ draws); `fresh`: start from the priors, not the last fit
        (after a model change, or when the last fit went somewhere implausible), and drop NUTS's checkpoint.
        NUTS goes on from its checkpoint (stopped, or finished: more draws / target_ess) with the data it
        started with (newer days follow in the daily updates); without one, with target_ess, from the saved
        fit's draws if they come from a run of this length on the same data history. A NUTS run holds LOCK (another
        one going on: RuntimeError). A mode-only fit keeps the stored draws (`kept`)."""
        if samples and nuts:
            from filelock import FileLock, Timeout
            try:
                with FileLock(str(self.processed / self.LOCK), timeout=0):
                    try:
                        return self.fitted(samples, maxiter, self.tracked(progress, "full"), data, model, nuts,
                                           fresh, target_ess, max_minutes, warmup)
                    finally:
                        self.untrack()
            except Timeout:
                raise RuntimeError("another NUTS run of this model is going on (a refresh, or a full run): "
                                   "try again when it's done") from None
        return self.fitted(samples, maxiter, progress, data, model, nuts, fresh, target_ess, max_minutes, warmup)

    def fitted(self, samples, maxiter, progress, data, model, nuts, fresh, target_ess, max_minutes, warmup):
        """`fit` (its lock held when it runs NUTS)."""
        from ..physio import CHAINS, PhysioFit, PhysioModel
        data = data if data is not None else self.inputs()
        checkpoint = self.processed / self.CHECKPOINT
        seed_draws = None
        if fresh:
            checkpoint.unlink(missing_ok=True)
        elif samples and nuts:
            ck = PhysioFit.load_checkpoint(checkpoint)
            end = int(ck.get("days", 0)) if ck else 0
            if 0 < end <= len(data.days) and ck["tag"] == self.nuts_tag(data.until(end), samples, warmup):
                data, model = data.until(end), None
            elif target_ess:
                st = self.stored()
                end = int(st["days"]) if st is not None else 0
                if (st is not None and len(st["samples"]) == CHAINS * samples and 0 < end <= len(data.days)
                        and str(st["hist_key"]) == self.history_key(data, max(end - self.RECENT_DAYS, 0))):
                    data, model = data.until(end), None
                    seed_draws = np.asarray(st["samples"]).reshape(CHAINS, samples, -1)
        model = model if model is not None else PhysioModel(data)
        fitter = PhysioFit(model)
        progress(f"physio model {self.structure().rsplit('|', 1)[-1]}: {len(data.days)} days, "
                 f"{len(data.effort_w)} best powers, {len(data.reading_w)} threshold readings, "
                 f"{model.spec.size} parameters ...")
        res = fitter.map(None if fresh else self.warm_start(model), maxiter=maxiter)
        progress(f"physio model: posterior mode after {res.iterations} iterations, -log posterior {res.neg_log_post:.1f} ({res.message})")
        draws, info, draws_day = np.zeros((0, model.spec.size)), None, str(data.days[-1])
        if samples and nuts:
            progress(f"physio model: {samples} posterior samples (NUTS, slow) ...")
            if seed_draws is not None and self.stored()["names"] != model.spec.names():
                seed_draws = None                         # (another parameter layout: not the same posterior)
            draws, nuts = fitter.nuts(res.z, num_warmup=warmup, num_samples=samples, checkpoint=checkpoint,
                                      tag=self.nuts_tag(data, samples, warmup), say=progress, target_ess=target_ess,
                                      max_minutes=max_minutes, seed_draws=seed_draws,
                                      whitening_to=self.processed / self.WHITENING,   # its own stages only (never read)
                                      structure=self.structure())
            info = nuts
            progress(f"physio model: NUTS {nuts['chains']} chains, {nuts['mean_steps']:.0f} steps a draw, "
                     f"step {nuts['step_size']:.2g}, {nuts['divergences']} divergences, worst R-hat "
                     f"{nuts['r_hat']:.3f}, smallest ESS {nuts['min_ess']:.0f}")
        elif samples:
            progress("physio model: the posterior mode only - ranges need NUTS (--samples N --nuts)")
        if not (samples and nuts) and not fresh:          # a mode-only refit keeps the last run's draws
            draws, draws_day = self.kept(model, self.stored(), data)
            progress(f"physio model: ranges from the run of {draws_day}" if len(draws) else
                     f"physio model: no ranges (no draws of this model): {self.FULL_RUN}")
        states = self.save(model, data, res.z, draws, draws_day, info, progress)
        return model, res, draws, states

    def refresh(self, progress=print):
        """today's ranges by a short NUTS run: the mode (warm-started, as `update`), then REFRESH_DRAWS draws a chain
        started from the stored run's last draw of each chain (aligned by name; new weeks' noise at the new mode),
        with the stored whitening's short warm-up (read, never written: the file is the last full run's) - or the
        full warm-up when older data changed since the stored fit. Its own checkpoint (REFRESH_CHECKPOINT, tag): never a full run's. Holds LOCK. Saved like `fit` (dated
        today). Returns (model, MapResult, draws, states) like `fit`, or None (said why): no stored draws of this
        model (a full run is needed), another NUTS run is going on, or the run ended (time, draws cap) short of
        REFRESH_ESS / REFRESH_R_HAT - then nothing is saved and the stored ranges keep their date."""
        from filelock import FileLock, Timeout
        try:
            with FileLock(str(self.processed / self.LOCK), timeout=0):
                try:
                    return self.refreshed(self.tracked(progress, "refresh"))
                finally:
                    self.untrack()
        except Timeout:
            progress("physio model: another NUTS run is going on: no refresh now")
            return None

    def refreshed(self, progress):
        """`refresh` (its lock held)."""
        from ..physio import CHAINS, PhysioFit, PhysioModel
        data = self.inputs()
        model = PhysioModel(data)
        st = self.stored()
        names = model.spec.names()
        if not self.same_model(model, st) or not len(st["samples"]) or len(st["samples"]) % CHAINS:
            progress(f"physio model: no stored posterior draws of this model - a full run is needed: {self.FULL_RUN}")
            return None
        seen = int(st["days"])
        older_same = (len(data.days) >= seen
                      and str(st.get("hist_key")) == self.history_key(data, max(seen - self.RECENT_DAYS, 0)))
        fitter = PhysioFit(model)
        progress(f"physio model {self.structure().rsplit('|', 1)[-1]}: {len(data.days)} days "
                 f"({len(data.days) - seen} new), {model.spec.size} parameters ...")
        res = fitter.map(model.spec.align(st["names"], st["z"])[0], maxiter=self.MAXITER)
        progress(f"physio model: posterior mode after {res.iterations} iterations, -log posterior {res.neg_log_post:.1f}")
        last = np.asarray(st["samples"]).reshape(CHAINS, -1, len(st["names"]))[:, -1]
        start = model.spec.align(st["names"], last)
        new = ~np.isin(names, st["names"])
        start[:, new] = res.z[new]                         # new weeks' noise: at the new mode
        if not older_same:
            progress("physio model: older data changed since the stored draws: the full warm-up")
        draws, info = fitter.nuts(res.z, num_warmup=self.WARMUP, num_samples=self.REFRESH_DRAWS,
                                  checkpoint=self.processed / self.REFRESH_CHECKPOINT,
                                  tag="refresh|" + self.nuts_tag(data, self.REFRESH_DRAWS, self.WARMUP), say=progress,
                                  target_ess=self.REFRESH_ESS, max_r_hat=self.REFRESH_R_HAT,
                                  max_minutes=self.REFRESH_MINUTES, start=start,
                                  seed=len(data.days), structure=self.structure(),
                                  whitening_from=self.processed / self.WHITENING if older_same else None)   # never written
        progress(f"physio model: NUTS {info['chains']} chains, {info['mean_steps']:.0f} steps a draw, "
                 f"step {info['step_size']:.2g}, {info['divergences']} divergences, worst R-hat "
                 f"{info['r_hat']:.3f}, smallest ESS {info['min_ess']:.0f}")
        if not info["reached"]:                            # draws that didn't settle would replace ranges that had
            progress(f"physio model: the refresh didn't reach ESS {self.REFRESH_ESS} / R-hat {self.REFRESH_R_HAT}: "
                     f"nothing saved, the ranges stay those of the run of {st['draws_day']}. The same command goes on "
                     f"from its checkpoint.")
            return None
        states = self.save(model, data, res.z, draws, str(data.days[-1]), info, progress)
        return model, res, draws, states

    def check(self, coverage: bool = False, folds: int | None = None, every: int | None = None,
              progress=print) -> dict:
        """`goodpush physio --backtest` / `--coverage` (goodpush.checks): the forecast of near-max days after
        each cut-off (PhysioCalibrated: the Model page's forecast error), or whether the 90 % ranges hold 90 % of
        held-out readings and best powers (PosteriorCoverage). Results in processed/ladder/ (CheckReport)."""
        from ..checks import EVERY_DAYS, FOLDS, CheckReport, PerformanceMarker, PhysioCalibrated, PointBacktest
        data = self.inputs()
        report = CheckReport(self.processed)
        if coverage:
            from ..checks.coverage import PosteriorCoverage
            cov = PosteriorCoverage(data, report.folder, self.processed / self.WHITENING, self.structure(),
                                    folds=folds or 24, every=every or 30)
            out = cov.run()
            for line in cov.lines(out["summary"]):
                progress(line)
            progress(f"saved to {cov.path}")
            return out
        marker = PerformanceMarker(data)
        daily = pd.read_parquet(self.processed / "daily.parquet").reindex(pd.DatetimeIndex(data.days))
        model = PhysioCalibrated(marker, daily["tss"].to_numpy(), data, say=progress)
        out = PointBacktest(model, marker, folds=folds or FOLDS, every=every or EVERY_DAYS).run(say=progress)
        report.save(model.name, "backtest", out)
        progress(f"saved to {report.folder} (the Model page shows it)")
        return out

    def nuts_tag(self, data, samples: int, warmup: int) -> str:
        """what a NUTS run samples: model version, chains, warm-up and the data (recovery z-scores aside: their
        baselines move a little every day). A checkpoint with another tag is not gone on from. Not the number of
        draws: a larger --samples goes on from a finished run's checkpoint (its draws cap is 4x --samples)."""
        from ..physio import CHAINS, PhysioFit
        from ..physio.model import NOISE_WEEKS
        return (f"{self.structure()}|{PhysioFit.SAMPLER}|{NOISE_WEEKS}|{CHAINS}|{warmup}|"
                f"{self.history_key(data, len(data.days))}")

    def picked(self, model, draws) -> dict | None:
        """daily trajectories (MAX_TRAJECTORIES, T) of evenly spaced draws; None without draws"""
        if not len(draws):
            return None
        return model.trajectories(draws[self.spaced(len(draws), min(len(draws), self.MAX_TRAJECTORIES))])

    def states(self, model, z, draws, T: dict | None = None) -> pd.DataFrame:
        tr = model.trajectory(z)
        out = pd.DataFrame({c: tr[c] for c in self.COLUMNS}, index=pd.DatetimeIndex(model.data.days, name="date"))
        if len(draws):
            T = T if T is not None else self.picked(model, draws)
            for c in self.COLUMNS:
                out[f"{c}_p05"], out[f"{c}_p95"] = np.percentile(T[c], [5, 95], axis=0)
        return out

    @staticmethod
    def spaced(n: int, k: int) -> np.ndarray:
        """k of n draw indices, evenly spaced (draws in chain order: every chain contributes)."""
        return np.linspace(0, n - 1, k).astype(int)

    @staticmethod
    def predict(model, z, days_ahead: int, durations) -> np.ndarray:
        """all-out power (W) per duration on the morning `days_ahead` days after the last data day, resting
        until then. z (D,) -> (len(durations),); draws (n, D) -> (n, len(durations))."""
        import jax
        B = model.data.hist_h.shape[1]
        rest, none = np.zeros((days_ahead + 1, B)), np.zeros(days_ahead + 1)
        Z = np.atleast_2d(z)
        ahead = jax.jit(jax.vmap(model.ahead, in_axes=(0, None, None, None)))
        P = model.batched(ahead, Z, rest, none, np.asarray(durations, float))     # BATCH draws at a time
        return P[:, days_ahead] if np.ndim(z) == 2 else P[0, days_ahead]


class PhysioDiagnostics:
    """How the latest physio fit is doing, for the app's Model page (processed/physio_diagnostics.json): the NUTS
    run's convergence (kept from the run that made the draws until another one), the posterior of today's
    thresholds, and the model's predictions next to what you measured - best powers and threshold readings.
    A diagnostic, not a verdict: n = 1, and an all-out effort is the model's call too."""
    FILE = "physio_diagnostics.json"
    TRACE_POINTS = 200               # per chain
    SLOWEST = 12                     # parameters listed (smallest ESS)
    BINS = 15
    TODAY = ("cp", "map", "lt1", "ftp", "w_prime", "pmax")
    EFFORT_DAYS = 365                # best powers shown (the scatter stays readable)
    READING_DAYS = 730
    # parameters the data may leave at their prior (`assumed`): name -> (what, short, why). One counts as assumed when
    # its posterior mean is within PRIOR_MEAN_SDS prior sds of the prior mean and its posterior sd above PRIOR_SD_SHARE
    # of the prior sd
    PRIOR_MEAN_SDS = 0.5
    PRIOR_SD_SHARE = 0.8
    MAYBE_ASSUMED = {"log_s_f": ("What fatigue costs you on the day", "assumed: your rides can't show this",
                                 "Your rides don't show how much fatigue takes off your all-out efforts (most likely "
                                 "they were ridden fresh): this number stays about where the model started (its "
                                 "assumption)."),
                     "x0[4]": (                         # durability's starting level (STATES[4])
                         "How well you hold power past an hour", "assumed: your rides can't show this",
                         "Steady all-out efforts longer than an hour are rare, so your rides can't show how well you "
                         "hold power past an hour: this stays about where the model started (its assumption)."),
                     "log_effect[2]": (                 # MAP from CP-MAP work (physio.model EFFECT_SD)
                         "What VO2max intervals do for your MAP", "assumed: from studies",
                         "Your rides can't tell the effect of VO2max work apart (there is little of it, or it came "
                         "together with more hours): the model uses what studies of trained cyclists found."),
                     "log_effect[3]": (                 # MAP from work above MAP
                         "What efforts above MAP do for your MAP", "assumed: from studies",
                         "Your rides hold too little work above MAP to show its effect: the model uses what studies of "
                         "trained cyclists found.")}
    # always assumed: the model's Pmax and why (sprints are almost never all-out; Pmax moved a lot between two equally
    # plausible shapes of the 'below capacity' distribution, 2026-11-10: notes)
    PMAX = ("pmax", "Pmax (sprint)", "a rough lower bound",
            "Your sprints are almost never all-out, so the model only knows Pmax is at least what you've sprinted; "
            "how far above it sits depends on assumptions. Take it as a rough lower bound.")

    def __init__(self, model, z, T: dict | None, draws=None):
        """draws (n, D): the posterior draws, if any (`assumed`)."""
        self.model, self.z, self.T = model, np.asarray(z), T
        self.draws = np.zeros((0, model.spec.size)) if draws is None else np.asarray(draws)

    def assumed(self) -> list[dict]:
        """what the model takes from its assumptions rather than your data, said by those numbers in the app: each
        MAYBE_ASSUMED parameter whose draws stay at its prior, and Pmax always. [] without draws."""
        if not len(self.draws):
            return []
        names, spec, out = self.model.spec.names(), self.model.spec, []
        for name, (what, short, why) in self.MAYBE_ASSUMED.items():
            i = names.index(name)
            mean, sd = float(self.draws[:, i].mean()), float(self.draws[:, i].std())
            if abs(mean - spec.mean[i]) <= self.PRIOR_MEAN_SDS * spec.sd[i] and sd > self.PRIOR_SD_SHARE * spec.sd[i]:
                out.append({"name": name, "what": what, "short": short, "why": why})
        name, what, short, why = self.PMAX
        return out + [{"name": name, "what": what, "short": short, "why": why}]

    @staticmethod
    def sampling(model, draws, info: dict) -> dict:
        """a NUTS run's convergence: tiles, the slowest parameters, and per chain a trace of the log posterior and
        of today's CP (draws in chain order, as PhysioFit.nuts returns them)"""
        import time
        chains = int(info["chains"])
        G = np.asarray(draws).reshape(chains, -1, model.spec.size)
        k = np.unique(np.linspace(0, G.shape[1] - 1, PhysioDiagnostics.TRACE_POINTS).astype(int))
        Z = G[:, k].reshape(-1, model.spec.size)
        lp = model.batched(model.log_posterior_draws, Z, model.spec.layout, model.arrays).reshape(chains, -1)
        cp = model.trajectories(Z)["cp"][:, model.T - 1].reshape(chains, -1)
        names, ess, r = model.spec.names(), np.asarray(info["ess_each"]), np.asarray(info["r_hat_each"])
        slow = np.argsort(ess)[:PhysioDiagnostics.SLOWEST]
        return {"date": time.strftime("%Y-%m-%d %H:%M"), "chains": chains, "draws": int(G.shape[1]),
                "warmup": int(info.get("warmup", 0)), "step_size": float(info.get("step_size", "nan")),
                "divergences": [int(d) for d in info.get("divergences_chain") or []],   # [] when not kept
                "steps": [round(float(v)) for v in info.get("steps_chain") or []],
                "r_hat": float(info["r_hat"]), "min_ess": float(info["min_ess"]), "parameters": len(names),
                "unjudged": int((~np.asarray(info["judged"], bool)).sum()),
                "ess_hist": PhysioDiagnostics.histogram(np.log10(np.maximum(ess, 1.0))),
                "slowest": [{"name": names[i], "ess": round(float(ess[i])), "r_hat": round(float(r[i]), 3),
                             "judged": bool(info["judged"][i]),
                             "chain_means": np.round(G[:, :, i].mean(1), 3).tolist()} for i in slow],
                "trace": {"draw": (k + 1).tolist(), "log_post": np.round(lp, 1).tolist(), "cp": np.round(cp, 1).tolist()}}

    @staticmethod
    def histogram(v) -> dict:
        n, e = np.histogram(np.asarray(v, float), bins=PhysioDiagnostics.BINS)
        return {"x": np.round((e[:-1] + e[1:]) / 2, 4).tolist(), "n": n.tolist()}

    def posterior(self) -> dict:
        """today's thresholds over the draws: histogram, median, 5-95 %"""
        if self.T is None:
            return {}
        out = {}
        for c in self.TODAY:
            v = np.asarray(self.T[c])[:, self.model.T - 1] / (1000.0 if c == "w_prime" else 1.0)
            q = np.percentile(v, [5, 50, 95])
            out[c] = {**self.histogram(v), "p05": round(float(q[0]), 1), "p50": round(float(q[1]), 1),
                      "p95": round(float(q[2]), 1)}
        return out

    def efforts(self) -> dict:
        """the last EFFORT_DAYS' ride best powers next to the model's all-out power that day (fatigue included):
        all-out efforts belong on the line, the others below it"""
        d, m = self.model.data, self.model
        X, F, _ = m.simulate(self.z)
        keep = d.effort_day >= len(d.days) - self.EFFORT_DAYS
        day = d.effort_day[keep]
        pred = np.asarray(m.all_out(self.z, np.asarray(X)[day], np.asarray(F)[day], d.effort_s[keep]))
        return {"date": [str(x) for x in d.days[day]], "s": d.effort_s[keep].astype(int).tolist(),
                "observed": np.round(d.effort_w[keep], 1).tolist(), "predicted": np.round(pred, 1).tolist(),
                "kind": [KINDS[k] for k in d.effort_kind[keep]]}

    def readings(self) -> dict:
        """each source's readings of the last READING_DAYS next to the model's figure (its bias included), and
        per source the learned bias and spread over all its readings"""
        d, m = self.model.data, self.model
        if not len(d.sources):
            return {"date": [], "source": [], "observed": [], "predicted": [], "sources": [], "model": {"date": []}}
        X, _, _ = m.simulate(self.z)
        p = m.spec.unpack(self.z)
        Q = np.asarray(m.quantities(np.asarray(X)[d.reading_day]))
        q = np.array([QUANTITIES.index(s.quantity) for s in d.sources])[d.reading_source]
        pred = Q[q, np.arange(len(q))] * np.exp(np.asarray(p["bias"])[d.reading_source])
        res = np.log(d.reading_w / pred)
        keep = d.reading_day >= len(d.days) - self.READING_DAYS
        sources = [{"name": s.name, "quantity": s.quantity, "bound": s.bound, "n": int((d.reading_source == i).sum()),
                    "bias_pct": round(100 * float(np.expm1(p["bias"][i])), 1),
                    "sd_pct": round(100 * float(np.std(res[d.reading_source == i])), 1)
                    if (d.reading_source == i).sum() > 1 else None}
                   for i, s in enumerate(d.sources)]
        first = max(len(d.days) - self.READING_DAYS, 0)
        lines = np.asarray(m.quantities(np.asarray(X)[first:]))          # the model's own figures, daily
        return {"date": [str(x) for x in d.days[d.reading_day[keep]]],
                "source": [d.sources[i].name for i in d.reading_source[keep]],
                "observed": np.round(d.reading_w[keep], 1).tolist(), "predicted": np.round(pred[keep], 1).tolist(),
                "sources": sources,
                "model": {"date": [str(x) for x in d.days[first:]],
                          **{q: np.round(lines[i], 1).tolist() for i, q in enumerate(QUANTITIES)}}}

    CHANGE_DAYS = 84                 # training(): the changes compared (12 weeks: a block)
    EXPLAINED = ("lt1", "cp", "map")

    def training(self) -> dict:
        """how much of the fitted path training alone gives: the model run again without its weekly unexplained
        changes (eta = 0). Per threshold (log watts): `share` of the fitted path's variance over time the
        training-only path reproduces (1 = all of it), the typical change over CHANGE_DAYS of the fitted path and
        of the training-only one (%), and how the two changes go together (`corr`). Changes that follow the fitted
        ones but are smaller: a forecast from planned training has the direction right and is too flat. {} with
        fewer than two blocks of days."""
        m = self.model
        h = self.CHANGE_DAYS
        if m.T < 2 * h:
            return {}
        o = m.spec.offsets["eta"]
        z0 = self.z.copy()
        z0[o:o + next(b.mean.size for b in m.spec.blocks if b.name == "eta")] = 0.0
        full, only = m.trajectory(self.z), m.trajectory(z0)
        out = {"days": h}
        for k in self.EXPLAINED:
            f, t = np.log(full[k]), np.log(only[k])
            df, dt = f[h:] - f[:-h], t[h:] - t[:-h]
            var = float(f.var())
            share = 1.0 - float(((f - f.mean()) - (t - t.mean())).var()) / var if var > 0 else None
            corr = float(np.corrcoef(df, dt)[0, 1]) if df.std() > 0 and dt.std() > 0 else None
            out[k] = {"share": None if share is None else round(share, 3), "corr": None if corr is None else round(corr, 3),
                      "change_pct": round(100 * float(df.std()), 2), "training_change_pct": round(100 * float(dt.std()), 2)}
        return out

    def save(self, processed, sampling: dict | None, draws_day: str = "") -> None:
        """`draws_day`: the data day of the run that made the draws ("" without draws): ranges older than the data
        say so in the app."""
        import json
        f = processed / self.FILE
        if sampling is None:                       # a daily update: the run that made the draws stays shown
            try:
                sampling = json.loads(f.read_text(encoding="utf-8")).get("sampling")
            except (OSError, ValueError):
                sampling = None
            if sampling is not None and self.T is None:
                sampling = None                    # its draws are gone (another model layout)
        from ..physio.model import VERSION
        out = {"day": str(self.model.data.days[-1]), "draws_day": draws_day, "sampling": sampling, "version": VERSION,
               "posterior": self.posterior(), "efforts": self.efforts(), "readings": self.readings(),
               "training": self.training(), "assumed": self.assumed()}
        f.write_text(json.dumps(out, ensure_ascii=True), encoding="utf-8")


class TrainingResponse:
    """What each kind of training does for you, from the fitted model (the Rider page):

    * gains - one extra typical session a week of a workout type for WEEKS weeks (on top of a repeat of your last
      WEEKS weeks), its work at today's watts for its textbook %FTP (planning.intensity) and easy recoveries: the
      change in your LT1, CP, MAP and 20 / 60-min power (fresh) at the end: the best fit's, or with posterior draws
      their median and 10-90 % range;
    * where each session's work lands in your intensity domains (today's LT1 / CP / MAP);
    * effects - each domain's training effects relative to the prior (near 1: your data barely moved it);
    * sources - which domains built your MAP, CP / MAP and LT1 / 1-h power over the last year (training effect x your
      average dose), with your hours per domain;
    * experience - hours per domain over the last two years (how much of each the model has seen).

    Written to processed/physio_response.json with every saved fit."""
    FILE = "physio_response.json"
    WEEKS = 12
    MAX_DRAWS = 40
    # (key, name, the session, its parts as (textbook fraction of FTP, hours)). Work plus easy recoveries; warm-ups
    # left out (the same for every session). No long-ride row: the model counts long rides as endurance hours
    EASY = 0.55
    WORKOUTS = (("endurance", "Endurance", "1.5 h steady, in a short or a long ride", ((0.65, 1.5),)),
                ("tempo", "Tempo", "3 x 15 min", ((0.82, 0.75), (EASY, 0.17))),
                ("sweetspot", "Sweet spot", "3 x 15 min", ((0.91, 0.75), (EASY, 0.17))),
                ("threshold", "Threshold", "2 x 20 min", ((0.98, 0.67), (EASY, 0.17))),
                ("vo2max", "VO2max intervals", "5 x 4 min", ((1.12, 0.33), (EASY, 0.25))),
                ("short", "Short hard intervals", "2 x 10 x 30/30 s", ((1.40, 0.17), (EASY, 0.25))))
    PRIOR_LIKE = 0.15                  # an effect within 15 % of its prior: barely moved by your data
    DOMAINS = (("z1", "Easy (below LT1)"), ("z2", "Tempo to threshold (LT1 - CP)"), ("z3", "VO2max / severe (above CP, power and heart rate)"),
               ("z4", "Short hard efforts (above MAP)"), ("sprint", "Sprints"), ("long", "Long rides (past 2 h)"))
    OUTCOMES = ("lt1", "cp", "map", "p20", "p60")
    SESSION_DAY = 2                    # the extra session's weekday in the repeated weeks

    def __init__(self, model, z, draws=None):
        from ..physio.model import DOSE_UNIT_H
        self.model, self.z = model, np.asarray(z, float)
        self.draws = None if draws is None or not len(draws) else np.asarray(draws, float)
        self.unit_h = DOSE_UNIT_H
        self.X = np.asarray(model.simulate(self.z)[0])                # fitted mornings (T, 5)
        self.today = {k: np.asarray(v) for k, v in model.capacity(self.X[-1]).items()}

    def baseline(self):
        """the last WEEKS weeks again: daily histograms (D, B) and long-ride hours (D,)."""
        n = 7 * self.WEEKS
        return self.model.data.hist_h[-n:].copy(), self.model.data.long_h[-n:].copy()

    def baseline_hr(self) -> np.ndarray:
        """the baseline days' time with heart rate by its evidence (D, B, E; the added session has none)."""
        return self.model.data.hr_hours[-7 * self.WEEKS:]

    def watts(self, frac: float) -> float:
        """today's watts for a textbook fraction of FTP."""
        from ..planning.intensity import FRAC, IntensityMap
        return float(np.interp(frac, FRAC, IntensityMap.knots_of(self.today)))

    def bin_of(self, frac: float) -> int:
        return int(np.argmin(np.abs(self.model.power_w - self.watts(frac))))

    def added(self, parts):
        """the baseline with one extra session a week: its parts ((fraction of FTP, hours), ...) on SESSION_DAY."""
        H, L = self.baseline()
        day = np.flatnonzero(np.arange(len(L)) % 7 == self.SESSION_DAY)
        for frac, h in parts:                                          # one or two parts
            H[day, self.bin_of(frac)] += h
        return H, L

    def end_of(self, z, H, L):
        """(OUTCOMES,) fresh at the end of a forecast with these days' training."""
        import jax.numpy as jnp
        from ..physio import PowerCurve
        X, _ = self.model.forecast(z, H, L, None, self.baseline_hr())
        c = self.model.capacity(X[-1])
        p = PowerCurve.power(jnp.array([1200.0, 3600.0]), c["cp"], c["w_prime"], c["pmax"], c["fade"])
        return jnp.stack([c["lt1"], c["cp"], c["map"], p[0], p[1]])

    def outcomes(self, Z, H, L) -> np.ndarray:
        """(sets, OUTCOMES) for parameter vectors Z (sets, D)."""
        import jax
        return self.model.batched(jax.vmap(self.end_of, in_axes=(0, None, None)), Z, H, L)    # BATCH sets at a time

    def parameter_sets(self) -> np.ndarray:
        """the posterior mode, then up to MAX_DRAWS draws (evenly spaced)."""
        if self.draws is None:
            return self.z[None]
        pick = PhysioRun.spaced(len(self.draws), min(len(self.draws), self.MAX_DRAWS))
        return np.vstack([self.z[None], self.draws[pick]])

    @staticmethod
    def day_doses(log_power, x, hist, long, hr):
        from ..physio import PhysioModel
        return PhysioModel.doses(log_power, PhysioModel.capacity(x), hist, long, hr)

    def doses(self, X, H, L, HR=None) -> np.ndarray:
        """(days, DOSES) doses of these days with the domain edges of states X (days, 5); HR: their time with heart
        rate by its evidence (days, B, E; default none)."""
        import jax
        import jax.numpy as jnp
        from ..physio.model import EVIDENCE
        HR = np.zeros(np.shape(H) + (len(EVIDENCE),)) if HR is None else HR
        return np.asarray(jax.vmap(self.day_doses, in_axes=(None, 0, 0, 0, 0))(
            jnp.log(self.model.power_w), jnp.asarray(X), jnp.asarray(H), jnp.asarray(L), jnp.asarray(HR)))

    def gains(self) -> list[dict]:
        Z = self.parameter_sets()
        H, L = self.baseline()
        base = self.outcomes(Z, H, L)
        now = np.repeat(self.X[-1:], len(L), axis=0)                    # today's domain edges
        out = []
        for key, label, session, parts in self.WORKOUTS:              # a handful of workout types
            H2, L2 = self.added(parts)
            g = 100 * (self.outcomes(Z, H2, L2) / base - 1)            # (sets, OUTCOMES) %
            work = parts[:1]
            Hw, Lw = H.copy(), np.zeros_like(L)
            Hw[self.SESSION_DAY, self.bin_of(work[0][0])] += work[0][1]
            HR = self.baseline_hr()
            hour = ((self.doses(now, Hw, Lw, HR) - self.doses(now, H, Lw, HR)).sum(0) * self.unit_h)[:5]
            share = hour / max(hour.sum(), 1e-9)
            row = {"key": key, "label": label, "session": session, "watts": round(self.watts(work[0][0])),
                   # with draws their median: the best fit alone can sit outside the draws' range (2026-11-12)
                   "gain": dict(zip(self.OUTCOMES, np.round(np.median(g[1:], axis=0) if len(g) > 1 else g[0], 2).tolist())),
                   "domains": {d: round(float(v), 2) for (d, _), v in zip(self.DOMAINS, share) if v > 0.02}}
            if len(g) > 1:
                row["range"] = {k: np.round(np.percentile(g[1:, i], [10, 90]), 2).tolist() for i, k in enumerate(self.OUTCOMES)}
            out.append(row)
        return out

    def sources(self, days: int = 365) -> dict:
        """per state (map, f_t, f_e): each domain's share of its training lift over the last `days` days (training
        effect x average dose), and the hours a week in each domain."""
        from ..physio.model import GAIN_EFFECT, GAIN_MASK, STATES
        d = self.model.data
        D = self.doses(self.X[-days:], d.hist_h[-days:], d.long_h[-days:], d.hr_hours[-days:]).mean(0)
        E = np.zeros(GAIN_MASK.shape)
        E[GAIN_MASK] = np.exp(np.asarray(self.model.spec.unpack(self.z)["log_effect"]))[GAIN_EFFECT]
        out = {}
        for s in ("map", "f_t", "f_e"):
            i = STATES.index(s)
            c = E[i] * D
            out[s] = {self.DOMAINS[j][0]: round(float(c[j] / max(c.sum(), 1e-12)), 3) for j in np.flatnonzero(GAIN_MASK[i])}
        out["hours_week"] = {k: round(float(v), 2) for (k, _), v in zip(self.DOMAINS, D * self.unit_h * 7)}
        return out

    def effects(self) -> dict:
        """per domain: what it trains (states) and its fitted effect / prior effect; prior_like when every one is
        within PRIOR_LIKE of its prior (your data barely moved it)."""
        from ..physio.model import GAIN_EFFECT, GAIN_INDEX, STATES
        fit = np.exp(np.asarray(self.model.spec.unpack(self.z)["log_effect"]))[GAIN_EFFECT]
        prior = np.exp(np.asarray(self.model.spec.unpack(self.model.spec.mean)["log_effect"]))[GAIN_EFFECT]
        out = {}
        for k, (s, j) in enumerate(zip(*GAIN_INDEX)):                 # the gain matrix's entries
            d = out.setdefault(self.DOMAINS[j][0], {"trains": {}, "prior_like": True})
            d["trains"][STATES[s]] = round(float(fit[k] / prior[k]), 2)
            d["prior_like"] &= bool(abs(fit[k] / prior[k] - 1) <= self.PRIOR_LIKE)
        return out

    def experience(self, days: int = 730) -> dict:
        """hours per domain over the last `days` days (at each day's own edges)."""
        d = self.model.data
        D = self.doses(self.X[-days:], d.hist_h[-days:], d.long_h[-days:], d.hr_hours[-days:]).sum(0)
        return {k: round(float(v), 1) for (k, _), v in zip(self.DOMAINS, D * self.unit_h)}

    def build(self) -> dict:
        return {"date": str(self.model.data.days[-1]), "weeks": self.WEEKS,
                "draws": 0 if self.draws is None else len(self.draws), "domains": dict(self.DOMAINS),
                "workouts": self.gains(), "sources": self.sources(), "effects": self.effects(),
                "experience": self.experience()}

    def save(self, processed) -> None:
        import json
        (Path(processed) / self.FILE).write_text(json.dumps(self.build()))
