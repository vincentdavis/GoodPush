"""Build the activity, daily and weekly analysis tables from the ingested data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Project
from .ingest import SUMMARY_TABLE
from .io import store
from .io.garmin import COLUMNS as WELLNESS_COLS, load_wellness
from .io.google_health import GoogleHealthDays
from .io.hrv import NightHrv
from .models.ftp_model import TEST_WEIGHT, FtpModel
from .progress import Progress
from .io.strava import mood_flags, project_activities, sport_class, weight_history
from .metrics import load as L
from .metrics.power import MMP_DURATIONS
from .metrics.power_check import PowerCheck
from .models.cp import cp_series, daily_envelope, estimate_ftp_series, rolling_envelope

ZONE_COLS = [f"pz{i + 1}_s" for i in range(len(L.POWER_ZONES))]
HRZ_COLS = [f"hz{i + 1}_s" for i in range(len(L.HR_ZONES))]


# ---------------------------------------------------------------------------
def build_activities(proj: Project, progress=print) -> tuple[pd.DataFrame, dict]:
    acts = project_activities(proj)
    streams = store.load(proj.processed, SUMMARY_TABLE)
    if streams is not None and not streams.empty:
        streams = streams.copy()
        streams["id"] = streams["id"].astype(str)
        streams = streams.drop(columns=[c for c in ("source", "source_key") if c in streams])
        acts = acts.merge(streams, on="id", how="left")
    for c in ("has_power", "n_seconds", "moving_s", "np", "avg_hr", "hr_hist", "power_hist",
              "best20_hr", "max_hr", "ef", "decoupling", "work_kj", "avg_power"):
        if c not in acts:
            acts[c] = np.nan
    for d in MMP_DURATIONS:
        if f"mmp_{d}" not in acts:
            acts[f"mmp_{d}"] = np.nan
    acts["has_power"] = acts["has_power"].fillna(False).astype(bool)
    if "fit_sub_sport" in acts:
        sub = acts["fit_sub_sport"].fillna("").astype(str)
        acts["indoor"] = acts["indoor"] | sub.isin(["indoor_cycling", "virtual_activity"])
    acts["sport"] = acts["type"].map(sport_class)
    acts["start_local"] = acts["start_utc"].dt.tz_convert(proj.tz)
    acts["date"] = acts["start_local"].dt.tz_localize(None).dt.normalize()
    # duration: prefer stream moving time, else CSV moving time, else elapsed
    acts["duration_s"] = acts["moving_s"].where(acts["moving_s"] > 0, acts["moving_s_csv"])
    acts["duration_s"] = acts["duration_s"].fillna(acts["elapsed_s"]).fillna(0)
    # use CSV weighted power when the file had no power stream
    use_csv = ~acts["has_power"] & acts["wap_csv"].gt(30) & acts["sport"].eq("ride")
    acts["np_used"] = acts["np"].where(acts["has_power"], acts["wap_csv"].where(use_csv))
    bad, good = zip(*[mood_flags(f"{n} {d} {p}") for n, d, p in
                      zip(acts["name"], acts["description"], acts["private_note"])])
    acts["mood_bad"], acts["mood_good"] = bad, good

    # --- athlete constants -------------------------------------------------
    ath = proj.athlete
    info = {}
    rides = acts[acts["sport"].eq("ride")]
    mh = pd.concat([acts["max_hr"], acts["max_hr_csv"]]).dropna()
    mh = mh[(mh > 120) & (mh < 225)]
    info["max_hr"] = float(ath.get("max_hr") or (np.percentile(mh, 99.5) if len(mh) else 190))
    b20 = rides["best20_hr"].dropna()
    b20 = b20[(b20 > 100) & (b20 < info["max_hr"])]
    info["lthr"] = float(ath.get("lthr") or (np.percentile(b20, 98) if len(b20) > 10
                                                 else 0.9 * info["max_hr"]))
    info["lthr_source"] = "config" if ath.get("lthr") else "98th pct of best 20-min HR"
    info["resting_hr"] = float(ath.get("resting_hr") or 0)  # filled after wellness load

    # --- weight ----------------------------------------------------------------
    wh = weight_history(acts)
    info["weight_source"] = "config" if ath.get("weight_kg") else ("strava" if len(wh) else "none")
    if wh.nunique() <= 1 and ath.get("weight_kg"):
        wh = pd.Series(dtype=float)
    info["weight_history"] = wh
    return acts, info


def build_ftp(proj: Project, acts: pd.DataFrame, days: pd.DatetimeIndex, wellness: pd.DataFrame | None = None,
              lthr: float | None = None, progress=print) -> pd.DataFrame:
    fcfg = proj.cfg["ftp"]
    day_env = daily_envelope(acts[acts["sport"].eq("ride")], MMP_DURATIONS, days)
    method = fcfg.get("auto_method", "fitted")
    auto = estimate_ftp_series(day_env, fcfg.get("window_days", 90), method if method in ("cp", "p20") else "cp",
                               fcfg.get("fallback", 200), fcfg.get("floor_days", 180))
    out = auto.copy()
    hist = fcfg.get("history") or {}
    fit = method in ("fitted", "model") and wellness is not None and bool(lthr)
    cp = FtpModel.physio_cp(proj.processed) if fit or method == "model" else None   # the model's LT2 (its last fit)
    if fit:                                        # fitted to Garmin's FTP, yours and your tests
        rides = acts[acts["sport"].eq("ride") & acts["has_power"].fillna(False).astype(bool)]
        rhr = wellness["rhr"].rolling(28, min_periods=5).median() if "rhr" in wellness else pd.Series(dtype=float)
        log_bests, hr = FtpModel.log_bests(rides, days), FtpModel.hr_threshold(rides, days, rhr, lthr)
        labels, tests = FtpModel.labels(wellness, hist), FtpModel.tests(FtpModel.last_ramps(proj.processed), cp)
        tests = tests[~tests.index.isin(pd.DatetimeIndex([pd.Timestamp(k) for k in hist]))]   # your entry wins
        labels = pd.concat([labels[~labels.index.isin(tests.index)], tests]).sort_index()      # a test over Garmin's
        weights = pd.Series(TEST_WEIGHT, index=tests.index)
        model = FtpModel.cached(proj.processed, labels, (log_bests, hr, days), lthr, progress, weights)
        if model is not None:
            fitted = pd.Series(model.predict_from(log_bests, hr), index=days).ffill()
            out["ftp_auto"] = fitted.fillna(out["ftp_auto"])
            out["ftp_cp"] = auto["ftp_auto"]
    if method == "model" and cp is not None:       # FTP = the model's LT2; before its first day, the above
        out["ftp_auto"] = cp.reindex(days.union(cp.index)).ffill().reindex(days).round().fillna(out["ftp_auto"])
    manual = pd.Series({pd.Timestamp(k): float(v) for k, v in hist.items()}).sort_index() \
        if hist else pd.Series(dtype=float)
    if fcfg.get("mode") == "manual" and len(manual):
        # each entry holds from its date on; before the first one the automatic estimate stays
        m = manual.reindex(days.union(manual.index)).ffill().reindex(days)
        out["ftp"] = m.fillna(out["ftp_auto"])
    else:
        out["ftp"] = out["ftp_auto"]
        out["ftp_manual"] = manual.reindex(days.union(manual.index)).ffill().reindex(days) \
            if len(manual) else np.nan
    # performance markers with shorter windows (more responsive)
    env42 = rolling_envelope(day_env, 42)
    cp42 = cp_series(env42)
    out["cp42"] = cp42["cp"]
    out["wprime42"] = cp42["wprime"]
    for d in (5, 60, 300, 1200):
        out[f"best{d}_42"] = env42[f"mmp_{d}"]
        out[f"best{d}_90"] = rolling_envelope(day_env, 90)[f"mmp_{d}"]
    out["day_env_ok"] = day_env.notna().any(axis=1)
    return out, day_env


def add_load(proj: Project, acts: pd.DataFrame, ftp: pd.DataFrame, info: dict) -> pd.DataFrame:
    rhr = info["resting_hr"] or 55.0
    lthr = info["lthr"]
    acts = acts.copy()
    acts["ftp"] = ftp["ftp"].reindex(acts["date"]).to_numpy()
    acts["if_"] = acts["np_used"] / acts["ftp"]
    with np.errstate(all="ignore"):
        acts["tss_power"] = acts["duration_s"] * (acts["np_used"] / acts["ftp"]) ** 2 / 3600.0 * 100.0
    acts.loc[~(acts["ftp"] > 0), "tss_power"] = np.nan
    from .metrics.power import HR_BINS, POWER_BINS
    PH = L.hist_matrix(acts["power_hist"], POWER_BINS)
    HH = L.hist_matrix(acts["hr_hist"], HR_BINS)
    hr_hist_tss = L.hr_tss_matrix(HH, rhr, lthr)
    with np.errstate(all="ignore"):
        f_avg = np.clip((acts["avg_hr_csv"].to_numpy(float) - rhr) / (lthr - rhr), 0, 1.3) \
            if lthr and lthr > rhr else np.full(len(acts), np.nan)
    hr_avg_tss = acts["duration_s"].to_numpy(float) / 3600.0 * f_avg ** 2 * 100.0 * 1.05
    acts["tss_hr"] = np.where(np.isfinite(HH).all(1), hr_hist_tss, hr_avg_tss)
    dflt = proj.cfg["load"]["default_tss_per_hour"]
    acts["tss_default"] = acts["duration_s"] / 3600 * acts["sport"].map(
        lambda s: dflt.get(s, dflt.get("other", 30)))
    tss = acts["tss_power"].where(acts["sport"].eq("ride") & acts["tss_power"].notna())
    tss = tss.fillna(acts["tss_hr"]).fillna(acts["tss_default"])
    acts["tss"] = tss.clip(upper=600)
    acts["tss_source"] = np.select(
        [acts["sport"].eq("ride") & acts["tss_power"].notna(), acts["tss_hr"].notna()],
        ["power", "hr"], "duration")
    acts[ZONE_COLS] = L.zone_seconds_matrix(PH, L._PCENTERS, acts["ftp"].to_numpy(float), L.POWER_ZONES)
    acts[HRZ_COLS] = L.zone_seconds_matrix(HH, L._HCENTERS, np.full(len(acts), float(lthr or np.nan)),
                                           L.HR_ZONES)
    above = acts[["pz5_s", "pz6_s", "pz7_s"]].sum(axis=1, min_count=1)
    acts["hard"] = (above >= 600) | (acts["if_"] >= 0.85) | (acts["pz4_s"] >= 1200)
    acts["long"] = acts["duration_s"] >= 3 * 3600
    # efficiency factor is only comparable on steady aerobic rides
    acts["aerobic_ride"] = (acts["sport"].eq("ride") & acts["has_power"] & (acts["if_"] < 0.80)
                            & (acts["duration_s"] >= 2700) & acts["ef"].notna())
    return acts


WORKOUT_VERSION = 2


def classify_rides(proj: Project, acts: pd.DataFrame, lthr: float, progress=print) -> pd.DataFrame:
    """Workout type for every ride (cached; needs the 1 Hz streams kept by ingest)."""
    from .models.workouts import classify
    cache = store.load(proj.processed, "workout_types")
    have = {}
    if cache is not None and len(cache):
        cache["id"] = cache["id"].astype(str)
        have = {k: v for k, v in zip(cache["id"], cache["key"])}
    from .io import streamstore
    rides = acts[acts["sport"].eq("ride")]
    rows, n_new = [], 0
    todo = []
    for r in rides.itertuples():
        key = f"{round(float(r.ftp)) if r.ftp == r.ftp else 0}-{round(lthr)}-v{WORKOUT_VERSION}"
        if have.get(r.id) != key:
            todo.append((r, key))
    # one read for all rides that need (re)classifying (e.g. after an FTP change: all of them)
    streams = streamstore.read_many(proj.processed, [r.id for r, _ in todo], "ride",
                                    ["power", "heart_rate"]) if todo else {}
    for r, key in todo:
        z = streams.get(str(r.id))
        p, hr = (z["power"], z["heart_rate"]) if z else (None, None)
        if not r.has_power:
            p = None
        c = classify(p, r.ftp, r.name if isinstance(r.name, str) else "", r.duration_s,
                     r.np_used if r.np_used == r.np_used else None, hr=hr, lthr=lthr,
                     indoor=bool(r.indoor))
        c.update(id=r.id, key=key)
        rows.append(c)
        n_new += 1
    if n_new:
        progress(f"classified {n_new} rides by workout type")
    new = pd.DataFrame(rows)
    if cache is not None and len(cache):
        out = pd.concat([cache[~cache["id"].isin(new["id"] if len(new) else [])], new],
                        ignore_index=True) if len(new) else cache
    else:
        out = new
    if len(out):
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].astype(str).replace({"None": "", "nan": ""})
        store.save(out.reset_index(drop=True), proj.processed, "workout_types")
    return out


def build_daily(proj: Project, acts: pd.DataFrame, ftp: pd.DataFrame, wellness: pd.DataFrame,
                days: pd.DatetimeIndex) -> pd.DataFrame:
    lcfg = proj.cfg["load"]
    use = acts if lcfg.get("include_sports", "all") == "all" else acts[acts["sport"].eq("ride")]
    g = use.groupby("date")
    d = pd.DataFrame(index=days)
    d["tss"] = g["tss"].sum().reindex(days).fillna(0)
    d["hours"] = (g["duration_s"].sum() / 3600).reindex(days).fillna(0)
    rides = acts[acts["sport"].eq("ride")]
    gr = rides.groupby("date")
    d["ride_hours"] = (gr["duration_s"].sum() / 3600).reindex(days).fillna(0)
    d["ride_km"] = (gr["distance_m"].sum() / 1000).reindex(days).fillna(0)
    d["kj"] = gr["work_kj"].sum().reindex(days).fillna(0)
    d["n_acts"] = g.size().reindex(days).fillna(0)
    d["hard"] = gr["hard"].sum().reindex(days).fillna(0)
    d["long"] = gr["long"].sum().reindex(days).fillna(0)
    d["indoor_h"] = (rides[rides["indoor"]].groupby("date")["duration_s"].sum() / 3600
                     ).reindex(days).fillna(0)
    for c in ZONE_COLS + HRZ_COLS:
        d[c] = gr[c].sum().reindex(days).fillna(0)
    d["mood_bad"] = g["mood_bad"].sum().reindex(days).fillna(0)
    from .models.workouts import TYPES as WTYPES
    wk_col = rides["workout"] if "workout" in rides else pd.Series("", index=rides.index)
    for t in WTYPES:
        d[f"n_{t}"] = (wk_col == t).groupby(rides["date"]).sum().reindex(days).fillna(0)
    d["relative_effort"] = g["relative_effort"].sum().reindex(days)
    # "tested" performance: 20-min-equivalent power on days with a hard effort
    eq = pd.concat([rides["mmp_1200"], 0.88 * rides["mmp_480"], 0.82 * rides["mmp_300"]],
                   axis=1).max(axis=1)
    d["eq20_test"] = eq.where(rides["hard"] & rides["has_power"]).groupby(rides["date"]).max() \
        .reindex(days)
    aer = acts[acts["aerobic_ride"]]
    d["ef"] = aer.groupby("date")["ef"].mean().reindex(days)
    d["decoupling"] = aer.groupby("date")["decoupling"].mean().reindex(days)

    d = d.join(L.pmc(d["tss"], lcfg["ctl_days"], lcfg["atl_days"]).drop(columns="tss"))
    d = d.join(ftp)
    # wellness + personal baselines (28-day rolling, excluding today)
    if wellness is not None and len(wellness):
        w = wellness.reindex(days).rename(columns={"weight": "weight_garmin"})
        d = d.join(w)
    based = ("sleep_h", "sleep_score", "rhr", "hrv", "stress", "body_battery_max", "readiness", "hrv_deep", "spo2")
    missing = [c for c in dict.fromkeys([*WELLNESS_COLS, *based]) if c not in d and c != "weight"]
    d = d.join(pd.DataFrame(np.nan, index=d.index, columns=missing))
    v = d[list(based)]                                  # baselines for all of them at once
    base = v.rolling(28, min_periods=10).median().shift(1)
    sd = v.rolling(28, min_periods=10).std().shift(1)
    d = pd.concat([d, base.add_suffix("_base"), ((v - base) / sd.replace(0, np.nan)).add_suffix("_z"),
                   d["ef"].rolling(42, min_periods=4).median().shift(1).rename("ef_base")], axis=1)
    # events
    d["event"] = ""
    for ev in proj.cfg.get("events", []) or []:
        try:
            s = pd.Timestamp(ev.get("date"))
            e = pd.Timestamp(ev.get("end") or ev.get("date"))
        except Exception:
            continue
        if "example" in str(ev.get("note", "")):
            continue
        d.loc[(d.index >= s) & (d.index <= e), "event"] = f"{ev.get('type', '')}: {ev.get('note', '')}"
    d.index.name = "date"
    return d.copy()                # one block in memory after the column-by-column build


def build_weekly(d: pd.DataFrame) -> pd.DataFrame:
    wk = d.resample("W-MON", label="left", closed="left")  # weeks starting Monday
    W = pd.DataFrame({
        "hours": wk["hours"].sum(), "ride_hours": wk["ride_hours"].sum(),
        "tss": wk["tss"].sum(), "kj": wk["kj"].sum(), "km": wk["ride_km"].sum(),
        "hard": wk["hard"].sum(), "long": wk["long"].sum(), "indoor_h": wk["indoor_h"].sum(),
        "rest_days": wk["tss"].apply(lambda x: int((x < 10).sum())),
        "ctl": wk["ctl"].last(), "atl": wk["atl"].last(), "tsb_min": wk["tsb"].min(),
        "acwr_max": wk["acwr"].max(), "monotony": wk["monotony"].last(), "strain": wk["strain"].last(),
        "ftp": wk["ftp"].last(), "cp42": wk["cp42"].last(), "wprime42": wk["wprime42"].last(),
        "perf_test": wk["eq20_test"].max(),
        "best1200_42": wk["best1200_42"].last(), "best300_42": wk["best300_42"].last(),
        "ef": wk["ef"].mean(), "decoupling": wk["decoupling"].mean(),
        "sleep_h": wk["sleep_h"].mean(), "sleep_score": wk["sleep_score"].mean(),
        "rhr": wk["rhr"].mean(), "hrv": wk["hrv"].mean(), "rhr_z": wk["rhr_z"].mean(),
        "hrv_z": wk["hrv_z"].mean(), "mood_bad": wk["mood_bad"].sum(),
        "sleep_z": wk["sleep_h_z"].mean(), "stress": wk["stress"].mean(),
        "readiness": wk["readiness"].mean(), "bedtime_h": wk["bedtime_h"].mean(),
        "vo2max_garmin": wk["vo2max_garmin"].last(),
        "event": wk["event"].agg(lambda x: "; ".join(sorted(set(v for v in x if v)))),
    })
    for c in ZONE_COLS + HRZ_COLS + [c for c in d.columns if c.startswith("n_")]:
        W[c] = wk[c].sum()
    tot = W[ZONE_COLS].sum(axis=1).replace(0, np.nan)
    W["f_z12"] = (W["pz1_s"] + W["pz2_s"]) / tot
    W["f_z3"] = W["pz3_s"] / tot
    W["f_z4"] = W["pz4_s"] / tot
    W["f_z5p"] = (W["pz5_s"] + W["pz6_s"] + W["pz7_s"]) / tot
    W["ramp"] = W["ctl"].diff()
    # performance marker: best tested 20-min-equivalent over the last 6 weeks.
    # NaN when nothing hard was ridden in 6 weeks ("untested" is not "detrained").
    W["perf"] = W["perf_test"].rolling(6, min_periods=1).max()
    W.index.name = "week"
    return W


def run_analysis(proj: Project, progress=print) -> dict:
    """the analysis (`analyse`) on the physio model's own daily states. Without states of this model version (a
    new version: the old fit's are set aside; or no fit yet) the model is fitted once on tables made without
    them, then everything is analysed again on its states - no input comes from another model's fit
    (models.physio.PhysioRun.reset; 2026-11-27)."""
    try:
        from .models.physio import PhysioRun
        run = PhysioRun(proj)
    except ImportError:                              # no physio extra: no model, nothing comes from one
        return analyse(proj, progress)
    first = run.reset(progress)
    r = analyse(proj, progress)
    run.mark(not first)
    if first:
        try:
            progress("physio model: its first fit of this version, then the analysis again on its daily states ...")
            run.fit(progress=progress)
        except Exception as e:  # noqa: BLE001 - e.g. too little data: the analysis stands without the model
            progress(f"physio model: no first fit ({e}) - the analysis without its states")
            return r
        r = analyse(proj, progress)
        run.mark(True)
    return r


def analyse(proj: Project, progress=print) -> dict:
    """the activities, daily and weekly tables (saved in processed/) and the athlete's info."""
    acts, info = build_activities(proj, progress)
    try:                                             # heart rate at the end of each ride's best efforts: all-out?
        acts = PowerCheck.effort_heart_rates(proj.processed, acts)
    except Exception as e:  # noqa: BLE001 - the analysis works without them
        progress(f"effort heart rates skipped: {e}")
    acts = PowerCheck.of(proj).apply(acts, info["lthr"])   # power you marked wrong, or heart rate says is: set aside
    n_bad = acts["power_ignored"].value_counts()
    if n_bad.drop("", errors="ignore").sum():
        progress("power set aside: " + ", ".join(f"{int(v)} ride(s) {k}" for k, v in n_bad.items() if k))
    Progress.step(progress, "Analysing", 1, 5)
    if acts["date"].isna().all():
        raise RuntimeError("no dated activities found")
    start = acts["date"].min()
    end = max(acts["date"].max(), pd.Timestamp.now(tz=proj.tz).tz_localize(None).normalize())
    days = pd.date_range(start, end, freq="D")

    wellness, inv = load_wellness(proj.path("garmin"), proj.tz, cache_dir=proj.processed,
                                   progress=progress)
    gh = GoogleHealthDays.load(proj.path("google_health"), proj.tz)      # Fitbit, via Google Health
    if len(gh):
        wellness = GoogleHealthDays.merge(wellness, gh)
        inv["sources"]["Google Health (Fitbit)"] = {"days": int(len(gh))}
    NightHrv.update(proj.path("garmin"), proj.processed, proj.tz, progress)   # every night's 5-min HRV
    nh = NightHrv.quantiles(proj.processed)
    if len(nh):
        wellness = wellness.join(nh, how="outer")
        inv["sources"]["overnight HRV readings"] = {"nights": int(nh["hrv_median"].notna().sum())}
    if not info["resting_hr"]:
        r = wellness["rhr"].dropna() if "rhr" in wellness else pd.Series(dtype=float)
        info["resting_hr"] = float(r.median()) if len(r) > 5 else 55.0
        info["resting_hr_source"] = "garmin" if len(r) > 5 else "default"
    else:
        info["resting_hr_source"] = "config"

    Progress.step(progress, "Analysing", 2, 5)
    ftp, day_env = build_ftp(proj, acts, days, wellness, info["lthr"], progress)
    acts = add_load(proj, acts, ftp, info)
    wt = classify_rides(proj, acts, info["lthr"], progress)
    if len(wt):
        cols = ["id", "type", "type_stream", "type_name", "vo2_s", "thr_s", "ss_s", "tempo_s",
                "intermittent_s", "anaer_reps"]
        wt = wt[[c for c in cols if c in wt]].rename(columns={"type": "workout"})
        acts = acts.merge(wt, on="id", how="left")
    else:
        acts["workout"] = ""
    acts.loc[~acts["sport"].eq("ride"), "workout"] = ""
    try:                                             # every ride's intervals (the physio model's HR readings)
        from .models.intervals import IntervalLog
        IntervalLog.update(proj.processed, acts, progress)
    except Exception as e:  # noqa: BLE001 - nice to have; the rest of the analysis doesn't need it
        progress(f"intervals skipped: {e}")
    Progress.step(progress, "Analysing", 3, 5)
    # DFA alpha1 (alphaHRV) warm-up / session values and their baselines
    from . import alpha as AL
    try:
        af = AL.features(proj, acts)
    except Exception as e:  # noqa: BLE001
        progress(f"alpha1 features skipped: {e}")
        af = pd.DataFrame()
    if len(af):
        af.to_csv(AL.path_table(proj), index=False)
        keep = [c for c in af.columns if c.startswith(("a1_", "hrvt", "ramp_"))]
        sid = acts["id"].astype(str).str.replace(r"\.0$", "", regex=True)
        acts = acts.drop(columns=[c for c in keep if c in acts]).assign(_sid=sid) \
            .merge(af[["id"] + keep].rename(columns={"id": "_sid"}), on="_sid", how="left").drop(columns="_sid")
        progress(f"alpha1: {int(af['a1_warm'].notna().sum())} rides with a usable warm-up")
    try:                                             # ramp tests from the plan: top, recovery, HR-based thresholds
        from .models.ramp import RampTest
        acts = RampTest.add_to(proj, acts, info["lthr"], progress)
    except Exception as e:  # noqa: BLE001 - the analysis works without them
        progress(f"ramp tests skipped: {e}")
    try:                                             # LSCTs from the plan: power at set heart rates, HR recovery
        from .models.lsct import LSCT
        acts = LSCT.add_to(proj, acts, progress)
    except Exception as e:  # noqa: BLE001 - the analysis works without them
        progress(f"LSCTs skipped: {e}")
    try:                                             # all-out steps from the plan: the power curve's own points
        from .models.all_out import AllOutStep
        acts = AllOutStep.add_to(proj, acts, progress)
    except Exception as e:  # noqa: BLE001 - the analysis works without them
        progress(f"all-out steps skipped: {e}")
    if "effort_hr_60" in acts:                       # merged first thing (PowerCheck)
        from .metrics.effort_hr import EffortHeartRate
        mx = EffortHeartRate.maxed(acts, info["max_hr"]).sum()
        progress("all-out by heart rate: " + ", ".join(f"{d // 60} min {int(c)}" for d, c in mx.items()))
    Progress.step(progress, "Analysing", 4, 5)
    daily = build_daily(proj, acts, ftp, wellness, days)
    ad = AL.to_daily(af, days)
    daily = pd.concat([daily.drop(columns=ad.columns, errors="ignore"), ad.astype(float)], axis=1)
    weekly = build_weekly(daily)

    # weight series
    wh = info.pop("weight_history")
    w_cfg = proj.athlete.get("weight_kg") or 0
    gw = wellness["weight"].dropna() if "weight" in wellness else pd.Series(dtype=float)
    if not w_cfg and gw.nunique() >= 2:          # Garmin weight log beats a constant Strava value
        wh = gw.groupby(level=0).last()
        info["weight_source"] = "garmin"
    if w_cfg:
        wd = pd.Series(float(w_cfg), index=days)
    elif len(wh):
        wd = wh.reindex(days.union(wh.index)).ffill().bfill().reindex(days)
    else:
        wd = pd.Series(float(w_cfg) if w_cfg else np.nan, index=days)
    daily["weight"] = wd
    info["weight_kg"] = float(wd.iloc[-1]) if wd.notna().any() else np.nan
    info["wellness_inventory"] = {k: v for k, v in inv.items() if k != "keys"}
    info["wellness_days"] = int(len(wellness))

    for name, df in (("activities", acts), ("daily", daily), ("weekly", weekly)):
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].astype(str)
        if "start_utc" in out:
            out["start_utc"] = out["start_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
            out["start_local"] = out["start_local"].dt.tz_localize(None)
        store.save(out, proj.processed, name)
    Progress.step(progress, "Analysing", 5, 5)
    progress(f"analysis tables: {len(acts)} activities, {len(daily)} days, {len(weekly)} weeks, "
             f"{info['wellness_days']} wellness days")
    from .models import markers                       # your illness markers, when there's something new to learn from
    try:
        markers.learn_if_due(proj, progress)
    except Exception as e:  # noqa: BLE001 - never let the markers break the analysis
        progress(f"illness markers skipped: {e}")
    return {"acts": acts, "daily": daily, "weekly": weekly, "info": info,
            "day_env": day_env, "wellness": wellness}
