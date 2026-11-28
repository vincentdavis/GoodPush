"""Physio state-space model: curve, simulation, posterior, fit, inputs from goodpush tables."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from goodpush.physio import PhysioData, PhysioFit, PhysioModel, PowerCurve, Source  # noqa: E402
from goodpush.physio.model import EFFECT_STATE, NOISE_SD, STATES, ParamSpec  # noqa: E402

SOURCES = (Source("garmin_ftp", "ftp", 0.05, 0.04), Source("alpha1_lt1", "lt1", 0.08, 0.08))


@pytest.fixture(autouse=True)
def short_trees(monkeypatch):
    """NUTS trees of at most 31 steps (real runs 1023): a toy run's few warm-up steps leave the step size untuned, so
    every draw went to the tree limit - 5-7 s a block, most of the suite's time. These tests check what a run does."""
    monkeypatch.setattr(PhysioFit, "MAX_STEPS", 2 ** 5 - 1)


def synthetic(n_days=84, seed=0, hr=False):
    """3 rides a week: 1.5 h at 170 W, with 4 x 5 min at 300 W on one of them; a few readings. hr: also steady
    stretches for the heart-rate curve (each ride at ~75 % of LTHR, the hard day's reps at ~95 %)."""
    rng = np.random.default_rng(seed)
    days = np.datetime64("2026-01-05") + np.arange(n_days)
    power = (np.arange(150) + 0.5) * 10
    hist = np.zeros((n_days, 150))
    ride = np.isin(np.arange(n_days) % 7, [1, 3, 5])
    hist[ride, 17] = 1.5
    hard = np.arange(n_days) % 7 == 3
    hist[hard, 30] = 20 / 60
    eff_day = np.repeat(np.nonzero(ride)[0], 3)
    eff_s = np.tile([60.0, 300.0, 1200.0], ride.sum())
    curve = np.asarray(PowerCurve.power(eff_s, 250.0, 18000.0, 950.0, 0.08))
    eff_w = curve * np.where(np.repeat(hard[ride], 3), 0.98, rng.uniform(0.55, 0.8, eff_s.size))
    steady = {}
    if hr:
        rd, hd = np.nonzero(ride)[0], np.nonzero(hard)[0]
        steady = dict(hr_day=np.r_[rd, hd, hd], hr_w=np.r_[rng.normal(170, 8, rd.size), rng.normal(300, 10, 2 * hd.size)],
                      hr_x=np.r_[rng.normal(0.76, 0.03, rd.size), rng.normal(0.95, 0.02, 2 * hd.size)],
                      hr_rep=np.r_[np.zeros(rd.size, bool), np.ones(2 * hd.size, bool)], hr_rest=0.28, hr_fatigue_ref=0.5)
    return PhysioData(days=days, power_w=power, hist_h=hist, long_h=np.zeros(n_days), **steady,
                      recovery_z=rng.normal(0, 1, (n_days, 4)), week=np.arange(n_days) // 7,
                      effort_day=eff_day, effort_s=eff_s, effort_w=eff_w, effort_kind=np.where(np.repeat(hard[ride], 3), 1, 0),
                      reading_day=np.array([10, 40, 70, 20]), reading_w=np.array([245.0, 248, 252, 190]),
                      reading_source=np.array([0, 0, 0, 1]), sources=SOURCES)


def test_chains_are_single_threaded_devices():
    import os
    from goodpush.physio import CHAINS
    flags = os.environ.get("XLA_FLAGS", "")
    assert "xla_cpu_multi_thread_eigen=false" in flags and "intra_op_parallelism_threads=1" in flags
    assert jax.local_device_count() >= min(CHAINS, 1)


def test_curve_shape():
    t = np.array([1.0, 5, 60, 300, 1200, 1800, 3600, 7200])
    p = np.asarray(PowerCurve.power(t, 250.0, 20000.0, 1000.0, 0.08))
    assert abs(p[0] - 1000) < 30 and np.all(np.diff(p) < 0)
    assert p[5] > 250 and p[6] == pytest.approx(250 + 20000 / 3600 - 0.08 * 250 * np.log(2), rel=1e-3)


def test_w_prime_from_map_and_the_chain_of_ceilings():
    cp, pmax = np.array([200.0, 250, 300]), np.array([900.0, 1200, 700])
    for wp in (5000.0, 15000.0, 30000.0):                     # W' -> 5-min power -> W' again
        mp = PowerCurve.power(300.0, cp, wp, pmax, 0.08)
        assert np.allclose(PowerCurve.w_prime_for(300.0, mp, cp, pmax), wp, rtol=1e-6)
    assert np.isfinite(jax.grad(lambda m: PowerCurve.w_prime_for(300.0, m, 250.0, 1000.0))(320.0))
    x = np.array([np.log(300.0), np.log(0.8 / 0.2), np.log(0.8 / 0.2), np.log(1000.0), np.log(12.5)])
    c = PhysioModel.capacity(x)                               # MAP 300, CP = 0.8 MAP, LT1 = 0.8 x 1-h power
    p60 = float(PhysioModel.curve(x, 3600.0))
    assert float(c["map"]) == pytest.approx(300) and float(c["cp"]) == pytest.approx(240) and float(c["lt1"]) == pytest.approx(0.8 * p60)
    assert float(c["lt1"]) < p60 < float(c["cp"]) < float(c["map"])
    assert float(PhysioModel.curve(x, 300.0)) == pytest.approx(300, rel=1e-6)   # the curve's 5-min power is MAP


def test_posterior_finite_and_fit_improves():
    m = PhysioModel(synthetic())
    f = PhysioFit(m)
    z0 = m.spec.mean.copy()
    v0, g = f.objective(z0)
    assert np.isfinite(v0) and np.isfinite(g).all() and g.shape == (m.spec.size,)
    r = f.map(z0, maxiter=300)
    assert r.neg_log_post < v0
    tr = m.trajectory(r.z)
    assert set(tr) >= {"map", "f_t", "cp", "lt1", "w_prime", "pmax", "fade", "ftp", "fatigue", "form"}
    assert np.all(tr["lt1"] < tr["cp"]) and np.all(tr["cp"] < tr["map"]) and np.all(tr["fatigue"] >= 0)
    assert 200 < tr["ftp"][-1] < 300          # the readings say ~250 W


def test_training_raises_capacity():
    """more threshold work -> higher CP than resting, everything else equal (prior means)."""
    d = synthetic()
    rest = PhysioData(**{**d.__dict__, "hist_h": d.hist_h * 0})
    z = PhysioModel(d).spec.mean                      # eta (weekly noise) prior mean = 0
    trained, idle = PhysioModel(d).trajectory(z), PhysioModel(rest).trajectory(z)
    assert trained["cp"][-1] > idle["cp"][-1]
    assert trained["fatigue"].max() > 0 and idle["fatigue"].max() == 0


def test_spec_pack_unpack_roundtrip():
    m = PhysioModel(synthetic())
    z = m.spec.pack({"x0": np.arange(5.0)})
    assert np.allclose(np.asarray(m.spec.unpack(z)["x0"]), np.arange(5.0))
    assert len(m.spec.names()) == m.spec.size


def test_inputs_from_tables():
    from goodpush.models.physio import PhysioInputs
    days = pd.date_range("2026-03-02", periods=21)
    h = np.zeros(200)
    h[18] = 3600
    h[30] = 600
    acts = pd.DataFrame({
        "sport": ["ride", "ride", "run"], "date": [days[1], days[8], days[9]],
        "power_hist": [json.dumps(h.tolist()), None, None], "power_coverage": [1.0, 0.0, 0.0],
        "moving_s": [9000.0, 3600.0, 1800.0], "duration_s": [9000.0, 3600.0, 1800.0], "ftp": [250.0, 250.0, 250.0],
        "workout": ["test", "endurance", None], "hrvt1_w": [190.0, np.nan, np.nan],
        **{f"hz{i}_s": [0.0, 1800.0 if i == 2 else 0.0, 0.0] for i in range(1, 8)},
        **{f"mmp_{s}": [400.0 - s / 30, np.nan, np.nan] for s in PhysioInputs.EFFORT_S},
        "avg_hr": [140.0, np.nan, np.nan], "avg_power": [200.0, np.nan, np.nan], "np_used": [210.0, np.nan, np.nan]})
    daily = pd.DataFrame({"sleep_h_z": 0.5, "hrv_z": np.nan, "rhr_z": 1.0, "rhr": 50.0, "weight": 70.0,
                          "vo2max_garmin": [50.0] * 10 + [52.0] * 11, "ftp_garmin": np.nan,
                          "ftp_garmin_auto": np.nan}, index=days)
    d = PhysioInputs.build(acts, daily, {"2026-03-10": 240}, 180.0)
    assert d.hist_h.shape == (21, PhysioInputs.BINS)
    assert d.hist_h[1].sum() == pytest.approx(7 / 6) and d.hist_h[8].sum() == pytest.approx(0.5)   # HR-zone ride
    assert d.long_h[1] == pytest.approx(0.5) and d.long_h.sum() == pytest.approx(0.5)
    assert np.allclose(d.recovery_z[0], [0.5, 0.0, -1.0, 0.0]) and list(d.week[:8]) == [0] * 7 + [1]
    assert d.recovery_seen[0].tolist() == [True, False, True, False]          # no HRV, no bedtime that day
    assert set(d.effort_day) == {1} and d.effort_s.max() == 7200
    short = d.effort_s <= PhysioInputs.TEST_MAX_S                       # a test is one up to 20 min (no history:
    assert (d.effort_kind[short] == 2).all() and (d.effort_kind[~short] == 1).all()   # every duration near "best")
    names = pd.Series(np.array([s.name for s in d.sources])[d.reading_source]).value_counts().to_dict()
    assert names == {"my_ftp": 1, "alpha1_lt1": 1}       # Garmin's estimates aren't readings; steady rides go in as
    assert d.hr_day.tolist() == [1] and d.hr_w.tolist() == [200.0] and d.hr_x.tolist() == [140 / 180]   # power at HR
    assert not d.hr_rep.any() and d.hr_rest == pytest.approx(50 / 180)
    # steady reps (>= 5 min, even power) at their settled HR: each one, as it was (no scaling to threshold HR)
    reps = pd.DataFrame({"date": [days[8]] * 3 + [days[12]], "seconds": [600, 600, 120, 900], "cv": [0.03, 0.04, 0.02, 0.3],
                         "watts": [230.0, 240.0, 300.0, 220.0], "hr_late": [165.0, 170.0, 175.0, 160.0]})
    d = PhysioInputs.build(acts, daily, {"2026-03-10": 240}, 180.0, reps)
    assert d.hr_day[d.hr_rep].tolist() == [8, 8] and d.hr_w[d.hr_rep].tolist() == [230.0, 240.0]   # not the 2-min,
    assert np.allclose(d.hr_x[d.hr_rep], [165 / 180, 170 / 180])                                  # not the uneven one


def test_usual_fatigue_is_where_the_heart_rate_fatigue_term_is_measured_from():
    """an hour a day at CP settles at 1 / (1 - e^(-1/7)) - the model's fatigue units (hours at CP), about ATL / 100."""
    from goodpush.models.physio import PhysioInputs
    hist = np.zeros((200, PhysioInputs.BINS))
    hist[:, 24] = 1.0                                                     # 1 h a day at 245 W
    f = PhysioInputs.usual_fatigue(hist, np.full(5, 1200.0), np.full(5, 245.0 / 0.95), np.arange(150, 200))
    assert f == pytest.approx(1 / (1 - np.exp(-1 / 7)), rel=1e-3)
    assert PhysioInputs.usual_fatigue(hist, np.zeros(0), np.zeros(0), np.arange(5)) == 0.0     # no 20-min bests


def test_threshold_heart_rate_moves_with_maximum_heart_rate():
    """2026-11-09: a fixed LTHR read the same watts at a lower heart rate years later (age) as fitness or fatigue."""
    from goodpush.models.physio import PhysioInputs
    days = pd.date_range("2024-01-01", periods=800)
    rides = pd.DataFrame({"date": days, "max_hr": np.where(days.year == 2024, 190.0, 185.0)})
    thr = PhysioInputs.lthr_then(rides, np.arange(800), 800, 170.0, 185.0)
    assert thr[-1] == pytest.approx(170.0) and thr[100] == pytest.approx(170.0 * 190 / 185, rel=0.01)
    assert (np.diff(thr) <= 1e-9).all()                                       # never rises
    assert (PhysioInputs.lthr_then(rides, np.arange(800), 800, 170.0, None) == 170.0).all()   # no max HR: fixed


def test_the_model_starts_at_the_first_ride_with_power():
    """2026-11-08: months of rides without power left the start to the prior (an old FTP setting read as the 'career best')."""
    from goodpush.models.physio import PhysioInputs
    days = pd.date_range("2026-03-02", periods=21)
    acts = pd.DataFrame({"sport": ["ride", "ride"], "date": [days[0], days[5]], "has_power": [False, True]})
    daily = pd.DataFrame(index=days)
    assert PhysioInputs.start(acts, daily) == days[5]
    assert PhysioInputs.start(acts.assign(has_power=False), daily) == days[0]      # no power at all: from the start
    assert PhysioInputs.start(acts.drop(columns="has_power"), daily) == days[0]


def test_since_days_fits_on_the_recent_rides_from_the_start_of_a_quarter():
    """2026-11-27: [physio] since_days - only about the last N days; the first day moves quarterly, not daily (the
    weekly noise terms count from it, so a daily move would make every update a full fit)."""
    from goodpush.models.physio import PhysioInputs
    days = pd.date_range("2019-01-01", "2026-11-27")
    acts = pd.DataFrame({"sport": ["ride", "ride"], "date": [days[10], days[-1]], "has_power": [True, True]})
    daily = pd.DataFrame(index=days)
    assert PhysioInputs.start(acts, daily) == days[10]                                    # 0: all your data
    assert PhysioInputs.start(acts, daily, 365) == pd.Timestamp("2025-10-01")             # 2025-11-28 -> its quarter
    assert PhysioInputs.start(acts, daily, 1825) == pd.Timestamp("2021-10-01")
    assert PhysioInputs.start(acts, daily, 100000) == days[10]                            # longer than your data
    for last in pd.date_range("2026-09-30", "2026-12-30"):                                # the same for a quarter
        assert PhysioInputs.window_start(last, 365) == pd.Timestamp("2025-10-01")
    assert PhysioInputs.window_start(pd.Timestamp("2026-12-31"), 365) == pd.Timestamp("2026-01-01")   # then on


def test_a_stored_fit_from_another_first_day_is_not_gone_on_from(tmp_path):
    """[physio] since_days moved the first day: the stored fit's noise weeks count from another day, so warm-starting
    or keeping its draws by name would shift them - a full fit instead."""
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    d = synthetic(n_days=84)

    class Run(PhysioRun):
        data = d

        def inputs(self):
            self._start = str(self.data.days[0])
            return self.data

    run = Run(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    run.inputs()
    run.fit(maxiter=200)
    assert run.first_day(run.stored()) == str(d.days[0]) and run.first_day({}) is None
    run.data = d.window(28)                                        # four weeks later start
    msgs = []
    run.current(progress=msgs.append)
    assert any("full fit" in m for m in msgs) and not any("new day(s)" in m for m in msgs)
    assert run.first_day(run.stored()) == str(d.days[28])


def test_a_test_counts_only_where_it_reached_your_recent_best():
    """2026-11-06: a 20-min FTP test's 5 s, a failed test's 20 min sat far below the recent best and were 'tests'."""
    from goodpush.models.physio import PhysioInputs
    days = pd.date_range("2026-03-01", periods=60)
    s = PhysioInputs.EFFORT_S
    best = {t: 1000.0 * (t / 5.0) ** -0.25 for t in s}                     # a plausible best curve
    def ride(day, workout, scale):                                           # noqa: E306 (test-local builder)
        return {"date": days[day], "workout": workout, "moving_s": 7200.0, "duration_s": 7200.0, "power_coverage": 1.0,
                **{f"mmp_{t}": best[t] * scale.get(t, 0.5) for t in s}}
    rides = pd.DataFrame([ride(5, "threshold", {t: 1.0 for t in s}),        # the 90-day best, all durations
                          ride(40, "test", {1200: 1.0}),                    # an honest 20-min test
                          ride(50, "test", {1200: 0.75})])                  # a failed one
    day, dur, _, kind = PhysioInputs.efforts(rides, np.array([5, 40, 50]))
    k = {(int(a), float(b)): int(c) for a, b, c in zip(day, dur, kind)}
    assert k[(40, 1200.0)] == 2 and k[(40, 5.0)] == 1                       # tested at 20 min only
    assert k[(50, 1200.0)] == 1                                              # failed: a hard ride, not a test


def test_forecast_rest_recovers_form_and_trajectories_match():
    from goodpush.models.physio import PhysioRun
    m = PhysioModel(synthetic())
    z = m.spec.mean
    X, F = m.forecast(z, np.zeros((10, 150)), np.zeros(10))
    assert np.all(np.diff(np.asarray(F)) < 0)                     # resting: fatigue only clears
    p0, p9 = PhysioRun.predict(m, z, 0, [1200.0]), PhysioRun.predict(m, z, 9, [1200.0])
    assert p9[0] > p0[0] * 0.99 and np.isfinite(p9).all()
    many = m.trajectories(np.stack([z, z]))
    assert np.allclose(many["cp"][1], m.trajectory(z)["cp"])


def test_run_warm_start_and_states(tmp_path):
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    m = PhysioModel(synthetic())
    run = PhysioRun(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    assert run.warm_start(m) is None
    z = m.spec.mean + 0.1
    np.savez_compressed(tmp_path / PhysioRun.FIT, z=z[:-3], names=np.array(m.spec.names()[:-3]), noise_weeks=1,
                        states=PhysioRun.structure())
    w = run.warm_start(m)
    assert np.allclose(w[:-3], z[:-3]) and np.allclose(w[-3:], m.spec.mean[-3:])   # unknown names: prior mean
    st = run.states(m, z, np.stack([z, z + 0.01]))
    assert len(st) == len(m.data.days) and (st["cp_p05"] <= st["cp_p95"]).all()


def stored_fit(folder, structure, derived):
    """a stored fit of model `structure` and the files (`derived`) worked out from its states."""
    from goodpush.models.physio import PhysioRun
    from goodpush.physio.model import NOISE_WEEKS
    np.savez_compressed(folder / PhysioRun.FIT, z=np.zeros(1), names=np.array(["a"]), noise_weeks=NOISE_WEEKS,
                        states=structure)
    for f in derived:
        (folder / f).write_text("x", encoding="utf-8")


def test_a_new_model_version_never_reads_the_old_fits_states(tmp_path):
    """2026-11-27: inputs worked out from the last fit's daily states (SevereSplit keeps them per ride) must not
    come from another model version's fit: they are set aside, and the analysis is marked as made without them."""
    from types import SimpleNamespace
    from goodpush.metrics.severe_hr import SevereSplit
    from goodpush.models.physio import PhysioRun
    run = PhysioRun(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    derived = (PhysioRun.STATES, SevereSplit.FILE)
    stored_fit(tmp_path, PhysioRun.structure().replace("|v", "|v0"), derived)   # another version's fit
    said = []
    assert run.reset(said.append) and not any((tmp_path / f).exists() for f in derived) and said
    assert run.reset(said.append) and len(said) == 1                     # nothing left: no fit of this one yet
    stored_fit(tmp_path, PhysioRun.structure(), derived)                 # this version's
    assert not run.reset(said.append) and all((tmp_path / f).exists() for f in derived)
    assert not run.analysed()
    run.mark(True)
    assert run.analysed()
    run.mark(False)
    assert not run.analysed()


def dose_problem(cap, days=28, weight=1.0, **kw):
    from goodpush.physio.plan import DosePlan, Target
    m = PhysioModel(synthetic())
    B = 150
    key = np.zeros((days, B))
    key[np.arange(days) % 7 == 1, 25] = 0.5                     # a 30-min hard block each Tuesday
    week = np.arange(days) // 7
    easy_days = np.isin(np.arange(days) % 7, [2, 3, 5, 6])
    share = easy_days / 4.0
    easy = np.zeros(B)
    easy[16] = 1.0
    return DosePlan(m, m.spec.mean, np.zeros((days, B)), key, share, week, easy, np.full(days // 7, 10.0),
                    [Target(days - 1, weight, (300.0, 1200.0), all_out=False)], cap, **kw)


def test_dose_plan_respects_bounds_and_fatigue_cap():
    loose, tight = dose_problem(1e6).solve(), dose_problem(1.0).solve()
    for r in (loose, tight):
        assert np.all((r.easy_h >= -1e-9) & (r.easy_h <= 10 + 1e-9)) and np.all((r.key_scale >= 0) & (r.key_scale <= 1))
        assert r.states.shape == (28, 5) and np.isfinite(r.objective)
    assert tight.easy_h.sum() < loose.easy_h.sum()
    assert tight.fatigue.max() < loose.fatigue.max()


def test_dose_plan_a_flat_week_follows_the_steady_hours():
    flat = dose_problem(1e6, weight=0.0)                   # nothing to gain: time cost alone takes the hours away
    steady = np.array([6.0, 6.6, 0.0, 7.3])                # a week without (recovery): no pull
    held = dose_problem(1e6, weight=0.0, steady=steady)
    r = flat.solve()
    h0 = flat.weekly_hours(r.easy_h, r.key_scale)
    r = held.solve()
    h = held.weekly_hours(r.easy_h, r.key_scale)
    assert np.all(h0 < 1.0)
    assert np.all(np.abs(h[steady > 0] - steady[steady > 0]) < 0.3) and h[2] < 1.0
    gain = dose_problem(1e6, steady=np.full(4, 3.0)).solve()          # a forecast that gains overrides the pull
    assert np.allclose(gain.easy_h, dose_problem(1e6).solve().easy_h, atol=0.05)


def test_physio_load_week_targets():
    from types import SimpleNamespace
    from goodpush.planning.load import LoadModel
    from goodpush.planning.physio_load import PhysioLoad
    from goodpush.planning.workouts import WorkoutLibrary as WL
    m = PhysioModel(synthetic())
    start = pd.Timestamp(m.data.days[-1]) + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=14)
    kinds = ["rest", "threshold", "endurance", "vo2max", "rest", "long", "endurance"] * 2
    wos = [WL.rest(d) if k == "rest" else WL.intervals(d, k, 0) if k in ("threshold", "vo2max")
           else WL.easy(d, k, 90) for d, k in zip(dates, kinds)]
    row = SimpleNamespace(phase=0, block=0)
    weeks = [{"idx": np.arange(7) + 7 * i, "days": 7, "cap_h": 10.0, "done": (0.0, 0.0), "extra": (0.0, 0.0),
              "kind": "load", "taper": False, "row": row, "start": dates[7 * i]} for i in range(2)]
    pl = PhysioLoad(m, m.spec.mean, 250.0)
    assert pl.hist(wos[1]).sum() == pytest.approx(wos[1].minutes / 60)
    out = pl.week_targets(weeks, wos, [], {}, LoadModel(), 40.0, 40.0)
    assert out["tss"].shape == (2,) and np.all(out["tss"] > 0) and out["ctl"].shape == (2,)
    assert set(out["physio"]) >= {"easy_h", "key_scale", "fatigue_cap"}


def test_training_response_for_the_rider_page(tmp_path):
    import json
    from goodpush.models.physio import TrainingResponse
    m = PhysioModel(synthetic())
    f = PhysioFit(m)
    z = f.map(maxiter=200).z
    r = TrainingResponse(m, z).build()
    keys = [w["key"] for w in r["workouts"]]
    assert keys == [w[0] for w in TrainingResponse.WORKOUTS] and r["draws"] == 0 and "long" not in keys   # long rides = endurance
    w = {x["key"]: x for x in r["workouts"]}
    assert w["endurance"]["watts"] < w["threshold"]["watts"] < w["vo2max"]["watts"] < w["short"]["watts"]
    assert all(np.isfinite(list(x["gain"].values())).all() for x in r["workouts"])
    assert max(w["endurance"]["domains"], key=w["endurance"]["domains"].get) == "z1"     # easy work lands below LT1
    assert set(r["sources"]) == {"map", "f_t", "f_e", "hours_week"}
    assert all(abs(sum(r["sources"][k].values()) - 1) < 1e-2 for k in ("map", "f_t", "f_e"))
    assert set(r["effects"]["z3"]["trains"]) == {"map", "f_t"} and isinstance(r["effects"]["z1"]["prior_like"], bool)
    draws = z + 0.05 * m.spec.sd * np.random.default_rng(6).standard_normal((8, z.size))   # with draws: a range per outcome
    t = TrainingResponse(m, z, draws)
    t.save(tmp_path)
    g = json.loads((tmp_path / TrainingResponse.FILE).read_text())["workouts"][0]
    assert g["range"]["p20"][0] <= g["gain"]["p20"] <= g["range"]["p20"][1]      # the draws' median, inside their range


def test_predict_for_the_mode_and_for_draws():
    from goodpush.models.physio import PhysioRun
    m = PhysioModel(synthetic())
    f = PhysioFit(m)
    r = f.map(maxiter=400)
    draws = r.z + 0.05 * m.spec.sd * np.random.default_rng(1).standard_normal((40, r.z.size))
    p = PhysioRun.predict(m, draws, 2, [300.0, 1200.0])
    assert p.shape == (40, 2) and np.all(p[:, 0] > p[:, 1])
    assert np.allclose(PhysioRun.predict(m, r.z, 2, [300.0]), PhysioRun.predict(m, r.z[None], 2, [300.0])[0])


def weighted(d: PhysioData, seed: int = 4) -> PhysioData:
    """d with best-power weights other than 1 (PhysioData.effort_weight)."""
    return replace(d, effort_weight=np.random.default_rng(seed).uniform(0.2, 1.0, len(d.effort_w)))


def test_numba_posterior_matches_jax():
    m = PhysioModel(weighted(synthetic(n_days=84, hr=True)))
    f = PhysioFit(m)
    w0 = f.to_w(f.map(maxiter=200).z)
    rng = np.random.default_rng(3)
    for w in (w0, w0 + 0.1 * rng.standard_normal(w0.size), w0 + 1.5 * rng.standard_normal(w0.size),
              np.zeros(w0.size)):                       # far points: clipped states, rides above the curve
        vj, gj = f.value_and_grad(jnp.asarray(w), m.spec.layout, m.arrays)
        vn, gn = f.fast.value_and_grad(w)
        assert np.isclose(vn, float(vj), rtol=1e-10, atol=1e-8)
        assert np.allclose(gn, np.asarray(gj), rtol=1e-7, atol=1e-7 * np.abs(gj).max())
    v, g = jax.value_and_grad(f.fast.potential)(jnp.asarray(w0))     # the JAX wrapper NUTS uses
    assert np.isclose(float(v), f.fast.value_and_grad(w0)[0]) and np.allclose(g, f.fast.value_and_grad(w0)[1])


def test_numba_hessian_and_low_rank_whitening():
    m = PhysioModel(weighted(synthetic(n_days=84, hr=True)))   # every reading within the days (JAX drops the gradient of others)
    f = PhysioFit(m)
    z = f.map(maxiter=200).z
    w = jnp.asarray(f.to_w(z))
    H = f.fast.hessian(np.asarray(w))
    Hj = np.asarray(jax.hessian(f.potential)(w))
    assert np.allclose(H, Hj, rtol=1e-5, atol=1e-6 * np.abs(Hj).max())
    lam, V = np.linalg.eigh(H)
    lam = np.maximum(lam, f.MIN_EIGEN)
    from goodpush.physio.fit import Whitening
    wh = Whitening.from_hessian(np.zeros(lam.size), lam, V, f.MIN_EIGEN)   # every informed direction kept: exact
    L = wh.to_v(np.eye(lam.size))
    assert np.allclose(L @ L.T, (V / lam) @ V.T, atol=1e-10)


def test_gain_shares_are_the_same_posterior_in_other_coordinates():
    from goodpush.physio.fit import GainShares, Whitening
    m = PhysioModel(synthetic(n_days=84))
    f = PhysioFit(m)
    z = f.map(maxiter=200).z
    sh = GainShares(m, GainShares.average_doses(m, z))
    w = f.to_w(z) + 0.2 * np.random.default_rng(5).standard_normal(z.size)
    v = sh.from_w(w)
    w_back, logdet = sh.to_w(jnp.asarray(v))
    assert np.allclose(w_back, w, atol=1e-12)                                  # one to one
    names = m.spec.names()                                                     # MAP's and CP / MAP's floors are levels:
    fl = [names.index(f"floor[{STATES.index(k)}]") for k in ("map", "f_t", "f_e")]
    lift = np.exp(v[sh.total])                                                 # floor + the lift after HORIZON days
    assert np.allclose((v[fl[:2]] - w[fl[:2]]) * m.spec.sd[fl[:2]], lift) and np.all(lift > 0) and v[fl[2]] == w[fl[2]]
    z_v = m.spec.unpack(f.to_z(w))
    for i, k in enumerate(("map", "f_t")):                                     # r = log total + log(1 - exp(-H / tau))
        g = np.flatnonzero(EFFECT_STATE == STATES.index(k))
        tau = np.exp(np.asarray(z_v["log_tau"])[STATES.index(k)])
        total = np.sum(np.exp(np.asarray(z_v["log_effect"])[g] + sh.log_dose[g]))
        assert np.isclose(lift[i], total * (1 - np.exp(-sh.HORIZON / tau)), rtol=1e-12)
    e = np.flatnonzero(EFFECT_STATE == STATES.index("f_e"))[0]                 # LT1's share: the long-run total, as before
    assert np.isclose(np.exp(v[sh.index[e]]), np.sum(np.exp(np.asarray(z_v["log_effect"])[EFFECT_STATE == STATES.index("f_e")]
                                                           + sh.log_dose[EFFECT_STATE == STATES.index("f_e")])), rtol=1e-12)
    J = np.asarray(jax.jacfwd(sh.w_of)(jnp.asarray(v)))
    assert np.isclose(float(logdet), np.linalg.slogdet(J)[1], rtol=1e-10)     # the Jacobian term is exact
    assert np.any(J[np.ix_(sh.total, sh.tau)] != 0)                           # ... and J has the log tau columns
    u = jnp.zeros(z.size)                                                      # NUTS's potential at v
    pot = f.sampler_potential(sh, Whitening(v), u)
    assert np.isclose(float(pot), f.fast.value_and_grad(w)[0] - float(logdet), rtol=1e-12)


def test_gain_shares_horizon_holds_at_long_time_constants():
    from goodpush.physio.fit import GainShares
    m = PhysioModel(synthetic(n_days=84))
    names = m.spec.names()
    sd = m.spec.sd[[names.index(f"log_tau[{i}]") for i in range(3)]]
    assert np.allclose(sd, [0.7, 0.4, 0.7])                     # LT1's share: a narrower detraining prior
    f = PhysioFit(m)
    sh = GainShares(m, GainShares.average_doses(m, f.map(maxiter=100).z))
    rng = np.random.default_rng(7)
    W = rng.standard_normal((6, m.spec.size))
    W[:, sh.tau] = np.array([[-3.0], [0.0], [2.0], [4.0], [6.0], [8.0]])   # up to 8 prior sds: tau of years
    V = sh.from_w(W)
    first = [int(np.flatnonzero(sh.index == t)[0]) for t in sh.total]       # the LEVEL states' totals within index
    for w, v in zip(W, V):
        w_back, logdet = sh.to_w(jnp.asarray(v))
        assert np.allclose(w_back, w, atol=1e-10, rtol=0)                 # one to one
        q = v[sh.index].copy()
        q[first] -= sh.horizon_share(v[sh.tau])                           # the long-run totals: the old coordinates
        old = sh.log_effects(jnp.asarray(q))[1] - np.sum(np.log(sh.sd))
        assert np.isclose(float(logdet), float(old), rtol=1e-12)           # the shears add nothing to it
    long = sh.horizon_share(np.array([8.0, 8.0]))                         # a long tau: ~ log(H / tau) - H / (2 tau)
    x = sh.HORIZON / np.exp(sh.tau_mean + 8.0 * sh.tau_sd)
    assert np.allclose(long, np.log(x) - x / 2, atol=1e-5)


def test_preconditioned_nuts_runs():
    m = PhysioModel(synthetic(n_days=28))
    f = PhysioFit(m)
    r = f.map(maxiter=300)
    draws, info = f.nuts(r.z, num_warmup=20, num_samples=10, progress=False, chains=2)
    assert draws.shape == (20, m.spec.size) and np.isfinite(draws).all() and np.isfinite(info["r_hat"])
    assert set(info) >= {"mean_steps", "divergences", "step_size", "r_hat", "min_ess"} and info["mean_steps"] >= 1
    assert len(set(info["step_size_chain"])) == 1      # one step size for all chains' draws (2026-11-09)


def test_nuts_checkpoint_goes_on_where_it_stopped(tmp_path):
    m = PhysioModel(synthetic(n_days=28))
    f = PhysioFit(m)
    r = f.map(maxiter=300)
    ck, said = tmp_path / "run.ckpt", []
    first, _ = f.nuts(r.z, num_warmup=20, num_samples=4, progress=False, chains=2, checkpoint=ck, tag="a",
                      block=2, say=said.append)
    saved = PhysioFit.load_checkpoint(ck)
    assert (saved["warmup_done"], saved["W"].shape[1], saved["days"]) == (20, 4, 28)
    assert sum("checkpoint saved" in x for x in said) == 4          # 2 warm-up blocks, then two blocks of draws
    assert any("whitening learned from 16 draws" in x for x in said)   # the learning block's second half, 2 chains
    assert sum("smallest ESS" in x and "worst R-hat" in x for x in said) == 2   # after each block, no ESS target
    said.clear()
    more, info = f.nuts(r.z, num_warmup=20, num_samples=8, progress=False, chains=2, checkpoint=ck, tag="a",
                        block=2, say=said.append)                    # stopped at 4 draws: goes on to 8
    assert "going on" in said[0] and sum("checkpoint saved" in x for x in said) == 2
    assert np.allclose(more.reshape(2, 8, -1)[:, :4], first.reshape(2, 4, -1))   # the first draws are kept
    assert np.isfinite(more).all() and np.isfinite(info["r_hat"])
    said.clear()
    f.nuts(r.z, num_warmup=20, num_samples=4, progress=False, chains=2, checkpoint=ck, tag="b", say=said.append)
    assert not any("going on" in x for x in said)                    # another posterior: starts over


def test_run_goes_on_with_the_data_it_started_with(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    data = synthetic(n_days=84)
    run = PhysioRun(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    save = PhysioFit.save_checkpoint

    def stop_after_warmup(path, ck, say, what, timing=""):   # Ctrl-C right after warm-up is saved
        save(path, ck, say, what, timing)
        raise KeyboardInterrupt
    monkeypatch.setattr(PhysioFit, "save_checkpoint", staticmethod(stop_after_warmup))
    with pytest.raises(KeyboardInterrupt):
        run.fit(samples=4, warmup=4, nuts=True, data=data.until(56), progress=lambda *a: None)
    assert PhysioFit.load_checkpoint(tmp_path / PhysioRun.CHECKPOINT)["days"] == 56
    monkeypatch.setattr(PhysioFit, "save_checkpoint", staticmethod(save))
    said = []
    model, res, draws, st = run.fit(samples=4, warmup=4, nuts=True, data=data, progress=said.append)   # 4 weeks later
    assert any("going on" in x for x in said) and len(model.data.days) == 56 and len(st) == 56
    assert len(draws) == 4 * 4 and (tmp_path / PhysioRun.CHECKPOINT).exists()     # kept: a later run goes on
    said.clear()
    run.fit(data=data, progress=said.append)                       # a mode-only refit (an older day changed) ...
    st = run.stored()
    assert len(st["samples"]) == 4 * 4 and "log_w" not in st       # ... keeps the NUTS draws as they are, ...
    assert st["draws_day"] == str(data.days[55]) and int(st["days"]) == 84      # ... with their own date
    assert any(f"ranges from the run of {data.days[55]}" in x for x in said)
    assert (tmp_path / PhysioRun.CHECKPOINT).exists()
    run.fit(data=data, progress=said.append, fresh=True)           # --fresh: thrown away
    assert len(run.stored()["samples"]) == 0


def test_whitening_learns_what_the_hessian_missed():
    """2026-11-09: directions 10-28x wider than the mode's Hessian said put chains at the tree limit. A stage learned
    from draws and gradients (fewer draws than dimensions) brings the posterior back to about unit scale."""
    from goodpush.physio.fit import Whitening
    rng = np.random.default_rng(11)
    D, n = 300, 120
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    var = np.ones(D)
    var[:4] = (28.0 ** 2, 12.0 ** 2, 5.0 ** 2, 0.3)         # a long ridge at an angle to every axis, and a narrow one
    var[4] = 0.01                                           # very narrow: squeezed only to MIN_VAR in one stage
    S = (Q * var) @ Q.T
    U = rng.multivariate_normal(np.zeros(D), S, size=n)
    G = U @ np.linalg.inv(S)                                # gradients of the Gaussian potential
    wh, stages = Whitening(np.zeros(D)).refined(U, G)
    Li = wh.to_u(np.eye(D))                                 # rows: L^-1 e_i
    left = np.linalg.eigvalsh(Li.T @ S @ Li)                 # the posterior's covariance in the new u
    assert np.isclose(left.max(), 1, atol=0.05)              # the ridges exactly (a joint PCA left ~4-25x)
    assert np.sum(left < 0.5) == 1 and np.isclose(left.min(), 0.01 / Whitening.MIN_VAR, rtol=0.05)  # squeezed so far
    assert stages[0][1].shape[1] == 5 and np.all(stages[1][0] == 1.0)
    u = rng.standard_normal((3, D))
    assert np.allclose(wh.to_u(wh.to_v(u)), u)
    I = rng.standard_normal((n, D))                         # already white: nothing learned
    _, none = Whitening(np.zeros(D)).refined(I, I)
    assert none[0][1].shape[1] == 0 and np.all(none[1][0] == 1.0)


def test_warmup_blocks_and_tree_report(monkeypatch):
    monkeypatch.setattr(PhysioFit, "MAX_STEPS", 2 ** 10 - 1)                 # the real runs' limit
    assert PhysioFit.kernel(lambda u: 0.0)._max_tree_depth == 10
    assert PhysioFit.warmup_blocks(400) == [(0, 50, True), (50, 150, True), (150, 300, True), (300, 400, False)]
    assert PhysioFit.warmup_blocks(20) == [(0, 15, True), (15, 20, False)]
    assert PhysioFit.warmup_blocks(1) == [(0, 1, False)]
    assert PhysioFit.warmup_blocks(400, stored=True) == [(0, 100, True), (100, 200, False)]   # from the stored whitening
    f = PhysioFit.__new__(PhysioFit)
    ok = f.tree_report(np.array([[500, 520], [480, 490]]), np.array([0.010, 0.0098]))
    assert "WARNING" not in ok and "steps a draw 510/485" in ok
    bad = f.tree_report(np.array([[511, 503], [1023, 1023]]), np.array([0.0099, 0.0045]))
    assert "WARNING: chain 2 at the tree limit" in bad


def test_a_longer_run_goes_on_from_a_finished_one(tmp_path):
    """2026-11-09: a run stopped at its 4x --samples cap with R-hat still falling; more --samples goes on from its
    checkpoint (the number of draws is not part of what the run samples)."""
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    data = synthetic(n_days=84)
    run = PhysioRun(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    run.fit(samples=4, warmup=4, nuts=True, data=data, progress=lambda *a: None)
    said = []
    _, _, draws, _ = run.fit(samples=6, warmup=4, nuts=True, data=data, progress=said.append)
    assert any("going on from the checkpoint" in x for x in said) and len(draws) == 4 * 6
    assert run.nuts_tag(data, 4, 4) == run.nuts_tag(data, 6, 4) != run.nuts_tag(data, 4, 8)


def test_full_runs_write_the_whitening_refreshes_only_read_it(tmp_path):
    """2026-11-10 (whitening file rule): a full NUTS run starts from the Hessian alone, takes the full warm-up and
    overwrites physio_whitening.pkl with the stages it learned (never reading the stored ones); a refresh starts from
    the stored stages (by name), learns its one block for itself, and never writes the file."""
    import pickle
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    from goodpush.physio import CHAINS
    d = synthetic(n_days=84)

    class Run(PhysioRun):
        data = d.until(56)
        WARMUP, REFRESH_DRAWS, REFRESH_ESS = 12, 4, 0

        def inputs(self):
            return self.data

    run = Run(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    said = []
    run.fit(samples=4, warmup=12, nuts=True, progress=said.append)
    path = tmp_path / PhysioRun.WHITENING
    first = PhysioFit.load_checkpoint(path)
    assert first["structure"] == run.structure() and first["sampler"] == PhysioFit.SAMPLER and len(first["stages"]) == 2
    assert not any("starting from the stored whitening" in x for x in said)
    before = path.read_bytes()
    for days in (63, 70):                                  # two refreshes: read the stages, never write them
        run.data = d.until(days)
        said.clear()
        _, _, draws, _ = run.refresh(said.append)
        assert any("starting from the stored whitening: 12 warm-up steps" in x for x in said)
        assert not any("learned whitening stored" in x for x in said) and len(draws) == CHAINS * 4
    assert path.read_bytes() == before                     # byte-identical
    run.data = d
    said.clear()
    _, _, draws, _ = run.fit(samples=4, warmup=12, nuts=True, progress=said.append, fresh=True)   # a full run
    assert not any("starting from the stored whitening" in x for x in said)                       # never reads it
    assert any("learned whitening stored" in x for x in said) and np.isfinite(draws).all()
    second = PhysioFit.load_checkpoint(path)
    assert second["names"] == PhysioModel(d).spec.names() and len(second["stages"]) == 2           # its own only
    used = PhysioFit.load_checkpoint(tmp_path / PhysioRun.CHECKPOINT)["whitening"].stages      # the run's own:
    assert len(used) == 1 + len(second["stages"])                     # the Hessian + what it learned, none stored
    assert all(np.array_equal(a, b) for s1, s2 in zip(used[1:], second["stages"]) for a, b in zip(s1, s2))
    for key, other in (("structure", "other|v0"), ("sampler", "other")):    # made otherwise: a refresh doesn't use it
        with open(path, "wb") as f:
            pickle.dump(second | {key: other}, f)
        said.clear()
        (tmp_path / PhysioRun.REFRESH_CHECKPOINT).unlink(missing_ok=True)
        run.refresh(said.append)
        assert any("from another model or sampler: not used" in x for x in said)
        assert not any("starting from the stored whitening" in x for x in said)


def test_stored_whitening_is_put_on_the_new_parameters_by_name():
    from goodpush.physio.fit import Whitening
    V = np.linalg.qr(np.random.default_rng(2).standard_normal((3, 2)))[0]
    stage = (np.array([2.0, 3.0, 4.0]), V, np.array([1.0, -0.5]))
    (s, V2, c), = Whitening.renamed((stage,), ["a", "b", "c"], ["b", "x", "a", "c"])
    assert np.allclose(s, [3.0, 1.0, 2.0, 4.0]) and np.allclose(V2[[2, 0, 3]], V) and np.all(V2[1] == 0)
    assert np.allclose(V2.T @ V2, np.eye(2)) and np.array_equal(c, stage[2])     # still orthonormal: new = identity
    (s3, V3, c3), = Whitening.renamed((stage,), ["a", "b", "c"], ["c", "a"])  # a parameter gone (a backtest fold's
    keep = np.eye(3)[[2, 0]]                                                   # later weeks): the exact restriction
    assert np.allclose(Whitening.apply((s3, V3, c3), np.eye(2)), Whitening.apply(stage, keep)[:, [2, 0]])
    assert np.allclose(V3.T @ V3, np.eye(V3.shape[1])) and np.all(c3 > -1)


def test_chain_progress_follows_the_slowest_chain():
    from goodpush.physio.fit import ChainProgress
    bar = ChainProgress(10, 2, 2)
    assert int(bar.tick(0, -1)) == 0 and int(bar.tick(0, -1)) == 1         # chains numbered as they report
    bar.tick(6, 0)
    bar.tick(2, 1)
    assert bar.bar.n == 2                                                  # the slower chain sets the line
    bar.tick(4, 0)
    bar.tick(8, 1)
    assert bar.bar.n == 10


def test_nuts_target_ess_and_going_on_from_saved_draws(tmp_path, monkeypatch):
    m = PhysioModel(synthetic(n_days=28))
    f = PhysioFit(m)
    r = f.map(maxiter=300)
    said = []
    kw = dict(num_warmup=20, progress=True, chains=2, block=2, say=said.append)    # parallel chains: one line
    first, _ = f.nuts(r.z, num_samples=4, checkpoint=tmp_path / "a.ckpt", tag="a", target_ess=1e9, max_minutes=0, **kw)
    assert any("time limit" in x for x in said) and first.shape == (2 * 4, m.spec.size)
    said.clear()
    with monkeypatch.context() as mp:            # toy draws' ESS is noise (even negative): one per draw instead
        mp.setattr(PhysioFit, "diagnostics", lambda self, shares, ck: (1.0, float(ck["W"].shape[1]), ""))
        f.nuts(r.z, num_samples=4, checkpoint=tmp_path / "a.ckpt", tag="a", target_ess=6, **kw)
    assert any("going on from the checkpoint" in x for x in said)         # the finished run's checkpoint ...
    assert any("target reached at 6 draws" in x for x in said)             # ... drawn on until the target
    said.clear()
    with monkeypatch.context() as mp:            # R-hat 1.015: short of a full run's 1.01, enough for a refresh's 1.02
        mp.setattr(PhysioFit, "diagnostics", lambda self, shares, ck: (1.015, float(ck["W"].shape[1]), ""))
        f.nuts(r.z, num_samples=4, checkpoint=tmp_path / "a.ckpt", tag="a", target_ess=6, max_r_hat=1.02, **kw)
        assert any("target reached at 6 draws" in x for x in said)
        said.clear()
        f.nuts(r.z, num_samples=4, checkpoint=tmp_path / "a.ckpt", tag="a", target_ess=6, max_minutes=0, **kw)
        assert any("time limit" in x for x in said) and not any("target reached" in x for x in said)
    said.clear()
    f.nuts(r.z, num_samples=2, checkpoint=tmp_path / "c.ckpt", tag="c", target_ess=1e9, **kw)
    assert any("stopped at 8 draws" in x for x in said)                    # never endless: 4x the draws asked for
    said.clear()
    more, info = f.nuts(r.z, num_samples=6, checkpoint=tmp_path / "b.ckpt", tag="b",
                        seed_draws=first.reshape(2, 4, -1), **kw)              # no checkpoint: from the draws
    assert any("going on from the saved draws (4 a chain)" in x for x in said)
    assert np.allclose(more.reshape(2, 6, -1)[:, :4], first.reshape(2, 4, -1), atol=1e-8)
    assert np.isfinite(more).all() and np.isfinite(info["mean_steps"])
    assert any(x.startswith("NUTS: not judged: pmax: ESS") for x in said)       # reported, not judged


def test_a_ride_day_counts_as_at_most_three_best_powers():
    """2026-11-10: a day's up to 12 best powers come from overlapping windows; each weighs min(3 / n, 1)."""
    from goodpush.models.physio import PhysioInputs
    day = np.array([0, 0, 0, 0, 0, 0, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])    # 6, 2 and 12 on three days
    w = PhysioInputs.effort_weights(day)
    assert np.allclose(w, np.r_[np.full(6, 0.5), 1.0, 1.0, np.full(12, 0.25)]) and PhysioInputs.EFFORTS_PER_DAY == 3
    assert PhysioInputs.effort_weights(np.zeros(0, int)).shape == (0,)
    d = weighted(synthetic(n_days=84))
    u, v = d.until(40), d.window(40)
    assert np.array_equal(u.effort_weight, d.effort_weight[d.effort_day < 40])
    assert np.array_equal(v.effort_weight, d.effort_weight[d.effort_day >= 40])
    assert synthetic().until(40).effort_weight is None and np.all(synthetic().weights == 1.0)
    m = PhysioModel(d)                                          # padding counts 0
    assert np.array_equal(np.asarray(m.arrays["effort_weight"])[:len(d.effort_w)], d.effort_weight)
    assert np.all(np.asarray(m.arrays["effort_weight"])[len(d.effort_w):] == 0.0)


def test_data_until_cuts_days_efforts_and_readings():
    from goodpush.models.physio import PhysioRun
    d = synthetic(n_days=84)
    u = d.until(40)
    assert len(u.days) == 40 and u.hist_h.shape[0] == 40 and u.effort_day.max() < 40 and u.reading_day.max() < 40
    assert PhysioRun.history_key(u, 40) == PhysioRun.history_key(d, 40)
    assert PhysioModel(u).spec.size <= PhysioModel(d).spec.size


def test_align():
    d = synthetic(n_days=84)
    m = PhysioModel(d)
    short = PhysioModel(replace(d, days=d.days[:56], hist_h=d.hist_h[:56], long_h=d.long_h[:56],
                                recovery_z=d.recovery_z[:56], week=d.week[:56]))
    A = m.spec.align(short.spec.names(), short.spec.mean[None] + 1.0)
    shared = np.isin(m.spec.names(), short.spec.names())
    assert np.allclose(A[0, shared], short.spec.mean + 1.0) and np.allclose(A[0, ~shared], m.spec.mean[~shared])


def test_run_current_updates_only_new_days(tmp_path):
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    d = synthetic(n_days=84)
    keep = d.effort_day < 70
    first = replace(d, days=d.days[:70], hist_h=d.hist_h[:70], long_h=d.long_h[:70], recovery_z=d.recovery_z[:70],
                    week=d.week[:70], effort_day=d.effort_day[keep], effort_s=d.effort_s[keep],
                    effort_w=d.effort_w[keep], effort_kind=d.effort_kind[keep],
                    reading_day=d.reading_day[d.reading_day < 70], reading_w=d.reading_w[d.reading_day < 70],
                    reading_source=d.reading_source[d.reading_day < 70])

    class Run(PhysioRun):
        data = first

        def inputs(self):
            return self.data

    run = Run(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    run.fit(maxiter=200)
    assert int(run.stored()["days"]) == 70
    run.data = d
    msgs = []
    model, z = run.current(progress=msgs.append)
    assert any("updated with 14 new day(s)" in s for s in msgs) and int(run.stored()["days"]) == 84
    assert z.size == model.spec.size and np.isfinite(z).all()
    model2, z2 = run.current(progress=msgs.append)                 # nothing new: the stored fit
    assert np.array_equal(z, z2)


def test_daily_update_reaches_the_full_fits_mode(tmp_path):
    """2026-11-10: the daily update is a full posterior-mode fit warm-started from the stored one (by name), so it
    ends where a full fit (`PhysioRun.fit`, warm-started from the same stored fit) on the same data does."""
    import shutil
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    d = synthetic(n_days=84)

    class Run(PhysioRun):
        data = d.until(70)

        def inputs(self):
            return self.data

    run = Run(SimpleNamespace(processed=tmp_path / "a", reports=tmp_path, cfg={}))
    run.processed.mkdir()
    run.fit()
    shutil.copytree(run.processed, tmp_path / "b")
    run.data = d
    msgs = []
    model, z = run.current(progress=msgs.append)
    assert any("updated with 14 new day(s)" in s for s in msgs)
    other = Run(SimpleNamespace(processed=tmp_path / "b", reports=tmp_path, cfg={}))
    other.data = d
    _, full, _, _ = other.fit(model=model, data=d, progress=lambda *a: None)
    assert float(model.neg_log_posterior(z)) == pytest.approx(full.neg_log_post, abs=1e-6)
    again = PhysioFit(model).map(z)                                    # and it is a mode: nothing left to gain
    assert again.neg_log_post == pytest.approx(full.neg_log_post, abs=1e-6)


def test_refresh_draws_anew_from_the_last_runs_draws(tmp_path, monkeypatch):
    """2026-11-10: daily ranges by a short NUTS run started from the stored run's last draws (not kept: another
    day's posterior), with the stored whitening's short warm-up; without stored draws a full run is asked for."""
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    from goodpush.physio import CHAINS
    d = synthetic(n_days=84)

    class Run(PhysioRun):
        data = d.until(70)
        WARMUP, REFRESH_DRAWS, REFRESH_ESS = 20, 4, 0

        def inputs(self):
            return self.data

    run = Run(SimpleNamespace(processed=tmp_path / "a", reports=tmp_path, cfg={}))
    run.processed.mkdir()
    run.fit(samples=6, warmup=20, nuts=True, progress=lambda *a: None)
    assert (run.processed / run.WHITENING).exists() and len(run.stored()["samples"]) == CHAINS * 6
    first = run.stored()["samples"]
    run.data = d.until(77)
    said = []
    run.current(progress=said.append)                     # a daily update: the mode only, the draws kept as they are
    st = run.stored()
    assert any(f"ranges from the run of {d.days[69]}" in x for x in said) and st["draws_day"] == str(d.days[69])
    idx = [st["names"].index(n) for n in PhysioModel(d.until(70)).spec.names()]
    assert np.array_equal(st["samples"][:, idx], first) and str(st["day"]) == str(d.days[76])   # by name, unchanged
    import json
    assert json.loads((run.processed / "physio_diagnostics.json").read_text())["draws_day"] == str(d.days[69])
    run.data = d
    said.clear()
    asked, nuts = {}, PhysioFit.nuts
    monkeypatch.setattr(PhysioFit, "nuts", lambda self, *a, **k: asked.update(k) or nuts(self, *a, **k))
    model, res, draws, states = run.refresh(said.append)
    assert asked["max_r_hat"] == PhysioRun.REFRESH_R_HAT == 1.02 and PhysioFit.MAX_R_HAT == 1.01   # refresh only
    assert draws.shape == (CHAINS * 4, model.spec.size) and np.isfinite(draws).all()      # only its own draws
    st = run.stored()
    assert len(st["samples"]) == CHAINS * 4 and int(st["days"]) == 84 and st["draws_day"] == str(d.days[-1])
    assert any("starting from the stored whitening" in x for x in said)
    assert any("start from the given positions" in x for x in said) and not any("going on" in x for x in said)
    assert (run.processed / run.REFRESH_CHECKPOINT).exists() and "cp_p05" in states
    # 2026-11-11 (coverage check: 7 of 24 short runs ended short of their targets): such a refresh saves nothing
    monkeypatch.setattr(PhysioFit, "nuts", lambda self, *a, **k: [(dr, info | {"reached": False})
                                                                  for dr, info in [nuts(self, *a, **k)]][0])
    files = {p.name: p.read_bytes() for p in run.processed.iterdir() if p.name != run.REFRESH_CHECKPOINT}
    said.clear()
    assert run.refresh(said.append) is None and any("nothing saved" in x for x in said)
    assert {p.name: p.read_bytes() for p in run.processed.iterdir() if p.name != run.REFRESH_CHECKPOINT} == files
    other = Run(SimpleNamespace(processed=tmp_path / "b", reports=tmp_path, cfg={}))
    other.processed.mkdir()
    other.fit(progress=lambda *a: None)                                    # a mode, no draws
    said.clear()
    assert other.refresh(said.append) is None and any("a full run is needed" in x for x in said)


def test_run_ignores_fit_with_other_noise_period(tmp_path):
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    m = PhysioModel(synthetic())
    np.savez_compressed(tmp_path / PhysioRun.FIT, z=m.spec.mean, names=np.array(m.spec.names()), noise_weeks=4)
    run = PhysioRun(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    assert run.stored() is None and run.warm_start(m) is None


def test_pmax_and_durability_only_drift():
    """no gains, no detraining: without process noise they stay at their start, trained or not."""
    d = synthetic()
    rest = PhysioData(**{**d.__dict__, "hist_h": d.hist_h * 0})
    z = PhysioModel(d).spec.mean
    trained, idle = PhysioModel(d).trajectory(z), PhysioModel(rest).trajectory(z)
    for c in ("pmax", "fade"):
        assert np.allclose(trained[c], trained[c][0]) and np.allclose(trained[c], idle[c])


def test_run_current_refits_when_the_model_changed(tmp_path):
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    d = synthetic(n_days=84)

    class Run(PhysioRun):
        def inputs(self):
            return d

    run = Run(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    run.fit(maxiter=50)
    st = run.stored()
    names = [n.replace("log_effect", "log_gain") for n in st["names"]]       # as an older model version
    np.savez_compressed(tmp_path / PhysioRun.FIT, **{**{k: v for k, v in st.items() if k != "names"},
                                                     "names": np.array(names), "key": "old"})
    msgs = []
    run.current(progress=msgs.append)
    assert any("full fit" in s for s in msgs) and not any("updated with" in s for s in msgs)
    st = run.stored()                                    # same names, other states (a structure change): refit, no warm start
    np.savez_compressed(tmp_path / PhysioRun.FIT, **{**st, "names": np.array(st["names"]), "states": "map,f_e,f_t,pmax,durability|v1", "key": "old"})
    assert run.stored() is None and run.warm_start(PhysioModel(d)) is None
    msgs = []
    run.current(progress=msgs.append)
    assert any("full fit" in s for s in msgs)



def test_dose_plan_block_shape_ramps_then_recovers():
    from goodpush.physio.plan import RAMP_MIN, RECOVERY_MAX
    p = dose_problem(1e6, days=28, ramp=[1.0, 1.0, 0.0],   # weeks 0 -> 1 -> 2 build, week 3 recovers
                     recover=[0.0, 0.0, 1.0])
    r = p.solve()
    h = p.weekly_hours(jax.numpy.asarray(r.easy_h), jax.numpy.asarray(r.key_scale))
    assert h[1] >= (1 + RAMP_MIN) * h[0] - 0.3 and h[2] >= (1 + RAMP_MIN) * h[1] - 0.3
    assert h[3] <= RECOVERY_MAX * h[2] + 0.3


def test_intensity_map_anchors_targets_to_thresholds(tmp_path):
    from goodpush.physio import PowerCurve
    from goodpush.planning.intensity import FRAC, REPS_1MIN, IntensityMap
    from goodpush.planning.workouts import WorkoutLibrary as WL
    m = PhysioModel(synthetic())
    X = np.asarray(m.simulate(m.spec.mean)[0])[-3:]
    days = pd.date_range("2026-11-02", periods=3)
    im = IntensityMap.from_states(m, days, X, 250.0)
    c = m.capacity(X[1])
    assert im.watts(days[1], 0.75) == pytest.approx(float(c["lt1"]))
    assert im.watts(days[1], 1.0) == pytest.approx(float(c["cp"])) and im.watts(days[1], 1.2) == pytest.approx(float(c["map"]))
    p60 = float(PowerCurve.power(60.0, c["cp"], c["w_prime"], c["pmax"], c["fade"]))
    assert im.watts(days[1], 1.5) == pytest.approx(REPS_1MIN * p60, rel=1e-6)
    assert np.all(np.diff(im.knots, axis=1) >= 0) and im.knots.shape == (3, len(FRAC))
    w = WL.intervals(days[1], "vo2max", 2, variety=False)
    a = im.apply(w)
    assert [s.name for s in a.expanded()] == [s.name for s in w.expanded()]
    k = next(i for i, s in enumerate(w.expanded()) if s.lo >= 1.0)                           # a VO2max rep
    assert a.expanded()[k].lo * 250 == pytest.approx(im.watts(days[1], w.expanded()[k].lo), abs=0.5)   # watts kept
    im.save(tmp_path)
    back = IntensityMap.load(tmp_path)
    assert np.allclose(back.knots, im.knots, atol=0.1) and back.ftp == 250.0


def test_physio_load_doses_use_your_watts():
    from goodpush.planning.physio_load import PhysioLoad
    from goodpush.planning.workouts import WorkoutLibrary as WL
    m = PhysioModel(synthetic())
    pl = PhysioLoad(m, m.spec.mean, 250.0)
    w = WL.easy(pd.Timestamp(m.data.days[-1]) + pd.Timedelta(days=2), "endurance", 60)
    h = pl.hist(w)
    z2 = w.steps[1]
    watts = pl.now.watts(w.date, (z2.lo + z2.hi) / 2)
    assert h[int(watts // 10)] == pytest.approx(z2.minutes / 60)              # the Z2 part at your mapped watts
    assert pl.intensity_map() is pl.now                                        # nothing solved: today's


def test_intensity_map_leaves_tests_alone():
    from goodpush.planning.intensity import IntensityMap
    from goodpush.planning.workouts import WorkoutLibrary as WL
    m = PhysioModel(synthetic())
    im = IntensityMap.from_states(m, pd.date_range("2026-11-02", periods=1), np.asarray(m.simulate(m.spec.mean)[0])[-1:], 250.0)
    for t in (WL.ramp_test(pd.Timestamp("2026-11-02")), WL.ftp_test(pd.Timestamp("2026-11-02"))):
        assert im.apply(t) is t


def test_mapped_session_tss_uses_the_days_ftp():
    from goodpush.planning.intensity import IntensityMap
    from goodpush.planning.workouts import WorkoutLibrary as WL
    days = pd.date_range("2026-11-02", periods=2)
    base = np.array([0, 180, 240, 300, 400, 900.0])
    im = IntensityMap(days, np.vstack([base, base * 1.1]), 240.0)          # 10 % fitter on day 2
    assert im.ftp_on(days[1]) == pytest.approx(264.0)
    w1, w2 = im.apply(WL.easy(days[0], "endurance", 60)), im.apply(WL.easy(days[1], "endurance", 60))
    assert w2.steps[1].lo == pytest.approx(w1.steps[1].lo * 1.1, abs=0.002)   # 10 % more watts ...
    assert w2.tss() == pytest.approx(w1.tss(), rel=0.01)                     # ... same TSS at that day's FTP



def test_no_backstop_without_history_and_forecast_frame():
    from goodpush.planning.physio_load import MIN_RIDE_DAYS, PhysioLoad
    m = PhysioModel(synthetic())                                        # 84 days, 36 with rides
    pl = PhysioLoad(m, m.spec.mean, 250.0)
    assert (np.asarray(m.data.hist_h).sum(1) > 0).sum() < MIN_RIDE_DAYS and pl.fatigue_cap == float("inf")
    assert pl.forecast_frame() is None
    long = PhysioModel(synthetic(n_days=400))
    from goodpush.planning.physio_load import BACKSTOP_GROWTH
    highest = float(np.max(long.trajectory(long.spec.mean)["fatigue"]))
    assert PhysioLoad(long, long.spec.mean, 250.0).fatigue_cap == pytest.approx(BACKSTOP_GROWTH * highest)   # room to grow


def test_recovery_markers_do_not_enter_the_model():
    """they act through readiness; inside the fit they fitted backwards or swamped fatigue (2026-10-26)."""
    d = synthetic()
    a, b = PhysioModel(d), PhysioModel(replace(d, recovery_z=d.recovery_z * 0 + 3.0))
    assert float(a.log_posterior(a.spec.mean)) == pytest.approx(float(b.log_posterior(b.spec.mean)))


def test_heart_rate_curve_weekly_shared_error_and_fatigue():
    """2026-11-09: steady stretches are power at their heart rate through the rider's own curve g (not scaled to
    threshold HR along a line from resting HR); a week's stretches share part of their error; fatigue holds HR down."""
    from goodpush.physio.model import HR_KNOTS
    d = synthetic(n_days=84, hr=True)
    m = PhysioModel(d)
    p = m.spec.unpack(m.spec.mean)
    x = np.array([0.6, 0.7, 0.75, 1.0, 1.05])
    g = np.asarray(PhysioModel.hr_curve(p, jnp.asarray(x)))
    k = np.asarray(p["hr_curve"])
    assert g[1] == pytest.approx(k[0]) and g[3] == pytest.approx(k[-1]) and g[2] == pytest.approx((k[0] + k[1]) / 2)
    assert g[0] == pytest.approx(k[0] - (k[1] - k[0])) and len(k) == len(HR_KNOTS)          # straight on past the ends
    X, F, _ = m.simulate(m.spec.mean)
    a = m.arrays
    # one week's terms, by hand: the closed form = the log density of N(0, s^2 I + w^2 1 1')
    from scipy.stats import multivariate_normal
    q = dict(p) | {"hr_log_sd": jnp.array([np.log(0.04)]), "hr_log_week_sd": jnp.array([np.log(0.03)])}
    wk = np.asarray(a["hr_group"]) == 2
    use = jnp.asarray(wk & np.asarray(a["hr_live"]))
    cp = np.asarray(PhysioModel.capacity(np.asarray(X)[d.hr_day[wk[:len(d.hr_w)]]])["cp"])
    r = (np.log(d.hr_w) - np.log(np.asarray(PhysioModel.capacity(np.asarray(X)[d.hr_day])["cp"]))
         - np.asarray(PhysioModel.hr_curve(q, jnp.asarray(d.hr_x))) - d.hr_rep * float(q["hr_rep"][0])
         - float(q["hr_fatigue"][0]) * np.asarray(F)[d.hr_day])[wk[:len(d.hr_w)]]
    C = 0.04 ** 2 * np.eye(len(r)) + 0.03 ** 2
    assert len(cp) > 1 and float(PhysioModel.hr_terms(q, a, X, F, use)) == pytest.approx(multivariate_normal(cov=C).logpdf(r))
    # tired: the same watts at a lower heart rate - with hr_fatigue > 0 that reads as tiredness, not fitness
    assert float(PhysioModel.hr_terms(q | {"hr_fatigue": jnp.array([0.02])}, a, X, F)) != float(PhysioModel.hr_terms(q, a, X, F))
    w = d.window(30)                                                     # windows keep their stretches, shifted
    assert w.hr_day.min() >= 0 and len(w.hr_day) == (d.hr_day >= 30).sum() and len(d.until(30).hr_day) == (d.hr_day < 30).sum()


def test_fewer_parameters_one_deficit_one_endurance_effect():
    """2026-11-09 removal series (Occam): forecasts as good with one deficit shape and a log scale straight in log
    duration, and one effect for z1 and z2 on LT1 / 1-h power. The weekly noise stays per state (shared, LT1 and
    CP/MAP could only move with MAP)."""
    from goodpush.physio.model import GAIN_INDEX
    m = PhysioModel(synthetic(n_days=84))
    p = m.spec.unpack(m.spec.mean + 0.3 * np.random.default_rng(1).standard_normal(m.spec.size))
    shape = dict(m.spec.layout)
    assert shape["eta"] == (m.data.n_periods, 5) and shape["log_shape_m2"] == (1,) and shape["log_deficit"] == (2,)
    G = np.asarray(PhysioModel.gains(p))
    assert G[1, 0] == G[1, 1] and G[1, 5] == 0                 # f_E <- z1 = z2; long rides only as their hours
    assert len(m.spec.unpack(m.spec.mean)["log_effect"]) == len(GAIN_INDEX[0]) - 1
    x = PhysioModel.standard_log([60.0, 300.0, 1200.0])
    assert np.isclose(x.mean(), 0) and np.isclose(x.std(), 1) and x[0] < x[1] < x[2]


def test_ess_target_leaves_out_pmax():
    from goodpush.physio import STATES
    m = PhysioModel(synthetic(n_days=28))
    keep, names = PhysioFit(m).judged(), m.spec.names()
    p = STATES.index("pmax")
    out = {n for n, k in zip(names, keep) if not k}
    assert out == {f"x0[{p}]", f"floor[{p}]", f"log_tau[{p}]"} | {
        n for n in names if n.startswith("eta[") and int(n[4:-1]) % len(STATES) == p}
    assert keep[[i for i, n in enumerate(names) if n.startswith(("log_effect", "bias", "eta"))]].sum() > 0


def test_run_refits_when_the_readings_are_weighed_differently(tmp_path):
    """2026-11-03 (tests weigh more): a fit made with other reading weights is redone, warm-started - the same
    data alone would otherwise keep the stored fit, and the new weights would never count."""
    from types import SimpleNamespace
    from goodpush.models.physio import SOURCES, PhysioRun
    d = synthetic(n_days=84)

    class Run(PhysioRun):
        def inputs(self):
            return d

    run = Run(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    run.fit(maxiter=50)
    msgs = []
    run.current(progress=msgs.append)
    assert not any("full fit" in s for s in msgs)                          # same data, same weights: kept
    st = run.stored()
    assert str(st["priors"]) == PhysioRun.priors() and "ramp_lt2:cp:0.01:0.035:bound" in PhysioRun.priors()
    np.savez_compressed(tmp_path / PhysioRun.FIT, **{**st, "names": np.array(st["names"]), "priors": "older"})
    assert run.warm_start(PhysioModel(d)) is not None                      # still a warm start
    msgs = []
    run.current(progress=msgs.append)
    assert any("weight of its readings changed" in s for s in msgs)
    by = {s.name: s for s in SOURCES}
    assert by["ramp_lt2"].bound and by["ramp_lt1"].bound and "ramp_top" not in by    # caps; its top is an effort
    assert not {"garmin_ftp", "garmin_vo2max"} & set(by)                           # independent of Garmin


def test_a_ramp_threshold_is_an_upper_bound():
    """2026-11-03: a ramp test reads its thresholds high, so its reading caps the state - only a state above it
    costs (soft: log Phi). The numba posterior matches JAX with such readings, inside and past the bound."""
    bound = Source("ramp_lt2", "cp", 0.01, 0.035, bound=True)
    d = synthetic(n_days=84)
    d = replace(d, sources=SOURCES + (bound,), reading_day=np.r_[d.reading_day, 30, 60],
                reading_w=np.r_[d.reading_w, 400.0, 150.0], reading_source=np.r_[d.reading_source, 2, 2])
    m = PhysioModel(d)
    f = PhysioFit(m)
    w0 = f.to_w(f.map(maxiter=200).z)
    rng = np.random.default_rng(5)
    for w in (w0, w0 + 0.3 * rng.standard_normal(w0.size), np.zeros(w0.size)):
        vj, gj = f.value_and_grad(jnp.asarray(w), m.spec.layout, m.arrays)
        vn, gn = f.fast.value_and_grad(w)
        assert np.isclose(vn, float(vj), rtol=1e-10, atol=1e-8)
        assert np.allclose(gn, np.asarray(gj), rtol=1e-7, atol=1e-7 * np.abs(gj).max())
    z = jnp.asarray(f.to_z(w0))
    X, _, _ = PhysioModel.simulate_of(z, m.spec.layout, m.arrays)
    ll = np.asarray(PhysioModel.reading_terms(ParamSpec.split(m.spec.layout, z), m.arrays, X))[:len(d.reading_w)]
    assert -0.01 < ll[-2] <= 0.0 and ll[-1] < -1            # well above the state: ~free; well below it: costs


def test_many_draws_run_in_batches_with_the_same_results(monkeypatch):
    """2026-11-13: the write-out simulated 100 / 800 draws' whole histories in one call and the machine reset three
    times there; `batched` runs BATCH at a time (the last padded: one compiled shape) and gives the same numbers."""
    from goodpush.physio import PhysioModel
    d = synthetic(n_days=56)
    m = PhysioModel(d)
    z = m.spec.mean
    Z = z + 0.01 * np.random.default_rng(1).standard_normal((7, len(z)))
    whole = {k: np.asarray(v)[:, :m.T] for k, v in m.daily_many(jnp.asarray(Z), m.spec.layout, m.arrays).items()}
    lp_whole = np.asarray(m.log_posterior_draws(jnp.asarray(Z), m.spec.layout, m.arrays))
    monkeypatch.setattr(PhysioModel, "BATCH", 3)                      # 7 draws = 3 + 3 + 1 (padded to 3)
    T = m.trajectories(Z)
    assert all(T[k].shape == whole[k].shape and np.allclose(T[k], whole[k]) for k in whole)
    lp = m.batched(m.log_posterior_draws, Z, m.spec.layout, m.arrays)
    assert lp.shape == (7,) and np.allclose(lp, lp_whole)


def test_an_app_on_older_model_code_never_refits(tmp_path, monkeypatch):
    """2026-11-13: one model everywhere - an app started before an update must not refit after a ride sync (it would
    write its old model over the fit and drop the draws); the fit records its version for the Model page."""
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioDiagnostics, PhysioRun
    from goodpush.physio.model import VERSION
    d = synthetic(n_days=84)
    monkeypatch.setattr(PhysioRun, "inputs", lambda self: d)
    run = PhysioRun(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    assert PhysioRun.version_on_disk() == VERSION
    run.fit(maxiter=50, progress=lambda s: None)
    assert json.loads((tmp_path / PhysioDiagnostics.FILE).read_text(encoding="utf-8"))["version"] == VERSION
    stored = (tmp_path / PhysioRun.FIT).read_bytes()
    monkeypatch.setattr(PhysioRun, "version_on_disk", staticmethod(lambda: VERSION + 1))
    with pytest.raises(RuntimeError, match=f"runs physio model v{VERSION}, the code on disk is v{VERSION + 1}"):
        run.current(progress=lambda s: None)
    assert (tmp_path / PhysioRun.FIT).read_bytes() == stored                     # untouched


def test_runs_name_their_model_version_and_say_when_stored_draws_are_dropped(tmp_path):
    """2026-11-13: a NUTS run sampled v14 (the code then); the first v16 refit deleted its draws without a word, and
    nothing in the run's log said which version it sampled."""
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    from goodpush.physio.model import VERSION
    d = synthetic(n_days=84)

    class Run(PhysioRun):
        def inputs(self):
            return d

    run = Run(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    said = []
    model, res, _, _ = run.fit(maxiter=50, progress=said.append)
    assert said[0].startswith(f"physio model v{VERSION}: 84 days")              # the version, first thing in the log
    run.save(model, d, res.z, np.tile(res.z, (3, 1)), "2026-11-04")             # a run's draws ...
    with np.load(tmp_path / PhysioRun.FIT) as f:                                 # ... of an older model version
        st = {k: f[k] for k in f.files}
    older = str(st["states"]).replace(f"|v{VERSION}", "|v1")
    np.savez_compressed(tmp_path / PhysioRun.FIT, **st | {"states": np.array(older)})
    said.clear()
    run.fit(maxiter=50, progress=said.append)                                    # this version: can't take them
    assert not (tmp_path / PhysioRun.DRAWS).exists()
    assert any("draws (model v1) don't belong to this fit" in s and f"(model v{VERSION})" in s for s in said)
    said.clear()
    run.fit(maxiter=50, progress=said.append)                                    # nothing left to drop: no message
    assert not any("don't belong" in s for s in said)


def test_draws_stay_on_their_machine_and_only_go_with_their_fit(tmp_path):
    """2026-11-04: an 800 MB fit synced to the other computer on every refit - the coefficients sync, the draws
    (physio_draws.npz) stay where they were made, and a fit synced from elsewhere never takes this machine's."""
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioRun
    from goodpush.sync.peer import PeerRules
    d = synthetic(n_days=84)

    class Run(PhysioRun):
        def inputs(self):
            return d

    run = Run(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    model, res, _, _ = run.fit(maxiter=50)
    assert not (tmp_path / PhysioRun.DRAWS).exists() and len(run.stored()["samples"]) == 0
    draws = np.tile(res.z, (3, 1))
    run.save(model, d, res.z, draws, "2026-11-04")
    with np.load(tmp_path / PhysioRun.FIT) as f:
        assert "samples" not in f.files                          # the synced file: coefficients only
    st = run.stored()
    assert st["samples"].shape == draws.shape and "log_w" not in st and st["draws_day"] == "2026-11-04"
    np.savez(tmp_path / PhysioRun.DRAWS, samples=draws, log_w=np.zeros(3), fit_id=st["fit_id"])   # before 2026-11-10:
    old = {k: v for k, v in st.items() if k not in ("samples", "draws_day")} | {"full_date": "2026-11-03"}   # weights,
    np.savez_compressed(tmp_path / PhysioRun.FIT, **old | {"names": np.array(st["names"])})              # full_date
    st = run.stored()
    assert st["samples"].shape == draws.shape and "log_w" not in st and st["draws_day"] == "2026-11-03"
    np.savez_compressed(tmp_path / PhysioRun.FIT, **{**{k: v for k, v in st.items() if k not in ("samples",)},
                                                     "names": np.array(st["names"]), "fit_id": "other"})
    assert len(run.stored()["samples"]) == 0                     # a fit from the other machine: not these draws
    assert PeerRules.kind("data/processed/physio_draws.npz") == "skip"
    assert PeerRules.kind("data/processed/physio_fit.npz") == "derived"


def test_numba_matches_jax_with_maxed_efforts():
    """2026-11-04: a fourth effort kind (maxed: heart rate at your maximum) - both posteriors agree."""
    d = synthetic(n_days=84)
    kind = d.effort_kind.copy()
    kind[::5] = 3
    m = PhysioModel(replace(d, effort_kind=kind))
    f = PhysioFit(m)
    w0 = f.to_w(f.map(maxiter=200).z)
    rng = np.random.default_rng(7)
    for w in (w0, w0 + 0.3 * rng.standard_normal(w0.size)):
        vj, gj = f.value_and_grad(jnp.asarray(w), m.spec.layout, m.arrays)
        vn, gn = f.fast.value_and_grad(w)
        assert np.isclose(vn, float(vj), rtol=1e-10, atol=1e-8)
        assert np.allclose(gn, np.asarray(gj), rtol=1e-7, atol=1e-7 * np.abs(gj).max())


def test_numba_matches_jax_with_bad_readings():
    """2026-11-26 (v17): bests far over the curve and in the crossover (both branches share them) - both agree."""
    d = synthetic(n_days=84)
    w = d.effort_w.copy()
    w[::7], w[3::11] = 3.0 * w[::7], 1.13 * w[3::11]
    m = PhysioModel(replace(d, effort_w=w))
    f = PhysioFit(m)
    w0 = f.to_w(f.map(maxiter=200).z)
    rng = np.random.default_rng(3)
    for x in (w0, w0 + 0.3 * rng.standard_normal(w0.size)):
        vj, gj = f.value_and_grad(jnp.asarray(x), m.spec.layout, m.arrays)
        vn, gn = f.fast.value_and_grad(x)
        assert np.isclose(vn, float(vj), rtol=1e-10, atol=1e-8)
        assert np.allclose(gn, np.asarray(gj), rtol=1e-7, atol=1e-7 * np.abs(gj).max())


def test_one_broken_power_meter_best_barely_moves_the_curve():
    """2026-11-26: a ride's 1000 W for 4:20 (heart rate 132) lifted a synthetic rider's CP +86 %, MAP +119 % over the
    whole history; with BAD_READING_SHARE a lone best at 3 x the curve moves MAP and CP under 2 %."""
    d = synthetic(n_days=84)
    base = PhysioModel(d)
    k = int(np.flatnonzero(d.effort_s <= 300)[len(d.effort_s) // 4])
    w = d.effort_w.copy()
    w[k] *= 3.0
    glitch = PhysioModel(replace(d, effort_w=w))
    caps = []
    for m in (base, glitch):
        X, _, _ = m.simulate(PhysioFit(m).map(maxiter=400).z)
        c = PhysioModel.capacity(X)
        caps.append(np.array([np.median(np.asarray(c["map"])), np.median(np.asarray(c["cp"]))]))
    assert np.all(np.abs(caps[1] / caps[0] - 1) < 0.02)


def test_nuts_takes_smaller_steps_than_the_default():
    """2026-11-04: chains stuck in a funnel at numpyro's 0.8 - the kernel asks for 0.9, and a run made otherwise is a
    different sampler (its checkpoint isn't gone on from)."""
    k = PhysioFit.kernel(lambda u: 0.0)
    assert k._target_accept_prob == PhysioFit.TARGET_ACCEPT == 0.9 and "ta0.9" in PhysioFit.SAMPLER


def test_model_page_diagnostics(tmp_path):
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioDiagnostics, PhysioRun
    m = PhysioModel(synthetic())
    f = PhysioFit(m)
    z = f.map(maxiter=300).z
    draws, info = f.nuts(z, num_warmup=20, num_samples=10, progress=False, chains=2)
    assert len(info["r_hat_each"]) == len(info["ess_each"]) == m.spec.size and len(info["divergences_chain"]) == 2
    run = PhysioRun(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    run.save(m, m.data, z, draws, str(m.data.days[-1]), info)
    d = json.loads((tmp_path / PhysioDiagnostics.FILE).read_text())
    g = d["sampling"]
    assert (g["chains"], g["draws"]) == (2, 10) and np.shape(g["trace"]["log_post"]) == (2, 10)
    assert np.isfinite(g["trace"]["cp"]).all() and len(g["slowest"]) == PhysioDiagnostics.SLOWEST
    assert sum(d["posterior"]["cp"]["n"]) == len(draws) and d["posterior"]["cp"]["p05"] <= d["posterior"]["cp"]["p95"]
    e = d["efforts"]
    assert len(e["observed"]) == len(e["predicted"]) == np.sum(m.data.effort_day >= 0) and set(e["kind"]) <= {"easy", "hard"}
    hard = np.array(e["kind"]) == "hard"                     # all-out-ish efforts (98 % of the curve) near the line
    assert np.median(np.array(e["observed"])[hard] / np.array(e["predicted"])[hard]) > 0.85
    assert d["training"] == {}                                # under two blocks of days: nothing to compare
    r = d["readings"]
    assert [s["n"] for s in r["sources"]] == [3, 1] and len(r["observed"]) == 4
    assert len(r["model"]["date"]) == len(r["model"]["cp"]) == len(m.data.days) and r["model"]["lt1"][-1] < r["model"]["cp"][-1]
    names = [a["name"] for a in d["assumed"]]
    assert "pmax" in names and all({"what", "short", "why"} <= set(a) for a in d["assumed"])
    i = m.spec.names().index("log_s_f")                       # 2026-11-10: what fatigue costs on the day, near its prior
    rng = np.random.default_rng(9)
    prior = np.tile(z, (200, 1))
    prior[:, i] = m.spec.mean[i] + m.spec.sd[i] * rng.standard_normal(200)
    assert [a["name"] for a in PhysioDiagnostics(m, z, None, prior).assumed()] == ["log_s_f", "pmax"]
    pulled = prior.copy()
    pulled[:, i] = m.spec.mean[i] + 3 * m.spec.sd[i] + 0.1 * m.spec.sd[i] * rng.standard_normal(200)   # far from it
    assert [a["name"] for a in PhysioDiagnostics(m, z, None, pulled).assumed()] == ["pmax"]
    narrow = prior.copy()
    narrow[:, i] = m.spec.mean[i] + 0.2 * m.spec.sd[i] * rng.standard_normal(200)   # at the prior mean, but learned
    assert [a["name"] for a in PhysioDiagnostics(m, z, None, narrow).assumed()] == ["pmax"]
    assert PhysioDiagnostics(m, z, None).assumed() == []      # no draws: nothing said
    run.save(m, m.data, z, draws, str(m.data.days[-1]))   # a daily update keeps the run's
    assert json.loads((tmp_path / PhysioDiagnostics.FILE).read_text())["sampling"] == g
    run.save(m, m.data, z, np.zeros((0, m.spec.size)), str(m.data.days[-1]))   # no draws: no run
    d = json.loads((tmp_path / PhysioDiagnostics.FILE).read_text())
    assert d["sampling"] is None and d["posterior"] == {} and d["efforts"]["observed"]


def test_diagnostics_split_the_path_into_training_and_unexplained_changes():
    """2026-11-10: with the weekly unexplained changes switched off the path is training alone; without any such
    changes in the fit the two are the same path."""
    from goodpush.models.physio import PhysioDiagnostics
    m = PhysioModel(synthetic(n_days=196))
    z = m.spec.mean.copy()
    same = PhysioDiagnostics(m, z, None).training()
    assert same["days"] == 84 and all(np.isclose(same[k]["share"], 1.0) for k in ("lt1", "cp", "map"))
    assert same["cp"]["change_pct"] == same["cp"]["training_change_pct"]
    o = m.spec.offsets["eta"]
    z[o:o + 5 * m.data.n_periods] = 3.0 * np.random.default_rng(2).standard_normal(5 * m.data.n_periods)
    split = PhysioDiagnostics(m, z, None).training()
    assert split["map"]["share"] < 0.9 and split["map"]["training_change_pct"] == same["map"]["training_change_pct"]
    assert -1 <= split["cp"]["corr"] <= 1


def test_priors_centre_on_the_riders_own_first_year():
    """2026-11-10: MAP 290 W / Pmax 900 W as fixed prior centres sat 2-3 sds from a much weaker or stronger rider."""
    from dataclasses import replace
    from goodpush.models.physio import PhysioInputs
    from goodpush.physio.model import FLOOR_SHARE, GENERIC_LEVEL_W
    d = synthetic(n_days=84)
    assert d.level_w is None and PhysioModel(d).spec.mean[0] == np.log(GENERIC_LEVEL_W[0])     # no level: generic
    day = np.repeat(np.r_[np.arange(0, 336, 3), 500], 2)                 # 112 rides, and one 500 days in: not the start
    n = day.size // 2
    s = np.tile([5.0, 300.0], n)
    w = np.tile([700.0, 240.0], n) * np.repeat(np.r_[0.8 + 0.2 * (np.arange(n - 1) % 9 == 0), 1.5], 2)
    lvl = PhysioInputs.own_level(day, s, w)
    assert lvl == (240.0, 700.0)                                         # each 8-week block's best: the near all-out ones
    w[2 * 5 + 1] = 1000.0                                                # 2026-11-26: a broken meter's 5 min at 1000 W
    assert PhysioInputs.own_level(day, s, w) == lvl                      # one block's best moves no median
    assert np.allclose(PhysioInputs.own_level(day, s, 0.5 * w), 0.5 * np.array(lvl))   # half the watts: half the level
    assert PhysioInputs.own_level(day[:10], s[:10], w[:10]) is None       # five rides: no level of their own yet
    own = replace(d, level_w=lvl)
    p = PhysioModel(own).spec.unpack(PhysioModel(own).spec.mean)
    assert np.isclose(p["x0"][0], np.log(lvl[0])) and np.isclose(p["x0"][3], np.log(lvl[1]))
    assert np.isclose(p["floor"][0], np.log(FLOOR_SHARE[0] * lvl[0])) and np.isclose(p["floor"][3], np.log(FLOOR_SHARE[1] * lvl[1]))
    assert own.until(56).level_w == lvl and own.window(28).level_w == lvl      # a fold or window: the same priors


def test_coverage_of_the_posterior_ranges(tmp_path, monkeypatch):
    """2026-11-10 (coverage check): per cut-off a posterior on the data before it, then each held-out observation's
    PIT from a predictive sample per draw. On data generated from the model itself the 90 % ranges hold about 90 %
    of the held-out best powers (loose: few draws, two folds); the result has its tables; a rerun fits nothing."""
    import jax
    from goodpush.checks.coverage import PosteriorCoverage
    from goodpush.physio import QUANTITIES
    d = synthetic(n_days=126)
    m = PhysioModel(d)
    z = PhysioFit(m).map(maxiter=300).z
    obs = {"e_day": jnp.asarray(d.effort_day), "e_s": jnp.asarray(d.effort_s, float),
           "e_x": jnp.asarray(m.duration_x[m.effort_t]), "e_kind": jnp.asarray(d.effort_kind),
           "r_day": jnp.asarray(d.reading_day), "r_src": jnp.asarray(d.reading_source),
           "r_q": jnp.asarray([QUANTITIES.index(d.sources[s].quantity) for s in d.reading_source])}
    E, R = PosteriorCoverage.samples(jnp.asarray(z)[None], jax.random.split(jax.random.PRNGKey(5), 1),
                                     m.spec.layout, m.arrays, obs)
    made = replace(d, effort_w=np.exp(np.asarray(E)[0]), reading_w=np.exp(np.asarray(R)[0]))   # from the model

    class Coverage(PosteriorCoverage):
        WARMUP, DRAWS, ESS, PREDICT_DRAWS, MAXITER = 20, 6, 0, 24, 300

    said = []
    cov = Coverage(made, tmp_path, folds=2, every=14, say=said.append)
    out = cov.run()
    assert [f["cut"] for f in out["folds"]] == [str(made.days[c]) for c in cov.cuts()] and len(cov.cuts()) == 2
    pit = np.array([v for f in out["folds"] for v in f["effort_pit"] + f["reading_pit"]])
    assert pit.size and np.all((pit >= 0) & (pit <= 1))
    s = out["summary"]
    assert set(s) == {"readings", "readings_by_days_ahead", "efforts_by_kind", "efforts_by_duration",
                      "efforts_by_each_duration", "efforts_by_days_ahead", "folds"}
    assert list(s["efforts_by_days_ahead"]) == ["days 0-13", "days 14-27", "days 28-41"]     # 2026-11-11: by horizon
    assert sum(t["n"] for t in s["efforts_by_days_ahead"].values()) == s["efforts_by_kind"]["all"]["n"] \
        == sum(t["n"] for t in s["efforts_by_each_duration"].values())
    assert all(0 <= a < 42 for f in out["folds"] for a in f["effort_ahead"] + f["reading_ahead"])
    assert set(s["efforts_by_duration"]) == {"<= 60 s", "3-20 min", ">= 40 min"} and s["efforts_by_kind"]["all"]["n"] > 50
    assert 0.75 <= s["efforts_by_kind"]["all"]["inside_90"] <= 1.0 and 0.3 <= s["efforts_by_kind"]["all"]["mean_pit"] <= 0.7
    assert {"inside_50", "inside_90", "mean_pit", "above_95"} <= set(s["efforts_by_kind"]["all"])
    assert any(x.startswith("Best powers by duration") for x in cov.lines(s))
    assert (tmp_path / PosteriorCoverage.FILE).exists() and any("coverage fold 2 of 2 done" in x for x in said)
    monkeypatch.setattr(PhysioFit, "map", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fitted again")))
    again = Coverage(made, tmp_path, folds=2, every=14, say=said.append).run()      # resumed: both folds there
    assert again["summary"] == s
    import json
    old = {**out, "folds": [{k: v for k, v in f.items() if not k.endswith("_ahead")} for f in out["folds"]]}
    (tmp_path / PosteriorCoverage.FILE).write_text(json.dumps(old), encoding="utf-8")   # a result from before 11-11
    assert Coverage(made, tmp_path, folds=2, every=14, say=said.append).run()["summary"] == s    # days ahead read again


def test_coverage_warns_when_the_stored_whitening_is_from_another_model(tmp_path):
    """2026-11-15: coverage says up front when the last full NUTS run's whitening can't be used (missing,
    an older model version, another sampler) - each fold then takes the full warm-up."""
    import pickle

    from goodpush.checks.coverage import PosteriorCoverage
    from goodpush.physio.model import VERSION
    path, structure = tmp_path / "physio_whitening.pkl", f"a,b|v{VERSION}"
    cov = PosteriorCoverage(synthetic(n_days=28), tmp_path, path, structure)
    assert "no stored whitening" in cov.whitening_note()
    old = {"names": [], "stages": (), "structure": f"a,b|v{VERSION - 2}", "sampler": PhysioFit.SAMPLER}
    path.write_bytes(pickle.dumps(old))
    assert f"model v{VERSION - 2}, not v{VERSION}" in cov.whitening_note()
    path.write_bytes(pickle.dumps({**old, "structure": structure, "sampler": "old"}))
    assert "another sampler" in cov.whitening_note()
    path.write_bytes(pickle.dumps({**old, "structure": structure}))
    assert cov.whitening_note() is None


def test_races_are_hard_rides_not_tests():
    """v10 (coverage 2026-11-11: races' 20+ min far below the curve): a race's best powers are kind 1 (hard) at every
    duration, even at its best - kind 3 where heart rate shows a maximal effort; a test ride keeps kind 2 up to
    TEST_MAX_S where it went near the previous best."""
    from goodpush.metrics.effort_hr import EffortHeartRate
    from goodpush.models.physio import PhysioInputs
    S = np.array(PhysioInputs.EFFORT_S, float)
    best = 400.0 - S / 30
    rides = pd.DataFrame({"workout": ["endurance", "race", "test"], "power_coverage": 1.0,
                          "moving_s": 9000.0, "duration_s": 9000.0, "date": pd.to_datetime(["2026-03-01", "2026-03-06", "2026-03-11"]),
                          **{f"mmp_{s}": best[i] for i, s in enumerate(PhysioInputs.EFFORT_S)},
                          **{f"effort_hr_{s}": [150.0, 186.0 if s == 300 else 150.0, 150.0] for s in EffortHeartRate.DURATIONS},
                          **{f"effort_coast_{s}": 0.0 for s in EffortHeartRate.DURATIONS}})
    day, s, w, kind = PhysioInputs.efforts(rides, np.array([0, 5, 10]), max_hr=187.0)
    race, test = day == 5, day == 10
    assert (kind[race & (s != 300)] == 1).all() and kind[race & (s == 300)].tolist() == [3]     # maxed: heart rate
    short = s <= PhysioInputs.TEST_MAX_S
    assert (kind[test & short] == 2).all() and (kind[test & ~short] == 1).all()                # tests as before
    assert PhysioInputs.KIND_OF_WORKOUT["race"] == 1 and PhysioInputs.KIND_OF_WORKOUT["test"] == 2


def test_durability_labelled_assumed_at_its_prior():
    """v10: durability's starting level (x0 of STATES' durability) is 'assumed' when its draws stay at the prior -
    steady all-out efforts past an hour are rare -, not when the data pull it in and away."""
    from goodpush.models.physio import PhysioDiagnostics
    from goodpush.physio.model import STATES
    m = PhysioModel(synthetic())
    z = PhysioFit(m).map(maxiter=200).z
    name = f"x0[{STATES.index('durability')}]"
    assert name in PhysioDiagnostics.MAYBE_ASSUMED
    i = m.spec.names().index(name)
    rng = np.random.default_rng(4)
    prior = np.tile(z, (200, 1))
    prior[:, i] = m.spec.mean[i] + m.spec.sd[i] * rng.standard_normal(200)
    out = PhysioDiagnostics(m, z, None, prior).assumed()
    assert [a["name"] for a in out] == [name, "pmax"]
    assert out[0]["what"] == "How well you hold power past an hour" and "rides can't show" in out[0]["short"]
    tight = prior.copy()
    tight[:, i] = m.spec.mean[i] + 2 * m.spec.sd[i] + 0.1 * m.spec.sd[i] * rng.standard_normal(200)
    assert [a["name"] for a in PhysioDiagnostics(m, z, None, tight).assumed()] == ["pmax"]


def test_an_illness_weeks_may_change_more():
    """v12 (2026-11-11): weeks that overlap a severe illness (onset .. comeback + ILLNESS_AFTER_DAYS) get
    ILLNESS_NOISE x the weekly noise, in JAX and numba alike. v13 (2026-11-12): a week's change comes in equal
    parts on its days (one step on its first day drew staircases)."""
    from goodpush.models.physio import PhysioInputs, PhysioRun
    days = pd.date_range("2026-01-05", periods=56)                     # 8 Monday weeks
    week = np.arange(56) // 7
    s = PhysioInputs.noise_scale(days, week, [("2026-01-15", "2026-01-18")])      # Thu .. Sun + 7 d = to 25 Jan
    k = PhysioInputs.ILLNESS_NOISE
    assert np.array_equal(s, np.where(np.isin(week, [1, 2]), k, 1.0))
    s2 = PhysioInputs.noise_scale(days, week, [("2026-02-23", "2026-02-24"), ("2026-01-05", "2026-01-05")])
    assert np.array_equal(s2, np.where(np.isin(week, [0, 1, 7]), k, 1.0))       # unsorted, overlapping a week edge
    assert np.all(PhysioInputs.noise_scale(days, week, []) == 1.0)
    d = synthetic(n_days=84)
    ill = replace(d, noise_scale=np.where(np.isin(d.week, [4, 5]), k, 1.0))
    assert np.array_equal(ill.until(40).noise_scale, ill.noise_scale[:40])
    assert np.array_equal(ill.window(30).noise_scale, ill.noise_scale[30:]) and np.all(d.noise_scales == 1.0)
    assert PhysioRun.key(ill) != PhysioRun.key(d) and PhysioRun.history_key(ill, 84) != PhysioRun.history_key(d, 84)
    assert PhysioRun.history_key(ill, 28) == PhysioRun.history_key(d, 28)       # before the illness: the same
    m0, m = PhysioModel(d), PhysioModel(ill)
    size = np.asarray(m.arrays["noise_size"])[:84, 0]
    assert np.allclose(size, ill.noise_scale / 7)
    z = m.spec.mean.copy()
    z[m.spec.offsets["eta"]:m.spec.offsets["eta"] + 5 * m.data.n_periods] = 1.0  # the same eta: k x the step in weeks 4-5
    X0, X = (np.asarray(mm.simulate(jnp.asarray(z))[0]) for mm in (m0, m))
    drift = np.diff(X0[:, STATES.index("pmax")])                     # Pmax only drifts: the same step every day,
    assert np.allclose(drift, NOISE_SD[STATES.index("pmax")] / 7)   # a week's seven the weekly size
    step = np.diff(X, axis=0)[27:42] - np.diff(X0, axis=0)[27:42]
    assert np.abs(step).max() > 0 and np.allclose(np.diff(X0, axis=0)[:27], np.diff(X, axis=0)[:27])
    f = PhysioFit(m)
    w0 = f.to_w(f.map(maxiter=200).z)
    for w in (w0, w0 + 0.3 * np.random.default_rng(8).standard_normal(w0.size)):
        vj, gj = f.value_and_grad(jnp.asarray(w), m.spec.layout, m.arrays)
        vn, gn = f.fast.value_and_grad(w)
        assert np.isclose(vn, float(vj), rtol=1e-10, atol=1e-8)
        assert np.allclose(gn, np.asarray(gj), rtol=1e-7, atol=1e-7 * np.abs(gj).max())


def test_severe_illnesses_from_the_analysis_reach_the_model(tmp_path, monkeypatch):
    """PhysioRun.inputs passes results.json's severe episodes (not milder ones) as illnesses."""
    from goodpush.models import physio as P
    from types import SimpleNamespace
    seen = {}

    def build(cls, *a):
        seen["ill"] = a[6]
        return SimpleNamespace(days=[])
    monkeypatch.setattr(P.PhysioInputs, "build", classmethod(build))
    monkeypatch.setattr(P.pd, "read_parquet", lambda p: pd.DataFrame())
    monkeypatch.setattr(P.PhysioRun, "keep_compiled", staticmethod(lambda path: None))
    import goodpush.models.intervals as I
    monkeypatch.setattr(I.IntervalLog, "load", staticmethod(lambda p: None))
    (tmp_path / "results.json").write_text(json.dumps({"athlete": {"lthr": 175}, "illness": {"episodes": [
        {"onset": "2023-01-19", "comeback_note": "2023-02-18", "severity": "severe"},
        {"onset": "2023-04-13", "comeback_note": "2023-04-16", "severity": "moderate"}]}}))
    proj = type("Proj", (), {"processed": tmp_path, "reports": tmp_path, "cfg": {}})()
    P.PhysioRun(proj).inputs()
    assert seen["ill"] == [("2023-01-19", "2023-02-18")]


def test_vo2_effects_on_map_held_near_the_studies_and_labelled_assumed():
    """v12: MAP's effects from CP-MAP and above-MAP work have a narrower prior (their dose always comes with the
    hours) and are 'assumed' when the draws stay there."""
    from goodpush.models.physio import PhysioDiagnostics
    from goodpush.physio.model import EFFECT_SD, EFFECT_STATE, GAIN_EFFECT, GAIN_INDEX, STATES
    m = PhysioModel(synthetic())
    names = m.spec.names()
    for e, dose in ((2, 2), (3, 3)):                                   # z3 = CP-MAP, z4 = above MAP
        k = int(np.flatnonzero(GAIN_EFFECT == e)[0])
        assert GAIN_INDEX[0][k] == STATES.index("map") and GAIN_INDEX[1][k] == dose and EFFECT_STATE[e] == 0
        assert m.spec.sd[names.index(f"log_effect[{e}]")] == 0.6 and f"log_effect[{e}]" in PhysioDiagnostics.MAYBE_ASSUMED
    assert np.allclose(np.delete(EFFECT_SD, [2, 3]), np.hypot(1.5, 0.7))
    z = PhysioFit(m).map(maxiter=200).z
    i = names.index("log_effect[2]")
    draws = np.tile(z, (200, 1))
    draws[:, i] = m.spec.mean[i] + m.spec.sd[i] * np.random.default_rng(5).standard_normal(200)
    out = PhysioDiagnostics(m, z, None, draws).assumed()
    assert [a["name"] for a in out] == ["log_effect[2]", "pmax"] and "studies" in out[0]["short"]


def with_heart_rate(d, seed=0):
    """d with heart-rate evidence on part of its riding (PhysioData.hr_hist_h): the hard day's 300 W partly strong
    evidence for severe, partly against; a third of the easy rides' 170 W at weak evidence."""
    from goodpush.physio.model import EVIDENCE
    rng = np.random.default_rng(seed)
    hr = np.zeros(d.hist_h.shape + (len(EVIDENCE),))
    f = rng.uniform(0.3, 0.7, len(d.days))
    hr[:, 30, 10] = d.hist_h[:, 30] * f
    hr[:, 30, 3] = d.hist_h[:, 30] * (1 - f) / 2
    hr[:, 17, 6] = d.hist_h[:, 17] / 3
    return replace(d, hr_hist_h=hr)


class Doses:
    """one day's doses (hours) at fixed edges: LT1 200, CP 250, MAP 310, Pmax 900 W."""
    C = dict(lt1=jnp.asarray(200.0), cp=jnp.asarray(250.0), map=jnp.asarray(310.0), pmax=jnp.asarray(900.0))
    LP = jnp.log((np.arange(150) + 0.5) * 10)

    @classmethod
    def of(cls, hist, hr=None):
        from goodpush.physio.model import DOSE_UNIT_H
        return np.asarray(PhysioModel.doses(cls.LP, cls.C, jnp.asarray(hist), jnp.asarray(0.0),
                                            None if hr is None else jnp.asarray(hr))) * DOSE_UNIT_H

    @classmethod
    def v14(cls, hist):
        """v14's split: the CP edge cut the band (z2 = a1 - a2, z3 = a2 - a3)."""
        edges = np.log([200.0, 250.0, 310.0, 310.0 + 0.5 * (900.0 - 310.0)])
        a = 1 / (1 + np.exp(-(np.asarray(cls.LP)[None, :] - edges[:, None]) / 0.05))
        return np.r_[np.concatenate([1 - a[:1], a[:-1] - a[1:], a[-1:]]) @ hist, 0.0]

    @staticmethod
    def at(watts, k, hours=1.0):
        """(hist, hr): `hours` at `watts`, all of it with heart rate in evidence bin k."""
        from goodpush.physio.model import EVIDENCE
        h, hr = np.zeros(150), np.zeros((150, len(EVIDENCE)))
        h[int(watts // 10)] = hours                  # bin centre = watts + 5
        hr[int(watts // 10), k] = hours
        return h, hr


def test_no_evidence_reproduces_the_v14_doses():
    """evidence 0 everywhere (no heart rate, planned sessions): the band split by the CP edge alone is v14's within a
    few % of the band (the products of soft edges differ only where two edges overlap)."""
    rng = np.random.default_rng(2)
    hist = np.zeros(150)
    hist[8:40] = rng.uniform(0, 0.3, 32)                   # 80-400 W, a spread of riding
    new, old = Doses.of(hist), Doses.v14(hist)
    assert np.allclose(new[[0, 3, 4]], old[[0, 3, 4]], rtol=1e-12)
    band = old[1] + old[2]
    assert new[1] + new[2] == pytest.approx(band, rel=1e-12)
    assert abs(new[2] - old[2]) < 0.03 * band and abs(new[1] - old[1]) < 0.03 * band
    zero = np.zeros((150, 12))
    assert np.array_equal(Doses.of(hist, zero), new)       # zero heart-rate time = no heart rate


def test_evidence_moves_band_hours_between_heavy_and_severe():
    from goodpush.physio.model import EVIDENCE
    top, bottom = len(EVIDENCE) - 1, 0
    h, hr = Doses.at(250, top)                              # at CP, strong evidence for severe
    plain, strong = Doses.of(h), Doses.of(h, hr)
    assert strong[2] > plain[2] + 0.3 and strong[1] + strong[2] == pytest.approx(plain[1] + plain[2], rel=1e-12)
    assert np.allclose(strong[[0, 3, 4]], plain[[0, 3, 4]])
    h, hr = Doses.at(100, top)                              # far below LT1: nothing to split
    assert np.allclose(Doses.of(h, hr), Doses.of(h), atol=1e-6)
    h, hr = Doses.at(280, bottom)                           # above CP, strong evidence against
    plain, weak = Doses.of(h), Doses.of(h, hr)
    assert weak[1] > plain[1] + 0.4 and weak[1] + weak[2] == pytest.approx(plain[1] + plain[2], rel=1e-12)


def test_heart_rate_evidence_zero_time_is_bit_identical():
    """zero time with heart rate: the same log posterior as without it, JAX and numba."""
    from goodpush.physio.model import EVIDENCE
    d = weighted(synthetic(n_days=84, hr=True))
    m0, m1 = PhysioModel(d), PhysioModel(replace(d, hr_hist_h=np.zeros(d.hist_h.shape + (len(EVIDENCE),))))
    f0, f1 = PhysioFit(m0), PhysioFit(m1)
    w = 0.3 * np.random.default_rng(1).standard_normal(m0.spec.size)
    assert float(f0.value_and_grad(jnp.asarray(w), m0.spec.layout, m0.arrays)[0]) == \
        float(f1.value_and_grad(jnp.asarray(w), m1.spec.layout, m1.arrays)[0])
    v0, g0 = f0.fast.value_and_grad(w)
    v1, g1 = f1.fast.value_and_grad(w)
    assert v0 == v1 and np.array_equal(g0, g1)


def test_numba_matches_jax_with_heart_rate_evidence():
    m = PhysioModel(with_heart_rate(weighted(synthetic(n_days=84, hr=True))))
    f = PhysioFit(m)
    w0 = f.to_w(f.map(maxiter=200).z)
    rng = np.random.default_rng(5)
    for w in (w0, w0 + 0.1 * rng.standard_normal(w0.size), w0 + 1.5 * rng.standard_normal(w0.size)):
        vj, gj = f.value_and_grad(jnp.asarray(w), m.spec.layout, m.arrays)
        vn, gn = f.fast.value_and_grad(w)
        assert np.isclose(vn, float(vj), rtol=1e-10, atol=1e-8)
        assert np.allclose(gn, np.asarray(gj), rtol=1e-7, atol=1e-7 * np.abs(gj).max())


def test_heart_rate_evidence_follows_until_window_and_forecast():
    d = with_heart_rate(synthetic(n_days=84))
    assert d.until(40).hr_hist_h.shape[0] == 40 and d.window(30).hr_hist_h.shape[0] == 54
    m = PhysioModel(d.until(56))
    z = m.spec.mean
    X0, _ = m.forecast(z, d.hist_h[56:], d.long_h[56:])
    X1, _ = m.forecast(z, **d.ridden(56, 84))
    assert not np.allclose(np.asarray(X0), np.asarray(X1))      # the ridden days' evidence counts


def test_planned_sessions_do_not_depend_on_heart_rate():
    """a forecast of planned days (no heart rate) is the same whatever the history's heart-rate evidence: the plan and
    a past ride without a strap share one formula."""
    from goodpush.physio.model import EVIDENCE
    d = synthetic(n_days=84)
    carry = (jnp.asarray(PhysioModel(d).spec.unpack(PhysioModel(d).spec.mean)["x0"]), jnp.zeros(5), jnp.asarray(0.0))
    m0, m1 = PhysioModel(d), PhysioModel(with_heart_rate(d))
    z = m0.spec.mean
    plan = d.hist_h[:28], d.long_h[:28]
    X0, _ = m0.forecast(z, *plan, carry)
    X1, _ = m1.forecast(z, *plan, carry)
    X2, _ = m1.forecast(z, *plan, carry, np.zeros(plan[0].shape + (len(EVIDENCE),)))
    assert np.array_equal(np.asarray(X0), np.asarray(X1)) and np.array_equal(np.asarray(X0), np.asarray(X2))


def mix_problem(Z, days=28):
    """two key slots (Tuesdays of weeks 1 and 3): an hour at 300 W, 10 min at 100 W, or (not allowed) 3 h at 300 W."""
    from goodpush.physio.plan import MixPlan, Target
    m = PhysioModel(synthetic())
    B = 150
    base = np.zeros((days, B))
    base[np.arange(days) % 7 == 3, 17] = 1.5                       # endurance rides
    cand = np.zeros((2, 3, B))
    cand[:, 0, 10] = 1 / 6
    cand[:, 1, 30] = 1.0
    cand[:, 2, 30] = 3.0
    ok = np.array([[True, True, False]] * 2)
    return MixPlan(m, Z, base, np.zeros(days), [1, 15], cand, ok, np.zeros(2, int),
                   [Target(days - 1, 1.0, (300.0, 1200.0), all_out=False)], float("inf"))


def test_mix_plan_picks_the_session_with_the_best_forecast():
    m = PhysioModel(synthetic())
    plan = mix_problem(m.spec.mean[None])
    r = plan.solve()
    assert list(r.choice) == [1, 1] and r.objective < r.start and r.rounds >= 1   # the hard hour, never the 3 h
    v1, _ = plan.evaluate(r.choice)
    v3, g3 = mix_problem(np.repeat(m.spec.mean[None], 3, axis=0)).evaluate(r.choice)
    assert v3 == pytest.approx(v1, rel=1e-6) and g3.shape[1] == 150                # padding draws weigh nothing


def test_physio_load_chooses_key_sessions():
    from goodpush.planning.physio_load import PhysioLoad
    from goodpush.planning.scheduler import MIX_TYPES
    from goodpush.planning.workouts import WorkoutLibrary as WL
    m = PhysioModel(synthetic())
    start = pd.Timestamp(m.data.days[-1]) + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=14)
    kinds = ["rest", "tempo", "endurance", "tempo", "rest", "long", "endurance"] * 2
    wos = [WL.rest(d) if k == "rest" else WL.intervals(d, k, 0) if k == "tempo" else WL.easy(d, k, 90)
           for d, k in zip(dates, kinds)]
    slots = {i: [WL.intervals(w.date, k, 0) for k in sorted(MIX_TYPES, key="tempo".__ne__)]
             for i, w in enumerate(wos) if w.type == "tempo"}
    draws = m.spec.mean[None] + 0.01 * np.random.default_rng(0).standard_normal((3, m.spec.size))
    for pl in (PhysioLoad(m, m.spec.mean, 250.0), PhysioLoad(m, m.spec.mean, 250.0, draws=draws)):
        picked, r = pl.choose_sessions(wos, [], slots, {})
        assert set(picked) == set(slots) and set(picked.values()) <= set(MIX_TYPES)
        assert r.objective <= r.start
    assert PhysioLoad(m, m.spec.mean, 250.0).choose_sessions(wos, [], {}, {}) == ({}, None)


def test_compiled_code_kept_apart_by_compression(tmp_path, monkeypatch):
    """jax names a cache entry the same for zstd and zlib, so each compression gets its own folder (the installed app
    on Python 3.12 and a 3.14 checkout shared one and read each other's entries wrong); entries from before go."""
    from jax._src import compilation_cache as cc
    from goodpush.models.physio import PhysioRun
    assert PhysioRun.cache_codec() == ("zstd" if cc.zstd else "zstandard" if cc.zstandard else "zlib")
    set_to = {}
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(jax.config, "update", lambda name, value: set_to.__setitem__(name, value))
    (tmp_path / "jit_f-abc-cache").write_bytes(b"x")
    (tmp_path / "jit_f-abc-atime").write_bytes(b"x")
    (tmp_path / "zlib").mkdir()
    (tmp_path / "zlib" / "jit_g-def-cache").write_bytes(b"x")
    PhysioRun.keep_compiled(tmp_path)
    assert set_to["jax_compilation_cache_dir"] == str(tmp_path / PhysioRun.cache_codec())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["zlib"]
    assert (tmp_path / "zlib" / "jit_g-def-cache").exists()


def test_a_nuts_run_says_how_far_it_is_while_it_holds_the_lock(tmp_path):
    """PhysioRun.tracked keeps PROGRESS up to date; nuts_progress reads it only while a run holds LOCK
    (the Model page shows a run started from a terminal, or the weekly refresh)."""
    from filelock import FileLock
    from goodpush.models.physio import PhysioRun
    run = PhysioRun.__new__(PhysioRun)
    run.processed = tmp_path
    said = []
    with FileLock(str(tmp_path / PhysioRun.LOCK)):
        say = run.tracked(said.append, "full")
        say("NUTS: warm-up 1-40 of 400")
        p = PhysioRun.nuts_progress(tmp_path)
        assert p["kind"] == "full" and p["line"] == "NUTS: warm-up 1-40 of 400" and p["t"] >= p["started"]
        run.untrack()
        assert PhysioRun.nuts_progress(tmp_path) is None              # held, but nothing said: nothing to show
    say("after the run")
    assert PhysioRun.nuts_progress(tmp_path) is None                  # no run holds LOCK
    assert said[-1] == "after the run" and said[0].startswith("physio model: starting")


def test_compiled_code_kept_apart_by_compression(tmp_path, monkeypatch):
    """jax names a cache entry the same for zstd and zlib, so each compression gets its own folder (the installed app
    on Python 3.12 and a 3.14 checkout shared one and read each other's entries wrong); entries from before go."""
    from jax._src import compilation_cache as cc
    from goodpush.models.physio import PhysioRun
    assert PhysioRun.cache_codec() == ("zstd" if cc.zstd else "zstandard" if cc.zstandard else "zlib")
    set_to = {}
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(jax.config, "update", lambda name, value: set_to.__setitem__(name, value))
    (tmp_path / "jit_f-abc-cache").write_bytes(b"x")
    (tmp_path / "jit_f-abc-atime").write_bytes(b"x")
    (tmp_path / "zlib").mkdir()
    (tmp_path / "zlib" / "jit_g-def-cache").write_bytes(b"x")
    PhysioRun.keep_compiled(tmp_path)
    assert set_to["jax_compilation_cache_dir"] == str(tmp_path / PhysioRun.cache_codec())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["zlib"]
    assert (tmp_path / "zlib" / "jit_g-def-cache").exists()


def test_model_page_says_what_the_fit_used_and_what_the_latest_rides_moved(tmp_path):
    """2026-11-28 Model page: the diagnostics name what the fit was given, today's best-fit numbers, and the best fit
    before the newest data day (kept through refits of the same day, dropped across model versions); the power check's
    set-aside rides are counted from the fit's first day."""
    import pandas as pd
    from types import SimpleNamespace
    from goodpush.models.physio import PhysioDiagnostics, PhysioRun
    from goodpush.physio.model import VERSION
    m = PhysioModel(synthetic(n_days=84))
    z = PhysioFit(m).map(maxiter=50).z
    run = PhysioRun(SimpleNamespace(processed=tmp_path, reports=tmp_path, cfg={}))
    run.save(m, m.data, z, np.zeros((0, m.spec.size)), "", None, progress=lambda s: None)
    f = tmp_path / PhysioDiagnostics.FILE
    d = json.loads(f.read_text(encoding="utf-8"))
    I = d["inputs"]
    assert I["first_day"] == str(m.data.days[0]) and I["days"] == len(m.data.days) == 84
    assert I["efforts"] == len(m.data.effort_w) and I["readings"] == len(m.data.reading_w)
    assert 0 < I["ride_days"] <= 84 and I["hours"] > 0
    assert d["best"]["cp"] == pytest.approx(float(m.trajectory(z)["cp"][-1]), abs=0.05)
    assert 5 < d["best"]["w_prime"] < 60 and d["previous"] is None   # kJ; nothing before the first fit
    old = {**d, "day": "2000-01-01", "best": {"cp": 1.0}}
    assert PhysioDiagnostics.previous(old, d["day"], VERSION) == {"day": "2000-01-01", "best": {"cp": 1.0}}
    same = {**d, "previous": {"day": "2000-01-01", "best": {"cp": 1.0}}}
    assert PhysioDiagnostics.previous(same, d["day"], VERSION) == same["previous"]   # a refit of the same day
    assert PhysioDiagnostics.previous(old, d["day"], VERSION + 1) is None            # another model
    acts = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-03-01", "2026-03-02", "2026-03-03"]),
                         "power_ignored": ["marked", "marked", "heart rate", ""]})
    assert PhysioDiagnostics.set_aside(acts, "2026-02-01") == {"marked": 1, "heart rate": 1}
    assert PhysioDiagnostics.set_aside(None, "2026-02-01") == {"marked": 0, "heart rate": 0}
