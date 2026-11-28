"""metrics.power_check: rides whose power is wrong - marked by you, or a best heart rate says no rider rode."""
import numpy as np
import pandas as pd

from goodpush.metrics.power import MMP_DURATIONS
from goodpush.metrics.power_check import PowerCheck


def rides(n=30, lthr=165.0):
    """n rides a day apart, all with power: 5-min bests 240-260 W at heart rates near threshold."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2017-04-01", periods=n)
    acts = pd.DataFrame({"id": [f"r{i}" for i in range(n)], "date": days, "sport": "ride", "has_power": True,
                         "power_coverage": 1.0, "np": 220.0, "np_used": 220.0, "avg_power": 200.0, "work_kj": 700.0,
                         "max_power": 800.0, "ef": 1.4, "decoupling": 2.0,
                         "power_hist": [[60] * 10 for _ in range(n)]})
    for d in MMP_DURATIONS:
        acts[f"mmp_{d}"] = 300.0 * (d / 300.0) ** -0.1
    for d in PowerCheck.SCREEN_S:
        acts[f"effort_w_{d}"] = 250.0 * (d / 300.0) ** -0.1 + rng.uniform(-10, 10, n)
        acts[f"effort_hr_{d}"] = lthr + rng.uniform(-3, 8, n)
    return acts


def test_marks_by_ride_day_and_date_range():
    acts = rides()
    pc = PowerCheck(["r3", "2017-04-10", "2017-04-20..2017-04-22", " r5 "])
    assert np.flatnonzero(pc.marked(acts)).tolist() == [3, 5, 9, 19, 20, 21]


def test_heart_rate_screen_finds_minutes_far_over_usual_at_an_easy_heart_rate():
    """2026-11-26: 4:20 at 1000 W with heart rate 132 - a broken meter, not a rider."""
    acts = rides()
    for d in (180, 300):
        acts.loc[7, f"effort_w_{d}"] = 1000.0 if d == 180 else 884.0
        acts.loc[7, f"effort_hr_{d}"] = 132.0
    acts.loc[12, "effort_w_300"], acts.loc[12, "effort_hr_300"] = 400.0, 182.0   # a real all-out: heart rate up
    assert np.flatnonzero(PowerCheck().screened(acts, 165.0)).tolist() == [7]
    assert not PowerCheck(keep=["r7"]).screened(acts, 165.0).any()                # you said it was real
    assert not PowerCheck().screened(acts.iloc[:10], 165.0).any()                  # too few rides to say what's usual
    assert not PowerCheck().screened(acts, 0.0).any()                              # no threshold heart rate


def test_apply_blanks_a_marked_rides_power_and_a_screened_rides_bests():
    acts = rides()
    acts.loc[7, ["effort_w_180", "effort_hr_180"]] = [1000.0, 132.0]
    out = PowerCheck(["r3"]).apply(acts, 165.0)
    assert out["power_ignored"].tolist().count("") == 28
    m, s = out.loc[3], out.loc[7]
    assert m["power_ignored"] == "marked" and s["power_ignored"] == "heart rate"
    assert not m["has_power"] and m["power_coverage"] == 0 and m["power_hist"] is None
    assert all(pd.isna(m[c]) for c in ("np_used", "avg_power", "mmp_300", "effort_w_300"))
    assert s["has_power"] and s["avg_power"] == 200.0 and s["power_hist"] is not None   # its time and load stay
    assert pd.isna(s["mmp_300"]) and pd.isna(s["mmp_5"])                                # ... not its bests
    assert acts.loc[3, "has_power"]                                                     # a copy: acts as it was


def test_the_ride_page_choice_edits_both_lists():
    assert PowerCheck.edited(["2017-04-10", "r1"], ["r2"], "r2", "bad") == (["2017-04-10", "r1", "r2"], [])
    assert PowerCheck.edited(["r1"], [], "r1", "keep") == ([], ["r1"])
    assert PowerCheck.edited(["r1"], ["r1"], "r1", "") == ([], [])


def test_physio_inputs_take_no_watts_from_a_ride_without_power():
    """a marked ride's alpha1 / ramp watts and steady reps don't reach the physio model."""
    from goodpush.models.physio import PhysioInputs
    acts = PowerCheck(["r3"]).apply(rides(), 165.0)
    acts["hrvt1_w"] = 150.0
    days = pd.DatetimeIndex(acts["date"])
    daily = pd.DataFrame(index=days)
    d, w, _ = PhysioInputs.readings(acts, np.arange(len(acts)), daily, None)
    assert len(d) == len(acts) - 1 and 3 not in d
    assert PhysioInputs.set_aside(acts).tolist() == ["r3"]
    reps = pd.DataFrame({"id": ["r3", "r4"], "date": acts["date"].iloc[3:5].to_numpy(), "seconds": 600, "cv": 0.03,
                         "watts": 230.0, "hr_late": 160.0})
    no_hr = acts.assign(duration_s=3600.0, avg_hr=np.nan)                              # no steady whole rides
    day, w, _, rep = PhysioInputs.steady(no_hr, np.arange(len(acts)), days, 165.0, reps)
    assert day.tolist() == [4] and rep.all()                                            # r3's rep left out
