"""The training plan: workouts, season (phases / races), load, scheduler, state, scoring, planner."""
import json
import threading
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goodpush.planning import (DEFAULT_PLAN, LADDERS, Event, LoadModel, Phase, Planner, PlanState, Scheduler, Season, SessionScorer,
                              Workout, WorkoutLibrary as WL)
from goodpush.planning.intensity import IntensityMap
from goodpush.planning.readiness import readiness
from goodpush.planning.load import RECOVERY_SHARE
from goodpush.planning.scheduler import FTP_TEST_LEVEL
from goodpush.planning.workouts import FORMAT_TITLES, KEY_TYPES, RAMP_STEPS

NO_RIDES = pd.DataFrame({"date": pd.to_datetime([]), "sport": [], "workout": []})
AVAIL = [1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 4.0]


def daily_until(day, rhr_bump=False) -> pd.DataFrame:
    """120 days of training up to `day` (rest Mondays, long Saturdays) with steady wellness."""
    from goodpush.metrics.load import pmc
    idx = pd.date_range(end=day, periods=120, freq="D")
    d = pd.DataFrame(index=idx)
    d["tss"] = np.where(idx.day_of_week == 0, 0, np.where(idx.day_of_week == 5, 150.0, 40.0))
    d["hours"] = np.where(idx.day_of_week == 0, 0, np.where(idx.day_of_week == 5, 4.0, 1.0))
    p = pmc(d["tss"])
    for c in ("ctl", "atl", "tsb", "monotony"):
        d[c] = p[c]
    d["rhr"] = 50 + np.random.default_rng(0).normal(0, .5, len(d))
    d["hrv"], d["sleep_h"], d["sleep_h_base"], d["hrv_z"] = 60.0, 7.5, 7.5, 0.0
    if rhr_bump:
        d.iloc[-2:, d.columns.get_loc("rhr")] = 56       # this morning and yesterday
    return d


def build_season(start="2026-10-19", weeks=52, build_weeks=3, kind="build", events=()) -> Season:
    s = Season([Phase(kind, start, weeks, build_weeks=build_weeks)])
    for e in events:
        s.add_event(**e)
    return s


def run_plan(st, today, acts=NO_RIDES, days=14, season=None, cfg=None, daily=None, ftp=200):
    daily = daily_until(today - pd.Timedelta(days=1)) if daily is None else daily
    ready = readiness(daily, today, override=st.get("overrides", {}).get(str(today.date())))
    span = pd.date_range(today, periods=days, freq="D") if days else None
    cfg = {"hours_per_week": 8.0, **(cfg or {})}           # the rider of daily_until rides ~9 h a week
    return Scheduler(cfg, season or build_season()).plan(st, daily, acts, ftp, today, ready, span)


@pytest.fixture
def st(tmp_path) -> PlanState:
    return PlanState.load(tmp_path / "plan_state.json")


def thr_ride(ftp, frac, reps=3, rep_min=10):
    """1 Hz power: 15 min warm-up, reps x rep_min at frac x FTP with 5 min easy, 20 min easy."""
    p = [np.full(900, 0.55 * ftp)]
    for _ in range(reps):
        p += [np.full(rep_min * 60, frac * ftp), np.full(300, 0.5 * ftp)]
    p.append(np.full(1200, 0.6 * ftp))
    return np.concatenate(p)


# --- workouts --------------------------------------------------------------------------------------
def test_workout_tss_dict_roundtrip_and_resize():
    d = pd.Timestamp("2026-10-27")
    thr = WL.intervals(d, "threshold", 2, variety=False)
    assert thr.minutes == 80 and thr.tss() == pytest.approx(sum(s.minutes / 60 * ((s.lo + s.hi) / 2) ** 2 * 100
                                                                 for s in thr.expanded()))
    assert Workout.from_dict("2026-10-27#2", thr.to_dict()).key == "2026-10-27#2"
    longer = thr.resized(100)
    assert sum(s.minutes for s in longer.expanded()) == 100 and longer.title == thr.title
    assert Workout.shape(longer.to_dict())[2] == Workout.shape(thr.to_dict())[2]      # same intervals
    lt = WL.long_tempo(d, 180, 20)
    r = lt.resized(210)
    assert r.minutes == 210 and "3:30 h" in r.title and Workout.shape(r.to_dict())[2] == Workout.shape(lt.to_dict())[2]
    with pytest.raises(ValueError):
        thr.resized(40)                                                  # shorter than the intervals
    assert WL.easy(d, "endurance", 60).resized(95).minutes == 95
    assert Workout.differs(thr.to_dict(), WL.intervals(d, "threshold", 1).to_dict())
    assert not Workout.differs(WL.easy(d, "endurance", 60).to_dict(), WL.easy(d, "endurance", 70).to_dict())


def test_zwo_matches_the_session():
    from goodpush.report.plan_out import session_zwo
    d = pd.Timestamp("2026-10-27")
    for s in (WL.intervals(d, "threshold", 0), WL.ramp_test(d), WL.ftp_test(d), WL.easy(d, "endurance", 90)):
        root = ET.fromstring(session_zwo(s))
        assert root.tag == "workout_file" and root.findtext("sportType") == "bike"
        assert root.findtext("name").startswith("GoodPush - ")
        tot = 0
        for el in root.find("workout"):
            a = el.attrib
            if el.tag == "IntervalsT":
                tot += int(a["Repeat"]) * (int(a["OnDuration"]) + int(a["OffDuration"]))
            else:
                tot += int(a["Duration"])
        assert tot == round(s.minutes * 60), s.title
    thr = ET.fromstring(session_zwo(WL.intervals(d, "threshold", 0, variety=False))).find("workout")
    assert [e.tag for e in thr] == ["Warmup", "IntervalsT", "SteadyState", "SteadyState"]
    ramp = ET.fromstring(session_zwo(WL.ramp_test(d))).find("workout")
    assert len(ramp) == len(RAMP_STEPS) + 4 and float(ramp[len(RAMP_STEPS)].attrib["Power"]) == 1.4   # + easy, Z2, cool down


def test_interval_formats_take_turns_and_score():
    from goodpush.planning import LEVELS
    from goodpush.planning.workouts import Repeat
    from goodpush.report.plan_out import blocks, session_zwo
    d = pd.Timestamp("2026-10-27")
    for kind, formats in LADDERS.items():                 # every format at every level builds and fits a file
        assert {len(f[4]) for f in formats.values()} == {LEVELS[kind]}
        for fmt in formats:
            for lv in range(LEVELS[kind]):
                w = WL.intervals(d, kind, lv, fmt)
                assert w.title.startswith(FORMAT_TITLES.get(fmt, fmt)) and w.note.startswith(formats[fmt][3]) and w.tss() > 0
                assert w.minutes == pytest.approx(sum(s.minutes for s in w.expanded()))
                tot = sum(int(a.get("Repeat", 1)) * (int(a["OnDuration"]) + int(a["OffDuration"])) if e.tag == "IntervalsT"
                          else int(a["Duration"]) for e in ET.fromstring(session_zwo(w)).find("workout") for a in [e.attrib])
                assert abs(tot - w.minutes * 60) <= len(w.expanded()), w.title       # whole seconds per step
    n = len(LADDERS["vo2max"])
    notes = [WL.intervals(d + pd.Timedelta(weeks=k), "vo2max", 3).note for k in range(n)]
    assert {next(f for f, v in LADDERS["vo2max"].items() if t.startswith(v[3])) for t in notes}         == set(LADDERS["vo2max"])                                                         # each in turn, week by week
    ou = WL.intervals(d, "threshold", 2, "Over-unders", variety=False)
    sets = ou.steps[1]                                            # 3 sets of (3 x over / under, 5 min between)
    assert isinstance(sets, Repeat) and sets.reps == 3 and sets.steps[0].reps == 3
    assert [s.kind for s in sets.steps[0].steps] == ["active", "active"]                    # the under is work
    # ridden as planned, each set scores as one rep (2026-11-10: 1-min steps split the sets apart, 0 %)
    ou1 = WL.intervals(d, "vo2max", 0, "Over-unders 1:1", variety=False)
    ride = np.concatenate([np.full(round(s.minutes * 60), (s.lo + s.hi) / 2 * 250) for s in ou1.expanded()])
    sc = SessionScorer.score(ou1, ride, IntensityMap.ftp_knots(250))
    assert sc["ratio"] >= 0.9 and {r["result"] for r in sc["reps"]} == {"hit"}
    # 2026-11-16: over-unders at most 5 overs a set; 6-min reps never longer (no pyramid), hard starts always surge
    assert all(lv[1] <= 5 for f in LADDERS.values() for v in f.values() if v[0] == "Over" for lv in v[4])
    assert [s.kind for s in ou1.steps[1].steps[0].steps] == ["active", "active"]        # 1:1 unders are still work
    for k in range(8):
        day = d + pd.Timedelta(days=3 * k)
        six = WL.intervals(day, "vo2max", k % 5, "6-min")
        assert max(s.minutes for s in six.expanded() if s.name == "Hard") == 6 and six.title.startswith("6-min ")
        for var in (True, False):
            hs = WL.intervals(day, "vo2max", k % 5, "Hard-start VO2max", variety=var)
            rep = hs.steps[-3].steps
            assert [s.name for s in rep] == ["Surge", "VO2max", "Recover"] and "hard starts" not in hs.title
            sd = WL.intervals(day, "vo2max", k % 5, "Step-down VO2max", variety=var)     # 5 min: 1.5 / 1.5 / 2, down
            rep = sd.steps[-3].steps
            assert [s.minutes for s in rep[:3]] == pytest.approx([1.5, 1.5, 2]) and sd.steps[-3].reps <= 5
            assert rep[0].lo > rep[1].lo > rep[2].lo >= 1 - 0.05 and rep[3].kind == "rest"
            assert sd.title == f"Step-down VO2max {sd.steps[-3].reps}x5 min"
    mi = WL.intervals(d, "vo2max", 4, "30/15s", variety=False)                   # 3 sets of 13 x 30 s / 15 s, 3 min between sets
    assert mi.steps[1].reps == 3 and mi.steps[1].steps[1].minutes == 3 and mi.title == "30/15 3x13"
    bl = blocks(mi.steps, 250)[1]
    assert bl["reps"] == 3 and bl["steps"][0]["reps"] == 13 and bl["steps"][1]["name"] == "Recover"
    # a ride that did the 30/15s scores on target (short reps get a short smoothing window)
    ftp = 250
    rep = np.r_[np.full(30, 1.23 * ftp), np.full(15, 0.5 * ftp)]
    ride = np.concatenate([np.full(900, 0.55 * ftp)] + [np.r_[np.tile(rep, 13), np.full(180, 0.5 * ftp)]] * 3
                          + [np.full(1200, 0.6 * ftp)])
    assert SessionScorer.score(mi, ride, IntensityMap.ftp_knots(ftp))["ratio"] >= 0.9
    assert SessionScorer.score(mi, np.full(ride.size, 0.9 * ftp), IntensityMap.ftp_knots(ftp))["ratio"] < 0.05
    from goodpush.planning.optimizer import SessionOptimizer          # specialty: anaerobic every other week
    mixes = [SessionOptimizer.phase_mix(SimpleNamespace(type="specialty", recovery=False, week=k, block=0), 0)
             for k in (0, 1)]
    assert mixes == [["threshold", "vo2max"], ["threshold", "anaerobic"]]


# --- season ----------------------------------------------------------------------------------------
def test_season_phases_events_and_storage(tmp_path):
    s = Season()
    s.add_phase(type="build", start="2026-10-19", weeks=12, build_weeks=2)
    with pytest.raises(ValueError, match="overlaps"):
        s.add_phase(type="base", start="2027-01-04", weeks=4)
    s.add_phase(type="specialty", start="2027-01-11", weeks=6)
    with pytest.raises(ValueError):
        s.add_phase(type="nap", start="2028-01-01", weeks=2)
    new = s.add_events_text("2027-02-07, A, road, Opener GP\n2027-02-14,B,crit  # comment\n\n2027-02-21", series="Spring")
    assert [(e.priority, e.kind, e.name, e.series) for e in new] == [("A", "road", "Opener GP", "Spring"),
                                                                     ("B", "crit", "Crit", "Spring"),
                                                                     ("B", "road", "Road", "Spring")]
    with pytest.raises(ValueError, match="line 2"):
        Season.parse_events("2027-03-01, A\n2027-03-02, Z")
    s.save(tmp_path)
    back = Season.load(tmp_path)
    assert [vars(p) for p in back.phases] == [vars(p) for p in s.phases] and len(back.events) == 3
    t = back.day_table(pd.date_range("2026-10-12", "2027-02-28"))
    assert (t.loc[:"2026-10-18", "phase"] == -1).all() and (t.loc["2027-02-22":, "phase"] == -1).all()
    rec = t.loc[t["recovery"]].index.to_period("W-SUN").unique().start_time
    assert [str(w.date()) for w in rec[:3]] == ["2026-11-02", "2026-11-23", "2026-12-14"]   # 2:1 cadence
    assert t.loc["2027-01-11", "type"] == "specialty" and t.loc["2027-01-11", "week"] == 0
    assert back.end == pd.Timestamp("2027-02-21")


def test_a_race_is_edited_in_place():
    s = Season()
    a, b = s.add_event(date="2027-02-07", priority="B", name="Opener"), s.add_event(date="2027-03-01", name="Later")
    e = s.edit_event(a.id, date="2027-03-08", priority="A", name="Opener GP", kind="crit")
    assert (e.id, e.date, e.priority, e.name, e.kind, e.series) == (a.id, "2027-03-08", "A", "Opener GP", "crit", "")
    assert [x.id for x in s.events] == [b.id, a.id] and len(s.events) == 2      # the same race, in date order
    with pytest.raises(ValueError):
        s.edit_event(a.id, priority="Z")                                      # checked like a new one
    assert s.events[1].priority == "A"                                        # ... and left as it was
    with pytest.raises(KeyError):
        s.edit_event("nope", name="x")



def test_phase_dates_edited_into_free_weeks_and_merged_on_request():
    s = Season()
    a = s.add_phase(type="build", start="2026-11-02", weeks=4)
    c = s.add_phase(type="base", start="2026-12-14", weeks=4)
    assert s.edit_phase(a.id, "2026-11-02", 6) == ""                       # into the free weeks after it
    assert s.phases[0].end == pd.Timestamp("2026-12-13")
    with pytest.raises(ValueError, match="overlaps Base"):
        s.edit_phase(a.id, "2026-11-02", 7)                                 # not over the next phase
    ask = s.place_phase(Phase(type="build", start="2026-10-19", weeks=2))   # right before the Build: same kind
    assert "8-week phase" in ask and "OK" not in ask and len(s.phases) == 2   # asked, nothing changed yet
    s.place_phase(Phase(type="build", start="2026-10-19", weeks=2), merge=True)
    assert [(p.id, p.start, p.weeks) for p in s.phases] == [(a.id, "2026-10-19", 8), (c.id, "2026-12-14", 4)]
    assert s.place_phase(Phase(type="build", start="2026-10-05", weeks=2, hours=8)) == ""   # other hours: no question
    assert len(s.phases) == 3


# --- load ------------------------------------------------------------------------------------------
def test_load_projection_matches_the_recurrence():
    lm, tss = LoadModel(), np.array([0, 50, 120, 0, 80.0])
    c, a = lm.project(tss, 50.0, 60.0)
    cc, aa = [], []
    x, y = 50.0, 60.0
    for t in tss:
        x += lm.k_ctl * (t - x)
        y += lm.k_atl * (t - y)
        cc.append(x)
        aa.append(y)
    assert np.allclose(c, cc) and np.allclose(a, aa)


def lp_week(kind="load", ramp=(1.5, 2.5), cap=1e4, races=(), days=7):
    return {"days": days, "kind": kind, "ramp": ramp, "cap": cap, "races": list(races)}


def test_week_targets_push_the_last_loading_week():
    from goodpush.planning.load import TSB_PUSH
    lm = LoadModel()
    weeks = [lp_week(), lp_week(), lp_week() | {"push": True}, lp_week("recovery")]
    r = lm.week_targets(weeks, 60.0, 62.0, 60.0)
    c, a = lm.project(np.repeat(r["tss"] / 7, 7), 60.0, 62.0)
    ends = np.r_[60.0, c[6::7]]
    assert np.diff(ends)[2] > 2.5                           # no ramp cap in the push week ...
    grown = r["tss"][2] / r["tss"][1]
    assert grown <= 1.15 + 1e-6                               # ... it ends at the push form, unless the
    assert c[20] - a[20] >= TSB_PUSH - 1e-6 and (c[20] - a[20] < TSB_PUSH + 0.5 or grown > 1.15 - 1e-6)  # growth cap stops it


def test_week_targets_ramp_recovery_and_taper():
    lm = LoadModel()
    weeks = [lp_week(), lp_week(), lp_week(), lp_week("recovery"), lp_week(), lp_week(races=[(6, "A")])]
    r = lm.week_targets(weeks, 50.0, 55.0, 50.0)
    assert r["ok"]
    daily = np.repeat(r["tss"] / 7, 7)                     # the LP assumes an even spread over the week
    c, a = lm.project(daily, 50.0, 55.0)
    ends = np.r_[50.0, c[6::7]]
    ramps = np.diff(ends)
    assert np.allclose(ends[1:], r["ctl"], atol=1e-6)
    assert np.all(ramps[[0, 1, 2, 4]] <= 2.5 + 1e-6) and np.all(ramps[[0, 1, 2]] >= 1.5 - 1e-6)
    assert r["tss"][3] <= RECOVERY_SHARE * 7 * ends[3] + 1e-6   # recovery week: a share of a steady week
    assert r["tss"][1] <= 1.15 * r["tss"][0] + 1e-6 and r["tss"][4] <= 1.15 * r["tss"][2] + 1e-6
    tsb_race = c[5 * 7 + 5] - a[5 * 7 + 5]                  # race on the 7th day: TSB the evening before
    assert 10 - 1e-6 <= tsb_race <= 25 + 1e-6 and r["tss"][5] < r["tss"][4]
    capped = lm.week_targets([lp_week(cap=300.0)] * 3, 50.0, 50.0, 50.0)
    assert np.allclose(capped["tss"], 300.0)               # hours bind: the most it can, ramp unmet (soft)
    rec = lm.week_targets([lp_week("recovery_phase")] * 2, 60.0, 60.0, 60.0)
    assert rec["tss"][0] <= 0.5 * 7 * 60 + 1e-6


# --- scheduler -------------------------------------------------------------------------------------
def test_plan_generate_and_adapt(st, tmp_path):
    from goodpush.report import plan_out
    today = pd.Timestamp("2026-10-20")              # Tuesday
    st["levels"]["threshold"] = FTP_TEST_LEVEL      # threshold work in: a 20-min test is worth riding
    out = run_plan(st, today)
    ss = out["sessions"]
    assert len(ss) == 14 and ss[0].date == today
    assert all(s.type == "rest" for s in ss if s.date.weekday() == 0)
    assert ss[0].type == "test"                      # no test on record -> test first
    assert out["readiness"]["status"] == "green" and all(w["hours_planned"] <= 8.3 for w in out["weeks"])
    assert any(s.type == "threshold" for s in ss)
    # red day: RHR +6 two days running -> today and tomorrow easy, next key day conditional
    st2 = PlanState.load(tmp_path / "s2.json")
    st2["levels"]["threshold"] = FTP_TEST_LEVEL
    out2 = run_plan(st2, today, daily=daily_until(today, rhr_bump=True))
    s2 = out2["sessions"]
    assert out2["readiness"]["status"] == "red" and s2[0].type in ("rest", "recovery") and s2[0].adjusted
    assert s2[1].type in ("rest", "recovery")
    nxt = next(s for s in s2[2:] if s.type in set(LADDERS) | {"test", "ramp"})
    assert "resting HR is back" in nxt.adjusted
    # progress: a threshold ride on a planned threshold day advances the ladder
    thr_day = next(s.date for s in ss if s.type == "threshold")
    lvl0 = st["levels"]["threshold"]
    st.update_progress(pd.DataFrame({"date": [thr_day], "sport": ["ride"], "workout": ["threshold"]}),
                       thr_day + pd.Timedelta(days=2))
    assert st["levels"]["threshold"] == lvl0 + 1
    out.update(ftp_source="test", today=today, cfg=dict(DEFAULT_PLAN))
    (tmp_path / "plan" / "workouts").mkdir(parents=True)
    (tmp_path / "plan" / "workouts" / "gp_2026-01-01_old.zwo").write_text("ours")         # an old plan's workout
    d = plan_out.write_all(tmp_path / "plan", out)
    assert not (d / "workouts" / "gp_2026-01-01_old.zwo").exists()
    ics = (d / "plan.ics").read_bytes().decode()
    assert ics.count("BEGIN:VEVENT") == 14 and all(len(x.encode()) <= 75 for x in ics.split("\r\n"))
    assert "SUMMARY:GoodPush - " in ics
    assert len(list((d / "workouts").glob("gp_*.fit"))) == sum(1 for s in ss if s.type != "rest")
    assert len(list((d / "workouts").glob("gp_*.txt"))) == len(list((d / "workouts").glob("gp_*.fit")))
    assert "Training plan" in (d / "plan.md").read_text(encoding="utf-8")
    pj = json.loads((d / "plan.json").read_text(encoding="utf-8"))["projection"]
    assert pj["date"][0] == "2026-10-20" and len(pj["date"]) == len(pj["ctl"]) == len(pj["tsb"]) == 14
    assert pj["tsb"][3] == pytest.approx(pj["ctl"][3] - pj["atl"][3], abs=0.11)


def test_no_ftp_test_before_threshold_work(st):
    """A new rider (or one back from a break: levels fade) gets the ramp test first; the 20-min test only once the
    plan's threshold / VO2max work has reached FTP_TEST_LEVEL."""
    today = pd.Timestamp("2026-10-20")
    out = run_plan(st, today)
    ss = out["sessions"]
    assert not any(s.type == "test" for s in ss) and not out["need_test"]
    assert any(s.type == "ramp" for s in ss)
    out = run_plan(PlanState.load(st.path.parent / "long.json"), today, days=None, season=build_season(weeks=12))
    ss = out["sessions"]
    first_test = next(s.date for s in ss if s.type == "test")
    hard = [s for s in ss if s.date < first_test and s.type in ("threshold", "vo2max")]
    assert max(sum(s.type == k for s in hard) for k in ("threshold", "vo2max")) >= FTP_TEST_LEVEL


def test_override_mark_and_lock(st):
    today = pd.Timestamp("2026-10-20")
    st["levels"]["threshold"] = FTP_TEST_LEVEL
    st["overrides"] = {"2026-10-20": "off"}
    out = run_plan(st, today, daily=daily_until(today))
    assert out["readiness"]["status"] == "amber" and "feel off" in out["readiness"]["reasons"][0]
    st["overrides"] = {}
    assert run_plan(st, today, daily=daily_until(today))["sessions"][0].type == "test"
    st.mark("2026-10-20", "done")                    # test done -> no new test, but today stays a test
    assert st["last_test"] == "2026-10-20"
    out = run_plan(st, today, daily=daily_until(today))
    assert out["sessions"][0].type == "test" and out["sessions"][0].locked
    assert st["planned"]["2026-10-20"]["status"] == "done" and not any(s.type == "test" for s in out["sessions"][1:])
    thr = next(s for s in out["sessions"] if s.type == "threshold")
    lv = st["levels"]["threshold"]
    st.mark(str(thr.date.date()), "done")
    assert st["levels"]["threshold"] == lv + 1
    st.mark(str(thr.date.date()), "")
    assert st["levels"]["threshold"] == lv


def test_ramp_test_scheduled_and_detected(st):
    today = pd.Timestamp("2026-10-20")
    st["levels"]["threshold"] = FTP_TEST_LEVEL
    out = run_plan(st, today)
    ramps = [s for s in out["sessions"] if s.type == "ramp"]
    assert out["need_ramp"] and len(ramps) == 1 and ramps[0].date.weekday() == 3       # second key day
    test_week = next(s.date for s in out["sessions"] if s.type == "test").isocalendar()[1]
    assert ramps[0].date.isocalendar()[1] != test_week
    assert max(s.hi for s in ramps[0].steps) > 1.35 and 40 < ramps[0].tss() < 120
    acts = pd.DataFrame({"date": [ramps[0].date], "sport": ["ride"], "workout": ["vo2max"], "ramp_top_w": [300.0],
                         "ramp_lt1_w": [210.0]})
    later = ramps[0].date + pd.Timedelta(days=2)
    out2 = run_plan(st, later, acts)
    assert not out2["need_ramp"] and not any(s.type == "ramp" for s in out2["sessions"])
    assert st["planned"][str(ramps[0].date.date())]["status"] == "done" and st["last_ramp"]


def test_replan_doesnt_turn_today_into_a_test(tmp_path):
    today = pd.Timestamp("2026-10-22")              # Thursday: second key day
    rode = pd.DataFrame({"date": [today], "sport": ["ride"], "workout": ["vo2max"]})
    for acts, want in ((rode, "vo2max"), (NO_RIDES, "ramp")):     # after the ride / in the morning
        st = PlanState({"planned": {"2026-10-22": {"type": "vo2max", "title": "VO2max 5x3 min", "level": 0, "minutes": 65}},
                        "last_test": "2026-10-01"}, tmp_path / f"{want}.json")
        assert run_plan(st, today, acts)["sessions"][0].type == want


def test_changes_wait_for_accept(st):
    st["last_test"] = "2026-10-15"
    mon = pd.Timestamp("2026-10-19")
    out = run_plan(st, mon)
    assert not out["proposal"] and st["accepted"]                        # first plan: taken as is
    thr = next(s for s in out["sessions"] if s.type == "threshold" and s.date > mon)
    st["levels"]["threshold"] += 1                                       # as if Tuesday's was on target
    st["planned"]["2026-10-13"] = {"type": "threshold", "status": "done", "compliance": {"result": "hit", "ratio": .95}}
    out = run_plan(st, mon)
    ch = [c for c in out["proposal"] if c["date"] == str(thr.date.date())]
    assert ch and ch[0]["from"]["title"] == thr.title and "on target" in ch[0]["why"]
    assert ch[0]["from"]["tss"] == round(thr.tss())                  # the card shows name, time and TSS
    assert next(s for s in out["sessions"] if s.date == thr.date).title == thr.title    # still the old one
    st.decide("reject", [str(thr.date.date())])
    assert not any(c["date"] == str(thr.date.date()) for c in run_plan(st, mon)["proposal"])
    st["rejected"] = {}
    run_plan(st, mon)
    assert st.decide("accept") and not st["proposal"]
    out = run_plan(st, mon)
    assert next(s for s in out["sessions"] if s.date == thr.date).title != thr.title and not out["proposal"]


def test_rejection_holds_against_small_replans(tmp_path):
    st = PlanState({}, tmp_path / "s.json")
    day = "2026-10-21"
    easy = {"type": "endurance", "title": "Endurance", "minutes": 70, "steps": []}
    st["proposal"] = {"created": day, "changes": [{"kind": "session", "date": day, "sig": "a", "session": easy}]}
    st.decide("reject")
    assert st.was_rejected(day, "a", easy)                                       # the very version
    assert st.was_rejected(day, "b", {**easy, "minutes": 60})                    # a new day's data shifted it a bit
    assert not st.was_rejected(day, "c", {**easy, "minutes": 40})                # a material change: asked again
    assert not st.was_rejected(day, "d", {**easy, "type": "threshold", "title": "Threshold"})
    st["rejected"][day] = "a"                                                    # an older entry: signature only
    assert st.was_rejected(day, "a", easy) and not st.was_rejected(day, "b", {**easy, "minutes": 60})


def test_replan_keeps_the_minutes_you_saw(st):
    st["last_test"] = st["last_ramp"] = "2026-10-15"
    mon = pd.Timestamp("2026-10-19")
    out = run_plan(st, mon, days=28)
    far = [s for s in out["sessions"] if s.type == "endurance" and s.minutes <= 100
           and (s.date - mon).days >= 14 and s.date.weekday() != 5]                 # (Saturday: the long-ride day)
    a, b = next((x, y) for x in far for y in far if x.minutes < y.minutes - 10
                and x.date.isocalendar()[1] == y.date.isocalendar()[1])
    da, db = str(a.date.date()), str(b.date.date())
    st["accepted"][da]["minutes"] += 10                          # the version you saw: 10 min moved from one to the other
    st["accepted"][db]["minutes"] -= 10
    seen = st["accepted"][da]["minutes"], st["accepted"][db]["minutes"]
    out = run_plan(st, mon, days=28)                             # the same week's load: the days stay as you saw them
    got = {str(s.date.date()): s.minutes for s in out["sessions"]}
    assert (got[da], got[db]) == seen
    week = [s.minutes for s in out["sessions"] if s.date.isocalendar()[1] == a.date.isocalendar()[1]]
    less = run_plan(st, mon, days=28, cfg={"hours_per_week": 4.0})          # a real change still moves them
    assert sum(s.minutes for s in less["sessions"] if s.date.isocalendar()[1] == a.date.isocalendar()[1]) < sum(week)


def test_missed_session_made_up_and_hard_ride_moves_next(tmp_path):
    wed = pd.Timestamp("2026-10-21")
    st = PlanState({"last_test": "2026-10-15", "last_ramp": "2026-10-15", "planned": {
        "2026-10-20": {"type": "threshold", "title": "Threshold 3x10 min", "level": 0, "status": "missed"}}}, tmp_path / "a")
    out = run_plan(st, wed, pd.DataFrame({"date": [pd.Timestamp("2026-10-20")], "sport": ["ride"], "workout": ["endurance"]}))
    mk = [s for s in out["sessions"] if s.type == "threshold" and s.date <= pd.Timestamp("2026-10-25")]
    assert len(mk) == 1 and "makeup" in mk[0].why_change
    i = out["sessions"].index(mk[0])
    assert not {out["sessions"][k].type for k in (i - 1, i + 1)} & KEY_TYPES
    tue = pd.Timestamp("2026-10-27")
    st2 = PlanState({"last_test": "2026-10-20", "last_ramp": "2026-10-20"}, tmp_path / "b")
    out = run_plan(st2, tue, pd.DataFrame({"date": [tue - pd.Timedelta(days=1)], "sport": ["ride"], "workout": ["race"]}))
    from goodpush.planning.optimizer import HARDNESS
    s0 = out["sessions"][0]                     # no hard session after the race (sweet spot / tempo at most)
    assert HARDNESS[s0.type] < 2 and "unplanned race" in s0.why_change
    assert any("yesterday's race" in s.why_change and s.type == "threshold" for s in out["sessions"][2:])



def test_ramp_test_added_to_a_day_counts_as_done(tmp_path):
    """a ramp test added to a day ("day#2"), linked and done: its day is the last ramp, and the next plan reads it."""
    wed = pd.Timestamp("2026-10-21")
    st = PlanState({"last_test": "2026-10-15", "planned": {
        "2026-10-20#2": {"type": "ramp", "title": "Ramp test", "minutes": 60, "ride": "r1", "status": "linked",
                         "compliance": {"ride": "r1", "result": "done"}}}}, tmp_path / "a")
    st.update_progress(pd.DataFrame({"date": [pd.Timestamp("2026-10-20")], "sport": ["ride"]}), wed)
    assert st["planned"]["2026-10-20#2"]["status"] == "done" and st["last_ramp"] == "2026-10-20"
    assert run_plan(st, wed)["sessions"]

def test_move_add_delete_and_resize_sessions(st):
    from goodpush.report import plan_out
    from goodpush.sync import incyclist
    thu = pd.Timestamp("2026-10-22")
    st["last_test"] = st["last_ramp"] = "2026-10-15"
    st["planned"]["2026-10-21"] = {"type": "endurance", "title": "Endurance 65 min", "minutes": 65, "status": "missed"}
    st.setdefault("accepted", {})["2026-10-21"] = WL.easy(pd.Timestamp("2026-10-21"), "endurance", 65).to_dict()
    tgt = next(s for s in run_plan(st, thu)["sessions"] if s.date > thu and s.type != "rest")
    day = str(tgt.date.date())
    st.set_lock(day, True, thu)                                         # tgt's day locked: a move lands next to it
    key = st.move("2026-10-21", day, thu)                               # yesterday's missed ride -> next to tgt
    assert key == f"{day}#2" and st["planned"]["2026-10-21"]["status"] == "moved"
    out = run_plan(st, thu)
    on_day = [s for s in out["sessions"] if s.date == tgt.date]
    assert [s.type for s in on_day][1] == "endurance" and on_day[1].key == key and not on_day[1].locked
    assert on_day[0].locked and on_day[0].type == tgt.type
    st.edit(key, 90)
    st.delete(day, thu)                                                 # locked day: a locked rest day of yours
    out = run_plan(st, thu)
    assert [(s.type, s.minutes) for s in out["sessions"] if s.date == tgt.date] == [("rest", 0), ("endurance", 90)]
    with pytest.raises(ValueError):
        st.delete("2026-10-21", thu)                                     # past: mark skipped instead
    out.update(ftp_source="t", today=thu, cfg=dict(DEFAULT_PLAN))
    names = [plan_out.fit_name(s) for s in out["sessions"] if s.type != "rest"]
    assert len(names) == len(set(names)) and any("_2_" in n for n in names)
    ids = [e["workoutId"] for e in incyclist.entries(out, "UTC")]
    assert f"goodpush-{day}-2" in ids and len(ids) == len(set(ids))
    assert any(r["key"] == key for r in plan_out.rows(out))
    nxt = next(s for s in out["sessions"] if s.type in LADDERS and s.date > tgt.date)
    k2 = st.move(str(nxt.date.date()), "2026-10-31", thu)                # future: in place of the plan's session there
    assert k2 == "2026-10-31" and str(nxt.date.date()) not in st["manual_plan"]   # no rest day left behind
    assert str(nxt.date.date()) not in st["accepted"]                   # ... the planner fills the old day afresh
    out = run_plan(st, thu)
    assert next(s for s in out["sessions"] if s.date == pd.Timestamp(k2) and not s.key).type == nxt.type   # kept
    st.edit(k2, reset=True)                                             # back to the plan's choice
    assert k2 not in st["manual_plan"]
    st.edit(str(nxt.date.date()), reset=True)
    assert next(s for s in run_plan(st, thu)["sessions"] if s.date == nxt.date and not s.key).type != "rest" or \
        str(nxt.date.date()) not in st.get("manual_plan", {})



def test_your_changes_are_unlocked_until_you_lock_them(st):
    """2026-11-15 (the rider): changes aren't locked by default - the plan starts from them but may change them; the
    calendar's lock keeps one exactly; Clear my changes gives the unlocked ones back to the plan."""
    thu = pd.Timestamp("2026-10-22")
    st["last_test"] = st["last_ramp"] = "2026-10-15"
    src = next(s for s in run_plan(st, thu)["sessions"] if s.date > thu and s.type in LADDERS)
    ds = str(src.date.date())
    st.edit(ds, src.minutes + 10)
    assert not st["manual_plan"][ds].get("locked") and st["accepted"][ds] == st["manual_plan"][ds]
    out = run_plan(st, thu)
    w = next(s for s in out["sessions"] if s.date == src.date and not s.key)
    assert not w.locked and (w.type, w.minutes) == (src.type, src.minutes + 10)   # kept: no reason to change it
    assert not [c for c in out["proposal"] if c.get("date") == ds]
    st.set_lock(ds, True, thu)
    assert next(s for s in run_plan(st, thu)["sessions"] if s.date == src.date and not s.key).locked
    other = next(s for s in out["sessions"] if s.date > src.date and s.type != "rest")
    st.set_lock(str(other.date.date()), True, thu)                      # a plan session pinned as it is
    assert st["manual_plan"][str(other.date.date())]["pinned"]
    st.set_lock(str(other.date.date()), False, thu)                     # unpinned: nothing of yours left there
    assert str(other.date.date()) not in st["manual_plan"]
    k = st.copy(ds, "2026-10-31", thu)
    assert st.clear_own(thu) == [k] and list(st["manual_plan"]) == [ds]  # the locked one stays
    st.set_lock(ds, False, thu)
    assert st.clear_own(thu) == [ds] and ds not in st["accepted"]
    with pytest.raises(ValueError):
        st.set_lock("2026-10-20", True, thu)


def test_your_own_session_is_never_proposed_back_to_you(st):
    """2026-11-15: a change made before they were also the accepted version (an older plan state) came back as a
    proposal "moved from ..." - your own session, kept by the plan, goes straight in."""
    thu = pd.Timestamp("2026-10-22")
    st["last_test"] = st["last_ramp"] = "2026-10-15"
    src = next(s for s in run_plan(st, thu)["sessions"] if s.date > thu and s.type in LADDERS)
    ds = str(src.date.date())
    st.edit(ds, src.minutes + 30)
    st["accepted"][ds] = src.to_dict()                                  # as an older state had it
    out = run_plan(st, thu)
    assert next(s for s in out["sessions"] if s.date == src.date and not s.key).minutes == src.minutes + 30
    assert not [c for c in out["proposal"] if c.get("date") == ds]


def test_a_linked_session_keeps_its_steps(st):
    """2026-11-15: a ride linked to a long ride with durability efforts turned it into plain endurance (rebuilt from
    the planned record), and unlinking proposed the original back."""
    from goodpush.planning.workouts import Step
    thu = pd.Timestamp("2026-10-22")
    ds = str(thu.date())
    steps = [Step("Z2", 60, .60, .70), Step("Durability", 20, .95, 1.0), Step("Z2", 70, .60, .70)]
    long = Workout(thu, "long", 150, "Long endurance 2:30 h, durability", steps)
    st["planned"][ds] = {"type": "long", "title": long.title, "minutes": 150, "level": None}
    st.setdefault("accepted", {})[ds] = long.to_dict()
    st.mark(ds, "done", ride="r1")
    run_plan(st, thu)
    assert [s["name"] for s in st["accepted"][ds]["steps"]] == ["Z2", "Durability", "Z2"]
    st.unlink(ds)
    out = run_plan(st, thu)
    assert not [c for c in out["proposal"] if c.get("date") == ds and Workout.differs(long.to_dict(), c["session"])
                and c["session"]["type"] == "long"]


def test_an_added_hard_session_is_one_of_the_weeks(tmp_path):
    """2026-11-15: an FTP test pasted next to a recovery-week day counted only as load - the week kept its own hard
    session and the plan trimmed easy riding instead. Now it is one of the week's quality sessions, with an easy
    day before it and no hard session next to it."""
    from goodpush.planning.optimizer import HARDNESS
    thu = pd.Timestamp("2026-10-22")
    mon = thu + pd.Timedelta(days=4)                                          # the next whole week
    week = lambda out: [s for s in out["sessions"] if mon <= s.date < mon + pd.Timedelta(days=7)]
    st0 = PlanState.load(tmp_path / "a.json")
    st0["last_test"] = st0["last_ramp"] = "2026-10-15"
    before = [s for s in week(run_plan(st0, thu, days=14)) if HARDNESS.get(s.type, 0) == 2]
    assert before                                                             # the week has its own hard sessions
    day = next(d for d in pd.date_range(mon + pd.Timedelta(days=2), periods=4)
               if all(abs((s.date - d).days) > 1 for s in before))           # a day away from them
    st1 = PlanState.load(tmp_path / "b.json")
    st1["last_test"] = st1["last_ramp"] = "2026-10-15"
    st1["manual_plan"] = {f"{day.date()}#2": WL.ftp_test(day).to_dict()}
    after = week(run_plan(st1, thu, days=14))
    mains = [s for s in after if "#" not in (s.key or "")]
    assert sum(HARDNESS.get(s.type, 0) == 2 for s in mains) == len(before) - 1     # the test is one of them
    near = {s.date: s for s in mains}
    assert near[day - pd.Timedelta(days=1)].type in ("rest", "recovery")      # an easy day before
    assert HARDNESS.get(near[day].type, 0) + HARDNESS.get(near[day + pd.Timedelta(days=1)].type, 0) <= 1


def test_a_test_you_moved_is_the_plans_test(st):
    """2026-11-15: an FTP test dragged 10 days later came back on its old day (still due by the interval)."""
    thu = pd.Timestamp("2026-10-22")
    st["last_ramp"] = "2026-10-15"                                            # no 20-min test yet: one is due
    tests = lambda out: sorted(str(s.date.date()) for s in out["sessions"] if s.type == "test")
    first = tests(run_plan(st, thu, days=42))[0]
    later = str((pd.Timestamp(first) + pd.Timedelta(days=10)).date())
    st.move(first, later, thu)
    assert tests(run_plan(st, thu, days=42)) == [later]                       # yours, and no other within 6 weeks


def test_copy_session_keeps_the_original(st):
    thu = pd.Timestamp("2026-10-22")
    st["last_test"] = st["last_ramp"] = "2026-10-15"
    src = next(s for s in run_plan(st, thu)["sessions"] if s.date > thu and s.type in LADDERS)
    k1 = st.copy(str(src.date.date()), "2026-10-31", thu)                # pasted on another day ...
    k2 = st.copy(str(src.date.date()), str(src.date.date()), thu)        # ... and on its own day: a second session
    out = run_plan(st, thu)
    assert k2 == f"{src.date.date()}#2" and "copied from" in st["manual_plan"][k1]["why_change"]
    same = [s for s in out["sessions"] if s.date == src.date]
    assert [s.type for s in same] == [src.type, src.type]                # the original stays
    assert any(s.key in (k1, "") and s.date == pd.Timestamp("2026-10-31") and s.type == src.type for s in out["sessions"])
    k3 = st.copy(str(src.date.date()), "2026-10-20", thu)                # into the past too: recorded there
    assert st["planned"][k3]["type"] == src.type


def test_plan_runs_to_the_end_of_the_season(st):
    thu = pd.Timestamp("2026-10-22")
    out = run_plan(st, thu, days=None, season=build_season(weeks=12))
    ss = out["sessions"]
    assert ss[-1].date == pd.Timestamp("2027-01-10") and sum(w["recovery"] for w in out["weeks"]) == 3
    tests = [s.date for s in ss if s.type == "test"]
    assert len(tests) == 2 and (tests[1] - tests[0]).days >= 42
    ramps = [s.date for s in ss if s.type == "ramp"]
    assert ramps and not {t.to_period("W").start_time for t in tests} & {r.to_period("W").start_time for r in ramps}
    thr = [s for s in ss if s.type == "threshold"]
    assert [s.level for s in thr] == sorted(s.level for s in thr) and thr[-1].level > thr[0].level
    st["levels"]["threshold"] += 1               # only the next 2 weeks ask; further out just updates
    out = run_plan(st, thu, days=None, season=build_season(weeks=12))
    assert out["proposal"] and all(pd.Timestamp(c["date"]) < thu + pd.Timedelta(days=14) for c in out["proposal"])


def test_specialty_gets_recovery_weeks_counted_around_races():
    s = Season()
    s.add_events_text("2027-02-21, B, mtb\n2027-02-28, B, mtb\n2027-03-07, A, mtb\n2027-05-30, A, mtb")
    long = s.add_phase(type="specialty", start="2027-04-12", weeks=7)           # no race until the taper
    assert long.build_weeks == 2 and long.recovery_weeks == [2]                 # 2 loading, easier, 2 loading, taper
    raced = s.add_phase(type="specialty", start="2027-01-18", weeks=7)          # race weeks 4-6 ease it off already
    assert raced.recovery_weeks == [2] and s.race_weeks(raced.start, 7).tolist() == [0, 0, 0, 0, 1, 1, 1]
    near = s.raced(Phase("build", "2027-02-01", 3, build_weeks=2))              # none right before a race week
    assert near.recovery_weeks == []
    assert Season().raced(Phase("specialty", "2027-04-12", 7)).recovery_weeks == [2, 5]   # no races: the cadence
    sugg = s.suggest(pd.Timestamp("2027-03-10"))
    assert sugg and all(x["build_weeks"] is None for x in sugg)


def test_phases_suggested_from_races():
    sat = pd.Timestamp("2026-10-24")
    s = Season([Phase("build", "2026-10-19", 12)])
    s.add_events_text("2027-01-31, C, mtb\n2027-02-21, B, mtb\n2027-03-21, B, mtb\n2027-04-04, A, mtb, Final")
    got = [(x["type"], x["start"], x["weeks"]) for x in s.suggest(sat)]
    assert got == [("build", "2027-01-11", 4), ("specialty", "2027-02-08", 8), ("recovery", "2027-04-05", 2)]
    assert "Final" in s.suggest(sat)[1]["why"]
    s2 = Season()                                     # nothing yours: base -> build -> specialty -> recovery
    s2.add_events_text("2027-04-04, A, mtb\n2027-01-24, B, mtb")      # racing starts 10 weeks out
    got = [(x["type"], x["start"], x["weeks"]) for x in s2.suggest(sat)]
    assert [g[0] for g in got] == ["base", "build", "specialty", "recovery"]
    assert got[2] == ("specialty", "2027-01-18", 11) and got[1][2] == 8           # from the first B race; Build 8 weeks at most
    s3 = Season()                                     # two A races: one recovery week between
    s3.add_events_text("2026-12-13, A, crit\n2027-03-07, A, road")
    kinds = [(x["type"], x["weeks"]) for x in s3.suggest(sat)]
    assert kinds == [("specialty", 7), ("recovery", 1), ("build", 3), ("specialty", 8), ("recovery", 2)]   # from next Monday
    assert Season([Phase("base", "2026-10-26", 4)]).suggest(sat) == []      # no A race, a phase ahead: yours
    for x in s.suggest(sat):                          # accepting them: they fit around yours
        s.add_phase(type=x["type"], start=x["start"], weeks=x["weeks"], build_weeks=x["build_weeks"])
    assert s.suggest(sat) == [] and Planner.signature([]) == ""


def test_deleting_every_phase_sticks(tmp_path, st):
    proj = SimpleNamespace(root=tmp_path)
    p = Planner.__new__(Planner)
    p.proj = proj
    thu = pd.Timestamp("2026-10-22")
    assert [x.type for x in p.season(st, thu).phases] == ["build"]          # first run: a 12-week Build
    s = Season.load(tmp_path)
    s.delete_phase(s.phases[0].id)
    s.save(tmp_path)
    assert p.season(st, thu).phases == []                                   # deleted: not recreated
    run_plan(st, thu, days=None, season=build_season(weeks=12))             # a season was planned ...
    assert max(st["planned"]) > "2027-01-01"
    st["planned"]["2026-10-20"] = {"type": "threshold", "status": "done"}
    out = run_plan(st, thu, days=None, season=Season())                     # ... then every phase deleted
    assert [x.date for x in out["sessions"] if x.date > thu] == []
    assert max(st["planned"]) < "2026-10-22" and st["planned"]["2026-10-20"]["status"] == "done"   # the past stays

def test_phases_races_and_availability(st):
    thu = pd.Timestamp("2026-10-22")
    season = Season([Phase("build", "2026-10-19", 3), Phase("recovery", "2026-11-16", 1),
                     Phase("specialty", "2026-11-30", 4)])                    # a gap: 23-29 Nov
    season.add_events_text("2026-12-13, A, crit, Final\n2026-12-06, B, road, Warm-up race")
    out = run_plan(st, thu, days=None, season=season, cfg={"availability": AVAIL})
    by_day = {s.date: s for s in out["sessions"]}
    assert pd.Timestamp("2026-11-25") not in by_day and max(by_day) == pd.Timestamp("2026-12-27")
    assert all(s.minutes <= AVAIL[s.date.weekday()] * 60 for s in out["sessions"] if s.type != "race")
    assert all(s.type in ("long", "rest") or s.date < thu for s in out["sessions"] if s.date.weekday() == 6
               and s.date < pd.Timestamp("2026-11-16"))                   # Sunday = the 4 h day
    rec = [s for s in out["sessions"] if pd.Timestamp("2026-11-16") <= s.date <= pd.Timestamp("2026-11-22")]
    assert not {s.type for s in rec} & set(LADDERS) and sum(s.minutes for s in rec) <= 4 * 60 + 1
    a = pd.Timestamp("2026-12-13")
    assert by_day[a].type == "race" and by_day[a - pd.Timedelta(days=1)].type == "openers"
    assert by_day[a + pd.Timedelta(days=1)].type == "recovery"
    assert by_day[a - pd.Timedelta(days=2)].type not in KEY_TYPES                    # no key day 1-2 days before A
    assert not any(s.type in ("test", "ramp") for s in out["sessions"]
                   if pd.Timestamp("2026-11-30") <= s.date <= pd.Timestamp("2026-12-13"))   # not in race weeks
    wk = {w["week"]: w for w in out["weeks"]}
    assert wk["2026-12-07"]["taper"] and wk["2026-12-07"]["tss_target"] < wk["2026-11-30"]["tss_target"]
    assert {w["phase"] for w in out["weeks"]} == {"Build", "Recovery / downtime", "Specialty"}


# --- state -----------------------------------------------------------------------------------------
def test_mark_links_ride_and_scores_decide():
    today = pd.Timestamp("2026-10-22")
    st = PlanState({"levels": {k: 2 for k in LADDERS}, "planned": {
        "2026-10-20": {"type": "threshold", "title": "Threshold 3x10", "status": "missed"},
        "2026-10-21": {"type": "endurance", "title": "Endurance"}, "2026-10-22": {"type": "vo2max", "title": "VO2max 5x3"}}})
    p = st.mark("2026-10-20", "done", ride="123")
    assert p["status"] == "linked" and p["ride"] == "123" and st["levels"]["threshold"] == 2
    p["compliance"] = {"ride": "123", "result": "hit", "ratio": 0.95}
    st.update_progress(NO_RIDES, today)
    assert p["status"] == "done" and st["levels"]["threshold"] == 3
    p = st.mark("2026-10-20", "done", ride="456")                     # another ride: step undone
    assert p["status"] == "linked" and "compliance" not in p and st["levels"]["threshold"] == 2
    p["compliance"] = {"ride": "456", "result": "low", "ratio": 0.3}
    st.update_progress(NO_RIDES, today)
    assert p["status"] == "low" and st["levels"]["threshold"] == 1
    st.mark("2026-10-20", "")
    assert not {"ride", "status", "step"} & set(st["planned"]["2026-10-20"]) and st["levels"]["threshold"] == 2
    t = st.mark("2026-10-22", "done", ride="9")                       # today's ride, linked today
    t["compliance"] = {"ride": "9", "result": "partial", "ratio": 0.7}
    st.update_progress(NO_RIDES, today)
    assert t["status"] == "partial" and st["levels"]["vo2max"] == 2
    st.mark("2026-10-21", "skipped", ride="789")
    assert "ride" not in st["planned"]["2026-10-21"] and st["planned"]["2026-10-21"]["status"] == "skipped"
    assert st.mark("2026-10-20", "done")["status"] == "done" and st["levels"]["threshold"] == 3
    st.mark("2026-10-20", "")
    assert st["levels"]["threshold"] == 2
    legacy = PlanState({"levels": {k: 1 for k in LADDERS}, "planned": {"2026-10-22": {
        "type": "vo2max", "status": "done", "manual": True, "ride": "x", "compliance": {"ride": "x", "result": "low"}}}})
    legacy.update_progress(NO_RIDES, pd.Timestamp("2026-10-23"))
    assert legacy["planned"]["2026-10-22"]["status"] == "low" and legacy["levels"]["vo2max"] == 0


def test_unlink_keeps_the_ride_off_the_session():
    st = PlanState({"levels": {k: 2 for k in LADDERS}, "planned": {
        "2026-10-20": {"type": "threshold", "status": "done", "step": 1,
                       "compliance": {"ride": "a", "result": "hit"}}}})          # matched automatically
    rides = pd.DataFrame({"sid": ["a", "b"], "day": ["2026-10-20"] * 2, "work_kj": [900.0, 300.0]})
    p = st.unlink("2026-10-20")
    assert p["not_rides"] == ["a"] and not {"status", "compliance", "step"} & set(p) and st["levels"]["threshold"] == 1
    assert SessionScorer.ride_for(p, "2026-10-20", rides, set())["sid"] == "b"      # the other ride, not the big one
    assert SessionScorer.ride_for(st.unlink("2026-10-20") | {"not_rides": ["a", "b"]}, "2026-10-20", rides, set()) is None
    p = st.mark("2026-10-20", "done", ride="a")                                      # linked after all
    assert p["ride"] == "a" and p["not_rides"] == []
    with pytest.raises(KeyError):
        st.unlink("2026-10-21")


def test_a_session_you_move_onto_a_readiness_day_is_kept_or_stays_put(st):
    """2026-11-18 (the rider): a hard session moved onto an amber morning vanished (the plan made the day easy and
    re-planned its old day). Amber: yours is kept, with the morning's note. Red: the move is refused unless locked."""
    thu = pd.Timestamp("2026-10-22")
    st["last_test"] = st["last_ramp"] = "2026-10-15"
    st["overrides"] = {str(thu.date()): "off"}                                   # amber
    out = run_plan(st, thu, daily=daily_until(thu))
    assert out["readiness"]["status"] == "amber"
    src = next(s for s in out["sessions"] if s.date > thu and s.type in ("threshold", "vo2max", "sweetspot"))
    st.move(str(src.date.date()), str(thu.date()), thu, out["readiness"])
    w = run_plan(st, thu, daily=daily_until(thu))["sessions"][0]
    assert (w.date, w.type, w.title) == (thu, src.type, src.title) and "your session, kept" in w.adjusted
    st["overrides"] = {str(thu.date()): "sick"}                                  # red
    out = run_plan(st, thu, daily=daily_until(thu))
    assert out["readiness"]["status"] == "red" and out["sessions"][0].type in ("rest", "recovery")
    hard = next(s for s in out["sessions"] if s.date > thu + pd.Timedelta(days=1) and s.type in LADDERS)
    ds, fri = str(hard.date.date()), str((thu + pd.Timedelta(days=1)).date())   # red keeps tomorrow easy too
    with pytest.raises(ValueError, match="readiness is red"):
        st.move(ds, fri, thu, out["readiness"])
    assert st.current(ds)["type"] == hard.type                                    # still where it was
    st.set_lock(ds, True, thu)                                                    # locked: your call
    key = st.move(ds, fri, thu, out["readiness"])
    assert key == fri and st["manual_plan"][key]["locked"]
    on_day = [s.type for s in run_plan(st, thu, daily=daily_until(thu))["sessions"] if s.date == pd.Timestamp(fri)]
    assert on_day == [hard.type]


def test_move_session_to_any_day_unless_linked():
    today = pd.Timestamp("2026-10-22")
    st = PlanState({"levels": {k: 2 for k in LADDERS}, "planned": {
        "2026-10-19": {"type": "rest", "title": "Rest"},
        "2026-10-20": {"type": "threshold", "title": "Threshold", "level": 2, "minutes": 60, "status": "missed"},
        "2026-10-21": {"type": "vo2max", "title": "VO2max", "level": 2, "minutes": 60, "ride": "r"},
        "2026-10-24": {"type": "tempo", "title": "Tempo", "level": 2, "minutes": 60}}})
    assert st.move("2026-10-20", "2026-10-19", today) == "2026-10-19"           # past -> past: recorded, to be scored
    assert st["planned"]["2026-10-19"]["type"] == "threshold" and "compliance" not in st["planned"]["2026-10-19"]
    assert st["planned"]["2026-10-20"]["status"] == "moved"
    assert st.move("2026-10-24", "2026-10-18", today) == "2026-10-18"           # future -> past: the future day loses it
    assert "2026-10-24" not in st.get("manual_plan", {}) and st["planned"]["2026-10-18"]["type"] == "tempo"
    with pytest.raises(ValueError):
        st.move("2026-10-21", "2026-10-25", today)                              # a ride counts for it
    assert st.copy("2026-10-19", "2026-10-15", today) == "2026-10-15"          # pasting: any day too
    assert st["planned"]["2026-10-15"]["type"] == "threshold" and st["planned"]["2026-10-19"]["type"] == "threshold"


def test_two_easy_rides_merge_into_one_longer_ride():
    today = pd.Timestamp("2026-10-22")
    st = PlanState({"planned": {
        "2026-10-20": {"type": "recovery", "title": "Recovery spin 45 min", "minutes": 45, "status": "missed"},
        "2026-10-23": {"type": "recovery", "title": "Recovery spin 45 min", "minutes": 45},
        "2026-10-24": {"type": "recovery", "title": "Recovery spin 45 min", "minutes": 45},
        "2026-10-25": {"type": "long", "title": "Long endurance 3:00 h", "minutes": 180},
        "2026-10-26": {"type": "threshold", "title": "Threshold", "level": 2, "minutes": 80},
        "2026-10-27": {"type": "endurance", "title": "Endurance 1:30 h", "minutes": 90, "ride": "r"}}})
    assert st.merge("2026-10-23", "2026-10-24", today) == "2026-10-24"
    w = st.current("2026-10-24")
    assert (w["type"], w["minutes"], w["title"]) == ("endurance", 90, "Endurance 1:30 h")     # two recoveries
    assert st.current("2026-10-23")["type"] == "rest" and "merged" in w["why_change"]        # ... and a day off
    assert all(st["manual_plan"][d].get("locked") for d in ("2026-10-23", "2026-10-24"))     # a re-plan keeps both
    st.merge("2026-10-24", "2026-10-25", today)
    assert (st.current("2026-10-25")["type"], st.current("2026-10-25")["minutes"]) == ("long", 270)
    st.merge("2026-10-20", "2026-10-25", today)                                 # a missed one from the past
    assert st["planned"]["2026-10-20"]["status"] == "moved" and st.current("2026-10-25")["minutes"] == 315
    for src, dst in (("2026-10-26", "2026-10-25"), ("2026-10-27", "2026-10-25"), ("2026-10-25", "2026-10-25#2")):
        with pytest.raises(ValueError):                                         # intervals, a ride on it, one day
            st.merge(src, dst, today)
    run_plan(st, today, days=7)                                                 # the plan keeps both days as set
    assert [st.current(d)["type"] for d in ("2026-10-23", "2026-10-24")] == ["rest", "rest"]
    assert st.current("2026-10-25")["minutes"] == 315


def test_levels_from_rides_and_by_hand(st, monkeypatch):
    from goodpush.io import streamstore
    ride = thr_ride(250, 0.97, reps=2, rep_min=20)                     # 2 x 20 min at threshold
    monkeypatch.setattr(streamstore, "read_many", lambda proc, ids, sport, cols: {"1": {"power": ride}})
    acts = pd.DataFrame({"id": [1.0, 2.0], "date": pd.to_datetime(["2026-10-10", "2026-10-12"]), "sport": ["ride"] * 2,
                         "workout": ["threshold", "vo2max"], "ftp": [200.0] * 2})         # the ride's FTP isn't used
    st["levels"]["vo2max"] = 3
    seeded = SessionScorer.seed_levels(st, acts, None, pd.Timestamp("2026-10-20"), 250.0)
    assert seeded == {"threshold": 4} and st["levels"]["vo2max"] == 3 and st["levels_seeded"] == "2026-10-20"
    st.set_level("sweetspot", 99)
    assert st["levels"]["sweetspot"] == len(LADDERS["sweetspot"]["Sweet spot"][4]) - 1
    with pytest.raises(ValueError):
        st.set_level("sprint", 1)


def test_plan_timeline_follows_the_ride():
    w = WL.intervals(pd.Timestamp("2026-10-27"), "threshold", 0, variety=False)
    knots = IntensityMap.ftp_knots(250.0)
    blocks = SessionScorer.planned_blocks(w, knots)
    n = len(blocks)
    reps = [{"start": 1500 + 900 * i, "ride_s": int(b[:, 0].sum()) - 60} for i, b in enumerate(blocks)]
    reps[n // 2] = {"planned_s": 600}                                              # one rep missed
    tl = SessionScorer.timeline(w, knots, reps, 99999)
    work = tl[tl[:, 4] > 0]
    starts = work[np.r_[True, np.diff(work[:, 0]) > 0] & np.r_[True, work[1:, 0] != work[:-1, 1]], 0]
    assert len(starts) == n and starts[0] == 1500 and (np.diff(tl[:, 0]) >= 0).all()   # pinned to the ride's reps
    assert (tl[:, 4] == 2).any() and tl[tl[:, 4] == 2, 0].min() > 1500                  # the missed one, between them
    plain = SessionScorer.timeline(w, knots, [], 99999)                         # nothing matched: the plan's clock
    assert tl[0, 0] == 1500 - plain[plain[:, 4] > 0][0, 0]                              # the warm-up just before rep 1
    late = SessionScorer.timeline(w, knots, [], 99999, 1151)                     # recorded 19 min before pedalling
    assert np.allclose(late[:, :2], plain[:, :2] + 1151) and np.allclose(SessionScorer.timeline(w, knots, reps, 99999, 1151), tl)
    assert tl[:, 1].max() <= 99999 and SessionScorer.timeline(w, knots, [], 600)[:, 1].max() == 600   # cut at the ride's end
    assert np.allclose(tl[tl[:, 4] == 1, 2], blocks[0][0, 1])                          # watts as scored
    assert np.isclose(work[(work[:, 0] >= 1500) & (work[:, 1] <= 1500 + blocks[0][:, 0].sum()), 1].max() - 1500,
                      blocks[0][:, 0].sum())                                             # the rep's planned length


def test_scores_move_the_ladder():
    st = PlanState({"levels": {k: 2 for k in LADDERS}, "planned": {
        "2026-10-13": {"type": "threshold", "compliance": {"result": "hit"}},
        "2026-10-15": {"type": "vo2max", "compliance": {"result": "partial"}},
        "2026-10-16": {"type": "sweetspot", "compliance": {"result": "low"}},
        "2026-10-17": {"type": "tempo", "compliance": {"result": "other"}}}})
    st.update_progress(NO_RIDES, pd.Timestamp("2026-10-20"))
    assert st["levels"] == {"threshold": 3, "vo2max": 2, "sweetspot": 1, "tempo": 2, "anaerobic": 2, "durability": 2}
    assert [st["planned"][d]["status"] for d in sorted(st["planned"])] == ["done", "partial", "low", "substituted"]


def test_plan_rebuild_keeps_session_marked_meanwhile(tmp_path, monkeypatch):
    from goodpush.config import init_project
    from goodpush.report import plan_out
    proj = init_project(tmp_path)
    sp = proj.processed / "plan_state.json"
    PlanState({"planned": {"2026-10-20": {"type": "threshold", "title": "T"}}}, sp).save()

    def app_link():                         # what the Calendar does when you drop a ride
        with PlanState.open(sp) as st:
            st.mark("2026-10-20", "done", ride="r1")
    click = threading.Thread(target=app_link)

    def slow_plan(self, state, *a, **k):
        click.start()                       # the drop arrives while the plan is being rebuilt
        time.sleep(0.3)
        state["planned"]["2026-10-27"] = {"type": "rest"}
        return {"readiness": {"status": "green"}, "proposal": []}
    monkeypatch.setattr(Scheduler, "plan", slow_plan)
    monkeypatch.setattr(Planner, "readiness", lambda *a: {"status": "green"})
    monkeypatch.setattr(plan_out, "write_all", lambda *a, **k: tmp_path)
    days = pd.date_range("2026-10-01", "2026-10-22")
    acts = pd.DataFrame({"date": [pd.Timestamp("2026-10-15")], "name": ["a ride"]})       # not a new rider
    Planner(proj).build({"daily": pd.DataFrame({"ftp": 200.0}, index=days), "acts": acts},
                        today=pd.Timestamp("2026-10-22"), progress=lambda *a: None)
    click.join()
    st = PlanState.load(sp)
    assert st["planned"]["2026-10-20"].get("ride") == "r1" and "2026-10-27" in st["planned"]
    assert [p.type for p in Season.load(tmp_path).phases] == ["build"]       # first run: your plan -> Build


# --- scoring ---------------------------------------------------------------------------------------
def test_session_score():
    s = WL.intervals(pd.Timestamp("2026-10-20"), "threshold", 2, "Threshold", variety=False)  # 3 x 10 min
    good = SessionScorer.score(s, thr_ride(250, 0.97), IntensityMap.ftp_knots(250))
    assert good["ratio"] >= 0.95 and good["planned_s"] == 1800 and 0.97 < good["avg_pct"] < 1.02
    assert SessionScorer.score(s, thr_ride(250, 0.97, reps=2), IntensityMap.ftp_knots(250))["ratio"] == pytest.approx(2 / 3, abs=0.03)
    assert SessionScorer.score(s, thr_ride(250, 0.80), IntensityMap.ftp_knots(250))["ratio"] < 0.05
    hot = SessionScorer.score(s, thr_ride(250, 1.03), IntensityMap.ftp_knots(250))
    assert hot["ratio"] >= 0.95 and hot["avg_pct"] > 1.03
    assert SessionScorer.score(WL.easy(pd.Timestamp("2026-10-20"), "endurance", 60), thr_ride(250, 0.6), IntensityMap.ftp_knots(250)) is None
    vo2 = SessionScorer.score(WL.intervals(pd.Timestamp("2026-10-22"), "vo2max", 0, "VO2max"), thr_ride(250, 1.35, 5, 3), IntensityMap.ftp_knots(250))
    assert vo2["ratio"] < 0.2 and vo2["above_s"] >= 0.5 * vo2["planned_s"] and vo2["avg_pct"] > 1.08   # over



def test_session_score_rep_by_rep():
    """5 x 3 min VO2max: an opener first (not a rep), rep 1 on target, rep 2 low, rep 3 cut short, then home."""
    s = WL.intervals(pd.Timestamp("2026-10-22"), "vo2max", 0, "VO2max", variety=False)       # 5 x 3 min at 108-115 %
    ftp, rng = 250.0, np.random.default_rng(3)
    on = lambda sec, f: np.full(sec, f * ftp)                                  # noqa: E731
    ride = np.concatenate([on(600, .6), on(60, 1.1), on(300, .55), on(180, 1.11), on(180, .5), on(180, .98), on(180, .5),
                           on(120, 1.11), on(1200, .55)]) + rng.normal(0, 3, 3000)
    sc = SessionScorer.score(s, ride, IntensityMap.ftp_knots(ftp))
    assert [r["result"] for r in sc["reps"]] == ["hit", "low", "short", "missed", "missed"]
    assert sc["reps"][0]["start"] == pytest.approx(960, abs=3) and sc["reps"][0]["lo"] == round(1.08 * ftp)
    assert sc["ratio"] == pytest.approx((180 + 120) / 900, abs=0.03)                     # rep 1 + rep 3's 2 min
    # your thresholds instead of textbook %FTP: CP 275 W moves the targets up, the same ride falls short
    hi = SessionScorer.score(s, ride, np.array([0, 201, 275, 339, 424, 953.0]))
    assert hi["reps"][0]["lo"] > sc["reps"][0]["lo"] and hi["reps"][0]["result"] == "low"


def test_easy_days_and_one_ride_per_session(tmp_path):
    today = pd.Timestamp("2026-10-26")
    st = PlanState({"planned": {
        "2026-10-21": {"type": "endurance", "title": "Endurance 65 min", "minutes": 65, "status": "n/a"},
        "2026-10-22": {"type": "endurance", "minutes": 60}, "2026-10-23": {"type": "recovery", "minutes": 45},
        "2026-10-23#2": {"type": "endurance", "minutes": 90},
        "2026-10-24": {"type": "long", "minutes": 220}, "2026-10-25": {"type": "rest", "minutes": 0}}})
    acts = pd.DataFrame({"id": list("abcd"), "date": pd.to_datetime(["2026-10-22", "2026-10-23", "2026-10-23", "2026-10-24"]),
                         "sport": "ride", "workout": ["endurance", "recovery", "endurance", "race"],
                         "duration_s": [3600.0, 1200.0, 5400.0, 7200.0], "work_kj": [600.0, 150.0, 900.0, 1500.0],
                         "ftp": 250.0})
    SessionScorer.update(st, acts, tmp_path, today)
    c = {k: (p["compliance"]["result"], p["compliance"]["ride"]) for k, p in st["planned"].items() if p.get("compliance")}
    # the day's main session takes the bigger ride, the added one the other ride
    assert c == {"2026-10-21": ("none", None), "2026-10-22": ("done", "a"), "2026-10-23": ("done", "c"),
                 "2026-10-23#2": ("short", "b"), "2026-10-24": ("other", "d")}
    st.update_progress(acts, today)
    assert st["planned"]["2026-10-21"]["status"] == "missed" and st["planned"]["2026-10-25"]["status"] == "n/a"
    assert st["levels"] == {k: 0 for k in LADDERS}                                 # easy days don't move ladders


def test_ftp_suggestion():
    today = pd.Timestamp("2026-10-30")
    hot = {"result": "hit", "ratio": 0.97, "avg_pct": 1.05}
    st = PlanState({"planned": {d: {"type": t, "compliance": dict(hot)} for d, t in
                                (("2026-10-13", "threshold"), ("2026-10-20", "sweetspot"), ("2026-10-27", "threshold"))}})
    assert SessionScorer.ftp_suggestion(st, 250, today)["to"] == 258
    st["planned"]["2026-10-20"]["compliance"]["avg_pct"] = 1.0
    assert SessionScorer.ftp_suggestion(st, 250, today) is None
    low = {"result": "low", "ratio": 0.3, "avg_pct": 0.92}
    st2 = PlanState({"planned": {d: {"type": "threshold", "compliance": low} for d in ("2026-10-20", "2026-10-27")}})
    assert SessionScorer.ftp_suggestion(st2, 250, today)["to"] == 242


# --- outputs ---------------------------------------------------------------------------------------
def test_incyclist_calendar_keeps_other_entries(st, tmp_path):
    from goodpush.sync import incyclist as I
    cfg = {"incyclist": {"sync": True, "folder": str(tmp_path / "Incyclist")}}
    (tmp_path / "Incyclist" / "workouts").mkdir(parents=True)
    cal = I.calendar_path(cfg)
    other = {"day": "2026-10-21T07:00:00.000Z", "workoutId": "x1", "source": "other", "workout": {"name": "theirs"}}
    old_ours = {"day": "2026-10-19T07:00:00.000Z", "workoutId": "goodpush-2026-10-19", "source": "goodpush", "workout": {}}
    stale = {"day": "2026-10-25T07:00:00.000Z", "workoutId": "goodpush-2026-10-25", "source": "goodpush", "workout": {}}
    cal.write_text(json.dumps([other, old_ours, stale]))
    today = pd.Timestamp("2026-10-20")
    out = run_plan(st, today)
    out["today"] = today
    r = I.sync(out, cfg, "America/Los_Angeles", progress=lambda *a: None)
    data = json.loads(cal.read_text())
    ours = [e for e in data if e["source"] == "goodpush"]
    workouts = [s for s in out["sessions"] if s.type != "rest"]
    assert other in data and old_ours in data and stale not in data
    assert r["written"] == len(workouts) == len(ours) - 1
    first = next(e for e in ours if e["workoutId"] == "goodpush-2026-10-20")
    assert first["day"] == "2026-10-20T07:00:00.000Z" and first["workout"]["name"] == "GoodPush - " + workouts[0].title

    def secs(steps):
        return sum(x["duration"] if x["type"] == "step" else x["repeat"] * secs(x["steps"]) for x in steps)
    for e, s in zip(sorted(ours[1:], key=lambda e: e["day"]), workouts):
        assert secs(e["workout"]["steps"]) == round(s.minutes * 60), s.title
    with pytest.raises(FileNotFoundError):
        I.sync(out, {"incyclist": {"folder": str(tmp_path / "nope")}}, "UTC", progress=lambda *a: None)


def test_planner_writes_incyclist_only_when_switched_on(tmp_path, monkeypatch):
    from goodpush.sync import incyclist as I
    proj = SimpleNamespace(cfg={"incyclist": {"sync": False}}, tz="UTC", processed=tmp_path)
    monkeypatch.setattr(I, "sync", lambda *a, **k: pytest.fail("must not write"))
    assert Planner(proj).sync_incyclist({}, progress=lambda *a: None) is None
    monkeypatch.setattr(I, "sync", lambda out, cfg, tz, progress: {"written": 3})
    proj.cfg["incyclist"]["sync"] = True
    assert Planner(proj).sync_incyclist({}, progress=lambda *a: None) == {"written": 3}
    monkeypatch.setattr(I, "sync", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no folder")))
    msgs = []
    assert Planner(proj).sync_incyclist({}, progress=msgs.append) is None and "no folder" in msgs[0]


def test_easy_variants_rotate_keep_length_and_respect_week_type():
    from goodpush.planning.workouts import EASY_VARIANTS, WorkoutLibrary as WL
    days = pd.date_range("2026-11-02", periods=10)
    rides = [WL.varied(WL.easy(d, "endurance", 90), middle_ok=True) for d in days]
    titles = {w.title for w in rides}
    assert len(titles) == len(EASY_VARIANTS["endurance"])               # every variant shows up
    assert all(sum(s.minutes for s in w.expanded()) == pytest.approx(90) for w in rides)
    assert all(rides[i].title != rides[i + 1].title for i in range(len(rides) - 1))
    calm = [WL.varied(WL.easy(d, "endurance", 90), middle_ok=False) for d in days]
    assert not any("low cadence" in w.title for w in calm)                # tempo-like only in loading weeks
    short = WL.varied(WL.easy(days[0], "endurance", 30), middle_ok=True)
    assert short.title == "Endurance 30 min"                             # too short for any variant
    long = [WL.varied(WL.easy(d, "long", 180), middle_ok=True) for d in days[:3]]
    assert {w.title.split(", ")[-1] for w in long} == {"Long endurance 3:00 h", "tempo finish", "late sweet spot"}


def test_varied_ride_resizes_keeping_its_structure():
    from goodpush.planning.workouts import WorkoutLibrary as WL
    d = next(d for d in pd.date_range("2026-11-02", periods=10)
             if "spin-ups" in WL.varied(WL.easy(d, "endurance", 90), True).title)
    w = WL.varied(WL.easy(d, "endurance", 90), True)
    r = w.resized(120)
    assert "spin-ups" in r.title and sum(s.minutes for s in r.expanded()) == pytest.approx(120)


def test_tempo_formats_take_turns():
    from goodpush.planning.workouts import WorkoutLibrary as WL
    titles = [WL.intervals(d, "tempo", 1).title for d in pd.date_range("2026-11-02", periods=3, freq="7D")]
    assert {next(f for f in sorted(LADDERS["tempo"], key=len, reverse=True) if t.startswith(f)) for t in titles}         == set(LADDERS["tempo"])


def test_polarized_quality_only_above_lt2():
    middle = {"tempo", "sweetspot", "threshold"}
    for phase in ("base", "build", "specialty", "maintenance"):
        for block in range(3):
            for week in range(3):
                pol = Scheduler.quality(phase, False, block, week, "polarized")
                assert pol and not middle & set(pol)
                for rec in (False, True):                     # as many as pyramidal: the count is one rule
                    assert (len(Scheduler.quality(phase, rec, block, week, "polarized"))
                            == len(Scheduler.quality(phase, rec, block, week)) == Scheduler.quality_count(phase, rec))
    assert Scheduler.quality("base", False, 0, 0) == ["sweetspot"]       # pyramidal base: one middle-zone day
    assert Scheduler.quality("build", True, 0, 0) == []                  # a recovery week: its LSCT instead


def test_tests_in_a_block_make_up_only_the_missing_load():
    d = pd.Timestamp("2026-11-03")
    for t in (WL.ftp_test(d), WL.ramp_test(d)):
        assert Scheduler.test_extra(t, 300, WL.easy(d, "recovery", 20)) is t        # as much TSS already: as it is
        big = WL.easy(d, "endurance", 30).resized(t.minutes + 180)                    # far more TSS than the test
        x = Scheduler.test_extra(t, 300, big)
        assert x.minutes > t.minutes + 10 and sum(s.minutes for s in x.expanded()) == pytest.approx(x.minutes)
        assert x.tss() >= big.tss() - 5 and x.tss() < big.tss() + 10            # made up, not more (5-min steps)
        assert [s for s in x.expanded() if s.lo >= 1.0] == [s for s in t.expanded() if s.lo >= 1.0]   # the test itself unchanged
        assert Scheduler.test_extra(t, 90, big).minutes == pytest.approx(90)               # within the day's time
        assert Scheduler.test_extra(t, t.minutes + 5, big).minutes == t.minutes           # no room: as it is


def test_week_totals_in_your_watts_scale_target_and_planned_together():
    from goodpush.planning.workouts import WorkoutLibrary as WL
    days = pd.date_range("2026-11-02", periods=3)
    generic = [WL.easy(d, "endurance", 60) for d in days]
    mapped = [replace_power(w, 1.1) for w in generic]                      # 10 % more power
    gen = sum(w.tss() for w in generic)
    out = {"sessions": mapped, "ctl_now": 40.0, "projection": [],
           "weeks": [{"week": "2026-11-02", "tss_planned": round(gen + 50), "tss_target": round(gen + 50 + 20), "ctl_end": 0}]}
    daily = pd.DataFrame({"ctl": [40.0], "atl": [45.0]}, index=[pd.Timestamp("2026-11-01")])
    sched = Scheduler({}, Season([], []))
    Planner.in_your_watts(out, generic, sched, daily, pd.Timestamp("2026-11-02"))
    r = out["weeks"][0]
    ratio = sum(w.tss() for w in mapped) / gen
    assert r["tss_planned"] == pytest.approx(50 + gen * ratio, abs=1)          # done rides (50) unchanged
    assert r["tss_target"] == pytest.approx(50 + (gen + 20) * ratio, abs=1)
    assert len(out["projection"]) == 3 and r["ctl_end"] == pytest.approx(out["projection"][-1][1], abs=0.1)


def replace_power(w, f):
    from dataclasses import replace
    return replace(w, steps=[replace(s, lo=s.lo * f, hi=s.hi * f) for s in w.steps])


def test_readiness_two_short_nights_and_late_bedtimes():
    days = pd.date_range("2026-09-01", periods=60)
    d = pd.DataFrame({"rhr": 48.0, "hrv": 60.0, "sleep_h": 7.6, "sleep_h_base": 7.6, "hrv_z": 0.0, "tsb": 0.0,
                      "monotony": 1.0, "bedtime_h": 22.75}, index=days)
    today = days[-1]
    assert readiness(d, today)["status"] == "green"
    one = d.copy()
    one.loc[today, "sleep_h"] = 6.0                                    # one short night: fine
    assert readiness(one, today)["status"] == "green"
    two = one.copy()
    two.loc[days[-2], "sleep_h"] = 6.0                                 # two: amber
    r = readiness(two, today)
    assert r["status"] == "amber" and any("two short nights" in x for x in r["reasons"])
    late = d.copy()
    late.loc[days[-2:], "bedtime_h"] = [23.75, 0.25]                    # 1 h and 1.5 h late (after midnight)
    r = readiness(late, today)                                          # one sign on its own: a note only
    assert r["status"] == "green" and any("bedtime 60 min later" in x for x in r["notes"])
    late["hrv_z"] = -1.2                                                # with a low HRV: amber
    r = readiness(late, today)
    assert r["status"] == "amber" and any("bedtime 60 min later" in x for x in r["reasons"])
    once = d.copy()
    once.loc[today, "bedtime_h"] = 0.5                                  # one late night: fine
    assert readiness(once, today)["status"] == "green"


def test_readiness_gives_no_monotony_note():
    days = pd.date_range("2026-09-01", periods=60)
    d = pd.DataFrame({"rhr": 48.0, "hrv": 60.0, "sleep_h": 7.6, "sleep_h_base": 7.6, "hrv_z": 0.0, "tsb": 0.0,
                      "monotony": 2.5, "bedtime_h": 22.75}, index=days)
    r = readiness(d, days[-1])
    assert r["status"] == "green" and not any("monotony" in x for x in r["notes"] + r["reasons"])


def test_endurance_is_lower_to_mid_zone_everywhere():
    from goodpush.planning.workouts import EASY, Z2, WorkoutLibrary as WL
    assert Z2 == EASY["endurance"] == EASY["long"] and Z2[1] <= 0.65 < 0.75              # well below LT1 (0.75)
    d = pd.Timestamp("2026-11-02")
    w = WL.easy(d, "endurance", 90)
    assert w.z2_band() == Z2 and w.resized(120).z2_band() == Z2
    sessions = [WL.varied(WL.easy(d + pd.Timedelta(days=i), k, 180), True) for i in range(5) for k in ("endurance", "long")]
    sessions += [WL.intervals(d, k, 0) for k in ("threshold", "vo2max", "sweetspot")]
    sessions += [WL.durability(WL.easy(d, "long", 180), 2), WL.long_tempo(d, 180, 20), WL.openers(d)]
    z2 = [s for w in sessions for s in w.expanded() if s.name == "Z2"]
    assert len(z2) > 20 and all((s.lo, s.hi) == Z2 for s in z2)


def test_nested_repeats_ramps_cadence_and_cues():
    from goodpush.planning.workouts import Repeat, Step
    from goodpush.report.plan_out import blocks, session_steps, session_zwo, steps_text
    d = pd.Timestamp("2026-10-27")
    w = Workout(d, "vo2max", 0, "Ramps and sets", [
        Step("Warm up", 10, .50, .60, "warmup", lo_end=.70, hi_end=.80),
        Repeat(2, [Repeat(3, [Step("On", 1, 1.15, 1.20, cadence=100, cue="spin it up"), Step("Off", 1, .50, .55, "rest")]),
                   Step("Recover", 4, .50, .55, "rest")]),
        Step("Big gear", 5, .80, .85, cadence=60, mode="sim"), Step("Cool down", 5, .55, .40, "cooldown")])
    w.minutes = w.total_minutes()
    assert w.minutes == 10 + 2 * (6 + 4) + 10 and len(w.expanded()) == 1 + 2 * 7 + 2
    ramp = w.steps[0]
    assert ramp.mean_sq() == pytest.approx((.55 ** 2 + .55 * .75 + .75 ** 2) / 3)       # TSS along the slope
    pieces = ramp.pieces()
    assert len(pieces) == 10 and sum(p.minutes for p in pieces) == pytest.approx(10) and pieces[0].mid < pieces[-1].mid
    back = Workout.from_dict("2026-10-27", w.to_dict())                                    # nested JSON round trip
    assert back.steps == w.steps and back.tss() == pytest.approx(w.tss())
    old = {"type": "openers", "steps": [s.to_dict() for s in WL.openers(d).expanded()[:3]] + [WL.openers(d).steps[-2].to_dict()],
           "repeat": [1, 2, 3]}                                                            # an older copy: one repeat
    assert isinstance(Workout.from_dict("2026-10-27", old).steps[1], Repeat)
    assert Workout.shape(old)[2] == Workout.shape(Workout.from_dict("2026-10-27", old).to_dict())[2]
    out = session_steps(w, 200)
    assert out["steps"][0]["lo_end"] == 140 and out["blocks"][1]["steps"][0]["reps"] == 3
    assert "→ 140-160 W" in out["workout"] and "2 x (3 x (" in steps_text(w.steps, 200)
    assert blocks(w.steps, 200, split_ramps=True)[0]["name"] == "Warm up" and len(blocks(w.steps, 200, True)) == 10 + 3
    zwo = ET.fromstring(session_zwo(w)).find("workout")
    tags = [e.tag for e in zwo]
    assert tags[0] == "Ramp" and tags.count("SteadyState") >= 2 * 3 * 2
    on = next(e for e in zwo if e.find("textevent") is not None)
    assert on.attrib["Cadence"] == "100" and on.find("textevent").attrib["message"] == "spin it up"
    assert sum(int(e.attrib["Duration"]) for e in zwo) == round(w.minutes * 60)


def test_interval_variety_keeps_the_work_and_takes_turns():
    from goodpush.planning.steps import WORK
    from goodpush.planning.variety import FEWEST, PATTERNS, SHORTEST, IntervalVariety as IV
    d = pd.Timestamp("2026-11-02")
    for kind, formats in LADDERS.items():
        for on_name, off_name, off_band, _, ladder in formats.values():
            for sets, reps, on, off, lo, hi, _ in ladder:
                if sets != 1:
                    continue
                plain = Workout(d, kind, 0, "", IV.main_set("straight", on_name, off_name, reps, on, off, (lo, hi), off_band)[0])
                work = lambda w: sum(s.minutes for s in w.expanded() if s.kind == "active" and s.lo >= WORK)  # noqa: E731
                for pat in PATTERNS.get(kind, ()):
                    if on < SHORTEST[pat] or reps < FEWEST.get(pat, 1):
                        continue
                    w = Workout(d, kind, 0, "", IV.main_set(pat, on_name, off_name, reps, on, off, (lo, hi), off_band)[0])
                    assert w.tss() == pytest.approx(plain.tss(), rel=.05), (kind, pat, reps, on)     # the same load
                    if pat != "ramp-in":                                  # its ramp minute starts below the band
                        assert abs(work(w) - work(plain)) <= 1, (kind, pat, reps, on)
    a, b = WL.intervals(d, "threshold", 2, "Threshold"), WL.intervals(d, "threshold", 2, "Threshold")
    assert a == b                                                         # a day keeps its session on every rebuild
    n = len(PATTERNS["threshold"])
    start = pd.Timestamp.fromordinal((d.toordinal() // 7 // n * n + n) * 7)       # the first week of a cycle
    seen = [IV.pattern(start + pd.Timedelta(weeks=k), "threshold", 20, 3) for k in range(n)]
    assert sorted(seen) == sorted(PATTERNS["threshold"])                  # every shape once a cycle
    assert "pyramid" not in {IV.pattern(start + pd.Timedelta(weeks=k), "threshold", 20, 2) for k in range(n)}
    assert WL.intervals(d, "vo2max", 3, "30/15s").steps[-3].reps == 3     # micro intervals keep their protocol
    tue, thu = pd.Timestamp("2026-11-03"), pd.Timestamp("2026-11-05")
    for kind in ("threshold", "vo2max", "sweetspot"):                   # twice in a week: two different sessions
        assert WL.intervals(tue, kind, 2).title != WL.intervals(thu, kind, 2).title, kind


def test_rest_or_a_spin_before_threshold_and_harder_days(tmp_path):
    from goodpush.planning.workouts import AT_THRESHOLD_UP, EASY_BEFORE
    today = pd.Timestamp("2026-11-02")
    for kind, style, avail in (("build", "pyramidal", None), ("build", "polarized", AVAIL), ("specialty", "polarized", AVAIL),
                               ("base", "pyramidal", AVAIL)):
        st = PlanState.load(tmp_path / f"{kind}{style}.json")
        cfg = {"intensity": style, **({"availability": avail} if avail else {})}
        out = run_plan(st, today, days=28, season=build_season(kind=kind), cfg=cfg)
        s = [w for w in out["sessions"] if not w.key]
        hard = [(a, b) for a, b in zip(s, s[1:]) if b.type in AT_THRESHOLD_UP and (b.date - a.date).days == 1]
        assert all(a.type in EASY_BEFORE for a, _ in hard), [(a.date.date(), a.type, b.type) for a, b in hard]
        assert all("Optional" in a.note for a, _ in hard if a.type == "recovery")
    d = pd.Timestamp("2026-11-02")                         # the fallback week: endurance before a test -> a spin
    sch = Scheduler({}, build_season())
    week = [WL.rest(d), WL.easy(d + pd.Timedelta(days=1), "endurance", 90), WL.ftp_test(d + pd.Timedelta(days=2)),
            WL.easy(d + pd.Timedelta(days=3), "long", 180), WL.intervals(d + pd.Timedelta(days=4), "vo2max", 1)]
    week[3].locked = True                                  # a day you set yourself stays
    sch.easy_before_hard(week)
    assert week[1].type == "recovery" and "test" in week[1].note and week[3].type == "long"


def test_ladders_push_but_dont_crush():
    """Each level asks more than the one before (more time on target, shorter rests or a little more power)
    and never less; reps stop at a format's proven size; level 1 is still a real session for its zone."""
    # min on target at level 1
    floor = {"threshold": 18, "vo2max": 12, "sweetspot": 18, "tempo": 18, "anaerobic": 4, "durability": 5}
    cap = {"threshold": 50, "vo2max": 48, "sweetspot": 60, "tempo": 60, "anaerobic": 12, "durability": 10}   # ... at the top
    for kind, formats in LADDERS.items():
        for fmt, (on_name, _, (off_lo, _), _, ladder) in formats.items():
            under = off_lo >= .76                       # over-unders / bursts: the part between is work too
            on = [sets * reps * (m + off * under) for sets, reps, m, off, *_ in ladder]
            hard = [(w, reps * (m + off) if sets > 1 or under else m, -off, -rest, lo)   # ... or longer reps / sets
                    for w, (sets, reps, m, off, lo, _, rest) in zip(on, ladder)]
            assert on[0] >= floor[kind] and max(on) <= cap[kind], (fmt, on)
            assert all(b >= a for a, b in zip(on, on[1:])), (fmt, on)                   # never less work
            assert all(b > a for a, b in zip(hard, hard[1:])), (fmt, hard)               # each level a step up


def test_day_limits_grow_with_your_own_longest_days():
    """without availability a day is at most the fixed default (2 h, the long day 4 h) - raised to your own longest
    recent day (all of it on the long day, OTHER_DAY_SHARE on others), at most RIDE_LIMIT_STEPS growth steps past it
    however far ahead; availability wins. A pro's 20-h weeks fit; a few-hours-a-day rider's limits don't change."""
    from goodpush.planning.load import VolumeRamp
    from goodpush.planning.optimizer import DAY_MINUTES
    from goodpush.planning.scheduler import OTHER_DAY_SHARE, RIDE_LIMIT_STEPS
    today = pd.Timestamp("2026-11-04")
    idx = pd.date_range("2026-09-01", "2026-11-03", freq="D")
    sch = Scheduler({}, build_season())
    sch.away = {}
    for longest, other_w, long_w in ((2.0, DAY_MINUTES["other"], DAY_MINUTES["long"]),      # the defaults as they were
                                     (5.5, OTHER_DAY_SHARE * 5.5 * 60 * 1.1, 5.5 * 60 * 1.1)):   # a pro's
        d = pd.DataFrame({"hours": 2.0, "ride_hours": 2.0}, index=idx)
        d.loc[idx[-3], ["hours", "ride_hours"]] = longest
        sch.ramp = VolumeRamp(d, today)
        ramp = 60 * longest * 1.1                                              # the ramp's own day cap (day_max)
        assert sch.ride_limit(today, DAY_MINUTES["other"], False) == pytest.approx(min(max(other_w, 120), ramp))
        assert sch.ride_limit(today, DAY_MINUTES["long"], True) == pytest.approx(min(max(long_w, 240), ramp))
    far = today + pd.Timedelta(weeks=30)                                       # ride_cap: 1.1 ** 31 by then
    assert sch.ride_limit(far, 240, True) == pytest.approx(5.5 * 60 * 1.1 ** RIDE_LIMIT_STEPS)
    sch.avail = [2.0] * 7                                                      # your time wins
    assert sch.ride_limit(today, 240, True) == 120
    sch.avail = []
    sch.ramp = None                                                            # no history: the defaults
    assert sch.ride_limit(today, 240, True) == 240


def test_physio_load_counts_z2_around_a_pros_intervals():
    """Scheduler.easy_max: an interval day of a loading week may carry Z2 up to the day's limit; the easy days follow
    your own longest days; a recovery week's interval day carries none."""
    from goodpush.planning.load import VolumeRamp
    from goodpush.planning.workouts import WorkoutLibrary as WL
    today = pd.Timestamp("2026-11-04")
    idx = pd.date_range("2026-09-01", "2026-11-03", freq="D")
    d = pd.DataFrame({"hours": 3.0, "ride_hours": 3.0}, index=idx)
    d.loc[idx[-3], ["hours", "ride_hours"]] = 5.5
    sch = Scheduler({}, build_season())
    sch.away, sch.ramp = {}, VolumeRamp(d, today)
    vo2 = WL.intervals(today, "vo2max", 3)
    days = [vo2, WL.easy(today, "endurance", 90), WL.easy(today, "long", 240), WL.easy(today, "recovery", 45)]
    load, rest = sch.easy_max(days, True), sch.easy_max(days, False)
    assert load[0] == pytest.approx(sch.ride_limit(today, 120, False) - vo2.minutes) and load[0] > 120
    assert load[1] > 150 and load[2] > 240 and load[3] == 90
    assert rest[0] == 0.0


def test_volume_ramp_from_the_recent_average():
    from goodpush.planning.load import VolumeRamp
    today = pd.Timestamp("2026-11-04")                                       # a Wednesday
    idx = pd.date_range("2026-09-01", "2026-11-03", freq="D")
    d = pd.DataFrame({"hours": 0.0, "ride_hours": 0.0}, index=idx)
    sundays = idx[idx.day_of_week == 6]
    d.loc[sundays, ["hours", "ride_hours"]] = 3.0                            # 3 h a week, one ride
    d.loc[sundays[-2], ["hours", "ride_hours"]] = 6.0                        # one big week: an average, not the max
    r = VolumeRamp(d, today)
    assert r.week_h == pytest.approx((3 * 3 + 6) / 4) and r.ride_h == 6.0
    assert r.week_cap(today) == pytest.approx(3.75 * 1.1)                    # this week: +10 %
    assert r.week_cap(today + pd.Timedelta(weeks=3)) == pytest.approx(3.75 * 1.1 ** 4)   # compounding, week by week
    assert r.ride_cap(today) == pytest.approx(6 * 60 * 1.1)
    assert VolumeRamp(d.iloc[:0], today).week_cap(today) == np.inf           # no history: no limit
    days = pd.date_range(today, periods=28, freq="D")
    table = pd.DataFrame({"type": "base", "recovery": days >= pd.Timestamp("2026-11-16")}, index=days)
    table.loc[days >= pd.Timestamp("2026-11-23"), "recovery"] = False        # week 3 a recovery week
    r = VolumeRamp(d, today, table)
    caps = [r.week_cap(today + pd.Timedelta(weeks=k)) / r.week_h for k in range(4)]
    assert caps == pytest.approx([1.1, 1.21, 1.331, 1.331])                  # after the recovery week: no extra step
    steady = [r.week_steady(today + pd.Timedelta(weeks=k)) / r.week_h for k in range(4)]
    assert steady == pytest.approx([1.0, 1.1, 1.21, 1.21])                   # one step below: your average this week
    assert VolumeRamp(d, today, start_hours=7.5).week_cap(today) == pytest.approx(7.5 * 1.1)   # what you can ride
    assert VolumeRamp(d, today, start_hours=7.5).ride_h == 6.0               # the day still from your rides
    assert VolumeRamp(d, today, start_hours=2.0).week_h == pytest.approx(3.75)   # ridden more: your average wins
    assert VolumeRamp(d.iloc[:0], today, start_hours=7.5).week_h == 7.5
    d[:] = 0.0
    assert VolumeRamp(d, today).week_h == VolumeRamp.MIN_WEEK_H              # after a break: a floor


def test_plan_weeks_ramp_up_from_what_you_rode(st):
    thu = pd.Timestamp("2026-10-22")
    daily = daily_until(thu - pd.Timedelta(days=1))
    daily["hours"] *= 0.5                                                     # 4.5 h a week, longest day 2 h
    daily["ride_hours"] = daily["hours"]
    out = run_plan(st, thu, days=56, cfg={"hours_per_week": 12.0, "availability": AVAIL}, daily=daily)
    caps = [w["hours_cap"] for w in out["weeks"]]
    ramped = [w["hours_cap"] for w in out["weeks"] if not w["push"]]
    assert caps[0] == pytest.approx(4.5 * 1.1, abs=0.05) and max(ramped) < 12   # not your 12 h at once
    push = [w for w in out["weeks"] if w["push"]]
    assert push and max(caps) < 12                                             # a push week stays in the ramp too
    load = [(i, w) for i, w in enumerate(out["weeks"]) if not w["recovery"] and not w["push"]]
    assert any(w["recovery"] for w in out["weeks"])
    steps = [1 + sum(w["push"] for w in out["weeks"][i + 1:j]) for (i, _), (j, _) in zip(load, load[1:])]
    assert all(b["hours_cap"] <= 1.1 ** k * a["hours_cap"] + 0.1    # also over a recovery week (0.1 h: rounded);
               for ((_, a), (_, b)), k in zip(zip(load, load[1:]), steps))       # a push week still counts as a step
    load = [w for _, w in load]
    assert any(w["hours_limited"] for w in load) and not any(w["hours_limited"] for w in out["weeks"] if w["recovery"])
    assert all(s.minutes <= 2 * 60 * 1.1 ** 9 + 1 for s in out["sessions"])   # one ride: the same rule
    by_week = pd.Series([s.minutes / 60 for s in out["sessions"]],
                        pd.DatetimeIndex([s.date for s in out["sessions"]])).resample("W-SUN").sum()
    assert (by_week.to_numpy()[:len(caps)] <= np.array(caps) + 0.3).all()
    out = run_plan(st, thu, days=14, cfg={"hours_per_week": 12.0, "availability": AVAIL, "plan_starting_hours": 7.5},
                   daily=daily)
    assert out["weeks"][0]["hours_cap"] == pytest.approx(7.5 * 1.1, abs=0.1)    # what you can already ride (0.1 h: rounded)


def test_plan_is_remade_after_a_code_change(monkeypatch):
    from goodpush.planning import Planner
    now = Planner.code_on_disk()
    monkeypatch.setattr(Planner, "CODE", now)                                 # this app runs the code on disk
    assert not Planner.code_stale()
    assert Planner.code_changed({"code": "older"}) and Planner.code_changed({})   # made by other code: re-make
    assert not Planner.code_changed({"code": now})                            # made by this code
    monkeypatch.setattr(Planner, "CODE", "older")                             # an app started before an update ...
    assert Planner.code_stale() and not Planner.code_changed({"code": "older"}) and not Planner.code_changed({})
    #                                         ... leaves the plan alone (the restarted app re-makes it) and says so


def test_code_on_disk_is_its_content_not_its_file_times(tmp_path):
    """2026-11-18 (the rider): the plan was re-made at every app start - files touched (a checkout, the line-ending
    hook) counted as new planning code. Only a change to the content does."""
    import os
    from goodpush import code_on_disk
    f = tmp_path / "a.py"
    f.write_bytes(b"x = 1\ny = 2\n")
    was = code_on_disk(tmp_path)
    os.utime(f, (1e9, 1e9))                                                   # touched
    f.write_bytes(b"x = 1\r\ny = 2\r\n")                                      # line endings rewritten
    assert code_on_disk(tmp_path) == was
    f.write_bytes(b"x = 1\ny = 3\n")
    assert code_on_disk(tmp_path) != was


def test_never_three_days_without_a_ride(st):
    from goodpush.planning.optimizer import SessionOptimizer
    thu = pd.Timestamp("2026-10-22")
    days = [WL.rest(thu + pd.Timedelta(days=k)) for k in range(5)]
    acts = pd.DataFrame({"date": [thu - pd.Timedelta(days=2)], "sport": ["ride"]})    # last ride: Tuesday
    assert SessionOptimizer.gap_windows(days, acts, thu) == [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]   # Wed off: Thu or Fri
    season = build_season(events=[{"date": "2026-11-15", "priority": "A", "kind": "road", "name": "Final"}])
    out = run_plan(st, thu, days=28, season=season, cfg={"availability": AVAIL})
    ride = pd.Series({s.date: s.minutes > 0 and s.type != "rest" for s in out["sessions"]}).groupby(level=0).max()
    off = (~ride).astype(int).groupby(ride.cumsum()).cumsum()
    assert off.max() <= 2                                                    # incl. the taper before the A race


def test_durability_finish_on_the_long_ride():
    from goodpush.planning.workouts import DURABILITY_AFTER_MIN
    d = pd.Timestamp("2026-11-08")
    long = WL.easy(d, "long", 240)
    w = WL.durability(long, 0)
    assert w.type == "long" and w.level == 0 and w.total_minutes() == 240 and "durability 2x2:30" in w.title
    assert [s.name for s in w.expanded()][-2:] == ["Durability", "Cool down"]         # at the very end
    top = WL.durability(WL.easy(d, "long", 110), 2)            # too short for the top
    work = [s for s in top.expanded() if s.name == "Durability"]
    assert top.level == 1 and len(work) == 3 and top.steps[1].minutes + 10 >= DURABILITY_AFTER_MIN   # 3 x 2.5 fits
    assert WL.durability(WL.easy(d, "long", 60), 0) is not None
    hard = [s for s in WL.durability(long, 0, polarized=True).expanded() if s.name == "Durability"]
    assert all(s.lo >= 1.05 for s in hard) and all(s.lo == .95 for s in work[:1] + [x for x in w.expanded() if x.name == "Durability"])   # polarized: above LT2


def test_plan_puts_the_finish_on_loading_weeks_long_rides_and_moves_its_ladder(st):
    thu = pd.Timestamp("2026-10-22")
    st["levels"]["durability"] = 1
    out = run_plan(st, thu, days=21, cfg={"availability": AVAIL})
    longs = [s for s in out["sessions"] if s.type == "long"]
    fin = [s for s in longs if s.level is not None]
    assert fin and "durability 3x2:30" in fin[0].title                               # your level ...
    assert [s.level for s in fin] == sorted(s.level for s in fin)                      # ... then a step a week
    day = str(fin[0].date.date())
    p = st["planned"][day]
    assert PlanState.ladder_of(p) == "durability" and p["level"] == 1
    p["compliance"] = {"ride": "1", "result": "hit"}                                    # the finish done ...
    st.update_progress(NO_RIDES, fin[0].date + pd.Timedelta(days=1))
    assert st["levels"]["durability"] == 2 and p["status"] == "done"                  # ... one level up


def test_durability_finish_scored_on_an_outdoor_ride():
    ftp = 250.0
    w = WL.durability(WL.easy(pd.Timestamp("2026-11-08"), "long", 240), 0)              # 2 x 2.5 min near the end
    p = np.full(4 * 3600, 150.0) + np.random.default_rng(0).normal(0, 15, 4 * 3600)     # outdoors: noisy Z2
    for start in (12000, 12600):                                                        # not where the plan has them
        p[start:start + 150] = 0.975 * ftp
    sc = SessionScorer.score(w, p, IntensityMap.ftp_knots(ftp))
    assert sc["ratio"] >= 0.9 and [r["result"] for r in sc["reps"]] == ["hit", "hit"]


def test_readiness_flags_signs_that_agree_not_one_alone():
    """one mild sign (low HRV) is a note; low HRV with a low resting HR is amber; 4 points red; every sign listed"""
    today = pd.Timestamp("2026-10-26")
    d = daily_until(today)
    d["sleep_score_z"], d["body_battery_max_z"], d["stress_z"] = 0.0, 0.0, 0.0
    r = readiness(d, today)
    assert r["status"] == "green" and {"HRV", "Resting HR", "Sleep", "Sleep score", "Load"} <= {g["name"] for g in r["signs"]}
    d.loc[today, "hrv_z"] = -1.4
    r = readiness(d, today)
    assert r["status"] == "green" and any("one sign on its own" in x for x in r["notes"])
    d.loc[today, "rhr"] = 46.0                                              # ~4 bpm below usual, with the low HRV
    r = readiness(d, today)
    assert r["status"] == "amber" and any("low together with a low HRV" in x for x in r["reasons"])
    d.loc[today, ["hrv_z", "body_battery_max_z", "stress_z"]] = [-2.5, -2.0, 2.0]   # 2 + 1 + 1 + 1
    assert readiness(d, today)["status"] == "red"


def test_a_run_of_bad_mornings_keeps_tomorrow_easy_too(st):
    from goodpush.planning.readiness import readiness, sustained
    mon = pd.Timestamp("2026-10-26")
    daily = daily_until(mon)
    one = daily.copy()
    one["sleep_score_z"] = 0.0
    one.iloc[-1:, one.columns.get_loc("hrv_z")] = -1.5                      # one low-HRV morning, a poor sleep score
    one.iloc[-1:, one.columns.get_loc("sleep_score_z")] = -2.0
    r1 = sustained(one, mon, readiness(one, mon))
    assert r1["status"] == "amber" and r1["easy_days"] == 1
    daily["sleep_score_z"] = 0.0
    daily.iloc[-5:, daily.columns.get_loc("hrv_z")] = -1.5                  # five in a row: a decline
    daily.iloc[-5:, daily.columns.get_loc("sleep_score_z")] = -2.0
    r = sustained(daily, mon, readiness(daily, mon))
    assert r["easy_days"] == 2 and r["streak"] >= 4 and "easy today and tomorrow" in r["reasons"][-1]
    assert [m["date"] for m in r["mornings"]][-1] == "2026-10-26" and len(r["mornings"]) == 7
    assert sum(m["status"] != "green" for m in r["mornings"]) == r["streak"] and "mornings" in r1
    out = Scheduler({}, build_season()).plan(st, daily, NO_RIDES, 200, mon, r, pd.date_range(mon, periods=7))
    s = out["sessions"]
    assert all(w.type not in ("threshold", "vo2max", "anaerobic", "test", "ramp") for w in s[:2])   # today, tomorrow
    assert any(w.type in ("threshold", "vo2max", "anaerobic", "sweetspot", "test", "ramp") for w in s[2:])   # later


def test_ramp_range_from_your_history_else_the_usual():
    from goodpush.planning.load import RampRange
    row = lambda band, weeks, ill, gain: {"ctl_ramp_per_week": band, "weeks": weeks, "illness_next_6wk": ill,
                                          "power_change_pct": gain}
    table = [row("falling (<0)", 115, .23, -4.2), row("0-1.5", 141, .18, .5), row("1.5-3", 76, .22, 8.2),
             row("3-4.5", 12, .42, 2.7)]
    r = RampRange({"ramp_table": table, "illness_base_rate": .21, "illness_episodes": 9})
    assert (r.lo, r.hi, r.lo_learned, r.hi_learned) == (1.5, 3.0, True, True)   # 3-4.5: too few weeks to go on
    table[3]["weeks"] = 30
    assert RampRange({"ramp_table": table, "illness_base_rate": .21, "illness_episodes": 9}).hi == 3.0   # ill more often
    table[3]["illness_next_6wk"] = .25
    assert RampRange({"ramp_table": table, "illness_base_rate": .21, "illness_episodes": 9}).hi == 4.5
    r = RampRange({"ramp_table": table, "illness_base_rate": .21, "illness_episodes": 1})      # no illnesses noted
    assert (r.lo, r.hi, r.hi_learned) == (1.5, RampRange.USUAL[1], False)
    assert (RampRange(None).lo, RampRange(None).hi) == RampRange.USUAL                         # a new user: textbook
    days = pd.date_range("2026-10-05", "2026-11-01")
    daily = pd.DataFrame({"ctl": np.linspace(40, 44, len(days))}, index=days)                  # +1 a week so far
    weeks = [{"week": "2026-11-02", "ctl_end": 62.0, "type": "base"},
             {"week": "2026-11-09", "ctl_end": 60.0, "type": "base", "recovery": True},
             {"week": "2026-11-16", "ctl_end": 46.0, "type": "build"},                          # +0.75 a week: slow
             {"week": "2026-11-23", "ctl_end": 46.0, "type": "specialty"}]                      # holding: fine
    RampRange(None).flag(weeks, daily)
    assert [w["ramp_flag"] for w in weeks] == ["fast", None, "slow", None]
    assert weeks[0]["ramp_4wk"] == pytest.approx((62 - 41.0) / 4, abs=0.1)


def test_days_away_get_no_session_and_the_plan_works_around_them(st):
    thu = pd.Timestamp("2026-10-22")
    plain = {str(s.date.date()): s for s in run_plan(st, thu, days=21, cfg={"availability": AVAIL})["sessions"]}
    hard = [d for d, s in plain.items() if s.type in KEY_TYPES and d > "2026-10-25"][0]
    st.set_away(hard, str((pd.Timestamp(hard) + pd.Timedelta(days=1)).date()), "work trip")
    st.replan(thu)
    out = run_plan(st, thu, days=21, cfg={"availability": AVAIL})
    by = {str(s.date.date()): s for s in out["sessions"]}
    assert by[hard].type == "rest" and by[hard].title == "Away" and by[hard].note == "work trip"
    assert by[str((pd.Timestamp(hard) + pd.Timedelta(days=1)).date())].minutes == 0
    wk = pd.Timestamp(hard) - pd.Timedelta(days=pd.Timestamp(hard).weekday())
    assert sum(s.type in KEY_TYPES for s in out["sessions"] if wk <= s.date < wk + pd.Timedelta(days=7)) >= 1   # moved, not lost
    st.set_away(hard, hard, away=False)
    assert hard not in st["away"] and len(st["away"]) == 1
    with pytest.raises(ValueError):
        st.set_away("2026-11-01", "2026-10-01")


def test_a_session_can_open_with_a_timed_max_effort():
    """2026-11-04: measure 5-min / 1-min power instead of inferring it - warm-up, openers, the effort all-out,
    a recovery, then the session's own set."""
    d = pd.Timestamp("2026-11-12")
    vo2 = WL.intervals(d, "vo2max", 3)          # (an early level's few reps can't all give way: never below 2)
    w = WL.max_effort(vo2, 300)
    names = [s.name for s in w.expanded()]
    assert w.type == "vo2max" and w.level == vo2.level and w.title.startswith("All-out 5 min + ")
    assert names[:2] == ["Warm up", "Opener"] and names.count("All-out 5 min") == 1 and "Primer" in names
    effort = next(s for s in w.expanded() if s.name == "All-out 5 min")
    assert effort.minutes == 5 and effort.mode == "resistance"                       # ride it by feel
    i = names.index("All-out 5 min")
    after = names[i + 2:]
    assert names[i + 1] == "Recover" and len(after) > 4 and [s.name for s in vo2.expanded()][-len(after):] == after
    assert w.total_minutes() <= vo2.total_minutes() + 5 and "heart rate" in w.note       # in the first rep's place
    one = WL.max_effort(WL.intervals(d, "anaerobic", 0), 60)
    assert any(s.name == "All-out 1 min" and s.minutes == 1 for s in one.expanded())


def test_plan_measures_at_the_end_of_build_and_in_specialty(st):
    from goodpush.planning.workouts import MAX_EFFORT_EVERY_WEEKS
    s = Season([Phase("build", "2026-11-02", 4, build_weeks=4), Phase("specialty", "2026-11-30", 8)])
    out = run_plan(st, pd.Timestamp("2026-11-02"), days=12 * 7, season=s, cfg={"availability": AVAIL})
    maxed = [w for w in out["sessions"] if w.title.startswith("All-out ")]
    assert maxed, "no timed max effort in build's last week or specialty"
    for kind in ("All-out 5 min", "All-out 1 min"):
        days = sorted(w.date for w in maxed if w.title.startswith(kind))
        assert all((b - a).days >= 7 * MAX_EFFORT_EVERY_WEEKS for a, b in zip(days, days[1:]))   # spread out
    assert all(w.date >= pd.Timestamp("2026-11-23") for w in maxed)                 # build: only its push week
    # a 5-min effort your heart rate confirmed last week: no 5-min max for the next few weeks
    acts = pd.DataFrame({"date": [pd.Timestamp("2026-10-28")], "sport": ["ride"], "workout": ["vo2max"],
                         "max_hr": [190.0], "effort_hr_300": [189.0], "effort_coast_300": [0.0]})
    st2 = PlanState.load(st.path.with_name("other.json"))
    out2 = run_plan(st2, pd.Timestamp("2026-11-02"), acts=acts, days=12 * 7, season=s, cfg={"availability": AVAIL})
    first = [w.date for w in out2["sessions"] if w.title.startswith("All-out 5 min")]
    assert all((d - pd.Timestamp("2026-10-28")).days >= 7 * MAX_EFFORT_EVERY_WEEKS for d in first)


def test_test_efforts_rotate_lengths_through_loading_weeks(st):
    """2026-11-22: [plan] test_efforts - key sessions open with 1, 3 or 5 min all-out now and then,
    so the power curve is measured at 1, 3 and 5 min (no sprint: Pmax stays inferred); at most one a week, each length
    not sooner than its TEST_EFFORT_WEEKS, before the session types it suits. Rebuilt from the plan's record, a
    session keeps its opening (the analysis finds the step in it)."""
    from goodpush.planning.workouts import MAX_EFFORTS, TEST_EFFORT_BEFORE, TEST_EFFORT_WEEKS, TestEfforts
    s = Season([Phase("build", "2026-11-02", 4, build_weeks=4), Phase("specialty", "2026-11-30", 8)])
    out = run_plan(st, pd.Timestamp("2026-11-02"), days=12 * 7, season=s,
                   cfg={"availability": AVAIL, "test_efforts": True})
    of_name = {v[0]: k for k, v in MAX_EFFORTS.items()}
    opened = [(w, of_name[w.title.split(" + ")[0]]) for w in out["sessions"] if w.title.startswith("All-out ")]
    assert {sec for _, sec in opened} >= {60, 180} and len(opened) >= 4
    weeks = [w.date.to_period("W") for w, _ in opened]
    assert len(set(weeks)) == len(weeks)                                       # one a week at most
    for sec in TEST_EFFORT_WEEKS:
        days = sorted(w.date for w, s_ in opened if s_ == sec)
        assert all((b - a).days >= 7 * TEST_EFFORT_WEEKS[sec] for a, b in zip(days, days[1:]))
    assert all(w.type in TEST_EFFORT_BEFORE[sec] for w, sec in opened)
    w, sec = opened[0]
    back = WL.rebuild(w.date, {"type": w.type, "level": w.level, "title": w.title, "minutes": w.minutes})
    assert any(x.name == MAX_EFFORTS[sec][0] for x in back.expanded())
    r = TestEfforts({300: pd.Timestamp("2026-11-02")})                        # 5 min last week: 3 min first
    assert r.pick("vo2max", "2026-11-09") == 180 and r.pick("sweetspot", "2026-11-09") is None
    r.placed(180, "2026-11-09")
    assert r.pick("vo2max", "2026-11-16") == 60 and r.pick("threshold", "2026-11-16") is None


def test_merged_short_reps_are_split_and_a_pause_keeps_the_reps_in_line():
    """2026-11-06: a 30/15 session's first two reps read as one 75-s effort (the first recovery too high); every later
    rep matched one off and the last two came out 'missed'. With a pause mid-set the reps still line up."""
    from goodpush.planning import SessionScorer
    from goodpush.planning.scoring import SPLIT_LONGER
    p = np.full(600, 150.0)
    starts = [100, 145, 190, 235, 400, 445]                                   # 4 reps, a 120-s pause, 2 more
    for s in starts:
        p[s:s + 30] = 310.0
    p[130:145] = 230.0                                                        # the first recovery drops too little
    ef = [{"start": 100, "seconds": 75, "watts": 300.0}, {"start": 190, "seconds": 30, "watts": 310.0}]
    out = SessionScorer.split_merged(p, ef, 0.85 * 296, 30)
    got = np.array([(x["start"], x["seconds"]) for x in out])
    assert got.shape == (3, 2) and np.allclose(got, [(100, 30), (145, 30), (190, 30)], atol=1)   # 3-s mean: +- 1 s
    short = [{"start": 190, "seconds": int(SPLIT_LONGER * 30), "watts": 300.0}]
    assert SessionScorer.split_merged(p, short, 250, 30) == short            # not clearly longer: as it is


def test_readiness_takes_the_training_before_out_of_the_morning_markers():
    """2026-11-10 (a loading block is meant to load you): HRV and resting HR follow the day before's training;
    low after a big day is the expected response, the same reading after a rest day is news."""
    from goodpush.planning.readiness import LoadResponse
    rng = np.random.default_rng(3)

    def responding(day):                              # HRV z falls 2 per 100 TSS ridden the day before
        d = daily_until(day)
        d["hrv_z"] = -0.02 * (d["tss"].shift(1).fillna(40.0) - 50.0) + rng.normal(0, 0.2, len(d))
        return d
    sunday = pd.Timestamp("2026-10-25")               # the morning after the long Saturday
    d = responding(sunday)
    assert d["hrv_z"].iloc[-1] < -1.5 and LoadResponse(d).slope(d["hrv_z"]) < -0.01
    r = readiness(d, sunday)
    hrv = next(s for s in r["signs"] if s["name"] == "HRV")
    assert r["status"] == "green" and hrv["points"] == 0 and "the training before" in hrv["text"]
    flat = d.assign(tss=40.0)                         # no training to explain it: the same morning is flagged
    assert readiness(flat, sunday)["status"] == "amber"
    tuesday = pd.Timestamp("2026-10-27")              # after the rest day HRV should be up: as low as that Sunday is news
    d2 = responding(tuesday)
    d2.iloc[-1, d2.columns.get_loc("hrv_z")] = d["hrv_z"].iloc[-1]
    r2 = readiness(d2, tuesday)
    assert r2["status"] == "amber" and "measured" in r2["reasons"][0]
    few = d.iloc[-60:]                                # under 90 mornings: nothing is learned, nothing taken out
    assert LoadResponse(few).slope(few["hrv_z"]) == 0.0 and LoadResponse(d).slope(pd.Series(rng.normal(0, 1, len(d)), d.index)) == 0.0


def test_not_recovering_is_judged_on_the_week_and_a_recovery_week_goes_on():
    """2026-11-10: one off morning isn't the signal - a marker that stays off through the easy days is; in a recovery
    week (the test of a block) the recovery then goes on before the next block's hard sessions."""
    from goodpush.planning.readiness import NOT_RECOVERING_DAYS, RECOVER_ON_DAYS, RecoveryTrend, recovering, sustained
    today = pd.Timestamp("2026-10-26")
    d = daily_until(today)
    d["rhr_z"] = 0.0
    green = readiness(d, today)
    assert RecoveryTrend(d, today).off() == [] and recovering(d, today, green)["status"] == "green"
    one = d.copy()
    one.loc[today, "hrv_z"] = -1.4                                  # one low morning: a note, no trend
    assert recovering(one, today, readiness(one, today)).get("trend") == []
    week = d.copy()
    week.loc[today - pd.Timedelta(days=6):, "hrv_z"] = -1.4          # a week of them, today too
    r = recovering(week, today, readiness(week, today))
    assert r["status"] == "amber" and r["easy_days"] == NOT_RECOVERING_DAYS and "not recovering: HRV" in r["reasons"][-1]
    back = week.copy()
    back.loc[today, "hrv_z"] = 0.2                                  # back this morning: the first good morning ends it
    assert recovering(back, today, readiness(back, today)).get("trend") == []
    r = sustained(week, today, readiness(week, today), recovery_week=True)
    assert r["easy_days"] == RECOVER_ON_DAYS and "the recovery week goes on" in r["reasons"][-1] and len(r["mornings"]) == 7
    up = d.copy()                                                   # HRV up, resting HR down, loaded: a note only
    up.loc[today - pd.Timedelta(days=6):, ["hrv_z", "rhr_z"]] = [1.3, -1.3]
    up["tsb"] = -15.0
    r = recovering(up, today, readiness(up, today))
    assert r["status"] == "green" and any("deep fatigue" in n for n in r["notes"])


def test_own_sessions_above_the_endurance_band_get_a_respec_proposal(st):
    from goodpush.planning.steps import Step
    from goodpush.planning.workouts import Z2, WorkoutLibrary as WL
    mon = pd.Timestamp("2026-10-19")
    day, old = "2026-10-21", (0.60, 0.72)
    mine = WL.easy(pd.Timestamp(day), "endurance", 90, old).to_dict()            # saved under an earlier, harder band
    st.place(day, mine, mon)
    st.add({"type": "tempo", "title": "Mine", "steps": [Step("Tempo", 20, .80, .85).to_dict(),
                                                        Step("Z2", 30, .50, .60).to_dict()]}, day, mon)   # easier: left alone
    out = run_plan(st, mon)
    ch = [c for c in out["proposal"] if c.get("own")]
    assert [c["date"] for c in ch] == [day] and "endurance band" in ch[0]["why"]
    assert next(s for s in out["sessions"] if str(s.date.date()) == day and not s.key).z2_band() == old   # kept until accepted
    st.decide("reject", [day])
    assert not [c for c in run_plan(st, mon)["proposal"] if c.get("own")]        # rejected: not asked again
    st["rejected"] = {}
    run_plan(st, mon)
    st.decide("accept", [day])
    out = run_plan(st, mon)
    w = next(s for s in out["sessions"] if str(s.date.date()) == day and not s.key)
    assert w.z2_band() == Z2 and w.minutes == 90 and day in st["manual_plan"] and not [c for c in out["proposal"] if c.get("own")]


class PickOne:
    """a stand-in for PhysioLoad.choose_sessions: always the same session type where it's offered (else the slot's
    first), for the slots from `start` on (the model's forecast starts after its last fitted day); remembers the
    slots it saw."""
    draws = None

    def __init__(self, kind: str, start: pd.Timestamp | None = None):
        self.kind, self.slots, self.start = kind, None, start

    def choose_sessions(self, workouts, extras, slots, events):
        self.slots = slots
        return ({i: self.kind if self.kind in {c.type for c in slots[i]} else slots[i][0].type
                 for i in slots if self.start is None or workouts[i].date >= self.start},
                SimpleNamespace(start=1.0, objective=0.5))


def test_sweet_spot_is_a_hard_session(st):
    from goodpush.planning.optimizer import HARDNESS
    assert HARDNESS["sweetspot"] == HARDNESS["threshold"] == HARDNESS["tempo"] > HARDNESS["lsct"]  # tempo: a hard day
    today = pd.Timestamp("2026-11-02")
    for kind in ("base", "build"):                        # base: sweet spot; build: threshold + sweet spot
        out = run_plan(PlanState({}), today, days=42, season=build_season(kind=kind),
                       cfg={"key_days": ["tue", "wed"]})  # side by side: only the hard-day rules keep them apart
        hard = pd.Series([HARDNESS.get(w.type, 0) == 2 for w in out["sessions"] if not w.key],
                         pd.DatetimeIndex([w.date for w in out["sessions"] if not w.key])).groupby(level=0).any()
        assert any(w.type == "sweetspot" for w in out["sessions"]), kind
        assert not (hard & hard.shift(1, fill_value=False)).any(), kind       # never two hard days in a row


def test_model_mix_puts_the_models_choice_on_the_key_days(st):
    from goodpush.planning.scheduler import MIX_TYPES
    today = pd.Timestamp("2026-11-02")
    pick = PickOne("vo2max")
    sch = Scheduler({"intensity": "model"}, build_season(), physio=pick)
    daily = daily_until(today - pd.Timedelta(days=1))
    out = sch.plan(st, daily, NO_RIDES, 200, today, readiness(daily, today), pd.date_range(today, periods=28))
    quality = [w.type for w in out["sessions"] if w.type in MIX_TYPES]
    assert quality and set(quality) == {"vo2max"}           # no pyramidal sweet spot / threshold / extra tempo
    mix = out["session_mix"]
    assert mix["sessions"] == {"vo2max": len(pick.slots)} and mix["gain"] == 0.5
    for cands in pick.slots.values():                       # every type, the template's first, fitting the day
        assert {c.type for c in cands} <= set(MIX_TYPES) | {"endurance"} and len(cands) >= 3
    row = SimpleNamespace(type="build", recovery=False, block=0, week=0)
    week = [k for d, k in sorted(sch.chosen.items()) if pd.Timestamp(d) < today + pd.Timedelta(days=7)]
    pyr = Scheduler({"intensity": "pyramidal"}, build_season())
    for hard in (0, 1, 2):                                  # how many: the same rule whatever picks the types
        n = len(pyr.week_mix(row, today, hard))
        assert sch.week_mix(row, today, hard) == week[:n] and n == max(0, 2 - hard)


def test_model_mix_holds_no_type_and_may_choose_an_easy_ride(st):
    """2026-11-17: the model chooses what each key day needs - no tempo held for a base week - and an endurance
    ride as long as the template's session is always one of its options; a base loading week has one hard session
    (BASE_HARD), its other key days are easy rides (2026-11-18: one quality day in base, and tempo counts as hard)."""
    from goodpush.planning.optimizer import HARDNESS
    from goodpush.planning.scheduler import BASE_HARD, MIX_TYPES
    today = pd.Timestamp("2026-11-03")                     # a Tuesday: a key day the model can't choose for
    daily = daily_until(today - pd.Timedelta(days=1))
    st["last_test"] = st["last_ramp"] = "2026-10-30"     # no test due today
    span = pd.date_range(today, periods=28)
    start = today + pd.Timedelta(days=1)
    pick = PickOne("vo2max", start=start)
    sch = Scheduler({"intensity": "model"}, build_season(kind="base"), physio=pick)
    out = sch.plan(PlanState(json.loads(json.dumps(st))), daily, NO_RIDES, 200, today, readiness(daily, today), span)
    assert str(today.date()) not in sch.chosen and "vo2max" in sch.chosen.values()
    assert all(c[-1].type == "endurance" and c[-1].minutes >= 30 for c in pick.slots.values())   # an easy option
    s = pd.Series([w.type for w in out["sessions"] if not w.key], pd.DatetimeIndex([w.date for w in out["sessions"]
                                                                                    if not w.key]))
    for monday, types in s.groupby(s.index - pd.to_timedelta(s.index.weekday, unit="D")):
        recovery = bool(sch.season.day_table(pd.DatetimeIndex([monday]))["recovery"].iloc[0])
        hard = [t for t in types if HARDNESS.get(t, 0) == 2]
        assert recovery or len(hard) <= BASE_HARD, (monday, list(types))       # VO2max wanted everywhere: one
    later = [c for i, c in pick.slots.items() if not any(HARDNESS[x.type] == 2 for x in c)]
    assert not later              # base has one quality day: the other key days are the template's easy rides
    easy = PickOne("endurance", start=start)
    sch = Scheduler({"intensity": "model"}, build_season(kind="base"), physio=easy)
    out = sch.plan(st, daily, NO_RIDES, 200, today, readiness(daily, today), span)
    assert set(sch.chosen.values()) == {None}
    assert not [w for w in out["sessions"] if w.type in MIX_TYPES and w.date >= start], \
        [(w.date, w.type) for w in out["sessions"] if w.type in MIX_TYPES]


def test_a_session_in_a_format_you_made_harder_keeps_its_day(st):
    from goodpush.planning.scheduler import MIX_TYPES
    today = pd.Timestamp("2026-11-02")
    daily = daily_until(today - pd.Timedelta(days=1))
    span = pd.date_range(today, periods=28)
    Scheduler({"intensity": "model"}, build_season(), physio=PickOne("threshold")).plan(
        st, daily, NO_RIDES, 200, today, readiness(daily, today), span)
    thr = {k: WL.format_of(pd.Timestamp(k), "threshold") for k, p in st["planned"].items()
           if p["type"] == "threshold" and "#" not in k}
    assert thr
    for f in set(thr.values()):
        st.set_offset("threshold", f, 2)
    pick = PickOne("vo2max")                                # a model that would rather have VO2max everywhere
    sch = Scheduler({"intensity": "model"}, build_season(), physio=pick)
    out = sch.plan(st, daily, NO_RIDES, 200, today, readiness(daily, today), span)
    on = {str(w.date.date()): w for w in out["sessions"] if not w.key}
    for d in thr:                                           # each stays a threshold session, harder, on its day
        assert on[d].type == "threshold" and on[d].offset > 0, (d, on[d].title)
        assert not [c for c in out["proposal"] if c.get("date") == d]    # your choice: in straight away
    assert not set(thr) & set(sch.chosen) and set(sch.chosen.values()) <= set(MIX_TYPES) | {None}


def test_model_mix_falls_back_to_pyramidal(st, tmp_path):
    today = pd.Timestamp("2026-11-02")
    daily = daily_until(today - pd.Timedelta(days=1))
    span = pd.date_range(today, periods=28)
    pyr = run_plan(PlanState.load(tmp_path / "a.json"), today, days=28, cfg={"intensity": "pyramidal"})
    for physio in (None, PickOne("vo2max")):                             # no model / not asked for: the template's mix
        cfg = {"intensity": "model" if physio is None else "pyramidal", "hours_per_week": 8.0}   # as run_plan
        sch = Scheduler(cfg, build_season(), physio=physio)
        out = sch.plan(PlanState.load(tmp_path / f"{cfg['intensity']}.json"), daily, NO_RIDES, 200, today,
                       readiness(daily, today), span)
        assert sch.chosen is None and [w.type for w in out["sessions"]] == [w.type for w in pyr["sessions"]]
        assert physio is None or physio.slots is None


def test_a_race_week_has_at_most_one_hard_session_besides_the_race(tmp_path):
    from goodpush.planning.scheduler import MIX_TYPES, RACE_WEEK_QUALITY
    today = pd.Timestamp("2026-11-02")
    races = [{"date": d, "priority": "B", "kind": "road", "name": "Series"} for d in ("2026-11-08", "2026-11-22")]
    daily = daily_until(today - pd.Timedelta(days=1))
    for style, physio in (("pyramidal", None), ("polarized", None), ("model", PickOne("vo2max"))):
        sch = Scheduler({"intensity": style}, build_season(events=races), physio=physio)
        out = sch.plan(PlanState.load(tmp_path / f"{style}.json"), daily, NO_RIDES, 200, today,
                       readiness(daily, today), pd.date_range(today, periods=28))
        for monday in (pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-16")):
            week = [w for w in out["sessions"] if monday <= w.date < monday + pd.Timedelta(days=7) and not w.key]
            assert any(w.type == "race" for w in week)
            assert sum(w.type in MIX_TYPES for w in week) <= RACE_WEEK_QUALITY, (style, [w.type for w in week])
            row = SimpleNamespace(type="build", recovery=False, block=0, week=0)
            assert len(sch.week_mix(row, monday, 0)) <= RACE_WEEK_QUALITY      # the optimiser's count too


def test_recovery_weeks_explicit_and_the_same_as_the_cadence():
    """a phase without recovery_weeks (an old plans.json) gets them from its cadence; the day table then reads as the
    cadence did (3:1: week 3 recovery, the week before it the push week); moved, the blocks follow."""
    p = Phase("build", "2026-11-02", 9, build_weeks=3)
    assert p.recovery_weeks == [3, 7] and Phase("recovery", "2026-11-02", 3).recovery_weeks == []
    days = pd.date_range("2026-11-02", periods=63)
    t = Season([p]).day_table(days)
    wk = np.arange(63) // 7
    assert (t["recovery"].to_numpy() == (wk % 4 == 3)).all() and (t["block"].to_numpy() == wk // 4).all()
    assert (t["week_in_block"].to_numpy() == wk % 4).all() and (t["push"].to_numpy() == (wk % 4 == 2)).all()
    moved = Season([Phase("build", "2026-11-02", 9, build_weeks=3, recovery_weeks=[4, 8, 20])]).day_table(days)
    assert moved["recovery"].to_numpy()[[28, 56]].all() and moved["push"].to_numpy()[[21, 49]].all()
    assert moved["block_weeks"].to_numpy()[0] == 4 and moved["week_in_block"].to_numpy()[35] == 0   # after week 4
    flat = Season([Phase("specialty", "2026-11-02", 4, build_weeks=0)]).day_table(days[:28])
    assert not flat["recovery"].any() and (flat["block_weeks"] == 0).all()


def test_season_as_blocks_back_to_back(tmp_path):
    s = Season([Phase("base", "2026-11-02", 6, build_weeks=3, hours=9.0),
                Phase("build", "2026-12-14", 4, build_weeks=3)])
    base, build = s.phases
    s.set_blocks("2026-11-04", [{"id": build.id, "type": "build", "weeks": 8},                # reordered and grown
                                {"id": base.id, "type": "base", "weeks": 6, "recovery_weeks": [1]},
                                {"type": "recovery", "weeks": 1}, {"type": "specialty", "weeks": 3}])
    a, b, c, d = s.phases
    assert [p.start for p in s.phases] == ["2026-11-02", "2026-12-28", "2027-02-08", "2027-02-15"]   # from its Monday
    assert (a.id, a.recovery_weeks) == (build.id, [3, 7])                  # the cadence fills the weeks it gained
    assert (b.id, b.recovery_weeks, b.hours) == (base.id, [1], 9.0)       # moved recovery week, settings kept
    assert c.type == "recovery" and d.recovery_weeks == Phase("specialty", "2026-01-05", 3).recovery_weeks
    s.set_blocks("2026-11-02", [{"id": base.id, "type": "maintenance", "weeks": 2}])
    assert s.phases[0].name == "Maintenance" and s.phases[0].id == base.id
    s.save(tmp_path)
    assert Season.load(tmp_path).phases[0].recovery_weeks == s.phases[0].recovery_weeks
    with pytest.raises(ValueError):
        s.set_blocks("2026-11-02", [{"type": "build", "weeks": 0}])


def test_season_changed_days_are_the_weeks_an_edit_changed():
    old = Season([Phase("base", "2026-11-02", 4, build_weeks=0), Phase("build", "2026-11-30", 4, recovery_weeks=[3])])
    new = Season([Phase("base", "2026-11-02", 4, build_weeks=0), Phase("build", "2026-11-30", 5, recovery_weeks=[1])])
    days = new.changed_days(old, pd.Timestamp("2026-11-04"))
    weeks = sorted({str((pd.Timestamp(d) - pd.Timedelta(days=pd.Timestamp(d).weekday())).date()) for d in days})
    # recovery week 4 -> 2 (and the push week before it: 3 -> 1), one week added; the base weeks stay
    assert weeks == ["2026-11-30", "2026-12-07", "2026-12-14", "2026-12-21", "2026-12-28"]
    assert len(days) == 35 and new.changed_days(new, pd.Timestamp("2026-11-04")) == []
    st = PlanState({"manual_plan": {"2026-11-10": {}, "2026-12-08": {}, "2026-12-08#2": {}},
                    "planned": {"2026-12-08#2": {}}})
    own = st.own_on(days)
    assert own == ["2026-12-08", "2026-12-08#2"]
    st.forget_own(own)
    assert list(st["manual_plan"]) == ["2026-11-10"] and st["planned"] == {}


def test_phase_grows_by_its_cadence():
    p = Phase("base", "2026-11-02", 6, build_weeks=3, recovery_weeks=[2])
    assert p.grown(12) == [2, 6, 10] and p.grown(4) == [2]
    assert Phase("base", "2026-11-02", 2, build_weeks=3).grown(9) == [3, 7]


def test_autoplan_replaces_the_season_from_a_races():
    s = Season([Phase("base", "2026-11-02", 30)], [Event("2027-04-11", "A", "road", "Goal")])
    s.autoplan(pd.Timestamp("2026-11-04"))
    assert [p.type for p in s.phases][-2:] == ["specialty", "recovery"] and s.phases[0].start == "2026-11-09"
    ends = [pd.Timestamp(p.start) for p in s.phases[1:]]
    assert all(e == q.end + pd.Timedelta(days=1) for e, q in zip(ends, s.phases))      # back to back
    s = Season([Phase("base", "2026-11-02", 30)], [Event("2026-12-06", "B")])    # no A race: the general block
    s.autoplan(pd.Timestamp("2026-11-04"))
    assert [(p.type, p.start, p.weeks) for p in s.phases] == [("base", "2026-11-09", 8), ("build", "2027-01-04", 4)]


def test_general_block_without_races_or_phases():
    """Nothing to plan for (no A race ahead, no phase from next Monday): a general Base -> Build block."""
    sat = pd.Timestamp("2026-10-24")
    got = Season(events=[Event("2027-01-10", "B")]).suggest(sat)
    assert [(x["type"], x["start"], x["weeks"]) for x in got] == [("base", "2026-10-26", 8), ("build", "2026-12-21", 4)]
    assert all(x["generic"] and x["why"] for x in got)
    assert Season([Phase("build", "2026-08-03", 8)]).suggest(sat)[0]["type"] == "base"     # your season has ended
    s = Season()
    for x in got:                                     # accepted: nothing more to suggest; 3:1 blocks
        s.add_phase(type=x["type"], start=x["start"], weeks=x["weeks"], build_weeks=x["build_weeks"])
    assert s.suggest(sat) == [] and [p.recovery_weeks for p in s.phases] == [[3, 7], [3]]
    new = Season()                                    # a new rider's first plan (no rides): the same block
    assert new.ensure_phase(pd.Timestamp("2026-10-19"), rides=False)
    assert [(p.type, p.start, p.weeks) for p in new.phases] == [("base", "2026-10-19", 8), ("build", "2026-12-14", 4)]


def test_plan_for_a_new_rider_without_rides(tmp_path):
    """No rides, no analysis, no season: `build` plans from nothing - zero load, the fallback FTP, the general
    block - and the plan has sessions on every week of it."""
    from goodpush.config import init_project, load_project
    proj = load_project(init_project(tmp_path / "p").root)
    assert proj.cfg["plan"]["hours_per_week"] == 4.0 == DEFAULT_PLAN["hours_per_week"]   # a block's hours unless set
    out = Planner(proj).build(today=pd.Timestamp("2026-10-22"), write=False, progress=lambda *a: None)
    assert out["ftp_source"] == "fallback" and out["ctl_now"] == 0
    assert [p["type"] for p in out["season"]["phases"]] == ["base", "build"]
    weeks = out["weeks"]
    assert len(weeks) == 12 and all(w["hours_planned"] > 0 for w in weeks)
    assert sum(w["recovery"] for w in weeks) == 3
    # little riding: easy days are endurance rides; a spin only after a hard day or before one at threshold
    from goodpush.planning.optimizer import HARDNESS
    from goodpush.planning.workouts import AT_THRESHOLD_UP, SPIN_FROM_HOURS
    rate = WL.easy(pd.Timestamp("2026-10-22"), "endurance", 60).tss()
    low = {pd.Timestamp(w["week"]) for w in weeks if w["tss_target"] < SPIN_FROM_HOURS * rate}
    by_day = {w.date: w.type for w in out["sessions"] if not w.key}
    spins = [d for d, t in by_day.items() if t == "recovery" and d - pd.Timedelta(days=d.weekday()) in low]
    assert low and sum(t == "endurance" for t in by_day.values()) > len(spins)
    assert all(HARDNESS.get(by_day.get(d - pd.Timedelta(days=1)), 0) == 2
               or by_day.get(d + pd.Timedelta(days=1)) in AT_THRESHOLD_UP for d in spins), spins
    p = Planner(proj)                                 # "I can already ride 6 h a week": load starts there
    p.cfg["plan_starting_hours"] = 6.0
    more = p.build(today=pd.Timestamp("2026-10-22"), write=False, progress=lambda *a: None)
    assert more["ctl_now"] == pytest.approx(6 * 45 / 7, abs=0.5)
    hours = lambda o: sum(w["hours_planned"] for w in o["weeks"])  # noqa: E731
    assert hours(more) > 1.3 * hours(out)


# --- easier / harder (workouts.Adjuster) ---------------------------------------------------------------
# every ladder session at offset 0 as built before the Adjuster came (sha1 of title, minutes, level, note and steps
# over levels -1 .. top + 1, ten dates, with and without variety): offset 0 must keep them byte for byte. A
# deliberate change to a ladder updates its digest here.
LADDER_DIGESTS = {                              # sha1 prefixes, not secrets
    "threshold/Threshold": "a1c74a9089a9f882",  # pragma: allowlist secret
    "threshold/Over-unders": "30c2071330c2e4e0",  # pragma: allowlist secret
    "vo2max/VO2max": "a8b3b952e07d8380",  # pragma: allowlist secret
    "vo2max/6-min": "ac32e23aa8964689",  # pragma: allowlist secret
    "vo2max/Over-unders 1:1": "4a1621a43524fe28",  # pragma: allowlist secret
    "vo2max/Hard-start VO2max": "0e8ed51b31f3c54e",  # pragma: allowlist secret
    "vo2max/Step-down VO2max": "037fea5e1c7db7e9",  # pragma: allowlist secret
    "vo2max/30/15s": "658d321979b36065",  # pragma: allowlist secret
    "vo2max/40/20s": "ebaf87aa09608850",  # pragma: allowlist secret
    "sweetspot/Sweet spot": "d888fa857c32ed2b",  # pragma: allowlist secret
    "sweetspot/Sweet spot bursts": "ae50f75deb818938",  # pragma: allowlist secret
    "tempo/Tempo": "34024d840e4b8c33",  # pragma: allowlist secret
    "tempo/Tempo bursts": "8395627b49cae5f2",  # pragma: allowlist secret
    "tempo/Tempo-plus": "f14f638f12bf6b70",  # pragma: allowlist secret
    "anaerobic/1-min efforts": "253416342f310981",  # pragma: allowlist secret
    "anaerobic/30-s efforts": "857cdc74351291c9",  # pragma: allowlist secret
    "durability/Durability": "77f3e6552acadd01",  # pragma: allowlist secret
}
ADJ_DAYS = pd.date_range("2026-01-05", periods=10, freq="5D")


def ladder_digest(kind: str, fmt: str, offsets) -> str:
    import hashlib
    rows = []
    for lv in range(-1, len(LADDERS[kind][fmt][4]) + 1):
        for d in ADJ_DAYS:
            for var in (True, False):
                w = WL.intervals(d, kind, lv, fmt, variety=var, offsets=offsets)
                d_ = {k: v for k, v in w.to_dict().items() if k != "format"}     # pinned before sessions kept it
                rows.append([w.title, w.minutes, w.level, w.note, d_])
    return hashlib.sha1(json.dumps(rows, default=str).encode()).hexdigest()[:16]


def work_load(w) -> float:
    """minutes x target² of the work steps: what easier takes away and harder adds."""
    from goodpush.planning.steps import WORK
    return sum(s.minutes * s.mean_sq() for s in w.expanded() if s.kind == "active" and s.lo >= WORK)


def test_offset_zero_keeps_every_ladder_session():
    zero = {k: {f: 0 for f in v} for k, v in LADDERS.items()}
    for k, v in LADDERS.items():
        for f in v:
            others = {k: {g: -2 for g in v if g != f}}           # another format's offset leaves this one alone
            for offsets in (None, zero, others):
                assert ladder_digest(k, f, offsets) == LADDER_DIGESTS[f"{k}/{f}"], (k, f, offsets)


def test_adjuster_walks_one_knob_step_at_a_time_within_its_limits():
    from goodpush.planning.steps import WORK, Repeat
    from goodpush.planning.workouts import Adjuster
    for k, v in LADDERS.items():
        for f, (*_, ladder) in v.items():
            for lv in range(len(ladder)):
                for d in ADJ_DAYS[:3]:
                    base = WL.intervals(d, k, lv, f)
                    adj = Adjuster.of(k, f, lv)
                    e, h = adj.reach(base)
                    assert e > 0 and h >= 0, (k, f, lv)
                    top = {s.hi for s in adj.apply(base, h).expanded() if s.kind == "active" and s.lo >= WORK}
                    assert top <= {s.hi for s in base.expanded()}, (k, f, lv)      # harder never adds power
                    assert adj.moves(base, -e - 3) == adj.moves(base, -e)
                    assert adj.moves(base, h + 3) == adj.moves(base, h)
                    prev, prev_w = adj.moves(base, -e), adj.apply(base, -e)
                    for n in range(-e + 1, h + 1):
                        mv, w = adj.moves(base, n), adj.apply(base, n)
                        step = {x: mv.get(x, 0) - prev.get(x, 0) for x in set(mv) | set(prev)}
                        step = {x: v for x, v in step.items() if v}
                        if set(step) <= {"reps", "sets"} and n > 0:    # harder: a rep, or a set VOLUME_BACK lower
                            assert step == {"reps": 1} or step == {"sets": 1, "reps": -Adjuster.VOLUME_BACK}, step
                        elif set(step) <= {"reps", "sets"}:        # easier: fewer reps in all each step back
                            v = adj.volume(base)
                            reps_in_all = [(v["sets"] + x.get("sets", 0)) * (v["reps"] + x.get("reps", 0))
                                           for x in (prev, mv)]
                            assert reps_in_all[0] < reps_in_all[1], (k, f, lv, n, step)
                        else:                                      # any other knob: one step
                            assert len(step) == 1 and sum(abs(v) for v in step.values()) == 1, (k, f, lv, n, step)
                        assert w.steps != prev_w.steps, (k, f, lv, n)
                        assert work_load(w) >= work_load(prev_w) - 1e-9, (k, f, lv, n)     # every step: more work
                        assert w.level == base.level
                        prev, prev_w = mv, w
                    for n in (-e, -1, 1, h):                       # the planner's path is the same walk
                        got = WL.intervals(d, k, lv, f, offsets={k: {f: n}})
                        assert got.steps == adj.apply(base, n).steps and got.minutes == got.total_minutes()
                    if f == "Over-unders":                         # never more than 5 overs a set
                        top = WL.intervals(d, k, lv, f, offsets={k: {f: h}})
                        sets = [x for x in top.steps if isinstance(x, Repeat)]
                        assert max(r.reps for x in sets for r in x.steps if isinstance(r, Repeat)) <= 5
                    for n in (-e, h):
                        w = adj.apply(base, n)
                        on = {s.minutes for s in w.expanded() if s.name in ("Hard", "On")}
                        if f == "6-min":
                            assert on == {6}                       # 6-min reps stay 6 min
                        if f in ("30/15s", "40/20s"):              # micro intervals keep their on; their off is
                            off = {s.minutes for s in w.expanded() if s.name == "Off"}       # never shorter, 3x easier
                            assert on == {ladder[0][2]} and off == {ladder[0][3] * (3 if n < 0 else 1)}
                            title = WL.intervals(d, k, lv, f, offsets={k: {f: n}}).title
                            assert title.startswith(f"{round(ladder[0][2] * 60)}/{round(next(iter(off)) * 60)} ")


def shaped_rep(w) -> list:
    """a rep of the main set without its rest (a step-down or hard-start rep)."""
    from goodpush.planning.workouts import Adjuster
    m = Adjuster.parts(w.steps)[2]
    return w.steps[m].steps[:-1]


def test_adjuster_scales_a_shaped_rep_as_a_whole():
    from goodpush.planning.workouts import Adjuster
    d = ADJ_DAYS[0]
    for f in ("Step-down VO2max", "Hard-start VO2max"):
        base = WL.intervals(d, "vo2max", 2, f)
        e, _ = Adjuster.of("vo2max", f, 2).reach(base)
        for n in (-e + 3, -6, 4):
            a, b = shaped_rep(base), shaped_rep(WL.intervals(d, "vo2max", 2, f, offsets={"vo2max": {f: n}}))
            ratio = [s.lo / t.lo for s, t in zip(b, a)]
            assert np.allclose(ratio, ratio[0]) and [s.name for s in a] == [s.name for s in b], (f, n)
            assert [s.cue for s in a] == [s.cue for s in b]
        low = WL.intervals(d, "vo2max", 2, f, offsets={"vo2max": {f: -e}})
        assert max(s.hi for s in low.expanded()) < max(s.hi for s in base.expanded())     # power went last


def test_easier_harder_is_a_position_per_format_and_the_level_keeps_its_meaning():
    from goodpush.planning.workouts import Adjuster
    d = pd.Timestamp("2026-11-24")
    key = str(d.date())
    st = PlanState({"levels": {k: 1 for k in LADDERS},
                    "planned": {key: {"type": "vo2max", "level": 1, "title": "x"},
                                "2026-11-25": {"type": "endurance", "level": None}}})
    a = st.adjustment(key)
    fmt, base = a["format"], WL.intervals(d, "vo2max", 1)
    assert fmt == WL.format_of(d, "vo2max") and a["offset"] == 0 and a["easier"] and a["harder"]
    for _ in range(3):
        st.adjust(key, -1)
    assert st.offsets() == {"vo2max": {fmt: -3}}
    w = WL.intervals(d, "vo2max", 1, offsets=st.offsets())
    assert w.steps == Adjuster.of("vo2max", fmt, 1).apply(base, -3).steps and w.offset == -3  # 3 clicks = 3 at once
    st.record([w])                                           # the plan remembers it: a rebuild from the record shows it
    assert st["planned"][key]["offset"] == -3 and Workout.from_dict(key, st.current(key)).steps == w.steps
    assert Workout.from_dict(key, w.to_dict()).offset == -3 and "offset" not in base.to_dict()
    for _ in range(3):
        st.adjust(key, 1)
    assert st.offsets() == {} and WL.intervals(d, "vo2max", 1, offsets=st.offsets()).to_dict() == base.to_dict()
    st.adjust(key, 1)
    st.adjust(key, -1)                                                                       # harder then easier
    assert st.offsets() == {} and st["levels"]["vo2max"] == 1
    e, h = Adjuster.of("vo2max", fmt, 1).reach(base)
    st.set_offset("vo2max", fmt, -e)
    assert not st.adjustment(key)["easier"]
    with pytest.raises(ValueError, match="as easy as"):
        st.adjust(key, -1)
    st.set_offset("vo2max", fmt, -e - 40)                    # set beyond this session's reach: counts from its limit
    assert st.adjust(key, 1)["offset"] == -e + 1
    st.set_offset("vo2max", fmt, h)
    with pytest.raises(ValueError, match="as hard as"):
        st.adjust(key, 1)
    assert st.adjustment("2026-11-25") is None
    st["manual_plan"] = {key: {**st.current(key), "level": None}}    # your own session: the builder's, not the format's
    assert st.adjustment(key) is None
    with pytest.raises(ValueError):
        st.adjust(key, -1)
    with pytest.raises(ValueError):
        st.set_offset("vo2max", "Sprints", 1)


def test_a_session_you_moved_keeps_its_format_and_easier_harder(st):
    """2026-11-18 (the rider): a session moved to another day lost easier / harder. It keeps its format (not the new
    day's turn), the buttons, and a click rebuilds your copy in place - its length, day and why stay."""
    today = pd.Timestamp("2026-10-20")
    st["last_test"] = st["last_ramp"] = "2026-10-15"
    out = run_plan(st, today, daily=daily_until(today))
    src = next(s for s in out["sessions"] if s.date > today + pd.Timedelta(days=1) and s.type in LADDERS)
    assert src.format and st.current(str(src.date.date()))["format"] == src.format
    key = st.move(str(src.date.date()), str((today + pd.Timedelta(days=1)).date()), today)
    a = st.adjustment(key)
    assert a and a["format"] == src.format and a["offset"] == 0
    before, offsets = dict(st["manual_plan"][key]), st.offsets()
    _, _, easier, harder, _ = st.reach(key)
    d = 1 if harder >= easier else -1
    n = min(2, max(easier, harder)) * d                                 # two clicks (or as far as it goes)
    seen, pa = st.preview(key, n)                                       # the session window: shown, not saved
    assert pa["offset"] == n and st.offsets() == offsets and st["manual_plan"][key] == before
    assert abs(seen.minutes - before["minutes"]) < 5 and seen.offset == n
    assert st.preview(key, 50 * d)[1]["offset"] == (harder if d > 0 else -easier)   # no further than it goes
    assert st.preview(key, 0)[1] == a
    after = st.adjust(key, d)
    mine = st["manual_plan"][key]
    assert after["offset"] == mine["offset"] == st.offsets()[src.type][src.format] != 0
    assert mine["steps"] != before["steps"] and abs(mine["minutes"] - before["minutes"]) < 5
    assert mine["why_change"] == before["why_change"] and st["accepted"][key]["steps"] == mine["steps"]
    st.adjust(key, -d)
    assert st.adjust(key, n)["offset"] == n                             # closing the window: its clicks at once
    assert st["manual_plan"][key]["steps"] == seen.to_dict()["steps"]   # what the window showed
    old = {"type": src.type, "level": src.level, "title": f"{src.format} 3x5"}       # saved before formats were
    assert WL.format_in(today, old) == src.format


def test_offsets_load_from_an_old_state_and_merge_between_computers(tmp_path):
    from goodpush.sync.peer import Merge
    p = tmp_path / "plan_state.json"
    PlanState({"planned": {"2026-11-05": {"type": "vo2max", "level": 1}}}, p).save()
    old = p.read_bytes()
    st = PlanState.load(p)
    assert st.offsets() == {}
    st.save()
    assert p.read_bytes() == old                              # nothing new written until you choose
    left, right = PlanState.load(p), PlanState.load(p)
    left.set_offset("vo2max", "VO2max", -2)
    right.set_offset("threshold", "Threshold", 1)
    L, R = (json.dumps(x, indent=1).encode() for x in (left, right))
    rel = "data/processed/plan_state.json"
    m = json.loads(Merge(True).json(old, L, R, rel))
    assert m["format_offsets"] == {"vo2max": {"VO2max": -2}, "threshold": {"Threshold": 1}}
    assert json.loads(Merge(False).json(old, L, old, rel))["format_offsets"] == {"vo2max": {"VO2max": -2}}
    assert json.loads(Merge(True).json(None, old, L, rel))["format_offsets"] == {"vo2max": {"VO2max": -2}}
    back = PlanState(json.loads(L))
    back.set_offset("vo2max", "VO2max", 0)                    # back to as planned on one computer
    assert json.loads(Merge(True).json(L, json.dumps(back).encode(), L, rel))["format_offsets"] == {}


def test_trimmed_goes_through_the_adjuster():
    from goodpush.planning.steps import Repeat
    from goodpush.planning.variety import IntervalVariety as IV
    from goodpush.planning.workouts import Adjuster
    d = next(x for x in pd.date_range("2026-01-05", periods=30)
             if any(isinstance(s, Repeat) for s in IV.warm_up(x, "vo2max")))       # spin-ups in the warm-up
    w = WL.intervals(d, "vo2max", 3, "6-min")                                     # 5 x 6 min
    t = w.trimmed("two short nights")
    warm = IV.warm_up(d, "vo2max")
    assert t.steps[:len(warm)] == warm                                            # the warm-up keeps its spin-ups
    main = next(x for x in t.steps[len(warm):] if isinstance(x, Repeat))
    hard = [s for s in t.expanded() if s.name == "Hard"]
    assert main.reps == 4 and {(s.lo, s.hi) for s in hard} == {(1.02 * .98, 1.05 * .98)}
    assert t.title == w.title + " (trimmed)" and t.adjusted == "two short nights" and t.minutes == t.total_minutes()
    assert t.steps == Adjuster().move(w, Adjuster.TRIM).steps and t.level == w.level
    two = WL.intervals(d, "threshold", 2, "Threshold", variety=False).trimmed("x")   # 2 x 20: keeps both reps
    assert sum(s.name == "Threshold" for s in two.expanded()) == 2
    micro = WL.intervals(d, "vo2max", 4, "30/15s").trimmed("x")                    # 3 x 13: one rep less a set
    sets = next(x for x in micro.steps if isinstance(x, Repeat) and isinstance(x.steps[0], Repeat))
    assert (sets.reps, sets.steps[0].reps, sets.steps[0].steps[1].minutes) == (3, 12, .25)


def main_repeat(w):
    from goodpush.planning.steps import Repeat
    return next(s for s in w.steps if isinstance(s, Repeat))


def test_adjuster_on_your_own_workouts():
    from goodpush.planning.builder import WorkoutText
    from goodpush.planning.workouts import Adjuster
    d, adj = pd.Timestamp("2026-11-24"), Adjuster()
    w = Workout(d, "own", steps=WorkoutText().parse("Warm up 10m 50%\n4x\n  Hard 4m 110%\n  Recover 3m 50%\n"
                                                     "Cool down 10m 50%"))
    assert adj.reach(w) == (2 + 4 + 5 + 10, 1)                # reps 4 -> 2, rests to 2x, reps to half, power -10 %;
                                                              # harder: rests to 0.75x (no sets, never power)
    easier, harder = main_repeat(adj.apply(w, -1)), main_repeat(adj.apply(w, 1))
    assert easier.reps == 3 and easier.steps[1].minutes == 3 and harder.reps == 4       # fewer reps first
    assert main_repeat(adj.apply(w, -5)).reps == 2 and main_repeat(adj.apply(w, -5)).steps[1].minutes == 5.25
    assert harder.steps[1].minutes == 2.25
    assert adj.apply(w, 0) is w and adj.apply(w, -40).steps == adj.apply(w, -21).steps
    sets = Workout(d, "own", steps=WorkoutText().parse("Warm up 10m 50%\n2x\n  3x\n    On 30s 130%\n    Off 30s 50%\n"
                                                        "  Recover 5m 50%\nCool down 10m 50%"))
    walk = [adj.moves(sets, n) for n in (-1, -2, -3, -4, -7, -8)]     # 2x3 -> 2x2 -> 1x3 -> 1x2, then rests, on
    assert walk == [{"reps": -1}, {"sets": -1}, {"reps": -1, "sets": -1}, {"reps": -1, "sets": -1, "rest": 1},
                    {"reps": -1, "sets": -1, "rest": 4}, {"reps": -1, "sets": -1, "rest": 4, "on": -1}]
    assert [adj.moves(sets, n) for n in (1, 2, 3)] == [{"reps": -1, "sets": 1}, {"sets": 1},     # 3x2, 3x3, rests
                                                       {"sets": 1, "rest": -1}]
    one = main_repeat(adj.apply(sets, -7))
    assert (one.reps, one.steps[0].reps, one.steps[0].steps[1].minutes, one.steps[1].minutes) == (1, 2, 1.0, 10)
    easy = WL.easy(d, "endurance", 90)
    assert adj.reach(easy) == (0, 0) and adj.apply(easy, -3) is easy and adj.move(easy, Adjuster.TRIM) is easy


def volume_walk(sets: int, reps: int, d: int, few: int, most: int, top: int) -> list[tuple[int, int]]:
    """(sets, reps) along the volume ladder from sets x reps: a rep at a time between most - VOLUME_BACK and most,
    then a set more (d = 1) or less (d = -1) starting over; one set goes on down to `few`. Easier skips a step
    that isn't fewer reps in all than every one before it."""
    from goodpush.planning.workouts import Adjuster
    lo, out, least = most - Adjuster.VOLUME_BACK, [], sets * reps
    while True:
        reps += d
        if reps > most:
            sets, reps = sets + 1, lo
        elif reps < lo and sets > 1:
            sets, reps = sets - 1, most
        if sets > top or reps < few:
            return out
        if d > 0 or sets * reps < least:
            out.append((sets, reps))
            least = min(least, sets * reps)


def test_volume_ladder_a_rep_a_rep_then_a_set_and_back_at_the_same_power():
    from goodpush.planning.workouts import Adjuster
    d, f = ADJ_DAYS[0], "Over-unders"
    ladder = LADDERS["threshold"][f][4]
    for lv in range(len(ladder)):
        base, adj = WL.intervals(d, "threshold", lv, f), Adjuster.of("threshold", f, lv)
        e, h = adj.reach(base)
        few, most = adj.limits["reps"]
        for sign, n, skip in ((1, h, 0), (-1, e, 0)):          # both ways: reps and sets first
            want = volume_walk(ladder[lv][0], ladder[lv][1], sign, few, most, adj.limits["sets"][1])
            got = []
            for i in range(1, len(want) + 1):
                w = WL.intervals(d, "threshold", lv, f, offsets={"threshold": {f: sign * (skip + i)}})
                _, _, m, k = Adjuster.parts(w.steps)
                got.append((w.steps[m].reps, w.steps[m].steps[k].reps))
            assert got == want, (lv, sign)
            assert n > skip + len(want)                          # then the rests (harder), on / power (easier)
        assert all("power" not in adj.moves(base, i) for i in range(1, h + 1))
    base = WL.intervals(d, "threshold", 0, f)
    assert volume_walk(3, 3, 1, 2, 5, 4)[:4] == [(3, 4), (3, 5), (4, 4), (4, 5)]        # the rule, spelled out
    assert volume_walk(3, 3, -1, 2, 5, 4)[:4] == [(2, 4), (1, 5), (1, 4), (1, 3)]     # 2x5 is more than 3x3
    top = WL.intervals(d, "threshold", 0, f, offsets={"threshold": {f: 99}})
    assert {s.hi for s in top.expanded()} == {s.hi for s in base.expanded()}


def test_the_plan_uses_your_easier_harder_per_format():
    today = pd.Timestamp("2026-10-19")
    plain = run_plan(PlanState({}), today, days=21)
    offs = {k: {f: -2 for f in v} for k, v in LADDERS.items()}
    out = run_plan(PlanState({"format_offsets": offs}), today, days=21)
    iv = [w for w in out["sessions"] if w.type in LADDERS and not w.adjusted and w.level is not None]
    assert iv and out["levels"] == plain["levels"]
    for w in iv:                                             # (a max effort opens some: "All-out 5 min + ...")
        assert w.title.split(" + ")[-1] == WL.intervals(w.date, w.type, w.level, offsets=offs).title, w.title
    assert [(w.title, w.minutes) for w in iv] != [(w.title, w.minutes) for w in plain["sessions"]
                                                  if w.type in LADDERS and not w.adjusted]


# --- levels fade without the work (planning.decay) -------------------------------------------------------
def intervals_table(rows, today):
    """an IntervalLog table from (days before today, zone, seconds, steps, cv) rows."""
    return pd.DataFrame([{"date": today - pd.Timedelta(days=d), "zone": z, "seconds": s, "steps": n, "cv": cv}
                         for d, z, s, n, cv in rows])


def test_levels_fade_without_the_work():
    from goodpush.planning.decay import GRACE_DAYS, STEP_DAYS, LevelDecay
    assert [LevelDecay.steps(g) for g in (GRACE_DAYS - 1, GRACE_DAYS, GRACE_DAYS + STEP_DAYS - 1,
                                          GRACE_DAYS + STEP_DAYS)] == [0, 1, 1, 2]
    today = pd.Timestamp("2026-06-01")
    iv = intervals_table([(3, "threshold", 480, 8, .12),       # over-unders: threshold work, not steady
                          (3, "tempo", 900, 0, .03),           # (a ramp test's steps: not steady either)
                          (40, "sweetspot", 1200, 0, .03),     # 20 min steady sweet spot
                          (60, "threshold", 600, 0, .04),      # 10 min steady threshold
                          (30, "vo2max", 240, 0, .05)], today)
    iv = pd.concat([iv, intervals_table([(2, "threshold", 600, 4, .11)], today)], ignore_index=True)
    work, steady = LevelDecay.last_work(iv, today)
    assert work["sweetspot"] == work["tempo"] == today - pd.Timedelta(days=2)         # higher work keeps them
    assert steady == {"threshold": today - pd.Timedelta(days=60), "sweetspot": today - pd.Timedelta(days=40),
                      "tempo": today - pd.Timedelta(days=3)}
    st = PlanState({"levels": {**{k: 0 for k in LADDERS}, "threshold": 6, "sweetspot": 4, "vo2max": 2}})
    assert st.decay_levels(iv, today) == {"threshold": (0, 5), "sweetspot": (0, 3), "vo2max": (2, 0)}
    assert st["levels"]["threshold"] == 6 and st.lags() == {"threshold": 5, "sweetspot": 3}
    assert st.decay_levels(iv, today) == {}                                   # the same day again: nothing
    later = today + pd.Timedelta(days=STEP_DAYS)
    assert st.decay_levels(iv, later) == {"threshold": (0, 1), "sweetspot": (0, 1)}  # a steady step per STEP_DAYS
    assert st.lags() == {"threshold": 6, "sweetspot": 4}                     # down to the first level
    st.set_lag("threshold", 5)
    st.set_lag("sweetspot", 3)
    d = next(x for x in pd.date_range(today, periods=21) if WL.format_of(x, "threshold") == "Threshold")
    lags = st.lags()
    assert WL.intervals(d, "threshold", 6, lags=lags).level == 6 - lags["threshold"]
    assert WL.intervals(d, "threshold", 6, "Over-unders", lags=lags).level == 6        # surges keep the type's
    assert WL.intervals(d, "vo2max", 2, lags=lags).level == 2
    assert PlanState().decay_levels(None, today) == {}                         # no history: no clock


def test_a_steady_session_on_target_makes_up_its_lag():
    today = pd.Timestamp("2026-06-01")
    d = next(x for x in pd.date_range(today - pd.Timedelta(days=20), periods=21)
             if WL.format_of(x, "threshold") == "Threshold")
    st = PlanState({"levels": {**{k: 0 for k in LADDERS}, "threshold": 6}, "sustained_lag": {"threshold": 3},
                    "planned": {str(d.date()): {"type": "threshold", "level": 3,
                                                "compliance": {"result": "hit"}}}})
    st.update_progress(pd.DataFrame({"date": []}), today)
    p = st["planned"][str(d.date())]
    assert p["status"] == "done" and st["levels"]["threshold"] == 6 and st.lags() == {"threshold": 2}
    st.mark(str(d.date()), "")                                                 # undone: the lag is back
    assert st.lags() == {"threshold": 3} and st["levels"]["threshold"] == 6
    st.set_lag("threshold", 9)
    assert st.lags() == {"threshold": 6}                                       # never below the first level


def test_old_plan_states_move_to_the_new_ladders(tmp_path):
    p = tmp_path / "plan_state.json"
    p.write_text(json.dumps({"levels": {"threshold": 0, "vo2max": 1, "sweetspot": 3, "tempo": 1},
                             "planned": {"2026-05-01": {"type": "tempo", "level": 1},
                                         "2026-05-02": {"type": "vo2max", "level": 1},
                                         "2026-05-03": {"type": "long", "level": 1}},
                             "accepted": {"2026-06-02": {"type": "threshold", "level": 0}}}), encoding="utf-8")
    st = PlanState.load(p)
    assert st["ladders"] == PlanState.LADDERS_VERSION
    assert {k: st["levels"][k] for k in ("threshold", "vo2max", "sweetspot", "tempo")} == \
        {"threshold": 2, "vo2max": 1, "sweetspot": 5, "tempo": 3}                  # the same sessions as before
    assert [st["planned"][k]["level"] for k in sorted(st["planned"])] == [3, 1, 1]
    assert st["accepted"]["2026-06-02"]["level"] == 2
    st.save()
    assert PlanState.load(p)["levels"] == st["levels"]                         # once only
    assert PlanState({"levels": {"threshold": 1}})["levels"]["threshold"] == 1     # in code: today's levels


def test_replay_of_the_decay_rule():
    from goodpush.planning.decay import LevelDecay
    today = pd.Timestamp("2026-06-01")
    rows = [(d, "sweetspot", 1200, 0, .03) for d in (100, 93, 86, 79)] + [(49, "sweetspot", 600, 0, .03)]
    r = LevelDecay.replay(intervals_table(rows, today))
    assert len(r) == 1 and r.iloc[0][["type", "gap_days", "steps"]].tolist() == ["sweetspot", 30, 2]
    assert r.iloc[0]["minutes"] == 10 and r.iloc[0]["ratio"] == pytest.approx(0.5)


def test_recovery_weeks_have_an_lsct_and_no_tempo_day(st):
    """2026-11-18: no stand-alone tempo (a tempo day costs a hard morning, little more gain than easy riding); a recovery
    week's first key day is the LSCT, whatever picks the hard days."""
    today = pd.Timestamp("2026-11-02")
    for style in ("pyramidal", "polarized"):
        for kind in ("base", "build"):
            season = build_season(kind=kind)
            out = run_plan(PlanState({}), today, days=56, season=season, cfg={"intensity": style})
            dt = season.day_table(pd.date_range(today, periods=56))
            main = [w for w in out["sessions"] if not w.key]
            rec = {w.date for w in main if dt.loc[w.date, "recovery"]}
            assert rec, (style, kind)
            lsct = [w for w in main if w.type == "lsct"]
            assert lsct and all(w.date in rec for w in lsct), (style, kind)
            assert not any(w.type == "tempo" for w in main), (style, kind)


def test_intervals_icu_text_matches_the_session():
    import re

    from goodpush.planning.workouts import Repeat, Step
    from goodpush.report.plan_out import session_icu
    d = pd.Timestamp("2026-10-27")
    w = Workout(d, "vo2max", 0, "Sets", [
        Step("Warm up", 10, .50, .60, "warmup", lo_end=.70, hi_end=.80),
        Repeat(2, [Repeat(3, [Step("On", 1.5, 1.15, 1.20, cadence=100, cue="spin it up"),
                              Step("Off", 1, .50, .50, "rest")]), Step("Recover", 4, .50, .55, "rest")]),
        Step("Hold 80 % HRmax", 6, .70, .76, mode="resistance", hr=.80), Step("Cool down", 5, .55, .40, "cooldown")])
    w.minutes = w.total_minutes()
    text = session_icu(w)
    lines = text.splitlines()
    assert "- Warm up 10m ramp 55-75%" in lines and "- spin it up 1m30s 115-120% 100rpm" in lines
    assert "- Hold HRmax 6m 80% HR" in lines and "- Cool down 5m 40-55%" in lines            # numbers left out of the text
    sets = text.split("\n\n")
    rep = next(b for b in sets if b.startswith("2x"))
    assert rep.splitlines()[0] == "2x" and len(rep.splitlines()) == 1 + 3 * 2 + 1     # the inner repeat written out
    secs = 0
    for b in sets:                                                                     # durations add up to the session
        reps = int(b.split("x")[0]) if re.match(r"^\d+x$", b.splitlines()[0]) else 1
        for line in (x for x in b.splitlines() if x.startswith("- ")):
            m = re.search(r" (?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)? ", line)
            secs += reps * sum(int(v or 0) * k for v, k in zip(m.groups(), (3600, 60, 1)))
    assert secs == round(w.minutes * 60)
    for s in (WL.intervals(d, "threshold", 0), WL.ramp_test(d), WL.ftp_test(d), WL.easy(d, "endurance", 90)):
        assert all(re.match(r"^- [^\d%]* ?\d+[hms]", x) for x in session_icu(s).splitlines() if x.startswith("-"))
