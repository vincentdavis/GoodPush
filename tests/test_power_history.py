"""models.power_history: the Rider page's power over time (PowerHistory) and its power-curve heatmap (CurveHistory)."""
import numpy as np
import pytest


def power_history_case():
    """200 days on one curve (CP 250, MAP 300 = its 5-min power), a fit's effort-mixture parameters and three ride days:
    a test on the curve (day 50), an easy ride far below it (day 100), an easy and a harder ride on one day (day 150)."""
    import pandas as pd
    from goodpush.models.physio import PhysioInputs
    from goodpush.physio import PowerCurve
    days = pd.date_range("2025-01-01", periods=200)
    wp = float(PowerCurve.w_prime_for(300.0, 300.0, 250.0, 1000.0))
    states = pd.DataFrame({"cp": 250.0, "map": 300.0, "w_prime": wp, "pmax": 1000.0, "fade": 0.08, "fatigue": 0.0,
                           "form": 1.0}, index=days)
    curve = np.asarray(PowerCurve.power(np.asarray(PhysioInputs.EFFORT_S, float), 250.0, wp, 1000.0, 0.08))
    rows = [(50, "t1", "test", 1.0), (100, "e1", None, 0.5), (150, "e2", None, 0.5), (150, "h2", "threshold", 0.8)]
    acts = pd.DataFrame([{"date": days[d] + pd.Timedelta(hours=9), "id": i, "sport": "ride", "workout": w,
                          "moving_s": 7200.0, "duration_s": 7200.0, "power_coverage": 1.0,
                          **{f"mmp_{s}": f * c for s, c in zip(PhysioInputs.EFFORT_S, curve)}} for d, i, w, f in rows])
    names = ["logit_pi[0]", "logit_pi[1]", "logit_pi[2]", "logit_pi[3]", "log_shape_m2", "log_deficit[0]",
             "log_deficit[1]", "log_scale_kind[0]", "log_scale_kind[1]", "log_scale_kind[2]", "x0[0]"]
    fit = {"names": names, "z": np.array([-3.0, -2.0, 0.0, 1.0, 0.0, np.log(0.3), 0.0, 0.0, 0.0, 0.0, 5.0])}
    fc = pd.DataFrame({"cp": np.linspace(250.0, 260.0, 42), "map": np.linspace(300.0, 310.0, 42)},
                      index=pd.date_range(days[-1] + pd.Timedelta(days=1), periods=42))
    return states, acts, fit, fc


def test_power_history_predicts_the_fresh_curve_and_forecasts_on_from_it():
    import json
    from goodpush.models.power_history import PowerHistory
    from goodpush.physio import PowerCurve
    states, acts, fit, fc = power_history_case()
    out = PowerHistory(states, acts, fit, fc).build()
    n, s = len(states), list(PowerHistory.DURATIONS)
    assert len(out["date"]) == n + 42 and out["today"] == "2025-07-19" and out["s"] == s
    want = np.asarray(PowerCurve.power(np.asarray(s, float)[:, None], 250.0, states["w_prime"].iloc[0], 1000.0, 0.08))
    pred = np.array(out["predicted"], float)
    assert np.allclose(pred[:, :n], np.broadcast_to(want, (len(s), n)), atol=0.06) and np.isnan(pred[:, n:]).all()
    f = np.array(out["forecast"], float)
    assert np.isnan(f[:, :n - 1]).all() and np.array_equal(f[:, n - 1], pred[:, n - 1])
    assert np.allclose(f[:, n], pred[:, n - 1], rtol=1e-3)                     # no jump at today: same curve, same day
    assert np.all(f[:, -1] > f[:, n] + 5)                                       # the plan lifts it
    assert PowerHistory(states, acts).build()["forecast"] == []                 # no forecast: none
    json.dumps(out)                                                              # plain JSON for the app


def test_power_history_observes_each_days_best_its_ride_and_whether_it_was_all_out():
    from goodpush.models.power_history import PowerHistory
    from goodpush.physio import KINDS
    states, acts, fit, fc = power_history_case()
    out = PowerHistory(states, acts, fit, fc).build()
    o = out["observed"][out["s"].index(1200)]
    assert o["i"] == [50, 100, 150] and o["id"] == ["t1", "e1", "h2"]          # day 150: the harder ride's best
    pred = out["predicted"][out["s"].index(1200)]
    assert o["w"][2] == pytest.approx(0.8 * pred[150], abs=0.1) and o["kind"] == ["test", "easy", "hard"]
    assert o["all_out"][0] > 99 and o["all_out"][1] < 1                         # on the curve as a test / far below, easy
    assert out["observed"][out["s"].index(3600)]["kind"][0] == "hard"           # a test is 20 min or shorter
    assert out["all_out_sd"] > 0 and set(o["kind"]) <= set(KINDS)
    assert PowerHistory(states, acts, None, fc).build()["observed"][0]["all_out"] is None   # no fit: no odds
    assert max(o["bad"]) < 1                                                     # near or below the curve: good readings


def test_a_best_far_over_the_curve_is_a_bad_reading_not_all_out():
    """2026-11-26: a broken power meter's 1000 W for minutes reads as a bad reading (physio v17 BAD_READING_SHARE)."""
    import pandas as pd
    from goodpush.models.power_history import PowerHistory
    states, acts, fit, fc = power_history_case()
    glitch = acts.iloc[[1]].assign(id="g1", date=states.index[175] + pd.Timedelta(hours=9))
    glitch[[c for c in acts if c.startswith("mmp_")]] *= 2 * 3.0                 # 3 x the curve (the row was half of it)
    out = PowerHistory(states, pd.concat([acts, glitch], ignore_index=True), fit, fc).build()
    o = out["observed"][out["s"].index(300)]
    q = o["id"].index("g1")
    assert o["bad"][q] > 99 and o["all_out"][q] < 1


def test_curve_history_gives_model_and_measured_curves_weekly():
    import json
    import pandas as pd
    from goodpush.models.power_history import CurveHistory
    days = pd.date_range("2025-01-06", periods=120)
    states = pd.DataFrame({"cp": np.linspace(240, 260, 120), "w_prime": 18000.0, "pmax": 1000.0, "fade": 0.08}, index=days)
    acts = pd.DataFrame({"date": days[::3], "sport": "ride", "mmp_5": 900.0, "mmp_300": 290.0, "mmp_1200": 250.0})
    out = CurveHistory(states, acts).build()
    n, D = len(out["date"]), len(out["s"])
    assert n == 18 and all(len(r) == n for r in out["model"]) and len(out["model"]) == D == len(out["measured"])
    m = np.array(out["model"], float)
    assert np.all(np.diff(m[:, -1]) < 0) and m[out["s"].index(1200), -1] > m[out["s"].index(1200), 0]   # falls with
    i300 = out["s"].index(300)                                          # duration, rises with CP; measured = 42-d best
    assert out["measured"][i300][-1] == 290 and out["measured"][out["s"].index(3600)][-1] is None
    p, meas = np.array(out["percent"], float), np.array(out["measured"], float)    # measured as % of the model
    ok = np.isfinite(meas)
    assert np.allclose(p[ok], 100 * meas[ok] / m[ok], atol=0.6) and np.isnan(p[~ok]).all()
    assert out["percent"][out["s"].index(3600)][-1] is None                          # nothing that long: empty
    json.dumps(out)


def test_curve_history_centres_the_measured_window_on_the_week():
    """2026-11-12: one big effort shows in the columns within 21 days of its week's middle - before and after it, as
    many weeks each side - not for the six weeks after it (which trailed the model)."""
    import pandas as pd
    from goodpush.models.power_history import CurveHistory
    days = pd.date_range("2025-01-06", periods=140)                   # Mondays .. 20 weeks
    states = pd.DataFrame({"cp": 250.0, "w_prime": 18000.0, "pmax": 1000.0, "fade": 0.08}, index=days)
    effort = pd.Timestamp("2025-03-06")                               # a Thursday: its week's middle
    acts = pd.DataFrame({"date": [*days[::7], effort], "sport": "ride", "mmp_300": [*[200.0] * 20, 400.0]})
    out = CurveHistory(states, acts).build()
    row = np.array(out["measured"][out["s"].index(300)], float)
    hit = np.flatnonzero(row == 400)
    week = out["date"].index("2025-03-09")                             # the effort's column (its Sunday)
    assert week in hit and hit.min() == week - 3 and hit.max() == week + 3   # 3 weeks each side, centred


def test_app_physio_carries_the_20_min_power_the_rider_page_predicts(tmp_path):
    """server._physio: p20 per day, the same fresh 20-min power as PowerHistory's predicted (Today's Your numbers)."""
    from goodpush.app import server
    from goodpush.models.power_history import PowerHistory
    states, acts, fit, fc = power_history_case()
    states = states.assign(lt1=200.0, ftp=245.0)
    states["cp"] = np.linspace(240.0, 260.0, len(states))           # moving, so each day's p20 is its own
    states.to_parquet(tmp_path / "physio_states.parquet")
    out = server._physio(tmp_path)
    pred = PowerHistory(states, acts, fit, fc).build()
    j = pred["s"].index(1200)
    assert len(out["p20"]) == len(out["date"]) == len(states)
    assert np.allclose(out["p20"], np.array(pred["predicted"][j][:len(states)], float), atol=0.06)
    states.drop(columns=["pmax"]).to_parquet(tmp_path / "physio_states.parquet")
    assert "p20" not in server._physio(tmp_path)                    # an older states file: no 20 min, no error
