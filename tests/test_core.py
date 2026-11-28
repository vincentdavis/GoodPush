import io
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goodpush.io import fit
from goodpush.io.streams import read_gpx, read_tcx
from goodpush.metrics.load import pmc, power_tss
from goodpush.metrics.power import mean_max, normalized_power, summarize_stream, to_1hz
from goodpush.models.cp import fit_cp


def make_fit(power, hr, t0=1_000_000_000, compressed_every=0):
    """Tiny FIT encoder: one record definition (timestamp, power, heart_rate)."""
    recs = bytearray()
    # definition: local 0, global 20, fields 253 uint32, 7 uint16, 3 uint8
    recs += bytes([0x40, 0, 0]) + struct.pack("<H", 20) + bytes([3])
    recs += bytes([253, 4, 0x86, 7, 2, 0x84, 3, 1, 0x02])
    # definition: local 1 (for compressed headers) without timestamp
    recs += bytes([0x41, 0, 0]) + struct.pack("<H", 20) + bytes([2])
    recs += bytes([7, 2, 0x84, 3, 1, 0x02])
    for i, (p, h) in enumerate(zip(power, hr)):
        ts = t0 + i
        if compressed_every and i % compressed_every:
            recs += bytes([0x80 | (1 << 5) | (ts & 0x1F)]) + struct.pack("<HB", p, h)
        else:
            recs += bytes([0x00]) + struct.pack("<IHB", ts, p, h)
    header = struct.pack("<BBHI4sH", 14, 0x20, 2100, len(recs), b".FIT", 0)
    return header + bytes(recs) + b"\x00\x00"


def test_fit_roundtrip_with_compressed_timestamps():
    p = [100 + i % 50 for i in range(200)]
    h = [120 + i % 30 for i in range(200)]
    data = make_fit(p, h, compressed_every=4)
    a = fit.read_activity(data)
    r = a["records"]
    assert r["power"] == p
    assert r["heart_rate"] == h
    t = np.array(r["time"])
    assert np.all(np.diff(t) == 1)
    assert t[0] == 1_000_000_000 + fit.FIT_EPOCH_OFFSET


def test_invalid_values_become_none():
    data = make_fit([0xFFFF, 200], [0xFF, 140])
    r = fit.read_activity(data)["records"]
    assert r["power"][0] is None and r["heart_rate"][0] is None
    assert r["power"][1] == 200


def test_np_and_meanmax():
    x = np.full(3600, 200.0)
    assert normalized_power(x) == pytest.approx(200)
    x[:60] = 500
    mm = mean_max(x, [1, 60, 3600])
    assert mm[0] == 500 and mm[1] == 500 and mm[2] == pytest.approx(205)


def test_cp_fit_recovers_parameters():
    cp, wp = 250.0, 20000.0
    ds = [180, 300, 600, 1200]
    pw = [cp + wp / d for d in ds]
    c, w, r2 = fit_cp(ds, pw)
    assert c == pytest.approx(cp, rel=1e-6) and w == pytest.approx(wp, rel=1e-6)


def test_pmc_steady_state():
    tss = pd.Series(100.0, index=pd.date_range("2024-01-01", periods=400))
    df = pmc(tss)
    assert df["ctl"].iloc[-1] == pytest.approx(100, rel=0.01)
    assert df["tsb"].iloc[-1] == pytest.approx(0, abs=1)
    assert power_tss(3600, 250, 250) == pytest.approx(100)


def test_resample_gaps():
    t = list(range(0, 100)) + list(range(150, 200)) + list(range(1000, 1100))
    rec = {"time": t, "power": [200] * len(t), "heart_rate": [140] * len(t)}
    s = to_1hz(rec)
    # 50 s stop filled with zeros, 800 s stop cut out
    assert s["n"] == 200 + 100
    assert (s["power"][100:150] == 0).all()
    summ = summarize_stream(s)
    assert summ["has_power"]


TCX = b"""  <?xml version="1.0"?><TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
 xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2"><Activities><Activity Sport="Biking"><Lap><Track>
<Trackpoint><Time>2021-01-01T10:00:00Z</Time><HeartRateBpm><Value>120</Value></HeartRateBpm><Extensions><ns3:TPX><ns3:Watts>210</ns3:Watts></ns3:TPX></Extensions></Trackpoint>
<Trackpoint><Time>2021-01-01T10:00:01Z</Time><HeartRateBpm><Value>121</Value></HeartRateBpm><Extensions><ns3:TPX><ns3:Watts>220</ns3:Watts></ns3:TPX></Extensions></Trackpoint>
</Track></Lap></Activity></Activities></TrainingCenterDatabase>"""

GPX = b"""<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">
<trk><type>VirtualRide</type><trkseg>
<trkpt lat="1" lon="1"><ele>5</ele><time>2020-04-18T17:52:30+00:00</time><extensions><power>103</power><gpxtpx:TrackPointExtension><gpxtpx:hr>99</gpxtpx:hr><gpxtpx:cad>78</gpxtpx:cad></gpxtpx:TrackPointExtension></extensions></trkpt>
<trkpt lat="1" lon="1"><ele>5</ele><time>2020-04-18T17:52:31+00:00</time><extensions><power>70</power></extensions></trkpt>
</trkseg></trk></gpx>"""


def test_tcx_gpx():
    t = read_tcx(TCX)["records"]
    assert t["power"] == [210, 220] and t["heart_rate"] == [120, 121]
    g = read_gpx(GPX)["records"]
    assert g["power"] == [103, 70] and g["heart_rate"][0] == 99 and g["cadence"][0] == 78


def test_end_to_end(tmp_path: Path):
    """Synthetic athlete: 26 weeks of rides as FIT files -> full pipeline."""
    from goodpush.analyze import run_analysis
    from goodpush.config import init_project, load_project
    from goodpush.ingest import ingest
    from goodpush.report import write_reports

    proj = init_project(tmp_path)
    sdir = proj.path("strava")
    (sdir / "activities").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2024-01-01 15:00", tz="UTC")
    aid = 1000
    for day in range(180):
        if day % 7 in (0, 4):
            continue
        ftp = 220 + day * 0.2
        hard = day % 7 == 2
        n = 3600 if not hard else 4200
        p = rng.normal(0.65 * ftp, 15, n)
        if hard:
            for k in range(4):
                p[600 + k * 900: 600 + k * 900 + 480] = 1.08 * ftp
        p = np.clip(p, 0, 1500).astype(int)
        h = np.clip(100 + p * 0.25 + rng.normal(0, 3, n), 60, 200).astype(int)
        t0 = int((start + pd.Timedelta(days=day)).timestamp()) - fit.FIT_EPOCH_OFFSET
        fn = f"activities/{aid}.fit"
        (sdir / fn).write_bytes(make_fit(p.tolist(), h.tolist(), t0=t0))
        dt = (start + pd.Timedelta(days=day)).strftime("%b %d, %Y, %I:%M:%S %p")
        rows.append({"Activity ID": aid, "Activity Date": dt, "Activity Name": "ride",
                     "Activity Type": "Virtual Ride", "Elapsed Time": n, "Distance": 30,
                     "Filename": fn, "Moving Time": n, "Average Heart Rate": h.mean()})
        aid += 1
    pd.DataFrame(rows).to_csv(sdir / "activities.csv", index=False)
    proj = load_project(tmp_path)
    ingest(proj, workers=1, progress=lambda *a: None)
    from goodpush.models import markers
    from goodpush.models.dips import BreakLabels
    BreakLabels(proj.processed).set_period("2024-02-07", "2024-04-01", "gain")     # marked on the load chart
    BreakLabels(proj.processed).set_period("2024-05-01", "2024-05-06", "illness")
    said = []
    r = run_analysis(proj, progress=said.append)
    from goodpush.models.physio import PhysioRun
    assert PhysioRun(proj).analysed() and any("first fit" in str(s) for s in said)   # analysed on its own states
    res = write_reports(proj, r, progress=lambda *a: None)
    assert (proj.reports / "dashboard.html").exists()
    assert 200 < res["power"]["ftp_now"] < 300
    assert res["load"]["ctl_peak"] > 20
    mine = [p for p in res["progress_periods"] if p["yours"]]
    assert [str(p["start"])[:10] for p in mine] == ["2024-02-05"] and mine[0]["what_was_different"]   # its weeks
    assert "2024-05-01" in [e["onset"] for e in markers.learn(proj, progress=lambda *a: None)["illness"]["episode_list"]]
    # second run uses the cache
    out = ingest(proj, workers=1, progress=lambda *a: None)
    assert len(out) == len(rows)


def _fit_msgs(defs_and_rows):
    """Generic tiny FIT encoder: [(global_num, [(field, size, base_type, fmt)], [values...])]."""
    body = bytearray()
    for local, (gnum, fields, rows) in enumerate(defs_and_rows):
        body += bytes([0x40 | local, 0, 0]) + struct.pack("<H", gnum) + bytes([len(fields)])
        for f, size, bt, _ in fields:
            body += bytes([f, size, bt])
        fmt = "<" + "".join(x[3] for x in fields)
        for row in rows:
            body += bytes([local]) + struct.pack(fmt, *row)
    return struct.pack("<BBHI4sH", 14, 0x20, 2100, len(body), b".FIT", 0) + bytes(body) + b"\0\0"


def test_garmin_wellness(tmp_path: Path):
    import json
    import zipfile
    from goodpush.io.garmin import load_wellness
    w = tmp_path / "DI_CONNECT" / "DI-Connect-Wellness"
    a = tmp_path / "DI_CONNECT" / "DI-Connect-Aggregator"
    u = tmp_path / "DI_CONNECT" / "DI-Connect-Uploaded-Files"
    for d in (w, a, u):
        d.mkdir(parents=True)
    (w / "2024-01-01_2024-04-01_1_sleepData.json").write_text(json.dumps([
        {"calendarDate": "2024-01-02", "sleepStartTimestampGMT": "2024-01-02T07:00:00.0",
         "deepSleepSeconds": 3600, "lightSleepSeconds": 18000, "remSleepSeconds": 5400,
         "awakeSleepSeconds": 600, "sleepScores": {"overallScore": 85}}, {"retro": False}]))
    (a / "UDSFile_2024-01-01_2024-04-01.json").write_text(json.dumps([
        {"calendarDate": "2024-01-02", "restingHeartRate": 47, "currentDayRestingHeartRate": 46,
         "allDayStress": {"aggregatorList": [{"type": "TOTAL", "averageStressLevel": 25}]}}]))
    (w / "2024-01-01_2024-04-01_1_healthStatusData.json").write_text(json.dumps([
        {"calendarDate": "2024-01-03", "metrics": [{"type": "HRV", "value": 66.0,
                                                    "baselineLowerLimit": 58, "baselineUpperLimit": 80}]}]))
    ts = int(pd.Timestamp("2024-01-02 15:00", tz="UTC").timestamp()) - fit.FIT_EPOCH_OFFSET
    hrv_fit = _fit_msgs([
        (0, [(0, 1, 0x00, "B"), (4, 4, 0x86, "I")], [(68, ts)]),
        (370, [(253, 4, 0x86, "I"), (0, 2, 0x84, "H"), (1, 2, 0x84, "H")], [(ts, 70 * 128, 72 * 128)]),
    ])
    with zipfile.ZipFile(u / "UploadedFiles_0-_Part1.zip", "w") as z:
        z.writestr("x_1.fit", hrv_fit)
        z.writestr("x_2.fit", make_fit([100], [100]))
    daily, inv = load_wellness(tmp_path, "America/Los_Angeles", cache_dir=tmp_path / "cache",
                               progress=lambda *a: None)
    d2 = daily.loc["2024-01-02"]
    assert d2["sleep_h"] == pytest.approx(7.5) and d2["sleep_score"] == 85
    assert d2["rhr"] == 46 and d2["stress"] == 25
    assert d2["hrv"] == pytest.approx(72) and d2["hrv_weekly"] == pytest.approx(70)
    assert daily.loc["2024-01-03", "hrv"] == 66
    assert d2["bedtime_h"] == pytest.approx(23.0)
    # cached second pass gives the same result
    daily2, _ = load_wellness(tmp_path, "America/Los_Angeles", cache_dir=tmp_path / "cache",
                              progress=lambda *a: None)
    assert daily2.loc["2024-01-02", "hrv"] == pytest.approx(72)


def test_illness_notes_and_episode():
    from goodpush.models.illness import build_episodes, find_notes
    days = pd.date_range("2024-01-01", periods=120)
    d = pd.DataFrame(index=days)
    d["tss"] = 60.0
    d["rhr"] = 45.0
    for c in ("hrv_z", "sleep_h", "sleep_h_base", "stress", "atl", "tsb", "acwr", "ctl", "eq20_test"):
        d[c] = np.nan
    d.loc["2024-03-10":"2024-03-16", "tss"] = 0      # a week off sick
    d.loc["2024-03-10":"2024-03-13", "rhr"] = 52     # resting HR up 7 bpm
    acts = pd.DataFrame({"date": pd.to_datetime(["2024-03-18", "2024-02-01", "2024-02-05"]),
                         "name": ["Back after the lurgy", "a cold cold wet day", "Easy"],
                         "description": ["", "", "sick kid all week"], "private_note": ["", "", ""],
                         "type": ["Ride"] * 3})
    notes = find_notes(acts)
    assert list(notes["date"].dt.strftime("%m-%d")) == ["03-18"]
    ep = build_episodes(d, notes)
    assert str(ep.loc[0, "onset"]) == "2024-03-10"
    assert ep.loc[0, "days_off"] == 7 and ep.loc[0, "rhr_peak_delta"] == 7
    assert ep.loc[0, "severity"] == "severe"


class _FakeStrava:
    """Stands in for sync.strava.Client."""

    def __init__(self, acts):
        self.acts = {a["id"]: a for a in acts}

    def get(self, path, **params):
        if path == "/athlete/activities":
            after = params["after"]
            items = [a for a in self.acts.values()
                     if pd.Timestamp(a["start_date"]).timestamp() > after]
            return items if params["page"] == 1 else []
        aid = int(path.split("/")[2])
        if path.endswith("/streams"):
            n = 1800
            return {"time": {"data": list(range(n))}, "watts": {"data": [200] * n},
                    "heartrate": {"data": [140] * n}, "cadence": {"data": [90] * n}}
        return dict(self.acts[aid], description="felt good", private_note="")


def test_strava_sync_appends(tmp_path: Path):
    from goodpush.config import init_project, load_project
    from goodpush.ingest import ingest
    from goodpush.io.strava import load_activities_csv
    from goodpush.sync import strava as S
    proj = init_project(tmp_path)
    sdir = proj.path("strava")
    acts = [{"id": 1000 + i, "name": f"ride {i}", "sport_type": "VirtualRide",
             "start_date": f"2026-10-{10 + i:02d}T15:00:00Z", "elapsed_time": 1800,
             "moving_time": 1800, "distance": 15000.0, "average_watts": 200,
             "weighted_average_watts": 200, "average_heartrate": 140, "manual": False}
            for i in range(3)]
    fake = _FakeStrava(acts)
    n = S.sync(sdir, client=fake, since="2026-10-01", progress=lambda *a: None)
    assert n == 3
    df = load_activities_csv(sdir)
    assert len(df) == 3 and set(df["type"]) == {"Virtual Ride"} and df["indoor"].all()
    # second sync finds nothing new
    assert S.sync(sdir, client=fake, progress=lambda *a: None) == 0
    out = ingest(load_project(tmp_path), workers=1, progress=lambda *a: None)
    assert (out["error"].fillna("") == "").all()
    assert out["np"].round().eq(200).all()


def test_garmin_sync_normalises(tmp_path: Path):
    from goodpush.io.garmin import load_wellness
    from goodpush.sync import garmin as G

    class FakeGarmin:
        def get_sleep_data(self, d):
            return {"dailySleepDTO": {"calendarDate": d, "sleepStartTimestampGMT": 1767250800000,
                                      "deepSleepSeconds": 3600, "lightSleepSeconds": 18000,
                                      "remSleepSeconds": 5400, "awakeSleepSeconds": 300,
                                      "sleepScores": {"overall": {"value": 81}}}}

        def get_user_summary(self, d):
            return {"calendarDate": d, "restingHeartRate": 44, "totalSteps": 9000,
                    "averageStressLevel": 22, "bodyBatteryHighestValue": 90}

        def get_hrv_data(self, d):
            return {"hrvSummary": {"calendarDate": d, "lastNightAvg": 71, "weeklyAvg": 69,
                                   "baseline": {"balancedLow": 60, "balancedUpper": 80}}}

        def get_training_readiness(self, d):
            return [{"calendarDate": d, "score": 66, "hrvWeeklyAverage": 69, "timestamp": d}]

        def get_body_composition(self, a, b):
            return {"dateWeightList": [{"calendarDate": a, "weight": 68500.0}]}

    n = G.sync(tmp_path, api=FakeGarmin(), since=(pd.Timestamp.now() - pd.Timedelta(days=2))
               .strftime("%Y-%m-%d"), pause=0, progress=lambda *a: None)
    assert n == 3
    daily, inv = load_wellness(tmp_path, "UTC", progress=lambda *a: None)
    last = daily.iloc[-1]
    assert last["sleep_h"] == pytest.approx(7.5) and last["sleep_score"] == 81
    assert last["rhr"] == 44 and last["hrv"] == 71 and last["readiness"] == 66
    assert last["stress"] == 22 and daily["weight"].dropna().iloc[0] == pytest.approx(68.5)
    assert G.last_known_day(tmp_path) == daily.index.max()


def test_garmin_token_saving_variants(tmp_path: Path, monkeypatch=None):
    import os
    os.environ["GOODPUSH_SECRETS"] = str(tmp_path)
    from goodpush.sync import garmin as G

    class New:  # newer garminconnect: api.client.dumps()
        class client:
            @staticmethod
            def dumps():
                return "x" * 600

    class Old:  # older garminconnect: api.garth.dump(dir)
        class garth:
            @staticmethod
            def dump(path):
                Path(path, "oauth1_token.json").write_text("{}")

    assert G._save_tokens(New())
    assert (tmp_path / "garmin_tokens.txt").read_text() == "x" * 600
    assert G._save_tokens(Old())
    assert (tmp_path / "garmin_tokens" / "oauth1_token.json").exists()
    assert "rate-limiting" in G._friendly(RuntimeError("Mobile login returned 429"))
    del os.environ["GOODPUSH_SECRETS"]


def test_weigh_ins_normaliser():
    from goodpush.sync.garmin import norm_weight
    r = norm_weight({"dailyWeightSummaries": [{"summaryDate": "2026-10-01",
                                               "latestWeight": {"weight": 68200.0}}]})
    assert r[0]["weight"]["weight"] == 68200.0 and r[0]["metaData"]["calendarDate"].startswith("2026-10-01")


def test_deauth_removes_tokens(tmp_path: Path):
    import os
    os.environ["GOODPUSH_SECRETS"] = str(tmp_path)
    from goodpush.sync import garmin as G, strava as S, write_secret
    write_secret("strava.json", {"client_id": "1", "client_secret": "x" * 40,
                                 "access_token": "t", "refresh_token": "r", "expires_at": 9e12})
    calls = []
    orig = S._request
    S._request = (lambda url, data=None, token=None, method=None: calls.append(url) or ({}, {}))
    try:
        S.deauthorize(progress=lambda *a: None)
    finally:
        S._request = orig
    assert calls == [S.DEAUTH_URL] and not (tmp_path / "strava.json").exists()
    (tmp_path / "garmin_tokens").mkdir()
    (tmp_path / "garmin_tokens" / "garmin_tokens.json").write_text("{}")
    G.logout(progress=lambda *a: None)
    assert not (tmp_path / "garmin_tokens").exists()
    del os.environ["GOODPUSH_SECRETS"]


def test_workout_classifier():
    from goodpush.models.workouts import classify, name_label
    ftp = 250.0
    base = np.full(3600, 175.0)
    thr = base.copy()
    for k in range(3):
        thr[600 + k * 900: 600 + k * 900 + 600] = 250
    assert classify(thr, ftp)["type"] == "threshold"
    vo2 = base.copy()
    for k in range(5):
        vo2[600 + k * 480: 600 + k * 480 + 240] = 290
    assert classify(vo2, ftp)["type"] == "vo2max"
    assert classify(base, ftp)["type"] == "endurance"
    # same "threshold" power but HR stays low -> FTP was underestimated, not a threshold ride
    hr = np.full(3600, 120.0)
    assert classify(thr, ftp, hr=hr, lthr=175)["type"] != "threshold"
    assert name_label("Zwift - VO2max: Bossi Intervals") == "vo2max"
    assert name_label("Lactate Clearance: Over-Unders") == "threshold"
    assert name_label("Test: Power Profile - FTP Protocol") == "test"


def test_import_exports(tmp_path: Path):
    import zipfile
    from goodpush.config import init_project
    from goodpush.importer import import_path
    proj = init_project(tmp_path / "proj")
    sz = tmp_path / "export_123.zip"
    with zipfile.ZipFile(sz, "w") as z:
        z.writestr("export_123/activities.csv", "Activity ID,Activity Date\n")
        z.writestr("export_123/activities/1.fit.gz", b"x")
        z.writestr("export_123/media/photo.jpg", b"x")
        z.writestr("../evil.txt", b"x")
    assert import_path(proj, sz, progress=lambda *a: None) == "strava"
    assert (proj.path("strava") / "activities.csv").exists()
    assert (proj.path("strava") / "activities" / "1.fit.gz").exists()
    assert not (proj.path("strava") / "media").exists()
    assert not (tmp_path / "proj" / "data" / "raw" / "evil.txt").exists()
    gz = tmp_path / "garmin.zip"
    with zipfile.ZipFile(gz, "w") as z:
        z.writestr("DI_CONNECT/DI-Connect-Wellness/a_sleepData.json", "[]")
    assert import_path(proj, gz, progress=lambda *a: None) == "garmin"
    assert (proj.path("garmin") / "DI_CONNECT" / "DI-Connect-Wellness" / "a_sleepData.json").exists()
    bad = tmp_path / "x.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("readme.txt", "hi")
    with pytest.raises(ValueError):
        import_path(proj, bad, progress=lambda *a: None)


def test_garmin_block_guard(tmp_path: Path):
    import os
    os.environ["GOODPUSH_SECRETS"] = str(tmp_path)
    from goodpush.sync import garmin as G
    G._note_block()
    assert G._blocked_until() is not None
    with pytest.raises(SystemExit):
        G.login()
    del os.environ["GOODPUSH_SECRETS"]


def test_fit_workout_roundtrip():
    from goodpush.io.fit import read_messages
    from goodpush.io.fit_workout import fit_crc, workout_fit
    steps = [dict(name="Warm up", min=10, lo=100, hi=130, kind="warmup"),
             {"reps": 3, "steps": [dict(name="Threshold", min=10, lo=190, hi=200),
                                   dict(name="Recover", min=5, lo=90, hi=110, kind="rest")]},
             dict(name="Cool down", min=5, lo=80, hi=100, kind="cooldown")]
    b = workout_fit("Thr 3x10", steps)
    assert fit_crc(b) == 0                       # whole-file CRC checks out
    m = read_messages(b)
    assert m[0][0][0] == 5                       # file type = workout
    assert m[26][0][8] == "Thr 3x10" and m[26][0][6] == 5
    ws = m[27]
    assert ws[1][5] == 1190 and ws[1][6] == 1200  # watts + 1000
    assert ws[3][1] == 6 and ws[3][2] == 1 and ws[3][4] == 3  # repeat steps 1..2 three times
    assert ws[4][7] == 3                          # cooldown intensity
    nested = [{"reps": 2, "steps": [{"reps": 3, "steps": [dict(name="On", min=1, lo=300, hi=320, cadence=100, cue="go")]},
                                    dict(name="Rest", min=4, lo=100, hi=120, kind="rest")]}]
    ws = read_messages(workout_fit("Sets", nested))[27]
    assert [(w.get(1), w.get(2), w.get(4)) for w in ws if w.get(1) == 6] == [(6, 0, 3), (6, 0, 2)]  # inner, then outer
    assert ws[0][8] == "go" and (ws[0][19], ws[0][21], ws[0][22]) == (3, 95, 105)                  # cue, cadence


def test_garmin_status_and_curl_guard(tmp_path: Path):
    import os
    os.environ["GOODPUSH_SECRETS"] = str(tmp_path)
    from goodpush.sync import garmin as G
    st = G.status()
    assert st["tokens"] is False and st["blocked_hours_left"] == 0
    old_v, old_c, old_imp = G._version, G._has_curl_cffi, G._import_garmin
    G._version, G._has_curl_cffi, G._import_garmin = (lambda: (0, 3)), (lambda: False), (lambda: object)
    try:
        try:
            G.login()
            raise AssertionError("login should refuse without curl_cffi")
        except SystemExit as e:
            assert "curl_cffi" in str(e)
        assert "curl_cffi is NOT installed" in G._friendly(Exception("429 Too Many Requests"))
    finally:
        G._version, G._has_curl_cffi, G._import_garmin = old_v, old_c, old_imp
        del os.environ["GOODPUSH_SECRETS"]


def test_garmin_browser_import(tmp_path: Path):
    import json
    from goodpush.config import init_project, load_project
    from goodpush.importer import import_path
    from goodpush.io.garmin import load_wellness
    from goodpush.sync import garmin as G
    js = G.browser_snippet(tmp_path / "g", since="2026-10-01")
    assert 'const START = "2026-10-01"' in js and "__START__" not in js and "__ACT_START__" not in js
    day = {"sleep": {"dailySleepDTO": {"calendarDate": "2026-10-19", "sleepTimeSeconds": 24300,
                                       "deepSleepSeconds": 5000, "lightSleepSeconds": 13000,
                                       "remSleepSeconds": 6300, "awakeSleepSeconds": 600,
                                       "sleepStartTimestampGMT": 1792395000000,
                                       "sleepEndTimestampGMT": 1792419300000,
                                       "sleepScores": {"overall": {"value": 81}}}},
           "summary": {"calendarDate": "2026-10-19", "restingHeartRate": 44, "totalSteps": 5739,
                       "averageStressLevel": 23, "bodyBatteryHighestValue": 96},
           "hrv": {"hrvSummary": {"calendarDate": "2026-10-19", "lastNightAvg": 72, "weeklyAvg": 71,
                                  "baseline": {"balancedLow": 65, "balancedUpper": 80}}},
           "readiness": [{"calendarDate": "2026-10-19", "score": 70, "timestamp": "2026-10-19T14:00:00.0"}]}
    f = tmp_path / "goodpush_garmin_2026-10-18_2026-10-20.json"
    f.write_text(json.dumps({"source": "goodpush-garmin-browser", "version": 1, "start": "2026-10-18",
                             "end": "2026-10-20", "days": {"2026-10-19": day,
                                                          "2026-10-20": {"sleep": None, "summary": None,
                                                                         "hrv": None, "readiness": []}},
                             "weight": {"dailyWeightSummaries": [
                                 {"summaryDate": "2026-10-18", "latestWeight": {"weight": 70500}}]}}))
    proj = load_project(init_project(tmp_path / "p").root)
    assert import_path(proj, f, progress=lambda *a: None) == "garmin_browser"
    d, _ = load_wellness(proj.path("garmin"), "UTC", cache_dir=None, progress=lambda *a: None,
                         scan_zips=False)
    row = d.loc["2026-10-19"]
    assert row["rhr"] == 44 and row["hrv"] == 72 and abs(row["sleep_h"] - 6.75) < 0.05
    assert not (proj.path("garmin") / "api" / "raw" / "2026-10-20.json").exists()


def test_tomledit(tmp_path: Path):
    from goodpush.app.tomledit import set_values
    from goodpush.config import init_project, load_project
    proj = init_project(tmp_path)
    set_values(proj.root / "goodpush.toml", "plan", {"hours_per_week": 9.5, "key_days": ["tue", "fri"],
                                                     "ftp": 231, "new_key": "x"})
    cfg = load_project(tmp_path).cfg["plan"]
    assert cfg["hours_per_week"] == 9.5 and cfg["key_days"] == ["tue", "fri"] and cfg["ftp"] == 231
    assert cfg["new_key"] == "x"
    txt = (proj.root / "goodpush.toml").read_text()
    assert "# the two days for hard sessions" in txt          # comments kept
    set_values(proj.root / "goodpush.toml", "brand_new", {"a": 1})
    assert load_project(tmp_path).cfg["brand_new"]["a"] == 1


def test_tomledit_tables_numbers_and_text(tmp_path: Path):
    """tomlkit keeps a changed line's comment, a section's sub-tables, numpy numbers as numbers, text as UTF-8"""
    import tomllib
    import numpy as np
    from goodpush.app.tomledit import set_table, set_tables, set_values
    p = tmp_path / "goodpush.toml"
    p.write_text('[ftp]\nmode = "auto"  # how FTP is set\nold = 1\n\n[ftp.history]\n"2026-10-22" = 250\n\n'
                 '[[events]]\ndate = "2026-01-01"\nnote = "x"\n', encoding="utf-8")
    set_values(p, "ftp", {"mode": "manual", "days": np.int64(60), "w": np.float64(1 / 3)})
    set_table(p, "ftp", {"mode": "manual", "days": 60})          # "old" and "w" go, [ftp.history] stays
    set_tables(p, "events", [{"date": "2026-03-01", "note": "Grippe – fièvre"}, {"date": "2026-04-01", "note": "y"}])
    set_values(p, "ftp.history", {"2026-10-30": 243})
    txt = p.read_text(encoding="utf-8")
    cfg = tomllib.loads(txt)
    assert cfg["ftp"] == {"mode": "manual", "days": 60, "history": {"2026-10-22": 250, "2026-10-30": 243}}
    assert [e["note"] for e in cfg["events"]] == ["Grippe – fièvre", "y"]
    assert 'mode = "manual"  # how FTP is set' in txt and '"2026-10-30" = 243' in txt
    set_values(p, "ftp", {"w": np.float64(1 / 3)})
    assert tomllib.loads(p.read_text(encoding="utf-8"))["ftp"]["w"] == 0.333
    set_tables(p, "events", [])
    assert "events" not in tomllib.loads(p.read_text(encoding="utf-8"))


def test_app_server_auth_and_state(tmp_path: Path):
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from goodpush.app.server import make_handler
    from goodpush.config import init_project, load_project
    init_project(tmp_path)
    proj = load_project(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proj, "k" * 20, lan=True))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        # from 127.0.0.1 no key is needed; state works even before any analysis
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state") as r:
            js = json.loads(r.read())
        assert "plan" in js and "ladders" in js and js["code_stale"] is False      # this process runs the code on disk
        from goodpush.app import server
        code, server.CODE = server.CODE, "0"             # goodpush updated on disk since the app started
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/job") as r:
                job = json.loads(r.read())
            assert job["code_stale"] is True                             # the page polls this: it says so at once
        finally:
            server.CODE = code
        # the poll also says how new the rides and plan are: a page that missed a job's end compares and reloads
        assert (job["data_version"], job["plan_version"]) == (js["data_version"], js["plan_version"])
        import os
        ps = proj.processed / "plan_state.json"
        ps.write_text("{}", encoding="utf-8")
        os.utime(ps, (1e9, 1e9))
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/job") as r:
            assert json.loads(r.read())["plan_version"] == 1e9 != js["plan_version"]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
            assert b"goodpush" in r.read()
        # POST without the custom header is refused (CSRF guard)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/action", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(req)
            raise AssertionError("POST without header accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 403
        # path traversal
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/..%2f..%2fgoodpush.toml")
            raise AssertionError("traversal served")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown()


def test_garmin_browser_activities(tmp_path: Path):
    import base64
    import io
    import json
    import zipfile
    from goodpush.config import init_project, load_project
    from goodpush.importer import import_path
    from goodpush.ingest import ingest
    from goodpush.io.strava import load_activities_csv
    proj = load_project(init_project(tmp_path / "p").root)
    # one Strava export ride, and two Garmin rides: one duplicate of it, one new
    sdir = proj.path("strava")
    sdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Activity ID": 1, "Activity Date": "Oct 10, 2026, 3:00:00 PM", "Activity Name": "strava ride",
                   "Activity Type": "Ride", "Elapsed Time": 3600, "Filename": ""}]).to_csv(sdir / "activities.csv", index=False)
    t0 = int(pd.Timestamp("2026-10-12 15:00", tz="UTC").timestamp()) - fit.FIT_EPOCH_OFFSET
    fitbytes = make_fit([200] * 1800, [140] * 1800, t0=t0)
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        z.writestr("123_ACTIVITY.fit", fitbytes)
    b64 = base64.b64encode(zb.getvalue()).decode()
    acts = [{"summary": {"activityId": 999, "activityName": "dup", "startTimeGMT": "2026-10-10 15:01:00",
                         "activityType": {"typeKey": "road_biking"}, "duration": 3600}, "file_b64": None},
            {"summary": {"activityId": 123, "activityName": "TR: Baird", "description": "felt ill",
                         "startTimeGMT": "2026-10-12 15:00:00", "activityType": {"typeKey": "indoor_cycling"},
                         "duration": 1800, "avgPower": 200, "averageHR": 140}, "file_b64": b64}]
    f = tmp_path / "goodpush_garmin_2026-10-09_2026-10-12.json"
    f.write_text(json.dumps({"source": "goodpush-garmin-browser", "version": 2, "start": "2026-10-09",
                             "end": "2026-10-12", "days": {}, "weight": None, "activities": acts}))
    assert import_path(proj, f, progress=lambda *a: None) == "garmin_browser"
    assert (proj.path("garmin") / "activities" / "garmin_123.fit").read_bytes() == fitbytes
    a = load_activities_csv(proj.path("strava"), proj.path("garmin"))
    assert list(a["name"]) == ["strava ride", "TR: Baird"]          # duplicate dropped
    g = a.iloc[1]
    assert g["type"] == "Virtual Ride" and g["indoor"] and g["id"] == "garmin_123"
    assert g["description"] == "felt ill"
    out = ingest(proj, workers=1, progress=lambda *a: None)
    row = out[out["id"].astype(str).eq("garmin_123")].iloc[0]
    assert abs(row["avg_power"] - 200) < 1
    # re-import without the file keeps the file we already have
    acts[1]["file_b64"] = None
    f.write_text(json.dumps({"source": "goodpush-garmin-browser", "version": 2, "days": {}, "activities": acts}))
    import_path(proj, f, progress=lambda *a: None)
    tab = pd.read_csv(proj.path("garmin") / "activities" / "activities_garmin.csv", dtype=str, keep_default_na=False)
    assert len(tab) == 2 and tab.loc[tab["id"].eq("garmin_123"), "filename"].iloc[0] == "garmin_123.fit"


def test_read_track_fit(tmp_path: Path):
    from goodpush.io.streams import read_track
    # a FIT with two records carrying positions (semicircles)
    k = 2 ** 31 / 180.0
    fields = [(253, 4, 0x86, "I"), (0, 4, 0x85, "i"), (1, 4, 0x85, "i")]
    rows = [[1_000_000_000 + i, int(47.6 * k), int((-122.3 + i * 1e-4) * k)] for i in range(5)]
    data = _fit_msgs([(20, fields, rows)])
    f = tmp_path / "a.fit"
    f.write_bytes(data)
    tr = read_track(f)
    assert len(tr) == 5 and abs(tr[0][0] - 47.6) < 1e-4 and abs(tr[-1][1] - (-122.2996)) < 1e-4


def test_illness_markers_learn_and_status():
    from goodpush.models import markers as M
    idx = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    rng = np.random.default_rng(3)
    d = pd.DataFrame(index=idx)
    d["rhr"] = 48 + rng.normal(0, 1, len(idx))
    d["hrv"] = 70 + rng.normal(0, 4, len(idx))
    d["sleep_h"] = 7.3 + rng.normal(0, 0.4, len(idx))
    d["tss"] = 60.0
    noticed = pd.to_datetime(["2022-04-10", "2022-09-01", "2023-02-14", "2023-07-20", "2024-01-05",
                              "2024-06-11"])
    for n in noticed:   # resting HR climbs 4 days before you notice; HRV unrelated
        d.loc[n - pd.Timedelta(days=4): n, "rhr"] += 6
    ep = pd.DataFrame({"onset": noticed.date, "noticed": noticed.date, "severity": "moderate",
                       "days_off": 5})
    L = M.discover(d, ep)
    assert not L["fallback"]
    assert L["chosen"][0]["key"] == "rhr" and L["chosen"][0]["caught"] == 6
    assert all(c["key"] != "hrv" for c in L["chosen"])
    assert L["combined"]["caught"] >= 5
    # daily tracking: a new rise today -> warning
    d2 = d.copy()
    d2.loc[d2.index[-3:], "rhr"] += 7
    st = M.status(d2, L)
    assert st["level"] == ("warning" if len(st["firing"]) >= 2 else "watch") and "Resting HR" in st["firing"][0]
    assert M.status(d, L)["level"] in ("clear", "watch")
    # too little history -> general rule
    L2 = M.discover(d, ep.iloc[:2])
    assert L2["fallback"] and L2["chosen"][0]["default"]


def test_marker_levels_one_watch_two_warning_general_rule_warns():
    from goodpush.models import markers as M
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"rhr": 48 + rng.normal(0, 1, len(idx)), "hrv": 70 + rng.normal(0, 2, len(idx))}, index=idx)
    rhr = {"key": "rhr", "label": M.LABELS["rhr"], "column": "rhr_d", "op": ">=", "threshold": 3, "days": 1,
           "unit": "bpm", "rule": "rhr"}
    hrv = {"key": "hrv", "label": M.LABELS["hrv"], "column": "hrv_z", "op": "<=", "threshold": -2.0, "days": 1,
           "unit": "z", "rule": "hrv"}
    learned = {"chosen": [rhr, hrv], "combined": {"min_markers": 1}, "fallback": False}
    assert M.status(d, learned)["level"] == "clear"
    d.loc[idx[-1], "rhr"] += 8
    assert M.status(d, learned)["level"] == "watch"                         # one marker, whatever min_markers says
    d.loc[idx[-1], "hrv"] -= 30
    assert M.status(d, learned)["level"] == "warning"                       # two together
    general = {"chosen": [{**rhr, "default": True}], "combined": {"min_markers": 1}, "fallback": True}
    assert M.status(d, general)["level"] == "warning"                       # the general rule alone


def test_marked_illness_absorbs_note_episodes_and_close_overreaching_is_one():
    from goodpush.models.markers import Episodes
    ep = pd.DataFrame({"onset": pd.to_datetime(["2024-04-25", "2024-05-03", "2024-08-01"]).date,
                       "noticed": pd.to_datetime(["2024-04-24", "2024-05-02", "2024-08-01"]).date,
                       "severity": ["unconfirmed (no gap)", "severe", "moderate"], "days_off": [3, 4, 5]})
    told = [{"label": "illness", "start": "2024-05-01", "end": "2024-05-06", "days": 6},
            {"label": "travel", "start": "2024-06-01", "end": "2024-06-10", "days": 10}]
    ill = Episodes.illness(ep, told).sort_values("noticed").reset_index(drop=True)
    assert len(ill) == 2                                                    # two note episodes + the period = one
    assert str(ill["noticed"][0]) == "2024-04-24" and str(ill["onset"][0]) == "2024-04-25"
    assert ill["severity"][0] == "severe" and ill["days_off"][0] == 6
    assert str(ill["noticed"][1]) == "2024-08-01"                            # far away: on its own
    assert len(Episodes.illness(ep, [])) == 3
    alone = Episodes.illness(pd.DataFrame(), told[:1])
    assert len(alone) == 1 and str(alone["noticed"][0]) == "2024-05-01" and alone["severity"][0] == "moderate"
    over = Episodes.overreaching([{"label": "overreaching", "start": "2024-03-16", "end": "2024-03-25"},
                                  {"label": "overreaching", "start": "2024-03-01", "end": "2024-03-08"},
                                  {"label": "overreaching", "start": "2024-06-01", "end": "2024-06-07"},
                                  {"label": "illness", "start": "2024-03-05", "end": "2024-03-07"}])
    assert [str(x) for x in over["noticed"]] == ["2024-03-01", "2024-06-01"]   # 15 days apart: one, from the first
    assert list(over["days_off"]) == [25, 7] and len(Episodes.overreaching([])) == 0


def breaks_daily(seed: int = 5, wellness: bool = True) -> pd.DataFrame:
    """two years of riding six days a week with body readings, and five breaks: one without body readings (2023-06),
    a quiet one (2023-10), after a heavy block with HRV down (2024-03), a trip (2024-05), an illness (2024-09)."""
    from goodpush.metrics.load import pmc
    idx = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(index=idx)
    d["tss"] = np.where(idx.day_of_week == 0, 0.0, 80.0) + rng.normal(0, 10, len(idx)).clip(-30, 30)
    body = ["rhr", "hrv", "sleep_h", "bedtime_h", "steps", "respiration"]
    for c, m, sd in zip(body, (46, 70, 7.4, 23.0, 6000, 14), (0.7, 3, 0.3, 0.3, 800, 0.3)):
        d[c] = m + rng.normal(0, sd, len(idx))
    d.loc["2024-05-01":"2024-05-12", ["tss", "bedtime_h", "steps"]] = [0.0, 14.0, 16000]   # travel
    d.loc["2024-09-10":"2024-09-20", "tss"] = 0.0                                          # illness
    d.loc["2024-09-09":"2024-09-15", "rhr"] += 7
    d.loc["2024-09-09":"2024-09-15", "hrv"] -= 15
    d.loc["2024-02-05":"2024-03-03", "tss"] *= 2.2                                         # heavy block ...
    d.loc["2024-02-20":"2024-03-03", "hrv"] -= 6                                           # ... HRV down before
    d.loc["2024-03-04":"2024-03-12", "tss"] = 0.0
    d.loc["2023-06-01":"2023-06-09", "tss"] = 0.0                                          # no body readings
    d.loc["2023-05-10":"2023-06-25", body] = np.nan
    d.loc["2023-10-02":"2023-10-10", "tss"] = 0.0                                          # quiet
    if not wellness:
        d[body] = np.nan
    p = pmc(d["tss"])
    for c in ("ctl", "atl", "tsb", "acwr", "monotony"):
        if c in p:
            d[c] = p[c]
    return d


def test_training_breaks_found_as_before():
    """2026-11-11 port to classes (TrainingBreaks): the breaks and their CTL numbers are the old analyze()'s on this
    table (checked equal to the old module, analyze and around, before the cause rules changed); detection is
    vectorised (run edges by diff)."""
    from goodpush.models.dips import TrainingBreaks
    res = TrainingBreaks(breaks_daily()).analyze()
    assert [str(s) for s in res["start"]] == ["2023-06-01", "2023-10-02", "2024-03-04", "2024-05-01", "2024-09-09"]
    assert res["days"].tolist() == [9, 9, 9, 13, 12]
    assert res["ctl_before"].tolist() == [67.3, 69.3, 111.5, 76.2, 68.6] and res["ctl_min"].tolist() == [54.4, 55.9, 89.7, 56.1, 51.6]
    # runs: 3+ low days; runs JOIN_DAYS or fewer apart are one break; 2-day rests don't chain
    days = pd.date_range("2025-01-01", periods=120)
    tss = np.full(120, 60.0)
    tss[[60, 61]] = 0.0                                   # two days off: a rest, not a run
    tss[70:74] = tss[77:82] = 0.0                         # 4 + 5 days off, 3 days apart: one break
    run = TrainingBreaks(pd.DataFrame({"tss": tss}, index=days)).runs()
    assert run == [(days[70], days[81])]


def test_training_breaks_causes_and_labels(tmp_path: Path):
    """the automatic causes (2026-11-11): travel, illness, overreaching only with a body sign, "unclear" without body
    readings (not "life - no stress signs"), "life" when the readings say nothing; your label wins."""
    from goodpush.models.dips import BreakLabels, TrainingBreaks
    d = breaks_daily()
    res = TrainingBreaks(d).analyze()
    by = {str(r["start"])[:7]: r for r in res.to_dict("records")}
    assert by["2024-05"]["auto_cause"] == "travel" and by["2024-09"]["auto_cause"] == "illness"
    assert by["2024-03"]["auto_cause"] == "overreaching"
    assert by["2023-06"]["auto_cause"] == "unclear" and by["2023-06"]["confidence"] == "low"
    assert by["2023-06"]["reasons"] == ["no sleep / HRV / resting-HR data around this break"]
    assert by["2023-10"]["auto_cause"] == "life"
    per, prof = TrainingBreaks.around(d, res)
    assert len(per) == len(res) and len(prof["offset"]) == 57 and {"travel", "illness"} <= set(prof["causes"])
    labels = BreakLabels(tmp_path)
    labels.set(str(by["2024-05"]["start"]), "life")
    r2 = TrainingBreaks(d, labels=labels.load()).analyze().set_index("start").loc[by["2024-05"]["start"]]
    assert r2["cause"] == "life" and r2["auto_cause"] == "travel"
    labels.set(str(by["2024-05"]["start"]), "")
    assert labels.load() == {}
    with pytest.raises(ValueError):
        labels.set("2024-05-01", "bored")


def test_periods_you_mark_on_the_load_chart(tmp_path: Path):
    """a period you mark (Fitness > Mark periods): kept beside a break label on the same day, its edges moved and its
    cause changed via `was`, removed; a gain period described on its weeks like the found ones."""
    from goodpush.models import insights as I
    from goodpush.models.dips import BreakLabels, TrainingBreaks
    lab = BreakLabels(tmp_path)
    lab.set("2024-05-06", "travel")
    lab.set_period("2024-05-12", "2024-05-06", "illness")                    # dragged right to left
    assert lab.periods() == [{"start": "2024-05-06", "end": "2024-05-12", "cause": "illness"}]
    assert lab.load()["2024-05-06"] == "travel"                                # the break's own label stays
    lab.set_period("2024-05-04", "2024-05-14", "overreaching", was=("2024-05-06", "2024-05-12"))
    lab.set_period("2024-01-01", "2024-03-10", "gain")
    assert [(p["start"], p["cause"]) for p in lab.periods()] == [("2024-01-01", "gain"), ("2024-05-04", "overreaching")]
    assert lab.periods("gain")[0]["end"] == "2024-03-10"
    lab.set_period("2024-05-04", "2024-05-14", "", was=("2024-05-04", "2024-05-14"))
    assert [p["cause"] for p in lab.periods()] == ["gain"]
    with pytest.raises(ValueError):
        lab.set_period("2024-05-01", "2024-05-03", "bored")
    d = breaks_daily()
    assert len(TrainingBreaks(d, labels=lab.load()).analyze())                 # the periods list is no break label
    W = pd.DataFrame({"perf": np.linspace(200, 240, 20)}, index=pd.date_range("2024-01-01", periods=20, freq="7D"))
    m = I.marked_periods(W, lab.periods("gain"))
    assert (m["start"].iloc[0], m["end"].iloc[0]) == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-04"))
    assert m["yours"].iloc[0] and m["weeks"].iloc[0] == 9 and 5 < m["change_pct"].iloc[0] < 10


def test_overreaching_needs_a_body_sign():
    """load alone - a hard block before a break - is no overreaching (2026-11-11): without HRV down / resting HR up
    beyond the load, or a slow bounce back, the break is a quiet one."""
    from goodpush.models.dips import TrainingBreaks
    d = breaks_daily()
    d.loc["2024-02-20":"2024-03-03", "hrv"] += 6           # the same block, the body fine
    r = TrainingBreaks(d).analyze().set_index("start").loc[pd.Timestamp("2024-03-04").date()]
    assert r["auto_cause"] != "overreaching" and r["evidence"]["pre_tsb_min"] < -25
    assert not any("beyond what the load explains" in x for x in r["reasons"])


def test_load_cutoffs_follow_your_own_history():
    """a fast build / deep fatigue against the rider's own 4-week ramps and form (9 in 10 of their days), the fixed
    numbers while the history is short."""
    from goodpush.models.dips import LoadCutoffs
    days = pd.date_range("2024-01-01", periods=400)
    ramp = np.linspace(0, 40, 400)                             # CTL rising 0.1 a day: 4-week ramps of 2.8
    c = LoadCutoffs(pd.DataFrame({"ctl": ramp, "tsb": np.linspace(-5, 5, 400)}, index=days))
    assert c.ramp == pytest.approx(2.8, abs=0.01) and c.tsb == pytest.approx(-4.0, abs=0.05)
    short = LoadCutoffs(pd.DataFrame({"ctl": ramp[:100], "tsb": 0.0}, index=days[:100]))
    assert (short.ramp, short.tsb) == (LoadCutoffs.RAMP, LoadCutoffs.TSB)
    assert (LoadCutoffs(pd.DataFrame()).ramp, LoadCutoffs(pd.DataFrame()).tsb) == (LoadCutoffs.RAMP, LoadCutoffs.TSB)


def test_known_facts_come_before_any_guess(tmp_path: Path):
    """what the app knows decides a break's cause directly, confidence high (2026-11-11): a day marked sick, days away
    (travel when sleep / steps show it, else life), a planned recovery phase ("planned"); your label still wins."""
    import json
    from goodpush.models.dips import CAUSES, KnownFacts, TrainingBreaks
    from goodpush.planning.season import Phase, Season
    d = breaks_daily()
    (tmp_path / "data").mkdir()
    (tmp_path / "plan_state.json").write_text(json.dumps({
        "overrides": {"2023-10-01": "sick", "2024-01-01": "easy"},
        "away": {"2024-05-03": "family visit", "2023-06-02": ""}}), encoding="utf-8")
    Season([Phase("recovery", "2024-03-04", 2)]).save(tmp_path)
    known = KnownFacts.of(tmp_path, tmp_path)
    by = {str(r["start"])[:7]: r for r in TrainingBreaks(d, known=known).analyze().to_dict("records")}
    assert (by["2023-10"]["auto_cause"], by["2023-10"]["confidence"], by["2023-10"]["reasons"]) == ("illness", "high", ["you marked yourself sick"])
    assert by["2024-05"]["auto_cause"] == "travel" and by["2024-05"]["reasons"][0] == "days away: family visit"
    assert (by["2023-06"]["auto_cause"], by["2023-06"]["reasons"]) == ("life", ["days away"])   # no travel signs
    assert (by["2024-03"]["auto_cause"], by["2024-03"]["reasons"]) == ("planned", ["a planned recovery phase"])
    assert by["2024-09"]["auto_cause"] == "illness" and "planned" in CAUSES        # untouched: the evidence's guess
    labelled = TrainingBreaks(d, labels={str(by["2024-03"]["start"]): "overreaching"}, known=known).analyze()
    assert labelled.set_index("start").loc[by["2024-03"]["start"], "cause"] == "overreaching"
    assert KnownFacts.of(tmp_path / "nowhere", tmp_path / "nowhere").cause(pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-12"),
                                                                           {"bedtime_shift_nights": 0, "steps_high_days": 0}) is None


def test_alpha1_features(tmp_path: Path):
    from goodpush import alpha as AL
    from goodpush.config import init_project, load_project
    proj = load_project(init_project(tmp_path).root)
    sdir = proj.processed / "streams"
    sdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    ftp = 220.0
    rows = []
    days = pd.date_range("2025-01-01", periods=30, freq="2D")
    for i, day in enumerate(days):
        lead = 1150 if i == len(days) - 2 else 0         # recording started 19 min before the first pedal stroke
        t = np.arange(0, 3600 + lead, 5)
        tr = t - lead
        p = np.where(tr < 0, 0.0, np.where(tr < 25 * 60, 130.0, np.where(tr < 50 * 60, 200.0, 240.0)))   # easy, moderate, hard
        a1 = np.clip(1.6 - 0.0045 * p, 0.2, 2) + rng.normal(0, 0.03, t.size)    # crosses 0.75 at ~189 W
        hr = 60 + 0.45 * p + rng.normal(0, 1, t.size)
        if i == len(days) - 1:
            a1[tr < 25 * 60] -= 0.35      # the last ride: alpha1 clearly low in the warm-up
        np.savez_compressed(sdir / f"r{i}.a1.npz", t=t, a1=a1, p=p, hr=hr, art=np.zeros(t.size))
        rows.append({"id": f"r{i}", "date": day, "ftp": ftp, "indoor": True})
    af = AL.features(proj, pd.DataFrame(rows))
    assert len(af) == 30
    last = af.iloc[-1]
    assert last["a1_warm_z"] <= -2.5
    assert abs(af.iloc[-2]["a1_warm_z"]) < 2 and af.iloc[-2]["a1_warm_s"] >= 14 * 60   # warm-up from the first pedal stroke
    est = af["hrvt1_28d"].dropna()
    assert len(est) and abs(est.iloc[-1] - 189) < 25
    d = AL.to_daily(af, pd.date_range("2025-01-01", "2025-03-01"))
    assert d["a1_warm_z"].notna().sum() >= 20


def test_garmin_personal_api_sync(tmp_path, monkeypatch):
    """Personal API: key -> token -> days + activities, stored like the browser download."""
    import io as _io
    import json as _json
    import threading
    import zipfile
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import pandas as pd
    from goodpush.sync import garmin_api as G

    monkeypatch.setenv("GOODPUSH_SECRETS", str(tmp_path / "secrets"))
    today = pd.Timestamp.now().normalize()
    d1 = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fit = b"\x0e\x10\x00\x00\x00\x00\x00\x00.FIT\x00\x00"
    zb = _io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        z.writestr("123_ACTIVITY.fit", fit)
    seen = {"token": 0, "auth": set(), "files": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj=None, raw=None):
            body = raw if raw is not None else _json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.headers["Authorization"] == "ApiKey test-key-0123456789abcdef"
            if "application/json" in (self.headers.get("Accept") or ""):
                return self._send(406, {"error": "NotAcceptableException"})   # what Garmin does
            assert not self.headers.get("Content-Type")
            seen["token"] += 1
            self._send(200, raw=f"tok{seen['token']}".encode())                # plain-text token

        def do_GET(self):
            seen["auth"].add(self.headers.get("Authorization"))
            p, _, q = self.path.partition("?")
            if p == "/sleep-service/sleep/dailySleep":
                ds = q.split("date=")[1][:10]
                return self._send(200, {"calendarDate": ds, "sleepTimeSeconds": 27000, "deepSleepSeconds": 5000,
                                        "lightSleepSeconds": 15000, "remSleepSeconds": 7000, "awakeSleepSeconds": 600,
                                        "sleepScores": {"overall": {"value": 81}}})
            if p == "/usersummary-service/usersummary/daily":
                ds = q.split("calendarDate=")[1][:10]
                return self._send(200, {"calendarDate": ds, "restingHeartRate": 44, "totalSteps": 9000})
            if p.startswith("/hrv-service/hrv/"):
                return self._send(200, {"hrvSummary": {"calendarDate": p[-10:], "lastNightAvg": 70,
                                                       "weeklyAvg": 68, "baseline": {"balancedLow": 60,
                                                                                     "balancedUpper": 80}}})
            if p.startswith("/weight-service/weight/range/"):
                if self.headers.get("Accept") != "*/*":
                    return self._send(406, {})                                 # retried with */*
                return self._send(200, {"dailyWeightSummaries": [
                    {"summaryDate": d1, "latestWeight": {"weight": 70500}}]})
            if p == "/activitylist-service/activities/search/activities":
                return self._send(200, [] if "start=0" not in q else [
                    {"activityId": 123, "activityName": "Tempo", "startTimeGMT": f"{d1} 15:00:00",
                     "activityType": {"typeKey": "virtual_ride"}, "duration": 3600, "avgPower": 220}])
            if p == "/download-service/files/activity/123":
                seen["files"] += 1
                return self._send(200, raw=zb.getvalue())
            return self._send(404, {})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setattr(G, "BASE", base)
    monkeypatch.setattr(G, "TOKEN_URL", base + "/token")
    monkeypatch.setattr(G, "PAUSE", 0.0)
    try:
        assert not G.has_key()
        G.save_key("ApiKey test-key-0123456789abcdef")
        assert G.has_key() and "test-key" not in _json.dumps(G.key_info())
        gdir = tmp_path / "garmin"
        r = G.sync(gdir, since=d1, progress=lambda *a: None)
        assert r["days"] == 2 and r["activities"] == 1 and r["files"] == 1 and not r["stopped"]
        assert seen["token"] == 1 and seen["auth"] == {"Bearer tok1"}
        rec = _json.loads((gdir / "api" / "raw" / f"{d1}.json").read_text())
        assert rec["sleep"]["dailySleepDTO"]["deepSleepSeconds"] == 5000
        assert (gdir / "activities" / "garmin_123.fit").read_bytes() == fit
        assert any(f.name.startswith("weight_") for f in (gdir / "api" / "raw").iterdir())
        # the export-shaped files the wellness loader reads were rebuilt
        from goodpush.io.garmin import load_wellness
        daily, _ = load_wellness(gdir, progress=lambda *a: None, cache_dir=None, scan_zips=False)
        row = daily.loc[pd.Timestamp(d1)]
        assert row["rhr"] == 44 and abs(row["sleep_h"] - 7.5) < 0.01 and row["hrv"] == 70
        # second sync: cached token, activity file not downloaded again
        r2 = G.sync(gdir, since=d1, progress=lambda *a: None)
        assert seen["token"] == 1 and seen["files"] == 1 and r2["activities"] == 1
        G.forget_key()
        assert not G.has_key()
    finally:
        srv.shutdown()


def test_garmin_sync_skips_what_garmin_drops_and_fetches_it_next_time(tmp_path, monkeypatch):
    """A service that drops the connection: skipped (not asked again that run), the rest still comes in,
    and its days are fetched on the next sync."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import pandas as pd
    from goodpush.sync import garmin_api as G

    monkeypatch.setenv("GOODPUSH_SECRETS", str(tmp_path / "secrets"))
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    today = pd.Timestamp.now().normalize()
    d1 = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    state = {"down": True, "summary": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, raw):
            self.send_response(code)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            self._send(200, b"tok")

        def do_GET(self):
            p, _, q = self.path.partition("?")
            if p == "/usersummary-service/usersummary/daily":
                state["summary"].append(q[-10:])
                if state["down"]:
                    self.close_connection = True            # "Remote end closed connection without response"
                    return
                return self._send(200, _json.dumps({"calendarDate": q[-10:], "restingHeartRate": 44}).encode())
            if p == "/sleep-service/sleep/dailySleep":
                return self._send(200, _json.dumps({"calendarDate": q[-10:], "sleepTimeSeconds": 27000}).encode())
            self._send(404, b"{}")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setattr(G, "BASE", base)
    monkeypatch.setattr(G, "TOKEN_URL", base + "/token")
    try:
        G.save_key("ApiKey test-key-0123456789abcdef")
        gdir, msgs = tmp_path / "garmin", []
        r = G.sync(gdir, since=d1, progress=msgs.append)
        assert r["days"] == 2 and r["skipped"] == 2 and r["retry"] == [d1, today.strftime("%Y-%m-%d")]
        assert len(state["summary"]) == 4                      # retried, then left alone for the rest of the run
        assert any("didn't answer 2 request(s) (usersummary-service)" in m for m in msgs)
        state.update(down=False, summary=[])
        r = G.sync(gdir, progress=lambda *a: None)              # next update: from the incomplete days again
        assert d1 in state["summary"] and r["skipped"] == 0 and r["retry"] == []
    finally:
        srv.shutdown()


def test_streamstore_pack_read_replace_prune(tmp_path):
    pytest.importorskip("h5py")
    from goodpush.io import streamstore as SS
    sd = tmp_path / "streams"
    sd.mkdir()
    rng = np.random.default_rng(1)

    def ride(aid, n, seed):
        r = np.random.default_rng(seed)
        np.savez_compressed(sd / f"{aid}.npz", start=np.array([1.7e9 + seed]),
                            power=r.uniform(0, 400, n).astype(np.float32),
                            heart_rate=np.where(r.random(n) < 0.05, np.nan, r.uniform(90, 180, n)).astype(np.float32),
                            cadence=r.uniform(0, 100, n).astype(np.float32), speed=r.uniform(0, 12, n).astype(np.float32),
                            altitude=r.uniform(0, 900, n).astype(np.float32), moving=r.random(n) > 0.1)
    ride("a", 500, 1); ride("b", 300, 2); ride("c", 700, 3)
    np.savez_compressed(sd / "b.a1.npz", t=np.arange(0, 600, 5, dtype=np.int32),
                        a1=rng.uniform(0.3, 1.5, 120).astype(np.float32), p=rng.uniform(100, 300, 120).astype(np.float32),
                        hr=rng.uniform(100, 160, 120).astype(np.float32), art=np.zeros(120, np.float32))

    def same(d, path):
        z = np.load(path)
        return all(np.array_equal(np.asarray(z[c], dt), np.asarray(d[c], dt), equal_nan=dt.startswith("f"))
                   for c, dt in SS.GROUPS["ride"]["channels"].items())

    assert SS.pack(tmp_path, progress=lambda *a: None) == {"ride": 3, "alpha": 1}
    assert SS.pack(tmp_path, progress=lambda *a: None) == {}                 # nothing new
    assert SS.ids(tmp_path, "ride") == {"a", "b", "c"}
    for a in "abc":
        d = SS.read(tmp_path, a)
        assert same(d, sd / f"{a}.npz") and d["moving"].dtype == bool and d["start"] > 1.7e9
    al = SS.read(tmp_path, "b", "alpha")
    assert al["t"].dtype == np.int32 and al["t"].size == 120
    keys, n, flat = SS.read_all(tmp_path, "ride", ["power"])
    assert keys == ["a", "b", "c"] and list(n) == [500, 300, 700] and flat["power"].size == 1500

    # a re-parsed ride: the staged file wins until pack() re-points the index
    import time as _t
    _t.sleep(0.01)
    ride("b", 400, 9)
    assert same(SS.read(tmp_path, "b"), sd / "b.npz")
    assert SS.pack(tmp_path, progress=lambda *a: None) == {"ride": 1}
    assert same(SS.read(tmp_path, "b"), sd / "b.npz")
    keys, n, flat = SS.read_all(tmp_path, "ride", ["power"])
    assert sorted(keys) == ["a", "b", "c"] and n.sum() == 1600 and flat["power"].size == 1600
    with SS._open(SS.h5_path(tmp_path)) as f:
        assert SS._dead_fraction(f) > 0
    SS.pack(tmp_path, progress=lambda *a: None, compact=True)
    with SS._open(SS.h5_path(tmp_path)) as f:
        assert SS._dead_fraction(f) == 0
    assert same(SS.read(tmp_path, "b"), sd / "b.npz") and same(SS.read(tmp_path, "c"), sd / "c.npz")

    # prune: the .npz files go, reads keep working from the HDF5 file
    want = {a: SS.read(tmp_path, a) for a in "abc"}
    assert SS.prune(tmp_path, progress=lambda *a: None) == 4
    assert not list(sd.glob("*.npz"))
    got = SS.read_many(tmp_path, ["a", "b", "c", "zz"])
    assert set(got) == {"a", "b", "c"}
    for a in "abc":
        assert np.array_equal(got[a]["heart_rate"], want[a]["heart_rate"], equal_nan=True)

    # deleted rides: out of the index (staged copies deleted); the file is only rewritten once the dead
    # samples pass COMPACT_AT (of 3000 + 1600 ride + 120 alpha samples)
    ride("e", 3000, 5)
    SS.pack(tmp_path, progress=lambda *a: None)
    want["e"] = SS.read(tmp_path, "e")
    ride("d", 200, 4)
    size = SS.h5_path(tmp_path).stat().st_size
    assert SS.remove(tmp_path, ["a", "b", "d", "zz"], progress=lambda *a: None) == 3
    assert SS.ids(tmp_path, "ride") == {"c", "e"} and SS.ids(tmp_path, "alpha") == set()
    assert [x.name for x in sd.glob("*.npz")] == ["e.npz"] and SS.read(tmp_path, "b") is None
    with SS._open(SS.h5_path(tmp_path)) as f:
        assert SS._dead_fraction(f) == pytest.approx((500 + 400 + 120) / 4720)     # 22 %: left as it is
    assert SS.h5_path(tmp_path).stat().st_size >= size
    for a in "ce":
        assert np.array_equal(SS.read(tmp_path, a)["power"], want[a]["power"])
    assert SS.remove(tmp_path, ["c"], progress=lambda *a: None) == 1            # 37 %: compacted
    with SS._open(SS.h5_path(tmp_path)) as f:
        assert SS._dead_fraction(f) == 0
    assert SS.h5_path(tmp_path).stat().st_size < size and SS.ids(tmp_path, "ride") == {"e"}
    assert np.array_equal(SS.read(tmp_path, "e")["power"], want["e"]["power"])


def test_stale_physio_forecast_hidden_and_replanned(tmp_path: Path, monkeypatch):
    import os
    from types import SimpleNamespace
    from goodpush.app import server
    days = pd.date_range("2026-10-29", periods=3, name="date")
    st = pd.DataFrame({c: 200.0 for c in ("cp", "lt1", "map", "ftp", "fatigue", "form")}, index=days)
    fc = pd.DataFrame({c: 190.0 for c in ("cp", "lt1", "map", "ftp", "fatigue", "fatigue_cap")},
                      index=pd.date_range("2026-11-01", periods=2, name="date"))
    st.to_parquet(tmp_path / "physio_states.parquet")
    fc.to_parquet(tmp_path / "physio_forecast.parquet")
    os.utime(tmp_path / "physio_states.parquet", (1e9, 1e9))                 # the plan came after the fit
    assert not server.forecast_stale(tmp_path) and "forecast" in server._physio(tmp_path)
    os.utime(tmp_path / "physio_states.parquet", (2e9, 2e9))                 # refitted since: another fit's forecast
    assert server.forecast_stale(tmp_path) and "forecast" not in server._physio(tmp_path)
    assert server._physio(tmp_path)["assumed"] == []                         # no diagnostics: nothing said
    from goodpush.models.physio import PhysioRun                               # Today's reminder: the full run's command,
    phys = server._physio(tmp_path)                                            # shown while there are no ranges
    assert phys["full_run"] == PhysioRun.FULL_RUN and "cp_p05" not in phys
    from goodpush.physio import CHAINS                                         # ... and the cores it keeps busy
    assert phys["full_run_cores"]["chains"] == CHAINS <= phys["full_run_cores"]["all"]
    (tmp_path / "physio_diagnostics.json").write_text('{"assumed": [{"name": "log_s_f"}]}', encoding="utf-8")
    assert server._physio(tmp_path)["assumed"] == [{"name": "log_s_f"}]      # the Rider page's fatigue hint
    proj = SimpleNamespace(processed=tmp_path, cfg={"plan": {"enabled": True}})
    monkeypatch.setattr(server.Planner, "__init__", lambda self, p: None)
    monkeypatch.setattr(server.Planner, "build", lambda self, **k: None)
    calls = []
    server.prewarm_plan(proj, lambda: calls.append(1))
    assert calls == [1]
    os.utime(tmp_path / "physio_states.parquet", (1e9, 1e9))
    server.prewarm_plan(proj, lambda: calls.append(1))
    assert calls == [1]                                                      # forecast current: no re-plan


def test_calendar_data(tmp_path: Path):
    import json
    from goodpush.app.server import calendar_data
    from goodpush.config import init_project
    proj = init_project(tmp_path)
    now = pd.Timestamp.now(tz=proj.tz).tz_localize(None).normalize()
    days = pd.date_range(now - pd.Timedelta(days=9), now, freq="D")
    daily = pd.DataFrame({"tss": np.arange(len(days), dtype=float)}, index=days)
    acts = pd.DataFrame({"id": [1.0, 2.0], "date": [days[0], days[-2]], "name": ["a", "b"], "sport": "ride",
                         "workout": ["endurance", np.nan], "duration_s": [3600.0, np.nan], "tss": [50.0, np.nan],
                         "indoor": [True, None]})
    fut = str((now + pd.Timedelta(days=3)).date())
    (proj.reports / "plan").mkdir(parents=True, exist_ok=True)
    (proj.processed / "plan_state.json").write_text(json.dumps(
        {"planned": {str(days[1].date()): {"type": "rest", "title": "Rest day", "status": "n/a"},
                     str(days[2].date()): {"type": "vo2max", "title": "VO2", "tss": 80, "status": "moved"}},
         "away": {fut: "work trip"}}))
    (proj.reports / "plan" / "plan.json").write_text(json.dumps(
        {"sessions": [{"date": fut, "type": "threshold", "title": "Threshold 3x10", "minutes": 80, "tss": 70}],
         "ramp_range": {"lo": 1.5, "hi": 3.0, "lo_learned": True, "hi_learned": True},
         "weeks": [{"week": "2026-10-19", "tss_target": 480, "hours_limited": True, "ramp_flag": "fast", "ramp_4wk": 3.4},
                   {"week": "2026-10-26", "recovery": True},
                   {"week": "2026-11-02", "recovery": True, "taper": True}]}))
    (proj.root / "data" / "plans.json").write_text(json.dumps(
        {"phases": [{"type": "base", "start": "2026-10-26", "weeks": 2}], "events": []}))
    c = calendar_data(proj, daily, acts)
    shown = [{k: p[k] for k in ("name", "type", "start", "end")} for p in c["phases"]]   # (+ the season editor's keys)
    assert shown == [{"name": "Base", "type": "base", "start": "2026-10-26", "end": "2026-11-08"}]
    assert c["week_kind"] == {"2026-10-26": "recovery", "2026-11-02": "taper"}
    assert c["hours_limited"] == ["2026-10-19"]
    assert c["ramp_flags"] == {"2026-10-19": ["fast", 3.4]} and c["ramp_range"]["hi"] == 3.0
    assert c["away"] == {fut: "work trip"}                                     # days you can't ride
    start = pd.Timestamp(c["start"])
    assert start.day_of_week == 0 and len(c["tss"]) % 7 == 0 and len(c["plan_tss"]) == len(c["tss"])
    assert start + pd.Timedelta(days=len(c["tss"]) - 1) >= pd.Timestamp(fut)
    i0 = (days[0] - start).days
    assert c["tss"][i0:i0 + len(days)] == list(range(len(days)))
    assert all(v is None for v in c["tss"][i0 + len(days):])          # no actual TSS after today
    assert c["plan_tss"][(pd.Timestamp(fut) - start).days] == 70
    assert c["planned"][str(days[1].date())]["tss"] == 0 and c["planned"][fut]["title"] == "Threshold 3x10"
    assert str(days[2].date()) not in c["planned"] and not c["plan_tss"][(days[2] - start).days]   # moved away: not shown
    assert [a["id"] for a in c["acts"]] == ["1", "2"]
    assert c["acts"][1]["tss"] is None and c["acts"][1]["min"] is None and c["acts"][1]["workout"] == ""
    assert c["weeks"] == {"2026-10-19": 480, "2026-10-26": None, "2026-11-02": None} and c["today"] == str(now.date())
    json.dumps(c, allow_nan=False)


def test_calendar_covers_season(tmp_path: Path):
    """A new rider: no rides, no plan, a season block ahead: the calendar runs to the block's end."""
    import json
    from goodpush.app.server import calendar_data
    from goodpush.config import init_project
    proj = init_project(tmp_path)
    now = pd.Timestamp.now(tz=proj.tz).tz_localize(None).normalize()
    start = now + pd.Timedelta(days=14 - now.day_of_week)                # a Monday two weeks out
    (proj.root / "data" / "plans.json").write_text(json.dumps(
        {"phases": [{"type": "base", "start": str(start.date()), "weeks": 6}], "events": []}))
    c = calendar_data(proj, None, None)
    first = pd.Timestamp(c["start"])
    assert first + pd.Timedelta(days=len(c["tss"]) - 1) == start + pd.Timedelta(weeks=6, days=-1)
    assert len(c["plan_tss"]) == len(c["tss"])


def test_manual_ftp_keeps_history():
    from types import SimpleNamespace
    from goodpush.analyze import build_ftp
    from goodpush.metrics.power import MMP_DURATIONS
    days = pd.date_range("2026-01-01", "2026-03-31")
    acts = pd.DataFrame({"date": days[::3], "sport": "ride", "has_power": True,
                         **{f"mmp_{d}": 250.0 * (60 / max(d, 60)) ** 0.07 for d in MMP_DURATIONS}})
    proj = SimpleNamespace(cfg={"ftp": {"mode": "manual", "history": {"2026-03-01": 250}, "window_days": 90}})
    out, _ = build_ftp(proj, acts, days)
    assert (out.loc["2026-03-01":, "ftp"] == 250).all()
    before = out.loc[:"2026-02-28"]
    assert np.allclose(before["ftp"], before["ftp_auto"]) and not (before["ftp"] == 250).any()


def test_sustained_crossing_ignores_a_single_dip():
    from goodpush.alpha import _sustained_crossing
    m = np.array([[1, 1, .9, .4, .8, .6, .3, .2],          # dip at bin 3 must not count
                  [1, .9, .8, .7, .7, .6, .6, .55]])        # never drops below 0.5
    w = _sustained_crossing(m, np.ones_like(m, bool), 0.5)
    assert abs(w[0] - (10 * (5 + 0.1 / 0.3) + 5)) < 1e-6 and np.isnan(w[1])
    w2 = _sustained_crossing(m, np.ones_like(m, bool), 0.5, after_w=np.array([200.0, 0.0]))
    assert np.isnan(w2).all()                                # nothing above 200 W


def _ramp_ride(kind):
    """5-s bins of a synthetic ride: a 2-min-step ramp to failure, or a warm-up into a 20-min interval."""
    if kind == "ramp":
        p1 = np.r_[np.full(600, 110.0), np.repeat(100 + 12.0 * np.arange(18), 120), np.full(300, 80.0)]
    else:
        p1 = np.r_[np.linspace(100, 200, 600), np.full(1200, 230.0), np.full(600, 120.0)]
    t = np.arange(0, p1.size, 5)
    return t, p1.reshape(-1, 5).mean(1)


def test_ramp_test_detection_and_thresholds():
    from goodpush import alpha as AL
    parts = [_ramp_ride("ramp"), _ramp_ride("interval")]
    t = np.concatenate([x[0] for x in parts])
    p = np.concatenate([x[1] for x in parts])
    rid = np.repeat(np.arange(2), [x[0].size for x in parts])
    R = {"n": 2, "rid": rid, "t": t, "p": p, "ftp": np.array([200.0, 200.0]), "art": np.zeros(t.size),
         "a1": np.zeros(t.size), "hr": 100 + 0.3 * p}
    p2, _ = AL._steady(R)
    # alpha1 follows the 2-min power it is computed over: 0.75 at 214 W, 0.5 at 250 W
    R["a1"] = np.clip(1.2 - (np.nan_to_num(p2, nan=100.0) - 150) * 0.007, 0.3, 1.4)
    out = AL._ramps(R, p2)
    assert np.isfinite(out["ramp_top_w"][0]) and np.isnan(out["ramp_top_w"][1])   # interval isn't a ramp
    assert out["ramp_min"][0] >= 30 and out["ramp_top_w"][0] > 280
    assert abs(out["ramp_lt1_w"][0] - 214) < 10 and abs(out["ramp_lt2_w"][0] - 250) < 10
    assert out["ramp_lt1_hr"][0] < out["ramp_lt2_hr"][0] <= out["ramp_max_hr"][0]
    # 2026-11-09: FTP for the day rose after the test (the model's): the ramp no longer reached 110 % of it and its
    # thresholds were dropped. A planned ramp session you rode is a test whatever FTP says
    R["ftp"] = np.array([300.0, 300.0])
    assert np.isnan(AL._ramps(R, p2)["ramp_lt1_w"][0])                              # unplanned: FTP's gates
    R["planned"] = np.array([True, True])
    planned = AL._ramps(R, p2)
    assert planned["ramp_lt1_w"][0] == out["ramp_lt1_w"][0] and planned["ramp_lt2_w"][0] == out["ramp_lt2_w"][0]
    assert np.isnan(planned["ramp_top_w"][1])                                        # an interval still isn't one


def test_threshold_summary_prefers_ramp():
    from goodpush.models.thresholds import summary
    days = pd.date_range("2026-08-01", "2026-10-22")
    daily = pd.DataFrame({"ftp": 250.0, "ftp_auto": 220.0, "cp": 210.0, "wprime": 22000.0, "cp42": 215.0,
                          "wprime42": 21000.0, "hrvt1_28d": 200.0, "hrvt1_hr_28d": 150.0,
                          "hrvt2_42d": 255.0, "hrvt2_hr_42d": 170.0}, index=days)
    acts = pd.DataFrame({"date": [pd.Timestamp("2026-10-10")], "id": ["9"], "name": ["Ramp"],
                         "ramp_lt1_w": [210.0], "ramp_lt1_hr": [148.0], "ramp_lt2_w": [255.0],
                         "ramp_lt2_hr": [168.0], "ramp_top_w": [320.0]})
    bests = {"15": {"w": 700}, "30": {"w": 560}, "60": {"w": 400}, "1200": {"w": 230}}
    cfg = {"mode": "manual", "history": {"2026-10-22": 250}}
    s = summary(daily, acts, bests, bests, cfg, days[-1])
    assert s["ftp"] == {"w": 250, "source": "your setting", "since": "2026-10-22", "model_w": 220}
    assert s["lt1"]["source"] == "ramp test" and s["lt1"]["w"] == 210 and s["lt2"]["hr"] == 168
    assert s["glycolytic"]["p30_cp"] == round(560 / 210, 2) and s["cp"]["wprime_90d_kj"] == 22.0
    s2 = summary(daily, acts.iloc[0:0], bests, bests, {"mode": "auto"}, days[-1])
    assert s2["lt2"]["w"] == 255 and "steady riding" in s2["lt2"]["source"] and s2["ftp"]["source"].startswith("model")


def test_garmin_ftp_from_fit_and_in_thresholds(tmp_path):
    from goodpush.io.garmin import load_wellness, parse_fit_wellness
    from goodpush.models.thresholds import summary
    recs = bytearray()                                  # file_id.time_created and zones_target FTP 262
    recs += bytes([0x40, 0, 0]) + struct.pack("<H", 0) + bytes([1, 4, 4, 0x86])
    recs += bytes([0x00]) + struct.pack("<I", 1_160_000_000)
    recs += bytes([0x41, 0, 0]) + struct.pack("<H", 7) + bytes([1, 3, 2, 0x84])
    recs += bytes([0x01]) + struct.pack("<H", 262)
    data = struct.pack("<BBHI4sH", 14, 0x20, 2100, len(recs), b".FIT", 0) + bytes(recs) + b"\x00\x00"
    day = pd.Timestamp(1_160_000_000 + fit.FIT_EPOCH_OFFSET, unit="s").strftime("%Y-%m-%d")
    assert list(parse_fit_wellness(data, "UTC")) == [{"date": day, "ftp_garmin": 262}]
    # a synced ride (garmin/activities): read up to its first record only - a later zones_target is ignored
    recs += bytes([0x42, 0, 0]) + struct.pack("<H", 20) + bytes([1, 253, 4, 0x86])
    recs += bytes([0x02]) + struct.pack("<I", 1_160_000_100) + bytes([0x01]) + struct.pack("<H", 999)
    ride = struct.pack("<BBHI4sH", 14, 0x20, 2100, len(recs), b".FIT", 0) + bytes(recs) + b"\x00\x00"
    (tmp_path / "activities").mkdir()
    (tmp_path / "activities" / "garmin_1.fit").write_bytes(ride)
    w, _ = load_wellness(tmp_path, "UTC", progress=lambda *a: None, scan_zips=False)
    assert w.loc[day, "ftp_garmin"] == 262
    days = pd.date_range("2026-04-01", "2026-10-22")
    daily = pd.DataFrame({"ftp": 250.0, "ftp_garmin": np.nan}, index=days)
    daily.loc["2026-04-10", "ftp_garmin"], daily.loc["2026-05-15":"2026-10-20", "ftp_garmin"] = 250.0, 262.0
    g = summary(daily, pd.DataFrame({"date": []}), {}, {}, {"mode": "auto"}, days[-1])["garmin"]
    assert g["w"] == 262 and g["since"] == "2026-05-15" and g["date"] == "2026-10-20"


def test_ride_timeline_cuts_long_stops_like_the_streams():
    from goodpush.metrics.power import ride_seconds, to_1hz
    t = np.r_[np.arange(0, 600), np.arange(1200, 1800), np.arange(1860, 2400)] + 1_700_000_000.0   # 600-s stop, 60-s stop
    sec = ride_seconds(t)
    assert sec[599] == 599 and sec[600] == 600 and sec[1199] == 1199 and sec[1200] == 1260   # long cut, short kept
    s = to_1hz({"time": list(t), "power": [200] * t.size, "heart_rate": [140] * t.size})
    assert s["n"] == sec[-1] + 1


def test_ride_selection_stats_and_zoom_streams(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from goodpush.app.server import activity_window, ride_stops, ride_streams
    from goodpush.io import streamstore
    n = 1800                                            # 10 min at 200 W, 5 min at 300 W, 15 min at 150 W
    z = {"power": np.r_[np.full(600, 200.0), np.full(300, 300.0), np.full(900, 150.0)],
         "heart_rate": np.full(n, 150.0), "cadence": np.full(n, 90.0), "speed": np.full(n, 10.0),
         "altitude": np.arange(n) * 0.1, "moving": np.ones(n, bool)}
    t = np.arange(0, n, 5)                              # alpha1 every 5 s: 1.0 for 10 min, then 0.6
    za = {"t": t, "a1": np.where(t < 600, 1.0, 0.6), "p": np.zeros(t.size), "hr": np.zeros(t.size)}
    monkeypatch.setattr(streamstore, "read", lambda proc, aid, group="ride": {"ride": z, "alpha": za}[group] if aid == "7" else None)
    acts = pd.DataFrame({"id": [7.0], "ftp": [250.0], "date": [pd.Timestamp("2026-11-02")]})
    w = activity_window(SimpleNamespace(processed=tmp_path), acts, "7", 600, 900)   # (no physio fit: no stamina)
    s = w["stats"]
    assert (s["seconds"], s["avg_power"], s["np"], s["if"], s["tss"]) == (300, 300, 300, 1.2, 12)
    assert s["avg_hr"] == 150 and s["distance_m"] == 3000 and s["speed_kmh"] == 36.0 and s["climb_m"] == 29
    assert (s["alpha1"], s["alpha1_min"], s["alpha1_max"]) == (0.6, 0.6, 0.6)
    assert w["streams"]["bucket_s"] == 1 and len(w["streams"]["t_min"]) == 300
    assert w["streams"]["t_min"][0] == pytest.approx(10 + 0.5 / 60, abs=1e-4) and w["streams"]["power"][0] == 300
    full = ride_streams(z, 0, n, points=600)            # whole ride: 3-s buckets
    assert full["bucket_s"] == 3 and len(full["power"]) == 600 and full["power"][250] == 300
    stopped = np.zeros(n, bool)
    stopped[300:420] = True                              # 2 min stopped: left out, the clock skips it
    cut = ride_streams(z, 0, 900, points=900, stopped=stopped)
    assert len(cut["power"]) == 780 and cut["t_min"][300] == pytest.approx(300.5 / 60, abs=1e-4) and cut["power"][-1] == 300
    z2 = {**z, "power": np.r_[np.zeros(10), z["power"][10:]], "speed": np.r_[np.zeros(5), np.full(n - 5, 10.0)]}
    assert ride_stops(z2).sum() == 5                     # no power but rolling: not a stop
    with pytest.raises(ValueError):
        activity_window(SimpleNamespace(processed=None), acts, "7", 100, 102)
    with pytest.raises(KeyError):
        activity_window(SimpleNamespace(processed=None), acts, "8", 0, 60)


def test_progress_bar_terminal_and_app_job(capsys):
    import threading
    from goodpush.app.server import Jobs
    from goodpush.progress import Progress, TerminalBar
    bar = TerminalBar()
    p = Progress(bar.message, bar.draw)
    Progress.step(p, "Reading ride files", 30, 120)     # (tqdm itself draws nothing here: not a terminal)
    first = bar.bar
    Progress.step(p, "Reading ride files", 90, 120)
    assert bar.label == "Reading ride files" and bar.bar is first     # same stage, same bar
    p("parsed 100/120")                                 # log lines go through tqdm.write
    assert "parsed 100/120" in capsys.readouterr().out
    Progress.step(p, "Reading ride files", 120, 120)    # finished: the bar closes
    assert bar.bar is None
    Progress.step(p, "Analysing", 1, 5)                 # a new stage gets its own bar
    assert bar.label == "Analysing"
    bar.close()
    Progress.step(print, "ignored", 1, 2)               # a plain callback: no bar, no error
    jobs, seen, go = Jobs(), {}, threading.Event()

    def job(log):
        log("starting")
        Progress.step(log, "Overnight HRV files", 3, 12)
        seen.update(jobs.status())
        go.set()
    assert jobs.start("test", job) and go.wait(5)
    assert seen["step"] == {"label": "Overnight HRV files", "done": 3, "total": 12} and seen["log"] == ["starting"]


def test_jobs_queue_a_replan_behind_a_running_job():
    import threading
    import time as _t
    from goodpush.app.server import Jobs
    jobs, hold, ran, seen = Jobs(), threading.Event(), [], []
    assert jobs.start("Garmin update", lambda log: hold.wait(5)) is True
    assert jobs.start("Other", lambda log: None) is False                      # not queued: refused
    assert jobs.start("Updating the plan", lambda log: ran.append(1), queue=True) == "queued"
    st = jobs.status()
    assert st["running"] and st["name"] == "Garmin update" and st["queued"] == "Updating the plan"
    hold.set()
    for _ in range(100):                                                        # never idle in between
        s = jobs.status()
        seen.append((s["running"], s["name"]))
        if ran and not s["running"]:
            break
        _t.sleep(0.02)
    assert ran == [1] and jobs.status()["queued"] is None and jobs.status()["name"] == "Updating the plan"
    assert all(r for r, n in seen if n == "Garmin update")                       # running until handed over


def test_toml_date_keys(tmp_path: Path):
    from goodpush.app.tomledit import set_values
    p = tmp_path / "goodpush.toml"
    p.write_text('[ftp]\nmode = "auto"\n\n[ftp.history]\n# "2024-01-15" = 265\n"2026-10-22" = 250\n')
    set_values(p, "ftp.history", {"2026-10-30": 243})
    set_values(p, "ftp.history", {"2026-10-22": 240})
    import tomllib
    assert tomllib.loads(p.read_text())["ftp"]["history"] == {"2026-10-22": 240, "2026-10-30": 243}




def post_settings(url: str, values: dict) -> dict:
    """POST a Settings > Plan save to a running app; its JSON answer (or error)."""
    import json
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, json.dumps({"action": "settings", "values": values}).encode(),
                                 {"Content-Type": "application/json", "X-GoodPush": "1"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def test_app_plan_settings_intensity_and_load(tmp_path: Path):
    import threading
    import tomllib
    from http.server import ThreadingHTTPServer
    from goodpush.app.server import make_handler
    from goodpush.config import init_project, load_project
    init_project(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(load_project(tmp_path), "k" * 20, lan=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/action"
    try:
        assert "error" in post_settings(url, {"intensity": "threshold-heavy"})
        assert "error" in post_settings(url, {"load_model": "guesswork"})
        post_settings(url, {"intensity": "polarized", "load_model": "physio"})
        plan = tomllib.loads((tmp_path / "goodpush.toml").read_text(encoding="utf-8"))["plan"]
        assert (plan["intensity"], plan["load_model"]) == ("polarized", "physio")
    finally:
        srv.shutdown()


def test_app_removes_several_days_away_at_once(tmp_path: Path):
    """2026-11-05: days away picked on the calendar go in one action, one re-plan."""
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from goodpush.app.server import make_handler
    from goodpush.config import init_project, load_project
    from goodpush.planning.state import PlanState
    init_project(tmp_path)
    proj = load_project(tmp_path)
    sp = proj.processed / "plan_state.json"
    with PlanState.open(sp) as st:
        st.set_away("2099-11-12", "2099-11-15", "work trip")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proj, "k" * 20, lan=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/action"
    try:
        req = urllib.request.Request(url, json.dumps({"action": "away", "days": ["2099-11-12", "2099-11-14"]}).encode(),
                                     {"Content-Type": "application/json", "X-GoodPush": "1"})
        urllib.request.urlopen(req).read()
        with PlanState.open(sp) as st:
            assert st["away"] == {"2099-11-13": "work trip", "2099-11-15": "work trip"}
    finally:
        srv.shutdown()


def test_app_asks_before_a_season_change_replans_your_own_days(tmp_path: Path):
    """2026-11-15: sessions moved by hand stayed pinned when the season moved under them (odd weeks). Locking a
    season change asks first when your own days are in the weeks that changed; yes = the planner takes them back."""
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from goodpush.app.server import make_handler
    from goodpush.config import init_project, load_project
    from goodpush.planning import Phase, Season
    from goodpush.planning.state import PlanState
    init_project(tmp_path)
    proj = load_project(tmp_path)
    td = pd.Timestamp.now().normalize()
    monday = td + pd.Timedelta(days=7 - td.weekday())
    base = Phase("base", str(monday.date()), 4, build_weeks=0)
    Season([base]).save(proj.root)
    kept, gone = (str((monday + pd.Timedelta(days=d)).date()) for d in (8, 30))    # week 2 / week 5 (new)
    sp = proj.processed / "plan_state.json"
    with PlanState.open(sp) as st:
        st["manual_plan"] = {kept: {"type": "rest"}, gone: {"type": "rest"}, gone + "#2": {"type": "endurance"}}
        st["planned"][gone + "#2"] = {"type": "endurance"}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proj, "k" * 20, lan=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/action"
    body = {"action": "season_blocks", "start": str(monday.date()),
            "blocks": [{"id": base.id, "type": "base", "weeks": 4}, {"type": "build", "weeks": 2}]}
    post = lambda b: json.loads(urllib.request.urlopen(urllib.request.Request(  # noqa: E731
        url, json.dumps(b).encode(), {"Content-Type": "application/json", "X-GoodPush": "1"})).read())
    try:
        assert post({"action": "lock_session", "key": kept, "locked": True}) == {"ok": True}   # saved, no re-plan
        assert PlanState.load(sp)["manual_plan"][kept]["locked"]
        post({"action": "lock_session", "key": kept, "locked": False})
        j = post(body)
        assert j["answer"] == "replan_own" and "2 day(s)" in j["ask"]
        assert j["title"] and j["yes"] and j["no"]                         # the dialog's title and buttons
        assert len(Season.load(proj.root).phases) == 1                     # nothing saved before the answer
        post({**body, "replan_own": True})
        assert len(Season.load(proj.root).phases) == 2
        st = PlanState.load(sp)
        assert list(st["manual_plan"]) == [kept] and gone + "#2" not in st["planned"]
    finally:
        srv.shutdown()


def test_app_offers_a_new_rider_the_general_block(tmp_path: Path, monkeypatch):
    """No rides, no plan, planning off: the state offers the general block (before any plan exists); taking it
    adds those phases and switches planning on."""
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from goodpush.app import server
    from goodpush.config import init_project, load_project
    from goodpush.planning import Season
    init_project(tmp_path)
    proj = load_project(tmp_path)
    assert not proj.cfg["plan"]["enabled"]
    started = []
    monkeypatch.setattr(server.Jobs, "start", lambda self, name, fn, *a, **k: started.append(name) or "started")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(proj, "k" * 20, lan=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        sugg = json.loads(urllib.request.urlopen(base + "/api/state").read())["season"]["suggestion"]
        assert [(s["type"], s["weeks"], s["generic"]) for s in sugg] == [("base", 8, True), ("build", 4, True)]
        assert not started                                       # planning off: no plan made by itself
        urllib.request.urlopen(urllib.request.Request(base + "/api/action", json.dumps({"action": "accept_phases"}).encode(),
                                                      {"Content-Type": "application/json", "X-GoodPush": "1"})).read()
        assert [p.type for p in Season.load(tmp_path).phases] == ["base", "build"] and started
        assert load_project(tmp_path).cfg["plan"]["enabled"]
        assert "# make a plan at all" in (tmp_path / "goodpush.toml").read_text(encoding="utf-8")   # comments kept
    finally:
        srv.shutdown()
    other = load_project(init_project(tmp_path / "b").root)        # blocks dragged in on the Calendar: on too
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(other, "k" * 20, lan=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        data = json.dumps({"action": "season_blocks", "start": "2027-01-04", "blocks": [{"type": "base", "weeks": 6}]})
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/action",
                                                      data.encode(),
                                                      {"Content-Type": "application/json", "X-GoodPush": "1"})).read()
        assert load_project(other.root).cfg["plan"]["enabled"]
    finally:
        srv.shutdown()


def test_app_marks_a_period(tmp_path: Path, monkeypatch):
    """Fitness > Mark periods: the action stores the period (moved via `was`) and /api/dips lists it; nothing is
    learned until marking ends (periods_done); a found break's own days confirm that break instead of a copy."""
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from goodpush.app import server
    from goodpush.config import init_project, load_project
    init_project(tmp_path)
    proj = load_project(tmp_path)
    started = []
    monkeypatch.setattr(server.Jobs, "start", lambda self, name, fn, *a, **k: started.append(name) or "started")
    (proj.processed / "dips.json").write_text(json.dumps({"dips": [{"start": "2026-09-03", "end": "2026-09-14"}]}))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(proj, "k" * 20, lan=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    post = lambda body: urllib.request.urlopen(urllib.request.Request(base + "/api/action", json.dumps(body).encode(),  # noqa: E731
                                                                      {"Content-Type": "application/json", "X-GoodPush": "1"})).read()
    try:
        post({"action": "label_period", "start": "2026-03-02", "end": "2026-03-08", "value": "travel"})
        post({"action": "label_period", "start": "2026-03-01", "end": "2026-03-09", "value": "illness", "was": ["2026-03-02", "2026-03-08"]})
        got = json.loads(urllib.request.urlopen(base + "/api/dips").read())["periods"]
        assert got == [{"start": "2026-03-01", "end": "2026-03-09", "cause": "illness"}] and not started
        post({"action": "label_period", "start": "2026-09-03", "end": "2026-09-14", "value": "travel"})   # the found break
        from goodpush.models.dips import BreakLabels
        assert BreakLabels(proj.processed).load()["2026-09-03"] == "travel" and len(BreakLabels(proj.processed).periods()) == 1
        post({"action": "periods_done", "gain": False})
        assert started == ["Re-learning from your labels"]
    finally:
        srv.shutdown()


def test_day_rides_main_ride_per_day():
    from goodpush.app.server import _day_rides
    days = pd.date_range("2026-10-01", periods=4)
    acts = pd.DataFrame({"id": [11.0, 12.0, 13.0, 14.0], "sport": ["ride", "ride", "run", "ride"],
                         "date": [days[0], days[0], days[1], days[3]], "tss": [30.0, 80.0, 50.0, np.nan],
                         "workout": ["endurance", "vo2max", "", None], "name": ["Spin", "Hills", "Airport walk", ""],
                         "description": ["", "legs heavy", None, ""], "private_note": [None, "", "flight to Oslo", None]})
    out = _day_rides(acts, days)
    assert out["ride"] == ["12", None, None, "14"]                  # most TSS that day; runs don't count
    assert out["kind"] == ["vo2max", None, None, None]              # its workout type (none: not classified)
    assert out["title"] == ["Spin · Hills", "Airport walk", None, None]                  # every activity: the hover
    assert out["note"] == ["legs heavy", "flight to Oslo", None, None]


def test_ride_intervals_steps_sets_and_bursty_power():
    from goodpush.models.intervals import RideIntervals
    from goodpush.planning.intensity import IntensityMap
    rng = np.random.default_rng(0)
    knots = IntensityMap.ftp_knots(230.0)                        # no physio fit: textbook %FTP, CP = FTP
    on = lambda s, w: np.full(s, float(w))                       # noqa: E731
    reps = [np.r_[on(180, 250), on(60, 100)] for _ in range(5)]
    reps[2][80:85] = 0.0                                         # a dropout inside the third rep
    p = np.concatenate([on(600, 160), *reps, on(240, 110), np.tile(np.r_[on(120, 210), on(60, 235)], 3), on(600, 120)])
    p = p + rng.normal(0, 4, p.size)
    r = RideIntervals(p, knots, w_prime=20000.0)
    iv, sets = r.intervals(), r.sets()
    assert [round(x["seconds"] / 5) * 5 for x in iv] == [180] * 5 + [540]       # the dip doesn't split a rep
    assert [x["zone"] for x in iv[:5]] == ["vo2max"] * 2 + ["threshold"] + ["vo2max"] * 2   # 5 s at 0 W: 243 W
    assert [x["watts"] for x in iv[:5]] == pytest.approx([250, 250, 243, 250, 250], abs=2)
    # 20 W over CP for 3 min = 18 % of W' (no refill above CP)
    assert 17 <= iv[0]["w_prime_used"] <= 19
    assert iv[4]["w_prime_used"] > iv[0]["w_prime_used"]                             # 1-min rests: not all back
    # your thresholds instead: CP 260 W puts the same 250 W reps at threshold
    lt1, cp, mp = 190.0, 260.0, 320.0
    zones = [x["zone"] for x in RideIntervals(p, [0, lt1, cp, mp, 400, 900]).intervals()[:5]]
    assert zones == ["threshold"] * 2 + ["sweetspot"] + ["threshold"] * 2       # threshold from 246 W
    assert [s["count"] for s in sets] == [5, 1] and sets[0]["rest"] == pytest.approx(60, abs=3)
    assert sets[0]["rest_watts"] == pytest.approx(100, abs=8) and sets[0]["rest_zone"] == "recovery"
    sec = dict(zip(r.summary()["zones"], r.summary()["zone_seconds"]))
    assert sec["endurance_high"] == pytest.approx(600, abs=30) and sec["endurance_low"] < 60   # 160 W warm-up: upper
    assert sum(sec.values()) == p.size
    assert [(round(t / 10) * 10, round(w / 5) * 5) for t, w in iv[5]["steps"]] == [(120, 210), (60, 235)] * 3   # over-unders
    # outdoors: bursty power around an endurance pace, one 8-min climb at threshold
    q = np.clip(150 + rng.normal(0, 90, 3600), 0, None)
    q[1800:2280] += 80
    r = RideIntervals(q, knots)
    assert r.bursty
    iv = r.intervals()
    assert len(iv) == 1 and abs(iv[0]["start"] - 1800) < 40 and 420 < iv[0]["seconds"] < 540


def test_ride_intervals_micro_bursts_over_unders_and_dropouts():
    from scipy.signal import lfilter
    from goodpush.models.intervals import RideIntervals
    rng = np.random.default_rng(1)
    knots = np.array([0.0, 200.0, 235.0, 280.0, 350.0, 720.0])   # LT1, CP, MAP, ...: tempo from ~201 W
    on = lambda s, w: np.full(s, float(w))                       # noqa: E731
    lag = lambda x: lfilter([0.5], [1.0, -0.5], x - x[0]) + x[0]  # a trainer's ~3-s ramp at each step  # noqa: E731
    micro = np.tile(np.r_[on(30, 290), on(15, 95)], 12)
    p = np.concatenate([on(600, 150), *[np.r_[micro, on(240, 110)] for _ in range(3)], on(300, 120)])
    p = lag(p) + rng.normal(0, 5, p.size)
    r = RideIntervals(p, knots)
    assert r.roughness > 0.2 and not r.bursty                    # rough, but the steps explain it
    sets = r.sets()
    assert [s["count"] for s in sets] == [12, 12, 12]            # 3 x 12, not one set of 36 or a sweet-spot block
    assert all(28 <= s["seconds"] <= 32 and 13 <= s["rest"] <= 17 for s in sets)
    # over-unders: 1 min over / 2 min under just below tempo (195 W), 3 blocks; a 10-s dropout in a rep
    ou = np.tile(np.r_[on(60, 255), on(120, 195)], 3)
    rep = on(240, 250)
    rep[100:110] = 0.0
    q = np.concatenate([on(600, 140), *[np.r_[ou, on(240, 100)] for _ in range(3)], rep, on(300, 120)])
    q = lag(q) + rng.normal(0, 4, q.size)
    iv = RideIntervals(q, knots).efforts()
    assert [round(x["seconds"] / 10) * 10 for x in iv] == [540, 540, 540, 240]   # unders kept, the last one too
    assert [round(w / 5) * 5 for _, w in iv[0]["steps"]] == [255, 195] * 3
    assert RideIntervals(q, knots).sets()[0]["count"] == 3
    # lactate clearance: 45-s overs at ~1.22 CP, 135-s unders just below LT1 (~0.68 of the over), 3 x 3
    lc = np.tile(np.r_[on(45, 287), on(135, 200)], 3)
    q = np.concatenate([on(600, 140), *[np.r_[lc, on(240, 100)] for _ in range(3)], on(300, 120)])
    q = lag(q) + rng.normal(0, 4, q.size)
    iv = RideIntervals(q, knots).efforts()
    assert [round(x["seconds"] / 10) * 10 for x in iv] == [540, 540, 540]
    assert np.allclose([w for _, w in iv[0]["steps"]], [287, 200] * 3, atol=10)
    assert {x["zone"] for x in iv} == {"threshold"}              # averages sweet spot, but over CP 3 times
    assert RideIntervals(q, knots).sets()[0]["zone"] == "threshold"
    # one hard start then sweet spot: crosses CP once, keeps its average's zone
    hs = np.r_[on(600, 140), on(30, 300), on(570, 220), on(300, 120)]
    assert [x["zone"] for x in RideIntervals(lag(hs) + rng.normal(0, 4, hs.size), knots).efforts()] == ["sweetspot"]
    # 3 x 1 min at tempo spread over endurance: a set, though each is short for tempo; 20-s blips aren't
    surge = lambda s: np.concatenate([on(600, 150), *[np.r_[on(s, 210), on(540, 165)] for _ in range(3)]])  # noqa: E731
    iv = RideIntervals(lag(surge(60)) + rng.normal(0, 4, 600 + 3 * 600), knots).intervals()
    assert [(round(x["seconds"] / 10) * 10, x["zone"]) for x in iv] == [(60, "tempo")] * 3
    assert not RideIntervals(lag(surge(20)) + rng.normal(0, 4, 600 + 3 * 560), knots).intervals()
    # bursty power never takes unders: an effort there is tempo and up
    out = np.clip(150 + rng.normal(0, 90, 3600), 0, None)
    r = RideIntervals(out, knots)
    assert r.bursty and all(x["watts"] >= r.edges[3] for x in r.efforts())


def test_interval_log_rows_and_keys():
    from goodpush.models.intervals import IntervalLog, RideIntervals
    from goodpush.planning.intensity import IntensityMap
    on = lambda s, w: np.full(s, float(w))                       # noqa: E731
    p = np.concatenate([on(600, 150), *[np.r_[on(180, 250), on(60, 100)] for _ in range(4)], on(600, 120)])
    hr = np.concatenate([on(600, 120), *[np.r_[np.linspace(140, 165, 180), on(60, 130)] for _ in range(4)], on(600, 115)])
    rows = IntervalLog.rows(RideIntervals(p, IntensityMap.ftp_knots(230.0), hr))
    assert [(r["set"], r["rep"], r["reps"]) for r in rows] == [(0, i, 4) for i in range(4)] and rows[0]["rest_s"] == 60
    assert rows[0]["hr_late"] > 155 and rows[0]["hr_drift"] > 5 and rows[0]["cv"] < 0.01
    k = IntervalLog.keys(np.vstack([IntensityMap.ftp_knots(250.0), IntensityMap.ftp_knots(250.6), IntensityMap.ftp_knots(260.0)]))
    assert k[0] == k[1] != k[2]                                  # thresholds in 5 W steps
    old = pd.DataFrame({"id": ["a", "b", "c"], "key": [k[0], k[0], "v0" + k[0][2:]]})
    cp = IntensityMap.ftp_knots(250.0)
    knots = np.vstack([cp * 1.012, cp * 1.06, cp, cp])            # a refit's drift over a step, a real change
    assert list(IntervalLog.stale(np.array(["a", "b", "c", "d"]), knots, old)) == [1, 2, 3]   # old method, new ride


def test_interval_heart_rate_recovery(monkeypatch):
    from goodpush.io import streamstore
    from goodpush.models.intervals import IntervalLog, RideIntervals
    from goodpush.planning.intensity import IntensityMap
    on = lambda s, w: np.full(s, float(w))                       # noqa: E731
    p = np.concatenate([on(600, 150), *[np.r_[on(180, 250), on(60, 100)] for _ in range(4)], on(600, 120)])
    hr = np.concatenate([on(600, 120), *[np.r_[on(180, 165), on(60, 130)] for _ in range(4)], on(600, 115)])
    rows = IntervalLog.rows(RideIntervals(p, IntensityMap.ftp_knots(230.0), hr))
    assert [(r["hr_end"], r["hrr_60"], r["rec_watts"]) for r in rows] == [(165, 35, 100)] * 4
    # the next interval sooner than a minute, a missing reading, no heart rate at all
    rec = RideIntervals.recovery(p, np.where(np.arange(p.size) == 1158, np.nan, hr), [600, 840, 1060], [780, 1020, 1100])
    assert np.isnan(rec["hrr_60"]).tolist() == [False, True, True] and np.isnan(rec["rec_watts"]).tolist() == [False, True, False]
    assert np.isnan(RideIntervals.recovery(p, None, [600], [780])["hrr_60"]).all()
    # a table stored before: filled from the streams, intervals as they were
    old = pd.DataFrame({"id": ["a"] * 4 + ["b"], "start": [600, 840, 1080, 1320, -1], "seconds": [180] * 4 + [0]})
    monkeypatch.setattr(streamstore, "read_many", lambda pr, ids, g, ch: {"a": {"power": p, "heart_rate": hr}})
    t = IntervalLog.add_recovery(None, old)
    assert t["hrr_60"].tolist()[:4] == [35] * 4 and np.isnan(t["hrr_60"].iloc[4])


def test_interval_recovery_share():
    from goodpush.models.intervals import IntervalLog, RideIntervals
    from goodpush.planning.intensity import IntensityMap
    # steady HR at 100 W: 50 + (170 - 50) x 100 / 250 = 98; from 165, 33.5 bpm is half the way; 5 bpm above it: none
    s = RideIntervals.share([165.0, 103.0, np.nan], [33.5, 2.0, 30.0], [100.0, 100.0, 100.0], 250.0, 50.0, 170.0)
    assert s[0] == 0.5 and np.isnan(s[1:]).all()
    daily = pd.DataFrame({"rhr_base": [50.0, np.nan, 48.0]}, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]))
    assert IntervalLog.rest_hr(daily, ["2025-12-30", "2026-01-03", "2026-01-03", "2026-01-06"]).tolist() == [50, 50, 50, 48]
    assert np.isnan(IntervalLog.rest_hr(None, ["2026-01-01"])).all()
    t = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "watts": [300.0], "pct_cp": [120.0], "hr_end": [165.0],
                      "hrr_60": [33.5], "rec_watts": [100.0]})
    assert IntervalLog.with_share(t, daily, 170.0)["hrr_share"].tolist() == [0.5]
    # on the ride page, given resting HR and LTHR
    on = lambda s, w: np.full(s, float(w))                       # noqa: E731
    p = np.concatenate([on(600, 150), *[np.r_[on(180, 250), on(60, 100)] for _ in range(4)], on(600, 120)])
    hr = np.concatenate([on(600, 120), *[np.r_[on(180, 165), on(60, 130)] for _ in range(4)], on(600, 115)])
    r = RideIntervals(p, IntensityMap.ftp_knots(230.0), hr)
    want = round(35 / (165 - (50 + 120 * 100 / r.cp)), 3)
    assert [x["hrr_share"] for x in r.summary(50.0, 170.0)["intervals"]] == [want] * 4
    assert "hrr_share" not in r.summary()["intervals"][0]


def fit_with_file_id(kind: int, created_unix: int) -> bytes:
    """a minimal FIT file: just its file_id (type, time_created)."""
    from goodpush.io import fit
    recs = bytes([0x40, 0, 0]) + struct.pack("<HB", 0, 2) + bytes([0, 1, 0x00, 4, 4, 0x86])
    recs += bytes([0x00, kind]) + struct.pack("<I", created_unix - fit.FIT_EPOCH_OFFSET)
    return struct.pack("<BBHI4sH", 14, 0x20, 2100, len(recs), b".FIT", 0) + recs + b"\x00\x00"


def test_garmin_export_activities_matched_to_their_files(tmp_path):
    import json as _json
    import zipfile
    from types import SimpleNamespace
    from goodpush.app.server import save_upload
    from goodpush.sync import garmin as G
    g = tmp_path / "garmin"
    fit_dir, up = g / "DI_CONNECT" / "DI-Connect-Fitness", g / "DI_CONNECT" / "DI-Connect-Uploaded-Files"
    fit_dir.mkdir(parents=True)
    up.mkdir(parents=True)
    t0 = 1_760_000_000                                                    # 2025-10-09
    acts = [{"activityId": 11, "name": "Hill repeats", "activityType": "cycling", "startTimeGmt": t0 * 1000.0,
             "duration": 3_600_000.0, "elapsedDuration": 3_700_000.0, "distance": 3_000_000.0, "avgPower": 210.0},
            {"activityId": 12, "name": "Zwift", "activityType": "virtual_ride", "startTimeGmt": (t0 + 86400) * 1000.0,
             "duration": 1_800_000.0}]
    (fit_dir / "me_0_summarizedActivities.json").write_text(_json.dumps([{"summarizedActivitiesExport": acts}]))
    with zipfile.ZipFile(up / "UploadedFiles_0-_Part1.zip", "w") as z:
        z.writestr("me_1.fit", fit_with_file_id(4, t0 + 2))                  # the ride's file, created 2 s in
        z.writestr("me_2.fit", fit_with_file_id(32, t0 + 3))                 # a monitoring file: not an activity
    assert G.import_export_activities(g, progress=lambda *a: None) == 2
    t = pd.read_csv(g / "activities" / "activities_garmin.csv", dtype=str, keep_default_na=False).set_index("id")
    assert t.loc["garmin_11", "filename"] == "garmin_11.fit" and t.loc["garmin_12", "filename"] == ""
    assert float(t.loc["garmin_11", "elapsed_s"]) == 3700 and float(t.loc["garmin_11", "distance_m"]) == 30000   # ms, cm
    assert t.loc["garmin_11", "type"] == "Ride" and t.loc["garmin_11", "start_utc"].startswith("2025-10-09")
    assert G.import_export_activities(g, progress=lambda *a: None) == 0      # again: nothing to do
    # dropped on the page: an export is kept in data/uploads/ (not deleted), anything else refused
    import io
    proj = SimpleNamespace(processed=tmp_path / "data" / "processed")
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        z.writestr("DI_CONNECT/DI-Connect-Fitness/x.json", "[]")
    path, kind = save_upload(proj, io.BytesIO(zb.getvalue()), len(zb.getvalue()), "../my export.zip")
    assert kind == "garmin" and path == tmp_path / "data" / "uploads" / "my_export.zip" and path.exists()
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        z.writestr("a.jpg", b"x")
    with pytest.raises(ValueError, match="isn't a Strava or Garmin export"):
        save_upload(proj, io.BytesIO(zb.getvalue()), len(zb.getvalue()), "photos.zip")
    with pytest.raises(ValueError, match="stopped part way"):
        save_upload(proj, io.BytesIO(b"PK"), 1000, "cut.zip")
    assert sorted(p.name for p in (tmp_path / "data" / "uploads").iterdir()) == ["my_export.zip"]   # no leftovers


def test_settings_cover_every_toml_option_and_round_trip(tmp_path: Path):
    """Every option in the goodpush.toml template is on the Settings page (SETTINGS, or the Incyclist card),
    and saving each section's values unchanged leaves the same settings."""
    import tomllib
    from goodpush.app.server import _save_settings, _settings_values
    from goodpush.config import SETTINGS, clean_settings, init_project, load_project, template_text
    template = tomllib.loads(template_text())
    on_page = {(sec, f["key"]) for sec, spec in SETTINGS.items() for f in spec["fields"]}
    for sec, table in template.items():
        if sec == "events":
            assert ("events", "events") in on_page
            continue
        for key in table:
            assert (sec, key) in on_page or sec == "incyclist", (sec, key)
    proj = init_project(tmp_path)
    before = load_project(tmp_path).cfg
    values = _settings_values(proj)
    for sec in SETTINGS:
        _save_settings(proj, sec, clean_settings(sec, values[sec]))
    after = load_project(tmp_path).cfg
    for sec in SETTINGS:
        if sec != "events":
            blank = "" if sec == "paths" else None                       # no Downloads folder set: "" or absent
            assert {k: after[sec].get(k, blank) for k in values[sec]} == \
                {k: before[sec].get(k, blank) for k in values[sec]}, sec


def test_settings_checks_and_tables(tmp_path: Path):
    import tomllib
    from goodpush.app.server import _save_settings
    from goodpush.config import clean_settings, init_project, load_project
    proj = init_project(tmp_path)
    with pytest.raises(ValueError, match="pick exactly 2"):
        clean_settings("plan", {"key_days": ["tue"]})
    with pytest.raises(ValueError, match="unknown time zone"):
        clean_settings("athlete", {"timezone": "Mars/Olympus"})
    with pytest.raises(ValueError, match="0 or 30-200"):
        clean_settings("athlete", {"weight_kg": 12})
    assert clean_settings("athlete", {"weight_kg": 0, "max_hr": "185"}) == {"weight_kg": 0, "max_hr": 185}
    assert clean_settings("plan", {"availability": []}) == {"availability": []}
    _save_settings(proj, "ftp", clean_settings("ftp", {"mode": "manual", "history": "2026-10-22 250\n2026-01-05=240"}))
    _save_settings(proj, "ftp", clean_settings("ftp", {"history": "2026-10-22 238"}))        # 2026-01-05 removed
    _save_settings(proj, "load", clean_settings("load", {"default_tss_per_hour": {"ride": 50, "run": 55}}))
    rows = clean_settings("events", {"events": "2026-02-10, 2026-02-03, illness, flu, rough\n2026-03-01, , travel, "})
    assert rows["events"][0] == {"date": "2026-02-10", "end": "2026-02-10", "type": "illness", "note": "flu, rough"}
    _save_settings(proj, "events", rows)
    _save_settings(proj, "events", rows)                                  # replaced, not added twice
    cfg = load_project(tmp_path).cfg
    assert cfg["ftp"]["mode"] == "manual" and cfg["ftp"]["history"] == {"2026-10-22": 238}
    assert cfg["load"]["default_tss_per_hour"] == {"ride": 50, "run": 55, "strength": 0, "yoga": 0, "other": 0}
    assert [e["type"] for e in cfg["events"]] == ["illness", "travel"]
    assert "[ride]" in (tmp_path / "goodpush.toml").read_text() and tomllib.loads((tmp_path / "goodpush.toml").read_text())
    with pytest.raises(ValueError, match="start, end, type"):
        clean_settings("events", {"events": "2026-02-03, 2026-02-10, holiday"})


def test_sick_days_count_as_illnesses(tmp_path: Path):
    import json
    from types import SimpleNamespace
    from goodpush.models import markers
    (tmp_path / "plan_state.json").write_text(json.dumps({"overrides": {
        "2026-03-01": "sick", "2026-03-02": "sick", "2026-03-04": "sick",      # one illness (a day's gap)
        "2026-03-10": "off", "2026-05-20": "sick"}}))                          # a day off isn't; a later one is
    ev = markers.sick_events(tmp_path)
    assert [(e["date"], e["end"]) for e in ev] == [("2026-03-01", "2026-03-04"), ("2026-05-20", "2026-05-20")]
    assert markers.sick_events(tmp_path / "nowhere") == []
    proj = SimpleNamespace(processed=tmp_path, cfg={"events": [{"date": "2025-01-05", "type": "illness", "note": "flu"}]})
    acts = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "name": ["easy, still a bit sick"],
                         "description": [""], "private_note": [""], "type": ["Ride"]})
    fp = markers.sources(proj, acts)
    assert len(markers.illness_events(proj)) == 3 and fp == markers.sources(proj, acts)
    (tmp_path / "plan_state.json").write_text(json.dumps({"overrides": {"2026-06-01": "sick"}}))
    assert markers.sources(proj, acts) != fp                                    # a new sick day: learn again


def test_insights_send_every_gain():
    from goodpush.app.server import _insights
    res = {"progress_periods": [{"start": f"20{k:02d}-01-01", "end": f"20{k:02d}-03-01", "change_pct": 5.0 + k} for k in range(9)]}
    out = _insights(res)
    assert len(out["gains"]) == 9 and len(out["progress"]) == 6 and set(out["gains"][0]) == {"start", "end", "change_pct"}


def test_installed_app_finds_its_project_and_runs_itself(tmp_path: Path, monkeypatch):
    import sys
    from goodpush import launcher
    from goodpush.config import default_root, init_project, load_project
    monkeypatch.setenv("GOODPUSH_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)                                      # started from a folder with no project
    assert default_root() == tmp_path / "home"
    with pytest.raises(FileNotFoundError):
        load_project()
    init_project(default_root())                                     # what the app does on first launch
    assert load_project().root == (tmp_path / "home").resolve()
    assert launcher.command("ride")[-4:] == ["-u", "-m", "goodpush.cli", "ride"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)          # the installed app: it runs itself
    assert launcher.command("--root", "x", "ride") == [sys.executable, "--root", "x", "ride"]
    assert launcher.child_env()["PYTHONUNBUFFERED"] == "1"


def test_installed_app_uses_the_default_athlete_and_leaves_another_goodpush_alone(tmp_path: Path, monkeypatch):
    from goodpush.athletes import Athletes
    from goodpush.launcher import App
    monkeypatch.setenv("GOODPUSH_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GOODPUSH_SECRETS", raising=False)
    monkeypatch.setattr(Athletes, "SET", None)
    monkeypatch.chdir(tmp_path)
    assert App().project().root == (tmp_path / "home").resolve()            # nobody yet: the app's own folder
    Athletes().add("athlete_name", tmp_path / "athlete_name")
    (tmp_path / "athlete_name").mkdir()
    (tmp_path / "athlete_name" / "goodpush.toml").write_text("")
    assert App().project().root == (tmp_path / "athlete_name").resolve()            # the machine's default athlete
    me = App.build()
    ports = {8765: "0123456789ab", 8766: ""}                                 # a development goodpush, another program
    monkeypatch.setattr(App, "build_at", staticmethod(lambda p: ports.get(p)))
    app = App()
    assert app.choose() is False and app.url == "http://127.0.0.1:8767/"   # starts on the first free port
    ports[8766] = me
    app = App()
    assert app.choose() is True and app.port == 8766                       # opened twice: just shows the page


def test_quit_in_the_app_stops_the_server_and_asks_while_a_ride_is_open(tmp_path: Path, monkeypatch):
    """closing the browser leaves goodpush running: the top bar's Quit stops it, asking first when it's busy."""
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from types import SimpleNamespace
    from goodpush.app.server import RideLauncher, make_handler
    from goodpush.config import init_project, load_project
    init_project(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(load_project(tmp_path), None, lan=False))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/action"
    post = lambda b: json.loads(urllib.request.urlopen(urllib.request.Request(  # noqa: E731
        url, json.dumps(b).encode(), {"Content-Type": "application/json", "X-GoodPush": "1"})).read())
    monkeypatch.setattr(RideLauncher, "proc", SimpleNamespace(poll=lambda: None))     # a ride window open
    try:
        j = post({"action": "quit"})
        assert "ride window" in j["ask"] and j["yes"] == "Quit" and j["danger"]
        assert th.is_alive()                                                # nothing stopped before the answer
        assert post({"action": "quit", "force": True}) == {"ok": True}
        th.join(5)
        assert not th.is_alive()
    finally:
        srv.shutdown() if th.is_alive() else None
        srv.server_close()


def test_console_quick_edit_off_is_safe_anywhere():
    """off in a real Windows console; elsewhere (CI, pytest's captured streams, macOS) just False."""
    from goodpush.progress import Console
    assert Console.no_quick_edit() in (True, False)


def test_dip_profiles_without_a_metric_raise_no_warning():
    """a metric missing around every break (e.g. no HRV yet): None in the profile, no 'Mean of empty slice'."""
    import warnings
    from goodpush.models.dips import TrainingBreaks
    days = pd.date_range("2025-01-01", periods=200)
    daily = pd.DataFrame({"tss": 50.0, "tsb": 0.0, "rhr": 50.0, "hrv": np.nan, "sleep_h": 7.5}, index=days)
    dips = pd.DataFrame({"start": [days[60], days[120]], "cause": ["illness", "illness"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        per, prof = TrainingBreaks.around(daily, dips)
    assert len(per) == 2 and prof["causes"]["illness"]["n"] == 2
    assert all(v is None for v in prof["causes"]["illness"]["hrv_z"])           # never measured
    assert set(prof["metrics"]) == set(TrainingBreaks.PROFILE_METRICS)


def test_app_remakes_a_stale_plan_before_opening(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from goodpush.app import server
    from goodpush.planning import Planner
    proj = SimpleNamespace(reports=tmp_path, processed=tmp_path, cfg={"plan": {"enabled": True}}, tz="UTC")
    (tmp_path / "daily.parquet").write_bytes(b"")
    made = []
    monkeypatch.setattr(server, "_plan_only", lambda proj, log: made.append(1))
    assert server.plan_at_start(proj) and made == [1]                          # no plan yet
    (tmp_path / "plan").mkdir()
    day = str(pd.Timestamp.now(tz="UTC").tz_localize(None).date())
    plan = tmp_path / "plan" / "plan.json"
    plan.write_text(json.dumps({"generated": day, "sessions": [], "code": "older"}))
    assert server.plan_at_start(proj) and made == [1, 1]                       # made by other code
    plan.write_text(json.dumps({"generated": day, "sessions": [], "code": Planner.CODE}))
    assert not server.plan_at_start(proj) and made == [1, 1]                   # today's, by this code


def test_alpha1_from_a_garmin_copy_of_a_goodpush_ride(tmp_path: Path):
    """a goodpush recording and a Garmin recording of the same ride (the Garmin one is dropped as a duplicate): the
    goodpush ride takes the Garmin file's alpha1, on its own timeline and with its own power."""
    from types import SimpleNamespace
    from goodpush.alpha import companions, on_ride
    g = tmp_path / "garmin" / "activities"
    g.mkdir(parents=True)
    (tmp_path / "strava").mkdir()
    pd.DataFrame({"id": ["garmin_1", "garmin_2"], "start_utc": ["2026-11-03T23:03:40Z", "2026-11-02T10:00:00Z"],
                  "type": "Virtual Ride", "filename": ["garmin_1.fit", "garmin_2.fit"]}).to_csv(g / "activities_garmin.csv", index=False)
    proj = SimpleNamespace(path=lambda k: tmp_path / k)
    rides = pd.DataFrame({"id": ["goodpush_1", "goodpush_2", "strava_3"], "type": "Virtual Ride",
                          "start_utc": pd.to_datetime(["2026-11-03T23:03:15Z", "2026-11-05T10:00:00Z", "2026-11-03T23:03:40Z"])})
    found = companions(proj, rides)
    assert list(found) == ["goodpush_1"] and found["goodpush_1"].name == "garmin_1.fit"     # 25 s apart; 2 has none
    t0 = 1_793_746_995.0                                # the goodpush ride starts 10 s before the Garmin one
    rec = {"time": list(t0 + np.arange(600.0)), "power": list(100 + np.arange(600.0))}
    df = pd.DataFrame({"t": t0 + 10 + np.arange(0.0, 700.0), "a1": 1.0, "p": np.nan, "hr": 120.0, "art": 0.0})
    df = on_ride(df, rec)
    assert df["t"].iloc[0] == 10 and df["p"].iloc[0] == 110 and df["t"].iloc[-1] == 599     # past its end: dropped


def test_dfa_alpha1_on_signals_of_known_scaling():
    """DFA alpha1 (metrics.dfa) against theory: uncorrelated beats 0.58 - not 0.5: with boxes of 4-16 beats DFA's
    expected F(n)^2 is (n^2 - 4) / 15n (Kantelhardt et al. 2001, Physica A 295:441), whose slope is 0.583 - then
    1/f noise 1.0 and a random walk 1.5; and the windows of a beat series. Whole-ride arrays at once."""
    from goodpush.metrics import dfa
    rng = np.random.default_rng(1)
    n = 120_000

    def series(beta):                            # power spectrum ~ 1/f^beta: alpha = (1 + beta) / 2
        f = np.fft.rfftfreq(n)
        amp = np.where(f > 0, f, np.inf) ** (-beta / 2)
        x = np.fft.irfft(amp * np.exp(2j * np.pi * rng.random(f.size)), n)
        return 0.6 + 0.01 * x / x.std()
    starts = np.arange(0, n - 200, 200)
    k = np.arange(4, 17)
    white = np.polyfit(np.log(k), 0.5 * np.log((k ** 2 - 4) / (15 * k)), 1)[0]
    for beta, want, tol in ((0.0, white, 0.02), (1.0, 1.0, 0.05), (2.0, 1.5, 0.06)):
        a = dfa.alpha1(series(beta), starts, np.full(starts.size, 200))
        assert np.nanmean(a) == pytest.approx(want, abs=tol), beta
    t = np.cumsum(np.full(600, 0.5))             # 5 min at 120 bpm
    a, k = dfa.windows(t, np.array([60.0, 200.0, 300.0]))
    assert list(k) == [120, 240, 240] and a[2] == t.searchsorted(180.0, "right")


def test_dfa_beat_correction_and_detrending():
    """odd beats: a missed one split in two, an extra one merged, a misplaced pair evened out, all flagged, the
    time kept; something unresolvable kept and flagged. A heart rate drifting up (a ramp) doesn't lift alpha1 once
    detrended."""
    from goodpush.metrics import dfa
    rr = np.full(40, 0.5)
    rr = np.r_[rr[:10], 1.0, rr[10:20], 0.2, 0.3, rr[20:30], 0.36, 0.64, rr[30:]]   # missed, extra, misplaced
    out, flag = dfa.corrected(rr)
    assert np.allclose(out, 0.5) and out.size == 45 and out.sum() == pytest.approx(rr.sum())   # 22.5 s of 0.5 s beats
    assert (flag == 1).sum() == 2 + 1 + 2
    odd = np.r_[np.full(10, 0.5), 0.8, np.full(10, 0.5)]                     # 1.6 beats: can't resolve
    out, flag = dfa.corrected(odd)
    assert np.array_equal(out, odd) and list(np.flatnonzero(flag)) == [10] and flag[10] == 2
    rng = np.random.default_rng(4)
    beats = 0.5 + 0.005 * rng.standard_normal(400)
    lost = np.r_[beats[:150], 3.5, 3.0, beats[150:]]                         # a strap losing contact for 6.5 s
    out, flag = dfa.corrected(lost)
    assert np.array_equal(out, lost) and list(np.flatnonzero(flag)) == [150, 151] and (flag == 2).all(where=flag > 0)
    t = np.array([100.0])                                                   # a window around the gap
    a, share = dfa.series(lost, 0.0, t)
    clean, _ = dfa.series(beats, 0.0, t)
    assert a[0] == pytest.approx(clean[0], abs=0.1) and share[0] < dfa.MAX_ARTIFACTS   # only its boxes left out
    rest = 0.9 + 0.12 * np.sin(np.arange(300) * 2 * np.pi / 5) + 0.01 * rng.standard_normal(300)
    assert not dfa.corrected(rest)[1].any()                                  # breathing swings at rest (+-13 %) aren't odd
    rng = np.random.default_rng(2)
    n = 20_000
    white = 0.5 + 0.01 * rng.standard_normal(n)
    ramp = white - np.linspace(0, 0.25, n)                                   # heart rate rising all the way
    s = np.arange(0, n - 300, 300)
    k = np.full(s.size, 300)
    raw, flat = dfa.alpha1(ramp, s, k).mean(), dfa.alpha1(dfa.detrended(ramp), s, k).mean()
    assert flat == pytest.approx(dfa.alpha1(white, s, k).mean(), abs=0.02) and raw > flat


def test_a_restarted_app_reloads_the_open_page_instead_of_opening_another(tmp_path: Path, monkeypatch):
    import http.client
    import json
    import threading
    import webbrowser
    from http.server import ThreadingHTTPServer
    from goodpush.app.server import make_handler
    from goodpush.config import init_project, load_project
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    init_project(tmp_path)
    handler = make_handler(load_project(tmp_path), None, lan=False)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert handler.pages.open_unless_seen("http://x/", wait=0.1) and opened == ["http://x/"]   # no page open
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c.request("GET", "/api/alive")
        r = c.getresponse()
        assert r.getheader("Content-Type") == "text/event-stream"
        assert r.fp.readline() == b"retry: 1000\n"
        run = r.fp.readline().decode().removeprefix("data: ").strip()
        c2 = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c2.request("GET", "/api/job")
        assert run == json.loads(c2.getresponse().read())["started"]   # the page compares it with its own run
        assert not handler.pages.open_unless_seen("http://x/", wait=0.1) and opened == ["http://x/"]   # a page is open
        c.close()
    finally:
        srv.shutdown()


def test_ranges_refresh_due_weekly_or_when_the_fit_leaves_its_range():
    """2026-11-12: the ranges are re-made once they are ranges_every_days data days older than the best fit, or at
    once when today's best-fit CP / LT1 / MAP lies outside its stored 5-95 % range; 0 / daily_ranges = false never."""
    from goodpush.app.server import DailyRanges as R
    inside = pd.Series({"cp": 250.0, "cp_p05": 240.0, "cp_p95": 260.0, "lt1": 200.0, "lt1_p05": 190.0,
                        "lt1_p95": 210.0, "map": 300.0, "map_p05": 290.0, "map_p95": 310.0, "ftp": 999.0})
    assert R.is_due("2026-11-10", "2026-11-03", 7, inside)          # a week older
    assert not R.is_due("2026-11-09", "2026-11-03", 7, inside)      # six days: not yet
    assert not R.is_due("2026-11-09", "2026-11-03", 7, None)
    assert R.is_due("2026-11-04", "2026-11-03", 7, inside.mask(inside.index == "lt1", 215.0))  # above its range
    assert R.is_due("2026-11-04", "2026-11-03", 7, inside.mask(inside.index == "map", 280.0))   # below its range
    assert not R.is_due("2026-11-04", "2026-11-03", 7, inside.mask(inside.index == "ftp", 1.0))  # not watched
    assert not R.is_due("2026-11-04", "2026-11-03", 7, inside.drop(["cp_p05", "cp_p95"]))      # no range: no reason
    assert not R.is_due("2026-11-30", "2026-11-03", 0, inside.mask(inside.index == "cp", 1.0))  # 0 = never
    assert not R.is_due("2026-11-30", "", 7, inside)                # no draws: a full run is needed (on command)
    assert not R.is_due("2026-11-10", "2026-11-10", 1, inside.mask(inside.index == "cp", 1.0))  # draws are today's
    assert R.every_days({}) == R.EVERY_DAYS == 7
    assert R.every_days({"physio": {"daily_ranges": False}}) == 0               # the old switch still means never
    assert R.every_days({"physio": {"daily_ranges": True}}) == 7
    assert R.every_days({"physio": {"daily_ranges": False, "ranges_every_days": 3}}) == 3   # the new key wins
    assert R.every_days({"physio": {"ranges_every_days": 0}}) == 0


class TerminalStub(io.StringIO):
    """a stand-in terminal (Tee.into only writes a log when output is a terminal)."""

    def isatty(self) -> bool:
        return True


def test_run_logs_stay_on_this_computer(tmp_path: Path, monkeypatch):
    """2026-11-13: run logs were redirected into the athlete folder and synced to every computer. They live beside
    this computer's credentials (Project.logs), a terminal run writes its own (Tee.into), and sync skips *.log."""
    from goodpush.config import Project
    from goodpush.progress import Tee
    from goodpush.sync.peer import PeerRules
    logs = Project(tmp_path / "athlete").logs
    assert logs.is_dir() and logs == tmp_path / "athlete" / "logs"     # in the folder (its drive), never synced
    term, err = TerminalStub(), TerminalStub()
    monkeypatch.setattr(sys, "stdout", term)
    monkeypatch.setattr(sys, "stderr", err)
    with Tee.into(logs / "run.log") as path:
        print("physio model v16: 84 days")
        print("a bar", file=sys.stderr)
        assert sys.stdout.isatty()                           # tqdm still sees a terminal
        kept = sys.stdout                                    # (colorama keeps the stream it wrapped ...)
    assert sys.stdout is term and sys.stderr is err          # put back
    kept.write("\x1b[0m")                                    # ... and writes its colour reset at exit: no error
    kept.flush()
    assert term.getvalue().endswith("\x1b[0m")
    text = path.read_text(encoding="utf-8")
    assert "physio model v16: 84 days" in text and "a bar" in text and "physio model v16" in term.getvalue()
    monkeypatch.setattr(sys, "stdout", io.StringIO())        # redirected (or started by the app): no own log
    with Tee.into(logs / "other.log") as path:
        print("x")
    assert path is None and not (logs / "other.log").exists()
    for rel in ("physio_v16.log", "data/processed/physio_refresh.log", "data/processed/ride_launch.log",
                "logs/physio-nuts_2026-11-13_1555.log", "logs/notes.txt"):
        assert PeerRules.kind(rel) == "skip"


def test_ranges_refresh_starts_once_a_data_day_on_the_home_computer(tmp_path: Path, monkeypatch):
    """2026-11-10: the app re-samples the model's ranges (`goodpush physio --refresh`) in its own process, low
    priority, after a page was served, on the home computer only, at most once a data day - and only when due
    (2026-11-12: a week older, or a best-fit number outside its range; test above)."""
    import sys
    from types import SimpleNamespace
    from goodpush import launcher
    from goodpush.app import server
    from goodpush.models.physio import PhysioRun
    monkeypatch.setattr(launcher, "command", lambda *a: [sys.executable, "-c", "pass"])   # the refresh: a no-op
    proj = SimpleNamespace(processed=tmp_path, root=tmp_path, cfg={}, logs=tmp_path / "logs")   # this computer's logs
    proj.logs.mkdir()
    np.savez(tmp_path / PhysioRun.FIT, day="2026-11-10", draws_day="2026-11-03")
    home = [False]
    r = server.DailyRanges(server.Jobs(), lambda: home[0], lambda: proj)
    assert not r.tick()                                     # no page served yet
    r.page_served()
    assert not r.tick()                                     # not the home computer
    home[0] = True
    assert r.tick() and r.proc is not None
    assert (proj.logs / r.LOG).read_text(encoding="utf-8").startswith("refresh for 2026-11-10")
    assert not (tmp_path / r.LOG).exists()                  # not in the athlete folder (synced)
    r.proc.wait()
    assert not r.tick() and r.proc is None                  # once a data day, even when it failed or saved nothing
    np.savez(tmp_path / PhysioRun.FIT, day="2026-11-11", draws_day="2026-11-03")
    proj.cfg = {"physio": {"daily_ranges": False}}
    assert not r.tick()                                     # turned off (the old switch)
    proj.cfg = {"physio": {"ranges_every_days": 0}}
    assert not r.tick()                                     # turned off
    proj.cfg = {"physio": {"ranges_every_days": 30}}
    assert not r.tick()                                     # not due yet (no states: no range to leave)
    pd.DataFrame({"cp": [300.0], "cp_p05": [240.0], "cp_p95": [260.0]},
                 index=pd.DatetimeIndex(["2026-11-11"])).to_parquet(tmp_path / PhysioRun.STATES)
    assert r.tick()                                         # the best fit left its range: due at once
    r.proc.wait()
    np.savez(tmp_path / PhysioRun.FIT, day="2026-11-11", draws_day="2026-11-04")
    assert not r.tick() and r.proc is None                  # once a data day
    (tmp_path / PhysioRun.STATES).unlink()
    np.savez(tmp_path / PhysioRun.FIT, day="2026-11-12", draws_day="2026-11-04")   # a new data day, 8 days on
    proj.cfg = {}
    from filelock import FileLock
    with FileLock(str(tmp_path / PhysioRun.LOCK)):
        assert not r.tick()                                 # another NUTS run holds the lock
    assert r.tick()                                         # a week older and a new data day
    r.proc.wait()
    np.savez(tmp_path / PhysioRun.FIT, day="2026-11-13", draws_day="2026-11-13")
    assert not r.tick()                                     # the draws are today's
    np.savez(tmp_path / PhysioRun.FIT, day="2026-11-13", draws_day="")
    assert not r.tick()                                     # no draws: a full run is needed (on command)
    assert "creationflags" in r.lower() if sys.platform == "win32" else "preexec_fn" in r.lower()


class StageStub:
    """FullModelRun's stage processes in a test (launcher.command): each prints its last argument; the NUTS one
    fails (fail), every one sleeps 30 s (slow) when asked."""

    def __init__(self):
        self.fail = self.slow = False

    def __call__(self, *a) -> list[str]:
        import sys
        code = (f"import sys, time; print({a[-1]!r}); time.sleep({30 if self.slow else 0}); "
                f"sys.exit({int(self.fail and '--nuts' in a)})")
        return [sys.executable, "-c", code]

    @staticmethod
    def finish(run, seconds: float = 60) -> None:
        import time
        t = time.time()
        while run.running() and time.time() - t < seconds:
            time.sleep(0.05)
        assert not run.running()


def test_full_model_run_samples_then_backtests(tmp_path: Path, monkeypatch):
    """2026-11-17: the Model page's Run button: a full NUTS run, then the backtest, in their own processes; the
    backtest only after a sampling run that finished; Stop ends it; never over another NUTS run; the weekly refresh
    waits for it."""
    import time
    from types import SimpleNamespace
    from filelock import FileLock
    from goodpush import launcher
    from goodpush.app import server
    from goodpush.models.physio import PhysioRun
    stub = StageStub()
    monkeypatch.setattr(launcher, "command", stub)
    proj = SimpleNamespace(processed=tmp_path, root=tmp_path, cfg={}, logs=tmp_path / "logs")
    proj.logs.mkdir()
    jobs = server.Jobs()
    r = server.FullModelRun(jobs, lambda: proj)
    finish = StageStub.finish
    assert r.argv("nuts") == tuple(PhysioRun.FULL_RUN.split()[1:]) and r.argv("backtest") == ("physio", "--backtest")
    assert r.status()["busy"] == "" and r.status()["stage"] == ""
    r.start()
    assert r.status()["stage"] == "nuts" and r.status()["step"] == 1
    with pytest.raises(ValueError):
        r.start()                                            # one at a time
    ranges = server.DailyRanges(jobs, lambda: True, lambda: proj, r)
    assert not ranges.idle(proj)                             # no refresh during it
    finish(r)
    log = (proj.logs / r.LOG).read_text(encoding="utf-8")
    assert r.result["ok"] and "== Sampling" in log and "== Backtest" in log and r.last_line(proj) == "--backtest"
    stub.fail = True
    r.start()
    finish(r)
    log = (proj.logs / r.LOG).read_text(encoding="utf-8")
    assert not r.result["ok"] and "failed" in r.result["say"] and "== Backtest" not in log   # none after a failure
    stub.fail, stub.slow = False, True
    r.start()
    t = time.time()
    while r.proc is None and time.time() - t < 30:
        time.sleep(0.05)
    r.stop()
    finish(r)
    log = (proj.logs / r.LOG).read_text(encoding="utf-8")
    assert not r.result["ok"] and r.result["say"].startswith("stopped") and "== Backtest" not in log
    with FileLock(str(tmp_path / PhysioRun.LOCK)):
        assert "sampling run" in r.busy(proj)               # a terminal's NUTS run or the refresh holds the lock
        with pytest.raises(ValueError):
            r.start()
    assert server.FullModelRun.file_day(proj.logs / r.LOG) == time.strftime("%Y-%m-%d")
    assert server.FullModelRun.file_day(tmp_path / "none.json") is None


def test_app_fit_from_scratch_is_refused_during_a_sampling_run(tmp_path: Path):
    """Settings → Data → Fit the model from scratch (action physio_fresh) drops the NUTS checkpoint and the draws:
    never while a sampling run holds the lock - refused with the reason, the checkpoint left as it was."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from filelock import FileLock
    from goodpush.app.server import make_handler
    from goodpush.config import init_project, load_project
    from goodpush.models.physio import PhysioRun
    init_project(tmp_path)
    proj = load_project(tmp_path)
    proj.processed.mkdir(parents=True, exist_ok=True)
    checkpoint = proj.processed / PhysioRun.CHECKPOINT
    checkpoint.write_bytes(b"a stopped run")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proj, "k" * 20, lan=False, warm=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/action",
                                 json.dumps({"action": "physio_fresh"}).encode(),
                                 {"Content-Type": "application/json", "X-GoodPush": "1"})
    try:
        with FileLock(str(proj.processed / PhysioRun.LOCK)):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
        assert e.value.code == 400 and "sampling run" in json.loads(e.value.read())["error"]
        assert checkpoint.read_bytes() == b"a stopped run"
    finally:
        srv.shutdown()


def test_app_state_never_starts_or_waits_for_a_refresh(tmp_path: Path, monkeypatch):
    """Hard rule (2026-11-10): nothing the app serves waits for the refresh. /api/state reads the stored fit (mode and
    ranges, with the ranges' date) and neither starts nor runs one; only the minute loop does, after a page."""
    import json
    import threading
    import time
    import urllib.request
    from http.server import ThreadingHTTPServer
    from goodpush.app import server
    from goodpush.config import init_project, load_project
    from goodpush.models.physio import PhysioRun
    started = []
    monkeypatch.setattr(server.DailyRanges, "start", lambda self, proj, day: started.append(day))
    monkeypatch.setattr(PhysioRun, "refresh", lambda *a, **k: started.append("refresh"))
    monkeypatch.setattr(PhysioRun, "current", lambda *a, **k: started.append("current"))
    init_project(tmp_path)
    proj = load_project(tmp_path)
    proj.processed.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2026-11-08", periods=3, name="date")
    pd.DataFrame({c: 200.0 for c in ("cp", "cp_p05", "cp_p95", "lt1", "map", "ftp", "fatigue", "form")},
                 index=days).to_parquet(proj.processed / PhysioRun.STATES)
    (proj.processed / "physio_diagnostics.json").write_text('{"draws_day": "2026-11-08"}', encoding="utf-8")
    np.savez(proj.processed / PhysioRun.FIT, day="2026-11-10", draws_day="2026-11-08")    # a refresh is due
    st = server.Data(proj).state()
    assert st["physio"]["draws_day"] == "2026-11-08" and st["physio"]["date"][-1] == "2026-11-10"
    assert started == []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(proj, None, lan=False, warm=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t = time.time()
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/api/state") as r:
            js = json.loads(r.read())
        assert time.time() - t < 10 and js["physio"]["draws_day"] == "2026-11-08"
        assert started == []                                # the page got its answer; no refresh started or run
    finally:
        srv.shutdown()


def test_model_page_reads_the_coverage_check(tmp_path: Path):
    """2026-11-11: the Model page's verdict row - /api/model carries the coverage check's held-out numbers
    (processed/ladder/physio_coverage.json, checks.coverage.PosteriorCoverage) when there is one, and whether it was
    made with this model version; nothing without the file. The physio payload carries W' for the Rider page."""
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from goodpush.app import server
    from goodpush.config import init_project, load_project
    from goodpush.checks.coverage import PosteriorCoverage
    from goodpush.models.physio import PhysioRun
    from goodpush.physio.model import VERSION
    init_project(tmp_path)
    proj = load_project(tmp_path)
    (proj.processed / "ladder").mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(proj, None, lan=False, warm=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/model"
    try:
        with urllib.request.urlopen(url) as r:
            first = json.loads(r.read())
        assert "coverage" not in first and "backtest" not in first and "ladder" not in first   # (the ladder is gone)
        assert first["versions"] == {"fit": None, "running": VERSION, "on_disk": VERSION}   # 2026-11-13: one model
        bt_file = proj.processed / "ladder" / "physio_calibrated.json"
        bt_file.write_text(json.dumps({"backtest": {"mae_pct": 4.1, "bias_pct": -0.5, "n": 90, "folds": []}}),
                           encoding="utf-8")
        with urllib.request.urlopen(url) as r:                  # with the day the backtest was written
            assert json.loads(r.read())["backtest"] == {"mae_pct": 4.1, "bias_pct": -0.5, "n": 90,
                                                         "date": server.FullModelRun.file_day(bt_file)}
        part = {"n": 40, "inside_50": 0.5, "inside_90": 0.9, "mean_pit": 0.5, "above_95": 0.05}
        (proj.processed / "ladder" / PosteriorCoverage.FILE).write_text(json.dumps(
            {"key": {"version": VERSION}, "folds": [{}, {}], "summary": {"efforts_by_kind": {"all": part},
                                                                       "readings": {"all": {**part, "n": 6}}}}), encoding="utf-8")
        with urllib.request.urlopen(url) as r:
            cov = json.loads(r.read())["coverage"]
        assert cov == {"efforts": part, "readings": {**part, "n": 6}, "folds": 2, "current": True,
                       "date": server.FullModelRun.file_day(proj.processed / "ladder" / PosteriorCoverage.FILE)}
    finally:
        srv.shutdown()
    days = pd.date_range("2026-11-08", periods=3, name="date")
    pd.DataFrame({c: 200.0 for c in ("cp", "lt1", "map", "ftp", "w_prime", "w_prime_p05", "w_prime_p95", "fatigue", "form")},
                 index=days).to_parquet(proj.processed / PhysioRun.STATES)
    assert server.Data(proj).state()["physio"]["w_prime_p95"][-1] == 200.0


def test_ride_stamina_w_balance_durability_and_day(tmp_path):
    from goodpush.app.server import ride_streams
    from goodpush.models.physio import PhysioRun
    from goodpush.models.stamina import RideStamina as RS
    cp, wp, lt1 = 250.0, 20000.0, 190.0
    p = np.r_[np.full(600, 100.0), np.full(300, 300.0), np.full(600, 100.0)]
    s = RS(p, cp, wp)                                            # no LT1: no durability decline
    bal = s.w_balance()
    assert bal[599] == pytest.approx(wp) and bal[899] == pytest.approx(wp - 50 * 300)    # spent: 50 W x 300 s
    k = s.skiba_tau(150.0) / s.skiba_tau(RS.REF_GAP)             # recovering 150 W under CP, slower than Chorley's
    left = RS.FAST * np.exp(-600 / (RS.TAU_FAST * k)) + (1 - RS.FAST) * np.exp(-600 / (RS.TAU_SLOW * k))
    assert bal[-1] == pytest.approx(wp - 15000 * left)
    assert s.current()[899] == pytest.approx(5000 / wp) and (s.potential() == 1).all()
    # the recursion second by second (two pools; the slow one slows once a second W' is spent), across BLOCKs
    rng = np.random.default_rng(1)
    q = np.clip(rng.normal(240, 110, 6000), 0, None)
    r = RS(q, cp, wp)
    ref, f, sl, e = np.empty(q.size), 0.0, 0.0, 0.0
    for i, w in enumerate(q):
        if w < cp:
            kk = r.skiba_tau(cp - w) / r.skiba_tau(RS.REF_GAP)
            f *= np.exp(-1 / (RS.TAU_FAST * kk))
            sl *= np.exp(-1 / (RS.TAU_SLOW * (1 + RS.SLOW_GROWTH * min(max(e / wp - 1, 0), 1)) * kk))
        else:
            f, sl, e = f + RS.FAST * (w - cp), sl + (1 - RS.FAST) * (w - cp), e + w - cp
        ref[i] = wp - f - sl
    assert np.maximum(q - cp, 0).sum() > 2 * wp                  # the slowing is exercised up to its cap
    assert r.w_balance() == pytest.approx(ref)
    r.BLOCK = 700
    assert r.w_balance() == pytest.approx(ref)
    # stamina is 0 once W' is spent; potential is not touched by W'
    out = RS(np.full(600, 350.0), cp, wp)
    assert out.current()[199] == pytest.approx(0.0) and (out.potential() == 1).all()
    # durability (Clark 2019): 2 h above LT1 -> CP -10 % (none in the first hour), W' -20 %; below LT1 nothing
    h = RS(np.full(7200, 220.0), cp, wp, lt1)
    fresh = cp + wp / RS.REF_S
    assert h.cp_now()[3599] == pytest.approx(cp) and h.cp_now()[-1] == pytest.approx(0.9 * cp)
    assert h.w_prime_now()[-1] == pytest.approx(0.8 * wp)
    assert h.potential()[-1] == pytest.approx((0.9 * cp + 0.8 * wp / RS.REF_S) / fresh)
    easy = RS(np.full(7200, 150.0), cp, wp, lt1)
    assert (easy.potential() == 1).all() and (easy.current() == 1).all()
    d = RS(q, cp, wp, lt1)
    cur, pot = d.current(), d.potential()
    assert (cur <= pot + 1e-12).all() and (np.diff(pot) <= 1e-12).all() and (cur >= 0).all()
    # the day's fit: its nearest day; none without a fit
    assert RS.for_day(tmp_path, "2026-01-02", q) is None
    days = pd.to_datetime(["2026-01-01", "2026-01-05"])
    pd.DataFrame({"cp": [240.0, 260.0], "w_prime": [18000.0, 21000.0], "lt1": [180.0, 195.0]}, index=days) \
        .to_parquet(tmp_path / PhysioRun.STATES)
    fd = RS.for_day(tmp_path, "2026-01-03", q)
    assert (fd.cp, fd.w_prime, fd.lt1) == (260.0, 21000.0, 195.0)
    # chart streams carry both lines in %
    z = {"power": q, "heart_rate": np.full(q.size, 140.0), "cadence": np.full(q.size, 90.0),
         "speed": np.full(q.size, 8.0), "altitude": np.zeros(q.size)}
    st = ride_streams(z, 0, q.size, points=q.size, stamina=d)
    assert st["stamina"][-1] == pytest.approx(cur[-1] * 100, abs=0.06)
    assert st["potential"][-1] == pytest.approx(pot[-1] * 100, abs=0.06)
    assert "stamina" not in ride_streams(z, 0, q.size)
