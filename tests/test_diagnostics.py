"""Settings > Diagnostics (app/diagnostics.py): the facts, what may be shared, the log viewer's limits, the app log,
and the page's wiring."""
import io
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from goodpush.app.diagnostics import AppLog, AppLogStream, Diagnostics
from goodpush.config import init_project, load_project
from goodpush.physio.model import VERSION

STATIC = Path(__file__).parents[1] / "src" / "goodpush" / "app" / "static"


def project(tmp_path: Path):
    init_project(tmp_path / "rider")
    return load_project(tmp_path / "rider")


def facts(report: dict) -> dict:
    return {g["id"]: {f["label"]: f for f in g["facts"]} for g in report["groups"]}


def test_the_report_has_versions_model_athletes_and_folders(tmp_path: Path):
    from goodpush import __version__
    proj = project(tmp_path)
    rep = Diagnostics(proj, {"name": "Updating plan", "running": False, "ok": True}, peers=2).report()
    f = facts(rep)
    assert f["app"]["GoodPush"]["value"] == __version__
    assert f["app"]["Last job"]["value"] == "Updating plan (finished)"
    assert f["model"]["Model version (this app runs)"]["value"] == f"v{VERSION}"
    assert f["model"]["Last fit"]["value"] == "none yet"            # a new rider: nothing fitted yet
    assert f["athletes"]["Athletes on this computer"]["value"] == "0"
    assert f["athletes"]["Paired computers"]["value"] == "2"
    assert f["data"]["Activities"]["value"] == "0"
    assert {x["folder"] for x in f["folders"].values()} == {"athlete", "logs", "app_logs", "machine"}
    assert all(x["private"] for x in f["folders"].values())
    assert {"numpy", "pandas"} <= set(f["packages"])
    # the model page's fit, when there is one: its version and days, a stale one said so
    (proj.processed / "physio_diagnostics.json").write_text(json.dumps(
        {"version": VERSION - 1, "day": "2026-01-31", "draws_day": ""}), encoding="utf-8")
    m = facts(Diagnostics(proj).report())["model"]
    assert m["Model version (last fit)"]["value"].endswith("refits on the next update")
    assert m["Last fit's data up to"]["value"] == "2026-01-31"


def test_what_may_be_shared_leaves_out_folders_and_names(tmp_path: Path):
    proj = project(tmp_path)
    d = Diagnostics(proj, port=8765)
    text = d.shareable_text()
    assert "GoodPush" in text and f"v{VERSION}" in text
    assert str(tmp_path) not in text and str(Path.home()) not in text     # no folder of this computer
    assert "Shown now" not in text and "Port" not in text
    assert d.shareable()["Folders"] == {}                                # its group stays, empty


def test_a_broken_group_shows_its_error_and_the_rest_still_shows(tmp_path: Path, monkeypatch):
    proj = project(tmp_path)
    monkeypatch.setattr(Diagnostics, "sources", lambda self: 1 / 0)
    f = facts(Diagnostics(proj).report())
    assert "ZeroDivisionError" in f["sources"]["Couldn't read"]["value"]
    assert "GoodPush" in f["app"]


def test_the_log_viewer_reads_listed_logs_only_and_their_end(tmp_path: Path, monkeypatch):
    proj = project(tmp_path)
    (proj.logs / "physio_2026-01-01_0800.log").write_text("\n".join(f"line {i}" for i in range(1000)), encoding="utf-8")
    (proj.root / "secret.txt").write_text("no", encoding="utf-8")
    d = Diagnostics(proj)
    assert [x["name"] for x in d.logs()] == ["physio_2026-01-01_0800.log"]
    t = d.tail("run", "physio_2026-01-01_0800.log", 10)
    assert t["text"].splitlines() == [f"line {i}" for i in range(990, 1000)] and t["cut"]
    for where, name in (("run", "../secret.txt"), ("run", "secret.txt"), ("app", "physio_2026-01-01_0800.log")):
        try:
            d.tail(where, name)
            raise AssertionError(f"{name} served")
        except KeyError:
            pass
    monkeypatch.setattr(Diagnostics, "LOG_BYTES", 100)                 # a long log: its end only, no cut line
    t = d.tail("run", "physio_2026-01-01_0800.log", 5000)
    assert t["text"].splitlines()[-1] == "line 999" and len(t["text"]) <= 100


def test_the_app_log_stamps_lines_keeps_the_run_before_and_starts_again_when_full(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(AppLog, "folder", staticmethod(lambda: tmp_path))
    (tmp_path / AppLog.NAME).write_text("the run before\n", encoding="utf-8")
    monkeypatch.setattr(AppLog, "file", AppLog.open_new())
    monkeypatch.setattr(AppLog, "at_line_start", True)
    try:
        out = AppLogStream(io.StringIO())
        out.write("job started: x\npart")
        out.write("ly\n")
        assert out.stream.getvalue() == "job started: x\npartly\n"          # the terminal sees it as it was
        AppLog.file.flush()
        lines = (tmp_path / AppLog.NAME).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2 and lines[0].endswith(" job started: x") and lines[1].endswith(" partly")
        assert lines[1][2] == ":" and not lines[1].startswith("partly")
        assert (tmp_path / AppLog.PREVIOUS).read_text(encoding="utf-8") == "the run before\n"
        assert AppLogStream(None).write("installed app: no terminal\n")    # stdout is None there
        monkeypatch.setattr(AppLog, "MAX_BYTES", 10)
        out.write("this line is longer than ten bytes\n")
        assert (tmp_path / AppLog.NAME).read_text(encoding="utf-8") == ""   # started again; the full one kept
        assert "longer than ten" in (tmp_path / AppLog.PREVIOUS).read_text(encoding="utf-8")
    finally:
        AppLog.file.close()


def post(base: str, body: dict) -> dict:
    req = urllib.request.Request(f"{base}/api/action", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-GoodPush": "1"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def test_the_server_serves_diagnostics_and_logs_and_reveals_only_known_places(tmp_path: Path, monkeypatch):
    from goodpush.app.server import make_handler
    from goodpush.io.strava import StravaUpload
    proj = project(tmp_path)
    (proj.logs / "ride_launch.log").write_text("ride window opened\n", encoding="utf-8")
    shown = []
    monkeypatch.setattr(StravaUpload, "reveal", staticmethod(shown.append))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proj, None, lan=False, warm=False))
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    try:
        with urllib.request.urlopen(f"{base}/api/diagnostics") as r:
            rep = json.loads(r.read())
        assert {g["id"] for g in rep["groups"]} >= {"app", "model", "athletes", "data", "sources", "folders"}
        assert "## App" in rep["shareable_text"]
        with urllib.request.urlopen(f"{base}/api/log?where=run&name=ride_launch.log&lines=50") as r:
            assert json.loads(r.read())["text"] == "ride window opened"
        try:
            urllib.request.urlopen(f"{base}/api/log?where=run&name=..%2Fgoodpush.toml")
            raise AssertionError("another file served")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        assert post(base, {"action": "reveal", "folder": "logs"}) == {"ok": True}
        assert post(base, {"action": "reveal", "where": "run", "name": "ride_launch.log"}) == {"ok": True}
        assert shown == [proj.logs, proj.logs / "ride_launch.log"]
        assert "error" in post(base, {"action": "reveal", "folder": "/etc"})
        assert "error" in post(base, {"action": "reveal", "where": "run", "name": "../goodpush.toml"})
        assert len(shown) == 2
    finally:
        srv.shutdown()


def test_the_page_has_the_section_and_its_card_loads_when_opened():
    settings = (STATIC / "js" / "pages" / "settings.js").read_text(encoding="utf-8")
    card = (STATIC / "js" / "diagnostics.js").read_text(encoding="utf-8")
    assert 'this.section("diagnostics", "Diagnostics"' in settings and 'diagnostics.wire($("#set-diagnostics"))' in settings
    assert '"/api/diagnostics"' in card and "/api/log?" in card and '"toggle"' in card
    assert "Copy for a bug report" in card and "shareable_text" in card
    help_ = (STATIC / "js" / "pages" / "help.js").read_text(encoding="utf-8")
    assert "#settings/diagnostics" in help_


def test_the_entry_points_start_the_app_log():
    src = Path(sys.modules["goodpush.app.diagnostics"].__file__).parents[1]
    assert "AppLog.start()" in (src / "launcher.py").read_text(encoding="utf-8")
    assert "AppLog.start()" in (src / "cli.py").read_text(encoding="utf-8")
