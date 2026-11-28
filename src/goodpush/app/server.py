"""`goodpush app`: a small local web app over your goodpush workspace.

Python standard library only (http.server). Pages are static files in ./static; data
comes from /api/state, which reads the files `goodpush run` / `goodpush plan` write.
Buttons call /api/action, which runs the work in a background thread.

Security: by default it listens on 127.0.0.1 only. With --lan it listens on your
network so your phone can open it; every request then needs the access key printed
at start-up (kept in the machine folder's app_key, folders.Folders; first visit with ?key=... sets a cookie).
"""
from __future__ import annotations

import hmac
import json
import warnings
import mimetypes
import os
import secrets
import socket
import threading
import subprocess
import sys
import time
import traceback
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from .. import __version__, code_on_disk
from ..config import SETTINGS, clean_settings
from ..progress import Progress
from ..planning import DEFAULT_PLAN, LADDERS, Planner, PlanState, Season, WorkoutLibrary
from ..planning.season import EVENT_KINDS, PHASE_TYPES, Phase

STATIC = Path(__file__).with_name("static")
COOKIE = "goodpush_key"


# --- data -------------------------------------------------------------------------------
def _num(v, nd=1):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, nd)


def _clean(o):
    """NaN/inf -> None so the browser can parse the JSON."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return _clean(o.item())
    return o


def _read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


class Data:
    """Loads (and caches by file time) what the pages need."""

    def __init__(self, proj):
        self.proj = proj
        self._cache = {}

    TABLES = ("daily.parquet", "daily.csv.gz", "activities.parquet", "activities.csv.gz")

    def stamp(self) -> tuple:
        """the tables' file times: they change on every analysis (rides, days)"""
        return tuple(_mtime(self.proj.processed / f) for f in self.TABLES)

    def versions(self, stamp: tuple | None = None) -> dict:
        """what a page compares to know its data is old (a ride added, a re-plan): cheap, so /api/job sends it
        on every poll and a page that missed a job's end (a window macOS throttled, ...) still catches up.
        `stamp`: the tables a state was made from (not newer ones written since)."""
        return {"plan_version": _mtime(self.proj.processed / "plan_state.json"),   # changes on every re-plan
                "data_version": max(stamp or self.stamp())}   # ... and on every analysis (rides, days)

    def _tables(self):
        from ..io import store
        base = self.proj.processed
        stamp = self.stamp()
        if self._cache.get("stamp") != stamp:
            daily = store.load(base, "daily")
            acts = store.load(base, "activities")
            if daily is not None:
                daily.index = pd.to_datetime(daily.index)
            if acts is not None:
                acts["date"] = pd.to_datetime(acts["date"])
            self._cache = {"stamp": stamp, "daily": daily, "acts": acts}
        return self._cache["daily"], self._cache["acts"]

    def state(self) -> dict:
        proj = self.proj
        res = _read_json(proj.reports / "results.json", {}) or {}
        plan = _read_json(proj.reports / "plan" / "plan.json", {}) or {}
        pstate = _read_json(proj.processed / "plan_state.json", {}) or {}
        daily, acts = self._tables()
        out = {"name": proj.cfg["athlete"].get("name") or "", "generated": res.get("generated"),
               "plan": plan, "levels": pstate.get("levels", {}),
               "plan_cfg": {**_plan_defaults(), **proj.cfg.get("plan", {})},
               "power": _pick(res.get("power", {}), ["ftp_now", "ftp_peak", "ftp_peak_date", "wkg_now",
                                                     "wkg_peak", "cp_90d", "wprime_90d_kj", "phenotype",
                                                     "phenotype_reason", "bests_90d", "bests_all"]),
               "load": res.get("load", {}), "athlete": res.get("athlete", {}),
               "thresholds": res.get("thresholds", {}),
               "season_meta": {"types": {k: {"label": v[0], "build_weeks": v[1], "about": v[2]}
                                         for k, v in PHASE_TYPES.items()}, "kinds": EVENT_KINDS},
               "insights": _insights(res), "notes": res.get("notes", []),
               **self.versions(self._cache["stamp"]),
               "about": {"version": __version__, "installed": bool(getattr(sys, "frozen", False))}}   # Help: issue forms
        season = Season.load(proj.root)            # what you entered, even before the plan has caught up
        day = pd.Timestamp.now(tz=proj.tz).tz_localize(None).normalize()
        out["season"] = {"phases": [vars(p) for p in season.phases], "events": [vars(e) for e in season.events],
                         **Planner.suggestion(season, pstate, day)}     # also before the first plan
        from ..planning.builder import MyWorkouts
        out["my_workouts"] = MyWorkouts(proj.root).load()
        if daily is not None and len(daily):
            out["today"] = _today(daily)
            out["series"] = _series(daily)
            out["pmc"] = _series(daily, days=None, cols=("ctl", "atl", "tsb", "tss", "hours"))   # the fitness chart: every day
            if acts is not None and len(acts):
                out["pmc"].update(_day_rides(acts, daily.index))            # click a day -> its ride; its type
            out["history"] = _history(daily)
        if acts is not None and len(acts):
            out["rides"] = _rides(acts)
            out["last_ride"] = str(acts["date"].max().date())
        out["ladders"] = WorkoutLibrary.level_titles(pstate.get("format_offsets"))
        tab = proj.processed / "alpha_rides.csv"
        if tab.exists():
            try:
                at = pd.read_csv(tab, parse_dates=["date"]).sort_values("date")
                at = at[at["date"] >= at["date"].max() - pd.Timedelta(days=400)]
                cols = ["date", "a1_warm", "a1_warm_p", "a1_warm_exp", "a1_warm_z", "a1_warm_hr", "a1_warm_hr_z",
                        "a1_session_z", "hrvt1_28d", "indoor", "id"]
                rows = [{c: (str(r["date"].date()) if c == "date" else _jsonable(r.get(c))) for c in cols if c in at}
                        for _, r in at.iterrows()]
                last = next((r for r in reversed(rows) if r.get("a1_warm_z") is not None), None)
                out["alpha"] = {"rides": rows, "latest": last}
            except Exception:  # noqa: BLE001
                pass
        phys = _physio(proj.processed)
        if phys:
            out["physio"] = phys
        from ..models.physio import TrainingResponse
        resp = _read_json(proj.processed / TrainingResponse.FILE, None)
        if resp:
            out["response"] = resp
        from ..models import markers as M
        learned = M.load(proj.processed)
        if learned:
            out["markers"] = {k: learned.get(k) for k in ("analyzed", "episodes", "confirmed_episodes",
                                                          "chosen", "combined", "candidates", "fallback",
                                                          "base_rate", "episode_list")}
            if daily is not None and len(daily):
                try:
                    out["marker_status"] = M.status(daily, learned)
                except Exception:  # noqa: BLE001
                    out["marker_status"] = {}
        lo = M.load(proj.processed, "overreaching")
        if lo:
            out["markers_over"] = {k: lo.get(k) for k in ("analyzed", "episodes", "confirmed_episodes", "chosen",
                                                          "combined", "candidates", "fallback", "base_rate",
                                                          "window_days", "min_episodes")}
            if lo.get("chosen") and daily is not None and len(daily):
                try:
                    out["over_status"] = M.status(daily, lo)
                except Exception:  # noqa: BLE001
                    out["over_status"] = {}
        from ..sync import garmin_api
        from ..sync.google_health import GoogleHealth
        out["garmin_api"] = {**garmin_api.key_info(), "auto_hours": _auto_hours(proj)}
        out["google_health"] = GoogleHealth.info()
        from ..sync import incyclist
        out["incyclist"] = incyclist.status(proj.cfg)
        from ..sync.intervals_icu import IntervalsIcu
        out["intervals_icu"] = IntervalsIcu.info()
        try:
            from ..cli import _downloads_dir
            dl = str(_downloads_dir(proj))
        except Exception:  # noqa: BLE001
            dl = ""
        out["import_guide"] = _import_guide(proj, acts)
        out["settings"] = {"schema": SETTINGS, "values": _settings_values(proj),
                           "auto_sync_hours": _auto_hours(proj), "downloads": dl,
                           "root": str(proj.root)}
        raw = proj.path("garmin") / "api" / "raw"
        days = sorted(raw.glob("????-??-??.json")) if raw.exists() else []
        if days:
            t = max(p.stat().st_mtime for p in days)
            out["garmin_import"] = {"latest_day": days[-1].stem,
                                    "at": pd.Timestamp(t, unit="s", tz="UTC").tz_convert(proj.tz)
                                    .strftime("%Y-%m-%d %H:%M")}
        return out


HISTORY_SHORT_DAYS = 150          # rides only this far back and no account export: show the guide to bringing in the rest


def _import_guide(proj, acts: pd.DataFrame | None) -> dict:
    """How much history there is, for the Today page's guide: the first ride and whether an account export
    was imported (the export itself arrives by drag and drop: save_upload)."""
    first = pd.to_datetime(acts["date"]).min() if acts is not None and len(acts) else None
    imported = (proj.path("garmin") / "DI_CONNECT").is_dir() or (proj.path("strava") / "activities.csv").exists()
    short = first is None or first >= pd.Timestamp.now().normalize() - pd.Timedelta(days=HISTORY_SHORT_DAYS)
    return {"first": str(first.date()) if first is not None else None, "show": short and not imported}


CODE = code_on_disk()              # the code this app runs


def code_stale() -> bool:
    """goodpush's code on disk is newer than this app runs (updated since it started): it needs a restart."""
    return code_on_disk() != CODE


RIDE_STARTUP_S = 20.0              # a ride window that dies this fast didn't start: its log is the error


class RideLauncher:
    """`goodpush ride` for a session, started from the app as its own process (its overlay and the trainer
    are on this computer), as a job that watches it start. One at a time; output in ride_launch.log (Project.logs).
    The process only saves the ride (--save-only) and ends with its window, so "a ride is open" means one is:
    `saved` (an app job: ingest, link, re-plan) is called with the ride it saved."""
    LOG = "ride_launch.log"
    proc = None

    @staticmethod
    def command(proj, key: str = "") -> list[str]:
        """`goodpush ride` for the session `key` (the installed app runs itself; see launcher.command)."""
        from ..launcher import command
        return command("--root", str(proj.root), "ride", "--save-only", *(["--session", key] if key else []))

    @classmethod
    def start(cls, proj, key: str = "", log=print, saved=lambda ride_id: None) -> None:
        if cls.proc is not None and cls.proc.poll() is None:
            raise ValueError("A ride is already open – finish that one first.")
        path = proj.logs / cls.LOG
        with open(path, "w", encoding="utf-8") as out:
            from ..launcher import child_env
            cls.proc = subprocess.Popen(cls.command(proj, key), cwd=proj.root, stdout=out, stderr=subprocess.STDOUT,
                                        start_new_session=True, env=child_env())
        try:
            code = cls.proc.wait(RIDE_STARTUP_S)
        except subprocess.TimeoutExpired:
            log("Ride window open: ride, then save it there (it's added to your rides and the plan updates).")
            threading.Thread(target=cls.watch, args=(cls.proc, path, saved), name="ride-watch", daemon=True).start()
            return
        tail = [x for x in path.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()][-3:]
        raise ValueError(f"The ride didn't start (exit {code}): " + (" ".join(tail) or "no output"))

    @staticmethod
    def watch(proc, log_path: Path, saved) -> None:
        """once the ride window closes: `saved` with the ride it saved (its log's last SAVED line), if any."""
        from ..cli import SAVED
        proc.wait()
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        path = next((x[len(SAVED):] for x in reversed(lines) if x.startswith(SAVED)), None)
        if path:
            saved(Path(path.strip()).stem)


class DailyRanges:
    """[physio] ranges_every_days: the model's 5-95 % ranges re-made by `goodpush physio --refresh` (PhysioRun.refresh)
    in its own process at below-normal priority, on the home computer only. The best fit follows every ride (about a
    second); the ranges move slowly, so a refresh is due (`is_due`) once they are EVERY_DAYS data days older than the
    fit, or sooner when today's best-fit CP, LT1 or MAP has left its stored 5-95 % range (a test or a maxed effort
    moved the fit). Nothing the app serves waits for it: `tick` (the app's minute loop, never a page) starts it once a
    page has been served, at most once per data day, and never while a job, a ride or another NUTS run
    (PhysioRun.LOCK) is going on. Pages keep using the mode and the stored draws and read the new ranges once the
    refresh has saved them (it replaces its files whole); then the plan is re-made (its forecast follows the fit).
    Output in physio_refresh.log (Project.logs: this computer's), first line the data day it was for. A failed or stopped refresh changes
    nothing: the ranges keep their date."""
    EVERY_DAYS = 7                   # [physio] ranges_every_days when not set (0 = never)
    WATCHED = ("cp", "lt1", "map")   # best-fit numbers that call a refresh at once when outside their stored range
    LOG = "physio_refresh.log"
    FOR = "refresh for "

    def __init__(self, jobs, home_here, project, full=None):
        """home_here(): this is the athlete's home computer; project(): the project now (settings may change);
        full: the Model page's full run (FullModelRun), never refreshed over."""
        self.jobs, self.home_here, self.project, self.full = jobs, home_here, project, full
        self.proc = None
        self.served = False

    def page_served(self) -> None:
        self.served = True

    @classmethod
    def every_days(cls, cfg: dict) -> int:
        """[physio] ranges_every_days; without it the old switch (before 2026-11-12): daily_ranges = false = never."""
        physio = cfg.get("physio") or {}
        if "ranges_every_days" in physio:
            return max(int(physio["ranges_every_days"] or 0), 0)
        return cls.EVERY_DAYS if physio.get("daily_ranges", True) else 0

    @classmethod
    def is_due(cls, day: str, drawn: str, every_days: int, latest: pd.Series | None) -> bool:
        """day: the fit's data day; drawn: its draws' ("" none: a full run is needed, never a refresh); latest: the
        stored states' last row (best fit + _p05 / _p95), None when unread."""
        if every_days <= 0 or not drawn or drawn >= day:
            return False
        if (pd.Timestamp(day) - pd.Timestamp(drawn)).days >= every_days:
            return True
        if latest is None:
            return False
        x = latest.reindex([c + e for c in cls.WATCHED for e in ("", "_p05", "_p95")]).to_numpy(float).reshape(-1, 3)
        return bool(np.any((x[:, 0] < x[:, 1]) | (x[:, 0] > x[:, 2])))    # NaN (no range) compares False

    def due(self, proj) -> str | None:
        """the data day to refresh for (is_due), unless a refresh was already started for it."""
        try:
            from ..models.physio import PhysioRun
            with np.load(proj.processed / PhysioRun.FIT, allow_pickle=False) as f:
                day, drawn = str(f["day"]), str(f["draws_day"])
        except (ImportError, OSError, KeyError, ValueError):     # no fit (or one from before 2026-11-10), no jax
            return None
        try:
            states = pd.read_parquet(proj.processed / PhysioRun.STATES)
            latest = states.iloc[-1] if len(states) else None
        except (OSError, ValueError):
            latest = None
        if self.tried(proj) == day or not self.is_due(day, drawn, self.every_days(proj.cfg), latest):
            return None
        return day

    def tried(self, proj) -> str:
        """the data day the last refresh was started for ("" none)."""
        try:
            first = (proj.logs / self.LOG).read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            return ""
        return first[len(self.FOR):] if first.startswith(self.FOR) else ""

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def idle(self, proj) -> bool:
        """no job, ride, refresh, full model run (FullModelRun) or NUTS run going on."""
        if self.running() or self.jobs.status()["running"] or (self.full is not None and self.full.running()):
            return False
        if RideLauncher.proc is not None and RideLauncher.proc.poll() is None:
            return False
        try:
            from ..models.physio import PhysioRun
            return not PhysioRun.nuts_running(proj.processed)
        except ImportError:
            return False

    def tick(self) -> bool:
        """after a refresh that saved: re-plan; then start one if it is due (True when it did)."""
        proj = self.project()
        if self.proc is not None and self.proc.poll() is not None:
            saved, self.proc = self.proc.returncode == 0, None
            if saved and (proj.cfg.get("plan") or {}).get("enabled") and forecast_stale(proj.processed):
                self.jobs.start("Updating the plan (new model ranges)", lambda log: _plan_only(proj, log), queue=True)
        if not (self.served and self.every_days(proj.cfg) and self.home_here()):
            return False
        day = self.due(proj)
        if day is None or not self.idle(proj):
            return False
        self.start(proj, day)
        return True

    def start(self, proj, day: str) -> None:
        from ..launcher import child_env, command
        with open(proj.logs / self.LOG, "w", encoding="utf-8") as out:
            out.write(f"{self.FOR}{day}\n")
            out.flush()
            self.proc = subprocess.Popen(command("--root", str(proj.root), "physio", "--refresh"), cwd=proj.root,
                                         stdout=out, stderr=subprocess.STDOUT, env=child_env(), **self.lower())

    @staticmethod
    def lower() -> dict:
        """Popen arguments for below-normal priority: the computer stays responsive while it samples."""
        if sys.platform == "win32":
            return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
        import functools
        return {"preexec_fn": functools.partial(os.nice, 10)}


class FullModelRun:
    """The Model page's Run button: a full NUTS run (PhysioRun.FULL_RUN: new 5-95 % ranges) and then the backtest
    (`goodpush physio --backtest`: the page's forecast error), one after the other, each in its own process at
    below-normal priority (DailyRanges.lower). The backtest only follows a NUTS run that finished. Stop ends the
    current process: a NUTS run goes on from its checkpoint next time. Output in physio_full_run.log (Project.logs).
    Not started while a job, a ride or another NUTS run (PhysioRun.LOCK: a terminal's or a refresh) is going on;
    a run outlives an app restart (the new app sees its lock), but its backtest then isn't started."""
    LOG = "physio_full_run.log"
    STAGES = (("nuts", "Sampling the model (NUTS)"), ("backtest", "Backtest"))
    TAIL = 4096                      # bytes of the log read for its last line

    def __init__(self, jobs, project):
        self.jobs, self.project = jobs, project
        self.proc = None
        self.stage = ""              # STAGES key now ("" none)
        self.started = 0.0           # this run's start (epoch s)
        self.stage_started = 0.0
        self.stopped = False
        self.result = None           # the last run's end: {"ok", "say", "t"}

    @staticmethod
    def argv(stage: str) -> tuple[str, ...]:
        from ..models.physio import PhysioRun
        return tuple(PhysioRun.FULL_RUN.split()[1:]) if stage == "nuts" else ("physio", "--backtest")

    def running(self) -> bool:
        return bool(self.stage)

    @staticmethod
    def file_day(path: Path) -> str | None:
        """when a check's result was written (YYYY-MM-DD; None: not there)."""
        try:
            return time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))
        except OSError:
            return None

    def busy(self, proj) -> str:
        """why a run can't start now ("" it can)."""
        from ..models.physio import PhysioRun
        if self.running():
            return "A full model run is going on already."
        if PhysioRun.nuts_running(proj.processed):
            return "Another sampling run is going on (a terminal's, or the weekly ranges refresh): wait for it to end."
        if self.jobs.status()["running"]:
            return "Busy with another job: try again when it's done."
        if RideLauncher.proc is not None and RideLauncher.proc.poll() is None:
            return "Not during a ride."
        return ""

    def start_fresh(self, proj):
        """Settings → Data → Fit the model from scratch (`fresh_fit`) as a job; raises ValueError (busy's reason)
        when it can't start now - never while a NUTS run (its checkpoint) or another job goes on."""
        why = self.busy(proj)
        if why:
            raise ValueError(why)
        return self.jobs.start("Fitting the model from scratch", lambda log: self.fresh_fit(proj, log))

    @staticmethod
    def fresh_fit(proj, log) -> None:
        """`goodpush physio --fresh` (the posterior mode from the priors, not the last fit; drops a NUTS run's
        checkpoint and the stored draws, so no ranges until the Model page's next full run), then the analysis, plan
        and reports from that fit."""
        from ..models.physio import PhysioRun
        run = PhysioRun(proj)
        run.same_code()
        log("Fitting the model from its priors (not the last fit) ...")
        run.fit(fresh=True, progress=log)
        _reanalyse(proj, log)

    def start(self) -> None:
        """raises ValueError (busy's reason) when it can't start now."""
        proj = self.project()
        why = self.busy(proj)
        if why:
            raise ValueError(why)
        self.stage, self.started, self.stopped, self.result = self.STAGES[0][0], time.time(), False, None
        threading.Thread(target=self.run, args=(proj,), name="goodpush-full-model-run", daemon=True).start()

    def run(self, proj) -> None:
        """the stages in turn; the next only after one that finished (exit code 0)."""
        from ..launcher import child_env, command
        say = "done"
        try:
            with open(proj.logs / self.LOG, "w", encoding="utf-8") as out:
                for stage, label in self.STAGES:
                    self.stage, self.stage_started = stage, time.time()
                    out.write(f"== {label}: goodpush {' '.join(self.argv(stage))}\n")
                    out.flush()
                    self.proc = subprocess.Popen(command("--root", str(proj.root), *self.argv(stage)), cwd=proj.root,
                                                 stdout=out, stderr=subprocess.STDOUT, env=child_env(),
                                                 **DailyRanges.lower())
                    code = self.proc.wait()
                    if stage == "nuts" and code == 0:
                        self.replan(proj)
                    if self.stopped or code != 0:
                        say = (f"stopped during: {label}" if self.stopped
                               else f"{label} failed (exit {code}): see the log")
                        break
        except OSError as e:
            say = f"couldn't start: {e}"
        finally:
            self.result = {"ok": say == "done", "say": say, "t": time.time()}
            self.stage, self.proc = "", None

    def replan(self, proj) -> None:
        """new ranges: the plan's forecast follows the fit (as after DailyRanges' refresh)."""
        if (proj.cfg.get("plan") or {}).get("enabled") and forecast_stale(proj.processed):
            self.jobs.start("Updating the plan (new model ranges)", lambda log: _plan_only(proj, log), queue=True)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.stopped = True
            self.proc.terminate()

    def last_line(self, proj) -> str:
        """the log's last non-empty line (progress bars rewrite theirs with \\r)."""
        try:
            with open(proj.logs / self.LOG, "rb") as f:
                f.seek(max(f.seek(0, os.SEEK_END) - self.TAIL, 0))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        parts = [p.strip() for p in tail.replace("\r", "\n").split("\n")]
        return next((p for p in reversed(parts) if p), "")

    def status(self) -> dict:
        """for the Model page: the stage now, its times, the log's last line, the last run's end, and why it
        couldn't start now."""
        proj = self.project()
        now = time.time()
        return {"stage": self.stage, "label": dict(self.STAGES).get(self.stage, ""),
                "step": next((i + 1 for i, (k, _) in enumerate(self.STAGES) if k == self.stage), None),
                "steps": len(self.STAGES), "elapsed": round(now - self.started) if self.stage else None,
                "stage_elapsed": round(now - self.stage_started) if self.stage else None,
                "last": self.last_line(proj) if self.stage or self.result else "", "result": self.result,
                "busy": "" if self.stage else self.busy(proj), "log": str(proj.logs / self.LOG),
                "other": None if self.stage else self.other(proj)}

    @staticmethod
    def other(proj) -> dict | None:
        """a NUTS run started elsewhere (a terminal's, or the weekly ranges refresh): its kind, start and last
        progress line (PhysioRun.PROGRESS), with its age in seconds; None when there is none."""
        from ..models.physio import PhysioRun
        p = PhysioRun.nuts_progress(proj.processed)
        if not p:
            return None
        now = time.time()
        return {"kind": p.get("kind", ""), "line": p.get("line", ""), "elapsed": round(now - float(p.get("started", now))),
                "quiet": round(now - float(p.get("t", now)))}


UPLOAD_MAX = 20 * 1024 ** 3       # bytes: an account export with years of data is a few GB
UPLOAD_CHUNK = 1 << 20


def save_upload(proj, rfile, length: int, name: str) -> tuple[Path, str]:
    """An account export dropped on the page: streamed to data/uploads/ (in chunks: it can be GBs) and
    checked to be a Strava or Garmin export. Returns (path, kind); raises ValueError otherwise."""
    import re
    from ..importer import export_kind
    if not 0 < length <= UPLOAD_MAX:
        raise ValueError("send the export's .zip (up to 20 GB)")
    d = proj.processed.parent / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / (re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name) or "export.zip")
    part, left = dest.with_name(dest.name + ".part"), length
    with open(part, "wb") as f:
        while left:                                    # chunks of a big file
            chunk = rfile.read(min(left, UPLOAD_CHUNK))
            if not chunk:
                break
            f.write(chunk)
            left -= len(chunk)
    kind = None if left else export_kind(part)
    if kind is None:
        part.unlink(missing_ok=True)
        raise ValueError("the upload stopped part way" if left else
                         "that isn't a Strava or Garmin export (send the .zip as you downloaded it)")
    part.replace(dest)
    return dest, kind


ACT_FIELDS = ["id", "name", "type", "sport", "indoor", "workout", "type_name", "duration_s", "moving_s",
              "distance_m", "elev_gain", "np_used", "avg_power", "max_power", "if_", "tss", "tss_source",
              "avg_hr", "max_hr", "work_kj", "ef", "decoupling", "avg_cadence", "ftp", "gear",
              "description", "private_note", "avg_temp", "calories", "also", "power_ignored"]


def _jsonable(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, np.floating, float, int)):
        return _num(v, 2)
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v)
    return "" if s == "nan" else s


def activities_list(acts: pd.DataFrame, q: str = "", sport: str = "ride", offset: int = 0, limit: int = 50):
    a = acts
    if sport and "sport" in a:
        a = a[a["sport"].astype(str).eq(sport)]
    if q:
        ql = q.lower()
        txt = (a["name"].astype(str) + " " + a.get("workout", "").astype(str) + " " +
               a["date"].dt.strftime("%Y-%m-%d"))
        a = a[txt.str.lower().str.contains(ql, regex=False)]
    a = a.sort_values("start_utc" if "start_utc" in a else "date", ascending=False)
    rows = []
    for _, x in a.iloc[offset: offset + limit].iterrows():
        rows.append({"id": str(x["id"]).removesuffix(".0"), "date": str(x["date"].date()), "name": str(x.get("name") or ""),
                     "workout": _jsonable(x.get("workout")) or "", "min": _num(x.get("duration_s", np.nan) / 60, 0),
                     "km": _num(x.get("distance_m", np.nan) / 1000, 1), "tss": _num(x.get("tss"), 0),
                     "np": _num(x.get("np_used"), 0), "if": _num(x.get("if_"), 2), "hr": _num(x.get("avg_hr"), 0),
                     "indoor": bool(x.get("indoor")) if x.get("indoor") == x.get("indoor") else False})
    return {"total": int(len(a)), "offset": offset, "rows": rows}


def activity_detail(proj, acts: pd.DataFrame, aid: str, res: dict) -> dict:
    sid = acts["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    a = acts[sid.eq(aid)]
    if not len(a):
        raise KeyError(f"no activity {aid}")
    x = a.iloc[0]
    out = {k: _jsonable(x.get(k)) for k in ACT_FIELDS if k in a}
    out["id"] = aid
    out["date"] = str(x["date"].date())
    st = x.get("start_local")
    out["start"] = str(pd.Timestamp(st))[:16] if st is not None and st == st else out["date"]
    out["zones_power"] = [_num(x.get(f"pz{i}_s"), 0) for i in range(1, 8)]
    out["zones_hr"] = [_num(x.get(f"hz{i}_s"), 0) for i in range(1, 8)]
    from ..metrics.power import MMP_DURATIONS
    out["mmp"] = {str(d): _num(x.get(f"mmp_{d}"), 0) for d in MMP_DURATIONS if f"mmp_{d}" in a}
    out["mmp_90d"] = {k: (v or {}).get("w") for k, v in ((res.get("power") or {}).get("bests_90d") or {}).items()}
    zs = res.get("zones") or {}
    out["zone_limits"] = zs
    # neighbours for prev / next
    rides = acts[acts["sport"].astype(str).eq(str(x.get("sport")))].sort_values("date")
    ids = rides["id"].astype(str).str.replace(r"\.0$", "", regex=True).tolist()
    i = ids.index(aid) if aid in ids else -1
    out["prev"] = ids[i - 1] if i > 0 else None
    out["next"] = ids[i + 1] if 0 <= i < len(ids) - 1 else None
    # streams (1 Hz, saved by ingest)
    from ..io import streamstore
    z = streamstore.read(proj.processed, aid, "ride")          # one hyperslab from streams.h5
    ftp = float(x.get("ftp") or 0) or 200.0
    lthr = float((res.get("athlete") or {}).get("lthr") or 0) or None
    if z:
        p = z["power"].astype(float)
        stopped = ride_stops(z)
        from ..models.stamina import RideStamina
        out["streams"] = ride_streams(z, 0, p.size, stopped=stopped,
                                      stamina=RideStamina.for_day(proj.processed, x.get("date"), p))
        run = np.flatnonzero(np.diff(np.r_[0, stopped.astype(int), 0])).reshape(-1, 2)
        out["stops"] = [[int(a), int(b - a)] for a, b in run]      # [ride second, seconds]: the charts leave them out
        from ..io import store
        from ..models.intervals import IntervalLog, RideIntervals
        from ..planning.intensity import IntensityMap
        knots, w_prime = IntensityMap.for_day(proj.processed, x.get("date"), ftp)   # your LT1 / CP / MAP that day
        rhr = float(IntervalLog.rest_hr(store.load(proj.processed, "daily"), [x.get("date")])[0]) if lthr else None
        out["intervals"] = {**RideIntervals(p, knots, z["heart_rate"], w_prime).summary(  # steps, efforts, sets
                                rhr if rhr == rhr else None, lthr),
                            "anchors": {k: round(float(v)) for k, v in zip(("lt1", "cp", "map"), knots[1:4])},
                            "w_prime": round(w_prime) if w_prime else None}
        out["lthr"] = lthr
        out["plan"] = planned_for_ride(proj, aid, knots, p.size, int(np.argmax(~stopped)) if (~stopped).any() else 0)
        if out["plan"] and out["plan"]["type"] == "ramp":
            out["ramp"] = ramp_test_detail(proj, acts, aid, res)
        if out["plan"] and out["plan"]["type"] == "lsct":
            out["lsct"] = lsct_detail(proj, acts, aid, out["plan"]["key"])
    # DFA alpha1 if recorded: goodpush's own from the beat intervals, or alphaHRV's (alpha.extract)
    z = streamstore.read(proj.processed, aid, "alpha")
    if z:
        from ..alpha import BIN_S, INDEX
        idx = pd.read_csv(proj.processed / INDEX, dtype=str, usecols=lambda c: c in ("id", "source"))
        src = idx.loc[idx["id"].eq(aid), "source"] if "source" in idx else pd.Series(dtype=str)
        s = str(src.iloc[-1]) if len(src) and src.iloc[-1] == src.iloc[-1] else "alphaHRV"
        out["alpha_source"] = s
        n = len(z["t"])
        b = max(1, int(np.ceil(n / 900)))
        m = n // b * b

        def bk(arr):
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = np.nanmean(arr[:m].astype(float).reshape(-1, b), axis=1) if m else []
            return [_num(q, 2) for q in v]
        out["alpha"] = {"t_min": [round(float(t) / 60, 2) for t in z["t"][:m:b]], "a1": bk(z["a1"]),
                        "p": bk(z["p"]), "hr": bk(z["hr"]), "bucket_s": b * BIN_S}
        tab = proj.processed / "alpha_rides.csv"
        if tab.exists():
            at = pd.read_csv(tab, dtype={"id": str})
            row = at[at["id"].str.replace(r"\.0$", "", regex=True).eq(aid)]
            if len(row):
                from ..models.ramp import RampTest
                out["alpha_check"] = {k: _jsonable(v) for k, v in row.iloc[0].items()}
                out["alpha_check"]["ramp_lt2_loose"] = RampTest.lt2_check(row.iloc[:1]).iloc[0] or None
    # GPS track (outdoor rides)
    out["track"] = []
    fn = str(x.get("filename") or "")
    if fn and fn != "nan" and not out.get("indoor"):
        f = proj.path("strava") / fn
        if f.exists():
            try:
                from ..io.streams import read_track
                out["track"] = read_track(f)
            except Exception:  # noqa: BLE001 - a map is a nice-to-have
                out["track"] = []
    return out


def planned_for_ride(proj, aid: str, knots, seconds: int, start: int = 0) -> dict | None:
    """The planned session this ride counts for (linked or matched) and its steps on the ride's clock
    (SessionScorer.timeline from `start`, the first second ridden; minutes); None when it counts for none."""
    from ..planning import PlanState, SessionScorer
    st = PlanState.load(proj.processed / "plan_state.json")
    key = next((k for k, p in st["planned"].items()
                if str(p.get("ride") or (p.get("compliance") or {}).get("ride") or "") == aid), None)
    if key is None:
        return None
    p = st["planned"][key]
    c = p.get("compliance") or {}
    tl = SessionScorer.timeline(st.workout(key), knots, c.get("reps") or [], seconds, start)
    return {"key": key, "type": p.get("type"), "title": p.get("title") or "", "result": c.get("result"),
            "linked": bool(p.get("ride")),
            "steps": [[round(a / 60, 4), round(b / 60, 4), round(lo), round(hi), int(k)] for a, b, lo, hi, k in tl]}


def ramp_test_detail(proj, acts: pd.DataFrame, aid: str, res: dict) -> dict | None:
    """A ramp test's summary for its ride page: the steps, top, max HR, recovery and heart rate at your thresholds
    (now), and your ramp tests side by side - heart rate at today's FTP watts in each, so they compare."""
    from ..models.ramp import RampTest
    from ..planning import PlanState
    T = res.get("thresholds") or {}
    watts = {k: float((T.get(k.lower()) or {}).get("w") or 0) for k in ("LT1", "FTP", "LT2")}
    st = PlanState.load(proj.processed / "plan_state.json")
    sid = acts["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    linked = {str(p["ride"]): k for k, p in st["planned"].items() if p.get("type") == "ramp" and p.get("ride")}
    detected = set(sid[acts["ramp_top_w"].notna()]) if "ramp_top_w" in acts else set()
    rows, this = [], None
    for i in np.flatnonzero(sid.isin(set(linked) | detected).to_numpy()):     # your ramp tests: a handful
        x, rid = acts.iloc[i], sid.iloc[i]
        r = RampTest.of_ride(proj, x, st.workout(linked[rid])) if rid in linked else None
        s = r.summary(watts, float(x.get("ftp") or 0)) if r is not None else {}
        if rid == aid:
            this = s
        alpha = lambda c: _num(x.get(c), 0) if c in acts else None  # noqa: E731
        rows.append({"id": rid, "date": str(x["date"].date()), "this": rid == aid,
                     "top_w": s.get("top_w") or alpha("ramp_top_w"), "max_hr": s.get("max_hr") or alpha("ramp_max_hr"),
                     "hr_ftp": (s.get("hr_at") or {}).get("FTP", {}).get("hr"), "recovery_60": (s.get("recovery") or {}).get(60),
                     "lt1_w": alpha("ramp_lt1_w") or alpha("ramp_hr_lt1_w"), "lt2_w": alpha("ramp_lt2_w") or alpha("ramp_hr_lt2_w"),
                     "by_hr": alpha("ramp_lt1_w") is None and alpha("ramp_lt2_w") is None})
        if rid == aid:                                 # the thresholds this test gave (stored at analysis)
            this["thresholds"] = {k: {"w": alpha(f"ramp_{k}_w"), "hr": alpha(f"ramp_{k}_hr"),
                                      "hr_w": alpha(f"ramp_hr_{k}_w"), "at_hr": alpha(f"ramp_hr_{k}_hr")} for k in ("lt1", "lt2")}
    if not this:
        return None
    ath = res.get("athlete") or {}
    return {**this, "history": sorted(rows, key=lambda r: r["date"]), "ftp_w": round(watts["FTP"]) or None,
            "max_hr_now": _num(ath.get("max_hr"), 0), "max_hr_set": int(proj.cfg["athlete"].get("max_hr") or 0)}


def lsct_detail(proj, acts: pd.DataFrame, aid: str, key: str) -> dict | None:
    """An LSCT's stages for its ride page (power and heart rate over each stage's second half, heart-rate recovery)
    and your LSCTs side by side (the lsct_* columns stored at analysis)."""
    from ..models.lsct import COLUMNS, LSCT
    from ..planning import PlanState
    st = PlanState.load(proj.processed / "plan_state.json")
    sid = acts["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    row = acts[sid.eq(aid)]
    if not len(row):
        return None
    r = LSCT.of_ride(proj, row.iloc[0], st.workout(key))
    if r is None:
        return None
    has = acts[COLUMNS[0]].notna() if COLUMNS[0] in acts else pd.Series(False, index=acts.index)
    hist = [{"id": i, "date": str(d.date()), "this": i == aid, **{c: _num(x.get(c), 0) for c in COLUMNS}}
            for i, d, (_, x) in zip(sid[has], acts.loc[has, "date"], acts[has].iterrows())]     # a handful
    return {"stages": [{"name": s.name, "share": s.share, "seconds": round(s.seconds), "power": _num(s.power, 0),
                        "hr": _num(s.hr, 0)} for s in r.stages.itertuples()],
            "recovery_60": r.recovery(), "history": sorted(hist, key=lambda h: h["date"])}


STOP_W, STOP_MS = 20.0, 0.447       # a stop on the ride charts: under 20 W and under 1 mph (no speed: power alone)


def ride_stops(z: dict) -> np.ndarray:
    """The seconds of a ride you were stopped (STOP_W / STOP_MS), as a mask."""
    return ((np.nan_to_num(np.asarray(z["power"], float)) < STOP_W)
            & (np.nan_to_num(np.asarray(z["speed"], float)) < STOP_MS))


def ride_streams(z: dict, a: int, b: int, points: int = 1200, stopped: np.ndarray | None = None,
                 stamina=None) -> dict:
    """Chart streams for seconds [a, b) of a ride without the `stopped` ones: means over equal buckets, at
    most `points` of them (1-s data once the stretch is short enough). t_min is riding time from the start.
    stamina (RideStamina of the whole ride): its current / potential lines too, in %."""
    keep = np.ones(np.asarray(z["power"]).size, bool) if stopped is None else ~stopped
    idx = a + np.flatnonzero(keep[a:b])
    bs = max(1, int(np.ceil(idx.size / points)))
    m = idx.size // bs * bs
    ch = {"power": z["power"], "hr": z["heart_rate"], "cadence": z["cadence"], "speed": z["speed"] * 3.6,
          "altitude": z["altitude"]}
    if stamina is not None:
        ch["stamina"], ch["potential"] = stamina.current() * 100, stamina.potential() * 100
    X = np.vstack([np.asarray(v, float)[idx[:m]] for v in ch.values()]).reshape(len(ch), -1, bs)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        M = np.nanmean(X, axis=2) if m else np.zeros((len(ch), 0))
    out = {k: [_num(q, 1) for q in row] for k, row in zip(ch, M)}
    ride_s = np.cumsum(keep) - 1                        # each kept second's place on the riding clock
    out["t_min"] = np.round((ride_s[idx[:m:bs]] + bs / 2) / 60, 4).tolist()
    out["bucket_s"] = bs
    return out


def activity_window(proj, acts: pd.DataFrame, aid: str, t0: float, t1: float) -> dict:
    """Seconds [t0, t1) of a ride: its statistics (a selection) and chart streams (the zoomed view)."""
    from ..io import streamstore
    from ..metrics.power import summarize_stream
    sid = acts["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    a = acts[sid.eq(aid)]
    z = streamstore.read(proj.processed, aid, "ride") if len(a) else None
    if not z:
        raise KeyError(f"no ride data for {aid}")
    n = z["power"].size
    i0, i1 = int(max(0, min(t0, n))), int(max(0, min(t1, n)))
    if i1 - i0 < 5:
        raise ValueError("select at least 5 seconds")
    sl = {k: np.asarray(z[k][i0:i1], float) for k in ("power", "heart_rate", "cadence", "speed", "altitude")}
    s = summarize_stream({"n": i1 - i0, **sl, "moving": np.asarray(z["moving"][i0:i1], bool)})
    ftp = float(a.iloc[0].get("ftp") or 0) or 200.0
    np_w = s["np"]
    alt = sl["altitude"][:(i1 - i0) // 10 * 10].reshape(-1, 10)        # 10-s means against GPS noise
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        alt = np.nanmean(alt, axis=1) if alt.size else alt
    dist = float(np.nansum(sl["speed"]))
    hr = sl["heart_rate"][np.isfinite(sl["heart_rate"])]
    stats = {"from_s": i0, "to_s": i1, "seconds": i1 - i0, "moving_s": s["moving_s"],
             "avg_power": _num(s["avg_power"], 0), "np": _num(np_w, 0), "max_power": _num(s["max_power"], 0),
             "work_kj": _num(s["work_kj"], 0), "if": _num(np_w / ftp, 2),
             "tss": _num((i1 - i0) * np_w * (np_w / ftp) / (ftp * 3600) * 100, 0), "ftp": _num(ftp, 0),
             "pct_ftp": _num(s["avg_power"] / ftp * 100, 0),
             "avg_hr": _num(s["avg_hr"], 0), "max_hr": _num(hr.max(), 0) if hr.size else None,
             "avg_cadence": _num(s["avg_cadence"], 0), "decoupling": _num(s["decoupling"], 1),
             "distance_m": _num(dist, 0) if dist > 0 else None,
             "speed_kmh": _num(dist / s["moving_s"] * 3.6, 1) if dist > 0 and s["moving_s"] else None,
             "climb_m": _num(np.nansum(np.clip(np.diff(alt), 0, None)), 0) if alt.size > 1 and np.isfinite(alt).any() else None}
    za = streamstore.read(proj.processed, aid, "alpha")      # DFA alpha1 (alphaHRV), when recorded
    if za:
        t, a1 = np.asarray(za["t"]), np.asarray(za["a1"], float)
        a1 = a1[(t >= i0) & (t < i1) & np.isfinite(a1)]
        if a1.size:
            stats.update(alpha1=_num(a1.mean(), 2), alpha1_min=_num(a1.min(), 2), alpha1_max=_num(a1.max(), 2))
    from ..models.stamina import RideStamina
    sta = RideStamina.for_day(proj.processed, a.iloc[0]["date"], z["power"])     # the whole ride: it builds up
    return {"stats": stats, "streams": ride_streams(z, i0, i1, stopped=ride_stops(z), stamina=sta)}


def _session_details(planned: dict, accepted: dict, ftp: float, processed=None) -> None:
    """Steps (watts), block layout and workout text for past planned days - the accepted version
    (kept 30 days), else rebuilt from its type and level - so the calendar can draw and open them."""
    from ..planning import Workout, WorkoutLibrary
    from ..planning.intensity import IntensityMap
    from ..report.plan_out import WHY, session_steps
    imap = IntensityMap.load(processed) if processed is not None else None      # your watts, when known
    ftp = imap.ftp if imap is not None else ftp
    for ds, p in planned.items():
        if "steps" in p or p.get("type") in (None, "rest"):
            continue
        a = accepted.get(ds)
        s = Workout.from_dict(ds, a) if a and a.get("type") == p.get("type") else \
            WorkoutLibrary.rebuild(pd.Timestamp(ds[:10]), p)
        s = imap.apply(s) if imap is not None else s
        p.update(**session_steps(s, ftp), why=WHY.get(s.type, ""), note=s.note)
        p.setdefault("tss", round(s.tss()))


def calendar_data(proj, daily: pd.DataFrame | None, acts: pd.DataFrame | None) -> dict:
    """The Calendar page: TSS for every day (actual + planned), every activity and every
    planned session, Monday of the first week to Sunday of the last planned week."""
    plan = _read_json(proj.reports / "plan" / "plan.json", {}) or {}
    pstate = _read_json(proj.processed / "plan_state.json", {}) or {}
    planned = {ds: dict(p) for ds, p in (pstate.get("planned") or {}).items()
               if p.get("status") != "moved"}            # moved away from a past day: shown only where it went
    for s in plan.get("sessions") or []:          # the current plan: TSS, steps in watts, files
        planned.setdefault(s.get("key") or s["date"], {}).update(
            {k: s.get(k) for k in ("type", "title", "minutes", "tss", "workout", "adjusted", "fit", "zwo", "icu", "steps",
                                   "blocks", "why", "note") if k in s})
    mine = pstate.get("manual_plan") or {}
    _session_details(planned, {**(pstate.get("accepted") or {}), **mine}, float(plan.get("ftp") or 0) or 200.0,
                     proj.processed)
    for k, p in planned.items():
        p["custom"] = k in mine and not mine[k].get("pinned")      # you moved / added / changed it
        p["locked"] = bool((mine.get(k) or {}).get("locked"))     # the plan keeps it exactly as it is
    for p in planned.values():
        if p.get("type") == "rest":
            p["tss"] = 0
    now = pd.Timestamp.now(tz=proj.tz).tz_localize(None).normalize()
    st = PlanState(pstate)
    for k, p in planned.items():               # interval sessions to come (the plan's, moved too): easier / harder
        if k[:10] >= str(now.date()) and p.get("type") in LADDERS:
            p["adjust"] = st.adjustment(k)
    tss = daily["tss"] if daily is not None and "tss" in daily else pd.Series(dtype=float)
    season = Season.load(proj.root)
    first = min([tss.index.min()] if len(tss) else [now])
    last = max([now, tss.index.max() if len(tss) else now] + [pd.Timestamp(d[:10]) for d in planned]
               + [p.end for p in season.phases])        # the season's blocks, planned or not
    first -= pd.Timedelta(days=first.day_of_week)
    last += pd.Timedelta(days=6 - last.day_of_week)
    days = pd.date_range(first, last, freq="D")
    actual = tss.reindex(days)
    actual[days > now] = np.nan
    pt = pd.DataFrame([(pd.Timestamp(k[:10]), p.get("tss")) for k, p in planned.items()], columns=["d", "tss"])
    plan_tss = (pt.dropna().groupby("d")["tss"].sum() if len(pt) else pd.Series(dtype=float)).reindex(days)
    out_acts = []
    if acts is not None and len(acts):
        a = acts.sort_values("start_utc" if "start_utc" in acts else "date")
        col = lambda c, fill="": a[c] if c in a else pd.Series(fill, index=a.index)
        f = pd.DataFrame({
            "date": a["date"].dt.strftime("%Y-%m-%d"), "id": a["id"].astype(str).str.replace(r"\.0$", "", regex=True),
            "name": col("name").fillna("").astype(str), "sport": col("sport").fillna("").astype(str),
            "workout": col("workout").fillna("").astype(str).replace("nan", ""),
            "min": (pd.to_numeric(col("duration_s", np.nan), errors="coerce") / 60).round(0),
            "tss": pd.to_numeric(col("tss", np.nan), errors="coerce").round(0),
            "indoor": col("indoor", False).fillna(False).astype(bool)})
        out_acts = f.astype(object).where(f.notna(), None).to_dict("records")
    rnd = lambda s: [None if v != v else round(float(v)) for v in s.to_numpy()]
    return {"start": str(first.date()), "today": str(now.date()), "tss": rnd(actual), "plan_tss": rnd(plan_tss),
            "acts": out_acts, "planned": planned, "proposal": plan.get("proposal") or [],
            "events": [vars(e) for e in season.events],
            "phases": [{"name": p.name, "type": p.type, "start": p.start, "end": str(p.end.date()), "id": p.id,
                        "weeks": p.weeks, "recovery_weeks": p.recovery_weeks, "hours": p.hours,
                        "build_weeks": p.build_weeks} for p in season.phases],
            "weeks": {w["week"]: w.get("tss_target") for w in plan.get("weeks") or []},
            "week_kind": {w["week"]: "taper" if w.get("taper") else "recovery"
                          for w in plan.get("weeks") or [] if w.get("taper") or w.get("recovery")},
            "hours_limited": [w["week"] for w in plan.get("weeks") or [] if w.get("hours_limited")],
            "push_weeks": [w["week"] for w in plan.get("weeks") or [] if w.get("push")],
            "ramp_flags": {w["week"]: [w["ramp_flag"], w.get("ramp_4wk")] for w in plan.get("weeks") or []
                           if w.get("ramp_flag")},
            "ramp_range": plan.get("ramp_range"), "away": pstate.get("away") or {}}


def _plan_defaults():
    return dict(DEFAULT_PLAN)


def _settings_values(proj) -> dict:
    """What the Settings page shows: each SETTINGS section's current values (the plan's with its defaults)."""
    cfg = proj.cfg
    out = {sec: {f["key"]: cfg.get(sec, {}).get(f["key"]) for f in spec["fields"]}
           for sec, spec in SETTINGS.items() if sec != "events"}
    out["plan"] = {k: {**_plan_defaults(), **cfg.get("plan", {})}.get(k) for k in out["plan"]}
    out["paths"]["downloads"] = cfg["paths"].get("downloads") or ""
    out["events"] = {"events": [e for e in cfg.get("events") or [] if "example" not in str(e.get("note", ""))]}
    out["physio"]["ranges_every_days"] = DailyRanges.every_days(cfg)     # (an old daily_ranges = false shows 0)
    return out


def _save_settings(proj, section: str, vals: dict) -> None:
    """Checked values into goodpush.toml: FTP history and life events replace what's there."""
    from .tomledit import set_table, set_tables, set_values
    path = proj.root / "goodpush.toml"
    if section == "events":
        set_tables(path, "events", vals.get("events", []))
        return
    if section == "ftp" and "history" in vals:
        set_table(path, "ftp.history", vals.pop("history"))
    if vals:
        set_values(path, section, vals)


def _pick(d, keys):
    return {k: d.get(k) for k in keys}


def _today(d: pd.DataFrame) -> dict:
    last = d.iloc[-1]
    well = d[["sleep_h", "rhr", "hrv"]].dropna(how="all")
    t = {"date": str(d.index[-1].date()),
         "wellness_date": str(well.index[-1].date()) if len(well) else None}
    row = d.loc[well.index[-1]] if len(well) else last
    for c in ("sleep_h", "sleep_score", "rhr", "hrv", "body_battery_max", "stress", "readiness",
              "hrv_base_low", "hrv_base_high", "hrv_weekly"):
        t[c] = _num(row.get(c))
    for c in ("sleep_h", "rhr", "hrv", "sleep_score"):
        t[c + "_base"] = _num(row.get(c + "_base"))
        t[c + "_z"] = _num(row.get(c + "_z"), 2)
    for c in ("ctl", "atl", "tsb", "monotony", "ftp"):
        t[c] = _num(last.get(c))
    return t


def _series(d: pd.DataFrame, days=400, cols=None) -> dict:
    x = d.iloc[-days:] if days else d
    cols = cols or ["ctl", "atl", "tsb", "tss", "ftp", "eq20_test", "ef", "rhr", "rhr_base", "hrv",
            "hrv_base_low", "hrv_base_high", "hrv_base", "sleep_h", "sleep_h_base", "sleep_score",
            "body_battery_max", "hours", "hrv_min", "hrv_q25", "hrv_median", "hrv_q75", "hrv_max", "hrv_n"]
    out = {"date": [str(i.date()) for i in x.index]}
    for c in cols:
        if c in x:
            out[c] = [_num(v, 2) for v in x[c].to_numpy()]
    return out


def _mtime(p) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def forecast_stale(processed) -> bool:
    """the physio fit is newer than the plan's forecast (refitted since the last plan, e.g. by the start-up warm-up
    or `goodpush physio`): the forecast continues another fit."""
    from ..models.physio import PhysioRun
    from ..planning.planner import PHYSIO_FORECAST
    return _mtime(processed / PhysioRun.STATES) > _mtime(processed / PHYSIO_FORECAST)


def _physio(processed) -> dict | None:
    """the physio model's daily capacities (with 5 / 95 % bands when there are posterior draws) and the
    plan's forecast, from processed/physio_states.parquet and physio_forecast.parquet (not when it belongs to an
    older fit: it would start somewhere else)."""
    from ..models.physio import PhysioRun
    from ..planning.planner import PHYSIO_FORECAST
    try:
        st = pd.read_parquet(processed / PhysioRun.STATES)
    except (OSError, ValueError):
        return None
    from ..models.physio import PhysioDiagnostics
    cols = [c for c in ("cp", "lt1", "map", "ftp", "w_prime", "fatigue", "form") for c in (c, c + "_p05", c + "_p95") if c in st]
    out = {"date": [str(i.date()) for i in st.index], **{c: [_num(v, 1) for v in st[c].to_numpy()] for c in cols}}
    if {"cp", "w_prime", "pmax", "fade"} <= set(st):      # 20-min power, fresh (as the Rider page's power over time)
        from ..physio import PowerCurve
        p20 = np.asarray(PowerCurve.power(1200.0, *(st[c].to_numpy(float) for c in ("cp", "w_prime", "pmax", "fade"))))
        out["p20"] = [_num(v, 1) for v in p20]
    diag =_read_json(processed / PhysioDiagnostics.FILE, {}) or {}
    out["assumed"] = diag.get("assumed") or []                          # said by the numbers
    out["draws_day"] = diag.get("draws_day") or diag.get("full_date") or None   # the ranges' run (pages: when older)
    from ..physio import CHAINS
    out["full_run"] = PhysioRun.FULL_RUN                                # Today's reminder when there are no ranges:
    out["full_run_cores"] = {"chains": CHAINS, "all": os.cpu_count() or CHAINS}   # a core per chain (numba: all, briefly)
    try:
        if forecast_stale(processed):
            raise OSError("forecast from an older fit")
        fc = pd.read_parquet(processed / PHYSIO_FORECAST)
        fc = fc[fc.index > st.index[-1]]
        out["forecast"] = {"date": [str(i.date()) for i in fc.index],
                           **{c: [_num(v, 1) for v in fc[c].to_numpy()] for c in ("cp", "lt1", "map", "ftp", "fatigue")}}
        out["fatigue_cap"] = _num(fc["fatigue_cap"].iloc[0], 2) if len(fc) else None
    except (OSError, ValueError, KeyError):
        pass
    return out


def _power_history(proj, acts: pd.DataFrame) -> dict:
    """the Rider page's power over time (models.power_history.PowerHistory): the physio model's daily states, its
    stored fit (all-out probabilities), the plan's forecast (unless it belongs to an older fit) and your maximum heart
    rate."""
    from ..models.power_history import PowerHistory
    from ..models.physio import PhysioRun
    from ..planning.planner import PHYSIO_FORECAST
    processed = proj.processed
    try:
        st = pd.read_parquet(processed / PhysioRun.STATES)
    except (OSError, ValueError):
        return {}
    try:
        fc = None if forecast_stale(processed) else pd.read_parquet(processed / PHYSIO_FORECAST)
        fc = None if fc is None else fc[fc.index > st.index[-1]]
    except (OSError, ValueError):
        fc = None
    ath = (_read_json(proj.reports / "results.json", {}) or {}).get("athlete") or {}
    max_hr = float(ath.get("max_hr") or (proj.cfg.get("athlete") or {}).get("max_hr") or 0) or None
    return PowerHistory(st, acts, PhysioRun(proj).stored(), fc, max_hr).build()


def _history(d: pd.DataFrame) -> dict:
    w = d.resample("W-SUN")
    col = lambda c: w[c].last() if c in d else pd.Series(np.nan, index=w["ftp"].last().index)  # noqa: E731
    h = pd.DataFrame({"ftp": w["ftp"].last(), "test": w["eq20_test"].max(), "ctl": w["ctl"].last(),
                      "hours": w["hours"].sum(), "cp": col("cp"), "lt1": col("hrvt1_28d"), "lt2": col("hrvt2_42d"),
                      "garmin": col("ftp_garmin")})
    return {"date": [str(i.date()) for i in h.index],
            **{c: [_num(v) for v in h[c].to_numpy()] for c in h.columns}}


def _day_rides(a: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    """per day: {"ride": the id of that day's main ride (most TSS), "kind": its workout type, "title": every
    activity's name, "note": their descriptions / private notes (Strava)}, None for none."""
    r = a[a["sport"].astype(str).eq("ride")] if "sport" in a else a
    out = {k: [None] * len(days) for k in ("ride", "kind", "title", "note")}
    if len(r):
        cols = ["id"] + (["workout"] if "workout" in r else [])
        main = r.assign(t=r["tss"].fillna(0) if "tss" in r else 0).sort_values("t").groupby("date")[cols].last().reindex(days)
        kind = main["workout"] if "workout" in main else pd.Series(None, index=days, dtype=object)
        out["ride"] = [None if pd.isna(v) else str(v).removesuffix(".0") for v in main["id"].to_numpy()]
        out["kind"] = [None if pd.isna(v) or not v else str(v) for v in kind.to_numpy()]
    if len(a) and "date" in a:                     # every activity of the day (a walk, a run: travel days too)
        txt = lambda c: a[c].fillna("").astype(str).str.strip() if c in a else pd.Series("", index=a.index)  # noqa: E731
        notes = (txt("description") + "\n" + txt("private_note")).str.strip()
        by = pd.DataFrame({"date": a["date"], "title": txt("name"), "note": notes}).groupby("date")
        for k in ("title", "note"):
            s = by[k].agg(lambda x: " · ".join(v for v in x if v)).reindex(days)
            out[k] = [v or None if isinstance(v, str) else None for v in s.to_numpy()]
    return out


def _rides(a: pd.DataFrame, n=25) -> list:
    r = a[a["sport"].astype(str).eq("ride")] if "sport" in a else a
    r = r.sort_values("date").tail(n).iloc[::-1]
    out = []
    for x in r.itertuples():
        out.append({"id": str(x.id).removesuffix(".0"), "date": str(x.date.date()), "name": str(getattr(x, "name", "") or ""),
                    "workout": str(getattr(x, "workout", "") or "").replace("nan", ""),
                    "type_name": str(getattr(x, "type_name", "") or "").replace("nan", ""),
                    "min": _num(getattr(x, "duration_s", None) and x.duration_s / 60, 0),
                    "tss": _num(getattr(x, "tss", None), 0), "np": _num(getattr(x, "np_used", None), 0),
                    "if": _num(getattr(x, "if_", None), 2), "hr": _num(getattr(x, "avg_hr", None), 0)})
    return out


def _insights(res: dict) -> dict:
    ill = res.get("illness", {}) or {}
    bu = res.get("buildup", {}) or {}
    drv = (res.get("drivers", {}) or {}).get("table") or []
    drv = sorted([x for x in drv if x.get("target") == "d_perf_pct" and x.get("rho") is not None],
                 key=lambda x: -abs(x["rho"]))[:8]
    keep = ["start", "end", "weeks", "perf_start_w", "perf_end_w", "change_pct", "hours_wk",
            "hard_wk", "what_was_different", "yours"]
    pp = res.get("progress_periods", [])
    found = sorted((p for p in pp if not p.get("yours")), key=lambda p: -(p.get("change_pct") or 0))[:6]
    return {"gains": [_pick(p, ["start", "end", "change_pct"]) for p in pp if not p.get("yours")],   # fitness chart
            # the 6 biggest found and every gain you marked, by date
            "progress": [_pick(p, keep) for p in sorted(found + [p for p in pp if p.get("yours")], key=lambda p: str(p["start"]))],
            "declines": [_pick(p, keep + ["illness_after"]) for p in res.get("decline_periods", [])[:6]],
            "illness": [_pick(e, ["onset", "days_off", "rhr_peak_delta", "severity", "days_to_regain_97pct"])
                        for e in (ill.get("episodes") or [])][-12:][::-1],
            "illness_warning": ill.get("warning"), "drivers": drv,
            "peak": {"week": bu.get("peak_week"), "value": bu.get("peak_value")},
            "builds": [_pick(b, ["label", "hours_wk", "ctl_end", "max_ramp_4wk", "quality_wk",
                                 "threshold", "vo2max", "sweetspot", "sleep_h", "gain_pct"])
                       for b in (bu.get("builds") or [])][:6],
            "years": res.get("years", [])}


# --- actions -----------------------------------------------------------------------------
class OpenPages:
    """The goodpush pages open in a browser: each holds /api/alive (a server-sent event stream) while it's open, and the
    browser reconnects it by itself within a second when the app restarts - also in a background tab, where timers
    are slowed. So a restarted app reloads the tab you have instead of opening another one."""
    WAIT_S = 5.0                                # how long a starting app waits for an open page before opening one
    BEAT_S = 15.0                               # a comment now and then: a page closed meanwhile is noticed

    def __init__(self):
        self.seen = threading.Event()           # a page connected since the app started

    def hold(self, handler, run_id: str) -> None:
        """the stream for one page: this run's id (a page from an earlier run reloads), then a beat until it goes"""
        self.seen.set()
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            handler.wfile.write(f"retry: 1000\ndata: {run_id}\n\n".encode())
            handler.wfile.flush()
            while True:
                time.sleep(self.BEAT_S)
                handler.wfile.write(b": beat\n\n")
                handler.wfile.flush()
        except (ConnectionError, OSError):
            pass
        handler.close_connection = True

    def open_unless_seen(self, url: str, wait: float | None = None) -> bool:
        """open the app in the browser unless a page already open connects within `wait` s; True if opened"""
        if self.seen.wait(self.WAIT_S if wait is None else wait):
            return False
        import webbrowser
        webbrowser.open(url)
        return True


class Jobs:
    """One background job at a time; `start(..., queue=True)` while one runs waits for it (one at most:
    a newer request replaces an older waiting one) and starts the moment it ends."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cur = {"name": None, "running": False, "log": [], "ok": None, "t": 0}
        self.next = None

    def status(self):
        with self.lock:
            return dict(self.cur, log=self.cur["log"][-30:], queued=self.next[0] if self.next else None)

    def start(self, name, fn, queue=False):
        with self.lock:
            if self.cur["running"]:
                if queue:
                    self.next = (name, fn)
                    return "queued"
                return False
            self.cur = {"name": name, "running": True, "log": [], "ok": None, "t": time.time(), "step": None}
        self.spawn(fn)
        return True

    def drop_queued(self) -> str | None:
        """forget the job waiting for the current one (its name; None: none waits)."""
        with self.lock:
            nxt, self.next = self.next, None
        return nxt[0] if nxt else None

    def spawn(self, fn):

        def add(*a):
            with self.lock:
                self.cur["log"].append(" ".join(str(x) for x in a))

        def step(label, done, total):
            with self.lock:
                self.cur["step"] = None if done >= total else {"label": label, "done": done, "total": total}
        log = Progress(add, step)                # callable like print, plus the progress bar

        def run():
            ok, name, t0 = True, self.cur["name"], time.time()
            print(f"job started: {name}")          # the app log's record of what ran (AppLog)
            try:
                fn(log)
            except BaseException as e:  # noqa: BLE001 - report every failure in the UI
                ok = False
                log(f"failed: {e}")
                traceback.print_exc()
            print(f"job {'finished' if ok else 'failed'}: {name} ({time.time() - t0:.0f} s)")
            with self.lock:                       # the waiting job takes over without an idle moment
                self.cur.update(running=False, ok=ok)
                nxt, self.next = self.next, None
                if nxt:
                    self.cur = {"name": nxt[0], "running": True, "log": [], "ok": None, "t": time.time(), "step": None}
            if nxt:
                self.spawn(nxt[1])
        threading.Thread(target=run, daemon=True).start()


def _full_rebuild(proj, log):
    from ..analyze import run_analysis
    from ..report import write_reports
    from ..ingest import ingest
    from ..sync.garmin import API_SUBDIR, rebuild
    if (proj.path("garmin") / API_SUBDIR / "raw").exists():   # Garmin's combined files from every day's download
        rebuild(proj.path("garmin"))                          # (another computer may have fetched some: peer sync)
    log("Reading new activity files ...")
    ingest(proj, progress=log)
    log("Analysing ...")
    r = run_analysis(proj, progress=log)
    r["plan"] = Planner(proj).build(r, progress=log)
    write_reports(proj, r, progress=log)
    log("Done.")


def _reanalyse(proj, log):
    """Analysis + plan + reports from cached rides (no file re-reading)."""
    from ..analyze import run_analysis
    from ..report import write_reports
    r = run_analysis(proj, progress=log)
    r["plan"] = Planner(proj).build(r, progress=log)
    write_reports(proj, r, progress=log)
    log("Done.")


def _add_ride(proj, key: str, ride_id: str, log):
    """a ride the ride window saved: into your rides, linked to its session, then analysis and the plan."""
    from ..ingest import add_ride
    add_ride(proj, key, ride_id, log)
    _reanalyse(proj, log)


def _forget_activity(proj, aid: str, log):
    """a deleted ride out of the processed folder: a job, so it waits for a running ingest (which packs
    streams.h5) instead of racing it. If a later job takes its place in the queue, the next ingest does it."""
    from ..ingest import forget
    forget(proj, [aid], log)
    _reanalyse(proj, log)


def _plan_stale(plan: dict, day: str) -> bool:
    """the plan (plan.json) needs re-making: from an earlier day, an older file format, or other planning code."""
    return ((plan.get("generated") or "") < day or any("blocks" not in x for x in plan.get("sessions", []))
            or Planner.code_changed(plan))


def plan_at_start(proj) -> bool:
    """At app start, before the first page: re-make a stale plan, so the app opens on today's plan made by this
    code (not the old one, then "Updating"). True when it did. No rides yet is fine (Planner.no_rides)."""
    plan_file = proj.reports / "plan" / "plan.json"
    if not (proj.cfg.get("plan") or {}).get("enabled"):
        return False
    plan = json.loads(plan_file.read_text(encoding="utf-8")) if plan_file.exists() else {}
    day = str(pd.Timestamp.now(tz=proj.tz).tz_localize(None).date())
    if not _plan_stale(plan, day):
        return False
    print("Updating the plan for today before opening ...")
    try:
        _plan_only(proj, print)
    except Exception:  # noqa: BLE001 - the app still opens; the page offers the update again
        traceback.print_exc()
        return False
    return True


def _plan_only(proj, log):
    Planner(proj).build(progress=log)
    log("Plan updated.")


def _sick_day(proj, log):
    """you marked a day sick: your illness markers learn from it, then the plan."""
    from ..models import markers
    markers.learn_if_due(proj, progress=log)
    _plan_only(proj, log)


def prewarm_plan(proj, refitted=None) -> None:
    """At app start, build the plan once without writing anything, so the model code is loaded / compiled
    before your first change (that one then takes as long as any other). If that refitted the physio model (new
    rides, a model change), `refitted()` is called: the plan should be made again with it."""
    if not (proj.cfg.get("plan") or {}).get("enabled"):
        return
    try:
        Planner(proj).build(write=False, progress=lambda *a, **k: None)
    except Exception:  # noqa: BLE001 - only a warm-up: the real build reports its own problems
        traceback.print_exc()
        return
    if refitted and forecast_stale(proj.processed):
        refitted()


def _auto_hours(proj) -> float:
    """0 on a machine that isn't this athlete's home ([sync] home): it gets the data from there"""
    from ..sync.peer import Peers, PeerSync
    if not PeerSync.home_here_for(proj.cfg, Peers().joined()):
        return 0.0
    try:
        return float((proj.cfg.get("garmin") or {}).get("auto_sync_hours", 0))
    except (TypeError, ValueError):
        return 0.0


def _garmin_sync(proj, log, force_rebuild=False) -> bool:
    """Personal API pull, then rebuild only if something new arrived (true then)."""
    from ..sync import garmin_api
    r = garmin_api.sync(proj.path("garmin"), proj.path("strava"), progress=log)
    if force_rebuild or r["activities"] or r["days"]:
        _full_rebuild(proj, log)
    else:
        log("Already up to date.")
    return bool(r["activities"] or r["days"])


def _google_sync(proj, gh, log, rebuild=True):
    """Google Health (Fitbit) pull, then rebuild if anything new arrived."""
    r = gh.sync(proj.path("google_health"), progress=log)
    if rebuild and r["new"]:
        _full_rebuild(proj, log)
    elif rebuild:
        log("Already up to date.")
    return r


def _health_sync(proj, gh, log) -> bool:
    """Garmin and / or Google Health, whichever is connected; one rebuild at the end (true: something new)."""
    from ..sync import garmin_api
    new = False
    if garmin_api.has_key():
        r = garmin_api.sync(proj.path("garmin"), proj.path("strava"), progress=log)
        new = bool(r["activities"] or r["days"])
    if gh.connected():
        new = bool(_google_sync(proj, gh, log, rebuild=False)["new"]) or new
    if new:
        _full_rebuild(proj, log)
    else:
        log("Already up to date.")
    return new


class PeerHub:
    """Settings > Other computers: pair with another computer's app and keep the athlete folder in step with it.
    The one that shows the code serves the folder (PeerServer) only while a code is shown, or while computers are
    paired with it and [sync] serve is on (per computer); one that joined syncs with it when you press Sync, after
    a ride, and every [sync] every_hours while both apps run (per computer; 0 = not by itself)."""

    def __init__(self, data: Data, jobs: Jobs, port: int, folder: Path | None = None):
        from ..sync.peer import PeerServer
        self.data, self.jobs, self.folder = data, jobs, folder
        self.port = port + PeerServer.PORT_OFFSET
        self.server = self.new_server()
        self.last_auto, self.ride_saved = time.time(), False
        self.me = socket.gethostname()             # this computer's name to the others
        self.error = None
        self.listen_if_needed()

    def new_server(self):
        """the PeerServer for the project's folder (its pairings and certificate: the athlete's credentials)"""
        from ..planning import PlanState
        from ..sync.peer import PeerServer
        return PeerServer(self.data.proj.root, self.port, self.folder, on_done=self.taken_in, busy=self.busy,
                          lock=PlanState.LOCK, on_garmin=self.garmin_asked)

    def moved(self) -> None:
        """the app now serves another athlete (AthleteSwitch): their folder, their paired computers"""
        self.server.stop()
        self.server = self.new_server()
        self.error = None
        self.listen_if_needed()

    def serving(self) -> bool:
        """computers paired with this one may sync now"""
        return bool(self.server.devices_now().peers) and bool((self.data.proj.cfg.get("sync") or {}).get("serve", True))

    def listen_if_needed(self) -> None:
        """listening only while that's needed: a code shown, or paired computers allowed to sync"""
        if self.serving() or self.server.pairing.active():
            self.listen()
        elif self.server.httpd:
            self.server.stop()

    def listen(self) -> None:
        try:
            self.server.start()
            self.error = None
        except OSError as e:
            self.error = f"Other computers can't reach this one (port {self.server.port}: {e})."

    def set_serve(self, on: bool) -> None:
        from ..config import CONFIG_NAME
        from . import tomledit
        tomledit.set_values(self.data.proj.root / CONFIG_NAME, "sync", {"serve": bool(on)})
        self.reload()
        self.listen_if_needed()

    def busy(self) -> bool:
        s = self.jobs.status()
        return bool(s["running"] or s["queued"])

    def peers(self):
        from ..sync.peer import Peers
        return Peers(self.folder)

    def home_here(self) -> bool:
        from ..sync.peer import PeerSync
        return PeerSync.home_here_for(self.data.proj.cfg, self.peers().joined(), self.me)

    def reload(self):
        """the project again from goodpush.toml (a sync or a change here wrote it)"""
        from ..config import load_project
        self.data.proj = load_project(self.data.proj.root)
        return self.data.proj

    def set_home(self, name: str, only_if_unset: bool = False) -> None:
        from ..config import CONFIG_NAME
        from . import tomledit
        if only_if_unset and (self.data.proj.cfg.get("sync") or {}).get("home"):
            return
        tomledit.set_values(self.data.proj.root / CONFIG_NAME, "sync", {"home": name})
        self.reload()

    def taken_in(self, name: str, kinds, wait: bool = True) -> bool:
        """a computer finished a sync that changed files here. Home: take them in (new rides / health data, your
        changes), then pass it all on to the other computers - to the sender too unless it waits to fetch it.
        Elsewhere: only your changes (the analysis came with them)."""
        kinds = set(kinds)
        if not self.home_here():
            if "decision" in kinds:
                self.reload()
            return False
        if not {"data", "rows", "index", "decision"} & kinds:
            return False
        skip = name if wait else None
        if {"data", "rows", "index"} & kinds:
            return bool(self.jobs.start(f"Adding what came from {self.label(name)}", lambda log: (
                _full_rebuild(self.data.proj, log), self.relay(log, skip)), queue=True))
        proj = self.reload()
        return bool(self.jobs.start(f"Updating with your changes from {self.label(name)}", lambda log: (
            _reanalyse(proj, log), self.relay(log, skip)), queue=True))

    def relay(self, log, skip: str | None = None) -> None:
        """home: send what's here to every other paired computer that can be reached now (quietly past the rest)"""
        for name, p in self.peers().peers.items():
            if name == skip or p.get("kind") != "app" or not self.reachable(p.get("address", "")):
                continue
            try:
                self.sync(name, log, "send")
            except (ConnectionError, PermissionError, FileNotFoundError, OSError) as e:
                log(f"{self.label(name)}: not sent ({e})")

    def garmin_asked(self, name: str) -> bool:
        """home: another computer pressed Update now - download from Garmin here, then send it to everyone"""
        from ..sync import garmin_api
        if not self.home_here() or not garmin_api.has_key():
            return False
        return bool(self.jobs.start(f"Updating from Garmin (asked by {self.label(name)})",
                                    lambda log: self.updated(_garmin_sync(self.data.proj, log), log), queue=True))

    def garmin_job(self, local):
        """Update now on a computer that isn't home: ask home to download from Garmin (it sends the result here), so
        Garmin's files come from one place; home not reachable: download here (`local`) - peer sync merges the two
        (Garmin's activity list by activity, its combined files rebuilt at home)."""
        return self.jobs.start("Updating from Garmin", lambda log: self.garmin_via_home(log, local), queue=True)

    def garmin_via_home(self, log, local) -> None:
        home = (self.data.proj.cfg.get("sync") or {}).get("home") or ""
        p = self.peers().peers.get(home) or {}
        if p.get("kind") == "app" and self.reachable(p.get("address", "")):
            side = self.peers().side(home)
            try:
                asked = side.garmin()
            except (ConnectionError, PermissionError, OSError) as e:
                asked = False
                log(f"{self.label(home)}: {e}")
            finally:
                side.close()
            if asked:
                log(f"{self.label(home)} is downloading from Garmin and sends the result here.")
                return
        log(f"{self.label(home) or 'The home computer'} isn't reachable: downloading here; the next sync merges it.")
        local(log)

    def updated(self, new: bool, log) -> None:
        """new data from Garmin / Google Health arrived (and was analysed) here: home passes it on"""
        if new and self.home_here():
            self.relay(log)

    def label(self, name: str) -> str:
        """what you call that computer here (Rename), else its own name"""
        for p in (self.peers().peers, self.server.devices_now().peers):
            if p.get(name, {}).get("label"):
                return p[name]["label"]
        return name

    def rename(self, name: str, label: str) -> None:
        """call a paired computer `label` on this one (empty: its own name again). Only shown: the computer's own
        name still identifies it (home, sync records)."""
        label = " ".join(str(label).split())[:40]
        peers, devices = self.peers(), self.server.devices_now()
        if name not in peers.peers and name not in devices.peers:
            raise ValueError(f"No paired computer {name}.")
        peers.update(name, label=label)
        devices.update(name, label=label)

    def status(self) -> dict:
        devices = self.server.devices_now().peers
        code = self.server.pairing.active()
        home = (self.data.proj.cfg.get("sync") or {}).get("home") or ""
        peers = self.peers().peers
        computers = [{"name": k, "label": (peers.get(k) or devices.get(k) or {}).get("label") or k,
                      "address": (peers.get(k) or {}).get("address") or (peers.get(k) or {}).get("host"),
                      "last": (peers.get(k) or {}).get("last"), "result": (peers.get(k) or {}).get("result"),
                      "can_start": k in peers, "lets_in": k in devices, "home": k == home}
                     for k in sorted(set(peers) | set(devices))]
        return {"name": self.me, "home": home, "home_label": self.label(home) if home else "", "computers": computers,
                "home_here": self.home_here(), "error": self.error,
                "joined": [{"name": k, "label": v.get("label") or k, "address": v.get("address") or v.get("host"),
                            "last": v.get("last"), "result": v.get("result"), "kind": v.get("kind") or "ssh"}
                           for k, v in sorted(self.peers().peers.items())],
                "devices": [{"name": k, "label": v.get("label") or k, "paired": v.get("paired")}
                            for k, v in sorted(devices.items())],
                "serve": bool((self.data.proj.cfg.get("sync") or {}).get("serve", True)),
                "every_hours": self.every_hours(),
                "listening": bool(self.server.httpd), "waiting": self.server.pairing.waiting(),
                "code": code, "addresses": self.server.addresses() if self.server.httpd else [],
                "until": time.strftime("%H:%M", time.localtime(self.server.pairing.until)) if code else None}

    def pair(self) -> dict:
        """show a code on this computer (and let the computers paired with it sync)"""
        self.listen()
        if self.error:
            raise ValueError(self.error)
        self.server.pairing.new()
        self.set_home(self.me, only_if_unset=True)     # the one others join is usually always on
        if not (self.data.proj.cfg.get("sync") or {}).get("serve", True):
            self.set_serve(True)
        return self.status()

    def join(self, address: str, code: str) -> str:
        """pair with the computer showing `code` - both ways: either can start a sync after"""
        from ..sync.peer import HttpSide, PeerCert
        if not address.strip() or not code.strip():
            raise ValueError("Enter the other machine's address and the code it shows.")
        entry = HttpSide.pair(address, code, self.me)
        name = entry.pop("name")
        self.peers().add(name, **entry)
        token = self.server.devices_now().pair(name)            # its way back to this one
        there = HttpSide(entry["address"], entry["fingerprint"], entry["token"], name)
        try:
            there.link(PeerCert(self.folder).fingerprint(), self.server.port, token)
        finally:
            there.close()
        self.set_home(name, only_if_unset=True)
        self.listen_if_needed()
        self.sync_job(name)
        return name

    def forget(self, name: str) -> None:
        self.peers().remove(name)
        self.server.devices_now().remove(name)
        self.listen_if_needed()                    # the last one gone: stop listening

    DIRECTION = {"both": "Syncing with", "send": "Sending to", "get": "Getting from"}

    def sync(self, name: str, log, direction: str = "both") -> dict:
        """one sync with a paired computer (a job): both ways, or only send / get its files (PeerSync.plan); then
        the other takes in what it got - both ways this one waits and fetches the result. Home, having got new
        rides: analyses them and passes them on."""
        from ..sync.peer import FolderSide, PeerSync
        peers = self.peers()
        proj = self.data.proj
        there = peers.side(name)
        log(f"{self.DIRECTION[direction]} {self.label(name)} ...")
        try:
            s = PeerSync(FolderSide(proj.root), there, name, self.home_here())
            steps = s.plan(direction)
            log(s.describe(steps))
            out = s.run(steps, progress=False)
            got = set(s.changed["here"])
            again = s.hand_over(log, wait=False)   # home sends the results itself (relay): no waiting here
            if again:
                got |= s.changed["here"]
                out["clashes"] += again["clashes"]
        finally:
            there.close()
        for c in out["clashes"]:
            log(f"Changed on both computers, {'this' if s.home_here else 'the home'} one's kept: {c}")
        if "decision" in got:
            proj = self.reload()
        if s.home_here and {"data", "rows", "index"} & got:
            _full_rebuild(proj, log)
            self.relay(log)                        # the analysis to everyone, this one included
        peers.update(name, last=time.strftime("%Y-%m-%dT%H:%M"), result="ok")
        log("In step." if steps or again else "Already in step.")
        return out

    def sync_job(self, name: str, direction: str = "both"):
        if direction not in self.DIRECTION:
            raise ValueError(f"No direction {direction}.")
        return self.jobs.start(f"{self.DIRECTION[direction]} {self.label(name)}",
                               lambda log: self.sync_logged(name, log, direction), queue=True)

    def sync_logged(self, name: str, log, direction: str = "both") -> None:
        try:
            self.sync(name, log, direction)
        except (ConnectionError, PermissionError, FileNotFoundError, OSError) as e:
            self.peers().update(name, last=time.strftime("%Y-%m-%dT%H:%M"), result=str(e)[:200])
            raise

    def sync_all_job(self):
        """Sync (the top bar): every computer this one joined, one after the other"""
        names = list(self.peers().peers)
        if not names:
            raise ValueError("This computer hasn't joined another one yet (Settings > Other computers).")
        return self.jobs.start("Syncing", lambda log: [self.sync_logged(n, log) for n in names], queue=True)

    def after_ride(self) -> None:
        """a ride was saved here: sync once it's processed (the next auto() when nothing runs)"""
        self.ride_saved = True

    def set_every(self, hours: float) -> None:
        from ..config import CONFIG_NAME
        from . import tomledit
        if not 0 <= hours <= 48:
            raise ValueError("Between 0 and 48 hours (0 = only when you press Sync).")
        tomledit.set_values(self.data.proj.root / CONFIG_NAME, "sync", {"every_hours": hours})
        self.reload()

    def every_hours(self) -> float:
        try:
            return float((self.data.proj.cfg.get("sync") or {}).get("every_hours", 2.0))
        except (TypeError, ValueError):
            return 0.0

    def auto(self) -> None:
        """every minute: stop listening once the code has run out (nothing paired / serve off); after a ride, and
        every [sync] every_hours on a computer that isn't home (0 = never by itself; home passes things on as they
        arrive): sync with the paired computers that are reachable (quietly if not)"""
        self.listen_if_needed()
        h = self.every_hours()
        due = self.ride_saved or (h > 0 and not self.home_here() and time.time() - self.last_auto >= h * 3600)
        if not due or self.busy():
            return
        self.last_auto, self.ride_saved = time.time(), False
        for name, p in self.peers().peers.items():
            if p.get("kind") == "app" and self.reachable(p.get("address", "")):
                self.sync_job(name)

    @staticmethod
    def reachable(address: str) -> bool:
        from ..sync.peer import HttpSide
        try:
            socket.create_connection(HttpSide.split(address), timeout=3).close()
            return True
        except OSError:
            return False


class AthleteSwitch:
    """The menu under the app's name: the athletes on this machine (athletes.Athletes), switch the app to another
    one, add one, remove one. Switching changes the project the app serves (Data.proj), whose credentials it uses and the
    computers it syncs with (PeerHub.moved); only while nothing runs that holds the old project (a job, a ride, the
    Model page's run, a ranges refresh). The registry's default athlete stays as it is."""

    def __init__(self, data: Data, hub: PeerHub, ranges: DailyRanges):
        self.data, self.hub, self.ranges = data, hub, ranges

    def status(self) -> dict:
        """who the app shows, the athletes to choose from and the folder a new one goes in by default."""
        from .. import athletes
        from ..config import CONFIG_NAME
        reg, root = athletes.Athletes(), self.data.proj.root
        parent = reg.folders[reg.default].parent if reg.default in reg.folders else root.parent
        return {"current": reg.name_of(root) or "", "name": self.data.proj.cfg["athlete"].get("name") or "",
                "default": reg.default, "parent": str(parent),
                "athletes": [{"name": k, "folder": str(v), "missing": not (v / CONFIG_NAME).exists()}
                             for k, v in sorted(reg.folders.items())]}

    def why_not(self) -> str:
        """empty when the app may switch now, else why not."""
        from .. import athletes
        jobs = self.ranges.jobs.status()
        if jobs["running"] or jobs["queued"]:
            return "Wait for the current job to finish, then switch."
        if RideLauncher.proc is not None and RideLauncher.proc.poll() is None:
            return "Finish the ride first."
        if self.ranges.running() or (self.ranges.full is not None and self.ranges.full.running()):
            return "The model is running for this athlete: switch when it's done."
        secrets_env = os.environ.get("GOODPUSH_SECRETS")
        if secrets_env is not None and secrets_env != athletes.Athletes.SET:
            return "GOODPUSH_SECRETS is set for this app, so it can't use another athlete's sign-ins."
        return ""

    def switch(self, name: str) -> None:
        """the app shows athlete `name` from now on (ValueError: why not now)."""
        from .. import athletes
        from ..config import CONFIG_NAME, load_project
        reason = self.why_not()
        if reason:
            raise ValueError(reason)
        try:
            folder = athletes.Athletes().folder(name)
        except KeyError:
            raise ValueError(f"There is no athlete '{name}' on this computer.") from None
        if not (folder / CONFIG_NAME).exists():
            raise ValueError(f"{name}'s folder isn't there any more: {folder}")
        self.data.proj = load_project(athlete=name)
        self.data._cache = {}
        self.hub.moved()

    def add(self, name: str, folder: str) -> None:
        """register athlete `name` with `folder` (default: next to the default athlete's; a new folder gets a
        goodpush.toml), then switch to them (ValueError: why not)."""
        from .. import athletes
        from ..config import CONFIG_NAME, init_project
        reg = athletes.Athletes()
        name = name.strip().lower()
        if name in reg.folders:
            raise ValueError(f"There is an athlete '{name}' already.")
        if not athletes.NAME.match(name):
            raise ValueError("A name is lower-case letters, digits, - or _ (up to 32).")
        reason = self.why_not()
        if reason:
            raise ValueError(reason)
        path = Path(folder.strip() or Path(self.status()["parent"]) / name).expanduser()
        if not path.is_absolute():
            raise ValueError("Give the whole folder path.")
        reg.add(name, path)
        if not (path / CONFIG_NAME).exists():
            init_project(path)
        self.switch(name)

    def move(self, name: str, folder: str):
        """athlete `name`'s folder to `folder` (athletes.FolderMove: a rename, or a copy checked and then the old one
        deleted; only re-pointed when you moved it yourself), as a job. The one the app shows: its paired computers
        wait meanwhile, then the app serves the new place. ValueError before anything changed: why not."""
        from .. import athletes
        reason = self.why_not()
        if reason:
            raise ValueError(reason.replace("switch", "move it"))
        move = athletes.FolderMove(athletes.Athletes(), name, folder)
        shown = move.src == self.data.proj.root.resolve()
        return self.ranges.jobs.start(f"Moving {name}'s folder", lambda log: self.moving(move, shown, log))

    def moving(self, move, shown: bool, log) -> None:
        """the job: the move; the app then on the new folder (a job asked for meanwhile, which holds the old
        folder, isn't run)."""
        from ..config import load_project
        move.progress = log
        if shown:
            self.hub.server.stop()
        try:
            move.run()
        finally:
            if shown:
                self.data.proj = load_project(move.dest if (move.dest / move.config).is_file() else move.src)
                self.data._cache = {}
                dropped = self.ranges.jobs.drop_queued()
                if dropped:
                    log(f"Not run (it was asked for during the move): {dropped}. Do it again.")
                self.hub.moved()

    def remove(self, name: str) -> None:
        """take athlete `name` off this computer (Athletes.remove: their folder stays); not the one the app shows."""
        from .. import athletes
        reg = athletes.Athletes()
        if name not in reg.folders:
            raise ValueError(f"There is no athlete '{name}' on this computer.")
        if reg.name_of(self.data.proj.root) == name:
            raise ValueError("Switch to another athlete first.")
        reg.remove(name)


def make_handler(proj, key: str | None, lan: bool, warm: bool = True, port: int = 8765):
    from ..config import load_project
    from ..sync.google_health import GoogleHealth, GoogleHealthError
    data = Data(proj)
    jobs = Jobs()
    auto_done = set()
    run_id = f"{time.time():.3f}"   # this run of the app: a page from an earlier run reloads (its code may be old)
    gh = GoogleHealth()            # holds the sign-in in progress between Connect and Google's redirect
    hub = PeerHub(data, jobs, port)
    full_run = FullModelRun(jobs, lambda: data.proj)     # the Model page's Run button
    ranges = DailyRanges(jobs, hub.home_here, lambda: data.proj, full_run)
    switcher = AthleteSwitch(data, hub, ranges)
    pages = OpenPages()

    def auto_sync():
        """While the app runs, if you turned it on ([garmin] auto_sync_hours > 0): pull from Garmin / Google
        Health every that many hours - never on opening the app (you sync with Update now). And the machines
        this one paired with (PeerHub.auto)."""
        from ..sync import garmin_api
        last = time.time()
        while True:
            h = _auto_hours(data.proj)
            if h > 0 and (garmin_api.has_key() or gh.connected()) and time.time() - last >= h * 3600:
                if jobs.start("Automatic update", lambda log: hub.updated(_health_sync(data.proj, gh, log), log)):
                    last = time.time()
            try:
                hub.auto()
            except Exception:  # noqa: BLE001 - never stop the loop
                traceback.print_exc()
            try:
                ranges.tick()                    # the model's ranges, in their own process (DailyRanges)
            except Exception:  # noqa: BLE001 - never stop the loop; the ranges keep their date
                traceback.print_exc()
            time.sleep(60)
    threading.Thread(target=auto_sync, daemon=True, name="goodpush-auto-sync").start()
    if warm:                        # (a plan just made at start-up has loaded the model code already)
        threading.Thread(target=prewarm_plan, daemon=True, name="goodpush-prewarm", args=(proj, lambda: jobs.start(
            "Updating the plan (model refitted)", lambda log: _plan_only(proj, log), queue=True))).start()

    def today():
        return pd.Timestamp.now(tz=proj.tz).tz_localize(None).normalize()

    def act(body: dict):
        nonlocal proj
        a = body.get("action")
        sp = proj.processed / "plan_state.json"
        if a == "label_dip":
            from ..models.dips import BreakLabels
            BreakLabels(proj.processed).set(str(body.get("start")), body.get("value") or "")

            def job(log):
                from ..models import markers
                markers.learn(proj, progress=log)
                _plan_only(proj, log)
            return jobs.start("Re-learning from your labels", job)
        if a == "label_period":                # a period you marked / moved / removed on the load chart: saved only;
            from ..models.dips import BreakLabels  # "periods_done" (marking switched off) learns from them all
            lab = BreakLabels(proj.processed)
            start, end, value, was = str(body.get("start")), str(body.get("end")), body.get("value") or "", body.get("was")
            found = {(d["start"], d["end"]) for d in (_read_json(proj.processed / "dips.json", {}) or {}).get("dips", [])}
            if (start, end) in found and value != "gain":    # a detected break's own days: you confirm (or relabel) it
                if was:
                    lab.set_period(*was, "", tuple(was))
                lab.set(start, value)
            else:
                lab.set_period(start, end, value, was and tuple(was))
            return "saved"
        if a == "periods_done":                # marking switched off: learn once from what you marked
            def job(log):
                if body.get("gain"):             # "Biggest gains" comes from the analysis (it learns markers when due)
                    _reanalyse(proj, log)
                    return
                from ..models import markers
                markers.learn(proj, progress=log)
                _plan_only(proj, log)
            return jobs.start("Re-analysing with your periods" if body.get("gain") else "Re-learning from your labels", job)
        if a == "learn_markers":
            def job(log):
                from ..models import markers
                markers.learn(proj, progress=log)
                _plan_only(proj, log)
            return jobs.start("Analysing your illness markers", job)
        if a == "delete_activity":            # a ride page's Delete (e.g. a stray goodpush recording)
            aid = str(body.get("id") or "")
            if not aid:
                raise ValueError("no activity given")
            from ..io.strava import DeletedActivities
            DeletedActivities(proj.root).delete(aid, proj.path("rides"))
            return jobs.start("Ride deleted - cleaning up and re-analysing", lambda log: _forget_activity(proj, aid, log),
                              queue=True)
        if a == "rebuild":
            return jobs.start("Rebuilding everything", lambda log: _full_rebuild(proj, log))
        if a == "physio_fresh":                # Settings → Data: the model from its priors
            return full_run.start_fresh(proj)
        if a in ("workout_text", "workout_format"):     # the workout builder: text <-> steps (nothing saved)
            from ..planning.builder import WorkoutText
            from ..planning.steps import items_from_dicts
            wt = WorkoutText(ftp=float(body.get("ftp") or 0))
            if a == "workout_text":
                items = wt.parse(str(body.get("text") or ""))
            elif body.get("key"):                      # a planned session, as % of FTP (the calendar shows watts)
                items = items_from_dicts((PlanState.load(sp).current(str(body["key"])) or {}).get("steps") or [])
            else:
                items = items_from_dicts(body.get("steps") or [])
            return {"steps": [x.to_dict() for x in items], "text": WorkoutText.format(items),
                    "type": WorkoutText.guess_type(items) if items else "", "errors": [vars(e) for e in wt.errors]}
        if a == "workout_adjust":            # the builder's easier / harder: `steps` as built, `offset` steps from them
            from ..planning.builder import WorkoutText
            from ..planning.steps import items_from_dicts
            from ..planning.workouts import Adjuster, Workout
            items = items_from_dicts(body.get("steps") or [])
            if not items:
                raise ValueError("an empty workout")
            w, adj = Workout(today(), "own", steps=items), Adjuster()
            easier, harder = adj.reach(w)
            n = int(min(max(int(body.get("offset") or 0), -easier), harder))
            moved = adj.apply(w, n).steps
            return {"steps": [x.to_dict() for x in moved], "text": WorkoutText.format(moved), "offset": n,
                    "easier": n > -easier, "harder": n < harder}
        if a in ("save_workout", "delete_workout"):     # your workout library (data/my_workouts.json)
            from ..planning.builder import MyWorkouts
            lib = MyWorkouts(proj.root)
            if a == "delete_workout":
                lib.delete(str(body.get("id") or ""))
                return {"ok": True}
            return {"ok": True, "id": lib.save(str(body.get("title") or ""), str(body.get("type") or ""),
                                               body.get("steps") or [], body.get("id") or None)}
        if a == "add_workout":                          # your workout on a day (or in place of a session)
            with PlanState.open(sp) as st:
                st.add(body.get("workout") or {}, str(body.get("date") or ""), today(), body.get("key") or None)
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a in ("move_session", "copy_session", "merge_session", "delete_session", "edit_session"):  # your changes
            plan = _read_json(proj.reports / "plan" / "plan.json", {}) or {}
            ready = plan.get("readiness") if plan.get("generated") == str(today().date()) else None
            with PlanState.open(sp) as st:
                if a == "move_session":
                    st.move(str(body.get("from")), str(body.get("to")), today(), ready)
                elif a == "merge_session":          # two easy rides as one (dragged onto the other, held there)
                    st.merge(str(body.get("from")), str(body.get("to")), today())
                elif a == "copy_session":
                    st.copy(str(body.get("from")), str(body.get("to")), today())
                elif a == "delete_session":
                    st.delete(str(body.get("key")), today())
                else:
                    st.edit(str(body.get("key")), body.get("minutes"), bool(body.get("reset")))
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a in ("lock_session", "clear_own"):          # the Calendar's lock on a session / clear your changes
            with PlanState.open(sp) as st:
                if a == "lock_session":     # only saved: the session stays as it is, the next re-plan respects it
                    st.set_lock(str(body.get("key")), bool(body.get("locked")), today())
                    return {"ok": True}
                st.clear_own(today())
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a == "dismiss_phases":              # "not now" to the phases suggested from your races
            with PlanState.open(sp) as st:
                st["season_dismissed"] = str(body.get("sig") or "")
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if (a in ("accept_phases", "autoplan_season", "season_blocks", "add_phase", "edit_phase")
                and not (proj.cfg.get("plan") or {}).get("enabled")):
            _save_settings(proj, "plan", {"enabled": True})      # you set up a season: planning is on
            proj = data.proj = load_project(proj.root)
        if a == "accept_phases":               # add the phases suggested (from your races, or the general block)
            season = Season.load(proj.root)
            for s in season.suggest(today()):
                season.add_phase(type=s["type"], start=s["start"], weeks=s["weeks"], build_weeks=s["build_weeks"])
            season.save(proj.root)
            with PlanState.open(sp) as st:
                st.replan(today())
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a in ("season_blocks", "autoplan_season"):   # the Calendar's season editor: blocks back to back / from
            # races (none ahead: the general block)
            season, old = Season.load(proj.root), Season.load(proj.root)
            if a == "season_blocks":
                season.set_blocks(str(body.get("start")), list(body.get("blocks") or []))
            else:
                season.autoplan(today())
            own = PlanState.load(sp).own_on(season.changed_days(old, today()))
            if own and body.get("replan_own") is None:   # days you set yourself in the weeks that changed: ask first
                first, last = (pd.Timestamp(k[:10]).strftime("%a %d %b") for k in (own[0], own[-1]))
                return {"answer": "replan_own", "title": "Plan your own days again?", "ask": (
                    f"You set {len(own)} day(s) yourself (moved, added or resized) in the weeks this changes "
                    f"({first} – {last}).\n\nThe planner can plan them for the new season, or they can stay as "
                    "you set them."), "yes": "Plan them again", "no": "Keep my days"}
            season.save(proj.root)
            with PlanState.open(sp) as st:
                if body.get("replan_own"):
                    st.forget_own(own)
                st.replan(today())
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a in ("add_phase", "edit_phase", "delete_phase", "add_event", "edit_event", "add_events",
                 "delete_event"):             # your season (Plan)
            season = Season.load(proj.root)
            v = body.get("values") or {}
            if a in ("add_phase", "edit_phase"):       # next to a phase of the same kind: ask to merge first
                merge = body.get("merge")
                ask = (season.place_phase(Phase(**{k: v.get(k) for k in ("type", "start", "weeks", "name", "build_weeks", "hours")}), merge)
                       if a == "add_phase" else season.edit_phase(str(body.get("id")), v.get("start"), v.get("weeks"), merge))
                if ask:
                    return {"ask": ask, "title": "Merge the phases?", "yes": "Merge them", "no": "Keep them apart"}
            elif a == "delete_phase":
                season.delete_phase(str(body.get("id")))
            elif a == "add_event":
                season.add_event(**{k: v.get(k) for k in ("date", "priority", "kind", "name", "series")})
            elif a == "edit_event":
                season.edit_event(str(body.get("id")), **{k: v.get(k) for k in ("date", "priority", "kind", "name", "series")})
            elif a == "add_events":
                season.add_events_text(str(v.get("text") or ""), str(v.get("series") or ""))
            else:
                season.delete_event(str(body.get("id")))
            season.save(proj.root)
            with PlanState.open(sp) as st:
                st.replan(today())
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a == "set_level":                  # your interval level for a type (Plan); applies straight away
            with PlanState.open(sp) as st:
                st.set_level(str(body.get("kind")), int(body.get("level") or 0))
                st.replan(today())
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a == "adjust_preview":             # the session window's easier / harder: the session `step` steps on, unsaved
            from ..planning.intensity import IntensityMap
            from ..report.plan_out import session_steps
            w, adj = PlanState.load(sp).preview(str(body.get("key")), int(body.get("step") or 0))
            imap = IntensityMap.load(proj.processed)               # your watts, when known
            ftp = imap.ftp if imap is not None else float(body.get("ftp") or 0) or 200.0
            v = imap.apply(w) if imap is not None else w
            return {**session_steps(v, ftp), "title": w.title, "minutes": w.minutes, "tss": round(v.tss()),
                    "adjust": adj}
        if a == "adjust_session":             # a planned session's format a step easier / harder (Calendar): from now
            with PlanState.open(sp) as st:    # on, like a level; as easy / hard as it gets -> an error, nothing saved
                st.adjust(str(body.get("key")), int(body.get("step") or 0))
                st.replan(today())
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a == "away":                       # days you can't ride (Calendar): add or take off a range,
            with PlanState.open(sp) as st:    # or take off the days picked on the calendar ("days")
                for ds in [str(d) for d in body.get("days") or []][:366]:
                    st.set_away(ds, ds, away=False)
                if not body.get("days"):
                    st.set_away(str(body.get("from") or ""), str(body.get("to") or ""), body.get("note") or "",
                                away=not body.get("remove"))
                st.replan(today())
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a == "plan_decide":                # accept / reject proposed plan changes (Calendar)
            decision = body.get("decision")
            with PlanState.open(sp) as st:
                done = st.decide(decision, body.get("dates"))
            ftp = next((c for c in done if c.get("kind") == "ftp"), None)
            if ftp and decision == "accept":      # FTP from today on; history keeps its own
                from .tomledit import set_values
                set_values(proj.root / "goodpush.toml", "ftp", {"mode": "manual"})
                set_values(proj.root / "goodpush.toml", "ftp.history", {str(today().date()): int(ftp["to"])})
                proj = load_project(proj.root)
                data.proj = proj
                return jobs.start(f"FTP {ftp['to']} W - re-analysing", lambda log: _reanalyse(proj, log)) or "saved"
            return jobs.start("Updating the plan", lambda log: _plan_only(proj, log), queue=True)
        if a in ("incyclist_settings", "incyclist_sync"):
            if a == "incyclist_settings":
                from .tomledit import set_values
                v = body.get("values") or {}
                folder = str(v.get("folder") or "").strip()
                if folder and not (Path(folder).expanduser() / "workouts").is_dir():
                    raise ValueError(f"No Incyclist data in {folder} (expected a 'workouts' folder there)")
                set_values(proj.root / "goodpush.toml", "incyclist", {"sync": bool(v.get("sync")), "folder": folder})
                proj = load_project(proj.root)
                data.proj = proj
                if not v.get("sync"):
                    return "saved"
            return jobs.start("Sending the plan to Incyclist", lambda log: _plan_only(proj, log), queue=True)
        if a == "icu_push":                   # Settings > intervals.icu: the next days of the plan to its calendar
            from ..sync.intervals_icu import IntervalsIcu
            return jobs.start("Sending the plan to intervals.icu", lambda log: IntervalsIcu().push(
                proj.reports / "plan", today(), log, rewrite=lambda: _plan_only(proj, log)), queue=True)
        if a == "garmin_sync":
            if not hub.home_here():                  # home downloads; this computer asks it (PeerHub.garmin_job)
                return hub.garmin_job(lambda log: _garmin_sync(proj, log))
            return jobs.start("Updating from Garmin", lambda log: hub.updated(_garmin_sync(proj, log), log))
        if a == "garmin_import":
            from ..cli import _downloads_dir
            from ..sync import garmin
            dl = _downloads_dir(proj)
            found = sorted(dl.glob("goodpush_garmin_*.json"), key=lambda p: p.stat().st_mtime)
            if not found:
                raise ValueError(f"No goodpush_garmin_*.json in {dl}. Run the Garmin script first.")

            def job(log):
                garmin.import_browser(proj.path("garmin"), found[-1], progress=log)
                _full_rebuild(proj, log)
            return jobs.start(f"Importing {found[-1].name}", job)
        if a in ("override", "mark", "link", "unlink", "test_result"):
            ds = body.get("date") or str(today().date())
            with PlanState.open(sp) as st:   # waits for a plan rebuild that's mid-way, then applies on top
                if a == "override":
                    v = body.get("value") or ""
                    if v not in ("", "off", "sick"):
                        raise ValueError("value must be '', off or sick")
                    ov = st.setdefault("overrides", {})
                    ov.pop(ds, None) if not v else ov.__setitem__(ds, v)
                elif a == "mark":
                    st.mark(ds, body.get("value") or "")
                elif a == "unlink":           # calendar: this ride doesn't count for this session
                    st.unlink(ds)
                elif a == "link":             # calendar: a ride dropped on a planned session
                    if ds[:10] > str(today().date()):
                        raise ValueError("that session is still to come")
                    if not body.get("ride"):
                        raise ValueError("no ride given")
                    st.mark(ds, "done", ride=str(body["ride"]))
                else:
                    p20 = float(body.get("p20") or 0)
                    if not 50 <= p20 <= 700:
                        raise ValueError("20-min power should be between 50 and 700 W")
                    from .tomledit import set_values
                    ftp = int(round(0.95 * p20))
                    set_values(proj.root / "goodpush.toml", "plan", {"ftp": ftp})
                    st.note_test("test", ds)
                    proj = load_project(proj.root)
                    data.proj = proj
            # saved either way; when another job is busy the plan files catch up on its next rebuild
            if a == "override" and body.get("value") == "sick":            # an illness: learn from it too
                return jobs.start("Updating plan", lambda log: _sick_day(proj, log), queue=True)
            return jobs.start("Updating plan", lambda log: _plan_only(proj, log), queue=True)
        if a == "ride_power":                 # Ride page: this ride's power is wrong / keep it / undo (metrics.power_check)
            from ..metrics.power_check import PowerCheck
            from .tomledit import set_values
            ride = str(body.get("ride") or "").removesuffix(".0")
            if not ride:
                raise ValueError("no ride given")
            ing = proj.cfg.get("ingest", {})
            bad, keep = PowerCheck.edited(list(ing.get("bad_power") or []), list(ing.get("keep_power") or []),
                                          ride, str(body.get("value") or ""))
            set_values(proj.root / "goodpush.toml", "ingest", {"bad_power": bad, "keep_power": keep})
            proj = load_project(proj.root)
            data.proj = proj
            return jobs.start("Re-analysing without that ride's power" if body.get("value") == "bad"
                              else "Re-analysing with that ride's power", lambda log: _reanalyse(proj, log))
        if a == "settings":                   # any section of SETTINGS (config.py); the Plan page sends plan
            section = body.get("section") or "plan"
            vals = clean_settings(section, body.get("values") or {})
            _save_settings(proj, section, vals)
            proj = load_project(proj.root)
            data.proj = proj
            then = SETTINGS[section]["then"]
            if then == "reanalyse":
                return jobs.start("Re-analysing with your new settings", lambda log: _reanalyse(proj, log))
            if then == "replan":
                with PlanState.open(sp) as st:
                    st.replan(today())
                return jobs.start("Updating plan", lambda log: _plan_only(proj, log))
            return "saved"
        raise ValueError(f"unknown action {a!r}")

    class H(BaseHTTPRequestHandler):
        server_version = "goodpush"

        def log_message(self, fmt, *args):  # quiet
            pass

        # auth: localhost always; LAN needs the key (query once -> cookie)
        def _authed(self, q) -> bool:
            if not lan:
                return True
            if self.client_address[0] in ("127.0.0.1", "::1"):
                return True
            c = cookies.SimpleCookie(self.headers.get("Cookie", ""))
            got = (q.get("key") or [None])[0] or (c[COOKIE].value if COOKIE in c else None)
            return bool(got) and hmac.compare_digest(got, key)

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:              # the page went away meanwhile (a reload, another page): nobody
                self.close_connection = True     # to answer - not an error

        def _json(self, obj, code=200):
            self._send(code, json.dumps(_clean(obj), default=str, allow_nan=False).encode())

        def do_GET(self):
            nonlocal proj
            proj = data.proj                     # a sync may have brought new settings (PeerHub.reload)
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._authed(q):
                return self._send(403, b"goodpush: open the link with ?key=... printed by `goodpush app`",
                                  "text/plain")
            if "key" in q and lan:   # remember the key, drop it from the address bar
                return self._send(302, b"", extra={
                    "Location": u.path or "/",
                    "Set-Cookie": f"{COOKIE}={key}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict"})
            if u.path in ("/oauth/google/start", "/oauth/google"):     # Fitbit / Google Health sign-in
                if self.client_address[0] not in ("127.0.0.1", "::1"):
                    return self._send(403, b"Connect Google Health from the PC running goodpush.", "text/plain")
                try:
                    if u.path == "/oauth/google/start":
                        port = self.server.server_address[1]
                        return self._send(302, b"", extra={"Location": gh.auth_url(f"http://127.0.0.1:{port}/oauth/google")})
                    if q.get("error"):
                        raise GoogleHealthError(f"Google said: {q['error'][0]}")
                    gh.finish((q.get("code") or [""])[0], (q.get("state") or [""])[0])
                    jobs.start("Fetching your Google Health data", lambda log: _google_sync(proj, gh, log))
                    return self._send(302, b"", extra={"Location": "/#settings"})
                except GoogleHealthError as e:
                    msg = f'<p>{e}</p><p><a href="/#settings">Back to goodpush</a></p>'.encode()
                    return self._send(400, msg, "text/html; charset=utf-8")
            try:
                if u.path == "/api/state":
                    st = data.state()
                    pl = st.get("plan") or {}
                    stale = _plan_stale(pl, str(today().date()))     # a new day while running, or updated code
                    has_data = any((proj.processed / f).exists() for f in ("daily.parquet", "daily.csv.gz"))
                    planning = has_data or (proj.cfg.get("plan") or {}).get("enabled")   # no rides yet: if asked
                    tkey = f"{proj.root}|{today().date()}"         # per athlete (AthleteSwitch)
                    if stale and planning and tkey not in auto_done:   # once a day, never in a loop
                        auto_done.add(tkey)
                        if jobs.start("Updating plan for today", lambda log: _plan_only(proj, log)):
                            st["refreshing"] = True
                    st["code_stale"] = code_stale()              # goodpush updated since the app started
                    st["peers"] = hub.status()
                    st["athletes"] = switcher.status()
                    self._json(st)
                    ranges.page_served()                 # from now on the minute loop may start a refresh
                    return None
                if u.path == "/api/activities":
                    _, acts = data._tables()
                    if acts is None:
                        return self._json({"total": 0, "rows": []})
                    return self._json(activities_list(
                        acts, q=(q.get("q") or [""])[0], sport=(q.get("sport") or ["ride"])[0],
                        offset=int((q.get("offset") or [0])[0]), limit=min(int((q.get("limit") or [50])[0]), 200)))
                if u.path.startswith("/api/activity/") and u.path.endswith("/window"):   # selection / zoom
                    _, acts = data._tables()
                    try:
                        return self._json(activity_window(proj, acts, u.path.split("/")[3],
                                                          float((q.get("from") or [0])[0]), float((q.get("to") or [0])[0])))
                    except (KeyError, ValueError) as e:
                        return self._json({"error": str(e)}, 404 if isinstance(e, KeyError) else 400)
                if u.path.startswith("/api/activity/"):
                    _, acts = data._tables()
                    res = _read_json(proj.reports / "results.json", {}) or {}
                    try:
                        return self._json(activity_detail(proj, acts, u.path.rsplit("/", 1)[1], res))
                    except KeyError as e:
                        return self._json({"error": str(e)}, 404)
                if u.path == "/api/calendar":
                    daily, acts = data._tables()
                    return self._json(calendar_data(proj, daily, acts))
                if u.path == "/api/dips":                  # detected breaks + the periods you marked
                    from ..models.dips import BreakLabels
                    return self._json({**(_read_json(proj.processed / "dips.json", {}) or {}),
                                       "periods": BreakLabels(proj.processed).periods()})
                if u.path == "/api/curve":              # the Rider page's power curve over time (models.power_history)
                    from ..models.power_history import CurveHistory
                    from ..models.physio import PhysioRun
                    _, acts = data._tables()
                    try:
                        st = pd.read_parquet(proj.processed / PhysioRun.STATES)
                    except (OSError, ValueError):
                        return self._json({})
                    return self._json(CurveHistory(st, acts).build() if acts is not None else {})
                if u.path == "/api/power":              # the Rider page's power over time (models.power_history)
                    _, acts = data._tables()
                    return self._json(_power_history(proj, acts) if acts is not None else {})
                if u.path == "/api/model":              # the Model page: models.physio.PhysioDiagnostics + the checks
                    from ..checks import CheckReport, PhysioCalibrated
                    from ..checks.coverage import PosteriorCoverage   # the ranges checked on held-out data
                    from ..models.physio import PhysioDiagnostics, PhysioRun
                    from ..physio.model import VERSION
                    report = CheckReport(proj.processed)
                    out = _read_json(proj.processed / PhysioDiagnostics.FILE, {}) or {}
                    out["versions"] = {"fit": out.get("version"), "running": VERSION,   # the model this app runs
                                       "on_disk": PhysioRun.version_on_disk()}       # and the one on disk
                    cov = _read_json(report.folder / PosteriorCoverage.FILE, {}) or {}
                    if (cov.get("summary") or {}).get("efforts_by_kind"):
                        out["coverage"] = {"efforts": cov["summary"]["efforts_by_kind"]["all"],
                                           "readings": (cov["summary"].get("readings") or {}).get("all"),
                                           "folds": len(cov.get("folds") or []),
                                           "current": (cov.get("key") or {}).get("version") == VERSION}
                    bt = report.read(PhysioCalibrated.name).get("backtest") or {}
                    if bt.get("mae_pct") is not None:   # the forecast error on near-max days (goodpush physio --backtest)
                        out["backtest"] = {k: bt.get(k) for k in ("mae_pct", "bias_pct", "n")}
                        out["backtest"]["date"] = full_run.file_day(report.folder / f"{PhysioCalibrated.name}.json")
                    if "coverage" in out:
                        out["coverage"]["date"] = full_run.file_day(report.folder / PosteriorCoverage.FILE)
                    if out.get("inputs"):                # rides whose power the power check set aside, in the fit's days
                        out["inputs"]["set_aside"] = PhysioDiagnostics.set_aside(data._tables()[1],
                                                                                 out["inputs"]["first_day"])
                    out["run"] = full_run.status()
                    return self._json(out)
                if u.path in ("/api/diagnostics", "/api/log"):     # Settings > Diagnostics (app/diagnostics.py)
                    from .diagnostics import Diagnostics
                    diag = Diagnostics.of_app(proj, jobs, hub, self.server.server_address[1], lan)
                    if u.path == "/api/diagnostics":
                        return self._json({**diag.report(), "shareable_text": diag.shareable_text()})
                    try:
                        return self._json(diag.tail((q.get("where") or [""])[0], (q.get("name") or [""])[0],
                                                    int((q.get("lines") or [diag.LOG_LINES])[0])))
                    except KeyError as e:
                        return self._json({"error": str(e)}, 404)
                if u.path == "/api/model_run":          # the Model page's full run, while it goes (FullModelRun)
                    return self._json(full_run.status())
                if u.path == "/api/alive":
                    return pages.hold(self, run_id)
                if u.path == "/api/job":
                    from ..launcher import App
                    return self._json(dict(jobs.status(), build=App.build(), code_stale=code_stale(), started=run_id,
                                           **data.versions(),
                                           peer_waiting=hub.server.pairing.waiting()
                                           if self.client_address[0] in ("127.0.0.1", "::1") else None))
                if u.path == "/api/snippet":
                    from ..sync import garmin
                    js = garmin.browser_snippet(proj.path("garmin"), strava_dir=proj.path("strava"))
                    return self._send(200, js.encode(), "text/plain; charset=utf-8")
                if u.path == "/files/plan.ics":
                    p = proj.reports / "plan" / "plan.ics"
                    return self._file(p, "text/calendar")
                if u.path.startswith("/files/workouts/"):
                    name = Path(u.path).name
                    if not name.startswith("gp_") or not name.endswith((".fit", ".zwo", ".txt")):
                        return self._send(404, b"not found", "text/plain")
                    return self._file(proj.reports / "plan" / "workouts" / name,
                                      "application/octet-stream",
                                      {"Content-Disposition": f'attachment; filename="{name}"'})
                if u.path == "/dashboard":
                    return self._file(proj.reports / "dashboard.html", "text/html; charset=utf-8")
                name = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
                p = (STATIC / name).resolve()
                if STATIC.resolve() not in p.parents or not p.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                return self._file(p, ctype)
            except ConnectionError:              # the page went away while we answered: nothing to tell it
                self.close_connection = True
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._json({"error": str(e)}, 500)

        def _file(self, p: Path, ctype, extra=None):
            if not p.is_file():
                return self._send(404, b"not found - run `goodpush run` first", "text/plain")
            return self._send(200, p.read_bytes(), ctype, extra)

        def do_POST(self):
            nonlocal proj
            proj = data.proj
            u = urlparse(self.path)
            if not self._authed({}) or self.headers.get("X-GoodPush") != "1":
                return self._send(403, b"forbidden", "text/plain")
            if u.path == "/api/upload":              # an account export dropped on the page
                try:
                    path, kind = save_upload(proj, self.rfile, int(self.headers.get("Content-Length") or 0),
                                             self.headers.get("X-Filename") or "export.zip")
                except ValueError as e:
                    return self._json({"error": str(e)}, 400)

                def job(log):
                    from ..importer import import_path
                    import_path(proj, path, progress=log)     # the upload is kept in data/uploads/
                    _full_rebuild(proj, log)
                started = jobs.start(f"Importing your {kind.title()} export", job, queue=True)
                return self._json({"kind": kind, "started": bool(started), "queued": started == "queued"})
            if u.path == "/api/strava_upload":       # Send to Strava: the body is the chart PNG (may be empty)
                if self.client_address[0] not in ("127.0.0.1", "::1"):
                    return self._json({"error": "Send a ride to Strava from the computer running goodpush."}, 400)
                from ..io.strava import StravaUpload
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 <= n <= StravaUpload.PNG_MAX:
                    return self._json({"error": "the chart picture is too big"}, 400)
                try:
                    StravaUpload(proj.path("rides"), proj.processed).open(
                        (parse_qs(u.query).get("id") or [""])[0], self.rfile.read(n) if n else b"")
                except (OSError, ValueError) as e:
                    return self._json({"error": str(e)}, 400)
                return self._json({"ok": True})
            if u.path != "/api/action":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(n, 100_000)) or b"{}")
                if body.get("action") == "garmin_test":
                    from ..sync import garmin_api
                    garmin_api.access_token(force=True)
                    return self._json({"ok": True, "message": "Garmin accepted your key"})
                if body.get("action") == "ride":         # the overlay opens on this computer, not on the phone
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self._json({"error": "Start the ride on the computer running goodpush (the overlay opens there)."}, 400)
                    key = str(body.get("key") or "")
                    saved = lambda rid: hub.after_ride() or jobs.start("Processing your ride",      # noqa: E731
                                                                       lambda log: _add_ride(proj, key, rid, log), queue=True)
                    started = jobs.start("Opening the ride window", lambda log: RideLauncher.start(proj, key, log, saved),
                                         queue=True)
                    return self._json({"started": bool(started), "busy": not started, "queued": started == "queued"})
                if body.get("action") == "quit":         # the top bar's Quit: stop goodpush on this computer
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self._json({"error": "Quit goodpush on the computer running it."}, 400)
                    busy = [f"“{jobs.status()['name']}” is still running and stops."] if jobs.status()["running"] else []
                    if RideLauncher.proc is not None and RideLauncher.proc.poll() is None:
                        busy.append("A ride window is open: its ride isn't added to your rides once you save it.")
                    if full_run.running():
                        busy.append("The model run goes on by itself, without its backtest.")
                    if busy and not body.get("force"):
                        return self._json({"title": "Quit goodpush anyway?", "ask": " ".join(busy), "yes": "Quit",
                                           "no": "Keep it running", "danger": True})
                    self._json({"ok": True})
                    threading.Thread(target=self.server.shutdown, name="goodpush quit", daemon=True).start()
                    return None
                if body.get("action") == "reveal":       # Diagnostics: a folder or log in Finder / Explorer
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self._json({"error": "Open folders on the computer running goodpush."}, 400)
                    from ..io.strava import StravaUpload
                    from .diagnostics import Diagnostics
                    diag = Diagnostics.of_app(proj, jobs, hub, self.server.server_address[1], lan)
                    if body.get("name"):
                        target = diag.log_path(str(body.get("where") or ""), str(body["name"]))
                    else:
                        target = diag.folders()[str(body.get("folder") or "")]
                    if not body.get("name"):
                        target.mkdir(parents=True, exist_ok=True)
                    StravaUpload.reveal(target)
                    return self._json({"ok": True})
                if body.get("action") == "incyclist_settings" and self.client_address[0] not in ("127.0.0.1", "::1"):
                    return self._json({"error": "Change the Incyclist settings on the PC running goodpush."}, 400)
                if body.get("action") in ("icu_key", "icu_forget"):     # the key only travels from this PC's browser
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self._json({"error": "Set the intervals.icu key from the PC running goodpush."}, 400)
                    from ..sync.intervals_icu import IntervalsIcu
                    if body["action"] == "icu_forget":
                        IntervalsIcu.forget()
                        return self._json({"ok": True})
                    return self._json({"ok": True, "name": IntervalsIcu.save_key(str(body.get("key") or ""))})
                if body.get("action") in ("garmin_key", "garmin_forget"):
                    # the key only ever travels from this PC's own browser to this PC
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self._json({"error": "Set the Garmin key from the PC running goodpush, "
                                                    "not from your phone."}, 400)
                    from ..sync import garmin_api
                    if body["action"] == "garmin_forget":
                        garmin_api.forget_key()
                        return self._json({"ok": True})
                    garmin_api.save_key(str(body.get("key") or ""))
                    garmin_api.access_token(force=True)       # fails loudly if Garmin rejects it
                    started = jobs.start("Updating from Garmin", lambda log: hub.updated(_garmin_sync(proj, log), log))
                    return self._json({"ok": True, "started": bool(started)})
                if body.get("action") in ("gh_client", "gh_forget", "gh_sync"):     # Fitbit via Google Health
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self._json({"error": "Set up Google Health from the PC running goodpush."}, 400)
                    if body["action"] == "gh_forget":
                        GoogleHealth.forget()
                        return self._json({"ok": True})
                    if body["action"] == "gh_client":
                        v = body.get("values") or {}
                        GoogleHealth.save_client(str(v.get("client_id") or ""), str(v.get("client_secret") or ""))
                        return self._json({"ok": True, "go": "/oauth/google/start"})
                    started = jobs.start("Updating from Google Health", lambda log: _google_sync(proj, gh, log))
                    return self._json({"ok": True, "started": bool(started)})
                if str(body.get("action")).startswith("peer_"):        # Settings > Other devices
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self._json({"error": "Pair and sync machines from the computer running goodpush."}, 400)
                    a, name = body["action"], str(body.get("name") or "")
                    if a == "peer_pair":
                        return self._json(hub.pair())
                    if a == "peer_join":
                        return self._json({"name": hub.join(str(body.get("address") or ""), str(body.get("code") or ""))})
                    if a == "peer_sync":
                        hub.peers().get(name)
                        return self._json({"started": bool(hub.sync_job(name, str(body.get("direction") or "both")))})
                    if a == "peer_forget":
                        hub.forget(name)
                        return self._json({"ok": True})
                    if a == "peer_home":
                        hub.set_home(name)
                        return self._json({"ok": True})
                    if a == "peer_sync_all":
                        return self._json({"started": bool(hub.sync_all_job())})
                    if a == "peer_every":
                        hub.set_every(float(body.get("hours") or 0))
                        return self._json({"ok": True})
                    if a == "peer_rename":
                        hub.rename(name, str(body.get("label") or ""))
                        return self._json({"ok": True})
                    if a == "peer_allow":
                        return self._json({"name": hub.server.pairing.answer(bool(body.get("allow")))})
                    if a == "peer_serve":
                        hub.set_serve(bool(body.get("on")))
                        return self._json({"ok": True})
                if body.get("action") in ("athlete_switch", "athlete_add", "athlete_remove", "athlete_move"):
                    if self.client_address[0] not in ("127.0.0.1", "::1"):     # the app name's menu
                        return self._json({"error": "Switch athletes on the computer running goodpush."}, 400)
                    if body["action"] == "athlete_move":
                        started = switcher.move(str(body.get("name") or ""), str(body.get("folder") or ""))
                        return self._json({"started": bool(started), "busy": not started})
                    if body["action"] == "athlete_remove":
                        switcher.remove(str(body.get("name") or ""))
                        return self._json({"athletes": switcher.status()})
                    if body["action"] == "athlete_add":
                        switcher.add(str(body.get("name") or ""), str(body.get("folder") or ""))
                    else:
                        switcher.switch(str(body.get("name") or ""))
                    proj = data.proj
                    return self._json({"ok": True})
                if body.get("action") == "model_run":       # Model page: NUTS, then the backtest (FullModelRun)
                    full_run.start()                         # ValueError: why not now (shown as the error)
                    return self._json({"ok": True})
                if body.get("action") == "model_run_stop":
                    full_run.stop()
                    return self._json({"ok": True})
                if body.get("action") == "copy_snippet":
                    # the app runs on your PC, so put the script on the PC's clipboard directly
                    # (browsers often refuse clipboard writes from a local http page)
                    from ..cli import _to_clipboard
                    from ..sync import garmin
                    js = garmin.browser_snippet(proj.path("garmin"), strava_dir=proj.path("strava"))
                    local = self.client_address[0] in ("127.0.0.1", "::1")
                    return self._json({"copied": bool(local and _to_clipboard(js))})
                started = act(body)
                if isinstance(started, dict):          # a question for you first (e.g. merge two phases?)
                    return self._json(started)
                if started == "saved":
                    return self._json({"ok": True, "saved": True})
                return self._json({"started": bool(started), "busy": not started, "queued": started == "queued"})
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)

    H.pages = pages
    return H


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent; picks the outgoing interface
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def app_key() -> str:
    from ..sync import secrets_dir
    p = secrets_dir() / "app_key"
    if p.exists():
        k = p.read_text().strip()
        if len(k) >= 16:
            return k
    k = secrets.token_urlsafe(18)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(k)
    return k


def serve(proj, port: int = 8765, lan: bool = False, open_browser: bool = True):
    key = app_key() if lan else None
    host = "0.0.0.0" if lan else "127.0.0.1"
    made = plan_at_start(proj)
    handler = make_handler(proj, key, lan, warm=not made, port=port)
    httpd = ThreadingHTTPServer((host, port), handler)
    local = f"http://127.0.0.1:{port}/"
    print(f"goodpush app running:  {local}")
    if lan:
        print(f"On your phone (same Wi-Fi):  http://{_lan_ip()}:{port}/?key={key}")
        print("  Keep that link private - it opens your training and health data.")
        print("  Windows may ask to allow Python through the firewall: allow it on Private networks.")
    print("Stop with Ctrl+C.")
    if open_browser:                        # a tab still open from before reloads itself instead
        threading.Thread(target=handler.pages.open_unless_seen, args=(local,), daemon=True).start()
    try:
        httpd.serve_forever()                # until Ctrl+C or the app's Quit (httpd.shutdown)
        print("stopped (Quit in the app)")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
