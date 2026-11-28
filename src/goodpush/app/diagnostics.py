"""Settings > Diagnostics: what a rider or a developer looks at first when something is off. The app's and the
model's versions, the athletes on this computer, how much data there is, the sources, the folders and the logs
(AppLog's, and each run's in Project.logs), with the last lines of any of them.

Each fact says whether it may leave the computer: `private` ones (folders, names, sign-in dates) stay out of
`shareable()`, the starting point for an athlete's diagnostic export later. A new feature adds its facts here.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

from ..folders import Folders


@dataclass
class Fact:
    """one line of Diagnostics: `value` as shown; `private`: never in shareable(); `folder`: a key of
    Diagnostics.folders() the page can show in Finder / Explorer."""
    label: str
    value: str
    private: bool = False
    folder: str = ""


class AppLog:
    """the app's own output (a job's start and end, what failed and its traceback) also into app.log in this
    computer's folder, not the athlete's: it covers every athlete the app showed. The installed app has no terminal,
    so without it that output was lost. Each line gets its time; the log of the run before is kept
    (app.previous.log), and a log past MAX_BYTES starts again. Only the entry points start it (launcher.main,
    `goodpush app`), never a test or a library import."""
    NAME = "app.log"
    PREVIOUS = "app.previous.log"
    MAX_BYTES = 5_000_000
    lock = threading.Lock()
    file = None
    at_line_start = True

    @staticmethod
    def folder() -> Path:
        return Folders.machine() / "logs"

    @classmethod
    def start(cls) -> Path | None:
        """stdout and stderr also into app.log from now on (once per process); None if it can't be written."""
        with cls.lock:
            if cls.file is not None:
                return cls.folder() / cls.NAME
            try:
                cls.folder().mkdir(parents=True, exist_ok=True)
                cls.file = cls.open_new()
            except OSError as e:
                print(f"goodpush: no app log ({e})", file=sys.stderr)
                return None
        sys.stdout, sys.stderr = AppLogStream(sys.stdout), AppLogStream(sys.stderr)
        print(f"goodpush {cls.version()} started; log: {cls.folder() / cls.NAME}")
        return cls.folder() / cls.NAME

    @classmethod
    def open_new(cls):
        """the last run's log becomes app.previous.log, and a new app.log is opened."""
        p = cls.folder() / cls.NAME
        if p.exists():
            os.replace(p, cls.folder() / cls.PREVIOUS)
        return open(p, "w", encoding="utf-8", buffering=1)

    @classmethod
    def write(cls, text: str) -> None:
        with cls.lock:
            if cls.file is None or cls.file.closed:
                return
            stamp = time.strftime("%H:%M:%S ")
            parts = text.split("\n")
            out = []
            for i, part in enumerate(parts):
                if part and cls.at_line_start:
                    part = stamp + part
                out.append(part)
                cls.at_line_start = i < len(parts) - 1 or (not part and cls.at_line_start)
            cls.file.write("\n".join(out))
            if cls.file.tell() > cls.MAX_BYTES:
                cls.file.close()
                cls.file = cls.open_new()

    @staticmethod
    def version() -> str:
        from .. import __version__
        return __version__


class AppLogStream:
    """stdout or stderr as before (the terminal, or nothing for the installed app), and AppLog."""

    def __init__(self, stream):
        self.stream = stream if stream is not None else open(os.devnull, "w", encoding="utf-8")

    def write(self, text: str) -> int:
        AppLog.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()

    def __getattr__(self, name):                 # isatty, encoding, ...: the stream's
        return getattr(self.stream, name)


class Diagnostics:
    """the facts and logs for one athlete's project, as the app runs now."""
    PACKAGES = ("numpy", "pandas", "scipy", "jax", "numba", "pyarrow", "h5py", "garminconnect", "PySide6")
    LOG_BYTES = 512_000            # a log's last half megabyte at most
    LOG_LINES = 400                # lines shown by default
    MAX_LINES = 5000
    RUN_LOGS = 40                  # the newest run logs listed
    STARTED = time.time()          # this process (the app run)

    def __init__(self, proj, jobs: dict | None = None, peers: int = 0, port: int = 0, lan: bool = False,
                 code_stale: bool = False):
        self.proj, self.jobs, self.peers, self.port, self.lan = proj, jobs or {}, peers, port, lan
        self.code_stale = code_stale

    @classmethod
    def of_app(cls, proj, jobs, hub, port: int, lan: bool) -> Diagnostics:
        """for the app's server: its jobs (server.Jobs), paired computers (server.PeerHub), port and code."""
        from .server import code_stale
        return cls(proj, jobs.status(), peers=len(hub.status().get("computers") or []), port=port, lan=lan,
                   code_stale=code_stale())

    # --- the facts ------------------------------------------------------------------------------------------
    def groups(self) -> list[dict]:
        out = []
        for gid, title, make in (("app", "App", self.app), ("model", "Model and analysis", self.model),
                                 ("athletes", "Athletes", self.athletes), ("data", "Data", self.data),
                                 ("sources", "Sources", self.sources), ("folders", "Folders", self.folder_facts),
                                 ("packages", "Libraries", self.packages)):
            try:
                facts = make()
            except Exception as e:  # noqa: BLE001 - one broken group mustn't hide the rest
                facts = [Fact("Couldn't read", f"{type(e).__name__}: {e}")]
            out.append({"id": gid, "title": title, "facts": [asdict(f) for f in facts]})
        return out

    def report(self) -> dict:
        return {"groups": self.groups(), "logs": self.logs(), "folders": {k: str(v) for k, v in self.folders().items()},
                "made": time.strftime("%Y-%m-%d %H:%M:%S")}

    def shareable(self) -> dict:
        """the facts that may leave this computer (no folders, names or sign-in details): what a rider can paste
        into a bug report now, and the start of the athlete diagnostic export."""
        return {g["title"]: {f["label"]: f["value"] for f in g["facts"] if not f["private"]} for g in self.groups()}

    def shareable_text(self) -> str:
        lines = []
        for title, facts in self.shareable().items():
            lines.append(f"## {title}")
            lines += [f"- {k}: {v}" for k, v in facts.items()]
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def app(self) -> list[Fact]:
        from .. import __version__
        installed = bool(getattr(sys, "frozen", False))
        mac = platform.mac_ver()[0]
        system = f"macOS {mac}" if mac else f"{platform.system()} {platform.release()}"
        job = self.jobs.get("name")
        state = ("running" if self.jobs.get("running") else "finished" if self.jobs.get("ok")
                 else "failed" if self.jobs.get("ok") is False else "")
        return [Fact("GoodPush", __version__),
                Fact("Running as", "the installed app" if installed else "from source (a development install)"),
                Fact("Updated since it started", "yes: restart it to run the new version" if self.code_stale else "no"),
                Fact("Started", time.strftime("%Y-%m-%d %H:%M", time.localtime(self.STARTED))),
                Fact("Computer", f"{system} ({platform.machine()})"),
                Fact("Python", platform.python_version()),
                Fact("Port", f"{self.port}" + (" (also on your network: --lan)" if self.lan else ""), private=True),
                Fact("Last job", f"{job} ({state})" if job else "none since it started")]

    def model(self) -> list[Fact]:
        from ..models.physio import PhysioDiagnostics, PhysioRun
        from ..physio.model import VERSION
        processed = self.proj.processed
        fit = self.read_json(processed / PhysioDiagnostics.FILE)
        res = self.read_json(self.proj.reports / "results.json")
        on_disk = PhysioRun.version_on_disk()
        facts = [Fact("Model version (this app runs)", f"v{VERSION}"),
                 Fact("Model version (code on disk)", f"v{on_disk}" + ("" if on_disk == VERSION else ": restart the app"))]
        if fit:
            v = fit.get("version")
            facts += [Fact("Model version (last fit)", f"v{v}" + ("" if v == VERSION else ": refits on the next update")),
                      Fact("Last fit's data up to", str(fit.get("day") or "–")),
                      Fact("Ranges (5–95 %) from", str(fit.get("draws_day") or "none yet: a full run on the Model page"))]
        else:
            facts.append(Fact("Last fit", "none yet"))
        facts += [Fact("Last analysis", str(res.get("generated") or "never")[:16].replace("T", " ")),
                  Fact("Last plan update", self.when(processed / "plan_state.json"))]
        return facts

    def athletes(self) -> list[Fact]:
        from ..athletes import Athletes
        reg = Athletes()
        current = reg.name_of(self.proj.root) or ""
        return [Fact("Athletes on this computer", str(len(reg.folders))),
                Fact("Shown now", current or "the app's own folder (not registered)", private=True),
                Fact("Default", reg.default or "–", private=True),
                Fact("Paired computers", str(self.peers))]

    def data(self) -> list[Fact]:
        from ..io import store
        processed = self.proj.processed
        acts, daily = store.load(processed, "activities"), store.load(processed, "daily")
        facts = [Fact("Activities", str(0 if acts is None else len(acts)))]
        if acts is not None and len(acts) and "date" in acts:
            d = acts["date"].astype(str).str[:10]
            facts.append(Fact("First / last activity", f"{d.min()} / {d.max()}"))
        facts.append(Fact("Days of data", str(0 if daily is None else len(daily))))
        facts.append(Fact("Athlete folder size", self.size(self.proj.root)))
        return facts

    def sources(self) -> list[Fact]:
        from ..sync import garmin_api, incyclist
        from ..sync.google_health import GoogleHealth
        from ..sync.intervals_icu import IntervalsIcu
        g, i, gh = garmin_api.key_info(), IntervalsIcu.info(), GoogleHealth.info()
        last = (g.get("last_sync") or {}).get("at")
        inc = incyclist.status(self.proj.cfg)
        auto = (self.proj.cfg.get("garmin") or {}).get("auto_sync_hours") or 0
        return [Fact("Garmin", ("key set" + (f", last check {str(last)[:16].replace('T', ' ')}" if last else ", not checked yet"))
                     if g.get("configured") else "no key"),
                Fact("Garmin check by itself", f"every {auto} h" if auto else "off"),
                Fact("intervals.icu", "key set" if i.get("configured") else "no key"),
                Fact("Fitbit (Google Health)", "connected" if gh.get("connected") else "not connected"),
                Fact("Incyclist", "writing the plan" if inc.get("sync") else "found, off" if inc.get("found") else "not found")]

    def folders(self) -> dict[str, Path]:
        """the folders the page names and can open: key -> path."""
        return {"athlete": self.proj.root, "logs": self.proj.logs, "app_logs": AppLog.folder(),
                "machine": Folders.machine()}

    def folder_facts(self) -> list[Fact]:
        names = {"athlete": "Athlete folder", "logs": "Run logs (this athlete, this computer)",
                 "app_logs": "App log", "machine": "This computer's goodpush folder (sign-ins, athletes)"}
        return [Fact(names[k], str(p), private=True, folder=k) for k, p in self.folders().items()]

    def packages(self) -> list[Fact]:
        out = []
        for name in self.PACKAGES:
            try:
                out.append(Fact(name, metadata.version(name)))
            except metadata.PackageNotFoundError:
                out.append(Fact(name, "not installed"))
        return out

    # --- the logs -------------------------------------------------------------------------------------------
    def logs(self) -> list[dict]:
        """the app log (now and the run before) and the newest run logs, newest first within each."""
        out = []
        app = AppLog.folder()
        for n in (AppLog.NAME, AppLog.PREVIOUS):
            if (app / n).is_file():
                out.append(self.log_entry("app", app / n))
        runs = sorted((p for p in self.proj.logs.glob("*.log") if p.is_file()), key=lambda p: p.stat().st_mtime,
                      reverse=True)
        out += [self.log_entry("run", p) for p in runs[:self.RUN_LOGS]]
        return out

    @staticmethod
    def log_entry(where: str, p: Path) -> dict:
        st = p.stat()
        return {"where": where, "name": p.name, "size": st.st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))}

    def log_path(self, where: str, name: str) -> Path:
        """a listed log only (no other file, no path): KeyError otherwise."""
        for e in self.logs():
            if e["where"] == where and e["name"] == name:
                return (AppLog.folder() if where == "app" else self.proj.logs) / name
        raise KeyError(f"no log {name!r}")

    def tail(self, where: str, name: str, lines: int = LOG_LINES) -> dict:
        """a log's last `lines` lines (from its last LOG_BYTES)."""
        p = self.log_path(where, name)
        lines = max(1, min(int(lines), self.MAX_LINES))
        size = p.stat().st_size
        with open(p, "rb") as f:
            f.seek(max(0, size - self.LOG_BYTES))
            text = f.read().decode("utf-8", errors="replace")
        rows = text.splitlines()
        if size > self.LOG_BYTES:
            rows = rows[1:]                       # the first one is cut
        return {"where": where, "name": name, "path": str(p), "size": size, "text": "\n".join(rows[-lines:]),
                "cut": len(rows) > lines or size > self.LOG_BYTES}

    # --- helpers --------------------------------------------------------------------------------------------
    @staticmethod
    def read_json(p: Path) -> dict:
        try:
            return json.loads(p.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def when(p: Path) -> str:
        try:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        except OSError:
            return "never"

    @staticmethod
    def size(root: Path) -> str:
        total = 0
        for dirpath, _, files in os.walk(root):
            for n in files:
                try:
                    total += os.stat(os.path.join(dirpath, n)).st_size
                except OSError:
                    pass
        for unit in ("bytes", "KB", "MB", "GB"):
            if total < 1024 or unit == "GB":
                return f"{total:.0f} {unit}" if unit == "bytes" else f"{total:.1f} {unit}"
            total /= 1024
        return ""
