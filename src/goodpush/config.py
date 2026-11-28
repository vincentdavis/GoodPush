"""Project configuration (goodpush.toml) and paths."""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path


if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib
    except ImportError:  # only needed when a goodpush.toml is actually read
        tomllib = None

CONFIG_NAME = "goodpush.toml"

DEFAULTS = {
    "athlete": {"name": "", "sex": "male", "birth_year": 0, "timezone": "UTC",
                "weight_kg": 0, "max_hr": 0, "resting_hr": 0, "lthr": 0},
    "paths": {"strava": "data/raw/strava", "garmin": "data/raw/garmin", "google_health": "data/raw/google_health",
              "rides": "data/raw/goodpush",
              "processed": "data/processed", "reports": "reports"},
    "ftp": {"mode": "auto", "window_days": 90, "auto_method": "fitted", "floor_days": 180,
            "fallback": 200, "history": {}},
    "load": {"ctl_days": 42, "atl_days": 7, "include_sports": "all",
             "default_tss_per_hour": {"ride": 45, "run": 50, "strength": 30,
                                      "yoga": 10, "other": 30}},
    "ingest": {"workers": 0, "keep_streams": True, "max_power": 2000, "bad_power": [], "keep_power": []},
    "analysis": {"driver_window_weeks": 8, "progress_min_gain": 0.04},
    "plan": {"enabled": False, "hours_per_week": 4.0, "plan_starting_hours": 0.0, "availability": [], "rest_days": ["mon"],
             "key_days": ["tue", "thu"], "long_day": "sat", "recovery_days": ["fri"],
             "min_ramp": 1.5, "max_ramp": 2.5, "test_every_weeks": 6, "ramp_every_weeks": 6, "test_efforts": False,
             "ftp": 0},
    "incyclist": {"sync": False, "folder": ""},
    "garmin": {"auto_sync_hours": 0},
    "physio": {"since_days": 0},     # since_days: models.physio.PhysioRun.since_days (0: all your data);
                                     # ranges_every_days: app.server.DailyRanges.every_days (7; old daily_ranges)
    "sync": {"home": "", "serve": True, "every_hours": 2.0},
    "ride": {"resistance": 30, "bike_kg": 9, "opacity": 0.6, "scale": 1.0, "x": -1, "y": -1, "click_through": False},
    "events": [],
}
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
EVENT_TYPES = ("illness", "injury", "race", "travel", "life", "note")


def setting(key: str, kind: str, label: str, help: str = "", **opts) -> dict:
    """One option of goodpush.toml for the Settings page. kind: text | path | tz | int | float | bool | choice |
    day | days | week | numbers | history | events | lines (one entry a line, a list); opts: lo / hi / step (numbers), choices [(value, label)],
    count (days: exactly this many), keys (numbers: the dict's keys), zero (0 = automatic, besides lo..hi)."""
    return {"key": key, "kind": kind, "label": label, "help": help, **opts}


# Every option, by section, as the Settings page shows them. `then`: what saving it redoes - reanalyse (the
# analysis, plan and reports from the cached rides), replan (the plan only), reload (the paths: read again)
# or nothing (used next time: the next import or ride). Incyclist has its own card (and handler).
SETTINGS = {
    "athlete": {"title": "You", "then": "reanalyse", "fields": [
        setting("name", "text", "Name", "shown in the report title", hi=40),
        setting("weight_kg", "float", "Weight kg", "0 = from your Garmin weight log", lo=30, hi=200, step=.1, zero=True),
        setting("max_hr", "int", "Max HR", "0 = from your rides", lo=120, hi=230, zero=True),
        setting("resting_hr", "int", "Resting HR", "0 = from Garmin (or 55 without data)", lo=25, hi=100, zero=True),
        setting("lthr", "int", "Threshold HR", "lactate threshold; 0 = from your best 20-min HR", lo=100, hi=210, zero=True),
        setting("sex", "choice", "Sex", "for power-profile percentiles", choices=[("male", "Male"), ("female", "Female")]),
        setting("birth_year", "int", "Birth year", "0 = unknown", lo=1920, hi=2020, zero=True),
        setting("timezone", "tz", "Time zone", "puts rides and sleep on the right days, e.g. America/Los_Angeles"),
    ]},
    "plan": {"title": "Training plan", "then": "replan", "fields": [
        setting("enabled", "bool", "Make a plan"),
        setting("hours_per_week", "float", "Hours per week", "the most in a week; recovery weeks get less",
                lo=2, hi=25, step=.5),
        setting("plan_starting_hours", "float", "Starting hours",
                "the build-up starts from at least this week (+10 % a loading week up to hours per week); "
                "0 = your last 4 weeks' average", lo=2, hi=25, step=.5, zero=True),
        setting("long_day", "day", "Long ride day", "only used without hours per day"),
        setting("ftp", "int", "FTP override", "0 = the FTP from your data", lo=50, hi=600, zero=True),
        setting("availability", "week", "Hours available per day",
                "the longest day gets the long ride, 0 = no riding; all blank = use the days below", hi=12, step=.5),
        setting("rest_days", "days", "Rest days"),
        setting("key_days", "days", "Key days", "the two hard days (tests go on the first)", count=2),
        setting("recovery_days", "days", "Recovery days", "easy spins, e.g. the day before a key day"),
        setting("intensity", "choice", "Hard days",
                "Model: your fitted physiology picks each key day's session for the biggest gain on your race days "
                "(pyramidal until it is fitted). Polarized: easy riding plus hard work above threshold only "
                "(VO2max, over-unders, anaerobic) - no tempo or sweet spot. Pyramidal: one sweet-spot day in "
                "base (tempo only in long rides), threshold and VO2max later. Recovery weeks: an LSCT.",
                choices=[("model", "Model"), ("polarized", "Polarized"), ("pyramidal", "Pyramidal")]),
        setting("mix_draws", "int", "Model: draws",
                "Hard days by the model: posterior draws the choice is averaged over (robust to effects the data "
                "barely pin down); 0 = the best fit alone (faster)", lo=0, hi=40, zero=True),
        setting("load_model", "choice", "Weekly load",
                "Model: your fitted physiology picks each week's riding for the best race-day form; blocks build "
                "up, then a recovery week, and what fatigue costs you is learned from your data. CTL rules: a "
                "fixed fitness ramp and tapers.",
                choices=[("physio", "Physiological model"), ("ctl", "CTL rules")]),
        setting("min_ramp", "float", "CTL ramp, least", "CTL rules: points a loading week adds at least",
                lo=0, hi=10, step=.1),
        setting("max_ramp", "float", "CTL ramp, most", "CTL rules: ... and at most", lo=0, hi=10, step=.1),
        setting("test_every_weeks", "int", "FTP test every (weeks)", "a 20-min test when the last is older",
                lo=2, hi=52),
        setting("ramp_every_weeks", "int", "Ramp test every (weeks)", "ramp test with alphaHRV for LT1 / LT2; 0 = never",
                lo=2, hi=52, zero=True),
        setting("test_efforts", "bool", "All-out efforts in key sessions",
                "now and then 1, 3 or 5 min all-out opens one: your power curve measured"),
    ]},
    "ftp": {"title": "FTP", "then": "reanalyse", "fields": [
        setting("mode", "choice", "FTP from", "Manual: your history below wins from each date on",
                choices=[("auto", "Your rides"), ("manual", "Your history")]),
        setting("auto_method", "choice", "Estimated by",
                "Model's LT2: the physio model's threshold, your ramp tests weighing most (until it has a fit: "
                "Fitted). Fitted: to Garmin's FTP, your history and your tests (needs 10+ values, else critical power)",
                choices=[("model", "Model's LT2"), ("fitted", "Fitted model"), ("cp", "Critical power"),
                         ("p20", "95 % of best 20 min")]),
        setting("window_days", "int", "Window (days)", "best efforts from this many days", lo=14, hi=365),
        setting("floor_days", "int", "Floor (days)", "never below 95 % of the best 20 min of this many days; 0 = off",
                lo=30, hi=730, zero=True),
        setting("fallback", "int", "Fallback FTP", "before there's any power data", lo=50, hi=600),
        setting("history", "history", "Your FTP history", "one per line: date and watts, e.g. 2025-03-01 250"),
    ]},
    "load": {"title": "Training load", "then": "reanalyse", "fields": [
        setting("ctl_days", "int", "Load (CTL) days", lo=14, hi=90),
        setting("atl_days", "int", "Fatigue (ATL) days", lo=3, hi=21),
        setting("include_sports", "choice", "Counts toward load",
                choices=[("all", "All activities"), ("cycling", "Cycling only")]),
        setting("default_tss_per_hour", "numbers", "TSS per hour without power or HR",
                keys=["ride", "run", "strength", "yoga", "other"], lo=0, hi=150),
    ]},
    "events": {"title": "Life events", "then": "reanalyse", "fields": [
        setting("events", "events", "Illness, injury, travel ...",
                "one per line: start, end, type, note - e.g. 2026-02-03, 2026-02-10, illness, flu. Types: "
                + ", ".join(EVENT_TYPES) + ". Shown on charts and used when reading overreaching."),
    ]},
    "ride": {"title": "Riding (goodpush ride)", "then": "nothing", "fields": [
        setting("resistance", "int", "Resistance %", "free rides and resistance mode", lo=0, hi=100),
        setting("bike_kg", "float", "Bike kg", "sim mode: a road, with you and the bike", lo=3, hi=30, step=.1),
        setting("opacity", "float", "Bar background", "0 = see-through, 1 = solid", lo=0, hi=1, step=.05),
        setting("scale", "float", "Bar size", "1 = normal", lo=.5, hi=2, step=.05),
        setting("x", "int", "Bar x", "-1 = top centre (dragging the bar sets it)", lo=-1, hi=10000),
        setting("y", "int", "Bar y", "-1 = top centre", lo=-1, hi=10000),
        setting("click_through", "bool", "Click-through", "clicks go to the window underneath (off from the tray icon)"),
    ]},
    "physio": {"title": "Physiological model", "then": "nothing", "fields": [
        setting("since_days", "int", "Fit on the last (days)",
                "only your rides of about this many days, e.g. 365 = the last year, 1825 = the last 5 years; the "
                "start moves back to the beginning of its quarter (1 Jan, 1 Apr, 1 Jul, 1 Oct) and on four times a "
                "year. A change means a full fit at the next update, and the ranges need a full run (Model page). "
                "0 = all your data", lo=30, hi=20000, zero=True),
        setting("ranges_every_days", "int", "Refresh the ranges every (days)",
                "on the home computer the app re-samples the model's 5-95 % ranges when they are this many days "
                "older than the data, or sooner when a best-fit number leaves its range (about 10 minutes in the "
                "background, low priority); 0 = only with goodpush physio --refresh", lo=1, hi=60, zero=True),
    ]},
    "garmin": {"title": "Garmin", "then": "nothing", "fields": [
        setting("auto_sync_hours", "float", "Check Garmin every (hours)", "0 = only when you press Update now",
                lo=0, hi=48, step=1),
    ]},
    "sync": {"title": "Several machines", "then": "nothing", "fields": [
        setting("home", "text", "Home machine", "the machine that downloads from Garmin / Strava (its name; empty = "
                "any). The others get the data from it (Settings > Other computers).", hi=64),
        setting("serve", "bool", "Let paired computers sync with this one", "this computer only: off = it doesn't "
                "listen for them (home network / Tailscale addresses only either way)"),
        setting("every_hours", "float", "Sync with the computer this one joined every (hours)", "this computer "
                "only; 0 = only when you press Sync (and after a ride)", lo=0, hi=48, step=0.5),
    ]},
    "ingest": {"title": "Importing", "then": "reanalyse", "fields": [
        setting("workers", "int", "Parallel readers", "0 = automatic (CPU count - 1)", lo=0, hi=64),
        setting("keep_streams", "bool", "Keep 1-s streams", "needed for the ride charts and interval detection"),
        setting("max_power", "int", "Max power (W)", "samples above are spikes and dropped", lo=500, hi=4000),
        setting("bad_power", "lines", "Power is wrong on", "a broken power meter: one a line - a ride id, a day "
                "(2017-05-03) or days (2017-06-01..2017-06-14); those rides count as rides without power. The Ride "
                "page's 'Power is wrong' adds a ride here"),
        setting("keep_power", "lines", "Power is right on", "ride ids whose best powers stay although heart rate "
                "says they're too high for it (the Ride page's 'They're real' adds a ride here)"),
    ]},
    "analysis": {"title": "Analysis", "then": "reanalyse", "fields": [
        setting("driver_window_weeks", "int", "Driver window (weeks)", "the training before a change in form",
                lo=2, hi=26),
        setting("progress_min_gain", "float", "Progress gain", "an FTP gain this big (0.04 = 4 %) marks progress",
                lo=.01, hi=.5, step=.01),
    ]},
    "paths": {"title": "Folders", "then": "reload", "fields": [
        setting("garmin", "path", "Garmin data"),
        setting("strava", "path", "Strava export", "the unzipped export (with activities.csv)"),
        setting("google_health", "path", "Fitbit (Google Health) data"),
        setting("rides", "path", "goodpush rides", "rides recorded with goodpush ride"),
        setting("processed", "path", "Processed data"),
        setting("reports", "path", "Reports"),
        setting("downloads", "path", "Downloads", "where the Garmin browser script saves; blank = your Downloads folder"),
    ]},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Project:
    root: Path
    cfg: dict = field(default_factory=dict)

    # convenience ---------------------------------------------------------
    def path(self, key: str) -> Path:
        p = Path(self.cfg["paths"][key])
        return p if p.is_absolute() else (self.root / p)

    @property
    def processed(self) -> Path:
        return self.path("processed")

    @property
    def reports(self) -> Path:
        return self.path("reports")

    @property
    def logs(self) -> Path:
        """this computer's run logs: logs/ in the athlete folder, which peer sync skips (sync.peer.PeerRules) - on
        the folder's drive, not the system drive (2026-11-13: logs had gone to the system drive)."""
        p = self.root / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def athlete(self) -> dict:
        return self.cfg["athlete"]

    @property
    def tz(self) -> str:
        return self.cfg["athlete"].get("timezone") or "UTC"


def find_root(start: Path | None = None) -> Path | None:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / CONFIG_NAME).exists():
            return cand
    return None


def default_root() -> Path:
    """Where the installed app keeps your project (goodpush.toml, your data, reports) when it isn't started
    from a project folder: GOODPUSH_HOME, else in the platform's place for app data (folders.Folders)."""
    from .folders import Folders
    return Folders.default_project()


def load_project(root: str | Path | None = None, athlete: str | None = None) -> Project:
    """Athlete `athlete`'s folder (or GOODPUSH_ATHLETE's), else the project at `root`, else the one this folder is
    in, else this machine's default athlete, else the installed app's (default_root). A registered athlete's
    credentials are used (athletes.Athletes)."""
    from .athletes import Athletes
    reg, home = Athletes(), default_root()
    athlete = athlete or os.environ.get("GOODPUSH_ATHLETE")
    if athlete:
        r = reg.folder(athlete).resolve()
    else:
        r = Path(root).resolve() if root else find_root() or (reg.folder(reg.default).resolve() if reg.default else None)             or (home if (home / CONFIG_NAME).exists() else None)
    if r is None:
        raise FileNotFoundError(
            f"No {CONFIG_NAME} found here or in parent folders. Run `goodpush init` first.")
    name = athlete or reg.name_of(r)
    if name:
        reg.use(name)
    cfg_file = r / CONFIG_NAME
    user = tomllib.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
    proj = Project(root=r, cfg=_merge(DEFAULTS, user))
    return proj


def template_text() -> str:
    return resources.files("goodpush").joinpath("templates/goodpush.toml").read_text(encoding="utf-8")


def init_project(root: str | Path) -> Project:
    r = Path(root).resolve()
    r.mkdir(parents=True, exist_ok=True)
    cfg_file = r / CONFIG_NAME
    if not cfg_file.exists():
        cfg_file.write_text(template_text(), encoding="utf-8")
    proj = load_project(r)
    for key in ("strava", "garmin", "processed", "reports"):
        proj.path(key).mkdir(parents=True, exist_ok=True)
    (r / "notes").mkdir(exist_ok=True)
    return proj


def clean_settings(section: str, values: dict) -> dict:
    """The values sent from the Settings page for one SETTINGS section, checked and in goodpush.toml's types
    (only the keys sent). Raises ValueError with a message for the page."""
    fields = {f["key"]: f for f in SETTINGS[section]["fields"]} if section in SETTINGS else None
    if fields is None:
        raise ValueError(f"no settings section {section!r}")
    return {k: clean_setting(fields[k], v) for k, v in values.items() if k in fields}


def clean_setting(f: dict, v):
    kind, label = f["kind"], f["label"]
    if kind in ("int", "float"):
        try:
            x = float(v or 0)
        except (TypeError, ValueError):
            raise ValueError(f"{label}: a number, please") from None
        if not (f.get("zero") and x == 0) and not f.get("lo", -1e18) <= x <= f.get("hi", 1e18):
            raise ValueError(f"{label} should be {'0 or ' if f.get('zero') else ''}{f.get('lo')}-{f.get('hi')}")
        return int(round(x)) if kind == "int" or x == int(x) and f.get("step", 1) >= 1 else round(x, 3)
    if kind == "bool":
        return bool(v)
    if kind in ("text", "path"):
        return str(v or "").strip()[:f.get("hi", 400)]
    if kind == "tz":
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo(str(v))
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"{label}: unknown time zone {v!r} (e.g. Europe/London)") from None
        return str(v)
    if kind in ("choice", "day"):
        allowed = [c[0] for c in f["choices"]] if kind == "choice" else DAYS
        if v not in allowed:
            raise ValueError(f"{label}: one of {', '.join(allowed)}")
        return v
    if kind == "days":
        days = [d for d in DAYS if d in (v or [])]
        if f.get("count") and len(days) != f["count"]:
            raise ValueError(f"{label}: pick exactly {f['count']}")
        return days
    if kind == "week":
        hours = [round(float(x or 0), 2) for x in (v or [])]
        if hours and (len(hours) != 7 or not all(0 <= x <= f["hi"] for x in hours)):
            raise ValueError(f"{label}: 7 values (Mon..Sun), 0-{f['hi']} hours each")
        return hours
    if kind == "numbers":
        return {k: clean_setting({"kind": "float", "label": f"{label} ({k})", "lo": f["lo"], "hi": f["hi"]},
                                 (v or {}).get(k, 0)) for k in f["keys"]}
    if kind == "lines":
        return setting_lines(v if isinstance(v, str) else "\n".join(map(str, v or [])))
    if kind == "history":
        return dict(sorted(parse_setting_line(line, 2, label, "date watts") for line in setting_lines(v)))
    if kind == "events":
        out = []
        for line in setting_lines(v):
            parts = [x.strip() for x in line.split(",", 3)]
            if len(parts) < 3 or parts[2] not in EVENT_TYPES:
                raise ValueError(f"{label}: '{line}' - start, end, type ({', '.join(EVENT_TYPES)}), note")
            start, end = (str(parse_date(x or parts[0], label)) for x in parts[:2])
            out.append({"date": start, "end": max(start, end), "type": parts[2],
                        "note": parts[3] if len(parts) > 3 else ""})
        return sorted(out, key=lambda e: e["date"])
    raise ValueError(f"{label}: unknown kind {kind}")


def setting_lines(text) -> list[str]:
    return [x.strip() for x in str(text or "").splitlines() if x.strip()]


def parse_date(s: str, label: str):
    from datetime import date
    try:
        return date.fromisoformat(s.strip())
    except ValueError:
        raise ValueError(f"{label}: {s!r} isn't a date (YYYY-MM-DD)") from None


def parse_setting_line(line: str, n: int, label: str, form: str) -> tuple[str, int]:
    """'2025-03-01 250' (or with = , :) -> ("2025-03-01", 250)."""
    parts = [x for x in line.replace("=", " ").replace(",", " ").replace(":", " ").split() if x]
    if len(parts) != n:
        raise ValueError(f"{label}: '{line}' - {form}")
    try:
        return str(parse_date(parts[0], label)), int(round(float(parts[1])))
    except ValueError as e:
        raise ValueError(f"{label}: '{line}' - {form}") from e
