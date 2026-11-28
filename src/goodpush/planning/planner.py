"""Builds the plan from an analysis result, writes its files and syncs Incyclist."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io.store import load_tables
from ..models import markers
from ..models.illness import find_notes
from ..models.intervals import IntervalLog
from .load import RampRange
from .readiness import readiness, sustained
from .scheduler import DEFAULT_PLAN, Scheduler
from .scoring import SessionScorer
from .season import Season
from .state import PlanState


PHYSIO_FORECAST = "physio_forecast.parquet"     # processed/: the plan's physio forecast (for the app)


class Planner:
    """One project's plan: `build()` scores what you did, re-plans, keeps what you accepted,
    proposes the rest and writes reports/plan/ (+ Incyclist when switched on)."""

    def __init__(self, proj):
        self.proj = proj
        self.cfg = {**DEFAULT_PLAN, **proj.cfg.get("plan", {})}
        self.state_path = proj.processed / "plan_state.json"

    CODE = ""                  # the planning code this process runs (code_on_disk() when it was imported)

    @staticmethod
    def code_on_disk() -> str:
        """the planning code as it is on disk now: its newest file time."""
        from .. import code_on_disk
        return code_on_disk(Path(__file__).parent)

    @classmethod
    def code_stale(cls) -> bool:
        """this process runs older planning code than is on disk: the app needs a restart."""
        return cls.CODE != cls.code_on_disk()

    @classmethod
    def code_changed(cls, plan: dict) -> bool:
        """the plan (plan.json) was made by other planning code than this process runs, and this process runs the
        code on disk: re-make it. An app still on older code leaves it alone - its plan would hide the change
        from the restarted app (it did: plans re-made by the old code looked up to date)."""
        return not cls.code_stale() and plan.get("code") != cls.CODE

    def ftp(self, daily: pd.DataFrame) -> tuple[float, str]:
        if float(self.cfg.get("ftp") or 0) > 0:
            return float(self.cfg["ftp"]), "goodpush.toml [plan] ftp"
        f = daily["ftp"].dropna()
        if len(f):
            mode = self.proj.cfg["ftp"].get("mode", "auto")
            return float(round(f.iloc[-1])), ("[ftp.history]" if mode == "manual" else "model estimate")
        return float(self.proj.cfg["ftp"].get("fallback", 200)), "fallback"

    def season(self, state: PlanState, today: pd.Timestamp, rides: bool = True) -> Season:
        """Your phases and races; the very first time (no plans.json yet) the plan you had becomes a
        12-week Build, or with no `rides` yet the general block (Season.GENERIC). Once you have deleted every
        phase, nothing is planned."""
        first = not Season.path(self.proj.root).exists()
        season = Season.load(self.proj.root)
        start = pd.Timestamp(state.get("start") or today - pd.Timedelta(days=today.weekday()))
        if first and season.ensure_phase(start, rides):
            season.save(self.proj.root)
        return season

    def readiness(self, daily: pd.DataFrame, acts: pd.DataFrame, state: PlanState, today: pd.Timestamp,
                  season=None) -> dict:
        """today's readiness; with the season, a recovery week (today's, or one that ended yesterday) is the test
        of its block: markers not back by then keep the recovery going (readiness.sustained)."""
        try:
            notes = find_notes(acts)
            ill = bool(len(notes) and (notes["date"] >= today - pd.Timedelta(days=10)).any())
        except Exception:  # noqa: BLE001 - no notes column etc.
            ill = False
        d = daily.loc[:today]
        learned, over = markers.load(self.proj.processed), markers.load(self.proj.processed, "overreaching")
        r = readiness(daily, today, ill, override=state.get("overrides", {}).get(str(today.date())),
                      markers=markers.status(d, learned) if learned else None,
                      markers_learned=bool(learned and not learned.get("fallback")),
                      overreach=markers.status(d, over) if over and over.get("chosen") else None)
        week = season.day_table(pd.DatetimeIndex([today - pd.Timedelta(days=1), today]))["recovery"] if season else None
        return sustained(daily, today, r, recovery_week=bool(week is not None and week.any()))

    @staticmethod
    def in_your_watts(out: dict, generic: list, sched, daily: pd.DataFrame, today: pd.Timestamp) -> None:
        """Week totals and the CTL forecast in the TSS of the sessions as you'll ride them (your watts).
        The weeks were sized in textbook %FTP; the same riding at your watts counts more (or less) TSS,
        so each week's remaining sessions and its target scale together and the CTL forecast follows."""
        mapped = out["sessions"]
        for r in out["weeks"]:
            start = pd.Timestamp(r["week"])
            inside = [i for i, s in enumerate(generic) if start <= s.date < start + pd.Timedelta(days=7)]
            gen = sum(generic[i].tss() for i in inside)
            if gen <= 0:
                continue
            ratio = sum(mapped[i].tss() for i in inside) / gen
            done = r["tss_planned"] - gen                              # rides already in this week
            r["tss_planned"] = round(done + gen * ratio)
            r["tss_target"] = round(done + (r["tss_target"] - done) * ratio)
        past = daily.loc[:today - pd.Timedelta(days=1)]
        if len(past):
            proj = sched.projection(mapped, float(past["ctl"].iloc[-1]), float(past["atl"].iloc[-1]))
            out["projection"] = proj
            if proj:
                out["ramp_per_week"] = (proj[-1][1] - out["ctl_now"]) / max(len(proj) / 7, 1)
            ends =pd.Series([p[1] for p in proj], pd.DatetimeIndex([p[0] for p in proj]))
            for r in out["weeks"]:
                e = ends.loc[:pd.Timestamp(r["week"]) + pd.Timedelta(days=6)]
                if len(e):
                    r["ctl_end"] = round(float(e.iloc[-1]), 1)

    def physio(self, ftp: float, progress=print):
        """PhysioLoad when [plan] load_model = "physio" or intensity = "model" (refitting the model if the data
        changed), with `mix_draws` of the stored posterior draws for the session choice; else None."""
        cfg = self.cfg
        if cfg["load_model"] != "physio" and cfg["intensity"] != "model":
            return None
        if self.stale_analysis():
            progress("physio model: the analysis isn't made from this model's own daily states yet - CTL rules / "
                     "pyramidal hard days until it is")
            return None
        try:
            from ..models.physio import PhysioRun
            from .physio_load import PhysioLoad
            from .workouts import Z2
            run = PhysioRun(self.proj)
            model, z = run.current(progress)
            draws = None
            n = int(cfg["mix_draws"] or 0) if cfg["intensity"] == "model" else 0
            if n > 0:
                draws, _ = run.kept(model, run.stored(), model.data)
                draws = draws[run.spaced(len(draws), min(len(draws), n))] if len(draws) else None
                if draws is None:
                    progress("physio model: no posterior draws yet, the session mix from the best fit alone "
                             f"({run.FULL_RUN})")
            return PhysioLoad(model, z, ftp, easy_ftp=sum(Z2) / 2, draws=draws)
        except Exception as e:  # noqa: BLE001 - e.g. jax missing: the CTL rules still make a plan
            progress(f"physio model unavailable, CTL rules / pyramidal hard days used: {e}")
            return None

    def stale_analysis(self) -> bool:
        """the plan uses the physio model and the analysis tables weren't made from its own daily states (a new
        model version, or no fit yet: models.physio.PhysioRun.reset) - the analysis is due first."""
        if self.cfg["load_model"] != "physio" and self.cfg["intensity"] != "model":
            return False
        try:
            from ..models.physio import PhysioRun
            return not PhysioRun(self.proj).analysed()
        except ImportError:                  # no physio extra: no model
            return False

    @staticmethod
    def log_mix(info: dict | None, progress=print) -> None:
        """one line on the key days' sessions the model chose (intensity "model")."""
        if not info:
            return
        if "error" in info:
            progress(f"session mix from the model failed, pyramidal used: {info['error']}")
            return
        by = "the best fit" if not info["draws"] else f"{info['draws']} posterior draws"
        what = ", ".join(f"{k} {v}" for k, v in info["sessions"].items()) or "no key days to choose"
        progress(f"key-day sessions from the model ({by}): {what}; "
                 f"forecast {100 * info['gain']:+.2f} % (summed over the target days) vs the pyramidal start")

    def no_rides(self, today: pd.Timestamp) -> dict:
        """the analysis tables before your first ride: yesterday, nothing measured (FTP from [ftp] fallback or
        [plan] ftp), no activities; load (CTL = ATL) as if you had ridden `plan_starting_hours` a week at
        [load] default_tss_per_hour (0 hours: from nothing)."""
        day = pd.DatetimeIndex([today - pd.Timedelta(days=1)], name="date")
        tph = float(((self.proj.cfg.get("load") or {}).get("default_tss_per_hour") or {}).get("ride", 45))
        load = float(self.cfg.get("plan_starting_hours") or 0) * tph / 7
        zero = {**dict.fromkeys(("tss", "hours", "ride_hours", "tsb"), 0.0), "ctl": load, "atl": load}
        unknown = dict.fromkeys(("ftp", "rhr", "hrv", "sleep_h"), float("nan"))
        acts = pd.DataFrame({"id": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[us]"),
                             "sport": pd.Series(dtype=str), "name": pd.Series(dtype=str),
                             "tss": pd.Series(dtype=float)})
        return {"daily": pd.DataFrame({**zero, **unknown}, index=day), "acts": acts}

    def build(self, tables: dict | None = None, today: pd.Timestamp | None = None, write: bool = True,
              progress=print) -> dict:
        """The plan from today. `tables`: an analysis result (daily, acts), default the stored one.
        With `today` given (plan as if...) nothing is pushed to Incyclist. No analysis yet (no rides): planned
        from nothing (`no_rides`)."""
        asked = today is not None
        today = today if asked else pd.Timestamp.now(tz=self.proj.tz).tz_localize(None).normalize()
        if not tables:
            try:
                tables = load_tables(self.proj.processed)
            except FileNotFoundError:
                tables = self.no_rides(today)
            else:
                if write and self.stale_analysis():   # e.g. the first plan after a model update: analysed again
                    from ..analyze import run_analysis
                    tables = run_analysis(self.proj, progress)
        daily, acts = tables["daily"], tables["acts"]
        ftp, src = self.ftp(daily)
        with PlanState.LOCK:                 # the app may tick a session off meanwhile
            state = PlanState.load(self.state_path)
            try:                             # how the last weeks' sessions went
                SessionScorer.update(state, acts, self.proj.processed, today)
            except Exception as e:  # noqa: BLE001 - scoring is a bonus; the plan works without it
                progress(f"session scoring skipped: {e}")
            if not state.get("levels_seeded"):   # first plan: start the interval levels where your rides are
                try:
                    seeded = SessionScorer.seed_levels(state, acts, self.proj.processed, today, ftp)
                    if seeded:
                        progress("interval levels from your rides: " + ", ".join(f"{k} {v + 1}" for k, v in seeded.items()))
                except Exception as e:  # noqa: BLE001
                    progress(f"interval levels from rides skipped: {e}")
            try:                             # levels fade without the work
                faded = state.decay_levels(IntervalLog.load(self.proj.processed), today)
                if faded:
                    progress("interval levels faded without the work: " + ", ".join(
                        f"{k} -{n}" + (f" (steady -{s})" if s else "") for k, (n, s) in faded.items()))
            except Exception as e:  # noqa: BLE001 - a bonus like the scoring
                progress(f"level decay skipped: {e}")
            season = self.season(state, today, rides=bool(len(acts)))
            sched = Scheduler(self.cfg, season, physio=self.physio(ftp, progress) if len(acts) else None)
            out = sched.plan(state, daily, acts, ftp, today, self.readiness(daily, acts, state, today, season))
            physio_load = sched.physio is not None and self.cfg["load_model"] == "physio"
            out["load_model"] = "physio" if physio_load and not sched.physio_error else "ctl"
            self.log_mix(out.get("session_mix"), progress)
            if sched.physio_error:
                progress(f"physio load targets failed, CTL rules used: {sched.physio_error}")
            elif physio_load and write:
                fc = sched.physio.forecast_frame()
                if fc is not None:
                    fc.assign(fatigue_cap=fc.attrs["fatigue_cap"]).to_parquet(self.proj.processed / PHYSIO_FORECAST)
            imap = sched.physio.intensity_map() if physio_load and not sched.physio_error else None
            if imap is not None:                     # targets in your watts (the plan state keeps textbook %FTP)
                generic = out["sessions"]
                out["sessions"] = [imap.apply(s) for s in generic]
                out["intensity"] = imap.today()
                self.in_your_watts(out, generic, sched, daily, today)
                if write:
                    imap.save(self.proj.processed)
            sug = SessionScorer.ftp_suggestion(state, ftp, today) if not float(self.cfg.get("ftp") or 0) else None
            if sug and state.get("rejected", {}).get("ftp") != sug["to"]:
                out["proposal"] = out["proposal"] + [{"kind": "ftp", **sug}]
                state["proposal"] = {"created": str(today.date()), "changes": out["proposal"]}
            if write:
                state.save()
        ramps = RampRange(self.buildup())
        ramps.flag(out.get("weeks") or [], daily)
        out["ramp_range"] = ramps.to_dict()
        out.update(ftp_source=src, planned=state["planned"], today=today, cfg=self.cfg, code=self.CODE,
                   override=state.get("overrides", {}).get(str(today.date()), ""),
                   season={"phases": [vars(p) for p in season.phases], "events": [vars(e) for e in season.events],
                           **self.suggestion(season, state, today)})
        if write:
            from ..report import plan_out
            d = plan_out.write_all(self.proj.reports / "plan", out, self.proj.cfg["athlete"].get("name") or "")
            progress(f"plan written to {d} (readiness today: {out['readiness']['status']})")
            if not asked:
                self.sync_incyclist(out, progress)
        return out

    def buildup(self) -> dict | None:
        """the analysis' build-up results (the ramp table), if there is one."""
        import json
        p = self.proj.reports / "results.json"
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("buildup") if p.exists() else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def suggestion(season: Season, state: dict, today: pd.Timestamp) -> dict:
        """the phases `season.suggest`s (none when you said "not now" to these) and their signature."""
        sugg = season.suggest(today)
        sig = Planner.signature(sugg)
        return {"suggestion": [] if state.get("season_dismissed") == sig else sugg, "suggestion_sig": sig}

    @staticmethod
    def signature(suggestion: list[dict]) -> str:
        """Which suggestion you said "not now" to (a new race or phase makes a new one)."""
        return ",".join(f"{s['type']}:{s['start']}:{s['weeks']}" for s in suggestion)

    def sync_incyclist(self, out: dict, progress=print) -> dict | None:
        """Plan -> Incyclist's workout calendar when [incyclist] sync = true (problems only reported)."""
        if not self.proj.cfg.get("incyclist", {}).get("sync"):
            return None
        from ..sync import incyclist
        try:
            return incyclist.sync(out, self.proj.cfg, self.proj.tz, progress=progress)
        except Exception as e:  # noqa: BLE001
            progress(f"Incyclist: plan not written - {e}")
            return None


Planner.CODE = Planner.code_on_disk()
