"""Your season: training phases and target events. Stored in data/plans.json (your input, not a cache).

Phase types follow Friel (base -> build -> peak/race -> transition) and TrainerRoad (base -> build
-> specialty), plus maintenance and recovery / downtime. Phases never overlap; gaps are allowed
and nothing is planned in them. The Calendar edits the season as blocks back to back (`set_blocks`).
`recovery_weeks` are a phase's recovery weeks (week numbers from 0), wherever you put them; a new phase
gets them from `build_weeks`, the cadence: that many loading weeks, then one recovery week (2 = 2:1,
3 = 3:1; 0 = none) - and so do the weeks a longer phase gains. A new phase counts its loading weeks around your
races (`Season.raced`): a week with an A or B race, or an A race's taper week before it, eases off already, so the
count starts again after it, and no recovery week goes right before one.

Events are races / target days with an A/B/C priority: A gets a ~2-week taper (volume -41..60 %,
intensity kept; Bosquet et al. 2007), B a short one, C is trained through.

`Season.suggest` proposes phases for the weeks your A races need that none of your phases cover,
working back from each A race, from next Monday (Friel's periodisation): Specialty for the last SPECIALTY weeks up to
the race (from your first B race before it, if earlier; at most SPECIALTY_MAX), Build before that
(at most BUILD_MAX), Base before that; RECOVERY_AFTER weeks of recovery after the last A race and one
week between A races. A suggested Specialty keeps its recovery weeks between the races.
Your own phases always win: suggestions only fill the gaps. With no A race ahead and no phase from next Monday on
(a new rider, or a season that has ended) it suggests GENERIC instead: a general Base -> Build block, so you can
train without a race in mind (`autoplan` lays it out too, and so does a new rider's first plan).
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

PHASE_TYPES = {
    "base": ("Base", 3, "Volume and durability: endurance with sweet spot and tempo, no VO2max."),
    "build": ("Build", 3, "Threshold every week with VO2max / sweet spot; load (CTL) climbs at your ramp."),
    "specialty": ("Specialty", 2, "Race-specific: VO2max and threshold; tapers into your events."),
    "maintenance": ("Maintenance", 0, "Hold fitness: one quality session a week, the rest easy."),
    "recovery": ("Recovery / downtime", 0, "Easy riding only, about half your hours."),
}
EVENT_KINDS = ["road", "crit", "tt", "mtb", "gravel", "cx", "hillclimb", "fondo", "other"]
PRIORITIES = ("A", "B", "C")
NS_WEEK = 7 * 86_400 * 10**9
SPECIALTY, SPECIALTY_MIN, SPECIALTY_MAX, BUILD_MAX, RECOVERY_AFTER = 8, 4, 12, 8, 2   # weeks (Friel: build 6-8)
WHY = {"base": "volume and durability before the hard work",
       "build": "raises fitness (threshold, VO2max) on the way to your race phase",
       "specialty": "race-specific work through your races, peaking for {race}",
       "recovery": "easy weeks to absorb the season after {race}"}
GENERIC = (("base", 8, "endurance and durability first (no race to plan back from)"),   # 2 x (3 loading + 1 easy)
           ("build", 4, "then threshold and VO2max on that base"))                     # 3 loading + 1 easy


def new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Phase:
    type: str
    start: str
    weeks: int
    name: str = ""
    build_weeks: int | None = None
    hours: float | None = None          # weekly hours for this phase (None = your usual)
    id: str = field(default_factory=new_id)
    recovery_weeks: list | None = None  # its recovery weeks (0 = the first week); None = from the cadence

    def __post_init__(self):
        if self.type not in PHASE_TYPES:
            raise ValueError(f"phase type must be one of {', '.join(PHASE_TYPES)}")
        self.start = str(pd.Timestamp(self.start).date())
        self.weeks = int(self.weeks)
        if not 1 <= self.weeks <= 52:
            raise ValueError("a phase is 1-52 weeks long")
        label, bw, _ = PHASE_TYPES[self.type]
        self.build_weeks = bw if self.build_weeks in (None, "") else int(self.build_weeks)
        if not 0 <= self.build_weeks <= 6:
            raise ValueError("loading weeks before a recovery week: 0 (none) to 6")
        self.hours = None if self.hours in (None, "", 0) else float(self.hours)
        self.name = str(self.name or label)
        recs = self.cadence(self.build_weeks, 0, self.weeks) if self.recovery_weeks is None else self.recovery_weeks
        # a recovery phase is all recovery: no weeks marked in it
        inside = {int(k) for k in recs if 0 <= int(k) < self.weeks}
        self.recovery_weeks = [] if self.type == "recovery" else sorted(inside)

    @staticmethod
    def cadence(build_weeks: int, first: int, weeks: int) -> list[int]:
        """recovery weeks by the cadence from week `first` (a block starting there) up to `weeks`."""
        return list(range(first + build_weeks, weeks, build_weeks + 1)) if build_weeks else []

    def grown(self, weeks: int) -> list[int]:
        """the recovery weeks with the phase `weeks` long: the cadence goes on into the weeks it gains."""
        recs = [k for k in self.recovery_weeks if k < weeks]
        if weeks <= self.weeks or not self.build_weeks or self.type == "recovery":
            return recs
        block = recs[-1] + 1 if recs else 0                  # the block the old last week is in started here
        nxt = max(block + self.build_weeks, self.weeks)      # its recovery week, at the earliest in the new weeks
        return recs + list(range(nxt, weeks, self.build_weeks + 1))

    @property
    def end(self) -> pd.Timestamp:
        """Last day."""
        return pd.Timestamp(self.start) + pd.Timedelta(weeks=self.weeks, days=-1)

    def overlaps(self, other: Phase) -> bool:
        return pd.Timestamp(self.start) <= other.end and pd.Timestamp(other.start) <= self.end

    def touches(self, other: Phase) -> bool:
        """right next to `other`: one starts the day after the other ends."""
        day = pd.Timedelta(days=1)
        return self.end + day == pd.Timestamp(other.start) or other.end + day == pd.Timestamp(self.start)

    def same_kind(self, other: Phase) -> bool:
        """same type, recovery-week cadence and hours: one phase could stand for both."""
        return (self.type, self.build_weeks, self.hours) == (other.type, other.build_weeks, other.hours)

    def weeks_table(self) -> dict:
        """per week of the phase: recovery, block (recovery weeks before it), week in its block, the block's loading
        weeks (0 in a phase without recovery weeks: no block structure), push (a loading week before a recovery
        week)."""
        w = np.arange(self.weeks)
        rec = np.isin(w, self.recovery_weeks)
        after = np.r_[False, rec[:-1]]                           # the week after a recovery week starts a block
        block = np.cumsum(after)
        start = np.maximum.accumulate(np.where(after, w, 0))
        loading = np.bincount(block, ~rec)[block] if rec.any() else np.zeros(self.weeks, int)
        return {"recovery": rec, "block": block, "week_in_block": w - start, "block_weeks": loading,
                "push": ~rec & np.r_[rec[1:], False]}


@dataclass
class Event:
    date: str
    priority: str = "B"
    kind: str = "road"
    name: str = ""
    series: str = ""
    id: str = field(default_factory=new_id)

    def __post_init__(self):
        self.date = str(pd.Timestamp(str(self.date).strip()).date())
        self.priority = str(self.priority or "B").strip().upper()
        if self.priority not in PRIORITIES:
            raise ValueError("priority is A, B or C")
        self.kind = str(self.kind or "road").strip().lower()
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"race type is one of {', '.join(EVENT_KINDS)}")
        self.name = str(self.name or self.kind.title()).strip()
        self.series = str(self.series or "").strip()


class Season:
    """Your phases and events, with the lookups the plan needs."""

    def __init__(self, phases=(), events=()):
        self.phases: list[Phase] = sorted(phases, key=lambda p: p.start)
        self.events: list[Event] = sorted(events, key=lambda e: e.date)

    # --- storage ---------------------------------------------------------------------------
    @staticmethod
    def path(root) -> Path:
        return Path(root) / "data" / "plans.json"

    @classmethod
    def load(cls, root) -> Season:
        try:
            d = json.loads(cls.path(root).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        return cls([Phase(**p) for p in d.get("phases", [])], [Event(**e) for e in d.get("events", [])])

    def save(self, root) -> None:
        p = self.path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"phases": [asdict(x) for x in self.phases],
                                   "events": [asdict(x) for x in self.events]}, indent=1), encoding="utf-8")
        tmp.replace(p)

    # --- editing ---------------------------------------------------------------------------
    def add_phase(self, **kw) -> Phase:
        p = Phase(**kw)
        p = self.raced(p) if kw.get("recovery_weeks") is None else p
        self.place_phase(p, merge=False)
        return p

    def edit_phase(self, pid: str, start, weeks, merge: bool | None = None) -> str:
        """new dates for a phase (into free weeks only); see place_phase for `merge` and the return."""
        old = next((p for p in self.phases if p.id == pid), None)
        if old is None:
            raise ValueError("that phase is gone (reload the page)")
        return self.place_phase(replace(old, start=start, weeks=weeks), merge, replacing=pid)

    def place_phase(self, p: Phase, merge: bool | None = None, replacing: str = "") -> str:
        """put p in the season (in place of phase `replacing`), never over another phase. Right next to
        phases of the same kind: merge None -> nothing changes, the question to ask is returned; True ->
        one phase over them all (the existing one's name and id); False -> kept apart. Returns "" once placed."""
        others = [o for o in self.phases if o.id != replacing]
        clash = next((o for o in others if o.overlaps(p)), None)
        if clash:
            raise ValueError(f"overlaps {clash.name} ({clash.start} - {clash.end.date()})")
        same = [o for o in others if o.same_kind(p) and o.touches(p)]
        if same and merge is None:
            weeks = p.weeks + sum(o.weeks for o in same)
            what = " and ".join(f"{o.name} ({o.start} - {o.end.date()})" for o in same)
            return (f"This is right next to {what}, with the same settings. Merged into one {weeks}-week phase, "
                    "its recovery weeks run through all of it; kept apart, each phase has its own.")
        if same and merge:
            p = replace(same[0], start=min(o.start for o in [p, *same]), weeks=p.weeks + sum(o.weeks for o in same))
            others = [o for o in others if o not in same]
        self.phases = sorted(others + [p], key=lambda x: x.start)
        return ""

    def set_blocks(self, start, blocks: list[dict]) -> None:
        """the season as blocks back to back from `start` (moved to its Monday): each {type, weeks, and optionally
        id, name, hours, build_weeks, recovery_weeks}; replaces the phases (the Calendar's season editor). A block
        with a phase's id keeps its settings; grown, the cadence fills its new weeks with recovery weeks
        (Phase.grown); a new block gets them from the cadence; a changed type takes that type's name."""
        old = {p.id: p for p in self.phases}
        at = pd.Timestamp(start).normalize()
        at -= pd.Timedelta(days=at.weekday())
        out = []
        for b in blocks:
            prev = old.get(str(b.get("id") or ""))
            kind, weeks = str(b.get("type")), int(b.get("weeks") or 0)
            same = prev is not None and prev.type == kind
            recs = b.get("recovery_weeks")
            if same and weeks > prev.weeks:
                recs = replace(prev, recovery_weeks=prev.recovery_weeks if recs is None else recs).grown(weeks)
            p = Phase(type=kind, start=at, weeks=weeks, recovery_weeks=recs,
                      name=b.get("name") or (prev.name if same else ""),
                      build_weeks=b.get("build_weeks", prev.build_weeks if same else None),
                      hours=b.get("hours", prev.hours if same else None),
                      **({"id": prev.id} if prev is not None else {}))
            out.append(self.raced(p) if recs is None else p)
            at += pd.Timedelta(weeks=weeks)
        self.phases = out

    def autoplan(self, today: pd.Timestamp) -> None:
        """the whole season back from your A races (`suggest` on no phases), from next Monday; replaces the phases.
        No A race ahead: the GENERIC block."""
        plan = Season(events=self.events).suggest(today)
        self.phases = [self.raced(Phase(type=s["type"], start=s["start"], weeks=s["weeks"],
                                        build_weeks=s["build_weeks"]))
                       for s in plan]

    def race_weeks(self, start, weeks: int) -> np.ndarray:
        """per week from `start`: eased by a race - an A or B race in it, or the taper week before an A race."""
        e = [((pd.Timestamp(x.date) - pd.Timestamp(start)).days // 7, x.priority) for x in self.events
             if x.priority in ("A", "B")]
        k = np.array([w for w, _ in e] + [w - 1 for w, pr in e if pr == "A"], int)
        out = np.zeros(weeks, bool)
        out[k[(k >= 0) & (k < weeks)]] = True
        return out

    def raced(self, p: Phase) -> Phase:
        """p with its recovery weeks by its cadence counted around your races (see the module doc)."""
        if not p.build_weeks or p.type == "recovery":
            return p
        eased, recs, run = self.race_weeks(p.start, p.weeks + 1), [], 0
        for k in range(p.weeks):                      # a phase's weeks (<= 52): the count carries week to week
            if eased[k]:
                run = 0
            elif run >= p.build_weeks and not eased[k + 1]:
                recs, run = recs + [k], 0
            else:
                run += 1
        return replace(p, recovery_weeks=recs)

    def delete_phase(self, pid: str) -> None:
        self.phases = [p for p in self.phases if p.id != pid]

    def add_event(self, **kw) -> Event:
        e = Event(**kw)
        self.events = sorted(self.events + [e], key=lambda x: x.date)
        return e

    def add_events_text(self, text: str, series: str = "") -> list[Event]:
        new = self.parse_events(text, series)
        self.events = sorted(self.events + new, key=lambda x: x.date)
        return new

    def edit_event(self, eid: str, **kw) -> Event:
        """race `eid` with new values (checked like a new one); it keeps its id, so it stays the same race."""
        old = next((e for e in self.events if e.id == eid), None)
        if old is None:
            raise KeyError(f"no race {eid}")
        e = Event(**{**asdict(old), **{k: v for k, v in kw.items() if v is not None}, "id": eid})
        self.events = sorted([x for x in self.events if x.id != eid] + [e], key=lambda x: x.date)
        return e

    def delete_event(self, eid: str) -> None:
        self.events = [e for e in self.events if e.id != eid]

    @staticmethod
    def parse_events(text: str, series: str = "") -> list[Event]:
        """Several races, one per line: `2026-11-08, B, crit, Race name` (priority, type and name
        optional; '#' starts a comment). Errors name the line."""
        out = []
        for n, line in enumerate(str(text).splitlines(), 1):
            parts = [x.strip() for x in re.split(r"[,;\t]", line.split("#")[0].strip())]
            if not parts[0]:
                continue
            try:
                out.append(Event(parts[0], *(parts[1:3]), name=", ".join(parts[3:]), series=series))
            except (ValueError, TypeError) as err:
                raise ValueError(f"line {n} ({line.strip()!r}): {err}") from None
        if not out:
            raise ValueError("no races in the text")
        return out

    def ensure_phase(self, start: pd.Timestamp, rides: bool = True) -> bool:
        """No phases yet: what you had becomes a 12-week Build from `start` (3:1); with no rides at all (a new
        rider) the GENERIC block from `start` instead."""
        if self.phases:
            return False
        if not rides:
            for s in self.generic(pd.Timestamp(start)):
                self.add_phase(type=s["type"], start=s["start"], weeks=s["weeks"])
            return True
        self.add_phase(type="build", start=start, weeks=12, build_weeks=3)
        return True

    @staticmethod
    def generic(monday: pd.Timestamp) -> list[dict]:
        """the GENERIC block from `monday`, as suggestions (marked `generic`)."""
        out, at = [], pd.Timestamp(monday)
        for kind, weeks, why in GENERIC:
            out.append({"type": kind, "start": str(at.date()), "weeks": weeks, "build_weeks": None, "why": why,
                        "generic": True})
            at += pd.Timedelta(weeks=weeks)
        return out

    def suggest(self, today: pd.Timestamp) -> list[dict]:
        """Phases for the weeks your A races need that none of your phases cover (see the module doc):
        [{type, start, weeks, why}], empty when nothing is missing."""
        today = pd.Timestamp(today).normalize()
        monday = today + pd.Timedelta(days=(7 - today.weekday()) % 7)      # from next Monday (today if Monday)
        week = lambda d: pd.Timestamp(d) - pd.Timedelta(days=pd.Timestamp(d).weekday())  # noqa: E731
        wk = pd.Timedelta(weeks=1)
        races = [e for e in self.events if e.priority == "A" and week(e.date) >= monday]
        if not races and (self.end is None or self.end < monday):      # nothing to plan for: a general block
            return self.generic(monday)
        want, seg = {}, monday                   # week -> (type, A race); seg = first week still free
        for i, a in enumerate(races):
            aw = week(a.date)
            first_b = min((week(e.date) for e in self.events if e.priority == "B" and seg <= week(e.date) <= aw), default=aw)
            spec = max(seg, aw - SPECIALTY_MAX * wk + wk, min(aw - (SPECIALTY - 1) * wk, first_b))
            spec = max(seg, min(spec, aw - (SPECIALTY_MIN - 1) * wk))
            build = max(seg, spec - BUILD_MAX * wk)
            for w in pd.date_range(seg, aw, freq="7D"):
                want[w] = ("specialty" if w >= spec else "build" if w >= build else "base", a)
            rest = RECOVERY_AFTER if i == len(races) - 1 else 1
            for w in pd.date_range(aw + wk, aw + rest * wk, freq="7D"):
                want[w] = ("recovery", a)
            seg = aw + (rest + 1) * wk
        covered = lambda w: any(pd.Timestamp(p.start) <= w <= p.end for p in self.phases)  # noqa: E731
        out = []
        for w in sorted(want):                   # merge free weeks of one type into phases
            kind, a = want[w]
            if covered(w):
                continue
            last = out[-1] if out else None
            if last and last["type"] == kind and pd.Timestamp(last["start"]) + last["weeks"] * wk == w:
                last["weeks"] += 1
            else:
                out.append({"type": kind, "start": str(w.date()), "weeks": 1,
                            "build_weeks": None,
                            "why": WHY[kind].format(race=f"{a.name} ({pd.Timestamp(a.date):%d %b})")})
        return out

    # --- lookups ---------------------------------------------------------------------------
    @property
    def end(self) -> pd.Timestamp | None:
        return max((p.end for p in self.phases), default=None)

    def events_on(self) -> dict[str, dict]:
        return {e.date: asdict(e) for e in self.events}

    def day_table(self, days: pd.DatetimeIndex) -> pd.DataFrame:
        """Per day: phase index (-1 outside every phase), type, phase_name, week in the phase, block,
        week in the block, the block's loading weeks, recovery week, push week, the phase's hours."""
        out = pd.DataFrame(index=days, data={"phase": -1, "type": "", "phase_name": "", "week": 0, "block": 0,
                                             "week_in_block": 0, "block_weeks": 0, "recovery": False, "push": False,
                                             "hours": np.nan})
        if not self.phases or not len(days):
            return out
        st = np.array([pd.Timestamp(p.start).value for p in self.phases])
        en = np.array([p.end.value for p in self.phases])
        v = days.values.astype("datetime64[ns]").astype(np.int64)
        k = np.searchsorted(st, v, side="right") - 1
        kk = np.maximum(k, 0)
        inside = (k >= 0) & (v <= en[kk])
        wk = (v - st[kk]) // NS_WEEK
        tables = [p.weeks_table() for p in self.phases]                  # a few phases, a few weeks each
        first = np.r_[0, np.cumsum([p.weeks for p in self.phases])[:-1]]
        at = first[kk] + np.clip(wk, 0, np.array([p.weeks for p in self.phases])[kk] - 1)
        col = {c: np.concatenate([t[c] for t in tables])[at] for c in tables[0]}
        out["phase"] = np.where(inside, k, -1)
        out["type"] = np.where(inside, np.array([p.type for p in self.phases])[kk], "")
        out["phase_name"] = np.where(inside, np.array([p.name for p in self.phases])[kk], "")
        out["week"] = wk
        out["block"] = np.where(inside, col["block"], 0)
        out["week_in_block"] = np.where(inside, col["week_in_block"], wk)
        out["block_weeks"] = np.where(inside, col["block_weeks"], 0)
        out["recovery"] = inside & col["recovery"]
        out["push"] = inside & col["push"]
        out["hours"] = np.array([p.hours or np.nan for p in self.phases], float)[kk]
        return out

    def changed_days(self, old: Season, today: pd.Timestamp) -> list[str]:
        """The days from `today` on whose phase type, recovery week or push week differ from `old` (what a season
        edit changed)."""
        ends = [e for e in (self.end, old.end) if e is not None]
        if not ends:
            return []
        days = pd.date_range(pd.Timestamp(today).normalize(), max(ends), freq="D")
        a, b = self.day_table(days), old.day_table(days)
        diff = np.zeros(len(days), bool)
        for c in ("type", "recovery", "push"):
            diff |= a[c].to_numpy() != b[c].to_numpy()
        return [str(d.date()) for d in days[diff]]
