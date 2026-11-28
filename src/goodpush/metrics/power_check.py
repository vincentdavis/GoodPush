"""Rides whose power is wrong, left out before anything reads their watts (2026-11-26).

A broken power meter can read minutes at 1000 W with heart rate at an easy 132: as a ride best it lifted a synthetic
rider's fitted CP 86 % and MAP 119 % over the whole history (the physio model now has a bad-reading share too,
physio BAD_READING_SHARE, but auto-FTP, load and the plan read the same bests). Two ways a ride's power is set aside:

* marked by you: goodpush.toml [ingest] bad_power lists ride ids, days ("2017-05-03") or date ranges
  ("2017-06-01..2017-06-14", a meter reading high for weeks). The ride counts as one without power: time and heart
  rate stay, its load comes from heart rate, no best powers, no FTP, no watts at heart rate from it;
* by heart rate: a best of SCREEN_S or longer OVER x the TOP_PCT percentile of your rides' bests at that length,
  while heart rate around its end stayed under HR_SHARE x threshold heart rate - not a rider's effort. Its best powers are
  dropped (its time in each power zone, and so its load, stay). [ingest] keep_power lists rides to leave alone.

Each ride's reason is in activities' `power_ignored`: "" (used), "marked" or "heart rate"."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .effort_hr import EffortHeartRate
from .power import MMP_DURATIONS


class PowerCheck:
    MARKED, HEART_RATE = "marked", "heart rate"
    SCREEN_S = (180, 300, 600, 1200)       # best-effort lengths the heart-rate screen reads (EffortHeartRate's)
    TOP_PCT = 99.0                         # ... OVER x this percentile of your rides' bests at that length or more
    OVER = 1.2
    HR_SHARE = 0.95                        # ... with heart rate around its end under this x threshold heart rate
    MIN_RIDES = 20                         # ... with at least this many rides to set the percentile
    RANGE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:\.\.(\d{4}-\d{2}-\d{2}))?$")
    # a ride without power: these are blanked (as metrics.power.summarize_stream leaves a ride without a power stream)
    POWER_COLUMNS = ("np", "np_used", "avg_power", "work_kj", "max_power", "ef", "decoupling")

    def __init__(self, bad: list | None = None, keep: list | None = None):
        bad = [str(x).strip() for x in bad or []]
        self.ranges = [(pd.Timestamp(m[1]), pd.Timestamp(m[2] or m[1])) for m in map(self.RANGE.match, bad) if m]
        self.bad_ids = {x for x in bad if not self.RANGE.match(x)}
        self.keep_ids = {str(x).strip() for x in keep or []}

    @classmethod
    def of(cls, proj) -> PowerCheck:
        ing = proj.cfg.get("ingest", {})
        return cls(ing.get("bad_power"), ing.get("keep_power"))

    @staticmethod
    def ids(acts: pd.DataFrame) -> pd.Series:
        return acts["id"].astype(str).str.replace(r"\.0$", "", regex=True)

    def marked(self, acts: pd.DataFrame) -> np.ndarray:
        """rides you marked: by id, or on a listed day / in a listed date range."""
        hit = self.ids(acts).isin(self.bad_ids).to_numpy().copy()
        day = pd.to_datetime(acts["date"]).dt.normalize()
        for a, b in self.ranges:                              # a few entries
            hit |= ((day >= a) & (day <= b)).to_numpy()
        return hit

    def screened(self, acts: pd.DataFrame, lthr: float) -> np.ndarray:
        """rides with a best of SCREEN_S+ far above your usual (TOP_PCT) at a heart rate under threshold."""
        out = np.zeros(len(acts), bool)
        has = acts["has_power"].fillna(False).to_numpy(bool) if "has_power" in acts else np.ones(len(acts), bool)
        if not lthr:
            return out
        for d in self.SCREEN_S:
            w, hr = (acts.get(f"effort_{k}_{d}") for k in ("w", "hr"))
            if w is None or hr is None:
                continue
            w, hr = w.to_numpy(float), hr.to_numpy(float)
            seen = has & np.isfinite(w)
            if seen.sum() < self.MIN_RIDES:
                continue
            top = np.percentile(w[seen], self.TOP_PCT, method="lower")      # one glitch never sets its own bar
            with np.errstate(invalid="ignore"):
                out |= seen & (w > self.OVER * top) & np.isfinite(hr) & (hr < self.HR_SHARE * lthr)
        return out & ~self.ids(acts).isin(self.keep_ids).to_numpy()

    def apply(self, acts: pd.DataFrame, lthr: float) -> pd.DataFrame:
        """`power_ignored` per ride, and the power those rides lose blanked (a copy)."""
        acts = acts.copy()
        marked = self.marked(acts)
        screened = self.screened(acts, lthr) & ~marked
        acts["power_ignored"] = np.where(marked, self.MARKED, np.where(screened, self.HEART_RATE, ""))
        bests = [f"mmp_{d}" for d in MMP_DURATIONS if f"mmp_{d}" in acts]
        acts.loc[marked | screened, bests] = np.nan
        acts.loc[marked, [c for c in self.POWER_COLUMNS if c in acts]] = np.nan
        acts.loc[marked, [c for c in acts if c.startswith("effort_w_")]] = np.nan
        if "power_hist" in acts:
            acts["power_hist"] = acts["power_hist"].where(~marked, None)
        acts.loc[marked, "has_power"] = False
        if "power_coverage" in acts:
            acts.loc[marked, "power_coverage"] = 0.0
        return acts

    @staticmethod
    def effort_heart_rates(processed, acts: pd.DataFrame) -> pd.DataFrame:
        """acts with EffortHeartRate's columns (best power and heart rate around its end, per length) merged in."""
        eh = EffortHeartRate.table(processed)
        if not len(eh):
            return acts
        sid = PowerCheck.ids(acts)
        return acts.drop(columns=[c for c in eh.columns if c != "id" and c in acts]).assign(_sid=sid) \
            .merge(eh.rename(columns={"id": "_sid"}), on="_sid", how="left").drop(columns="_sid")

    @classmethod
    def edited(cls, bad: list, keep: list, ride: str, value: str) -> tuple[list, list]:
        """the two lists after the Ride page's choice for one ride: "bad" (its power is wrong), "keep" (the heart-rate
        screen leaves it alone) or "" (neither)."""
        if value not in ("bad", "keep", ""):
            raise ValueError("value must be bad, keep or ''")
        bad = [x for x in bad if str(x) != ride] + ([ride] if value == "bad" else [])
        keep = [x for x in keep if str(x) != ride] + ([ride] if value == "keep" else [])
        return bad, keep
