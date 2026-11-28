"""State-space model of one athlete's physiology, as a log posterior over everything unknown.

States (daily, unconstrained): a chain of ceilings - log MAP (the aerobic ceiling: 5-min power, about a ramp
test's MAP), logit f_T (CP / MAP), logit f_E (LT1 / 1-h power) - plus log Pmax, log durability (fade = 1 /
durability, see PowerCurve) and a fatigue F (hours at CP-equivalent). CP = f_T MAP, LT1 = f_E P(1 h), so
LT1 < 1-h power < CP < MAP always (LT1 as a share of CP passed 1-h power, 2026-11-01); W' is what puts the
power-duration curve's 5-min power at MAP. Severe / VO2max work raises the ceiling (and CP and LT1 with it);
threshold work fills CP toward MAP and endurance fills LT1 toward CP, each saturating as its share nears 1.

Each day:

* the day's power histogram is split into domains by the *current* LT1, CP, MAP and a sprint line
  (soft edges), so "threshold work" moves up as you get fitter; plus hours past 2 h in a ride (long). The band
  LT1-MAP splits into heavy (z2) and severe (z3) by power and heart rate together (v16): a second's severe share is
  logistic((log P - log CP) / EDGE_WIDTH + e), e the heart-rate evidence (metrics.severe_hr: corrected heart rate
  above threshold heart rate, in units of its scatter) - 0 without heart rate (planned sessions too), so the plain
  power edge. Power alone can't tell steady from severe within a few % of CP; heart rate alone counted long hot rides
  and climbs (PhysioData.hr_hist_h, EVIDENCE);
* dose -> an adaptation pool per state (gain matrix: which domain trains which state); the pool turns
  into capacity at rate rho * exp(-gamma F): you get stronger while recovering, slower while tired - the
  cost of fatigue is learned, not a set limit;
* capacity decays toward a detrained floor with its own time constant (TRAINED states only);
* Pmax and durability are slowly drifting levels (process noise only, no gains, no detraining): the
  data (no all-out sprints or long all-out efforts) can't say how they respond to training - with
  gains, those chains didn't mix (R-hat 2, 2026-10-25). Their floor / tau entries are then prior only;
* fatigue F builds with load sum_bins hours * (P / CP)^2 and clears with time constant tau_F; it lowers
  what you can do all-out by exp(-s_F F) (not capacity);
* process noise every NOISE_WEEKS weeks (non-centred, fixed size NOISE_SD per week, scaled by
  sqrt(NOISE_WEEKS) so the drift per unit time is the same) on each state: what the model doesn't explain.
  A period's change comes in equal parts on its days, not as one step.
  An illness's weeks get a larger size (PhysioData.noise_scale): unaware of one, the fit read the drop as fast
  detraining (2026-11-11).
  Its size is fixed, not fitted: with a fitted size the posterior mode runs off (big noise, every
  reading matched exactly). One term a week shared by all states was tried (2026-11-09): near-max power forecast
  as well, but LT1 and CP/MAP could then only move in step with MAP - the fit ran to 6-day detraining and put LT1
  45 % below alpha1 (into its bias).

Observations:

* ride best powers, censored - most rides aren't all-out. A mixture: all-out (Gaussian in log power,
  sd ALL_OUT_SD, around the fatigued curve) or not (log deficit below the *capacity* curve ~ Gamma,
  one shape; its mean a straight line in log duration (log), plus a step per ride kind: easy / hard /
  test-race - by mean and shape, not scale and shape: those two slid along a ridge, correlation -0.94). Deficits are measured from capacity so an easy
  day after a hard one reads as a choice, not as fatigue. Either way a small fixed share (BAD_READING_SHARE) may
  be bad readings, spread evenly in log power: one broken power meter's best no longer drags the curve to it;
* threshold readings (your FTP entries, alpha1, ramp tests) with a fitted bias and error per source;
* steady stretches (rides, >= 5-min reps): power at their heart rate. log power = log CP + g(HR / LTHR) + an offset
  for reps + c F + error: g is the rider's own heart-rate-to-power curve (HR_KNOTS, linear between), c how fatigue
  holds heart rate down (tired: more watts at the same HR), the error partly shared within a week (heat, hydration,
  sleep), integrated out (`hr_terms`). Replaces scaling each ride's power to threshold HR along a straight line from
  resting HR (2026-11-09): that read 0 % low from easy rides and 11 % low from tempo rides, so a change in the week's
  ride mix looked like a change in CP, and the fit made adaptation near-instant to follow it.

Recovery markers (sleep, HRV, resting HR, bedtime; PhysioData.recovery_z) are not in the model
(2026-10-26): as causes of recovery they fitted backwards (you sleep more *because* you are tired); as
daily signals of fatigue (with fatigue noise) they swamped everything - thousands of strongly
autocorrelated days turned fatigue into a slow drift (tau_F 115-340 d). They act through readiness
instead; a measurement model on weekly averages is the next thing to try.

Priors are Normal on the unconstrained values, so the negative log posterior is minimised directly
(MAP) and sampled with NUTS on the same vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gamma, norm

from .curve import PowerCurve

STATES = ("map", "f_e", "f_t", "pmax", "durability")
VERSION = 17        # bumped when a state's meaning or the parameters change (2: f_E = LT1 / 1-h power; 3: shared
#                      noise, one deficit shape, f_E's z1 / z2 effect shared; 4: noise per state again; 5: the
#                      heart-rate curve instead of HR readings scaled to threshold; 6: a reservoir rate per trained
#                      state; 7: LT1 trained by endurance hours alone, the deficit by its mean, HR fatigue centred;
#                      8: the MAP and Pmax priors centred on the rider's own first year, PhysioData.level_w, and
#                      half the weekly noise; 9: each best power weighted, PhysioData.effort_weight - a ride-day's
#                      overlapping windows count as a few observations, not up to 12; 10: an offset per duration on
#                      the shortfall's scale - removed: the shortfall's shape sharpened and the ceiling floated above
#                      tested power; 11: v9's likelihood, races hard rides instead of tests; 12: an illness's
#                      weeks may change more (PhysioData.noise_scale), VO2 effects on MAP held near the studies; 13: a week's
#                      unexplained change spread over its days; 14: LT1 share's detraining time constant prior narrower;
#                      15: z3 = severe by heart rate where there is heart rate; 16: the LT1-MAP band split by power and
#                      heart-rate evidence together, metrics.severe_hr; 17: a ride best may be a bad reading,
#                      BAD_READING_SHARE): a stored fit is refitted
# without a level of the rider's own (PhysioData.level_w): the start of MAP and Pmax (W) a trained rider's; and the
# detrained floors as shares of the start, whoever the rider
GENERIC_LEVEL_W = (290.0, 900.0)
FLOOR_SHARE = (250.0 / 290.0, 800.0 / 900.0)
DOSES = ("z1", "z2", "z3", "z4", "sprint", "long")
# a dose of 1 = 1 h below LT1 / 1 h LT1-CP / 10 min CP-MAP / 5 min above MAP / 1 min sprint / 1 h past 2 h.
# VO2max intervals (~105-120 % FTP) are mostly CP-MAP (z3): above MAP (z4) is above 5-min all-out power. z2 / z3 =
# the LT1-MAP band x (1 - s) / s, s the severe share from power and heart-rate evidence (PhysioModel.doses)
DOSE_UNIT_H = np.array([1.0, 1.0, 1 / 6, 1 / 12, 1 / 60, 1.0])
#                      z1 z2 z3 z4 sp long
GAIN_MASK = np.array([[1, 1, 1, 1, 0, 0],     # MAP: any aerobic work may raise the ceiling (the data say which)
                      [1, 1, 0, 0, 0, 0],     # f_E: endurance hours raise LT1 toward 1-h power (any ride length)
                      [0, 1, 1, 0, 0, 0],     # f_T: threshold / sweet spot (LT1-CP) and CP-MAP raise CP toward MAP
                      [0, 0, 0, 0, 0, 0],     # Pmax: drifts (see TRAINED)
                      [0, 0, 0, 0, 0, 0]],    # durability: drifts
                     dtype=bool)
GAIN_INDEX = np.nonzero(GAIN_MASK)
# the training effect (entry of log_effect) of each gain in GAIN_INDEX order: f_E's z1 and z2 share one - the data
# fix only their sum (both come with every easy ride; their two effects slid along a ridge, 2026-11-01). Long rides
# (hours past 2 h) no longer train LT1 on their own: they are endurance hours too, and with both the chains split
# between "endurance hours" and "long rides" (either alone fit as well, 2026-11-09)
GAIN_EFFECT = np.array([0, 1, 2, 3, 4, 4, 5, 6])
EFFECT_STATE = GAIN_INDEX[0][np.unique(GAIN_EFFECT, return_index=True)[1]]   # the state each effect trains
# prior sd of each training effect (log): wide, except MAP's from CP-MAP (VO2max intervals) and above-MAP work. A rider's
# 20-25 min a week there always comes with the hours, so their data can't tell its effect apart (2026-11-11): held
# near the studies' size instead (trained cyclists, ~0.05-0.18 % of MAP per 10 min at a 200-300-day time constant;
# the centre ~0.1 % there), not left to slide to nearly nothing. Shown as assumed when the data leave it there
EFFECT_SD = np.where(np.isin(np.arange(EFFECT_STATE.size), (2, 3)), 0.6, np.hypot(1.5, 0.7))
TRAINED = np.array([1.0, 1.0, 1.0, 0.0, 0.0])    # states with gains and detraining; the others only drift
# each state's reservoir rate (entry of log_rate): the trained states each their own; Pmax and durability have no
# reservoir (no gains), so theirs is never used
RATE_OF = np.array([0, 1, 2, 0, 0])
# the parameters of the daily dynamics (PhysioModel.step); fixed shapes, unlike eta (one row per week)
DYNAMICS = ("floor", "log_tau", "log_effect", "log_rate", "log_gamma", "log_tau_f")
QUANTITIES = ("lt1", "cp", "map", "ftp")      # what a threshold reading measures; ftp = 1-h power
KINDS = ("easy", "hard", "test", "maxed")     # effort kinds for the effort likelihood; maxed = heart rate shows it
                                              # went all-out (metrics.effort_hr.EffortHeartRate)
MAP_S = 300.0                                 # MAP := power for this long
EDGE_WIDTH = 0.05                             # soft domain edges, in log power
# heart-rate evidence that a second is severe (log-odds, added to the CP edge's): the centres of its bins in
# PhysioData.hr_hist_h; metrics.severe_hr clips evidence to +-EVIDENCE_MAX (past it the band split hardly moves)
EVIDENCE_MAX = 6.0
EVIDENCE = np.linspace(-EVIDENCE_MAX + 0.5, EVIDENCE_MAX - 0.5, 12)
RECOVERY = ("sleep", "hrv", "rhr", "bed")     # recovery markers (sign: + = better): 2-night sleep z, HRV z,
                                              # -resting HR z, -(hours later to bed than usual)
NOISE_SD = np.array([0.005, 0.015, 0.015, 0.01, 0.015])   # weekly unexplained change per state (unconstrained)
# (log MAP 0.5 %; the logits 0.015 = about 0.25 % of the share at 0.8). Halved 2026-11-10 (v8): at twice this the
# noise took changes training explains - forecasts from training ~20 % too flat, held-out LT1 readings 16.8 % off
# against 12.2 %, LT1's level partly in its readings' bias. A quarter forecast a little better still, but a tenth
# (or none) broke: LT1's level went into the bias (2.5-2.7 prior sds), near-max days 4.5 % against 4.0 %
NOISE_WEEKS = 1                                       # weekly (4 weekly: Pmax drifts to its limit, 2026-10-25)
MIN_READING_SD = 0.01
HR_KNOTS = np.array([0.70, 0.80, 0.90, 1.00])     # heart rate / LTHR where the heart-rate curve g has its values
HR_LOG_2PI = float(np.log(2.0 * np.pi))
# all-out efforts scatter this much (log) around the curve: power meter + day-to-day. Fixed: when fitted
# it widens until the "all-out" part of the mixture describes ordinary rides
ALL_OUT_SD = 0.03
# a ride best may be a bad reading (a broken power meter: 4:20 at 1000 W, heart rate 132): this fixed share of them
# are, spread evenly over BAD_READING_WIDTH of log power. Without it one such best sat ~40 ALL_OUT_SDs above the curve
# and lifted CP +86 %, MAP +119 % over a whole synthetic history (2026-11-26); with it a lone best more than ~12 %
# (easy ride) to ~15 % (test) over the curve costs about -log(SHARE / WIDTH) ~ 10 and pulls no further, while
# several that agree still move the curve. Fixed, not fitted (a fitted share would explain ordinary rides away)
BAD_READING_SHARE = 1e-4
BAD_READING_WIDTH = 2.0
MIN_SHAPE = 2.0              # Gamma shape of the effort deficit (see effort_terms)
# hard physical limits on the unconstrained states (MAP 40..1500 W, LT1/1-h 0.2..0.98, CP/MAP 0.4..0.98,
# Pmax 100..3000 W, fade 0.005..1): keeps wild optimiser trial points finite; nothing real is near them
STATE_LO = np.log([40.0, 0.25, 0.4 / 0.6, 100.0, 1.0])
STATE_HI = np.log([1500.0, 0.98 / 0.02, 0.98 / 0.02, 3000.0, 200.0])
# the model's arrays are padded to these sizes (masked), so its compiled code is reused as days, efforts and
# readings are added (and kept on disk across runs): a daily update or re-fit compiles once per bucket
DAY_BUCKET = 28              # days
OBS_BUCKET = 64              # efforts, readings


def bucket(n: int, size: int) -> int:
    """n rounded up to a multiple of `size` (at least one)."""
    return max(-(-n // size), 1) * size


def pad(a, n: int, fill=0) -> np.ndarray:
    """a with `fill` appended along the first axis up to length n."""
    a = np.asarray(a)
    return np.concatenate([a, np.full((n - len(a),) + a.shape[1:], fill, a.dtype)])


@dataclass(frozen=True)
class Source:
    """A kind of threshold reading: which quantity, prior bias sd and typical error (both in log power). `bound`:
    the reading is an upper bound (a ramp test reads high: the threshold is at most this), so only a state above
    it costs - a one-sided term, log Phi((log reading - log state - bias) / sd)."""
    name: str
    quantity: str
    bias_sd: float
    sd: float
    bound: bool = False


@dataclass
class PhysioData:
    """Everything the model sees, as plain arrays (T days, B power bins, N efforts, M readings)."""
    days: np.ndarray                 # (T,) datetime64[D]
    power_w: np.ndarray              # (B,) bin centre power (W)
    hist_h: np.ndarray               # (T, B) hours per power bin
    long_h: np.ndarray               # (T,) hours past 2 h, summed over the day's rides
    recovery_z: np.ndarray           # (T, R) recovery markers (RECOVERY), 0 where missing
    week: np.ndarray                 # (T,) week number (Monday weeks, first = 0)
    effort_day: np.ndarray           # (N,) day index of a ride best power
    effort_s: np.ndarray             # (N,) duration (s)
    effort_w: np.ndarray             # (N,) best power (W)
    effort_kind: np.ndarray          # (N,) index into KINDS
    reading_day: np.ndarray          # (M,) day index
    reading_w: np.ndarray            # (M,) value (W)
    reading_source: np.ndarray       # (M,) index into sources
    sources: tuple = field(default_factory=tuple)
    recovery_seen: np.ndarray | None = None   # (T, R) which markers were measured (default: non-zero)
    hr_day: np.ndarray = field(default_factory=partial(np.zeros, 0, int))     # (H,) steady stretches: day index,
    hr_w: np.ndarray = field(default_factory=partial(np.zeros, 0))             # power (W),
    hr_x: np.ndarray = field(default_factory=partial(np.zeros, 0))             # heart rate / LTHR,
    hr_rep: np.ndarray = field(default_factory=partial(np.zeros, 0, bool))     # a rep (else a whole ride)
    hr_rest: float = 0.3             # resting HR / LTHR (the heart-rate curve's prior: a straight line from rest)
    hr_fatigue_ref: float = 0.0      # the fatigue the heart-rate fatigue term is measured from (about the usual on
    #                                  steady days, PhysioInputs.usual_fatigue): curve and term then don't trade off
    level_w: tuple | None = None     # (MAP, Pmax) about where this rider starts, from their own first year
    #                                  (PhysioInputs.own_level): the centre of those priors, which in fixed watts sat
    #                                  2-3 sds from a much weaker or stronger rider. None: GENERIC_LEVEL_W
    effort_weight: np.ndarray | None = None   # (N,) each best power's weight in the likelihood (None: all 1); a
    #                                  day's best powers come from overlapping windows of the same rides (PhysioInputs)
    noise_scale: np.ndarray | None = None     # (T,) the unexplained weekly change's size, x NOISE_SD, each day
    #                                  (None: all 1); larger in an illness's weeks (PhysioInputs)
    # the part of hist_h with heart rate, by its heart-rate evidence (T, B, len(EVIDENCE); metrics.severe_hr.SevereSplit);
    # the rest of hist_h has evidence 0. None: none
    hr_hist_h: np.ndarray | None = None

    @property
    def weights(self) -> np.ndarray:
        """effort_weight, ones when not given."""
        return np.ones(len(self.effort_w)) if self.effort_weight is None else np.asarray(self.effort_weight, float)

    @property
    def noise_scales(self) -> np.ndarray:
        """noise_scale, ones when not given."""
        return np.ones(len(self.days)) if self.noise_scale is None else np.asarray(self.noise_scale, float)

    @property
    def hr_hours(self) -> np.ndarray:
        """hr_hist_h, zeros when not given."""
        return (np.zeros(np.shape(self.hist_h) + (len(EVIDENCE),)) if self.hr_hist_h is None
                else np.asarray(self.hr_hist_h, float))

    def ridden(self, a: int, b: int) -> dict:
        """PhysioModel.forecast's training for days a .. b - 1 (ridden days: their heart-rate parts too)."""
        return dict(hist_h=self.hist_h[a:b], long_h=self.long_h[a:b],
                    hr_hist_h=None if self.hr_hist_h is None else self.hr_hist_h[a:b])

    @property
    def n_periods(self) -> int:
        """process-noise periods (NOISE_WEEKS weeks each) these days span."""
        return int(self.week.max()) // NOISE_WEEKS - int(self.week.min()) // NOISE_WEEKS + 1 if len(self.week) else 0

    @property
    def first_period(self) -> int:
        return int(self.week[0]) // NOISE_WEEKS if len(self.week) else 0

    def until(self, end: int) -> PhysioData:
        """the first `end` days, with their efforts and readings (the data as it was on day `end` - 1)."""
        e, r = self.effort_day < end, self.reading_day < end
        return replace(self, days=self.days[:end], hist_h=self.hist_h[:end], long_h=self.long_h[:end],
                       recovery_z=self.recovery_z[:end], week=self.week[:end],
                       recovery_seen=None if self.recovery_seen is None else self.recovery_seen[:end],
                       noise_scale=None if self.noise_scale is None else self.noise_scale[:end],
                       hr_hist_h=None if self.hr_hist_h is None else self.hr_hist_h[:end],
                       effort_day=self.effort_day[e], effort_s=self.effort_s[e], effort_w=self.effort_w[e],
                       effort_kind=self.effort_kind[e], reading_day=self.reading_day[r], reading_w=self.reading_w[r],
                       reading_source=self.reading_source[r], **self.weight_part(e), **self.hr_part(self.hr_day < end, 0))

    def window(self, start: int) -> PhysioData:
        """the days from `start` on, with their efforts and readings (day indices from 0 again)."""
        e, r = self.effort_day >= start, self.reading_day >= start
        return replace(self, days=self.days[start:], hist_h=self.hist_h[start:], long_h=self.long_h[start:],
                       recovery_z=self.recovery_z[start:], week=self.week[start:],
                       recovery_seen=None if self.recovery_seen is None else self.recovery_seen[start:],
                       noise_scale=None if self.noise_scale is None else self.noise_scale[start:],
                       hr_hist_h=None if self.hr_hist_h is None else self.hr_hist_h[start:],
                       effort_day=self.effort_day[e] - start, effort_s=self.effort_s[e], effort_w=self.effort_w[e],
                       effort_kind=self.effort_kind[e], reading_day=self.reading_day[r] - start,
                       reading_w=self.reading_w[r], reading_source=self.reading_source[r],
                       **self.weight_part(e), **self.hr_part(self.hr_day >= start, start))

    def weight_part(self, keep: np.ndarray) -> dict:
        """effort_weight of the efforts where `keep` (for `until` / `window`)."""
        return dict(effort_weight=None if self.effort_weight is None else np.asarray(self.effort_weight)[keep])

    def hr_part(self, keep: np.ndarray, shift: int) -> dict:
        """the steady stretches where `keep`, their days `shift` earlier (for `until` / `window`)."""
        return dict(hr_day=self.hr_day[keep] - shift, hr_w=self.hr_w[keep], hr_x=self.hr_x[keep], hr_rep=self.hr_rep[keep])


@dataclass(frozen=True)
class Block:
    name: str
    mean: np.ndarray
    sd: np.ndarray


class ParamSpec:
    """Named blocks of one flat parameter vector, each with a Normal prior."""

    def __init__(self, blocks: list[Block]):
        self.blocks = blocks
        sizes = [b.mean.size for b in blocks]
        self.offsets = dict(zip([b.name for b in blocks], np.cumsum([0] + sizes[:-1]).tolist()))
        self.mean = np.concatenate([np.ravel(b.mean) for b in blocks]).astype(float)
        self.sd = np.concatenate([np.ravel(b.sd) for b in blocks]).astype(float)
        self.layout = tuple((b.name, np.shape(b.mean)) for b in blocks)    # hashable: static under jit

    @property
    def size(self) -> int:
        return self.mean.size

    @staticmethod
    def split(layout, z) -> dict:
        """z cut into its named blocks by `layout` ((name, shape), ...)."""
        out, o = {}, 0
        for name, shape in layout:                   # a handful of blocks
            n = int(np.prod(shape))
            out[name] = z[o:o + n].reshape(shape)
            o += n
        return out

    def unpack(self, z) -> dict:
        return self.split(self.layout, z)

    def pack(self, values: dict) -> np.ndarray:
        z = self.mean.copy()
        for b in self.blocks:
            if b.name in values:
                z[self.offsets[b.name]:self.offsets[b.name] + b.mean.size] = np.ravel(values[b.name])
        return z

    @staticmethod
    def prior(z, mean, sd):
        return jnp.sum(norm.logpdf(z, mean, sd))

    def log_prior(self, z):
        return self.prior(z, self.mean, self.sd)

    def names(self) -> list[str]:
        return [b.name if b.mean.size == 1 else f"{b.name}[{i}]" for b in self.blocks for i in range(b.mean.size)]

    def align(self, names, Z, rng=None) -> np.ndarray:
        """vectors (n, D_old) named `names` (an older fit) on this spec: shared names copied, the rest at
        the prior mean, or with `rng` drawn from the prior (e.g. new noise periods for posterior draws)."""
        Z = np.atleast_2d(np.asarray(Z, float))
        old = {n: i for i, n in enumerate(list(names))}
        src = np.array([old.get(n, -1) for n in self.names()], int)
        out = (np.broadcast_to(self.mean, (len(Z), self.size)).copy() if rng is None
               else self.mean + self.sd * rng.standard_normal((len(Z), self.size)))
        out[:, src >= 0] = Z[:, src[src >= 0]]
        return out


class PhysioModel:
    """Log posterior and simulation for one athlete's data. `z` = flat unconstrained parameter vector."""
    # whole-history simulations one JAX call runs at once (`batched`): many draws in one call held every draw's
    # intermediates together; 2026-11-13 the machine reset three times in a NUTS run's write-out (100 and 800 at once)
    BATCH = 10

    @classmethod
    def batched(cls, fn, Z, *args):
        """``fn(Z, *args)`` for parameter vectors Z (n, D) in batches of BATCH rows, results (arrays or a dict of them,
        first axis n) joined. The last batch is padded with copies of the last row: one compiled shape."""
        Z = np.asarray(Z)
        n = len(Z)
        Z = np.concatenate([Z, np.repeat(Z[-1:], -n % cls.BATCH, axis=0)])
        parts = [jax.device_get(fn(jnp.asarray(Z[i:i + cls.BATCH]), *args)) for i in range(0, len(Z), cls.BATCH)]
        return jax.tree.map(lambda *a: np.concatenate(a)[:n], *parts)

    def __init__(self, data: PhysioData, durations=None):
        """`durations`: the effort durations with their own deficit shape / scale (default: the data's);
        a window of the data passes the full model's, so its parameters line up."""
        self.data = data
        self.effort_s = np.unique(np.asarray(data.effort_s, float)) if durations is None else np.asarray(durations, float)
        self.effort_t = np.searchsorted(self.effort_s, np.asarray(data.effort_s, float))
        self.duration_x = self.standard_log(self.effort_s)
        self.spec = ParamSpec(self.blocks(data, len(self.effort_s)))
        self.T = len(data.days)
        self.arrays = self.padded_arrays(data, self.spec, self.effort_t, self.duration_x)
        self.power_w = self.arrays["power_w"]

    @staticmethod
    def standard_log(durations) -> np.ndarray:
        """log durations, centred and scaled to sd 1: where the deficit's log scale line is read."""
        x = np.log(np.asarray(durations, float))
        return (x - x.mean()) / (x.std() or 1.0) if x.size else x

    @staticmethod
    def padded_arrays(data: PhysioData, spec: ParamSpec, effort_t, duration_x) -> dict:
        """what the likelihood reads, as jax arrays padded to DAY_BUCKET / OBS_BUCKET: padded days have no
        training and are not run (`live`), padded efforts and readings count 0 (`effort_live`, `reading_live`;
        their values are just finite). Passed to the class-level compiled functions as arguments."""
        f = np.asarray
        T, N, M = len(data.days), len(data.effort_w), len(data.reading_w)
        nd, ne, nr = bucket(T, DAY_BUCKET), bucket(N, OBS_BUCKET), bucket(M, OBS_BUCKET)
        period = f(data.week, int) // NOISE_WEEKS - data.first_period            # row of eta
        # the day's share of its period's unexplained change: the same on each of its days (as one step on the
        # period's first day the fitted paths were staircases, a jump each week; 2026-11-12)
        noise_size = data.noise_scales / (7.0 * NOISE_WEEKS)
        per_source = f([QUANTITIES.index(s.quantity) for s in data.sources], int)
        quantity = per_source[f(data.reading_source, int)] if len(per_source) else np.zeros(M, int)
        bound = f([s.bound for s in data.sources], bool)[f(data.reading_source, int)] if len(per_source)             else np.zeros(M, bool)
        a = dict(power_w=f(data.power_w, float), mean=spec.mean, sd=spec.sd,
                 hist_h=pad(f(data.hist_h, float), nd), long_h=pad(f(data.long_h, float), nd),
                 live=pad(np.ones(T, bool), nd),
                 noise_size=pad(noise_size, nd)[:, None], period=pad(period, nd, period[-1] if T else 0),
                 effort_day=pad(f(data.effort_day, int), ne), effort_s=pad(f(data.effort_s, float), ne, MAP_S),
                 effort_w=pad(f(data.effort_w, float), ne, 1.0), effort_kind=pad(f(data.effort_kind, int), ne),
                 effort_t=pad(f(effort_t, int), ne), effort_x=pad(f(duration_x, float)[f(effort_t, int)], ne),
                 effort_live=pad(np.ones(N, bool), ne), effort_weight=pad(data.weights, ne),
                 reading_day=pad(f(data.reading_day, int), nr), reading_w=pad(f(data.reading_w, float), nr, 1.0),
                 reading_source=pad(f(data.reading_source, int), nr), reading_quantity=pad(quantity, nr),
                 reading_bound=pad(bound, nr),
                 reading_live=pad(np.ones(M, bool), nr))
        H = len(data.hr_w)
        nh = bucket(H, OBS_BUCKET)
        hd = f(data.hr_day, int)
        a |= dict(hr_day=pad(hd, nh), hr_logw=pad(np.log(f(data.hr_w, float)), nh), hr_x=pad(f(data.hr_x, float), nh, 1.0),
                  hr_rep=pad(f(data.hr_rep, float), nh), hr_live=pad(np.ones(H, bool), nh),
                  hr_group=pad(f(data.week, int)[hd] - (int(data.week[0]) if T else 0), nh),   # the week, from 0
                  hr_fatigue_ref=np.asarray(float(data.hr_fatigue_ref)))
        if data.hr_hist_h is not None:
            a["hr_hist_h"] = pad(f(data.hr_hist_h, float), nd)
        return {k: jnp.asarray(v) for k, v in a.items()}

    # --- priors ----------------------------------------------------------------------------
    @staticmethod
    def blocks(data: PhysioData, n_durations: int) -> list[Block]:
        a, lg = np.asarray, np.log
        logit = lg(0.75 / 0.25)
        g0 = a([0.004, 0.02, 0.01, 0.004, 0.02])[EFFECT_STATE]
        S = len(data.sources)
        map0, pmax0 = data.level_w or GENERIC_LEVEL_W
        return [
            # CP / MAP: 0.68-0.85 (90 %) - trained riders' range. Wider (sd 0.5), a huge never-tested ceiling with every
            # effort below it fitted as well as the real one (MAP and W' far above any test): the floor sets the level
            Block("x0", a([lg(map0), logit, lg(0.78 / 0.22), lg(pmax0), lg(1 / 0.08)]), a([0.3, 0.7, 0.3, 0.4, 0.5])),
            Block("floor", a([lg(FLOOR_SHARE[0] * map0), lg(0.70 / 0.30), lg(0.72 / 0.28), lg(FLOOR_SHARE[1] * pmax0),
                              lg(1 / 0.12)]), a([0.2, 0.5, 0.3, 0.3, 0.4])),
            # detraining time constants. LT1's share of 1-h power narrower (95 % about 27-130 days): after complete
            # cessation the threshold's share of the ceiling falls slowly and only partly, and has levelled within 8-12
            # weeks (Coyle et al. 1984, 1985; Mujika & Padilla 2000); at 0.7 the data left it wider than the prior and
            # NUTS stalled in its long tail (2026-11-12)
            Block("log_tau", lg(a([60.0, 60, 60, 90, 90])), a([0.7, 0.4, 0.7, 0.7, 0.7])),
            # training effect = gain x detraining time constant: how far a steady daily dose of 1 moves the
            # state in the long run. Sampled instead of the gain: the data pin down the product, so gain and
            # time constant slid along a ridge (R-hat 1.17, 27 divergences, 2026-10-25)
            Block("log_effect", lg(g0) + lg(a([60.0, 60, 60, 90, 90]))[EFFECT_STATE], EFFECT_SD),
            Block("eta", np.zeros((data.n_periods, 5)), np.ones((data.n_periods, 5))),
            # reservoir turnover per trained state (MAP, LT1 / 1-h share, CP / MAP share), from physiology: central /
            # VO2 adaptations show in about 1-2 weeks, threshold in 2-3, the oxidative (mitochondria, fat use) in 4-6;
            # median 10 / 35 / 17 days, 90 % about half to twice that. One shared rate fitted to ~1 day: the HR
            # readings' day-to-day swings were being followed (2026-11-09)
            Block("log_rate", lg(a([1 / 10, 1 / 35, 1 / 17])), np.full(3, 0.4)),
            Block("log_gamma", a([lg(0.05)]), a([1.0])),
            Block("log_tau_f", a([lg(7.0)]), a([0.3])),
            Block("log_s_f", a([lg(0.01)]), a([1.0])),
            Block("log_shape_m2", a([lg(2.0)]), a([0.7])),                   # shape - MIN_SHAPE, all durations
            # log mean deficit = level + slope x standardised log duration (PhysioModel.standard_log); scale = mean /
            # shape. By mean and shape: the data fix the mean, and scale and shape slid along a ridge (2026-11-09)
            Block("log_deficit", a([lg(0.4), 0.0]), a([1.0, 1.0])),
            Block("log_scale_kind", np.zeros(3), np.full(3, 0.5)),          # hard / test / maxed vs easy rides
            # chance an effort is all-out: easy 1 %, hard 5 %, test 40 %, maxed (heart rate at your max) 90 %
            Block("logit_pi", a([lg(0.01 / 0.99), lg(0.05 / 0.95), lg(0.4 / 0.6), lg(0.9 / 0.1)]),
                  a([0.5, 0.5, 1.0, 0.5])),
            Block("bias", np.zeros(S), a([s.bias_sd for s in data.sources])),
            Block("log_sd", lg(a([s.sd for s in data.sources]) - MIN_READING_SD), np.full(S, 0.3)),
            # the heart-rate curve: log (power / CP) at HR_KNOTS x LTHR; prior around the straight line from resting HR
            # (power share = (x - rest) / (1 - rest)), loose - the rider's own steady rides set it
            Block("hr_curve", lg(np.maximum(HR_KNOTS - data.hr_rest, 0.05) / (1.0 - data.hr_rest)),
                  np.full(HR_KNOTS.size, 0.2)),
            Block("hr_rep", a([0.0]), a([0.05])),            # reps (settled HR of a >= 5-min rep) vs whole rides
            Block("hr_fatigue", a([0.0]), a([0.02])),        # log power at the same HR per unit fatigue (HR held down)
            Block("hr_log_sd", a([lg(0.05)]), a([0.5])),     # a stretch's own error
            Block("hr_log_week_sd", a([lg(0.03)]), a([0.7])),   # the error a week's stretches share
        ]

    # --- states ----------------------------------------------------------------------------
    @staticmethod
    def capacity(x) -> dict:
        """unconstrained states (..., 5) -> watts / joules / fade: MAP, CP = f_T MAP, the W' that puts the curve's
        MAP_S power at MAP, and LT1 = f_E x the curve's 1-h power."""
        x = jnp.moveaxis(jnp.clip(jnp.asarray(x), STATE_LO, STATE_HI), -1, 0)
        c = dict(map=jnp.exp(x[0]), pmax=jnp.exp(x[3]), fade=jnp.exp(-x[4]))
        c["cp"] = jax.nn.sigmoid(x[2]) * c["map"]
        c["w_prime"] = PowerCurve.w_prime_for(MAP_S, c["map"], c["cp"], c["pmax"])
        c["lt1"] = jax.nn.sigmoid(x[1]) * PowerCurve.power(3600.0, c["cp"], c["w_prime"], c["pmax"], c["fade"])
        return c

    @staticmethod
    def gains(p) -> jnp.ndarray:
        """gain per unit dose (states x doses) = training effect / time constant."""
        g = jnp.exp(p["log_effect"][GAIN_EFFECT] - p["log_tau"][GAIN_INDEX[0]])
        return jnp.zeros(GAIN_MASK.shape).at[GAIN_INDEX].set(g)

    @staticmethod
    def doses(log_power, c: dict, hist, long, hr=None) -> jnp.ndarray:
        """the day's dose per domain (DOSES), domain edges from today's capacity: z1 below LT1, the band LT1-MAP split
        into z2 / z3 by the severe share s = logistic((log P - log CP) / EDGE_WIDTH + evidence), z4 MAP to the sprint
        line, sprint above. hr: (B, len(EVIDENCE)) the part of `hist` with heart rate by its evidence; the rest has
        evidence 0 (s = the CP edge)."""
        edges = jnp.log(jnp.stack([c["lt1"], c["cp"], c["map"], c["map"] + 0.5 * (c["pmax"] - c["map"])]))
        x = (log_power[None, :] - edges[:, None]) / EDGE_WIDTH
        above = jax.nn.sigmoid(x)
        band = above[0] - above[2]                                # LT1-MAP
        if hr is None:
            severe = (band * above[1]) @ hist
        else:
            s = jax.nn.sigmoid(x[1][:, None] + jnp.asarray(EVIDENCE)[None, :])
            severe = (band * above[1]) @ (hist - hr.sum(-1)) + jnp.sum(band[:, None] * s * hr)
        d = jnp.stack([(1.0 - above[0]) @ hist, band @ hist - severe, severe, (above[2] - above[3]) @ hist,
                       above[3] @ hist, long])
        return d / DOSE_UNIT_H

    @staticmethod
    def step(log_power, p, gain, carry, inputs):
        """one day. `work` = the day's hours x power^2 (its load is work / CP^2). Everything that doesn't
        depend on the state is computed for all days in `run`. A day that isn't `live` (padding) leaves the
        state as it was."""
        x, pool, fat = carry
        hist, long, work, noise, live = inputs[:5]
        c = PhysioModel.capacity(x)
        dose = PhysioModel.doses(log_power, c, hist, long, inputs[5] if len(inputs) > 5 else None)
        r = 1.0 - jnp.exp(-jnp.exp(p["log_rate"])[RATE_OF] * jnp.exp(-jnp.exp(p["log_gamma"][0]) * fat))
        x_new = x + r * pool - TRAINED * jnp.exp(-p["log_tau"]) * (x - p["floor"]) + noise
        pool_new = pool * (1.0 - r) + gain @ dose
        fat_new = fat * jnp.exp(-1.0 / jnp.exp(p["log_tau_f"][0])) + work / c["cp"] ** 2
        return jax.tree.map(partial(jnp.where, live), (x_new, pool_new, fat_new), carry), (x, fat)

    @staticmethod
    @jax.jit
    def run(power_w, p, hist, long, noise, carry, live=None, hr=None):
        """simulate from `carry` = (x, pool, fatigue): returns the end carry and each morning's (states, fatigue).
        live: the days to run (default all); on the others the state stays. hr: (days, B, len(EVIDENCE)) the part of
        `hist` with heart rate by its evidence (default none: evidence 0, the power edges alone).
        Compiled once per shape for the class (the data are arguments, not baked in): a re-plan reuses it."""
        live = jnp.ones(long.shape, bool) if live is None else live
        inputs = (hist, long, hist @ power_w ** 2, noise, live) + (() if hr is None else (hr,))
        return jax.lax.scan(partial(PhysioModel.step, jnp.log(power_w), p, PhysioModel.gains(p)), carry, inputs)

    @staticmethod
    def dynamics(p: dict) -> dict:
        """the parameters a forward run uses (DYNAMICS): their shapes don't change as days are added."""
        return {k: p[k] for k in DYNAMICS}

    @staticmethod
    def noise(p, a) -> jnp.ndarray:
        """each day's process noise: its share of the period's change, x the day's scale (PhysioData.noise_scale)."""
        return a["noise_size"] * NOISE_SD * np.sqrt(NOISE_WEEKS) * p["eta"][a["period"]]

    @staticmethod
    def simulate_of(z, layout, a, carry0=None, live=None):
        """`simulate` on the padded arrays `a` (padded days at the end: the state stays). live: the days to
        run, default the real ones (e.g. only the days before a window)."""
        p = ParamSpec.split(layout, z)
        if carry0 is None:
            carry0 = (p["x0"], jnp.zeros(len(STATES)), jnp.asarray(0.0))
        end, (X, F) = PhysioModel.run(a["power_w"], PhysioModel.dynamics(p), a["hist_h"], a["long_h"],
                                      PhysioModel.noise(p, a), carry0, a["live"] if live is None else live,
                                      a.get("hr_hist_h"))
        return X, F, end

    def simulate(self, z, carry0=None):
        """morning states (T, 5), fatigue (T,), end carry, for the data's days; from x0 (rested, nothing
        pending) or from `carry0` = (x, pool, fatigue), e.g. the state at the start of a window."""
        X, F, end = self.simulate_of(jnp.asarray(z), self.spec.layout, self.arrays, carry0)
        return X[:self.T], F[:self.T], end

    def forecast(self, z, hist_h, long_h, carry=None, hr_hist_h=None):
        """morning states (n, 5) and fatigue (n,) for the n days after the data's last day, given their
        training (power histograms, hours past 2 h; hr_hist_h: the part with heart rate by its evidence, as
        PhysioData's - PhysioData.ridden; planned days have none), without process noise.
        Day 0 = the morning after the last data day. `carry`: simulate(z)'s end state, if already known."""
        z = jnp.asarray(z)
        p = self.spec.unpack(z)
        end = self.simulate(z)[2] if carry is None else carry
        n = len(long_h)
        hr = None if hr_hist_h is None else jnp.asarray(hr_hist_h, float)
        _, (X, F) = self.run(self.power_w, self.dynamics(p), jnp.asarray(hist_h, float), jnp.asarray(long_h, float),
                             jnp.zeros((n, len(STATES))), end, None, hr)
        return X, F

    def all_out(self, z, x, fat, t):
        """what you could do all-out for t seconds with states x and fatigue (capacity x form)."""
        p = self.spec.unpack(jnp.asarray(z))
        return self.curve(x, t) * jnp.exp(-jnp.exp(p["log_s_f"][0]) * fat)

    def ahead(self, z, hist_h, long_h, t, hr_hist_h=None):
        """all-out power (n, len(t)) on each forecast morning for these future days' training."""
        X, F = self.forecast(z, hist_h, long_h, None, hr_hist_h)
        return self.all_out(z, X[:, None, :], F[:, None], jnp.asarray(t, float))

    # --- likelihood ------------------------------------------------------------------------
    @staticmethod
    def curve(x, t):
        """capacity power at duration t (s) for states x (..., 5)."""
        c = PhysioModel.capacity(x)
        return PowerCurve.power(t, c["cp"], c["w_prime"], c["pmax"], c["fade"])

    @staticmethod
    def effort_terms(p, a, X, F):
        """log likelihood of each ride best power, times its weight (effort_weight; 0 for padding)."""
        k, day = a["effort_kind"], a["effort_day"]
        u = jnp.log(a["effort_w"]) - jnp.log(jnp.maximum(PhysioModel.curve(X[day], a["effort_s"]), 1.0))
        u_fat = u + jnp.exp(p["log_s_f"][0]) * F[day]                  # vs the fatigued curve
        all_out = jax.nn.log_sigmoid(p["logit_pi"])[k] + norm.logpdf(u_fat, 0.0, ALL_OUT_SD)
        # not all-out: log deficit below capacity ~ Gamma(one shape, log mean = line in log duration + ride kind).
        # shape > MIN_SHAPE = 2: density and slope both 0 at no deficit (shape < 1: infinite density there,
        # the fit puts the curve on one ride; 1..2: infinite gradient there, NUTS can't take a step)
        shape = MIN_SHAPE + jnp.exp(p["log_shape_m2"][0])
        scale = jnp.exp(p["log_deficit"][0] + p["log_deficit"][1] * a["effort_x"] + jnp.r_[0.0, p["log_scale_kind"]][k]) / shape
        lg = gamma.logpdf(jnp.maximum(-u, 1e-9), shape, scale=scale)
        submax = jnp.where(u < 0, jax.nn.log_sigmoid(-p["logit_pi"])[k] + lg, -jnp.inf)
        good = np.log1p(-BAD_READING_SHARE) + jnp.logaddexp(all_out, submax)
        term = jnp.logaddexp(good, np.log(BAD_READING_SHARE / BAD_READING_WIDTH))   # or a bad reading
        return jnp.where(a["effort_live"], a["effort_weight"] * term, 0.0)

    @staticmethod
    def quantities(x) -> jnp.ndarray:
        """(4, ...) LT1, CP, MAP, 1-h power (capacity, no fatigue) in QUANTITIES order."""
        c = PhysioModel.capacity(x)
        ftp = PowerCurve.power(3600.0, c["cp"], c["w_prime"], c["pmax"], c["fade"])
        return jnp.stack([c["lt1"], c["cp"], c["map"], ftp])

    @staticmethod
    def reading_terms(p, a, X):
        """log likelihood of each threshold reading (0 for padding)."""
        if p["bias"].shape[0] == 0:                  # no reading sources
            return jnp.zeros(a["reading_w"].shape)
        Q = PhysioModel.quantities(X[a["reading_day"]])
        pred = jnp.log(Q[a["reading_quantity"], jnp.arange(a["reading_day"].shape[0])]) + p["bias"][a["reading_source"]]
        sd = (MIN_READING_SD + jnp.exp(p["log_sd"]))[a["reading_source"]]
        r = jnp.log(a["reading_w"])
        ll = jnp.where(a["reading_bound"], norm.logcdf((r - pred) / sd), norm.logpdf(r, pred, sd))   # bound: one-sided
        return jnp.where(a["reading_live"], ll, 0.0)

    @staticmethod
    def hr_curve(p, x):
        """g(x): log (power / CP) at heart rate x x LTHR, linear between HR_KNOTS and beyond the ends."""
        k = jnp.asarray(HR_KNOTS)
        j = jnp.clip(jnp.searchsorted(k, x) - 1, 0, HR_KNOTS.size - 2)
        t = (x - k[j]) / (k[j + 1] - k[j])
        return (1.0 - t) * p["hr_curve"][j] + t * p["hr_curve"][j + 1]

    @staticmethod
    def hr_terms(p, a, X, F, use=None):
        """log likelihood of the steady stretches (`use`: which count, default the real ones). Residual
        r = log power - (log CP + g + rep offset + c F); within a week r ~ N(0, s^2 I + w^2 1 1'), per week in closed
        form: -(sum r^2 - w^2 S^2 / (s^2 + n w^2)) / (2 s^2) - ((n - 1) log s^2 + log(s^2 + n w^2) + n log 2 pi) / 2."""
        m = (a["hr_live"] if use is None else use).astype(float)
        day = a["hr_day"]
        cp = PhysioModel.capacity(X[day])["cp"]
        r = (a["hr_logw"] - jnp.log(cp) - PhysioModel.hr_curve(p, a["hr_x"]) - a["hr_rep"] * p["hr_rep"][0]
             - p["hr_fatigue"][0] * (F[day] - a["hr_fatigue_ref"])) * m
        G = a["hist_h"].shape[0] // 7 + 2                          # weeks the padded days can span
        S = jax.ops.segment_sum(r, a["hr_group"], G)
        Q = jax.ops.segment_sum(r * r, a["hr_group"], G)
        n = jax.ops.segment_sum(m, a["hr_group"], G)
        s2, w2 = jnp.exp(2.0 * p["hr_log_sd"][0]), jnp.exp(2.0 * p["hr_log_week_sd"][0])
        den = s2 + n * w2
        return -0.5 * jnp.sum((Q - w2 * S * S / den) / s2 + (n - 1.0) * jnp.log(s2) + jnp.log(den) + n * HR_LOG_2PI)

    @staticmethod
    def log_posterior_of(z, layout, a, carry0=None):
        """log posterior on the padded arrays `a` (the class-level compiled fits call this)."""
        p = ParamSpec.split(layout, z)
        X, F, _ = PhysioModel.simulate_of(z, layout, a, carry0)
        return (ParamSpec.prior(z, a["mean"], a["sd"]) + jnp.sum(PhysioModel.effort_terms(p, a, X, F))
                + jnp.sum(PhysioModel.reading_terms(p, a, X)) + PhysioModel.hr_terms(p, a, X, F))

    @staticmethod
    def loglik_of(z, layout, a, first_day):
        """log likelihood of the efforts and readings on day `first_day` and later (whole simulation)."""
        p = ParamSpec.split(layout, z)
        X, F, _ = PhysioModel.simulate_of(z, layout, a)
        return (jnp.sum(jnp.where(a["effort_day"] >= first_day, PhysioModel.effort_terms(p, a, X, F), 0.0))
                + jnp.sum(jnp.where(a["reading_day"] >= first_day, PhysioModel.reading_terms(p, a, X), 0.0))
                + PhysioModel.hr_terms(p, a, X, F, a["hr_live"] & (a["hr_day"] >= first_day)))

    # compiled once for the class per shape bucket: (draws, D) -> (draws,)
    loglik_draws = staticmethod(jax.jit(jax.vmap(loglik_of.__func__, in_axes=(0, None, None, None)), static_argnums=1))
    log_posterior_draws = staticmethod(jax.jit(jax.vmap(log_posterior_of.__func__, in_axes=(0, None, None)),
                                               static_argnums=1))

    def log_posterior(self, z, carry0=None):
        return self.log_posterior_of(z, self.spec.layout, self.arrays, carry0)

    def loglik_from(self, z, first_day: int):
        return self.loglik_of(z, self.spec.layout, self.arrays, first_day)

    def neg_log_posterior(self, z):
        return -self.log_posterior(z)

    # --- outputs ---------------------------------------------------------------------------
    @staticmethod
    def daily(z, layout, a) -> dict:
        """daily capacities (W, J), f_T, 1-h power, fatigue and its all-out factor (padded days included)."""
        p = ParamSpec.split(layout, z)
        X, F, _ = PhysioModel.simulate_of(z, layout, a)
        out = PhysioModel.capacity(X)
        out["f_t"] = out["cp"] / out["map"]       # = sigmoid(x_f_t), kept as a column
        out["ftp"] = PhysioModel.quantities(X)[3]
        out["fatigue"] = F
        out["form"] = jnp.exp(-jnp.exp(p["log_s_f"][0]) * F)
        return out

    daily_one = staticmethod(jax.jit(daily.__func__, static_argnums=1))
    daily_many = staticmethod(jax.jit(jax.vmap(daily.__func__, in_axes=(0, None, None)), static_argnums=1))

    def trajectory(self, z) -> dict:
        """`daily` for one parameter vector, as numpy arrays (T,)."""
        return {k: np.asarray(v)[:self.T] for k, v in self.daily_one(jnp.asarray(z), self.spec.layout, self.arrays).items()}

    def trajectories(self, Z) -> dict:
        """`daily` for many parameter vectors (draws, D), BATCH at a time: arrays (draws, T)."""
        return {k: v[:, :self.T] for k, v in self.batched(self.daily_many, Z, self.spec.layout, self.arrays).items()}
