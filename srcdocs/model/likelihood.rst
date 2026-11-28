Likelihood
==========

Given the simulated states, the data's log likelihood has two parts
(:meth:`goodpush.physio.PhysioModel.effort_terms`, :meth:`goodpush.physio.PhysioModel.reading_terms`). The
log posterior is the sum of these and the log prior.

Ride best powers: a censored mixture
------------------------------------

Take a ride's best power :math:`y` (W) over a duration :math:`\Delta` (one of the 12 durations) on day :math:`t`,
whose kind is :math:`k` (easy, hard workout or race, test, or maxed - heart rate confirmed it all-out, :doc:`inputs`). Let :math:`C(\Delta; x_t)` be the capacity curve
:math:`P(\Delta)` of :doc:`overview` for that day's states :math:`x_t` (fresh). Then

.. math::

   v = \log y - \log C(\Delta; x_t), \qquad v_F = v + s_F F_t

:math:`v` is how far the effort was above (> 0) or below (< 0) capacity, in log power; :math:`v_F` is the same
against the fatigued curve :math:`C\,e^{-s_F F_t}` (fatigue :math:`F_t` and its cost :math:`s_F`, see
:doc:`dynamics`). Either the effort was **all-out**, with probability :math:`\psi_k`, or it wasn't:

.. math::

   p(y) = (1 - \beta)\left[\psi_k\; \mathcal N\!\left(v_F;\ 0,\ 0.03\right)
        \;+\; (1 - \psi_k)\; \mathbb 1[v < 0]\;
        \mathrm{Gamma}\!\left(-v;\ \alpha,\ \mu_\Delta\, e^{\kappa_k} / \alpha\right)\right] \;+\; \beta / w

where :math:`p(y)` is the likelihood of the best power, :math:`\mathcal N(v; 0, 0.03)` the normal density with
mean 0 and standard deviation 0.03 at :math:`v`, :math:`\mathrm{Gamma}(v; \alpha, \theta)` the gamma density with
shape :math:`\alpha` and scale :math:`\theta`, :math:`\mathbb 1[v < 0]` is 1 below capacity, else 0, and
:math:`\beta = 10^{-4}`, :math:`w = 2` the bad-reading share and width (below). Fitted:

* :math:`\psi_k` - the chance a best power is all-out, per kind: an easy ride is rarely all-out at any
  duration, a test often is, a maxed effort almost always. A race counts as a hard ride, not a test: its best
  powers come inside repeated efforts (surges, sprints, sitting in), not from one fresh steady effort, so its 20 min
  and longer fall far below capacity like a hard workout's (in a coverage check they sat well under the curve when
  races were tests);
* :math:`\alpha, \mu_\Delta` - shape and mean of how far below capacity (log) a best power that isn't all-out
  falls: **one** shape for all durations, and a log mean on a straight line in log duration,
  :math:`\log \mu_\Delta = a + b\, x_\Delta` (:math:`x_\Delta` = log duration, centred and scaled to sd 1). One
  shape and a line forecast as well as a shape and scale per duration (12 + 12 parameters). By mean and shape, not
  scale and shape: the data fix the mean, and scale and shape slid along a ridge (correlation -0.94 in NUTS);
* :math:`\kappa_k` - a log multiplier of the mean for hard (and race), test and maxed efforts (:math:`\kappa_{easy} = 0`).

What the two parts mean:

* **All-out**: Gaussian in log power around the *fatigued* curve, sd 3 % (power meter + day to day). The sd is fixed:
  when fitted, it widens until the "all-out" part describes ordinary rides.
* **Not all-out**: the log deficit below *capacity*, :math:`-v`, is gamma distributed. Deficits are measured from capacity, not the
  fatigued curve, so an easy day after a hard one reads as a choice, not as fatigue. A best power *above* capacity can only be all-out - or a bad reading.
* **A bad reading**: a small fixed share :math:`\beta` of best powers are a broken power meter's, spread evenly over
  :math:`w` in log power. Without it a lone best far above the curve (minutes at 1000 W, heart rate barely above
  easy) sat some 40 sds over it, and the cheapest way to explain it was to lift the states for the whole history:
  on a synthetic rider one such ride put CP 86 % and MAP 119 % over the truth. With it, a lone best more than about
  12 % (easy ride) to 15 % (test) over the curve costs a fixed :math:`-\log(\beta / w) \approx 10` and pulls no
  further, while several bests that agree still move the curve. Fixed: a fitted share would explain ordinary rides
  away. Rides whose power is known to be wrong are left out before the model sees them (:doc:`inputs`).
* :math:`\alpha \ge 2`: then the deficit density and its slope are both 0 at no deficit. With shape < 1 the
  density is infinite there (the fit puts the curve through one ride); between 1 and 2 its gradient is infinite
  (NUTS can't take a step).

**A weight per best power.** A ride-day gives a best power for each of up to 12 durations, all from overlapping
windows of the same riding: one hard effort shows at several durations. Counted as independent, they made the
posterior ranges too narrow. Each best power's log likelihood is therefore multiplied by

.. math::

   \omega = \min\!\left(3 / n_t,\ 1\right)

with :math:`n_t` the number of best powers on its day, so a ride-day counts as at most three observations. The
weight changes how sure the model is, hardly where it puts you: fitted with and without it, the levels moved by a
few watts at most.

This is how the model reads the **whole power curve**: every day contributes 12 durations from 5 s to 2 h, each
either an all-out point on the curve or a lower bound under it. Durations that are rarely ridden all-out (sprints,
very long efforts) are mostly lower bounds, so the parts of the curve they set (Pmax, durability) are weakly
identified.

Threshold readings
------------------

A reading :math:`r` (W) on day :math:`t` from source :math:`j` (:doc:`inputs`), which measures the quantity
:math:`q` - one of LT1, CP, MAP or 1-h power:

.. math::

   \log r \sim \mathcal N\!\left(\log Q_q(x_t) + b_j,\ \ 0.01 + s_j\right)

where :math:`Q_q(x_t)` is that quantity computed from the day's states (capacity, no fatigue), :math:`b_j` the
source's bias (log power: 0.05 ≈ 5 % high) and :math:`s_j` its error standard deviation, both fitted per source;
0.01 is a floor on the standard deviation. :math:`\mathcal N(m, s)` is a normal distribution with mean :math:`m`
and standard deviation :math:`s`. A source with a consistent offset
(say, alpha1 thresholds from ordinary rides reading high) gets a bias instead of dragging the state; the priors on
the biases (:doc:`priors`) decide how far each source is trusted to set the level.

Upper bounds
^^^^^^^^^^^^

A ramp test reads its thresholds high: alpha1 needs a couple of minutes of data :cite:p:`gronwald2020` and heart rate trails the rising
power, so by the time either crosses its threshold value the power is already past the threshold, and
steady-state thresholds sit at or below a ramp's. Its LT1 and LT2 (``bound`` sources) therefore say "at most
this", and enter one-sided:

.. math::

   \log \Phi\!\left(\frac{\log r - \log Q_q(x_t) - b_j}{0.01 + s_j}\right)

with :math:`\Phi` the standard normal distribution function. A state well below the reading costs nothing; one
above it costs about as much as a normal reading would. It is soft (:math:`s_j` about 3.5 %), as alpha1 is noisy,
a strap can drop beats and a test can come out low. Ordinary rides push the curve up from below (best powers are
lower bounds, see above); tests cap it from above, and a test whose crossing came only at exhaustion is a loose
cap rather than a wrong value.

Steady stretches: power at heart rate
-------------------------------------

A steady stretch (:doc:`inputs`) with power :math:`P` at heart rate :math:`h` on day :math:`t`:

.. math::

   \log P = \log \mathrm{CP}(x_t) + g(h / \mathrm{LTHR}) + o\,[\text{rep}] + c\,F_t + e

* :math:`g` - **your heart-rate curve**: log (power / CP) at a heart rate, given at 70, 80, 90 and 100 % of
  threshold heart rate and straight between (and beyond). Fitted: the zones are your own, not a formula;
* :math:`o` - a rep's offset against whole rides (a rep's settled heart rate vs a ride's average with its warm-up
  and drift);
* :math:`c` - how fatigue holds heart rate down: tired, the same heart rate comes with more watts (and is hard to
  raise), so the model does not read that as fitness. Fatigue is measured from about its usual value on steady
  days (so :math:`g` is your curve at usual fatigue, and the two don't trade off); about half of :math:`c` overlaps
  slow trends (fatigue rose over the years), so it is a correction, not a result;
* :math:`e` - the error, partly shared by a week's stretches (heat, hydration, sleep):
  :math:`e \sim \mathcal N(0, s^2 I + w^2 \mathbf 1 \mathbf 1^\top)` within a week, independent between weeks.
  The shared part is integrated out in closed form per week, so it adds two parameters (:math:`s, w`), not one
  per week.

A heart rate that is too *high* for the watts can also mean illness or poor recovery; that side is not modelled
(it goes into the error and the weekly part; readiness handles it day to day).

What is assumed independent
---------------------------

Given the states, all best powers and readings are independent. Two consequences:

* one hard ride gives up to 12 best powers that are really one effort seen at several durations; the weight above
  counts them as at most three pieces of evidence, a rough correction rather than a model of how they overlap;
* readings from the same ride (e.g. a steady stretch and an alpha1 reading) are treated as separate looks.

Both make the posterior more confident than the data justify. See :doc:`limits`.
