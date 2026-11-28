# Model page: design critique and fixes

The screenshots named below are in the pull request's description (the repository holds no binary files).

A design critique of the Model page (`#model`), done on 2026-11-27 against `main` after the Tailwind CSS + daisyUI
restyle, and the fixes for its five highest-priority findings. The other findings are listed at the end for later.

**How it was checked:** a synthetic rider (`goodpush synth new`, 180 days, 86 rides) with `goodpush run`, then a
short sampling run (`goodpush physio --nuts --samples 150 --warmup 120`), so the page showed both the "best fit only"
state and the "sampling run didn't pass its checks" state. The page was captured with Playwright at 1366 × 860 and
at iPhone 13 width, in light and dark mode, and contrast was measured on the rendered page. It had no console errors
and no horizontal scroll. The backtest and coverage sections weren't covered, because they need hour-long runs.

## Overall impression

The page is careful and honest. It shows uncertainty, bias, and how far the model can be trusted, which most training
apps hide. The biggest opportunity was to make it answer its own question at the top. Its hint says the question is
*"Can you trust the physiology model behind the Rider page and the plan?"*, but the page opened with a version number
and two tiles that gave opposite answers about the ranges.

## The five fixes

### 1. One answer about the ranges, at the top

**Before:** Status had two tiles both labelled **Ranges**. After a sampling run, the first said "Nov 26, 2026" (it has
ranges) and the second said "not yet" (they can't be trusted). Before a run they said "none" and "none yet". The
"Run sampling + backtest" button sat alone in a full-width strip between them, and "Today, with its uncertainty"
showed its 5–95 % ranges with no warning.

_Screenshot in the PR description: Before: two Ranges tiles (`before-status.jpg`)._

**After:** a single verdict card opens the page: **Ranges: none yet / rough / trustworthy / from an earlier run**,
with one sentence on why and the run button underneath it. The button is the primary action only when the ranges
need a run. Status keeps Version, Fitted to and Backtest, plus the forecast-check tiles when there are any. Dates are
short ("Nov 26"), so the tiles no longer wrap. When the checks failed, the uncertainty table says its ranges are rough.

_Screenshot in the PR description: After: one verdict (`after-status.jpg`)._

### 2. Predicted vs observed reads as predicted vs observed

**Before:** neither axis had a title, and the two axes used different ranges (x 220–264 W, y 100–300 W). The
"observed = predicted" line came out almost flat, so "on the line" and "well above it" were hard to judge.

_Screenshot in the PR description: Before: flat identity line (`before-predicted.jpg`)._

**After:** both axes are titled (*Predicted: the model's all-out power that day*, *Observed: your best*) and share one
range and tick step, so the identity line runs corner to corner. This uses three new `EChart.draw` options in
`charts.js` (`xName` / `yName`, `yDomain`, `tick`), which other charts can use too.

_Screenshot in the PR description: After: titled, equal axes (`after-predicted.jpg`)._

### 3. Intensity colours keep their order in dark mode

**Before:** the intensity scale (`--z1` … `--z7`) ran yellow (easy) → navy (hard) in light mode, but slate (easy) →
bright yellow (hard) in dark mode. The scale flipped, so in dark mode the hardest riding got the colour that meant
easiest in light mode. This shows on the Model, Rider, Calendar and Ride pages.

_Screenshot in the PR description: Before: reversed scale in dark mode (`before-riding-themes.jpg`)._

**After:** dark mode keeps the yellow → blue order, with lighter blues that stay visible on the dark background
(each at least 4.2:1 against the card). Text drawn on those colours uses new `--on-z1/3/5/7` tokens, so it stays
readable in both themes.

_Screenshot in the PR description: After: same order in both themes (`after-riding-themes.jpg`)._

### 4. Threshold readings on the same footing as the model

**Before:** the chart drew raw alpha1 readings around 145 W above a model LT1 line around 122 W. The +20.8 % bias
that explains the gap was only in the table underneath, so at a glance the model looked about 20 W off. Sources had
code names (`alpha1 lt1`, `ramp lt2`).

_Screenshot in the PR description: Before: readings 20 W above the line (`before-threshold.jpg`)._

**After:** a **Bias removed / As read** switch (bias removed by default, remembered per browser) divides each reading
by its source's learned offset, so readings can be compared with the model's line. Sources have plain names
("alpha1 on rides (LT1)", "Ramp test, alpha1 (LT2)", "FTP you entered"…) in the legend, the tooltip and the table.

_Screenshot in the PR description: After: bias removed (`after-threshold.jpg`)._

### 5. Progress for a sampling run started elsewhere

**Before:** while a sampling run from a terminal (or the weekly ranges refresh) was going, the button just greyed out
with "Another sampling run is going on … wait for it to end". Nothing on the page changed until it was reloaded, even
though the run itself prints its progress.

**After:** a NUTS run now writes its start and last progress line to `processed/physio_nuts_progress.log` while it
holds the NUTS lock (`PhysioRun.tracked`). Peer sync already skips `*.log`. `/api/model_run` returns it as
`other`, and the page shows "A sampling run started outside the app is going on · 4 min so far", with that line. It
keeps polling and reloads the model when the run ends.

_Screenshot in the PR description: After: a run started from a terminal (`after-run-elsewhere.jpg`)._

## Second pass (after upstream 0.4.1)

Most of the other findings were fixed in a second pass:

| Finding | Severity | Done |
|---|---|---|
| Phone: the riding table and the threshold table were cut off on the right, with no sign that they scroll | Moderate | Both become one card per row on a phone (`table.stacktable`; `.kv` puts short values beside their names) |
| Fatigue y axis (0–4) had no unit | Minor | Titled "Recent hours at CP" |
| Red was used for "training explains part of the change" | Minor | Amber for part or "too small", red only when training explains little (under 20 %) |
| "What each kind of riding did for you" was a blue disclosure styled unlike every other section, yet open by default | Minor | An ordinary sub-heading under "What training explains" |
| The sections mixed "can I trust it?" with "what it says about you" | Minor | Two groups under the verdict: *What the model says about you* (today's ranges, fatigue, training), then *How far to trust it* (status, sampler, predicted vs observed, readings) |
| Mixed date formats ("2026-11-27 17:00" in sampler details) | Minor | "Sampling run of Nov 27, 17:00" |
| The three main charts had no text alternative | Moderate (a11y) | `EChart.draw(..., {label})` sets `role="img"` and an `aria-label` with a one-sentence summary (now, highest, how many efforts sit above the prediction) |
| Hard ride, Test and Heart rate maxed were told apart by colour only | Minor (a11y) | Circle, diamond and triangle markers, with the same shapes in the legend |
| Segmented buttons were 40 px tall | Minor (a11y) | 44 px on phones |
| Range bars used a different hue per row | Minor | Upstream's zone colour map (0.4.1) now colours them by intensity |

_Screenshot in the PR description: Phone: the readings table, one card per row (`after-phone-readings.jpg`)._

Still open: the 💡 hints (25 after a sampling run) could become one-line captions where they hold a section's main
idea. That's a copy change best made with the guide's author.

## Third pass (2026-11-28, model v17, before any sampling run)

A fresh look at the page as a new model version leaves it: best fit only, no ranges.

| Finding | Severity | Done |
|---|---|---|
| The verdict said "today's numbers come from the best fit only", but the page didn't show them: "Today, with its uncertainty" disappears without ranges | Moderate | **Today, from the best fit**: CP, MAP, FTP, LT1 and W′ (in kJ, as on the Rider page), with "after a sampling run" where the range will go |
| The fatigue legend named a dashed "Planned" line that isn't drawn when there's no plan | Minor | "Planned" shows only when the plan's forecast reaches past today |
| The riding table's 💡 sat alone on a row at the top of its card | Minor | It sits in the heading, like every other section |

_Screenshot in the PR description: After: today's numbers from the best fit (`after-best-fit.jpg`)._

## Additions (2026-11-28)

Five things the page didn't say, added after the third pass:

| Added | Where | From |
|---|---|---|
| **6-week change** beside each of today's numbers | Today table (best fit or ranges) | the best fit's daily line (`S.physio`) |
| **What the latest rides moved**: today's best fit against the one before the newest data day; a move over 3 % is flagged as worth checking that ride's power | under the Today table | `physio_diagnostics.json` `best` / `previous` (new, `PhysioDiagnostics.previous`) |
| **What the fit used**: days and first day, days with riding and hours, best powers, threshold readings, rides whose power the power check set aside | under Status | `inputs` (new, `PhysioDiagnostics.inputs`); set-aside counts by the server (`PhysioDiagnostics.set_aside`) |
| **Predicted vs observed in one sentence**: how many efforts beat the prediction, and how close the five closest to all-out sit | under the chart, follows the duration buttons | the efforts already on the page |
| **How fresh the readings are**: the latest LT1 and CP reading, and a ramp test named when one is older than 6 weeks | under the readings table | the readings already on the page |

_Screenshot in the PR description: Today's numbers with their 6-week change and what the latest rides moved (`after-trend-moved.jpg`)._

_Screenshot in the PR description: What the fit used (`after-fit-used.jpg`)._

## What works well

* The page shows its doubts: ranges, divergences, bias and spread, and "≈ as assumed" where the data barely moved a
  prior.
* "Today, with its uncertainty" is the clearest section: median, range and a bar each, with "assumed" chips where a
  value isn't learned.
* Sampler details sits behind a disclosure, so experts can check R-hat and chain traces without cluttering the page.
