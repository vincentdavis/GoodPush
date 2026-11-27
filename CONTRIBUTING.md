# Contributing to GoodPush

Thank you for helping. GoodPush is a beta, built by one person in their spare time, and the most useful help needs
**no programming at all**: using it, saying what goes wrong, and saying whether its numbers match what you know
about your own body.

| You can... | Time | How |
|---|---|---|
| Report a problem | 5 min | In the app: **Beta** (top right) > **Report a problem**, or [a bug report](https://github.com/GoodPush/GoodPush/issues/new?template=bug_report.yml) |
| Tell us whether the numbers make sense for you | 10-20 min | [Do the numbers make sense?](https://github.com/GoodPush/GoodPush/issues/new?template=numbers.yml) |
| Suggest an idea or ask a question | 5 min | [Idea or question](https://github.com/GoodPush/GoodPush/issues/new?template=idea.yml) |
| Fix a confusing sentence in the app or the guide | 10 min | [Wording and docs](#wording-and-docs) below |
| Write code | any | [Code](#code) below |

All of these need a free [GitHub account](https://github.com/signup). Everything on GitHub issues is **public**.

## Before you post: your data is personal

GoodPush works with health data. Issues are public and stay online, so share only what you're happy for anyone to
read:

* **Never attach** your ride files (.fit / .gpx / .tcx), account exports, `goodpush.toml`, or anything from the
  `data/` or `reports/` folders. Never paste a Garmin key, Strava secret or other password.
* **Screenshots** are very helpful, but check them first for your name, ride maps (they show where you live and
  ride), dates you'd rather not share and health numbers.
* **Relative numbers are enough.** "GoodPush's LT1 is about 10 % below my lab result" tells us as much as the watts.
* If a problem only shows with your data, say so in the issue: we'll find a private way to look at it together.

## Report a problem

The bug form asks for:

1. **What you did**, step by step ("opened Plan, clicked Add phase, picked Build").
2. **What you expected**, and **what happened instead**, with any error message as text.
3. **Your GoodPush version and computer.** The app's **Report a problem** button fills these in for you. Otherwise:
   the version is at the bottom of **Settings**.

One problem per issue. If someone already reported it, add a thumbs-up or a comment there instead of a new issue:
it tells us how many people it affects.

## Does the model work for you?

GoodPush builds a model of *your* physiology from your rides: critical power, LT1, MAP, W', sprint power,
durability, fatigue, and how each kind of training changes them. It was built and tested on very few people, so the
biggest open question is simple: **do its numbers make sense for other riders?**

You don't need to understand the model to help. You need to know something about your own body that you trust:

* a **lab test** (lactate or gas exchange: LT1 / LT2, VO2max power)
* a **ramp test** or **20-minute FTP test**, a recent **all-out effort** or a **race**
* **heart-rate zones** you've confirmed, or simply **how sessions feel**: is the plan's "easy" easy? Can you hold its
  threshold intervals?
* how your **fitness changed** over a season, an injury or an illness, and whether GoodPush's charts follow it

Then compare it with what GoodPush shows (mostly the **Rider** page; the **Model** page shows how sure it is) and
[tell us](https://github.com/GoodPush/GoodPush/issues/new?template=numbers.yml). Useful answers include:

* **"It matches."** Agreement is evidence too, and much rarer to hear about than problems.
* **"It's off by about X %, in this direction."** Say how you measured, and when.
* **"It was right until..."** a break, a new power meter, a change in training.
* **"The plan is too hard / too easy"**: which sessions, and how they felt.
* **How much data GoodPush has**: months or years of rides, power meter or not, heart-rate strap or wrist, Garmin
  sleep / HRV or not. The model needs some history; telling us what you have helps us see where it struggles.

Everything the model finds is about one person's data and observational: hypotheses, not proof. Your comparisons
are how we find out which parts hold up across riders.

## Ideas and questions

[Open an idea or question](https://github.com/GoodPush/GoodPush/issues/new?template=idea.yml). Say what you'd like to do
and why, rather than how it should look: "I want to see whether my long rides are getting easier" is more useful
than "add a chart". GoodPush is kept deliberately simple, so not every idea gets built; every one is read.

## Wording and docs

If a sentence in the app or [the guide](https://goodpush.github.io/GoodPush/) confused you, that's a bug. Either:

* open an [idea or question](https://github.com/GoodPush/GoodPush/issues/new?template=idea.yml) quoting the sentence
  and saying what you thought it meant; or
* fix it yourself on GitHub, no tools needed: open the file (the guide's pages are in
  [`srcdocs/`](https://github.com/GoodPush/GoodPush/tree/main/srcdocs), this file is `CONTRIBUTING.md`), click the
  pencil icon, edit, and choose **Propose changes**. GitHub walks you through the rest.

## Code

### Setup

```bash
git clone https://github.com/GoodPush/GoodPush.git
cd GoodPush
uv sync --extra dev                            # .venv with Python 3.14 (.python-version), versions from uv.lock
uv run pre-commit install                      # the checks every commit must pass
uv run pytest tests -q                         # ~5 minutes; the physio tests are most of it
```

[uv](https://docs.astral.sh/uv/) installs Python 3.14 itself if you don't have it. `uv run <command>` runs in the
project's `.venv` (or activate it: `source .venv/bin/activate`, Windows `.venv\Scripts\activate`). After changing
dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock`.

Intel Macs: pip has no recent JAX or numba wheels there, so use conda instead:
`conda env create -f environment.yml` (Windows without uv: `environment-windows.yml`), then `conda activate goodpush`.
GoodPush needs Python 3.12 or newer (garminconnect needs 3.12); 3.14 is what it's developed and built with.

Try your change in the app with `goodpush app`, on a **copy** of a workspace (`goodpush --root path/to/copy app`):
the app can start a Garmin sync when it opens, and some steps rewrite files in the workspace.

### The checks

`pre-commit` runs on every commit. Don't skip it (`--no-verify`); if a check is wrong, say so in the pull request.

* **privacy** (`tools/privacy_check.py`): no ride / health files, e-mail addresses, home-folder paths, GPS
  coordinates, tokens.
* **hygiene** (`tools/hygiene_check.py`): the code rules below that a tool can check.
* **ruff** (syntax errors, undefined names, unused imports; lines up to 120), **detect-secrets**, and file
  basics (line endings, trailing spaces, large files).

### How the code is written

* **For any rider.** No one's numbers or personal facts in code, comments, tests or docs.
* **Physiological parameters stay physiological**: LTHR, LT1 and the like are inputs or estimates, never free
  fitting knobs.
* **Vectorised numpy / scipy**, no Python loops over samples, rows or days; numba only where a loop can't be avoided.
  Small batches, not one huge array. Check speed before and after; results must stay the same.
* **Classes with composition**, type hints, domain names (`LoadModel`, not `Helper`). No nested functions and no
  module-level `_helpers`: put them on the class. Keep it simple; build what's needed now.
* **Reuse established packages** (scipy, tqdm...) rather than writing your own.
* **Every change comes with a test** in `tests/`. Run the tests you touched; the full suite before a pull request.
* **The physiological model** (`src/goodpush/physio/`): a change to its parameters needs the hand-written numba
  gradient changed too, and a test matching it to the JAX version. Read [the model pages](https://goodpush.github.io/GoodPush/model/overview.html)
  first, and open an issue to discuss model changes before writing them.
* **The app's front end** (`src/goodpush/app/static/js/`): plain ES modules, no build step. Named imports and
  exports only, no `window.` globals, no inline `onclick`. One `Page` subclass per page in `pages/`; shared
  pieces are classes (`TimeWindow`, `Segmented`, `Pref`, `Lazy`) - look for the one that does it before writing
  a new one. Styles: `styles/app.css` (Tailwind CSS + daisyUI), compiled into `static/app.css` with
  `python styles/build.py` (see `styles/README.md`).
* **Plain language** in the app: say what it means for the rider, briefly.

### Pull requests

1. For anything bigger than a small fix, open an issue first so we can agree on the approach.
2. Keep each pull request to one change, with a test, and say in its description what it changes and why.
3. Contributions are released under the project's licence, the [GNU AGPL v3.0 or later](https://github.com/GoodPush/GoodPush/blob/main/LICENSE)
   with the additional terms in [NOTICE](https://github.com/GoodPush/GoodPush/blob/main/NOTICE).

Reviews may take a while: this is a spare-time project. Thank you for your patience.

### The guide (GitHub Pages)

The guide's sources are in `srcdocs/` (Markdown and reStructuredText); the built site is committed to `docs/`,
which GitHub Pages serves. After changing `srcdocs/` (or the physio model's docstrings):

```bash
uv run --extra docs python -m sphinx -b html -d build/doctrees srcdocs docs
```

and commit `docs/` with the change. The guide is public: no personal numbers in it.

## Code of conduct

Be kind, assume good faith, and remember that people share their health here. See
[CODE_OF_CONDUCT.md](https://github.com/GoodPush/GoodPush/blob/main/CODE_OF_CONDUCT.md).
