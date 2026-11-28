# AGENTS.md

Notes for AI coding agents working in this repository. Human contributors: see `CONTRIBUTING.md`, which this file
follows and does not replace.

GoodPush is a personal cycling training analysis app: Strava, Garmin and other sources in; power profile, training
load, a physiological model and an adaptive plan out. It runs as a local web app in the browser, with a tray icon.

## Layout

| Path | What it is |
|---|---|
| `src/goodpush/` | the Python package (hatchling, version from git tags via hatch-vcs) |
| `src/goodpush/cli.py` | the `goodpush` command (argparse subcommands: `app`, `sync`, `update`, `plan`, `physio`, `ride`, ...) |
| `src/goodpush/launcher.py` | `goodpush-app`: starts the server, opens the browser, shows the tray icon (what the installers run) |
| `src/goodpush/app/server.py` | the app's backend: a stdlib `http.server.ThreadingHTTPServer` with one handler (`make_handler`) serving the JSON API and the static files; background work in `Jobs`, the Garmin auto-sync thread in `serve` |
| `src/goodpush/app/static/` | the front end: `index.html`, `app.css` (compiled from `styles/app.css`: never edit it), plain ES modules in `js/` (one `Page` subclass per page in `js/pages/`, calendar in `js/calendar/`), charts on ECharts |
| `styles/` | the stylesheet's source (Tailwind CSS v4 + daisyUI 5), `build.py`, vendored daisyUI: see "Styles" below |
| `src/goodpush/app/static/vendor/` | third-party files copied in unchanged (`echarts.esm.min.js` and its licences). Don't edit; pre-commit skips this folder |
| `src/goodpush/sync/` | downloads from each source: `garmin.py`, `garmin_api.py`, `garmin_browser.js`, `strava.py`, `google_health.py`, `intervals_icu.py`, `incyclist.py`, and `peer.py` (your folder on another machine) |
| `src/goodpush/io/` | file formats and stores (FIT, streams in HDF5, Strava / Garmin / intervals.icu readers, ZWO) |
| `src/goodpush/metrics/`, `models/`, `physio/`, `planning/` | the analysis: metrics, models, the physiological model (JAX + a hand-written numba gradient), the planner |
| `src/goodpush/ride/` | riding on the trainer: ANT+ / Bluetooth sensors, recorder, the PySide6 overlay |
| `src/goodpush/report/`, `checks/`, `synth/` | reports, back-tests and coverage checks, synthetic riders with a known physiology |
| `src/goodpush/templates/goodpush.toml` | the config template a new workspace starts from |
| `tests/` | pytest suite (`testpaths = ["tests"]`) |
| `tools/` | `privacy_check.py` and `hygiene_check.py`, run by pre-commit |
| `packaging/` | PyInstaller specs (`goodpush.spec`), the icon, third-party notices, the Windows (Inno Setup) and Linux installers, `RELEASE_NOTES.md` |
| `srcdocs/` → `docs/` | the guide's sources (Sphinx, MyST) and the built site that GitHub Pages serves. `docs/` is generated |

## Set up, run, test

Use **uv** with **Python 3.14**. `.python-version` pins 3.14 (uv downloads it if missing), `uv.lock` pins every
dependency, and `uv sync` / `uv run` keep an editable install of the checkout in `.venv` (git-ignored).

```bash
uv sync --extra dev                      # .venv with goodpush (editable) + all extras + test / commit tools
uv run pre-commit install                # installs the pre-commit and commit-msg hooks
```

- Prefix commands with `uv run` (or `source .venv/bin/activate` first). Don't use `pip install` in `.venv`.
- Extras are in `pyproject.toml`: `garmin`, `physio`, `ride`, `sync`, `webview`, `all`, `dev`, `build`, `docs`.
  Add one with `uv sync --extra NAME` (`uv sync` removes extras you don't name: list every one you need).
- Changing dependencies: edit `pyproject.toml`, run `uv lock`, commit `uv.lock` with it.
- `requires-python` is `>=3.12` (garminconnect needs 3.12; Intel Macs need conda-forge's builds, which stop
  short of 3.14: `conda env create -f environment.yml`). Don't raise it to 3.14 without checking Intel Macs.

Run the app (the dev server: the same server the installed app runs, from the checkout):

```bash
uv run goodpush app --no-browser         # server on 127.0.0.1:8765 (--port N, --lan for your phone)
uv run goodpush-app                      # as installed: opens the browser, tray icon
uv run goodpush --root path/to/copy app  # use a COPY of a workspace: the app can sync and rewrites files
```

The server sends everything with `Cache-Control: no-store` and reads `static/` from the checkout, so a front-end
change shows on a plain browser reload; Python changes need the server restarted. Styles are compiled (Tailwind +
daisyUI): run `uv run python styles/build.py --watch` alongside while editing them (see `styles/README.md`).

If 8765 is taken by another goodpush, `goodpush-app` tries the next ports.

Test:

```bash
uv run pytest tests -q                   # full suite, about 5 minutes; the physio tests are most of it
uv run pytest tests/test_app_static.py -q   # run the files you touched while working
uv run pre-commit run --all-files        # every commit check
```

Rebuild the guide after changing `srcdocs/` or the physio model's docstrings, and commit `docs/` with it:

```bash
uv run --extra docs python -m sphinx -b html -d build/doctrees srcdocs docs
```

A Windows build writes backslashes in `docs/searchindex.js`'s file names: put forward slashes back.

Build the installed app (PyInstaller; on macOS `dist/goodpush.app`). uv runs the tools; it doesn't make the
app itself (`uv build` only makes the Python wheel / sdist):

```bash
uv sync --extra build --extra dev        # PyInstaller + Pillow; name dev too or uv sync drops it
cd packaging && uv run python make_icon.py && uv run python notices.py && uv run pyinstaller goodpush.spec --noconfirm
```

## Conventions

From `CONTRIBUTING.md` and the checks; `tools/hygiene_check.py` enforces some of them on the lines a commit adds.

- **No personal data anywhere**: no one's numbers, names, e-mail addresses, home-folder paths, GPS coordinates,
  tokens or ride files in code, comments, tests, docs or commit messages. `tools/privacy_check.py` checks this.
  Never commit a `goodpush.toml`, `data/` or `reports/` folder.
- **Never skip the checks** (`--no-verify`). If a check is wrong, say so in the PR.
- **Python**: type hints, classes with composition and domain names (`LoadModel`, not `Helper`). No nested
  functions and no new module-level `_helpers`: put them on the class (older code has some; don't add more).
  Lines up to 120. Ruff runs with `E9` and `F` only.
- **Vectorised numpy / scipy**, no Python loops over samples, rows or days; numba only where a loop can't be
  avoided. Reuse established packages over writing your own.
- **Physiological parameters stay physiological**: LTHR, LT1 and the like are inputs or estimates, never free
  fitting knobs. A change to the physio model's parameters needs the numba gradient changed too, plus a test
  matching it to the JAX version.
- **Front end**: plain ES modules, no build step. Named imports and exports only, no `import *`, no `window.x =`
  globals, no inline `onclick=` (add listeners in `wire()`). Reuse the shared classes (`TimeWindow`,
  `Segmented`, `Pref`, `Lazy`) before writing new ones.
- **Every change comes with a test** in `tests/`.
- **Plain language** in the app's text: what it means for the rider, briefly.
- The PR template is `.github/pull_request_template.md`: fill in what changed, why, and how it was tested.

## Styles: Tailwind CSS + daisyUI

The app's stylesheet is Tailwind CSS v4 with daisyUI 5. Details: `styles/README.md`.

* Edit `styles/app.css`, never `src/goodpush/app/static/app.css` (compiled output, committed so the app, pip
  installs and `packaging/` need no build step).
* After changing `styles/app.css`, or adding a Tailwind / daisyUI class to `static/index.html` or `static/js/`, run
  `uv run python styles/build.py` and commit both files. No Node, no npm: `build.py` downloads Tailwind's standalone CLI.
* Versions are pinned: Tailwind CLI 4.3.5 in `styles/build.py` (with release SHA-256s), daisyUI 5.8.3 vendored in
  `styles/vendor/`. Don't upgrade them as a side effect; an upgrade updates the hashes in
  `tests/test_app_static.py` and the versions in `packaging/notices.py` too.
* Themes are the daisyUI themes `light` and `dark` (`data-theme` on `<html>`; none follows the system). Colours the
  app and the charts use (`--ink`, `--muted`, `--s1`...) are defined in those theme blocks, in hex (ECharts reads
  them and can't parse `oklch`). Keep text colours at WCAG AA (4.5:1) in both themes.
* Tailwind's preflight is off, and daisyUI's `card`, `loading`, `stack`, `status`, `steps`, `toast`, `tab` and
  `modal` are excluded because the app uses those class names itself. `.btn` is daisyUI's button.
* Keep the global `:focus-visible` ring; don't add `outline:none`.
* Check a style change in light and dark mode at laptop and phone widths (the phone bottom bar has Today, Calendar,
  Fitness, Plan and More).
* No GitHub Actions builds: the compiled CSS is built locally and committed.

## Scientific and backend code

The analysis, the physiological model and the planner. Read the module's docstrings and `srcdocs/model/` before
changing one; the reasons for most choices are written there.

- **Rider-agnostic.** No one's numbers or facts in code, comments, tests or docs: rates, priors and thresholds are
  the codebase's constants for every rider, and a rider's data only tests them. Tests use synthetic riders
  (`goodpush.synth`) or made-up tables.
- **Results are n = 1 and observational.** Show correlations as hypotheses, never as effects. Docs and the app's
  text say what a number is and how sure it is.
- **New analyses go in the package with a test**, not in one-off scripts. Experiments run on a copy of a
  workspace, never on a real one (fits and checks write into `data/processed/`).
- **Vectorise.** Prefix sums, `np.bincount` / `np.lexsort`, `scipy.signal.lfilter` instead of loops over samples,
  rows or days. A loop that can't be avoided goes in a numba kernel. Profile before and after; results stay
  identical. Smoothing uses the recursive exponential filter (Dave Hale's; `expSmooth` in `util.js`), not box or
  median windows.
- **Low memory, small batches.** No large one-shot batch operations (vmap over hundreds of draws, every ride's
  streams at once): fixed small batches (`PhysioModel.batched`, `BATCH` constants). Long runs name each stage in
  their log and report progress.
- **Our own numba kernels for the heavy compute; JAX is the test reference.** The posterior's value and gradient
  are `physio/numba_posterior.py` (a hand-written adjoint). A change to the model's parameters or likelihood
  changes both, with a test matching numba to JAX.
- **The physio model's version.** Bump `VERSION` in `physio/model.py` when the model changes what its parameters
  mean. Stored fits of another version are refitted, and their daily states, and anything worked out from them,
  are set aside (`PhysioRun.reset`), so a new version never builds on an old fit. A process running another
  version than the code on disk never refits (`PhysioRun.same_code`). Rebuild `docs/` when the model changes.
- **Don't start a full NUTS run** (`PhysioRun.FULL_RUN`, hours of every core): hand the command to the person.
  Posterior-mode fits (`PhysioFit.map`, seconds) are fine. Don't run tests or benchmarks while a fit runs.
- **Physiological parameters stay physiological** (see Conventions), and inputs stay measurements: threshold
  heart rate, FTP entries and the like go in as data or estimates with their own error, never as tuning knobs.
- **Data the model can't trust is set aside before it** (`metrics/power_check.py`), and the likelihood keeps a
  small share for bad readings: a single impossible value must not move a whole fit.
- **Files:** read and write text as UTF-8 explicitly (the installed app runs Python without UTF-8 mode). Replace
  files whole (`PhysioRun.write_whole`): the app may read them while another process writes.
- KISS / YAGNI: reuse established packages (scipy, tqdm, ...), and ask before adding a new public name or setting.

## Planned modernization of the app (front end and its server)

For work on the app, not the analysis. One PR each, each starting after the previous one is merged:

1. **Tailwind CSS v4 + daisyUI** restyle of the front end (done: see "Styles").
2. **FastAPI** replacing the stdlib `http.server` backend in `app/server.py`, keeping the same URLs and JSON.

Not decided: what replaces the auto-sync loop and `Jobs` in `app/server.py` (APScheduler was one idea). Don't
start it without asking.

Until each lands, work with the current code as described above.
