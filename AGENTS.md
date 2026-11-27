# AGENTS.md

Notes for AI coding agents working in this repository. Human contributors: see `CONTRIBUTING.md`, which this file
follows and does not replace.

GoodPush is a personal cycling training analysis app: Strava, Garmin and other sources in; power profile, training
load, a physiological model and an adaptive plan out. It runs as a local web app in the browser, with a tray icon.

This repository is a fork of `GoodPush/GoodPush` (the upstream). Its own rules are under "Rules for this fork" below.

## Layout

| Path | What it is |
|---|---|
| `src/goodpush/` | the Python package (hatchling, version from git tags via hatch-vcs) |
| `src/goodpush/cli.py` | the `goodpush` command (argparse subcommands: `app`, `sync`, `update`, `plan`, `physio`, `ride`, ...) |
| `src/goodpush/launcher.py` | `goodpush-app`: starts the server, opens the browser, shows the tray icon (what the installers run) |
| `src/goodpush/app/server.py` | the app's backend: a stdlib `http.server.ThreadingHTTPServer` with one handler (`make_handler`) serving the JSON API and the static files; background work in `Jobs`, the Garmin auto-sync thread in `serve` |
| `src/goodpush/app/static/` | the front end: `index.html`, `app.css` (compiled from `styles/`), plain ES modules in `js/` (one `Page` subclass per page in `js/pages/`, calendar in `js/calendar/`), charts on ECharts |
| `src/goodpush/app/static/vendor/` | third-party files copied in unchanged (`echarts.esm.min.js`, licences). Don't edit; pre-commit skips this folder |
| `styles/` | the stylesheet source (`app.css`, Tailwind CSS v4 + daisyUI 5), `build.py` that compiles it, vendored daisyUI in `vendor/` |
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

```bash
uv sync --extra dev                      # .venv with Python 3.14 (.python-version), versions from uv.lock
uv run pre-commit install                # installs the pre-commit and commit-msg hooks
```

`uv run <command>` runs in the project's `.venv` (or activate it). After changing dependencies in
`pyproject.toml`, run `uv lock` and commit `uv.lock`. Python 3.12+ is required (`requires-python`); 3.14 is what
it's developed and built with. Intel Macs: use conda instead (`conda env create -f environment.yml`, Windows
without uv: `environment-windows.yml`), since pip has no recent JAX or numba wheels there. Extras are in
`pyproject.toml`: `garmin`, `physio`, `ride`, `sync`, `webview`, `all`, `dev`, `build`, `docs`.

Run the app:

```bash
uv run goodpush-app                      # server on 127.0.0.1:8765, opens the browser, tray icon
uv run goodpush app --no-browser         # just the server, same port (--port N, --lan for your phone)
uv run goodpush --root path/to/copy app  # use a COPY of a workspace: the app can sync and rewrites files
```

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
- **Front end**: plain ES modules, no JS build step (styles: see below). Named imports and exports only, no `import *`, no `window.x =`
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
  `python styles/build.py` and commit both files. No Node, no npm: `build.py` downloads Tailwind's standalone CLI.
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

## Rules for this fork

- **Don't build on GitHub.** GitHub Actions are turned off on this fork, so ignore `.github/workflows/build.yml`
  (it is upstream's) and don't add or rely on workflows or CI checks. Releases are built, signed and notarized
  locally on the maintainer's Mac.
- **Never force-push**, to any branch.
- **Upstream sync is a fast-forward** of `main` from `GoodPush/GoodPush` main. If it can't fast-forward (diverged
  history, conflicts, uncommitted changes), stop and ask instead of merging, rebasing or forcing. Don't push
  upstream's tags unless asked.

## Planned modernization

In this order, one PR each, each starting after the previous one is merged:

1. **Tailwind CSS v4 + daisyUI** restyle of the front end. Done: see "Styles" above.
2. **FastAPI** replacing the stdlib `http.server` backend in `app/server.py`, keeping the same URLs and JSON.
3. **APScheduler** replacing the auto-sync loop and `Jobs` in `app/server.py`.

Until steps 2 and 3 land, work with the current backend as described above.
