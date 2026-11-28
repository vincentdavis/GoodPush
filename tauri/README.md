# GoodPush Desktop (Tauri)

GoodPush in a window of its own instead of a browser tab. This folder is the only thing the desktop app adds to
the repository: the app itself (pages, server, analysis, Ride now) is the same code as the browser app, so it
keeps up with upstream by merging, with nothing to port.

## How it works

| Part | What it is |
|---|---|
| `src-tauri/` | the Tauri shell (Rust): the window, the tray icon, starting and stopping goodpush's server |
| `sidecar.py` | the server's entry point: `goodpush --tauri-serve PORT` (the installed app's server, no browser or tray); any other arguments are the command line |
| `spec_rewrite.py` | writes the server's PyInstaller spec from `packaging/goodpush.spec` with a few edits, so `packaging/` stays upstream's |
| `ui/index.html` | what the window shows while the server starts |
| `build_mac.sh` | builds, signs, notarizes and staples the macOS disk image, on your Mac |
| `sync.sh` | merges upstream `main`, then your own branch, into the desktop app's branch |

When it starts, the shell runs the server on the first free port from 8765 (another goodpush running, such as a
development one, keeps its port) and shows its page once it listens. Closing the window leaves goodpush running,
as closing the browser tab did: the Dock icon or the tray icon opens it again. Quit (the menu, the tray, or the
page's Quit) stops the server; if the shell goes another way, the server notices and stops too.

What a browser did for the page, the shell does: links to other sites and links that open a new tab go to your
browser, downloads (workout files, plan.ics) go into Downloads, and a dropped account export reaches the page's own
drop zone.

**Ride now** works as in the installed app: the server starts the ride window (the PySide6 overlay) as its own
process from the same bundle, here `GoodPush Desktop.app/Contents/Helpers/goodpush-server.app`. That bundle is
started by the desktop app, so macOS asks the desktop app for Bluetooth; `src-tauri/Info.plist` says why.

## Develop

Rust and Node, then once:

```bash
(cd packaging && uv run python make_icon.py)        # the icon, drawn (not committed)
cd tauri && npm ci && npx tauri icon ../packaging/icon.png -o src-tauri/icons
```

Run the shell on the checkout's goodpush (`uv run`, so a front-end change shows on a reload, Cmd+R):

```bash
cd tauri && npx tauri dev
```

It uses your real athlete folder, like `goodpush app` does: try changes on a copy (AGENTS.md).

## Build for macOS

```bash
tauri/build_mac.sh "Developer ID Application: NAME (TEAMID)" NOTARY_PROFILE
```

It writes `build/tauri/goodpush-desktop-VERSION-mac-arm64.dmg`. Without arguments it builds an ad-hoc signed app
(`build/tauri/GoodPush Desktop.app`) to try on this Mac only. The server is built with Python 3.12 in
`.venv-build`, as the installed app is (`GOODPUSH_BUILD_PYTHON` changes it). GitHub Actions doesn't build it.

The app is called GoodPush Desktop so that it doesn't replace the installed browser app (`goodpush.app`) in
Applications; both use the same athlete folders.

## Keep it up to date

On the desktop app's branch, with nothing uncommitted:

```bash
tauri/sync.sh                 # upstream main, then origin's VMD-local-dev
tauri/sync.sh other-branch    # upstream main, then origin's other-branch
```

It merges (never rebases or pushes), then checks that `packaging/goodpush.spec` still has what `spec_rewrite.py`
edits, and runs `tests/test_tauri_shell.py`. Build, try it, and push the branch.

### When a sync breaks the build

- `spec_rewrite.py --check` fails: upstream changed a line of `packaging/goodpush.spec` the desktop app edits.
  Change that `Edit` in `spec_rewrite.py` to the new text. Fixes marked `required=False` are skipped once upstream
  has them; delete them then.
- The server doesn't start: the window says so and names its log
  (`~/Library/Logs/io.github.goodpush.desktop/server.log`).
- A merge conflict in `tauri/`: nothing upstream touches `tauri/`, so it came from your own branch.
