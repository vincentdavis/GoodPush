# The app's stylesheet (Tailwind CSS v4 + daisyUI 5)

`styles/app.css` is the source; `python styles/build.py` compiles it into `src/goodpush/app/static/app.css`,
which is committed. The app, pip installs and the Mac / Windows packaging (`packaging/`) use the committed file and
never run a build. Node isn't needed.

```
python styles/build.py           # build once (minified): do this before committing a style change
python styles/build.py --watch   # rebuild on every change while you edit
```

Rebuild whenever you change `styles/app.css`, or add a Tailwind / daisyUI class to `index.html` or `js/`: Tailwind
only writes the classes it finds in those files.

## Pinned versions

| What | Version | Where |
|---|---|---|
| Tailwind CSS standalone CLI | 4.3.5 | `build.py` (`TailwindBuild.VERSION`, SHA-256 per platform); downloaded once into `styles/.bin/` (git-ignored) |
| daisyUI | 5.8.3 | `vendor/daisyui.mjs`, `vendor/daisyui-theme.mjs`: the release files, unchanged (`tests/test_app_static.py` checks their hashes) |

The Tailwind CLI is a ~100 MB binary per platform, so it isn't committed; `build.py` fetches the pinned release and
refuses it if its checksum differs. To upgrade either one: replace the files / version and checksums, rebuild,
update the hashes and banner check in `tests/test_app_static.py` and the versions in `packaging/notices.py`,
and check every page in light and dark at laptop and phone widths.

## How it fits together

* **Themes**: two daisyUI themes named `light` and `dark` in `app.css`. `<html data-theme="light|dark">` picks one
  (the top bar's theme button, `ThemeSwitch` in `js/main.js`); no attribute follows the system.
* **The app's colour names** (`--bg`, `--surface`, `--ink`, `--ink2`, `--muted`, `--line`, `--accent`, `--s1`...,
  `--good`, `--warn`, `--bad`, `--z1`...) are defined inside the same theme blocks, in hex. The charts read them
  with `css("--ink")` (`js/util.js`) when they draw, so ECharts follows the theme; ECharts can't parse `oklch`,
  so keep them hex.
* **Contrast**: text colours (`--muted`, `--warn`, `--good`, `--bad`, `--accent`) are at least 4.5:1 (WCAG AA) on the
  page and on cards in both themes. Check a new one before adding it.
* **No preflight**: Tailwind's reset isn't loaded, so the browser's base styles (paragraph margins, list bullets)
  stay as the pages expect.
* **daisyUI components the app doesn't take**: `card`, `loading`, `stack`, `status`, `steps`, `toast`, `tab`,
  `modal` are excluded because the app has its own classes with those names. `.btn` is daisyUI's button; the app's
  `.btn.primary`, `.btn.sm`, `.btn.danger`, `.btn.ghost`, `.btn.circle` set its variables.
* **Focus**: one `:focus-visible` ring (2 px, `--accent`) for everything keyboard-reachable. Don't remove outlines.
* The rest of `app.css` is the app's own component CSS (cards, tiles, calendar, charts...), written with the theme
  variables. New UI can use Tailwind utilities and daisyUI classes directly in the markup.
