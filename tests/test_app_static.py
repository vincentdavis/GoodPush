"""The app's front end (app/static/js, ES modules): every import names an export of an existing module, the
page loads the entry module, and the package ships every file."""
import re
from pathlib import Path

import goodpush.app.server as server

STATIC = Path(server.__file__).with_name("static")
JS = STATIC / "js"
IMPORT = re.compile(r'^import \{([^}]*)\} from "([^"]+)";', re.M)
EXPORT = re.compile(r"^export (?:async function|function|class|const|let) ([\w$]+)", re.M)


def modules() -> list[Path]:
    return sorted(JS.rglob("*.js"))


def vendored() -> dict[Path, set[str]]:
    """static/vendor: third-party ES modules (minified); their names from the closing export{a as name, ...}."""
    out = {}
    for p in (STATIC / "vendor").glob("*.js"):
        block = p.read_text(encoding="utf-8").rsplit("export{", 1)[-1]
        out[p.resolve()] = {part.split(" as ")[-1].strip(" };\n") for part in block.split(",")}
    return out


def test_every_import_names_an_export_of_an_existing_module():
    exports = {p.resolve(): set(EXPORT.findall(p.read_text(encoding="utf-8"))) for p in modules()} | vendored()
    problems = []
    for p in modules():
        for names, target in IMPORT.findall(p.read_text(encoding="utf-8")):
            t = (p.parent / target).resolve()
            if not target.startswith(".") or t not in exports:
                problems.append(f"{p.relative_to(JS)}: no module {target}")
                continue
            for n in (x.strip() for x in names.split(",") if x.strip()):
                if n not in exports[t]:
                    problems.append(f"{p.relative_to(JS)}: {target} doesn't export {n}")
    assert not problems, problems


def test_vendored_libraries_are_the_released_files():
    """static/vendor is skipped by the pre-commit hooks: its files must stay byte for byte what was released
    (ECharts 6.1.0, dist/echarts.esm.min.js of the npm package; update the hash together with the file)."""
    import hashlib
    js = (STATIC / "vendor" / "echarts.esm.min.js").read_bytes()
    released = "28515f26aa57f87eb1e4ed4eb446927eaf329b5f5e02b4bfd4b0264fea40d1da"  # pragma: allowlist secret
    assert hashlib.sha256(js).hexdigest() == released
    assert (STATIC / "vendor" / "echarts.LICENSE").is_file() and (STATIC / "vendor" / "echarts.NOTICE").is_file()


def test_page_loads_the_entry_module_and_no_single_script_is_left():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert '<script type="module" src="js/main.js"></script>' in html
    assert not (STATIC / "app.js").exists()
    pages = [p for p in (JS / "pages").glob("*.js") if p.name != "page.js"]
    assert pages and all(re.search(r"^class \w+Page extends Page \{", p.read_text(encoding="utf-8"), re.M) for p in pages)


def test_the_wheel_ships_every_front_end_file():
    """What hatchling would put in the wheel (no build): every static file, the template and sync's .js."""
    from hatchling.builders.wheel import WheelBuilder
    project = Path(server.__file__).parents[3]
    shipped = {f.distribution_path.replace("\\", "/") for f in WheelBuilder(str(project)).recurse_included_files()}
    package = STATIC.parents[1]                                # the goodpush package
    wanted = [p for p in [*STATIC.rglob("*"), *(package / "templates").glob("*"), *(package / "sync").glob("*.js")]
              if p.is_file() and "__pycache__" not in p.parts]
    missing = [p for p in wanted if "goodpush/" + p.relative_to(package).as_posix() not in shipped]
    assert wanted and not missing, missing


def test_send_to_strava_copies_the_title_before_strava_opens():
    """The title goes on the clipboard before anything is awaited: once Strava's page opens this page loses the
    focus (and the click's activation) the browser's clipboard asks for. It says where the ride came from."""
    ride = (JS / "pages" / "ride.js").read_text(encoding="utf-8")
    send = ride.split("  async sendToStrava() {")[1].split("\n  }\n")[0]
    assert send.index("navigator.clipboard.writeText(this.stravaTitle(") < send.index("await ")
    assert "return `GoodPush - ${" in ride.split("  stravaTitle(a) {")[1].split("\n  }\n")[0]


def test_charts_smooth_with_the_recursive_exponential_filter_not_a_box():
    util = (JS / "util.js").read_text(encoding="utf-8")
    assert "export function expSmooth(arr, sigma)" in util and "export const boxSigma" in util
    for path in modules():
        text = path.read_text(encoding="utf-8")
        assert "boxSmooth" not in text and "smoothC" not in text, path.name
    for name in ("pages/ride.js", "dips.js"):
        assert "expSmooth(" in (JS / name).read_text(encoding="utf-8"), name


def test_ride_power_starts_raw_with_a_two_minute_smoothing_slider():
    ride = (JS / "pages" / "ride.js").read_text(encoding="utf-8")
    assert "  rideSmooth = 0;" in ride
    assert '<input type="range" id="smooth" min="0" max="120"' in ride


def test_session_buttons_dont_share_an_attribute_with_the_workout_profile():
    """The profile's bars carry data-m (minutes): the session window's Mark done / Skipped listener must not
    pick them up (a click on an interval sent its minutes as the status)."""
    session = (JS / "calendar" / "session.js").read_text(encoding="utf-8")
    listened = set(re.findall(r'querySelectorAll\("\[(data-[\w-]+)\]"\)', session))
    widgets = (JS / "widgets.js").read_text(encoding="utf-8")
    profile = set(re.findall(r'(data-[\w-]+)=', widgets.split("export function profileSVG")[1].split("\nexport ")[0]))
    assert listened and profile and not listened & profile


def test_header_and_favicon_use_the_logo_files():
    """The header draws logo.svg's #logo (its letters take the page's text colour), the tab icon is the bike."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert '<link rel="icon" href="mark.svg" type="image/svg+xml">' in html
    assert '<use href="logo.svg#logo"/>' in html
    logo = (STATIC / "logo.svg").read_text(encoding="utf-8")
    assert 'id="logo"' in logo and 'fill="currentColor"' in logo
    assert (STATIC / "mark.svg").is_file()


def test_the_guide_opens_in_the_browser_from_the_header_help_and_model():
    """The guide (the built docs, published online) is one click away: the header, Help and Model link to it in a
    new tab; Help and Model point at the model's own pages."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    help_js = (JS / "pages" / "help.js").read_text(encoding="utf-8")
    guide = re.search(r'export const GUIDE = "([^"]+)"', help_js).group(1)
    assert f'<a class="guidelink" href="{guide}" target="_blank" rel="noopener"' in html
    model_js = (JS / "pages" / "model.js").read_text(encoding="utf-8")
    assert 'import {GUIDE} from "./help.js";' in model_js
    for js in (help_js, model_js):
        assert '${GUIDE}model/overview.html" target="_blank" rel="noopener"' in js
    assert (Path(server.__file__).parents[3] / "docs" / "model" / "overview.html").is_file()


def test_the_stylesheet_is_built_from_the_pinned_tailwind_and_daisyui():
    """app.css is compiled from styles/app.css (python styles/build.py) and committed. daisyUI is vendored in
    styles/vendor (its 5.8.3 release files, byte for byte); Tailwind's CLI is pinned in styles/build.py. The
    committed app.css must come from those versions."""
    import hashlib
    styles = Path(server.__file__).parents[3] / "styles"
    released = {"daisyui.mjs": "b1e758e7a719ca213ec51b2ebcc58678dd5b79a845b2853cb8a7426b53680762",  # pragma: allowlist secret
                "daisyui-theme.mjs": "e3dc6a029c5e98dadb1eb108aa7559dfc6dbe35b0334677093ce044f93e3197c"}  # pragma: allowlist secret
    for name, sha in released.items():
        assert hashlib.sha256((styles / "vendor" / name).read_bytes()).hexdigest() == sha, name
    assert 'VERSION = "4.3.5"' in (styles / "build.py").read_text(encoding="utf-8")
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert css.startswith("/*! tailwindcss v4.3.5 ") and "daisyUI 5.8.3" in css
    assert (STATIC / "vendor" / "tailwindcss.LICENSE").is_file() and (STATIC / "vendor" / "daisyui.LICENSE").is_file()


def test_questions_use_the_apps_dialog_not_the_browsers():
    """confirm() / prompt() / alert() stop the page and look like the browser, not the app: every question goes
    through Ask (controls.js), a <dialog> with a title and buttons that say what they do."""
    native = re.compile(r"(?<![\w.$])(?:window\.)?(?:confirm|prompt|alert)\(")
    method = re.compile(r"^  (?:async )?(?:confirm|prompt)\(\{.*\) \{$", re.M)        # Ask's own methods
    found = []
    for p in modules():
        code = re.sub(r"//.*$", "", method.sub("", p.read_text(encoding="utf-8")), flags=re.M)
        found += [f"{p.relative_to(JS)}: {m.group()}" for m in native.finditer(code)]
    assert not found, found
    controls = (JS / "controls.js").read_text(encoding="utf-8")
    assert "export const ask = new Ask();" in controls and "d.showModal();" in controls
    css = (STATIC / "app.css").read_text(encoding="utf-8")                       # its styles built in
    assert "dialog.ask{" in css and "dialog.ask::backdrop{" in css


def test_the_sync_icon_is_two_arrows_not_a_cloud():
    """goodpush keeps everything on your computer: nothing in the app or the guide may look like a cloud (2026-11-27,
    the Today redesign's sync icon)."""
    status = (JS / "status_bar.js").read_text(encoding="utf-8")
    assert "export const syncIcon" in status and "const ARROWS" in status
    code = "\n".join(re.sub(r"//.*$", "", p.read_text(encoding="utf-8"), flags=re.M) for p in modules())
    words = re.findall(r"\bcloud\w*", re.sub(r"console\.cloud\.google\.com", "", code), flags=re.I)
    assert not words, words
    guide = Path(__file__).parents[1] / "srcdocs"
    said = [p.name for p in guide.rglob("*.md") if re.search(r"cloud icon", p.read_text(encoding="utf-8"), re.I)]
    assert not said, said


def test_the_status_bar_sync_page_and_setup_checklist_are_wired():
    """the top bar's sync icon and Setup pill (status_bar.js), the Sync page (#sync) and the links into Settings'
    sections."""
    main = (JS / "main.js").read_text(encoding="utf-8")
    assert "sync: syncPage" in main and 'id="statusbar"' in (STATIC / "index.html").read_text(encoding="utf-8")
    status = (JS / "status_bar.js").read_text(encoding="utf-8")
    keys = re.findall(r'\{key: "(\w+)", title:', status)
    assert keys == ["garmin", "history", "model", "rideapp"]
    settings = (JS / "pages" / "settings.js").read_text(encoding="utf-8")
    from goodpush.config import SETTINGS                                       # settingsCard: one section each
    sections = set(re.findall(r'this\.section\("(\w+)"', settings)) | set(SETTINGS)
    links = set(re.findall(r"#settings/(\w+)", "".join(p.read_text(encoding="utf-8") for p in modules())))
    assert links and links <= sections, links - sections


def test_today_steps_through_days_without_module_state():
    """#today/YYYY-MM-DD shows that day; the arrow keys and a swipe step through days (DaySteps: its state on the
    class, not a module-level variable)."""
    today = (JS / "pages" / "today.js").read_text(encoding="utf-8")
    assert "router.routeArg()" in today and "ISO_DAY.test(a)" in today
    assert "class DaySteps" in today and "new DaySteps().listen();" in today
    assert not re.findall(r"^(?:let|var) ", today, re.M)


def test_today_shows_its_why_and_steps_open_and_no_next_line():
    """2026-11-27: the verdict's Why starts open, the session's steps are shown (no fold), and the links to the next
    session and the week plan at the bottom are gone."""
    today = (JS / "pages" / "today.js").read_text(encoding="utf-8")
    assert 'id="todaywhy" open>' in today
    assert 'id="todaysteps"' in today and '<details class="dayfold" id="todaysteps"' not in today
    assert "nextline" not in today and "Week plan" not in today


# colour-vision simulation (Machado, Oliveira & Fernandes 2009, full severity) and OKLab (Ottosson 2020): published
# colour-science constants (the privacy check reads pairs of decimals as coordinates)
CVD = {"protan": [[0.152286, 1.052583, -0.204868],      # privacy: ok
                  [0.114503, 0.786281, 0.099216],       # privacy: ok
                  [-0.003882, -0.048116, 1.051998]],    # privacy: ok
       "deutan": [[0.367322, 0.860646, -0.227968],      # privacy: ok
                  [0.280085, 0.672501, 0.047413],       # privacy: ok
                  [-0.011820, 0.042940, 0.968881]],     # privacy: ok
       "tritan": [[1.255528, -0.076749, -0.178779],     # privacy: ok
                  [-0.078411, 0.930809, 0.147602],      # privacy: ok
                  [0.004733, 0.691367, 0.303900]]}      # privacy: ok
OKLAB_LMS = [[0.4122214708, 0.5363325363, 0.0514459929],     # privacy: ok
             [0.2119034982, 0.6806995451, 0.1073969566],     # privacy: ok
             [0.0883024619, 0.2817188376, 0.6299787005]]     # privacy: ok
OKLAB_LAB = [[0.2104542553, 0.7936177850, -0.0040720468],    # privacy: ok
             [1.9779984951, -2.4285922050, 0.4505937099],    # privacy: ok
             [0.0259040371, 0.7827717662, -0.8086757660]]    # privacy: ok


def oklab(hex_colour: str, sim=None):
    """a colour (#rrggbb) in OKLab, as seen with the colour-vision simulation `sim` (Machado 2009, full severity)."""
    import numpy as np
    c = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    rgb = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    rgb = np.clip(np.asarray(sim) @ rgb, 0, 1) if sim is not None else rgb
    return np.asarray(OKLAB_LAB) @ np.cbrt(np.asarray(OKLAB_LMS) @ rgb)


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two colours (#rrggbb)."""
    import numpy as np
    lum = []
    for h in (a, b):
        c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
        lum.append(float(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) @ [0.2126, 0.7152, 0.0722]))
    return (max(lum) + 0.05) / (min(lum) + 0.05)


def test_red_amber_green_is_colour_blind_safe_everywhere():
    """every red / amber / green in the app is --good / --warn / --bad (verdicts, chips, the morning ovals, the sync
    icon and its lights; 2026-11-27). In each theme every pair stays apart for normal vision (OKLab Delta E x100 >= 15)
    and for protan, deutan and tritan vision (>= 8), and as text each keeps 4.5:1 on its own tint (--good-bg ...).
    The HTML report uses the same colours, and each morning oval carries its verdict in its label too."""
    import itertools
    import numpy as np
    css = (Path(__file__).parents[1] / "styles" / "app.css").read_text(encoding="utf-8")
    inks = re.findall(r"--good: (#\w{6}); --warn: (#\w{6}); --bad: (#\w{6});", css)
    tints = re.findall(r"--good-bg: (#\w{6}); --warn-bg: (#\w{6}); --bad-bg: (#\w{6});", css)
    assert len(inks) == len(tints) == 2
    for cols, bgs in zip(inks, tints):
        for a, b in itertools.combinations(cols, 2):
            assert 100 * np.linalg.norm(oklab(a) - oklab(b)) >= 15, (a, b)
            for name, sim in CVD.items():
                assert 100 * np.linalg.norm(oklab(a, sim) - oklab(b, sim)) >= 8, (name, a, b)
        assert all(contrast(ink, bg) >= 4.5 for ink, bg in zip(cols, bgs)), (cols, bgs)
    report = (Path(__file__).parents[1] / "src" / "goodpush" / "report" / "dashboard.py").read_text(encoding="utf-8")
    assert set(re.findall(r"--good:(#\w{6}); --bad:(#\w{6});", report)) == {(c[0], c[2]) for c in inks}
    assert "--mgreen" not in css and ".mdot7.green { background:var(--good) }" in css
    widgets = (JS / "widgets.js").read_text(encoding="utf-8")
    assert 'class="mdot7 ${esc(m.status)}" role="img" title="${esc(t)}" aria-label="${esc(t)}"' in widgets


def test_the_physio_lines_take_their_zone_colours():
    """2026-11-27: the model's LT1 / CP / FTP / MAP / W′ / Pmax lines (Rider and Model pages) and their legends are
    coloured where their intensity lands on the zone colour map (util.js INTENSITY_COL), not the series colours; each
    stays visible as a line (3:1 on the chart surface) in both themes."""
    util = (JS / "util.js").read_text(encoding="utf-8")
    col = dict(re.findall(r'(\w+): "(--z\d)"', util.split("export const INTENSITY_COL = ", 1)[1].split("};", 1)[0]))
    assert col == {"lt1": "--z3", "cp": "--z4", "ftp": "--z4", "map": "--z5", "w_prime": "--z6", "pmax": "--z7"}
    model = (JS / "pages" / "model.js").read_text(encoding="utf-8")
    rider = (JS / "pages" / "rider.js").read_text(encoding="utf-8")
    for line in (model.split("const MODEL_Q", 1)[1].split("};", 1)[0], model.split("const READ_LINE", 1)[1].split("};", 1)[0],
                 rider.split("const PHYSIO_LINES", 1)[1].split("\n", 1)[0]):
        assert "INTENSITY_COL" in line and not re.search(r'"--s\d"', line), line
    assert "lines = PHYSIO_LINES" in rider.split("chartPhysio(", 1)[1].split("const series", 1)[0]   # the chart ...
    legend = next(x for x in rider.splitlines() if '<div class="legend">${PHYSIO_LINES.map(' in x)       # ... its legend
    assert "INTENSITY_COL.cp" in legend and "--s" not in legend
    assert "background:var(${qCol})" in model                                   # the readings chart's legend
    css = (Path(__file__).parents[1] / "styles" / "app.css").read_text(encoding="utf-8")
    surfaces = re.findall(r"--surface: (#\w{6});", css)
    zones = [dict(re.findall(r"--(z\d): (#\w{6})", z)) for z in re.findall(r"(--z1: #\w{6};.*)", css)]
    assert len(surfaces) == len(zones) == 2
    for surface, z in zip(surfaces, zones):
        assert all(contrast(z[c[2:]], surface) >= 3 for c in col.values()), (surface, z)


def test_the_model_page_answers_can_i_trust_it_at_the_top():
    """2026-11-27 Model page critique (reviews/model-page/README.md): one ranges verdict opens the page (no Ranges tile
    beside it to disagree with), predicted vs observed has titled axes over one range, the readings chart can take
    each source's bias out, chart kinds have shapes as well as colours, and the charts have a text alternative."""
    model = (JS / "pages" / "model.js").read_text(encoding="utf-8")
    charts = (JS / "charts.js").read_text(encoding="utf-8")
    assert 'tile("Ranges"' not in model and "rangesVerdict()" in model
    assert all(f'["Ranges: {w}' in model for w in ("none yet", "rough", "trustworthy"))
    effort = model.split('EChart.draw("mdEffort"', 1)[1].split("series:", 1)[0]
    assert all(k in effort for k in ("xDomain: [lo, hi]", "yDomain: [lo, hi]", "xName:", "yName:", "tick", "label:"))
    assert "yDomain" in charts.split("yRange(view) {", 1)[1].split("}", 1)[0]
    assert 'id="mdReadFair"' in model and "bias[n]" in model and "READ_NAME" in model
    assert all(f'"{s}"]' in model for s in ("diamond", "triangle")) and "symbol: s.shape" in charts
    assert 'box.setAttribute("role", "img")' in charts
    assert all(f'EChart.draw("{c}"' in model for c in ("mdEffort", "mdRead", "mdFatigue"))
    assert model.count("label: `") >= 3
    assert "j.stage || j.other" in model                         # keeps polling while a run started elsewhere goes on


def test_the_model_page_shows_todays_best_fit_before_any_ranges():
    """2026-11-28 Model page review: before a sampling run the verdict says today's numbers come from the best fit,
    so the page shows them (W' in kJ, as on the Rider page); the fatigue legend names "Planned" only when the plan's
    forecast draws that dashed line; the riding table's hint sits in its heading, not alone in the card."""
    model = (JS / "pages" / "model.js").read_text(encoding="utf-8")
    assert "if (!rows) return this.bestFitSection();" in model
    best = model.split("bestFitSection() {", 1)[1].split("\n  }\n", 1)[0]
    assert "Today, from the best fit" in best and "after a sampling run" in best and "/ 1000" in best
    fatigue = model.split("fatigueSection() {", 1)[1].split("\n  }\n", 1)[0]
    assert "this.fatiguePlanned() ?" in fatigue
    riding = model.split("ridingSection() {", 1)[1].split("\n  }\n", 1)[0]
    assert '<div class="card">${hint(' not in riding


def test_the_model_page_says_where_you_are_heading_and_what_the_fit_used():
    """2026-11-28 Model page ideas: today's numbers with their 6-week change and what the latest rides moved (a big
    move flagged), what the fit was given (days, rides, best powers, readings, rides set aside), the predicted vs
    observed chart in one sentence, and how fresh the LT1 / CP readings are, with a ramp test named when they're old."""
    model = (JS / "pages" / "model.js").read_text(encoding="utf-8")
    assert model.count("this.trendCell(k)") == 2 and model.count("${this.movedLine()}") == 2
    assert "const TREND_DAYS = 42;" in model and "const MOVED_BIG = 0.03;" in model and "const FRESH_DAYS = 42;" in model
    assert "${this.inputsCard()}" in model and "I.set_aside" in model
    assert 'id="mdEffortSay"' in model and '$("#mdEffortSay").textContent = this.effortSay(pick);' in model
    assert "${this.readingsFresh()}" in model and "a ramp test would add a fresh one" in model
    server = (Path(__file__).parents[1] / "src" / "goodpush" / "app" / "server.py").read_text(encoding="utf-8")
    assert "PhysioDiagnostics.set_aside(" in server


def test_the_intensity_colours_keep_their_order_in_both_themes():
    """--z1 (easy) is the yellow end and --z7 (hard) the blue end in light and dark alike (the dark theme reversed
    the scale before 2026-11-27); text on a zone colour (--on-z*) stays readable on it."""
    css = (Path(__file__).parents[1] / "styles" / "app.css").read_text(encoding="utf-8")
    zones = [dict(re.findall(r"--(z\d): (#\w{6})", z)) for z in re.findall(r"(--z1: #\w{6};.*)", css)]
    on = [dict(re.findall(r"--on-(z\d): (#\w{6})", z)) for z in re.findall(r"(--on-z1: #\w{6};.*)", css)]
    assert len(zones) == len(on) == 2
    for z, t in zip(zones, on):
        r, g, b = (int(z["z1"][i:i + 2], 16) for i in (1, 3, 5))
        assert r > b + 60 and g > b + 60                         # z1 yellow
        r, g, b = (int(z["z7"][i:i + 2], 16) for i in (1, 3, 5))
        assert b > r + 40                                        # z7 blue
        assert all(contrast(t[k], z[k]) >= 4.5 for k in t), (z, t)
