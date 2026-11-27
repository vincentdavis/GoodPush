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
