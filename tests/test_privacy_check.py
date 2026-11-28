import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("privacy_check", ROOT / "tools" / "privacy_check.py")
privacy_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(privacy_check)
PrivacyCheck = privacy_check.PrivacyCheck


def check(text, name="x.py", terms=None, tmp=None):
    terms_file = None
    if terms is not None:
        terms_file = tmp / "terms.txt"
        terms_file.write_text("# my private words\n" + "\n".join(terms))
    c = PrivacyCheck(terms_file or ROOT / "no_such_file")
    c.check_text(name, text)
    return c.problems


def test_personal_data_is_caught():
    assert check("contact jane.doe@gmail.com")
    assert check('trainer = "ant:fec:43210"')
    assert check(r"path C:\Users\jdoe\rides")
    assert check("/home/jdoe/data")
    assert check(r"F:\codes\goodpush")
    assert check("start 47.123456, -122.654321")
    assert check("id 123456789-abcdefghijklmnopqrstuvwxyz012345.apps.googleusercontent.com")
    assert check('refresh_token = "a1b2c3d4e5f6g7h8i9j0"')
    assert check('trainer = "x"', name="goodpush.toml")
    assert check(r"cache in C:\ProgramData\goodpush")              # the system drive too (2026-11-13)
    assert check('password = "hunter22"') and check("{'pwd': 'abcd1234'}")  # pragma: allowlist secret


def test_ordinary_code_passes():
    assert not check('password = ""  # asked at sign-in') and not check("password=os.environ['PW']")
    assert not check("Co-Authored-By: Claude <noreply@anthropic.com>")
    assert not check('Device.from_key("ant:fec:12345")  hub = {"trainer": "sim:fec:1"}')
    assert not check(r"C:\Users\you\.goodpush and ~/.goodpush, https://www.strava.com/api/v3")
    assert not check("power = 250  # watts, 1.5 h at 0.75 FTP")
    assert not check('trainer = "ant:fec:43210"  # privacy: ok')


def test_own_terms_and_blocked_files(tmp_path):
    assert check("ride with athlete_name today", terms=["athlete_name"], tmp=tmp_path)
    assert not check("ride alone", terms=["athlete_name"], tmp=tmp_path)
    c = PrivacyCheck(tmp_path / "none")
    for p in ("goodpush.toml", "data/raw/strava/activities.csv", "rides/a.fit", "notes/x.md", ".env",
              "x/ride_devices.json"):
        assert not c.check_path(p), p
    assert c.check_path("src/goodpush/templates/goodpush.toml") and c.check_path("README.md")


def test_code_names_no_rider(tmp_path):
    terms = ["code: athlete_name"]
    assert check("# the cap athlete_name liked", name="src/goodpush/plan.py", terms=terms, tmp=tmp_path)
    assert check('"""athlete_name\'s data: 6 W"""', name="tests/test_x.py", terms=terms, tmp=tmp_path)
    assert not check('authors = [{ name = "athlete_name" }]', name="pyproject.toml", terms=terms, tmp=tmp_path)
    assert not check("# the cap a rider likes", name="src/goodpush/plan.py", terms=terms, tmp=tmp_path)


def test_binary_files_are_refused_except_a_design_reviews_screenshots(tmp_path, monkeypatch):
    """2026-11-28: a design review (reviews/<page>/README.md) keeps its screenshots of a synthetic rider beside it, in
    reviews/<page>/img/; any other binary file, an image elsewhere included, is still refused."""
    monkeypatch.chdir(tmp_path)
    jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 64
    for rel in ("reviews/model-page/img/after.jpg", "reviews/model-page/after.jpg", "docs/shot.png",
                "reviews/model-page/img/sub/after.jpg", "reviews/model-page/img/data.bin"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(jpg)
    c = PrivacyCheck(ROOT / "no_such_file")
    for rel in ("reviews/model-page/img/after.jpg", "reviews/model-page/after.jpg", "docs/shot.png",
                "reviews/model-page/img/sub/after.jpg", "reviews/model-page/img/data.bin"):
        c.check_file(rel)
    assert [p.split(":")[0] for p in c.problems] == ["reviews/model-page/after.jpg", "docs/shot.png",
                                                     "reviews/model-page/img/sub/after.jpg",
                                                     "reviews/model-page/img/data.bin"]
