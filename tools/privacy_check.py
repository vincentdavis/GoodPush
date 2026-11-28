"""Pre-commit privacy check: refuse commits that carry personal data or goodpush's private files.

    python tools/privacy_check.py FILE...          check files (pre-commit passes the staged ones)
    python tools/privacy_check.py --message FILE   check a commit message (commit-msg hook)

Secrets in general (keys, tokens, high-entropy strings) are detect-secrets' job; this catches what is
specific to goodpush: ride / health data files, the workspace goodpush.toml, device numbers, e-mail
addresses, home-folder and drive paths (C: too), password values, GPS coordinates, Google / Strava credentials,
and your own terms in
privacy_terms.txt in goodpush's machine folder (TERMS_FOLDERS; one per line: names, address, serial numbers ...; that file stays private).
A term written "code: NAME" applies only to the package's code, tests and model docs (CODE_PATHS): the
software is for any rider, so it names none - while NAME may still sign the licence and the author fields.
A line with "privacy: ok" in it is let through. Binary files are refused, except a design review's screenshots
(REVIEW_IMAGES: reviews/<page>/img/, made from a synthetic rider).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path, PurePosixPath


class PrivacyCheck:
    # files that hold your rides, health data, settings or credentials
    BLOCKED_PATHS = [re.compile(p, re.I) for p in (
        r"^goodpush\.toml$", r"(^|/)(data|reports|notes)/", r"(^|/)\.goodpush/",
        r"(^|/)\.env(\..*)?$",
        r"\.(fit|gpx|tcx|h5|hdf5|parquet|npz|zip|pem|key|p12|pfx)$", r"(^|/)ride_devices\.json$",
        r"(token|secret|credential)s?[^/]*\.json$")]
    # a design review's screenshots (reviews/<page>/img/), the one place a binary file may go in: made from a synthetic
    # rider (goodpush synth), never your own data
    REVIEW_IMAGES = re.compile(r"^reviews/[^/]+/img/[^/]+\.(jpe?g|png|webp)$", re.I)
    SAFE_EMAIL = re.compile(r"@(example\.(com|org|net)|users\.noreply\.github\.com|anthropic\.com)$", re.I)
    SAFE_HOME = {"you", "user", "username", "name", "<name>", "<you>", "<user>", "me", "runner", "public",
                 "default", "%username%"}
    FAKE_DEVICE = {"12345"}
    PATTERNS = {
        "e-mail address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
        "device number": re.compile(r"\b(ant|ble):(fec|power|hr|cadence|speed_cadence):(\d{3,})\b"),
        "home folder path": re.compile(r"([A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)([^\\/\s\"'`]+)"),
        "personal drive path": re.compile(r"\b[D-Zd-z]:[\\/]+[A-Za-z]"),
        "system drive path": re.compile(r"\b[Cc]:[\\/]+(?![Uu]sers[\\/])[A-Za-z]"),   # (home folders: above)
        "password value": re.compile(r"(password|passwd|pwd)[\"']?\s*[:=]\s*[\"'][^\"'\s]{4,}[\"']", re.I),
        "GPS coordinates": re.compile(r"-?\b\d{1,3}\.\d{5,}\s*[,; ]\s*-?\d{1,3}\.\d{5,}\b"),
        "Google OAuth client": re.compile(r"\d{6,}-[a-z0-9]{20,}\.apps\.googleusercontent\.com|GOCSPX-[\w-]{10,}"),
        "token value": re.compile(r"(access|refresh|bearer|api)[_ -]?(token|key)[\"']?\s*[:=]\s*[\"'][\w.\-]{16,}[\"']",
                                  re.I),
    }
    TOML_DEVICE = re.compile(r"^\s*(trainer|power|cadence|hr)\s*=\s*\"[^\"]+\"")
    OK = "privacy: ok"
    CODE = "code:"
    CODE_PATHS = re.compile(r"^(src|tests|srcdocs/model)/")

    TERMS = "privacy_terms.txt"

    def __init__(self, terms_file: Path | None = None):
        if terms_file is None:
            terms_file = next((d / self.TERMS for d in self.terms_folders() if (d / self.TERMS).is_file()),
                              Path(self.TERMS))
        lines = terms_file.read_text(encoding="utf-8").splitlines() if terms_file.exists() else []
        terms = [t.strip() for t in lines if t.strip() and not t.lstrip().startswith("#")]
        self.terms = [t for t in terms if not t.lower().startswith(self.CODE)]
        self.code_terms = [t[len(self.CODE):].strip() for t in terms if t.lower().startswith(self.CODE)]
        self.problems: list[str] = []

    @staticmethod
    def terms_folders() -> list[Path]:
        """where privacy_terms.txt may be, first found wins: goodpush's machine folder (GOODPUSH_CONFIG, else the
        platform's: goodpush.folders.Folders, which this hook's own environment can't import), then ~/.goodpush
        (before 2026-11-25, until goodpush moves it)."""
        out = [Path(os.environ[k]) for k in ("GOODPUSH_CONFIG", "GOODPUSH_SECRETS") if os.environ.get(k)]
        try:
            import platformdirs
            out.append(Path(platformdirs.user_config_dir("goodpush", appauthor=False, roaming=False)))
        except ImportError:
            pass
        return out + [Path.home() / ".goodpush"]

    def check_path(self, path: str) -> bool:
        rel = PurePosixPath(path.replace("\\", "/")).as_posix()
        if any(p.search(rel) for p in self.BLOCKED_PATHS):
            self.problems.append(f"{rel}: private file (rides, health data, settings or credentials)")
            return False
        return True

    def check_text(self, name: str, text: str) -> None:
        toml = name.endswith(".toml")
        code = bool(self.CODE_PATHS.search(PurePosixPath(name.replace("\\", "/")).as_posix()))
        for i, line in enumerate(text.splitlines(), 1):
            if self.OK in line:
                continue
            for what in self.found(line, toml, code):
                self.problems.append(f"{name}:{i}: {what}: {line.strip()[:100]}")

    def found(self, line: str, toml: bool, code: bool = False) -> list[str]:
        out = []
        for what, rx in self.PATTERNS.items():
            for m in rx.finditer(line):
                if what == "e-mail address" and self.SAFE_EMAIL.search(m.group(0)):
                    continue
                if what == "device number" and m.group(3) in self.FAKE_DEVICE:
                    continue
                if what == "home folder path" and m.group(2).lower() in self.SAFE_HOME:
                    continue
                out.append(what)
                break
        if toml and self.TOML_DEVICE.search(line):
            out.append("device choice in a .toml")
        low = line.lower()
        out += [f"your term {t!r}" for t in self.terms if t.lower() in low]
        if code:
            out += [f"a rider named in code ({t!r})" for t in self.code_terms if t.lower() in low]
        return out

    def check_file(self, path: str) -> None:
        if not self.check_path(path):
            return
        try:
            data = Path(path).read_bytes()
        except OSError:
            return                                   # deleted / moved
        if b"\0" in data[:8000]:
            if self.REVIEW_IMAGES.search(PurePosixPath(path.replace("\\", "/")).as_posix()):
                return
            self.problems.append(f"{path}: binary file - commit only source and docs")
            return
        self.check_text(path, data.decode("utf-8", errors="replace"))

    def run(self, argv: list[str]) -> int:
        if argv[:1] == ["--message"]:
            text = Path(argv[1]).read_text(encoding="utf-8", errors="replace")
            self.check_text("commit message", "\n".join(l for l in text.splitlines() if not l.startswith("#")))
        else:
            for path in argv:
                self.check_file(path)
        if self.problems:
            print("Privacy check failed - remove these (or mark a line 'privacy: ok' if it's really fine):")
            print("\n".join(f"  {p}" for p in self.problems))
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(PrivacyCheck().run(sys.argv[1:]))
