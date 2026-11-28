"""The PyInstaller spec of goodpush's server for the desktop app (tauri/), written from packaging/goodpush.spec.

    python tauri/spec_rewrite.py OUT.spec      write it (tauri/build_mac.sh does, then runs pyinstaller OUT.spec)
    python tauri/spec_rewrite.py --check       only check that packaging/goodpush.spec still has what it changes

The desktop app keeps no spec of its own, so a change upstream makes to packaging/goodpush.spec (a new library, a
file left out) reaches the desktop app with the next sync, and nothing in packaging/ differs from upstream. What
the desktop app needs differently is a list of edits here. A required edit whose text upstream changed stops the
build with the text it looked for (tauri/README.md, "When a sync breaks the build"); a fix is skipped once
upstream has it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Edit:
    """one change to the spec's text: `old` -> `new`; `required` ones must find `old` (fixes may be upstream)."""
    why: str
    old: str
    new: str
    required: bool = True


class SpecRewrite:
    """packaging/goodpush.spec with the desktop app's edits, its relative paths made absolute (so the result can
    be written anywhere, e.g. build/tauri/)."""
    SOURCE = REPO / "packaging" / "goodpush.spec"
    PACKAGING = REPO / "packaging"
    ENTRY = REPO / "tauri" / "sidecar.py"
    BUNDLE_NAME = "goodpush-server.app"
    BUNDLE_ID = "io.github.goodpush.desktop.server"

    def __init__(self, text: str | None = None):
        self.source = self.SOURCE.read_text(encoding="utf-8") if text is None else text

    def edits(self) -> list[Edit]:
        p = self.PACKAGING.as_posix()
        return [
            Edit("the server entry point (tauri/sidecar.py), not the tray launcher",
                 '["goodpush_app.py"]', f'[r"{self.ENTRY.as_posix()}"]'),
            Edit("the notices, from packaging/", '("../LICENSE", "."), ("../NOTICE", "."), ("THIRD_PARTY_NOTICES.txt", ".")',
                 f'(r"{REPO.as_posix()}/LICENSE", "."), (r"{REPO.as_posix()}/NOTICE", "."), '
                 f'(r"{p}/THIRD_PARTY_NOTICES.txt", ".")'),
            Edit("the icon, from packaging/", 'icon="icon.png"', f'icon=r"{p}/icon.png"'),
            Edit("a bundle of its own inside the desktop app", 'name="goodpush.app"', f'name="{self.BUNDLE_NAME}"'),
            Edit("its own bundle id, apart from the installed tray app's", 'bundle_identifier="io.github.goodpush.goodpush"',
                 f'bundle_identifier="{self.BUNDLE_ID}"'),
            Edit("no Qt WebEngine: the desktop app's window shows the page (about 200 MB less)",
                 '"PySide6.QtQml",', '"PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml",'),
            Edit("fix: the physio model and numba's cache need goodpush's .py files in the bundle",
                 'collect_data_files("goodpush")', 'collect_data_files("goodpush", include_py_files=True)',
                 required=False),
            Edit("fix: pyarrow.lib links libarrow_substrait (pyarrow 26): leaving it out breaks parquet",
                 '("flight", "arrow_s3", "substrait", ', '("flight", "arrow_s3", ', required=False),
        ]

    def missing(self) -> list[Edit]:
        """the required edits whose text packaging/goodpush.spec no longer has."""
        return [e for e in self.edits() if e.required and e.old not in self.source]

    def text(self) -> str:
        gone = self.missing()
        if gone:
            raise ValueError("packaging/goodpush.spec changed where the desktop app edits it; update "
                             "tauri/spec_rewrite.py:\n" + "\n".join(f"  {e.why}: looked for {e.old!r}" for e in gone))
        out = self.source
        for e in self.edits():
            out = out.replace(e.old, e.new)
        return "# Written by tauri/spec_rewrite.py from packaging/goodpush.spec: don't edit.\n" + out

    @classmethod
    def main(cls, argv: list[str]) -> None:
        if argv[:1] == ["--check"]:
            gone = cls().missing()
            for e in gone:
                print(f"packaging/goodpush.spec no longer has {e.old!r} ({e.why})")
            sys.exit(1 if gone else 0)
        if len(argv) != 1:
            sys.exit(__doc__)
        Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[0]).write_text(cls().text(), encoding="utf-8")


if __name__ == "__main__":
    SpecRewrite.main(sys.argv[1:])
