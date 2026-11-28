"""The desktop app (tauri/): its server entry point and the PyInstaller spec it writes from packaging/goodpush.spec.

The spec test is the one a sync with upstream trips first: a change to packaging/goodpush.spec where the desktop app
edits it fails here, before a build does."""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

TAURI = Path(__file__).resolve().parents[1] / "tauri"


class Load:
    """tauri/'s scripts as modules (tauri/ is not a package: PyInstaller and the build script run them as files)."""

    @staticmethod
    def module(name: str):
        spec = importlib.util.spec_from_file_location(f"tauri_{name}", TAURI / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


rewrite = Load.module("spec_rewrite")
sidecar = Load.module("sidecar")


def test_spec_rewrite_applies_to_packaging_spec():
    s = rewrite.SpecRewrite()
    assert s.missing() == [], "packaging/goodpush.spec changed where tauri/spec_rewrite.py edits it"
    text = s.text()
    compile(text, "goodpush-server.spec", "exec")            # still Python
    assert "goodpush_app.py" not in text
    assert (TAURI / "sidecar.py").as_posix() in text
    assert f'name="{s.BUNDLE_NAME}"' in text and s.BUNDLE_ID in text
    assert "PySide6.QtWebEngineCore" in text.split("UNUSED = ")[1].split("]")[0]
    assert '"icon.png"' not in text and '"../LICENSE"' not in text   # every path absolute


def test_spec_rewrite_names_what_upstream_changed():
    s = rewrite.SpecRewrite(rewrite.SpecRewrite().source.replace('["goodpush_app.py"]', '["launcher.py"]'))
    with pytest.raises(ValueError, match="goodpush_app.py"):
        s.text()


def test_spec_rewrite_skips_fixes_upstream_already_has():
    fixed = rewrite.SpecRewrite().source.replace('collect_data_files("goodpush")',
                                                 'collect_data_files("goodpush", include_py_files=True)')
    text = rewrite.SpecRewrite(fixed).text()
    assert text.count("include_py_files=True") == 1


def test_spec_rewrite_check_and_write(tmp_path):
    with pytest.raises(SystemExit) as e:
        rewrite.SpecRewrite.main(["--check"])
    assert e.value.code == 0
    out = tmp_path / "build" / "goodpush-server.spec"
    rewrite.SpecRewrite.main([str(out)])
    assert out.read_text(encoding="utf-8").startswith("# Written by tauri/spec_rewrite.py")


class FakeApp:
    """launcher.App's part the server entry uses: stopped, quit()."""

    def __init__(self):
        self.stopped = threading.Event()
        self.quits = 0

    def quit(self):
        self.quits += 1
        self.stopped.set()


def test_server_stops_when_the_desktop_app_goes(monkeypatch):
    srv = sidecar.ShellServer(0)
    srv.app = FakeApp()
    monkeypatch.setattr(srv, "WATCH_S", 0.01)
    monkeypatch.setattr(sidecar.os, "getppid", lambda: 1)     # re-parented: the desktop app is gone
    srv.watch(parent=4242)
    assert srv.app.quits == 1


def test_server_stops_on_sigterm():
    srv = sidecar.ShellServer(0)
    srv.app = FakeApp()
    srv.stop()
    assert srv.app.stopped.wait(2) and srv.app.quits == 1


def test_other_arguments_are_the_command_line(monkeypatch):
    seen = []
    import goodpush.launcher
    monkeypatch.setattr(goodpush.launcher, "main", lambda args: seen.append(args))
    sidecar.ShellServer.main(["--root", "x", "ride", "--save-only"])      # how the server starts Ride now
    assert seen == [["--root", "x", "ride", "--save-only"]]


def test_serve_needs_a_port():
    with pytest.raises(SystemExit):
        sidecar.ShellServer.main([sidecar.ShellServer.FLAG])
