"""The installed app. Double-click goodpush: it opens in your browser, with an icon in the menu bar / system
tray to open it again or quit. Opened again while it's running, it just shows the page.

Started with arguments it is the command line (`goodpush ride --session ...`): that's how the app starts the
ride window from a bundle that has no separate Python. Your project is the one in the folder goodpush starts
from, else this machine's default athlete (athletes.Athletes), else the app's own folder (config.default_root),
made on first launch. Another goodpush on its port (e.g. a development one) is left alone: it takes the next port.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PORT = 8765


def command(*args: str) -> list[str]:
    """goodpush run with `args` as its own process: the bundle itself when installed, else python -m goodpush.cli."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-u", "-m", "goodpush.cli", *args]


def child_env() -> dict:
    """the environment for that process: unbuffered, so its log fills as it goes."""
    return {**os.environ, "PYTHONUNBUFFERED": "1"}


class App:
    """the server on this computer and the tray icon that opens / stops it."""

    PORTS = 10                     # ports tried from PORT on when another goodpush (e.g. a development one) has it

    def __init__(self, port: int = PORT):
        self.port = port
        self.url = f"http://127.0.0.1:{port}/"
        self.httpd = None
        self.handler = None
        self.stopped = threading.Event()           # the server ended (Quit)

    @staticmethod
    def build() -> str:
        """which goodpush this is (the installed app, a development env, ...): its program and version."""
        from . import __version__
        return hashlib.sha1(f"{Path(sys.executable).resolve()}|{__version__}".encode()).hexdigest()[:12]

    @staticmethod
    def build_at(port: int) -> str | None:
        """the build of the goodpush serving `port` ("" = another program, or an older goodpush), None = free."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/job", timeout=1) as r:
                return str(json.loads(r.read()).get("build") or "")
        except urllib.error.HTTPError:
            return ""
        except (OSError, ValueError):
            return None

    def choose(self) -> bool:
        """the first port that is free or runs this very build; True = it's already running (opened twice)."""
        me = self.build()
        for port in range(self.port, self.port + self.PORTS):
            b = self.build_at(port)
            if b is None or b == me:
                self.port, self.url = port, f"http://127.0.0.1:{port}/"
                return b is not None
        raise RuntimeError(f"ports {self.port}-{self.port + self.PORTS - 1} are all used by other programs")

    def project(self):
        """the project `goodpush` would use (this folder's, else this machine's default athlete), else the app's
        own folder, made on first launch."""
        from .config import default_root, init_project, load_project
        try:
            return load_project()
        except FileNotFoundError:
            return init_project(default_root())

    def start(self) -> None:
        from http.server import ThreadingHTTPServer

        from .app.server import make_handler
        self.handler = make_handler(self.project(), None, False, port=self.port)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self.handler)
        threading.Thread(target=self.serve, name="goodpush server", daemon=True).start()

    def serve(self) -> None:
        """the server until it's shut down (the page's Quit, or the tray's), then the tray goes too."""
        self.httpd.serve_forever()
        self.stopped.set()
        try:
            from PySide6.QtCore import QMetaObject, Qt
            from PySide6.QtWidgets import QApplication
        except ImportError:
            return
        if QApplication.instance() is not None:          # its event loop runs on the main thread
            QMetaObject.invokeMethod(QApplication.instance(), "quit", Qt.ConnectionType.QueuedConnection)

    def open(self) -> None:
        webbrowser.open(self.url)

    def quit(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()

    def run(self) -> None:
        if self.choose():
            self.open()
            return
        self.start()
        if os.environ.get(AppWindow.ENV):        # spike: the app in its own window (Qt WebEngine)
            AppWindow(self.url).run()
            self.quit()
            return
        self.handler.pages.open_unless_seen(self.url)   # a goodpush tab still open reloads itself instead
        if not self.tray():                      # no tray (some Linux desktops): until Quit in the page
            self.stopped.wait()

    def tray(self) -> bool:
        """the menu bar / system tray icon (Open goodpush, Quit) until you quit; False when there's none."""
        try:
            from PySide6.QtGui import QAction
            from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon
        except ImportError:
            return False
        app = QApplication.instance() or QApplication([])
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return False
        app.setQuitOnLastWindowClosed(False)
        from .ride.overlay import Overlay
        icon = QSystemTrayIcon(Overlay.icon())
        menu = QMenu()
        for text, slot in (("Open goodpush", self.open), ("Quit goodpush", app.quit)):
            act = QAction(text, menu)
            act.triggered.connect(slot)
            menu.addAction(act)
        icon.setContextMenu(menu)
        icon.setToolTip("goodpush")
        icon.activated.connect(self.clicked)
        icon.show()
        app.exec()
        self.quit()
        return True

    def clicked(self, reason) -> None:
        """a click on the tray icon itself (Windows, Linux) opens the page; the menu is on right-click."""
        from PySide6.QtWidgets import QSystemTrayIcon
        if reason == QSystemTrayIcon.ActivationReason.Trigger and sys.platform != "darwin":
            self.open()


class AppWindow:
    """Spike (branch spike/qt-webengine): the page in a window of its own (Qt WebEngine) instead of a browser tab.
    A profile that keeps localStorage, links to other sites in your browser, downloads into Downloads with a notice
    at the bottom (Show in folder), reload with the platform's key (Cmd+R, F5) as in a browser. With
    GOODPUSH_WEBVIEW_CHECK=<file> it loads the page, downloads a file next to <file>, writes what it found as JSON
    and quits (the build check)."""
    ENV = "GOODPUSH_WEBVIEW"
    CHECK = "GOODPUSH_WEBVIEW_CHECK"
    CHECK_S = 60                                 # the check gives up after this long
    NOTICE_S = 8                                 # a download's notice shows this long
    NOTICE_CSS = ("QFrame#notice { background:#1f1f1d; border-radius:10px; } QLabel { color:#f0efec; font-size:13px; }"
                  " QPushButton { color:#f0efec; background:#3a3a37; border:0; border-radius:6px; padding:5px 10px; }"
                  " QPushButton:hover { background:#4a4a46; }")

    def __init__(self, url: str):
        self.url = url
        self.found = {}
        self.saved = None                        # the file the notice is about

    def run(self) -> None:
        from PySide6.QtCore import QStandardPaths, QTimer
        from PySide6.QtWebEngineCore import QWebEngineProfile
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QApplication
        from .folders import Folders
        app = QApplication.instance() or QApplication([])
        self.app = app
        self.profile = QWebEngineProfile("goodpush", app)            # named: kept on disk (the default one isn't)
        self.profile.setPersistentStoragePath(str(Folders.machine() / "webview"))   # this computer's, not an athlete's
        self.profile.downloadRequested.connect(self.download)
        self.downloads = str(Path(os.environ[self.CHECK]).parent) if os.environ.get(self.CHECK) else \
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        self.view = QWebEngineView(self.profile)
        self.view.page().newWindowRequested.connect(self.elsewhere)
        self.view.setWindowTitle("goodpush")
        self.view.resize(1280, 860)
        self.shortcuts()
        self.notice_bar()
        if os.environ.get(self.CHECK):
            self.view.loadFinished.connect(self.loaded)
            QTimer.singleShot(self.CHECK_S * 1000, self.report)
        self.view.load(self.url)
        self.view.show()
        app.exec()

    def shortcuts(self) -> None:
        """what a browser tab would let you do: reload the page (Cmd+R on a Mac, F5 / Ctrl+R elsewhere)."""
        from PySide6.QtGui import QKeySequence, QShortcut
        for keys in QKeySequence.keyBindings(QKeySequence.StandardKey.Refresh):
            QShortcut(keys, self.view, self.view.reload)

    def elsewhere(self, request) -> None:
        """a link to another site (target=_blank): in your browser."""
        webbrowser.open(request.requestedUrl().toString())

    def notice_bar(self) -> None:
        """the bar a finished download shows at the bottom of the window (a browser would show it in its own)."""
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton
        self.notice = QFrame(self.view)
        self.notice.setObjectName("notice")
        self.notice.setStyleSheet(self.NOTICE_CSS)
        row = QHBoxLayout(self.notice)
        row.setContentsMargins(14, 8, 8, 8)
        self.notice_text, self.notice_show, close = QLabel(), QPushButton("Show in folder"), QPushButton("×")
        self.notice_show.clicked.connect(self.reveal)
        close.clicked.connect(self.notice.hide)
        for w in (self.notice_text, self.notice_show, close):
            row.addWidget(w)
        self.notice.hide()
        self.notice_timer = QTimer(self.notice)
        self.notice_timer.setSingleShot(True)
        self.notice_timer.timeout.connect(self.notice.hide)

    def download(self, item) -> None:
        from functools import partial
        item.setDownloadDirectory(self.downloads)
        item.isFinishedChanged.connect(partial(self.downloaded, item))
        item.accept()

    def downloaded(self, item) -> None:
        """a download ended: say where it went (or that it failed)."""
        from PySide6.QtWebEngineCore import QWebEngineDownloadRequest
        name = item.downloadFileName()
        if item.state() == QWebEngineDownloadRequest.DownloadState.DownloadCompleted:
            self.show_notice(f"Saved to Downloads: {name}", Path(item.downloadDirectory()) / name)
        else:
            self.show_notice(f"Download failed: {name} ({item.interruptReasonString()})", None)
        if os.environ.get(self.CHECK):
            self.found.update(download=name, saved=bool(self.saved and self.saved.is_file()),
                              notice=self.notice_text.text(), notice_shown=self.notice.isVisible())
            self.report()

    def show_notice(self, text: str, path: Path | None) -> None:
        self.saved = path
        self.notice_text.setText(text)
        self.notice_show.setVisible(path is not None)
        self.notice.adjustSize()
        self.notice.move((self.view.width() - self.notice.width()) // 2, self.view.height() - self.notice.height() - 18)
        self.notice.raise_()
        self.notice.show()
        self.notice_timer.start(self.NOTICE_S * 1000)

    def reveal(self) -> None:
        """the downloaded file selected in Explorer / Finder (StravaUpload.reveal)."""
        from .io.strava import StravaUpload
        if self.saved is not None:
            StravaUpload.reveal(self.saved)
        self.notice.hide()

    def loaded(self, ok: bool) -> None:
        """the check: the page loaded, the app's scripts ran (its tabs), localStorage works."""
        self.found["loaded"] = ok
        js = ("localStorage.setItem('gp_check', '1');"
              "JSON.stringify({title: document.title, tabs: document.querySelectorAll('#tabs a').length,"
              " storage: localStorage.getItem('gp_check'), agent: navigator.userAgent})")
        self.view.page().runJavaScript(js, 0, self.scripted)

    def scripted(self, result) -> None:
        self.found.update(json.loads(result or "{}"))
        from PySide6.QtCore import QTimer
        QTimer.singleShot(3000, self.drawn)        # the app's first /api/state and render

    def drawn(self) -> None:
        self.view.page().runJavaScript("document.querySelector('#main')?.innerText.slice(0, 200) || ''", 0, self.content)

    def content(self, text) -> None:
        """the page drawn (more than its "Loading…"), else a second later again (report() ends it at CHECK_S)."""
        self.found["main_text"] = text
        if len(text or "") > 20:
            js = "document.querySelector('link[rel=stylesheet]')?.href || ''"
            self.view.page().runJavaScript(js, 0, self.fetch)       # then a download (downloaded reports)
            return
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1000, self.drawn)

    def fetch(self, href) -> None:
        """the check's download: the page's own stylesheet, a file the app always serves."""
        from PySide6.QtCore import QUrl
        if not href:
            self.report()
            return
        self.view.page().download(QUrl(href), "")

    def report(self) -> None:
        Path(os.environ[self.CHECK]).write_text(json.dumps(self.found), encoding="utf-8")
        self.app.quit()


def main(argv: list[str] | None = None) -> None:
    multiprocessing.freeze_support()             # the import workers re-run this program (installed app)
    args = sys.argv[1:] if argv is None else argv
    if args:
        from .cli import main as cli_main
        cli_main(args)
        return
    from .folders import Folders
    from .progress import Console
    Folders.migrate()                            # the folders from before, once (cli.main does it for commands)
    Console.no_quick_edit()                      # started from a terminal: a click must not freeze it
    from .app.diagnostics import AppLog
    AppLog.start()                               # its output into app.log (Settings > Diagnostics shows it)
    App().run()


if __name__ == "__main__":
    main()
