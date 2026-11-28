"""Entry point of goodpush inside the desktop app (tauri/): the server only, for the app's own window to show.

    goodpush --tauri-serve PORT      the server on 127.0.0.1:PORT, no browser and no tray icon (the window has them);
                                     it stops when the desktop app goes, or on the page's Quit (then the app goes)
    goodpush ARGS...                 the command line, as packaging/goodpush_app.py: the server starts the ride window
                                     this way (launcher.command), so Ride now runs in the same bundle

PyInstaller builds it with tauri/spec_rewrite.py's copy of packaging/goodpush.spec; in development the desktop app
runs it with `uv run python tauri/sidecar.py --tauri-serve PORT`.
"""
from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path


class ShellServer:
    """goodpush's server for the desktop app's window: the installed app's (launcher.App) without its browser tab
    and tray icon, ended by the app (SIGTERM, or the app gone) or by the page's Quit."""
    FLAG = "--tauri-serve"
    WATCH_S = 1.0                                # how often it checks the desktop app is still there

    def __init__(self, port: int):
        self.port = port
        self.app = None

    @staticmethod
    def caches() -> None:
        """numba's compiled kernels in this computer's cache folder, never inside the signed app."""
        if getattr(sys, "frozen", False) and sys.platform == "darwin":
            os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / "Library" / "Caches" / "goodpush" / "numba"))

    def run(self) -> None:
        from goodpush.folders import Folders
        from goodpush.launcher import App
        Folders.migrate()
        self.app = App(self.port)
        self.app.start()
        signal.signal(signal.SIGTERM, self.stop)
        threading.Thread(target=self.watch, args=(os.getppid(),), name="desktop app watch", daemon=True).start()
        print(f"goodpush serving {self.app.url}", flush=True)
        self.app.stopped.wait()

    def stop(self, *_signal) -> None:
        """the desktop app quits: the server stops (the main thread's wait then returns)."""
        if self.app is not None:
            threading.Thread(target=self.app.quit, name="desktop app quit", daemon=True).start()

    def watch(self, parent: int) -> None:
        """the desktop app that started this went (crashed, force-quit): stop, rather than hold its port."""
        while not self.app.stopped.is_set():
            if os.getppid() != parent:
                self.app.quit()
                return
            time.sleep(self.WATCH_S)

    @classmethod
    def main(cls, argv: list[str] | None = None) -> None:
        multiprocessing.freeze_support()         # the import workers re-run this program
        cls.caches()
        args = sys.argv[1:] if argv is None else argv
        if args[:1] == [cls.FLAG]:
            if len(args) != 2 or not args[1].isdigit():
                sys.exit(f"usage: goodpush {cls.FLAG} PORT")
            cls(int(args[1])).run()
            return
        from goodpush.launcher import main
        main(args)


if __name__ == "__main__":
    ShellServer.main()
