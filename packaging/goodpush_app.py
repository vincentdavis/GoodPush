"""Entry point of the installed app (PyInstaller): the launcher, or the command line when given arguments."""
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False) and sys.platform == "darwin":
    # numba's cache next to the code would write into the signed app (and break its seal): the user's caches instead
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / "Library" / "Caches" / "goodpush" / "numba"))

from goodpush.launcher import main  # noqa: E402

if __name__ == "__main__":
    main()
