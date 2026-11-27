"""Build the app's stylesheet: styles/app.css (Tailwind CSS v4 + daisyUI 5) -> src/goodpush/app/static/app.css.

    python styles/build.py           build it once (minified)
    python styles/build.py --watch   rebuild on every change (while editing the CSS, index.html or js/)

Tailwind's standalone CLI (one binary, no Node) at the pinned version below is downloaded once into styles/.bin/
(not in git) and checked against its release SHA-256. daisyUI is vendored in styles/vendor/ (its release .mjs files,
pinned). The compiled app.css is committed: the app, pip installs and packaging/ never need this script.
"""
from __future__ import annotations

import hashlib
import platform
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path


class TailwindBuild:
    VERSION = "4.3.5"
    URL = "https://github.com/tailwindlabs/tailwindcss/releases/download/v{v}/{name}"
    # the release's sha256sums.txt, for the platforms goodpush is built and developed on
    SHA256 = {
        "tailwindcss-macos-arm64": "f3f4f5d1727e2b693b8f1c9a1492b5dd87008c86ea286bee1589cb458e6235fc",  # pragma: allowlist secret
        "tailwindcss-macos-x64": "a63702686a3ed048b3135d84a73be49ea9d47f880c9b20e54b3ef60934ad8cdf",  # pragma: allowlist secret
        "tailwindcss-linux-x64": "e47695cee3f0d90623a91f3939b08bdee610eb5ad99340adff9d4145b284d6cc",  # pragma: allowlist secret
        "tailwindcss-windows-x64.exe": "2803583268c53f4713e89943e211081fddb186c0a4a7d07b5c342c6ae853ec95",  # pragma: allowlist secret
    }
    HERE = Path(__file__).resolve().parent
    SOURCE = HERE / "app.css"
    OUTPUT = HERE.parent / "src" / "goodpush" / "app" / "static" / "app.css"

    @classmethod
    def binary_name(cls) -> str:
        system, machine = platform.system(), platform.machine().lower()
        arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
        if system == "Darwin":
            return f"tailwindcss-macos-{arch}"
        if system == "Windows":
            return "tailwindcss-windows-x64.exe"
        return f"tailwindcss-linux-{arch}"

    @classmethod
    def binary(cls) -> Path:
        name = cls.binary_name()
        exe = cls.HERE / ".bin" / cls.VERSION / name
        if exe.is_file():
            return exe
        exe.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading Tailwind CSS {cls.VERSION} ({name})")
        data = urllib.request.urlopen(cls.URL.format(v=cls.VERSION, name=name), timeout=120).read()
        want = cls.SHA256.get(name)
        got = hashlib.sha256(data).hexdigest()
        if want and got != want:
            sys.exit(f"{name}: SHA-256 {got} isn't the release's {want}; not using it")
        if not want:
            print(f"note: no pinned checksum for {name}; add {got} to TailwindBuild.SHA256 once checked")
        exe.write_bytes(data)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return exe

    @classmethod
    def run(cls, argv: list[str]) -> int:
        cmd = [str(cls.binary()), "-i", str(cls.SOURCE), "-o", str(cls.OUTPUT)]
        cmd += ["--watch"] if "--watch" in argv else ["--minify"]
        return subprocess.run(cmd, cwd=cls.HERE).returncode


if __name__ == "__main__":
    sys.exit(TailwindBuild.run(sys.argv[1:]))
