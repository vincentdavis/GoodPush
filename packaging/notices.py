"""THIRD_PARTY_NOTICES.txt for the installers: every library the app bundles (from the build environment) with its
version, licence and licence / notice files, plus Python's own licence. Run before PyInstaller; it goes into every
download next to goodpush's LICENSE (the MIT, BSD, Apache and LGPL licences of what's bundled ask for this).
Copyleft code (LGPL, GPL, MPL) also gets a pointer to its source at the exact version bundled: the app ships it
compiled (.pyc, libraries), and those licences ask that its source can be had."""
import re
import sys
from importlib import metadata
from pathlib import Path

# build tools, not part of the app
SKIP = {"goodpush", "pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "macholib", "pefile", "pywin32-ctypes",
        "pip", "setuptools", "wheel", "pytest", "pluggy", "iniconfig", "pillow",
        "pre-commit", "pre_commit", "cfgv", "identify", "nodeenv", "virtualenv", "detect-secrets", "distlib",
        "python-discovery",                                      # ... and dev tools, when built locally
        "garminconnect", "garth", "curl_cffi", "curl-cffi"}      # ... and what goodpush.spec leaves out (UNUSED)
NAMES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS")
LICENSES = Path(__file__).with_name("licenses")
COPYLEFT = re.compile(r"\bL?GPL|\bMPL|General Public|Mozilla Public", re.I)
PYPI = "https://pypi.org/project/{name}/{version}/#files"      # a package's sdist at that version
# packages with no sdist on PyPI: where their source is instead ({v}: the version)
SOURCE = {"pyside6-essentials": "https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v{v}",
          "shiboken6": "https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v{v}"}
# bundled code whose packages don't carry its licence text:
# (what, the package it comes in, licence, its source ({v}: that package's version), licence files)
ALSO = [("Qt (in PySide6 / shiboken6: the ride window and the tray icon)", "PySide6_Essentials", "LGPL-3.0-only",
         "https://download.qt.io/archive/qt/{mm}/{v}/single/ (qt-everywhere-src-{v}; the Qt libraries are separate "
         "files in the app and can be replaced)", ["LGPL-3.0.txt", "GPL-3.0.txt"]),
        ("libusb (in libusb-package: the ANT+ USB stick)", "libusb-package", "LGPL-2.1-or-later",
         "https://github.com/libusb/libusb/releases/tag/v{v3} (a separate library file in the app; it can be "
         "replaced)", ["LGPL-2.1.txt"])]
# front-end libraries copied into the app (app/static/vendor), not pip packages: (what, licence, their files there)
VENDOR = Path(__file__).resolve().parents[1] / "src" / "goodpush" / "app" / "static" / "vendor"
VENDORED = [("Apache ECharts 6.1.0 (the app's charts; with code from d3)", "Apache-2.0, d3: BSD-3-Clause",
             ["echarts.LICENSE", "echarts.NOTICE", "echarts.LICENSE-d3"]),
            ("Tailwind CSS 4.3.5 and daisyUI 5.8.3 (compiled into the app's stylesheet, app.css; styles/)", "MIT",
             ["tailwindcss.LICENSE", "daisyui.LICENSE"])]


def key(name: str) -> str:
    """A package name as PyPI compares them (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def source(dist) -> str | None:
    """Where a copyleft package's source at this exact version is; None for permissive ones."""
    lic = " ".join([licence(dist)] + [c for c in dist.metadata.get_all("Classifier") or [] if c.startswith("License")])
    if not COPYLEFT.search(lic):
        return None
    name, version = dist.metadata["Name"], dist.version
    return SOURCE.get(key(name), PYPI).format(name=name, version=version, v=version)


def also_source(template: str, version: str) -> str:
    """ALSO's source for the bundled package's version: {v} it, {mm} major.minor, {v3} its first three parts."""
    parts = version.split(".")
    return template.format(v=version, mm=".".join(parts[:2]), v3=".".join(parts[:3]))


def bundled_also() -> list[tuple[str, str, str, list[str]]]:
    """ALSO's entries whose package is in the build: (what + version, licence, source, licence files)."""
    out = []
    for what, pkg, lic, src, files in ALSO:
        try:
            version = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
        out.append((f"{what}, from {pkg} {version}", lic, also_source(src, version), files))
    return out


def licence(dist) -> str:
    md = dist.metadata
    if md.get("License-Expression"):
        return md["License-Expression"]
    lic = (md.get("License") or "").strip()
    if lic and len(lic) < 100 and "\n" not in lic:
        return lic
    cls = [c.split(" :: ")[-1] for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
    return ", ".join(cls) or "see the licence text below"


def texts(dist) -> list[tuple[str, str]]:
    out = []
    for f in dist.files or []:                     # a package's files: its licence ones are few
        name = Path(str(f)).name.upper()
        if name.startswith(NAMES) and (".dist-info" in str(f) or "licenses" in str(f).lower()):
            try:
                out.append((str(f), Path(dist.locate_file(f)).read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    return out


def python_licence() -> str:
    base = Path(sys.base_prefix)
    for p in (base / "LICENSE.txt", base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "LICENSE.txt"):
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    return "Python Software Foundation License: https://docs.python.org/3/license.html"


def main(out: Path) -> None:
    dists = sorted({d.metadata["Name"].lower(): d for d in metadata.distributions()
                    if d.metadata["Name"] and d.metadata["Name"].lower() not in SKIP}.values(),
                   key=lambda d: d.metadata["Name"].lower())
    head = ["goodpush bundles the software below. Each is the copyright of its authors and used under its licence,",
            "reproduced here. goodpush's own licence is in LICENSE + NOTICE. On Windows, tools\\ holds Zadig (GPLv3) with its",
            "notice.", "The source of each copyleft (LGPL / GPL / MPL) part, at the version bundled, is at its 'source:'.",
            "", "Summary", "-------"]
    also = bundled_also()
    for d in dists:
        src = source(d)
        head += [f"{d.metadata['Name']} {d.version}: {licence(d)}"] + ([f"    source: {src}"] if src else [])
    head += [f"Python {sys.version.split()[0]}: PSF License"]
    for what, lic, src, _ in also:
        head += [f"{what}: {lic}", f"    source: {src}"]
    head += [f"{what}: {lic}" for what, lic, _ in VENDORED]
    head += [""]
    body = []
    for d in dists:
        src = source(d)
        for path, text in texts(d):
            body += ["=" * 79, f"{d.metadata['Name']} {d.version} - {path}"] + ([f"Source: {src}"] if src else [])
            body += ["=" * 79, text.strip(), ""]
    body += ["=" * 79, f"Python {sys.version.split()[0]}", "=" * 79, python_licence().strip(), ""]
    for what, lic, src, files in also:
        body += ["=" * 79, f"{what}: {lic}", f"Source: {src}", "=" * 79]
        body += [(LICENSES / f).read_text(encoding="utf-8").strip() + "\n" for f in files]
    for what, lic, files in VENDORED:
        for f in files:
            body += ["=" * 79, f"{what} - {f}", "=" * 79, (VENDOR / f).read_text(encoding="utf-8").strip(), ""]
    out.write_text("\n".join(head + body), encoding="utf-8")
    print(f"{out.name}: {len(dists)} packages")


if __name__ == "__main__":
    main(Path(__file__).with_name("THIRD_PARTY_NOTICES.txt"))
