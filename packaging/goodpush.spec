# PyInstaller: the installed goodpush app (one folder; on macOS a .app). Build with
#   uv sync --extra build && cd packaging && uv run python make_icon.py && uv run python notices.py && uv run pyinstaller goodpush.spec --noconfirm
#   (or, in a pip / conda environment: pip install -e ".[build]", then the same commands without "uv run")
# The installers wrap dist/ (packaging/README.md); GitHub Actions builds every platform (.github/workflows/build.yml).
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, collect_submodules

datas, binaries, hidden = [], [], []
# goodpush itself: every module (many are imported lazily) and its page, templates and scripts
hidden += collect_submodules("goodpush")
datas += collect_data_files("goodpush")
datas += [("../LICENSE", "."), ("../NOTICE", "."), ("THIRD_PARTY_NOTICES.txt", ".")]      # notices.py writes the second
# libraries with native parts or plugins PyInstaller can't see from the imports
BLUETOOTH = {"win32": ("winrt",), "darwin": ("CoreBluetooth", "objc")}.get(sys.platform, ())   # bleak's backends
for pkg in ("jax", "jaxlib", "numpyro", "openant", "libusb_package", "bleak", "tzdata") + BLUETOOTH:
    d, b, h = collect_all(pkg)
    datas, binaries, hidden = datas + d, binaries + b, hidden + h
binaries += collect_dynamic_libs("usb")

# What the app never uses, left out to keep the download small (~130 MB):
# * the community Garmin login (garminconnect, garth, curl_cffi): the app uses the Personal API key / browser snippet
# * pyarrow beyond local parquet files: Flight, cloud file systems, ORC, Substrait, CUDA, Gandiva
# * Pillow (only draws the icon at build time), Qt's translations (the app is in English)
UNUSED = ["tkinter", "matplotlib", "IPython", "pytest", "PIL",
          "garminconnect", "garth", "curl_cffi",
          "pyarrow.flight", "pyarrow._flight", "pyarrow._s3fs", "pyarrow._gcsfs", "pyarrow._azurefs", "pyarrow._hdfs",
          "pyarrow.orc", "pyarrow._orc", "pyarrow.substrait", "pyarrow._substrait", "pyarrow.cuda", "pyarrow._cuda",
          "pyarrow.gandiva",
          "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtPdf", "PySide6.Qt3DCore"]
# ... and their files, plus pyarrow's C++ headers / sources and packages' metadata of what's left out
UNUSED_FILES = ("flight", "arrow_s3", "substrait", "gandiva", "arrow_cuda", "translations",
                "pyarrow/include/", "pyarrow/src/", ".pyx", ".pxd", "garminconnect-", "garth-", "curl_cffi-")

a = Analysis(
    ["goodpush_app.py"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden + ["usb.backend.libusb1"],
    excludes=UNUSED,
    noarchive=False,
)
# spike/qt-webengine: the app's own window (launcher.AppWindow) needs Qt WebEngine, and WebEngine its locales
KEEP = ("qtwebengine_locales/en-US.pak",)
a.binaries = [b for b in a.binaries if not any(k in b[0].replace("\\", "/") for k in UNUSED_FILES)]
a.datas = [d for d in a.datas if any(k in d[0].replace("\\", "/") for k in KEEP)
           or not any(k in d[0].replace("\\", "/") for k in UNUSED_FILES)]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="goodpush",
    console=sys.platform.startswith("linux"),      # Linux: a terminal shows it's running; elsewhere the tray icon does
    icon="icon.png",                               # make_icon.py draws it; PyInstaller converts it (Pillow)
)
coll = COLLECT(exe, a.binaries, a.datas, name="goodpush")
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="goodpush.app",
        icon="icon.png",
        bundle_identifier="io.github.goodpush.goodpush",
        info_plist={"LSUIElement": True,                   # a menu-bar app: no Dock icon, the page is in your browser
                    "NSHighResolutionCapable": True,
                    # macOS asks before an app uses Bluetooth, and quits one that doesn't say why
                    "NSBluetoothAlwaysUsageDescription": "goodpush connects to your trainer and sensors over Bluetooth.",
                    "CFBundleShortVersionString": __import__("goodpush").__version__},
    )
