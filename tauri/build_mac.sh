#!/bin/bash
# The desktop app for macOS: goodpush's server (PyInstaller, from tauri/spec_rewrite.py) inside the Tauri shell,
# signed (Developer ID, hardened runtime), in a disk image, notarized and stapled. Runs on this Mac, never on GitHub.
#   tauri/build_mac.sh "Developer ID Application: NAME (TEAMID)" NOTARY_PROFILE
# NOTARY_PROFILE: a notarytool keychain profile (xcrun notarytool store-credentials). Leave it "" to sign only, and
# leave out both to build an unsigned app to try on this Mac. Run from anywhere; writes build/tauri/.
set -euo pipefail
ID="${1:-}"; PROFILE="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT=build/tauri; ENT=tauri/src-tauri/Entitlements.plist
PY_VERSION="${GOODPUSH_BUILD_PYTHON:-3.12}"          # the installed app's Python (Intel Macs: conda-forge's)
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv-build"   # its own environment: .venv (development) stays as it is

echo "== 1/6 the server's environment (Python $PY_VERSION, uv.lock)"
uv sync --frozen --python "$PY_VERSION" --extra build --reinstall-package goodpush   # its version from git, now
VERSION="$(uv run --no-sync python -c 'import goodpush; print(goodpush.__version__)')"
SEMVER="$(echo "$VERSION" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+')"
echo "   goodpush $VERSION (app version $SEMVER)"

echo "== 2/6 the server bundle (PyInstaller)"
(cd packaging && uv run --no-sync python make_icon.py && uv run --no-sync python notices.py)
uv run --no-sync python tauri/spec_rewrite.py "$OUT/goodpush-server.spec"
uv run --no-sync pyinstaller "$OUT/goodpush-server.spec" --noconfirm --distpath "$OUT/dist" --workpath "$OUT/work"
SERVER="$OUT/dist/goodpush-server.app"

echo "== 3/6 the Tauri shell"
(cd tauri && npm ci --no-fund --no-audit && npx tauri icon ../packaging/icon.png -o src-tauri/icons >/dev/null &&
  npx tauri build --bundles app --config "{\"version\": \"$SEMVER\"}")
APP_SRC="$(ls -d tauri/src-tauri/target/release/bundle/macos/*.app | head -1)"
APP="$OUT/$(basename "$APP_SRC")"
rm -rf "$APP" && ditto "$APP_SRC" "$APP"
mkdir -p "$APP/Contents/Helpers" && ditto "$SERVER" "$APP/Contents/Helpers/goodpush-server.app"
NAME="goodpush-desktop-$SEMVER-mac-$(uname -m)"

if [ -z "$ID" ]; then
  echo "== no signing identity: ad-hoc signed, for this Mac only"
  codesign --force --deep --sign - --entitlements "$ENT" "$APP"
  echo "built $APP"; exit 0
fi

echo "== 4/6 signing, inside out"
sign() { codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$ID" "$@"; }
INNER="$APP/Contents/Helpers/goodpush-server.app"
find "$INNER/Contents" -type f ! -path "*/Contents/MacOS/goodpush" -print0 | xargs -0 file | grep Mach-O |
  cut -d: -f1 | while IFS= read -r f; do sign "$f"; done
find "$INNER" -name "*.framework" -type d -print0 | xargs -0 -I{} codesign --force --timestamp --options runtime --sign "$ID" {}
sign "$INNER"
sign "$APP/Contents/MacOS/"*
sign "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

echo "== 5/6 the disk image"
rm -rf "$OUT/dmg" "$OUT/$NAME.dmg" && mkdir "$OUT/dmg"
cp -R "$APP" LICENSE DISCLAIMER.md packaging/THIRD_PARTY_NOTICES.txt "$OUT/dmg/"
ln -s /Applications "$OUT/dmg/Applications"
hdiutil create -volname "GoodPush Desktop" -srcfolder "$OUT/dmg" -ov -format UDZO "$OUT/$NAME.dmg" && rm -rf "$OUT/dmg"
codesign --force --timestamp --sign "$ID" "$OUT/$NAME.dmg"

if [ -n "$PROFILE" ]; then
  echo "== 6/6 notarizing (a few minutes)"
  xcrun notarytool submit "$OUT/$NAME.dmg" --keychain-profile "$PROFILE" --wait
  xcrun stapler staple "$OUT/$NAME.dmg"
  xcrun stapler validate "$OUT/$NAME.dmg"
  spctl --assess --type open --context context:primary-signature -vv "$OUT/$NAME.dmg"
fi
echo "built $OUT/$NAME.dmg"
