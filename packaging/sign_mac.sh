#!/bin/bash
# Developer ID signing (hardened runtime), notarization and stapling of the macOS app and its disk image.
#   packaging/sign_mac.sh "Developer ID Application: NAME (TEAMID)" NOTARY_PROFILE [dmg-name]
# NOTARY_PROFILE: a notarytool keychain profile (xcrun notarytool store-credentials). Run from the repo root
# after pyinstaller has written dist/goodpush.app. Leave NOTARY_PROFILE empty ("") to sign only.
set -euo pipefail
ID="$1"; PROFILE="${2:-}"; NAME="${3:-goodpush-mac-apple-silicon}"
APP=dist/goodpush.app; ENT=packaging/entitlements.plist
# inside out: every Mach-O file, then the Qt frameworks as bundles, then the app
find "$APP/Contents" -type f ! -path "*/Contents/MacOS/goodpush" -print0 | xargs -0 file | grep Mach-O | cut -d: -f1 |
  while IFS= read -r f; do codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$ID" "$f"; done
find "$APP" -name "*.framework" -type d -print0 | xargs -0 -I{} codesign --force --timestamp --options runtime --sign "$ID" {}
codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$ID" "$APP"
codesign --verify --deep --strict "$APP"
# the disk image, as in .github/workflows/build.yml
rm -rf dmg "$NAME.dmg" && mkdir dmg && cp -R "$APP" LICENSE DISCLAIMER.md packaging/THIRD_PARTY_NOTICES.txt dmg/
ln -s /Applications dmg/Applications
hdiutil create -volname goodpush -srcfolder dmg -ov -format UDZO "$NAME.dmg" && rm -rf dmg
codesign --force --timestamp --sign "$ID" "$NAME.dmg"
if [ -n "$PROFILE" ]; then
  xcrun notarytool submit "$NAME.dmg" --keychain-profile "$PROFILE" --wait
  xcrun stapler staple "$NAME.dmg"
  xcrun stapler validate "$NAME.dmg"
  spctl --assess --type open --context context:primary-signature -vv "$NAME.dmg"
fi
