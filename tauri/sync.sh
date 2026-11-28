#!/bin/bash
# Bring the desktop app's branch up to date: upstream's main (GoodPush/GoodPush), then your own line of work
# (origin's VMD-local-dev, by default), merged into the branch you're on. Nothing is pushed or rebased.
#   tauri/sync.sh [YOUR_BRANCH]
# Then build (tauri/build_mac.sh) and push this branch yourself. A merge that stops on a conflict leaves it for you
# to finish (git status); tauri/ is the desktop app's alone, so a conflict there means a sync went wrong.
set -euo pipefail
MINE="${1:-VMD-local-dev}"
cd "$(dirname "$0")/.."
git diff --quiet && git diff --cached --quiet || { echo "commit or stash your changes first"; exit 1; }
BRANCH="$(git branch --show-current)"
case "$BRANCH" in main|"$MINE") echo "run this on the desktop app's branch, not $BRANCH"; exit 1;; esac
git remote get-url upstream >/dev/null 2>&1 || git remote add upstream https://github.com/GoodPush/GoodPush.git
echo "== fetching upstream main and origin $MINE"
git fetch upstream main --tags          # the tags: the version hatch-vcs gives the build
git fetch origin "$MINE"
echo "== merging upstream/main into $BRANCH"
git merge --no-edit upstream/main
echo "== merging origin/$MINE into $BRANCH"
git merge --no-edit "origin/$MINE"
echo "== checking the desktop app still fits"
uv run python tauri/spec_rewrite.py --check
uv run pytest tests/test_tauri_shell.py -q
echo "== up to date: $(git log --oneline -1)"
echo "next: tauri/build_mac.sh to build, then git push origin $BRANCH"
