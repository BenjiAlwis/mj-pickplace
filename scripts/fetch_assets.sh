#!/usr/bin/env bash
# Fetch the UR5e and Robotiq 2F-85 models from MuJoCo Menagerie at a pinned commit.
set -euo pipefail
COMMIT="98a70c569c4c1b6da467a20012f766ee1f8dc521"
DEST="${1:-third_party/mujoco_menagerie}"
if [ -f "$DEST/universal_robots_ur5e/ur5e.xml" ] && [ -f "$DEST/robotiq_2f85/2f85.xml" ]; then
  echo "assets already present in $DEST"; exit 0
fi
rm -rf "$DEST"
git init -q "$DEST"
git -C "$DEST" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git -C "$DEST" sparse-checkout set universal_robots_ur5e robotiq_2f85
git -C "$DEST" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$DEST" checkout -q FETCH_HEAD
echo "fetched Menagerie @ ${COMMIT:0:8} into $DEST"
