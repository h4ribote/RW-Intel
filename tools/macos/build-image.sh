#!/bin/bash
# Builds the amd64 Linux runtime image the other macOS scripts run the game in.
#
# This is the macOS analogue of New-RwInstance.ps1: it prepares the environment a run needs. It does not create instance directories, because in the container model those are made inside the container at launch time (see rw-run.sh), where the game install is mounted read-only and each instance is only a current directory of symlinks.
#
#   ./tools/macos/build-image.sh
#
# The image is tagged rw-linux:latest by default; override with RW_IMAGE.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_docker

echo "building $RW_IMAGE for linux/amd64 from $RW_MACOS/Dockerfile"
docker build --platform linux/amd64 -t "$RW_IMAGE" "$RW_MACOS"
echo "built $RW_IMAGE"
