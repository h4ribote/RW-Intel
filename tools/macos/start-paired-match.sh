#!/bin/bash
# Starts the two game instances of one paired lockstep match.
#
# The macOS analogue of Start-RwPairedMatch.ps1. Which instance hosts and which joins is decided by the control process, which sends host, port and join alongside the start command. Both instances run in one container and so share its loopback, which is where the host binds its match port and the joiner reaches it; there is therefore no host-side port to check, unlike on Windows where each instance is a separate process on the machine's own network.
#
#   python -m rwintel.control --host 0.0.0.0 --instances 2 --paired --opponents 0 --map Lake --max-seconds 180
#   ./tools/macos/start-paired-match.sh -Speed 10
#
# A host compares a checksum of the core unit definitions before admitting the joiner. Both instances mount one master copy of the install and launch with mods off, so they agree by construction.
set -euo pipefail
here=$(dirname "$0")

# The pair is exactly two instances; everything else is forwarded to start-agents.sh unchanged.
exec "$here/start-agents.sh" -Count 2 "$@"
