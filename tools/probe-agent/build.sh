#!/bin/bash
# Builds the probe agent jar on macOS or Linux.
#
# On Windows the sibling build.ps1 compiles against the JDK bundled with the game. The Linux distribution bundles only a JRE, and the macOS copy bundles nothing, so this looks for a JDK to compile with. Any JDK 9 or newer with javac will do; the agent is compiled to Java 8 bytecode so that it loads in the game's own Java 8 Linux JVM, and that same class file version also loads in the Windows build's JVM.
#
#   ./tools/probe-agent/build.sh
#   JAVA_HOME=/path/to/jdk ./tools/probe-agent/build.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)

if [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/javac" ]; then
    jdk="$JAVA_HOME"
elif [ -x /usr/libexec/java_home ] && /usr/libexec/java_home >/dev/null 2>&1; then
    jdk=$(/usr/libexec/java_home)
elif command -v javac >/dev/null 2>&1; then
    jdk=$(dirname "$(dirname "$(command -v javac)")")
else
    echo "no JDK found; set JAVA_HOME to a JDK 9+ (one that includes javac)" >&2
    exit 1
fi

javac="$jdk/bin/javac"
jar="$jdk/bin/jar"

classes="$here/classes"
rm -rf "$classes"
mkdir -p "$classes"

"$javac" --release 8 -Xlint:-options -d "$classes" "$here/RwProbeAgent.java"
"$jar" --create --file "$here/rwprobe.jar" --manifest "$here/manifest.txt" -C "$classes" .

echo "built $here/rwprobe.jar"
