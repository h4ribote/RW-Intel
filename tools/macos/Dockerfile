# The amd64 Linux runtime for Rusted Warfare.
#
# On Apple Silicon the game cannot run natively: LWJGL 2.9.3's macOS natives are x86_64-only and its display-mode natives fault on current macOS. The Linux distribution runs cleanly instead, so the game is put in an amd64 container and driven under Rosetta, while the control process stays native on the host. The same image runs without emulation on a real x86-64 Linux host.
#
# The game ships its own JVM (jvm-linux, a JRE) and its own natives (liblwjgl64.so, librocketConnector.so and the libRocket libraries), so this image supplies only what those link against: a virtual display, a software OpenGL rasteriser, the X client libraries LWJGL needs, and freetype and libstdc++ for libRocket. The window the engine opens is 10x10, so software rasterisation costs nothing worth measuring.
#
# Build for amd64 explicitly, since the host is arm64:
#   docker build --platform linux/amd64 -t rw-linux:latest tools/macos
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        x11-utils \
        x11-xserver-utils \
        libgl1-mesa-dri libglx-mesa0 libglu1-mesa \
        libx11-6 libxext6 libxcursor1 libxrandr2 libxxf86vm1 libxi6 libxt6 \
        libfreetype6 libstdc++6 \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Force Mesa's software renderer; there is no GPU in the VM and the 10x10 window makes it free.
ENV LIBGL_ALWAYS_SOFTWARE=1
WORKDIR /rw
