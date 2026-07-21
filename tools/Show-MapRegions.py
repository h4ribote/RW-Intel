"""Reports how the built-in skirmish maps decompose into regions.

Regions are the names the command layers use for places, so their count and size decide the shape of the operational layer's action space.
Run this to see what the merge distance actually produces before fixing it in the model, and to see which maps stay inside a given slot count.

    python tools/Show-MapRegions.py
    python tools/Show-MapRegions.py --distances 300 400 500
    python tools/Show-MapRegions.py --json local/regions.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rwdata import AssetPaths, decompose, list_skirmish_maps, read_map
from rwdata.regions import DEFAULT_MERGE_DISTANCE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", default=None, help="asset directory, defaults to local/rw/assets")
    parser.add_argument("--distances", type=float, nargs="+", default=[DEFAULT_MERGE_DISTANCE],
                        help="merge distances in world units to compare")
    parser.add_argument("--json", default=None, help="write the decomposition at the first distance to this file")
    arguments = parser.parse_args()

    paths = AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default()
    maps = list_skirmish_maps(paths)
    if not maps:
        print("no skirmish maps found", file=sys.stderr)
        return 1

    distances = arguments.distances
    header = f"{'map':46s} {'size':9s} {'p':>2s} {'res':>4s} " + " ".join(f"{('d' + str(int(d))):>6s}" for d in distances)
    print(header)
    print("-" * len(header))

    counts = {d: [] for d in distances}
    radii = {d: [] for d in distances}
    export = {}

    for path in maps:
        content = read_map(path, paths)
        cells = []
        for distance in distances:
            regions = decompose(content, distance)
            counts[distance].append(len(regions))
            radii[distance].extend(r.radius for r in regions)
            cells.append(f"{len(regions):6d}")
            if distance == distances[0]:
                export[content.name] = {
                    "width": content.width,
                    "height": content.height,
                    "tile_size": content.tile_size,
                    "players": content.players,
                    "merge_distance": distance,
                    "regions": [vars(r) for r in regions],
                }
        print(f"{content.name[:46]:46s} {content.width:4d}x{content.height:<4d} {content.players:2d} "
              f"{len(content.resources):4d} " + " ".join(cells))

    print()
    print(f"{'distance':>9s} {'median':>7s} {'mean':>7s} {'min':>5s} {'max':>5s} {'p95radius':>10s} {'<=24':>6s}")
    for distance in distances:
        values = counts[distance]
        within = sum(1 for v in values if v <= 24)
        percentile = sorted(radii[distance])[int(0.95 * (len(radii[distance]) - 1))]
        print(f"{distance:9.0f} {statistics.median(values):7.1f} {statistics.mean(values):7.1f} "
              f"{min(values):5d} {max(values):5d} {percentile:10.0f} {within:5d}/{len(values)}")

    print()
    for label, selector in (("2 players", 2), ("4 players", 4), ("6 or more", 6)):
        subset = []
        for path in maps:
            content = read_map(path, paths)
            if (content.players == selector) or (selector == 6 and content.players >= 6):
                subset.append(len(decompose(content, distances[0])))
        if subset:
            print(f"{label:10s} n={len(subset):2d}  regions median={statistics.median(subset):5.1f} "
                  f"min={min(subset):3d} max={max(subset):3d}")

    if arguments.json:
        os.makedirs(os.path.dirname(os.path.abspath(arguments.json)), exist_ok=True)
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(export, handle, indent=1, ensure_ascii=False)
        print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
