"""Reports what the game ships on disk, without launching it.

    python -m rwintel.data regions
    python -m rwintel.data regions --distances 300 400 500
    python -m rwintel.data regions --json
    python -m rwintel.data units
    python -m rwintel.data units --all --json
    python -m rwintel.data terrain Beach Lake "Big Island" --png

`regions` decomposes every built-in skirmish map into regions, which are the names the command layers use for places, so their count and size decide the shape of the operational layer's action space. `units` lists the unit catalogue the definition files describe and checks how well price tracks what decides a fight, which is what makes price usable as the measure of military value. `terrain` reports what a map's ground is made of and, per movement type, which connected components it can cross and what each side starts with in which, and with `--png` draws it.

`--json` without a path writes to local/reports/regions.json or local/reports/units.json; `--png` without a directory writes to local/reports/terrain/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import List, Sequence

from .. import paths
from . import AssetPaths, components, decompose, list_skirmish_maps, read_map, read_terrain, read_unit_catalog, render_png
from .regions import DEFAULT_MERGE_DISTANCE
from .terrain import COLOURS, KINDS, MOVEMENTS, OTHER_UNIT, nearest_component

#: The region slot count of the observation, which a map has to decompose within to be playable as it stands.
REGION_SLOTS = 24


def _write_json(path: str, content) -> None:
    with paths.replacing(path) as handle:
        json.dump(content, handle, indent=1, ensure_ascii=False)
    print(f"\nwrote {path}")


def regions(arguments) -> int:
    assets = AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default()
    maps = list_skirmish_maps(assets)
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
    contents = [read_map(path, assets) for path in maps]

    for content in contents:
        cells = []
        for distance in distances:
            found = decompose(content, distance)
            counts[distance].append(len(found))
            radii[distance].extend(r.radius for r in found)
            cells.append(f"{len(found):6d}")
            if distance == distances[0]:
                export[content.name] = {
                    "width": content.width,
                    "height": content.height,
                    "tile_size": content.tile_size,
                    "players": content.players,
                    "merge_distance": distance,
                    "regions": [vars(r) for r in found],
                }
        print(f"{content.name[:46]:46s} {content.width:4d}x{content.height:<4d} {content.players:2d} "
              f"{len(content.resources):4d} " + " ".join(cells))

    print()
    print(f"{'distance':>9s} {'median':>7s} {'mean':>7s} {'min':>5s} {'max':>5s} {'p95radius':>10s} {'<=' + str(REGION_SLOTS):>6s}")
    for distance in distances:
        values = counts[distance]
        within = sum(1 for v in values if v <= REGION_SLOTS)
        percentile = sorted(radii[distance])[int(0.95 * (len(radii[distance]) - 1))]
        print(f"{distance:9.0f} {statistics.median(values):7.1f} {statistics.mean(values):7.1f} "
              f"{min(values):5d} {max(values):5d} {percentile:10.0f} {within:5d}/{len(values)}")

    print()
    for players in sorted({content.players for content in contents}):
        subset = [len(decompose(content, distances[0])) for content in contents if content.players == players]
        print(f"{players:2d} players  n={len(subset):2d}  regions median={statistics.median(subset):5.1f} "
              f"min={min(subset):3d} max={max(subset):3d}")

    if arguments.json is not None:
        _write_json(arguments.json or os.path.join(paths.reports(), "regions.json"), export)
    return 0


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation on logs, since prices and durability both span orders of magnitude."""
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 3:
        return float("nan")
    xs2: List[float] = [p[0] for p in pairs]
    ys2: List[float] = [p[1] for p in pairs]
    mx, my = statistics.mean(xs2), statistics.mean(ys2)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs2) * sum((y - my) ** 2 for y in ys2))
    return numerator / denominator if denominator else float("nan")


def units(arguments) -> int:
    assets = AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default()
    catalog = read_unit_catalog(assets)

    priced = [u for u in catalog.values() if u.price > 0]
    buildings = [u for u in priced if u.is_building]
    mobile = [u for u in priced if not u.is_building]
    fighters = [u for u in mobile if u.can_attack and u.damage_per_second > 0 and u.max_hp > 0]

    print(f"definitions read: {len(catalog)}   priced: {len(priced)}   buildings: {len(buildings)}   mobile: {len(mobile)}   armed mobile with a resolved weapon: {len(fighters)}")
    print()

    listed = sorted(priced if arguments.all else fighters, key=lambda u: (u.tech_level, u.price, u.name))
    print(f"{'name':26s} {'tech':>4s} {'price':>6s} {'hp':>7s} {'dps':>7s} {'range':>6s} {'sight':>6s} {'speed':>6s} {'move':>9s} {'built from':22s}")
    for unit in listed:
        print(f"{unit.name[:26]:26s} {unit.tech_level:4d} {unit.price:6d} {unit.max_hp:7.0f} "
              f"{unit.damage_per_second:7.1f} {unit.attack_range:6.0f} {unit.sight_range:6.0f} "
              f"{unit.move_speed:6.2f} {unit.movement_type[:9]:9s} {','.join(unit.built_from)[:22]:22s}")

    print()
    print("price against what decides a fight, over the armed mobile units (log-log correlation):")
    print(f"  price vs hp            {correlation([u.price for u in fighters], [u.max_hp for u in fighters]):.2f}")
    print(f"  price vs dps           {correlation([u.price for u in fighters], [u.damage_per_second for u in fighters]):.2f}")
    print(f"  price vs hp*dps        {correlation([u.price for u in fighters], [u.max_hp * u.damage_per_second for u in fighters]):.2f}")
    print(f"  price vs range         {correlation([u.price for u in fighters], [u.attack_range for u in fighters]):.2f}")

    ranges = [u.attack_range for u in fighters if u.attack_range > 0]
    sights = [u.sight_range for u in priced if u.sight_range > 0]
    print()
    print(f"attack range over armed mobile units: median {statistics.median(ranges):.0f}  max {max(ranges):.0f} world units")
    print(f"sight range over priced units:        median {statistics.median(sights):.0f}  max {max(sights):.0f} world units")

    producers = sorted((u for u in priced if u.builds), key=lambda u: -len(u.builds))
    print()
    print("producers:")
    for unit in producers[:12]:
        print(f"  {unit.name:22s} tech {unit.tech_level}  price {unit.price:6d}  builds {len(unit.builds):2d}: {', '.join(unit.builds[:8])}")

    if arguments.json is not None:
        _write_json(arguments.json or os.path.join(paths.reports(), "units.json"),
                    {name: vars(unit) for name, unit in sorted(catalog.items())})
    return 0


def find_maps(names: Sequence[str], assets: AssetPaths) -> List[str]:
    """The skirmish map each name means: a TMX path as it is, otherwise the one built-in map whose title (the file name after its `[pN]` tag) starts with it, or failing that the one whose file name contains it, ignoring case. A name matching none or several is refused with what it could have meant."""
    found = []
    shipped = list_skirmish_maps(assets)

    def title(path: str) -> str:
        return os.path.basename(path).split("]", 1)[-1].strip().lower()

    for name in names:
        if os.path.isfile(name):
            found.append(name)
            continue
        matches = [path for path in shipped if title(path).startswith(name.lower())]
        if len(matches) != 1:
            matches = [path for path in shipped if name.lower() in os.path.basename(path).lower()]
        if len(matches) != 1:
            listed = ", ".join(os.path.basename(path) for path in (matches or shipped))
            raise SystemExit(f"{name!r} names {'no' if not matches else 'more than one'} skirmish map: {listed}")
        found.append(matches[0])
    return found


def terrain(arguments) -> int:
    assets = AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default()
    maps = find_maps(arguments.maps, assets)
    directory = None
    if arguments.png is not None:
        directory = arguments.png or os.path.join(paths.reports(), "terrain")
        os.makedirs(directory, exist_ok=True)
        print("colours: " + ", ".join(f"{name} {rgb}" for name, rgb in COLOURS.items()) + f", other units {OTHER_UNIT}")
    for path in maps:
        ground = read_terrain(path, assets)
        print()
        print(f"{ground.name}: {ground.width}x{ground.height} tiles of {ground.tile_size}, "
              + ", ".join(f"{kind} {ground.share(kind):.2f}" for kind in KINDS if ground.share(kind) > 0))
        for movement in arguments.movement or ("LAND", "HOVER", "WATER"):
            found, labels = components(ground, movement)
            print(f"  {movement}: {len(found)} component(s)")
            shown = found[:arguments.components]
            for component in shown:
                units = ", ".join(f"{unit}:{team}" for unit, team in component.units) or "-"
                print(f"    component {component.index}: {component.tiles} tiles, x {component.x[0]}-{component.x[1]}, "
                      f"y {component.y[0]}-{component.y[1]}, resources {component.resources}, starting units {units}")
            if len(found) > len(shown):
                rest = found[len(shown):]
                print(f"    {len(rest)} smaller component(s) of {sum(c.tiles for c in rest)} tiles holding "
                      f"{sum(c.resources for c in rest)} resource point(s)")
            off = [mark for mark in ground.marks if nearest_component(labels, mark.cell) < 0]
            if off:
                print(f"    out of reach: "
                      + ", ".join(f"{mark.kind}{':' + mark.team if mark.team else ''} at {mark.cell}" for mark in off))
        if directory is not None:
            print(f"  wrote {render_png(ground, os.path.join(directory, ground.name + '.png'), arguments.scale)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rwintel.data", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("regions", help="decompose the built-in skirmish maps into regions")
    p.add_argument("--distances", type=float, nargs="+", default=[DEFAULT_MERGE_DISTANCE],
                   help="merge distances in world units to compare")
    p.add_argument("--json", nargs="?", const="", default=None,
                   help="write the decomposition at the first distance to this file, or to local/reports/regions.json")
    p.set_defaults(handler=regions)

    p = commands.add_parser("units", help="list the unit catalogue and check price against combat value")
    p.add_argument("--all", action="store_true", help="list every priced unit rather than the armed mobile ones")
    p.add_argument("--json", nargs="?", const="", default=None,
                   help="write the whole catalogue to this file, or to local/reports/units.json")
    p.set_defaults(handler=units)

    p = commands.add_parser("terrain", help="report and draw the ground of skirmish maps and what each movement type can reach")
    p.add_argument("maps", nargs="+", help="part of a built-in map's file name, or the path of a TMX file")
    p.add_argument("--movement", action="append", choices=list(MOVEMENTS),
                   help="movement type to list the connected components of; repeated for several, LAND HOVER WATER unless given")
    p.add_argument("--components", type=int, default=6, help="components to list one by one, the largest first")
    p.add_argument("--png", nargs="?", const="", default=None,
                   help="draw each map into this directory, or into local/reports/terrain")
    p.add_argument("--scale", type=int, default=4, help="pixels to a tile in the drawing")
    p.set_defaults(handler=terrain)

    for sub in commands.choices.values():
        sub.add_argument("--assets", default=None, help=f"the game's asset directory, defaulting to local/{paths.GAME_DIRECTORY}/assets")

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
