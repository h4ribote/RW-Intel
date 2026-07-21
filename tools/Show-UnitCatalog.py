"""Reports the built-in unit catalogue and checks whether price works as a measure of military value.

The commander needs one number per unit to weigh a trade, a loss budget, or an army composition against another.
Price is the obvious candidate because it is exact, already in the same currency as the economy, and readable from a live unit through its type. This report shows what the catalogue actually contains and how well price tracks the things that decide a fight.

    python tools/Show-UnitCatalog.py
    python tools/Show-UnitCatalog.py --json local/units.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.data import AssetPaths, read_unit_catalog


def correlation(xs, ys) -> float:
    """Pearson correlation on logs, since prices and durability both span orders of magnitude."""
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 3:
        return float("nan")
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    mx, my = statistics.mean(xs2), statistics.mean(ys2)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs2) * sum((y - my) ** 2 for y in ys2))
    return numerator / denominator if denominator else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", default=None, help="asset directory, defaults to local/rw/assets")
    parser.add_argument("--json", default=None, help="write the whole catalogue to this file")
    parser.add_argument("--all", action="store_true", help="list every unit rather than the combat units")
    arguments = parser.parse_args()

    paths = AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default()
    catalog = read_unit_catalog(paths)

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

    if arguments.json:
        os.makedirs(os.path.dirname(os.path.abspath(arguments.json)), exist_ok=True)
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump({name: vars(unit) for name, unit in sorted(catalog.items())}, handle, indent=1, ensure_ascii=False)
        print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
