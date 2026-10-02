"""Which regions a movement type reaches, and where transports pick up and set down, on a hand-drawn map.

The map is two islands of land parted by a strait of water, with a resource region on each. Land units reach only their own island, hovercraft and ships reach both shores, and a hovercraft sets down on land of the far island's own land component.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.reach import Reach, members_reach, narrowest, within
from rwintel.wire.terrain import TILE, decode_terrain, encode_terrain

#: 12 columns: land 0-3, water 4-7, land 8-11.
WIDTH, HEIGHT = 12, 6


def _passage():
    land = np.full((HEIGHT, WIDTH), -1)
    land[:, 0:4] = 0
    land[:, 8:12] = 1
    water = np.full((HEIGHT, WIDTH), -1)
    water[:, 4:8] = 0
    hover = np.zeros((HEIGHT, WIDTH), dtype=int)
    return decode_terrain(encode_terrain(WIDTH, HEIGHT, {"LAND": land, "HOVER": hover, "WATER": water}))


def _at(column, row):
    return ((column + 0.5) * TILE, (row + 0.5) * TILE)


REGIONS = [SimpleNamespace(id=0, x=_at(1, 2)[0], y=_at(1, 2)[1]), SimpleNamespace(id=1, x=_at(10, 3)[0], y=_at(10, 3)[1])]


def test_land_reaches_its_own_island_and_hover_and_air_reach_both():
    reach = Reach(_passage(), REGIONS)
    home = _at(2, 2)
    assert reach.walkable("LAND", *home, 0) and not reach.walkable("LAND", *home, 1)
    assert reach.walkable("HOVER", *home, 1) and reach.walkable("AIR", *home, 1)
    assert members_reach(reach, [("LAND", *home), ("HOVER", *home)], 0)
    assert not members_reach(reach, [("LAND", *home), ("HOVER", *home)], 1)


def test_a_region_is_reached_by_water_from_the_water_off_its_shore():
    reach = Reach(_passage(), REGIONS)
    assert reach.walkable("WATER", *_at(5, 1), 1)
    assert reach.regions[1].coastal and reach.regions[0].coastal


def test_a_ship_is_not_sent_to_a_region_whose_centre_stands_inland():
    """The game side orders a contract out only to members that reach its target point, the region's centre, within the passage's own few tiles of it; water further off still reaches the region for landings, but gives a ship no way to walk there."""
    inland = SimpleNamespace(id=2, x=_at(0, 2)[0], y=_at(0, 2)[1])
    reach = Reach(_passage(), REGIONS + [inland])
    sea = _at(5, 1)
    assert reach.regions[2].components["WATER"] >= 0
    assert not reach.walkable("WATER", *sea, 2) and not members_reach(reach, [("WATER", *sea)], 2)
    assert reach.walkable("LAND", *_at(2, 4), 2)
    for region in reach.regions.values():
        for movement in ("LAND", "HOVER", "WATER"):
            centre = (region.x, region.y)
            assert reach.walkable(movement, *sea, region.region) == reach.passage.reachable(movement, sea, centre)


def test_a_hovercraft_sets_down_on_the_far_islands_own_land():
    reach = Reach(_passage(), REGIONS)
    x, y = reach.landing(1, "HOVER")
    assert x >= 8 * TILE
    assert reach.walkable("LAND", x, y, 1)
    assert reach.landing(1, "AIR") == reach.landing(1, "HOVER")


def test_a_pickup_is_where_both_the_passengers_and_the_transport_can_go():
    reach = Reach(_passage(), REGIONS)
    tank = _at(1, 4)
    assert reach.pickup("LAND", tank, "HOVER", _at(6, 0)) == tank
    # A ship cannot come onto land, so nothing on the tank's island is a place it can pick from.
    assert reach.pickup("LAND", tank, "WATER", _at(6, 0)) is None


def test_a_ship_builds_from_the_water_within_its_reach():
    reach = Reach(_passage(), REGIONS)
    shore = _at(3, 2)
    inland = _at(0, 2)
    assert reach.ship_site(shore, _at(6, 3), 2 * TILE) is not None
    assert reach.ship_site(inland, _at(6, 3), 2 * TILE) is None


def _uncached_approaches(reach, movement, start, point, tiles):
    """`Reach.approaches` as a scan of the whole map, without the cache or the window."""
    if movement == "AIR":
        return True
    mask = reach._component_mask(movement, start)
    return mask is not None and reach._nearest(mask, point, tiles) is not None


def test_the_cached_approach_answers_what_a_scan_of_the_whole_map_answers():
    """A map of several components per movement type, blocked tiles scattered through it, with points near every edge."""
    rng = np.random.default_rng(11)
    width, height = 47, 33
    rows, columns = np.mgrid[0:height, 0:width]
    land = np.where(rng.random((height, width)) < 0.3, -1, columns // 9 + 6 * (rows // 11))
    water = np.where(rng.random((height, width)) < 0.6, -1, (columns + rows) // 13)
    hover = np.where(rng.random((height, width)) < 0.1, -1, rows // 17)
    passage = decode_terrain(encode_terrain(width, height, {"LAND": land, "WATER": water, "HOVER": hover}))
    reach = Reach(passage, [SimpleNamespace(id=0, x=_at(3, 3)[0], y=_at(3, 3)[1])])
    uncached = Reach(passage, [])

    def world():
        return float(rng.uniform(-TILE, (width + 1) * TILE)), float(rng.uniform(-TILE, (height + 1) * TILE))

    asked = [(movement, world(), world(), int(tiles)) for movement in ("LAND", "WATER", "HOVER", "AIR", "OVER_CLIFF")
             for tiles in (0, 1, 3, 8, 30) for _ in range(60)]
    answers = [reach.approaches(movement, start, point, tiles) for movement, start, point, tiles in asked]
    assert answers == [_uncached_approaches(uncached, *query) for query in asked]
    assert any(answers) and not all(answers)
    assert answers == [reach.approaches(movement, start, point, tiles) for movement, start, point, tiles in asked]
    assert reach._approaches and not uncached._approaches


def test_the_narrowest_movement_is_what_the_whole_squad_can_cross():
    assert narrowest(["LAND", "HOVER"]) == "LAND"
    assert narrowest(["WATER", "HOVER"]) == "WATER"
    assert narrowest(["LAND", "WATER"]) == ""
    assert within("LAND", "OVER_CLIFF") and not within("OVER_CLIFF", "HOVER")
