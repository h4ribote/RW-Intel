"""What the terrain reader promises: each movement type crosses exactly the tiles the engine's path finder lets it, properties on the editor's helper layers count for nothing, every resource point and starting unit is counted in the component it stands in, and the drawing is a PNG of the map at its scale. The map here is written by hand, so nothing needs the game installed; the one check on a shipped map is skipped where the game is not."""

from __future__ import annotations

import base64
import os
import struct
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from rwintel.data import AssetPaths, components, read_map, read_terrain, render_png
from rwintel.data.terrain import (
    AIR, COLOURS, HOVER, LAND, OVER_CLIFF, OVER_CLIFF_WATER, RESOURCE, WATER, WATER_MOVE, allowed_by,
    nearest_component,
)

#: The ground of the test map, one character a tile: land, water, large cliff and soft cliff.
GROUND = [
    "..^...~..#..",
    "..^...~..#..",
    "..^...~..#..",
    "..^...~..#..",
    "..^...~..#..",
    "..^...~..#..",
]
GROUND_GIDS = {".": 1, "~": 2, "#": 3, "^": 4}
#: Ground tileset ids 0-3 are plain, water, large cliff and soft cliff. Its tile 15 claims to be a command centre and lies past the ids the map reserved for the tileset, on an id the units tileset leaves plain (`PLAIN_UNIT`), so a tile placed there must be read as nothing.
UNITS_FIRST = 10
HQ_0, HQ_1, TREE, POOL = UNITS_FIRST, UNITS_FIRST + 1, UNITS_FIRST + 2, UNITS_FIRST + 3
PLAIN_UNIT = 1 + 15


def _layer(name: str, gids) -> str:
    data = base64.b64encode(zlib.compress(struct.pack("<%dI" % len(gids), *gids))).decode("ascii")
    return (f'<layer name="{name}" width="12" height="6"><data encoding="base64" compression="zlib">{data}</data>'
            f"</layer>")


def _write_map(folder: str) -> str:
    def tile(tile_id, **properties):
        rows = "".join(f'<property name="{key.replace("_", "-") if key != "res_pool" else key}" value="{value}"/>'
                       for key, value in properties.items())
        return f'<tile id="{tile_id}"><properties>{rows}</properties></tile>'

    ground = [GROUND_GIDS[char] for row in GROUND for char in row]
    units = [0] * (12 * 6)
    items = [0] * (12 * 6)
    helper = [0] * (12 * 6)
    units[0 * 12 + 0] = HQ_0
    units[5 * 12 + 10] = HQ_1
    units[1 * 12 + 1] = TREE
    units[3 * 12 + 3] = PLAIN_UNIT
    items[2 * 12 + 7] = POOL
    items[4 * 12 + 4] = POOL
    items[3 * 12 + 6] = POOL
    # Water painted on the editor's helper layer over the whole left half, which the engine does not read.
    for row in range(6):
        for column in range(6):
            helper[row * 12 + column] = GROUND_GIDS["~"]
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<map version="1.0" orientation="orthogonal" width="12" height="6" tilewidth="20" tileheight="20">'
        '<tileset firstgid="1" name="ground" tilewidth="20" tileheight="20">'
        + tile(1, water="") + tile(2, large_cliff="") + tile(3, cliff_soft="")
        + '<tile id="15"><properties><property name="unit" value="commandCenter"/><property name="team" value="5"/></properties></tile>'
        '</tileset>'
        f'<tileset firstgid="{UNITS_FIRST}" name="units" tilewidth="20" tileheight="20">'
        '<tile id="0"><properties><property name="unit" value="commandCenter"/><property name="team" value="0"/></properties></tile>'
        '<tile id="1"><properties><property name="unit" value="commandCenter"/><property name="team" value="1"/></properties></tile>'
        '<tile id="2"><properties><property name="unit" value="tree"/><property name="team" value="none"/></properties></tile>'
        '<tile id="3"><properties><property name="res_pool" value=""/></properties></tile>'
        '</tileset>'
        + _layer("Ground", ground) + _layer("Units", units) + _layer("Items", items) + _layer("set", helper) + "</map>")
    path = os.path.join(folder, "[p2]Test (2p).tmx")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def test_each_tile_property_lets_across_what_the_engine_lets_across():
    """The path finder's own grids decide this, and the table is pinned to what they say, property by property."""
    everything = {LAND, OVER_CLIFF, HOVER, WATER_MOVE, OVER_CLIFF_WATER, AIR}
    expected = {
        (): everything - {WATER_MOVE},
        ("small-rock",): everything - {WATER_MOVE},
        ("water",): {HOVER, WATER_MOVE, OVER_CLIFF_WATER, AIR},
        ("cliff-soft",): {OVER_CLIFF, HOVER, OVER_CLIFF_WATER, AIR},
        ("cliff",): {OVER_CLIFF, HOVER, OVER_CLIFF_WATER, AIR},
        ("large-cliff",): {OVER_CLIFF, OVER_CLIFF_WATER, AIR},
        ("trees",): {OVER_CLIFF, OVER_CLIFF_WATER, AIR},
        ("res_pool",): {OVER_CLIFF, HOVER, OVER_CLIFF_WATER, AIR},
        ("lava",): {AIR},
        ("lava-cliff",): {AIR},
        ("large-rock",): {AIR},
        ("cliff-soft", "water"): {HOVER, OVER_CLIFF_WATER, AIR},
    }
    for properties, allowed in expected.items():
        assert set(allowed_by(properties)) == allowed, properties


def test_each_movement_type_is_parted_by_what_stops_it():
    with tempfile.TemporaryDirectory() as folder:
        terrain = read_terrain(_write_map(folder), AssetPaths(folder))
    assert terrain.kind_at((6, 0)) == WATER and abs(terrain.share(WATER) - 1 / 12) < 1e-9
    # The helper layer's water over the left half counts for nothing.
    assert terrain.kind_at((0, 0)) == "land"
    sizes = {movement: sorted((c.tiles for c in components(terrain, movement)[0]), reverse=True)
             for movement in (LAND, HOVER, OVER_CLIFF, WATER_MOVE, AIR)}
    # Land is cut by the soft cliff, the water and the large cliff, and loses the two pools that stand on land.
    assert sizes[LAND] == [17, 12, 12, 11]
    # A hovercraft crosses the water and the soft cliff but not the large one.
    assert sizes[HOVER] == [54, 12]
    # A mech climbs both cliffs but stops at the water.
    assert sizes[OVER_CLIFF] == [36, 30]
    # The pool standing in the water column stops boats as well, which cuts the water in two.
    assert sizes[WATER_MOVE] == [3, 2]
    assert sizes[AIR] == [72]


def test_resource_points_and_starting_units_are_counted_in_the_component_they_stand_in():
    with tempfile.TemporaryDirectory() as folder:
        terrain = read_terrain(_write_map(folder), AssetPaths(folder))
    found, labels = components(terrain, LAND)
    by_size = {c.tiles: c for c in found}
    # The pools are not crossable by land, so each is counted in the land beside it.
    assert by_size[17].resources == 2 and by_size[11].resources == 1
    assert [c.units for c in found if c.units] == [[("commandCenter", "0")], [("commandCenter", "1")]]
    assert all(mark.kind != "tree" for mark in terrain.marks)
    assert sum(1 for mark in terrain.marks if mark.kind == RESOURCE) == 3
    assert nearest_component(labels, (6, 3), reach=0) == -1 and nearest_component(labels, (6, 3)) >= 0


def test_the_map_reader_still_clamps_each_tileset_to_its_reserved_ids():
    with tempfile.TemporaryDirectory() as folder:
        content = read_map(_write_map(folder), AssetPaths(folder))
    assert content.spawns == [(0, 0), (10, 5)]
    assert content.resources == [(4, 4), (6, 3), (7, 2)]


def test_the_drawing_is_a_png_of_the_map_at_its_scale():
    with tempfile.TemporaryDirectory() as folder:
        terrain = read_terrain(_write_map(folder), AssetPaths(folder))
        path = render_png(terrain, os.path.join(folder, "map.png"), scale=3)
        with open(path, "rb") as handle:
            data = handle.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (36, 18)
    start = data.index(b"IDAT") + 4
    length = struct.unpack(">I", data[start - 8:start - 4])[0]
    rows = np.frombuffer(zlib.decompress(data[start:start + length]), dtype=np.uint8).reshape(height, 1 + width * 3)
    pixels = rows[:, 1:].reshape(height, width, 3)
    # The water tile at the top of the water column is drawn in the water colour, and the marks of the command centre and the resource points in theirs.
    assert tuple(pixels[0 * 3 + 1, 6 * 3 + 1]) == COLOURS[WATER]
    assert tuple(pixels[0 * 3 + 1, 0 * 3 + 1]) == COLOURS["commandCenter"]
    assert tuple(pixels[2 * 3 + 1, 7 * 3 + 1]) == COLOURS[RESOURCE]


def test_beach_landing_puts_each_side_on_an_island_of_its_own_apart_from_the_largest():
    assets = AssetPaths.default()
    path = os.path.join(assets.skirmish_maps, "[p2]Beach landing (2p) [by hxyy].tmx")
    if not os.path.exists(path):
        pytest.skip("the game's assets are not installed")
    terrain = read_terrain(path, assets)
    found, labels = components(terrain, LAND)
    centres = [nearest_component(labels, mark.cell) for mark in terrain.marks if mark.kind == "commandCenter"]
    assert len(centres) == 2 and len(set(centres)) == 2 and 0 not in centres
    assert found[0].resources > max(found[c].resources for c in centres)
    # A hovercraft reaches the whole map from either side.
    hover, hover_labels = components(terrain, HOVER)
    assert {nearest_component(hover_labels, mark.cell) for mark in terrain.marks if mark.kind == "commandCenter"} == {0}
