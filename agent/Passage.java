import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The ground each movement type can cross, as connected components of the path finder's own grids, for both halves to decide what can reach what.
 *
 * Read once an episode's map is loaded, kept by the world for the reachability a contract and a lift are checked against, and sent to the control process, again to one that reconnects. A tile belongs to a component when its movement type can cross it and it joins the rest edge to edge; a blocked tile carries {@link #BLOCKED}. Air crosses everything and has no grid.
 *
 * Body, little-endian: width u16, height u16, grids u8; per grid: name length u8, name ASCII, components u16, runs u32, then the runs row by row from the top left, each a label u16 and a length u16.
 */
final class Passage {

    /** The movement types with a grid, by the engine's own names. */
    static final String[] MOVEMENTS = {"LAND", "OVER_CLIFF", "HOVER", "WATER", "OVER_CLIFF_WATER"};

    /** The movement type that crosses everything. */
    static final String AIR = "AIR";

    /** The label of a tile its movement type cannot cross. */
    static final int BLOCKED = 0xFFFF;

    /** World units to a tile. */
    static final float TILE = 20f;

    /** How many tiles around a blocked position the component of a position is looked for in, matching `Passage.component_at` in `rwintel/wire/terrain.py`. */
    static final int REACH = 3;

    /** The component labels of every movement type's grid, and how many components each has. */
    static final class Grids {
        int width;
        int height;
        final Map<String, int[]> labels = new LinkedHashMap<String, int[]>();
        final Map<String, Integer> counts = new LinkedHashMap<String, Integer>();

        /** The component of a world position for a movement type: the one under it, or the first found within {@link #REACH} tiles for a position on a tile it cannot cross; -1 when there is none or no grid for the movement type, and 0 everywhere for air. */
        int componentAt(String movement, float x, float y) {
            if (AIR.equals(movement)) return 0;
            int[] grid = labels.get(movement);
            if (grid == null || width == 0) return -1;
            int column = Math.min(width - 1, Math.max(0, (int) Math.floor(x / TILE)));
            int row = Math.min(height - 1, Math.max(0, (int) Math.floor(y / TILE)));
            int here = grid[row * width + column];
            if (here != BLOCKED) return here;
            for (int distance = 1; distance <= REACH; distance++) {
                for (int dy = -distance; dy <= distance; dy++) {
                    for (int dx = -distance; dx <= distance; dx++) {
                        int nx = column + dx;
                        int ny = row + dy;
                        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                        int label = grid[ny * width + nx];
                        if (label != BLOCKED) return label;
                    }
                }
            }
            return -1;
        }

        /** Whether a unit of the movement type can go from one world position to the other under its own power. */
        boolean reachable(String movement, float x0, float y0, float x1, float y1) {
            int from = componentAt(movement, x0, y0);
            return from >= 0 && from == componentAt(movement, x1, y1);
        }
    }

    private Passage() {
    }

    /** The grids of the map now loaded, labelled by component. Movement types the path finder keeps no grid for are left out. */
    static Grids read(Engine engine, Object game) {
        Grids grids = new Grids();
        for (String movement : MOVEMENTS) {
            int[] blocked = engine.blockedTiles(game, movement);
            if (blocked == null) continue;
            grids.width = blocked[0];
            grids.height = blocked[1];
            int[] labels = new int[grids.width * grids.height];
            grids.counts.put(movement, Integer.valueOf(label(blocked, grids.width, grids.height, labels)));
            grids.labels.put(movement, labels);
        }
        return grids;
    }

    static byte[] encode(Grids grids) {
        Map<String, int[]> runs = new LinkedHashMap<String, int[]>();
        int size = 5;
        for (Map.Entry<String, int[]> grid : grids.labels.entrySet()) {
            int[] encoded = runs(grid.getValue());
            runs.put(grid.getKey(), encoded);
            size += 1 + grid.getKey().length() + 2 + 4 + encoded.length * 2;
        }
        ByteBuffer out = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        out.putShort((short) grids.width);
        out.putShort((short) grids.height);
        out.put((byte) runs.size());
        for (Map.Entry<String, int[]> grid : runs.entrySet()) {
            byte[] name = grid.getKey().getBytes(java.nio.charset.StandardCharsets.US_ASCII);
            out.put((byte) name.length);
            out.put(name);
            out.putShort((short) grids.counts.get(grid.getKey()).intValue());
            int[] encoded = grid.getValue();
            out.putInt(encoded.length / 2);
            for (int value : encoded) out.putShort((short) value);
        }
        return out.array();
    }

    /** Labels the crossable tiles by connected component, edge to edge, in the order they are first met; returns how many components there are. */
    static int label(int[] blocked, int width, int height, int[] labels) {
        java.util.Arrays.fill(labels, BLOCKED);
        int[] queue = new int[width * height];
        int count = 0;
        for (int start = 0; start < width * height; start++) {
            if (labels[start] != BLOCKED || blocked[2 + start] != 0) continue;
            int head = 0;
            int tail = 0;
            queue[tail++] = start;
            labels[start] = count;
            while (head < tail) {
                int at = queue[head++];
                int x = at % width;
                int y = at / width;
                if (x + 1 < width) tail = visit(blocked, labels, queue, tail, at + 1, count);
                if (x > 0) tail = visit(blocked, labels, queue, tail, at - 1, count);
                if (y + 1 < height) tail = visit(blocked, labels, queue, tail, at + width, count);
                if (y > 0) tail = visit(blocked, labels, queue, tail, at - width, count);
            }
            count++;
        }
        return count;
    }

    private static int visit(int[] blocked, int[] labels, int[] queue, int tail, int at, int label) {
        if (labels[at] != BLOCKED || blocked[2 + at] != 0) return tail;
        labels[at] = label;
        queue[tail] = at;
        return tail + 1;
    }

    /** Runs of equal labels as label, length pairs, each length at most what a u16 holds. */
    static int[] runs(int[] labels) {
        java.util.List<Integer> out = new java.util.ArrayList<Integer>();
        int i = 0;
        while (i < labels.length) {
            int value = labels[i];
            int length = 1;
            while (i + length < labels.length && labels[i + length] == value && length < 0xFFFF) length++;
            out.add(Integer.valueOf(value));
            out.add(Integer.valueOf(length));
            i += length;
        }
        int[] encoded = new int[out.size()];
        for (int k = 0; k < encoded.length; k++) encoded[k] = out.get(k).intValue();
        return encoded;
    }

    // Movement classes.

    /** Movement types as the wire numbers them in a squad's passage byte; nought is none, or members that share no one narrowest type. */
    static final String[] CLASSES = {"", "LAND", "OVER_CLIFF", "HOVER", "WATER", "OVER_CLIFF_WATER", AIR};

    static int classOf(String movement) {
        for (int i = 1; i < CLASSES.length; i++) if (CLASSES[i].equals(movement)) return i;
        return 0;
    }

    /** Whether every tile the first movement type crosses the second crosses too, from the per-movement passability of land, water and the two heights of cliff. */
    static boolean within(String narrow, String wide) {
        if (narrow.equals(wide) || AIR.equals(wide)) return true;
        if ("LAND".equals(narrow)) return "OVER_CLIFF".equals(wide) || "HOVER".equals(wide) || "OVER_CLIFF_WATER".equals(wide);
        if ("OVER_CLIFF".equals(narrow) || "HOVER".equals(narrow)) return "OVER_CLIFF_WATER".equals(wide);
        if ("WATER".equals(narrow)) return "HOVER".equals(wide) || "OVER_CLIFF_WATER".equals(wide);
        return false;
    }

    /** The narrowest of a set of movement types: the one every other crosses all the ground of, or the empty name when the set is empty or holds two neither of which is within the other. */
    static String narrowest(java.util.Collection<String> movements) {
        String best = null;
        for (String movement : movements) {
            if (best == null || within(movement, best)) best = movement;
        }
        if (best == null) return "";
        for (String movement : movements) if (!within(best, movement)) return "";
        return best;
    }
}
