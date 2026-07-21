import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The agent's own view of the match: the region table, squad membership, and whatever the last scan of the world found.
 *
 * Squads live here rather than in the control process only in the sense of being addressable: which units belong to which squad is decided by a policy and arrives with the actions. The game side keeps the membership because it is what lets it address a squad with one command and describe it back in the observation.
 *
 * Regions likewise arrive from the control process. Deriving them needs the map file, which is read there; doing it here as well would put the same rule in two implementations, and the one that can be checked without launching the game is the better place for it.
 */
final class World {

    /** A building standing on a resource point is within this of it, and nothing else is. */
    private static final float ON_RESOURCE = 40f;

    static final class Region {
        final int id;
        final float x;
        final float y;
        final int resources;
        final float distanceFromHome;
        int heldByUs;
        int heldByEnemy;
        float ourValue;
        float enemyValue;
        int enemySeenAtMs;

        Region(int id, float x, float y, int resources, float distanceFromHome) {
            this.id = id;
            this.x = x;
            this.y = y;
            this.resources = resources;
            this.distanceFromHome = distanceFromHome;
        }
    }

    static final class Squad {
        final int id;
        int commander;
        final List<Long> units = new ArrayList<Long>();
        float value;
        float formedValue;
        float x;
        float y;
        int task;
        int status;
        int targetRegion;
        /** Value lost since the current contract was issued, which is what the contract's budget is spent against. */
        float losses;
        int costBudget;
        int deadlineMs;
        /** Value held when the contract was issued, the baseline the losses are measured from. */
        float valueAtIssue;
        int issuedAtMs;

        Squad(int id) {
            this.id = id;
        }
    }

    static final class Seen {
        long id;
        int squad;
        int typeIndex;
        float x;
        float y;
        float health;
        float maxHealth;
        int built;
        int orders;
        int targetSquad;
        int stance;
        /** Game time since the unit was last hit, which is the cheap trigger for a squad needing attention. */
        int sinceHitMs;
        boolean hostile;
        Object handle;
        boolean building;
        int price;
    }

    private final Engine engine;

    final List<Region> regions = new ArrayList<Region>();
    final Map<Integer, Squad> squads = new LinkedHashMap<Integer, Squad>();
    final List<Seen> visible = new ArrayList<Seen>();

    /** Reported type name to the index the control process knows the type by, and back again. */
    private final Map<String, Integer> typeIndex = new HashMap<String, Integer>();
    private final List<Object> typesByIndex = new ArrayList<Object>();

    private final Map<Long, Integer> unitToSquad = new HashMap<Long, Integer>();
    private final Map<Long, Object> unitHandles = new HashMap<Long, Object>();

    /** Every enemy is reported while this holds; otherwise only what the engine says is visible. */
    boolean omniscient = true;

    int commandedUnits;
    private float homeX = Float.NaN;
    private float homeY = Float.NaN;
    private String resourceTypeName = "";

    World(Engine engine) {
        this.engine = engine;
    }

    void indexTypes(List<Object> types) throws Exception {
        typeIndex.clear();
        typesByIndex.clear();
        for (Object type : types) {
            String name = engine.typeName(type);
            if (typeIndex.containsKey(name)) continue;
            typeIndex.put(name, Integer.valueOf(typesByIndex.size()));
            typesByIndex.add(type);
        }
        Object resource = engine.typeNamed("crystalResource");
        resourceTypeName = resource == null ? "" : engine.typeName(resource);
    }

    int indexOf(String typeName) {
        Integer index = typeIndex.get(typeName);
        return index == null ? 0xFFFF : index.intValue();
    }

    Object typeAt(int index) {
        return index >= 0 && index < typesByIndex.size() ? typesByIndex.get(index) : null;
    }

    List<Object> types() {
        return typesByIndex;
    }

    /** Replaces the region table, which happens once per episode when the control process has seen which map is loaded. */
    void setRegions(List<float[]> table) {
        regions.clear();
        for (int i = 0; i < table.size(); i++) {
            float[] row = table.get(i);
            regions.add(new Region(i, row[0], row[1], (int) row[2], row[3]));
        }
    }

    void reset() {
        squads.clear();
        unitToSquad.clear();
        unitHandles.clear();
        visible.clear();
        regions.clear();
        homeX = Float.NaN;
        homeY = Float.NaN;
    }

    Squad squad(int id) {
        Squad squad = squads.get(Integer.valueOf(id));
        if (squad == null) squads.put(Integer.valueOf(id), squad = new Squad(id));
        return squad;
    }

    /** Applies a membership decision. A unit belongs to at most one squad, so it is taken out of whatever held it. */
    void assign(int squadId, List<Long> units) {
        Squad squad = squad(squadId);
        for (Long unit : squad.units) unitToSquad.remove(unit);
        squad.units.clear();
        for (Long unit : units) {
            Integer previous = unitToSquad.get(unit);
            if (previous != null) {
                Squad old = squads.get(previous);
                if (old != null) old.units.remove(unit);
            }
            squad.units.add(unit);
            unitToSquad.put(unit, Integer.valueOf(squadId));
        }
    }

    Object handle(long id) {
        return unitHandles.get(Long.valueOf(id));
    }

    /**
     * Rescans the world and recomputes everything derived from it.
     * A full scan every period is affordable: the engine keeps no per player index, but reading a unit costs well under a microsecond once the code is warm, so a few hundred units is a fraction of a step.
     */
    void refresh(Object game, Object self) throws Exception {
        int selfTeam = engine.team(self);
        int now = engine.gameTime(game);

        for (Region region : regions) {
            region.heldByUs = 0;
            region.heldByEnemy = 0;
            region.ourValue = 0f;
            region.enemyValue = 0f;
        }
        visible.clear();
        unitHandles.clear();
        commandedUnits = 0;

        List<Seen> resources = new ArrayList<Seen>();
        List<Seen> buildings = new ArrayList<Seen>();
        Map<Long, Float> valueById = new HashMap<Long, Float>();

        Object[] units = engine.unitArray();
        int count = engine.unitCount();
        for (int i = 0; i < count && i < units.length; i++) {
            Object unit = units[i];
            if (unit == null || engine.dead(unit)) continue;
            Object owner = engine.owner(unit);
            if (owner == null) continue;
            Object type = engine.type(unit);
            if (type == null) continue;

            int team = engine.team(owner);
            boolean ours = team == selfTeam;
            boolean hostile = !ours;
            if (hostile && !omniscient && !engine.visibleTo(unit, self)) continue;

            Seen seen = new Seen();
            seen.handle = unit;
            seen.id = engine.id(unit);
            seen.x = engine.x(unit);
            seen.y = engine.y(unit);
            seen.health = engine.health(unit);
            seen.maxHealth = engine.maxHealth(unit);
            float progress = engine.built(unit);
            seen.built = (int) Math.max(0f, Math.min(255f, progress * 255f));
            seen.hostile = hostile;
            seen.sinceHitMs = Math.max(0, now - engine.lastHit(unit));
            String typeName = engine.typeName(type);
            seen.typeIndex = indexOf(typeName);
            seen.building = engine.typeIsBuilding(type);
            seen.price = engine.price(unit);
            if (engine.armedClass.isInstance(unit)) {
                seen.stance = engine.stanceOf(unit);
                seen.orders = engine.orderCount(unit);
                Object target = engine.attackTarget(unit);
                seen.targetSquad = target == null ? 0xFFFF : squadOf(engine.id(target));
            } else {
                seen.targetSquad = 0xFFFF;
            }
            Integer squadId = unitToSquad.get(Long.valueOf(seen.id));
            seen.squad = squadId == null ? 0xFFFF : squadId.intValue();

            unitHandles.put(Long.valueOf(seen.id), unit);
            valueById.put(Long.valueOf(seen.id), Float.valueOf(seen.price));
            visible.add(seen);

            if (typeName.equals(resourceTypeName)) {
                resources.add(seen);
                continue;
            }
            if (seen.building) buildings.add(seen);
            if (ours) {
                commandedUnits++;
                if (Float.isNaN(homeX) && seen.building) {
                    homeX = seen.x;
                    homeY = seen.y;
                }
            }

            Region region = nearest(seen.x, seen.y);
            if (region != null) {
                if (ours) region.ourValue += seen.price;
                else {
                    region.enemyValue += seen.price;
                    region.enemySeenAtMs = now;
                }
            }
        }

        for (Seen resource : resources) {
            Region region = nearest(resource.x, resource.y);
            if (region == null) continue;
            for (Seen building : buildings) {
                if (building.built < 255) continue;
                float dx = building.x - resource.x;
                float dy = building.y - resource.y;
                if (dx * dx + dy * dy > ON_RESOURCE * ON_RESOURCE) continue;
                if (building.hostile) region.heldByEnemy++;
                else region.heldByUs++;
                break;
            }
        }

        refreshSquads(valueById, now);
    }

    private int squadOf(long unitId) {
        Integer squadId = unitToSquad.get(Long.valueOf(unitId));
        return squadId == null ? 0xFFFF : squadId.intValue();
    }

    private void refreshSquads(Map<Long, Float> valueById, int now) {
        for (Squad squad : squads.values()) {
            float value = 0f;
            float sumX = 0f;
            float sumY = 0f;
            int alive = 0;
            for (int i = squad.units.size() - 1; i >= 0; i--) {
                Long id = squad.units.get(i);
                Float price = valueById.get(id);
                if (price == null) {
                    // Gone from the world, so gone from the squad. The loss stays on the books through valueAtIssue.
                    squad.units.remove(i);
                    unitToSquad.remove(id);
                    continue;
                }
                Object handle = unitHandles.get(id);
                value += price.floatValue();
                alive++;
                try {
                    sumX += engine.x(handle);
                    sumY += engine.y(handle);
                } catch (Exception ignored) {
                    // a unit that vanished between the scan and here simply does not move the centre
                }
            }
            squad.value = value;
            if (alive > 0) {
                squad.x = sumX / alive;
                squad.y = sumY / alive;
            }
            if (squad.formedValue <= 0f) squad.formedValue = value;
            squad.losses = Math.max(0f, squad.valueAtIssue - value);
            squad.status = statusOf(squad, now);
        }
    }

    private int statusOf(Squad squad, int now) {
        if (squad.units.isEmpty()) return 3;                                   // nothing left to run the task
        if (squad.deadlineMs > 0 && now > squad.deadlineMs) return 4;          // expired
        if (squad.costBudget > 0 && squad.losses > squad.costBudget * 0.7f) return 2;  // losing
        return 0;
    }

    Region nearest(float x, float y) {
        Region best = null;
        float bestDistance = Float.MAX_VALUE;
        for (Region region : regions) {
            float dx = region.x - x;
            float dy = region.y - y;
            float distance = dx * dx + dy * dy;
            if (distance < bestDistance) {
                bestDistance = distance;
                best = region;
            }
        }
        return best;
    }

    Region regionAt(int id) {
        return id >= 0 && id < regions.size() ? regions.get(id) : null;
    }

    float homeX() {
        return homeX;
    }

    float homeY() {
        return homeY;
    }
}
