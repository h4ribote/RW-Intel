import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The agent's own view of the match: the region table, squad membership, and whatever the last scan of the world found.
 *
 * Squads live here rather than in the control process only in the sense of being addressable: which units belong to which squad is decided by a policy and arrives with the actions. The game side keeps the membership because it is what lets it address a squad with one command and describe it back in the observation.
 *
 * Regions likewise arrive from the control process. Deriving them needs the map file, which is read there; doing it here as well would put the same rule in two implementations, and the one that can be checked without launching the game is the better place for it. What is added here is the distance of each region from our own base, which is not knowable when the table is sent because nothing has been built yet.
 *
 * What is counted as our own strength is strength under our own command. A squad a human has taken over still belongs to the same player and still shows up in the engine's aggregates, so it is described in the observation but left out of every total; a command chain that plans with units it cannot move is worse off than one that knows it is short.
 */
final class World {

    /** A building standing on a resource point is within this of it, and nothing else is. */
    private static final float ON_RESOURCE = 40f;

    /** The value the wire uses where a squad would go and there is none. */
    static final int NO_SQUAD = 0xFFFF;

    /** Fixed slot counts, so the layers above see an action space that does not change shape with the map or with how many squads happen to exist. */
    static final int REGION_SLOTS = 24;
    static final int SQUAD_SLOTS = 8;

    /** Bits of a squad's commander field: which layers of its command a human has taken over. */
    static final int HUMAN_OPERATIONS = 1;
    static final int HUMAN_TACTICS = 2;

    /** Mission status, matching `rwintel/wire/action.py`. */
    static final int ACTIVE = 0;
    static final int STALLED = 1;
    static final int LOSING = 2;
    static final int COMPLETE = 3;
    static final int EXPIRED = 4;
    /** No member can reach the target under its own power, so no order went out. */
    static final int UNREACHABLE = 5;
    /** The squad is the cargo of a lift whose transports are still on their way to it. */
    static final int AWAITING_LIFT = 6;
    /** The squad is being loaded, carried or set down. */
    static final int LIFTING = 7;

    /** What a contract's target names, matching `rwintel/wire/action.py`. */
    static final int TARGET_REGION = 0;
    static final int TARGET_SQUAD = 1;
    static final int TARGET_UNIT = 2;

    /** A mission is stalled when the balance of force in its target region has not moved this much for this long. */
    private static final float STALL_MOVEMENT = 0.10f;
    private static final int STALL_WINDOW_MS = 30000;

    /** Losses past this share of the contract's budget are what "losing" means. */
    private static final float LOSING_SHARE = 0.7f;

    /** A squad worn down to this share of the value it was formed with is worth reporting as an event, because it is the point at which the organisation layer starts merging. */
    private static final float DEPLETED_SHARE = 0.4f;

    /** Event kinds, matching `rwintel/wire/observation.py`. */
    static final int EVENT_UNIT_COMPLETED = 1;
    static final int EVENT_UNIT_LOST = 2;
    static final int EVENT_SQUAD_DEPLETED = 3;
    static final int EVENT_BOARDED = 4;
    static final int EVENT_DISEMBARKED = 5;
    static final int EVENT_LIFT_DONE = 6;
    static final int EVENT_LIFT_FAILED = 7;

    static final class Region {
        final int id;
        final float x;
        final float y;
        final int resources;
        final boolean spawn;
        float distanceFromHome;
        int heldByUs;
        int heldByEnemy;
        float ourValue;
        float enemyValue;
        int enemySeenAtMs;

        Region(int id, float x, float y, int resources, boolean spawn) {
            this.id = id;
            this.x = x;
            this.y = y;
            this.resources = resources;
            this.spawn = spawn;
        }
    }

    static final class Squad {
        final int id;
        int commander;
        /** Whose orders move this squad, as a player slot, or -1 for the player this process is. Only a constructed engagement uses anything else: with the sandbox flag set, one process drives both sides of a fight, and a command has to be taken out in the name of the player whose units it addresses. */
        int owner = -1;
        final List<Long> units = new ArrayList<Long>();
        float value;
        float formedValue;
        float x;
        float y;
        /** How far the members are scattered: the standard deviation of their distance from the centre. */
        float spread;
        int task;
        int stance;
        int status;
        /** What the contract's target names, and the region, squad or unit it is. */
        int targetKind;
        long target;
        /** The region the target lies in: the target itself for a region, and the region nearest a squad or unit target as of the last scan. What completion and stalling are judged in. */
        int targetRegion;
        /** Set when the contract was taken on and no member could reach its target. */
        boolean unreachable;
        /** Set when the squad or unit the contract names is gone. */
        boolean targetLost;
        /** The lift the squad is the cargo of, or null. While there is one the lift moves the squad and nothing else does. */
        Lift lift;
        /** Members aboard a transport, and the narrowest movement type among the members as a {@link Passage#CLASSES} number. */
        int aboard;
        int passage;
        /** Members aboard as of the last upkeep, which is how members set down since are noticed. */
        int aboardSeen;
        /** The unit a raid is attacking, and the unit an escort is guarding, so that each is reissued only when it changes. */
        long raidTarget;
        long escortLead;
        /** Value lost since the current contract was issued, which is what the contract's budget is spent against. */
        float losses;
        float costBudget;
        int deadlineMs;
        int issuedAtMs;
        /** Value held when the contract was issued, the baseline the losses are measured from. */
        float valueAtIssue;
        /** The balance of force in the target region when it last moved, and when that was, which is how a stall is recognised. */
        float lastBalance;
        int balanceMovedAtMs;
        boolean reportedDepleted;
        /** What the squad was last told to do instead of its contract, so that returning to the contract is issued once rather than every period. */
        int lastDeviation;
        /** Set when a contract has just been issued, so that the next scan takes the value its losses are measured from. */
        boolean rebaseline;

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
        /** The kind of order being carried out, or Engine.NO_ORDER for a unit that has never had one. */
        int order;
        /** How many orders are queued. Nought is what "free to be given something to do" means: the kind of the last order stays set after it has been carried out, so it does not answer that question. */
        int queued;
        long target;
        int stance;
        /** Game time since the unit was last hit, which is the cheap trigger for a squad needing attention. */
        int sinceHitMs;
        boolean hostile;
        Object handle;
        boolean building;
        int price;
        /** The tier a building of ours has been raised to, and the price of raising it to the next one, which is nought when it offers no further tier. Read only for our own finished buildings; nought for everything else. */
        int level;
        int upgradePrice;
        /** The transport this unit is aboard, by id, 0 for none; and for a transport, how many slots it has filled. */
        long carrier;
        int aboard;
        boolean builder;
        /** A building that may only stand on a resource pool, finished or not. */
        boolean extractor;
    }

    static final class Event {
        final int kind;
        final int squad;
        final long unit;
        final int typeIndex;
        final float value;

        Event(int kind, int squad, long unit, int typeIndex, float value) {
            this.kind = kind;
            this.squad = squad;
            this.unit = unit;
            this.typeIndex = typeIndex;
            this.value = value;
        }
    }

    private final Engine engine;

    final List<Region> regions = new ArrayList<Region>();
    /** Each resource point as x, y and the region it lies in, supplied with the region table. */
    private final List<float[]> resourcePoints = new ArrayList<float[]>();
    final Map<Integer, Squad> squads = new LinkedHashMap<Integer, Squad>();
    final List<Seen> visible = new ArrayList<Seen>();
    /** Things that happened since the control process last drained them. The organisation layer runs on events rather than on a period. */
    final List<Event> events = new ArrayList<Event>();

    /** Reported type name to the index the control process knows the type by, and back again. */
    private final Map<String, Integer> typeIndex = new HashMap<String, Integer>();
    private final List<Object> typesByIndex = new ArrayList<Object>();

    /** How far each type can shoot, resolved once. Range is not on the type interface and digging it out per unit per period would be paid for every unit in the world. */
    private final Map<Object, Float> rangeByType = new java.util.IdentityHashMap<Object, Float>();

    /** Each type's movement, transport capacity and whether it builds, resolved once for the same reason. */
    private final Map<Object, String> movementByType = new java.util.IdentityHashMap<Object, String>();
    private final Map<Object, Integer> capacityByType = new java.util.IdentityHashMap<Object, Integer>();
    private final Map<Object, Boolean> builderByType = new java.util.IdentityHashMap<Object, Boolean>();

    /** What each movement type can cross on the map in play, or null until the episode's terrain has been read. */
    Passage.Grids passage;

    /** The lifts under way by id, and the ones that ended since the last frame that reported them. */
    final Map<Integer, Lift> lifts = new LinkedHashMap<Integer, Lift>();
    final List<Lift> endedLifts = new ArrayList<Lift>();

    private final Map<Long, Integer> unitToSquad = new HashMap<Long, Integer>();
    private final Map<Long, Object> unitHandles = new HashMap<Long, Object>();

    /** What the last scan saw of our own units, so this one can tell what was finished and what was lost. */
    private final Map<Long, int[]> ownLastSeen = new HashMap<Long, int[]>();

    /** Every enemy is reported while this holds; otherwise only what the engine says is visible. */
    boolean omniscient = true;

    /** Units and value under our own command, which is the total with any other commander's holdings taken out. */
    int commandedUnits;
    float commandedValue;

    private float homeX = Float.NaN;
    private float homeY = Float.NaN;

    World(Engine engine) {
        this.engine = engine;
    }

    void indexTypes(List<Object> types) throws Exception {
        typeIndex.clear();
        typesByIndex.clear();
        rangeByType.clear();
        movementByType.clear();
        capacityByType.clear();
        builderByType.clear();
        for (Object type : types) {
            String name = engine.typeName(type);
            if (typeIndex.containsKey(name)) continue;
            typeIndex.put(name, Integer.valueOf(typesByIndex.size()));
            typesByIndex.add(type);
            rangeByType.put(type, Float.valueOf(engine.typeRange(type)));
        }
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
        for (int i = 0; i < table.size() && i < REGION_SLOTS; i++) {
            float[] row = table.get(i);
            regions.add(new Region(i, row[0], row[1], (int) row[2], row[3] != 0f));
        }
    }

    /**
     * Replaces the list of resource points, each one a position and the region it lies in.
     *
     * These arrive with the region table rather than being found in the world, because a resource pool is not a thing in the world: it is a flag on a map tile, and no object stands on it until an extractor is built there. The control process reads the same map file the engine does, so the positions are the engine's own.
     */
    void setResourcePoints(List<float[]> points) {
        resourcePoints.clear();
        resourcePoints.addAll(points);
    }

    void reset() {
        squads.clear();
        unitToSquad.clear();
        unitHandles.clear();
        ownLastSeen.clear();
        visible.clear();
        events.clear();
        regions.clear();
        resourcePoints.clear();
        lifts.clear();
        endedLifts.clear();
        passage = null;
        homeX = Float.NaN;
        homeY = Float.NaN;
        commandedUnits = 0;
        commandedValue = 0f;
    }

    Squad squad(int id) {
        Squad squad = squads.get(Integer.valueOf(id));
        if (squad == null) squads.put(Integer.valueOf(id), squad = new Squad(id));
        return squad;
    }

    /** Applies a membership decision. A unit belongs to at most one squad, so it is taken out of whatever held it. A squad handed an empty roster is disbanded, which is how the organisation layer retires one. */
    void assign(int squadId, int commander, int owner, List<Long> units) {
        Squad squad = squad(squadId);
        squad.commander = commander;
        squad.owner = owner;
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
        if (units.isEmpty()) squads.remove(Integer.valueOf(squadId));
    }

    Object handle(long id) {
        return unitHandles.get(Long.valueOf(id));
    }

    /** How far a unit can shoot, in world units, or zero for one that cannot. */
    float rangeOf(Object unit) {
        try {
            Float range = rangeByType.get(engine.type(unit));
            return range == null ? 0f : range.floatValue();
        } catch (Exception e) {
            return 0f;
        }
    }

    /** How far a type can shoot, for the catalogue the control process classifies units with. */
    float rangeOfType(Object type) {
        Float range = rangeByType.get(type);
        return range == null ? 0f : range.floatValue();
    }

    /** The engine's name for the movement type of a unit's type, the empty name when it cannot be read. */
    String movementOf(Object unit) {
        try {
            return movementOfType(engine.type(unit));
        } catch (Exception e) {
            return "";
        }
    }

    String movementOfType(Object type) {
        if (type == null) return "";
        String movement = movementByType.get(type);
        if (movement == null) {
            try {
                movement = engine.typeMovement(type);
            } catch (Exception e) {
                movement = "";
            }
            movementByType.put(type, movement);
        }
        return movement;
    }

    /** Transport capacity of a type in slots, -1 for one that carries nothing. */
    int capacityOfType(Object type) {
        if (type == null) return -1;
        Integer capacity = capacityByType.get(type);
        if (capacity == null) capacityByType.put(type, capacity = Integer.valueOf(engine.typeCapacity(type)));
        return capacity.intValue();
    }

    boolean builderType(Object type) {
        if (type == null) return false;
        Boolean builder = builderByType.get(type);
        if (builder == null) {
            try {
                builder = Boolean.valueOf(engine.typeIsBuilder(type));
            } catch (Exception e) {
                builder = Boolean.FALSE;
            }
            builderByType.put(type, builder);
        }
        return builder.booleanValue();
    }

    /** Whether the unit can get to the position under its own power. True while the map's passage is unknown, so that nothing is withheld for want of it. */
    boolean reachable(Object unit, float x, float y) {
        if (passage == null) return true;
        try {
            return passage.reachable(movementOf(unit), engine.x(unit), engine.y(unit), x, y);
        } catch (Exception e) {
            return true;
        }
    }

    /** Whether a unit of this movement type at one position can get to the other, by the same rule. */
    boolean reachable(String movement, float x0, float y0, float x1, float y1) {
        return passage == null || passage.reachable(movement, x0, y0, x1, y1);
    }

    /** Whether a squad is one this process still commands, rather than one a human has taken over. */
    static boolean ours(Squad squad) {
        return squad == null || squad.commander == 0;
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
        commandedValue = 0f;

        List<Seen> extractors = new ArrayList<Seen>();
        Map<Long, Float> valueById = new HashMap<Long, Float>();
        Set<Long> ownNow = new HashSet<Long>();

        Object[] units = engine.unitArray();
        int count = engine.unitCount();
        for (int i = 0; i < count && i < units.length; i++) {
            Object unit = units[i];
            if (unit == null || engine.dead(unit)) continue;
            Object type = engine.type(unit);
            if (type == null) continue;
            String typeName = engine.typeName(type);

            Object owner = engine.owner(unit);
            if (owner == null) continue;
            int team = engine.team(owner);
            boolean ours = team == selfTeam;
            // What a negative team owns is the scenery and the spectators. Reading "not on our team" as "hostile" would report every tree on the map as an enemy unit, and the tactical layer would spend the match shooting at them.
            if (!ours && team < 0) continue;
            if (!ours && !omniscient && !engine.visibleTo(unit, self)) continue;

            Seen seen = new Seen();
            seen.handle = unit;
            seen.id = engine.id(unit);
            seen.x = engine.x(unit);
            seen.y = engine.y(unit);
            seen.health = engine.health(unit);
            seen.maxHealth = engine.maxHealth(unit);
            float progress = engine.built(unit);
            seen.built = (int) Math.max(0f, Math.min(255f, progress * 255f));
            seen.hostile = !ours;
            seen.sinceHitMs = Math.max(0, now - engine.lastHit(unit));
            seen.typeIndex = indexOf(typeName);
            seen.building = engine.typeIsBuilding(type);
            seen.price = engine.price(unit);
            if (ours && seen.building && seen.built >= 255) {
                seen.level = engine.level(unit);
                Object upgrade = engine.upgradeOffered(unit);
                seen.upgradePrice = upgrade == null ? 0 : engine.actionPrice(upgrade);
            }
            seen.order = Engine.NO_ORDER;
            if (engine.armedClass.isInstance(unit)) {
                seen.stance = engine.stanceOf(unit);
                seen.order = engine.orderKind(unit);
                seen.queued = engine.orderCount(unit);
                Object target = engine.attackTarget(unit);
                seen.target = target == null ? 0L : engine.id(target);
            }
            Object carrier = engine.carrier(unit);
            seen.carrier = carrier == null ? 0L : engine.id(carrier);
            if (capacityOfType(type) > 0) seen.aboard = engine.aboard(unit);
            seen.builder = builderType(type);
            seen.extractor = seen.building && engine.typeOnResourcePool(type);
            Integer squadId = unitToSquad.get(Long.valueOf(seen.id));
            seen.squad = squadId == null ? NO_SQUAD : squadId.intValue();

            unitHandles.put(Long.valueOf(seen.id), unit);
            valueById.put(Long.valueOf(seen.id), Float.valueOf(seen.price));
            visible.add(seen);

            if (seen.extractor && seen.built >= 255) extractors.add(seen);

            boolean commanded = ours && ours(squads.get(squadId));
            if (commanded) {
                commandedUnits++;
                commandedValue += seen.price;
                if (Float.isNaN(homeX) && seen.building) {
                    homeX = seen.x;
                    homeY = seen.y;
                }
            }
            if (ours) noteOwnUnit(seen, ownNow);

            Region region = nearest(seen.x, seen.y);
            if (region != null) {
                if (commanded) region.ourValue += seen.price;
                else if (!ours) {
                    region.enemyValue += seen.price;
                    region.enemySeenAtMs = now;
                }
            }
        }

        for (float[] point : resourcePoints) {
            Region region = regionAt((int) point[2]);
            if (region == null) continue;
            for (Seen extractor : extractors) {
                float dx = extractor.x - point[0];
                float dy = extractor.y - point[1];
                if (dx * dx + dy * dy > ON_RESOURCE * ON_RESOURCE) continue;
                if (extractor.hostile) region.heldByEnemy++;
                else region.heldByUs++;
                break;
            }
        }

        noteLostUnits(ownNow);
        for (Region region : regions) region.distanceFromHome = distanceFromHome(region);
        refreshSquads(valueById, now);
    }

    /** Remembers what our units looked like, and reports the ones that finished building, went aboard a transport or came off one since the last scan. */
    private void noteOwnUnit(Seen seen, Set<Long> ownNow) {
        Long key = Long.valueOf(seen.id);
        ownNow.add(key);
        int aboard = seen.carrier != 0L ? 1 : 0;
        int[] previous = ownLastSeen.get(key);
        if (previous == null) {
            ownLastSeen.put(key, new int[]{seen.built, seen.typeIndex, seen.price, aboard});
            if (seen.built >= 255) events.add(new Event(EVENT_UNIT_COMPLETED, seen.squad, seen.id, seen.typeIndex, seen.price));
            return;
        }
        if (previous[0] < 255 && seen.built >= 255) {
            events.add(new Event(EVENT_UNIT_COMPLETED, seen.squad, seen.id, seen.typeIndex, seen.price));
        }
        if (previous[3] != aboard) {
            events.add(new Event(aboard != 0 ? EVENT_BOARDED : EVENT_DISEMBARKED, seen.squad, seen.id, seen.typeIndex, seen.price));
        }
        previous[0] = seen.built;
        previous[1] = seen.typeIndex;
        previous[2] = seen.price;
        previous[3] = aboard;
    }

    private void noteLostUnits(Set<Long> ownNow) {
        java.util.Iterator<Map.Entry<Long, int[]>> gone = ownLastSeen.entrySet().iterator();
        while (gone.hasNext()) {
            Map.Entry<Long, int[]> entry = gone.next();
            if (ownNow.contains(entry.getKey())) continue;
            int[] last = entry.getValue();
            Integer squadId = unitToSquad.get(entry.getKey());
            events.add(new Event(EVENT_UNIT_LOST, squadId == null ? NO_SQUAD : squadId.intValue(),
                    entry.getKey().longValue(), last[1], last[2]));
            gone.remove();
        }
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
            squad.spread = spreadOf(squad);
            noteCarriage(squad);
        }
        for (Squad squad : squads.values()) {
            if (squad.targetKind == TARGET_REGION) {
                squad.targetRegion = (int) squad.target;
                continue;
            }
            float[] point = targetPoint(squad);
            squad.targetLost = point == null;
            Region where = point == null ? null : nearest(point[0], point[1]);
            if (where != null) squad.targetRegion = where.id;
        }
        for (Squad squad : squads.values()) {
            float value = squad.value;
            if (squad.rebaseline) {
                squad.rebaseline = false;
                squad.valueAtIssue = value;
            }
            // The strength a squad is measured against is the most it has ever held, not what it held on the day it was formed. Reinforcement is what the organisation layer does to a worn squad, and a denominator that ignored it would call a squad healthy for having been small.
            if (value > squad.formedValue) squad.formedValue = value;
            squad.losses = Math.max(0f, squad.valueAtIssue - value);
            squad.status = statusOf(squad, now);
            boolean depleted = squad.formedValue > 0f && value < squad.formedValue * DEPLETED_SHARE;
            if (depleted && !squad.reportedDepleted) {
                events.add(new Event(EVENT_SQUAD_DEPLETED, squad.id, 0L, 0xFFFF, value));
            }
            squad.reportedDepleted = depleted;
        }
    }

    /** How many members are aboard a transport, and the narrowest movement type among all of them. */
    private void noteCarriage(Squad squad) {
        int aboard = 0;
        Set<String> movements = new HashSet<String>();
        for (Long id : squad.units) {
            Object handle = unitHandles.get(id);
            if (handle == null) continue;
            try {
                if (engine.carrier(handle) != null) aboard++;
            } catch (Exception ignored) {
                // a unit that cannot be read is counted as not aboard
            }
            String movement = movementOf(handle);
            if (!movement.isEmpty()) movements.add(movement);
        }
        squad.aboard = aboard;
        squad.passage = Passage.classOf(Passage.narrowest(movements));
    }

    /** Where a squad's contract points: the centre of its target region, the centre of its target squad, or its target unit; null when the squad or unit named is gone or the region does not exist. */
    float[] targetPoint(Squad squad) {
        if (squad.targetKind == TARGET_SQUAD) {
            Squad other = squads.get(Integer.valueOf((int) squad.target));
            if (other == null || other.units.isEmpty()) return null;
            return new float[]{other.x, other.y};
        }
        if (squad.targetKind == TARGET_UNIT) {
            Object unit = handle(squad.target);
            if (unit == null) return null;
            try {
                return new float[]{engine.x(unit), engine.y(unit)};
            } catch (Exception e) {
                return null;
            }
        }
        Region region = regionAt((int) squad.target);
        return region == null ? null : new float[]{region.x, region.y};
    }

    private float spreadOf(Squad squad) {
        int alive = 0;
        double sum = 0;
        for (Long id : squad.units) {
            Object handle = unitHandles.get(id);
            if (handle == null) continue;
            try {
                double dx = engine.x(handle) - squad.x;
                double dy = engine.y(handle) - squad.y;
                sum += dx * dx + dy * dy;
                alive++;
            } catch (Exception ignored) {
                // a unit that cannot be read does not contribute to how scattered the squad is
            }
        }
        return alive == 0 ? 0f : (float) Math.sqrt(sum / alive);
    }

    /**
     * Where a mission stands, in the terms the contract is written in.
     *
     * The order the cases are tried in is what decides which one a caller hears about when several hold at once. A lift comes first, since while one is under way the lift and not the contract is moving the squad. A target nothing can reach or that is gone comes next, since nothing further happens on the contract until it is replaced. Completion is reported ahead of expiry because a mission that took the region and ran late is a success that was slow, not a failure. Losing is reported ahead of a stall because the budget is the thing the operational layer has to act on soonest.
     */
    private int statusOf(Squad squad, int now) {
        Region target = regionAt(squad.targetRegion);
        // Tracked before the ladder rather than inside it. The stall clock only means anything if it runs whatever else is true; left to be updated only on the periods where no other status won, it would sit still through a spell of losing and then report a stall the instant the squad recovered.
        boolean stalled = target != null && trackBalance(squad, target, now);
        if (squad.units.isEmpty()) return ACTIVE;
        if (squad.lift != null) return squad.lift.phase == Lift.APPROACH ? AWAITING_LIFT : LIFTING;
        if (squad.unreachable) return UNREACHABLE;
        if (squad.targetLost) return STALLED;
        if (target != null && target.enemyValue <= 0f && holding(squad, target)) return COMPLETE;
        if (squad.deadlineMs > 0 && now > squad.deadlineMs) return EXPIRED;
        if (squad.costBudget > 0f && squad.losses > squad.costBudget * LOSING_SHARE) return LOSING;
        if (stalled) return STALLED;
        return ACTIVE;
    }

    /** Whether the squad itself is sitting on the region, rather than whether anything of ours happens to be there. A base counts as our value in its own region, so a mission to defend home would otherwise report itself complete before the squad had walked anywhere. */
    private boolean holding(Squad squad, Region target) {
        if (squad.units.isEmpty()) return false;
        Region where = nearest(squad.x, squad.y);
        return where != null && where.id == target.id;
    }

    /** A mission is stalled when neither side has shifted the balance of the target region for long enough that going on is unlikely to shift it either. */
    private boolean trackBalance(Squad squad, Region target, int now) {
        float total = target.ourValue + target.enemyValue;
        float balance = total <= 0f ? 0f : (target.ourValue - target.enemyValue) / total;
        if (squad.balanceMovedAtMs == 0 || Math.abs(balance - squad.lastBalance) >= STALL_MOVEMENT) {
            squad.lastBalance = balance;
            squad.balanceMovedAtMs = now;
            return false;
        }
        return now - squad.balanceMovedAtMs >= STALL_WINDOW_MS;
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

    private float distanceFromHome(Region region) {
        if (Float.isNaN(homeX)) return 0f;
        return (float) Math.hypot(region.x - homeX, region.y - homeY);
    }

    float homeX() {
        return homeX;
    }

    float homeY() {
        return homeY;
    }
}
