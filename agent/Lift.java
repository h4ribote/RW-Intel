import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One lift: transports taking a squad or a list of units from where they stand to a point on land, carried out step by step on the game side.
 *
 * The control process decides which transports, what cargo and where to; the order of loading, moving and setting down, and the waiting between those steps, are settled here, one tactical period at a time. Loading is the transport's own pick-up order, appended so that it fetches every passenger in turn, because a passenger told to board a transport it cannot reach stands holding the order for ever. Setting down is the transport's unload action issued on the move, which sets the passengers down when the transport stops on land and holds over water.
 *
 * Phases run approach, loading, carrying, unloading, then done or failed. A failure carries its reason.
 */
final class Lift {

    /** Phases, matching `rwintel/wire/observation.py`. */
    static final int APPROACH = 0;
    static final int LOADING = 1;
    static final int CARRYING = 2;
    static final int UNLOADING = 3;
    static final int DONE = 4;
    static final int FAILED = 5;

    /** Why a lift failed, matching `rwintel/wire/observation.py`. */
    static final int NO_REASON = 0;
    static final int SUNK = 1;
    static final int UNREACHABLE_PICKUP = 2;
    static final int UNREACHABLE_DROP = 3;
    static final int REFUSED = 4;
    static final int EXPIRED = 5;
    static final int CANCELLED = 6;
    static final int CARGO_LOST = 7;

    /** What the cargo names, matching `rwintel/wire/action.py`. */
    static final int CARGO_SQUAD = 0;
    static final int CARGO_UNITS = 1;

    /** A passenger or transport this close to the pick-up point has arrived there. */
    private static final float PICKUP_RADIUS = 160f;

    /** The longest the transports and passengers are given to gather before loading starts with whoever is there. */
    private static final int APPROACH_MS = 30000;

    /** How long the gathering orders are given to be taken up before anyone standing still counts as having arrived. */
    private static final int GATHER_MS = 1000;

    /** The longest loading may take before the transports leave with what is aboard. */
    private static final int LOADING_MS = 60000;

    /** How long a transport that has stopped without the passengers it still has to fetch, or with passengers it has not set down, is left before its orders are issued again. */
    private static final int RETRY_MS = 4000;

    /** How far apart transports of one lift are set down, so that they do not stop on top of one another. */
    private static final float DROP_SPACING = 80f;

    final int id;
    final List<Long> transports = new ArrayList<Long>();
    final int cargoKind;
    final int cargoSquad;
    final List<Long> cargoUnits = new ArrayList<Long>();
    final float pickupX;
    final float pickupY;
    final int dropRegion;
    final float dropX;
    final float dropY;
    int deadlineMs;

    int phase = APPROACH;
    int reason = NO_REASON;
    /** Passengers assigned to each transport, in the order it fetches them. */
    final Map<Long, List<Long>> load = new LinkedHashMap<Long, List<Long>>();
    int expected;
    int loaded;
    float health;
    /** Game time the lift is expected to have set its cargo down by. */
    int etaMs;
    private int phaseSinceMs;
    private int orderedAtMs;
    private boolean unloadIssued;
    /** How many were aboard when the transports set out, which is the count setting down is measured from. */
    private int departed;

    private final Engine engine;
    private final World world;

    Lift(Engine engine, World world, int id, List<Long> transports, int cargoKind, int cargoSquad, List<Long> cargoUnits,
         float pickupX, float pickupY, int dropRegion, float dropX, float dropY, int deadlineMs) {
        this.engine = engine;
        this.world = world;
        this.id = id;
        this.transports.addAll(transports);
        this.cargoKind = cargoKind;
        this.cargoSquad = cargoSquad;
        this.cargoUnits.addAll(cargoUnits);
        this.pickupX = pickupX;
        this.pickupY = pickupY;
        this.dropRegion = dropRegion;
        this.dropX = dropX;
        this.dropY = dropY;
        this.deadlineMs = deadlineMs;
    }

    /** Whether a lift row restates this lift rather than describing a different one. */
    boolean sameAs(List<Long> transports, int cargoKind, int cargoSquad, List<Long> cargoUnits, int dropRegion, float dropX, float dropY) {
        return this.transports.equals(transports) && this.cargoKind == cargoKind && this.cargoSquad == cargoSquad
                && this.cargoUnits.equals(cargoUnits) && this.dropRegion == dropRegion
                && Math.abs(this.dropX - dropX) < 1f && Math.abs(this.dropY - dropY) < 1f;
    }

    boolean ended() {
        return phase == DONE || phase == FAILED;
    }

    /** Whether the unit is cargo of this lift: assigned a place aboard, or named in its list of units. */
    boolean holds(long unit) {
        Long id = Long.valueOf(unit);
        if (cargoKind == CARGO_UNITS && cargoUnits.contains(id)) return true;
        for (List<Long> passengers : load.values()) if (passengers.contains(id)) return true;
        return false;
    }

    /** The passengers as named now: the squad's members, or the units listed, leaving out whatever is gone. */
    List<Object> passengers() {
        List<Object> out = new ArrayList<Object>();
        List<Long> ids = cargoUnits;
        if (cargoKind == CARGO_SQUAD) {
            World.Squad squad = world.squads.get(Integer.valueOf(cargoSquad));
            ids = squad == null ? new ArrayList<Long>() : squad.units;
        }
        for (Long passenger : ids) {
            Object unit = world.handle(passenger.longValue());
            if (unit != null && alive(unit)) out.add(unit);
        }
        return out;
    }

    /** Assigns the passengers to the transports and sends everyone towards the pick-up point; ends the lift at once when it cannot be carried out. */
    void start(Object game, Object self, int now) throws Exception {
        phaseSinceMs = now;
        List<Object> carriers = new ArrayList<Object>();
        boolean standing = false;
        for (Long transport : transports) {
            Object unit = world.handle(transport.longValue());
            if (unit == null || !alive(unit)) continue;
            standing = true;
            if (world.capacityOfType(engine.type(unit)) <= 0) continue;
            carriers.add(unit);
            load.put(Long.valueOf(engine.id(unit)), new ArrayList<Long>());
        }
        List<Object> cargo = passengers();
        if (carriers.isEmpty()) {
            finish(FAILED, standing ? REFUSED : SUNK);
            return;
        }
        if (cargo.isEmpty()) {
            finish(FAILED, CARGO_LOST);
            return;
        }
        // The largest passengers go first, so that what takes the most slots is not left without room by what takes the least.
        java.util.Collections.sort(cargo, new java.util.Comparator<Object>() {
            public int compare(Object a, Object b) {
                return Integer.compare(slotsOf(b), slotsOf(a));
            }
        });
        Map<Object, Integer> free = new java.util.IdentityHashMap<Object, Integer>();
        for (Object carrier : carriers) free.put(carrier, Integer.valueOf(world.capacityOfType(engine.type(carrier)) - engine.aboard(carrier)));
        boolean carriable = false;
        boolean reachable = false;
        for (Object passenger : cargo) {
            Object best = null;
            for (Object carrier : carriers) {
                if (!engine.typeCarries(engine.type(carrier), engine.type(passenger))) continue;
                carriable = true;
                if (engine.carrier(passenger) == null && !world.reachable(carrier, engine.x(passenger), engine.y(passenger))) continue;
                reachable = true;
                if (free.get(carrier).intValue() < slotsOf(passenger)) continue;
                if (best == null || free.get(carrier).intValue() > free.get(best).intValue()) best = carrier;
            }
            if (best == null) continue;
            free.put(best, Integer.valueOf(free.get(best).intValue() - slotsOf(passenger)));
            load.get(Long.valueOf(engine.id(best))).add(Long.valueOf(engine.id(passenger)));
            expected++;
        }
        if (expected == 0) {
            finish(FAILED, !carriable ? REFUSED : !reachable ? UNREACHABLE_PICKUP : REFUSED);
            return;
        }
        for (Object carrier : carriers) {
            if (load.get(Long.valueOf(engine.id(carrier))).isEmpty()) continue;
            if (!world.reachable(carrier, dropX, dropY)) {
                finish(FAILED, UNREACHABLE_DROP);
                return;
            }
        }
        // Transports with nobody assigned to them sit this lift out.
        java.util.Iterator<Map.Entry<Long, List<Long>>> idle = load.entrySet().iterator();
        while (idle.hasNext()) if (idle.next().getValue().isEmpty()) idle.remove();

        for (Long transport : load.keySet()) {
            Object carrier = world.handle(transport.longValue());
            Object command = engine.command(game, self);
            engine.addUnit(command, carrier);
            engine.moveTo(command, pickupX, pickupY);
        }
        Object command = null;
        for (Object passenger : assigned()) {
            if (engine.carrier(passenger) != null || !world.reachable(passenger, pickupX, pickupY)) continue;
            if (command == null) command = engine.command(game, self);
            engine.addUnit(command, passenger);
        }
        if (command != null) engine.moveTo(command, pickupX, pickupY);
        orderedAtMs = now;
        measure(now);
    }

    /** Advances the lift by one period. */
    void step(Object game, Object self, int now) throws Exception {
        if (ended()) return;
        for (Long transport : load.keySet()) {
            Object carrier = world.handle(transport.longValue());
            if (carrier == null || !alive(carrier)) {
                // Everyone aboard the one sunk is lost with it; the other transports set theirs down the next time they stop on land.
                abandon(game, self, SUNK);
                return;
            }
        }
        if (deadlineMs > 0 && now > deadlineMs) {
            abandon(game, self, EXPIRED);
            return;
        }
        List<Object> cargo = assigned();
        if (cargo.isEmpty()) {
            finish(phase >= CARRYING ? DONE : FAILED, phase >= CARRYING ? NO_REASON : CARGO_LOST);
            return;
        }
        loaded = 0;
        for (Object passenger : cargo) if (engine.carrier(passenger) != null) loaded++;

        if (phase == APPROACH) {
            if ((now - phaseSinceMs >= GATHER_MS && gathered(cargo)) || now - phaseSinceMs >= APPROACH_MS) {
                enter(LOADING, now);
                fetch(game, self, now, true);
            }
        } else if (phase == LOADING) {
            if (loaded == cargo.size() || (now - phaseSinceMs >= LOADING_MS && loaded > 0)) {
                enter(CARRYING, now);
                departed = loaded;
                carry(game, self, now);
            } else if (now - phaseSinceMs >= LOADING_MS) {
                abandon(game, self, REFUSED);
                return;
            } else if (now - orderedAtMs >= RETRY_MS) {
                fetch(game, self, now, false);
            }
        } else {
            if (!unloadIssued) {
                // The unload is issued a period after the move, once the move is the order being carried out, so that it rides the move rather than standing in for it.
                for (Long transport : load.keySet()) unload(game, self, world.handle(transport.longValue()));
                unloadIssued = true;
            }
            if (phase == CARRYING && loaded < departed) enter(UNLOADING, now);
            if (phase == UNLOADING && loaded == 0) {
                finish(DONE, NO_REASON);
                return;
            }
            if (stalled(now)) carry(game, self, now);
        }
        measure(now);
    }

    /** Ends the lift without carrying it further: anything aboard is set down the next time its transport stops on land. */
    void abandon(Object game, Object self, int why) throws Exception {
        for (Long transport : load.keySet()) {
            Object carrier = world.handle(transport.longValue());
            if (carrier != null && alive(carrier) && engine.aboard(carrier) > 0) unload(game, self, carrier);
        }
        finish(FAILED, why);
    }

    private void finish(int phase, int reason) {
        this.phase = phase;
        this.reason = reason;
    }

    private void enter(int phase, int now) {
        this.phase = phase;
        phaseSinceMs = now;
    }

    /** Each transport fetches the passengers it was given that are not yet aboard, in turn: the first replaces whatever it was doing and the rest are appended after it. */
    private void fetch(Object game, Object self, int now, boolean first) throws Exception {
        for (Map.Entry<Long, List<Long>> entry : load.entrySet()) {
            Object carrier = world.handle(entry.getKey().longValue());
            if (carrier == null) continue;
            if (!first && engine.orderCount(carrier) > 0) continue;
            boolean append = false;
            for (Long passenger : entry.getValue()) {
                Object unit = world.handle(passenger.longValue());
                if (unit == null || !alive(unit) || engine.carrier(unit) != null) continue;
                Object command = engine.command(game, self);
                if (append) engine.append(command);
                engine.addUnit(command, carrier);
                engine.loadUp(command, unit);
                append = true;
            }
        }
        orderedAtMs = now;
    }

    /** Sends every transport to its own spot beside the drop point, spaced along a line through it, with the unload to follow next period. */
    private void carry(Object game, Object self, int now) throws Exception {
        int index = 0;
        int count = load.size();
        for (Long transport : load.keySet()) {
            Object carrier = world.handle(transport.longValue());
            float offset = (index++ - (count - 1) / 2f) * DROP_SPACING;
            Object command = engine.command(game, self);
            engine.addUnit(command, carrier);
            engine.moveTo(command, dropX + offset, dropY);
        }
        unloadIssued = false;
        orderedAtMs = now;
    }

    private void unload(Object game, Object self, Object carrier) throws Exception {
        if (carrier == null) return;
        Object command = engine.command(game, self);
        engine.addUnit(command, carrier);
        engine.unitAction(command, carrier, "109");
    }

    /** Whether a transport still carrying passengers has stood without an order for RETRY_MS since the transports were last sent: stopped short by something in the way, or stopped where nobody could get off. Its move and unload are then issued again. */
    private boolean stalled(int now) throws Exception {
        if (now - orderedAtMs < RETRY_MS) return false;
        for (Long transport : load.keySet()) {
            Object carrier = world.handle(transport.longValue());
            if (carrier != null && engine.aboard(carrier) > 0 && engine.orderCount(carrier) == 0) return true;
        }
        return false;
    }

    /** Whether everyone has got to the pick-up point or stopped trying: a passenger that cannot reach it is fetched from where it stands. */
    private boolean gathered(List<Object> cargo) throws Exception {
        for (Object passenger : cargo) {
            if (engine.carrier(passenger) != null || !world.reachable(passenger, pickupX, pickupY)) continue;
            boolean there = Math.hypot(engine.x(passenger) - pickupX, engine.y(passenger) - pickupY) <= PICKUP_RADIUS;
            if (!there && engine.orderCount(passenger) > 0) return false;
        }
        for (Long transport : load.keySet()) {
            Object carrier = world.handle(transport.longValue());
            boolean there = Math.hypot(engine.x(carrier) - pickupX, engine.y(carrier) - pickupY) <= PICKUP_RADIUS * 2;
            if (!there && engine.orderCount(carrier) > 0) return false;
        }
        return true;
    }

    /** The passengers assigned to a transport that are still alive. */
    private List<Object> assigned() throws Exception {
        List<Object> out = new ArrayList<Object>();
        for (List<Long> passengers : load.values()) {
            for (Long passenger : passengers) {
                Object unit = world.handle(passenger.longValue());
                if (unit != null && alive(unit)) out.add(unit);
            }
        }
        return out;
    }

    /** The transports' combined health and when the cargo is expected to be down, from the slowest transport's distance still to go. */
    private void measure(int now) throws Exception {
        health = 0f;
        float slowest = 0f;
        for (Long transport : load.keySet()) {
            Object carrier = world.handle(transport.longValue());
            if (carrier == null) continue;
            health += engine.health(carrier);
            float speed = engine.typeSpeed(engine.type(carrier));
            float distance = (float) Math.hypot(engine.x(carrier) - dropX, engine.y(carrier) - dropY);
            if (phase <= LOADING) {
                distance = (float) (Math.hypot(engine.x(carrier) - pickupX, engine.y(carrier) - pickupY)
                        + Math.hypot(pickupX - dropX, pickupY - dropY));
            }
            float seconds = speed > 0f ? distance / speed : 0f;
            if (seconds > slowest) slowest = seconds;
        }
        etaMs = now + (int) (slowest * 1000f);
    }

    private int slotsOf(Object unit) {
        try {
            return Math.max(1, engine.typeSlots(engine.type(unit)));
        } catch (Exception e) {
            return 1;
        }
    }

    private boolean alive(Object unit) {
        try {
            return !engine.dead(unit);
        } catch (Exception e) {
            return false;
        }
    }
}
