import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Turns a decoded action into engine commands, on the game thread.
 *
 * A squad under a contract carries out the contract's task in the way that task is written: an attack advances on the target, an encirclement comes at it from both sides, a defence and a withdrawal move to it under the contract's stance without stopping to chase, a raid goes for the enemy's extractors and builders, and an escort guards the squad or unit it is given. What the tactical layer chooses is only whether to depart from that and how; a squad choosing to hold gets no command at all until its contract changes.
 *
 * The two arrive on different periods and so travel in different sections: a contract is rewritten when the operational layer runs, a departure from it is chosen at the tactical rate. Folding them together would make the slower decision be re-sent at the faster rate, and re-issuing a contract is what resets the losses it is measured against.
 *
 * A contract whose target no member can reach under its own power issues nothing, and the squad reports it as unreachable rather than standing on the shore holding an attack order. Members aboard a transport are never given an order: one given aboard is replaced by the step off when the transport sets them down. A squad that is the cargo of a lift is moved by the lift alone, and its contract is carried out again once the lift has set it down.
 *
 * Units taking the same order go into one command. The engine accepts a list, and a squad is by construction a set of units that were meant to act together.
 */
final class Commander {

    /** How far a withdrawing squad pulls back, roughly a squad's own frontage plus a tank's reach. */
    private static final float FALL_BACK = 400f;

    /** How far apart a spreading squad ends up, chosen to clear the radius of an area weapon. */
    private static final float SPREAD = 140f;

    /** How far either side of the target the two halves of an encircling squad are aimed. */
    private static final float ENCIRCLE_OFFSET = 250f;

    /** How far around the target a raid looks for extractors and builders to go for. */
    private static final float RAID_RADIUS = 600f;

    /** Tasks, matching `rwintel/wire/action.py`. */
    private static final int ATTACK = 0;
    private static final int DEFEND = 1;
    private static final int RAID = 2;
    private static final int WITHDRAW = 3;
    private static final int ESCORT = 4;
    private static final int ENCIRCLE = 5;

    /** Deviations, matching `rwintel/wire/action.py`. */
    private static final int HOLD = 0;
    private static final int WITHDRAW_BACK = 1;
    private static final int FOCUS = 2;
    private static final int SPREAD_OUT = 3;
    private static final int KITE = 4;
    private static final int WITHDRAW_FAR = 5;
    private static final int FOCUS_THREAT = 6;

    /** The stance a squad breaking off contact is put into, so that nothing turns around to fight on the way out. */
    private static final int HOLD_FIRE = 3;

    /** A raid that has found nothing to go for and is advancing on its target instead. */
    private static final long ADVANCING = -1L;

    /**
     * How far from a squad an opposing unit has to be for the squad to be backing away from something.
     *
     * Backing away is defined against an enemy. With nothing in reach there is nothing to back away from, and the manoeuvre has no destination that means anything, so it is not carried out. Without that, a squad ordered to break off with no enemy anywhere near it is sent this far again every period for as long as the order stands, and walks off across the map: measured, that was enough to scatter the survivors of finished fights over the whole board and leave a third of the next fights with nowhere clear to be built.
     */
    private static final float CONTACT = 900f;

    /**
     * How far from the ground it was sent to a squad may be backed off before it stops being backed off further.
     *
     * A squad that has broken contact and kept walking is no longer carrying out its contract, whatever it was told to do about the fight in front of it. This is the distance at which backing away has plainly finished, and it bounds the walk even where the enemy follows.
     */
    private static final float MAX_WITHDRAWAL = 1400f;

    private static final int KIND_UNIT = 0;
    private static final int KIND_BUILDING = 1;
    private static final int KIND_UPGRADE = 2;

    /** Bits of a lift row's flags, matching `rwintel/wire/action.py`. */
    private static final int LIFT_OVERRIDE = 1;
    private static final int LIFT_CANCEL = 2;

    private final Engine engine;
    private final World world;

    /**
     * Keeps the books of an action without carrying any of it out.
     *
     * Set while a replay plays back. The recorded commands are the only ones the match may receive, so nothing goes to the engine; squad membership and contracts are still taken on, because the observation describes squads from them, and a command chain run beside a person's play forms its squads and issues its contracts over the person's units. Lifts and unit actions are commands and nothing else, so they are not taken on at all.
     */
    boolean shadow = false;

    Commander(Engine engine, World world) {
        this.engine = engine;
        this.world = world;
    }

    void apply(Object game, byte[] body) throws Exception {
        Object self = shadow ? null : engine.local(game);
        if (!shadow && self == null) return;
        ByteBuffer in = ByteBuffer.wrap(body).order(ByteOrder.LITTLE_ENDIAN);

        int squadCount = in.getShort() & 0xFFFF;
        for (int i = 0; i < squadCount; i++) {
            int squadId = in.getShort() & 0xFFFF;
            int commander = in.get() & 0xFF;
            // Nought is the player this process is; anything else is that player's slot plus one.
            int owner = (in.get() & 0xFF) - 1;
            int members = in.getShort() & 0xFFFF;
            in.getShort();  // padding
            List<Long> units = new ArrayList<Long>(members);
            for (int j = 0; j < members; j++) units.add(Long.valueOf(in.getInt() & 0xFFFFFFFFL));
            world.assign(squadId, commander, owner, units);
        }

        int contractCount = in.getShort() & 0xFFFF;
        for (int i = 0; i < contractCount; i++) {
            int squadId = in.getShort() & 0xFFFF;
            int task = in.get() & 0xFF;
            int stance = in.get() & 0xFF;
            int targetKind = in.get() & 0xFF;
            boolean override = (in.get() & 0xFF) != 0;
            in.getShort();  // padding
            long target = in.getInt() & 0xFFFFFFFFL;
            float budget = in.getFloat();
            int deadline = in.getInt();
            int issuedAt = in.getInt();
            applyContract(game, self, squadId, task, stance, targetKind, target, budget, deadline, issuedAt, override);
        }

        int deviationCount = in.getShort() & 0xFFFF;
        for (int i = 0; i < deviationCount; i++) {
            int squadId = in.getShort() & 0xFFFF;
            int deviation = in.get() & 0xFF;
            boolean override = (in.get() & 0xFF) != 0;
            if (!shadow) applyDeviation(game, self, squadId, deviation, override);
        }

        int productionCount = in.getShort() & 0xFFFF;
        for (int i = 0; i < productionCount; i++) {
            long producer = in.getInt() & 0xFFFFFFFFL;
            int typeIndex = in.getShort() & 0xFFFF;
            int kind = in.get() & 0xFF;
            boolean cancel = (in.get() & 0xFF) != 0;
            float x = in.getFloat();
            float y = in.getFloat();
            if (!shadow) produce(game, self, producer, typeIndex, kind, cancel, x, y);
        }

        int liftCount = in.getShort() & 0xFFFF;
        for (int i = 0; i < liftCount; i++) {
            int liftId = in.getShort() & 0xFFFF;
            int cargoKind = in.get() & 0xFF;
            int flags = in.get() & 0xFF;
            int dropRegion = in.get() & 0xFF;
            int transportCount = in.get() & 0xFF;
            int cargoCount = in.getShort() & 0xFFFF;
            float pickupX = in.getFloat();
            float pickupY = in.getFloat();
            float dropX = in.getFloat();
            float dropY = in.getFloat();
            int deadline = in.getInt();
            List<Long> transports = new ArrayList<Long>(transportCount);
            for (int j = 0; j < transportCount; j++) transports.add(Long.valueOf(in.getInt() & 0xFFFFFFFFL));
            List<Long> cargo = new ArrayList<Long>(cargoCount);
            for (int j = 0; j < cargoCount; j++) cargo.add(Long.valueOf(in.getInt() & 0xFFFFFFFFL));
            if (!shadow) {
                applyLift(game, self, liftId, cargoKind, (flags & LIFT_OVERRIDE) != 0, (flags & LIFT_CANCEL) != 0,
                        dropRegion, transports, cargo, pickupX, pickupY, dropX, dropY, deadline);
            }
        }

        int actionCount = in.getShort() & 0xFFFF;
        for (int i = 0; i < actionCount; i++) {
            long unitId = in.getInt() & 0xFFFFFFFFL;
            boolean append = (in.get() & 0xFF) != 0;
            int length = in.get() & 0xFF;
            byte[] name = new byte[length];
            in.get(name);
            if (!shadow) unitAction(game, self, unitId, append, new String(name, java.nio.charset.StandardCharsets.US_ASCII));
        }
    }

    /**
     * Carries the period's standing work forward: lifts take their next step, a lift that has ended hands its squad back to its contract, a raid moves on to its next quarry and an escort keeps to the one it guards.
     *
     * Runs once a period after the action is applied and before the world is scanned, so the scan reports what this did.
     */
    void upkeep(Object game) throws Exception {
        if (shadow) return;
        Object self = engine.local(game);
        if (self == null) return;
        int now = engine.gameTime(game);
        java.util.Iterator<Lift> lifts = world.lifts.values().iterator();
        while (lifts.hasNext()) {
            Lift lift = lifts.next();
            lift.step(game, self, now);
            if (!lift.ended()) continue;
            lifts.remove();
            release(game, self, lift);
        }
        for (World.Squad squad : world.squads.values()) {
            // Members set down since the last period hold no order, whatever set them down, so the contract is carried out again with them in it.
            boolean landed = squad.aboard < squad.aboardSeen;
            squad.aboardSeen = squad.aboard;
            if (squad.issuedAtMs == 0 || squad.lift != null || squad.units.isEmpty()) continue;
            if (landed) {
                execute(game, self, squad);
                continue;
            }
            if (squad.lastDeviation != HOLD || squad.unreachable) continue;
            if (squad.task == RAID) raid(game, self, squad);
            else if (squad.task == ESCORT) escort(game, self, squad);
        }
    }

    /**
     * Takes a contract on, and starts the squad on it.
     *
     * A contract that is the same one again is not re-applied. Re-issuing resets the value the losses are measured from, so a contract re-sent every period would report a squad as having lost nothing however much of it had been destroyed.
     */
    private void applyContract(Object game, Object self, int squadId, int task, int stance, int targetKind, long target,
                               float budget, int deadline, int issuedAt, boolean override) throws Exception {
        World.Squad squad = world.squads.get(Integer.valueOf(squadId));
        // A contract for a squad that does not exist is not an instruction to invent one. Squads are formed by handing over a roster, and creating one here would put a phantom into a slot the observation reports.
        if (squad == null) return;
        // A squad someone else has taken the operational command of is not the operational layer's to re-task, and the override bit is how the one who did take it says so. Refusing both would make taking a squad over a way of silencing it rather than a way of commanding it, which is the opposite of what the intervention interface is for.
        if ((squad.commander & World.HUMAN_OPERATIONS) != 0 && !override) return;
        boolean changed = squad.task != task || squad.targetKind != targetKind || squad.target != target
                || squad.stance != stance || squad.issuedAtMs != issuedAt;
        squad.task = task;
        squad.stance = stance;
        squad.targetKind = targetKind;
        squad.target = target;
        if (targetKind == World.TARGET_REGION) squad.targetRegion = (int) target;
        squad.costBudget = budget;
        squad.deadlineMs = deadline;
        if (changed) {
            // The value the losses are measured from is taken at the next scan rather than now. A squad formed in this same action has not been counted yet, and a baseline of zero would report it as having lost nothing however much of it was destroyed.
            squad.rebaseline = true;
            squad.issuedAtMs = issuedAt > 0 ? issuedAt : engine.gameTime(game);
            squad.balanceMovedAtMs = 0;
            squad.unreachable = false;
            squad.targetLost = false;
        }
        // A squad being lifted takes its contract on and carries it out once it is set down.
        if (shadow || !changed || squad.units.isEmpty() || squad.lift != null) return;
        execute(game, self, squad);
    }

    /**
     * Departs from the contract, or returns to it.
     *
     * Everything but holding has to be re-issued every period, because each is a reaction to where things are at that moment. Holding is issued once, when the squad returns to its contract, and then left to the engine, which is already carrying it out.
     */
    private void applyDeviation(Object game, Object self, int squadId, int deviation,
                                boolean override) throws Exception {
        World.Squad squad = world.squads.get(Integer.valueOf(squadId));
        if (squad == null || squad.units.isEmpty() || squad.lift != null) return;
        // A departure is a departure from a contract, so a squad that has never been given one is left alone. Without this a squad nobody has tasked reads its target as region zero and is marched to whatever happens to be there, which on a map between two players is the other player's base.
        if (squad.issuedAtMs == 0) return;
        if ((squad.commander & World.HUMAN_TACTICS) != 0 && !override) return;
        float[] target = world.targetPoint(squad);
        if (target == null) return;

        if (deviation == FOCUS) focus(game, self, squad);
        else if (deviation == FOCUS_THREAT) focusThreat(game, self, squad);
        else if (deviation == SPREAD_OUT) spread(game, self, squad);
        else if (deviation == WITHDRAW_BACK) fallBack(game, self, squad, target, HOLD_FIRE, FALL_BACK);
        // The whole way out rather than the short step WITHDRAW takes: the distance is bounded to the same limit fallBack backs any withdrawal off to, so this asks for that limit and gets as much of it as the squad has not already used.
        else if (deviation == WITHDRAW_FAR) fallBack(game, self, squad, target, HOLD_FIRE, MAX_WITHDRAWAL);
        else if (deviation == KITE) kite(game, self, squad, target);
        else if (deviation == HOLD && squad.lastDeviation != HOLD) execute(game, self, squad);
        // Holding is the one departure that is issued once and then left to the engine, so it is only recorded when the order actually went out. Recording it after an order that could not be issued would leave the squad believing it was carrying out its contract with nothing to carry it out.
        if (deviation != HOLD) squad.lastDeviation = deviation;
    }

    /**
     * The player whose name an order to this squad goes out in.
     *
     * Ordinarily that is this process's own player and the question does not arise. It arises in a constructed engagement, where the sandbox flag lets one process drive both sides: a command is taken out of the pool for a player, and one taken out for the wrong player addresses units that are not that player's.
     */
    private Object issuer(Object game, Object self, World.Squad squad) throws Exception {
        if (squad.owner < 0) return self;
        Object player = engine.playerAt(squad.owner);
        return player == null ? self : player;
    }

    // Carrying out a contract.

    /** Carries out the squad's contract with the members that can reach its target; reports the contract unreachable when none can. */
    private void execute(Object game, Object self, World.Squad squad) throws Exception {
        squad.raidTarget = 0L;
        squad.escortLead = 0L;
        float[] point = world.targetPoint(squad);
        if (point == null) {
            squad.targetLost = true;
            squad.lastDeviation = -1;
            return;
        }
        List<Object> free = free(squad);
        List<Object> able = new ArrayList<Object>();
        for (Object unit : free) if (world.reachable(unit, point[0], point[1])) able.add(unit);
        squad.unreachable = !free.isEmpty() && able.isEmpty();
        if (able.isEmpty()) {
            squad.lastDeviation = -1;
            return;
        }
        Object issuer = issuer(game, self, squad);
        if (squad.task == ENCIRCLE && able.size() >= 2) {
            encircle(game, issuer, squad, able, point);
        } else if (squad.task == DEFEND || squad.task == WITHDRAW) {
            Object command = order(game, issuer, able, squad.stance);
            engine.moveTo(command, point[0], point[1]);
        } else if (squad.task == ESCORT) {
            escort(game, self, squad);
        } else if (squad.task == RAID) {
            raid(game, self, squad);
        } else {
            Object command = order(game, issuer, able, squad.stance);
            engine.attackMoveTo(command, point[0], point[1]);
        }
        squad.lastDeviation = HOLD;
    }

    /** Splits the squad across its line of approach and sends each half at the target from its own side. */
    private void encircle(Object game, Object issuer, World.Squad squad, List<Object> able, float[] point) throws Exception {
        float ax = point[0] - squad.x;
        float ay = point[1] - squad.y;
        float length = (float) Math.sqrt(ax * ax + ay * ay);
        if (length < 1f) {
            ax = 1f;
            ay = 0f;
            length = 1f;
        }
        final float px = -ay / length;
        final float py = ax / length;
        final World.Squad centre = squad;
        List<Object> ordered = new ArrayList<Object>(able);
        Collections.sort(ordered, new Comparator<Object>() {
            public int compare(Object a, Object b) {
                return Float.compare(side(a, centre, px, py), side(b, centre, px, py));
            }
        });
        int half = ordered.size() / 2;
        Object left = order(game, issuer, ordered.subList(0, half), squad.stance);
        engine.attackMoveTo(left, point[0] - px * ENCIRCLE_OFFSET, point[1] - py * ENCIRCLE_OFFSET);
        Object right = order(game, issuer, ordered.subList(half, ordered.size()), squad.stance);
        engine.attackMoveTo(right, point[0] + px * ENCIRCLE_OFFSET, point[1] + py * ENCIRCLE_OFFSET);
    }

    private float side(Object unit, World.Squad squad, float px, float py) {
        try {
            return (engine.x(unit) - squad.x) * px + (engine.y(unit) - squad.y) * py;
        } catch (Exception e) {
            return 0f;
        }
    }

    /** Goes for the nearest enemy extractor or builder around the target, moving to the next when it falls, and advances on the target when there is none. */
    private void raid(Object game, Object self, World.Squad squad) throws Exception {
        float[] point = world.targetPoint(squad);
        if (point == null) return;
        World.Seen quarry = null;
        float best = RAID_RADIUS * RAID_RADIUS;
        for (World.Seen seen : world.visible) {
            if (!opposes(squad, seen) || seen.handle == null || seen.carrier != 0L) continue;
            if (!seen.builder && !(seen.building && seen.extractor)) continue;
            float dx = seen.x - point[0];
            float dy = seen.y - point[1];
            float distance = dx * dx + dy * dy;
            if (distance < best) {
                best = distance;
                quarry = seen;
            }
        }
        long wanted = quarry == null ? ADVANCING : quarry.id;
        if (wanted == squad.raidTarget) return;
        List<Object> able = reaching(squad, point);
        if (able.isEmpty()) return;
        Object command = order(game, issuer(game, self, squad), able, squad.stance);
        if (quarry == null) engine.attackMoveTo(command, point[0], point[1]);
        else engine.attack(command, quarry.handle);
        squad.raidTarget = wanted;
    }

    /** Guards the unit the contract names, or the member of the squad it names that is furthest forward, reissued only when that changes. */
    private void escort(Object game, Object self, World.Squad squad) throws Exception {
        Object lead = null;
        if (squad.targetKind == World.TARGET_UNIT) {
            lead = world.handle(squad.target);
        } else if (squad.targetKind == World.TARGET_SQUAD) {
            World.Squad other = world.squads.get(Integer.valueOf((int) squad.target));
            if (other != null) lead = front(other);
        }
        if (lead == null) {
            // An escort sent to a region has nobody to guard, and stands on the region instead.
            if (squad.targetKind == World.TARGET_REGION && squad.escortLead != ADVANCING) {
                float[] point = world.targetPoint(squad);
                List<Object> able = point == null ? new ArrayList<Object>() : reaching(squad, point);
                if (able.isEmpty()) return;
                Object command = order(game, issuer(game, self, squad), able, squad.stance);
                engine.moveTo(command, point[0], point[1]);
                squad.escortLead = ADVANCING;
            }
            return;
        }
        long id = engine.id(lead);
        if (id == squad.escortLead) return;
        List<Object> able = reaching(squad, engine.x(lead), engine.y(lead));
        if (able.isEmpty()) return;
        Object command = order(game, issuer(game, self, squad), able, squad.stance);
        engine.guard(command, lead);
        squad.escortLead = id;
    }

    /** The member of a squad nearest the point its own contract aims at, which is the one at its head; the member nearest its centre when it has no contract. */
    private Object front(World.Squad squad) throws Exception {
        float[] aim = squad.issuedAtMs == 0 ? null : world.targetPoint(squad);
        float tx = aim == null ? squad.x : aim[0];
        float ty = aim == null ? squad.y : aim[1];
        Object best = null;
        float bestDistance = Float.MAX_VALUE;
        for (Long id : squad.units) {
            Object unit = world.handle(id.longValue());
            if (unit == null || engine.carrier(unit) != null) continue;
            float dx = engine.x(unit) - tx;
            float dy = engine.y(unit) - ty;
            float distance = dx * dx + dy * dy;
            if (distance < bestDistance) {
                bestDistance = distance;
                best = unit;
            }
        }
        return best;
    }

    private List<Object> reaching(World.Squad squad, float[] point) throws Exception {
        return reaching(squad, point[0], point[1]);
    }

    private List<Object> reaching(World.Squad squad, float x, float y) throws Exception {
        List<Object> able = new ArrayList<Object>();
        for (Object unit : free(squad)) if (world.reachable(unit, x, y)) able.add(unit);
        return able;
    }

    /** The members that can take an order: alive, able to hold orders, not aboard a transport, and not the cargo of a lift. */
    private List<Object> free(World.Squad squad) throws Exception {
        List<Object> out = new ArrayList<Object>();
        for (Long id : squad.units) {
            Object unit = world.handle(id.longValue());
            if (unit == null || !engine.armedClass.isInstance(unit) || engine.carrier(unit) != null) continue;
            if (lifted(id.longValue())) continue;
            out.add(unit);
        }
        return out;
    }

    private boolean lifted(long unit) {
        for (Lift lift : world.lifts.values()) if (lift.holds(unit)) return true;
        return false;
    }

    private Object order(Object game, Object issuer, List<Object> units, int stance) throws Exception {
        Object command = engine.command(game, issuer);
        for (Object unit : units) engine.addUnit(command, unit);
        engine.setStance(command, stance);
        return command;
    }

    // Departures from a contract.

    /**
     * Whether a unit is on the other side from this squad.
     *
     * Hostility is recorded from this process's own point of view, which is the only point of view an ordinary match has. A constructed engagement drives both sides, so the squad standing in for the opponent has to read the flag the other way round or it would concentrate its fire on its own side.
     */
    private static boolean opposes(World.Squad squad, World.Seen seen) {
        return seen.hostile == (squad.owner < 0);
    }

    /** Every unit onto the weakest enemy within reach, which removes an enemy from the fight sooner than spreading the damage. Reach is the squad's own weapon range, not a fixed distance: an order to attack something the squad has to walk to is an order to break formation. */
    private void focus(Object game, Object self, World.Squad squad) throws Exception {
        float reach = reachOf(squad);
        World.Seen best = null;
        float bestHealth = Float.MAX_VALUE;
        for (World.Seen seen : world.visible) {
            if (!opposes(squad, seen)) continue;
            float dx = seen.x - squad.x;
            float dy = seen.y - squad.y;
            if (dx * dx + dy * dy > reach * reach) continue;
            if (seen.health < bestHealth) {
                bestHealth = seen.health;
                best = seen;
            }
        }
        if (best == null || best.handle == null) return;
        Object command = engine.command(game, issuer(game, self, squad));
        if (!addAll(command, squad)) return;
        engine.setStance(command, squad.stance);
        engine.attack(command, best.handle);
    }

    /** Every unit onto the longest-ranged enemy within reach, rather than the weakest focus picks. The gun that out-reaches the squad does the most damage while it lives and dies to a focused volley like anything else once the squad is close enough to it, so taking it out first buys more than shortening the count by one. Reach is the squad's own weapon range, as in focus, because an order to attack something the squad has to walk to is an order to break formation. */
    private void focusThreat(Object game, Object self, World.Squad squad) throws Exception {
        float reach = reachOf(squad);
        World.Seen best = null;
        float bestRange = -1f;
        for (World.Seen seen : world.visible) {
            if (!opposes(squad, seen) || seen.handle == null) continue;
            float dx = seen.x - squad.x;
            float dy = seen.y - squad.y;
            if (dx * dx + dy * dy > reach * reach) continue;
            float range = world.rangeOf(seen.handle);
            if (range > bestRange) {
                bestRange = range;
                best = seen;
            }
        }
        if (best == null) return;
        Object command = engine.command(game, issuer(game, self, squad));
        if (!addAll(command, squad)) return;
        engine.setStance(command, squad.stance);
        engine.attack(command, best.handle);
    }

    /** Pushes the squad apart radially, which is what an area weapon is answered with. Necessarily one command per unit. */
    private void spread(Object game, Object self, World.Squad squad) throws Exception {
        int index = 0;
        List<Object> members = free(squad);
        for (Object unit : members) {
            double angle = 2 * Math.PI * index++ / Math.max(1, members.size());
            Object command = engine.command(game, issuer(game, self, squad));
            engine.addUnit(command, unit);
            engine.moveTo(command, squad.x + (float) (Math.cos(angle) * SPREAD),
                    squad.y + (float) (Math.sin(angle) * SPREAD));
        }
    }

    /**
     * Pulls the squad back from its target, along the line it came in on.
     *
     * Bounded at both ends, and both bounds are about the same thing: a squad ordered to back away is given the order afresh every period, so an order with nothing to back away from and no limit on how far is an order to leave the map. So there has to be an enemy close enough to be backing away from, and there is a distance from the contracted ground past which the squad has plainly finished backing away and is simply walking.
     */
    private void fallBack(Object game, Object self, World.Squad squad, float[] target,
                          int stance, float distance) throws Exception {
        if (!inContact(squad)) return;
        float dx = squad.x - target[0];
        float dy = squad.y - target[1];
        float length = (float) Math.sqrt(dx * dx + dy * dy);
        if (length < 1f) {
            dx = 1f;
            dy = 0f;
            length = 1f;
        }
        if (length >= MAX_WITHDRAWAL) return;
        distance = Math.min(distance, MAX_WITHDRAWAL - length);
        Object command = engine.command(game, issuer(game, self, squad));
        if (!addAll(command, squad)) return;
        engine.setStance(command, stance);
        engine.moveTo(command, squad.x + dx / length * distance, squad.y + dy / length * distance);
    }

    /**
     * Backs off just far enough to be out of the enemy's reach while staying inside our own, and goes on shooting.
     *
     * This is the whole of what kiting is, and it is why it is a different departure from withdrawing rather than the same one with the safety off: a squad that outranges what it is fighting wins by keeping exactly that difference, and one that does not gains nothing by trying.
     */
    private void kite(Object game, Object self, World.Squad squad, float[] target) throws Exception {
        float advantage = reachOf(squad) - enemyReachNear(squad);
        if (advantage <= 0f) {
            fallBack(game, self, squad, target, HOLD_FIRE, FALL_BACK);
            return;
        }
        fallBack(game, self, squad, target, squad.stance, advantage);
    }

    /** Whether anything on the other side is close enough for this squad to be manoeuvring against it. */
    private boolean inContact(World.Squad squad) {
        for (World.Seen seen : world.visible) {
            if (!opposes(squad, seen) || seen.handle == null) continue;
            float dx = seen.x - squad.x;
            float dy = seen.y - squad.y;
            if (dx * dx + dy * dy <= CONTACT * CONTACT) return true;
        }
        return false;
    }

    /** How far the squad can shoot, taken as the shortest reach among its armed members, since that is the range at which all of it is in the fight. */
    private float reachOf(World.Squad squad) {
        float reach = Float.MAX_VALUE;
        for (Long id : squad.units) {
            Object unit = world.handle(id.longValue());
            if (unit == null || !engine.armedClass.isInstance(unit)) continue;
            float range = world.rangeOf(unit);
            if (range > 0f && range < reach) reach = range;
        }
        return reach == Float.MAX_VALUE ? FALL_BACK : reach;
    }

    /** How far the enemies close enough to matter can shoot, taken as the longest among them. */
    private float enemyReachNear(World.Squad squad) {
        float reach = 0f;
        for (World.Seen seen : world.visible) {
            if (!opposes(squad, seen) || seen.handle == null) continue;
            float dx = seen.x - squad.x;
            float dy = seen.y - squad.y;
            if (dx * dx + dy * dy > FALL_BACK * FALL_BACK) continue;
            float range = world.rangeOf(seen.handle);
            if (range > reach) reach = range;
        }
        return reach;
    }

    private boolean addAll(Object command, World.Squad squad) throws Exception {
        List<Object> members = free(squad);
        for (Object unit : members) engine.addUnit(command, unit);
        return !members.isEmpty();
    }

    // Lifts and unit actions.

    /**
     * Starts a lift, restates one already under way, or cancels one.
     *
     * A row that restates a lift under way changes only its deadline, so the lift goes on from the step it has reached. A row that describes a different lift under an id already in use replaces it. A squad someone else holds the operational command of is lifted only on that commander's word, by the same rule its contracts follow.
     */
    private void applyLift(Object game, Object self, int liftId, int cargoKind, boolean override, boolean cancel,
                           int dropRegion, List<Long> transports, List<Long> cargo,
                           float pickupX, float pickupY, float dropX, float dropY, int deadline) throws Exception {
        Lift existing = world.lifts.get(Integer.valueOf(liftId));
        int squadId = cargoKind == Lift.CARGO_SQUAD && !cargo.isEmpty() ? cargo.get(0).intValue() : World.NO_SQUAD;
        World.Squad squad = squadId == World.NO_SQUAD ? null : world.squads.get(Integer.valueOf(squadId));
        World.Squad carried = existing == null ? null : world.squads.get(Integer.valueOf(existing.cargoSquad));
        if (!override && (heldByHuman(squad) || heldByHuman(carried))) return;
        if (cancel) {
            if (existing != null) {
                existing.abandon(game, self, Lift.CANCELLED);
                world.lifts.remove(Integer.valueOf(liftId));
                release(game, self, existing);
            }
            return;
        }
        List<Long> units = cargoKind == Lift.CARGO_UNITS ? cargo : new ArrayList<Long>();
        if (existing != null && existing.sameAs(transports, cargoKind, squadId, units, dropRegion, dropX, dropY)) {
            existing.deadlineMs = deadline;
            return;
        }
        if (cargoKind == Lift.CARGO_SQUAD && squad == null) return;
        if (existing != null) {
            existing.abandon(game, self, Lift.CANCELLED);
            world.lifts.remove(Integer.valueOf(liftId));
            release(game, self, existing);
        }
        if (squad != null && squad.lift != null) {
            Lift held = squad.lift;
            held.abandon(game, self, Lift.CANCELLED);
            world.lifts.remove(Integer.valueOf(held.id));
            release(game, self, held);
        }
        Lift lift = new Lift(engine, world, liftId, transports, cargoKind, squadId, units,
                pickupX, pickupY, dropRegion, dropX, dropY, deadline);
        if (squad != null) squad.lift = lift;
        lift.start(game, self, engine.gameTime(game));
        if (lift.ended()) {
            release(game, self, lift);
            return;
        }
        world.lifts.put(Integer.valueOf(liftId), lift);
    }

    /** Reports a lift that has ended and hands its squad back to its contract, which is carried out afresh from wherever the lift left it. */
    private void release(Object game, Object self, Lift lift) throws Exception {
        world.endedLifts.add(lift);
        world.events.add(new World.Event(lift.phase == Lift.DONE ? World.EVENT_LIFT_DONE : World.EVENT_LIFT_FAILED,
                lift.cargoSquad, lift.id, 0xFFFF, lift.reason));
        World.Squad squad = unbind(lift);
        if (squad == null || squad.issuedAtMs == 0 || squad.units.isEmpty()) return;
        squad.unreachable = false;
        execute(game, self, squad);
    }

    private static boolean heldByHuman(World.Squad squad) {
        return squad != null && (squad.commander & World.HUMAN_OPERATIONS) != 0;
    }

    private World.Squad unbind(Lift lift) {
        if (lift.cargoKind != Lift.CARGO_SQUAD) return null;
        World.Squad squad = world.squads.get(Integer.valueOf(lift.cargoSquad));
        if (squad == null || squad.lift != lift) return null;
        squad.lift = null;
        return squad;
    }

    /** Issues one of the unit's own actions by its action id. */
    private void unitAction(Object game, Object self, long unitId, boolean append, String id) throws Exception {
        Object unit = world.handle(unitId);
        if (unit == null || !engine.armedClass.isInstance(unit)) return;
        Object command = engine.command(game, self);
        if (append) engine.append(command);
        engine.addUnit(command, unit);
        engine.unitAction(command, unit, id);
    }

    // Production.

    /**
     * Produces a unit at a factory, places a building with a builder, or raises a building to its next tier.
     *
     * Both are special actions rather than orders, and the action id is built from the name the type reports, which is not always the name it was looked up under. Placing a building needs the position as well, so the build order carries the placement and the action names what is being placed.
     */
    private void produce(Object game, Object self, long producerId, int typeIndex, int kind,
                         boolean cancel, float x, float y) throws Exception {
        Object producer = world.handle(producerId);
        if (producer == null || !engine.armedClass.isInstance(producer)) return;
        if (kind == KIND_UPGRADE) {
            // A tier raise names no type: the building offers at most one at its current tier, and that is the one issued.
            Object upgrade = engine.upgradeOffered(producer);
            if (upgrade == null) return;
            Object command = engine.command(game, self);
            engine.addUnit(command, producer);
            engine.offeredAction(command, upgrade);
            if (cancel) engine.setField(command, "g", Boolean.TRUE);
            return;
        }
        Object type = world.typeAt(typeIndex);
        if (type == null) return;

        Object command = engine.command(game, self);
        engine.addUnit(command, producer);
        if (kind == KIND_BUILDING) {
            engine.specialAction(command, "b_" + engine.typeName(type));
            engine.build(command, x, y, type, 1);
        } else if (kind == KIND_UNIT) {
            engine.specialAction(command, "u_" + engine.typeName(type));
        } else {
            return;
        }
        if (cancel) engine.setField(command, "g", Boolean.TRUE);
    }
}
