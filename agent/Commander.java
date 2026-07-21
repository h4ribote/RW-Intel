import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Turns a decoded action into engine commands, on the game thread.
 *
 * The default for a squad under a contract is to advance on the target region with the contract's stance, which leaves the engine's own path finding, target acquisition and engagement to do the work. What the tactical layer chooses is only whether to depart from that and how; a squad choosing to hold gets no command at all until its contract changes.
 *
 * The two arrive on different periods and so travel in different sections: a contract is rewritten when the operational layer runs, a departure from it is chosen at the tactical rate. Folding them together would make the slower decision be re-sent at the faster rate, and re-issuing a contract is what resets the losses it is measured against.
 *
 * Units taking the same order go into one command. The engine accepts a list, and a squad is by construction a set of units that were meant to act together.
 */
final class Commander {

    /** How far a withdrawing squad pulls back, roughly a squad's own frontage plus a tank's reach. */
    private static final float FALL_BACK = 400f;

    /** How far apart a spreading squad ends up, chosen to clear the radius of an area weapon. */
    private static final float SPREAD = 140f;

    /** Deviations, matching `rwintel/wire/action.py`. */
    private static final int HOLD = 0;
    private static final int WITHDRAW = 1;
    private static final int FOCUS = 2;
    private static final int SPREAD_OUT = 3;
    private static final int KITE = 4;

    /** The stance a squad breaking off contact is put into, so that nothing turns around to fight on the way out. */
    private static final int HOLD_FIRE = 3;

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

    private final Engine engine;
    private final World world;

    Commander(Engine engine, World world) {
        this.engine = engine;
        this.world = world;
    }

    void apply(Object game, byte[] body) throws Exception {
        Object self = engine.local(game);
        if (self == null) return;
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
            int region = in.get() & 0xFF;
            boolean override = (in.get() & 0xFF) != 0;
            in.getShort();  // padding
            float budget = in.getFloat();
            int deadline = in.getInt();
            int issuedAt = in.getInt();
            applyContract(game, self, squadId, task, stance, region, budget, deadline, issuedAt, override);
        }

        int deviationCount = in.getShort() & 0xFFFF;
        for (int i = 0; i < deviationCount; i++) {
            int squadId = in.getShort() & 0xFFFF;
            int deviation = in.get() & 0xFF;
            boolean override = (in.get() & 0xFF) != 0;
            applyDeviation(game, self, squadId, deviation, override);
        }

        int productionCount = in.getShort() & 0xFFFF;
        for (int i = 0; i < productionCount; i++) {
            long producer = in.getInt() & 0xFFFFFFFFL;
            int typeIndex = in.getShort() & 0xFFFF;
            int kind = in.get() & 0xFF;
            boolean cancel = (in.get() & 0xFF) != 0;
            float x = in.getFloat();
            float y = in.getFloat();
            produce(game, self, producer, typeIndex, kind, cancel, x, y);
        }
    }

    /**
     * Takes a contract on, and starts the squad on it.
     *
     * A contract that is the same one again is not re-applied. Re-issuing resets the value the losses are measured from, so a contract re-sent every period would report a squad as having lost nothing however much of it had been destroyed.
     */
    private void applyContract(Object game, Object self, int squadId, int task, int stance,
                               int region, float budget, int deadline, int issuedAt,
                               boolean override) throws Exception {
        World.Squad squad = world.squads.get(Integer.valueOf(squadId));
        // A contract for a squad that does not exist is not an instruction to invent one. Squads are formed by handing over a roster, and creating one here would put a phantom into a slot the observation reports.
        if (squad == null) return;
        // A squad someone else has taken the operational command of is not the operational layer's to re-task, and the override bit is how the one who did take it says so. Refusing both would make taking a squad over a way of silencing it rather than a way of commanding it, which is the opposite of what the intervention interface is for.
        if ((squad.commander & World.HUMAN_OPERATIONS) != 0 && !override) return;
        boolean changed = squad.task != task || squad.targetRegion != region
                || squad.stance != stance || squad.issuedAtMs != issuedAt;
        squad.task = task;
        squad.stance = stance;
        squad.targetRegion = region;
        squad.costBudget = budget;
        squad.deadlineMs = deadline;
        if (changed) {
            // The value the losses are measured from is taken at the next scan rather than now. A squad formed in this same action has not been counted yet, and a baseline of zero would report it as having lost nothing however much of it was destroyed.
            squad.rebaseline = true;
            squad.issuedAtMs = issuedAt > 0 ? issuedAt : engine.gameTime(game);
            squad.balanceMovedAtMs = 0;
        }
        if (!changed || squad.units.isEmpty()) return;

        World.Region target = world.regionAt(region);
        if (target != null) advance(game, self, squad, target, stance);
    }

    /**
     * Departs from the contract, or returns to it.
     *
     * Everything but holding has to be re-issued every period, because each is a reaction to where things are at that moment. Holding is issued once, when the squad returns to its contract, and then left to the engine, which is already advancing it.
     */
    private void applyDeviation(Object game, Object self, int squadId, int deviation,
                                boolean override) throws Exception {
        World.Squad squad = world.squads.get(Integer.valueOf(squadId));
        if (squad == null || squad.units.isEmpty()) return;
        // A departure is a departure from a contract, so a squad that has never been given one is left alone. Without this a squad nobody has tasked reads its target as region zero and is marched to whatever happens to be there, which on a map between two players is the other player's base.
        if (squad.issuedAtMs == 0) return;
        if ((squad.commander & World.HUMAN_TACTICS) != 0 && !override) return;
        World.Region target = world.regionAt(squad.targetRegion);
        if (target == null) return;

        if (deviation == FOCUS) focus(game, self, squad);
        else if (deviation == SPREAD_OUT) spread(game, self, squad);
        else if (deviation == WITHDRAW) fallBack(game, self, squad, target, HOLD_FIRE, FALL_BACK);
        else if (deviation == KITE) kite(game, self, squad, target);
        else if (deviation == HOLD && squad.lastDeviation != HOLD) advance(game, self, squad, target, squad.stance);
        // Holding is the one departure that is issued once and then left to the engine, so it is only recorded when the order actually went out. Recording it after an order that could not be issued would leave the squad believing it was advancing with nothing to advance it.
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

    private void advance(Object game, Object self, World.Squad squad, World.Region target, int stance) throws Exception {
        Object command = engine.command(game, issuer(game, self, squad));
        if (!addAll(command, squad)) {
            squad.lastDeviation = -1;
            return;
        }
        engine.setStance(command, stance);
        engine.attackMoveTo(command, target.x, target.y);
        squad.lastDeviation = HOLD;
    }

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

    /** Pushes the squad apart radially, which is what an area weapon is answered with. Necessarily one command per unit. */
    private void spread(Object game, Object self, World.Squad squad) throws Exception {
        int index = 0;
        int members = squad.units.size();
        for (Long id : squad.units) {
            Object unit = world.handle(id.longValue());
            if (unit == null || !engine.armedClass.isInstance(unit)) continue;
            double angle = 2 * Math.PI * index++ / Math.max(1, members);
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
    private void fallBack(Object game, Object self, World.Squad squad, World.Region target,
                          int stance, float distance) throws Exception {
        if (!inContact(squad)) return;
        float dx = squad.x - target.x;
        float dy = squad.y - target.y;
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
    private void kite(Object game, Object self, World.Squad squad, World.Region target) throws Exception {
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
        boolean any = false;
        for (Long id : squad.units) {
            Object unit = world.handle(id.longValue());
            if (unit == null || !engine.armedClass.isInstance(unit)) continue;
            engine.addUnit(command, unit);
            any = true;
        }
        return any;
    }

    /**
     * Produces a unit at a factory, or places a building with a builder.
     *
     * Both are special actions rather than orders, and the action id is built from the name the type reports, which is not always the name it was looked up under. Placing a building needs the position as well, so the build order carries the placement and the action names what is being placed.
     */
    private void produce(Object game, Object self, long producerId, int typeIndex, int kind,
                         boolean cancel, float x, float y) throws Exception {
        Object producer = world.handle(producerId);
        Object type = world.typeAt(typeIndex);
        if (producer == null || type == null || !engine.armedClass.isInstance(producer)) return;

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
