import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Turns a decoded action into engine commands, on the game thread.
 *
 * The default for a squad under a contract is to advance on the target region with the contract's stance, which leaves the engine's own path finding, target acquisition and engagement to do the work. What the tactical layer chooses is only whether to depart from that and how; a squad choosing to hold gets no command at all until its contract changes.
 *
 * Units taking the same order go into one command. The engine accepts a list, and a squad is by construction a set of units that were meant to act together.
 */
final class Commander {

    /** How far a withdrawing or kiting squad pulls back, roughly a squad's own frontage plus a tank's reach. */
    private static final float FALL_BACK = 400f;

    /** How far apart a spreading squad ends up, chosen to clear the radius of an area weapon. */
    private static final float SPREAD = 140f;

    /** Deviations, matching `rwintel/wire/action.py`. */
    private static final int HOLD = 0;
    private static final int WITHDRAW = 1;
    private static final int FOCUS = 2;
    private static final int SPREAD_OUT = 3;
    private static final int KITE = 4;

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
            int members = in.getShort() & 0xFFFF;
            List<Long> units = new ArrayList<Long>(members);
            for (int j = 0; j < members; j++) units.add(Long.valueOf(in.getInt() & 0xFFFFFFFFL));
            world.assign(squadId, units);
        }

        int contractCount = in.getShort() & 0xFFFF;
        for (int i = 0; i < contractCount; i++) {
            int squadId = in.getShort() & 0xFFFF;
            int task = in.get() & 0xFF;
            int stance = in.get() & 0xFF;
            int region = in.get() & 0xFF;
            int deviation = in.get() & 0xFF;
            in.getShort();  // padding
            int budget = in.getInt();
            int deadline = in.getInt();
            applyContract(game, self, squadId, task, stance, region, deviation, budget, deadline);
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

    private void applyContract(Object game, Object self, int squadId, int task, int stance,
                               int region, int deviation, int budget, int deadline) throws Exception {
        World.Squad squad = world.squad(squadId);
        boolean changed = squad.task != task || squad.targetRegion != region || squad.deadlineMs != deadline;
        squad.task = task;
        squad.targetRegion = region;
        squad.costBudget = budget;
        squad.deadlineMs = deadline;
        if (changed) {
            squad.valueAtIssue = squad.value;
            squad.issuedAtMs = engine.gameTime(game);
        }
        if (squad.units.isEmpty()) return;

        World.Region target = world.regionAt(region);
        if (target == null) return;

        // A squad holding its contract is left to the engine, which is already advancing it. Anything else has to be re-issued every period, because it is a reaction to where things are now.
        if (deviation == HOLD) {
            if (changed) advance(game, self, squad, target, stance);
            return;
        }
        if (deviation == FOCUS) focus(game, self, squad, stance);
        else if (deviation == SPREAD_OUT) spread(game, self, squad);
        else fallBack(game, self, squad, target, stance, deviation == KITE);
    }

    private void advance(Object game, Object self, World.Squad squad, World.Region target, int stance) throws Exception {
        Object command = engine.command(game, self);
        if (!addAll(command, squad)) return;
        engine.setStance(command, stance);
        engine.attackMoveTo(command, target.x, target.y);
    }

    /** Every unit onto the weakest enemy within reach, which removes an enemy from the fight sooner than spreading the damage. */
    private void focus(Object game, Object self, World.Squad squad, int stance) throws Exception {
        World.Seen best = null;
        float bestHealth = Float.MAX_VALUE;
        for (World.Seen seen : world.visible) {
            if (!seen.hostile) continue;
            float dx = seen.x - squad.x;
            float dy = seen.y - squad.y;
            if (dx * dx + dy * dy > FALL_BACK * FALL_BACK) continue;
            if (seen.health < bestHealth) {
                bestHealth = seen.health;
                best = seen;
            }
        }
        if (best == null || best.handle == null) return;
        Object command = engine.command(game, self);
        if (!addAll(command, squad)) return;
        engine.setStance(command, stance);
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
            Object command = engine.command(game, self);
            engine.addUnit(command, unit);
            engine.moveTo(command, squad.x + (float) (Math.cos(angle) * SPREAD),
                    squad.y + (float) (Math.sin(angle) * SPREAD));
        }
    }

    /**
     * Pulls the squad back from its target.
     * Kiting differs from withdrawing only in the stance it keeps: it goes on shooting on the way out, which is the point of it against a shorter ranged enemy.
     */
    private void fallBack(Object game, Object self, World.Squad squad, World.Region target,
                          int stance, boolean shooting) throws Exception {
        float dx = squad.x - target.x;
        float dy = squad.y - target.y;
        float length = (float) Math.sqrt(dx * dx + dy * dy);
        if (length < 1f) {
            dx = 1f;
            dy = 0f;
            length = 1f;
        }
        float toX = squad.x + dx / length * FALL_BACK;
        float toY = squad.y + dy / length * FALL_BACK;

        Object command = engine.command(game, self);
        if (!addAll(command, squad)) return;
        engine.setStance(command, shooting ? stance : 3);  // holdFire while breaking off, so nothing turns to fight
        engine.moveTo(command, toX, toY);
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

    private void produce(Object game, Object self, long producerId, int typeIndex, int kind,
                         boolean cancel, float x, float y) throws Exception {
        Object producer = world.handle(producerId);
        Object type = world.typeAt(typeIndex);
        if (producer == null || type == null || !engine.armedClass.isInstance(producer)) return;

        Object command = engine.command(game, self);
        engine.addUnit(command, producer);
        if (kind == KIND_BUILDING) {
            engine.build(command, x, y, type, 1);
            return;
        }
        if (kind != KIND_UNIT) return;
        // The action id is built from the name the type reports, which is not always the name it was looked up under.
        engine.specialAction(command, "u_" + engine.typeName(type));
        if (cancel) engine.setField(command, "g", Boolean.TRUE);
    }
}
