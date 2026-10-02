import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * The orders the built-in AI players give their units, read off the command pool for the control process to learn from.
 *
 * The pool's queue (`c.b`) is replaced by a {@link Tap} holding the same commands, which also keeps a reference to every command appended to it. A command is filled in by the AI after the pool hands it over, so it is read at the head of the next step, once the AI's call has returned; the pool makes a fresh command each time, so the reference still holds what was issued after the pool has carried it out. Commands issued as a computer player other than the local player are kept, which excludes everything the controlled side issues, since that is always issued as the local player. Orders accumulate until the next operational observation carries them.
 */
final class AiOrders {

    /** Kinds of order on the wire, matching `rwintel.wire.AiOrderKind`. */
    static final int MOVE = 0;
    static final int ATTACK_MOVE = 1;
    static final int ATTACK = 2;
    static final int OTHER_MOVEMENT = 3;
    static final int LOAD_INTO = 4;
    static final int LOAD_UP = 5;
    static final int UNLOAD = 6;
    static final int CANCEL_UNLOAD = 7;
    static final int STOP = 8;

    static final int FLAG_APPEND = 1;
    static final int NO_UNIT = 0xFFFFFFFF;

    /** Bytes of one order before its unit ids. */
    static final int ORDER_SIZE = 22;

    /** The engine's order kinds (`av`) by position, as `rwintel/replay/commands.py` lists them. */
    private static final int KIND_MOVE = 0;
    private static final int KIND_ATTACK = 1;
    private static final int KIND_LOAD_INTO = 4;
    private static final int KIND_ATTACK_MOVE = 7;
    private static final int KIND_LOAD_UP = 8;
    private static final int KIND_PATROL = 9;
    private static final int KIND_GUARD = 10;
    private static final int KIND_GUARD_AT = 11;
    private static final int KIND_FOLLOW = 13;

    /** The command pool's queue with every append also noted in a list of its own, until reading the orders has failed. */
    static final class Tap extends ArrayList<Object> {
        private final AiOrders owner;

        Tap(Collection<?> existing, AiOrders owner) {
            super(existing);
            this.owner = owner;
        }

        @Override
        public boolean add(Object command) {
            if (!owner.failed) owner.pending.add(command);
            return super.add(command);
        }

        @Override
        public void add(int index, Object command) {
            if (!owner.failed) owner.pending.add(command);
            super.add(index, command);
        }

        @Override
        public boolean addAll(Collection<?> commands) {
            if (!owner.failed) owner.pending.addAll(commands);
            return super.addAll(commands);
        }
    }

    /** One order as the wire carries it. */
    static final class Order {
        int timeMs;
        int issuer;
        int kind;
        int flags;
        float x = Float.NaN;
        float y = Float.NaN;
        int target = NO_UNIT;
        int[] units;
    }

    private final Engine engine;
    private final List<Object> pending = Collections.synchronizedList(new ArrayList<Object>());
    /** Orders read since the last operational observation carried them. */
    final List<Order> taken = new ArrayList<Order>();
    private int written;
    private Object unloadHandle;
    private Object cancelHandle;
    /** Set once reading the orders has failed; from then on the tap notes nothing and the block is not sent for the rest of the process. */
    volatile boolean failed;

    AiOrders(Engine engine) {
        this.engine = engine;
    }

    /** Puts a tap on the pool's queue unless one is on it already; the pool is rebuilt for each match, so this is asked every step. */
    void install(Object game) throws Exception {
        Object queue = engine.commandQueue(game);
        if (queue == null || queue instanceof Tap) return;
        engine.setCommandQueue(game, new Tap((Collection<?>) queue, this));
    }

    /** Gives up reading the orders for the rest of the process and forgets those held. */
    void fail() {
        failed = true;
        reset();
    }

    /** Reads the commands appended since the last call and keeps the AI players' orders among them. Called on the game thread. */
    void drain(Object game) throws Exception {
        List<Object> batch;
        synchronized (pending) {
            if (pending.isEmpty()) return;
            batch = new ArrayList<Object>(pending);
            pending.clear();
        }
        if (unloadHandle == null) {
            unloadHandle = engine.actionHandle("109");
            cancelHandle = engine.actionHandle("110");
        }
        Object local = engine.local(game);
        for (Object command : batch) {
            Order order = read(command, local);
            if (order != null) taken.add(order);
        }
    }

    /** Forgets everything, as a new episode begins. */
    void reset() {
        pending.clear();
        taken.clear();
        written = 0;
    }

    private Order read(Object command, Object local) throws Exception {
        Object issuer = engine.issuer(command);
        if (issuer == null || issuer == local || !engine.isAi(issuer) || engine.undoes(command)) return null;
        Order out = new Order();
        Object special = engine.specialOf(command);
        Object given = engine.orderOf(command);
        if (special != null && special.equals(unloadHandle)) {
            out.kind = UNLOAD;
        } else if (special != null && special.equals(cancelHandle)) {
            out.kind = CANCEL_UNLOAD;
        } else if (given != null) {
            out.kind = kind(engine.kindOf(given));
            if (out.kind < 0) return null;
            Object target = engine.orderTarget(given);
            if (target != null) {
                out.target = (int) engine.id(target);
            } else {
                out.x = engine.orderX(given);
                out.y = engine.orderY(given);
            }
        } else if (engine.stops(command)) {
            out.kind = STOP;
        } else {
            return null;
        }
        out.timeMs = engine.issuedAt(command);
        out.issuer = engine.slot(issuer);
        out.flags = engine.appended(command) ? FLAG_APPEND : 0;
        List<?> units = engine.unitsOf(command);
        int count = units == null ? 0 : units.size();
        out.units = new int[count];
        for (int i = 0; i < count; i++) out.units[i] = (int) engine.id(units.get(i));
        return out;
    }

    /** The wire kind of an engine order kind, or -1 for one that is not kept: building, repairing, reclaiming and the rest. */
    private static int kind(int engineKind) {
        switch (engineKind) {
            case KIND_MOVE: return MOVE;
            case KIND_ATTACK_MOVE: return ATTACK_MOVE;
            case KIND_ATTACK: return ATTACK;
            case KIND_PATROL:
            case KIND_GUARD:
            case KIND_GUARD_AT:
            case KIND_FOLLOW: return OTHER_MOVEMENT;
            case KIND_LOAD_INTO: return LOAD_INTO;
            case KIND_LOAD_UP: return LOAD_UP;
            default: return -1;
        }
    }

    /** Bytes the block needs for the orders taken. */
    int size() {
        int size = 2;
        for (Order order : taken) size += ORDER_SIZE + 4 * order.units.length;
        return size;
    }

    /** Writes the block; at most 65535 orders go in one block and the rest wait for the next. */
    void write(ByteBuffer out) {
        int count = Math.min(taken.size(), 0xFFFF);
        written = count;
        out.putShort((short) count);
        for (int i = 0; i < count; i++) {
            Order order = taken.get(i);
            int units = Math.min(order.units.length, 0xFFFF);
            out.putInt(order.timeMs);
            out.put((byte) order.issuer);
            out.put((byte) order.kind);
            out.put((byte) order.flags);
            out.put((byte) 0);
            out.putFloat(order.x);
            out.putFloat(order.y);
            out.putInt(order.target);
            out.putShort((short) units);
            for (int j = 0; j < units; j++) out.putInt(order.units[j]);
        }
    }

    /** Forgets the orders the last block written carried, once the frame carrying it has gone. */
    void sent() {
        taken.subList(0, Math.min(written, taken.size())).clear();
        written = 0;
    }
}
