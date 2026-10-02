import java.nio.ByteBuffer;
import java.util.List;

/**
 * Builds the observation the control process consumes.
 *
 * Runs on the game thread. Read from anywhere else the values would come from part way through a simulation step, and no two units would be observed at the same instant.
 *
 * What each layer is told is the strength under its own command, not the player's total. A unit whose squad a human has taken over still belongs to the same player and still counts in the engine's aggregates, so using those directly would have the command chain planning with units it cannot move. Such a squad is still described, because the display the human works from is the same observation, but it is left out of every total.
 *
 * The region and squad blocks are written at a fixed length with a validity byte per slot rather than at the length that happens to be needed. What the layers above address is a slot, and an action space that changed shape with the map or with how many squads exist would have to be learned again each time it did.
 */
final class Observer {

    /** Grown as needed; the largest observation seen so far decides the size, so steady state does not allocate. */
    private ByteBuffer scratch = Wire.buffer(1 << 16);

    private final Engine engine;
    private final World world;

    /** The slot of the player the observation is taken for, or -1 for the player this process is. A replay played back is watched rather than played, and it is observed from the side of whichever player is being studied. */
    int viewpoint = -1;

    /** The number of the observation whose answer was applied at the head of the current step, -1 when none was; written in the timing block. */
    int answered = Link.NONE;

    /** The built-in AI players' orders, written in the AI orders block. */
    final AiOrders aiOrders;

    Observer(Engine engine, World world) {
        this.engine = engine;
        this.world = world;
        this.aiOrders = new AiOrders(engine);
    }

    /** The player whose side the observation is taken from. */
    Object self(Object game) throws Exception {
        return viewpoint >= 0 ? engine.playerAt(viewpoint) : engine.local(game);
    }

    byte[] build(Object game, int episode, int blocks) throws Exception {
        Object self = self(game);
        if (self == null) return null;

        world.refresh(game, self);

        int size = 64 + World.REGION_SLOTS * 32 + World.SQUAD_SLOTS * 60
                + world.visible.size() * 44 + world.events.size() * 20
                + (world.lifts.size() + world.endedLifts.size()) * 20 + 8;
        if (scratch.capacity() < size) scratch = Wire.buffer(Integer.highestOneBit(size) * 2);
        ByteBuffer out = scratch;
        out.clear();

        Object record = engine.record(game, self);
        out.putInt(engine.frame(game));
        out.putInt(engine.gameTime(game));
        out.putInt(episode);
        out.putShort((short) blocks);
        out.put((byte) engine.slot(self));
        out.put((byte) 0);
        out.putFloat((float) engine.credits(self));
        out.putFloat(engine.income(self));
        out.putShort((short) world.commandedUnits);
        out.putShort((short) engine.aggregateInt(self, "a", 0));
        out.putShort((short) engine.aggregateInt(self, "f", 0));
        out.putShort((short) engine.recordInt(record, "c"));
        out.putShort((short) engine.recordInt(record, "d"));
        out.putShort((short) engine.recordInt(record, "f"));
        out.putShort((short) engine.recordInt(record, "g"));
        out.putShort((short) 0);  // padding, so the block ends on a four byte boundary

        if ((blocks & Wire.BLOCK_REGIONS) != 0) writeRegions(out);
        if ((blocks & Wire.BLOCK_SQUADS) != 0) writeSquads(out);
        if ((blocks & Wire.BLOCK_UNITS) != 0) writeUnits(out);
        if ((blocks & Wire.BLOCK_LIFTS) != 0) writeLifts(out);
        if ((blocks & Wire.BLOCK_EVENTS) != 0) writeEvents(out);
        if ((blocks & Wire.BLOCK_MENUS) != 0) {
            List<int[]> menus = menus();
            int needed = out.position() + 2;
            for (int[] menu : menus) needed += 5 + 2 * (menu.length - 1);
            if (out.capacity() < needed) {
                ByteBuffer larger = Wire.buffer(Integer.highestOneBit(needed) * 2);
                out.flip();
                larger.put(out);
                scratch = out = larger;
            }
            writeMenus(out, menus);
        }
        if ((blocks & Wire.BLOCK_TIMING) != 0) {
            if (out.remaining() < 4) {
                ByteBuffer larger = Wire.buffer(out.capacity() * 2);
                out.flip();
                larger.put(out);
                scratch = out = larger;
            }
            out.putInt(answered);
        }
        if ((blocks & Wire.BLOCK_AI_ORDERS) != 0) {
            int needed = out.position() + aiOrders.size();
            if (out.capacity() < needed) {
                ByteBuffer larger = Wire.buffer(Integer.highestOneBit(needed) * 2);
                out.flip();
                larger.put(out);
                scratch = out = larger;
            }
            aiOrders.write(out);
        }

        byte[] body = new byte[out.position()];
        out.flip();
        out.get(body);
        return body;
    }

    private void writeRegions(ByteBuffer out) {
        List<World.Region> regions = world.regions;
        out.putShort((short) World.REGION_SLOTS);
        for (int slot = 0; slot < World.REGION_SLOTS; slot++) {
            World.Region region = slot < regions.size() ? regions.get(slot) : null;
            if (region == null) {
                out.put((byte) 0);
                skip(out, 27);
                continue;
            }
            out.put((byte) 1);
            out.put((byte) Math.min(255, region.resources));
            out.put((byte) Math.min(255, region.heldByUs));
            out.put((byte) Math.min(255, region.heldByEnemy));
            out.putFloat(region.x);
            out.putFloat(region.y);
            out.putFloat(region.ourValue);
            out.putFloat(region.enemyValue);
            out.putInt(region.enemySeenAtMs);
            out.putFloat(region.distanceFromHome);
        }
    }

    /** A squad is written into the slot its own identifier names, not into the next free one. A slot that meant a different squad from one period to the next would be worthless as a place to carry state, which is the entire reason the block is a fixed length. */
    private void writeSquads(ByteBuffer out) {
        out.putShort((short) World.SQUAD_SLOTS);
        for (int slot = 0; slot < World.SQUAD_SLOTS; slot++) {
            World.Squad squad = world.squads.get(Integer.valueOf(slot));
            if (squad == null) {
                out.put((byte) 0);
                skip(out, 59);
                continue;
            }
            out.put((byte) 1);
            out.putShort((short) squad.id);
            out.put((byte) squad.commander);
            out.put((byte) Math.min(255, squad.units.size()));
            out.put((byte) Math.min(255, squad.aboard));
            out.put((byte) squad.passage);
            out.put((byte) squad.targetRegion);
            out.putFloat(squad.value);
            out.putFloat(squad.formedValue);
            out.putFloat(squad.x);
            out.putFloat(squad.y);
            out.putFloat(squad.spread);
            out.put((byte) squad.task);
            out.put((byte) squad.stance);
            out.put((byte) squad.targetKind);
            out.put((byte) squad.status);
            out.putFloat(squad.costBudget);
            // The budget is a contract in credits, but what a policy has to reason with is what it is worth against everything still under command, so both go over.
            out.putFloat(world.commandedValue <= 0f ? 0f : squad.costBudget / world.commandedValue);
            out.putInt(squad.deadlineMs);
            out.putInt(squad.issuedAtMs);
            out.putFloat(squad.losses);
            out.putInt((int) squad.target);
            out.putShort((short) (squad.lift == null ? World.NO_SQUAD : squad.lift.id));
            skip(out, 2);
        }
    }

    /** Every lift under way, and every lift that ended since the last frame, which is reported once with how it ended. */
    private void writeLifts(ByteBuffer out) {
        int count = world.lifts.size() + world.endedLifts.size();
        out.putShort((short) Math.min(65535, count));
        for (Lift lift : world.lifts.values()) writeLift(out, lift);
        for (Lift lift : world.endedLifts) writeLift(out, lift);
    }

    private void writeLift(ByteBuffer out, Lift lift) {
        out.putShort((short) lift.id);
        out.put((byte) lift.phase);
        out.put((byte) lift.reason);
        out.put((byte) Math.min(255, lift.loaded));
        out.put((byte) Math.min(255, lift.expected));
        out.put((byte) Math.min(255, lift.load.size()));
        skip(out, 1);
        out.putFloat(lift.health);
        out.putInt(lift.etaMs);
        out.putShort((short) lift.cargoSquad);
        out.put((byte) lift.dropRegion);
        skip(out, 1);
    }

    private void writeUnits(ByteBuffer out) {
        List<World.Seen> seen = world.visible;
        out.putShort((short) Math.min(65535, seen.size()));
        for (World.Seen unit : seen) {
            out.putInt((int) unit.id);
            out.putShort((short) unit.squad);
            out.putShort((short) unit.typeIndex);
            out.putFloat(unit.x);
            out.putFloat(unit.y);
            out.putFloat(unit.health);
            out.putFloat(unit.maxHealth);
            out.putInt((int) unit.target);
            out.putShort((short) Math.min(65535, unit.sinceHitMs));
            out.put((byte) unit.built);
            out.put((byte) unit.order);
            out.put((byte) unit.stance);
            out.put((byte) (unit.hostile ? 1 : 0));
            out.put((byte) Math.min(255, unit.queued));
            out.put((byte) Math.min(255, Math.max(0, unit.level)));
            out.putShort((short) Math.min(65535, Math.max(0, unit.upgradePrice)));
            out.put((byte) Math.min(255, unit.aboard));
            skip(out, 1);
            out.putInt((int) unit.carrier);
        }
    }

    private void writeEvents(ByteBuffer out) {
        List<World.Event> events = world.events;
        out.putShort((short) Math.min(65535, events.size()));
        for (World.Event event : events) {
            out.put((byte) event.kind);
            out.put((byte) 0);
            out.putShort((short) event.squad);
            out.putInt((int) event.unit);
            out.putShort((short) event.typeIndex);
            out.putShort((short) 0);
            out.putFloat(event.value);
        }
    }

    /**
     * What each of our finished buildings and builders offers to produce or place right now, as the unit id followed by type indices.
     * Only units that offer something are listed. The list changes with a building's tier, which is why it is read live rather than sent once with the catalogue.
     */
    private List<int[]> menus() throws Exception {
        List<int[]> menus = new java.util.ArrayList<int[]>();
        for (World.Seen unit : world.visible) {
            if (unit.hostile || unit.built < 255) continue;
            Object type = engine.type(unit.handle);
            if (type == null || !(unit.building || engine.typeIsBuilder(type))) continue;
            List<Object> offered = engine.producible(unit.handle);
            if (offered.isEmpty()) continue;
            int[] menu = new int[1 + Math.min(255, offered.size())];
            menu[0] = (int) unit.id;
            for (int i = 1; i < menu.length; i++) menu[i] = world.indexOf(engine.typeName(offered.get(i - 1)));
            menus.add(menu);
        }
        return menus;
    }

    private static void writeMenus(ByteBuffer out, List<int[]> menus) {
        out.putShort((short) Math.min(65535, menus.size()));
        for (int[] menu : menus) {
            out.putInt(menu[0]);
            out.put((byte) (menu.length - 1));
            for (int i = 1; i < menu.length; i++) out.putShort((short) menu[i]);
        }
    }

    private static void skip(ByteBuffer out, int bytes) {
        for (int i = 0; i < bytes; i++) out.put((byte) 0);
    }
}
