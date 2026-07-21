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

    Observer(Engine engine, World world) {
        this.engine = engine;
        this.world = world;
    }

    byte[] build(Object game, int episode, int blocks) throws Exception {
        Object self = engine.local(game);
        if (self == null) return null;

        world.refresh(game, self);

        int size = 64 + World.REGION_SLOTS * 32 + World.SQUAD_SLOTS * 56
                + world.visible.size() * 48 + world.events.size() * 20;
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
        if ((blocks & Wire.BLOCK_EVENTS) != 0) writeEvents(out);

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
                skip(out, 51);
                continue;
            }
            out.put((byte) 1);
            out.putShort((short) squad.id);
            out.put((byte) squad.commander);
            out.put((byte) Math.min(255, squad.units.size()));
            skip(out, 3);
            out.putFloat(squad.value);
            out.putFloat(squad.formedValue);
            out.putFloat(squad.x);
            out.putFloat(squad.y);
            out.putFloat(squad.spread);
            out.put((byte) squad.task);
            out.put((byte) squad.stance);
            out.put((byte) squad.targetRegion);
            out.put((byte) squad.status);
            out.putFloat(squad.costBudget);
            // The budget is a contract in credits, but what a policy has to reason with is what it is worth against everything still under command, so both go over.
            out.putFloat(world.commandedValue <= 0f ? 0f : squad.costBudget / world.commandedValue);
            out.putInt(squad.deadlineMs);
            out.putInt(squad.issuedAtMs);
            out.putFloat(squad.losses);
        }
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
            skip(out, 5);
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

    private static void skip(ByteBuffer out, int bytes) {
        for (int i = 0; i < bytes; i++) out.put((byte) 0);
    }
}
