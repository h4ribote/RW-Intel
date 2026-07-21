import java.nio.ByteBuffer;
import java.util.List;

/**
 * Builds the observation the control process consumes.
 *
 * Runs on the game thread. Read from anywhere else the values would come from part way through a simulation step, and no two units would be observed at the same instant.
 *
 * What each layer is told is the strength under its own command, not the player's total. A unit whose squad a human has taken over still belongs to the same player and still counts in the engine's aggregates, so using those directly would have the command chain planning with units it cannot move.
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

        int size = 64 + world.regions.size() * 32 + world.squads.size() * 48 + world.visible.size() * 32;
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

        byte[] body = new byte[out.position()];
        out.flip();
        out.get(body);
        return body;
    }

    private void writeRegions(ByteBuffer out) {
        List<World.Region> regions = world.regions;
        out.putShort((short) regions.size());
        for (World.Region region : regions) {
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

    private void writeSquads(ByteBuffer out) {
        java.util.Collection<World.Squad> squads = world.squads.values();
        out.putShort((short) squads.size());
        for (World.Squad squad : squads) {
            out.putShort((short) squad.id);
            out.put((byte) squad.commander);
            out.put((byte) Math.min(255, squad.units.size()));
            out.putFloat(squad.value);
            out.putFloat(squad.formedValue);
            out.putFloat(squad.x);
            out.putFloat(squad.y);
            out.put((byte) squad.task);
            out.put((byte) squad.status);
            out.putShort((short) squad.targetRegion);
            out.putFloat(squad.losses);
            out.putInt(squad.costBudget);
            out.putInt(squad.deadlineMs);
        }
    }

    private void writeUnits(ByteBuffer out) throws Exception {
        List<World.Seen> seen = world.visible;
        out.putShort((short) seen.size());
        for (World.Seen unit : seen) {
            out.putInt((int) unit.id);
            out.putShort((short) unit.squad);
            out.putShort((short) unit.typeIndex);
            out.putFloat(unit.x);
            out.putFloat(unit.y);
            out.putFloat(unit.health);
            out.putFloat(unit.maxHealth);
            out.put((byte) unit.built);
            out.put((byte) unit.orders);
            out.putShort((short) unit.targetSquad);
            out.put((byte) unit.stance);
            out.put((byte) (unit.hostile ? 1 : 0));
            out.putShort((short) Math.min(65535, unit.sinceHitMs));
        }
    }
}
