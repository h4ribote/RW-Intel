import java.io.File;
import java.util.Arrays;
import java.util.Locale;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Starts, watches and ends skirmish episodes on instruction from the control process.
 *
 * The battleroom route is used rather than the quick start: beginning a match with no network session makes the engine redraw the seed and flatten the income multiplier, so the settings asked for would not be the settings played.
 *
 * Each episode rebuilds the server. The engine's own return to the battleroom cannot be used here, because its countdown is driven from the network tick and that tick stops being taken part way through a single player match, leaving the timer armed forever. Rebuilding costs a map load but always lands in a known state, and it also clears the player slots that would otherwise accumulate.
 */
final class MatchDriver {

    /** The team the engine gives to players who watch rather than play. */
    private static final int SPECTATOR = -3;

    private static final String SKIRMISH_DIRECTORY = "assets/maps/skirmish";

    static final class Settings {
        String map = "";
        int opponents = 1;
        int difficulty = 1;
        /** Leaves this many AI players in the match and moves everyone else, including the local player, to the spectators. Zero leaves the room alone. */
        int contestants = 0;
        int credits = 0;
        int startingUnits = 1;
        float income = 1.0f;
        int fog = 2;
        int seed = 12345;
        int maxSeconds = 0;
    }

    private final Engine engine;
    private Settings settings = new Settings();
    private String resolvedMap = "";
    private int episode = 0;

    MatchDriver(Engine engine) {
        this.engine = engine;
    }

    Settings settings() {
        return settings;
    }

    String map() {
        return resolvedMap;
    }

    int episode() {
        return episode;
    }

    /** Resolves the requested map substring to a path the engine accepts. Names contain spaces, which an agent argument cannot. */
    private String resolveMap(String wanted) {
        File directory = new File(SKIRMISH_DIRECTORY);
        String[] names = directory.list();
        if (names == null) throw new IllegalStateException("no map directory at " + directory.getAbsolutePath());
        Arrays.sort(names);
        String needle = wanted.toLowerCase(Locale.ENGLISH);
        for (String name : names) {
            if (!name.endsWith(".tmx")) continue;
            if (name.toLowerCase(Locale.ENGLISH).contains(needle)) return "maps/skirmish/" + name;
        }
        throw new IllegalStateException("no built-in map matching '" + wanted + "'");
    }

    /** Brings up a single player server on the requested map. Runs on the game thread: loading a map needs the OpenGL context. */
    void start(Object game, Settings requested) throws Exception {
        settings = requested;
        resolvedMap = resolveMap(requested.map);

        Object net = engine.net(game);
        engine.invoke(net, "b", String.class, "rw-intel setup");
        engine.resetPlayers();
        engine.invoke(engine.getField(game, "bS"), "g");
        engine.invoke(game, "L");

        synchronized (game) {
            engine.setField(game, "dm", null);
            engine.setField(game, "dl", resolvedMap);
        }

        Object normal = engine.staticField(engine.loadModeClass, "b");
        engine.loadModeClass.getClass();
        java.lang.reflect.Method load = game.getClass().getMethod("a", boolean.class, engine.loadModeClass);
        load.setAccessible(true);
        load.invoke(game, Boolean.TRUE, normal);

        engine.setField(net, "y", "You");
        engine.setField(net, "o", Boolean.TRUE);
        if (!Boolean.TRUE.equals(engine.invoke(net, "S"))) {
            throw new IllegalStateException("single player server did not start");
        }

        Object config = engine.getField(net, "ay");
        engine.setField(config, "a", engine.staticField(engine.mapKindClass, "a"));
        engine.setField(net, "az", resolvedMap);
        engine.setField(config, "b", resolvedMap.substring(resolvedMap.lastIndexOf('/') + 1));
        engine.setField(config, "c", Integer.valueOf(settings.credits));
        engine.setField(config, "d", Integer.valueOf(settings.fog));
        engine.setField(config, "e", Boolean.FALSE);
        engine.setField(config, "f", Integer.valueOf(settings.difficulty));
        engine.setField(config, "g", Integer.valueOf(settings.startingUnits));
        engine.setField(config, "h", Float.valueOf(settings.income));
        engine.setField(config, "i", Boolean.FALSE);
        engine.setField(config, "l", Boolean.FALSE);

        for (int i = 0; i < settings.opponents; i++) engine.invoke(net, "ap");
        engine.invoke(net, "f");
        engine.invoke(net, "P");
        engine.invoke(net, "L");

        // Taken out after the room has finished populating itself, because it fills every free slot with an AI whatever was asked for.
        if (settings.contestants > 0) chooseContestants();

        // The seed is written last, because returning to the battleroom redraws it.
        engine.setField(config, "q", Integer.valueOf(settings.seed));
        engine.invoke(net, "ae");
        episode++;
    }

    private void chooseContestants() throws Exception {
        int slots = engine.slotCount();
        int kept = 0;
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player == null) continue;
            if (kept < settings.contestants && engine.isAi(player)) {
                engine.setTeam(player, kept++);
            } else {
                engine.setTeam(player, SPECTATOR);
            }
        }
        if (kept < settings.contestants) {
            throw new IllegalStateException("only " + kept + " AI players available");
        }
    }

    /** True once the match cannot usefully continue: one side left, the engine has called it, or the time limit is up. */
    boolean finished(Object game) throws Exception {
        Object net = engine.net(game);
        if (!engine.getBoolean(net, "aW")) return true;
        // The engine's victory and defeat flags speak for the local player, which means nothing once that player is watching.
        if (settings.contestants == 0 && (engine.victory(game) || engine.defeat(game))) return true;
        if (settings.maxSeconds > 0 && engine.gameTime(game) / 1000 >= settings.maxSeconds) return true;
        return aliveTeams().size() <= 1;
    }

    Set<Integer> aliveTeams() throws Exception {
        int slots = engine.slotCount();
        Set<Integer> teams = new TreeSet<Integer>();
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player == null) continue;
            int team = engine.team(player);
            if (team == SPECTATOR) continue;
            if (engine.defeated(player) || engine.wiped(player) || engine.surrendered(player)) continue;
            teams.add(Integer.valueOf(team));
        }
        return teams;
    }

    /** Units and their worth per playing team, which is what a position with no winner has to be scored from. */
    String standing(Object game) throws Exception {
        TreeMap<Integer, int[]> byTeam = new TreeMap<Integer, int[]>();
        Object[] units = engine.unitArray();
        int count = engine.unitCount();
        for (int i = 0; i < count && i < units.length; i++) {
            Object unit = units[i];
            if (unit == null || engine.dead(unit)) continue;
            Object owner = engine.owner(unit);
            if (owner == null) continue;
            int team = engine.team(owner);
            // Negative teams are the spectators and the neutral owner that holds resource crystals and scenery, neither of which is a side in the match.
            if (team < 0) continue;
            if (engine.built(unit) < 1f) continue;
            int[] tally = byTeam.get(Integer.valueOf(team));
            if (tally == null) byTeam.put(Integer.valueOf(team), tally = new int[2]);
            tally[0]++;
            tally[1] += engine.price(unit);
        }
        StringBuilder out = new StringBuilder("[");
        for (java.util.Map.Entry<Integer, int[]> entry : byTeam.entrySet()) {
            if (out.length() > 1) out.append(',');
            out.append("{\"team\":").append(entry.getKey())
                    .append(",\"units\":").append(entry.getValue()[0])
                    .append(",\"value\":").append(entry.getValue()[1]).append('}');
        }
        return out.append(']').toString();
    }

    String players() throws Exception {
        int slots = engine.slotCount();
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player == null) continue;
            if (out.length() > 1) out.append(',');
            out.append("{\"slot\":").append(engine.slot(player))
                    .append(",\"team\":").append(engine.team(player))
                    .append(",\"ai\":").append(engine.isAi(player))
                    .append(",\"level\":").append(engine.aiLevel(player)).append('}');
        }
        return out.append(']').toString();
    }
}
