import java.lang.instrument.Instrumentation;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

/**
 * Measures how fast the Rusted Warfare simulation can be driven, applies the levers that control it, and can dump the live field values of game objects so the obfuscated data model can be confirmed against the disassembly.
 *
 * The engine advances game time by deltaSpeed * 16.667 ms per frame, where deltaSpeed is derived from real elapsed time and then multiplied by the field H.
 * So H is the wall-clock speed multiple, while the frame rate decides how coarse each simulation step is.
 *
 * Agent options, comma separated:
 *   speed=<float>     value to force into the engine speed multiplier, omit to leave it alone
 *   uncap=<bool>      ask the Slick container to drop its frame rate cap
 *   interval=<millis> reporting period
 *   dump=<count>      once the world is populated, dump this many game objects and all players
 *   act=move          order one unit to move, then report whether it obeyed
 *   catalog=<bool>    dump every registered unit type with the numbers a commander needs, then carry on
 *   obs=<bool>        build a full observation snapshot on the game thread and report what it costs
 *   spawn=<typeName>  create one unit of that type through the host spawn command, and report whether it appeared
 *   match=<mapName>   run skirmish episodes, picking the first built-in map whose file name contains this text
 *                     A substring is used rather than a path because map names contain spaces, which would split the agent argument.
 *   ai=<count>        opponents to add, default 1
 *   difficulty=<int>  AI difficulty from -2 (very easy) to 3 (impossible), default 1
 *   contestants=<n>   leave exactly this many AI players in the match and move everyone else, the local
 *                     player included, to the spectators
 *                     This is what makes an episode's outcome informative: with the local player sitting in the match and doing nothing, every episode ends the same way and the spread of results cannot be seen at all.
 *                     The local player cannot simply be flagged as an AI, because the built-in AI is a separate player class that only the host's add-AI path creates.
 *                     The room fills every slot regardless of how many opponents were asked for, so the contestants have to be chosen by taking the rest out rather than by adding the right number.
 *                     The map needs a starting position for each contestant's slot, so two contestants need a map for four.
 *   levels=<a,b,...>  difficulty for each contestant in turn, overriding the room's single setting
 *                     The room applies one difficulty to every AI it adds, so an uneven matchup can only be had by
 *                     setting each contestant afterwards. An uneven matchup is what gives a known effect size to
 *                     size a comparison against.
 *   seed=<int>        random seed for every episode, default 12345
 *   episodes=<count>  how many episodes to run, default 2
 *   maxSeconds=<int>  end an episode after this much game time even if undecided, 0 to disable
 */
public class RwProbeAgent {

    /** A dump is only useful once the world has been populated. */
    private static final int DUMP_OBJECT_THRESHOLD = 50;

    private static volatile float targetSpeed = -1f;
    private static volatile boolean uncap = false;
    private static volatile long intervalMs = 5000L;
    private static volatile int dumpCount = 0;
    private static volatile String action = "";
    private static volatile boolean dumpCatalog = false;
    private static volatile boolean measureObservation = false;
    private static volatile String spawnType = "";
    private static volatile String matchMap = "";
    /** The map path resolved from the requested substring. The request itself is kept so later episodes can resolve it again. */
    private static volatile String resolvedMap = "";
    private static volatile int opponents = 1;
    private static volatile int difficulty = 1;
    private static volatile int contestants = 0;
    private static volatile int[] levels = new int[0];
    private static volatile int seed = 12345;
    private static volatile int episodeLimit = 2;
    private static volatile int maxSeconds = 0;

    /** Unit picked by the move test, watched afterwards to see whether it obeyed. */
    private static volatile Object watchedUnit;
    private static volatile float watchTargetX;
    private static volatile float watchTargetY;

    public static void premain(String args, Instrumentation inst) {
        parseOptions(args);
        log("started: speed=" + targetSpeed + " uncap=" + uncap + " interval=" + intervalMs
                + "ms dump=" + dumpCount);
        Thread thread = new Thread(new Runnable() {
            public void run() {
                try {
                    monitor();
                } catch (Throwable e) {
                    log("monitor died: " + e);
                    e.printStackTrace();
                }
            }
        }, "rw-probe");
        thread.setDaemon(true);
        thread.start();
    }

    private static void parseOptions(String args) {
        if (args == null) return;
        for (String pair : args.split(",")) {
            int eq = pair.indexOf('=');
            if (eq < 0) continue;
            String key = pair.substring(0, eq).trim();
            String value = pair.substring(eq + 1).trim();
            if (key.equals("speed")) targetSpeed = Float.parseFloat(value);
            else if (key.equals("uncap")) uncap = Boolean.parseBoolean(value);
            else if (key.equals("interval")) intervalMs = Long.parseLong(value);
            else if (key.equals("dump")) dumpCount = Integer.parseInt(value);
            else if (key.equals("act")) action = value;
            else if (key.equals("catalog")) dumpCatalog = Boolean.parseBoolean(value);
            else if (key.equals("obs")) measureObservation = Boolean.parseBoolean(value);
            else if (key.equals("spawn")) spawnType = value;
            else if (key.equals("match")) matchMap = value;
            else if (key.equals("ai")) opponents = Integer.parseInt(value);
            else if (key.equals("difficulty")) difficulty = Integer.parseInt(value);
            else if (key.equals("contestants")) contestants = Integer.parseInt(value);
            // Separated by ';' because ',' already separates the agent's own options.
            else if (key.equals("levels")) {
                String[] parts = value.split(";");
                int[] parsed = new int[parts.length];
                for (int i = 0; i < parts.length; i++) parsed[i] = Integer.parseInt(parts[i].trim());
                levels = parsed;
            }
            else if (key.equals("seed")) seed = Integer.parseInt(value);
            else if (key.equals("episodes")) episodeLimit = Integer.parseInt(value);
            else if (key.equals("maxSeconds")) maxSeconds = Integer.parseInt(value);
        }
    }

    private static void monitor() throws Exception {
        Class<?> engineClass = awaitClass("com.corrodinggames.rts.gameFramework.l");
        Method getEngine = engineClass.getMethod("B");

        Object engine = null;
        while (engine == null) {
            engine = getEngine.invoke(null);
            if (engine == null) Thread.sleep(200);
        }
        log("engine ready: " + engine.getClass().getName());

        Field frameCounter = field(engineClass, "bx");
        Field gameTimeMs = field(engineClass, "by");
        Field speedMultiplier = field(engine.getClass(), "H");
        log("H initial value = " + speedMultiplier.getFloat(engine));

        Method allObjects = allObjectsMethod();
        Method setTargetFrameRate = null;
        Object container = null;
        boolean dumped = false;
        boolean acted = false;

        long lastNanos = System.nanoTime();
        int lastFrame = frameCounter.getInt(engine);
        int lastGameMs = gameTimeMs.getInt(engine);

        while (true) {
            Thread.sleep(intervalMs);

            // The engine singleton is not recreated between matches, but re-resolving is cheap and keeps the counters honest if that ever changes.
            Object current = getEngine.invoke(null);
            if (current == null) continue;
            if (current != engine) {
                engine = current;
                speedMultiplier = field(engine.getClass(), "H");
                lastFrame = frameCounter.getInt(engine);
                lastGameMs = gameTimeMs.getInt(engine);
                lastNanos = System.nanoTime();
                log("engine replaced, counters reset");
                continue;
            }

            if (targetSpeed > 0f && speedMultiplier.getFloat(engine) != targetSpeed) {
                speedMultiplier.setFloat(engine, targetSpeed);
            }

            if (uncap) {
                if (container == null) {
                    container = findContainer();
                    if (container != null) {
                        setTargetFrameRate = container.getClass().getMethod("setTargetFrameRate", int.class);
                        log("container found: " + container.getClass().getName());
                    }
                }
                if (setTargetFrameRate != null) setTargetFrameRate.invoke(container, -1);
            }

            long nanos = System.nanoTime();
            int frame = frameCounter.getInt(engine);
            int ms = gameTimeMs.getInt(engine);

            double wallSeconds = (nanos - lastNanos) / 1e9;
            int frames = frame - lastFrame;
            int advancedMs = ms - lastGameMs;

            // Starting a new episode rewinds the frame counter and the clock, which would otherwise be reported as a large negative rate.
            if (frames < 0 || advancedMs < 0) {
                log("counters rewound, re-baselining");
                lastNanos = nanos;
                lastFrame = frame;
                lastGameMs = ms;
                continue;
            }

            double fps = frames / wallSeconds;
            double speedRatio = (advancedMs / 1000.0) / wallSeconds;
            double stepMs = frames > 0 ? advancedMs / (double) frames : Double.NaN;
            int objects = countObjects(allObjects);

            log(String.format("fps=%.1f speed=%.2fx step=%.1fms objects=%d gameTime=%.1fs",
                    fps, speedRatio, stepMs, objects, ms / 1000.0));

            if (dumpCount > 0 && !dumped && objects >= DUMP_OBJECT_THRESHOLD) {
                dumped = true;
                dumpWorld(allObjects, engine);
            }

            if (action.equals("move") && !acted && objects >= DUMP_OBJECT_THRESHOLD) {
                acted = true;
                postToGameThread(engine, new Runnable() {
                    public void run() {
                        issueMoveTest();
                    }
                });
            }
            if (dumpCatalog && objects >= DUMP_OBJECT_THRESHOLD) {
                dumpCatalog = false;
                postToGameThread(engine, new Runnable() {
                    public void run() {
                        printUnitCatalog();
                    }
                });
            }

            if (measureObservation && objects >= DUMP_OBJECT_THRESHOLD) {
                postToGameThread(engine, new Runnable() {
                    public void run() {
                        measureObservation();
                    }
                });
            }

            // A spawn is a host system command, so it is only meaningful once a match is actually running.
            if (!spawnType.isEmpty() && (matchMap.isEmpty() ? objects >= DUMP_OBJECT_THRESHOLD : matchState == RUNNING)) {
                final String type = spawnType;
                spawnType = "";
                postToGameThread(engine, new Runnable() {
                    public void run() {
                        spawnTest(type);
                    }
                });
            }

            reportWatchedUnit();
            reportSpawnTest();
            if (!matchMap.isEmpty()) driveMatch(engine);

            lastNanos = nanos;
            lastFrame = frame;
            lastGameMs = ms;
        }
    }

    // ---- match control -------------------------------------------------------------------

    /** Team number the engine uses for players who watch rather than play. */
    private static final int SPECTATOR_TEAM = -3;

    private static final int NEED_SERVER = 0;
    private static final int NEED_EPISODE = 1;
    private static final int RUNNING = 2;

    private static int matchState = NEED_SERVER;
    private static int episode = 0;
    private static volatile boolean matchBusy = false;

    /**
     * Runs skirmish episodes: brings up a single player server once, then starts, watches and resets matches.
     * The battleroom route is used rather than the quick start, because beginning a match with no network session makes the engine redraw the random seed and flatten the income multiplier, which would make runs unreproducible.
     * Everything that changes engine state runs on the game thread, because loading a map builds textures and so needs the OpenGL context that only that thread holds.
     */
    private static void driveMatch(Object engine) {
        if (matchBusy) return;
        try {
            if (matchState == NEED_SERVER) {
                runMatchStep(engine, "startServer");
            } else if (matchState == NEED_EPISODE) {
                if (episode < episodeLimit) runMatchStep(engine, "startEpisode");
            } else {
                logCheckpoint(engine);
                reportMatch(engine);
                if (matchFinished(engine)) runMatchStep(engine, "endEpisode");
            }
        } catch (Throwable e) {
            log("match failed: " + e);
            e.printStackTrace();
            matchMap = "";
        }
    }

    private static void runMatchStep(final Object engine, final String step) {
        matchBusy = true;
        postToGameThread(engine, new Runnable() {
            public void run() {
                try {
                    if (step.equals("startServer")) startServer(engine);
                    else if (step.equals("startEpisode")) startEpisode(engine);
                    else endEpisode(engine);
                } catch (Throwable e) {
                    log("match step " + step + " failed: " + e);
                    e.printStackTrace();
                    matchMap = "";
                } finally {
                    matchBusy = false;
                }
            }
        });
    }

    /** Built-in maps live under this directory, reachable because each instance directory links to the master copy's assets. */
    private static final String SKIRMISH_DIRECTORY = "assets/maps/skirmish";

    /** Resolves the map substring given on the command line to a path the engine accepts. */
    private static String resolveMap() {
        java.io.File directory = new java.io.File(SKIRMISH_DIRECTORY);
        String[] names = directory.list();
        if (names == null) throw new IllegalStateException("no map directory at " + directory.getAbsolutePath());
        java.util.Arrays.sort(names);
        String wanted = matchMap.toLowerCase(java.util.Locale.ENGLISH);
        for (String name : names) {
            if (!name.endsWith(".tmx")) continue;
            if (name.toLowerCase(java.util.Locale.ENGLISH).contains(wanted)) return "maps/skirmish/" + name;
        }
        throw new IllegalStateException("no built-in map matching '" + matchMap + "'");
    }

    private static void startServer(Object engine) throws Exception {
        resolvedMap = resolveMap();
        log("match: bringing up a single player server for " + resolvedMap + " (slots in use: " + populatedSlots() + ")");
        Object net = getField(engine, "bX");
        Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");

        callWith(net, "b", String.class, "rw-intel setup");
        playerClass.getMethod("F").invoke(null);
        call(getField(engine, "bS"), "g");
        call(engine, "L");

        synchronized (engine) {
            setField(engine, "dm", null);
            setField(engine, "dl", resolvedMap);
        }

        Class<?> loadMode = Class.forName("com.corrodinggames.rts.gameFramework.s");
        Object normal = loadMode.getField("b").get(null);
        findMethod(engine.getClass(), "a", boolean.class, loadMode).invoke(engine, Boolean.TRUE, normal);

        setField(net, "y", "You");
        setField(net, "o", Boolean.TRUE);
        Object started = call(net, "S");
        log("match: single player server started = " + started + " " + netFlags(net));
        if (!Boolean.TRUE.equals(started)) throw new IllegalStateException("server did not start");
        matchState = NEED_EPISODE;
    }

    private static String netFlags(Object net) throws Exception {
        return "networked=" + getBoolean(net, "B") + " host=" + getBoolean(net, "C")
                + " singlePlayer=" + getBoolean(net, "F") + " started=" + getBoolean(net, "aW");
    }

    private static int populatedSlots() throws Exception {
        Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");
        int slots = playerClass.getField("c").getInt(null);
        Method slot = playerClass.getMethod("k", int.class);
        int used = 0;
        for (int i = 0; i < slots; i++) if (slot.invoke(null, Integer.valueOf(i)) != null) used++;
        return used;
    }

    private static void startEpisode(Object engine) throws Exception {
        Object net = getField(engine, "bX");
        Object config = getField(net, "ay");

        Class<?> mapKind = Class.forName("com.corrodinggames.rts.gameFramework.j.ai");
        setField(config, "a", mapKind.getField("a").get(null));
        setField(net, "az", resolvedMap);
        setField(config, "b", resolvedMap.substring(resolvedMap.lastIndexOf('/') + 1));
        setField(config, "c", Integer.valueOf(0));              // starting credits index 0 is 4000
        setField(config, "d", Integer.valueOf(2));              // line of sight fog
        setField(config, "e", Boolean.FALSE);                   // do not reveal the map
        setField(config, "f", Integer.valueOf(difficulty));
        setField(config, "g", Integer.valueOf(1));              // one builder
        setField(config, "h", Float.valueOf(1.0f));             // income multiplier
        setField(config, "i", Boolean.FALSE);                   // nukes allowed
        setField(config, "l", Boolean.FALSE);                   // no shared control

        for (int i = 0; i < opponents; i++) call(net, "ap");

        call(net, "f");
        call(net, "P");
        call(net, "L");

        // Taken out after the room has finished populating itself, because it fills every free slot with an AI whatever was asked for.
        if (contestants > 0) chooseContestants();

        // The seed is written last because returning to the battleroom redraws it.
        setField(config, "q", Integer.valueOf(seed));
        Object accepted = call(net, "ae");

        episode++;
        matchState = RUNNING;
        lastCheckpointMinute = -1;
        log("match: episode " + episode + " start accepted=" + accepted + " seed=" + seed
                + " opponents=" + opponents + " difficulty=" + difficulty + " " + netFlags(net));
    }

    /**
     * Leaves the first few AI players in the match and moves everyone else, including the local player, to the spectators.
     * Each contestant is put on a team of its own so that the survivors are exactly the players still in the fight.
     */
    private static void chooseContestants() throws Exception {
        Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");
        int slots = playerClass.getField("c").getInt(null);
        Method slot = playerClass.getMethod("k", int.class);
        Field teamField = playerClass.getField("r");
        Field aiField = playerClass.getField("w");

        int kept = 0;
        StringBuilder chosen = new StringBuilder();
        for (int i = 0; i < slots; i++) {
            Object player = slot.invoke(null, Integer.valueOf(i));
            if (player == null) continue;
            if (kept < contestants && aiField.getBoolean(player)) {
                teamField.setInt(player, kept);
                int level = kept < levels.length ? levels[kept] : difficulty;
                playerClass.getField("x").setInt(player, level);
                if (chosen.length() > 0) chosen.append(',');
                chosen.append(i).append("->team").append(kept).append("@").append(level);
                kept++;
            } else {
                teamField.setInt(player, SPECTATOR_TEAM);
            }
        }
        log("match: contestants " + chosen + ", everyone else is watching");
        if (kept < contestants) throw new IllegalStateException("only " + kept + " AI players available");
    }

    private static boolean matchFinished(Object engine) throws Exception {
        Object net = getField(engine, "bX");
        if (!getBoolean(net, "aW")) return true;
        // The engine's own victory and defeat flags are written from the local player's point of view, which is meaningless once that player is watching rather than playing.
        if (contestants == 0 && (getBoolean(engine, "dq") || getBoolean(engine, "dt"))) return true;
        if (maxSeconds > 0 && gameSeconds(engine) >= maxSeconds) return true;
        return aliveTeams().size() <= 1;
    }

    private static int gameSeconds(Object engine) throws Exception {
        return ((Integer) getField(engine, "by")).intValue() / 1000;
    }

    /**
     * Logs a comparable snapshot every minute of game time.
     * Two runs of the same seed can be lined up on these to see whether the simulation actually reproduces, which matters because the frame delta comes from real elapsed time rather than a fixed step.
     */
    private static void logCheckpoint(Object engine) throws Exception {
        int minute = gameSeconds(engine) / 60;
        if (minute <= lastCheckpointMinute) return;
        lastCheckpointMinute = minute;

        Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");
        int slots = playerClass.getField("c").getInt(null);
        Method slot = playerClass.getMethod("k", int.class);
        StringBuilder credits = new StringBuilder();
        for (int i = 0; i < slots; i++) {
            Object player = slot.invoke(null, Integer.valueOf(i));
            if (player == null) continue;
            if (credits.length() > 0) credits.append(',');
            credits.append((long) playerClass.getField("o").getDouble(player));
        }
        log("checkpoint: episode=" + episode + " minute=" + minute + " frame=" + getField(engine, "bx")
                + " objects=" + countObjects(allObjectsMethod()) + " credits=" + credits);
    }

    private static int lastCheckpointMinute = -1;

    private static void endEpisode(Object engine) throws Exception {
        Object net = getField(engine, "bX");
        log("match: episode " + episode + " finished at gameTime=" + getField(engine, "by")
                + "ms frame=" + getField(engine, "bx") + " victory=" + getBoolean(engine, "dq")
                + " defeat=" + getBoolean(engine, "dt"));
        // One line per episode with everything an outcome distribution is built from, so a run of episodes can be summarised without reading the narrative around it.
        java.util.Set<Integer> alive = aliveTeams();
        int seconds = gameSeconds(engine);
        log("result: episode=" + episode + " seed=" + seed + " seconds=" + seconds
                + " frames=" + getField(engine, "bx")
                + " winner=" + (alive.size() == 1 ? alive.iterator().next() : -1)
                + " aliveTeams=" + alive.size()
                + " timeout=" + (maxSeconds > 0 && seconds >= maxSeconds)
                + " units=" + countUnits()
                + " " + contestantStanding());
        // The engine's own return-to-battleroom path cannot be used here.
        // Its countdown is driven from the network tick, and that tick is skipped once the session flag drops, which it does part way through a single player match; the timer then sits armed forever.
        // Rebuilding the server for the next episode is slower but always lands in a known state, and it also clears the player slots that would otherwise accumulate.
        log("match: rebuilding the server for the next episode " + netFlags(net));
        matchState = NEED_SERVER;
    }

    /**
     * Reports each playing team's standing at the end of an episode: how many units it holds and what they are worth.
     * A win or a loss is too coarse to compare two sides that both survive, and an episode that is cut off at a time limit has no winner at all. Value held is what a score of such a position has to be built from, and it can only be had by walking the units, because the per player aggregates count units rather than what they cost.
     */
    private static String contestantStanding() throws Exception {
        Class<?> unitClass = Class.forName("com.corrodinggames.rts.game.units.am");
        Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");
        Field allUnits = unitClass.getDeclaredField("bE");
        allUnits.setAccessible(true);
        Object collection = allUnits.get(null);
        Object[] units = (Object[]) collection.getClass().getMethod("a").invoke(collection);
        int count = ((Integer) collection.getClass().getMethod("size").invoke(collection)).intValue();

        Field ownerField = unitClass.getField("bX");
        Field deadField = unitClass.getField("bV");
        Field builtField = unitClass.getField("cm");
        Method price = findMethod(unitClass, "cL");
        Field teamField = playerClass.getField("r");

        java.util.Map<Integer, int[]> byTeam = new java.util.TreeMap<Integer, int[]>();
        for (int i = 0; i < count && i < units.length; i++) {
            Object unit = units[i];
            if (unit == null || deadField.getBoolean(unit)) continue;
            Object owner = ownerField.get(unit);
            if (owner == null) continue;
            int team = teamField.getInt(owner);
            if (team == SPECTATOR_TEAM) continue;
            if (builtField.getFloat(unit) < 1f) continue;
            int[] tally = byTeam.get(Integer.valueOf(team));
            if (tally == null) byTeam.put(Integer.valueOf(team), tally = new int[2]);
            tally[0]++;
            tally[1] += ((Integer) price.invoke(unit)).intValue();
        }

        StringBuilder out = new StringBuilder();
        for (java.util.Map.Entry<Integer, int[]> entry : byTeam.entrySet()) {
            out.append(" team").append(entry.getKey()).append("Units=").append(entry.getValue()[0]);
            out.append(" team").append(entry.getKey()).append("Value=").append(entry.getValue()[1]);
        }
        return out.toString().trim();
    }

    /** A team counts as alive while it still holds a player that is not a spectator and has not lost or surrendered. */
    private static java.util.Set<Integer> aliveTeams() throws Exception {
        Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");
        int slots = playerClass.getField("c").getInt(null);
        Method slot = playerClass.getMethod("k", int.class);
        java.util.Set<Integer> teams = new java.util.TreeSet<Integer>();
        for (int i = 0; i < slots; i++) {
            Object player = slot.invoke(null, Integer.valueOf(i));
            if (player == null) continue;
            int team = playerClass.getField("r").getInt(player);
            if (team == SPECTATOR_TEAM) continue;
            if (getBoolean(player, "F") || getBoolean(player, "G") || getBoolean(player, "E")) continue;
            teams.add(Integer.valueOf(team));
        }
        return teams;
    }

    private static void reportMatch(Object engine) throws Exception {
        Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");
        int slots = playerClass.getField("c").getInt(null);
        Method slot = playerClass.getMethod("k", int.class);
        StringBuilder players = new StringBuilder();
        for (int i = 0; i < slots; i++) {
            Object player = slot.invoke(null, Integer.valueOf(i));
            if (player == null) continue;
            if (players.length() > 0) players.append(' ');
            players.append(String.format("[%d team=%d ai=%s credits=%.0f%s%s]", i,
                    playerClass.getField("r").getInt(player),
                    playerClass.getField("w").getBoolean(player),
                    playerClass.getField("o").getDouble(player),
                    getBoolean(player, "F") ? " defeated" : "",
                    getBoolean(player, "G") ? " wiped" : ""));
        }
        log("match: aliveTeams=" + aliveTeams() + " " + players);
    }

    // ---- reflection helpers --------------------------------------------------------------

    private static Field findField(Class<?> type, String name) throws Exception {
        for (Class<?> c = type; c != null; c = c.getSuperclass()) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                return f;
            } catch (NoSuchFieldException ignored) {
                // keep walking up
            }
        }
        throw new NoSuchFieldException(name + " on " + type.getName());
    }

    private static Method findMethod(Class<?> type, String name, Class<?>... parameters) throws Exception {
        for (Class<?> c = type; c != null; c = c.getSuperclass()) {
            try {
                Method m = c.getDeclaredMethod(name, parameters);
                m.setAccessible(true);
                return m;
            } catch (NoSuchMethodException ignored) {
                // keep walking up
            }
        }
        throw new NoSuchMethodException(name + " on " + type.getName());
    }

    private static Object getField(Object target, String name) throws Exception {
        return findField(target.getClass(), name).get(target);
    }

    private static boolean getBoolean(Object target, String name) throws Exception {
        return findField(target.getClass(), name).getBoolean(target);
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        findField(target.getClass(), name).set(target, value);
    }

    private static Object call(Object target, String name) throws Exception {
        return findMethod(target.getClass(), name).invoke(target);
    }

    private static Object callWith(Object target, String name, Class<?> type, Object argument) throws Exception {
        return findMethod(target.getClass(), name, type).invoke(target, argument);
    }

    // ---- issuing orders ------------------------------------------------------------------

    /**
     * Runs a task on the game thread.
     * The engine drains this queue in full at the top of every simulation step, before any unit is updated.
     * Orders must be issued from there: the command pool behind l.cf is a plain ArrayList with no synchronisation, so building a command from another thread races with the loop that consumes it.
     */
    private static void postToGameThread(Object engine, Runnable task) {
        try {
            Field queueField = engine.getClass().getField("k");
            queueField.setAccessible(true);
            Object queue = queueField.get(engine);
            queue.getClass().getMethod("add", Object.class).invoke(queue, task);
        } catch (Throwable e) {
            log("post to game thread failed: " + e);
        }
    }

    /**
     * Picks a mobile unit and orders it to move, to confirm that orders built by reflection are accepted by the engine.
     * A command obtained from the pool is already queued for execution, so filling in its fields is the whole of the submission.
     */
    private static void issueMoveTest() {
        try {
            Class<?> engineClass = Class.forName("com.corrodinggames.rts.gameFramework.l");
            Object engine = engineClass.getMethod("B").invoke(null);
            Class<?> armedClass = Class.forName("com.corrodinggames.rts.game.units.y");
            Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");

            Object unit = findMobileUnit(armedClass);
            if (unit == null) {
                log("act: no suitable unit found");
                return;
            }

            Field ownerField = unit.getClass().getField("bX");
            Object owner = ownerField.get(unit);
            Field xField = field(unit.getClass(), "eo");
            Field yField = field(unit.getClass(), "ep");
            float x = xField.getFloat(unit);
            float y = yField.getFloat(unit);

            // Somewhere clearly away from the unit, so movement is unambiguous.
            float targetX = x + 600f;
            float targetY = y + 600f;

            Field poolField = field(engineClass, "cf");
            Object pool = poolField.get(engine);
            Method obtain = pool.getClass().getMethod("b", playerClass);
            obtain.setAccessible(true);
            Object command = obtain.invoke(pool, owner);

            // h asks the engine to drop a duplicate trailing waypoint, exactly as the UI does.
            Field dedupe = command.getClass().getDeclaredField("h");
            dedupe.setAccessible(true);
            dedupe.setBoolean(command, true);

            Method addUnit = command.getClass().getMethod("a", armedClass);
            addUnit.setAccessible(true);
            addUnit.invoke(command, unit);

            Method moveTo = command.getClass().getMethod("a", float.class, float.class);
            moveTo.setAccessible(true);
            moveTo.invoke(command, targetX, targetY);

            watchedUnit = unit;
            watchTargetX = targetX;
            watchTargetY = targetY;
            log(String.format("act: ordered %s (owner %s) from (%.0f,%.0f) to (%.0f,%.0f)",
                    unit.getClass().getName(), describe(owner), x, y, targetX, targetY));
        } catch (Throwable e) {
            log("act failed: " + e);
            e.printStackTrace();
        }
    }

    /** Finds a completed, non-building unit that is free to take a movement order. */
    private static Object findMobileUnit(Class<?> armedClass) throws Exception {
        Class<?> unitClass = Class.forName("com.corrodinggames.rts.game.units.am");
        Field allUnits = unitClass.getDeclaredField("bE");
        allUnits.setAccessible(true);
        Object list = allUnits.get(null);
        Method backing = list.getClass().getMethod("a");
        backing.setAccessible(true);
        Method size = list.getClass().getMethod("size");
        size.setAccessible(true);
        Object[] units = (Object[]) backing.invoke(list);
        int count = ((Integer) size.invoke(list)).intValue();

        Field ownerField = unitClass.getField("bX");
        Field deadField = unitClass.getField("bV");
        Field builtField = unitClass.getField("cm");
        Field typeField = unitClass.getField("dz");

        // Unit types are anonymous non-public classes, so the method has to be resolved through the public interface they implement.
        // as.j() reports whether the type is a building.
        Class<?> typeInterface = Class.forName("com.corrodinggames.rts.game.units.as");
        Method isBuilding = typeInterface.getMethod("j");
        isBuilding.setAccessible(true);

        for (int i = 0; i < count && i < units.length; i++) {
            Object unit = units[i];
            if (unit == null || !armedClass.isInstance(unit)) continue;
            if (ownerField.get(unit) == null) continue;
            if (deadField.getBoolean(unit)) continue;
            if (builtField.getFloat(unit) < 1f) continue;
            Object type = typeField.get(unit);
            if (type == null) continue;
            if (Boolean.TRUE.equals(isBuilding.invoke(type))) continue;
            return unit;
        }
        return null;
    }

    /** Logs where the ordered unit is now and how many waypoints it is holding. */
    private static void reportWatchedUnit() {
        Object unit = watchedUnit;
        if (unit == null) return;
        try {
            float x = field(unit.getClass(), "eo").getFloat(unit);
            float y = field(unit.getClass(), "ep").getFloat(unit);
            Method queueLength = unit.getClass().getMethod("av");
            queueLength.setAccessible(true);
            Object waypoints = queueLength.invoke(unit);
            double remaining = Math.hypot(watchTargetX - x, watchTargetY - y);
            log(String.format("act: unit at (%.0f,%.0f) distanceToTarget=%.0f waypoints=%s",
                    x, y, remaining, waypoints));
        } catch (Throwable e) {
            log("act: watch failed: " + e);
            watchedUnit = null;
        }
    }

    // ---- unit catalogue ------------------------------------------------------------------

    /**
     * Prints every unit type the engine has registered.
     * Most built-in types are generated from the `.ini` files under assets, which can be read without running the game, but not all of them: the command centre, the builder and the factories have no definition file and exist only as code, so their numbers can only be had from a live process.
     * This is the authoritative list, and it also covers whatever a loaded mod added.
     */
    private static void printUnitCatalog() {
        try {
            Class<?> typeInterface = Class.forName("com.corrodinggames.rts.game.units.as");
            Class<?> builtIn = Class.forName("com.corrodinggames.rts.game.units.ar");

            java.util.List<Object> types = new java.util.ArrayList<Object>();
            java.util.Set<String> seen = new java.util.HashSet<String>();

            // The engine keeps every registered type, built-in and custom alike, in one static list.
            try {
                Field registry = builtIn.getDeclaredField("ae");
                registry.setAccessible(true);
                Object list = registry.get(null);
                if (list instanceof java.util.Collection) {
                    for (Object entry : (java.util.Collection<?>) list) {
                        if (typeInterface.isInstance(entry)) types.add(entry);
                    }
                }
                log("catalog: registry ar.ae holds " + types.size() + " types");
            } catch (Throwable e) {
                log("catalog: registry unavailable (" + e + "), falling back to the enum constants");
            }
            for (Object constant : builtIn.getEnumConstants()) types.add(constant);

            // A definition file that names an existing type in overrideAndReplace leaves both in the registry, so only the lookup says which one the game actually uses.
            Method resolve = findMethod(builtIn, "a", String.class);
            Method name = typeInterface.getMethod("v");
            Method display = typeInterface.getMethod("e");
            Method price = typeInterface.getMethod("c");
            Method tech = typeInterface.getMethod("g");
            Method building = typeInterface.getMethod("j");
            Method buildable = typeInterface.getMethod("l");
            Method buildSpeed = typeInterface.getMethod("D");
            Method movement = typeInterface.getMethod("o");

            log("---- catalog begin ----");
            // lookupName is what ar.a(String) accepts; reportedName is what the type calls itself and therefore what a live unit shows and what the production action id is built from. They differ wherever a definition file replaced a built-in.
            log("catalog: lookupName|reportedName|display|price|tech|building|builder|buildSpeed|movement|source");
            int printed = 0;
            for (Object type : types) {
                String internal;
                try {
                    internal = String.valueOf(name.invoke(type));
                } catch (Throwable e) {
                    continue;
                }
                if (!seen.add(internal)) continue;
                Object effective = type;
                try {
                    Object resolved = resolve.invoke(null, internal);
                    if (resolved != null) effective = resolved;
                } catch (Throwable ignored) {
                    // keep the registry entry
                }
                boolean overridden = effective != type;
                type = effective;
                StringBuilder line = new StringBuilder("catalog: ").append(internal);
                line.append('|').append(safeCall(name, type));
                line.append('|').append(safeCall(display, type));
                line.append('|').append(safeCall(price, type));
                line.append('|').append(safeCall(tech, type));
                line.append('|').append(safeCall(building, type));
                line.append('|').append(safeCall(buildable, type));
                line.append('|').append(safeCall(buildSpeed, type));
                line.append('|').append(safeCall(movement, type));
                line.append('|').append(overridden ? "overridden" : "original");
                log(line.toString());
                printed++;
            }
            log("---- catalog end: " + printed + " types ----");
        } catch (Throwable e) {
            log("catalog failed: " + e);
            e.printStackTrace();
        }
    }

    private static String safeCall(Method method, Object target) {
        try {
            return String.valueOf(method.invoke(target));
        } catch (Throwable e) {
            return "?";
        }
    }

    // ---- observation cost ----------------------------------------------------------------

    /**
     * Builds the observation the control layers would consume and reports what it cost.
     * There is no per-player index of units, so every snapshot is a full scan of the unit collection; whether that is affordable at the tactical layer's rate decides whether the interface can hand out whole snapshots or has to hand out deltas.
     * Running on the game thread is not just for safety: read from the monitor thread the values would come from part way through a simulation step, and no two units would be observed at the same instant.
     */
    private static void measureObservation() {
        try {
            Class<?> unitClass = Class.forName("com.corrodinggames.rts.game.units.am");
            Class<?> objectClass = Class.forName("com.corrodinggames.rts.gameFramework.w");
            Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");

            Field allUnits = unitClass.getDeclaredField("bE");
            allUnits.setAccessible(true);
            Object collection = allUnits.get(null);
            Method backing = collection.getClass().getMethod("a");
            backing.setAccessible(true);
            Method size = collection.getClass().getMethod("size");
            size.setAccessible(true);

            Field ownerField = unitClass.getField("bX");
            Field deadField = unitClass.getField("bV");
            Field builtField = unitClass.getField("cm");
            Field typeField = unitClass.getField("dz");
            Field healthField = unitClass.getField("cu");
            Field maxHealthField = unitClass.getField("cv");
            Field idField = objectClass.getField("eh");
            Field xField = objectClass.getField("eo");
            Field yField = objectClass.getField("ep");
            for (Field f : new Field[]{ownerField, deadField, builtField, typeField, healthField,
                    maxHealthField, idField, xField, yField}) {
                f.setAccessible(true);
            }

            int slots = playerClass.getField("c").getInt(null);
            Method slot = playerClass.getMethod("k", int.class);
            Field aggregateField = findField(playerClass, "T");

            // Repeat so the reported figure is a steady-state cost rather than the first pass through cold code.
            long[] samples = new long[OBSERVATION_SAMPLES];
            int observed = 0;
            long checksum = 0;
            for (int pass = 0; pass < OBSERVATION_SAMPLES; pass++) {
                long start = System.nanoTime();
                Object[] units = (Object[]) backing.invoke(collection);
                int count = ((Integer) size.invoke(collection)).intValue();
                observed = 0;
                for (int i = 0; i < count && i < units.length; i++) {
                    Object unit = units[i];
                    if (unit == null || deadField.getBoolean(unit)) continue;
                    Object owner = ownerField.get(unit);
                    if (owner == null) continue;
                    checksum += idField.getLong(unit);
                    checksum += (long) xField.getFloat(unit);
                    checksum += (long) yField.getFloat(unit);
                    checksum += (long) healthField.getFloat(unit);
                    checksum += (long) maxHealthField.getFloat(unit);
                    checksum += (long) (builtField.getFloat(unit) * 100f);
                    checksum += System.identityHashCode(typeField.get(unit));
                    checksum += System.identityHashCode(owner);
                    observed++;
                }
                for (int i = 0; i < slots; i++) {
                    Object player = slot.invoke(null, Integer.valueOf(i));
                    if (player == null) continue;
                    Object aggregate = aggregateField.get(player);
                    if (aggregate != null) checksum += System.identityHashCode(aggregate);
                }
                samples[pass] = System.nanoTime() - start;
            }
            java.util.Arrays.sort(samples);
            long median = samples[samples.length / 2];

            // Eight fields at four to eight bytes each is what a snapshot of one unit costs on the wire before any compaction.
            int bytes = observed * OBSERVATION_BYTES_PER_UNIT;
            log(String.format("obs: units=%d medianBuild=%.2fms perUnit=%.2fus snapshotBytes=%d checksum=%d",
                    observed, median / 1e6, observed > 0 ? median / 1000.0 / observed : 0.0, bytes, checksum));
        } catch (Throwable e) {
            log("obs failed: " + e);
            e.printStackTrace();
        }
    }

    private static final int OBSERVATION_SAMPLES = 9;
    private static final int OBSERVATION_BYTES_PER_UNIT = 40;

    // ---- host spawn command --------------------------------------------------------------

    /**
     * Creates one unit through the host's spawn command rather than by writing engine state.
     * The tactical layer is meant to train on constructed engagements instead of whole matches, and that only works if a situation can be built without breaking lockstep, which rules out assigning to fields directly.
     * A system command travels the same route as a player's order, so it should be safe; this confirms it does something at all.
     */
    private static void spawnTest(String typeName) {
        try {
            Class<?> engineClass = Class.forName("com.corrodinggames.rts.gameFramework.l");
            Object engine = engineClass.getMethod("B").invoke(null);
            Class<?> typeInterface = Class.forName("com.corrodinggames.rts.game.units.as");
            Class<?> builtIn = Class.forName("com.corrodinggames.rts.game.units.ar");

            Object type = findMethod(builtIn, "a", String.class).invoke(null, typeName);
            if (type == null) {
                log("spawn: no unit type named '" + typeName + "'");
                return;
            }
            // A definition file that replaces a built-in keeps its own name, so a type found under one name reports another. The unit that appears carries the reported one.
            String reported = String.valueOf(typeInterface.getMethod("v").invoke(type));
            log("spawn: resolved '" + typeName + "' to " + type.getClass().getName()
                    + " reporting name '" + reported + "'");

            Object anchor = findMobileUnit(Class.forName("com.corrodinggames.rts.game.units.y"));
            if (anchor == null) {
                log("spawn: no unit to take a position and an owner from");
                return;
            }
            Object owner = anchor.getClass().getField("bX").get(anchor);
            float x = field(anchor.getClass(), "eo").getFloat(anchor) + SPAWN_OFFSET;
            float y = field(anchor.getClass(), "ep").getFloat(anchor) + SPAWN_OFFSET;

            // A command taken with no player is not queued, so a system command has to be submitted explicitly.
            Object pool = field(engineClass, "cf").get(engine);
            Object command = findMethod(pool.getClass(), "b").invoke(pool);
            setField(command, "i", owner);
            setField(command, "r", Boolean.TRUE);
            setField(command, "u", Integer.valueOf(SYSTEM_SPAWN));
            findMethod(command.getClass(), "a", float.class, float.class, typeInterface, int.class)
                    .invoke(command, Float.valueOf(x), Float.valueOf(y), type, Integer.valueOf(1));

            Object net = getField(engine, "bX");
            findMethod(net.getClass(), "a", command.getClass()).invoke(net, command);

            spawnWatchType = reported;
            spawnWatchX = x;
            spawnWatchY = y;
            spawnWatchBefore = countUnits();
            log(String.format("spawn: submitted %s at (%.0f,%.0f) for %s, units before=%d",
                    typeName, x, y, describe(owner), spawnWatchBefore));
        } catch (Throwable e) {
            log("spawn failed: " + e);
            e.printStackTrace();
        }
    }

    /** Reports whether a unit of the requested type actually turned up near the requested position. */
    private static void reportSpawnTest() {
        if (spawnWatchType == null) return;
        try {
            Class<?> unitClass = Class.forName("com.corrodinggames.rts.game.units.am");
            Class<?> typeInterface = Class.forName("com.corrodinggames.rts.game.units.as");
            Method name = typeInterface.getMethod("v");
            Field allUnits = unitClass.getDeclaredField("bE");
            allUnits.setAccessible(true);
            Object collection = allUnits.get(null);
            Object[] units = (Object[]) collection.getClass().getMethod("a").invoke(collection);
            int count = ((Integer) collection.getClass().getMethod("size").invoke(collection)).intValue();
            Field typeField = unitClass.getField("dz");
            Field deadField = unitClass.getField("bV");

            int matches = 0;
            double nearest = Double.MAX_VALUE;
            long newestId = -1;
            String newest = "";
            for (int i = 0; i < count && i < units.length; i++) {
                Object unit = units[i];
                if (unit == null || deadField.getBoolean(unit)) continue;
                Object type = typeField.get(unit);
                if (type == null) continue;
                String typeName = String.valueOf(name.invoke(type));
                long id = field(unit.getClass(), "eh").getLong(unit);
                if (id > newestId) {
                    newestId = id;
                    newest = typeName;
                }
                if (!spawnWatchType.equals(typeName)) continue;
                matches++;
                double distance = Math.hypot(field(unit.getClass(), "eo").getFloat(unit) - spawnWatchX,
                        field(unit.getClass(), "ep").getFloat(unit) - spawnWatchY);
                if (distance < nearest) nearest = distance;
            }
            log(String.format("spawn: %s alive=%d nearestToTarget=%.0f unitsNow=%d (was %d) newestUnit=%s#%d",
                    spawnWatchType, matches, nearest == Double.MAX_VALUE ? -1.0 : nearest,
                    countUnits(), spawnWatchBefore, newest, newestId));
        } catch (Throwable e) {
            log("spawn: watch failed: " + e);
            spawnWatchType = null;
        }
    }

    private static int countUnits() throws Exception {
        Class<?> unitClass = Class.forName("com.corrodinggames.rts.game.units.am");
        Field allUnits = unitClass.getDeclaredField("bE");
        allUnits.setAccessible(true);
        Object collection = allUnits.get(null);
        return ((Integer) collection.getClass().getMethod("size").invoke(collection)).intValue();
    }

    /** Host system command that creates a unit outright. */
    private static final int SYSTEM_SPAWN = 5;
    /** Far enough from the unit the position was taken from that the new unit is not sitting on top of it. */
    private static final float SPAWN_OFFSET = 120f;

    private static volatile String spawnWatchType;
    private static volatile float spawnWatchX;
    private static volatile float spawnWatchY;
    private static volatile int spawnWatchBefore;

    // ---- dumping -------------------------------------------------------------------------

    /**
     * Prints the live field values of a sample of game objects and of every player.
     * This is ground truth for the obfuscated data model: it shows which field actually holds health, position and owner, which the disassembly alone can only suggest.
     */
    private static void dumpWorld(Method allObjects, Object engine) {
        try {
            log("---- dump begin ----");
            Object[] objects = objectArray(allObjects);
            if (objects == null) {
                log("dump: object array unavailable");
                return;
            }

            int dumped = 0;
            java.util.Set<String> seenClasses = new java.util.HashSet<String>();
            for (Object object : objects) {
                if (object == null) continue;
                // Show a spread of types rather than many copies of the same one.
                if (!seenClasses.add(object.getClass().getName())) continue;
                dumpObject("object[" + dumped + "]", object);
                if (++dumped >= dumpCount) break;
            }

            dumpPlayers();
            log("---- dump end ----");
        } catch (Throwable e) {
            log("dump failed: " + e);
        }
    }

    private static void dumpPlayers() {
        try {
            Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");
            for (Field f : playerClass.getDeclaredFields()) {
                if (!Modifier.isStatic(f.getModifiers()) || !f.getType().isArray()) continue;
                f.setAccessible(true);
                Object array = f.get(null);
                if (array == null) continue;
                int length = Array.getLength(array);
                log("players: static " + playerClass.getName() + "." + f.getName() + " length=" + length);
                for (int i = 0; i < length; i++) {
                    Object player = Array.get(array, i);
                    if (player != null) dumpObject("player[" + i + "]", player);
                }
                return;
            }
            log("players: no static array field found on " + playerClass.getName());
        } catch (Throwable e) {
            log("player dump failed: " + e);
        }
    }

    private static void dumpObject(String label, Object object) {
        log(label + " = " + object.getClass().getName());
        for (Class<?> c = object.getClass(); c != null && c != Object.class; c = c.getSuperclass()) {
            StringBuilder line = new StringBuilder();
            for (Field f : c.getDeclaredFields()) {
                if (Modifier.isStatic(f.getModifiers())) continue;
                f.setAccessible(true);
                String value;
                try {
                    value = describe(f.get(object));
                } catch (Throwable e) {
                    value = "<inaccessible>";
                }
                if (line.length() > 0) line.append("  ");
                line.append(f.getName()).append(':').append(shortType(f.getType())).append('=').append(value);
                if (line.length() > 300) {
                    log("  [" + c.getSimpleName() + "] " + line);
                    line.setLength(0);
                }
            }
            if (line.length() > 0) log("  [" + c.getSimpleName() + "] " + line);
        }
    }

    private static String describe(Object value) {
        if (value == null) return "null";
        Class<?> type = value.getClass();
        if (type.isArray()) return shortType(type.getComponentType()) + "[" + Array.getLength(value) + "]";
        if (value instanceof String) {
            String s = (String) value;
            if (s.length() > 40) s = s.substring(0, 40) + "...";
            return '"' + s + '"';
        }
        if (type.getName().startsWith("java.lang.")) return String.valueOf(value);
        if (value instanceof java.util.Collection) {
            return shortType(type) + "(" + ((java.util.Collection<?>) value).size() + ")";
        }
        return shortType(type) + "@" + Integer.toHexString(System.identityHashCode(value));
    }

    private static String shortType(Class<?> type) {
        String name = type.getName();
        int dot = name.lastIndexOf('.');
        if (dot < 0) return name;
        return name.substring(dot + 1);
    }

    // ---- engine access -------------------------------------------------------------------

    private static Object findContainer() {
        try {
            Class<?> mainClass = Class.forName("com.corrodinggames.rts.java.Main");
            Field self = mainClass.getDeclaredField("m");
            self.setAccessible(true);
            Object main = self.get(null);
            if (main == null) return null;
            Field containerField = mainClass.getDeclaredField("k");
            containerField.setAccessible(true);
            return containerField.get(main);
        } catch (Throwable e) {
            log("container lookup failed: " + e);
            return null;
        }
    }

    private static Method allObjectsMethod() {
        try {
            Class<?> objectClass = Class.forName("com.corrodinggames.rts.gameFramework.w");
            Method all = objectClass.getMethod("dK");
            all.setAccessible(true);
            return all;
        } catch (Throwable e) {
            log("object list lookup failed: " + e);
            return null;
        }
    }

    private static int countObjects(Method allObjects) {
        if (allObjects == null) return -1;
        try {
            Object list = allObjects.invoke(null);
            if (list == null) return -1;
            Method size = list.getClass().getMethod("size");
            size.setAccessible(true);
            return ((Integer) size.invoke(list)).intValue();
        } catch (Throwable e) {
            return -1;
        }
    }

    /** The engine's object collection exposes its backing array; entries may be null. */
    private static Object[] objectArray(Method allObjects) {
        if (allObjects == null) return null;
        try {
            Object list = allObjects.invoke(null);
            if (list == null) return null;
            Method backing = list.getClass().getMethod("b");
            backing.setAccessible(true);
            return (Object[]) backing.invoke(list);
        } catch (Throwable e) {
            log("object array lookup failed: " + e);
            return null;
        }
    }

    private static Class<?> awaitClass(String name) throws Exception {
        while (true) {
            try {
                return Class.forName(name);
            } catch (ClassNotFoundException e) {
                Thread.sleep(200);
            }
        }
    }

    private static Field field(Class<?> owner, String name) throws Exception {
        Field f = owner.getField(name);
        f.setAccessible(true);
        return f;
    }

    private static void log(String message) {
        System.out.println("[rw-probe] " + message);
        System.out.flush();
    }
}
