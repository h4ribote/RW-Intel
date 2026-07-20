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
 *   match=<mapName>   run skirmish episodes, picking the first built-in map whose file name contains this text
 *                     A substring is used rather than a path because map names contain spaces, which would split the agent argument.
 *   ai=<count>        opponents to add, default 1
 *   difficulty=<int>  AI difficulty from -2 (very easy) to 3 (impossible), default 1
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
    private static volatile String matchMap = "";
    /** The map path resolved from the requested substring. The request itself is kept so later episodes can resolve it again. */
    private static volatile String resolvedMap = "";
    private static volatile int opponents = 1;
    private static volatile int difficulty = 1;
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
            else if (key.equals("match")) matchMap = value;
            else if (key.equals("ai")) opponents = Integer.parseInt(value);
            else if (key.equals("difficulty")) difficulty = Integer.parseInt(value);
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
            reportWatchedUnit();
            if (!matchMap.isEmpty()) driveMatch(engine);

            lastNanos = nanos;
            lastFrame = frame;
            lastGameMs = ms;
        }
    }

    // ---- match control -------------------------------------------------------------------

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

        // The seed is written last because returning to the battleroom redraws it.
        setField(config, "q", Integer.valueOf(seed));
        Object accepted = call(net, "ae");

        episode++;
        matchState = RUNNING;
        lastCheckpointMinute = -1;
        log("match: episode " + episode + " start accepted=" + accepted + " seed=" + seed
                + " opponents=" + opponents + " difficulty=" + difficulty + " " + netFlags(net));
    }

    private static boolean matchFinished(Object engine) throws Exception {
        Object net = getField(engine, "bX");
        if (!getBoolean(net, "aW")) return true;
        if (getBoolean(engine, "dq") || getBoolean(engine, "dt")) return true;
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
        // The engine's own return-to-battleroom path cannot be used here.
        // Its countdown is driven from the network tick, and that tick is skipped once the session flag drops, which it does part way through a single player match; the timer then sits armed forever.
        // Rebuilding the server for the next episode is slower but always lands in a known state, and it also clears the player slots that would otherwise accumulate.
        log("match: rebuilding the server for the next episode " + netFlags(net));
        matchState = NEED_SERVER;
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
            if (team == -3) continue;
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
