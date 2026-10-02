import java.lang.instrument.Instrumentation;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Experiment agent: starts a skirmish, halts the computer players unless told not to, then plays a script of timed steps against the live engine and logs what the units do.
 *
 * It exists so that what the engine does with units, transports and orders is settled by running the game rather than by reading the disassembly, and so that it can be settled again after anything changes. The scripts it is run with live in scenarios/ and are started by `python -m rwintel.runtime lab`.
 *
 * Options, comma separated, besides the frame layer's:
 *   map=<substring>  built-in map to play, the first whose file name contains it
 *   script=<path>    the steps, one per line: "@<seconds> <verb> <args...>", seconds counted from the script's start
 *   seed=<int>, difficulty=<int>, delay=<seconds after the match starts before the script does>, track=<ms between tracking lines>
 *   halt=<bool>      halt every computer player first, true by default
 *
 * Verbs:
 *   spawn <label> <type> <x> <y> [enemy]       create a unit through the host spawn command and bind the label to it
 *   bind <label> <type> [<x> <y>] [enemy]      bind the label to an existing unit of the type, the nearest to the point
 *   track <label>...                           log the units' state every track interval
 *   move|amove|moveQ <label> <x> <y>           move, attack-move, or move appended to the order queue
 *   loadInto <passenger> <transport>           the passenger boards the transport
 *   loadUp|loadUpQ <transport> <passenger>     the transport picks the passenger up, appended to the queue with Q
 *   actions <label>                            list the unit's actions
 *   action|actionQ <label> <index or text>     issue one of the unit's own actions
 *   build <builder> <type> <x> <y>, produce <factory> <type>, upgrade <building>, menu <label>
 *   catalog                                    every type's capabilities read off its sample unit
 *   compat                                     which types each transport type's sample would load
 *   passmap <prefix>                           the path finder's passability grids, one file per movement type
 *   census <x0> <x1>                           units by player and type, and how many stand between x0 and x1 or are aboard
 *   near <x> <y> <radius>, end
 *
 * Verbs that drive the agent's own command code (World, Commander and Lift from agent/), through the same action frame the control process sends:
 *   regions <x> <y> [<x> <y>...]               the region table, numbered in order
 *   squad <id> <label>...                      form a squad of the labelled units
 *   contract <squad> <task> region|squad|unit <target> [stance]   task is attack, defend, raid, withdraw, escort or encircle; a unit target is a label
 *   lift <id> squad <squad>|units <label>... via <transport>... pickup <x> <y> drop <region> <x> <y> [deadline <seconds>]
 *   liftcancel <id>, status
 * Once one of these has run, the world is scanned and the command code's upkeep runs every tactical period, and every change of a squad's status, a lift's phase and every event is logged.
 */
public class RwLab {

    static String mapWanted = "Lake";
    static String resolvedMap = "";
    static String scriptPath = "";
    static int seed = 12345;
    static int difficulty = -2;
    static int delayS = 3;
    static int trackMs = 1000;
    static boolean haltAi = true;

    static Engine eng;
    static volatile Object engineObj;
    static final int NEED_SERVER = 0, NEED_EPISODE = 1, RUNNING = 2, DONE = 3;
    static volatile int state = NEED_SERVER;
    static volatile boolean busy = false;
    static int startedAtMs = -1;
    static int scriptStartMs = -1;
    static final List<String[]> steps = new ArrayList<String[]>();
    static final List<Integer> stepTimes = new ArrayList<Integer>();
    static int nextStep = 0;
    static final Map<String, Long> labels = new LinkedHashMap<String, Long>();
    static final List<String> tracked = new ArrayList<String>();
    static int lastTrackMs = Integer.MIN_VALUE;
    static final Map<String, String> pendingSpawns = new HashMap<String, String>();
    static final Map<String, float[]> pendingAt = new HashMap<String, float[]>();
    static final Map<String, Long> spawnFloor = new HashMap<String, Long>();

    /** The agent's own command code, driven once a verb has asked for it. */
    static World world;
    static Commander commander;
    static boolean chain = false;
    static int lastChainMs = Integer.MIN_VALUE;
    static final int CHAIN_MS = 200;
    static final Map<Integer, String> lastSquadState = new HashMap<Integer, String>();
    static final Map<Integer, String> lastLiftState = new HashMap<Integer, String>();
    static final String[] TASKS = {"attack", "defend", "raid", "withdraw", "escort", "encircle"};
    static final String[] STATUSES = {"ACTIVE", "STALLED", "LOSING", "COMPLETE", "EXPIRED", "UNREACHABLE", "AWAITING_LIFT", "LIFTING"};
    static final String[] PHASES = {"APPROACH", "LOADING", "CARRYING", "UNLOADING", "DONE", "FAILED"};
    static final String[] REASONS = {"", "SUNK", "UNREACHABLE_PICKUP", "UNREACHABLE_DROP", "REFUSED", "EXPIRED", "CANCELLED", "CARGO_LOST"};

    public static void premain(String args, Instrumentation inst) {
        if (args != null) {
            for (String pair : args.split(",")) {
                int eq = pair.indexOf('=');
                if (eq < 0) continue;
                String key = pair.substring(0, eq).trim();
                String value = pair.substring(eq + 1).trim();
                if (Frame.option(key, value)) continue;
                if (key.equals("map")) mapWanted = value;
                else if (key.equals("script")) scriptPath = value;
                else if (key.equals("seed")) seed = Integer.parseInt(value);
                else if (key.equals("difficulty")) difficulty = Integer.parseInt(value);
                else if (key.equals("delay")) delayS = Integer.parseInt(value);
                else if (key.equals("track")) trackMs = Integer.parseInt(value);
                else if (key.equals("halt")) haltAi = Boolean.parseBoolean(value);
            }
        }
        Frame.install();
        Thread thread = new Thread(new Runnable() {
            public void run() {
                try {
                    boot();
                } catch (Throwable e) {
                    log("boot failed: " + e);
                    e.printStackTrace();
                }
            }
        }, "rw-lab");
        thread.setDaemon(true);
        thread.start();
    }

    static void boot() throws Exception {
        Class<?> engineClass = null;
        while (engineClass == null) {
            try {
                engineClass = Class.forName("com.corrodinggames.rts.gameFramework.l");
            } catch (ClassNotFoundException e) {
                Thread.sleep(200);
            }
        }
        Method getEngine = engineClass.getMethod("B");
        Object engine = null;
        while (engine == null) {
            engine = getEngine.invoke(null);
            if (engine == null) Thread.sleep(200);
        }
        engineObj = engine;
        eng = new Engine();
        loadScript();
        log("ready: map=" + mapWanted + " script=" + scriptPath + " steps=" + steps.size());
        Frame.addListener(new Frame.Listener() {
            public void onFrame(int gameTimeMs) {
                tick(gameTimeMs);
            }
        });
    }

    static void loadScript() throws Exception {
        if (scriptPath.isEmpty()) return;
        java.io.BufferedReader in = new java.io.BufferedReader(new java.io.FileReader(scriptPath));
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split("\\s+");
            if (!parts[0].startsWith("@")) continue;
            stepTimes.add(Integer.valueOf((int) (Double.parseDouble(parts[0].substring(1)) * 1000)));
            String[] rest = new String[parts.length - 1];
            System.arraycopy(parts, 1, rest, 0, rest.length);
            steps.add(rest);
        }
        in.close();
    }

    static void tick(final int now) {
        if (busy) return;
        try {
            if (state == NEED_SERVER) {
                Frame.setIdle(false);
                post("server", new Task() { public void run() throws Exception { startServer(); } });
            } else if (state == NEED_EPISODE) {
                post("episode", new Task() { public void run() throws Exception { startEpisode(); } });
            } else if (state == RUNNING) {
                if (startedAtMs < 0) startedAtMs = now;
                if (scriptStartMs < 0) {
                    if (now - startedAtMs >= delayS * 1000) {
                        post("setup", new Task() { public void run() throws Exception { setup(now); } });
                    }
                    return;
                }
                final int at = now - scriptStartMs;
                resolveSpawns();
                while (nextStep < steps.size() && stepTimes.get(nextStep).intValue() <= at) {
                    final String[] step = steps.get(nextStep++);
                    try {
                        run(step, at);
                    } catch (Throwable e) {
                        log("step failed: " + join(step) + " -> " + e);
                        e.printStackTrace();
                    }
                }
                if (chain && (long) at - (long) lastChainMs >= CHAIN_MS) {
                    lastChainMs = at;
                    try {
                        chainStep(at);
                    } catch (Throwable e) {
                        log("chain failed: " + e);
                        e.printStackTrace();
                    }
                }
                if ((long) at - (long) lastTrackMs >= trackMs) {
                    lastTrackMs = at;
                    trackAll(at);
                }
            }
        } catch (Throwable e) {
            log("tick failed: " + e);
            e.printStackTrace();
        }
    }

    interface Task {
        void run() throws Exception;
    }

    static void post(final String name, final Task task) {
        busy = true;
        try {
            Field queueField = engineObj.getClass().getField("k");
            Object queue = queueField.get(engineObj);
            queue.getClass().getMethod("add", Object.class).invoke(queue, new Runnable() {
                public void run() {
                    try {
                        task.run();
                    } catch (Throwable e) {
                        log(name + " failed: " + e);
                        e.printStackTrace();
                        state = DONE;
                    } finally {
                        busy = false;
                    }
                }
            });
        } catch (Throwable e) {
            busy = false;
            log("post failed: " + e);
        }
    }

    // ---- match start, as the probe agent does it -------------------------------------------

    static void startServer() throws Exception {
        java.io.File directory = new java.io.File("assets/maps/skirmish");
        String[] names = directory.list();
        java.util.Arrays.sort(names);
        for (String name : names) {
            if (name.endsWith(".tmx") && name.toLowerCase().contains(mapWanted.toLowerCase())) {
                resolvedMap = "maps/skirmish/" + name;
                break;
            }
        }
        Object engine = engineObj;
        Object net = getField(engine, "bX");
        Class<?> playerClass = Class.forName("com.corrodinggames.rts.game.n");
        findMethod(net.getClass(), "b", String.class).invoke(net, "rw-lab setup");
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
        log("server started=" + started + " map=" + resolvedMap);
        state = NEED_EPISODE;
    }

    static void startEpisode() throws Exception {
        Object engine = engineObj;
        Object net = getField(engine, "bX");
        Object config = getField(net, "ay");
        Class<?> mapKind = Class.forName("com.corrodinggames.rts.gameFramework.j.ai");
        setField(config, "a", mapKind.getField("a").get(null));
        setField(net, "az", resolvedMap);
        setField(config, "b", resolvedMap.substring(resolvedMap.lastIndexOf('/') + 1));
        setField(config, "c", Integer.valueOf(0));
        setField(config, "d", Integer.valueOf(2));
        setField(config, "e", Boolean.FALSE);
        setField(config, "f", Integer.valueOf(difficulty));
        setField(config, "g", Integer.valueOf(1));
        setField(config, "h", Float.valueOf(1.0f));
        setField(config, "i", Boolean.FALSE);
        setField(config, "l", Boolean.FALSE);
        call(net, "ap");
        call(net, "f");
        call(net, "P");
        call(net, "L");
        setField(config, "q", Integer.valueOf(seed));
        Object accepted = call(net, "ae");
        log("episode accepted=" + accepted + " seed=" + seed);
        state = RUNNING;
    }

    // ---- the script ------------------------------------------------------------------------

    static void setup(int now) throws Exception {
        int halted = 0;
        for (int slot = 0; slot < 10; slot++) {
            Object player = eng.playerAt(slot);
            if (player == null || !haltAi) continue;
            if (eng.haltAi(player)) halted++;
        }
        Object self = eng.local(engineObj);
        Object hq = null;
        for (Object unit : liveUnits()) {
            if (eng.owner(unit) == self && eng.typeName(eng.type(unit)).equals("commandCenter")) hq = unit;
        }
        log("setup: halted=" + halted + " selfSlot=" + slotOf(self) + " selfTeam=" + eng.team(self)
                + (hq == null ? " no hq" : String.format(" hq=(%.0f,%.0f)", eng.x(hq), eng.y(hq))));
        for (int slot = 0; slot < 10; slot++) {
            Object player = eng.playerAt(slot);
            if (player != null) log("setup: player slot=" + slot + " team=" + eng.team(player) + (player == self ? " (self)" : ""));
        }
        scriptStartMs = now;
        lastTrackMs = Integer.MIN_VALUE;
    }

    static void run(String[] step, int at) throws Exception {
        String verb = step[0];
        log(String.format("t=%.1f do %s", at / 1000.0, join(step)));
        if (verb.equals("spawn")) {
            boolean enemy = step.length > 5 && step[5].equals("enemy");
            Object owner = enemy ? enemyPlayer() : eng.local(engineObj);
            Object type = eng.typeNamed(step[2]);
            if (type == null) {
                log("spawn: no type " + step[2]);
                return;
            }
            float x = Float.parseFloat(step[3]);
            float y = Float.parseFloat(step[4]);
            spawnFloor.put(step[1], Long.valueOf(maxId()));
            eng.spawn(engineObj, owner, type, x, y);
            pendingSpawns.put(step[1], eng.typeName(type));
            pendingAt.put(step[1], new float[]{x, y});
        } else if (verb.equals("bind")) {
            Object unit = nearestOwn(step[2], step.length > 4 ? Float.parseFloat(step[3]) : Float.NaN,
                    step.length > 4 ? Float.parseFloat(step[4]) : Float.NaN, step.length > 5 && step[5].equals("enemy"));
            if (unit == null) log("bind: no " + step[2]);
            else {
                labels.put(step[1], Long.valueOf(eng.id(unit)));
                log("bind: " + step[1] + " = " + describe(unit));
            }
        } else if (verb.equals("track")) {
            for (int i = 1; i < step.length; i++) if (!tracked.contains(step[i])) tracked.add(step[i]);
        } else if (verb.equals("move") || verb.equals("amove")) {
            Object unit = unit(step[1]);
            Object command = commandFor(unit);
            eng.addUnit(command, unit);
            if (verb.equals("move")) eng.moveTo(command, Float.parseFloat(step[2]), Float.parseFloat(step[3]));
            else eng.attackMoveTo(command, Float.parseFloat(step[2]), Float.parseFloat(step[3]));
        } else if (verb.equals("loadInto")) {
            Object unit = unit(step[1]);
            Object transport = unit(step[2]);
            Object command = commandFor(unit);
            eng.addUnit(command, unit);
            eng.loadInto(command, transport);
        } else if (verb.equals("loadUp") || verb.equals("loadUpQ")) {
            Object transport = unit(step[1]);
            Object unit = unit(step[2]);
            Object command = commandFor(transport);
            if (verb.endsWith("Q")) eng.append(command);
            eng.addUnit(command, transport);
            eng.loadUp(command, unit);
        } else if (verb.equals("moveQ")) {
            Object unit = unit(step[1]);
            Object command = commandFor(unit);
            eng.append(command);
            eng.addUnit(command, unit);
            eng.moveTo(command, Float.parseFloat(step[2]), Float.parseFloat(step[3]));
        } else if (verb.equals("actionQ")) {
            Object unit = unit(step[1]);
            Object action = findAction(unit, step[2]);
            Object command = commandFor(unit);
            eng.append(command);
            eng.addUnit(command, unit);
            eng.offeredAction(command, action);
        } else if (verb.equals("actions")) {
            Object unit = unit(step[1]);
            listActions(unit);
        } else if (verb.equals("action")) {
            Object unit = unit(step[1]);
            Object action = findAction(unit, step[2]);
            if (action == null) {
                log("action: none matching " + step[2]);
                return;
            }
            Object command = commandFor(unit);
            eng.addUnit(command, unit);
            eng.offeredAction(command, action);
        } else if (verb.equals("build")) {
            Object unit = unit(step[1]);
            Object type = eng.typeNamed(step[2]);
            Object command = commandFor(unit);
            eng.addUnit(command, unit);
            eng.specialAction(command, "b_" + eng.typeName(type));
            eng.build(command, Float.parseFloat(step[3]), Float.parseFloat(step[4]), type, 1);
        } else if (verb.equals("produce")) {
            Object unit = unit(step[1]);
            Object type = eng.typeNamed(step[2]);
            Object command = commandFor(unit);
            eng.addUnit(command, unit);
            eng.specialAction(command, "u_" + eng.typeName(type));
        } else if (verb.equals("upgrade")) {
            Object unit = unit(step[1]);
            Object upgrade = eng.upgradeOffered(unit);
            if (upgrade == null) {
                log("upgrade: none offered");
                return;
            }
            Object command = commandFor(unit);
            eng.addUnit(command, unit);
            eng.offeredAction(command, upgrade);
        } else if (verb.equals("menu")) {
            Object unit = unit(step[1]);
            StringBuilder out = new StringBuilder();
            for (Object type : eng.producible(unit)) out.append(eng.typeName(type)).append(' ');
            log("menu " + step[1] + " level=" + eng.level(unit) + ": " + out);
        } else if (verb.equals("catalog")) {
            catalog();
        } else if (verb.equals("compat")) {
            compatibility();
        } else if (verb.equals("passmap")) {
            passMap(step[1]);
        } else if (verb.equals("census")) {
            // Per player: units by type, how many stand on the main island's x range, and how many are aboard something.
            float x0 = Float.parseFloat(step[1]);
            float x1 = Float.parseFloat(step[2]);
            Map<String, int[]> counts = new java.util.TreeMap<String, int[]>();
            for (Object unit : liveUnits()) {
                Object owner = eng.owner(unit);
                if (owner == null || eng.team(owner) < 0) continue;
                String key = "slot" + slotOf(owner) + " " + eng.typeName(eng.type(unit));
                int[] c = counts.get(key);
                if (c == null) counts.put(key, c = new int[3]);
                c[0]++;
                if (eng.x(unit) >= x0 && eng.x(unit) <= x1) c[1]++;
                if (getField(unit, "cN") != null) c[2]++;
            }
            for (Map.Entry<String, int[]> entry : counts.entrySet()) {
                log("census " + entry.getKey() + " total=" + entry.getValue()[0] + " onMain=" + entry.getValue()[1] + " aboard=" + entry.getValue()[2]);
            }
        } else if (verb.equals("near")) {
            float x = Float.parseFloat(step[1]);
            float y = Float.parseFloat(step[2]);
            float r = Float.parseFloat(step[3]);
            for (Object unit : liveUnits()) {
                if (Math.hypot(eng.x(unit) - x, eng.y(unit) - y) <= r) log("near: " + describe(unit));
            }
        } else if (verb.equals("regions") || verb.equals("squad") || verb.equals("contract") || verb.equals("lift")
                || verb.equals("liftcancel") || verb.equals("status")) {
            command(step);
        } else if (verb.equals("end")) {
            log("end");
            state = DONE;
            Frame.setIdle(true);
        } else {
            log("unknown verb " + verb);
        }
    }

    /** Binds a spawned label to the newest unit of its type near where it was asked for, once the spawn command has gone through. */
    static void resolveSpawns() throws Exception {
        if (pendingSpawns.isEmpty()) return;
        for (String label : new ArrayList<String>(pendingSpawns.keySet())) {
            String type = pendingSpawns.get(label);
            float[] at = pendingAt.get(label);
            long floor = spawnFloor.get(label).longValue();
            Object best = null;
            double bestDistance = 400;
            for (Object unit : liveUnits()) {
                if (eng.id(unit) <= floor || labels.containsValue(Long.valueOf(eng.id(unit)))) continue;
                if (!eng.typeName(eng.type(unit)).equals(type)) continue;
                double distance = Math.hypot(eng.x(unit) - at[0], eng.y(unit) - at[1]);
                if (distance > bestDistance) continue;
                bestDistance = distance;
                best = unit;
            }
            if (best != null) {
                labels.put(label, Long.valueOf(eng.id(best)));
                pendingSpawns.remove(label);
                log("spawned: " + label + " = " + describe(best));
            }
        }
    }

    // The agent's own command code.

    /** Carries out a verb that drives the command code, building the same action frame the control process would send. */
    static void command(String[] step) throws Exception {
        if (world == null) {
            world = new World(eng);
            commander = new Commander(eng, world);
            List<Object> types = new ArrayList<Object>();
            for (Object type : eng.allTypes()) types.add(type);
            world.indexTypes(types);
            world.passage = Passage.read(eng, engineObj);
            world.refresh(engineObj, eng.local(engineObj));
            chain = true;
            log("chain: passage " + world.passage.labels.keySet());
        }
        String verb = step[0];
        if (verb.equals("regions")) {
            List<float[]> table = new ArrayList<float[]>();
            for (int i = 1; i + 1 < step.length; i += 2) table.add(new float[]{Float.parseFloat(step[i]), Float.parseFloat(step[i + 1]), 0f, 0f});
            world.setRegions(table);
            log("chain: " + table.size() + " regions");
            return;
        }
        if (verb.equals("status")) {
            for (World.Squad squad : world.squads.values()) log("chain: " + squadState(squad));
            for (Lift lift : world.lifts.values()) log("chain: " + liftState(lift));
            return;
        }
        world.refresh(engineObj, eng.local(engineObj));
        java.nio.ByteBuffer out = java.nio.ByteBuffer.allocate(4096).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        if (verb.equals("squad")) {
            out.putShort((short) 1);
            out.putShort((short) Integer.parseInt(step[1]));
            out.put((byte) 0);
            out.put((byte) 0);
            out.putShort((short) (step.length - 2));
            out.putShort((short) 0);
            for (int i = 2; i < step.length; i++) out.putInt((int) labels.get(step[i]).longValue());
            empty(out, 5);
        } else if (verb.equals("contract")) {
            int task = java.util.Arrays.asList(TASKS).indexOf(step[2]);
            int kind = step[3].equals("squad") ? 1 : step[3].equals("unit") ? 2 : 0;
            long target = kind == 2 ? labels.get(step[4]).longValue() : Long.parseLong(step[4]);
            int stance = step.length > 5 ? Integer.parseInt(step[5]) : (task == 1 || task == 4 ? 4 : task == 3 ? 2 : 5);
            empty(out, 1);
            out.putShort((short) 1);
            out.putShort((short) Integer.parseInt(step[1]));
            out.put((byte) task);
            out.put((byte) stance);
            out.put((byte) kind);
            out.put((byte) 0);
            out.putShort((short) 0);
            out.putInt((int) target);
            out.putFloat(5000f);
            out.putInt(0);
            out.putInt(eng.gameTime(engineObj));
            empty(out, 4);
        } else if (verb.equals("lift") || verb.equals("liftcancel")) {
            empty(out, 4);
            writeLift(out, step);
            empty(out, 1);
        }
        byte[] body = new byte[out.position()];
        out.flip();
        out.get(body);
        commander.apply(engineObj, body);
    }

    /** One lift row: "lift <id> squad <squad>|units <label>... via <transport>... pickup <x> <y> drop <region> <x> <y> [deadline <seconds>]", or "liftcancel <id>". */
    static void writeLift(java.nio.ByteBuffer out, String[] step) throws Exception {
        out.putShort((short) 1);
        int id = Integer.parseInt(step[1]);
        if (step[0].equals("liftcancel")) {
            out.putShort((short) id);
            out.put((byte) 0);
            out.put((byte) 2);
            out.put((byte) 0);
            out.put((byte) 0);
            out.putShort((short) 0);
            for (int i = 0; i < 4; i++) out.putFloat(0f);
            out.putInt(0);
            return;
        }
        boolean squad = step[2].equals("squad");
        List<Long> cargo = new ArrayList<Long>();
        List<Long> transports = new ArrayList<Long>();
        float[] pickup = new float[2];
        float[] drop = new float[3];
        int deadline = 0;
        int i = 3;
        for (; i < step.length && !step[i].equals("via"); i++) cargo.add(squad ? Long.valueOf(step[i]) : labels.get(step[i]));
        for (i++; i < step.length && !step[i].equals("pickup"); i++) transports.add(labels.get(step[i]));
        pickup[0] = Float.parseFloat(step[i + 1]);
        pickup[1] = Float.parseFloat(step[i + 2]);
        i += 3;
        drop[0] = Float.parseFloat(step[i + 1]);
        drop[1] = Float.parseFloat(step[i + 2]);
        drop[2] = Float.parseFloat(step[i + 3]);
        i += 4;
        if (i + 1 < step.length && step[i].equals("deadline")) deadline = eng.gameTime(engineObj) + (int) (Float.parseFloat(step[i + 1]) * 1000);
        out.putShort((short) id);
        out.put((byte) (squad ? 0 : 1));
        out.put((byte) 0);
        out.put((byte) drop[0]);
        out.put((byte) transports.size());
        out.putShort((short) cargo.size());
        out.putFloat(pickup[0]);
        out.putFloat(pickup[1]);
        out.putFloat(drop[1]);
        out.putFloat(drop[2]);
        out.putInt(deadline);
        for (Long transport : transports) out.putInt((int) transport.longValue());
        for (Long unit : cargo) out.putInt((int) unit.longValue());
    }

    static void empty(java.nio.ByteBuffer out, int sections) {
        for (int i = 0; i < sections; i++) out.putShort((short) 0);
    }

    /** One tactical period of the command code: upkeep, a scan, and a line for every change of a squad's status, a lift's phase and every event. */
    static void chainStep(int at) throws Exception {
        commander.upkeep(engineObj);
        world.refresh(engineObj, eng.local(engineObj));
        for (World.Squad squad : world.squads.values()) {
            String state = squadState(squad);
            if (!state.equals(lastSquadState.get(Integer.valueOf(squad.id)))) log(String.format("chain t=%.1f %s", at / 1000.0, state));
            lastSquadState.put(Integer.valueOf(squad.id), state);
        }
        for (Lift lift : world.lifts.values()) {
            String state = liftState(lift);
            if (!state.equals(lastLiftState.get(Integer.valueOf(lift.id)))) log(String.format("chain t=%.1f %s", at / 1000.0, state));
            lastLiftState.put(Integer.valueOf(lift.id), state);
        }
        for (Lift lift : world.endedLifts) log(String.format("chain t=%.1f ended %s", at / 1000.0, liftState(lift)));
        world.endedLifts.clear();
        for (World.Event event : world.events) {
            log(String.format("chain t=%.1f event kind=%d squad=%d unit=#%d value=%.0f", at / 1000.0, event.kind, event.squad, event.unit, event.value));
        }
        world.events.clear();
    }

    static String squadState(World.Squad squad) {
        return "squad " + squad.id + " " + STATUSES[squad.status] + " task=" + (squad.issuedAtMs == 0 ? "-" : TASKS[squad.task])
                + " units=" + squad.units.size() + " aboard=" + squad.aboard + " passage=" + Passage.CLASSES[squad.passage]
                + (squad.lift == null ? "" : " lift=" + squad.lift.id);
    }

    static String liftState(Lift lift) {
        return "lift " + lift.id + " " + PHASES[lift.phase] + (lift.reason == 0 ? "" : " " + REASONS[lift.reason])
                + " loaded=" + lift.loaded + "/" + lift.expected + " transports=" + lift.load.size();
    }

    // ---- what is logged --------------------------------------------------------------------

    static void trackAll(int at) throws Exception {
        for (String label : tracked) {
            Long id = labels.get(label);
            if (id == null) {
                log(String.format("track t=%.1f %s unbound", at / 1000.0, label));
                continue;
            }
            Object unit = byIdAll(id.longValue());
            if (unit == null) {
                log(String.format("track t=%.1f %s #%d absent from the unit list", at / 1000.0, label, id));
                continue;
            }
            log(String.format("track t=%.1f %s %s", at / 1000.0, label, describe(unit)));
        }
    }

    static String describe(Object unit) {
        StringBuilder out = new StringBuilder();
        try {
            out.append('#').append(eng.id(unit)).append(' ').append(eng.typeName(eng.type(unit)));
            out.append(String.format(" (%.0f,%.0f)", eng.x(unit), eng.y(unit)));
            out.append(" hp=").append((int) eng.health(unit)).append('/').append((int) eng.maxHealth(unit));
            out.append(eng.dead(unit) ? " DEAD" : "");
            out.append(" built=").append(String.format("%.2f", eng.built(unit)));
            Object owner = eng.owner(unit);
            out.append(" slot=").append(owner == null ? "-" : String.valueOf(slotOf(owner)));
            Object carrier = getField(unit, "cN");
            if (carrier != null) out.append(" inside=#").append(eng.id(carrier));
            Object other = getField(unit, "cO");
            if (other != null) out.append(" cO=#").append(eng.id(other));
            int capacity = intCall(unit, "bZ");
            if (capacity >= 0) {
                out.append(" load=").append(intCall(unit, "bY")).append('/').append(capacity);
                try {
                    out.append(" unloading=").append(getField(unit, "g"));
                } catch (Exception ignored) {
                    // a transport without the hovercraft's flag
                }
            }
            if (eng.armedClass.isInstance(unit)) {
                out.append(" orders=").append(findMethod(unit.getClass(), "av").invoke(unit));
                Object order = findMethod(unit.getClass(), "ar").invoke(unit);
                if (order != null) out.append(" order=").append(describeOrder(order));
            }
            out.append(" inList=").append(inList(unit));
        } catch (Throwable e) {
            out.append(" <").append(e).append('>');
        }
        return out.toString();
    }

    static String describeOrder(Object order) {
        try {
            for (Field f : order.getClass().getDeclaredFields()) {
                f.setAccessible(true);
                Object value = f.get(order);
                if (value != null && value.getClass().getName().equals("com.corrodinggames.rts.game.units.av")) return String.valueOf(value);
            }
        } catch (Throwable ignored) {
            // fall through
        }
        return "?";
    }

    static void listActions(Object unit) throws Exception {
        Object actions = findMethod(unit.getClass(), "N").invoke(unit);
        if (!(actions instanceof List)) {
            log("actions: none");
            return;
        }
        int index = 0;
        for (Object action : (List<?>) actions) {
            log("actions " + index++ + ": " + actionText(action));
        }
    }

    static String actionText(Object action) {
        StringBuilder out = new StringBuilder(action.getClass().getName());
        for (String name : new String[]{"a", "b", "f", "i", "N"}) {
            try {
                Method m = findMethod(action.getClass(), name);
                out.append(' ').append(name).append('=').append(m.invoke(action));
            } catch (Throwable ignored) {
                // not every action answers every question
            }
        }
        return out.toString();
    }

    static Object findAction(Object unit, String wanted) throws Exception {
        Object actions = findMethod(unit.getClass(), "N").invoke(unit);
        if (!(actions instanceof List)) return null;
        List<?> list = (List<?>) actions;
        try {
            int index = Integer.parseInt(wanted);
            return index < list.size() ? list.get(index) : null;
        } catch (NumberFormatException ignored) {
            // matched by text
        }
        for (Object action : list) {
            if (actionText(action).toLowerCase().contains(wanted.toLowerCase())) return action;
        }
        return null;
    }

    /** What the engine says about every type through its sample units, which is the capability table a control process could be handed. */
    static void catalog() throws Exception {
        Class<?> unitClass = Class.forName("com.corrodinggames.rts.game.units.am");
        Class<?> typeInterface = Class.forName("com.corrodinggames.rts.game.units.as");
        Method sampleOf = findStatic(unitClass, "a", typeInterface);
        Class<?> utility = Class.forName("com.corrodinggames.rts.gameFramework.utility.y");
        Method passenger = findStatic(utility, "a", unitClass, boolean.class, boolean.class);
        java.util.LinkedHashMap<String, Object> effective = new java.util.LinkedHashMap<String, Object>();
        for (Object type : eng.allTypes()) {
            String lookup = eng.typeName(type);
            Object resolved = eng.typeNamed(lookup);
            effective.put(lookup, resolved == null ? type : resolved);
        }
        log("catalog: lookup|class|movement|price|tech|building|builder|canAttack|range|hitsAir|hitsLand|capacity|slots|transportable|speed|hp|makes");
        for (Map.Entry<String, Object> entry : effective.entrySet()) {
            Object type = entry.getValue();
            Object sample = null;
            try {
                sample = sampleOf.invoke(null, type);
            } catch (Throwable ignored) {
                // no sample
            }
            StringBuilder line = new StringBuilder("catalog: ").append(entry.getKey());
            line.append('|').append(sample == null ? "-" : sample.getClass().getName().replace("com.corrodinggames.rts.game.units.", ""));
            line.append('|').append(eng.typeMovement(type));
            line.append('|').append(eng.typePrice(type)).append('|').append(eng.typeTech(type));
            line.append('|').append(eng.typeIsBuilding(type)).append('|').append(eng.typeIsBuilder(type));
            line.append('|').append(sample == null ? "?" : String.valueOf(boolCall(sample, "l")));
            line.append('|').append(eng.typeRange(type));
            line.append('|').append(eng.typeHitsAir(type)).append('|').append(eng.typeHitsLand(type));
            line.append('|').append(sample == null ? "?" : String.valueOf(intCall(sample, "bZ")));
            line.append('|').append(sample == null ? "?" : String.valueOf(intCall(sample, "cw")));
            String transportable = "?";
            if (sample != null) {
                try {
                    transportable = String.valueOf(passenger.invoke(null, sample, Boolean.FALSE, Boolean.FALSE));
                } catch (Throwable e) {
                    transportable = "err";
                }
            }
            line.append('|').append(transportable);
            String speed = "?";
            if (sample != null) {
                try {
                    speed = String.valueOf(findMethod(sample.getClass(), "z").invoke(sample));
                } catch (Throwable ignored) {
                    // no speed
                }
            }
            line.append('|').append(speed);
            line.append('|').append((int) eng.typeMaxHealth(type));
            StringBuilder makes = new StringBuilder();
            if (sample != null) {
                try {
                    for (Object made : eng.producible(sample)) makes.append(eng.typeName(made)).append(' ');
                } catch (Throwable ignored) {
                    // no menu
                }
            }
            line.append('|').append(makes.toString().trim());
            log(line.toString());
        }
        log("catalog: end");
    }

    /** For every transport type, which mobile types its sample unit says it would take, by the engine's own load test on two sample units. */
    static void compatibility() throws Exception {
        Class<?> unitClass = Class.forName("com.corrodinggames.rts.game.units.am");
        Class<?> typeInterface = Class.forName("com.corrodinggames.rts.game.units.as");
        Method sampleOf = findStatic(unitClass, "a", typeInterface);
        Method canLoad = unitClass.getDeclaredMethod("d", unitClass, boolean.class);
        canLoad.setAccessible(true);
        java.util.LinkedHashMap<String, Object> samples = new java.util.LinkedHashMap<String, Object>();
        for (Object type : eng.allTypes()) {
            String lookup = eng.typeName(type);
            Object resolved = eng.typeNamed(lookup);
            Object effective = resolved == null ? type : resolved;
            try {
                Object sample = sampleOf.invoke(null, effective);
                if (sample != null && !eng.typeIsBuilding(effective) && eng.typePrice(effective) > 0) samples.put(lookup, sample);
            } catch (Throwable ignored) {
                // no sample
            }
        }
        for (Map.Entry<String, Object> transport : samples.entrySet()) {
            if (intCall(transport.getValue(), "bZ") <= 0) continue;
            StringBuilder yes = new StringBuilder();
            StringBuilder no = new StringBuilder();
            for (Map.Entry<String, Object> passenger : samples.entrySet()) {
                String verdict;
                try {
                    verdict = String.valueOf(canLoad.invoke(transport.getValue(), passenger.getValue(), Boolean.FALSE));
                } catch (Throwable e) {
                    verdict = "err:" + e.getClass().getSimpleName();
                }
                (verdict.equals("true") ? yes : no).append(passenger.getKey()).append(verdict.equals("true") || verdict.equals("false") ? "" : "(" + verdict + ")").append(' ');
            }
            log("compat " + transport.getKey() + " takes: " + yes.toString().trim());
            log("compat " + transport.getKey() + " refuses: " + no.toString().trim());
        }
        log("compat: end");
    }

    /** Writes, per movement type, the path finder's own passability of every tile and the contents of its grid's arrays, to `<prefix>-<MOVEMENT>.txt`. */
    static void passMap(String prefix) throws Exception {
        Object finder = getField(engineObj, "bU");
        Class<?> movementClass = Class.forName("com.corrodinggames.rts.game.units.ao");
        Class<?> gridClass = Class.forName("com.corrodinggames.rts.gameFramework.k.i");
        Method passable = finder.getClass().getDeclaredMethod("a", movementClass, int.class, int.class);
        passable.setAccessible(true);
        Method passableStrict = finder.getClass().getDeclaredMethod("b", movementClass, int.class, int.class);
        passableStrict.setAccessible(true);
        Method gridOf = finder.getClass().getDeclaredMethod("a", movementClass);
        gridOf.setAccessible(true);
        for (Object movement : movementClass.getEnumConstants()) {
            Object grid = gridOf.invoke(finder, movement);
            if (grid == null) {
                log("passmap " + movement + ": no grid");
                continue;
            }
            int width = findField(gridClass, "b").getInt(grid);
            int height = findField(gridClass, "c").getInt(grid);
            java.io.PrintWriter out = new java.io.PrintWriter(new java.io.FileWriter(prefix + "-" + movement + ".txt"));
            out.println("width " + width + " height " + height);
            StringBuilder rows = new StringBuilder();
            int open = 0;
            for (int y = 0; y < height; y++) {
                StringBuilder row = new StringBuilder();
                for (int x = 0; x < width; x++) {
                    boolean a = ((Boolean) passable.invoke(finder, movement, Integer.valueOf(x), Integer.valueOf(y))).booleanValue();
                    boolean b = ((Boolean) passableStrict.invoke(finder, movement, Integer.valueOf(x), Integer.valueOf(y))).booleanValue();
                    row.append(a ? (b ? '#' : '+') : (b ? '-' : '.'));
                    if (a) open++;
                }
                out.println(row);
            }
            for (String name : new String[]{"d", "e", "f", "j"}) {
                byte[] values = (byte[]) findField(gridClass, name).get(grid);
                out.println("array " + name + " " + (values == null ? "null" : String.valueOf(values.length)));
                if (values != null) {
                    StringBuilder text = new StringBuilder();
                    for (byte value : values) text.append(value).append(',');
                    out.println(text);
                }
            }
            short[] labels = (short[]) findField(gridClass, "g").get(grid);
            out.println("array g " + (labels == null ? "null" : String.valueOf(labels.length)));
            if (labels != null) {
                StringBuilder text = new StringBuilder();
                for (short value : labels) text.append(value).append(',');
                out.println(text);
            }
            out.close();
            log("passmap " + movement + ": " + width + "x" + height + " open=" + open);
        }
        log("passmap: end");
    }

    // ---- engine helpers --------------------------------------------------------------------

    static List<Object> liveUnits() throws Exception {
        List<Object> out = new ArrayList<Object>();
        Object[] units = eng.unitArray();
        int count = eng.unitCount();
        for (int i = 0; i < count && i < units.length; i++) {
            Object unit = units[i];
            if (unit != null && !eng.dead(unit)) out.add(unit);
        }
        return out;
    }

    static Object byIdAll(long id) throws Exception {
        Object[] units = eng.unitArray();
        int count = eng.unitCount();
        for (int i = 0; i < count && i < units.length; i++) {
            Object unit = units[i];
            if (unit != null && eng.id(unit) == id) return unit;
        }
        return lastKnown.get(Long.valueOf(id));
    }

    static final Map<Long, Object> lastKnown = new HashMap<Long, Object>();

    static boolean inList(Object unit) throws Exception {
        Object[] units = eng.unitArray();
        int count = eng.unitCount();
        for (int i = 0; i < count && i < units.length; i++) if (units[i] == unit) return true;
        return false;
    }

    static Object unit(String label) throws Exception {
        Long id = labels.get(label);
        if (id == null) throw new IllegalStateException("no unit bound to " + label);
        Object unit = byIdAll(id.longValue());
        if (unit == null) throw new IllegalStateException("unit " + label + " #" + id + " is gone");
        lastKnown.put(id, unit);
        return unit;
    }

    static long maxId() throws Exception {
        long best = -1;
        for (Object unit : liveUnits()) best = Math.max(best, eng.id(unit));
        return best;
    }

    static Object nearestOwn(String typeName, float x, float y, boolean enemy) throws Exception {
        Object self = eng.local(engineObj);
        Object best = null;
        double bestDistance = Double.MAX_VALUE;
        for (Object unit : liveUnits()) {
            Object owner = eng.owner(unit);
            if (owner == null || (owner == self) == enemy) continue;
            if (enemy && eng.team(owner) < 0) continue;
            if (!eng.typeName(eng.type(unit)).equals(typeName)) continue;
            if (labels.containsValue(Long.valueOf(eng.id(unit)))) continue;
            double distance = Float.isNaN(x) ? 0 : Math.hypot(eng.x(unit) - x, eng.y(unit) - y);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = unit;
            }
        }
        return best;
    }

    static Object enemyPlayer() throws Exception {
        Object self = eng.local(engineObj);
        for (int slot = 0; slot < 10; slot++) {
            Object player = eng.playerAt(slot);
            if (player != null && player != self && eng.team(player) >= 0 && eng.team(player) != eng.team(self)) return player;
        }
        return null;
    }

    static Object commandFor(Object unit) throws Exception {
        return eng.command(engineObj, eng.owner(unit));
    }

    static int slotOf(Object player) {
        try {
            return getField(player, "k") instanceof Integer ? ((Integer) getField(player, "k")).intValue() : -1;
        } catch (Exception e) {
            return -1;
        }
    }

    static int intCall(Object target, String name) {
        try {
            return ((Integer) findMethod(target.getClass(), name).invoke(target)).intValue();
        } catch (Throwable e) {
            return -9;
        }
    }

    static String boolCall(Object target, String name) {
        try {
            return String.valueOf(findMethod(target.getClass(), name).invoke(target));
        } catch (Throwable e) {
            return "?";
        }
    }

    static Field findField(Class<?> type, String name) throws Exception {
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

    static Method findMethod(Class<?> type, String name, Class<?>... parameters) throws Exception {
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

    static Method findStatic(Class<?> type, String name, Class<?>... parameters) throws Exception {
        Method m = type.getDeclaredMethod(name, parameters);
        m.setAccessible(true);
        return m;
    }

    static Object getField(Object target, String name) throws Exception {
        return findField(target.getClass(), name).get(target);
    }

    static void setField(Object target, String name, Object value) throws Exception {
        findField(target.getClass(), name).set(target, value);
    }

    static Object call(Object target, String name) throws Exception {
        return findMethod(target.getClass(), name).invoke(target);
    }

    static String join(String[] parts) {
        StringBuilder out = new StringBuilder();
        for (String part : parts) {
            if (out.length() > 0) out.append(' ');
            out.append(part);
        }
        return out.toString();
    }

    static void log(String message) {
        System.out.println("[rw-lab] " + message);
        System.out.flush();
    }
}
