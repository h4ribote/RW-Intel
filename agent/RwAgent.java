import java.lang.instrument.Instrumentation;
import java.util.List;

/**
 * The javaagent that puts a Rusted Warfare process under outside command.
 *
 * It does three things, all of them on the game thread: it builds the observation, it turns arriving decisions into engine commands, and it starts and ends episodes. Everything that touches the network happens on another thread, and the two exchange single element slots (see {@link Link}).
 *
 * Everything must be on the game thread because the engine gives no other safe point: the command pool is an unsynchronised list, loading a map needs the OpenGL context that only that thread holds, and reading unit state from elsewhere samples a step in progress and returns units observed at different instants. The engine drains a queue of tasks in full at the top of every simulation step, and a task that queues itself again is therefore a per step hook.
 *
 * Agent options, comma separated:
 *   host=&lt;name&gt;       control process to connect to, default 127.0.0.1
 *   port=&lt;number&gt;     default 8642
 *   instance=&lt;n&gt;      which instance this is, so the control process can tell them apart
 *   speed=&lt;float&gt;     engine speed multiplier to hold, omit to leave it alone
 *   tactical=&lt;ms&gt;     game time between observations, default 200 which is 5Hz
 *   operational=&lt;ms&gt;  game time between region blocks, default 2000
 *   omniscient=&lt;bool&gt; report every enemy rather than only what the engine says is visible, default true
 */
public final class RwAgent {

    private static volatile String host = "127.0.0.1";
    private static volatile int port = 8642;
    private static volatile int instance = 0;
    private static volatile float speed = -1f;
    private static volatile int tacticalMs = 200;
    private static volatile int operationalMs = 2000;
    private static volatile boolean omniscient = true;

    private static Engine engine;
    private static World world;
    private static Observer observer;
    private static Commander commander;
    private static MatchDriver driver;
    private static Link link;

    /** Set once the world is loaded and the episode is under way, cleared when it ends. */
    private static boolean episodeRunning = false;
    private static int lastTacticalMs = Integer.MIN_VALUE;
    private static int lastOperationalMs = Integer.MIN_VALUE;
    private static boolean catalogueSent = false;

    public static void premain(String arguments, Instrumentation instrumentation) {
        parse(arguments);
        log("starting: host=" + host + " port=" + port + " instance=" + instance
                + " speed=" + speed + " tactical=" + tacticalMs + "ms");
        Thread starter = new Thread(new Runnable() {
            public void run() {
                try {
                    startUp();
                } catch (Throwable e) {
                    log("start up failed: " + e);
                    e.printStackTrace();
                }
            }
        }, "rw-agent");
        starter.setDaemon(true);
        starter.start();
    }

    private static void parse(String arguments) {
        if (arguments == null) return;
        for (String pair : arguments.split(",")) {
            int equals = pair.indexOf('=');
            if (equals < 0) continue;
            String key = pair.substring(0, equals).trim();
            String value = pair.substring(equals + 1).trim();
            if (key.equals("host")) host = value;
            else if (key.equals("port")) port = Integer.parseInt(value);
            else if (key.equals("instance")) instance = Integer.parseInt(value);
            else if (key.equals("speed")) speed = Float.parseFloat(value);
            else if (key.equals("tactical")) tacticalMs = Integer.parseInt(value);
            else if (key.equals("operational")) operationalMs = Integer.parseInt(value);
            else if (key.equals("omniscient")) omniscient = Boolean.parseBoolean(value);
        }
    }

    /**
     * Frames the game loop must have run before the agent resolves anything.
     *
     * Loading a game class runs its static initialiser, and the player class builds units in its own. Reaching it before the game has got there itself throws, and a class whose initialiser has thrown stays broken for the rest of the process, so the game then dies too. Waiting for the loop to have run means every class the agent names has already been initialised by the game.
     */
    private static final int SAFE_FRAME = 300;

    private static void startUp() throws Exception {
        Class<?> engineClass;
        while (true) {
            try {
                engineClass = Class.forName("com.corrodinggames.rts.gameFramework.l");
                break;
            } catch (ClassNotFoundException e) {
                Thread.sleep(200);
            }
        }
        java.lang.reflect.Method singleton = engineClass.getMethod("B");
        java.lang.reflect.Field frames = engineClass.getDeclaredField("bx");
        frames.setAccessible(true);

        Object game = null;
        while (game == null || frames.getInt(game) < SAFE_FRAME) {
            game = singleton.invoke(null);
            Thread.sleep(100);
        }
        log("engine ready at frame " + frames.getInt(game));

        engine = new Engine();

        world = new World(engine);
        world.omniscient = omniscient;
        observer = new Observer(engine, world);
        commander = new Commander(engine, world);
        driver = new MatchDriver(engine);
        link = new Link(host, port, instance);

        while (!link.connect()) {
            log("waiting for the control process at " + host + ":" + port);
            Thread.sleep(1000);
        }
        log("connected");
        pump(game);
    }

    /** Set while a step is queued or running, so only ever one is outstanding. */
    private static final java.util.concurrent.atomic.AtomicBoolean queued =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    private static final Runnable STEP = new Runnable() {
        public void run() {
            try {
                Object game = engine.engine();
                if (game != null) step(game);
            } catch (Throwable e) {
                log("step failed: " + e);
                e.printStackTrace();
            } finally {
                queued.set(false);
            }
        }
    };

    /**
     * Decides when the game thread has work, and queues one task when it does.
     *
     * The work itself has to run on the game thread, but it cannot be scheduled from there. The engine drains its task queue until the queue is empty, so a task that queues itself again is drained again in the same frame and the frame never ends; the first attempt at this froze the game outright. Deciding here and queueing one task at a time keeps that from happening.
     *
     * Reading the clock from this thread is fine because nothing is done with it but scheduling. Reading the world would not be: the values would come from part way through a step.
     */
    private static void pump(Object game) throws Exception {
        while (true) {
            if (link.connected() && !queued.get()) {
                int now = engine.gameTime(game);
                boolean due = !catalogueSent
                        || link.hasControl()
                        || (episodeRunning && (lastTacticalMs == Integer.MIN_VALUE || now - lastTacticalMs >= tacticalMs));
                if (due && queued.compareAndSet(false, true)) {
                    try {
                        engine.post(game, STEP);
                    } catch (Exception e) {
                        queued.set(false);
                        throw e;
                    }
                }
            }
            // Fine enough against a tactical period, which is twenty milliseconds of wall clock at ten times speed.
            java.util.concurrent.locks.LockSupport.parkNanos(1000000L);
        }
    }

    private static void step(Object game) throws Exception {
        if (!link.connected()) return;

        if (speed > 0f && engine.speed(game) != speed) engine.setSpeed(game, speed);

        if (!catalogueSent) {
            sendHello(game);
            catalogueSent = true;
        }

        String control;
        while ((control = link.takeControl()) != null) handleControl(game, control);

        if (!episodeRunning) return;

        if (driver.finished(game)) {
            endEpisode(game);
            return;
        }

        int now = engine.gameTime(game);

        // The action built from the previous observation is applied first, which is what fixes the decision lag at exactly one period.
        byte[] action = link.takeAction();
        if (action != null) commander.apply(game, action);

        int blocks = Wire.BLOCK_SQUADS | Wire.BLOCK_UNITS;
        if (lastOperationalMs == Integer.MIN_VALUE || now - lastOperationalMs >= operationalMs) {
            blocks |= Wire.BLOCK_REGIONS;
            lastOperationalMs = now;
        }
        lastTacticalMs = now;

        byte[] body = observer.build(game, driver.episode(), blocks);
        if (body != null) link.send(Wire.KIND_OBSERVATION, body);
    }

    private static void sendHello(Object game) throws Exception {
        // A definition file that replaces a built-in leaves both in the registry, and only the lookup says which one the game uses. Both names are reported: one is what a type is asked for by, the other is what it calls itself and what a production action id is built from.
        java.util.LinkedHashMap<String, Object> effective = new java.util.LinkedHashMap<String, Object>();
        for (Object type : engine.allTypes()) {
            String lookup = engine.typeName(type);
            Object resolved = engine.typeNamed(lookup);
            effective.put(lookup, resolved == null ? type : resolved);
        }
        world.indexTypes(new java.util.ArrayList<Object>(effective.values()));

        java.util.Map<String, String> lookupByName = new java.util.HashMap<String, String>();
        for (java.util.Map.Entry<String, Object> entry : effective.entrySet()) {
            String reported = engine.typeName(entry.getValue());
            if (!lookupByName.containsKey(reported)) lookupByName.put(reported, entry.getKey());
        }

        StringBuilder catalogue = new StringBuilder("[");
        for (Object type : world.types()) {
            String reported = engine.typeName(type);
            String lookup = lookupByName.get(reported);
            if (catalogue.length() > 1) catalogue.append(',');
            catalogue.append("{\"name\":\"").append(reported).append('"')
                    .append(",\"lookup\":\"").append(lookup == null ? reported : lookup).append('"')
                    .append(",\"price\":").append(engine.typePrice(type))
                    .append(",\"tech\":").append(engine.typeTech(type))
                    .append(",\"building\":").append(engine.typeIsBuilding(type))
                    .append(",\"builder\":").append(engine.typeIsBuilder(type))
                    .append(",\"movement\":\"").append(engine.typeMovement(type)).append('"')
                    .append('}');
        }
        catalogue.append(']');

        Wire.Json hello = new Wire.Json();
        hello.put("instance", instance);
        hello.put("tacticalMs", tacticalMs);
        hello.put("operationalMs", operationalMs);
        hello.put("omniscient", omniscient);
        hello.put("slots", engine.slotCount());
        hello.raw("unitTypes", catalogue.toString());
        link.send(Wire.KIND_HELLO, hello.toBytes());
        log("sent the catalogue: " + world.types().size() + " types");
    }

    private static void handleControl(Object game, String json) throws Exception {
        String command = Wire.field(json, "command");
        if (command == null) return;
        if (command.equals("start")) {
            MatchDriver.Settings settings = new MatchDriver.Settings();
            settings.map = Wire.field(json, "map") == null ? "" : Wire.field(json, "map");
            settings.opponents = Wire.intField(json, "opponents", 1);
            settings.difficulty = Wire.intField(json, "difficulty", 1);
            settings.contestants = Wire.intField(json, "contestants", 0);
            settings.credits = Wire.intField(json, "credits", 0);
            settings.startingUnits = Wire.intField(json, "startingUnits", 1);
            settings.income = Wire.floatField(json, "income", 1.0f);
            settings.fog = Wire.intField(json, "fog", 2);
            settings.seed = Wire.intField(json, "seed", 12345);
            settings.maxSeconds = Wire.intField(json, "maxSeconds", 0);
            world.reset();
            driver.start(game, settings);
            episodeRunning = true;
            lastTacticalMs = Integer.MIN_VALUE;
            lastOperationalMs = Integer.MIN_VALUE;

            Wire.Json event = new Wire.Json();
            event.put("event", "started");
            event.put("episode", driver.episode());
            event.put("map", driver.map());
            event.put("seed", settings.seed);
            event.raw("players", driver.players());
            link.send(Wire.KIND_EPISODE, event.toBytes());
        } else if (command.equals("regions")) {
            world.setRegions(parseRegions(json));
            log("region table: " + world.regions.size() + " regions");
        } else if (command.equals("abort")) {
            if (episodeRunning) endEpisode(game);
        } else if (command.equals("speed")) {
            speed = Wire.floatField(json, "value", speed);
            if (speed > 0f) engine.setSpeed(game, speed);
        } else if (command.equals("omniscient")) {
            world.omniscient = Wire.boolField(json, "value", world.omniscient);
        }
    }

    /**
     * Reads the region table out of a control frame.
     * The rows are a flat array of x, y, resource count and distance from home, four numbers each, because that avoids a general parser for the one message that carries a list.
     */
    private static List<float[]> parseRegions(String json) {
        java.util.List<float[]> rows = new java.util.ArrayList<float[]>();
        int start = json.indexOf("\"regions\"");
        if (start < 0) return rows;
        int open = json.indexOf('[', start);
        int close = json.indexOf(']', open);
        if (open < 0 || close < 0) return rows;
        String[] numbers = json.substring(open + 1, close).split(",");
        for (int i = 0; i + 3 < numbers.length; i += 4) {
            try {
                rows.add(new float[]{
                        Float.parseFloat(numbers[i].trim()),
                        Float.parseFloat(numbers[i + 1].trim()),
                        Float.parseFloat(numbers[i + 2].trim()),
                        Float.parseFloat(numbers[i + 3].trim()),
                });
            } catch (NumberFormatException e) {
                break;
            }
        }
        return rows;
    }

    private static void endEpisode(Object game) throws Exception {
        episodeRunning = false;
        java.util.Set<Integer> alive = driver.aliveTeams();
        int seconds = engine.gameTime(game) / 1000;

        Wire.Json event = new Wire.Json();
        event.put("event", "finished");
        event.put("episode", driver.episode());
        event.put("seconds", seconds);
        event.put("frames", engine.frame(game));
        event.put("winner", alive.size() == 1 ? alive.iterator().next().intValue() : -1);
        event.put("aliveTeams", alive.size());
        Object self = engine.local(game);
        event.put("team", self == null ? -1 : engine.team(self));
        event.put("timeout", driver.settings().maxSeconds > 0 && seconds >= driver.settings().maxSeconds);
        event.raw("standing", driver.standing(game));
        link.send(Wire.KIND_EPISODE, event.toBytes());
        log("episode " + driver.episode() + " finished at " + seconds + "s, alive teams " + alive);
    }

    static void log(String message) {
        System.out.println("[rw-agent] " + message);
        System.out.flush();
    }
}
