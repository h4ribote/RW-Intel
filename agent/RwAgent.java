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

    private static volatile Engine engine;
    private static volatile World world;
    private static volatile Observer observer;
    private static volatile Commander commander;
    private static volatile MatchDriver driver;
    private static volatile Link link;

    /** Set once the world is loaded and the episode is under way, cleared when it ends. */
    private static volatile boolean episodeRunning = false;
    /** Set on a process that has joined another's match and has nothing to do but watch for the host to start one. */
    private static volatile boolean awaitingHost = false;
    /** Set on a process that has opened a room for a match another process is to join, and is waiting for it to appear. */
    private static volatile boolean awaitingPeer = false;
    private static volatile int lastTacticalMs = Integer.MIN_VALUE;
    private static volatile int lastOperationalMs = Integer.MIN_VALUE;
    private static volatile boolean catalogueSent = false;
    /** Set when an episode event could not be sent because the link was down, so that it goes out again once there is a link. */
    private static volatile String pendingEpisodeEvent = null;

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
        long lastAttempt = 0;
        long lastJoinPoll = 0;
        while (true) {
            long wall = System.currentTimeMillis();
            if (!link.connected()) {
                // The episode is left running, and the squads keep the contracts they were last given: the engine goes on advancing them, which is a better thing to be doing while out of touch than standing still.
                if (wall - lastAttempt >= 1000) {
                    lastAttempt = wall;
                    if (link.connect()) {
                        catalogueSent = false;
                        log("reconnected to the control process");
                    }
                }
            } else if (!queued.get()) {
                int now = engine.gameTime(game);
                // A process waiting on a host has no episode of its own to pace against, and cannot pace against the game clock either, because the clock it will be running on is the one the host is about to hand it. So it is looked at on the wall clock, often enough that the wait costs nothing worth measuring.
                boolean waiting = (awaitingHost || awaitingPeer) && wall - lastJoinPoll >= 200;
                boolean due = !catalogueSent
                        || link.hasControl()
                        || waiting
                        || (episodeRunning && (lastTacticalMs == Integer.MIN_VALUE || now - lastTacticalMs >= tacticalMs));
                if (due && queued.compareAndSet(false, true)) {
                    if (waiting) lastJoinPoll = wall;
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

        if (pendingEpisodeEvent != null) {
            String held = pendingEpisodeEvent;
            pendingEpisodeEvent = null;
            sendEpisodeEvent(held);
        }

        String control;
        while ((control = link.takeControl()) != null) handleControl(game, control);

        // A host with an open room has nothing to report until somebody is in it, and the room only fills while this loop keeps running.
        if (awaitingPeer) {
            if (!driver.beginWhenReady(game)) return;
            awaitingPeer = false;
            beginEpisode(game);
            return;
        }

        // A process that joined another's match has nothing of its own to report until the host starts one, and no say in when that is.
        if (awaitingHost) {
            if (!driver.running(game)) return;
            awaitingHost = false;
            driver.joined(game);
            beginEpisode(game);
            return;
        }

        if (!episodeRunning) return;

        if (driver.finished(game)) {
            endEpisode(game);
            return;
        }

        int now = engine.gameTime(game);

        // The action built from the previous observation is applied first, which is what fixes the decision lag at exactly one period.
        byte[] action = link.takeAction();
        if (action != null) commander.apply(game, action);

        sampleStanding(game, now);

        int blocks = Wire.BLOCK_SQUADS | Wire.BLOCK_UNITS;
        if (lastOperationalMs == Integer.MIN_VALUE || now - lastOperationalMs >= operationalMs) {
            // Events ride the operational frame because the layer that consumes them, the one that forms and retires squads, runs there. They are accumulated meanwhile rather than dropped.
            blocks |= Wire.BLOCK_REGIONS | Wire.BLOCK_EVENTS;
            lastOperationalMs = now;
        }
        lastTacticalMs = now;

        byte[] body = observer.build(game, driver.episode(), blocks);
        if (body != null && link.send(Wire.KIND_OBSERVATION, body) && (blocks & Wire.BLOCK_EVENTS) != 0) {
            // Cleared only once the frame carrying them has actually gone. An event describes a change, so reporting one twice is reporting a change that did not happen, but dropping one is worse: the layer that forms and retires squads has no other way to learn of it.
            world.events.clear();
        }
    }

    /**
     * How far before the end the board the scoring weights are fitted on is taken from, in game milliseconds.
     *
     * The design fits the weights on the condition that the score just BEFORE a decision agrees with who won it, and what the finished event carries is the board the episode ended ON. For a decided match that board is taken after the loser has been destroyed, so a fit made on it is a fit on a question nobody has to ask: every weighting calls a board with one side left correctly. The board the score is actually used on is one where both sides are still standing, which is what this is.
     *
     * Half a game minute, because that is comfortably longer than the last exchange of a match and comfortably shorter than the stretch over which a match is decided. It is reported beside the sample rather than assumed, since an episode shorter than this has no such board and says so.
     */
    private static final int BEFORE_MS = 30000;

    /** How often the rolling board is taken. Fine enough that the sample handed over is within this of the moment asked for, coarse enough that walking every unit for it costs nothing against a tactical period. */
    private static final int STANDING_SAMPLE_MS = 5000;

    private static int lastStandingMs = Integer.MIN_VALUE;
    private static final java.util.ArrayDeque<int[]> standingWhen = new java.util.ArrayDeque<int[]>();
    private static final java.util.ArrayDeque<String> standingWhat = new java.util.ArrayDeque<String>();

    /**
     * Keeps a short history of where the sides stood, so that the end of the episode can be described by a board taken before it.
     *
     * Only as much of it as {@link #BEFORE_MS} asks for is kept: everything older than that is already past the moment that will be wanted, so holding it would be holding a match's worth of boards to hand over one of them.
     */
    private static void sampleStanding(Object game, int now) throws Exception {
        if (lastStandingMs != Integer.MIN_VALUE && now - lastStandingMs < STANDING_SAMPLE_MS) return;
        lastStandingMs = now;
        standingWhen.addLast(new int[]{now});
        standingWhat.addLast(driver.standing(game));
        while (standingWhen.size() > 1 && now - standingWhen.getFirst()[0] > BEFORE_MS + STANDING_SAMPLE_MS) {
            standingWhen.removeFirst();
            standingWhat.removeFirst();
        }
    }

    private static void clearStanding() {
        lastStandingMs = Integer.MIN_VALUE;
        standingWhen.clear();
        standingWhat.clear();
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

        // Range and whether a type can shoot at aircraft are not on the type interface. A type a definition file produced carries both, and one that exists only as code in the game carries neither, which is right for those: every one of them is a building or the builder. The organisation layer sorts units into doctrines from these, so they travel with the catalogue rather than being worked out again on each side.
        StringBuilder catalogue = new StringBuilder("[");
        for (Object type : world.types()) {
            String reported = engine.typeName(type);
            String lookup = lookupByName.get(reported);
            Boolean hitsAir = engine.typeHitsAir(type);
            Boolean hitsLand = engine.typeHitsLand(type);
            if (catalogue.length() > 1) catalogue.append(',');
            catalogue.append("{\"name\":\"").append(reported).append('"')
                    .append(",\"lookup\":\"").append(lookup == null ? reported : lookup).append('"')
                    .append(",\"price\":").append(engine.typePrice(type))
                    .append(",\"tech\":").append(engine.typeTech(type))
                    .append(",\"building\":").append(engine.typeIsBuilding(type))
                    .append(",\"builder\":").append(engine.typeIsBuilder(type))
                    .append(",\"extractor\":").append(engine.typeOnResourcePool(type))
                    .append(",\"movement\":\"").append(engine.typeMovement(type)).append('"')
                    .append(",\"range\":").append(world.rangeOfType(type))
                    .append(",\"hitsAir\":").append(hitsAir == null ? "true" : hitsAir.toString())
                    .append(",\"hitsLand\":").append(hitsLand == null ? "true" : hitsLand.toString())
                    .append('}');
        }
        catalogue.append(']');

        Wire.Json hello = new Wire.Json();
        hello.put("instance", instance);
        hello.put("build", engine.buildNumber());
        hello.put("tacticalMs", tacticalMs);
        hello.put("operationalMs", operationalMs);
        hello.put("omniscient", omniscient);
        hello.put("slots", engine.slotCount());
        hello.put("map", driver.map());
        // A reconnection is a fresh HELLO in the middle of whatever was already going on, so it has to say what that was: the control process rebuilds its side from the squad identifiers rather than starting the episode over.
        hello.put("episode", driver.episode());
        hello.put("running", episodeRunning);
        hello.raw("unitTypes", catalogue.toString());
        link.send(Wire.KIND_HELLO, hello.toBytes());
        log("sent the catalogue: " + world.types().size() + " types");
    }

    private static void handleControl(Object game, String json) throws Exception {
        String command = Wire.field(json, "command");
        if (command == null) return;
        if (command.equals("start")) {
            MatchDriver.Settings settings = new MatchDriver.Settings();
            settings.map = text(json, "map", "");
            settings.opponents = Wire.intField(json, "opponents", 1);
            settings.difficulty = Wire.intField(json, "difficulty", 1);
            settings.contestants = Wire.intField(json, "contestants", 0);
            settings.credits = Wire.intField(json, "credits", 0);
            settings.startingUnits = Wire.intField(json, "startingUnits", 1);
            settings.income = Wire.floatField(json, "income", 1.0f);
            settings.fog = Wire.intField(json, "fog", 2);
            settings.seed = Wire.intField(json, "seed", 12345);
            settings.maxSeconds = Wire.intField(json, "maxSeconds", 0);
            settings.arena = Wire.boolField(json, "arena", false);
            settings.networked = Wire.boolField(json, "host", false);
            settings.networkPort = Wire.intField(json, "port", settings.networkPort);
            settings.joinAddress = text(json, "join", "");
            // Named after the instance by default, because the name is what a host's desync report calls each client by and the instance number is the only thing that tells one agent from another.
            settings.name = text(json, "name", "rw-" + instance);
            world.reset();
            if (settings.joinAddress.isEmpty()) {
                if (driver.start(game, settings)) {
                    beginEpisode(game);
                } else {
                    // A networked host has opened its room and is waiting for the other process to finish joining it, which cannot happen while this thread stands still.
                    awaitingPeer = true;
                    link.takeAction();
                    log("hosting on port " + settings.networkPort + ", waiting for another process to join");
                }
            } else {
                driver.join(game, settings);
                awaitingHost = true;
                // A decision left over from the episode that just ended names units that no longer exist, and the wait for the host is long enough for one to arrive.
                link.takeAction();
                log("joined " + settings.joinAddress + ", waiting for the host to start a match");
            }
        } else if (command.equals("sync")) {
            Wire.Json event = new Wire.Json();
            event.put("event", "sync");
            event.put("episode", driver.episode());
            event.put("running", episodeRunning);
            event.raw("sync", driver.synchronisation(game));
            // The engine ships its own assertion over the same counters, which throws rather than answers, and which counts a client it has not yet sent a checksum to as a failure. It is asked only when the caller wants the engine's own wording.
            if (Wire.boolField(json, "assert", false)) event.put("complaint", engine.desyncComplaint(game));
            sendEpisodeEvent(event.toString());
        } else if (command.equals("regions")) {
            world.setRegions(parseRows(json, "regions", 4));
            world.setResourcePoints(parseRows(json, "resourcePoints", 3));
            log("region table: " + world.regions.size() + " regions");
        } else if (command.equals("abort")) {
            awaitingHost = false;
            if (episodeRunning) endEpisode(game);
        } else if (command.equals("speed")) {
            speed = Wire.floatField(json, "value", speed);
            if (speed > 0f) engine.setSpeed(game, speed);
        } else if (command.equals("omniscient")) {
            world.omniscient = Wire.boolField(json, "value", world.omniscient);
        } else if (command.equals("scenario")) {
            buildScenario(game, json);
        }
    }

    /** A string field of a control frame, or a default when it is absent. */
    private static String text(String json, String key, String fallback) {
        String value = Wire.field(json, key);
        return value == null ? fallback : value;
    }

    /**
     * Marks an episode as under way and tells the control process about it.
     *
     * This is the same whether the match was started here or by a host this process joined, and it is written once for that reason: the control process counts episodes off these frames, and a joined episode that announced itself differently would be an episode it had to count differently.
     */
    private static void beginEpisode(Object game) throws Exception {
        episodeRunning = true;
        lastTacticalMs = Integer.MIN_VALUE;
        lastOperationalMs = Integer.MIN_VALUE;
        // The rolling board belongs to the episode that has just ended; carried over, the first sample of this one would be a board from the last one.
        clearStanding();

        // Any decision left over from the episode that just ended names units that no longer exist.
        link.takeAction();

        Wire.Json event = new Wire.Json();
        event.put("event", "started");
        event.put("episode", driver.episode());
        event.put("map", driver.map());
        // Read back from the engine rather than repeated from what was asked for, because a process that joined a match was not asked and the host's seed is the one being played.
        event.put("seed", driver.seed(game));
        event.raw("players", driver.players());
        // Which player an arena episode's opposing side belongs to. Decided here because it is decided by how the room filled itself, which only this side sees.
        event.put("sparringSlot", driver.sparringSlot());
        event.raw("sync", driver.synchronisation(game));
        sendEpisodeEvent(event.toString());
    }

    /**
     * Builds an engagement to train the tactical layer on, without playing a match to reach it.
     *
     * Everything here goes through the host's spawn system command, which travels the route a player's order does. Nothing is assigned to engine state, because that is what breaks a lockstep session, and the whole reason for constructing situations is to train on them and then play with what was learnt.
     *
     * There is no instruction to clear the board, because the game offers no command that removes a unit: a scenario episode is started with no starting units instead, and then filled in. Its sandbox flag makes every player's units answerable here, which is what lets one process drive both sides of the engagement.
     */
    private static void buildScenario(Object game, String json) throws Exception {
        if (Wire.field(json, "sandbox") != null) engine.setSandbox(game, Wire.boolField(json, "sandbox", false));

        int made = 0;
        for (float[] row : parseRows(json, "spawns", 5)) {
            Object type = world.typeAt((int) row[0]);
            Object owner = engine.playerAt((int) row[1]);
            if (type == null || owner == null) continue;
            int count = Math.max(1, (int) row[4]);
            for (int i = 0; i < count; i++) {
                engine.spawn(game, owner, type, row[2], row[3]);
                made++;
            }
        }
        log("scenario: spawned " + made + " unit(s)");
    }

    /**
     * Reads a list out of a control frame as a flat array of numbers, a fixed count per row.
     * Every list the control frames carry is a table of numbers, so this is enough and a general parser would be answering a question nobody asked. Regions are x, y, resource count and whether the region holds a starting position; scenario spawns are type, player slot, x, y and how many.
     */
    private static List<float[]> parseRows(String json, String key, int width) {
        java.util.List<float[]> rows = new java.util.ArrayList<float[]>();
        int start = json.indexOf('"' + key + '"');
        if (start < 0) return rows;
        int open = json.indexOf('[', start);
        int close = json.indexOf(']', open);
        if (open < 0 || close < 0) return rows;
        String[] numbers = json.substring(open + 1, close).split(",");
        for (int i = 0; i + width - 1 < numbers.length; i += width) {
            float[] row = new float[width];
            try {
                for (int j = 0; j < width; j++) row[j] = Float.parseFloat(numbers[i + j].trim());
            } catch (NumberFormatException e) {
                break;
            }
            rows.add(row);
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
        // And the board as it stood half a minute earlier, which is the one the scoring weights have to be fitted on: the board an episode ENDS on has the loser already destroyed, and every weighting calls that one correctly. The oldest sample still held is the one nearest the moment asked for, since anything older than that is dropped as it is taken.
        if (!standingWhat.isEmpty()) {
            int at = standingWhen.getFirst()[0];
            if (seconds * 1000 - at >= BEFORE_MS - STANDING_SAMPLE_MS) {
                event.raw("before", standingWhat.getFirst());
                event.put("beforeSeconds", at / 1000);
            }
        }
        // An episode that fell out of step half way through is not one whose result may be used, so the verdict travels with the result rather than having to be asked for afterwards, by which time the next episode has already reset it.
        event.raw("sync", driver.synchronisation(game));
        sendEpisodeEvent(event.toString());
        log("episode " + driver.episode() + " finished at " + seconds + "s, alive teams " + alive);
    }

    /**
     * Sends an episode event, holding on to it if the link is down.
     *
     * These are the two frames the control process counts episodes with. One lost frame and the two sides disagree about how many have been run: the control process would start another to make up a number it had already reached, or wait for one that had already finished.
     */
    private static void sendEpisodeEvent(String json) {
        byte[] body;
        try {
            body = json.getBytes("UTF-8");
        } catch (java.io.UnsupportedEncodingException e) {
            throw new IllegalStateException(e);
        }
        if (!link.send(Wire.KIND_EPISODE, body)) pendingEpisodeEvent = json;
    }

    static void log(String message) {
        System.out.println("[rw-agent] " + message);
        System.out.flush();
    }
}
