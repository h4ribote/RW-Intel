import java.lang.instrument.Instrumentation;
import java.util.List;

/**
 * The javaagent that puts a Rusted Warfare process under outside command.
 *
 * It does three things, all of them on the game thread: it builds the observation, it turns arriving decisions into engine commands, and it starts and ends episodes. Everything that touches the network happens on another thread, and the two exchange single element slots (see {@link Link}).
 *
 * Everything must be on the game thread because the engine gives no other safe point: the command pool is an unsynchronised list, loading a map needs the OpenGL context that only that thread holds, and reading unit state from elsewhere samples a step in progress and returns units observed at different instants. The engine drains a queue of tasks in full at the top of every simulation step, and the frame layer's per frame hook decides when a task goes into it.
 *
 * Agent options, comma separated, besides the frame layer's (see {@link Frame}):
 *   host=&lt;name&gt;       control process to connect to, default 127.0.0.1
 *   port=&lt;number&gt;     default 8642
 *   instance=&lt;n&gt;      which instance this is, so the control process can tell them apart
 *   tactical=&lt;ms&gt;     game time between observations, default 200 which is 5Hz
 *   operational=&lt;ms&gt;  game time between region blocks, default 2000
 *   omniscient=&lt;bool&gt; report every enemy rather than only what the engine says is visible, default true
 */
public final class RwAgent {

    private static volatile String host = "127.0.0.1";
    private static volatile int port = 8642;
    private static volatile int instance = 0;
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
    /** Game time the last standing of an ordinary episode was sent at. Touched only on the game thread. */
    private static int lastStandingMs = Integer.MIN_VALUE;
    private static volatile boolean catalogueSent = false;
    /** Set when an episode event could not be sent because the link was down, so that it goes out again once there is a link. */
    private static volatile String pendingEpisodeEvent = null;
    /** The sequence number of the last observation sent in this episode and not yet answered, or {@link Link#NONE}. Touched only on the game thread. */
    private static int unanswered = Link.NONE;
    /** Observations sent and answers taken in the episode running, reported at its end. On the fixed clock every observation but the last is answered. Touched only on the game thread. */
    private static int observationsSent = 0;
    private static int answersTaken = 0;
    /** Whether the region table of the episode running has arrived. The control process sends it in answer to the episode's start, so it can never be in hand in the step that starts the match. Touched only on the game thread. */
    private static boolean regionsArrived = false;
    /** Wall clock the first observation of an episode waits for the region table before going without one. */
    private static final long REGIONS_WAIT_MS = 15000L;
    /** Wall clock between attempts to reach the control process while it has work for this game. */
    private static final long REDIAL_MS = 1000L;
    /** The longest wait between attempts once the control process has closed a link without anything for this game to play. */
    private static final long REDIAL_MAX_MS = 30000L;
    /** Set while the control process has nothing for this game, during which redialling backs off and is not logged. */
    private static volatile boolean quiet = false;

    public static void premain(String arguments, Instrumentation instrumentation) {
        parse(arguments);
        log("starting: host=" + host + " port=" + port + " instance=" + instance + " tactical=" + tacticalMs + "ms");
        if (Frame.fixedClock() && (tacticalMs % Frame.stepMs() != 0 || operationalMs % Frame.stepMs() != 0)) {
            log("warning: the periods are not whole numbers of " + Frame.stepMs() + " ms steps, so each runs up to a step long");
        }
        Frame.install();
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
            if (Frame.option(key, value)) continue;
            if (key.equals("host")) host = value;
            else if (key.equals("port")) port = Integer.parseInt(value);
            else if (key.equals("instance")) instance = Integer.parseInt(value);
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
        // Nothing happens until the control process starts an episode, and a menu running flat out would take a processor from the processes that have one.
        Frame.setIdle(true);

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
        final Object running = game;
        Frame.addListener(new Frame.Listener() {
            public void onFrame(int gameTimeMs) throws Exception {
                schedule(running, gameTimeMs);
            }
        });
        keepConnected();
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

    /** Wall clock time of the last look at whether a match to join has started. Touched only on the game thread. */
    private static long lastJoinPoll = 0;

    /**
     * Decides, on the game thread once every frame, whether there is work, and queues one task when there is.
     *
     * The work runs in the engine's task queue rather than here, because that is where the engine takes outside changes. The engine drains the queue until it is empty, so a task that queued itself again would be drained again in the same frame and the frame would never end; deciding here, once a frame, is what keeps one task at a time.
     * A period starts on the first frame whose game time has reached it, so its boundaries are exact whatever the frame rate.
     */
    private static void schedule(Object game, int now) throws Exception {
        if (!link.connected() || queued.get()) return;
        long wall = System.currentTimeMillis();
        // A process waiting on a host has no episode of its own to pace against, and cannot pace against the game clock either, because the clock it will be running on is the one the host is about to hand it. So it is looked at on the wall clock.
        boolean waiting = (awaitingHost || awaitingPeer) && wall - lastJoinPoll >= 200;
        // During an episode a control frame waits for the next period, so a frame such as an arena's scenario takes effect at a game time that does not depend on when it arrived in real time.
        boolean due = !catalogueSent
                || (link.hasControl() && !episodeRunning)
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

    /**
     * Remakes the link whenever it drops.
     * The episode is left running meanwhile, and the squads keep the contracts they were last given: the engine goes on advancing them, which is a better thing to be doing while out of touch than standing still.
     * A control process that has run everything it wanted from this game closes each new link right after the HELLO. Once a link has closed without a single control or action frame on it, redialling backs off by doubling up to {@link #REDIAL_MAX_MS}, is logged once rather than per attempt, and returns to every second with a log line as soon as a link carries work again.
     */
    private static void keepConnected() throws InterruptedException {
        long delay = REDIAL_MS;
        long nextAttempt = 0;
        while (true) {
            Thread.sleep(REDIAL_MS);
            if (link.connected()) {
                if (quiet && link.heard()) {
                    quiet = false;
                    delay = REDIAL_MS;
                    log("reconnected to the control process, which has work for this game again");
                }
                continue;
            }
            if (!quiet && link.closedIdle()) {
                quiet = true;
                delay = REDIAL_MS;
                log("the control process closed the link with nothing for this game to play; redialling quietly, at most " + REDIAL_MAX_MS / 1000 + "s apart");
            }
            long now = System.currentTimeMillis();
            if (now < nextAttempt) continue;
            if (link.connect()) {
                catalogueSent = false;
                if (!quiet) log("reconnected to the control process");
            }
            if (quiet) delay = Math.min(REDIAL_MAX_MS, delay * 2);
            nextAttempt = now + (quiet ? delay : 0);
        }
    }

    private static void step(Object game) throws Exception {
        if (!link.connected()) return;

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
            if (!driver.beginWhenReady(game)) {
                String failure = driver.waitFailure();
                if (failure != null) {
                    awaitingPeer = false;
                    failEpisode(failure);
                }
                return;
            }
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

        boolean aiOrders = !replaying && driver.settings() != null && driver.settings().aiOrders && !observer.aiOrders.failed;
        if (aiOrders) {
            // Read at the head of the step after the one they were issued in, when the AI that filled them in has returned.
            try {
                observer.aiOrders.drain(game);
                observer.aiOrders.install(game);
            } catch (Exception e) {
                // The orders are an addition to the observation, so a failure to read them drops the block for the rest of the process and the episode goes on without it.
                observer.aiOrders.fail();
                aiOrders = false;
                log("reading the built-in AI's orders failed, so operational observations go without them: " + e);
            }
        }

        int now = engine.gameTime(game);

        // A step that falls between periods, such as the one that sends the HELLO after a reconnect, stops here, so the pending answer is still applied on the period boundary one period after its observation and the boundaries do not move.
        if (lastTacticalMs != Integer.MIN_VALUE && now - lastTacticalMs < tacticalMs) return;

        if (driver.finished(game)) {
            endEpisode(game);
            return;
        }

        // Every episode is first observed the same way: once the match clock has run a full tactical period, so the step that loaded the map and the engine's own start of the match are behind it, and once the region table has arrived.
        if (observationsSent == 0 && !readyToObserve(game, now)) return;

        // The action built from the previous observation is applied first, which is what fixes the decision lag at exactly one period.
        // On the fixed and the replay clock the game thread waits for it: a frame advances the same steps however long it took, so waiting changes nothing the simulation sees, and every period then gets the answer to the one before it however busy the machine is.
        // On the wall clock waiting would turn into one long step, so an answer that has not arrived is simply not applied.
        byte[] action = Frame.steady() && unanswered != Link.NONE ? link.awaitAction(unanswered) : link.takeAction();
        unanswered = Link.NONE;
        // A control frame sent while the previous observation was being decided, such as an arena's scenario, comes before its answer on the link, so it is handled here, ahead of the answer, whether it arrived before this step began or while it waited.
        int episode = driver.episode();
        while ((control = link.takeControl()) != null) handleControl(game, control);
        if (!episodeRunning || driver.episode() != episode) return;
        observer.answered = action != null ? link.takenNumber() : Link.NONE;
        if (action != null) answersTaken++;
        // An empty action is the control process saying it has nothing to change.
        if (action != null && action.length > 0) commander.apply(game, action);
        commander.upkeep(game);

        int blocks = Wire.BLOCK_SQUADS | Wire.BLOCK_UNITS | Wire.BLOCK_LIFTS | Wire.BLOCK_TIMING;
        if (lastOperationalMs == Integer.MIN_VALUE || now - lastOperationalMs >= operationalMs) {
            // Events ride the operational frame because the layer that consumes them, the one that forms and retires squads, runs there. They are accumulated meanwhile rather than dropped.
            // Production menus ride it too: the build order that reads them runs on the operational period.
            blocks |= Wire.BLOCK_REGIONS | Wire.BLOCK_EVENTS | Wire.BLOCK_MENUS;
            if (aiOrders) blocks |= Wire.BLOCK_AI_ORDERS;
            lastOperationalMs = now;
        }
        lastTacticalMs = now;
        if (replaying) {
            sendProgress(game);
        } else {
            int standingMs = driver.settings().standingMs;
            if (standingMs > 0 && (lastStandingMs == Integer.MIN_VALUE || now - lastStandingMs >= standingMs)) {
                sendProgress(game);
                lastStandingMs = now;
            }
        }

        byte[] body = observer.build(game, driver.episode(), blocks);
        if (body == null) return;
        int sent = link.sendObservation(body);
        if (sent == Link.NONE) return;
        unanswered = sent;
        observationsSent++;
        if ((blocks & Wire.BLOCK_EVENTS) != 0) {
            // Cleared only once the frame carrying them has actually gone. An event describes a change, so reporting one twice is reporting a change that did not happen, but dropping one is worse: the layer that forms and retires squads has no other way to learn of it.
            world.events.clear();
        }
        if ((blocks & Wire.BLOCK_AI_ORDERS) != 0) observer.aiOrders.sent();
        // An ended lift is reported in one frame and then forgotten, by the same rule.
        world.endedLifts.clear();
    }

    /**
     * Whether the first observation of the episode running may go now, waiting on the game thread for the region table when the clock is ready and the table is not.
     * Waiting holds the simulation still on every clock, so the first observation is taken at the same game time with or without the wait. A table that does not come within {@link #REGIONS_WAIT_MS} is given up on, and the episode is observed without one.
     */
    private static boolean readyToObserve(Object game, int now) throws Exception {
        if (now < tacticalMs) return false;
        int episode = driver.episode();
        long deadline = System.currentTimeMillis() + REGIONS_WAIT_MS;
        while (!regionsArrived && episodeRunning && driver.episode() == episode && link.connected() && System.currentTimeMillis() < deadline) {
            String control = link.takeControl();
            if (control != null) handleControl(game, control);
            else Thread.sleep(1);
        }
        // A control frame taken while waiting may have ended the episode or begun another, whose own clock has to run its period first.
        if (!episodeRunning || driver.episode() != episode || !link.connected()) return false;
        if (!regionsArrived) {
            log("episode " + driver.episode() + ": no region table within " + REGIONS_WAIT_MS / 1000 + "s, observing without one");
            regionsArrived = true;
        }
        return true;
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

        // What each type can do, read off the engine's sample unit of it wherever the type interface does not say: whether it can attack at all, how fast it moves, what it carries and in how many slots, what it makes, and which types a transport would load. The command layers classify units from these and nothing else, so they travel with the catalogue rather than being worked out again on each side.
        java.util.List<Object> types = world.types();
        StringBuilder catalogue = new StringBuilder("[");
        for (int index = 0; index < types.size(); index++) {
            Object type = types.get(index);
            String reported = engine.typeName(type);
            String lookup = lookupByName.get(reported);
            Boolean hitsAir = engine.typeHitsAir(type);
            Boolean hitsLand = engine.typeHitsLand(type);
            int capacity = engine.typeCapacity(type);
            if (catalogue.length() > 1) catalogue.append(',');
            catalogue.append("{\"name\":\"").append(reported).append('"')
                    .append(",\"lookup\":\"").append(lookup == null ? reported : lookup).append('"')
                    .append(",\"price\":").append(engine.typePrice(type))
                    .append(",\"tech\":").append(engine.typeTech(type))
                    .append(",\"building\":").append(engine.typeIsBuilding(type))
                    .append(",\"builder\":").append(engine.typeIsBuilder(type))
                    .append(",\"extractor\":").append(engine.typeOnResourcePool(type))
                    .append(",\"movement\":\"").append(engine.typeMovement(type)).append('"')
                    .append(",\"canAttack\":").append(engine.typeCanAttack(type))
                    .append(",\"range\":").append(world.rangeOfType(type))
                    .append(",\"speed\":").append(engine.typeSpeed(type))
                    .append(",\"hp\":").append(engine.typeMaxHealth(type))
                    .append(",\"hitsAir\":").append(hitsAir == null ? "true" : hitsAir.toString())
                    .append(",\"hitsLand\":").append(hitsLand == null ? "true" : hitsLand.toString())
                    .append(",\"capacity\":").append(capacity)
                    .append(",\"slots\":").append(engine.typeSlots(type))
                    .append(",\"upgradable\":").append(engine.typeUpgradable(type))
                    .append(",\"menu\":").append(indices(engine.typeMenu(type)));
            if (capacity > 0) {
                java.util.List<Object> carried = new java.util.ArrayList<Object>();
                for (Object passenger : types) {
                    if (engine.typeCarries(type, passenger)) carried.add(passenger);
                }
                catalogue.append(",\"carries\":").append(indices(carried));
            }
            catalogue.append('}');
        }
        catalogue.append(']');

        Wire.Json hello = new Wire.Json();
        hello.put("instance", instance);
        hello.put("build", engine.buildNumber());
        hello.put("tacticalMs", tacticalMs);
        hello.put("operationalMs", operationalMs);
        hello.put("steady", Frame.steady());
        hello.put("omniscient", omniscient);
        hello.put("slots", engine.slotCount());
        // The instance number is the run's name for this process, not its directory, and the replays folder a playback is loaded from is in the directory.
        hello.put("directory", new java.io.File(".").getAbsoluteFile().getParent());
        hello.put("map", driver.map());
        // A reconnection is a fresh HELLO in the middle of whatever was already going on, so it has to say what that was: the control process rebuilds its side from the squad identifiers rather than starting the episode over.
        hello.put("episode", driver.episode());
        hello.put("running", episodeRunning);
        hello.raw("unitTypes", catalogue.toString());
        link.send(Wire.KIND_HELLO, hello.toBytes());
        if (!quiet) log("sent the catalogue: " + world.types().size() + " types");
        if (episodeRunning) sendTerrain(game);
    }

    /** Types as a JSON list of their catalogue indices, leaving out any the catalogue does not hold. */
    private static String indices(java.util.List<Object> types) throws Exception {
        StringBuilder out = new StringBuilder("[");
        for (Object type : types) {
            int index = world.indexOf(engine.typeName(type));
            if (index == 0xFFFF) continue;
            if (out.length() > 1) out.append(',');
            out.append(index);
        }
        return out.append(']').toString();
    }

    private static void handleControl(Object game, String json) throws Exception {
        String command = Wire.field(json, "command");
        if (command == null) return;
        if (command.equals("start")) {
            replaying = false;
            commander.shadow = false;
            observer.viewpoint = -1;
            world.omniscient = omniscient;
            MatchDriver.Settings settings = new MatchDriver.Settings();
            settings.map = text(json, "map", "");
            settings.opponents = Wire.intField(json, "opponents", 1);
            settings.difficulty = Wire.intField(json, "difficulty", 1);
            settings.contestants = Wire.intField(json, "contestants", 0);
            settings.aiOrders = Wire.boolField(json, "aiOrders", false);
            settings.watch = Wire.intField(json, "watch", -1);
            settings.credits = Wire.intField(json, "credits", 0);
            settings.startingUnits = Wire.intField(json, "startingUnits", 1);
            settings.income = Wire.floatField(json, "income", 1.0f);
            settings.fog = Wire.intField(json, "fog", 2);
            settings.seed = Wire.intField(json, "seed", 12345);
            settings.maxSeconds = Wire.intField(json, "maxSeconds", 0);
            settings.standingMs = Wire.intField(json, "standingMs", 0);
            settings.arena = Wire.boolField(json, "arena", false);
            settings.networked = Wire.boolField(json, "host", false);
            settings.networkPort = Wire.intField(json, "port", settings.networkPort);
            settings.peerWaitSeconds = Wire.intField(json, "peerWait", settings.peerWaitSeconds);
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
                    log("hosting on port " + settings.networkPort + ", waiting for another player to join"
                            + (settings.peerWaitSeconds > 0 ? " for up to " + settings.peerWaitSeconds + "s" : ""));
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
            regionsArrived = true;
            log("region table: " + world.regions.size() + " regions");
        } else if (command.equals("abort")) {
            awaitingHost = false;
            if (episodeRunning) endEpisode(game);
        } else if (command.equals("speed")) {
            Frame.setSpeed(Wire.floatField(json, "value", Frame.speed()));
        } else if (command.equals("omniscient")) {
            world.omniscient = Wire.boolField(json, "value", world.omniscient);
        } else if (command.equals("scenario")) {
            buildScenario(game, json);
        } else if (command.equals("replay")) {
            startReplay(game, json);
        }
    }

    /**
     * Plays a recorded match back under observation, as an episode like any other.
     *
     * The observation is taken for the player in the `viewpoint` slot, and the actions that come back are kept as books and never carried out: the recorded commands are the whole of what the match receives, and a single command more would make it a different match. What the playback reports beyond an ordinary episode is where every side stands once per operational period, which the observation of one side cannot say about the others.
     */
    private static void startReplay(Object game, String json) throws Exception {
        String name = text(json, "name", "");
        world.reset();
        world.omniscient = Wire.boolField(json, "omniscient", true);
        observer.viewpoint = Wire.intField(json, "viewpoint", -1);
        commander.shadow = true;
        replaying = true;
        // Anything left over from the episode that just ended names units the replay's world does not have.
        link.takeAction();
        if (!driver.playReplay(game, name, Wire.intField(json, "untilMs", 0), Wire.intField(json, "steps", 1))) {
            replaying = false;
            commander.shadow = false;
            failEpisode("the replay " + name + " did not load");
            return;
        }
        // A playback knows nobody as a computer player: every side is replayed from its commands. So the side to watch is named, or named as the one that is not the team the recording side's own policy played.
        int otherThan = Wire.intField(json, "otherThanTeam", -1);
        if (observer.viewpoint < 0 && otherThan >= 0) observer.viewpoint = driver.onlyOpponentOf(otherThan);
        if (observer.viewpoint < 0) {
            driver.stopReplay(game);
            replaying = false;
            commander.shadow = false;
            failEpisode("no side to observe " + name + " from: give the slot");
            return;
        }
        log("playing " + name + " back from slot " + observer.viewpoint + "'s side");
        driver.countReplay();
        beginEpisode(game);
    }

    /** Set while a replay plays back, for the episode it is. */
    private static volatile boolean replaying = false;

    /** Where every side stands, sent once per tactical period of a playback and once per standing interval of an ordinary episode. Not held back while the link is down, as the frames that count episodes are: the next one says the same thing again with newer figures. */
    private static void sendProgress(Object game) throws Exception {
        Wire.Json event = new Wire.Json();
        event.put("event", "progress");
        event.put("episode", driver.episode());
        event.put("timeMs", engine.gameTime(game));
        event.put("frame", engine.frame(game));
        event.raw("standing", driver.standing(game));
        // The world checksum is taken far less often than a period, so every one the playback takes is seen here, and each can be set beside the one the recording side took at the same frame.
        event.put("checksumFrame", engine.checksumFrame(game));
        event.put("checksum", engine.checksum(game));
        link.send(Wire.KIND_EPISODE, event.toString().getBytes("UTF-8"));
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
        Frame.setIdle(false);
        lastTacticalMs = Integer.MIN_VALUE;
        lastOperationalMs = Integer.MIN_VALUE;
        lastStandingMs = Integer.MIN_VALUE;
        unanswered = Link.NONE;
        observationsSent = 0;
        answersTaken = 0;
        regionsArrived = false;
        observer.aiOrders.reset();
        MatchDriver.Settings settings = driver.settings();
        if (!replaying && settings != null && settings.watch >= 0) {
            // Watching an AI contestant is observed like a playback: from that player's side, with the answers kept as books.
            observer.viewpoint = driver.aiSlot(settings.watch);
            commander.shadow = observer.viewpoint >= 0;
            log("episode " + driver.episode() + ": watching AI player " + settings.watch + " in slot " + observer.viewpoint);
        }

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
        if (replaying) {
            event.put("replay", driver.replay());
            event.put("viewpoint", observer.viewpoint);
        }
        sendEpisodeEvent(event.toString());
        sendTerrain(game);
    }

    /** What each movement type can cross on the map now loaded, which both halves decide reachability from: the world keeps it for checking contracts and lifts, and the control process is sent it. Sent after the episode's start and again after a HELLO in the middle of one, so a control process never decides on a map it was not told about. */
    private static void sendTerrain(Object game) throws Exception {
        Passage.Grids grids = Passage.read(engine, game);
        world.passage = grids;
        byte[] body = Passage.encode(grids);
        link.send(Wire.KIND_TERRAIN, body);
        log("sent the terrain: " + body.length + " bytes");
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

    /** Tells the control process that the episode it asked for could not be played, and why. No episode is counted, since none began. */
    private static void failEpisode(String reason) {
        Wire.Json event = new Wire.Json();
        event.put("event", "failed");
        event.put("episode", driver.episode() + 1);
        event.put("reason", reason);
        sendEpisodeEvent(event.toString());
        log("episode " + (driver.episode() + 1) + " could not be played: " + reason);
    }

    private static void endEpisode(Object game) throws Exception {
        episodeRunning = false;
        Frame.setIdle(true);
        unanswered = Link.NONE;
        java.util.Set<Integer> alive = driver.aliveTeams();
        int seconds = engine.gameTime(game) / 1000;

        Wire.Json event = new Wire.Json();
        event.put("event", "finished");
        event.put("episode", driver.episode());
        event.put("seconds", seconds);
        event.put("frames", engine.frame(game));
        event.put("winner", alive.size() == 1 ? alive.iterator().next().intValue() : -1);
        event.put("aliveTeams", alive.size());
        Object self = observer.self(game);
        event.put("team", self == null ? -1 : engine.team(self));
        event.put("timeout", driver.settings().maxSeconds > 0 && seconds >= driver.settings().maxSeconds);
        event.put("peerLeft", driver.peerLeft(game));
        event.raw("standing", driver.standing(game));
        // An episode that fell out of step half way through is not one whose result may be used, so the verdict travels with the result rather than having to be asked for afterwards, by which time the next episode has already reset it.
        event.raw("sync", driver.synchronisation(game));
        if (replaying) {
            // A playback that disagreed with a recorded checksum is no longer the recorded match, and nothing observed after that point describes it.
            event.put("replay", driver.replay());
            event.put("mismatches", engine.replayMismatches(game));
            event.put("ended", driver.replayFinished(game));
            event.put("exhausted", driver.replayExhausted(game));
            driver.stopReplay(game);
        } else {
            event.put("replay", driver.closeRecording(game));
        }
        sendEpisodeEvent(event.toString());
        log("episode " + driver.episode() + " finished at " + seconds + "s, alive teams " + alive
                + ", " + answersTaken + " answers to " + observationsSent + " observations");
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
