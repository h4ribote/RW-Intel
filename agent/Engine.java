import java.lang.reflect.Field;
import java.lang.reflect.Method;

/**
 * Every piece of reflection the agent needs, resolved once and held here.
 *
 * The game's class and field names are obfuscated, so each name below is a finding from disassembly and runtime checking rather than something readable; `docs/game` records which is which and how each was confirmed. Resolving them all at startup means a game update breaks the agent loudly at that point instead of somewhere in the middle of an episode.
 *
 * Nothing in this class enforces which thread it is called from, but almost all of it must run on the game thread: the command pool is an unsynchronised list, loading a map needs the OpenGL context that only that thread holds, and reading unit state from anywhere else samples a simulation step in progress and yields units observed at different instants.
 */
final class Engine {

    // ---- classes -------------------------------------------------------------------------

    final Class<?> engineClass;          // gameFramework.l, the engine base
    final Class<?> unitClass;            // game.units.am, the unit base
    final Class<?> armedClass;           // game.units.y, units that can hold orders
    final Class<?> objectClass;          // gameFramework.w, the game object base
    final Class<?> playerClass;          // game.n
    final Class<?> typeInterface;        // game.units.as, a unit type
    final Class<?> typeRegistry;         // game.units.ar, the built-in types and the lookup
    final Class<?> stanceClass;          // game.units.a, the engagement stances
    final Class<?> actionClass;          // game.units.a.c, a special action handle
    final Class<?> loadModeClass;        // gameFramework.s
    final Class<?> mapKindClass;         // gameFramework.j.ai
    final Class<?> orderClass;           // game.units.au, one entry of a unit's order queue
    final Class<?> orderKindClass;       // game.units.av, what kind of order that is
    /** game.units.custom.l, the type implementation a definition file produces. Null if the game ever stops having one. */
    final Class<?> definedTypeClass;
    /** The unobfuscated condition object a definition file's boolean keys are compiled into. */
    final Class<?> logicBooleanClass;

    // ---- engine fields -------------------------------------------------------------------

    private final Method engineSingleton;
    private final Field frameCounter;    // l.bx
    private final Field gameTimeMs;      // l.by
    private final Field commandPool;     // l.cf
    private final Field netEngine;       // l.bX
    private final Field localPlayer;     // l.bs
    private final Field victory;         // l.dq
    private final Field defeat;          // l.dt

    // ---- unit fields ---------------------------------------------------------------------

    private final Field unitCollection;  // am.bE, every unit in the world
    private final Field unitOwner;       // am.bX
    private final Field unitDead;        // am.bV
    private final Field unitBuilt;       // am.cm, build progress from 0 to 1
    private final Field unitType;        // am.dz
    private final Field unitHealth;      // am.cu
    private final Field unitMaxHealth;   // am.cv
    private final Field unitLastHit;     // am.bs, game time of the last hit taken
    private final Method sampleOfType;   // static am.a(as), the engine's own sample unit of a type
    private final Method armedRange;     // y.m(), a unit's maximum attack range
    private final Field objectId;        // w.eh
    private final Field objectX;         // w.eo
    private final Field objectY;         // w.ep
    private final Method unitPrice;      // am.cL()
    private final Method unitVisibleTo;  // am.d(n)
    private final Field unitStance;      // y.P
    private final Method unitTarget;     // y.ab()
    private final Method unitOrderCount; // y.av()
    private final Method unitOrder;      // y.ar(), the order being carried out
    private final Field orderKind;       // au.a, the av constant saying which kind it is
    private final Object[] orderKinds;   // av.values(), so a kind can be reported as its ordinal
    private final Field unitCarrier;     // am.cN, the transport a unit is aboard, null when it is aboard nothing
    private final Method transportLoad;  // am.bY(), how many slots a transport has filled

    // ---- type internals ------------------------------------------------------------------

    /** custom.l.cL, the stat block holding the numbers the type interface does not expose. */
    private final Field definedTypeStats;
    /** custom.as.i, maximum attack range in world units. */
    private final Field statsRange;
    /** custom.l.eq and custom.l.er, the canAttackFlyingUnits and canAttackLandUnits conditions. */
    private final Field definedTypeHitsAir;
    private final Field definedTypeHitsLand;
    /** custom.l.aJ, the definition's placeOnlyOnResPool, which is what makes a building an extractor rather than a building that happens to stand near a pool. */
    private final Field definedTypeOnResourcePool;
    private final Method logicIsStaticTrue;
    private final Method logicIsStaticFalse;
    private final Method logicRead;

    // ---- player fields -------------------------------------------------------------------

    private final Field playerCredits;   // n.o
    private final Field playerSlot;      // n.k
    private final Field playerTeam;      // n.r
    private final Field playerName;      // n.v
    private final Field playerIsAi;      // n.w
    private final Field playerAiLevel;   // n.x
    private final Field playerDefeated;  // n.F
    private final Field playerWiped;     // n.G
    private final Field playerSurrender; // n.E
    private final Field playerAggregate; // n.T
    private final Method playerIncome;   // n.v()
    private final Method playerBySlot;   // n.k(int)
    private final Field playerSlotCount; // n.c
    private final Method playerReset;    // n.F(), the static slot reset

    /** game.a.a, the class the room creates a computer player as. The thinking lives here and not on the ordinary player, which is why an ordinary player cannot be turned into one. */
    private final Class<?> aiPlayerClass;
    /** a.a.aX, read at the top of the computer player's own update and returning immediately when it is set. */
    private final Field aiHalted;

    // ---- type methods --------------------------------------------------------------------

    private final Method typeName;       // as.v()
    private final Method typeDisplay;    // as.e()
    private final Method typePrice;      // as.c()
    private final Method typeTech;       // as.g()
    private final Method typeBuilding;   // as.j()
    private final Method typeBuilder;    // as.l()
    private final Method typeBuildSpeed; // as.D()
    private final Method typeMovement;   // as.o()
    private final Method typeLookup;     // ar.a(String)
    private final Field typeAll;         // ar.ae, every registered type

    // ---- commands ------------------------------------------------------------------------

    private final Method poolObtain;     // c.b(n), which queues the command as it hands it over
    private final Method poolTake;       // c.b(), which does not queue, and is how a system command is taken
    private final Method netSubmit;      // ad.a(e), the explicit submission a system command needs
    private final Method commandAddUnit; // e.a(am)
    private final Method commandMove;    // e.a(float, float)
    private final Method commandAttackMove; // e.b(float, float)
    private final Method commandAttack;  // e.a(am)  -- resolved separately, same erasure as addUnit
    private final Method commandBuild;   // e.a(float, float, as, int)
    private final Method commandStance;  // e.a(a)
    private final Method commandAction;  // e.a(a.c)
    private final Method commandGuard;   // e.c(am)
    private final Method commandLoadInto; // e.e(am), the passenger boards the transport
    private final Method commandLoadUp;  // e.f(am), the transport picks the passenger up
    private final Field commandAppend;   // e.e, keep the orders already queued and add this one after them
    private final Method actionHandle;   // a.c.a(String)
    private final Field poolQueue;       // c.b, the commands queued since the pool last carried them out
    private final Field commandIssuer;   // e.i, the player the command is issued as
    private final Field commandTime;     // e.d, the game time it was issued at
    private final Field commandUndo;     // e.g, stopOrUndo
    private final Field commandStop;     // e.o, set by e.h()
    private final Field commandOrder;    // e.j, the order it gives, or null
    private final Field commandSpecial;  // e.k, the special action it names
    private final Field commandUnits;    // e.v, the units it addresses
    private final Field orderX;          // au.e
    private final Field orderY;          // au.f
    private final Field orderTarget;     // au.h, the unit the order is aimed at, or null

    // ---- what a unit offers ----------------------------------------------------------------

    private final Method unitLevel;      // am.V(), the tier a building has been raised to
    private final Method unitActions;    // am.N(), the actions the unit offers at its current tier
    private final Method actionHandleOf; // a.s.N(), the handle a command names the action by
    private final Method offeredPrice;   // a.s.c()
    private final Method actionKind;     // a.s.f(), an a.t
    private final Method offeredType;    // a.s.i(), the type an action produces or places
    private final Object upgradeKind;    // a.t.c, the kind every tier raise reports
    private final Object queueKind;      // a.t.d (queueUnit), the kind a factory's production reports
    private final Object placeKind;      // a.t.e (building), the kind a builder's placement reports

    private final Object[] stances;

    // ---- the network session ---------------------------------------------------------------

    /**
     * Everything a real multiplayer session needs, and everything the engine will say about whether one has drifted apart.
     *
     * These are resolved leniently rather than in the strict style of the rest of this class, and every accessor over them answers something harmless when its member is missing. The reason is what they are for: a synchronisation report exists to tell a run that an episode's result cannot be trusted, and it must not be the thing that stops the run because one obfuscated name moved.
     */
    private final Field settingsEngine;      // l.bQ, the settings object, whose own class name is not obfuscated
    private final Field settingsPort;        // SettingsEngine.networkPort, which is where a host reads the port it binds
    private final Method netHost;            // ad.b(boolean), the multiplayer host call
    private final Method netConnect;         // ad.a(String, boolean, Runnable), which starts a connector and hands it back
    private final Method netAdopt;           // ad.a(Socket), which takes a connected socket over as this process's session
    private final Method netName;            // ad.a(String), which is how the name this process joins under is set
    private final Field netStarted;          // ad.B, a session of either kind is up
    private final Field netHosting;          // ad.C, and this process is its host
    private final Field netChecksumFrame;    // ad.ah, the frame the current checksum belongs to, -1 for none
    private final Field netChecksumInterval; // ad.ai, 300 frames
    private final Field netChecksum;         // ad.am, the checksum record
    private final Field netMatches;          // ad.aq, checksums this process has agreed with as a client
    private final Field netConnections;      // ad.aM, the host's connection to each client
    private final Method netAssert;          // ad.x()
    private final Field checksumTotal;       // ak.a, the aggregate over every subsystem
    private final Field connectorError;      // an.e, non-null once a connection attempt has failed
    private final Field connectorSocket;     // an.g, the connected socket once one has succeeded
    private final Method connectorCancel;    // an.a()
    private final Field peerDesynced;        // c.v, this client is out of step right now
    private final Field peerBroken;          // c.w, and could not be brought back
    private final Field peerMatches;         // c.x, checksums it has agreed with
    private final Field peerDesyncs;         // c.y, times it has fallen out of step
    private final Method peerName;           // c.e()

    /**
     * How long a join waits for the connector before giving up.
     *
     * The wait happens on the game thread, which stalls the simulation for as long as it lasts, so this is a bound on how long a mistyped address costs rather than a generous allowance. A host on the same machine answers in milliseconds, and one that is not listening refuses immediately.
     */
    private static final int JOIN_TIMEOUT_SECONDS = 90;

    /** How long one attempt to reach a host is given before it is abandoned and tried again. The engine's own connect gives up after seven seconds, so this only has to be longer than that. */
    private static final int CONNECT_TIMEOUT_SECONDS = 10;

    Engine() throws Exception {
        engineClass = Class.forName("com.corrodinggames.rts.gameFramework.l");
        unitClass = Class.forName("com.corrodinggames.rts.game.units.am");
        armedClass = Class.forName("com.corrodinggames.rts.game.units.y");
        objectClass = Class.forName("com.corrodinggames.rts.gameFramework.w");
        playerClass = Class.forName("com.corrodinggames.rts.game.n");
        aiPlayerClass = Class.forName("com.corrodinggames.rts.game.a.a");
        typeInterface = Class.forName("com.corrodinggames.rts.game.units.as");
        typeRegistry = Class.forName("com.corrodinggames.rts.game.units.ar");
        stanceClass = Class.forName("com.corrodinggames.rts.game.units.a");
        actionClass = Class.forName("com.corrodinggames.rts.game.units.a.c");
        loadModeClass = Class.forName("com.corrodinggames.rts.gameFramework.s");
        mapKindClass = Class.forName("com.corrodinggames.rts.gameFramework.j.ai");
        orderClass = Class.forName("com.corrodinggames.rts.game.units.au");
        orderKindClass = Class.forName("com.corrodinggames.rts.game.units.av");
        definedTypeClass = Class.forName("com.corrodinggames.rts.game.units.custom.l");
        logicBooleanClass = Class.forName(
                "com.corrodinggames.rts.game.units.custom.logicBooleans.LogicBoolean");

        aiHalted = field(aiPlayerClass, "aX");

        engineSingleton = engineClass.getMethod("B");
        frameCounter = field(engineClass, "bx");
        gameTimeMs = field(engineClass, "by");
        commandPool = field(engineClass, "cf");
        netEngine = field(engineClass, "bX");
        localPlayer = field(engineClass, "bs");
        victory = field(engineClass, "dq");
        defeat = field(engineClass, "dt");

        unitCollection = field(unitClass, "bE");
        unitOwner = field(unitClass, "bX");
        unitDead = field(unitClass, "bV");
        unitBuilt = field(unitClass, "cm");
        unitType = field(unitClass, "dz");
        unitHealth = field(unitClass, "cu");
        unitMaxHealth = field(unitClass, "cv");
        unitLastHit = field(unitClass, "bs");
        sampleOfType = method(unitClass, "a", typeInterface);
        armedRange = method(armedClass, "m");
        objectId = field(objectClass, "eh");
        objectX = field(objectClass, "eo");
        objectY = field(objectClass, "ep");
        unitPrice = method(unitClass, "cL");
        unitVisibleTo = method(unitClass, "d", playerClass);
        unitStance = field(armedClass, "P");
        unitTarget = method(armedClass, "ab");
        unitOrderCount = method(armedClass, "av");
        unitOrder = method(armedClass, "ar");
        orderKind = fieldOfType(orderClass, orderKindClass);
        orderKinds = (Object[]) orderKindClass.getMethod("values").invoke(null);
        unitCarrier = field(unitClass, "cN");
        transportLoad = method(unitClass, "bY");

        Class<?> statsClass = Class.forName("com.corrodinggames.rts.game.units.custom.as");
        definedTypeStats = field(definedTypeClass, "cL");
        statsRange = field(statsClass, "i");
        definedTypeHitsAir = field(definedTypeClass, "eq");
        definedTypeHitsLand = field(definedTypeClass, "er");
        definedTypeOnResourcePool = field(definedTypeClass, "aJ");
        logicIsStaticTrue = method(logicBooleanClass, "isStaticTrue", logicBooleanClass);
        logicIsStaticFalse = method(logicBooleanClass, "isStaticFalse", logicBooleanClass);
        logicRead = method(logicBooleanClass, "read", armedClass);

        playerCredits = field(playerClass, "o");
        playerSlot = field(playerClass, "k");
        playerTeam = field(playerClass, "r");
        playerName = field(playerClass, "v");
        playerIsAi = field(playerClass, "w");
        playerAiLevel = field(playerClass, "x");
        playerDefeated = field(playerClass, "F");
        playerWiped = field(playerClass, "G");
        playerSurrender = field(playerClass, "E");
        playerAggregate = field(playerClass, "T");
        playerIncome = method(playerClass, "v");
        playerBySlot = method(playerClass, "k", int.class);
        playerSlotCount = field(playerClass, "c");
        playerReset = method(playerClass, "F");

        typeName = typeInterface.getMethod("v");
        typeDisplay = typeInterface.getMethod("e");
        typePrice = typeInterface.getMethod("c");
        typeTech = typeInterface.getMethod("g");
        typeBuilding = typeInterface.getMethod("j");
        typeBuilder = typeInterface.getMethod("l");
        typeBuildSpeed = typeInterface.getMethod("D");
        typeMovement = typeInterface.getMethod("o");
        typeLookup = method(typeRegistry, "a", String.class);
        typeAll = field(typeRegistry, "ae");

        Class<?> poolClass = Class.forName("com.corrodinggames.rts.gameFramework.c");
        Class<?> commandClass = Class.forName("com.corrodinggames.rts.gameFramework.e");
        Class<?> netClass = Class.forName("com.corrodinggames.rts.gameFramework.j.ad");
        poolObtain = method(poolClass, "b", playerClass);
        poolTake = method(poolClass, "b");
        netSubmit = method(netClass, "a", commandClass);
        commandAddUnit = method(commandClass, "a", armedClass);
        commandAttack = method(commandClass, "a", unitClass);
        commandMove = method(commandClass, "a", float.class, float.class);
        commandAttackMove = method(commandClass, "b", float.class, float.class);
        commandBuild = method(commandClass, "a", float.class, float.class, typeInterface, int.class);
        commandStance = method(commandClass, "a", stanceClass);
        commandAction = method(commandClass, "a", actionClass);
        commandGuard = method(commandClass, "c", unitClass);
        commandLoadInto = method(commandClass, "e", unitClass);
        commandLoadUp = method(commandClass, "f", unitClass);
        commandAppend = field(commandClass, "e");
        actionHandle = method(actionClass, "a", String.class);
        poolQueue = field(poolClass, "b");
        commandIssuer = field(commandClass, "i");
        commandTime = field(commandClass, "d");
        commandUndo = field(commandClass, "g");
        commandStop = field(commandClass, "o");
        commandOrder = field(commandClass, "j");
        commandSpecial = field(commandClass, "k");
        commandUnits = field(commandClass, "v");
        orderX = field(orderClass, "e");
        orderY = field(orderClass, "f");
        orderTarget = field(orderClass, "h");

        Class<?> offeredClass = Class.forName("com.corrodinggames.rts.game.units.a.s");
        Class<?> offeredKindClass = Class.forName("com.corrodinggames.rts.game.units.a.t");
        unitLevel = method(unitClass, "V");
        unitActions = method(unitClass, "N");
        actionHandleOf = method(offeredClass, "N");
        offeredPrice = method(offeredClass, "c");
        actionKind = method(offeredClass, "f");
        offeredType = method(offeredClass, "i");
        upgradeKind = field(offeredKindClass, "c").get(null);
        queueKind = field(offeredKindClass, "d").get(null);
        placeKind = field(offeredKindClass, "e").get(null);

        stances = (Object[]) stanceClass.getMethod("values").invoke(null);

        Class<?> settingsClass = classOrNull("com.corrodinggames.rts.gameFramework.SettingsEngine");
        Class<?> checksumClass = classOrNull("com.corrodinggames.rts.gameFramework.j.ak");
        Class<?> connectorClass = classOrNull("com.corrodinggames.rts.gameFramework.j.an");
        Class<?> peerClass = classOrNull("com.corrodinggames.rts.gameFramework.j.c");
        settingsEngine = fieldOrNull(engineClass, "bQ");
        settingsPort = fieldOrNull(settingsClass, "networkPort");
        netHost = methodOrNull(netClass, "b", boolean.class);
        netConnect = methodOrNull(netClass, "a", String.class, boolean.class, Runnable.class);
        netAdopt = methodOrNull(netClass, "a", java.net.Socket.class);
        netName = methodOrNull(netClass, "a", String.class);
        netStarted = fieldOrNull(netClass, "B");
        netHosting = fieldOrNull(netClass, "C");
        netChecksumFrame = fieldOrNull(netClass, "ah");
        netChecksumInterval = fieldOrNull(netClass, "ai");
        netChecksum = fieldOrNull(netClass, "am");
        netMatches = fieldOrNull(netClass, "aq");
        netConnections = fieldOrNull(netClass, "aM");
        netAssert = methodOrNull(netClass, "x");
        checksumTotal = fieldOrNull(checksumClass, "a");
        connectorError = fieldOrNull(connectorClass, "e");
        connectorSocket = fieldOrNull(connectorClass, "g");
        connectorCancel = methodOrNull(connectorClass, "a");
        peerDesynced = fieldOrNull(peerClass, "v");
        peerBroken = fieldOrNull(peerClass, "w");
        peerMatches = fieldOrNull(peerClass, "x");
        peerDesyncs = fieldOrNull(peerClass, "y");
        peerName = methodOrNull(peerClass, "e");
    }

    // ---- reflection helpers --------------------------------------------------------------

    private static Field field(Class<?> owner, String name) throws Exception {
        for (Class<?> c = owner; c != null; c = c.getSuperclass()) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                return f;
            } catch (NoSuchFieldException ignored) {
                // keep walking up
            }
        }
        throw new NoSuchFieldException(name + " on " + owner.getName());
    }

    /**
     * The one field of a class that holds a given type.
     * Used where the obfuscated name would be a guess but the type makes the field unambiguous, which is the case for an order's kind: only one field of an order is an order kind.
     */
    private static Field fieldOfType(Class<?> owner, Class<?> type) throws Exception {
        Field found = null;
        for (Class<?> c = owner; c != null; c = c.getSuperclass()) {
            for (Field candidate : c.getDeclaredFields()) {
                if (candidate.getType() != type) continue;
                if (found != null) throw new NoSuchFieldException(
                        "more than one field of " + type.getName() + " on " + owner.getName());
                candidate.setAccessible(true);
                found = candidate;
            }
        }
        if (found == null) throw new NoSuchFieldException("no field of " + type.getName() + " on " + owner.getName());
        return found;
    }

    /**
     * The lenient forms, for members whose absence must be survivable.
     *
     * Everything resolved strictly above is something without which an episode cannot be run at all, so failing at start up is the right answer for it. The network session members are not like that: a run that only ever hosts single player never touches one, and even a paired run would rather report that it cannot see the checksums than refuse to start.
     */
    private static Class<?> classOrNull(String name) {
        try {
            return Class.forName(name);
        } catch (Throwable e) {
            return null;
        }
    }

    private static Field fieldOrNull(Class<?> owner, String name) {
        try {
            return owner == null ? null : field(owner, name);
        } catch (Throwable e) {
            // Throwable rather than Exception because enumerating a class's declared members loads the types in their signatures, and this game's classes declare methods over Android types that are not on the class path of a desktop run. What comes back is an Error, and an Error escaping here would stop the agent from starting at all rather than leaving one accessor unresolved.
            return null;
        }
    }

    private static Method methodOrNull(Class<?> owner, String name, Class<?>... parameters) {
        try {
            return owner == null ? null : method(owner, name, parameters);
        } catch (Throwable e) {
            return null;
        }
    }

    private static Method method(Class<?> owner, String name, Class<?>... parameters) throws Exception {
        for (Class<?> c = owner; c != null; c = c.getSuperclass()) {
            try {
                Method m = c.getDeclaredMethod(name, parameters);
                m.setAccessible(true);
                return m;
            } catch (NoSuchMethodException ignored) {
                // keep walking up
            }
        }
        throw new NoSuchMethodException(name + " on " + owner.getName());
    }

    // ---- engine --------------------------------------------------------------------------

    Object engine() throws Exception {
        return engineSingleton.invoke(null);
    }

    int frame(Object engine) throws Exception {
        return frameCounter.getInt(engine);
    }

    int gameTime(Object engine) throws Exception {
        return gameTimeMs.getInt(engine);
    }

    Object net(Object engine) throws Exception {
        return netEngine.get(engine);
    }

    Object local(Object engine) throws Exception {
        return localPlayer.get(engine);
    }

    boolean victory(Object engine) throws Exception {
        return victory.getBoolean(engine);
    }

    boolean defeat(Object engine) throws Exception {
        return defeat.getBoolean(engine);
    }

    /**
     * The engine's sandbox flag, which makes every player's units answerable to this process and forces the fog off.
     * A constructed engagement needs both sides driven from one process, and this is what allows it without a network session.
     */
    boolean sandbox(Object engine) throws Exception {
        return field(engineClass, "bv").getBoolean(engine);
    }

    void setSandbox(Object engine, boolean on) throws Exception {
        field(engineClass, "bv").setBoolean(engine, on);
    }

    // ---- replays --------------------------------------------------------------------------

    /** l.cb, the replay player and recorder (gameFramework.ba). */
    private Object replays(Object engine) throws Exception {
        return field(engineClass, "cb").get(engine);
    }

    /** Loads a replay by file name, resolved against the replays folder of the working directory, and says whether it loaded. The simulation does not step it until the menu is closed. */
    boolean loadReplay(Object engine, String name) throws Exception {
        Object replays = replays(engine);
        return Boolean.TRUE.equals(method(replays.getClass(), "c", String.class).invoke(replays, name));
    }

    /**
     * Closes the menu the way the menu's own replay load does once it has loaded one: the active document, the document history, and the interface itself.
     * The pause test counts an open menu document as a pause, so a replay loaded with the menu still open stays at frame 0.
     */
    void closeMenu() throws Exception {
        Class<?> gui = Class.forName("com.corrodinggames.librocket.a");
        Object guiEngine = gui.getMethod("a").invoke(null);
        Object documents = gui.getField("b").get(guiEngine);
        documents.getClass().getMethod("closeActiveDocument").invoke(documents);
        documents.getClass().getMethod("clearHistory").invoke(documents);
        gui.getMethod("a", boolean.class).invoke(guiEngine, Boolean.FALSE);
    }

    /** ba.i(): a replay is being played back or a match recorded. After a replay has been loaded it says whether the playback is still going. */
    boolean replayActive(Object engine) throws Exception {
        Object replays = replays(engine);
        return Boolean.TRUE.equals(method(replays.getClass(), "i").invoke(replays));
    }

    /** ba.k(): the match is being recorded. */
    boolean replayRecording(Object engine) throws Exception {
        Object replays = replays(engine);
        return Boolean.TRUE.equals(method(replays.getClass(), "k").invoke(replays));
    }

    /** ba.s: playback has read the block that closes a recording. */
    boolean replayEnded(Object engine) throws Exception {
        Object replays = replays(engine);
        return field(replays.getClass(), "s").getBoolean(replays);
    }

    /** ba.t: the file name a recording is being written to, or null when nothing is being recorded. */
    String replayFile(Object engine) throws Exception {
        Object replays = replays(engine);
        Object name = field(replays.getClass(), "t").get(replays);
        return name == null ? null : String.valueOf(name);
    }

    /** ba.l: how many of the recorded checksums the playback has disagreed with. */
    int replayMismatches(Object engine) throws Exception {
        Object replays = replays(engine);
        return field(replays.getClass(), "l").getInt(replays);
    }

    /** ba.v: world steps the playback runs for each step's worth of elapsed time. More steps a frame, not longer steps, so the match played is the same one. */
    void setReplaySteps(Object engine, int steps) throws Exception {
        Object replays = replays(engine);
        field(replays.getClass(), "v").setInt(replays, steps);
    }

    /** ba.e(): stops recording or playing back, flushing and closing the file. */
    void stopReplay(Object engine) throws Exception {
        Object replays = replays(engine);
        method(replays.getClass(), "e").invoke(replays);
    }

    /**
     * Queues a task for the game thread.
     * The engine drains this queue in full at the top of every simulation step, before any unit is updated, which is the only point at which the world can be touched safely from outside.
     */
    void post(Object engine, Runnable task) throws Exception {
        Field queueField = engine.getClass().getField("k");
        queueField.setAccessible(true);
        Object queue = queueField.get(engine);
        queue.getClass().getMethod("add", Object.class).invoke(queue, task);
    }

    // ---- units ---------------------------------------------------------------------------

    /** The backing array of every unit in the world. Entries past `unitCount` are stale and must not be read. */
    Object[] unitArray() throws Exception {
        Object collection = unitCollection.get(null);
        return (Object[]) collection.getClass().getMethod("a").invoke(collection);
    }

    int unitCount() throws Exception {
        Object collection = unitCollection.get(null);
        return ((Integer) collection.getClass().getMethod("size").invoke(collection)).intValue();
    }

    Object owner(Object unit) throws Exception { return unitOwner.get(unit); }
    boolean dead(Object unit) throws Exception { return unitDead.getBoolean(unit); }
    float built(Object unit) throws Exception { return unitBuilt.getFloat(unit); }
    Object type(Object unit) throws Exception { return unitType.get(unit); }
    float health(Object unit) throws Exception { return unitHealth.getFloat(unit); }
    float maxHealth(Object unit) throws Exception { return unitMaxHealth.getFloat(unit); }
    int lastHit(Object unit) throws Exception { return unitLastHit.getInt(unit); }
    long id(Object unit) throws Exception { return objectId.getLong(unit); }
    float x(Object unit) throws Exception { return objectX.getFloat(unit); }
    float y(Object unit) throws Exception { return objectY.getFloat(unit); }
    int price(Object unit) throws Exception { return ((Integer) unitPrice.invoke(unit)).intValue(); }

    /** The transport the unit is aboard, or null. A unit aboard stays in the unit list at its transport's position and holds no order of its own. */
    Object carrier(Object unit) throws Exception { return unitCarrier.get(unit); }

    /** How many slots of a transport are filled; nought for a unit that carries nothing. */
    int aboard(Object transport) {
        try {
            Object value = transportLoad.invoke(transport);
            return value instanceof Integer ? Math.max(0, ((Integer) value).intValue()) : 0;
        } catch (Exception e) {
            return 0;
        }
    }

    boolean visibleTo(Object unit, Object player) throws Exception {
        return ((Boolean) unitVisibleTo.invoke(unit, player)).booleanValue();
    }

    int stanceOf(Object unit) throws Exception {
        Object value = unitStance.get(unit);
        for (int i = 0; i < stances.length; i++) if (stances[i] == value) return i;
        return 0;
    }

    Object attackTarget(Object unit) throws Exception {
        return unitTarget.invoke(unit);
    }

    int orderCount(Object unit) throws Exception {
        Object value = unitOrderCount.invoke(unit);
        return value instanceof Integer ? ((Integer) value).intValue() : 0;
    }

    /** The kind of the order a unit is carrying out, as its position in the engine's own enumeration, or {@link #NO_ORDER} when it has none. */
    static final int NO_ORDER = 255;

    int orderKind(Object unit) throws Exception {
        Object order = unitOrder.invoke(unit);
        if (order == null) return NO_ORDER;
        Object kind = orderKind.get(order);
        for (int i = 0; i < orderKinds.length; i++) if (orderKinds[i] == kind) return i;
        return NO_ORDER;
    }

    // ---- players -------------------------------------------------------------------------

    int slotCount() throws Exception {
        return playerSlotCount.getInt(null);
    }

    Object playerAt(int slot) throws Exception {
        return playerBySlot.invoke(null, Integer.valueOf(slot));
    }

    void resetPlayers() throws Exception {
        playerReset.invoke(null);
    }

    double credits(Object player) throws Exception { return playerCredits.getDouble(player); }
    int slot(Object player) throws Exception { return playerSlot.getInt(player); }
    int team(Object player) throws Exception { return playerTeam.getInt(player); }
    void setTeam(Object player, int team) throws Exception { playerTeam.setInt(player, team); }

    /**
     * Stops a computer player from thinking, and says whether there was one to stop.
     *
     * This is the one place anything outside the command route is written, and it is written to the deciding half of a player rather than to the simulation: the flag is read at the top of the computer player's own update and makes it return without doing anything, so no unit, no health and no credit is touched by it. Nothing it would have decided is lost either, because a player that never decides never issues a command.
     *
     * It is safe only where there is no second process to disagree with. A lockstep peer runs the same computer players over the same frames and reaches the same commands, so silencing one on one side and not the other is exactly the divergence the whole design is arranged to avoid. The caller is therefore expected to use this only on a board it alone is simulating.
     */
    boolean haltAi(Object player) throws Exception {
        if (player == null || !aiPlayerClass.isInstance(player)) return false;
        aiHalted.setBoolean(player, true);
        return true;
    }
    String name(Object player) throws Exception { return (String) playerName.get(player); }
    boolean isAi(Object player) throws Exception { return playerIsAi.getBoolean(player); }
    int aiLevel(Object player) throws Exception { return playerAiLevel.getInt(player); }
    boolean defeated(Object player) throws Exception { return playerDefeated.getBoolean(player); }
    boolean wiped(Object player) throws Exception { return playerWiped.getBoolean(player); }
    boolean surrendered(Object player) throws Exception { return playerSurrender.getBoolean(player); }

    int income(Object player) throws Exception {
        return ((Integer) playerIncome.invoke(player)).intValue();
    }

    /** The per frame aggregate the engine keeps for each player, read for the fields that cost a full scan otherwise. */
    Object aggregate(Object player) throws Exception {
        return playerAggregate.get(player);
    }

    int aggregateInt(Object player, String name, int fallback) {
        try {
            Object stats = aggregate(player);
            if (stats == null) return fallback;
            return field(stats.getClass(), name).getInt(stats);
        } catch (Exception e) {
            return fallback;
        }
    }

    /** Cumulative kills and losses, which the engine keeps per player and which the reward uses directly. */
    Object record(Object engine, Object player) throws Exception {
        Object holder = field(engineClass, "bY").get(engine);
        return method(holder.getClass(), "a", playerClass).invoke(holder, player);
    }

    int recordInt(Object record, String name) {
        try {
            if (record == null) return 0;
            return field(record.getClass(), name).getInt(record);
        } catch (Exception e) {
            return 0;
        }
    }

    // ---- unit types ----------------------------------------------------------------------

    java.util.List<Object> allTypes() throws Exception {
        java.util.List<Object> out = new java.util.ArrayList<Object>();
        Object registry = typeAll.get(null);
        if (registry instanceof java.util.Collection) {
            for (Object entry : (java.util.Collection<?>) registry) {
                if (typeInterface.isInstance(entry)) out.add(entry);
            }
        }
        for (Object constant : typeRegistry.getEnumConstants()) out.add(constant);
        return out;
    }

    /**
     * Resolves a type by name.
     * A definition file that replaces a built-in leaves both in the registry, and only this lookup says which one the game actually uses.
     */
    Object typeNamed(String name) throws Exception {
        return typeLookup.invoke(null, name);
    }

    String typeName(Object type) throws Exception { return String.valueOf(typeName.invoke(type)); }
    String typeDisplay(Object type) throws Exception { return String.valueOf(typeDisplay.invoke(type)); }
    int typePrice(Object type) throws Exception { return ((Integer) typePrice.invoke(type)).intValue(); }
    int typeTech(Object type) throws Exception { return ((Integer) typeTech.invoke(type)).intValue(); }
    boolean typeIsBuilding(Object type) throws Exception { return ((Boolean) typeBuilding.invoke(type)).booleanValue(); }
    boolean typeIsBuilder(Object type) throws Exception { return ((Boolean) typeBuilder.invoke(type)).booleanValue(); }
    float typeBuildSpeed(Object type) throws Exception { return ((Float) typeBuildSpeed.invoke(type)).floatValue(); }
    String typeMovement(Object type) throws Exception { return String.valueOf(typeMovement.invoke(type)); }

    /**
     * Which tiles a movement type cannot cross, row by row from the top left, as the path finder holds them (`l.bU`, a `gameFramework.k.l`, whose `a(ao, x, y)` is true on a blocked tile), or null when it keeps no grid for the movement type or no map is loaded.
     *
     * Standing buildings block their footprint here as well as the ground does. The width and height are the first two entries of what is returned, as the array's first row would otherwise have to be guessed.
     */
    int[] blockedTiles(Object game, String movement) {
        try {
            Object finder = field(engineClass, "bU").get(game);
            if (finder == null) return null;
            Class<?> movementClass = Class.forName("com.corrodinggames.rts.game.units.ao");
            Object kind = null;
            for (Object constant : movementClass.getEnumConstants()) {
                if (String.valueOf(constant).equals(movement)) kind = constant;
            }
            if (kind == null) return null;
            Object grid = method(finder.getClass(), "a", movementClass).invoke(finder, kind);
            if (grid == null) return null;
            int width = field(grid.getClass(), "b").getInt(grid);
            int height = field(grid.getClass(), "c").getInt(grid);
            Method blocked = method(finder.getClass(), "a", movementClass, int.class, int.class);
            int[] out = new int[2 + width * height];
            out[0] = width;
            out[1] = height;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (Boolean.TRUE.equals(blocked.invoke(finder, kind, Integer.valueOf(x), Integer.valueOf(y)))) out[2 + y * width + x] = 1;
                }
            }
            return out;
        } catch (Exception e) {
            return null;
        }
    }

    /** The engine's own sample unit of a type, or null when it keeps none. Every capability below is read off it, since it is an instance of the type's own class and answers as a unit of the type would. */
    Object sample(Object type) {
        try {
            return sampleOfType.invoke(null, type);
        } catch (Exception e) {
            return null;
        }
    }

    /** Whether units of the type can attack at all: the sample's own `l()`. A transport or a builder reports a range without having a weapon, so a range is not this answer. */
    boolean typeCanAttack(Object type) {
        Object sample = sample(type);
        try {
            return sample != null && Boolean.TRUE.equals(method(sample.getClass(), "l").invoke(sample));
        } catch (Exception e) {
            return false;
        }
    }

    /** Move speed in world units a second: the sample's `z()`, which the engine keeps per sixtieth of a second. Nought for what does not move. */
    float typeSpeed(Object type) {
        Object sample = sample(type);
        try {
            return sample == null ? 0f : ((Float) method(sample.getClass(), "z").invoke(sample)).floatValue() * 60f;
        } catch (Exception e) {
            return 0f;
        }
    }

    /** Transport capacity in slots, the sample's `bZ()`; -1 for a type that carries nothing. */
    int typeCapacity(Object type) {
        return sampleInt(type, "bZ", -1);
    }

    /** Slots a unit of the type takes aboard a transport, the sample's `cw()`. */
    int typeSlots(Object type) {
        return sampleInt(type, "cw", 1);
    }

    private int sampleInt(Object type, String name, int otherwise) {
        Object sample = sample(type);
        try {
            return sample == null ? otherwise : ((Integer) method(sample.getClass(), name).invoke(sample)).intValue();
        } catch (Exception e) {
            return otherwise;
        }
    }

    /** What a unit of the type offers to produce or place at its first tier, read off the sample's action list. */
    java.util.List<Object> typeMenu(Object type) {
        Object sample = sample(type);
        try {
            return sample == null ? new java.util.ArrayList<Object>() : producible(sample);
        } catch (Exception e) {
            return new java.util.ArrayList<Object>();
        }
    }

    /** Whether a unit of the type offers a tier raise at its first tier. */
    boolean typeUpgradable(Object type) {
        Object sample = sample(type);
        try {
            return sample != null && upgradeOffered(sample) != null;
        } catch (Exception e) {
            return false;
        }
    }

    /** Whether a transport of the first type would load a unit of the second: the engine's own load test, `d(am, boolean)`, between the two samples. It agrees with what live units do. */
    boolean typeCarries(Object transport, Object passenger) {
        Object carrier = sample(transport);
        Object cargo = sample(passenger);
        if (carrier == null || cargo == null) return false;
        try {
            return Boolean.TRUE.equals(method(unitClass, "d", unitClass, boolean.class).invoke(carrier, cargo, Boolean.FALSE));
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Maximum attack range in world units, or zero for a type that has none.
     *
     * Range is not on the type interface. A type that came from a definition file carries its numbers in a stat block, which is where the parser writes the file's maxAttackRange. A type that exists only as code carries no such block, and some of those shoot (the hover tank is one); for them the range is read off the engine's own sample unit of the type, which answers it in code.
     */
    float typeRange(Object type) {
        try {
            if (!definedTypeClass.isInstance(type)) {
                Object sample = sampleOfType.invoke(null, type);
                return sample != null && armedClass.isInstance(sample) ? ((Float) armedRange.invoke(sample)).floatValue() : 0f;
            }
            Object stats = definedTypeStats.get(type);
            return stats == null ? 0f : statsRange.getFloat(stats);
        } catch (Exception e) {
            return 0f;
        }
    }

    /** A type's maximum health, read off the engine's sample unit of it, or nought when the engine keeps no sample of the type. */
    float typeMaxHealth(Object type) {
        try {
            Object sample = sampleOfType.invoke(null, type);
            return sample == null ? 0f : unitMaxHealth.getFloat(sample);
        } catch (Exception e) {
            return 0f;
        }
    }

    /** Whether this type may only be placed on a resource pool, which is the definition of an extractor and the only way to tell one from any other building standing next to one. */
    boolean typeOnResourcePool(Object type) {
        try {
            return definedTypeClass.isInstance(type) && definedTypeOnResourcePool.getBoolean(type);
        } catch (Exception e) {
            return false;
        }
    }

    /** Whether a type can shoot at flying units, or at land units. Null when the answer is a condition that only a live unit can settle. */
    Boolean typeHitsAir(Object type) {
        return staticCondition(type, definedTypeHitsAir);
    }

    /** A type that exists only as code has no condition to read; one whose sample unit shoots is taken to shoot at the ground, which is what every such type in the game does. */
    Boolean typeHitsLand(Object type) {
        if (!definedTypeClass.isInstance(type)) return Boolean.valueOf(typeRange(type) > 0f);
        return staticCondition(type, definedTypeHitsLand);
    }

    private Boolean staticCondition(Object type, Field which) {
        try {
            if (!definedTypeClass.isInstance(type)) return Boolean.FALSE;
            Object condition = which.get(type);
            if (condition == null) return Boolean.FALSE;
            if (((Boolean) logicIsStaticTrue.invoke(null, condition)).booleanValue()) return Boolean.TRUE;
            if (((Boolean) logicIsStaticFalse.invoke(null, condition)).booleanValue()) return Boolean.FALSE;
            return null;
        } catch (Exception e) {
            return Boolean.FALSE;
        }
    }

    /** Settles a condition that was not constant, using a unit of that type as the context it is asked about. */
    private boolean readCondition(Object type, Field which, Object unit) {
        try {
            Object condition = which.get(type);
            if (condition == null || !armedClass.isInstance(unit)) return false;
            return ((Boolean) logicRead.invoke(condition, unit)).booleanValue();
        } catch (Exception e) {
            return false;
        }
    }

    boolean unitHitsAir(Object type, Object unit) {
        Boolean known = typeHitsAir(type);
        return known != null ? known.booleanValue() : readCondition(type, definedTypeHitsAir, unit);
    }

    boolean unitHitsLand(Object type, Object unit) {
        Boolean known = typeHitsLand(type);
        return known != null ? known.booleanValue() : readCondition(type, definedTypeHitsLand, unit);
    }

    /** The build number the game reports at start up, so a control process can refuse a build the field names were not read from. */
    String buildNumber() {
        try {
            Class<?> main = Class.forName("com.corrodinggames.rts.java.Main");
            Object instance = field(main, "m").get(null);
            if (instance == null) return "";
            Object value = field(main, "e").get(instance);
            return value == null ? "" : String.valueOf(value);
        } catch (Exception e) {
            return "";
        }
    }

    // ---- commands ------------------------------------------------------------------------

    /**
     * Takes a command for a player.
     * It is already queued for execution when it is handed over, so filling in its fields is the whole of the submission and there is nothing to send.
     */
    Object command(Object engine, Object player) throws Exception {
        return poolObtain.invoke(commandPool.get(engine), player);
    }

    void addUnit(Object command, Object unit) throws Exception {
        commandAddUnit.invoke(command, unit);
    }

    void moveTo(Object command, float x, float y) throws Exception {
        commandMove.invoke(command, Float.valueOf(x), Float.valueOf(y));
    }

    void attackMoveTo(Object command, float x, float y) throws Exception {
        commandAttackMove.invoke(command, Float.valueOf(x), Float.valueOf(y));
    }

    void attack(Object command, Object target) throws Exception {
        commandAttack.invoke(command, target);
    }

    /** Follows the unit and fights what threatens it. */
    void guard(Object command, Object target) throws Exception {
        commandGuard.invoke(command, target);
    }

    /** The units of the command walk to the transport and board it. A transport they cannot reach leaves them standing with the order held. */
    void loadInto(Object command, Object transport) throws Exception {
        commandLoadInto.invoke(command, transport);
    }

    /** The transport of the command goes to the passenger and takes it aboard. One at a time: a second one issued in the same step replaces the first unless the command is appended. */
    void loadUp(Object command, Object passenger) throws Exception {
        commandLoadUp.invoke(command, passenger);
    }

    /** Adds the command's order after the ones its units already hold instead of replacing them. */
    void append(Object command) throws Exception {
        commandAppend.set(command, Boolean.TRUE);
    }

    // ---- reading commands back -----------------------------------------------------------

    /** The pool's queue of commands not yet carried out, or null before a match has a pool. */
    Object commandQueue(Object engine) throws Exception {
        Object pool = commandPool.get(engine);
        return pool == null ? null : poolQueue.get(pool);
    }

    void setCommandQueue(Object engine, java.util.ArrayList<Object> queue) throws Exception {
        Object pool = commandPool.get(engine);
        if (pool != null) poolQueue.set(pool, queue);
    }

    Object issuer(Object command) throws Exception { return commandIssuer.get(command); }
    int issuedAt(Object command) throws Exception { return commandTime.getInt(command); }
    boolean appended(Object command) throws Exception { return commandAppend.getBoolean(command); }
    boolean undoes(Object command) throws Exception { return commandUndo.getBoolean(command); }
    boolean stops(Object command) throws Exception { return commandStop.getBoolean(command); }
    Object orderOf(Object command) throws Exception { return commandOrder.get(command); }
    Object specialOf(Object command) throws Exception { return commandSpecial.get(command); }
    java.util.List<?> unitsOf(Object command) throws Exception { return (java.util.List<?>) commandUnits.get(command); }
    float orderX(Object order) throws Exception { return orderX.getFloat(order); }
    float orderY(Object order) throws Exception { return orderY.getFloat(order); }
    Object orderTarget(Object order) throws Exception { return orderTarget.get(order); }

    /** The kind of an order, as its position in the engine's own enumeration, or {@link #NO_ORDER}. */
    int kindOf(Object order) throws Exception {
        Object kind = orderKind.get(order);
        for (int i = 0; i < orderKinds.length; i++) if (orderKinds[i] == kind) return i;
        return NO_ORDER;
    }

    /** The handle a command names a special action by, made from its id as the engine makes it. */
    Object actionHandle(String id) throws Exception { return actionHandle.invoke(null, id); }

    /**
     * Issues one of the unit's own actions by its action id, such as a transport's "109" (unload) and "110" (cancel the unload); false when the unit offers no action of that id.
     *
     * The id is matched against the handles of the unit's action list rather than sent blind, so an id the unit does not offer issues nothing.
     */
    boolean unitAction(Object command, Object unit, String id) throws Exception {
        Object wanted = actionHandle.invoke(null, id);
        Object actions = unitActions.invoke(unit);
        if (!(actions instanceof java.util.List)) return false;
        for (Object action : (java.util.List<?>) actions) {
            if (action == null) continue;
            Object handle = actionHandleOf.invoke(action);
            if (handle == wanted || (handle != null && handle.equals(wanted))) {
                commandAction.invoke(command, handle);
                return true;
            }
        }
        return false;
    }

    void build(Object command, float x, float y, Object type, int size) throws Exception {
        commandBuild.invoke(command, Float.valueOf(x), Float.valueOf(y), type, Integer.valueOf(size));
    }

    void setStance(Object command, int stance) throws Exception {
        if (stance < 0 || stance >= stances.length) return;
        commandStance.invoke(command, stances[stance]);
    }

    /**
     * Creates one unit of a type at a position, owned by a player, through the host's spawn system command.
     *
     * This is how a training situation is built. It is not an assignment to engine state: the command travels the same route a player's order does, which is what keeps a lockstep session in step. A command taken with no player is not queued as it is handed over, so unlike an ordinary order this one has to be submitted.
     */
    void spawn(Object engine, Object owner, Object type, float x, float y) throws Exception {
        Object command = poolTake.invoke(commandPool.get(engine));
        setField(command, "i", owner);
        setField(command, "r", Boolean.TRUE);
        setField(command, "u", Integer.valueOf(SYSTEM_SPAWN));
        commandBuild.invoke(command, Float.valueOf(x), Float.valueOf(y), type, Integer.valueOf(1));
        netSubmit.invoke(net(engine), command);
    }

    /** The system command that creates a unit outright. The engine refuses it unless the command also carries a build order with a type. */
    private static final int SYSTEM_SPAWN = 5;

    /** Production and upgrades travel as a special action whose name is built from the type's own reported name. */
    void specialAction(Object command, String handle) throws Exception {
        commandAction.invoke(command, actionHandle.invoke(null, handle));
    }

    /** Names an action by the handle the unit's own action list carries, which is how a tier raise is issued: its id is a number the unit class chose, not a name built from a type. */
    void offeredAction(Object command, Object action) throws Exception {
        commandAction.invoke(command, actionHandleOf.invoke(action));
    }

    int level(Object unit) throws Exception { return ((Integer) unitLevel.invoke(unit)).intValue(); }

    /** The tier raise this unit offers at its current tier, or null when it offers none. Buildings offer at most one at a time: the next tier. */
    Object upgradeOffered(Object unit) throws Exception {
        Object actions = unitActions.invoke(unit);
        if (!(actions instanceof java.util.List)) return null;
        for (Object action : (java.util.List<?>) actions) {
            if (action != null && actionKind.invoke(action) == upgradeKind) return action;
        }
        return null;
    }

    int actionPrice(Object action) throws Exception { return ((Integer) offeredPrice.invoke(action)).intValue(); }

    /**
     * The types this unit offers to produce or to place at its current tier, which is the engine's own answer to what a factory makes or a builder builds.
     * Read off the unit's action list: the actions of the production and placement kinds each name the type they make. A definition file cannot say this reliably, since a type whose definition names no maker may be a core unit or a part of something else.
     */
    java.util.List<Object> producible(Object unit) throws Exception {
        java.util.List<Object> out = new java.util.ArrayList<Object>();
        Object actions = unitActions.invoke(unit);
        if (!(actions instanceof java.util.List)) return out;
        for (Object action : (java.util.List<?>) actions) {
            if (action == null) continue;
            Object kind = actionKind.invoke(action);
            if (kind != queueKind && kind != placeKind) continue;
            Object type = offeredType.invoke(action);
            if (type != null) out.add(type);
        }
        return out;
    }

    void setField(Object target, String name, Object value) throws Exception {
        field(target.getClass(), name).set(target, value);
    }

    Object getField(Object target, String name) throws Exception {
        return field(target.getClass(), name).get(target);
    }

    boolean getBoolean(Object target, String name) throws Exception {
        return field(target.getClass(), name).getBoolean(target);
    }

    Object invoke(Object target, String name) throws Exception {
        return method(target.getClass(), name).invoke(target);
    }

    Object invoke(Object target, String name, Class<?> parameter, Object argument) throws Exception {
        return method(target.getClass(), name, parameter).invoke(target, argument);
    }

    Object staticField(Class<?> owner, String name) throws Exception {
        return field(owner, name).get(null);
    }

    // ---- the network session ---------------------------------------------------------------

    /** The port a host binds, as the settings currently hold it. Zero when the settings cannot be reached. */
    int networkPort(Object engine) {
        try {
            Object settings = settingsEngine == null ? null : settingsEngine.get(engine);
            return settings == null || settingsPort == null ? 0 : settingsPort.getInt(settings);
        } catch (Exception e) {
            return 0;
        }
    }

    boolean setNetworkPort(Object engine, int port) {
        try {
            Object settings = settingsEngine == null ? null : settingsEngine.get(engine);
            if (settings == null || settingsPort == null) return false;
            settingsPort.setInt(settings, port);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Opens a real multiplayer session, which is what lets a second process join this match.
     *
     * This is a different call from the single player server, and the difference is the whole point of it. A single player server is the entire session inside one process, so nothing can ever connect to it and the engine never has a second world to compare its own against; a host binds a TCP and a UDP acceptor, admits other processes, and from then on checksums the world for each of them.
     *
     * The port is written into the settings first because the engine reads it from there as it binds and never takes it as an argument, which is also why two hosts on one machine have to be given different ones. The call refuses outright if a session is already up, so whatever brought the previous one down has to have run first.
     */
    boolean hostNetworked(Object engine, int port) throws Exception {
        if (netHost == null) return false;
        // Refused rather than attempted when the port cannot be written. The engine reads the port from the settings as it binds and reports success either way, so a host that could not be told which port to use would come up listening on whatever was left in the preferences and every attempt to join it would be refused at an address that looks correct.
        if (!setNetworkPort(engine, port)) return false;
        return Boolean.TRUE.equals(netHost.invoke(net(engine), Boolean.FALSE));
    }

    /**
     * Connects to a match another process is hosting and hands the connected socket to the engine, returning the reason it did not work or null once this process is part of the session.
     *
     * The engine's connector runs on a thread of its own and reports through a callback rather than blocking, but the wait is done here anyway: the caller has nothing useful to do until the answer is known, and the socket has to be given to the engine from the same place that asked for it. The connector touches nothing but the socket, so waiting on the game thread costs a stall and risks no deadlock.
     *
     * The address is host[:port] and a bare host means the engine's default port. TCP is forced because the alternative is the engine's own hole punching, which has nothing to do on a machine talking to itself.
     */
    String join(Object engine, String address) throws Exception {
        if (netConnect == null || netAdopt == null) return "the engine's connect call could not be resolved";
        Object net = net(engine);
        long deadline = System.currentTimeMillis() + JOIN_TIMEOUT_SECONDS * 1000L;
        String last = "no attempt was made";
        // Tried again rather than once, because a host that is not listening yet refuses instantly rather than making the caller wait: the two processes are started together and the one that hosts has a map to load first, so the first several attempts are expected to fail and mean nothing.
        while (System.currentTimeMillis() < deadline) {
            String failure = attemptJoin(net, address);
            if (failure == null) return null;
            last = failure;
            Thread.sleep(1000);
        }
        return last;
    }

    private String attemptJoin(Object net, String address) throws Exception {
        final java.util.concurrent.CountDownLatch finished = new java.util.concurrent.CountDownLatch(1);
        Object connector = netConnect.invoke(net, address, Boolean.TRUE, new Runnable() {
            public void run() {
                finished.countDown();
            }
        });
        if (!finished.await(CONNECT_TIMEOUT_SECONDS, java.util.concurrent.TimeUnit.SECONDS)) {
            if (connectorCancel != null) connectorCancel.invoke(connector);
            return "no answer from " + address;
        }
        Object failure = connectorError == null ? null : connectorError.get(connector);
        if (failure != null) return String.valueOf(failure);
        Object socket = connectorSocket == null ? null : connectorSocket.get(connector);
        if (socket == null) return "connected to " + address + " but the socket was not handed back";
        // The engine tears down whatever it was doing before it takes the socket over, and it does so inside the call below rather than needing to be asked; asking separately would go through the strict resolver and turn a renamed method into a failure to start an episode at all, which is what resolving this whole block leniently was meant to rule out.
        return Boolean.TRUE.equals(netAdopt.invoke(net, socket))
                ? null : "the engine refused the connection to " + address;
    }

    /**
     * Waits in the room until another process has connected, or gives up.
     *
     * A match is started by the host, and once it has been there is nothing more to join: the other process has to be in the room before the start, not after it. Since the host is the only side that knows when it has finished loading, waiting here is where the two are brought into order, and it costs a stall in a battleroom rather than anything a match would notice.
     */
    boolean awaitPeer(Object engine, int seconds) throws Exception {
        long deadline = System.currentTimeMillis() + seconds * 1000L;
        while (System.currentTimeMillis() < deadline) {
            if (!connections(engine).isEmpty()) return true;
            Thread.sleep(200);
        }
        return false;
    }

    /**
     * The name this process answers to in a session.
     *
     * Written through the engine's own call rather than into the field behind it, because that call is also what strips the spaces the protocol cannot carry and what remembers the name for the next session. It is what a host's per client desync report names each client by, so it is worth setting to something that identifies which process this is.
     */
    void setNetworkName(Object engine, String name) throws Exception {
        if (netName != null) netName.invoke(net(engine), name);
    }

    /** Whether a session of either kind is up. */
    boolean networked(Object engine) {
        return netBoolean(engine, netStarted);
    }

    /** Whether this process hosts the session it is in, which says nothing at all unless {@link #networked} is true. */
    boolean isHost(Object engine) {
        return netBoolean(engine, netHosting);
    }

    /** The frame the current checksum was taken at, or -1 when none has been taken yet. */
    int checksumFrame(Object engine) {
        return netInt(engine, netChecksumFrame, -1);
    }

    /** Frames between checksums, which is how far apart two processes can drift before either of them finds out. */
    int checksumInterval(Object engine) {
        return netInt(engine, netChecksumInterval, 0);
    }

    /** The aggregate over every checksummed subsystem, which is the number the two sides actually compare. */
    long checksum(Object engine) {
        try {
            Object net = net(engine);
            Object record = net == null || netChecksum == null ? null : netChecksum.get(net);
            return record == null || checksumTotal == null ? 0L : checksumTotal.getLong(record);
        } catch (Exception e) {
            return 0L;
        }
    }

    /** How many checksums this process has agreed with as a client. Stays at zero on a host, which compares nothing of its own. */
    int checksumMatches(Object engine) {
        return netInt(engine, netMatches, 0);
    }

    /**
     * The connections this process holds, which on a host is one per client and is where the engine records that client's verdict.
     *
     * A client has one of these too : its connection to the server : but the four counters on it stay at their initial values for the whole match, because only the host's side of the checksum exchange ever writes them. A client's own evidence that it is in step is its count of matching checksums instead, so a reader that judged an episode by the counters alone would call every joined episode unsynchronised and throw away exactly the results a paired run exists to collect.
     */
    java.util.List<Object> connections(Object engine) {
        java.util.List<Object> out = new java.util.ArrayList<Object>();
        try {
            Object net = net(engine);
            Object queue = net == null || netConnections == null ? null : netConnections.get(net);
            if (queue instanceof java.util.Collection) out.addAll((java.util.Collection<?>) queue);
        } catch (Exception e) {
            // A report that cannot be built comes back empty rather than thrown, because it is asked for in order to decide whether an episode counts and not in order to run one.
        }
        return out;
    }

    String peerName(Object connection) {
        try {
            return peerName == null ? "" : String.valueOf(peerName.invoke(connection));
        } catch (Exception e) {
            return "";
        }
    }

    /** Whether this client is out of step at this moment. */
    boolean peerDesynced(Object connection) {
        return peerBoolean(connection, peerDesynced);
    }

    /** Whether this client is out of step and could not be brought back, which is the end of the match as a comparable thing. */
    boolean peerBroken(Object connection) {
        return peerBoolean(connection, peerBroken);
    }

    /** How many checksums this client has agreed with. None at all after a while of playing is as bad a sign as a disagreement. */
    int peerMatches(Object connection) {
        return peerInt(connection, peerMatches);
    }

    /** How many separate times this client has fallen out of step, which counts the ones it was brought back from as well. */
    int peerDesyncs(Object connection) {
        return peerInt(connection, peerDesyncs);
    }

    /**
     * The engine's own verdict on the session, as the message it would fail with, or null when it is content.
     *
     * The engine ships this as an assertion that throws, which is no use as a report: a run wants to record that an episode drifted and then go on to the next one, not stop at the point of asking. The counters above say the same thing without the exception, and this exists for when the engine's own wording is what is wanted. It is a harsh test as well as a loud one, and complains about a client that has simply not been sent a checksum yet.
     */
    String desyncComplaint(Object engine) {
        try {
            if (netAssert == null) return null;
            netAssert.invoke(net(engine));
            return null;
        } catch (java.lang.reflect.InvocationTargetException e) {
            Throwable cause = e.getCause();
            return cause == null ? e.toString() : String.valueOf(cause.getMessage());
        } catch (Exception e) {
            return e.toString();
        }
    }

    private boolean netBoolean(Object engine, Field which) {
        try {
            Object net = net(engine);
            return net != null && which != null && which.getBoolean(net);
        } catch (Exception e) {
            return false;
        }
    }

    private int netInt(Object engine, Field which, int fallback) {
        try {
            Object net = net(engine);
            return net == null || which == null ? fallback : which.getInt(net);
        } catch (Exception e) {
            return fallback;
        }
    }

    private static boolean peerBoolean(Object connection, Field which) {
        try {
            return connection != null && which != null && which.getBoolean(connection);
        } catch (Exception e) {
            return false;
        }
    }

    private static int peerInt(Object connection, Field which) {
        try {
            return connection == null || which == null ? 0 : which.getInt(connection);
        } catch (Exception e) {
            return 0;
        }
    }
}
