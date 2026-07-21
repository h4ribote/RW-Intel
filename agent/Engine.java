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
    private final Field objectId;        // w.eh
    private final Field objectX;         // w.eo
    private final Field objectY;         // w.ep
    private final Method unitPrice;      // am.cL()
    private final Method unitVisibleTo;  // am.d(n)
    private final Field unitStance;      // y.P
    private final Method unitTarget;     // y.ab()
    private final Method unitOrderCount; // y.av()

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
    private final Method commandAddUnit; // e.a(am)
    private final Method commandMove;    // e.a(float, float)
    private final Method commandAttackMove; // e.b(float, float)
    private final Method commandAttack;  // e.a(am)  -- resolved separately, same erasure as addUnit
    private final Method commandBuild;   // e.a(float, float, as, int)
    private final Method commandStance;  // e.a(a)
    private final Method commandAction;  // e.a(a.c)
    private final Method actionHandle;   // a.c.a(String)

    private final Object[] stances;

    Engine() throws Exception {
        engineClass = Class.forName("com.corrodinggames.rts.gameFramework.l");
        unitClass = Class.forName("com.corrodinggames.rts.game.units.am");
        armedClass = Class.forName("com.corrodinggames.rts.game.units.y");
        objectClass = Class.forName("com.corrodinggames.rts.gameFramework.w");
        playerClass = Class.forName("com.corrodinggames.rts.game.n");
        typeInterface = Class.forName("com.corrodinggames.rts.game.units.as");
        typeRegistry = Class.forName("com.corrodinggames.rts.game.units.ar");
        stanceClass = Class.forName("com.corrodinggames.rts.game.units.a");
        actionClass = Class.forName("com.corrodinggames.rts.game.units.a.c");
        loadModeClass = Class.forName("com.corrodinggames.rts.gameFramework.s");
        mapKindClass = Class.forName("com.corrodinggames.rts.gameFramework.j.ai");

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
        objectId = field(objectClass, "eh");
        objectX = field(objectClass, "eo");
        objectY = field(objectClass, "ep");
        unitPrice = method(unitClass, "cL");
        unitVisibleTo = method(unitClass, "d", playerClass);
        unitStance = field(armedClass, "P");
        unitTarget = method(armedClass, "ab");
        unitOrderCount = method(armedClass, "av");

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
        poolObtain = method(poolClass, "b", playerClass);
        commandAddUnit = method(commandClass, "a", armedClass);
        commandAttack = method(commandClass, "a", unitClass);
        commandMove = method(commandClass, "a", float.class, float.class);
        commandAttackMove = method(commandClass, "b", float.class, float.class);
        commandBuild = method(commandClass, "a", float.class, float.class, typeInterface, int.class);
        commandStance = method(commandClass, "a", stanceClass);
        commandAction = method(commandClass, "a", actionClass);
        actionHandle = method(actionClass, "a", String.class);

        stances = (Object[]) stanceClass.getMethod("values").invoke(null);
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

    void setSpeed(Object engine, float multiplier) throws Exception {
        field(engine.getClass(), "H").setFloat(engine, multiplier);
    }

    float speed(Object engine) throws Exception {
        return field(engine.getClass(), "H").getFloat(engine);
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

    void build(Object command, float x, float y, Object type, int size) throws Exception {
        commandBuild.invoke(command, Float.valueOf(x), Float.valueOf(y), type, Integer.valueOf(size));
    }

    void setStance(Object command, int stance) throws Exception {
        if (stance < 0 || stance >= stances.length) return;
        commandStance.invoke(command, stances[stance]);
    }

    /** Production and upgrades travel as a special action whose name is built from the type's own reported name. */
    void specialAction(Object command, String handle) throws Exception {
        commandAction.invoke(command, actionHandle.invoke(null, handle));
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
}
