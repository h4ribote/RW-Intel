import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.locks.LockSupport;

import org.newdawn.slick.GameContainer;
import org.newdawn.slick.opengl.renderer.SGL;

/**
 * How a game process draws, how its clock advances, and a hook that runs on the game thread once every frame.
 *
 * Both agents carry this and start it from their premain; it touches the game only by reflection and through Slick's public renderer interface.
 * With drawing off, Slick's renderer is replaced by one that drops everything that draws and the display is marked minimised, which makes LWJGL skip the buffer swap.
 * The fixed clock advances every frame by exactly one step of game time and lets frames run as fast as the processor allows, optionally held to a multiple of real time; the wall clock is the game's own, driven by real elapsed time under a frame rate cap.
 * The replay clock is for playing a recorded match back: it holds the engine speed multiplier at one, because a replay's world step scales with it and a step of another length is a different match, and gives every frame the same elapsed time, which the replay player turns into whole steps; frames otherwise run as on the fixed clock.
 *
 * Agent options it takes, comma separated with the agent's own:
 *   draw=&lt;bool&gt;       draw frames, default false
 *   clock=fixed|wall|replay  default fixed
 *   step=&lt;ms&gt;         game time per frame on the fixed clock, and elapsed time per frame on the replay clock, default 25
 *   speed=&lt;float&gt;     fixed and replay clock: the most game time per real time, 0 for no limit; wall clock: the engine speed multiplier, 0 to leave it alone
 *   fps=&lt;n&gt;           frame rate cap on the wall clock, default 300
 */
final class Frame {

    /** Something to run on the game thread every frame, after the frame's clock is set and before its simulation step. */
    interface Listener {
        void onFrame(int gameTimeMs) throws Exception;
    }

    /** Static fields that hold Slick's shared renderer on the drawing path. GameContainer.GL is not among them: it always gets an instance of its own, the one carrying the hooks. */
    private static final String[][] RENDERER_HOLDERS = {
        {"org.newdawn.slick.opengl.renderer.Renderer", "renderer"},
        {"org.newdawn.slick.Graphics", "GL"},
        {"org.newdawn.slick.Image", "GL"},
        {"org.newdawn.slick.geom.ShapeRenderer", "GL"},
        {"org.newdawn.slick.CachedRender", "GL"},
        {"org.newdawn.slick.BigImage", "GL"},
        {"org.newdawn.slick.AngelCodeFont", "GL"},
        {"org.newdawn.slick.util.MaskUtil", "GL"},
        {"com.corrodinggames.rts.java.e", "W"},
        {"com.corrodinggames.rts.java.d.a", "k"},
    };

    /** Frames per second of real time while an agent has said nothing is going on, on either clock. */
    static final int IDLE_FPS = 30;

    /** How far behind its pace the fixed clock may fall before the pace is measured from the present instead, so that a stall is not followed by a burst. */
    private static final long REBASE_NANOS = 250000000L;

    private static volatile boolean draw = false;
    private static volatile boolean fixedClock = true;
    private static volatile boolean replayClock = false;
    private static volatile int stepMs = 25;
    private static volatile float speed = 0f;
    private static volatile int fps = 300;
    private static volatile boolean idle = false;

    private static volatile GameContainer container;
    private static volatile Object game;
    private static volatile Object engine;
    private static volatile Field elapsed;
    private static volatile Field gameTime;
    private static volatile Field multiplier;
    private static volatile float fixedMultiplier;

    private static final CopyOnWriteArrayList<Listener> listeners = new CopyOnWriteArrayList<Listener>();

    // Touched only on the game thread.
    private static int lastGameTime = -1;
    private static int stepMismatches = 0;
    private static int anchorGameTime = 0;
    // Also cleared by setSpeed from whichever thread changes the speed.
    private static volatile long anchorNanos = 0L;
    private static long lastIdleNanos = 0L;

    private Frame() {
    }

    /** Takes one agent option if it is one of this layer's, and says whether it was. */
    static boolean option(String key, String value) {
        if (key.equals("draw")) draw = Boolean.parseBoolean(value);
        else if (key.equals("clock")) parseClock(value);
        else if (key.equals("step")) stepMs = Integer.parseInt(value);
        else if (key.equals("speed")) speed = Float.parseFloat(value);
        else if (key.equals("fps")) fps = Integer.parseInt(value);
        else return false;
        return true;
    }

    private static void parseClock(String value) {
        if (!value.equals("fixed") && !value.equals("wall") && !value.equals("replay")) {
            throw new IllegalArgumentException("clock is fixed, wall or replay, not " + value);
        }
        fixedClock = value.equals("fixed");
        replayClock = value.equals("replay");
    }

    /** Starts the thread that applies all of this once the game has come up, and keeps it applied. */
    static void install() {
        if (stepMs < 1) throw new IllegalArgumentException("step has to be at least 1 ms, not " + stepMs);
        fixedMultiplier = multiplierFor(stepMs);
        String clock = fixedClock ? "fixed step=" + stepMs + "ms" : replayClock ? "replay elapsed=" + stepMs + "ms" : "wall fps=" + fps;
        log("starting: draw=" + draw + " clock=" + clock + " speed=" + speed);
        Thread thread = new Thread(new Runnable() {
            public void run() {
                maintain();
            }
        }, "rw-frame");
        thread.setDaemon(true);
        thread.start();
    }

    static void addListener(Listener listener) {
        listeners.add(listener);
    }

    static boolean fixedClock() {
        return fixedClock;
    }

    static boolean replayClock() {
        return replayClock;
    }

    /** Whether how long a frame took in real time changes nothing the simulation sees, which holds on the fixed and the replay clock and is what lets the game thread wait on the control process. */
    static boolean steady() {
        return fixedClock || replayClock;
    }

    static int stepMs() {
        return stepMs;
    }

    static float speed() {
        return speed;
    }

    /** On the fixed clock the most game time per real time, 0 for no limit; on the wall clock the engine speed multiplier. */
    static void setSpeed(float value) {
        speed = value;
        anchorNanos = 0L;
    }

    /** Whether the agent has nothing going on, in which case frames are held to {@link #IDLE_FPS} so that a waiting process costs next to nothing. */
    static void setIdle(boolean value) {
        idle = value;
    }

    /**
     * The engine speed multiplier that makes one millisecond of elapsed time advance the game clock by exactly the step.
     * The engine turns elapsed milliseconds into sixtieths of a second with 0.06f, scales them by the multiplier, adds the product with 16.666666f to its clock and truncates, so the step itself can fall one short.
     */
    static float multiplierFor(int step) {
        float multiplier = step;
        while ((int) (0.06f * multiplier * 16.666666f) < step) multiplier = Math.nextUp(multiplier);
        return multiplier;
    }

    // The hooks, on the game thread.

    /** Sets the clock for the frame about to be simulated and runs the listeners. */
    static void beforeRender() {
        Object current = engine;
        if (current == null) return;
        int now;
        try {
            if (fixedClock) {
                elapsed.setInt(game, 1);
                multiplier.setFloat(current, fixedMultiplier);
            } else if (replayClock) {
                // A replay's world step is its step rate times the multiplier, so the multiplier stays at the one the match was recorded at and the elapsed time decides only how many whole steps a frame runs.
                elapsed.setInt(game, stepMs);
                multiplier.setFloat(current, 1f);
            } else if (speed > 0f) {
                multiplier.setFloat(current, speed);
            }
            now = gameTime.getInt(current);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
        checkStep(now);
        for (Listener listener : listeners) {
            try {
                listener.onFrame(now);
            } catch (Throwable e) {
                log("listener failed: " + e);
                e.printStackTrace();
            }
        }
    }

    /** Replaces the game's frame rate cap and paces the frame, just before the container would sync to the cap. */
    static void beforeSync() {
        // The engine is published last, so once it is there the container is too.
        Object currentEngine = engine;
        if (currentEngine == null) return;
        GameContainer current = container;
        if (!steady()) {
            current.setTargetFrameRate(idle ? IDLE_FPS : fps);
            return;
        }
        current.setTargetFrameRate(-1);
        if (idle) {
            paceIdle();
        } else if (speed > 0f) {
            try {
                pace(gameTime.getInt(currentEngine));
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
        }
    }

    /** How many frames that advanced the fixed clock by something other than one step are reported one by one. */
    private static final int STEP_MISMATCHES_REPORTED = 5;

    /** Reports the first few frames on the fixed clock that advanced the game clock by anything but nothing or one step, which would mean the engine turns time into steps differently from {@link #multiplierFor}. */
    private static void checkStep(int now) {
        int advanced = now - lastGameTime;
        if (fixedClock && lastGameTime >= 0 && advanced > 0 && advanced != stepMs && stepMismatches < STEP_MISMATCHES_REPORTED) {
            stepMismatches++;
            log("warning: a frame advanced the game clock by " + advanced + " ms, not the " + stepMs + " ms step, at game time " + now + " ms");
        }
        lastGameTime = now;
    }

    /** Holds game time to at most `speed` times real time, measured from an anchor that moves to the present when a new episode rewinds the clock or the game falls well behind. */
    private static void pace(int now) {
        long wall = System.nanoTime();
        if (anchorNanos == 0L || now < anchorGameTime) {
            anchorNanos = wall;
            anchorGameTime = now;
            return;
        }
        long due = anchorNanos + (long) ((now - anchorGameTime) * 1e6 / speed);
        if (due > wall) {
            LockSupport.parkNanos(due - wall);
        } else if (wall - due > REBASE_NANOS) {
            anchorNanos = wall;
            anchorGameTime = now;
        }
    }

    private static void paceIdle() {
        long wall = System.nanoTime();
        long due = lastIdleNanos + 1000000000L / IDLE_FPS;
        if (due > wall) {
            LockSupport.parkNanos(due - wall);
            wall = due;
        }
        lastIdleNanos = wall;
    }

    // Applying all of it and keeping it applied.

    private static void maintain() {
        boolean installed = false;
        while (true) {
            try {
                Thread.sleep(installed ? 1000 : 50);
                if (!loaded("com.corrodinggames.rts.java.Main") || !loaded("com.corrodinggames.rts.gameFramework.l")) continue;
                if (!installed) {
                    installed = resolve();
                    if (installed) log("installed");
                }
                if (installed) apply();
            } catch (InterruptedException e) {
                return;
            } catch (Throwable e) {
                log("failed: " + e);
                e.printStackTrace();
            }
        }
    }

    /** Finds the container, the Slick game and the engine, and says whether all three are there yet. */
    private static boolean resolve() throws Exception {
        Class<?> mainClass = Class.forName("com.corrodinggames.rts.java.Main");
        Object main = field(mainClass, "m").get(null);
        if (main == null) return false;
        Object foundContainer = field(mainClass, "k").get(main);
        Object foundGame = field(mainClass, "j").get(main);
        Class<?> engineClass = Class.forName("com.corrodinggames.rts.gameFramework.l");
        Object foundEngine = engineClass.getMethod("B").invoke(null);
        if (foundContainer == null || foundGame == null || foundEngine == null) return false;
        elapsed = field(foundGame.getClass(), "t");
        gameTime = field(engineClass, "by");
        multiplier = field(foundEngine.getClass(), "H");
        game = foundGame;
        container = (GameContainer) foundContainer;
        GameContainer resolved = container;
        if (!draw) {
            // A minimised window stops the container from rendering unless told to render regardless, and the simulation runs inside the render.
            resolved.setAlwaysRender(true);
            resolved.setUpdateOnlyWhenVisible(false);
        }
        engine = foundEngine;
        return true;
    }

    /** Applied again every second, because the game may bring in a class holding the renderer, recreate its settings, or be told by the display that it is visible. */
    private static void apply() throws Exception {
        suppressCrashReports();
        if (!draw) markMinimised();
        installRenderers();
    }

    private static boolean crashReportsChecked = false;

    /** The game uploads a report of any uncaught exception to its developer while its sendReports setting is on. */
    private static void suppressCrashReports() throws Exception {
        Object settings = field(Class.forName("com.corrodinggames.rts.gameFramework.l"), "bQ").get(engine);
        if (settings == null) return;
        Field sendReports = field(settings.getClass(), "sendReports");
        boolean on = sendReports.getBoolean(settings);
        if (on) sendReports.setBoolean(settings, false);
        if (!crashReportsChecked || on) log("crash report upload: " + (on ? "was on, turned off" : "off"));
        crashReportsChecked = true;
    }

    /** LWJGL swaps buffers only while the display is visible or dirty, and on Linux visible means not minimised. */
    private static void markMinimised() throws Exception {
        Object display = field(Class.forName("org.lwjgl.opengl.Display"), "display_impl").get(null);
        if (display != null) field(display.getClass(), "minimized").setBoolean(display, true);
    }

    private static void installRenderers() throws Exception {
        Field shared = field(Class.forName("org.newdawn.slick.opengl.renderer.Renderer"), "renderer");
        Object current = shared.get(null);
        SGL real = (SGL) (current instanceof FrameRenderer ? ((FrameRenderer) current).real : current);
        if (real == null) return;
        Field containerHolder = field(GameContainer.class, "GL");
        if (!(containerHolder.get(null) instanceof FrameRenderer)) {
            containerHolder.set(null, new FrameRenderer(real, !draw, true));
            log("frame hooks in place");
        }
        if (draw) return;
        FrameRenderer dropping = current instanceof FrameRenderer ? (FrameRenderer) current : new FrameRenderer(real, true, false);
        for (String[] holder : RENDERER_HOLDERS) {
            // Only classes the game has loaded already: naming one here would run its static initialiser ahead of the game, and one loaded later takes the shared renderer, which is replaced first.
            if (!loaded(holder[0])) continue;
            Field field = field(Class.forName(holder[0]), holder[1]);
            if (field.get(null) == real) field.set(null, dropping);
        }
    }

    // Reflection.

    private static Method findLoadedClass;

    /** Whether the application class loader has loaded the class already, found without loading or initialising it. */
    static boolean loaded(String name) throws Exception {
        if (findLoadedClass == null) {
            Method method = ClassLoader.class.getDeclaredMethod("findLoadedClass", String.class);
            method.setAccessible(true);
            findLoadedClass = method;
        }
        return findLoadedClass.invoke(ClassLoader.getSystemClassLoader(), name) != null;
    }

    private static Field field(Class<?> owner, String name) throws NoSuchFieldException {
        for (Class<?> type = owner; type != null; type = type.getSuperclass()) {
            try {
                Field found = type.getDeclaredField(name);
                found.setAccessible(true);
                return found;
            } catch (NoSuchFieldException e) {
                // declared further up
            }
        }
        throw new NoSuchFieldException(name + " on " + owner.getName());
    }

    static void log(String message) {
        System.out.println("[rw-frame] " + message);
        System.out.flush();
    }
}
