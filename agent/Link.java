import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * The socket, and nothing else.
 *
 * The game thread never touches this class's socket. It sends each observation with a sequence number in the frame header, and the control process answers every observation with one action carrying the same number, empty when it has nothing to change.
 * The game thread picks up the newest answer from a single element slot at the top of the next period. On the fixed clock it waits there for the answer to its own last observation, which makes the decision lag exactly one period whatever the load; on the wall clock it takes whatever is there, because waiting would become one long simulation step.
 */
final class Link {

    /** No observation: none sent yet in this episode, or the one sent went on a connection that has since been replaced. */
    static final int NONE = -1;

    private final String host;
    private final int port;
    private final int instance;

    private volatile Socket socket;
    private volatile OutputStream out;
    private volatile boolean running;
    /** Whether the current or last connection has carried a control or action frame, which is what a control process with work for this game sends. */
    private volatile boolean heard;
    /** Whether any connection has been made yet. */
    private volatile boolean made;
    /** Raised every time a connection is made. A reader belongs to one connection, and only the connection it belongs to is its to close. */
    private final java.util.concurrent.atomic.AtomicInteger generation = new java.util.concurrent.atomic.AtomicInteger();

    /** Guards the answer slot, and is what a waiting game thread waits on. */
    private final Object answers = new Object();
    /** The newest action from the control process, and the observation number it answers. */
    private byte[] answer;
    private int answered = NONE;

    // The observation numbering, touched only on the game thread.
    private int nextObservation = 0;
    private int lastSentGeneration = 0;
    private int takenNumber = NONE;

    /** How often a thread still waiting for an answer says so, in milliseconds. */
    private static final long WAIT_REPORT_MS = 30000L;

    /** Control frames, which are rare and must all be seen rather than only the newest. */
    private final ConcurrentLinkedQueue<String> control = new ConcurrentLinkedQueue<String>();

    Link(String host, int port, int instance) {
        this.host = host;
        this.port = port;
        this.instance = instance;
    }

    boolean connected() {
        return running;
    }

    /** Whether the current connection, or the last one when there is none, has carried any frame from the control process. */
    boolean heard() {
        return heard;
    }

    /** Whether the link is down and the connection that ended carried nothing from the control process, which is how a control process with nothing for this game to play treats it. */
    boolean closedIdle() {
        return made && !running && !heard;
    }

    /** Connects and starts the reader. Returns false if the control process is not listening yet, so the caller can retry. */
    boolean connect() {
        try {
            Socket connection = new Socket();
            connection.connect(new InetSocketAddress(host, port), 3000);
            connection.setTcpNoDelay(true);
            socket = connection;
            out = connection.getOutputStream();
            heard = false;
            made = true;
            running = true;
            final int mine = generation.incrementAndGet();
            Thread reader = new Thread(new Runnable() {
                public void run() {
                    read(mine);
                }
            }, "rw-link-reader");
            reader.setDaemon(true);
            reader.start();
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private void read(int mine) {
        try {
            DataInputStream in = new DataInputStream(socket.getInputStream());
            while (running && generation.get() == mine) {
                Wire.Frame frame = Wire.read(in);
                if (frame == null) break;
                heard = true;
                if (frame.kind == Wire.KIND_ACTION) {
                    synchronized (answers) {
                        answer = frame.body;
                        answered = frame.flags;
                        answers.notifyAll();
                    }
                } else if (frame.kind == Wire.KIND_CONTROL) {
                    control.add(new String(frame.body, "UTF-8"));
                }
            }
        } catch (Exception e) {
            RwAgent.log("link: reader stopped: " + e);
        } finally {
            // Only if this reader's own connection is still the current one. A reader that unblocks after the link has been remade would otherwise close the connection that replaced it, and the agent would drop itself the moment it reconnected.
            if (generation.get() == mine) close();
        }
    }

    /** Sends one frame. Returns whether it went, so that a caller with something that must not be lost can hold on to it. */
    boolean send(int kind, byte[] body) {
        return send(kind, body, 0);
    }

    private boolean send(int kind, byte[] body, int flags) {
        OutputStream stream = out;
        if (stream == null || !running) return false;
        try {
            synchronized (this) {
                Wire.write(stream, kind, instance, flags, body);
            }
            return true;
        } catch (IOException e) {
            RwAgent.log("link: send failed: " + e);
            close();
            return false;
        }
    }

    /** Sends an observation under the next number, and returns the number, or {@link #NONE} when it did not go. Called on the game thread. */
    int sendObservation(byte[] body) {
        int number = nextObservation;
        nextObservation = (nextObservation + 1) & Wire.FLAGS_MASK;
        int connection = generation.get();
        if (!send(Wire.KIND_OBSERVATION, body, number)) return NONE;
        lastSentGeneration = connection;
        return number;
    }

    /** The newest action, cleared as it is taken so the same decision is never applied twice. */
    byte[] takeAction() {
        synchronized (answers) {
            byte[] taken = answer;
            answer = null;
            takenNumber = taken != null ? answered : NONE;
            return taken;
        }
    }

    /** The observation number the action last returned by {@link #takeAction} or {@link #awaitAction} answers, or {@link #NONE} when that call returned none. Called on the game thread. */
    int takenNumber() {
        return takenNumber;
    }

    /**
     * Waits for the answer to observation `number` and takes it, or returns null once the connection it went on is gone.
     * An answer to an older observation is left for the next taker rather than returned, because it was decided on a world that has moved on.
     */
    byte[] awaitAction(int number) {
        synchronized (answers) {
            long started = System.currentTimeMillis();
            long reported = started;
            while (running && generation.get() == lastSentGeneration && !(answer != null && answered == number)) {
                try {
                    answers.wait(1000L);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return null;
                }
                long now = System.currentTimeMillis();
                if (now - reported >= WAIT_REPORT_MS) {
                    reported = now;
                    RwAgent.log("link: still waiting for the answer to observation " + number + " after " + (now - started) / 1000 + "s");
                }
            }
            if (answer == null || answered != number) {
                takenNumber = NONE;
                return null;
            }
            byte[] taken = answer;
            answer = null;
            takenNumber = number;
            return taken;
        }
    }

    String takeControl() {
        return control.poll();
    }

    boolean hasControl() {
        return !control.isEmpty();
    }

    void close() {
        running = false;
        Socket connection = socket;
        socket = null;
        out = null;
        synchronized (answers) {
            answer = null;
            answered = NONE;
            answers.notifyAll();
        }
        if (connection != null) {
            try {
                connection.close();
            } catch (IOException ignored) {
                // closing a socket that is already gone is not a failure worth reporting
            }
        }
    }
}
