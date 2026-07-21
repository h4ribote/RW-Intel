import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The socket, and nothing else.
 *
 * The game thread never touches this class's socket and never waits on it. It leaves an observation in a slot and picks up whatever action is in another, both single element and swapped atomically. That is what makes the decision lag exactly one control period and keeps it there: an action that has not arrived is simply not applied, rather than the simulation stalling until it does. A lag that depended on how busy the machine was would make the policy that is actually being run change with the load.
 */
final class Link {

    private final String host;
    private final int port;
    private final int instance;

    private volatile Socket socket;
    private volatile OutputStream out;
    private volatile boolean running;
    /** Raised every time a connection is made. A reader belongs to one connection, and only the connection it belongs to is its to close. */
    private final java.util.concurrent.atomic.AtomicInteger generation = new java.util.concurrent.atomic.AtomicInteger();

    /** The newest action from the control process, taken by the game thread at the top of a period. */
    private final AtomicReference<byte[]> pendingAction = new AtomicReference<byte[]>();

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

    /** Connects and starts the reader. Returns false if the control process is not listening yet, so the caller can retry. */
    boolean connect() {
        try {
            Socket connection = new Socket();
            connection.connect(new InetSocketAddress(host, port), 3000);
            connection.setTcpNoDelay(true);
            socket = connection;
            out = connection.getOutputStream();
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
                if (frame.kind == Wire.KIND_ACTION) {
                    pendingAction.set(frame.body);
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
        OutputStream stream = out;
        if (stream == null || !running) return false;
        try {
            synchronized (this) {
                Wire.write(stream, kind, instance, body);
            }
            return true;
        } catch (IOException e) {
            RwAgent.log("link: send failed: " + e);
            close();
            return false;
        }
    }

    /** The newest action, cleared as it is taken so the same decision is never applied twice. */
    byte[] takeAction() {
        return pendingAction.getAndSet(null);
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
        pendingAction.set(null);
        if (connection != null) {
            try {
                connection.close();
            } catch (IOException ignored) {
                // closing a socket that is already gone is not a failure worth reporting
            }
        }
    }
}
