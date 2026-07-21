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
            Thread reader = new Thread(new Runnable() {
                public void run() {
                    read();
                }
            }, "rw-link-reader");
            reader.setDaemon(true);
            reader.start();
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private void read() {
        try {
            DataInputStream in = new DataInputStream(socket.getInputStream());
            while (running) {
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
            close();
        }
    }

    void send(int kind, byte[] body) {
        OutputStream stream = out;
        if (stream == null || !running) return;
        try {
            synchronized (this) {
                Wire.write(stream, kind, instance, body);
            }
        } catch (IOException e) {
            RwAgent.log("link: send failed: " + e);
            close();
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
